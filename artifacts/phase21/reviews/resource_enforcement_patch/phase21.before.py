"""P2.1 replay implementation, measured storage, and complete pilot collection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

from world_models.data.audit import audit_initial_dataset
from world_models.data.benchmark import benchmark_fsync_directories, run_storage_benchmark
from world_models.data.collection import collect_initial_dataset, verify_initial_collection
from world_models.data.ledger import atomic_json, file_sha256
from world_models.data.resources import CollectionResources
from world_models.env.acceptance_metrics import cpu_budget
from world_models.env.provenance import runtime_manifest
from world_models.infrastructure.config import resolve_config


ROOT = Path(__file__).resolve().parents[1]


def execution_sources() -> dict:
    paths = set()
    for folder, extensions in (("src", ("*.py",)), ("scripts", ("*.py",)), ("tests", ("*.py",)),
                               ("configs", ("*.json", "*.yaml")), ("scenarios", ("*.acs",))):
        for pattern in extensions:
            paths.update((ROOT / folder).rglob(pattern))
    paths.update(ROOT / name for name in ("pyproject.toml", "requirements-phase0.lock", "requirements-phase1.lock"))
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sorted(paths)}


def dataset_cases(profile: str, task: str, seeds: list[int] | None) -> list[tuple[str, int]]:
    if profile not in ("pilot", "final") or task not in ("relay", "cover", "both"):
        raise ValueError("choose a declared experiment profile and task family")
    families = ("relay", "cover") if task == "both" else (task,)
    configured = resolve_config("relay", "B", profile, ROOT / "configs").settings["experiment"]["master_seeds"]
    selected = configured if seeds is None else seeds
    if not selected or len(set(selected)) != len(selected) or any(type(seed) is not int or seed not in configured for seed in selected):
        raise ValueError("choose unique training seeds from the selected profile")
    return [(family, seed) for family in families for seed in selected]


def save_report(path: Path, report: dict) -> None:
    atomic_json(path, report)
    path.with_suffix(".sha256").write_text(file_sha256(path) + "  " + path.name + "\n", encoding="ascii")


def collect_with_resource_report(report: dict, key: str, cpu_ids: list[int], callback):
    """Retain observed resources when a collection future raises.

    A secondary export failure is recorded without masking the original native
    failure. On an otherwise successful stage, missing resources still fail
    closed as they did before this reporting fix.
    """
    resources = CollectionResources(cpu_ids)
    original_error = None
    try:
        with resources:
            return callback()
    except BaseException as error:
        original_error = error
        raise
    finally:
        try:
            report[key] = resources.report()
        except BaseException as resource_error:
            report[key] = {"status": "unavailable", "error": repr(resource_error),
                           "scope": "resource export failed; values were not inferred or replaced"}
            if original_error is None:
                raise


def check_disk(path: Path, transitions: int, cases: int) -> dict:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    # Conservative raw RGB + metadata/journals and temporary working margin.
    required = int(cases * transitions * 73728 * 1.5 + 2 * 1024**3)
    if free < required:
        raise RuntimeError(f"insufficient free disk for raw replay and overhead: {free} < {required}")
    return {"free_bytes_at_start": free, "conservative_required_bytes": required,
            "estimate": "uncompressed images x1.5 plus 2GiB; no assumed compression saving"}


def progress(record: dict) -> None:
    # An episode update is persisted by the collector; stdout remains compact.
    count = record["published_joint_transitions"]
    if count % 1000 == 0 or record["status"] != "collecting":
        print(json.dumps({key: record[key] for key in ("status", "task_family", "run_seed", "published_joint_transitions", "target_transitions", "completed_episodes", "elapsed_seconds")}), flush=True)


def collection_origin(journal_root: Path | None) -> dict:
    runtime = runtime_manifest(ROOT)
    return {"execution_source_sha256": execution_sources(),
            "runtime": {key: value for key, value in runtime.items() if key != "recorded_at_utc"},
            "journal_mode": "durable_temporary_spool" if journal_root is not None else "dataset_local"}


def collect_cases(cases, directory: Path, *, profile: str, transitions: int,
                  jobs: int, storage: dict, reuse: bool, journal_root: Path | None = None,
                  journal_spool_max_bytes: int = 512 * 1024**2) -> list[dict]:
    origin = collection_origin(journal_root)
    origin["storage_setting"] = {key: storage[key] for key in ("compression", "compression_level", "chunk_transitions")}
    existing = {}
    # Reject any incompatible/partial output before starting a different case.
    for family, seed in cases:
        target = directory / family / f"seed_{seed}"
        if not target.exists():
            continue
        if not reuse:
            raise FileExistsError(f"dataset output exists: {target}; use --reuse only for verified complete data")
        verified = verify_initial_collection(target)
        report = json.loads((target / "collection_report.json").read_text(encoding="utf-8"))
        spec = json.loads((target / "collection_spec.json").read_text(encoding="utf-8"))
        if (report["target_transitions"] != transitions or report["task_family"] != family
                or report["run_seed"] != seed or report["profile"] != profile
                or spec.get("origin_provenance") != origin):
            raise ValueError("existing collection case/origin does not match the current frozen procedure")
        existing[family, seed] = {"path": str(target.resolve()), "reused": True, "report": report, "verification": verified}

    def run(case):
        family, seed = case
        target = directory / family / f"seed_{seed}"
        if case in existing:
            print(json.dumps({"status": "reused_verified_complete_dataset", "task_family": family, "run_seed": seed}), flush=True)
            return existing[case]
        report = collect_initial_dataset(family, seed, target, profile=profile, total_transitions=transitions,
                                         chunk_transitions=storage["chunk_transitions"], compression=storage["compression"],
                                         compression_level=storage["compression_level"], progress_callback=progress,
                                         journal_root=journal_root / family / f"seed_{seed}" if journal_root is not None else None,
                                         journal_spool_max_bytes=journal_spool_max_bytes, origin_provenance=origin)
        return {"path": str(target.resolve()), "reused": False, "report": report}
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(run, case): case for case in cases}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: (item["report"]["task_family"], item["report"]["run_seed"]))


def unit_suite(output: Path) -> dict:
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    report = {"status": "passed" if result.returncode == 0 else "failed", "returncode": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr}
    save_report(output / "unit_tests_report.json", report)
    if result.returncode:
        raise RuntimeError("unit suite failed before native replay collection")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("benchmark", "preflight", "collect", "verify", "complete"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "phase21")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "initial_replay" / "pilot")
    parser.add_argument("--task", choices=("relay", "cover", "both"), default="both")
    parser.add_argument("--profile", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--transitions", type=int)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--sample-datasets", type=Path, nargs="+", help="Sealed replay directories for a standalone production-data storage benchmark")
    parser.add_argument("--preflight-data-dir", type=Path, help="Optional directory for complete's four 1,000-transition native preflights")
    parser.add_argument("--journal-root", type=Path, help="Parent for an exclusively created temporary hot-journal spool; default is the OS temporary directory")
    parser.add_argument("--journal-spool-max-mib", type=int, default=512, help="Per-dataset hot-journal capacity guard; images and verified closed archives remain on --data-dir")
    parser.add_argument("--prior-diagnostic-report", type=Path, action="append", default=[], help="Retained aborted/development diagnostic report to include in campaign resource accounting")
    parser.add_argument("--reuse", action="store_true", help="Reuse only fully verified sealed datasets; never restart or erase failed attempts")
    parser.add_argument("--frozen-protocol", type=Path, help="Required before any final-seed collection; no final collection is part of the current P2.1 pilot scope")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / ("phase21_acceptance.json" if args.command == "complete" else f"{args.command}_report.json")
    if path.exists():
        parser.error("report output exists; use a fresh --output-dir to preserve historical evidence")
    try:
        cases = dataset_cases(args.profile, args.task, args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    configured_count = resolve_config("relay", "B", args.profile, ROOT / "configs").settings["experiment"]["initial_transitions"]
    transitions = args.transitions if args.transitions is not None else (1000 if args.command == "preflight" else configured_count)
    if type(transitions) is not int or not 0 < transitions <= configured_count or transitions % 10:
        parser.error("transition count must be positive, no larger than the approved initialization, and permit exact 70/30 quotas")
    if args.journal_spool_max_mib < 1:
        parser.error("journal spool capacity must be positive")
    if args.command == "complete" and (args.profile != "pilot" or set(cases) != {("relay", 0), ("relay", 1), ("cover", 0), ("cover", 1)} or transitions != configured_count):
        parser.error("complete requires all four approved pilot datasets at 50,000 transitions each")
    if args.profile == "final" and args.command in ("collect", "complete", "preflight"):
        if args.frozen_protocol is None:
            parser.error("confirmatory seeds remain untouched until a frozen protocol manifest is supplied")
        protocol = json.loads(args.frozen_protocol.read_text(encoding="utf-8"))
        if protocol.get("status") != "frozen" or protocol.get("execution_source_sha256") != execution_sources():
            parser.error("frozen protocol must pin the current execution sources")
    report = {"schema_version": "phase21-acceptance-v1", "status": "running", "scope": f"P2.1 {args.profile} real replay only",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "command": args.command,
              "profile": args.profile, "cases": [list(case) for case in cases], "target_transitions_per_dataset": transitions,
              "data_directory": str(args.data_dir.resolve()), "source_sha256_at_start": execution_sources(),
              "runtime_at_start": runtime_manifest(ROOT), "stages": {},
              "model_training_started": False,
              "final_seed_collection_started": args.profile == "final" and args.command in ("collect", "preflight")}
    started = time.perf_counter()
    try:
        if not all(pin["matches"] for pin in report["runtime_at_start"]["packages"].values()):
            raise RuntimeError("installed runtime differs from the pinned Phase 1/2.1 stack")
        report["prior_diagnostic_reports"] = []
        for previous_path in args.prior_diagnostic_report:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            rows = previous.get("cases", [])
            report["prior_diagnostic_reports"].append({
                "path": str(previous_path.resolve()), "sha256": file_sha256(previous_path),
                "status": previous["status"], "enters_initial_training_replay": False,
                "published_validation_transitions": sum(row.get("published_validation_transitions", 0) for row in rows),
                "physical_joint_decisions_including_initialization": sum(row.get("physical_joint_decisions_from_native_journals", 0) for row in rows)})
        if args.command in ("complete", "collect", "preflight"):
            report["disk_provisioning"] = check_disk(args.data_dir.resolve(), transitions, len(cases))
            journal_parent = (args.journal_root or Path(tempfile.gettempdir())).resolve()
            journal_parent.mkdir(parents=True, exist_ok=True)
            journal_reserve = args.jobs * args.journal_spool_max_mib * 1024**2 + 1024**3
            if shutil.disk_usage(journal_parent).free < journal_reserve:
                raise RuntimeError("insufficient temporary journal space; choose another --journal-root")
            journal_directory = Path(tempfile.mkdtemp(prefix="world-models-p21-", dir=journal_parent)).resolve()
            report["journal_storage"] = {"temporary_campaign_root": str(journal_directory),
                                         "per_dataset_capacity_bytes": args.journal_spool_max_mib * 1024**2,
                                         "free_bytes_at_start": shutil.disk_usage(journal_parent).free,
                                         "images_and_verified_archives_on_data_drive": True,
                                         "policy": "Retain durable active journals in bounded temporary storage; archive closed files with verified hashes before removing their hot originals."}
        if args.command == "complete":
            report["stages"]["unit_tests"] = unit_suite(output)
            report["stages"]["journal_io_probe"] = benchmark_fsync_directories(
                output / "journal_io_probe", locations={"hot_journals": journal_directory, "cold_replay": args.data_dir.resolve()})
            preflight_directory = (args.preflight_data_dir or output / "preflight_data").resolve()
            provisional = {"compression": "deflate", "compression_level": 1, "chunk_transitions": 128}
            with cpu_budget(4) as allocation:
                preflight = collect_with_resource_report(report, "preflight_resources", allocation["cpu_ids"],
                    lambda: collect_cases(cases, preflight_directory, profile=args.profile, transitions=1000,
                                              jobs=args.jobs, storage=provisional, reuse=args.reuse,
                                              journal_root=journal_directory / "preflight", journal_spool_max_bytes=args.journal_spool_max_mib * 1024**2))
                preflight_resources = report["preflight_resources"]
            preflight_audits = [audit_initial_dataset(row["path"], expected_transitions=1000) for row in preflight]
            report["stages"]["preflight"] = {"datasets": preflight, "audits": preflight_audits,
                                                 "resources": preflight_resources,
                                                 "enters_initial_training_replay": False,
                                                 "published_diagnostic_joint_transitions": 4000}
            save_report(output / "native_preflight_report.json", report["stages"]["preflight"])
        if args.command in ("benchmark", "complete"):
            if args.command == "complete":
                benchmark = run_storage_benchmark(output / "storage_benchmark", dataset_paths=[Path(row["path"]) / "replay" for row in preflight])
            elif args.sample_datasets:
                benchmark = run_storage_benchmark(output / "storage_benchmark", dataset_paths=args.sample_datasets)
            else:
                benchmark = run_storage_benchmark(output / "storage_benchmark", sample_root=ROOT / "artifacts" / "phase1_complete" / "mechanics")
            report["stages"]["storage_benchmark"] = benchmark
            storage = benchmark["selected"]
            if benchmark["status"] != "passed":
                raise RuntimeError("storage benchmark did not meet full measurement requirements")
        elif args.benchmark_report:
            benchmark = json.loads(args.benchmark_report.read_text(encoding="utf-8"))
            if benchmark["status"] != "passed" or not benchmark["all_payloads_exact"]:
                raise ValueError("supplied storage benchmark did not pass")
            storage = benchmark["selected"]
            report["storage_benchmark_source"] = {"path": str(args.benchmark_report.resolve()), "sha256": file_sha256(args.benchmark_report)}
        else:
            storage = {"compression": "deflate", "compression_level": 1, "chunk_transitions": 128}
            report["storage_choice_note"] = "Provisional setting; complete always measures storage before collection."
        report["storage_setting"] = storage
        if args.command in ("complete", "collect", "preflight"):
            with cpu_budget(4) as allocation:
                collected = collect_with_resource_report(report, "collection_resources", allocation["cpu_ids"],
                    lambda: collect_cases(cases, args.data_dir.resolve(), profile=args.profile, transitions=transitions,
                                              jobs=args.jobs, storage=storage, reuse=args.reuse,
                                              journal_root=journal_directory / "initial", journal_spool_max_bytes=args.journal_spool_max_mib * 1024**2))
            report["stages"]["collection"] = collected
        if args.command != "benchmark":
            audits = []
            for family, seed in cases:
                directory = args.data_dir.resolve() / family / f"seed_{seed}"
                audit = audit_initial_dataset(directory, expected_transitions=transitions,
                                              require_full_coverage=transitions == configured_count)
                audits.append(audit)
                print(json.dumps({"status": "dataset_audit_passed", "task_family": family, "run_seed": seed, "transitions": audit["transition_count"]}), flush=True)
            report["stages"]["dataset_audits"] = audits
            report["physical_initial_replay_joint_transitions"] = sum(row["transition_count"] for row in audits)
            report["attributed_initial_joint_transitions_per_variant"] = {variant: sum(row["transition_count"] for row in audits) for variant in ("B", "D")}
        report["source_sha256_at_end"] = execution_sources()
        report["runtime_at_end"] = runtime_manifest(ROOT)
        if report["source_sha256_at_end"] != report["source_sha256_at_start"]:
            raise RuntimeError("execution sources changed during the campaign")
        core = lambda value: {key: item for key, item in value.items() if key != "recorded_at_utc"}
        if core(report["runtime_at_end"]) != core(report["runtime_at_start"]):
            raise RuntimeError("runtime assets changed during the campaign")
        report["status"] = "complete_p21_pilot" if args.command == "complete" else "passed_preflight" if transitions < configured_count and args.command != "benchmark" else "passed"
        report["elapsed_seconds"] = time.perf_counter() - started
        save_report(path, report)
        print(json.dumps({"status": report["status"], "report": str(path)}), flush=True)
        return 0
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        save_report(path, report)
        print(json.dumps({"status": "failed", "report": str(path), "error": repr(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
