"""Complete P1.5-P1.7 with measured native evidence, never model-training data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def execution_sources() -> dict:
    paths = set()
    for directory, patterns in (("src",("*.py",)),("scripts",("*.py",)),("tests",("*.py",)),
                                ("configs",("*.yaml","*.json")),("scenarios",("*.acs",))):
        for pattern in patterns:
            paths.update((ROOT/directory).rglob(pattern))
    paths.update(ROOT/name for name in ("requirements-phase1.lock","requirements-phase0.lock","pyproject.toml"))
    return {str(path.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def documentation_sources() -> dict:
    paths = {ROOT/"README.md",ROOT/"detailed_research_implementation_plan.md"}
    for folder in ("docs","scenarios"):
        paths.update((ROOT/folder).rglob("*.md"))
    return {str(path.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def artifact_hashes(name: str, output: Path) -> dict:
    directories = {"events":"event_probe"}
    folder = output/directories.get(name,name)
    files = set(folder.rglob("*")) if folder.is_dir() else set()
    if name == "prepare":
        files.update(output/item for item in ("privileged_registry.json","split_manifest.json"))
    return {str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files) if path.is_file()}


def verify_artifact_hashes(records: dict) -> None:
    for name, expected in records.items():
        path = Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"acceptance evidence bytes are missing or changed: {path}")


def run_stage(name: str, output: Path, parameters: dict, callback, *, reuse=False) -> dict:
    from world_models.env.acceptance import save_json
    from world_models.env.provenance import runtime_manifest
    before = execution_sources()
    runtime = runtime_manifest(ROOT)
    runtime_core = {key:value for key,value in runtime.items() if key != "recorded_at_utc"}
    if not all(pin["matches"] for pin in runtime.get("packages",{}).values()):
        raise RuntimeError("installed runtime does not match requirements-phase1.lock")
    fingerprint = digest({"sources":before,"runtime":runtime_core,"parameters":parameters})
    path = output/f"{name}_report.json"
    if reuse and path.exists():
        receipt = path.with_suffix(".sha256")
        if not receipt.is_file() or receipt.read_text(encoding="ascii").split()[0] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError("cached acceptance report bytes differ from their receipt")
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("dependency_fingerprint") != fingerprint:
            raise RuntimeError(f"{name} evidence is stale; use a new output directory for changed execution sources/runtime/parameters")
        if cached.get("status") not in ("passed","passed_preflight"):
            raise RuntimeError(f"{name} previous stage did not pass; use a new output directory")
        verify_artifact_hashes(cached.get("artifact_sha256",{}))
        print(json.dumps({"progress":"reused_validated_stage","stage":name}),flush=True)
        return cached
    started = time.perf_counter()
    print(json.dumps({"progress":"stage_started","stage":name}),flush=True)
    try:
        result = callback()
        result.update(stage=name, parameters=parameters, dependency_fingerprint=fingerprint,
                      source_sha256=before, runtime=runtime,wall_seconds=time.perf_counter()-started)
        if execution_sources() != before:
            raise RuntimeError(f"execution sources changed during {name}")
        after = runtime_manifest(ROOT)
        if {key:value for key,value in after.items() if key != "recorded_at_utc"} != runtime_core:
            raise RuntimeError(f"runtime assets changed during {name}")
        result["execution_sources_unchanged"] = True
        result["runtime_assets_unchanged"] = True
        result["artifact_sha256"] = artifact_hashes(name,output)
        save_json(path,result)
        path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+path.name+"\n",encoding="ascii")
        print(json.dumps({"progress":"stage_finished","stage":name,"status":result["status"]}),flush=True)
        return result
    except BaseException as exc:
        save_json(output/f"{name}_error.json", {"stage":name,"status":"failed","parameters":parameters,
                 "dependency_fingerprint":fingerprint,"source_sha256_at_start":before,
                 "error":repr(exc),"details":getattr(exc,"record",None),"traceback":traceback.format_exc()})
        raise


def prepare(output: Path) -> dict:
    from world_models.env.acceptance import save_json
    from world_models.env.splits import write_split_manifest
    from world_models.env.targets import registry_manifest
    registry = registry_manifest()
    if registry != json.loads((ROOT/"configs/privileged_registry_v1.json").read_text(encoding="utf-8")):
        raise RuntimeError("versioned registry artifact differs from executable schema")
    save_json(output/"privileged_registry.json",registry)
    split = write_split_manifest(output/"split_manifest.json")
    return {"status":"passed","registry":registry,"split_manifest":split}


def unit_suite(output: Path) -> dict:
    completed = subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],
                               cwd=ROOT,capture_output=True,text=True,timeout=300,check=False)
    return {"status":"passed" if completed.returncode == 0 else "failed",
            "exit_code":completed.returncode,"stdout":completed.stdout,"stderr":completed.stderr}


def replay_campaign(output: Path, mechanics: dict) -> dict:
    from world_models.env.replay_diagnostics import run_fixed_action_replay
    reports = []
    for family in ("relay","cover"):
        case = next(case for case in mechanics["cases"] if case["task_family"] == family)
        trace = json.loads(Path(case["trace_path"]).read_text(encoding="utf-8"))
        if hashlib.sha256(Path(case["trace_path"]).read_bytes()).hexdigest() != case["trace_sha256"]:
            raise RuntimeError("mechanics action trace bytes differ from its verified hash")
        scenario = case["scenario_manifest"]
        if hashlib.sha256(Path(scenario["wad_path"]).read_bytes()).hexdigest() != scenario["hashes"]["wad_sha256"]:
            raise RuntimeError("mechanics scenario bytes differ from its verified hash")
        actions = [row["actions"] for row in trace["records"]]
        for mode in ("fresh_process","coordinated_reset"):
            report = run_fixed_action_replay(family, actions, seed=case["seed"],repeats=3,
                                              scenario_manifest=case["scenario_manifest"],output_dir=output,reset_mode=mode)
            reports.append(report)
            if report["status"] != "passed":
                raise RuntimeError(f"fixed-action replay did not reproduce {family}/{mode}; inspect {report['report_path']}")
    return {"status":"passed","reports":reports,
            "all_state_events_rewards_exact":all(case["state_exact"] and case["events_rewards_exact"] for case in reports),
            "all_rgb_exact":all(case["rgb_exact"] for case in reports),
            "scope":"Exact fixed joint-action replay on the pinned development platform, both families and both restart/reset paths. This does not assert cross-platform determinism."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","mechanics","boundaries","events","stress","benchmark","replay","verify"))
    parser.add_argument("--output-dir",type=Path,default=ROOT/"artifacts/phase1_complete")
    parser.add_argument("--reuse",action="store_true",help="Reuse only completed stages with identical execution/runtime/parameter fingerprints")
    parser.add_argument("--template-limit",type=int)
    parser.add_argument("--decisions",type=int,default=10000)
    parser.add_argument("--resets",type=int,default=100)
    parser.add_argument("--benchmark-steps",type=int,default=128)
    parser.add_argument("--benchmark-repetitions",type=int,default=3)
    args = parser.parse_args()
    if args.template_limit is not None and not 1 <= args.template_limit <= 10:
        parser.error("template limit must be between one and ten per family")
    if args.command == "verify" and (args.template_limit is not None or args.decisions < 10000 or args.resets < 100
                                      or args.benchmark_steps < 128 or args.benchmark_repetitions < 3):
        parser.error("full verify requires every template, >=10000 decisions, >=100 resets and benchmarks with >=128 steps and >=3 repetitions")
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    from world_models.env.acceptance import authoritative_event_probe, boundary_campaign, mechanics_campaign, save_json, stress_campaign
    from world_models.env.acceptance_metrics import benchmark_collectors, cpu_budget
    from world_models.env.validation_data import ValidationLedger
    from world_models.env.provenance import runtime_manifest
    campaign_runtime = runtime_manifest(ROOT)
    campaign_runtime_core = {key:value for key,value in campaign_runtime.items() if key != "recorded_at_utc"}
    report = {"schema_version":"phase1-acceptance-v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),
              "scope":["P1.5","P1.6","P1.7"],"status":"running","stages":{},
              "research_comparison":["B","D"],"training_started":False,
              "runtime_at_start":campaign_runtime,
              "source_sha256_at_start":execution_sources(),"documentation_sha256_at_start":documentation_sources()}
    path = output/("phase1_acceptance.json" if args.command == "verify" else f"{args.command}_campaign.json")
    if path.exists() and not args.reuse:
        parser.error("campaign output exists; preserve it with a new directory, or use --reuse for unchanged evidence")
    save_json(path,report)
    try:
        with cpu_budget(4) as budget:
            report["cpu_budget"] = budget
            stages = ["prepare","unit_tests","mechanics","events","boundaries","replay","stress","benchmark"] if args.command == "verify" else [args.command]
            mechanics = None
            for name in stages:
                if name == "prepare":
                    parameters,callback = {}, lambda: prepare(output)
                elif name == "unit_tests":
                    parameters,callback = {}, lambda: unit_suite(output)
                elif name == "mechanics":
                    parameters = {"template_limit":args.template_limit}
                    callback = lambda: mechanics_campaign(output/"mechanics",template_limit=args.template_limit)
                elif name == "events":
                    parameters,callback = {}, lambda: authoritative_event_probe(output/"event_probe")
                elif name == "boundaries":
                    parameters,callback = {}, lambda: boundary_campaign(output/"boundaries")
                elif name == "stress":
                    parameters = {"decisions":args.decisions,"coordinated_resets":args.resets}
                    callback = lambda: stress_campaign(output/"stress",decisions=args.decisions,coordinated_resets=args.resets)
                elif name == "benchmark":
                    parameters = {"steps_per_env":args.benchmark_steps,"repetitions":args.benchmark_repetitions,"cpu_ids":budget["cpu_ids"]}
                    callback = lambda: benchmark_collectors(output/"benchmark",cpu_ids=budget["cpu_ids"],steps_per_env=args.benchmark_steps,repetitions=args.benchmark_repetitions)
                elif name == "replay":
                    if mechanics is None:
                        candidate = output/"mechanics_report.json"
                        if not candidate.exists():
                            raise ValueError("run mechanics in this output directory before replay")
                        mechanics = json.loads(candidate.read_text(encoding="utf-8"))
                        if mechanics["source_sha256"] != execution_sources():
                            raise RuntimeError("mechanics replay source is stale")
                    parameters = {"repeats":3,"mechanics_traces":[case["trace_sha256"] for case in mechanics["cases"]]}
                    callback = lambda: replay_campaign(output/"replay",mechanics)
                result = run_stage(name,output,parameters,callback,reuse=args.reuse)
                if {key:value for key,value in result["runtime"].items() if key != "recorded_at_utc"} != campaign_runtime_core:
                    raise RuntimeError("runtime assets changed between acceptance stages")
                report["stages"][name] = result
                if name == "mechanics":
                    mechanics = result
                save_json(path,report)
                if result["status"] not in ("passed","passed_preflight"):
                    raise RuntimeError(f"{name} did not pass")
            report["validation_data"] = ValidationLedger(output).manifest()
            if args.command == "verify":
                base = report["validation_data"]
                trials = report["stages"]["benchmark"]["trials"]
                replays = [case for replay in report["stages"]["replay"]["reports"] for case in replay["cases"]]
                total_consumed = (base["consumed_joint_decisions"]
                                  +sum(trial["consumed_joint_decisions"] for trial in trials)
                                  +sum(case["consumed_joint_decisions_delta"] for case in replays))
                total_initialization = (base["initialization_joint_decisions"]
                                        +sum(trial["initialization_joint_decisions"] for trial in trials)
                                        +sum(case["initialization_joint_decisions_delta"] for case in replays))
                total_published = (base["published_transitions"]
                                   +sum(trial["joint_transitions"]+trial["warmup_joint_decisions"] for trial in trials)
                                   +sum(case["decisions"] for case in replays))
                if total_consumed != total_initialization+total_published+base["failed_dispatch_attempts"]:
                    raise RuntimeError("whole-campaign diagnostic interaction counts do not reconcile")
                report["all_validation_interaction_accounting"] = {
                    "consumed_joint_decisions_including_initialization_and_failed_attempts":total_consumed,
                    "published_transitions_including_benchmark_warmup":total_published,
                    "initialization_joint_decisions":total_initialization,
                    "failed_dispatch_attempts":base["failed_dispatch_attempts"],
                    "enters_training_replay":False,"training_budget_charged_implicitly":False}
            if args.command == "verify":
                complete = all(stage["status"] == "passed" for stage in report["stages"].values())
                complete &= report["stages"]["stress"]["acceptance_threshold_met"]
                complete &= report["stages"]["mechanics"]["all_template_mechanics_passed"]
                complete &= report["stages"]["replay"]["all_state_events_rewards_exact"]
                report["status"] = "complete_phase1" if complete else "incomplete"
                report["collector_selection"] = {
                    "selected_collectors":report["stages"]["benchmark"]["selected_collectors"],
                    "source":"measured benchmark policy; common to B and D",
                    "note":"No learner overlap was benchmarked; remeasure when the training host/learner load changes."}
                from world_models.infrastructure.config import resolve_config
                configured = resolve_config("relay","B","pilot").settings["resources"]["initial_collectors"]
                report["collector_selection"]["configured_initial_collectors"] = configured
                if configured != report["collector_selection"]["selected_collectors"]:
                    raise RuntimeError("measured collector choice differs from the shared configuration; update it consistently before freezing acceptance")
                save_json(output/"collector_selection.json",report["collector_selection"])
            else:
                report["status"] = next(iter(report["stages"].values()))["status"]
            report["execution_sources_unchanged"] = report["source_sha256_at_start"] == execution_sources()
            report["runtime_at_end"] = runtime_manifest(ROOT)
            report["runtime_unchanged_between_stages"] = campaign_runtime_core == {key:value for key,value in report["runtime_at_end"].items() if key != "recorded_at_utc"}
            report["documentation_sha256_at_end"] = documentation_sources()
            if not report["execution_sources_unchanged"]:
                raise RuntimeError("execution source drift during campaign")
            if not report["runtime_unchanged_between_stages"]:
                raise RuntimeError("runtime asset drift during campaign")
            for stage in report["stages"].values():
                verify_artifact_hashes(stage.get("artifact_sha256",{}))
            save_json(path,report)
    except BaseException as exc:
        report.update(status="failed",error=repr(exc),details=getattr(exc,"record",None),traceback=traceback.format_exc())
        save_json(path,report)
        print(json.dumps({"status":"failed","report":str(path),"error":repr(exc)}),flush=True)
        return 1
    print(json.dumps({"status":report["status"],"report":str(path)}),flush=True)
    return 0 if report["status"] in ("passed","passed_preflight","complete_phase1") else 1


if __name__ == "__main__":
    raise SystemExit(main())
