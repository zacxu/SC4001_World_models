"""Import stopped P2.4 fits onto another CUDA host without resetting training.

The explicit import records allowed runtime changes; normal resume remains strict.
Preparation never performs optimizer updates or restores process-global RNG state.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import signal

import torch

from scripts import phase24 as cli
from scripts import phase24_relocate as relocate
from world_models.data.replay import canonical_hash, sha256_file
from world_models.infrastructure.reproducibility import derive_seed
from world_models.training import checkpoints, runner


SCHEMA = "p24-cross-host-import-v1"
_json = relocate._json
_allocation = relocate._allocation
_source_locks = relocate._source_locks
_latest = relocate._latest
_require_hash = relocate._require_hash


def _gpu_inventory(indices):
    if not torch.cuda.is_available() or max(indices) >= torch.cuda.device_count():
        raise ValueError("selected allocated CUDA GPUs must be visible in this environment")
    return relocate._gpu_inventory(indices)


def _inventory(indices):
    """Inventory physical devices once, independently of workers sharing them."""
    inventory = _gpu_inventory(indices)
    if (len(inventory) != len(indices) or [row["logical_gpu"] for row in inventory] != indices
            or len({row["uuid"] for row in inventory}) != len(indices)
            or len({row["mask_token"] for row in inventory}) != len(indices)):
        raise ValueError("import requires distinct allocated GPUs for the selected indices")
    return inventory


def _runtime_compatible(original, current):
    """Allow recorded hardware/CUDA changes, retaining state-format versions."""
    keys = {"python", "torch", "numpy", "cuda_runtime", "cudnn", "gpu_name", "gpu_capability"}
    if not isinstance(original, dict) or not isinstance(current, dict) or set(original) != keys or set(current) != keys:
        raise ValueError("original/imported runtime contract is incomplete")
    def python_minor(value):
        match = re.fullmatch(r"(\d+\.\d+)\.\d+", str(value))
        if not match:
            raise ValueError("unsupported Python version in checkpoint runtime")
        return match.group(1)

    def torch_base(value):
        match = re.fullmatch(r"(2\.\d+\.\d+)(?:\+[^\s]+)?", str(value))
        if not match:
            raise ValueError("unsupported PyTorch version in checkpoint runtime")
        return match.group(1)

    if (set(original) != set(current)
            or python_minor(original["python"]) != python_minor(current["python"])
            or original["numpy"] != current["numpy"]
            or torch_base(original["torch"]) != torch_base(current["torch"])):
        raise ValueError("import requires the same Python major/minor, NumPy and PyTorch base version")
    return {key: {"original": original[key], "imported": current[key]}
            for key in original if original[key] != current[key]}


def _validate_cuda_state(payload, logical_gpu):
    """Test CUDA RNG compatibility using isolated generators, never global RNG."""
    if set(payload["rng"]["torch_cuda"]) != {"0"}:
        raise ValueError("import requires a checkpoint from a single-GPU worker")
    states = {"global CUDA": payload["rng"]["torch_cuda"]["0"], **payload["streams"]}
    for name, state in states.items():
        try:
            generator = torch.Generator(device=f"cuda:{logical_gpu}")
            generator.set_state(state)
            if not torch.equal(generator.get_state().cpu(), state):
                raise ValueError(f"CUDA RNG state changed when restoring {name}")
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"saved {name} RNG state is incompatible with the allocated CUDA runtime") from error


def _require_checkpoint(path, expected):
    path = Path(path)
    _require_hash(path, expected)
    receipt = path.with_suffix(".sha256").read_text(encoding="ascii").strip().split()
    if receipt != [expected, path.name] or path.with_name(path.name + ".lock").exists():
        raise ValueError(f"checkpoint commit receipt/lock changed: {path}")


def _parse_original(command):
    if (not isinstance(command, list) or not all(isinstance(part, str) for part in command)
            or command[1:4] != ["-m", "scripts.phase24", "initial"]):
        raise ValueError("unrecognized original P2.4 initial-fit command")
    try:
        return cli.parser().parse_args(command[3:])
    except SystemExit as error:
        raise ValueError("original launch command is invalid; preserve the original plan") from error


def _source_plan(source):
    plan = _json(source / "launch_plan.json")
    if (plan.get("schema_version") != "p24-launch-plan-v1" or plan.get("stage") != "initial"
            or plan.get("worker_count") != 4 or len(plan.get("jobs", [])) != 4
            or type(plan.get("workers_per_gpu")) is not int or plan["workers_per_gpu"] < 1):
        raise ValueError("source must contain an original four-case initial launch plan")
    devices = plan.get("gpus")
    if (not isinstance(devices, list) or not devices
            or any(type(index) is not int or index < 0 for index in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError("original launch GPU layout is invalid")
    slots = devices * plan["workers_per_gpu"]
    if len(slots) < 4:
        raise ValueError("original launch has fewer than four worker slots")
    arguments = []
    for (task, seed), gpu, job in zip(runner.CASES, slots, plan["jobs"]):
        args = _parse_original(job.get("command"))
        suffix = str(job.get("output_dir", "")).replace("\\", "/").rstrip("/").split("/")[-2:]
        if (job.get("task") != task or job.get("seed") != seed or job.get("logical_gpu") != gpu
                or args.task != task or args.seed != seed or args.profile != "pilot" or args.device != "cuda:0"
                or str(args.output_dir).replace("\\", "/") != str(job.get("output_dir")).replace("\\", "/")
                or suffix != [task, f"seed_{seed}"]
                or type(job.get("target_stage_updates")) is not int or job["target_stage_updates"] < 1
                or args.updates != job["target_stage_updates"]):
            raise ValueError("original launch cases, options or update targets are inconsistent")
        if arguments and any(getattr(args, key) != getattr(arguments[0], key)
                             for key in ("data_dir", "statistics_dir", "config_dir", "profile", "updates")):
            raise ValueError("original launch commands disagree on shared inputs or target")
        arguments.append(args)
    return plan, arguments


def _inspect_case(source, job, args, gpu, plan_settings):
    task, seed = job["task"], job["seed"]
    case = source / task / f"seed_{seed}"
    path = _latest(case)
    payload = checkpoints.read_checkpoint(path)
    manifest = _json(case / "run_manifest.json")
    cfg, _, _, provenance = cli.load_case(task, seed, data_root=args.data_dir,
        statistics_root=args.statistics_dir, config_root=args.config_dir, profile="pilot")
    common = deepcopy(cfg.settings)
    common.pop("variant")
    settings = cli.settings_from_args(args, task)
    metadata = payload["metadata"]
    expected = {"task_family": task, "run_seed": seed, "profile": "pilot", "stage": "initial",
        "variant": "shared_initial:B,D", "common_config_sha256": canonical_hash(common),
        "settings": asdict(settings), "precision": settings.precision, "device_type": "cuda",
        "device": "cuda:0", "cuda_visible_devices": metadata.get("cuda_visible_devices"),
        "target_stage_updates": job["target_stage_updates"], "source_sha256": runner.source_identity(),
        **provenance, "fitting_seed": derive_seed(task, seed, "world_model_fitting_initial", "B"),
        "runtime_contract": metadata.get("runtime_contract"),
        "normalization_policy": "frozen initial fitting statistics; no updates from diagnostics or refresh"}
    if (manifest.get("schema_version") != "p24-run-v1" or manifest.get("metadata") != metadata
            or manifest.get("resolved_config") != cfg.as_dict() or metadata != expected
            or asdict(settings) != plan_settings or job.get("data") != provenance):
        raise ValueError(f"original checkpoint/configuration/data/source/settings mismatch: {case}")
    if (not 0 <= payload["update"] <= job["target_stage_updates"]
            or path.name != f"update_{payload['update']:08d}.pt"):
        raise ValueError(f"checkpoint update does not match its committed filename/target: {path}")
    changes = _runtime_compatible(metadata["runtime_contract"], gpu["runtime_contract"])
    if settings.precision == "bfloat16" and gpu["runtime_contract"]["gpu_capability"][0] < 8:
        raise ValueError("the allocated GPU must support bfloat16 fitting")
    _validate_cuda_state(payload, gpu["logical_gpu"])
    ignored = [str(item) for item in sorted((case / "checkpoints").iterdir()) if item.is_file()
               and (".tmp" in item.name or item.name.endswith(".lock")
                    or (item.suffix == ".pt" and not item.with_suffix(".sha256").is_file()))]
    return {"task": task, "seed": seed, "model_updates": payload["update"],
        "target_stage_updates": job["target_stage_updates"], "source_checkpoint": str(path),
        "source_checkpoint_sha256": sha256_file(path), "source_manifest": str(case / "run_manifest.json"),
        "source_manifest_sha256": sha256_file(case / "run_manifest.json"),
        "original_cuda_visible_devices": metadata["cuda_visible_devices"],
        "gpu": gpu, "runtime_changes": changes, "ignored_uncommitted_files": ignored}


def prepare(source_dir, output_dir, *, gpus="0,1,2,3", workers_per_gpu=1,
            data_dir=None, statistics_dir=None, config_dir=None):
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and destination must be separate, non-overlapping run directories")
    if output.exists():
        raise FileExistsError("choose a fresh output directory; original and partial runs are preserved")
    indices = cli.gpu_indices(gpus)
    if len(indices) > len(runner.CASES):
        raise ValueError("select at most four allocated visible logical GPU indices")
    if type(workers_per_gpu) is not int or workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be a positive integer")
    if len(indices) * workers_per_gpu < len(runner.CASES):
        raise ValueError("four initial fits need at least four parallel worker slots")
    with ExitStack() as stack:
        _source_locks(stack, source)
        plan, original_args = _source_plan(source)
        inventory = _inventory(indices)
        slots = inventory * workers_per_gpu
        for args in original_args:
            args.data_dir = Path(data_dir or "data/initial_replay/pilot").resolve()
            args.statistics_dir = Path(statistics_dir or "artifacts/phase23/normalization").resolve()
            args.config_dir = Path(config_dir or runner.ROOT / "configs").resolve()
        rows = [_inspect_case(source, job, args, gpu, plan["settings"])
                for job, args, gpu in zip(plan["jobs"], original_args, slots)]
        arguments = relocate._launch_arguments(original_args[0], output, gpus, workers_per_gpu)
        output.mkdir(parents=True, exist_ok=False)
        stack.enter_context(runner.run_lock(output))
        runner.write_report(output / "import_pending.json", {"source_dir": str(source)})
        for row in rows:
            original = Path(row["source_checkpoint"])
            _require_hash(original, row["source_checkpoint_sha256"])
            payload = relocate.relocate_payload(checkpoints.read_checkpoint(original), row["gpu"]["mask_token"])
            payload["metadata"]["runtime_contract"] = deepcopy(row["gpu"]["runtime_contract"])
            case = output / row["task"] / f"seed_{row['seed']}"
            destination = case / "checkpoints" / original.name
            digest = relocate.write_relocated_checkpoint(destination, payload)
            if not relocate._same_state(checkpoints.read_checkpoint(destination), payload):
                raise ValueError("serialized checkpoint state differs; import is not ready")
            manifest = _json(row["source_manifest"])
            manifest["metadata"] = payload["metadata"]
            manifest["cross_host_import"] = {"source_manifest": row["source_manifest"],
                "source_manifest_sha256": row["source_manifest_sha256"], "gpu": row["gpu"],
                "runtime_field_describes": "original worker before cross-host import",
                "runtime_changes": row["runtime_changes"], "gpu_numerical_equivalence_measured": False}
            runner.write_report(case / "run_manifest.json", manifest)
            runner.write_report(case / "status.json", {"status": "imported_ready",
                "model_updates": row["model_updates"], "target_stage_updates": row["target_stage_updates"]})
            row.update(checkpoint=str(destination), checkpoint_sha256=digest,
                manifest=str(case / "run_manifest.json"), manifest_sha256=sha256_file(case / "run_manifest.json"))
            del payload
        new_plan = cli.launch_plan(cli.parser().parse_args(arguments))
        runner.write_report(output / "launch_plan.json", new_plan)
        report = {"schema_version": SCHEMA, "status": "ready_to_resume", "source_dir": str(source),
            "output_dir": str(output), "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "allocation": _allocation(), "gpus": inventory, "workers_per_gpu": workers_per_gpu,
            "resume_arguments": arguments, "cases": rows,
            "source_plan_sha256": sha256_file(source / "launch_plan.json"),
            "launch_plan_sha256": sha256_file(output / "launch_plan.json"),
            "helper_sha256": sha256_file(Path(__file__)), "relocation_helper_sha256": sha256_file(Path(relocate.__file__)),
            "fitting_source_sha256": runner.source_identity(), "optimizer_updates_executed": 0,
            "new_environment_interactions": 0, "changed_checkpoint_metadata_fields": ["cuda_visible_devices", "runtime_contract"],
            "training_state_preserved": True, "gpu_numerical_equivalence_measured": False,
            "gpu_uuid_policy": "recorded for audit; a later job may allocate different physical GPUs of the same model/runtime"}
        runner.write_report(output / "import.json", report)
        with (output / "import.sha256").open("x", encoding="ascii", newline="\n") as stream:
            stream.write(f"{sha256_file(output / 'import.json')}  import.json\n")
            stream.flush()
            os.fsync(stream.fileno())
        checkpoints._fsync_directory(output)
        (output / "import_pending.json").unlink()
    return report


def resume(output_dir):
    output = Path(output_dir).resolve()
    if (output / "import_pending.json").exists():
        raise ValueError("import preparation did not finish; preserve it and choose a fresh destination")
    report_path = output / "import.json"
    receipt = (output / "import.sha256").read_text(encoding="ascii").strip().split()
    if len(receipt) != 2 or receipt[1] != report_path.name:
        raise ValueError("import receipt is malformed")
    _require_hash(report_path, receipt[0])
    report = _json(report_path)
    if (report["schema_version"] != SCHEMA or report["status"] != "ready_to_resume"
            or report["output_dir"] != str(output) or report["allocation"] != _allocation()
            or report["fitting_source_sha256"] != runner.source_identity()):
        raise ValueError("import source/environment/allocation changed; preserve the prepared run")
    _require_hash(Path(__file__), report["helper_sha256"])
    _require_hash(Path(relocate.__file__), report["relocation_helper_sha256"])
    inventory = _inventory([gpu["logical_gpu"] for gpu in report["gpus"]])
    def without_uuid(gpus):
        return [{key: value for key, value in gpu.items() if key != "uuid"} for gpu in gpus]
    if without_uuid(inventory) != without_uuid(report["gpus"]):
        raise ValueError("GPU model, assigned mask or software runtime changed since import")
    _require_hash(output / "launch_plan.json", report["launch_plan_sha256"])
    source = Path(report["source_dir"])
    with ExitStack() as stack:
        _source_locks(stack, source)
        _require_hash(source / "launch_plan.json", report["source_plan_sha256"])
        with runner.run_lock(output):
            for row in report["cases"]:
                for path, digest in ((row["source_manifest"], row["source_manifest_sha256"]),
                                     (row["manifest"], row["manifest_sha256"])):
                    _require_hash(path, digest)
                _require_checkpoint(row["source_checkpoint"], row["source_checkpoint_sha256"])
                _require_checkpoint(row["checkpoint"], row["checkpoint_sha256"])
                if _latest(Path(row["source_manifest"]).parent) != Path(row["source_checkpoint"]):
                    raise ValueError("synchronized original run advanced after import; preserve both histories")
                payload = checkpoints.read_checkpoint(_latest(Path(row["manifest"]).parent))
                if payload["metadata"] != _json(row["manifest"])["metadata"]:
                    raise ValueError("latest continuation checkpoint differs from its imported manifest")
                _validate_cuda_state(payload, row["gpu"]["logical_gpu"])
        return cli.main(report["resume_arguments"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("prepare", help="import all four stopped initial fits; no training")
    item.add_argument("--source-dir", type=Path, required=True)
    item.add_argument("--output-dir", type=Path, required=True)
    item.add_argument("--gpus", default="0,1,2,3", help="distinct indices within CUDA_VISIBLE_DEVICES")
    item.add_argument("--workers-per-gpu", type=int, default=1,
                      help="parallel workers per selected GPU; four worker slots are required")
    item.add_argument("--data-dir", type=Path)
    item.add_argument("--statistics-dir", type=Path)
    item.add_argument("--config-dir", type=Path)
    item = commands.add_parser("resume", help="resume all four imported fits in parallel")
    item.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}; resume from the latest committed checkpoint")
    previous = {name: signal.getsignal(name) for name in (signal.SIGINT, signal.SIGTERM)}
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.command == "resume":
            return resume(args.output_dir)
        report = prepare(args.source_dir, args.output_dir, gpus=args.gpus,
                         workers_per_gpu=args.workers_per_gpu, data_dir=args.data_dir,
                         statistics_dir=args.statistics_dir, config_dir=args.config_dir)
        for row in report["cases"]:
            print(f"{row['task']}/seed_{row['seed']}: {row['model_updates']} / {row['target_stage_updates']}"
                  f" optimizer updates; GPU {row['gpu']['mask_token']}")
        print(f"Import report: {Path(report['output_dir']) / 'import.json'}")
        print(f"Ready to resume: {report['output_dir']}; GPU numerical equivalence is unmeasured.")
        return 0
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


if __name__ == "__main__":
    raise SystemExit(main())
