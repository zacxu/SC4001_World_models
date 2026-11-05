"""Move stopped P2.4 initial fits to four GPUs without changing fitting code.

Prepare preserves training state; resume delegates to the original parallel
launcher. This separate module intentionally leaves hash-bound P2.4 files alone.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import torch

from scripts import phase24 as cli
from world_models.data.replay import canonical_hash, sha256_file
from world_models.infrastructure.reproducibility import derive_seed
from world_models.training import checkpoints, runner


SCHEMA = "p24-relocation-v1"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _allocation():
    return {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}


def _gpu_inventory(indices):
    if not torch.cuda.is_available() or max(indices) >= torch.cuda.device_count():
        raise ValueError("four allocated CUDA GPUs must be visible in this environment")
    result = []
    for index in indices:
        device = torch.device(f"cuda:{index}")
        identity = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        if identity is None:
            raise ValueError("this PyTorch build cannot identify GPU UUIDs; preserve the existing run")
        result.append({"logical_gpu": index,
                       "mask_token": cli.gpu_environment(index)["CUDA_VISIBLE_DEVICES"],
                       "uuid": str(identity), "runtime_contract": runner.runtime_contract(device)})
    return result


def _inventory(indices):
    inventory = _gpu_inventory(indices)
    if (len(inventory) != 4 or [row["logical_gpu"] for row in inventory] != indices
            or len({row["uuid"] for row in inventory}) != 4
            or len({row["mask_token"] for row in inventory}) != 4):
        raise ValueError("relocation requires four distinct allocated GPUs")
    return inventory


def _launch_arguments(case_args, output, gpus, workers_per_gpu, resume=True):
    arguments = ["launch", "--output-dir", str(Path(output).resolve()),
                 "--gpus", gpus, "--workers-per-gpu", str(workers_per_gpu),
                 "--stage", "initial"]
    for name in ("data_dir", "statistics_dir", "config_dir", "profile", "microbatch",
                 "effective_batch", "burn_in", "loss_steps", "precision", "checkpoint_every",
                 "validate_every", "validation_windows", "log_every", "updates"):
        arguments.extend(("--" + name.replace("_", "-"), str(getattr(case_args, name))))
    if resume:
        arguments.append("--resume")
    return arguments


def _stable(plan):
    plan = deepcopy(plan)
    for job in plan["jobs"]:
        job["command"] = [part for part in job["command"] if part != "--resume"]
    return plan


def _source_plan(source):
    plan = _json(source / "launch_plan.json")
    if (plan.get("schema_version") != "p24-launch-plan-v1" or plan.get("stage") != "initial"
            or len(plan.get("jobs", [])) != 4):
        raise ValueError("source must be an existing four-case initial P2.4 launch")
    first = plan["jobs"][0]["command"]
    if first[1:4] != ["-m", "scripts.phase24", "initial"]:
        raise ValueError("unrecognized original launch command")
    args = cli.parser().parse_args(first[3:])
    original_args = _launch_arguments(args, source, ",".join(map(str, plan["gpus"])),
                                      plan["workers_per_gpu"])
    regenerated = cli.launch_plan(cli.parser().parse_args(original_args))
    if _stable(regenerated) != _stable(plan):
        raise ValueError("original launch plan, environment, settings or replay provenance changed")
    return plan, args


def _source_locks(stack, source):
    for directory in (source, *(source / task / f"seed_{seed}" for task, seed in runner.CASES)):
        if not directory.is_dir():
            raise ValueError(f"source run directory is missing: {directory}")
        stack.enter_context(runner.run_lock(directory))


def _latest(case):
    # A receipt without its payload is corrupt committed evidence, not a fallback.
    for receipt in (case / "checkpoints").glob("update_*.sha256"):
        if not receipt.with_suffix(".pt").is_file():
            raise ValueError(f"committed checkpoint payload is missing: {receipt}")
    choices = runner.checkpoint_candidates(case)
    if not choices:
        raise ValueError(f"no committed checkpoint in {case}")
    return choices[0]


def _inspect_case(job, args, gpu):
    case = Path(job["output_dir"])
    path = _latest(case)
    payload = checkpoints.read_checkpoint(path)
    manifest = _json(case / "run_manifest.json")
    cfg, _, _, provenance = cli.load_case(job["task"], job["seed"], data_root=args.data_dir,
        statistics_root=args.statistics_dir, config_root=args.config_dir, profile=args.profile)
    common = dict(cfg.settings)
    common.pop("variant")
    settings = cli.settings_from_args(args, job["task"])
    metadata = payload["metadata"]
    expected = {"task_family": job["task"], "run_seed": job["seed"], "profile": "pilot",
        "stage": "initial", "variant": "shared_initial:B,D", "common_config_sha256": canonical_hash(common),
        "settings": asdict(settings), "precision": settings.precision, "device_type": "cuda",
        "device": "cuda:0", "cuda_visible_devices": metadata.get("cuda_visible_devices"),
        "target_stage_updates": job["target_stage_updates"], "source_sha256": runner.source_identity(),
        **provenance, "fitting_seed": derive_seed(job["task"], job["seed"], "world_model_fitting_initial", "B"),
        "runtime_contract": gpu["runtime_contract"],
        "normalization_policy": "frozen initial fitting statistics; no updates from diagnostics or refresh"}
    if (manifest.get("schema_version") != "p24-run-v1" or manifest.get("metadata") != metadata
            or manifest.get("resolved_config") != cfg.as_dict() or metadata != expected):
        raise ValueError(f"checkpoint/configuration/data/source/runtime mismatch: {case}")
    if (set(payload["rng"]["torch_cuda"]) != {"0"}
            or not 0 <= payload["update"] <= job["target_stage_updates"]
            or path.name != f"update_{payload['update']:08d}.pt"):
        raise ValueError(f"checkpoint is not a single-GPU initial fitting state: {path}")
    orphans = [str(item) for item in sorted((case / "checkpoints").iterdir()) if item.is_file()
               and (".tmp" in item.name or item.name.endswith(".lock")
                    or (item.suffix == ".pt" and not item.with_suffix(".sha256").is_file()))]
    return {"task": job["task"], "seed": job["seed"], "model_updates": payload["update"],
            "target_stage_updates": job["target_stage_updates"], "source_checkpoint": str(path),
            "source_checkpoint_sha256": sha256_file(path), "source_manifest": str(case / "run_manifest.json"),
            "source_manifest_sha256": sha256_file(case / "run_manifest.json"),
            "original_cuda_visible_devices": metadata["cuda_visible_devices"],
            "gpu": gpu, "ignored_uncommitted_files": orphans}


def relocate_payload(payload, mask_token):
    """Change only device-assignment metadata, retaining every saved state byte."""
    if not isinstance(mask_token, str) or not mask_token.strip() or "," in mask_token:
        raise ValueError("each worker must receive a single CUDA_VISIBLE_DEVICES token")
    checkpoints._validate_payload(payload, restore_rng=False)
    result = deepcopy(payload)
    result["metadata"]["cuda_visible_devices"] = mask_token
    return result


def write_relocated_checkpoint(path, payload):
    """Serialize the supplied RNG state; save_checkpoint would recapture it."""
    checkpoints._validate_payload(payload, restore_rng=False)
    path = Path(path)
    receipt, lock = path.with_suffix(".sha256"), path.with_name(path.name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or receipt.exists():
        raise FileExistsError("relocated checkpoints are immutable")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    suffix = ".tmp-" + uuid.uuid4().hex
    temporary = path.with_name(path.name + suffix)
    temporary_receipt = receipt.with_name(receipt.name + suffix)
    try:
        if path.exists() or receipt.exists():
            raise FileExistsError("relocated checkpoint already exists")
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        with temporary_receipt.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(f"{digest}  {path.name}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        checkpoints._fsync_directory(path.parent)
        os.replace(temporary_receipt, receipt)
        checkpoints._fsync_directory(path.parent)
    finally:
        for owned in (temporary, temporary_receipt, lock):
            owned.unlink(missing_ok=True)
    return digest


def _same_state(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same_state(a, b) for a, b in zip(left, right))
    return left == right


def prepare(source_dir, output_dir, *, gpus="0,1,2,3"):
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and destination must be separate, non-overlapping run directories")
    if output.exists():
        raise FileExistsError("choose a fresh output directory; original and partial runs are preserved")
    indices = cli.gpu_indices(gpus)
    if len(indices) != 4:
        raise ValueError("provide exactly four visible logical GPU indices")
    with ExitStack() as stack:
        _source_locks(stack, source)
        plan, args = _source_plan(source)
        inventory = _inventory(indices)
        rows = [_inspect_case(job, args, gpu) for job, gpu in zip(plan["jobs"], inventory)]
        arguments = _launch_arguments(args, output, gpus, 1)
        output.mkdir(parents=True, exist_ok=False)
        stack.enter_context(runner.run_lock(output))
        runner.write_report(output / "relocation_pending.json", {"source_dir": str(source)})
        for row in rows:
            original = Path(row["source_checkpoint"])
            payload = relocate_payload(checkpoints.read_checkpoint(original), row["gpu"]["mask_token"])
            case = output / row["task"] / f"seed_{row['seed']}"
            destination = case / "checkpoints" / original.name
            digest = write_relocated_checkpoint(destination, payload)
            if not _same_state(checkpoints.read_checkpoint(destination), payload):
                raise ValueError("serialized checkpoint state differs; destination is not ready")
            manifest = _json(row["source_manifest"])
            manifest["metadata"] = payload["metadata"]
            # Keep the original execution inventory explicitly labelled as history.
            manifest["relocation"] = {"source_manifest": row["source_manifest"],
                "source_manifest_sha256": row["source_manifest_sha256"], "gpu": row["gpu"],
                "runtime_field_describes": "original worker before relocation"}
            runner.write_report(case / "run_manifest.json", manifest)
            runner.write_report(case / "status.json", {"status": "relocated_ready",
                "model_updates": row["model_updates"], "target_stage_updates": row["target_stage_updates"]})
            row.update(checkpoint=str(destination), checkpoint_sha256=digest,
                       manifest=str(case / "run_manifest.json"),
                       manifest_sha256=sha256_file(case / "run_manifest.json"))
            del payload
        new_plan = cli.launch_plan(cli.parser().parse_args(arguments))
        runner.write_report(output / "launch_plan.json", new_plan)
        report = {"schema_version": SCHEMA, "status": "ready_to_resume", "source_dir": str(source),
            "output_dir": str(output), "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "allocation": _allocation(), "gpus": inventory, "resume_arguments": arguments, "cases": rows,
            "source_plan_sha256": sha256_file(source / "launch_plan.json"),
            "launch_plan_sha256": sha256_file(output / "launch_plan.json"),
            "helper_sha256": sha256_file(Path(__file__)), "fitting_source_sha256": runner.source_identity(),
            "optimizer_updates_executed": 0, "new_environment_interactions": 0,
            "changed_checkpoint_metadata_fields": ["cuda_visible_devices"],
            "training_state_preserved": True, "gpu_numerical_equivalence_measured": False}
        runner.write_report(output / "relocation.json", report)
        receipt = output / "relocation.sha256"
        with receipt.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(f"{sha256_file(output / 'relocation.json')}  relocation.json\n")
            stream.flush()
            os.fsync(stream.fileno())
        checkpoints._fsync_directory(output)
        (output / "relocation_pending.json").unlink()
    return report


def _require_hash(path, expected):
    if not Path(path).is_file() or sha256_file(path) != expected:
        raise ValueError(f"relocation evidence changed or is missing: {path}")


def resume(output_dir):
    output = Path(output_dir).resolve()
    report_path = output / "relocation.json"
    receipt = (output / "relocation.sha256").read_text(encoding="ascii").strip().split()
    if len(receipt) != 2 or receipt[1] != report_path.name:
        raise ValueError("relocation receipt is malformed")
    _require_hash(report_path, receipt[0])
    report = _json(report_path)
    if (report["schema_version"] != SCHEMA or report["status"] != "ready_to_resume"
            or report["output_dir"] != str(output) or report["allocation"] != _allocation()
            or report["fitting_source_sha256"] != runner.source_identity()):
        raise ValueError("relocation source/environment/allocation changed; preserve the prepared run")
    _require_hash(Path(__file__), report["helper_sha256"])
    if _inventory([gpu["logical_gpu"] for gpu in report["gpus"]]) != report["gpus"]:
        raise ValueError("GPU identity, mask or software runtime changed since preparation")
    _require_hash(output / "launch_plan.json", report["launch_plan_sha256"])
    source = Path(report["source_dir"])
    # Keep the old campaign stopped for the whole continuation, including retries.
    with ExitStack() as stack:
        _source_locks(stack, source)
        _require_hash(source / "launch_plan.json", report["source_plan_sha256"])
        for row in report["cases"]:
            for path, digest in ((row["source_checkpoint"], row["source_checkpoint_sha256"]),
                                 (row["source_manifest"], row["source_manifest_sha256"]),
                                 (row["checkpoint"], row["checkpoint_sha256"]),
                                 (row["manifest"], row["manifest_sha256"])):
                _require_hash(path, digest)
            if _latest(Path(row["source_manifest"]).parent) != Path(row["source_checkpoint"]):
                raise ValueError("original run advanced after relocation; preserve both histories")
            checkpoints.read_checkpoint(row["checkpoint"])
        # Existing launcher revalidates replay/config/plan, locks the new root,
        # and starts all four subprocesses before waiting for any of them.
        return cli.main(report["resume_arguments"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("prepare", help="copy saved state to a fresh four-GPU run; no training")
    item.add_argument("--source-dir", type=Path, required=True)
    item.add_argument("--output-dir", type=Path, required=True)
    item.add_argument("--gpus", default="0,1,2,3", help="four indices within CUDA_VISIBLE_DEVICES")
    item = commands.add_parser("resume", help="resume the prepared run with one parallel worker per GPU")
    item.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "resume":
        return resume(args.output_dir)
    report = prepare(args.source_dir, args.output_dir, gpus=args.gpus)
    for row in report["cases"]:
        print(f"{row['task']}/seed_{row['seed']}: {row['model_updates']} / {row['target_stage_updates']}"
              f" optimizer updates; GPU {row['gpu']['mask_token']}")
    print(f"Ready to resume: {report['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
