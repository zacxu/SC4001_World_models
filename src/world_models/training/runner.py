"""P2.4 single-GPU stage orchestration and auditable fitting checkpoints."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from world_models.data.replay import ReplayDataset, canonical_hash, sha256_file
from world_models.infrastructure.config import resolve_config, validate_pair
from world_models.infrastructure.reproducibility import derive_seed
from world_models.models.world_model import SamplingStreams, build_model
from .checkpoints import load_checkpoint, read_checkpoint, save_checkpoint
from .fitting import FitSettings, compare_burn_in, evaluate_windows, optimizer_update
from .normalization import NormalizationStats
from .objective import ObjectiveConfig, WorldModelObjective
from .sampling import ReplaySampler


ROOT = Path(__file__).resolve().parents[3]
CASES = (("relay", 0), ("relay", 1), ("cover", 0), ("cover", 1))


def write_report(path, value):
    """Atomic mutable run-status publication (checkpoints themselves are immutable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_event(path, row):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def run_lock(directory):
    """OS advisory lock releases on process exit; no stale PID lock deletion."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / ".worker.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if stream.read(1) == b"":
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError("another worker owns this run directory") from exc
    try:
        yield
    finally:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def source_identity(root=ROOT):
    fitting_modules = ("checkpoints", "fitting", "runner", "sampling", "validation", "objective", "objective_batch",
                       "losses", "normalization", "diagnostics")
    paths = sorted([*Path(root, "src/world_models/models").glob("*.py"),
                    *(Path(root, f"src/world_models/training/{name}.py") for name in fitting_modules),
                    Path(root, "scripts/phase24.py"), Path(root, "src/world_models/data/replay.py"),
                    Path(root, "src/world_models/data/schema.py"), Path(root, "src/world_models/infrastructure/config.py"),
                    Path(root, "src/world_models/infrastructure/reproducibility.py")])
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def load_case(task, seed, *, data_root, statistics_root, config_root=None, profile="pilot", additional_replays=()):
    config_root = Path(config_root or ROOT / "configs")
    left = resolve_config(task, "B", profile, config_root)
    right = resolve_config(task, "D", profile, config_root)
    validate_pair(left, right)
    if seed not in left.settings["experiment"]["master_seeds"]:
        raise ValueError("seed is outside the selected experimental profile")
    case = Path(data_root) / task / f"seed_{seed}"
    initial = ReplayDataset(case / "replay", verify=False)
    metadata = initial.manifest["metadata"]
    if metadata["task_family"] != task or metadata["run_seed"] != seed or not metadata["learning_eligible"]:
        raise ValueError("initial replay task, seed or eligibility mismatch")
    if initial.manifest["transition_count"] != left.settings["experiment"]["initial_transitions"]:
        raise ValueError("initial replay does not contain the configured interaction budget")
    for variant in ("B", "D"):
        bound = ReplayDataset.from_binding(case / "bindings" / f"{variant}.json", variant=variant, verify=False)
        if bound.path != initial.path or bound.manifest_sha256 != initial.manifest_sha256:
            raise ValueError("B/D initial bindings are not paired")
    normalization_path = Path(statistics_root) / task / f"seed_{seed}.json"
    stats = NormalizationStats.load(normalization_path)
    stats.validate_for_dataset(initial)
    datasets = [initial, *(ReplayDataset(path, verify=False) for path in additional_replays)]
    for data in datasets[1:]:
        meta = data.manifest["metadata"]
        if (meta["task_family"], meta["run_seed"], meta["learning_eligible"]) != (task, seed, True):
            raise ValueError("refresh replay is not eligible real data for this task and seed")
        if data.manifest["schema"] != initial.manifest["schema"]:
            raise ValueError("refresh replay schema differs from frozen normalization source")
    provenance = {"initial_normalization": {"identity_sha256": stats.as_dict()["identity_sha256"],
                    "file_sha256": sha256_file(normalization_path)},
                  "replays": [{"manifest_sha256": data.manifest_sha256,
                               "dataset_identity_sha256": data.manifest["dataset_identity_sha256"],
                               "transitions": data.manifest["transition_count"]} for data in datasets]}
    return left, datasets, stats, provenance


def selected_device(device, precision):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("P2.4 supports a single CPU or CUDA device per worker")
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise RuntimeError("requested CUDA GPU is unavailable in this allocation")
        torch.cuda.set_device(device)
        if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("selected GPU does not support bfloat16")
    return device


def runtime_inventory(device):
    return {"python": __import__('platform').python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "device": str(device), "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_runtime": torch.version.cuda, "cpu_threads": torch.get_num_threads()}


def runtime_contract(device):
    return {"python": __import__('platform').python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None}


def checkpoint_candidates(output):
    candidates = sorted(Path(output).glob("checkpoints/update_*.pt"), reverse=True)
    return [path for path in candidates if path.with_suffix(".sha256").is_file()]


def quarantine_uncommitted_checkpoints(output):
    """Preserve crash debris under the exclusive run lock, without trusting it."""
    directory = Path(output) / "checkpoints"
    if not directory.exists():
        return []
    debris = []
    for path in directory.iterdir():
        if not path.is_file() or not path.name.startswith("update_"):
            continue
        if ".tmp-" in path.name or path.name.endswith(".lock"):
            debris.append(path)
        elif path.suffix == ".pt" and not path.with_suffix(".sha256").is_file():
            debris.append(path)
        elif path.suffix == ".sha256" and not path.with_suffix(".pt").is_file():
            debris.append(path)
    moved = []
    for path in debris:
        destination = directory / "uncommitted" / (uuid.uuid4().hex + "_" + path.name)
        destination.parent.mkdir(exist_ok=True)
        os.replace(path, destination)
        moved.append({"original": path.name, "preserved": destination.relative_to(directory).as_posix()})
    return moved


def fixed_diagnostic_windows(datasets, seed, settings):
    sampler = ReplaySampler(datasets, seed=seed, burn_in=settings.burn_in,
                            loss_steps=settings.loss_steps, partition="diagnostic")
    return sampler.sample(settings.validation_windows)


def _event_fixture(sampler, batch_size):
    """Tiny fit-only event-containing batch; never a policy imitation buffer."""
    windows = []
    found = set()
    required = {"latch_activations"} if sampler.task_family == "relay" else {
        "console_1_completions", "console_2_completions"}
    if batch_size < len(required)+1:
        raise ValueError("tiny overfit batch needs each task milestone plus an ordinary window")
    for dataset_index, dataset in enumerate(sampler.datasets):
        events = dataset.manifest["schema"]["event_fields"]
        for episode in dataset.episodes:
            if episode["partition"] != "fit":
                continue
            view = dataset.load_episode(episode["episode_id"], purpose="world_model_fit")
            for chunk, arrays in view.chunks():
                for name in sorted(required-found):
                    column = events.index(name)
                    positive = np.flatnonzero((arrays["event_deltas"][:, column] > 0) &
                                              arrays["event_delta_validity_mask"][:, column])
                    if positive.size:
                        anchor = int(chunk["transition_start"] + positive[0])
                        try:
                            windows.append(sampler.window(dataset_index, episode["episode_id"], anchor))
                        except ValueError:
                            continue
                        found.add(name)
                if found == required:
                    break
            if found == required:
                break
        if found == required:
            break
    if found != required:
        raise ValueError(f"tiny overfit fixture lacks positive task milestone labels: {sorted(required-found)}")
    ordinary = sampler.sample(max(1, batch_size-len(windows)))
    values = windows + ordinary
    return [values[index % len(values)] for index in range(batch_size)]


def fit_case(task, seed, output, *, data_root, statistics_root, settings=None, device="cuda:0",
             profile="pilot", config_root=None, updates=None, resume=False, stage="initial",
             variant=None, parent_checkpoint=None, additional_replays=(), fixed_batch=False):
    """Execute one stage on one assigned GPU. Called only by explicit user CLI."""
    if stage not in ("initial", "refresh", "overfit"):
        raise ValueError("stage must be initial, refresh, or overfit")
    if stage == "refresh" and (variant not in ("B", "D") or not parent_checkpoint or not additional_replays):
        raise ValueError("refresh needs a B/D branch, parent checkpoint and ordered branch replay shards")
    if stage != "refresh" and (variant is not None or parent_checkpoint or additional_replays):
        raise ValueError("initial/overfit uses shared replay and paired initialization only")
    cfg, datasets, stats, provenance = load_case(task, seed, data_root=data_root, statistics_root=statistics_root,
        config_root=config_root, profile=profile, additional_replays=additional_replays)
    settings = settings or FitSettings.from_experiment(cfg.settings)
    target = updates if updates is not None else (50 if stage == "overfit" else
        cfg.settings["world_model"]["online_updates" if stage == "refresh" else "initial_updates"])
    if type(target) is not int or target < 1:
        raise ValueError("updates must be positive")
    device = selected_device(device, settings.precision)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    # Exact-resume reference, including single-GPU kernels. Unsupported kernels
    # fail explicitly rather than silently changing reproducibility semantics.
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
    branch = variant or "B"
    fit_seed = derive_seed(task, seed, "world_model_fitting_online" if stage == "refresh" else
                           "world_model_fitting_initial", branch)
    if parent_checkpoint:
        # Each branch refresh has a reproducible independent stream rather than
        # restarting the same online stream at every collection block.
        fit_seed = int(canonical_hash({"base_seed": fit_seed, "parent_checkpoint_sha256":
                                      sha256_file(parent_checkpoint)})[:8], 16)
    random.seed(fit_seed)
    np.random.seed(fit_seed)
    torch.random.default_generator.manual_seed(fit_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(fit_seed)
    sampler = ReplaySampler(datasets, seed=fit_seed, burn_in=settings.burn_in, loss_steps=settings.loss_steps,
        recent_fraction=cfg.settings["world_model"]["recent_fraction"] if stage == "refresh" else 0.,
        recent_transitions=cfg.settings["world_model"]["recent_transitions"])
    model = build_model(cfg.settings, master_seed=seed, variant=branch).to(device)
    objective = WorldModelObjective(model.task, stats, ObjectiveConfig.from_experiment(cfg.settings))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay,
                                 foreach=False)
    streams = SamplingStreams.from_seed(fit_seed, device)
    common_cfg = dict(cfg.settings)
    common_cfg.pop("variant")
    metadata = {"task_family": task, "run_seed": seed, "profile": profile, "stage": stage,
                "variant": variant or "shared_initial:B,D", "common_config_sha256": canonical_hash(common_cfg),
                "settings": asdict(settings), "precision": settings.precision, "device_type": device.type,
                "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "target_stage_updates": target, "source_sha256": source_identity(), **provenance,
                "fitting_seed": fit_seed,
                "runtime_contract": runtime_contract(device),
                "normalization_policy": "frozen initial fitting statistics; no updates from diagnostics or refresh"}
    parent_update = 0
    if parent_checkpoint:
        parent = load_checkpoint(parent_checkpoint, model=model, optimizer=optimizer, restore_rng=False,
            expected_metadata={"task_family": task, "run_seed": seed, "common_config_sha256": metadata["common_config_sha256"],
                               "initial_normalization": metadata["initial_normalization"],
                               "source_sha256": metadata["source_sha256"], "precision": settings.precision})
        if parent["metadata"]["variant"] not in ("shared_initial:B,D", variant):
            raise ValueError("refresh checkpoint belongs to another branch")
        if parent["metadata"]["stage"] == "overfit" or parent["update"] != (
                parent["metadata"].get("parent_model_updates", 0) + parent["metadata"]["target_stage_updates"]):
            raise ValueError("refresh requires a completed initial/refresh checkpoint")
        parent_replays = parent["metadata"]["replays"]
        if metadata["replays"][:len(parent_replays)] != parent_replays or len(metadata["replays"]) <= len(parent_replays):
            raise ValueError("refresh must extend the parent ordered replay cohort")
        added = sum(row["transitions"] for row in metadata["replays"][len(parent_replays):])
        if added != cfg.settings["experiment"]["collection_block"]:
            raise ValueError("one model refresh requires exactly one counted collection block of new replay")
        for shard in datasets[len(parent_replays):]:
            if shard.manifest["metadata"].get("training_variant") != variant:
                raise ValueError("new online shards must explicitly record matching training_variant B or D")
        parent_update = parent["update"]
        metadata["parent_checkpoint_sha256"] = sha256_file(parent_checkpoint)
        metadata["parent_model_updates"] = parent_update
    output = Path(output).resolve()
    if output.exists() and not resume:
        raise FileExistsError("use a fresh run directory or explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        manifest_path = output / "run_manifest.json"
        if resume and not manifest_path.is_file():
            raise ValueError("resume requires an existing run manifest")
        update, resumed_from = parent_update, None
        if resume:
            if json.loads(manifest_path.read_text(encoding="utf-8"))["metadata"] != metadata:
                raise ValueError("run configuration/provenance changed; refusing resume")
            quarantined = quarantine_uncommitted_checkpoints(output)
            if quarantined:
                append_event(output / "events.jsonl", {"event": "preserved_uncommitted_checkpoint_artifacts", "files": quarantined})
            choices = checkpoint_candidates(output)
            if not choices:
                raise ValueError("no committed checkpoint available; preserve failed run and use a fresh output")
            resumed_from = choices[0]
            state = load_checkpoint(resumed_from, model=model, optimizer=optimizer, streams=streams,
                                    sampler=sampler, expected_metadata=metadata)
            update = state["update"]
        else:
            write_report(manifest_path, {"schema_version": "p24-run-v1", "metadata": metadata,
                "resolved_config": cfg.as_dict(), "runtime": runtime_inventory(device),
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "training_budget_interactions_added": 0})
            save_checkpoint(output / "checkpoints" / f"update_{update:08d}.pt", model=model, optimizer=optimizer,
                            streams=streams, sampler=sampler, update=update, metadata=metadata)
        # A deterministic diagnostic sample list is reconstructed on resume and
        # uses neither optimizer state nor the fitting/sampling RNG streams.
        held_out = fixed_diagnostic_windows(datasets, fit_seed+1, settings)
        fixed = None
        # Fixed fixture construction never consumes the fitting sampler.
        if fixed_batch or stage == "overfit":
            fixed_sampler = ReplaySampler(datasets, seed=fit_seed+2, burn_in=settings.burn_in, loss_steps=settings.loss_steps)
            fixed = _event_fixture(fixed_sampler, settings.effective_batch)
            append_event(output / "events.jsonl", {"event": "fixed_overfit_fixture", "windows": [
                {"episode_id": w.episode_id, "start": w.start, "anchor": w.anchor, "stop": w.stop} for w in fixed]})
        started = time.monotonic()
        invocation = str(time.time_ns())
        append_event(output / "events.jsonl", {"event": "resume" if resume else "start", "invocation": invocation,
                     "model_updates": update, "checkpoint": str(resumed_from) if resumed_from else None})
        last_checkpoint = update
        status = {"schema_version": "p24-status-v1", "status": "running", "stage": stage,
                  "model_updates": update, "stage_updates": update-parent_update, "target_stage_updates": target,
                  "task_family": task, "run_seed": seed, "gpu_training_measured": device.type == "cuda"}
        try:
            while update < parent_update+target:
                windows = fixed if fixed is not None else sampler.sample(settings.effective_batch)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                before = time.monotonic()
                metrics = optimizer_update(model, optimizer, windows, streams, objective, settings)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.monotonic()-before
                update += 1
                status.update(model_updates=update, stage_updates=update-parent_update,
                              last_update_seconds=elapsed, last_total_loss=metrics["total"])
                if update % settings.log_every == 0 or update == parent_update+1:
                    append_event(output / "metrics.jsonl", {"event": "fit", "invocation": invocation,
                        "model_updates": update, "stage_updates": update-parent_update, "seconds": elapsed,
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                        "real_joint_transitions_collected": 0, **metrics})
                    print(json.dumps({"task": task, "seed": seed, "update": update, "loss": metrics["total"],
                                      "seconds": elapsed}), flush=True)
                final = update == parent_update+target
                if update % settings.validate_every == 0 or final:
                    validation = evaluate_windows(model, held_out, objective, seed=fit_seed+3,
                                                   microbatch=settings.microbatch, precision=settings.precision)
                    append_event(output / "metrics.jsonl", {"event": "model_diagnostic", "invocation": invocation,
                        "model_updates": update, "partition": "diagnostic", "geometry_split": "train", **validation})
                if update % settings.checkpoint_every == 0 or final:
                    save_checkpoint(output / "checkpoints" / f"update_{update:08d}.pt", model=model,
                        optimizer=optimizer, streams=streams, sampler=sampler, update=update, metadata=metadata)
                    last_checkpoint = update
                write_report(output / "status.json", status)
            final_checkpoint = output / "checkpoints" / f"update_{update:08d}.pt"
            result = {**status, "status": "complete_fit_stage", "checkpoint": str(final_checkpoint),
                "checkpoint_sha256": sha256_file(final_checkpoint), "normalization": metadata["initial_normalization"],
                "invocation_seconds": time.monotonic()-started, "new_environment_interactions": 0,
                "quality_gate": "P2.5 pending; completed optimizer schedule is not model-quality acceptance"}
            if stage == "initial":
                result["shared_by_variants"] = ["B", "D"]
                for arm in ("B", "D"):
                    write_report(output / "bindings" / f"{arm}.json", {"variant": arm,
                        "checkpoint": os.path.relpath(final_checkpoint, output / "bindings").replace("\\", "/"),
                        "checkpoint_sha256": result["checkpoint_sha256"], "normalization": metadata["initial_normalization"],
                        "replay": provenance["replays"][0], "model_updates": update})
            write_report(output / "result.json", result)
            write_report(output / "status.json", result)
            return result
        except BaseException as exc:
            write_report(output / "status.json", {**status, "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error_type": type(exc).__name__, "error": str(exc), "last_committed_update": last_checkpoint,
                "recovery": "resume replays work since the last committed optimizer boundary; physical extra work remains in logs"})
            raise


def evaluate_checkpoint(checkpoint, *, data_root, statistics_root, settings, device="cuda:0",
                        validation_bank=None, burn_in_study=False, config_root=None):
    # Inspect safely using a temporary model only after task/seed metadata peek.
    payload = read_checkpoint(checkpoint)
    meta = payload["metadata"]
    cfg, datasets, stats, provenance = load_case(meta["task_family"], meta["run_seed"], data_root=data_root,
        statistics_root=statistics_root, config_root=config_root, profile=meta["profile"])
    if provenance["initial_normalization"] != meta["initial_normalization"]:
        raise ValueError("evaluation normalization differs from checkpoint")
    device = selected_device(device, settings.precision)
    torch.set_num_threads(1)
    model = build_model(cfg.settings, master_seed=meta["run_seed"], variant="B").to(device)
    load_checkpoint(checkpoint, model=model, restore_rng=False,
                    expected_metadata={"source_sha256": source_identity(),
                                       "initial_normalization": provenance["initial_normalization"]})
    objective = WorldModelObjective(model.task, stats, ObjectiveConfig.from_experiment(cfg.settings))
    common_cfg = dict(cfg.settings)
    common_cfg.pop("variant")
    if canonical_hash(common_cfg) != meta["common_config_sha256"]:
        raise ValueError("evaluation configuration differs from checkpoint")
    seed = derive_seed(meta["task_family"], meta["run_seed"], "p24_model_diagnostics", "B")
    if validation_bank is not None:
        from .validation import ValidationBank
        bank = ValidationBank(validation_bank)
        if bank.task_family != meta["task_family"] or bank.run_seed != meta["run_seed"]:
            raise ValueError("validation bank belongs to another task/seed")
        if bank.manifest["configuration_sha256"] != meta["common_config_sha256"]:
            raise ValueError("validation bank was collected under another experimental configuration")
        windows = bank.windows(burn_in=settings.burn_in, loss_steps=settings.loss_steps,
                              max_windows=settings.validation_windows, seed=seed)
        split = "validation"
        bank_identity = {"manifest_sha256": bank.manifest_sha256, "cohort_identity": bank.cohort_identity}
    else:
        bank = ReplaySampler(datasets, seed=seed, burn_in=settings.burn_in,
                              loss_steps=settings.loss_steps, partition="diagnostic")
        windows = bank.sample(settings.validation_windows)
        split = "train/model_diagnostic"
        bank_identity = {"cohort_identity": bank.cohort_identity, "replays": provenance["replays"]}
    if burn_in_study:
        if split != "validation":
            raise ValueError("burn-in selection requires a real validation-geometry bank")
        result = compare_burn_in(model, bank, windows, objective, seed=seed+1,
                                microbatch=settings.microbatch, precision=settings.precision)
    else:
        result = evaluate_windows(model, windows, objective, seed=seed+1,
                                 microbatch=settings.microbatch, precision=settings.precision)
    return {"checkpoint_sha256": sha256_file(checkpoint), "model_updates": payload["update"],
            "geometry_split": split, "optimizer_updates": 0, "new_environment_interactions": 0,
            "bank": bank_identity, "sampling_seed": seed,
            "settings": asdict(settings), "runtime": runtime_inventory(device),
            "windows": [{"episode_id": w.episode_id, "start": w.start, "anchor": w.anchor, "stop": w.stop} for w in windows],
            "diagnostics": result, "P2.5_quality_acceptance": False}
