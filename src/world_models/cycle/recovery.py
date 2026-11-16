"""Explicit, checkpoint-preserving recovery for the initial BF16 actor defect.

Ordinary resume still requires identical sources. This narrowly scoped amendment
accepts the recorded pre-fix implementation, preserves its evidence, and creates
a fresh run with IEEE FP32 actor computation. It never optimizes or collects data.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from world_models.cycle.checkpoints import load_cycle_checkpoint, save_cycle_checkpoint
from world_models.cycle.controller import CycleController, CycleOptions, _latest_checkpoint, _runtime
from world_models.cycle.provenance import source_identity
from world_models.data.ledger import atomic_json, file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.training.runner import run_lock, selected_device


AMENDMENT = "p35-actor-ieee-fp32-v1"
LEGACY_SOURCE_IDENTITY = "f1ef116fb4092e039c33e943540da5290dd6bee5d3953f5cfa7ed9877462e63f"
LEGACY_PATCH_FILES = {
    "src/world_models/policies/actor.py": "1f802db8e63b02b9a6b1716d8c9583a611f89b89bf88ff6ac709553ebc022127",
    "src/world_models/mappo/updater.py": "b0472a0adb52af5dcb17074bc9214543671f564929f33fe16fcc6d805fe4bff1",
    "src/world_models/cycle/controller.py": "bd9dd1bf35a149f8ee35051e5dbac82e3f92cb24a76335a99e8b2998d77a42ae",
    "scripts/phase35.py": "3a6e7e083e25af2878cff8b60bb14bab7778b23ae2d55c2945eaf8372ff57798",
    "tests/test_local_actor.py": "5d84eac2f376b242e61a461e0dca3fcf6770d268422675daf7e7635fad7b1a97",
    "tests/test_mappo_recurrent.py": "183eac000b42db30f6ce469a0d050a100f45dc73944a958bb8e58a3468bd6e0d",
    "tests/test_mappo_updater.py": "8337fa89dc90030116df71ae09d166f7dc5f1c8091871baa28173de46a715c11",
}
ADDED_SOURCES = {"src/world_models/cycle/recovery.py"}
REPLAY_ERROR = "initial behavior replay does not reproduce stored log probabilities"


def _path(value):
    path = Path(value)
    if ".." in path.parts or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("recovery paths cannot traverse parents or symbolic links")
    return path.resolve()


def _validate_sources(previous, current):
    if not isinstance(previous, dict) or canonical_hash(previous) != LEGACY_SOURCE_IDENTITY:
        raise ValueError("recovery requires the recognized pre-fix P3.5 source identity")
    if set(current) != set(previous) | ADDED_SOURCES:
        raise ValueError("recovery source file set contains an unrelated change")
    unexpected = [name for name, digest in previous.items()
                  if name not in LEGACY_PATCH_FILES and current.get(name) != digest]
    if unexpected:
        raise ValueError("recovery refuses unrelated source changes: " + ", ".join(sorted(unexpected)))
    if any(previous.get(name) != digest for name, digest in LEGACY_PATCH_FILES.items()):
        raise ValueError("recovery pre-fix file hashes differ")
    if current["src/world_models/policies/actor.py"] == previous["src/world_models/policies/actor.py"]:
        raise ValueError("install the actor precision fix before recovering")


def recover_run(source_run, run_dir, *, device="cuda:0") -> dict:
    """Create a fresh numerical continuation on the original runtime/device.

    Only a failed initial local block is supported. There is no uncommitted real
    collection to discard. A new source identity is recorded, never substituted
    into the original run. The baseline export remains the same weight artifact.
    """
    source, destination = _path(source_run), _path(run_dir)
    if not (source / "run_manifest.json").is_file():
        raise ValueError("source must be an existing P3.5 run")
    if destination.exists():
        raise FileExistsError("recovery requires a fresh run directory; preserve existing evidence")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("recovery destination must be separate from the source run")
    with run_lock(source):
        manifest_path = source / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_sources = source_identity()
        if manifest.get("schema_version") != "p35-run-v1" or "recovery" in manifest:
            raise ValueError("unsupported recovery source run")
        _validate_sources(manifest.get("source_sha256"), current_sources)
        options = CycleOptions(**manifest["options"])
        selected = selected_device(device, options.precision)
        if manifest["device"] != str(selected) or manifest["runtime"] != _runtime(selected):
            raise ValueError("recovery requires the same logical device and runtime as the original run")
        status = json.loads((source / "status.json").read_text(encoding="utf-8"))
        if (status.get("status") != "failed" or status.get("error_type") != "ValueError"
                or not str(status.get("error", "")).startswith(REPLAY_ERROR)):
            raise ValueError("recovery is only for the recorded initial behavior replay failure")
        checkpoint = _latest_checkpoint(source)
        if checkpoint is None:
            raise ValueError("source has no committed checkpoint")
        saved, metadata = load_cycle_checkpoint(checkpoint)
        if metadata.get("run_identity") != canonical_hash(manifest):
            raise ValueError("source checkpoint is not bound to its run manifest")
        state, learner = saved["controller"], saved["mappo"]
        if state["phase"] != "local" or state["block_index"] != 0:
            raise ValueError("this recovery supports the initial local block only")
        completed = learner["cycles_completed"]
        if (status.get("local_cycles_completed") != completed
                or status.get("sequence") != state["sequence"]):
            raise ValueError("status leads the committed checkpoint; reconcile uncommitted work before recovery")
        # This validates tensors, optimizer/RNG contracts, replay identities,
        # frozen initial selection and all schedule counters before publishing.
        controller = CycleController._restore_checkpoint(source, manifest, checkpoint, device=selected)
        exports = state["evaluation_exports"]
        if len(exports) != 1 or exports[0]["real_joint_transitions"] != 0:
            raise ValueError("initial recovery requires exactly the paired untrained actor export")
        row = exports[0]
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parent != Path("evaluation"):
            raise ValueError("baseline export path escapes the source evaluation directory")
        baseline = _path(source / relative)
        from scripts.phase36_support.evaluation import load_actor_export
        actor, baseline_record = load_actor_export(baseline, device="cpu", source_device=manifest["device"])
        del actor
        baseline_meta = baseline_record["metadata"]
        if (not baseline_meta["untrained_zero_interaction_baseline"]
                or baseline_meta["actor_version"] != row["actor_version"]
                or baseline_meta["model_updates"] != controller.schedule.initial_model_updates):
            raise ValueError("baseline export differs from the initial run")
        metrics_path = source / "metrics.jsonl"
        metrics_content = metrics_path.read_bytes() if metrics_path.is_file() else b""
        rows = [json.loads(line) for line in metrics_content.splitlines()]
        if (len(rows) != completed or any(record.get("event") != "local_ppo"
                or record.get("cycles_completed") != index for index, record in enumerate(rows, 1))):
            raise ValueError("source metrics do not match its committed initial cycles")
        recovered = deepcopy(manifest)
        recovered.update(source_sha256=current_sources, created_at=datetime.now(timezone.utc).isoformat())
        recovered["recovery"] = {
            "amendment": AMENDMENT, "source_run": str(source),
            "source_run_identity": canonical_hash(manifest),
            "source_manifest_sha256": file_sha256(manifest_path),
            "source_checkpoint": checkpoint.name,
            "source_checkpoint_manifest_sha256": file_sha256(checkpoint / "manifest.json"),
            "source_checkpoint_payload_sha256": file_sha256(checkpoint / "state.pt"),
            "source_failure": status["error"], "completed_cycles_preserved": completed,
            "actor_forward_precision": "ieee_float32",
            "world_model_and_critic_precision": options.precision,
            "weights_optimizer_and_sampling_state_preserved": True,
            "exact_original_numerical_trajectory": False,
            "purpose": "development_recovery; freeze corrected sources before fresh B/D cohort runs",
        }
        if source_identity() != current_sources:
            raise ValueError("source changed during recovery validation")
        # Publication starts only after validation. A failed write remains a
        # separate incomplete destination; the original checkpoint is retained.
        destination.mkdir(parents=True, exist_ok=False)
        with run_lock(destination):
            shutil.copyfile(manifest_path, destination / "recovery_parent_manifest.json")
            shutil.copyfile(source / "status.json", destination / "recovery_parent_status.json")
            (destination / "metrics.jsonl").write_bytes(metrics_content)
            target_baseline = destination / relative
            target_baseline.mkdir(parents=True)
            for member in ("manifest.json", "state.pt", "manifest.sha256"):
                shutil.copyfile(baseline / member, target_baseline / member)
            load_cycle_checkpoint(target_baseline)
            atomic_json(destination / "run_manifest.json", recovered)
            atomic_json(destination / "evaluation_checkpoints.json", exports)
            target_checkpoint = destination / "checkpoints" / checkpoint.name
            save_cycle_checkpoint(target_checkpoint, state=saved,
                                  metadata={**metadata, "run_identity": canonical_hash(recovered)})
            controller.directory, controller.manifest = destination, recovered
            controller.last_checkpoint = target_checkpoint
            controller._status("ready")
            report = {"status": "recovered", "amendment": AMENDMENT,
                      "source_run": str(source), "run_dir": str(destination),
                      "local_cycles_completed": completed,
                      "initial_local_cycles": controller.schedule.initial_local_cycles,
                      "remaining_initial_units": controller.schedule.initial_local_cycles - completed + 1,
                      "training_executed": False, "new_environment_interactions": 0,
                      "recovery": recovered["recovery"]}
            atomic_json(destination / "recovery.json", report)
        return report
