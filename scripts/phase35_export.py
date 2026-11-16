"""Project a final local actor from an immutable P3.5 checkpoint, on CPU.

This is an export operation: it never restores a controller, optimizer or world
model and never changes the original run. Historical source differences are
recorded, rather than attributed to the completed training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

import torch

from scripts.phase36_support.evaluation import actor_version, load_actor_export
from world_models.cycle.checkpoints import load_cycle_checkpoint, save_cycle_checkpoint
from world_models.cycle.schedule import Schedule
from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.policies.actor import LocalActor
from world_models.policies.config import PolicyConfig


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "p35-terminal-actor-export-v1"
_CHECKPOINT = re.compile(r"state_(\d{8})_[0-9a-f]{8}\Z")
_ACTOR_SOURCES = (
    "src/world_models/policies/actor.py", "src/world_models/policies/config.py",
    "src/world_models/contracts.py",
)


def _path(value):
    path = Path(value)
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("export paths cannot traverse parents or symbolic links")
    return path.resolve()


def _sha(value):
    return (type(value) is str and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _latest(root):
    directory = _path(root / "checkpoints")
    choices = {}
    for path in directory.iterdir():
        if path.is_symlink():
            raise ValueError("checkpoint paths cannot be symbolic links")
        if not path.is_dir() or not (path / "manifest.sha256").is_file():
            continue  # An uncommitted publication is never a saved state.
        match = _CHECKPOINT.fullmatch(path.name)
        if match is None:
            raise ValueError("malformed committed checkpoint name")
        sequence = int(match[1])
        if sequence in choices:
            raise ValueError("duplicate committed checkpoint sequence")
        choices[sequence] = path
    if not choices:
        raise ValueError("run has no committed checkpoint")
    sequence = max(choices)
    return choices[sequence], sequence


@contextmanager
def _exclusive_existing_run(root):
    """Use the existing worker lock without creating or writing any run file."""
    lock = root / ".worker.lock"
    if lock.is_symlink() or not lock.is_file():
        raise ValueError("run has no existing worker lock file")
    with lock.open("r+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("another worker owns this run; wait for it to exit") from error
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _sources(recorded):
    if type(recorded) is not dict or not recorded:
        raise ValueError("run must record its training sources")
    current = {}
    for name, digest in recorded.items():
        if type(name) is not str:
            raise ValueError("invalid recorded training source")
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or path.drive
                or "\\" in name or not _sha(digest)):
            raise ValueError("invalid recorded training source")
        source = _path(ROOT / path)
        if not source.is_relative_to(ROOT):
            raise ValueError("recorded training source escapes the project")
        if source.is_file():
            current[name] = file_sha256(source)
    paths = [*sorted((ROOT / "src/world_models").rglob("*.py")), Path(__file__).resolve()]
    current.update({path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths})
    for name in _ACTOR_SOURCES:
        if name not in recorded or current.get(name) != recorded[name]:
            raise ValueError("deployable actor implementation changed since training: " + name)
    return current, {
        "changed": {name: {"training_sha256": digest, "current_sha256": current.get(name)}
                    for name, digest in recorded.items() if current.get(name) != digest},
        "added": {name: digest for name, digest in current.items() if name not in recorded},
    }


def _schedule(manifest):
    settings = manifest["settings"]
    fields = ("initial_transitions", "real_budget", "collection_block", "initial_model_updates",
              "online_model_updates", "initial_local_cycles", "online_local_cycles")
    row = manifest["schedule"]
    schedule = Schedule(**{name: row[name] for name in fields})
    if row != schedule.describe():
        raise ValueError("run schedule is inconsistent")
    registered = Schedule.from_experiment(settings)
    if any(getattr(registered, name) != getattr(schedule, name)
           for name in fields if name != "initial_model_updates"):
        raise ValueError("saved schedule disagrees with the registered settings")
    if registered.initial_model_updates != schedule.initial_model_updates:
        selection = manifest.get("initial_selection", {})
        if (schedule.initial_model_updates > registered.initial_model_updates
                or selection.get("model_updates") != schedule.initial_model_updates
                or selection.get("original_target_stage_updates") != registered.initial_model_updates
                or manifest.get("registered_schedule") != registered.describe()
                or manifest.get("initial_training_policy") != "frozen_last_saved_checkpoints"
                or not _sha(selection.get("selection_sha256"))
                or selection.get("checkpoint_sha256") != manifest.get("initial_checkpoint_sha256")):
            raise ValueError("shortened initial model budget lacks its original selection binding")
    return schedule


def _validate_terminal(saved, metadata, manifest, sequence, schedule):
    state, learner, refresher = saved["controller"], saved["mappo"], saved["refresher"]
    expected = {"sequence", "block_index", "phase", "real_joint_transitions", "model_updates_in_block",
                "local_cycles_in_block", "imagined_joint_steps", "local_player_decisions", "evaluation_exports"}
    if type(state) is not dict or set(state) != expected:
        raise ValueError("unsupported saved controller state")
    for name in expected - {"phase", "evaluation_exports"}:
        _integer(state[name], name)
    if (state["sequence"] != sequence or state["phase"] not in ("local", "complete")
            or state["block_index"] != schedule.blocks
            or state["local_cycles_in_block"] != schedule.online_local_cycles
            or state["model_updates_in_block"] != schedule.online_model_updates
            or state["real_joint_transitions"] != schedule.real_budget
            or state["local_player_decisions"] != 2 * state["imagined_joint_steps"]):
        raise ValueError("checkpoint has not completed the final training budget")
    if (metadata.get("run_identity") != canonical_hash(manifest)
            or metadata.get("phase") != state["phase"]
            or metadata.get("block_index") != state["block_index"]
            or metadata.get("real_joint_transitions") != state["real_joint_transitions"]
            or type(metadata.get("completed_block")) is not bool
            or metadata.get("evaluation_executed") is not False):
        raise ValueError("checkpoint metadata is not bound to this run and controller")
    ppo = manifest["settings"]["mappo"]
    cycles = schedule.total_local_cycles
    steps = cycles * _integer(ppo["epochs"], "epochs", minimum=1) * _integer(ppo["recurrent_minibatches"], "recurrent_minibatches", minimum=1)
    if (learner.get("schema_version") != "p34-optimizer-v1"
            or learner.get("cycles_completed") != cycles
            or learner.get("actor_optimizer_steps") != steps or learner.get("critic_optimizer_steps") != steps
            or learner.get("config", {}).get("epochs") != ppo["epochs"]
            or learner.get("config", {}).get("minibatches") != ppo["recurrent_minibatches"]):
        raise ValueError("saved PPO counters disagree with the completed schedule")
    for name in ("cycles_completed", "actor_optimizer_steps", "critic_optimizer_steps"):
        _integer(learner[name], name)
    versions = learner.get("network_versions")
    for name in ("initial_model_updates", "total_model_updates"):
        _integer(refresher.get(name), name, minimum=1)
    if (type(versions) not in (tuple, list) or len(versions) != 3 or not all(_sha(v) for v in versions)
            or refresher.get("schema_version") != "p35-model-refresher-v1"
            or refresher.get("initial_model_updates") != schedule.initial_model_updates
            or refresher.get("total_model_updates") != schedule.total_model_updates
            or refresher.get("model_version") != versions[0]):
        raise ValueError("saved model version or update counters disagree")
    return state, learner, versions


@torch.inference_mode(False)
def export_final_actor(run_dir, output_dir):
    """Publish a fresh, validated actor-only projection of the terminal state."""
    root, output = _path(run_dir), _path(output_dir)
    if output.exists():
        raise FileExistsError("actor export output must be a fresh directory; preserve earlier evidence")
    if output == root or output in root.parents or (output.is_relative_to(root)
            and output.parent != root / "evaluation"):
        raise ValueError("within the source run, exports must be new direct children of evaluation")
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("run has no ordinary run_manifest.json")
    with _exclusive_existing_run(root):
        manifest_hash = file_sha256(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "p35-run-v1":
            raise ValueError("export requires a P3.5 run manifest")
        _integer(manifest["master_seed"], "master_seed")
        settings = manifest["settings"]
        if settings["variant"]["name"] != "B":
            raise ValueError("terminal export supports the existing variant B controller")
        schedule = _schedule(manifest)
        sources, differences = _sources(manifest.get("source_sha256"))
        checkpoint, sequence = _latest(root)
        checkpoint_hash = file_sha256(checkpoint / "manifest.json")
        saved, checkpoint_meta = load_cycle_checkpoint(checkpoint, device="cpu")
        state, learner, versions = _validate_terminal(saved, checkpoint_meta, manifest, sequence, schedule)
        # Construct the deployment network only, preserving the caller's CPU RNG.
        config = PolicyConfig(image_shape=tuple(settings["environment"]["image_shape"]),
            encoder_channels=tuple(settings["world_model"]["encoder_channels"]),
            embedding_size=settings["actor"]["embedding_size"], hidden_size=settings["actor"]["hidden_size"],
            action_count=settings["environment"]["action_count"],
            world_tokens=settings["world_model"]["memory_tokens"] + settings["world_model"]["latent_variables"],
            world_width=settings["world_model"]["width"], critic_hidden_sizes=tuple(settings["critic"]["hidden_sizes"]),
            value_huber_delta=settings["mappo"]["value_huber_delta"])
        with torch.random.fork_rng(devices=[]):
            actor = LocalActor(config)
        expected = actor.state_dict()
        if type(saved["actor"]) is not dict or set(saved["actor"]) != set(expected) or any(
                not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or tuple(value.shape) != tuple(expected[name].shape) for name, value in saved["actor"].items()):
            raise ValueError("saved actor tensor names, shapes, or FP32 dtypes disagree")
        actor.load_state_dict(saved["actor"], strict=True)
        origin = manifest["device"]
        if (type(origin) is not str or re.fullmatch(r"cpu|cuda:\d+", origin) is None
                or actor_version(actor, origin_device=origin) != versions[1]):
            raise ValueError("saved actor weights do not match their original version/device")
        export_meta = {"schema_version": "p35-local-actor-v1", "task_family": settings["task"]["name"],
            "variant": "B", "run_seed": manifest["master_seed"], "real_joint_transitions": schedule.real_budget,
            "actual_initial_replay_interactions": schedule.initial_transitions,
            "untrained_zero_interaction_baseline": False, "actor_version": versions[1],
            "actor_config": json.loads(json.dumps(asdict(config))), "world_model_version": versions[0],
            "model_updates": schedule.total_model_updates, "local_ppo_cycles": learner["cycles_completed"],
            "actor_optimizer_steps": learner["actor_optimizer_steps"], "critic_optimizer_steps": learner["critic_optimizer_steps"],
            "evaluation_split": settings["experiment"]["evaluation_split"],
            "on_registered_evaluation_grid": schedule.real_budget in settings["experiment"]["checkpoints"],
            "evaluation_executed": False, "phase3_acceptance": False}
        if (_latest(root) != (checkpoint, sequence) or file_sha256(manifest_path) != manifest_hash
                or file_sha256(checkpoint / "manifest.json") != checkpoint_hash
                or _sources(manifest["source_sha256"])[0] != sources):
            raise ValueError("run or export sources changed during validation")
        save_cycle_checkpoint(output, state={"actor": saved["actor"]}, metadata=export_meta)
        del saved
        _, checked = load_actor_export(output, device="cpu", source_device=origin)
        payload = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))["payload"]
        report = {"schema_version": SCHEMA_VERSION, "status": "exported",
            "created_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(root), "output_dir": str(output),
            "source_run_manifest_sha256": manifest_hash, "source_run_identity": canonical_hash(manifest),
            "source_checkpoint": checkpoint.name, "source_checkpoint_manifest_sha256": checkpoint_hash,
            "source_checkpoint_payload_sha256": payload["sha256"],
            "source_checkpoint_completed_block": checkpoint_meta["completed_block"],
            "source_checkpoint_phase": state["phase"], "original_training_source_sha256": manifest["source_sha256"],
            "export_source_sha256": sources, "source_differences": differences,
            "model_version_validation": "matching receipt-bound refresher and PPO network identities; model not instantiated",
            "actor_export": checked, "training_budget_complete": True,
            "controller_finalization_performed": False, "final_actor_export_only": True,
            "training_executed": False, "new_environment_interactions": 0,
            "new_model_updates": 0, "new_actor_optimizer_steps": 0, "new_critic_optimizer_steps": 0,
            "world_model_instantiated": False, "critic_instantiated": False,
            "phase3_acceptance": False, "original_run_modified": False}
        report_path = output / "export_provenance.json"
        with report_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        with (output / "export_provenance.sha256").open("x", encoding="ascii", newline="\n") as stream:
            stream.write(f"{file_sha256(report_path)}  export_provenance.json\n")
            stream.flush(); os.fsync(stream.fileno())
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    # Full checkpoint integrity checking is much faster with bounded CPU threads.
    torch.set_num_threads(1)
    try:
        report = export_final_actor(args.run_dir, args.output_dir)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"P3.5 actor export error ({type(error).__name__}): {error}", file=sys.stderr)
        return 1
    print(json.dumps({name: report[name] for name in ("status", "output_dir", "training_budget_complete",
                                                     "training_executed", "controller_finalization_performed")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
