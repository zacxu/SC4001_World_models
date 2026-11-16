"""Read-only import of a completed, receipt-bound P4.1 teacher for P4.2.

This is a stage handoff, not a teacher resume: it never restores optimizers or
sampling streams and permits a different inference device. Every source file
recorded by P4.1 must still have the same bytes. Newly added distillation code
does not invalidate that historical manifest or rewrite its receipts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath

import torch

from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.imagination.versions import module_version
from world_models.infrastructure.reproducibility import canonical_hash, derive_seed
from world_models.mappo.config import MAPPOConfig
from world_models.models.world_model import build_model
from world_models.rollouts.streams import RolloutStreams
from world_models.teacher.runner import (ROOT, _initial_inputs, _latest, _manifest,
                                        _policies, _root)
from world_models.training.runner import selected_device
from .versions import module_version_at_device, origin_device


@dataclass
class CompletedTeacher:
    directory: Path
    manifest: dict
    handoff: dict
    checkpoint: Path
    settings: dict
    master_seed: int
    precision: str
    model: torch.nn.Module
    teacher: torch.nn.Module
    critic: torch.nn.Module
    dataset: object
    normalization: object
    binding: dict
    provenance: dict


def _receipt_json(path):
    receipt = path.with_suffix(".sha256")
    if (not path.is_file() or not receipt.is_file() or path.is_symlink() or receipt.is_symlink()
            or receipt.read_text(encoding="ascii").strip() != f"{file_sha256(path)}  {path.name}"):
        raise ValueError(f"completed teacher needs a valid {path.name} receipt")
    return json.loads(path.read_text(encoding="utf-8"))


def _source_compatibility(recorded):
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("teacher recorded source manifest is missing")
    # P4.1 records all of world_models plus its own CLI. Only this new stage's
    # package may be absent from the historical registration.
    required = {path.relative_to(ROOT).as_posix()
                for path in (ROOT / "src/world_models").rglob("*.py")
                if not path.relative_to(ROOT).as_posix().startswith("src/world_models/distillation/")}
    required.add("scripts/phase41.py")
    if not required <= recorded.keys():
        raise ValueError("teacher recorded source manifest omits an existing dependency")
    for name, expected in recorded.items():
        member = PurePosixPath(name)
        if (type(name) is not str or member.is_absolute() or ".." in member.parts
                or "\\" in name or member.as_posix() != name
                or not (name.startswith("src/world_models/") and name.endswith(".py")
                        or name == "scripts/phase41.py")):
            raise ValueError("teacher source manifest contains an unsupported path")
        path = ROOT.joinpath(*member.parts)
        if (not path.is_file() or any(parent.is_symlink() for parent in (path, *path.parents))
                or file_sha256(path) != expected):
            raise ValueError(f"teacher source changed: {name}; preserve its original checkout")


def _adam_counters(state, network, steps, learning_rate, epsilon):
    """Check recorded Adam work without constructing or restoring an optimizer."""
    parameters = list(network.parameters())
    if (type(state) is not dict or set(state) != {"state", "param_groups"}
            or type(state["param_groups"]) is not list or len(state["param_groups"]) != 1
            or type(state["state"]) is not dict or set(state["state"]) != set(range(len(parameters)))):
        raise ValueError("teacher Adam slots do not match its completed parameter updates")
    group = state["param_groups"][0]
    expected = {"lr": learning_rate, "eps": epsilon, "weight_decay": 0,
                "betas": (0.9, 0.999), "amsgrad": False, "maximize": False,
                "params": list(range(len(parameters)))}
    if any(group.get(key) != value for key, value in expected.items()):
        raise ValueError("teacher Adam configuration differs from its fixed schedule")
    for index, slots in state["state"].items():
        if type(slots) is not dict or set(slots) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("teacher Adam parameter slots are malformed")
        step = slots["step"]
        if (not isinstance(step, torch.Tensor) or step.ndim != 0 or step.dtype != torch.float32
                or step.item() != steps):
            raise ValueError("teacher Adam step differs from completed optimizer counter")
        for name in ("exp_avg", "exp_avg_sq"):
            value = slots[name]
            if (not isinstance(value, torch.Tensor) or value.shape != parameters[index].shape
                    or value.dtype != torch.float32):
                raise ValueError("teacher Adam moments differ from its parameters")
        if (slots["exp_avg_sq"] < 0).any().item():
            raise ValueError("teacher Adam second moments must be nonnegative")


def _validate_completion(saved, metadata, manifest, handoff, checkpoint, model, teacher, critic):
    p = manifest["protocol"]
    cycles = p["teacher_cycles"]
    steps = cycles * p["epochs"] * p["minibatches"]
    if set(saved) != {"controller", "model", "teacher", "critic", "learner", "streams", "sampler", "last_report"}:
        raise ValueError("teacher checkpoint has missing or unsupported state")
    state = saved["controller"]
    if (type(state) is not dict
            or set(state) != {"sequence", "phase", "imagined_joint_steps", "teacher_player_decisions"}
            or state["phase"] != "complete"
            or any(type(state[key]) is not int for key in
                   ("sequence", "imagined_joint_steps", "teacher_player_decisions"))
            or state["sequence"] != cycles + 1
            or int(checkpoint.name.split("_")[1]) != state["sequence"]
            or not cycles * p["starts"] <= state["imagined_joint_steps"] <= cycles * p["starts"] * p["horizon"]
            or state["teacher_player_decisions"] != 2 * state["imagined_joint_steps"]):
        raise ValueError("teacher checkpoint is not a fully completed fixed-schedule run")
    expected_metadata = {"schema_version": "p41-teacher-checkpoint-v1", "policy_role": "global_teacher",
        "run_identity": canonical_hash(manifest), "sequence": state["sequence"], "phase": "complete",
        "teacher_cycles_completed": cycles, "teacher_actor_optimizer_steps": steps,
        "teacher_critic_optimizer_steps": steps, "initial_model_version": manifest["initial_model_version"],
        "model_updates": p["initial_model_updates"], "real_joint_transitions": p["initial_transitions"],
        "new_environment_interactions": 0, "new_model_updates": 0,
        "local_student_optimizer_steps": 0, "frozen_for_distillation": True}
    if canonical_hash(metadata) != canonical_hash(expected_metadata):
        raise ValueError("teacher completed checkpoint metadata or run identity differs")
    versions = tuple(module_version_at_device(network, manifest["runtime"]["logical_device"])
                     for network in (model, teacher, critic))
    if versions[0] != manifest["initial_model_version"]:
        raise ValueError("teacher changed its frozen initial world model")
    expected_handoff = {"schema_version": "p41-frozen-teacher-v1", "policy_role": "global_teacher",
        "run_identity": canonical_hash(manifest), "checkpoint": checkpoint.relative_to(checkpoint.parents[1]).as_posix(),
        "checkpoint_manifest_sha256": file_sha256(checkpoint / "manifest.json"),
        "teacher_version": versions[1], "critic_version": versions[2], "model_version": versions[0],
        "teacher_config": asdict(teacher.config), "attention_heads": teacher.attention_heads,
        "teacher_cycles": cycles, "frozen_for_distillation": True, "deployable_local_actor": False,
        "student_training_executed": False, "research_quality_acceptance": False}
    if canonical_hash(handoff) != canonical_hash(expected_handoff):
        raise ValueError("frozen teacher handoff differs from the latest completed checkpoint")
    learner = saved["learner"]
    config = MAPPOConfig.from_experiment(manifest["input_binding"]["settings"])
    expected_keys = {"schema_version", "config", "precision", "seed", "network_versions", "cycles_completed",
                     "actor_optimizer_steps", "critic_optimizer_steps", "shuffle_rng", "actor_optimizer", "critic_optimizer"}
    if (type(learner) is not dict or set(learner) != expected_keys
            or learner["schema_version"] != "p41-teacher-optimizer-v1"
            or learner["config"] != asdict(config) or learner["precision"] != manifest["precision"]
            or learner["seed"] != manifest["rng_namespaces"]["teacher_updates"]
            or learner["network_versions"] != versions
            or any(type(learner[key]) is not int for key in
                   ("cycles_completed", "actor_optimizer_steps", "critic_optimizer_steps"))
            or (learner["cycles_completed"], learner["actor_optimizer_steps"], learner["critic_optimizer_steps"])
                != (cycles, steps, steps)):
        raise ValueError("teacher optimizer state does not describe completed bound weights")
    _adam_counters(learner["actor_optimizer"], teacher, steps, config.actor_lr, config.adam_epsilon)
    _adam_counters(learner["critic_optimizer"], critic, steps, config.critic_lr, config.adam_epsilon)
    rng = learner["shuffle_rng"]
    if (not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1
            or rng.device.type != "cpu"):
        raise ValueError("teacher shuffle RNG state is malformed")
    try:
        trial_rng = torch.Generator(device="cpu")
        trial_rng.set_state(rng)
    except RuntimeError as error:
        raise ValueError("teacher shuffle RNG state is malformed") from error
    if trial_rng.initial_seed() != learner["seed"]:
        raise ValueError("teacher shuffle RNG belongs to a different namespace")
    if saved["streams"].get("identity") != manifest["rollout_stream_identity"]:
        raise ValueError("teacher recorded rollout streams differ from its manifest")
    expected_streams = RolloutStreams.from_seed(manifest["rng_namespaces"]["teacher_rollouts"], device="cpu").identity
    expected_streams["device"] = manifest["runtime"]["logical_device"]
    if manifest["rollout_stream_identity"] != expected_streams:
        raise ValueError("teacher rollout streams belong to a different namespace")
    last = saved["last_report"]
    if (type(last) is not dict or last.get("schema_version") != "p41-teacher-update-v1"
            or last.get("status") != "complete" or last.get("policy_role") != "global_teacher"
            or (last.get("cycles_completed"), last.get("actor_optimizer_steps"), last.get("critic_optimizer_steps"))
                != (cycles, steps, steps)
            or (last.get("world_model_version"), last.get("actor_version_after"), last.get("critic_version_after")) != versions
            or last.get("new_environment_interactions") != 0):
        raise ValueError("teacher final update report differs from the completed weights")


def _restore_initial_teacher(directory, manifest, model, teacher, critic):
    """Use receipt-bound initial values, not cross-build random regeneration."""
    candidates = [path for path in (directory / "checkpoints").glob("state_00000001_*")
                  if (path / "manifest.sha256").is_file()]
    if len(candidates) != 1:
        raise ValueError("teacher handoff requires exactly one committed initial checkpoint (sequence 1)")
    checkpoint = _root(candidates[0])
    saved, metadata = load_cycle_checkpoint(checkpoint)
    p = manifest["protocol"]
    expected_metadata = {"schema_version": "p41-teacher-checkpoint-v1", "policy_role": "global_teacher",
        "run_identity": canonical_hash(manifest), "sequence": 1, "phase": "teacher",
        "teacher_cycles_completed": 0, "teacher_actor_optimizer_steps": 0, "teacher_critic_optimizer_steps": 0,
        "initial_model_version": manifest["initial_model_version"],
        "model_updates": p["initial_model_updates"], "real_joint_transitions": p["initial_transitions"],
        "new_environment_interactions": 0, "new_model_updates": 0,
        "local_student_optimizer_steps": 0, "frozen_for_distillation": False}
    expected_state = {"sequence": 1, "phase": "teacher", "imagined_joint_steps": 0, "teacher_player_decisions": 0}
    if (metadata != expected_metadata or saved.get("controller") != expected_state):
        raise ValueError("teacher initial model, policy or critic identity changed: initial checkpoint binding differs")
    # The selected P2.4 model was already loaded independently. Match its exact
    # state against the initial teacher snapshot before restoring policy values.
    selected = model.state_dict()
    if (set(saved["model"]) != set(selected)
            or any(not torch.equal(saved["model"][name], value.cpu()) for name, value in selected.items())):
        raise ValueError("teacher initial model differs from its selected P2.4 checkpoint")
    for network, name in ((teacher, "teacher"), (critic, "critic")):
        network.load_state_dict(saved[name], strict=True)
    label = manifest["runtime"]["logical_device"]
    versions = tuple(module_version_at_device(network, label) for network in (model, teacher, critic))
    expected_versions = tuple(manifest.get(f"initial_{name}_version") for name in ("model", "teacher", "critic"))
    if versions != expected_versions:
        raise ValueError("teacher initial model, policy or critic identity changed: saved weight hash mismatch")
    learner = saved["learner"]
    if (learner.get("network_versions") != versions
            or any(type(learner.get(key)) is not int or learner[key] != 0 for key in
                   ("cycles_completed", "actor_optimizer_steps", "critic_optimizer_steps"))
            or any(learner.get(key, {}).get("state") != {} for key in ("actor_optimizer", "critic_optimizer"))):
        raise ValueError("teacher initial checkpoint already contains optimizer updates")
    return {"checkpoint": checkpoint.relative_to(directory).as_posix(),
            "checkpoint_manifest_sha256": file_sha256(checkpoint / "manifest.json"),
            "checkpoint_payload_sha256": file_sha256(checkpoint / "state.pt")}


@torch.inference_mode(False)
def load_completed_teacher(run_dir, *, device="cpu") -> CompletedTeacher:
    """Import a finished stage into frozen inference modules without any writes.

    Input paths remain bound to the original registration. Device/runtime may
    differ for the new stage; this does not assert numerical equivalence with
    P4.1 or permit resuming its optimizer trajectory under changed sources.
    """
    directory = _root(run_dir)
    _receipt_json(directory / "run_manifest.json")
    manifest = _manifest(directory)
    _source_compatibility(manifest.get("source_sha256"))
    recorded_device = origin_device(manifest.get("runtime", {}).get("logical_device"))
    if manifest.get("rollout_stream_identity", {}).get("device") != recorded_device:
        raise ValueError("teacher recorded logical device differs from its sampling identity")
    handoff = _receipt_json(directory / "frozen_teacher.json")
    checkpoint = _latest(directory)
    if (handoff.get("checkpoint") != checkpoint.relative_to(directory).as_posix()
            or handoff.get("checkpoint_manifest_sha256") != file_sha256(checkpoint / "manifest.json")):
        raise ValueError("frozen teacher must name the latest committed checkpoint")
    if manifest.get("precision") not in ("float32", "bfloat16"):
        raise ValueError("teacher precision is unsupported")
    settings, data, stats, initial, binding = _initial_inputs(**manifest["inputs"])
    if canonical_hash(binding) != canonical_hash(manifest.get("input_binding")):
        raise ValueError("teacher frozen replay, normalization, selection or settings changed")
    seed = manifest["inputs"]["master_seed"]
    namespaces = {name: derive_seed("relay", seed, name, "D") for name in
                  ("teacher_initialization", "teacher_critic_initialization", "teacher_anchor_sampling",
                   "teacher_anchor_prefix", "teacher_rollouts", "teacher_updates")}
    if manifest.get("rng_namespaces") != namespaces:
        raise ValueError("teacher initialization or sampling namespaces changed")
    target = selected_device(device, manifest["precision"])
    model = build_model(settings, master_seed=seed, variant="D")
    model.load_state_dict(initial["model"], strict=True)
    teacher, critic = _policies(settings, seed, "cpu")
    initial_snapshot = _restore_initial_teacher(directory, manifest, model, teacher, critic)
    del initial
    saved, metadata = load_cycle_checkpoint(checkpoint)
    for network, name in ((model, "model"), (teacher, "teacher"), (critic, "critic")):
        network.load_state_dict(saved[name], strict=True)
    _validate_completion(saved, metadata, manifest, handoff, checkpoint, model, teacher, critic)
    for network in (model, teacher, critic):
        network.zero_grad(set_to_none=True)
        network.requires_grad_(False).eval().to(target)
    provenance = {"schema_version": "p42-completed-teacher-input-v1",
        "run_identity": canonical_hash(manifest), "run_manifest_sha256": file_sha256(directory / "run_manifest.json"),
        "frozen_teacher_sha256": file_sha256(directory / "frozen_teacher.json"),
        "checkpoint": checkpoint.relative_to(directory).as_posix(),
        "checkpoint_manifest_sha256": file_sha256(checkpoint / "manifest.json"),
        "checkpoint_payload_sha256": file_sha256(checkpoint / "state.pt"),
        "teacher_version": handoff["teacher_version"], "critic_version": handoff["critic_version"],
        "model_version": handoff["model_version"], "teacher_cycles": handoff["teacher_cycles"],
        "source_compatibility": "all recorded paths unchanged; new distillation modules allowed",
        "teacher_runtime": deepcopy(manifest["runtime"]), "handoff_device": str(target),
        "verified_origin_device": recorded_device, "initial_snapshot": initial_snapshot,
        "imported_network_versions": {name: module_version(network) for name, network in
                                      (("model", model), ("teacher", teacher), ("critic", critic))},
        "cross_runtime_numerical_equivalence_measured": False}
    # Reject concurrent changes without opening the original run's write lock.
    if (_latest(directory) != checkpoint or canonical_hash(_manifest(directory)) != canonical_hash(manifest)
            or canonical_hash(_receipt_json(directory / "frozen_teacher.json")) != canonical_hash(handoff)):
        raise ValueError("teacher publication changed while importing the completed stage")
    return CompletedTeacher(directory, manifest, handoff, checkpoint, settings, seed, manifest["precision"],
                            model, teacher, critic, data, stats, binding, provenance)
