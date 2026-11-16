"""Read-only P4.2 to P4.3 handoff, without teachers or inherited optimizers.

Historical source bytes and device-labelled identities remain authoritative.
Only the newly added refinement package may be absent from a P4.2 manifest.
The supervised buffer is audited one shard at a time, never concatenated or
passed to local PPO. Native environment interactions are impossible here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path, PurePosixPath
import re

import torch

from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.distillation.buffer import DistillationBuffer
from world_models.distillation.handoff import _adam_counters, _receipt_json
from world_models.distillation.learning import DistillationConfig
from world_models.distillation.protocol import load_protocol as distillation_protocol
from world_models.distillation.protocol import protocol_identity as distillation_identity
from world_models.distillation.runner import _tree_equal
from world_models.distillation.versions import module_version_at_device, origin_device
from world_models.imagination import AnchorSampler
from world_models.imagination.replay import AnchorRef
from world_models.imagination.versions import module_version
from world_models.infrastructure.reproducibility import canonical_hash, derive_seed
from world_models.models.world_model import build_model
from world_models.policies.initialization import build_actor_critic
from world_models.rollouts.streams import RolloutStreams
from world_models.teacher.runner import ROOT, _initial_inputs, _latest, _root
from world_models.teacher.protocol import load_protocol as teacher_protocol
from world_models.teacher.protocol import protocol_identity as teacher_identity
from world_models.training.runner import selected_device


@dataclass
class WarmStartedStudent:
    directory: Path
    manifest: dict
    handoff: dict
    checkpoint: Path
    settings: dict
    master_seed: int
    precision: str
    model: torch.nn.Module
    actor: torch.nn.Module
    critic: torch.nn.Module
    dataset: object
    normalization: object
    binding: dict
    provenance: dict


def _source_compatibility(recorded, *, stage):
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError(f"{stage} recorded source manifest is missing")
    allowed_new = ("src/world_models/refinement/",)
    if stage == "teacher":
        allowed_new += ("src/world_models/distillation/",)
    required = {path.relative_to(ROOT).as_posix()
                for path in (ROOT / "src/world_models").rglob("*.py")
                if not path.relative_to(ROOT).as_posix().startswith(allowed_new)}
    required.add("scripts/phase41.py")
    if stage == "distillation":
        required.add("scripts/phase42.py")
    if not required <= recorded.keys():
        raise ValueError(f"{stage} source manifest omits an existing dependency")
    for name, expected in recorded.items():
        if type(name) is not str:
            raise ValueError(f"{stage} source manifest contains an unsupported path")
        member = PurePosixPath(name)
        if (member.is_absolute() or ".." in member.parts or "\\" in name or member.as_posix() != name
                or not (name.startswith("src/world_models/") and name.endswith(".py")
                        or name in ("scripts/phase41.py", "scripts/phase42.py"))):
            raise ValueError(f"{stage} source manifest contains an unsupported path")
        path = _root(ROOT.joinpath(*member.parts))
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"{stage} source changed: {name}; preserve its original checkout")


def _snapshot(directory):
    paths = [path for path in (directory / "checkpoints").glob("state_00000001_*")
             if (path / "manifest.sha256").is_file()]
    if len(paths) != 1:
        raise ValueError("student handoff needs exactly one committed initial checkpoint")
    return _root(paths[0])


def _diagnostics(value, *, updates, trajectories, joint_steps):
    expected = {"schema_version", "forward_kl", "student_entropy", "valid_player_decisions",
                "trajectories", "optimizer_updates", "scope"}
    if (type(value) is not dict or set(value) != expected
            or value["schema_version"] != "p42-training-diagnostics-v1"
            or value["scope"] != "fixed training buffer; not validation or real performance"
            or value["trajectories"] != trajectories or value["optimizer_updates"] != updates
            or value["valid_player_decisions"] != 2 * joint_steps
            or any(type(value[key]) not in (int, float) or not math.isfinite(value[key])
                   or value[key] < -1e-6 for key in ("forward_kl", "student_entropy"))
            or value["student_entropy"] > math.log(12) + 1e-5):
        raise ValueError("distillation before/after training diagnostics differ from completed work")


def _stream_shape(state, identity):
    """Audit historical CUDA RNG bytes without allocating that CUDA device."""
    if (type(state) is not dict or set(state) != {"schema_version", "identity", "world", "actors"}
            or state["schema_version"] != "p33-rollout-streams-v1" or state["identity"] != identity
            or type(state["world"]) is not dict or set(state["world"]) != {"prior", "posterior", "outcome"}
            or type(state["actors"]) is not dict or set(state["actors"]) != {"0", "1"}):
        raise ValueError("distillation rollout stream identity differs")
    for value in (*state["world"].values(), *state["actors"].values()):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0):
            raise ValueError("distillation historical sampling RNG is malformed")
    if identity["device"] == "cpu":
        for name, value in [*state["world"].items(), *[(f"actor_{key}", value)
                                                for key, value in state["actors"].items()]]:
            probe = torch.Generator(device="cpu")
            try:
                probe.set_state(value)
            except RuntimeError as error:
                raise ValueError("distillation historical sampling RNG is malformed") from error
            if probe.initial_seed() != identity["seeds"][name]:
                raise ValueError("distillation historical sampling RNG namespace differs")


def _audit_shards(directory, manifest, saved, initial, sampler):
    p, refs = manifest["protocol"], tuple(AnchorRef(**row) for row in manifest["references"])
    if refs != sampler.sample(p["trajectories"]):
        raise ValueError("distillation anchors differ from their registered common quota")
    if not _tree_equal(sampler.state_dict(), saved["sampler"]):
        raise ValueError("distillation sampler differs from its completed fixed draw")
    for ref in refs:
        sampler.validate_reference(ref)
    previous = initial["streams"]
    identity = manifest["rollout_stream_identity"]
    _stream_shape(previous, identity)
    total, bindings = 0, []
    rows = saved["shards"]
    if type(rows) is not list or len(rows) != p["trajectories"]:
        raise ValueError("completed student must bind every distillation trajectory")
    seen = set()
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {"path", "manifest_sha256"} or type(row["path"]) is not str:
            raise ValueError("distillation shard binding is malformed")
        member = PurePosixPath(row["path"])
        if (member.is_absolute() or ".." in member.parts or "\\" in row["path"]
                or member.as_posix() != row["path"] or row["path"] in seen):
            raise ValueError("distillation shard path escapes or is duplicated")
        seen.add(row["path"])
        path = _root(directory.joinpath(*member.parts))
        if (not path.is_relative_to(directory / "buffer")
                or file_sha256(path / "manifest.json") != row["manifest_sha256"]):
            raise ValueError("distillation shard receipt binding differs")
        payload, metadata = load_cycle_checkpoint(path)
        expected = {"schema_version": "p42-buffer-shard-v1", "run_identity": canonical_hash(manifest),
            "trajectory_index": index, "reference": asdict(refs[index]),
            "frozen_versions": manifest["frozen_versions"], "learning_eligible_for_world_model": False,
            "on_policy_for_student": False, "data_source": "teacher_world_model_imagination"}
        if metadata != expected or set(payload) != {"buffer"}:
            raise ValueError("distillation shard metadata differs from its student run")
        part = DistillationBuffer.from_state_dict(payload["buffer"])
        if (part.batch_size != 1 or not 1 <= part.steps <= p["horizon"]
                or part.cutoff.any().item() and part.steps != p["horizon"]
                or refs[index].decision_index + part.imagined_joint_steps > sampler.deadline
                or part.prefixes.references != (refs[index],)
                or part.metadata["teacher_version"] != manifest["frozen_versions"]["teacher"]
                or part.metadata["world_model_version"] != manifest["frozen_versions"]["model"]
                or part.metadata["cohort_identity"] != sampler.cohort_identity
                or part.metadata["task_family"] != sampler.task_family or part.metadata["deadline"] != sampler.deadline
                or len(part.streams) != 1 or not _tree_equal(part.streams[0]["before"], previous)):
            raise ValueError("distillation shard anchor, frozen model or RNG chain differs")
        # Validate prefixes against real replay rather than trusting copied RGB.
        prefix = sampler.read_prefix(refs[index])
        if not all((getattr(prefix, key) == getattr(part.prefixes.prefixes[0], key)).all()
                   for key in ("rgb", "actions", "decision_indices")):
            raise ValueError("distillation prefix differs from the original real replay")
        previous = part.streams[0]["after"]
        _stream_shape(previous, identity)
        total += part.imagined_joint_steps
        bindings.append({**row, "payload_sha256": file_sha256(path / "state.pt")})
        del payload, part
    if not _tree_equal(previous, saved["streams"]) or total != saved["controller"]["imagined_joint_steps"]:
        raise ValueError("distillation final RNG chain or valid-step count differs")
    return bindings


def _metadata(manifest, controller, actor_version, critic_version, updates):
    return {"schema_version": "p42-distillation-checkpoint-v1", "run_identity": canonical_hash(manifest),
            **controller, "distillation_optimizer_steps": updates, "actor_version": actor_version,
            "critic_version": critic_version, "new_environment_interactions": 0,
            "new_model_updates": 0, "local_ppo_optimizer_steps": 0, "teacher_optimizer_steps": 0}


@torch.inference_mode(False)
def load_warm_started_student(run_dir, *, device="cpu") -> WarmStartedStudent:
    """Verify completed P4.2 and return local modules ready for fresh PPO.

    This operation never writes to either upstream run or restores any optimizer.
    Recorded absolute input paths remain intact; a moved checkout must retain
    access to those original paths, rather than silently rewriting identities.
    """
    directory = _root(run_dir)
    manifest = _receipt_json(directory / "run_manifest.json")
    p = distillation_protocol()
    if (manifest.get("schema_version") != "p42-distillation-run-v1" or manifest.get("protocol") != p
            or manifest.get("protocol_sha256") != distillation_identity()
            or type(manifest.get("master_seed")) is not int
            or manifest.get("master_seed") not in p["master_seeds"]
            or manifest.get("precision") not in ("float32", "bfloat16")
            or manifest.get("fresh_ppo_optimizer_required_after_warm_start") is not True):
        raise ValueError("distillation protocol, seed, precision or fresh PPO requirement differs")
    _source_compatibility(manifest.get("source_sha256"), stage="distillation")
    label = origin_device(manifest.get("runtime", {}).get("logical_device"))
    frozen = manifest.get("frozen_versions")
    if (type(frozen) is not dict or set(frozen) != {"model", "teacher", "teacher_critic", "student_critic"}
            or any(type(value) is not str or re.fullmatch("[a-f0-9]{64}", value) is None
                   for value in frozen.values())):
        raise ValueError("student frozen input versions are malformed")
    handoff = _receipt_json(directory / "warm_started_student.json")
    checkpoint = _latest(directory)
    if (handoff.get("checkpoint") != checkpoint.relative_to(directory).as_posix()
            or handoff.get("checkpoint_manifest_sha256") != file_sha256(checkpoint / "manifest.json")):
        raise ValueError("warm-started student must name the latest committed checkpoint")
    teacher_directory = _root(manifest["teacher_run"])
    teacher_manifest = _receipt_json(teacher_directory / "run_manifest.json")
    teacher_handoff = _receipt_json(teacher_directory / "frozen_teacher.json")
    if (teacher_manifest.get("schema_version") != "p41-teacher-run-v1"
            or teacher_manifest.get("protocol") != teacher_protocol()
            or teacher_manifest.get("protocol_sha256") != teacher_identity()):
        raise ValueError("upstream teacher protocol differs")
    _source_compatibility(teacher_manifest.get("source_sha256"), stage="teacher")
    teacher_checkpoint = _latest(teacher_directory)
    if (teacher_handoff.get("checkpoint") != teacher_checkpoint.relative_to(teacher_directory).as_posix()
            or teacher_handoff.get("checkpoint_manifest_sha256") != file_sha256(teacher_checkpoint / "manifest.json")
            or teacher_handoff.get("run_identity") != canonical_hash(teacher_manifest)
            or teacher_handoff.get("teacher_cycles") != p["teacher_cycles"]
            or teacher_handoff.get("frozen_for_distillation") is not True):
        raise ValueError("upstream teacher handoff differs from its completed checkpoint")
    # Verify immutable teacher payload bytes without instantiating a teacher or
    # importing its optimizer. Its weights are never passed into local PPO.
    teacher_checkpoint_manifest = _receipt_json(teacher_checkpoint / "manifest.json")
    specification = teacher_checkpoint_manifest.get("payload", {})
    if (specification.get("path") != "state.pt"
            or (teacher_checkpoint / "state.pt").stat().st_size != specification.get("bytes")
            or file_sha256(teacher_checkpoint / "state.pt") != specification.get("sha256")):
        raise ValueError("upstream teacher checkpoint payload differs")
    teacher_p = teacher_manifest["protocol"]
    teacher_steps = teacher_p["teacher_cycles"] * teacher_p["epochs"] * teacher_p["minibatches"]
    expected_teacher_metadata = {"schema_version": "p41-teacher-checkpoint-v1", "policy_role": "global_teacher",
        "run_identity": canonical_hash(teacher_manifest), "sequence": teacher_p["teacher_cycles"] + 1,
        "phase": "complete", "teacher_cycles_completed": teacher_p["teacher_cycles"],
        "teacher_actor_optimizer_steps": teacher_steps, "teacher_critic_optimizer_steps": teacher_steps,
        "initial_model_version": teacher_manifest["initial_model_version"],
        "model_updates": p["initial_model_updates"], "real_joint_transitions": p["initial_transitions"],
        "new_environment_interactions": 0, "new_model_updates": 0,
        "local_student_optimizer_steps": 0, "frozen_for_distillation": True}
    if (teacher_checkpoint_manifest.get("metadata") != expected_teacher_metadata
            or int(teacher_checkpoint.name.split("_")[1]) != expected_teacher_metadata["sequence"]):
        raise ValueError("upstream teacher checkpoint is not its fully completed fixed schedule")
    teacher_label = origin_device(teacher_manifest["runtime"]["logical_device"])
    settings, data, stats, selected, binding = _initial_inputs(**teacher_manifest["inputs"])
    seed = manifest["master_seed"]
    if (teacher_manifest["inputs"]["master_seed"] != seed
            or canonical_hash(binding) != canonical_hash(teacher_manifest.get("input_binding"))
            or manifest.get("teacher_input_identity") != canonical_hash(
                {"manifest": teacher_manifest, "handoff": teacher_handoff, "binding": binding})):
        raise ValueError("student teacher, replay, normalization or selected model identity changed")
    model = build_model(settings, master_seed=seed, variant="D")
    model.load_state_dict(selected["model"], strict=True)
    del selected
    if (module_version_at_device(model, teacher_label)
            != teacher_handoff.get("model_version")
            or teacher_handoff.get("model_version") != teacher_manifest.get("initial_model_version")
            or module_version_at_device(model, label) != manifest.get("frozen_versions", {}).get("model")):
        raise ValueError("student frozen world model differs from selected P2.4 weights")
    # A teacher version includes the historical device label. Compare it
    # directly when both upstream stages used the same device; different device
    # transfers remain bound by the immutable P4.2 manifest and context identity.
    if (label == teacher_label
            and any(manifest["frozen_versions"].get(key) != teacher_handoff.get(version)
                    for key, version in (("teacher", "teacher_version"), ("teacher_critic", "critic_version")))):
        raise ValueError("distillation frozen teacher identities differ from its completed handoff")
    paired = build_actor_critic(settings, master_seed=seed, variant="D")
    actor, critic = paired.actor, paired.critic
    initial_path = _snapshot(directory)
    initial, initial_meta = load_cycle_checkpoint(initial_path)
    saved, metadata = load_cycle_checkpoint(checkpoint)
    expected_keys = {"controller", "actor", "critic", "learner", "streams", "sampler", "shards",
                     "diagnostics_before", "diagnostics_after", "last_report"}
    if set(initial) != expected_keys or set(saved) != expected_keys:
        raise ValueError("student checkpoint state has missing or unsupported keys")
    for network, key in ((actor, "actor"), (critic, "critic")):
        network.load_state_dict(initial[key], strict=True)
    paired_versions = {key: module_version_at_device(network, label)
                       for key, network in (("actor", actor), ("critic", critic))}
    expected_initial = {"sequence": 1, "phase": "buffer", "trajectories_completed": 0, "imagined_joint_steps": 0}
    if (manifest.get("paired_initial_versions") != paired_versions or initial["controller"] != expected_initial
            or initial_meta != _metadata(manifest, expected_initial, paired_versions["actor"], paired_versions["critic"], 0)
            or initial["shards"] != [] or any(initial[key] is not None for key in
                                            ("diagnostics_before", "diagnostics_after", "last_report"))
            or initial["learner"].get("updates_completed") != 0
            or initial["learner"].get("optimizer", {}).get("state") != {}):
        raise ValueError("student initial checkpoint differs from paired untrained weights")
    for network, key in ((actor, "actor"), (critic, "critic")):
        network.load_state_dict(saved[key], strict=True)
    actor_version, critic_version = (module_version_at_device(network, label) for network in (actor, critic))
    if (critic_version != paired_versions["critic"]
            or manifest["frozen_versions"].get("student_critic") != paired_versions["critic"]):
        raise ValueError("distillation changed the paired initial student critic")
    controller = saved["controller"]
    if (type(controller) is not dict or set(controller) != set(expected_initial)
            or any(type(controller.get(key)) is not int for key in
                   ("sequence", "trajectories_completed", "imagined_joint_steps"))
            or controller.get("phase") != "complete" or controller.get("trajectories_completed") != p["trajectories"]
            or controller.get("sequence") != 1 + p["trajectories"] + p["updates"]
            or type(controller.get("imagined_joint_steps")) is not int
            or not p["trajectories"] <= controller["imagined_joint_steps"] <= p["trajectories"] * p["horizon"]
            or int(checkpoint.name.split("_")[1]) != controller["sequence"]
            or metadata != _metadata(manifest, controller, actor_version, critic_version, p["updates"])):
        raise ValueError("student checkpoint is not a completed fixed-schedule warm start")
    namespaces = {name: derive_seed("relay", seed, name, "D") for name in
                  ("distillation_anchors", "distillation_prefix", "distillation_rollouts", "distillation_updates")}
    if manifest.get("rng_namespaces") != namespaces:
        raise ValueError("distillation random namespaces changed")
    expected_stream_identity = RolloutStreams.from_seed(namespaces["distillation_rollouts"]).identity
    expected_stream_identity["device"] = label
    if manifest.get("rollout_stream_identity") != expected_stream_identity:
        raise ValueError("distillation rollout stream namespace differs")
    learner = saved["learner"]
    expected_config = asdict(DistillationConfig(updates=p["updates"], sequences_per_update=p["sequences_per_update"],
        learning_rate=p["learning_rate"], epsilon=p["adam_epsilon"], betas=tuple(p["adam_betas"]),
        weight_decay=p["weight_decay"], gradient_clip=p["gradient_clip"]))
    expected_initial_learner = {"schema_version": "p42-student-optimizer-v1", "config": expected_config,
        "updates_completed": 0, "paired_actor_version": paired_versions["actor"],
        "paired_critic_version": paired_versions["critic"], "current_actor_version": paired_versions["actor"],
        "sampling_seed": namespaces["distillation_updates"], "next_local_ppo_optimizer": "fresh"}
    initial_learner = initial["learner"]
    if (set(initial_learner) != {*expected_initial_learner, "sampling_rng", "optimizer"}
            or any(initial_learner[key] != value for key, value in expected_initial_learner.items())
            or not torch.equal(initial_learner["sampling_rng"],
                               torch.Generator().manual_seed(namespaces["distillation_updates"]).get_state())
            or not _tree_equal(initial["sampler"], saved["sampler"])):
        raise ValueError("student initial optimizer or sampler namespaces differ")
    if label == "cpu" and not _tree_equal(initial["streams"],
            RolloutStreams.from_seed(namespaces["distillation_rollouts"]).state_dict()):
        raise ValueError("student initial rollout RNG differs from its independent namespace")
    if (set(learner) != {"schema_version", "config", "updates_completed", "paired_actor_version", "paired_critic_version",
                        "current_actor_version", "sampling_seed", "sampling_rng", "optimizer", "next_local_ppo_optimizer"}
            or learner["schema_version"] != "p42-student-optimizer-v1" or learner["config"] != expected_config
            or learner["updates_completed"] != p["updates"] or learner["current_actor_version"] != actor_version
            or learner["paired_actor_version"] != paired_versions["actor"]
            or learner["paired_critic_version"] != paired_versions["critic"]
            or learner["sampling_seed"] != namespaces["distillation_updates"]
            or learner["next_local_ppo_optimizer"] != "fresh"):
        raise ValueError("student completed supervised optimizer identity or counters differ")
    _adam_counters(learner["optimizer"], actor, p["updates"], p["learning_rate"], p["adam_epsilon"])
    if any(learner["optimizer"]["param_groups"][0].get(key, False) is not False
           for key in ("capturable", "differentiable", "decoupled_weight_decay")):
        raise ValueError("student supervised Adam configuration differs")
    rng = torch.Generator(device="cpu").manual_seed(namespaces["distillation_updates"])
    # Reproduce the exact fixed-buffer minibatch index draws, without a network
    # query or optimizer. This verifies the completed supervised sampling stream.
    for _ in range(p["updates"]):
        last_indices = torch.randperm(p["trajectories"], generator=rng)[:p["sequences_per_update"]].tolist()
    if not torch.equal(rng.get_state(), learner["sampling_rng"]):
        raise ValueError("student completed supervised sampling RNG differs")
    sampler = AnchorSampler.from_experiment([data], settings, seed=namespaces["distillation_anchors"])
    shards = _audit_shards(directory, manifest, saved, initial, sampler)
    for name, updates in (("diagnostics_before", 0), ("diagnostics_after", p["updates"])):
        _diagnostics(saved[name], updates=updates, trajectories=p["trajectories"],
                     joint_steps=controller["imagined_joint_steps"])
    last = saved["last_report"]
    if (type(last) is not dict or last.get("schema_version") != "p42-student-update-v1"
            or last.get("optimizer_update") != p["updates"]
            or last.get("sequence_indices") != last_indices
            or any(last.get(key) != 0 for key in ("teacher_optimizer_steps", "critic_optimizer_steps",
                                                 "model_optimizer_steps", "new_environment_interactions"))):
        raise ValueError("student final supervised update report differs from completed work")
    expected_handoff = {"schema_version": "p42-warm-started-student-v1", "run_identity": canonical_hash(manifest),
        "checkpoint": checkpoint.relative_to(directory).as_posix(),
        "checkpoint_manifest_sha256": file_sha256(checkpoint / "manifest.json"),
        "actor_version": actor_version, "critic_version": critic_version, "actor_config": asdict(actor.config),
        "paired_initial_versions": paired_versions, "teacher_input_identity": manifest["teacher_input_identity"],
        "frozen_versions": manifest["frozen_versions"], "trajectories": p["trajectories"],
        "imagined_joint_steps": controller["imagined_joint_steps"], "distillation_optimizer_steps": p["updates"],
        "diagnostics_before": saved["diagnostics_before"], "diagnostics_after": saved["diagnostics_after"],
        "diagnostic_scope": "training buffer only; not validation or real performance",
        "fresh_local_ppo_optimizers_required": True, "local_ppo_optimizer_steps": 0,
        "new_environment_interactions": 0, "new_model_updates": 0,
        "full_d_pipeline_implemented": False, "phase4_acceptance": False}
    if canonical_hash(handoff) != canonical_hash(expected_handoff):
        raise ValueError("warm-started student handoff differs from completed checkpoint")
    target = selected_device(device, manifest["precision"])
    for network in (model, actor, critic):
        network.zero_grad(set_to_none=True)
        network.eval().to(target)
    model.requires_grad_(False)
    actor.requires_grad_(True)
    critic.requires_grad_(True)
    provenance = {"schema_version": "p43-completed-student-input-v1", "run_identity": canonical_hash(manifest),
        "run_manifest_sha256": file_sha256(directory / "run_manifest.json"),
        "warm_started_student_sha256": file_sha256(directory / "warm_started_student.json"),
        "checkpoint": checkpoint.relative_to(directory).as_posix(),
        "checkpoint_manifest_sha256": file_sha256(checkpoint / "manifest.json"),
        "checkpoint_payload_sha256": file_sha256(checkpoint / "state.pt"),
        "initial_snapshot": {"checkpoint": initial_path.relative_to(directory).as_posix(),
            "checkpoint_manifest_sha256": file_sha256(initial_path / "manifest.json"),
            "checkpoint_payload_sha256": file_sha256(initial_path / "state.pt")},
        "teacher_run": str(teacher_directory), "teacher_run_manifest_sha256": file_sha256(teacher_directory / "run_manifest.json"),
        "teacher_checkpoint": teacher_checkpoint.relative_to(teacher_directory).as_posix(),
        "frozen_teacher_sha256": file_sha256(teacher_directory / "frozen_teacher.json"),
        "teacher_checkpoint_manifest_sha256": file_sha256(teacher_checkpoint / "manifest.json"),
        "teacher_checkpoint_payload_sha256": file_sha256(teacher_checkpoint / "state.pt"),
        "verified_origin_device": label, "handoff_device": str(target), "buffer_shards": shards,
        "paired_initial_versions": paired_versions,
        "imported_network_versions": {key: module_version(network)
                                      for key, network in (("model", model), ("actor", actor), ("critic", critic))},
        "upstream_optimizers_restored": False, "teacher_constructed_or_queried": False,
        "cross_runtime_numerical_equivalence_measured": False}
    teacher_initial = _snapshot(teacher_directory)
    # Keep a compact immutable file list, not the gigabyte-scale training
    # buffer or supervised Adam slots, for subsequent invocation guards.
    paths = [directory / "run_manifest.json", directory / "run_manifest.sha256",
             directory / "warm_started_student.json", directory / "warm_started_student.sha256",
             teacher_directory / "run_manifest.json", teacher_directory / "run_manifest.sha256",
             teacher_directory / "frozen_teacher.json", teacher_directory / "frozen_teacher.sha256"]
    for publication in (checkpoint, initial_path, teacher_checkpoint, teacher_initial,
                        *(directory / row["path"] for row in shards)):
        paths.extend(publication / name for name in ("manifest.json", "manifest.sha256", "state.pt"))
    # Teacher initial publication is not needed as a live module; verify its
    # receipt and bytes before retaining its immutable binding.
    teacher_initial_manifest = _receipt_json(teacher_initial / "manifest.json")
    teacher_initial_payload = teacher_initial_manifest.get("payload", {})
    if (teacher_initial_payload.get("path") != "state.pt"
            or (teacher_initial / "state.pt").stat().st_size != teacher_initial_payload.get("bytes")
            or file_sha256(teacher_initial / "state.pt") != teacher_initial_payload.get("sha256")):
        raise ValueError("upstream teacher initial checkpoint payload differs")
    expected_teacher_initial = {**expected_teacher_metadata, "sequence": 1, "phase": "teacher",
        "teacher_cycles_completed": 0, "teacher_actor_optimizer_steps": 0,
        "teacher_critic_optimizer_steps": 0, "frozen_for_distillation": False}
    if teacher_initial_manifest.get("metadata") != expected_teacher_initial:
        raise ValueError("upstream teacher initial checkpoint binding differs")
    provenance["bound_files"] = [{"path": str(path), "sha256": file_sha256(path)} for path in paths]
    provenance["initial_inputs"] = deepcopy(teacher_manifest["inputs"])
    if (_latest(directory) != checkpoint
            or canonical_hash(_receipt_json(directory / "run_manifest.json")) != canonical_hash(manifest)
            or canonical_hash(_receipt_json(directory / "warm_started_student.json")) != canonical_hash(handoff)
            or _latest(teacher_directory) != teacher_checkpoint
            or canonical_hash(_receipt_json(teacher_directory / "run_manifest.json")) != canonical_hash(teacher_manifest)
            or canonical_hash(_receipt_json(teacher_directory / "frozen_teacher.json")) != canonical_hash(teacher_handoff)):
        raise ValueError("completed student publication changed during import")
    return WarmStartedStudent(directory, manifest, handoff, checkpoint, settings, seed, manifest["precision"],
                              model, actor, critic, data, stats, binding, provenance)


def verify_frozen_inputs(context):
    """Recheck upstream publications and physical fitting data before an invocation.

    No optimizer or teacher object is created. This checks completed input
    publications, not status files, and never updates their historical receipts.
    """
    for row in context.provenance["bound_files"]:
        path = _root(row["path"])
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise ValueError(f"completed student or teacher input changed: {path}")
    _source_compatibility(context.manifest["source_sha256"], stage="distillation")
    teacher_manifest = _receipt_json(_root(context.provenance["teacher_run"]) / "run_manifest.json")
    _source_compatibility(teacher_manifest["source_sha256"], stage="teacher")
    settings, data, stats, payload, binding = _initial_inputs(**context.provenance["initial_inputs"])
    del settings, data, stats, payload
    if canonical_hash(binding) != canonical_hash(context.binding):
        raise ValueError("student frozen replay, normalization, selection or settings changed")
    if _latest(context.directory) != context.checkpoint:
        raise ValueError("completed student publication advanced after import")
    if (_latest(_root(context.provenance["teacher_run"]))
            != _root(Path(context.provenance["teacher_run"]) / context.provenance["teacher_checkpoint"])):
        raise ValueError("completed teacher publication advanced after import")
