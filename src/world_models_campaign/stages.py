"""Adapters for immutable P4.1/P4.2/P4.3 stages, without changing their sources.

The enclosing controller owns scheduling. Only ``run_one`` executes scientific
work; input validation, initial snapshots and inspection never train or sample.
Historical runs are imported explicitly, with their original chronology intact.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch

from world_models.cycle.checkpoints import load_cycle_checkpoint, save_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.distillation.versions import module_version_at_device
from world_models.imagination.versions import module_version
from world_models.infrastructure.reproducibility import canonical_hash, derive_seed
from world_models.policies.initialization import build_actor_critic
from world_models.teacher.runner import _initial_inputs, _latest, _root


NAMES = ("teacher", "buffer", "distillation", "local_cycle")


def _name(name):
    if name not in NAMES:
        raise ValueError("unknown D campaign stage")
    return name


def _json(path):
    path = _root(path)
    receipt = path.with_suffix(".sha256")
    if (not path.is_file() or not receipt.is_file() or receipt.is_symlink()
            or receipt.read_text(encoding="ascii").strip() != f"{file_sha256(path)}  {path.name}"):
        raise ValueError(f"campaign stage receipt mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _initial(directory):
    matches = [path for path in (directory / "checkpoints").glob("state_00000001_*")
               if (path / "manifest.sha256").is_file()]
    if len(matches) != 1:
        raise ValueError("campaign requires one committed initial student checkpoint")
    return _root(matches[0])


def validate_initial_inputs(inputs):
    """Return the existing strict five-part shared-model input context."""
    if type(inputs) is not dict:
        raise ValueError("campaign initial inputs must be a dictionary")
    return _initial_inputs(**inputs)


def _pair(settings, seed):
    student = build_actor_critic(settings, master_seed=seed, variant="D")
    reference_settings = deepcopy(settings)
    reference_settings["variant"] = {"name": "B", "teacher_enabled": False,
                                      "distillation_enabled": False}
    reference = build_actor_critic(reference_settings, master_seed=seed, variant="B")
    for name in ("actor", "critic"):
        if module_version(getattr(student, name)) != module_version(getattr(reference, name)):
            raise ValueError("campaign B/D local initialization is not bit-identical")
    return student


def save_initial_student(directory, *, inputs, mode="fresh", distillation_run=None):
    """Publish actor/critic values before fresh teacher work, or honest adoption.

    Adopted values come from the historical P4.2 sequence-one checkpoint, never
    from random regeneration under a different PyTorch build. This mode does
    not claim that a new snapshot existed before the historical teacher ran.
    """
    directory = _root(directory)
    if directory.exists():
        raise FileExistsError("campaign initial student needs a fresh directory")
    if mode not in ("fresh", "adopted") or (mode == "fresh" and distillation_run is not None):
        raise ValueError("unsupported campaign initial-student mode")
    settings, _, _, payload, binding = validate_initial_inputs(inputs)
    del payload
    seed = inputs["master_seed"]
    if type(seed) is not int or seed not in (0, 1):
        raise ValueError("campaign student requires registered seed 0 or 1")
    provenance = {"mode": mode, "saved_before_teacher_training": mode == "fresh",
                  "chronology": "fresh_before_teacher" if mode == "fresh" else
                                "historical_p42_initial_snapshot_after_teacher"}
    if mode == "fresh":
        paired = _pair(settings, seed)
        state = {name: getattr(paired, name).state_dict() for name in ("actor", "critic")}
        versions = {name: module_version(getattr(paired, name)) for name in state}
    else:
        if distillation_run is None:
            raise ValueError("adopted initial values require their historical P4.2 run")
        distillation_run = _root(distillation_run)
        manifest = _json(distillation_run / "run_manifest.json")
        snapshot = _initial(distillation_run)
        initial, _ = load_cycle_checkpoint(snapshot)
        if (manifest.get("schema_version") != "p42-distillation-run-v1"
                or manifest.get("master_seed") != seed
                or initial.get("learner", {}).get("updates_completed") != 0
                or initial.get("learner", {}).get("optimizer", {}).get("state") != {}):
            raise ValueError("historical student snapshot is not an untrained paired actor/critic")
        paired = build_actor_critic(settings, master_seed=seed, variant="D")
        state = {name: initial[name] for name in ("actor", "critic")}
        for name in state:
            getattr(paired, name).load_state_dict(state[name], strict=True)
        origin = manifest["runtime"]["logical_device"]
        if manifest.get("paired_initial_versions") != {
                name: module_version_at_device(getattr(paired, name), origin) for name in state}:
            raise ValueError("historical paired student snapshot has different registered weights")
        versions = {name: module_version(getattr(paired, name)) for name in state}
        provenance.update({"distillation_run": str(distillation_run),
            "initial_checkpoint": str(snapshot),
            "initial_checkpoint_manifest_sha256": file_sha256(snapshot / "manifest.json"),
            "initial_checkpoint_payload_sha256": file_sha256(snapshot / "state.pt")})
    metadata = {"schema_version": "p44-paired-initial-student-v1", "master_seed": seed,
        "inputs": deepcopy(inputs), "input_binding": deepcopy(binding), "paired_versions_cpu": versions,
        "actor_config": json.loads(json.dumps(asdict(paired.actor.config))), "provenance": provenance,
        "initialization_seeds": {name: derive_seed("relay", seed, name, "D") for name in
                                 ("local_actor_initialization", "local_critic_initialization")},
        "actor_optimizer_steps": 0, "critic_optimizer_steps": 0,
        "new_environment_interactions": 0, "new_model_updates": 0}
    save_cycle_checkpoint(directory, state=state, metadata=metadata)
    return verify_initial_student(directory)


def verify_initial_student(directory, *, distillation_run=None):
    directory = _root(directory)
    saved, metadata = load_cycle_checkpoint(directory)
    if (set(saved) != {"actor", "critic"}
            or metadata.get("schema_version") != "p44-paired-initial-student-v1"
            or type(metadata.get("master_seed")) is not int
            or metadata["master_seed"] not in (0, 1)
            or metadata.get("inputs", {}).get("master_seed") != metadata["master_seed"]
            or any(metadata.get(key) != 0 for key in ("actor_optimizer_steps", "critic_optimizer_steps",
                    "new_environment_interactions", "new_model_updates"))):
        raise ValueError("campaign initial student is malformed or already trained")
    provenance = metadata.get("provenance", {})
    mode = provenance.get("mode")
    if (mode not in ("fresh", "adopted")
            or provenance.get("saved_before_teacher_training") is not (mode == "fresh")
            or metadata.get("initialization_seeds") != {
                name: derive_seed("relay", metadata["master_seed"], name, "D") for name in
                ("local_actor_initialization", "local_critic_initialization")}):
        raise ValueError("campaign initial student chronology or initialization namespace differs")
    settings, _, _, payload, binding = validate_initial_inputs(metadata["inputs"])
    del payload
    if canonical_hash(binding) != canonical_hash(metadata.get("input_binding")):
        raise ValueError("campaign initial replay/model/normalization changed")
    paired = build_actor_critic(settings, master_seed=metadata["master_seed"], variant="D")
    for name in saved:
        getattr(paired, name).load_state_dict(saved[name], strict=True)
    if (metadata.get("paired_versions_cpu") != {name: module_version(getattr(paired, name)) for name in saved}
            or canonical_hash(metadata.get("actor_config")) != canonical_hash(asdict(paired.actor.config))):
        raise ValueError("campaign paired initial values changed")
    if distillation_run is not None:
        stage = _root(distillation_run)
        initial, _ = load_cycle_checkpoint(_initial(stage))
        for name in saved:
            if (set(saved[name]) != set(initial[name])
                    or any(not torch.equal(value, initial[name][key]) for key, value in saved[name].items())):
                raise ValueError("distillation initial student differs from saved pre-teacher paired values")
    return {"path": str(directory), "manifest_sha256": file_sha256(directory / "manifest.json"),
            "payload_sha256": file_sha256(directory / "state.pt"), "metadata": metadata}


def prepare_stage(name, directory, *, inputs, device="cuda:0", precision="bfloat16",
                  teacher_run=None, distillation_run=None):
    _name(name)
    if name == "teacher":
        from world_models.teacher.runner import TeacherRunner
        return TeacherRunner.prepare(directory, **inputs, device=device, precision=precision)
    if name in ("buffer", "distillation"):
        from world_models.distillation.runner import DistillationRunner
        if teacher_run is None:
            raise ValueError("buffer/distillation stage needs its completed teacher")
        return DistillationRunner.prepare(directory, teacher_run=teacher_run, device=device)
    from world_models.refinement.runner import RefinementRunner
    if distillation_run is None:
        raise ValueError("local stage needs completed action distillation")
    return RefinementRunner.prepare(directory, distillation_run=distillation_run, device=device)


def resume_stage(name, directory, *, device="cuda:0"):
    _name(name)
    if name == "teacher":
        from world_models.teacher.runner import TeacherRunner
        return TeacherRunner.resume(directory, device=device)
    if name in ("buffer", "distillation"):
        from world_models.distillation.runner import DistillationRunner
        return DistillationRunner.resume(directory, device=device)
    from world_models.refinement.runner import RefinementRunner
    return RefinementRunner.resume(directory, device=device)


def run_one(name, worker):
    """Commit exactly one logical unit using the stage's original optimizer/RNG."""
    _name(name)
    expected = {"teacher": "teacher", "buffer": "buffer", "distillation": "distillation", "local_cycle": "local"}
    if worker.state["phase"] not in (expected[name], "complete"):
        raise ValueError("campaign attempted work in the wrong stage or after completion")
    if name in ("buffer", "distillation"):
        return worker.run(max_units=1)
    return worker.run(max_cycles=1)


def _publication(path):
    _json(path)
    return [{"path": str(path), "sha256": file_sha256(path)},
            {"path": str(path.with_suffix(".sha256")), "sha256": file_sha256(path.with_suffix(".sha256"))}]


def _checkpoint_publications(directory):
    """Audit payload bytes without instantiating its model/optimizer classes."""
    manifest = _json(directory / "manifest.json")
    specification = manifest.get("payload", {})
    path = directory / "state.pt"
    if (specification.get("path") != "state.pt" or path.is_symlink() or not path.is_file()
            or specification.get("bytes") != path.stat().st_size
            or specification.get("sha256") != file_sha256(path)):
        raise ValueError("campaign stage checkpoint payload changed")
    return _publication(directory / "manifest.json") + [
        {"path": str(path), "sha256": specification["sha256"]}]


def _physical(directory, name):
    """Keep torn journal bytes as unknown physical work, never a restart block.

    Durable checkpoints establish logical progress. A killed append can damage
    a journal row independently of that progress; this parser records the exact
    damaged locations and hashes the bytes it inspected without repairing them.
    """
    path = directory / "attempts.jsonl"
    if not path.exists():
        return {"journal": str(path), "started_units": 0, "committed_units": 0,
                "unmatched_attempts": [], "recorded_elapsed_seconds": 0.0,
                "malformed_rows": 0, "malformed_locations": [],
                "unattributed_orphan_commits": [], "unmeasured_elapsed_commits": 0,
                "partial_interrupted_compute": "none recorded"}
    raw, rows, malformed, offset = path.read_bytes(), [], [], 0
    for number, line in enumerate(raw.splitlines(keepends=True), start=1):
        try:
            row = json.loads(line)
            if type(row) is not dict or type(row.get("event")) is not str:
                raise ValueError("journal row has no event")
            if (row["event"].endswith(("_started", "_committed"))
                    and (type(row.get("attempt")) is not str or not row["attempt"])):
                raise ValueError("journal work row has no attempt identity")
        except (ValueError, UnicodeError):
            malformed.append({"line": number, "byte_offset": offset, "bytes": len(line)})
        else:
            if name not in ("buffer", "distillation") or row.get("phase") == name:
                rows.append((number, row))
        offset += len(line)
    started = [row for _, row in rows if row["event"].endswith("_started")]
    committed = [row for _, row in rows if row["event"].endswith("_committed")]
    starts, commits = {row["attempt"] for row in started}, {row["attempt"] for row in committed}
    if len(starts) != len(started) or len(commits) != len(committed):
        raise ValueError("stage physical-work attempt identities are inconsistent")
    orphan_rows = [(number, row) for number, row in rows
                   if row["event"].endswith("_committed") and row["attempt"] not in starts]
    if any(not any(damage["line"] < number for damage in malformed) for number, _ in orphan_rows):
        raise ValueError("stage physical-work attempt identities are inconsistent")
    missing = sorted(starts - commits)
    elapsed, unmeasured = 0.0, 0
    for row in committed:
        duration = row.get("elapsed_seconds", row.get("elapsed_seconds_including_checkpoint"))
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            unmeasured += 1
        else:
            elapsed += duration
    unknown = bool(missing or malformed or orphan_rows or unmeasured)
    return {"journal": str(path), "journal_sha256": hashlib.sha256(raw).hexdigest(), "started_units": len(starts),
        "committed_units": len(commits), "unmatched_attempts": missing,
        "recorded_elapsed_seconds": elapsed, "malformed_rows": len(malformed),
        "malformed_locations": malformed,
        "unattributed_orphan_commits": sorted(row["attempt"] for _, row in orphan_rows),
        "unmeasured_elapsed_commits": unmeasured,
        "partial_interrupted_compute": "unknown; not counted as zero" if unknown else "none recorded"}


def inspect_stage(name, directory):
    """Verify durable checkpoint counters; never trust an ahead status.json."""
    _name(name)
    directory = _root(directory)
    manifest = _json(directory / "run_manifest.json")
    checkpoint = _latest(directory)
    saved, metadata = load_cycle_checkpoint(checkpoint)
    p = manifest["protocol"]
    if manifest.get("protocol_sha256") != canonical_hash(p):
        raise ValueError("campaign stage protocol receipt binding differs")
    if metadata.get("run_identity") != canonical_hash(manifest):
        raise ValueError("campaign stage checkpoint belongs to a different manifest")
    controller = saved["controller"]
    if (metadata.get("sequence") != controller.get("sequence")
            or metadata.get("phase") != controller.get("phase")
            or int(checkpoint.name.split("_")[1]) != controller.get("sequence")):
        raise ValueError("campaign stage controller sequence or phase differs from its checkpoint")
    from world_models.refinement.handoff import _source_compatibility
    if name in ("teacher", "buffer", "distillation"):
        _source_compatibility(manifest.get("source_sha256"),
                              stage="teacher" if name == "teacher" else "distillation")
    else:
        from world_models.refinement.runner import source_identity
        if manifest.get("source_sha256") != source_identity():
            raise ValueError("campaign refinement sources changed")
    counters = {"model_updates": p["initial_model_updates"],
        "real_joint_transitions": p["initial_transitions"], "new_model_updates": 0,
        "new_environment_interactions": 0, "teacher_actor_optimizer_steps": 0,
        "teacher_critic_optimizer_steps": 0, "distillation_optimizer_steps": 0,
        "local_actor_optimizer_steps": 0, "local_critic_optimizer_steps": 0,
        "imagined_joint_steps": saved["controller"]["imagined_joint_steps"]}
    publications = _publication(directory / "run_manifest.json")
    publications += _checkpoint_publications(checkpoint)
    learner = saved["learner"]
    if name == "teacher":
        from world_models.teacher.protocol import load_protocol
        if manifest.get("schema_version") != "p41-teacher-run-v1" or p != load_protocol():
            raise ValueError("campaign teacher protocol differs")
        completed, target = learner["cycles_completed"], p["teacher_cycles"]
        expected = completed * p["epochs"] * p["minibatches"]
        if (metadata.get("teacher_cycles_completed") != completed
                or any(learner[key] != expected or metadata.get(f"teacher_{key}") != expected
                       for key in ("actor_optimizer_steps", "critic_optimizer_steps"))
                or saved["controller"]["sequence"] != completed + 1
                or (controller["phase"] == "complete") != (completed == target)
                or controller.get("teacher_player_decisions") != 2 * controller["imagined_joint_steps"]
                or not completed * p["starts"] <= controller["imagined_joint_steps"]
                       <= completed * p["starts"] * p["horizon"]):
            raise ValueError("campaign teacher counters differ from checkpoint")
        counters.update(teacher_actor_optimizer_steps=expected, teacher_critic_optimizer_steps=expected)
        if (directory / "frozen_teacher.sha256").is_file():
            handoff = _json(directory / "frozen_teacher.json")
            if (completed != target or handoff.get("run_identity") != canonical_hash(manifest)
                    or handoff.get("checkpoint") != checkpoint.relative_to(directory).as_posix()
                    or handoff.get("checkpoint_manifest_sha256") != file_sha256(checkpoint / "manifest.json")
                    or handoff.get("teacher_cycles") != completed or handoff.get("frozen_for_distillation") is not True
                    or handoff.get("model_version") != manifest["initial_model_version"]):
                raise ValueError("campaign frozen teacher publication differs from completed checkpoint")
            publications += _publication(directory / "frozen_teacher.json")
    elif name in ("buffer", "distillation"):
        from world_models.distillation.protocol import load_protocol
        if manifest.get("schema_version") != "p42-distillation-run-v1" or p != load_protocol():
            raise ValueError("campaign distillation protocol differs")
        trajectories = saved["controller"]["trajectories_completed"]
        updates = learner["updates_completed"]
        completed, target = ((trajectories, p["trajectories"]) if name == "buffer" else (updates, p["updates"]))
        if (metadata.get("trajectories_completed") != trajectories
                or metadata.get("distillation_optimizer_steps") != updates
                or saved["controller"]["sequence"] != trajectories + updates + 1
                or len(saved["shards"]) != trajectories
                or not 0 <= trajectories <= p["trajectories"] or not 0 <= updates <= p["updates"]
                or (controller["phase"] == "buffer") != (trajectories < p["trajectories"])
                or (controller["phase"] == "complete") != (updates == p["updates"])
                or not trajectories <= controller["imagined_joint_steps"] <= trajectories * p["horizon"]):
            raise ValueError("campaign buffer/distillation counters differ from checkpoint")
        counters["distillation_optimizer_steps"] = updates
        for shard in saved["shards"]:
            path = _root(directory / shard["path"])
            if not path.is_relative_to(directory / "buffer"):
                raise ValueError("campaign buffer shard escaped its registered directory")
            if file_sha256(path / "manifest.json") != shard["manifest_sha256"]:
                raise ValueError("campaign buffer shard binding changed")
            publications += _checkpoint_publications(path)
        if (directory / "warm_started_student.sha256").is_file():
            handoff = _json(directory / "warm_started_student.json")
            if (updates != p["updates"] or handoff.get("run_identity") != canonical_hash(manifest)
                    or handoff.get("checkpoint") != checkpoint.relative_to(directory).as_posix()
                    or handoff.get("checkpoint_manifest_sha256") != file_sha256(checkpoint / "manifest.json")
                    or handoff.get("actor_version") != metadata.get("actor_version")
                    or handoff.get("critic_version") != metadata.get("critic_version")
                    or handoff.get("trajectories") != trajectories
                    or handoff.get("distillation_optimizer_steps") != updates):
                raise ValueError("campaign warm-start publication differs from completed checkpoint")
            publications += _publication(directory / "warm_started_student.json")
    else:
        from world_models.refinement.protocol import load_protocol
        if manifest.get("schema_version") != "p43-local-refinement-run-v1" or p != load_protocol():
            raise ValueError("campaign local refinement protocol differs")
        completed, target = learner["cycles_completed"], p["local_cycles"]
        expected = completed * p["epochs"] * p["minibatches"]
        if (metadata.get("local_cycles_completed") != completed
                or any(learner[key] != expected or metadata.get(key) != expected
                       for key in ("actor_optimizer_steps", "critic_optimizer_steps"))
                or saved["controller"]["sequence"] != completed + 1
                or (controller["phase"] == "complete") != (completed == target)
                or controller.get("local_player_decisions") != 2 * controller["imagined_joint_steps"]
                or not completed * p["starts"] <= controller["imagined_joint_steps"]
                       <= completed * p["starts"] * p["horizon"]
                or metadata.get("teacher_queries") != 0 or metadata.get("teacher_kl_weight") != 0):
            raise ValueError("campaign local refinement counters or guidance differ")
        counters.update(local_actor_optimizer_steps=expected, local_critic_optimizer_steps=expected)
        if (directory / "refined_student.sha256").is_file():
            handoff = _json(directory / "refined_student.json")
            if (completed != target or handoff.get("run_identity") != canonical_hash(manifest)
                    or handoff.get("checkpoint") != checkpoint.relative_to(directory).as_posix()
                    or handoff.get("checkpoint_manifest_sha256") != file_sha256(checkpoint / "manifest.json")
                    or handoff.get("actor_version") != metadata.get("actor_version")
                    or handoff.get("critic_version") != metadata.get("critic_version")
                    or handoff.get("model_version") != metadata.get("model_version")
                    or handoff.get("local_cycles_completed") != completed
                    or handoff.get("actor_optimizer_steps") != expected
                    or handoff.get("critic_optimizer_steps") != expected
                    or handoff.get("warm_start_input_identity") != manifest.get("warm_start_input_identity")
                    or any(handoff.get(key) != 0 for key in
                           ("teacher_kl_weight", "teacher_queries", "new_model_updates", "new_environment_interactions"))):
                raise ValueError("campaign refined student publication differs from completed checkpoint")
            publications += _publication(directory / "refined_student.json")
            export = _root(directory / handoff["actor_export"])
            if not export.is_relative_to(directory):
                raise ValueError("campaign actor export escaped its refinement stage")
            exported, export_metadata = load_cycle_checkpoint(export)
            if (export_metadata.get("schema_version") != "p35-local-actor-v1"
                    or export_metadata.get("actor_version") != handoff.get("actor_version")
                    or export_metadata.get("world_model_version") != metadata.get("model_version")
                    or export_metadata.get("variant") != "D" or export_metadata.get("task_family") != "relay"
                    or export_metadata.get("run_seed") != manifest.get("master_seed")
                    or export_metadata.get("local_ppo_cycles") != completed
                    or export_metadata.get("actor_optimizer_steps") != expected
                    or export_metadata.get("critic_optimizer_steps") != expected
                    or handoff.get("actor_export_manifest_sha256") != file_sha256(export / "manifest.json")
                    or set(exported) != {"actor"} or set(exported["actor"]) != set(saved["actor"])
                    or any(not torch.equal(value, exported["actor"][key]) for key, value in saved["actor"].items())):
                raise ValueError("campaign final actor export differs from local checkpoint")
            publications += _checkpoint_publications(export)
    if (type(completed) is not int or not 0 <= completed <= target
            or metadata.get("new_environment_interactions") != 0 or metadata.get("new_model_updates") != 0
            or checkpoint != _latest(directory)):
        raise ValueError("campaign stage completion or frozen-input accounting differs")
    return {"stage": name, "path": str(directory), "completed": completed, "target": target,
        "checkpoint": str(checkpoint), "checkpoint_manifest_sha256": file_sha256(checkpoint / "manifest.json"),
        "manifest_sha256": file_sha256(directory / "run_manifest.json"),
        "manifest_identity": canonical_hash(manifest), "counters": counters,
        "publications": publications, "physical_work": _physical(directory, name)}


def adoption_lineage(refinement_run, *, device="cpu"):
    """Validate a complete existing stage chain without running or rewriting it."""
    directory = _root(refinement_run)
    manifest = _json(directory / "run_manifest.json")
    from world_models.refinement.runner import RefinementRunner
    # Resume validates exact PPO optimizer moments, RNG, current local weights
    # and the full immutable teacher/buffer chain. It does no scientific work.
    worker = RefinementRunner.resume(directory, device=device)
    if worker.state["phase"] != "complete":
        raise ValueError("adoption requires P4.3 to finish; leave its running job unchanged")
    handoff = _json(directory / "refined_student.json")
    if (handoff.get("checkpoint") != worker.last_checkpoint.relative_to(directory).as_posix()
            or handoff.get("checkpoint_manifest_sha256") != file_sha256(worker.last_checkpoint / "manifest.json")):
        raise ValueError("adopted refinement handoff differs from final checkpoint")
    context = worker.context
    teacher = _root(context.provenance["teacher_run"])
    paths = {"teacher": str(teacher), "buffer": str(context.directory),
             "distillation": str(context.directory), "local_cycle": str(directory)}
    evidence = {name: inspect_stage(name, path) for name, path in paths.items()}
    if any(row["completed"] != row["target"] for row in evidence.values()):
        raise ValueError("adoption requires every D training stage to be durably complete")
    return {"inputs": deepcopy(context.provenance["initial_inputs"]), "paths": paths,
        "provenance": {"chronological_adoption": "completed_standalone_p41_p42_p43_chain",
            "fresh_cohort_claim": False, "initial_snapshot_created_before_historical_teacher": False,
            "refinement_manifest_identity": canonical_hash(manifest),
            "warm_start_input_identity": manifest["warm_start_input_identity"],
            "input_binding": deepcopy(context.binding), "bound_files": deepcopy(context.provenance["bound_files"]),
            "stage_evidence": evidence, "new_training_executed_during_adoption": False,
            "evaluation_executed": False}}
