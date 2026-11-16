"""Durable P4.2 warm-start, separate from teacher training and local PPO.

Only run() generates synthetic trajectories or optimizes. Immutable shards and
checkpoints publish receipts last; status is informational, never a resume point.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import torch

from world_models.cycle.checkpoints import load_cycle_checkpoint, save_cycle_checkpoint, _guard_real_replay
from world_models.data.ledger import atomic_json, file_sha256
from world_models.imagination import AnchorSampler
from world_models.imagination.replay import AnchorRef
from world_models.imagination.versions import module_version
from world_models.infrastructure.reproducibility import canonical_hash, derive_seed
from world_models.policies.initialization import build_actor_critic
from world_models.rollouts.streams import RolloutStreams
from world_models.teacher.learning import TeacherAnchorBuilder, TeacherRolloutGenerator
from world_models.teacher.runner import _initial_inputs, _latest, _NUMBERED, _publish_receipt, _root, _runtime
from world_models.training.runner import append_event, run_lock
from .buffer import DistillationBuffer
from .handoff import load_completed_teacher
from .learning import StudentDistiller, DistillationConfig
from .protocol import load_protocol, protocol_identity
from .versions import module_version_at_device


ROOT = Path(__file__).resolve().parents[3]
RUN_SCHEMA = "p42-distillation-run-v1"


def source_identity():
    paths = sorted((ROOT / "src/world_models").rglob("*.py"))
    paths += [ROOT / "scripts/phase41.py", ROOT / "scripts/phase42.py"]
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}


def _read_receipted(path):
    receipt = path.with_suffix(".sha256")
    if (not path.is_file() or not receipt.is_file()
            or receipt.read_text(encoding="ascii").strip() != f"{file_sha256(path)}  {path.name}"):
        raise ValueError(f"missing or invalid receipt: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(directory):
    result = _read_receipted(directory / "run_manifest.json")
    if (result.get("schema_version") != RUN_SCHEMA or result.get("protocol") != load_protocol()
            or result.get("protocol_sha256") != protocol_identity()):
        raise ValueError("distillation protocol or run schema changed")
    return result


def _context_identity(context):
    return canonical_hash({"manifest": context.manifest, "handoff": context.handoff,
                           "binding": context.binding})


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def _paired_students(context):
    """Construct B and D independently; require bit-identical paired starts."""
    settings = deepcopy(context.settings)
    student = build_actor_critic(settings, master_seed=context.master_seed, variant="D")
    settings["variant"] = {"name": "B", "teacher_enabled": False, "distillation_enabled": False}
    reference = build_actor_critic(settings, master_seed=context.master_seed, variant="B")
    expected = {key: module_version(getattr(reference, key)) for key in ("actor", "critic")}
    if expected != {key: module_version(getattr(student, key)) for key in expected}:
        raise ValueError("B and D initial actor/critic hashes differ")
    device = next(context.model.parameters()).device
    student.to(device)
    # B/D equality above is checked before transfer. The learner's identities
    # must then describe its actual device, because the historical hash includes
    # device metadata. Verify the reference bytes with the target label too.
    expected = {key: module_version_at_device(getattr(reference, key), str(device)) for key in expected}
    if expected != {key: module_version(getattr(student, key)) for key in expected}:
        raise ValueError("paired student weights changed during device transfer")
    return student, expected


class DistillationRunner:
    """Fixed teacher buffer then student-only Adam; no local PPO optimizer."""

    def __init__(self, directory, context):
        self.directory, self.context = _root(directory), context
        self.device = next(context.model.parameters()).device
        self.settings, self.master_seed = context.settings, context.master_seed
        self.protocol = load_protocol()
        if (type(self.master_seed) is not int or self.master_seed not in self.protocol["master_seeds"]
                or any(context.manifest["protocol"].get(key) != self.protocol[key] for key in
                       ("teacher_cycles", "initial_transitions", "initial_model_updates"))):
            raise ValueError("completed teacher differs from the P4.2 fixed-study inputs")
        paired, self.paired_versions = _paired_students(context)
        self.actor, self.critic = paired.actor, paired.critic
        self.critic.requires_grad_(False).eval()
        self.critic.zero_grad(set_to_none=True)
        self.frozen_versions = {name: module_version(network) for name, network in self._frozen().items()}
        self.namespaces = {name: derive_seed("relay", self.master_seed, name, "D") for name in (
            "distillation_anchors", "distillation_prefix", "distillation_rollouts", "distillation_updates")}
        self.sampler = AnchorSampler.from_experiment([context.dataset], self.settings,
                                                     seed=self.namespaces["distillation_anchors"])
        self.builder = TeacherAnchorBuilder(context.model, context.teacher,
            seed=self.namespaces["distillation_prefix"], precision=context.precision)
        self.generator = TeacherRolloutGenerator(context.model, context.teacher, context.critic,
            horizon=self.protocol["horizon"], precision=context.precision)
        self.streams = RolloutStreams.from_seed(self.namespaces["distillation_rollouts"], device=self.device)
        self.learner = StudentDistiller(self.actor, critic=self.critic,
            paired_actor_version=self.paired_versions["actor"], paired_critic_version=self.paired_versions["critic"],
            seed=self.namespaces["distillation_updates"], config=DistillationConfig(
                updates=self.protocol["updates"], sequences_per_update=self.protocol["sequences_per_update"],
                learning_rate=self.protocol["learning_rate"], epsilon=self.protocol["adam_epsilon"],
                betas=tuple(self.protocol["adam_betas"]), weight_decay=self.protocol["weight_decay"],
                gradient_clip=self.protocol["gradient_clip"]))
        self.state = {"sequence": 0, "phase": "buffer", "trajectories_completed": 0, "imagined_joint_steps": 0}
        self.shards, self.references = [], ()
        self.diagnostics_before = self.diagnostics_after = self.last_report = None
        self.manifest, self.last_checkpoint, self.buffer = {}, None, None
        self._failed = self._stop_requested = False

    def _frozen(self):
        return {"model": self.context.model, "teacher": self.context.teacher,
                "teacher_critic": self.context.critic, "student_critic": self.critic}

    @classmethod
    def prepare(cls, directory, *, teacher_run, device="cuda:0"):
        directory = _root(directory)
        _guard_real_replay(directory)
        if directory.exists():
            raise FileExistsError("distillation prepare requires a fresh output directory")
        context = load_completed_teacher(teacher_run, device=device)
        if directory.is_relative_to(context.directory) or context.directory.is_relative_to(directory):
            raise ValueError("distillation and teacher directories must be separate")
        worker = cls(directory, context)
        # Draw the complete quota once: sampling single-item batches would round
        # every recent fraction of 0.5 upward and change the registered mixture.
        worker.references = worker.sampler.sample(worker.protocol["trajectories"])
        worker.manifest = {"schema_version": RUN_SCHEMA, "protocol": worker.protocol,
            "protocol_sha256": protocol_identity(), "source_sha256": source_identity(),
            "runtime": _runtime(worker.device), "teacher_run": str(context.directory),
            "teacher_input_identity": _context_identity(context), "master_seed": context.master_seed,
            "precision": context.precision, "paired_initial_versions": worker.paired_versions,
            "frozen_versions": worker.frozen_versions, "rng_namespaces": worker.namespaces,
            "rollout_stream_identity": deepcopy(worker.streams.identity),
            "references": [asdict(ref) for ref in worker.references],
            "sampling_rule": "one common uniform/recent quota draw; no reward/event filtering",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_executed_at_prepare": False, "full_d_pipeline_implemented": False,
            "phase4_acceptance": False, "fresh_ppo_optimizer_required_after_warm_start": True}
        directory.mkdir(parents=True, exist_ok=False)
        with run_lock(directory):
            atomic_json(directory / "run_manifest.json", worker.manifest)
            _publish_receipt(directory / "run_manifest.json")
            worker._checkpoint()
            worker._status("prepared")
        return worker

    @classmethod
    def resume(cls, directory, *, device="cuda:0"):
        directory = _root(directory)
        if not directory.is_dir():
            raise ValueError("distillation run does not exist")
        with run_lock(directory):
            manifest = _manifest(directory)
            if manifest["source_sha256"] != source_identity():
                raise ValueError("distillation source changed; restore the prepared checkout")
            context = load_completed_teacher(manifest["teacher_run"], device=device)
            if _context_identity(context) != manifest["teacher_input_identity"]:
                raise ValueError("frozen teacher/input identity changed")
            worker = cls(directory, context)
            if _runtime(worker.device) != manifest["runtime"]:
                raise ValueError("exact distillation resume requires the same logical device and runtime")
            if (worker.paired_versions != manifest["paired_initial_versions"]
                    or worker.frozen_versions != manifest["frozen_versions"]
                    or worker.namespaces != manifest["rng_namespaces"]):
                raise ValueError("paired initialization or frozen weights changed")
            checkpoint = _latest(directory)
            saved, metadata = load_cycle_checkpoint(checkpoint)
            if metadata.get("run_identity") != canonical_hash(manifest):
                raise ValueError("distillation checkpoint belongs to a different run")
            if set(saved) != {"controller", "actor", "critic", "learner", "streams", "sampler", "shards",
                              "diagnostics_before", "diagnostics_after", "last_report"}:
                raise ValueError("distillation checkpoint state keys differ")
            worker.manifest, worker.state = manifest, deepcopy(saved["controller"])
            worker.references = tuple(AnchorRef(**row) for row in manifest["references"])
            if (worker.references != worker.sampler.sample(worker.protocol["trajectories"])
                    or not _tree_equal(worker.sampler.state_dict(), saved["sampler"])):
                raise ValueError("distillation anchors differ from the common seeded sampling rule")
            for ref in worker.references:
                worker.sampler.validate_reference(ref)
            worker.actor.load_state_dict(saved["actor"], strict=True)
            worker.critic.load_state_dict(saved["critic"], strict=True)
            worker.learner.load_state_dict(saved["learner"])
            worker.streams.load_state_dict(saved["streams"])
            worker.sampler.load_state_dict(saved["sampler"])
            worker.shards = deepcopy(saved["shards"])
            for name in ("diagnostics_before", "diagnostics_after", "last_report"):
                setattr(worker, name, deepcopy(saved[name]))
            worker.last_checkpoint = checkpoint
            worker._validate()
            if (metadata != worker._metadata()
                    or int(_NUMBERED.fullmatch(checkpoint.name)[1]) != worker.state["sequence"]
                    or worker.state["sequence"] != 1 + worker.state["trajectories_completed"]
                                                       + worker.learner.updates_completed):
                raise ValueError("distillation checkpoint metadata or sequence differs")
            # Validate every committed shard even before buffer completion.
            worker._load_buffer()
            return worker

    def request_stop(self):
        self._stop_requested = True

    def _validate(self):
        p, s = self.protocol, self.state
        if (set(s) != {"sequence", "phase", "trajectories_completed", "imagined_joint_steps"}
                or s["phase"] not in ("buffer", "distillation", "complete")
                or any(type(s[k]) is not int or s[k] < 0 for k in
                       ("sequence", "trajectories_completed", "imagined_joint_steps"))):
            raise ValueError("invalid distillation controller state")
        count, updates = s["trajectories_completed"], self.learner.updates_completed
        if (not 0 <= count <= p["trajectories"] or len(self.shards) != count
                or len(self.references) != p["trajectories"]
                or not count <= s["imagined_joint_steps"] <= count * p["horizon"]
                or not 0 <= updates <= p["updates"]
                or (s["phase"] == "buffer" and (count == p["trajectories"] or updates != 0))
                or (s["phase"] != "buffer" and count != p["trajectories"])
                or (s["phase"] == "complete") != (updates == p["updates"])):
            raise ValueError("distillation stage and optimizer/trajectory accounting differ")
        if (updates == 0 and module_version(self.actor) != self.paired_versions["actor"]):
            raise ValueError("untrained student differs from B paired initialization")
        if ((s["phase"] == "buffer") != (self.diagnostics_before is None)
                or (s["phase"] == "complete") != (self.diagnostics_after is not None)):
            raise ValueError("fixed-buffer before/after diagnostics are missing or premature")
        for name, network in self._frozen().items():
            if (module_version(network) != self.frozen_versions[name]
                    or any(v.requires_grad or v.grad is not None for v in network.parameters())):
                raise ValueError(f"frozen {name} changed or acquired gradients")
        if self.streams.identity != self.manifest["rollout_stream_identity"]:
            raise ValueError("distillation rollout sampling namespace changed")

    def _metadata(self):
        return {"schema_version": "p42-distillation-checkpoint-v1", "run_identity": canonical_hash(self.manifest),
            **deepcopy(self.state), "distillation_optimizer_steps": self.learner.updates_completed,
            "actor_version": module_version(self.actor), "critic_version": module_version(self.critic),
            "new_environment_interactions": 0, "new_model_updates": 0,
            "local_ppo_optimizer_steps": 0, "teacher_optimizer_steps": 0}

    def _checkpoint(self):
        self._validate()
        self.state["sequence"] += 1
        path = self.directory / "checkpoints" / f"state_{self.state['sequence']:08d}_{uuid.uuid4().hex[:8]}"
        saved = {"controller": deepcopy(self.state), "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(), "learner": self.learner.state_dict(),
            "streams": self.streams.state_dict(), "sampler": self.sampler.state_dict(),
            "shards": deepcopy(self.shards), "diagnostics_before": self.diagnostics_before,
            "diagnostics_after": self.diagnostics_after, "last_report": self.last_report}
        save_cycle_checkpoint(path, state=saved, metadata=self._metadata())
        self.last_checkpoint = path

    def _shard_metadata(self, index):
        return {"schema_version": "p42-buffer-shard-v1", "run_identity": canonical_hash(self.manifest),
            "trajectory_index": index, "reference": asdict(self.references[index]),
            "frozen_versions": self.frozen_versions, "learning_eligible_for_world_model": False,
            "on_policy_for_student": False, "data_source": "teacher_world_model_imagination"}

    def _load_buffer(self):
        if not self.shards:
            return None
        parts = []
        previous_streams = RolloutStreams.from_seed(self.namespaces["distillation_rollouts"],
                                                    device=self.device).state_dict()
        for index, row in enumerate(self.shards):
            member = Path(row["path"])
            path = _root(self.directory / member)
            if (member.is_absolute() or not path.is_relative_to(self.directory / "buffer")
                    or file_sha256(path / "manifest.json") != row["manifest_sha256"]):
                raise ValueError("buffer shard path or receipt binding differs")
            saved, metadata = load_cycle_checkpoint(path)
            if metadata != self._shard_metadata(index) or set(saved) != {"buffer"}:
                raise ValueError("buffer shard belongs to another run, teacher or anchor")
            part = DistillationBuffer.from_state_dict(saved["buffer"])
            if part.batch_size != 1 or not 1 <= part.steps <= self.protocol["horizon"]:
                raise ValueError("buffer shard trajectory dimensions differ")
            if ((part.cutoff.any().item() and part.steps != self.protocol["horizon"])
                    or self.references[index].decision_index + part.imagined_joint_steps > self.sampler.deadline):
                raise ValueError("buffer cutoff or decision clock differs from the registered horizon/deadline")
            if (part.prefixes.references != (self.references[index],)
                    or part.metadata["teacher_version"] != self.frozen_versions["teacher"]
                    or part.metadata["world_model_version"] != self.frozen_versions["model"]
                    or part.metadata["task_family"] != self.sampler.task_family
                    or part.metadata["deadline"] != self.sampler.deadline
                    or len(part.streams) != 1
                    or not _tree_equal(part.streams[0]["before"], previous_streams)):
                raise ValueError("buffer payload anchor, frozen weights or sampling chain differs")
            after = part.streams[0]["after"]
            probe = RolloutStreams.from_seed(self.namespaces["distillation_rollouts"], device=self.device)
            probe.load_state_dict(after)
            previous_streams = after
            parts.append(part)
        if not _tree_equal(previous_streams, self.streams.state_dict()):
            raise ValueError("buffer sampling continuation differs from its committed checkpoint")
        buffer = DistillationBuffer.concat(parts)
        if int(buffer.valid.sum().item()) != self.state["imagined_joint_steps"]:
            raise ValueError("buffer valid-step count differs from committed progress")
        if len(parts) == self.protocol["trajectories"]:
            self.buffer = buffer
        return buffer

    def _generate_one(self):
        index = self.state["trajectories_completed"]
        anchors = self.builder.build(self.sampler, (self.references[index],))
        rollout = self.generator.collect(anchors, streams=self.streams)
        buffer = DistillationBuffer.from_teacher_rollout(rollout, self.sampler, self.context.teacher)
        path = self.directory / "buffer" / f"trajectory_{index:04d}_{uuid.uuid4().hex[:8]}"
        save_cycle_checkpoint(path, state={"buffer": buffer.state_dict()}, metadata=self._shard_metadata(index))
        self.shards.append({"path": path.relative_to(self.directory).as_posix(),
                            "manifest_sha256": file_sha256(path / "manifest.json")})
        self.state["trajectories_completed"] += 1
        self.state["imagined_joint_steps"] += int(buffer.valid.sum().item())
        self.last_report = {"event": "teacher_trajectory", "trajectory_index": index,
                            "imagined_joint_steps": int(buffer.valid.sum().item())}
        if self.state["trajectories_completed"] == self.protocol["trajectories"]:
            self._load_buffer()
            self.diagnostics_before = self.learner.diagnostics(self.buffer)
            self.state["phase"] = "distillation"

    def _update_one(self):
        if self.buffer is None:
            self._load_buffer()
        self.last_report = self.learner.step(self.buffer)
        if self.learner.updates_completed == self.protocol["updates"]:
            self.diagnostics_after = self.learner.diagnostics(self.buffer)
            self.state["phase"] = "complete"

    def _publish_handoff(self):
        self._validate()
        if self.state["phase"] != "complete":
            raise ValueError("only a complete warm-start may hand off to local PPO")
        report = {"schema_version": "p42-warm-started-student-v1", "run_identity": canonical_hash(self.manifest),
            "checkpoint": self.last_checkpoint.relative_to(self.directory).as_posix(),
            "checkpoint_manifest_sha256": file_sha256(self.last_checkpoint / "manifest.json"),
            "actor_version": module_version(self.actor), "critic_version": module_version(self.critic),
            "actor_config": asdict(self.actor.config), "paired_initial_versions": self.paired_versions,
            "teacher_input_identity": self.manifest["teacher_input_identity"], "frozen_versions": self.frozen_versions,
            "trajectories": len(self.shards), "imagined_joint_steps": self.state["imagined_joint_steps"],
            "distillation_optimizer_steps": self.learner.updates_completed,
            "diagnostics_before": self.diagnostics_before, "diagnostics_after": self.diagnostics_after,
            "diagnostic_scope": "training buffer only; not validation or real performance",
            "fresh_local_ppo_optimizers_required": True, "local_ppo_optimizer_steps": 0,
            "new_environment_interactions": 0, "new_model_updates": 0,
            "full_d_pipeline_implemented": False, "phase4_acceptance": False}
        path = self.directory / "warm_started_student.json"
        if path.with_suffix(".sha256").exists():
            if canonical_hash(_read_receipted(path)) != canonical_hash(report):
                raise ValueError("committed student handoff differs from checkpoint")
            return
        atomic_json(path, report)
        temporary = path.with_suffix(".sha256.tmp")
        if temporary.exists():
            temporary.unlink()
        _publish_receipt(path)

    def _status(self, status, **extra):
        report = {"schema_version": "p42-distillation-status-v1", "status": status, **deepcopy(self.state),
            "task_family": "relay", "variant": "D", "run_seed": self.master_seed,
            "target_trajectories": self.protocol["trajectories"],
            "distillation_optimizer_steps": self.learner.updates_completed,
            "target_distillation_optimizer_steps": self.protocol["updates"],
            "new_environment_interactions": 0, "new_model_updates": 0,
            "local_ppo_optimizer_steps": 0, "teacher_optimizer_steps": 0,
            "full_d_pipeline_implemented": False,
            "last_checkpoint": None if self.last_checkpoint is None else str(self.last_checkpoint), **extra}
        atomic_json(self.directory / "status.json", report)
        return report

    def status(self):
        return json.loads((self.directory / "status.json").read_text(encoding="utf-8"))

    def run(self, *, max_units=None):
        """Commit each trajectory/update; stop requests finish the current unit."""
        if max_units is not None and (type(max_units) is not int or max_units < 1):
            raise ValueError("max_units must be a positive integer")
        if self._failed:
            raise RuntimeError("failed worker must resume its committed checkpoint in a fresh instance")
        with run_lock(self.directory):
            if canonical_hash(_manifest(self.directory)) != canonical_hash(self.manifest):
                raise ValueError("distillation run manifest changed")
            if source_identity() != self.manifest["source_sha256"]:
                raise ValueError("distillation sources changed; restore the prepared checkout")
            if _runtime(self.device) != self.manifest["runtime"]:
                raise ValueError("distillation runtime changed")
            if self.last_checkpoint != _latest(self.directory):
                raise RuntimeError("another worker advanced this run; resume its latest checkpoint")
            if (canonical_hash(_read_receipted(self.context.directory / "run_manifest.json"))
                    != canonical_hash(self.context.manifest)
                    or canonical_hash(_read_receipted(self.context.directory / "frozen_teacher.json"))
                    != canonical_hash(self.context.handoff)):
                raise ValueError("completed teacher receipt changed")
            _, _, _, _, binding = _initial_inputs(**self.context.manifest["inputs"])
            initial_snapshot = self.context.provenance["initial_snapshot"]
            initial_path = _root(self.context.directory / initial_snapshot["checkpoint"])
            if (canonical_hash(binding) != canonical_hash(self.context.binding)
                    or file_sha256(self.context.checkpoint / "state.pt")
                    != self.context.provenance["checkpoint_payload_sha256"]
                    or file_sha256(self.context.checkpoint / "manifest.json")
                    != self.context.provenance["checkpoint_manifest_sha256"]
                    or file_sha256(initial_path / "manifest.json") != initial_snapshot["checkpoint_manifest_sha256"]
                    or file_sha256(initial_path / "state.pt") != initial_snapshot["checkpoint_payload_sha256"]):
                raise ValueError("frozen teacher weights, replay, selection or normalization changed")
            self._validate()
            units, invocation, attempt = 0, uuid.uuid4().hex, None
            stop_file = self.directory / "stop.request"
            started = None
            try:
                while (self.state["phase"] != "complete" and not self._stop_requested and not stop_file.exists()
                       and (max_units is None or units < max_units)):
                    self._status("running")
                    attempt, started, phase = uuid.uuid4().hex, time.perf_counter(), self.state["phase"]
                    append_event(self.directory / "attempts.jsonl", {"event": "unit_started", "phase": phase,
                        "attempt": attempt, "invocation": invocation,
                        "last_committed_checkpoint": str(self.last_checkpoint)})
                    self._generate_one() if phase == "buffer" else self._update_one()
                    self._checkpoint()
                    append_event(self.directory / "attempts.jsonl", {"event": "unit_committed", "phase": phase,
                        "attempt": attempt, "invocation": invocation, "checkpoint": str(self.last_checkpoint),
                        "elapsed_seconds": time.perf_counter() - started,
                        "trajectories_completed": self.state["trajectories_completed"],
                        "distillation_optimizer_steps": self.learner.updates_completed})
                    append_event(self.directory / "metrics.jsonl", {"checkpoint": str(self.last_checkpoint),
                        "checkpoint_sequence": self.state["sequence"], "report": self.last_report})
                    units += 1
                    attempt, started = None, None
                    self._status("running")
                if self.state["phase"] == "complete":
                    self._publish_handoff()
                if stop_file.is_file():
                    stop_file.unlink()
                return self._status("complete" if self.state["phase"] == "complete" else "stopped", units_executed=units)
            except BaseException as error:
                self._failed = True
                try:
                    append_event(self.directory / "attempts.jsonl", {"event": "invocation_failed", "attempt": attempt,
                        "invocation": invocation, "last_committed_checkpoint": str(self.last_checkpoint),
                        "partial_physical_work": "unknown; interrupted work is not zero" if attempt else "no current unit",
                        "error_type": type(error).__name__, "error": str(error)})
                except OSError:
                    pass
                self._status("failed", error_type=type(error).__name__, error=str(error))
                raise
