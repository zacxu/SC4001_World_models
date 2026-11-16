"""Fresh matched B: twenty frozen-data local PPO cycles with exact continuation.

Training is confined to run(); preparation constructs paired initial networks
and a durable zero-cycle boundary. No teacher, simulator or model optimizer.
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

from world_models.cycle.checkpoints import _guard_real_replay, load_cycle_checkpoint, save_cycle_checkpoint
from world_models.data.ledger import atomic_json, file_sha256
from world_models.imagination import AnchorBuilder, AnchorSampler, module_version
from world_models.infrastructure.reproducibility import canonical_hash, derive_seed
from world_models.mappo.config import MAPPOConfig
from world_models.mappo.updater import RecurrentMAPPO
from world_models.rollouts import RolloutGenerator, RolloutStreams
from world_models.teacher.runner import _latest, _NUMBERED, _publish_receipt, _root, _runtime
from world_models.training.runner import append_event, run_lock
from .b_inputs import load_initial_b, verify_frozen_inputs, load_b_protocol as load_protocol, safe_inputs, input_identity
from world_models.distillation.versions import module_version_at_device
from world_models_study.evidence import source_identity
from world_models.policies.initialization import build_actor_critic


def protocol_identity():
    return canonical_hash(load_protocol())


ROOT = Path(__file__).resolve().parents[2]
RUN_SCHEMA = "p52-b-local-run-v1"


def _receipt_json(path):
    receipt = path.with_suffix(".sha256")
    if (not path.is_file() or not receipt.is_file() or path.is_symlink() or receipt.is_symlink()
            or receipt.read_text(encoding="ascii").strip() != f"{file_sha256(path)}  {path.name}"):
        raise ValueError(f"missing or invalid receipt: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(directory):
    result = _receipt_json(directory / "run_manifest.json")
    if (result.get("schema_version") != RUN_SCHEMA or result.get("protocol") != load_protocol()
            or result.get("protocol_sha256") != protocol_identity()):
        raise ValueError("matched B run schema or fixed protocol changed")
    return result


def inspect(directory):
    """Read a durable B boundary on CPU; an ahead live status is informational.

    This never resumes a worker, opens an engine or allocates a CUDA tensor.
    Concurrent checkpoint publication is detected so callers can retry the read.
    """
    directory = _root(directory)
    manifest = _manifest(directory)
    if manifest["source_sha256"] != source_identity():
        raise ValueError("matched B source changed; restore the prepared checkout")
    checkpoint = _latest(directory)
    saved, metadata = load_cycle_checkpoint(checkpoint, device="cpu")
    if set(saved) != {"controller", "actor", "critic", "learner", "streams", "sampler", "last_report"}:
        raise ValueError("matched B checkpoint keys differ")
    p, state, learner = manifest["protocol"], saved["controller"], saved["learner"]
    cycles, steps = learner["cycles_completed"], p["epochs"] * p["minibatches"]
    if (set(state) != {"sequence", "phase", "imagined_joint_steps", "local_player_decisions"}
            or type(cycles) is not int or not 0 <= cycles <= p["local_cycles"]
            or any(type(state[name]) is not int or state[name] < 0 for name in
                   ("sequence", "imagined_joint_steps", "local_player_decisions"))
            or state["sequence"] != cycles + 1
            or state["sequence"] != int(_NUMBERED.fullmatch(checkpoint.name)[1])
            or state["phase"] != ("complete" if cycles == p["local_cycles"] else "local")
            or state["local_player_decisions"] != 2 * state["imagined_joint_steps"]
            or not cycles * p["starts"] <= state["imagined_joint_steps"] <= cycles * p["starts"] * p["horizon"]
            or learner["actor_optimizer_steps"] != cycles * steps
            or learner["critic_optimizer_steps"] != cycles * steps
            or saved["streams"]["identity"] != manifest["rollout_stream_identity"]):
        raise ValueError("matched B durable counters or rollout identity differ")
    policies = build_actor_critic(manifest["input_binding"]["B_settings"],
                                 master_seed=manifest["master_seed"], variant="B")
    label = manifest["runtime"]["logical_device"]
    for name in ("actor", "critic"):
        getattr(policies, name).load_state_dict(saved[name], strict=True)
    versions = {name: module_version_at_device(getattr(policies, name), label) for name in ("actor", "critic")}
    expected = {"schema_version": "p52-b-local-checkpoint-v1", "run_identity": canonical_hash(manifest),
        **state, "local_cycles_completed": cycles, "actor_optimizer_steps": cycles * steps,
        "critic_optimizer_steps": cycles * steps,
        "model_version": manifest["initial_imported_versions"]["model"],
        "actor_version": versions["actor"], "critic_version": versions["critic"],
        "teacher_kl_weight": 0.0, "teacher_queries": 0, "teacher_optimizer_steps": 0,
        "distillation_optimizer_steps": 0, "new_model_updates": 0, "new_environment_interactions": 0}
    if metadata != expected:
        raise ValueError("matched B durable checkpoint metadata differ")
    publication = directory / "final_student.json"
    final = None
    if publication.with_suffix(".sha256").exists():
        final = _receipt_json(publication)
        if (state["phase"] != "complete" or final.get("run_identity") != canonical_hash(manifest)
                or final.get("checkpoint") != checkpoint.relative_to(directory).as_posix()
                or final.get("checkpoint_manifest_sha256") != file_sha256(checkpoint / "manifest.json")
                or final.get("actor_version") != versions["actor"]
                or final.get("local_cycles_completed") != cycles):
            raise ValueError("matched B final publication differs from its checkpoint")
        export = _root(directory / final["actor_export"])
        if not export.is_relative_to(directory):
            raise ValueError("matched B actor export escapes its run")
        exported, export_meta = load_cycle_checkpoint(export, device="cpu")
        if (file_sha256(export / "manifest.json") != final["actor_export_manifest_sha256"]
                or set(exported) != {"actor"} or set(exported["actor"]) != set(saved["actor"])
                or any(not torch.equal(value, exported["actor"][name]) for name, value in saved["actor"].items())
                or export_meta.get("actor_version") != versions["actor"] or export_meta.get("variant") != "B"):
            raise ValueError("matched B actor export differs from its checkpoint")
    if checkpoint != _latest(directory):
        raise RuntimeError("B checkpoint advanced during inspection; retry this read")
    live, live_error = {}, None
    try:
        if (directory / "status.json").is_file():
            live = json.loads((directory / "status.json").read_text(encoding="utf-8"))
            if not isinstance(live, dict):
                raise ValueError("live status is not a JSON object")
    except (OSError, UnicodeError, ValueError) as error:
        live, live_error = {}, f"{type(error).__name__}: {error}"
    return {"schema_version": "p52-b-inspection-v1", "status": "complete" if final else
            ("publication_pending" if state["phase"] == "complete" else "prepared" if cycles == 0 else "incomplete"),
        "task_family": "relay", "variant": "B", "run_seed": manifest["master_seed"],
        "phase": state["phase"], "local_cycles_completed": cycles, "target_local_cycles": p["local_cycles"],
        "actor_optimizer_steps": cycles * steps, "critic_optimizer_steps": cycles * steps,
        "model_updates": p["initial_model_updates"], "new_model_updates": 0,
        "real_joint_transitions": p["initial_transitions"], "new_environment_interactions": 0,
        "imagined_joint_steps": state["imagined_joint_steps"], "local_player_decisions": state["local_player_decisions"],
        "last_checkpoint": str(checkpoint), "actor_export": None if final is None else str(directory / final["actor_export"]),
        "live_process_status": live.get("status"), "live_status_is_durable_evidence": False,
        "live_status_read_error": live_error,
        "cohort_acceptance": False, "evaluation_executed": False}


class BRunner:
    """One paired local B actor, fresh PPO Adam state, twenty total cycles."""

    inspect = staticmethod(inspect)

    def __init__(self, directory, context):
        self.directory, self.context = _root(directory), context
        self.settings, self.master_seed, self.precision = context.settings, context.master_seed, context.precision
        self.model, self.actor, self.critic = context.model, context.actor, context.critic
        self.device = next(self.model.parameters()).device
        self.protocol = load_protocol()
        p = self.protocol
        if (self.precision != p["world_model_compute_precision"]
                or type(self.master_seed) is not int or self.master_seed not in p["master_seeds"]
                or self.settings["task"]["name"] != "relay" or self.settings["variant"]["name"] != "B"
                or context.dataset.manifest["transition_count"] != p["initial_transitions"]
                or context.binding["shared_initial_binding"]["selection"]["model_updates"] != p["initial_model_updates"]
                or (self.settings["mappo"]["starts"], self.settings["mappo"]["horizon"])
                    != (p["starts"], p["horizon"])):
            raise ValueError("B inputs differ from the registered P5.2 model/data/PPO recipe")
        self.model.requires_grad_(False).eval().zero_grad(set_to_none=True)
        for network in (self.actor, self.critic):
            network.requires_grad_(True)
            network.zero_grad(set_to_none=True)
        self.initial_versions = {name: module_version(network) for name, network in
                                 (("model", self.model), ("actor", self.actor), ("critic", self.critic))}
        self.paired_initialization = {
            "actor_cpu": module_version_at_device(self.actor, "cpu"),
            "critic_cpu": module_version_at_device(self.critic, "cpu"),
            "input_binding_sha256": canonical_hash(context.binding["shared_initial_binding"]),
            "world_model_checkpoint_sha256": context.binding["shared_initial_binding"]["selection"]["checkpoint_sha256"],
            "normalization_identity_sha256": context.binding["shared_initial_binding"]["normalization"]["identity_sha256"],
            "replay_manifest_sha256": context.binding["shared_initial_binding"]["provenance"]["replays"][0]["manifest_sha256"]}
        config = MAPPOConfig.from_experiment(self.settings)
        if config != MAPPOConfig(epochs=p["epochs"], minibatches=p["minibatches"]):
            raise ValueError("matched B must preserve B's original MAPPO mathematics and optimizer settings")
        # Fresh paired initialization has no inherited actor/critic optimizer.
        # No teacher, distillation optimizer or KL adapter is involved.
        self.learner = RecurrentMAPPO.from_experiment(self.model, self.actor, self.critic, self.settings,
                                                    master_seed=self.master_seed, precision=self.precision)
        self.namespaces = {name: derive_seed("relay", self.master_seed, name, "B") for name in
                           ("imagination_anchors", "imagination_prefixes", "local_policy_updates")}
        self.anchor_sampler = AnchorSampler.from_experiment([context.dataset], self.settings,
                                                            seed=self.namespaces["imagination_anchors"])
        self.builder = AnchorBuilder(self.model, self.actor, seed=self.namespaces["imagination_prefixes"],
                                     precision=self.precision)
        self.generator = RolloutGenerator.from_experiment(self.model, self.actor, self.critic, self.settings,
                                                         precision=self.precision)
        self.streams = RolloutStreams.from_experiment(self.settings, master_seed=self.master_seed, device=self.device)
        self.state = {"sequence": 0, "phase": "local", "imagined_joint_steps": 0, "local_player_decisions": 0}
        self.manifest, self.last_checkpoint, self.last_report = {}, None, None
        self._failed = self._stop_requested = False

    @classmethod
    def prepare(cls, directory, *, initial_selection, master_seed,
                data_root=ROOT / "data/initial_replay/pilot",
                statistics_root=ROOT / "artifacts/phase23/normalization",
                config_root=ROOT / "configs", device="cuda:0", precision="bfloat16", cohort_binding=None):
        directory = _root(directory)
        _guard_real_replay(directory)
        if directory.exists():
            raise FileExistsError("B prepare requires a fresh directory")
        inputs = safe_inputs(initial_selection=initial_selection, master_seed=master_seed,
                             data_root=data_root, statistics_root=statistics_root, config_root=config_root)
        for key in ("data_root", "statistics_root", "config_root"):
            source = Path(inputs[key])
            if directory.is_relative_to(source) or source.is_relative_to(directory):
                raise ValueError("B run must be separate from immutable input directories")
        selection = Path(inputs["initial_selection"])
        if selection.is_relative_to(directory):
            raise ValueError("B output may not contain its selection input")
        context = load_initial_b(inputs, device=device, precision=precision)
        worker = cls(directory, context)
        worker.manifest = {"schema_version": RUN_SCHEMA, "protocol": worker.protocol,
            "protocol_sha256": protocol_identity(), "source_sha256": source_identity(),
            "runtime": _runtime(worker.device), "inputs": inputs,
            "input_binding": deepcopy(context.binding), "input_identity": input_identity(context),
            "input_provenance": deepcopy(context.provenance),
            "master_seed": worker.master_seed, "precision": worker.precision,
            "cohort_binding": deepcopy(cohort_binding),
            "initial_imported_versions": worker.initial_versions,
            "paired_initialization": worker.paired_initialization,
            "rng_namespaces": worker.namespaces, "rollout_stream_identity": deepcopy(worker.streams.identity),
            "created_at": datetime.now(timezone.utc).isoformat(), "training_executed_at_prepare": False,
            "fresh_local_ppo_optimizers": True, "evaluation_executed": False, "cohort_acceptance": False}
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
            raise ValueError("matched B run does not exist")
        with run_lock(directory):
            manifest = _manifest(directory)
            if manifest["source_sha256"] != source_identity():
                raise ValueError("matched B source changed; restore the prepared checkout")
            context = load_initial_b(manifest["inputs"], device=device, precision=manifest["precision"])
            if (input_identity(context) != manifest["input_identity"]
                    or canonical_hash(context.binding) != canonical_hash(manifest["input_binding"])):
                raise ValueError("saved B input identity changed")
            worker = cls(directory, context)
            if manifest["runtime"] != _runtime(worker.device):
                raise ValueError("exact B resume requires the same logical device and compatible runtime")
            if (worker.paired_initialization != manifest["paired_initialization"]
                    or worker.initial_versions != manifest["initial_imported_versions"]
                    or worker.namespaces != manifest["rng_namespaces"]
                    or worker.streams.identity != manifest["rollout_stream_identity"]):
                raise ValueError("paired initial weights or local sampling identities changed")
            checkpoint = _latest(directory)
            saved, metadata = load_cycle_checkpoint(checkpoint)
            if metadata.get("run_identity") != canonical_hash(manifest):
                raise ValueError("matched B checkpoint belongs to another run")
            if set(saved) != {"controller", "actor", "critic", "learner", "streams", "sampler", "last_report"}:
                raise ValueError("matched B checkpoint keys differ")
            worker.manifest, worker.state = manifest, deepcopy(saved["controller"])
            for name, network in (("actor", worker.actor), ("critic", worker.critic)):
                network.load_state_dict(saved[name], strict=True)
            worker.learner.load_state_dict(saved["learner"])
            worker.streams.load_state_dict(saved["streams"])
            worker.anchor_sampler.load_state_dict(saved["sampler"])
            worker.last_report, worker.last_checkpoint = saved["last_report"], checkpoint
            worker._validate_state()
            if (metadata != worker._metadata()
                    or int(_NUMBERED.fullmatch(checkpoint.name)[1]) != worker.state["sequence"]
                    or worker.state["sequence"] != worker.learner.cycles_completed + 1):
                raise ValueError("matched B checkpoint counters or metadata differ")
            return worker

    def pairing(self):
        """Original paired initialization, independent of current trained weights/device."""
        return deepcopy(self.paired_initialization)

    def request_stop(self):
        self._stop_requested = True

    def _validate_state(self):
        p, s, learner = self.protocol, self.state, self.learner
        if (set(s) != {"sequence", "phase", "imagined_joint_steps", "local_player_decisions"}
                or s["phase"] not in ("local", "complete")
                or any(type(s[key]) is not int or s[key] < 0 for key in
                       ("sequence", "imagined_joint_steps", "local_player_decisions"))):
            raise ValueError("invalid matched B controller state")
        cycles, per_cycle = learner.cycles_completed, p["epochs"] * p["minibatches"]
        if (type(cycles) is not int or not 0 <= cycles <= p["local_cycles"]
                or learner.actor_optimizer_steps != cycles * per_cycle
                or learner.critic_optimizer_steps != cycles * per_cycle
                or s["local_player_decisions"] != 2 * s["imagined_joint_steps"]
                or not cycles * p["starts"] <= s["imagined_joint_steps"] <= cycles * p["starts"] * p["horizon"]
                or (s["phase"] == "complete") != (cycles == p["local_cycles"])):
            raise ValueError("matched B cycle, optimizer or imagined-step accounting differs")
        if (module_version(self.model) != self.manifest["initial_imported_versions"]["model"]
                or any(parameter.requires_grad or parameter.grad is not None for parameter in self.model.parameters())):
            raise ValueError("frozen shared world model changed or acquired gradients")
        if (cycles == 0 and {name: module_version(network) for name, network in
                            (("model", self.model), ("actor", self.actor), ("critic", self.critic))}
                != self.manifest["initial_imported_versions"]):
            raise ValueError("fresh matched B must start from the exact saved actor/paired critic")
        if (self.streams.identity != self.manifest["rollout_stream_identity"]
                or learner.seed != self.namespaces["local_policy_updates"]):
            raise ValueError("matched B sampling namespace changed")
        if self.last_report is not None:
            if (self.last_report.get("cycles_completed") != cycles
                    or self.last_report.get("world_model_version") != self.initial_versions["model"]
                    or self.last_report.get("actor_version_after") != module_version(self.actor)
                    or self.last_report.get("critic_version_after") != module_version(self.critic)
                    or self.last_report.get("teacher_kl_weight") != 0
                    or self.last_report.get("teacher_queries") != 0):
                raise ValueError("matched B report is stale or contains teacher guidance")
        elif cycles:
            raise ValueError("completed local cycles require their saved PPO report")

    def _metadata(self):
        return {"schema_version": "p52-b-local-checkpoint-v1", "run_identity": canonical_hash(self.manifest),
            **deepcopy(self.state), "local_cycles_completed": self.learner.cycles_completed,
            "actor_optimizer_steps": self.learner.actor_optimizer_steps,
            "critic_optimizer_steps": self.learner.critic_optimizer_steps,
            "model_version": module_version(self.model), "actor_version": module_version(self.actor),
            "critic_version": module_version(self.critic), "teacher_kl_weight": 0.0,
            "teacher_queries": 0, "teacher_optimizer_steps": 0, "distillation_optimizer_steps": 0,
            "new_model_updates": 0, "new_environment_interactions": 0}

    def _checkpoint(self):
        self._validate_state()
        self.state["sequence"] += 1
        path = self.directory / "checkpoints" / f"state_{self.state['sequence']:08d}_{uuid.uuid4().hex[:8]}"
        saved = {"controller": deepcopy(self.state), "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "learner": self.learner.state_dict(), "streams": self.streams.state_dict(),
            "sampler": self.anchor_sampler.state_dict(), "last_report": deepcopy(self.last_report)}
        save_cycle_checkpoint(path, state=saved, metadata=self._metadata())
        self.last_checkpoint = path

    def _export_actor(self):
        """Pure local-only publication compatible with the existing actor loader."""
        p = self.protocol
        metadata = {"schema_version": "p35-local-actor-v1", "task_family": "relay", "variant": "B",
            "run_seed": self.master_seed, "real_joint_transitions": p["initial_transitions"],
            "actual_initial_replay_interactions": p["initial_transitions"], "untrained_zero_interaction_baseline": False,
            "actor_version": module_version(self.actor), "actor_config": json.loads(json.dumps(asdict(self.actor.config))),
            "world_model_version": module_version(self.model), "model_updates": p["initial_model_updates"],
            "local_ppo_cycles": self.learner.cycles_completed, "actor_optimizer_steps": self.learner.actor_optimizer_steps,
            "critic_optimizer_steps": self.learner.critic_optimizer_steps, "evaluation_split": "test",
            "on_registered_evaluation_grid": False, "evaluation_executed": False, "phase3_acceptance": False}
        choices = sorted(path for path in self.directory.glob("actor_export*")
                         if path.is_dir() and (path / "manifest.sha256").is_file())
        if choices:
            for path in choices:
                saved, previous = load_cycle_checkpoint(path)
                expected = self.actor.state_dict()
                if (previous != metadata or set(saved) != {"actor"} or set(saved["actor"]) != set(expected)
                        or any(not torch.equal(saved["actor"][name], value.cpu()) for name, value in expected.items())):
                    raise ValueError("existing local actor export differs from the final B checkpoint")
            return choices[0]
        path = self.directory / "actor_export"
        if path.exists():
            path = self.directory / f"actor_export_attempt_{uuid.uuid4().hex[:8]}"
        save_cycle_checkpoint(path, state={"actor": self.actor.state_dict()}, metadata=metadata)
        return path

    def _publish_final_student(self):
        self._validate_state()
        if self.state["phase"] != "complete":
            raise ValueError("only the completed fixed B schedule can publish a final actor")
        export = self._export_actor()
        report = {"schema_version": "p52-final-B-student-v1", "run_identity": canonical_hash(self.manifest),
            "checkpoint": self.last_checkpoint.relative_to(self.directory).as_posix(),
            "checkpoint_manifest_sha256": file_sha256(self.last_checkpoint / "manifest.json"),
            "actor_export": export.relative_to(self.directory).as_posix(),
            "actor_export_manifest_sha256": file_sha256(export / "manifest.json"),
            "actor_version": module_version(self.actor), "critic_version": module_version(self.critic),
            "model_version": module_version(self.model), "actor_config": asdict(self.actor.config),
            "input_identity": self.manifest["input_identity"],
            "initial_imported_versions": self.manifest["initial_imported_versions"],
            "local_cycles_completed": self.learner.cycles_completed,
            "actor_optimizer_steps": self.learner.actor_optimizer_steps,
            "critic_optimizer_steps": self.learner.critic_optimizer_steps,
            "imagined_joint_steps": self.state["imagined_joint_steps"],
            "local_player_decisions": self.state["local_player_decisions"],
            "real_joint_transitions": self.protocol["initial_transitions"],
            "model_updates": self.protocol["initial_model_updates"],
            "teacher_kl_weight": 0.0, "teacher_queries": 0, "distillation_optimizer_steps_in_stage": 0,
            "teacher_optimizer_steps_in_stage": 0, "new_model_updates": 0, "new_environment_interactions": 0,
            "local_ppo_started_with_fresh_optimizers": True,
            "cohort_acceptance": False, "evaluation_executed": False}
        path = self.directory / "final_student.json"
        if path.with_suffix(".sha256").exists():
            if canonical_hash(_receipt_json(path)) != canonical_hash(report):
                raise ValueError("committed B student handoff differs from the final checkpoint")
            return
        atomic_json(path, report)
        temporary = path.with_suffix(".sha256.tmp")
        if temporary.exists():
            temporary.unlink()
        _publish_receipt(path)

    def _status(self, status, **extra):
        p = self.protocol
        steps = p["local_cycles"] * p["epochs"] * p["minibatches"]
        report = {"schema_version": "p52-b-status-v1", "status": status, **deepcopy(self.state),
            "task_family": "relay", "variant": "B", "run_seed": self.master_seed,
            "local_cycles_completed": self.learner.cycles_completed, "target_local_cycles": p["local_cycles"],
            "actor_optimizer_steps": self.learner.actor_optimizer_steps, "target_actor_optimizer_steps": steps,
            "critic_optimizer_steps": self.learner.critic_optimizer_steps, "target_critic_optimizer_steps": steps,
            "real_joint_transitions": p["initial_transitions"], "model_updates": p["initial_model_updates"],
            "new_model_updates": 0, "new_environment_interactions": 0,
            "teacher_queries": 0, "teacher_kl_weight": 0.0, "teacher_optimizer_steps": 0,
            "distillation_optimizer_steps": 0, "cohort_acceptance": False,
            "last_checkpoint": None if self.last_checkpoint is None else str(self.last_checkpoint), **extra}
        atomic_json(self.directory / "status.json", report)
        return report

    def status(self):
        return json.loads((self.directory / "status.json").read_text(encoding="utf-8"))

    def run(self, *, max_cycles=None):
        if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 1):
            raise ValueError("max_cycles must be a positive integer")
        if self._failed:
            raise RuntimeError("failed matched B worker must resume its committed checkpoint in a fresh instance")
        with run_lock(self.directory):
            if canonical_hash(_manifest(self.directory)) != canonical_hash(self.manifest):
                raise ValueError("matched B run manifest changed")
            if self.manifest["source_sha256"] != source_identity():
                raise ValueError("matched B source changed; restore its prepared checkout")
            if self.manifest["runtime"] != _runtime(self.device):
                raise ValueError("matched B runtime changed")
            if self.last_checkpoint != _latest(self.directory):
                raise RuntimeError("another worker advanced matched B; resume its latest committed checkpoint")
            verify_frozen_inputs(self.context)
            self._validate_state()
            cycles, invocation, attempt, started = 0, uuid.uuid4().hex, None, None
            stop_file = self.directory / "stop.request"
            try:
                while (self.state["phase"] != "complete" and not self._stop_requested and not stop_file.exists()
                       and (max_cycles is None or cycles < max_cycles)):
                    self._status("running")
                    attempt, started = uuid.uuid4().hex, time.perf_counter()
                    append_event(self.directory / "attempts.jsonl", {"event": "local_cycle_started", "attempt": attempt,
                        "invocation": invocation, "local_cycle": self.learner.cycles_completed + 1,
                        "last_committed_checkpoint": str(self.last_checkpoint)})
                    anchors = self.builder.build(self.anchor_sampler)
                    rollout = self.generator.collect(anchors, streams=self.streams)
                    report = self.learner.update(rollout, self.anchor_sampler)
                    self.last_report = {**report, "teacher_kl_weight": 0.0, "teacher_queries": 0,
                                        "distillation_optimizer_steps": 0, "teacher_optimizer_steps": 0}
                    self.state["imagined_joint_steps"] += report["imagined_joint_steps"]
                    self.state["local_player_decisions"] += 2 * report["imagined_joint_steps"]
                    if self.learner.cycles_completed == self.protocol["local_cycles"]:
                        self.state["phase"] = "complete"
                    self._checkpoint()
                    append_event(self.directory / "attempts.jsonl", {"event": "local_cycle_committed", "attempt": attempt,
                        "invocation": invocation, "local_cycle": self.learner.cycles_completed,
                        "checkpoint": str(self.last_checkpoint), "imagined_joint_steps": report["imagined_joint_steps"],
                        "actor_optimizer_steps": report["cycle_actor_optimizer_steps"],
                        "critic_optimizer_steps": report["cycle_critic_optimizer_steps"],
                        "elapsed_seconds": time.perf_counter() - started})
                    append_event(self.directory / "metrics.jsonl", {"checkpoint": str(self.last_checkpoint),
                        "checkpoint_sequence": self.state["sequence"], **self.last_report})
                    cycles += 1
                    attempt, started = None, None
                    self._status("running")
                if self.state["phase"] == "complete":
                    self._publish_final_student()
                if stop_file.is_file():
                    stop_file.unlink()
                return self._status("complete" if self.state["phase"] == "complete" else "stopped", cycles_executed=cycles)
            except BaseException as error:
                self._failed = True
                try:
                    append_event(self.directory / "attempts.jsonl", {"event": "local_invocation_failed", "attempt": attempt,
                        "invocation": invocation, "last_committed_checkpoint": str(self.last_checkpoint),
                        "partial_physical_optimizer_work": "unknown; interrupted work is not zero" if attempt else "no current cycle",
                        "elapsed_attempt_seconds": None if started is None else time.perf_counter() - started,
                        "error_type": type(error).__name__, "error": str(error)})
                except OSError:
                    pass
                self._status("failed", error_type=type(error).__name__, error=str(error))
                raise
