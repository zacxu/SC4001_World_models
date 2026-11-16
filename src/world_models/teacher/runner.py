"""Receipt-bound, resumable initial teacher stage; only run() trains.

The shared initial world model remains frozen. This module neither constructs
a student nor opens a native environment, and has no world-model optimizer.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import uuid

import torch

from world_models.cycle.checkpoints import load_cycle_checkpoint, save_cycle_checkpoint
from world_models.data.ledger import atomic_json, file_sha256
from world_models.data.replay import ReplayDataset
from world_models.imagination import AnchorSampler
from world_models.imagination.versions import module_version
from world_models.infrastructure.config import resolve_config, validate_config
from world_models.infrastructure.reproducibility import canonical_hash, derive_seed
from world_models.models.world_model import build_model
from world_models.policies.config import PolicyConfig
from world_models.policies.critic import CentralizedCritic
from world_models.policies.teacher import GlobalTeacher
from world_models.rollouts.streams import RolloutStreams
from world_models.training.checkpoints import read_checkpoint
from world_models.training.runner import (append_event, load_case, run_lock, runtime_contract,
                                          selected_device, source_identity as p24_sources)
from world_models.training.selection import resolve_selected_case
from .learning import TeacherAnchorBuilder, TeacherMAPPO, TeacherRolloutGenerator
from .protocol import TEACHER_PROTOCOL, load_protocol, protocol_identity


ROOT = Path(__file__).resolve().parents[3]
RUN_SCHEMA = "p41-teacher-run-v1"
STATUS_SCHEMA = "p41-teacher-status-v1"
_NUMBERED = re.compile(r"state_([0-9]{8})_[a-f0-9]{8}\Z")


def source_identity():
    """Bind runtime dependencies and CLI without importing the CLI itself."""
    paths = sorted((ROOT / "src/world_models").rglob("*.py"))
    paths += [ROOT / "scripts/phase41.py"]
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}


def _runtime(device):
    return {**runtime_contract(device), "logical_device": str(device),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cpu_threads": torch.get_num_threads(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _root(path):
    value = Path(path)
    if ".." in value.parts or any(part.is_symlink() for part in (value, *value.parents)):
        raise ValueError("teacher run path may not traverse parents or symbolic links")
    return value.resolve()


def _latest(directory):
    committed = [path for path in (directory / "checkpoints").glob("state_*")
                 if (path / "manifest.sha256").is_file()]
    if not committed:
        raise ValueError("no committed teacher checkpoint; preserve the incomplete run")
    numbers = []
    for path in committed:
        match = _NUMBERED.fullmatch(path.name)
        if match is None:
            raise ValueError("malformed committed teacher checkpoint name")
        numbers.append(int(match[1]))
    if len(numbers) != len(set(numbers)):
        raise ValueError("ambiguous duplicate committed teacher checkpoint sequence")
    return committed[numbers.index(max(numbers))]


def _publish_receipt(path):
    temporary = path.with_suffix(".sha256.tmp")
    with temporary.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{file_sha256(path)}  {path.name}\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path.with_suffix(".sha256"))


def _manifest(directory):
    path = directory / "run_manifest.json"
    receipt = path.with_suffix(".sha256")
    if not path.is_file() or not receipt.is_file():
        raise ValueError("teacher run manifest and receipt are both required")
    if receipt.read_text(encoding="ascii").strip() != f"{file_sha256(path)}  {path.name}":
        raise ValueError("teacher run manifest receipt mismatch")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("schema_version") != RUN_SCHEMA
            or result.get("protocol") != load_protocol()
            or result.get("protocol_sha256") != protocol_identity()):
        raise ValueError("teacher run protocol or schema changed")
    return result


def _initial_inputs(*, initial_selection, master_seed, data_root, statistics_root, config_root):
    p = TEACHER_PROTOCOL
    if type(master_seed) is not int or master_seed not in p["master_seeds"]:
        raise ValueError("P4.1 permits Relay master seeds 0 and 1 only")
    selected = resolve_selected_case(initial_selection, "relay", master_seed)
    cfg, datasets, stats, provenance = load_case("relay", master_seed, data_root=data_root,
        statistics_root=statistics_root, config_root=config_root, profile="pilot")
    if len(datasets) != 1:
        raise ValueError("teacher requires exactly the shared initial replay")
    # Eager validation also checks held-out bytes: later mutation of either
    # partition must not retain the same declared replay identity.
    data = ReplayDataset(datasets[0].path)
    stats.validate_for_dataset(data)
    parent = cfg.settings
    common = deepcopy(parent)
    common.pop("variant")
    payload = read_checkpoint(selected["checkpoint_path"])
    meta = payload["metadata"]
    expected = {"task_family": "relay", "run_seed": master_seed, "stage": "initial",
                "variant": "shared_initial:B,D", "common_config_sha256": canonical_hash(common),
                "source_sha256": p24_sources(),
                "target_stage_updates": parent["world_model"]["initial_updates"], **provenance}
    if (selected["model_updates"] != p["initial_model_updates"]
            or payload["update"] != p["initial_model_updates"]
            or selected["checkpoint_sha256"] != file_sha256(selected["checkpoint_path"])
            or data.manifest["transition_count"] != p["initial_transitions"]
            or parent["experiment"]["initial_transitions"] != p["initial_transitions"]
            or any(meta.get(key) != value for key, value in expected.items())
            or selected.get("initial_normalization") != provenance["initial_normalization"]
            or selected.get("replays") != provenance["replays"]):
        raise ValueError("teacher needs the exact frozen 4500-update / 50000-transition shared P2.4 inputs")
    settings = deepcopy(parent)
    settings["variant"] = deepcopy(resolve_config("relay", "D", "pilot", config_root).settings["variant"])
    validate_config(settings)
    recipe = settings["mappo"]
    if ((recipe["starts"], recipe["horizon"], recipe["epochs"], recipe["recurrent_minibatches"])
            != (p["starts"], p["horizon"], p["epochs"], p["minibatches"])
            or PolicyConfig.from_experiment(settings).world_width != p["world_width"]):
        raise ValueError("teacher settings differ from the registered PPO or architecture recipe")
    binding = {"selection": selected, "provenance": provenance,
               "replay_path": str(data.path.resolve()), "normalization": stats.as_dict(),
               "settings": settings}
    return settings, data, stats, payload, binding


def _policies(settings, seed, device):
    config = PolicyConfig.from_experiment(settings)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed("relay", seed, "teacher_initialization", "D"))
        teacher = GlobalTeacher(config, attention_heads=TEACHER_PROTOCOL["attention_heads"])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed("relay", seed, "teacher_critic_initialization", "D"))
        critic = CentralizedCritic(config)
    return teacher.to(device), critic.to(device)


class TeacherRunner:
    """One independently resumable, single-device teacher; no student training."""

    def __init__(self, directory, settings, master_seed, model, teacher, critic, data, *, precision):
        self.directory, self.settings = _root(directory), deepcopy(settings)
        self.master_seed, self.precision = master_seed, precision
        self.model, self.teacher, self.critic, self.dataset = model, teacher, critic, data
        self.device = next(model.parameters()).device
        self.model.requires_grad_(False).eval()
        self.model.zero_grad(set_to_none=True)
        self.learner = TeacherMAPPO.from_experiment(model, teacher, critic, settings,
            master_seed=master_seed, precision=precision)
        self.anchor_sampler = AnchorSampler.from_experiment([data], settings,
            seed=derive_seed("relay", master_seed, "teacher_anchor_sampling", "D"))
        self.builder = TeacherAnchorBuilder(model, teacher,
            seed=derive_seed("relay", master_seed, "teacher_anchor_prefix", "D"), precision=precision)
        self.generator = TeacherRolloutGenerator.from_experiment(model, teacher, critic, settings,
            precision=precision)
        # from_seed deliberately omits the local-action experiment namespace.
        # The teacher root namespace and all child seeds are stored in manifest.
        self.streams = RolloutStreams.from_seed(
            derive_seed("relay", master_seed, "teacher_rollouts", "D"), device=self.device)
        self.state = {"sequence": 0, "phase": "teacher", "imagined_joint_steps": 0,
                      "teacher_player_decisions": 0}
        self.manifest, self.last_checkpoint = {}, None
        self._stop_requested = self._failed = False

    @classmethod
    def prepare(cls, directory, *, initial_selection, master_seed,
                data_root=ROOT / "data/initial_replay/pilot",
                statistics_root=ROOT / "artifacts/phase23/normalization",
                config_root=ROOT / "configs", device="cuda:0", precision="bfloat16"):
        directory = _root(directory)
        if directory.exists():
            raise FileExistsError("teacher prepare requires a fresh directory")
        if precision not in ("float32", "bfloat16"):
            raise ValueError("teacher precision must be float32 or bfloat16")
        inputs = {"initial_selection": str(Path(initial_selection).resolve()), "master_seed": master_seed,
                  "data_root": str(Path(data_root).resolve()),
                  "statistics_root": str(Path(statistics_root).resolve()),
                  "config_root": str(Path(config_root).resolve())}
        settings, data, _, payload, binding = _initial_inputs(**inputs)
        device = selected_device(device, precision)
        model = build_model(settings, master_seed=master_seed, variant="D").to(device)
        model.load_state_dict(payload["model"], strict=True)
        teacher, critic = _policies(settings, master_seed, device)
        result = cls(directory, settings, master_seed, model, teacher, critic, data, precision=precision)
        result.manifest = {"schema_version": RUN_SCHEMA, "protocol": load_protocol(),
            "protocol_sha256": protocol_identity(), "source_sha256": source_identity(),
            "runtime": _runtime(device), "precision": precision, "inputs": inputs,
            "input_binding": binding, "initial_model_version": module_version(model),
            "initial_teacher_version": module_version(teacher), "initial_critic_version": module_version(critic),
            "rng_namespaces": {name: derive_seed("relay", master_seed, name, "D") for name in
                ("teacher_initialization", "teacher_critic_initialization", "teacher_anchor_sampling",
                 "teacher_anchor_prefix", "teacher_rollouts", "teacher_updates")},
            "rollout_stream_identity": deepcopy(result.streams.identity),
            "created_at": datetime.now(timezone.utc).isoformat(), "training_executed_at_prepare": False,
            "research_quality_acceptance": False, "full_d_pipeline_implemented": False}
        directory.mkdir(parents=True, exist_ok=False)
        with run_lock(directory):
            atomic_json(directory / "run_manifest.json", result.manifest)
            _publish_receipt(directory / "run_manifest.json")
            result._checkpoint()
            result._status("prepared")
        return result

    @classmethod
    def resume(cls, directory, *, device="cuda:0"):
        directory = _root(directory)
        # Lock while selecting and loading. run() checks the committed pointer
        # again, closing the gap after this lock is released.
        if not directory.is_dir():
            raise ValueError("teacher run does not exist")
        with run_lock(directory):
            manifest = _manifest(directory)
            if manifest["source_sha256"] != source_identity():
                raise ValueError("teacher run source changed; restore its saved checkout")
            device = selected_device(device, manifest["precision"])
            if manifest["runtime"] != _runtime(device):
                raise ValueError("teacher exact resume requires the same logical device and runtime")
            settings, data, _, _, binding = _initial_inputs(**manifest["inputs"])
            if canonical_hash(binding) != canonical_hash(manifest["input_binding"]):
                raise ValueError("teacher frozen input identities changed")
            checkpoint = _latest(directory)
            saved, metadata = load_cycle_checkpoint(checkpoint)
            if metadata.get("run_identity") != canonical_hash(manifest):
                raise ValueError("teacher checkpoint belongs to a different run manifest")
            if set(saved) != {"controller", "model", "teacher", "critic", "learner", "streams", "sampler", "last_report"}:
                raise ValueError("teacher checkpoint state has missing or unsupported keys")
            seed = manifest["inputs"]["master_seed"]
            model = build_model(settings, master_seed=seed, variant="D").to(device)
            teacher, critic = _policies(settings, seed, device)
            for network, name in ((model, "model"), (teacher, "teacher"), (critic, "critic")):
                network.load_state_dict(saved[name], strict=True)
            result = cls(directory, settings, seed, model, teacher, critic, data, precision=manifest["precision"])
            result.manifest, result.state = manifest, deepcopy(saved["controller"])
            result.learner.load_state_dict(saved["learner"])
            result.streams.load_state_dict(saved["streams"])
            result.anchor_sampler.load_state_dict(saved["sampler"])
            result.last_report = saved["last_report"]
            result.last_checkpoint = checkpoint
            result._validate_state()
            if (metadata != result._checkpoint_metadata()
                    or int(_NUMBERED.fullmatch(checkpoint.name)[1]) != result.state["sequence"]):
                raise ValueError("teacher checkpoint counters or metadata are inconsistent")
            if result.state["phase"] == "complete":
                result._freeze_teacher()
            return result

    def request_stop(self):
        """Stop after the current whole PPO cycle has been committed."""
        self._stop_requested = True

    def _validate_state(self):
        if (set(self.state) != {"sequence", "phase", "imagined_joint_steps", "teacher_player_decisions"}
                or self.state["phase"] not in ("teacher", "complete")):
            raise ValueError("invalid teacher controller state")
        if any(type(self.state[key]) is not int or self.state[key] < 0
               for key in ("sequence", "imagined_joint_steps", "teacher_player_decisions")):
            raise ValueError("teacher counters must be nonnegative integers")
        cycles, p = self.learner.cycles_completed, self.manifest["protocol"]
        if (type(cycles) is not int or not 0 <= cycles <= p["teacher_cycles"]
                or self.learner.actor_optimizer_steps != cycles * p["epochs"] * p["minibatches"]
                or self.learner.critic_optimizer_steps != cycles * p["epochs"] * p["minibatches"]
                or self.state["teacher_player_decisions"] != 2 * self.state["imagined_joint_steps"]
                or not cycles * p["starts"] <= self.state["imagined_joint_steps"] <= cycles * p["starts"] * p["horizon"]
                or (self.state["phase"] == "complete") != (cycles == p["teacher_cycles"])):
            raise ValueError("teacher cycle, optimizer or imagined-step accounting differs")
        if (module_version(self.model) != self.manifest["initial_model_version"]
                or any(parameter.requires_grad or parameter.grad is not None for parameter in self.model.parameters())):
            raise ValueError("frozen shared world model was modified or acquired gradients")
        if self.streams.identity != self.manifest["rollout_stream_identity"]:
            raise ValueError("teacher rollout RNG namespace changed")

    def _checkpoint_metadata(self):
        return {"schema_version": "p41-teacher-checkpoint-v1", "policy_role": "global_teacher",
                "run_identity": canonical_hash(self.manifest), "sequence": self.state["sequence"],
                "phase": self.state["phase"], "teacher_cycles_completed": self.learner.cycles_completed,
                "teacher_actor_optimizer_steps": self.learner.actor_optimizer_steps,
                "teacher_critic_optimizer_steps": self.learner.critic_optimizer_steps,
                "initial_model_version": self.manifest["initial_model_version"],
                "model_updates": self.manifest["protocol"]["initial_model_updates"],
                "real_joint_transitions": self.manifest["protocol"]["initial_transitions"],
                "new_environment_interactions": 0, "new_model_updates": 0,
                "local_student_optimizer_steps": 0, "frozen_for_distillation": self.state["phase"] == "complete"}

    def _checkpoint(self):
        self._validate_state()
        self.state["sequence"] += 1
        path = self.directory / "checkpoints" / f"state_{self.state['sequence']:08d}_{uuid.uuid4().hex[:8]}"
        saved = {"controller": deepcopy(self.state), "model": self.model.state_dict(),
                 "teacher": self.teacher.state_dict(), "critic": self.critic.state_dict(),
                 "learner": self.learner.state_dict(), "streams": self.streams.state_dict(),
                 "sampler": self.anchor_sampler.state_dict(), "last_report": getattr(self, "last_report", None)}
        save_cycle_checkpoint(path, state=saved, metadata=self._checkpoint_metadata())
        self.last_checkpoint = path
        return path

    def _freeze_teacher(self):
        for network in (self.teacher, self.critic):
            network.zero_grad(set_to_none=True)
            network.requires_grad_(False).eval()

    def _publish_frozen_teacher(self):
        """Publish receipt-bound handoff metadata to the completed checkpoint."""
        self._validate_state()
        if self.state["phase"] != "complete":
            raise ValueError("only a completed teacher can be frozen for P4.2")
        self._freeze_teacher()
        report = {"schema_version": "p41-frozen-teacher-v1", "policy_role": "global_teacher",
                  "run_identity": canonical_hash(self.manifest),
                  "checkpoint": self.last_checkpoint.relative_to(self.directory).as_posix(),
                  "checkpoint_manifest_sha256": file_sha256(self.last_checkpoint / "manifest.json"),
                  "teacher_version": module_version(self.teacher), "critic_version": module_version(self.critic),
                  "model_version": module_version(self.model), "teacher_config": asdict(self.teacher.config),
                  "attention_heads": self.teacher.attention_heads, "teacher_cycles": self.learner.cycles_completed,
                  "frozen_for_distillation": True, "deployable_local_actor": False,
                  "student_training_executed": False, "research_quality_acceptance": False}
        path = self.directory / "frozen_teacher.json"
        # A failed publication can be regenerated deterministically from its
        # completed checkpoint; training is never repeated to repair this file.
        receipt = path.with_suffix(".sha256")
        if receipt.exists():
            if (not path.is_file() or receipt.read_text(encoding="ascii").strip()
                    != f"{file_sha256(path)}  {path.name}"):
                raise ValueError("committed frozen teacher handoff receipt mismatch")
            if canonical_hash(json.loads(path.read_text(encoding="utf-8"))) != canonical_hash(report):
                raise ValueError("committed frozen teacher handoff differs from completed checkpoint")
            return
        atomic_json(path, report)
        temporary = path.with_suffix(".sha256.tmp")
        if temporary.exists():
            temporary.unlink()
        _publish_receipt(path)

    def _status(self, status, **extra):
        p = self.manifest["protocol"]
        report = {"schema_version": STATUS_SCHEMA, "status": status, **deepcopy(self.state),
            "task_family": "relay", "variant": "D", "policy_role": "global_teacher", "run_seed": self.master_seed,
            "teacher_cycles_completed": self.learner.cycles_completed, "target_teacher_cycles": p["teacher_cycles"],
            "teacher_actor_optimizer_steps": self.learner.actor_optimizer_steps,
            "teacher_critic_optimizer_steps": self.learner.critic_optimizer_steps,
            "target_teacher_actor_optimizer_steps": p["teacher_cycles"] * p["epochs"] * p["minibatches"],
            "target_teacher_critic_optimizer_steps": p["teacher_cycles"] * p["epochs"] * p["minibatches"],
            "model_updates": p["initial_model_updates"], "new_model_updates": 0,
            "real_joint_transitions": p["initial_transitions"], "new_environment_interactions": 0,
            "local_student_optimizer_steps": 0, "full_d_pipeline_implemented": False,
            "frozen_teacher_available": (self.directory / "frozen_teacher.sha256").is_file(),
            "last_checkpoint": None if self.last_checkpoint is None else str(self.last_checkpoint), **extra}
        atomic_json(self.directory / "status.json", report)
        return report

    def status(self):
        """Read the most recently published status without running any work."""
        return json.loads((self.directory / "status.json").read_text(encoding="utf-8"))

    def run(self, *, max_cycles=None):
        """Run additional whole cycles, checkpointing after every complete one."""
        if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 1):
            raise ValueError("max_cycles must be a positive integer")
        if self._failed:
            raise RuntimeError("teacher worker failed; resume its last committed checkpoint in a fresh instance")
        with run_lock(self.directory):
            if canonical_hash(_manifest(self.directory)) != canonical_hash(self.manifest):
                raise ValueError("teacher run manifest changed")
            if self.manifest["source_sha256"] != source_identity():
                raise ValueError("teacher source changed; restore its prepared checkout")
            if self.manifest["runtime"] != _runtime(self.device):
                raise ValueError("teacher runtime changed after preparation or resume")
            if self.last_checkpoint != _latest(self.directory):
                raise RuntimeError("another worker advanced this teacher; resume the latest committed checkpoint")
            _, _, _, _, binding = _initial_inputs(**self.manifest["inputs"])
            if canonical_hash(binding) != canonical_hash(self.manifest["input_binding"]):
                raise ValueError("teacher frozen inputs changed")
            self._validate_state()
            cycles, invocation, attempt, cycle_started = 0, uuid.uuid4().hex, None, None
            stop_file = self.directory / "stop.request"
            try:
                while (self.state["phase"] != "complete" and not self._stop_requested and not stop_file.exists()
                       and (max_cycles is None or cycles < max_cycles)):
                    self._status("running")
                    attempt = uuid.uuid4().hex
                    cycle_started = time.perf_counter()
                    # Intent is fsynced before any new sampling or PPO work.
                    # An unmatched intent means interrupted physical work;
                    # optimizer counts below describe committed logical work.
                    append_event(self.directory / "attempts.jsonl", {
                        "event": "teacher_cycle_started", "invocation": invocation, "attempt": attempt,
                        "teacher_cycle": self.learner.cycles_completed + 1,
                        "last_committed_checkpoint": str(self.last_checkpoint),
                        "created_at": datetime.now(timezone.utc).isoformat()})
                    anchors = self.builder.build(self.anchor_sampler)
                    rollout = self.generator.collect(anchors, streams=self.streams)
                    report = self.learner.update(rollout, self.anchor_sampler)
                    self.last_report = report
                    self.state["imagined_joint_steps"] += report["imagined_joint_steps"]
                    self.state["teacher_player_decisions"] += 2 * report["imagined_joint_steps"]
                    if self.learner.cycles_completed == self.manifest["protocol"]["teacher_cycles"]:
                        self.state["phase"] = "complete"
                    # Metrics become scientific evidence only when linked to
                    # the immutable checkpoint containing the same report.
                    self._checkpoint()
                    append_event(self.directory / "attempts.jsonl", {
                        "event": "teacher_cycle_committed", "invocation": invocation, "attempt": attempt,
                        "teacher_cycle": self.learner.cycles_completed,
                        "checkpoint": str(self.last_checkpoint),
                        "actor_optimizer_steps": report["cycle_actor_optimizer_steps"],
                        "critic_optimizer_steps": report["cycle_critic_optimizer_steps"],
                        "imagined_joint_steps": report["imagined_joint_steps"],
                        "elapsed_seconds_including_checkpoint": time.perf_counter() - cycle_started,
                        "created_at": datetime.now(timezone.utc).isoformat()})
                    append_event(self.directory / "metrics.jsonl", {"event": "teacher_ppo", "invocation": invocation,
                        "attempt": attempt, "checkpoint_sequence": self.state["sequence"],
                        "checkpoint": str(self.last_checkpoint), **report})
                    self._status("running")
                    cycles += 1
                    attempt, cycle_started = None, None
                if self.state["phase"] == "complete":
                    self._publish_frozen_teacher()
                if stop_file.is_file():
                    stop_file.unlink()
                return self._status("complete" if self.state["phase"] == "complete" else "stopped",
                                    cycles_executed=cycles)
            except BaseException as error:
                self._failed = True
                try:
                    append_event(self.directory / "attempts.jsonl", {
                        "event": "teacher_invocation_failed", "invocation": invocation, "attempt": attempt,
                        "last_committed_checkpoint": str(self.last_checkpoint),
                        "partial_physical_optimizer_work": ("unknown; do not count interrupted work as zero"
                            if attempt is not None else "none; no current teacher cycle"),
                        "elapsed_attempt_seconds": None if cycle_started is None else time.perf_counter() - cycle_started,
                        "error_type": type(error).__name__, "error": str(error)})
                except OSError:
                    # Preserve the original error; a durable unmatched start
                    # still exposes interrupted work if the disk is unavailable.
                    pass
                self._status("failed", error_type=type(error).__name__, error=str(error))
                raise
