"""Variant B's durable real-data, world-model and local-policy learning cycle."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import torch

from world_models.data.ledger import atomic_json, file_sha256
from world_models.data.replay import ReplayDataset
from world_models.imagination import AnchorBuilder, AnchorSampler
from world_models.imagination.versions import module_version
from world_models.infrastructure.config import validate_config
from world_models.infrastructure.reproducibility import canonical_hash, derive_seed
from world_models.mappo import RecurrentMAPPO
from world_models.models.world_model import build_model
from world_models.policies import build_actor_critic
from world_models.rollouts import RolloutGenerator, RolloutStreams
from world_models.training.checkpoints import read_checkpoint
from world_models.training.fitting import FitSettings
from world_models.training.normalization import NormalizationStats
from world_models.training.runner import append_event, run_lock, runtime_contract, selected_device, source_identity as p24_sources
from .checkpoints import load_cycle_checkpoint, save_cycle_checkpoint
from .collection import collect_policy_block
from .provenance import source_identity
from .refresh import ModelRefresher
from .schedule import Schedule


@dataclass(frozen=True)
class CycleOptions:
    precision: str = "bfloat16"
    checkpoint_cycles: int = 10
    checkpoint_model_updates: int = 250
    model_microbatch: int = 4

    def __post_init__(self):
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")
        for name in ("checkpoint_cycles", "checkpoint_model_updates", "model_microbatch"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


def _replay_record(dataset):
    return {"path": str(dataset.path.resolve()), "manifest_sha256": dataset.manifest_sha256,
            "dataset_identity_sha256": dataset.manifest["dataset_identity_sha256"],
            "transitions": dataset.manifest["transition_count"]}


def _runtime(device):
    return {**runtime_contract(device), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cpu_threads": torch.get_num_threads(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _latest_checkpoint(directory):
    choices = [path for path in (directory / "checkpoints").glob("state_*")
               if (path / "manifest.sha256").is_file()]
    return max(choices, key=lambda path: path.name) if choices else None


class CycleController:
    """Run one B task/seed on one selected GPU, with explicit prepare/resume.

    Only run() performs training/collection. Construction and plan/status tools
    never start a research job. Teacher/distillation integration is Phase 4;
    claiming a D run here would silently omit that treatment, so it is rejected.
    """

    def __init__(self, directory, settings, master_seed, model, actor, critic, normalization, *, options,
                 collector=collect_policy_block, fit_settings=None):
        validate_config(settings)
        if settings["variant"]["name"] != "B":
            raise ValueError("P3.5 runs variant B; variant D requires the Phase 4 teacher/distillation controller")
        if type(master_seed) is not int or master_seed not in settings["experiment"]["master_seeds"]:
            raise ValueError("master_seed must belong to the configured profile")
        self.directory = Path(directory).resolve()
        self.settings, self.master_seed, self.options = deepcopy(settings), master_seed, options
        self.task, self.variant = settings["task"]["name"], "B"
        self.schedule = Schedule.from_experiment(settings)
        self.model, self.actor, self.critic, self.normalization = model, actor, critic, normalization
        self.collector = collector
        self.fit_settings = fit_settings or FitSettings.from_experiment(settings, microbatch=options.model_microbatch,
                                                                       precision=options.precision)
        self.learner = RecurrentMAPPO.from_experiment(model, actor, critic, settings, master_seed=master_seed,
                                                      precision=options.precision)
        self.rollout_streams = RolloutStreams.from_experiment(settings, master_seed=master_seed,
                                                              device=next(model.parameters()).device)
        self.anchor_seed = derive_seed(self.task, master_seed, "imagination_anchors", "B")
        self.prefix_seed = derive_seed(self.task, master_seed, "imagination_prefixes", "B")
        self.builder = AnchorBuilder(model, actor, seed=self.prefix_seed, precision=options.precision)
        self.collector_model = RolloutGenerator.from_experiment(model, actor, critic, settings, precision=options.precision)
        self.refresher = None
        self.datasets = []
        self.anchor_sampler = None
        self.state = {}
        self.manifest = {}
        self._stop_requested = False
        self._failed = False
        self.last_checkpoint = None

    @classmethod
    def prepare(cls, directory, *, settings, master_seed, initial_dataset, normalization, initial_checkpoint=None,
                initial_selection=None, study_amendment=None,
                device="cuda:0", options=None, collector=collect_policy_block):
        """Bind shared inputs, including an explicitly frozen shortened pilot."""
        validate_config(settings)
        if settings["variant"]["name"] != "B":
            raise ValueError("P3.5 preparation supports B only; D needs Phase 4")
        options = options or CycleOptions(precision=settings["resources"]["mixed_precision"])
        device = selected_device(device, options.precision)
        if (initial_checkpoint is None) == (initial_selection is None):
            raise ValueError("provide exactly one initial checkpoint or frozen initial selection")
        parent_settings = settings
        if study_amendment is not None:
            from .study import parent_settings_for_study
            if initial_selection is None:
                raise ValueError("study amendment requires a frozen initial selection")
            parent_settings = parent_settings_for_study(settings, study_amendment, master_seed=master_seed)
        registered_schedule = Schedule.from_experiment(settings)
        selection_binding = None
        if initial_selection is None:
            schedule = registered_schedule
        else:
            amended = {"study_amendment": study_amendment} if study_amendment is not None else {}
            schedule, selection_binding = Schedule.from_initial_selection(settings, master_seed, initial_selection, **amended)
            initial_checkpoint = selection_binding["checkpoint_path"]
        data = initial_dataset if isinstance(initial_dataset, ReplayDataset) else ReplayDataset(initial_dataset)
        stats = normalization if isinstance(normalization, NormalizationStats) else NormalizationStats.load(normalization)
        stats.validate_for_dataset(data)
        payload = read_checkpoint(initial_checkpoint)
        metadata = payload["metadata"]
        common = deepcopy(parent_settings)
        common.pop("variant")
        if (payload["update"] != schedule.initial_model_updates or metadata.get("stage") != "initial"
                or metadata.get("variant") != "shared_initial:B,D"
                or metadata.get("target_stage_updates") != registered_schedule.initial_model_updates
                or metadata.get("task_family") != settings["task"]["name"] or metadata.get("run_seed") != master_seed
                or metadata.get("common_config_sha256") != canonical_hash(common)
                or metadata.get("source_sha256") != p24_sources()
                or metadata.get("initial_normalization", {}).get("identity_sha256") != stats.as_dict()["identity_sha256"]):
            raise ValueError("initial checkpoint must match the shared P2.4 fit, frozen normalization and selected update budget")
        if selection_binding is not None and file_sha256(initial_checkpoint) != selection_binding["checkpoint_sha256"]:
            raise ValueError("selected initial checkpoint changed during preparation")
        replay_rows = metadata.get("replays", [])
        if (len(replay_rows) != 1 or replay_rows[0].get("dataset_identity_sha256") != data.manifest["dataset_identity_sha256"]
                or replay_rows[0].get("manifest_sha256") != data.manifest_sha256
                or replay_rows[0].get("transitions") != schedule.initial_transitions
                or data.manifest["transition_count"] != schedule.initial_transitions):
            raise ValueError("P2.4 input replay differs from the approved shared initial budget")
        model = build_model(settings, master_seed=master_seed, variant="B").to(device)
        model.load_state_dict(payload["model"], strict=True)
        policies = build_actor_critic(settings, master_seed=master_seed, variant="B").to(device)
        result = cls(directory, settings, master_seed, model, policies.actor, policies.critic, stats,
                     options=options, collector=collector)
        result.schedule = schedule
        result.refresher = ModelRefresher(model, stats, settings, master_seed=master_seed, precision=options.precision,
                                           fit_settings=result.fit_settings, initial_checkpoint=payload,
                                           study_amendment=study_amendment)
        result._set_replay([data])
        result.state = {"sequence": 0, "block_index": 0, "phase": "local", "real_joint_transitions": schedule.initial_transitions,
                        "model_updates_in_block": 0, "local_cycles_in_block": 0,
                        "imagined_joint_steps": 0, "local_player_decisions": 0, "evaluation_exports": []}
        result.manifest = {"schema_version": "p35-run-v1", "settings": deepcopy(settings), "master_seed": master_seed,
                           "options": asdict(options), "fit_settings": asdict(result.fit_settings),
                           "schedule": result.schedule.describe(), "source_sha256": source_identity(),
                           "runtime": _runtime(device), "device": str(device),
                           "normalization": stats.as_dict(), "initial_checkpoint": str(Path(initial_checkpoint).resolve()),
                           "initial_checkpoint_sha256": file_sha256(initial_checkpoint),
                           "initial_replay": _replay_record(data), "created_at": datetime.now(timezone.utc).isoformat(),
                           "research_quality_acceptance": False}
        if selection_binding is not None:
            result.manifest.update(initial_selection=selection_binding,
                initial_training_policy="frozen_last_saved_checkpoints", study_scope="exploratory_pilot",
                registered_schedule=registered_schedule.describe())
        if study_amendment is not None:
            result.manifest.update(study_amendment=deepcopy(study_amendment),
                                   parent_schedule=Schedule.from_experiment(parent_settings).describe())
        if result.directory.exists():
            raise FileExistsError("prepare requires a fresh run directory")
        result.directory.mkdir(parents=True)
        with run_lock(result.directory):
            atomic_json(result.directory / "run_manifest.json", result.manifest)
            # The zero-interaction baseline is a paired untrained local actor;
            # initial collection/model fitting have already happened externally.
            result._export_actor(0, baseline=True)
            result._checkpoint()
        return result

    @classmethod
    def resume(cls, directory, *, device="cuda:0", collector=collect_policy_block):
        directory = Path(directory).resolve()
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "p35-run-v1" or manifest.get("source_sha256") != source_identity():
            raise ValueError("run source identity changed; use the saved source checkout")
        options = CycleOptions(**manifest["options"])
        device = selected_device(device, options.precision)
        if manifest["device"] != str(device) or manifest["runtime"] != _runtime(device):
            raise ValueError("exact resume requires the same logical device and runtime")
        checkpoint = _latest_checkpoint(directory)
        if checkpoint is None:
            raise ValueError("no committed P3.5 checkpoint; preserve this run directory")
        return cls._restore_checkpoint(directory, manifest, checkpoint, device=device, collector=collector)

    @classmethod
    def _restore_checkpoint(cls, directory, manifest, checkpoint, *, device, collector=collect_policy_block):
        """Restore verified state after the caller validates source and runtime.

        Ordinary resume requires identical sources. The explicit numerical
        recovery command separately validates its narrowly allowed amendment
        before using this shared restoration and state-validation path.
        """
        options = CycleOptions(**manifest["options"])
        saved, metadata = load_cycle_checkpoint(checkpoint, device="cpu")
        if metadata.get("run_identity") != canonical_hash(manifest):
            raise ValueError("checkpoint is not bound to this run manifest")
        settings, seed = manifest["settings"], manifest["master_seed"]
        schedule = Schedule.from_manifest(manifest)
        model = build_model(settings, master_seed=seed, variant="B").to(device)
        policies = build_actor_critic(settings, master_seed=seed, variant="B").to(device)
        for network, key in ((model, "model"), (policies.actor, "actor"), (policies.critic, "critic")):
            network.load_state_dict(saved[key], strict=True)
        result = cls(directory, settings, seed, model, policies.actor, policies.critic,
                     NormalizationStats.from_dict(manifest["normalization"]), options=options, collector=collector,
                     fit_settings=FitSettings(**manifest["fit_settings"]))
        result.manifest = manifest
        result.schedule = schedule
        result.state = deepcopy(saved["controller"])
        result.datasets = []
        for record in saved["replays"]:
            dataset = ReplayDataset(record["path"])
            if _replay_record(dataset) != record:
                raise ValueError("saved real replay bytes or locations changed")
            result.datasets.append(dataset)
        result.refresher = ModelRefresher(model, result.normalization, settings, master_seed=seed,
                                           precision=options.precision, fit_settings=result.fit_settings,
                                           study_amendment=manifest.get("study_amendment"))
        result._set_replay(result.datasets)
        result.refresher.load_state_dict(saved["refresher"])
        result.learner.load_state_dict(saved["mappo"])
        result.rollout_streams.load_state_dict(saved["rollout_streams"])
        result.anchor_sampler.load_state_dict(saved["anchor_sampler"])
        result._validate_state()
        result.last_checkpoint = checkpoint
        return result

    def request_stop(self):
        """Finish the current unit (a complete real block, fit update or PPO cycle)."""
        self._stop_requested = True

    def _set_replay(self, datasets):
        for index, data in enumerate(datasets):
            meta = data.manifest["metadata"]
            if (meta.get("task_family") != self.task or meta.get("run_seed") != self.master_seed
                    or (index and meta.get("training_variant") != self.variant)):
                raise ValueError("real replay belongs to a different task, seed or training branch")
        previous = None if self.anchor_sampler is None else self.anchor_sampler.state_dict()["rng_state"]
        sampler = AnchorSampler.from_experiment(datasets, self.settings, seed=self.anchor_seed)
        if previous is not None:
            state = sampler.state_dict()
            state["rng_state"] = previous
            sampler.load_state_dict(state)
        self.datasets = list(datasets)
        self.anchor_sampler = sampler
        self.refresher.set_replay(datasets)

    def _validate_state(self):
        state, schedule = self.state, self.schedule
        expected = {"sequence", "block_index", "phase", "real_joint_transitions", "model_updates_in_block",
                    "local_cycles_in_block", "imagined_joint_steps", "local_player_decisions", "evaluation_exports"}
        if set(state) != expected or state["phase"] not in ("collect", "refresh", "local", "complete"):
            raise ValueError("invalid cycle controller state")
        for name in expected - {"phase", "evaluation_exports"}:
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError("cycle counters must be nonnegative integers")
        block = state["block_index"]
        attached = block - (state["phase"] == "collect")
        if (block > schedule.blocks or attached < 0 or len(self.datasets) != attached + 1
                or state["real_joint_transitions"] != schedule.initial_transitions + attached * schedule.collection_block
                or sum(data.manifest["transition_count"] for data in self.datasets) != state["real_joint_transitions"]
                or state["local_player_decisions"] != 2 * state["imagined_joint_steps"]):
            raise ValueError("replay, real-budget or imagined-decision accounting is inconsistent")
        if state["local_cycles_in_block"] > self.schedule.local_cycles(block):
            raise ValueError("too many local cycles in this block")
        if state["model_updates_in_block"] > (schedule.online_model_updates if block else 0):
            raise ValueError("too many model updates in this block")
        if state["phase"] == "collect" and (state["model_updates_in_block"] or state["local_cycles_in_block"]):
            raise ValueError("new real collection must start with zero block optimizer counters")
        if state["phase"] == "refresh" and (block == 0 or state["local_cycles_in_block"]
                or state["model_updates_in_block"] >= schedule.online_model_updates):
            raise ValueError("refresh must precede all local PPO cycles in an online block")
        if state["phase"] in ("local", "complete") and block and state["model_updates_in_block"] != schedule.online_model_updates:
            raise ValueError("local PPO requires the complete model-refresh schedule")
        prior_cycles = 0 if block == 0 else schedule.initial_local_cycles + (block - 1) * schedule.online_local_cycles
        if self.learner.cycles_completed != prior_cycles + state["local_cycles_in_block"]:
            raise ValueError("PPO optimizer cycle count differs from the controller schedule")
        expected_model = schedule.initial_model_updates + max(0, attached - 1) * schedule.online_model_updates
        if attached and state["phase"] != "collect":
            expected_model += state["model_updates_in_block"]
        elif state["phase"] == "collect":
            expected_model = schedule.initial_model_updates + attached * schedule.online_model_updates
        if (self.refresher.initial_model_updates != schedule.initial_model_updates
                or self.refresher.model_updates != expected_model):
            raise ValueError("world-model update count differs from the controller schedule")
        if state["phase"] == "complete" and (block != schedule.blocks
                or state["local_cycles_in_block"] != schedule.local_cycles(block)
                or state["real_joint_transitions"] != schedule.real_budget
                or self.refresher.model_updates != schedule.total_model_updates
                or self.learner.cycles_completed != schedule.total_local_cycles):
            raise ValueError("complete requires the full real budget and local PPO schedule")

    def _checkpoint(self, *, completed_block=False):
        self._validate_state()
        self.state["sequence"] += 1
        destination = self.directory / "checkpoints" / f"state_{self.state['sequence']:08d}_{uuid.uuid4().hex[:8]}"
        state = {"controller": deepcopy(self.state), "model": self.model.state_dict(), "actor": self.actor.state_dict(),
                 "critic": self.critic.state_dict(), "refresher": self.refresher.state_dict(),
                 "mappo": self.learner.state_dict(), "rollout_streams": self.rollout_streams.state_dict(),
                 "anchor_sampler": self.anchor_sampler.state_dict(), "replays": [_replay_record(data) for data in self.datasets]}
        save_cycle_checkpoint(destination, state=state, metadata={"run_identity": canonical_hash(self.manifest),
            "phase": self.state["phase"], "block_index": self.state["block_index"], "completed_block": completed_block,
            "real_joint_transitions": self.state["real_joint_transitions"], "evaluation_executed": False})
        self.last_checkpoint = destination
        self._status("complete" if self.state["phase"] == "complete" else "ready")
        return destination

    def _export_actor(self, interactions, *, baseline=False):
        """Immutable local-only weights and evaluation metadata; no evaluation run."""
        base = self.directory / "evaluation" / f"real_{interactions:09d}"
        actor_version = module_version(self.actor)
        metadata = {"schema_version": "p35-local-actor-v1", "task_family": self.task, "variant": self.variant,
                    "run_seed": self.master_seed, "real_joint_transitions": interactions,
                    "actual_initial_replay_interactions": self.schedule.initial_transitions,
                    "untrained_zero_interaction_baseline": baseline, "actor_version": actor_version,
                    "actor_config": json.loads(json.dumps(asdict(self.actor.config))),
                    "world_model_version": module_version(self.model), "model_updates": self.refresher.model_updates,
                    "local_ppo_cycles": self.learner.cycles_completed,
                    "actor_optimizer_steps": self.learner.actor_optimizer_steps,
                    "critic_optimizer_steps": self.learner.critic_optimizer_steps,
                    "evaluation_split": self.settings["experiment"]["evaluation_split"],
                    "on_registered_evaluation_grid": interactions in self.settings["experiment"]["checkpoints"],
                    "evaluation_executed": False, "phase3_acceptance": False}
        choices = sorted(path for path in base.parent.glob(base.name + "*") if (path / "manifest.sha256").is_file())
        if choices:
            destination = choices[0]
            for path in choices:
                _, previous = load_cycle_checkpoint(path)
                if previous != metadata:
                    raise ValueError("existing evaluation export differs; preserve the earlier result")
        else:
            # An interrupted pure-data export contains no consumed simulator
            # work. Retain its incomplete directory and publish a fresh attempt.
            destination = base if not base.exists() else base.with_name(base.name + "_attempt_" + uuid.uuid4().hex[:8])
            save_cycle_checkpoint(destination, state={"actor": self.actor.state_dict()}, metadata=metadata)
        row = {"real_joint_transitions": interactions, "path": destination.relative_to(self.directory).as_posix(),
               "actor_version": actor_version}
        if row not in self.state["evaluation_exports"]:
            self.state["evaluation_exports"].append(row)
        atomic_json(self.directory / "evaluation_checkpoints.json", self.state["evaluation_exports"])

    def _status(self, status, **extra):
        report = {"schema_version": "p35-status-v1", "status": status, **deepcopy(self.state),
                  "task_family": self.task, "run_seed": self.master_seed, "variant": self.variant,
                  "real_budget": self.schedule.real_budget, "local_cycles_completed": self.learner.cycles_completed,
                  "target_local_cycles": self.schedule.total_local_cycles, "model_updates": self.refresher.model_updates,
                  "initial_model_updates": self.schedule.initial_model_updates,
                  "registered_initial_model_updates": self.settings["world_model"]["initial_updates"],
                  "target_model_updates": self.schedule.total_model_updates,
                  "actor_optimizer_steps": self.learner.actor_optimizer_steps,
                  "critic_optimizer_steps": self.learner.critic_optimizer_steps,
                  "last_checkpoint": None if self.last_checkpoint is None else str(self.last_checkpoint), **extra}
        atomic_json(self.directory / "status.json", report)
        return report

    def _attach_collection(self):
        index = self.state["block_index"]
        destination = self.directory / "real_replay" / f"block_{index:04d}"
        if destination.exists():
            report_path = destination / "collection_report.json"
            if not report_path.is_file():
                raise RuntimeError("interrupted real collection must be reconciled from its durable ledger; automatic recollection is refused")
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = self.collector(self.task, self.master_seed, self.variant, destination, actor=self.actor,
                settings=self.settings, block_index=index - 1, total_transitions=self.schedule.collection_block)
        self._validate_collection_report(report, destination)
        dataset = ReplayDataset(destination / report["dataset_path"])
        if (dataset.manifest_sha256 != report["dataset_manifest_sha256"]
                or dataset.manifest["dataset_identity_sha256"] != report["dataset_identity_sha256"]
                or dataset.manifest["transition_count"] != self.schedule.collection_block
                or dataset.manifest["metadata"].get("collection_block") != index - 1):
            raise ValueError("sealed collection manifest, identity or block budget differs")
        self._set_replay([*self.datasets, dataset])
        self.state["real_joint_transitions"] += self.schedule.collection_block
        self.state["phase"] = "refresh"
        self._checkpoint()

    def _validate_collection_report(self, report, destination):
        if (report.get("status") != "complete_online_replay"
                or report.get("task_family") != self.task or report.get("run_seed") != self.master_seed
                or report.get("training_variant") != self.variant
                or report.get("block_index") != self.state["block_index"] - 1
                or report.get("published_joint_transitions") != self.schedule.collection_block
                or report.get("actor_version_before") != module_version(self.actor)
                or report.get("actor_version_after") != module_version(self.actor)
                or report.get("dataset_path") != "replay"):
            raise ValueError("existing online collection is incomplete or belongs to another policy/budget")
        accounting = report.get("interaction_accounting", {})
        if (accounting.get("published_real_joint_transitions") != self.schedule.collection_block
                or accounting.get("replay_committed_joint_transitions") != self.schedule.collection_block
                or accounting.get("failed_dispatch_attempts") != 0):
            raise ValueError("online collection ledger is incomplete or contains failed consumed work")
        if (accounting.get("ledger_path") != "training_interactions.jsonl"
                or file_sha256(destination / "training_interactions.jsonl") != accounting.get("ledger_sha256")
                or file_sha256(destination / "collection_spec.json") != report.get("collection_spec_sha256")):
            raise ValueError("online collection ledger or specification changed since sealing")

    def run(self, *, max_units=None):
        """Continue until complete, a safe stop, or an optional unit limit.

        A unit is one entire real block, one model optimizer update, one PPO
        collection/update cycle, or one completed-block export transition.
        """
        if max_units is not None and (type(max_units) is not int or max_units < 1):
            raise ValueError("max_units must be a positive integer")
        if self._failed:
            raise RuntimeError("this controller failed; create a fresh resume instance from the last committed checkpoint")
        units = 0
        with run_lock(self.directory):
            if self.manifest.get("source_sha256") != source_identity():
                raise ValueError("run source changed; restore the prepared checkout before continuing")
            if self.last_checkpoint != _latest_checkpoint(self.directory):
                raise RuntimeError("another worker advanced this run; reload the latest checkpoint before continuing")
            self._validate_state()
            invocation = uuid.uuid4().hex
            try:
                stop_file = self.directory / "stop.request"
                while (self.state["phase"] != "complete" and not self._stop_requested and not stop_file.exists()
                       and (max_units is None or units < max_units)):
                    phase = self.state["phase"]
                    self._status("running")
                    if phase == "collect":
                        self._attach_collection()
                    elif phase == "refresh":
                        before_actor, before_critic = module_version(self.actor), module_version(self.critic)
                        metrics = self.refresher.update()
                        self.state["model_updates_in_block"] += 1
                        if (before_actor, before_critic) != (module_version(self.actor), module_version(self.critic)):
                            raise RuntimeError("world-model refresh changed policy parameters")
                        append_event(self.directory / "metrics.jsonl", {"event": "model_fit", "invocation": invocation,
                            "checkpoint_sequence_before": self.state["sequence"], "block_index": self.state["block_index"], **metrics})
                        if self.state["model_updates_in_block"] == self.schedule.online_model_updates:
                            self.state["phase"] = "local"
                            self._checkpoint()
                        elif self.state["model_updates_in_block"] % self.options.checkpoint_model_updates == 0:
                            self._checkpoint()
                    else:
                        block = self.state["block_index"]
                        if self.state["local_cycles_in_block"] == self.schedule.local_cycles(block):
                            self._export_actor(self.state["real_joint_transitions"])
                            if block == self.schedule.blocks:
                                self.state["phase"] = "complete"
                            else:
                                self.state.update(block_index=block + 1, phase="collect", model_updates_in_block=0, local_cycles_in_block=0)
                            self._checkpoint(completed_block=True)
                        else:
                            anchors = self.builder.build(self.anchor_sampler)
                            rollout = self.collector_model.collect(anchors, streams=self.rollout_streams)
                            report = self.learner.update(rollout, self.anchor_sampler)
                            self.state["local_cycles_in_block"] += 1
                            self.state["imagined_joint_steps"] += report["imagined_joint_steps"]
                            self.state["local_player_decisions"] += 2 * report["imagined_joint_steps"]
                            append_event(self.directory / "metrics.jsonl", {"event": "local_ppo", "invocation": invocation,
                                "checkpoint_sequence_before": self.state["sequence"], "block_index": block,
                                "real_joint_transitions": self.state["real_joint_transitions"], **report})
                            if self.state["local_cycles_in_block"] % self.options.checkpoint_cycles == 0:
                                self._checkpoint()
                    units += 1
                self._checkpoint()
                if stop_file.is_file():
                    stop_file.unlink()
                return self._status("complete" if self.state["phase"] == "complete" else "stopped", units_executed=units)
            except BaseException as error:
                # Do not publish a partially-mutated model/PPO state. Resume
                # from the last receipt; completed real blocks can be attached.
                self._failed = True
                extra = {}
                if self.state["phase"] == "collect":
                    failure = self.directory / "real_replay" / f"block_{self.state['block_index']:04d}" / "collection_error.json"
                    if failure.is_file():
                        extra["collection_failure"] = json.loads(failure.read_text(encoding="utf-8"))
                self._status("failed", error_type=type(error).__name__, error=str(error), **extra)
                raise
