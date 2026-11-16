"""Persistent online world-model fitting on guarded branch real replay.

This adapter borrows a model; its owner checkpoints model tensors separately.
Initial normalization and AdamW moments continue from the P2.4 checkpoint.
Only the branch's private fitting streams advance during online refreshes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import torch

from world_models.data.replay import ReplayDataset, canonical_hash, sha256_file
from world_models.imagination.versions import module_version
from world_models.infrastructure.config import validate_config
from world_models.infrastructure.reproducibility import derive_seed
from world_models.models.core import ModelConfig
from world_models.models.heads import TaskSpec
from world_models.models.world_model import SamplingStreams, SharedWorldModel
from world_models.training.checkpoints import read_checkpoint
from world_models.training.fitting import FitSettings, optimizer_update
from world_models.training.normalization import NormalizationStats
from world_models.training.objective import ObjectiveConfig, WorldModelObjective
from world_models.training.runner import source_identity as p24_source_identity
from world_models.training.sampling import ReplaySampler


SCHEMA_VERSION = "p35-model-refresher-v1"


def _copy(value):
    """Owned restricted-loader-compatible state, including detached CPU tensors."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_complex():
            raise ValueError("model refresher state requires dense real tensors")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("model refresher state contains nonfinite tensors")
        return value.detach().to("cpu", copy=True)
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not __import__("math").isfinite(value):
            raise ValueError("model refresher state contains a nonfinite scalar")
        return value
    if isinstance(value, dict) and all(type(key) in (str, int) for key in value):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy(item) for item in value]
    raise ValueError("unsupported model refresher state value")


class ModelRefresher:
    """One persistent AdamW optimizer and private fitting stream per branch.

    ``initial_checkpoint`` accepts a path or the result of ``read_checkpoint``.
    The caller loads model weights first. A payload dictionary avoids reading a
    large checkpoint twice; its receipt must have been verified by that reader.
    ``fit_settings`` permits explicit tiny CPU implementation fixtures without
    changing the registered production settings or normalization statistics.
    """

    def __init__(self, model, normalization, settings, *, master_seed,
                 precision="float32", fit_settings=None, initial_model_updates=0,
                 initial_checkpoint=None, study_amendment=None):
        validate_config(settings)
        if not isinstance(model, SharedWorldModel) or not isinstance(normalization, NormalizationStats):
            raise ValueError("refresh requires a SharedWorldModel and frozen NormalizationStats")
        if type(initial_model_updates) is not int or initial_model_updates < 0:
            raise ValueError("initial_model_updates must be a nonnegative integer")
        self.settings = deepcopy(settings)
        self._settings_identity = canonical_hash(self.settings)
        self._initial_parent_settings = deepcopy(settings)
        self._study_amendment = deepcopy(study_amendment)
        if study_amendment is not None:
            from .study import parent_settings_for_study
            self._initial_parent_settings = parent_settings_for_study(settings, study_amendment, master_seed=master_seed)
        self._initial_parent_identity = canonical_hash(self._initial_parent_settings)
        self._study_identity = canonical_hash(self._study_amendment)
        self.task_family = settings["task"]["name"]
        self.variant = settings["variant"]["name"]
        if master_seed not in settings["experiment"]["master_seeds"] or type(master_seed) is not int:
            raise ValueError("master_seed is outside the selected experiment")
        self.master_seed = master_seed
        if model.task != TaskSpec.from_experiment(settings) or model.config != ModelConfig.from_experiment(settings):
            raise ValueError("refresh model configuration differs from the resolved experiment")
        if normalization.task_family != self.task_family:
            raise ValueError("normalization task differs from the refresh model")
        # Frozen dataclasses still contain mutable mappings: take a deep,
        # schema-validated copy rather than retaining caller-owned statistics.
        self.normalization = NormalizationStats.from_dict(normalization.as_dict())
        self._normalization_identity = self.normalization.as_dict()["identity_sha256"]
        if precision not in ("float32", "bfloat16"):
            raise ValueError("refresh precision must be float32 or bfloat16")
        if fit_settings is not None and not isinstance(fit_settings, FitSettings):
            raise ValueError("fit_settings must be FitSettings")
        self.fit_settings = (replace(fit_settings, precision=precision) if fit_settings is not None else
                             FitSettings.from_experiment(settings, precision=precision))
        self._fit_settings_identity = asdict(self.fit_settings)
        self.model = model
        self.device = self._model_device()
        self.seed = derive_seed(self.task_family, master_seed, "world_model_fitting_online", self.variant)
        self.streams = SamplingStreams.from_seed(self.seed, self.device)
        self.optimizer = self._optimizer()
        self.objective = WorldModelObjective(model.task, self.normalization,
                                             ObjectiveConfig.from_experiment(settings))
        self.sampler = None
        self.datasets = ()
        self._cohort = []
        self._source_identity = {**p24_source_identity(),
            "src/world_models/cycle/refresh.py": sha256_file(Path(__file__)),
            "src/world_models/cycle/study.py": sha256_file(Path(__file__).with_name("study.py"))}
        self.initial_model_updates = initial_model_updates
        if initial_checkpoint is not None:
            parent = read_checkpoint(initial_checkpoint) if not isinstance(initial_checkpoint, dict) else initial_checkpoint
            self.initial_model_updates = self._load_initial(parent, initial_model_updates)
        elif initial_model_updates:
            raise ValueError("continuing nonzero model updates requires the P2.4 checkpoint optimizer moments")
        self.total_model_updates = self.initial_model_updates

    @property
    def model_updates(self):
        return self.total_model_updates

    def _model_device(self):
        tensors = (*self.model.parameters(), *self.model.buffers())
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1 or next(iter(devices)).type not in ("cpu", "cuda"):
            raise ValueError("refresh model must occupy one CPU or CUDA device")
        if any(tensor.is_floating_point() and tensor.dtype != torch.float32 for tensor in tensors):
            raise ValueError("refresh model weights and optimizer moments must remain float32")
        if any(not parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("all world-model parameters must permit refresh gradients")
        return next(iter(devices))

    def _optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.fit_settings.learning_rate,
                                 weight_decay=self.fit_settings.weight_decay, foreach=False)

    def _optimizer_candidate(self, state, updates):
        if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
            raise ValueError("invalid refresh optimizer state")
        candidate = self._optimizer()
        expected_groups = candidate.state_dict()["param_groups"]
        groups = state["param_groups"]
        if not isinstance(groups, list) or len(groups) != len(expected_groups):
            raise ValueError("refresh optimizer parameter groups differ")
        parameters = list(self.model.parameters())
        identifiers = []
        for actual, expected in zip(groups, expected_groups):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                raise ValueError("refresh optimizer configuration differs")
            if any(canonical_hash(actual[name]) != canonical_hash(value)
                   for name, value in expected.items() if name != "params"):
                raise ValueError("refresh optimizer hyperparameters differ")
            if not isinstance(actual["params"], list) or len(actual["params"]) != len(expected["params"]):
                raise ValueError("refresh optimizer parameter count differs")
            identifiers.extend(actual["params"])
        if (any(type(key) is not int or key < 0 for key in identifiers)
                or len(set(identifiers)) != len(identifiers) or not isinstance(state["state"], dict)
                or not set(state["state"]).issubset(identifiers)):
            raise ValueError("refresh optimizer parameter identities differ")
        for identifier, parameter in zip(identifiers, parameters):
            entry = state["state"].get(identifier)
            if entry is None:
                continue
            if not isinstance(entry, dict) or set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid refresh AdamW moments")
            for name in ("exp_avg", "exp_avg_sq"):
                value = entry[name]
                if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                        or value.shape != parameter.shape or not bool(torch.isfinite(value).all())):
                    raise ValueError("refresh AdamW moment shape, dtype or values differ")
            if bool((entry["exp_avg_sq"] < 0).any()):
                raise ValueError("refresh AdamW second moment must be nonnegative")
            step = entry["step"]
            if (not isinstance(step, torch.Tensor) or step.dtype != torch.float32 or step.shape != ()
                    or not bool(torch.isfinite(step)) or not 0 <= float(step) <= updates
                    or float(step) != int(float(step))):
                raise ValueError("refresh AdamW step is invalid")
        if updates and not state["state"]:
            raise ValueError("nonzero model updates require preserved AdamW moments")
        candidate.load_state_dict(_copy(state))
        return candidate

    def _load_initial(self, parent, requested_updates):
        if not isinstance(parent, dict) or not {"model", "optimizer", "metadata", "update"}.issubset(parent):
            raise ValueError("invalid initial P2.4 checkpoint payload")
        updates = parent["update"]
        if type(updates) is not int or updates < 0 or requested_updates not in (0, updates):
            raise ValueError("initial model update count differs from checkpoint")
        metadata = parent["metadata"]
        common = deepcopy(self._initial_parent_settings)
        common.pop("variant")
        expected = {"task_family": self.task_family, "run_seed": self.master_seed,
                    "common_config_sha256": canonical_hash(common),
                    "source_sha256": p24_source_identity()}
        if any(metadata.get(name) != value for name, value in expected.items()):
            raise ValueError("initial P2.4 checkpoint task, seed, configuration or source differs")
        if metadata.get("stage") != "initial" or metadata.get("variant") != "shared_initial:B,D":
            raise ValueError("online branch starts from the shared initial P2.4 checkpoint")
        if metadata.get("initial_normalization", {}).get("identity_sha256") != self._normalization_identity:
            raise ValueError("initial checkpoint normalization differs from frozen initial statistics")
        current = self.model.state_dict()
        if (set(current) != set(parent["model"]) or any(
                current[name].dtype != value.dtype or current[name].shape != value.shape
                or not torch.equal(current[name].detach().cpu(), value.detach().cpu())
                for name, value in parent["model"].items())):
            raise ValueError("load the exact P2.4 checkpoint model weights before attaching its optimizer")
        self.optimizer = self._optimizer_candidate(parent["optimizer"], updates)
        self._initial_replays = deepcopy(metadata.get("replays", []))
        return updates

    def _new_sampler(self, datasets):
        if not datasets or any(not isinstance(item, ReplayDataset) for item in datasets):
            raise ValueError("refresh requires sealed real ReplayDataset stores")
        for index, dataset in enumerate(datasets):
            source = dataset.manifest["metadata"]
            if (source.get("task_family"), source.get("run_seed")) != (self.task_family, self.master_seed):
                raise ValueError("refresh replay belongs to another task or training seed")
            if index and source.get("training_variant") != self.variant:
                raise ValueError("online refresh replay must record the matching B/D training_variant")
            # A large transferred store can be opened with eager verification
            # disabled, but fitting must still enforce terminal/clock integrity.
            if not dataset._verify:
                for episode in dataset.episodes:
                    if episode["partition"] == "fit":
                        dataset._verify_episode(episode)
        self.normalization.validate_for_dataset(datasets[0])
        initial = getattr(self, "_initial_replays", [])
        cohort = self._replay_identity(datasets)
        if initial and cohort[:len(initial)] != initial:
            raise ValueError("refresh cohort must extend the initial P2.4 replay cohort")
        sampler = ReplaySampler(datasets, seed=self.seed, burn_in=self.fit_settings.burn_in,
            loss_steps=self.fit_settings.loss_steps, partition="fit",
            recent_fraction=self.settings["world_model"]["recent_fraction"],
            recent_transitions=self.settings["world_model"]["recent_transitions"])
        return sampler, cohort

    @staticmethod
    def _replay_identity(datasets):
        return [{"manifest_sha256": item.manifest_sha256,
                 "dataset_identity_sha256": item.manifest["dataset_identity_sha256"],
                 "transitions": item.manifest["transition_count"]} for item in datasets]

    def set_replay(self, datasets):
        """Extend the ordered branch cohort without resetting its private RNG."""
        self._check_frozen()
        datasets = tuple(datasets)
        sampler, cohort = self._new_sampler(datasets)
        if self._cohort and cohort[:len(self._cohort)] != self._cohort:
            raise ValueError("refresh replay must preserve and extend its ordered cohort")
        if self.sampler is not None:
            state = self.sampler.state_dict()
            state["cohort_identity"] = sampler.cohort_identity
            sampler.load_state_dict(state)
        self.sampler, self.datasets, self._cohort = sampler, datasets, cohort

    def _check_frozen(self):
        if (canonical_hash(self.settings) != self._settings_identity
                or asdict(self.fit_settings) != self._fit_settings_identity
                or canonical_hash(self._initial_parent_settings) != self._initial_parent_identity
                or canonical_hash(self._study_amendment) != self._study_identity):
            raise ValueError("resolved refresh settings were mutated")
        if self.normalization.as_dict()["identity_sha256"] != self._normalization_identity:
            raise ValueError("initial fitting normalization was mutated")
        if self._model_device() != self.device:
            raise ValueError("refresh model device changed")

    @torch.inference_mode(False)
    def update(self):
        """Fit exactly one effective real-window batch; no simulator or policy update."""
        self._check_frozen()
        if self.sampler is None:
            raise ValueError("set_replay must attach the guarded real fitting cohort first")
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            with torch.enable_grad():
                windows = self.sampler.sample(self.fit_settings.effective_batch)
                metrics = optimizer_update(self.model, self.optimizer, windows, self.streams,
                                           self.objective, self.fit_settings)
            self.total_model_updates += 1
            return {**metrics, "model_updates": self.total_model_updates,
                    "online_model_updates": self.total_model_updates-self.initial_model_updates,
                    "real_joint_transitions_collected": 0,
                    "normalization_identity": self._normalization_identity,
                    "replay_cohort_identity": self.sampler.cohort_identity}
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            for module, mode in modes:
                module.training = mode

    def state_dict(self):
        self._check_frozen()
        return _copy({"schema_version": SCHEMA_VERSION, "identity": self._identity(),
            "initial_model_updates": self.initial_model_updates,
            "total_model_updates": self.total_model_updates, "model_version": module_version(self.model),
            "replays": self._cohort, "optimizer": self.optimizer.state_dict(),
            "streams": self.streams.state_dict(),
            "sampler": self.sampler.state_dict() if self.sampler is not None else None})

    def _identity(self):
        amendment = ({"study_amendment": deepcopy(self._study_amendment)}
                     if self._study_amendment is not None else {})
        return {"settings_sha256": canonical_hash(self.settings), "fit_settings": asdict(self.fit_settings),
                "task_family": self.task_family, "master_seed": self.master_seed,
                "variant": self.variant, "seed": self.seed, "device": str(self.device),
                "normalization": self.normalization.as_dict(), "source_sha256": self._source_identity, **amendment}

    def load_state_dict(self, state):
        """Validate all components before replacing any live optimizer or RNG state."""
        self._check_frozen()
        state = _copy(state)
        keys = {"schema_version", "identity", "initial_model_updates", "total_model_updates",
                "model_version", "replays", "optimizer", "streams", "sampler"}
        pristine = (self.initial_model_updates == self.total_model_updates == 0
                    and not self.optimizer.state)
        if (not isinstance(state, dict) or set(state) != keys or state["schema_version"] != SCHEMA_VERSION
                or state["identity"] != self._identity()
                or type(state["initial_model_updates"]) is not int or state["initial_model_updates"] < 0
                or (state["initial_model_updates"] != self.initial_model_updates and not pristine)
                or type(state["total_model_updates"]) is not int
                or state["total_model_updates"] < state["initial_model_updates"]
                or state["model_version"] != module_version(self.model)
                or state["replays"] != self._cohort):
            raise ValueError("model refresh checkpoint identity, model weights or replay cohort differs")
        optimizer = self._optimizer_candidate(state["optimizer"], state["total_model_updates"])
        streams = SamplingStreams.from_seed(self.seed, self.device)
        try:
            streams.load_state_dict(state["streams"])
        except (KeyError, TypeError, RuntimeError, ValueError) as error:
            raise ValueError("invalid model refresh sampling streams") from error
        if self.sampler is None:
            if state["sampler"] is not None:
                raise ValueError("attach the saved replay cohort before restoring refresh sampling")
            sampler = None
        else:
            sampler, _ = self._new_sampler(self.datasets)
            sampler.load_state_dict(state["sampler"])
        self.optimizer, self.streams, self.sampler = optimizer, streams, sampler
        self.initial_model_updates = state["initial_model_updates"]
        self.total_model_updates = state["total_model_updates"]
