"""Pure real-interaction boundaries and fixed local-learning budgets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace

from world_models.infrastructure.config import validate_config
from world_models.infrastructure.reproducibility import canonical_hash


@dataclass(frozen=True)
class Schedule:
    """Block zero is the common initialization; later blocks are online data."""

    initial_transitions: int
    real_budget: int
    collection_block: int
    initial_model_updates: int
    online_model_updates: int
    initial_local_cycles: int
    online_local_cycles: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("schedule budgets must be positive integers")
        if (self.real_budget <= self.initial_transitions or
                (self.real_budget - self.initial_transitions) % self.collection_block):
            raise ValueError("real budget must contain whole online collection blocks")

    @classmethod
    def from_experiment(cls, settings: dict) -> "Schedule":
        validate_config(settings)
        run, model, ppo = settings["experiment"], settings["world_model"], settings["mappo"]
        return cls(run["initial_transitions"], run["real_budget"], run["collection_block"],
                   model["initial_updates"], model["online_updates"],
                   ppo["initial_local_cycles"], ppo["online_local_cycles"])

    @classmethod
    def from_initial_selection(cls, settings, master_seed, selection_path, *, checkpoint=None,
                               study_amendment=None):
        """Use the frozen checkpoint's real count without rewriting the registration.

        The selection reader verifies its receipt and original P2.4 provenance.
        An explicit study amendment may change downstream budgets. Its parent
        still matches the full original P2.4 configuration exactly.
        """
        # Actor-only evaluation imports cycle.checkpoints, which first imports
        # this package's Schedule. Keep fitting/model imports out of that path.
        from world_models.training.selection import resolve_selected_case

        registered = cls.from_experiment(settings)
        parent = settings
        if study_amendment is not None:
            from .study import parent_settings_for_study
            parent = parent_settings_for_study(settings, study_amendment, master_seed=master_seed)
        if settings["experiment"]["name"] != "pilot":
            raise ValueError("a shortened initial selection is for the exploratory pilot only")
        row = resolve_selected_case(selection_path, settings["task"]["name"], master_seed,
                                    checkpoint=checkpoint)
        common = deepcopy(parent)
        common.pop("variant")
        updates = row["model_updates"]
        if (row["original_target_stage_updates"] != registered.initial_model_updates
                or row["common_config_sha256"] != canonical_hash(common)
                or type(updates) is not int or not 0 < updates <= registered.initial_model_updates):
            raise ValueError("selected initial checkpoint differs from the registered configuration or budget")
        binding = {name: row[name] for name in (
            "selection_path", "selection_sha256", "task", "seed", "checkpoint_path", "checkpoint_sha256",
            "model_updates", "original_target_stage_updates")}
        return replace(registered, initial_model_updates=updates), binding

    @classmethod
    def from_manifest(cls, manifest):
        """Revalidate the frozen selection; never restore a shortened run as 5,000."""
        settings = manifest["settings"]
        registered = cls.from_experiment(settings)
        amendment = manifest.get("study_amendment")
        if amendment is not None:
            from .study import parent_settings_for_study
            parent = parent_settings_for_study(settings, amendment, master_seed=manifest["master_seed"])
            if manifest.get("parent_schedule") != cls.from_experiment(parent).describe():
                raise ValueError("study amendment parent schedule differs")
        binding = manifest.get("initial_selection")
        if binding is None:
            if amendment is not None:
                raise ValueError("study amendment requires a frozen initial selection")
            schedule = registered
            if manifest.get("initial_training_policy", "completed_registered_schedule") != "completed_registered_schedule":
                raise ValueError("shortened training requires its frozen initial selection")
        else:
            if not isinstance(binding, dict):
                raise ValueError("invalid initial selection binding")
            options = {"study_amendment": amendment} if amendment is not None else {}
            schedule, verified = cls.from_initial_selection(settings, manifest["master_seed"],
                binding["selection_path"], checkpoint=manifest["initial_checkpoint"], **options)
            if (binding != verified
                    or manifest.get("initial_checkpoint_sha256") != verified["checkpoint_sha256"]
                    or manifest.get("registered_schedule") != registered.describe()
                    or manifest.get("initial_training_policy") != "frozen_last_saved_checkpoints"
                    or manifest.get("study_scope") != "exploratory_pilot"):
                raise ValueError("run's frozen initial selection or original schedule differs")
        if manifest.get("schedule") != schedule.describe():
            raise ValueError("run schedule differs from its verified initial checkpoint")
        return schedule

    @property
    def blocks(self) -> int:
        """Number of online collection blocks, excluding block zero."""
        return (self.real_budget - self.initial_transitions) // self.collection_block

    @property
    def total_local_cycles(self) -> int:
        return self.initial_local_cycles + self.blocks * self.online_local_cycles

    @property
    def total_model_updates(self) -> int:
        return self.initial_model_updates + self.blocks * self.online_model_updates

    def _index(self, block_index: int) -> None:
        if type(block_index) is not int or not 0 <= block_index <= self.blocks:
            raise ValueError(f"block index must be an integer from 0 through {self.blocks}")

    def block_target(self, block_index: int) -> int:
        """Absolute real count; recovery always collects toward this boundary."""
        self._index(block_index)
        return self.initial_transitions + block_index * self.collection_block

    def local_cycles(self, block_index: int) -> int:
        self._index(block_index)
        return self.initial_local_cycles if block_index == 0 else self.online_local_cycles

    def model_updates(self, block_index: int) -> int:
        self._index(block_index)
        return self.initial_model_updates if block_index == 0 else self.online_model_updates

    def completed_local_cycles(self, block_index: int) -> int:
        self._index(block_index)
        return self.initial_local_cycles + block_index * self.online_local_cycles

    def describe(self) -> dict:
        return {**asdict(self), "online_blocks": self.blocks,
                "total_local_cycles": self.total_local_cycles,
                "total_model_updates": self.total_model_updates,
                "block_targets": [self.block_target(index) for index in range(self.blocks + 1)]}
