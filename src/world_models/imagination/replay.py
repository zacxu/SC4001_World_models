"""Guarded fitting-replay anchors and observation/action-only real prefixes.

Sampling is a fixed-quota mixture of uniform eligible source states from all
current replay and its recent physical-transition suffix.  Rewards, outcomes,
events and privileged simulator targets never determine sampling weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
import re

import numpy as np

from world_models.contracts import ACTION_COUNT, LOCAL_RGB_SHAPE
from world_models.data.replay import ReplayDataset, canonical_hash


_VERSION = "imagination-anchor-sampler-v1"


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identity(value, name: str) -> None:
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 identity")


@dataclass(frozen=True)
class AnchorRef:
    """A nonterminal source observation in an exact, ordered replay cohort."""

    cohort_identity: str
    dataset_index: int
    dataset_identity: str
    episode_id: str
    decision_index: int

    def __post_init__(self) -> None:
        _identity(self.cohort_identity, "cohort_identity")
        _identity(self.dataset_identity, "dataset_identity")
        _integer(self.dataset_index, "dataset_index")
        _integer(self.decision_index, "decision_index")
        if type(self.episode_id) is not str or not self.episode_id:
            raise ValueError("episode_id must be a nonempty string")

    @property
    def prefix_length(self) -> int:
        """Number of real observations, including reset and the anchor image."""
        return self.decision_index + 1


@dataclass(frozen=True)
class AnchorPrefix:
    """Owned, read-only history: o_0..o_t and a_0..a_(t-1), with no suffix.

    Columns are the two agents.  An actor must receive only its own column;
    the shared world posterior uses both synchronized observations/actions.
    """

    ref: AnchorRef
    rgb: np.ndarray
    actions: np.ndarray
    decision_indices: np.ndarray

    def __post_init__(self) -> None:
        if type(self.ref) is not AnchorRef:
            raise ValueError("prefix requires an AnchorRef")
        count = self.ref.prefix_length
        specifications = (("rgb", np.dtype("uint8"), (count, 2, *LOCAL_RGB_SHAPE)),
                          ("actions", np.dtype("int64"), (count - 1, 2)),
                          ("decision_indices", np.dtype("int64"), (count,)))
        for name, dtype, shape in specifications:
            value = getattr(self, name)
            if type(value) is not np.ndarray or value.dtype != dtype or value.shape != shape:
                raise ValueError(f"prefix {name} requires dtype {dtype} and shape {shape}")
        if np.any(self.actions < 0) or np.any(self.actions >= ACTION_COUNT):
            raise ValueError("executed prefix actions must be in [0,12)")
        if not np.array_equal(self.decision_indices, np.arange(count, dtype=np.int64)):
            raise ValueError("prefix decision clocks must be contiguous from reset through the anchor")
        for name, _, _ in specifications:
            owned = np.array(getattr(self, name), copy=True, order="C")
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)


class AnchorSampler:
    """Versioned real-replay source sampling, independent of training RNGs.

    Each batch contains round-half-up(batch_size * recent_fraction) draws from
    the recent pool and the remaining draws from all fitting replay. Both
    populations sample with replacement; a final shuffle hides mixture order.
    The pools overlap, so a uniform draw may also belong to recent replay.

    Final destination observations, including collection cutoffs, are excluded
    consistently with the replay's source-anchor rule. Earlier nonterminal
    sources remain eligible even if their outgoing real transition terminates.
    """

    def __init__(self, datasets: Sequence[ReplayDataset], *, seed: int, deadline: int,
                 batch_size: int = 256, recent_fraction: float = .5,
                 recent_transitions: int = 100000):
        self.datasets = tuple(datasets)
        if not self.datasets or any(not isinstance(item, ReplayDataset) for item in self.datasets):
            raise ValueError("anchor sampling requires at least one sealed ReplayDataset")
        _integer(seed, "seed")
        self.deadline = _integer(deadline, "deadline", 1)
        self.batch_size = _integer(batch_size, "batch_size", 1)
        self.recent_transitions = _integer(recent_transitions, "recent_transitions", 1)
        if (isinstance(recent_fraction, bool) or not isinstance(recent_fraction, (int, float))
                or not math.isfinite(recent_fraction) or not 0 <= recent_fraction <= 1):
            raise ValueError("recent_fraction must be finite and between zero and one")
        self.recent_fraction = float(recent_fraction)
        schema = self.datasets[0].manifest["schema"]
        metadata = self.datasets[0].manifest["metadata"]
        self.task_family, self.run_seed = metadata["task_family"], metadata["run_seed"]
        self._episodes: dict[tuple[int, str], dict] = {}
        self._valid_indices: dict[tuple[int, str], frozenset[int]] = {}
        dataset_ids, episode_ids = set(), set()
        all_coordinates, recent_coordinates = [], []
        total = sum(dataset.manifest["transition_count"] for dataset in self.datasets)
        cutoff, offset = max(0, total - self.recent_transitions), 0
        for dataset_index, dataset in enumerate(self.datasets):
            manifest, source = dataset.manifest, dataset.manifest["metadata"]
            if (source.get("data_source") != "real_environment" or source.get("learning_eligible") is not True
                    or source.get("budget_scope") == "collection_preflight"):
                raise ValueError("preflight, diagnostic-only, and synthetic stores are excluded from learning anchors")
            if (source.get("task_family") != self.task_family or source.get("run_seed") != self.run_seed
                    or manifest["schema"] != schema):
                raise ValueError("all replay stores must share task family, run seed, and schema")
            identity = manifest["dataset_identity_sha256"]
            if identity in dataset_ids:
                raise ValueError("duplicate dataset content in anchor replay cohort")
            dataset_ids.add(identity)
            for episode in dataset.episodes:
                identifier, provenance = episode["episode_id"], episode["split_provenance"]
                if identifier in episode_ids:
                    raise ValueError("duplicate episode IDs in anchor replay cohort")
                episode_ids.add(identifier)
                if (episode.get("data_source") != "real_environment" or episode.get("learning_eligible") is not True
                        or episode.get("task_family") != self.task_family or episode.get("run_seed") != self.run_seed
                        or episode.get("split") != "train" or provenance.get("split") != "train"
                        or provenance.get("purpose") != "training" or provenance.get("learning_eligible") is not True):
                    raise ValueError("anchors require real learning-eligible training geometry")
                if episode["partition"] != "fit":
                    continue
                dataset.load_episode(identifier, purpose="imagination")
                # A caller may have skipped eager validation when opening a large
                # store. Never let that bypass terminal/reset/clock integrity.
                if not dataset._verify:
                    dataset._verify_episode(episode)
                key = dataset_index, identifier
                valid = []
                for chunk in episode["chunks"]:
                    arrays = dataset._chunk(chunk)
                    state_start = chunk["state_start"]
                    for local in range(chunk["state_count"]):
                        index = state_start + local
                        if index >= episode["valid_length"]:
                            continue
                        if not bool(arrays["image_valid"][local].all()):
                            raise ValueError("a complete anchor prefix requires valid nonterminal images from reset")
                        if index < self.deadline:
                            valid.append(index)
                    expected = np.arange(chunk["transition_start"],
                                         chunk["transition_start"] + chunk["transition_count"], dtype=np.int64)
                    if not np.array_equal(arrays["decision_index"], expected):
                        raise ValueError("recorded decision clocks must be contiguous from reset")
                self._episodes[key] = episode
                self._valid_indices[key] = frozenset(valid)
                for index in valid:
                    row = dataset_index, identifier, index
                    all_coordinates.append(row)
                    if offset + episode["transition_id_start"] + index >= cutoff:
                        recent_coordinates.append(row)
            offset += dataset.manifest["transition_count"]
        if not all_coordinates:
            raise ValueError("no valid nonterminal source anchors before the task deadline in fitting replay")
        if self.recent_fraction and not recent_coordinates:
            raise ValueError("recent physical-transition suffix contains no eligible anchors")
        self.cohort_identity = canonical_hash({
            "version": _VERSION,
            "ordered_datasets": [{"manifest_sha256": dataset.manifest_sha256,
                                  "dataset_identity_sha256": dataset.manifest["dataset_identity_sha256"]}
                                 for dataset in self.datasets],
            "deadline": self.deadline, "batch_size": self.batch_size,
            "recent_fraction": self.recent_fraction, "recent_transitions": self.recent_transitions,
            "partition": "fit", "mixture": "exact-round-half-up-shuffled-with-replacement",
            "eligibility": "valid-complete-prefix-nonterminal-source-before-deadline",
        })
        self._anchors = tuple(self.reference(*row) for row in all_coordinates)
        self._recent = tuple(self.reference(*row) for row in recent_coordinates)
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_experiment(cls, datasets: Sequence[ReplayDataset], experiment: Mapping, *, seed: int):
        from world_models.infrastructure.config import validate_config

        validate_config(experiment)
        model = experiment["world_model"]
        result = cls(datasets, seed=seed, deadline=experiment["task"]["deadline"],
                     batch_size=experiment["mappo"]["starts"], recent_fraction=model["recent_fraction"],
                     recent_transitions=model["recent_transitions"])
        if result.task_family != experiment["task"]["name"]:
            raise ValueError("experiment task differs from anchor replay task family")
        return result

    @property
    def anchor_count(self) -> int:
        return len(self._anchors)

    @property
    def recent_anchor_count(self) -> int:
        return len(self._recent)

    def reference(self, dataset_index: int, episode_id: str, decision_index: int) -> AnchorRef:
        _integer(dataset_index, "dataset_index")
        _integer(decision_index, "decision_index")
        if type(episode_id) is not str or not episode_id:
            raise ValueError("episode_id must be a nonempty string")
        if (dataset_index, episode_id) not in self._episodes:
            raise ValueError("episode is outside this sampler's guarded fitting cohort")
        if decision_index not in self._valid_indices[dataset_index, episode_id]:
            raise ValueError("anchor must be a nonterminal source with a complete valid prefix before its deadline")
        return AnchorRef(self.cohort_identity, dataset_index,
                         self.datasets[dataset_index].manifest["dataset_identity_sha256"], episode_id, decision_index)

    def validate_reference(self, ref: AnchorRef) -> None:
        if type(ref) is not AnchorRef:
            raise ValueError("expected an AnchorRef")
        if ref != self.reference(ref.dataset_index, ref.episode_id, ref.decision_index):
            raise ValueError("anchor reference belongs to a different replay cohort or dataset identity")

    def sample(self, batch_size: int | None = None) -> tuple[AnchorRef, ...]:
        size = self.batch_size if batch_size is None else _integer(batch_size, "batch_size", 1)
        recent_count = math.floor(size * self.recent_fraction + .5)
        rows = []
        for population, count in ((self._anchors, size - recent_count), (self._recent, recent_count)):
            if count:
                rows.extend(population[int(index)] for index in self._rng.integers(len(population), size=count))
        return tuple(rows[int(index)] for index in self._rng.permutation(size))

    def read_prefix(self, ref: AnchorRef) -> AnchorPrefix:
        """Read intersecting chunks once; export no action/observation after t.

        NPZ decompression operates on whole chunks; only the prefix's RGB,
        validity and executed-action/clock slices are consumed here.
        """
        self.validate_reference(ref)
        dataset = self.datasets[ref.dataset_index]
        episode = dataset.load_episode(ref.episode_id, purpose="imagination").metadata
        count, stop = ref.prefix_length, ref.decision_index
        images, actions, clocks = [], [], []
        for chunk in episode["chunks"]:
            state_low, state_high = chunk["state_start"], min(count, chunk["state_start"] + chunk["state_count"])
            action_low, action_high = chunk["transition_start"], min(stop, chunk["transition_start"] + chunk["transition_count"])
            if state_low >= state_high and action_low >= action_high:
                continue
            arrays = dataset._chunk(chunk)
            if state_low < state_high:
                length = state_high - state_low
                if not bool(arrays["image_valid"][:length].all()):
                    raise ValueError("a complete anchor prefix requires valid images from reset")
                images.append(arrays["rgb"][:length])
            if action_low < action_high:
                length = action_high - action_low
                actions.append(arrays["actions"][:length])
                clocks.append(arrays["decision_index"][:length])
        image_values = np.concatenate(images, axis=0)
        action_values = (np.concatenate(actions, axis=0).astype(np.int64)
                         if actions else np.empty((0, 2), dtype=np.int64))
        recorded = np.concatenate(clocks).astype(np.int64) if clocks else np.empty(0, dtype=np.int64)
        decision_values = np.concatenate((recorded, np.array([ref.decision_index], dtype=np.int64)))
        return AnchorPrefix(ref, image_values, action_values, decision_values)

    def state_dict(self) -> dict:
        return {"version": _VERSION, "cohort_identity": self.cohort_identity,
                "rng_state": deepcopy(self._rng.bit_generator.state)}

    def load_state_dict(self, state: dict) -> None:
        if (type(state) is not dict or set(state) != {"version", "cohort_identity", "rng_state"}
                or state["version"] != _VERSION or state["cohort_identity"] != self.cohort_identity):
            raise ValueError("sampler checkpoint cohort or sampling configuration differs")
        generator = np.random.default_rng()
        try:
            generator.bit_generator.state = deepcopy(state["rng_state"])
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("invalid anchor sampler RNG checkpoint") from error
        self._rng = generator
