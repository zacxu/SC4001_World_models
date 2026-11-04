"""Episode-safe, read-only real replay windows for world-model fitting.

Sampling uses a dedicated NumPy generator and a compact source-anchor index.
RGB data is read through the replay's bounded chunk cache, never materialized
for an entire episode. Store order defines recency for later refresh fits.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from world_models.data.replay import ReplayDataset, canonical_hash
from world_models.data.schema import STATE_ARRAYS, TRANSITION_ARRAYS


@dataclass(frozen=True)
class ReplayWindow:
    """States ``[start, stop]`` and transitions ``[start, stop)`` of one episode."""

    dataset_index: int
    episode_id: str
    start: int
    anchor: int
    stop: int
    states: dict[str, np.ndarray]
    transitions: dict[str, np.ndarray]
    event_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("dataset_index", "start", "anchor", "stop"):
            _integer(getattr(self, name), name, 0)
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("window requires a stable episode ID")
        if not self.start <= self.anchor < self.stop:
            raise ValueError("window requires start <= anchor < stop")
        for values, names, count in ((self.states, STATE_ARRAYS, self.stop-self.start+1),
                                     (self.transitions, TRANSITION_ARRAYS, self.stop-self.start)):
            if set(values) != set(names):
                raise ValueError("window arrays must contain the complete typed replay schema")
            if any(not isinstance(value, np.ndarray) or value.ndim < 1 or len(value) != count
                   for value in values.values()):
                raise ValueError("window state/transition arrays have inconsistent leading lengths")

    @property
    def burn_in(self) -> int:
        return self.anchor - self.start

    @property
    def loss_steps(self) -> int:
        return self.stop - self.anchor


def _integer(value, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class ReplaySampler:
    """Uniform valid transition anchors with an optional recent-data mixture.

    Fit and diagnostic samplers have independent generator state. Diagnostic
    mode selects only the frozen diagnostic partition; it cannot sample fit
    episodes. A recent fraction adds uniform sampling from eligible anchors
    among the last ``recent_transitions`` physical transitions of the ordered
    stores. Diagnostic transitions consume physical positions in that suffix.
    """

    def __init__(self, datasets: Sequence[ReplayDataset], *, seed: int,
                 burn_in: int = 16, loss_steps: int = 48,
                 recent_fraction: float = 0., recent_transitions: int = 100000,
                 partition: str = "fit"):
        self.datasets = tuple(datasets)
        if not self.datasets or any(not isinstance(dataset, ReplayDataset) for dataset in self.datasets):
            raise ValueError("sampling requires at least one sealed ReplayDataset")
        _integer(seed, "seed", 0)
        self.burn_in = _integer(burn_in, "burn_in", 0)
        self.loss_steps = _integer(loss_steps, "loss_steps", 1)
        self.recent_transitions = _integer(recent_transitions, "recent_transitions", 1)
        if (isinstance(recent_fraction, bool) or not isinstance(recent_fraction, (int, float))
                or not math.isfinite(recent_fraction) or not 0 <= recent_fraction <= 1):
            raise ValueError("recent_fraction must be a finite number between 0 and 1")
        self.recent_fraction = float(recent_fraction)
        if partition not in ("fit", "diagnostic"):
            raise ValueError("partition must be fit or diagnostic")
        self.partition = partition
        self._purpose = "world_model_fit" if partition == "fit" else "diagnostic"
        self._rng = np.random.default_rng(seed)
        self._episodes: dict[tuple[int, str], dict] = {}
        identities, episode_ids = set(), set()
        schema = self.datasets[0].manifest["schema"]
        metadata = self.datasets[0].manifest["metadata"]
        self.task_family, self.run_seed = metadata["task_family"], metadata["run_seed"]
        for dataset_index, dataset in enumerate(self.datasets):
            manifest, source = dataset.manifest, dataset.manifest["metadata"]
            if (source.get("data_source") != "real_environment" or source.get("learning_eligible") is not True
                    or source.get("budget_scope") == "collection_preflight"):
                raise ValueError("preflight, diagnostic-only, and synthetic stores are excluded from learning cohorts")
            if (source.get("task_family") != self.task_family or source.get("run_seed") != self.run_seed
                    or manifest["schema"] != schema):
                raise ValueError("all replay stores must share task family, run seed, and schema")
            identity = manifest["dataset_identity_sha256"]
            if identity in identities:
                raise ValueError("duplicate dataset content in replay cohort")
            identities.add(identity)
            for episode in dataset.episodes:
                identifier, provenance = episode["episode_id"], episode["split_provenance"]
                if identifier in episode_ids:
                    raise ValueError("duplicate episode IDs in replay cohort")
                episode_ids.add(identifier)
                if (episode.get("data_source") != "real_environment" or episode.get("learning_eligible") is not True
                        or episode.get("task_family") != self.task_family or episode.get("run_seed") != self.run_seed
                        or episode.get("split") != "train" or provenance.get("split") != "train"
                        or provenance.get("purpose") != "training" or provenance.get("learning_eligible") is not True):
                    raise ValueError("replay sampling requires real learning-eligible training geometry")
                if episode["partition"] == partition:
                    # Apply the replay's own purpose guard as well as cohort guards.
                    dataset.load_episode(identifier, purpose=self._purpose)
                    self._episodes[dataset_index, identifier] = episode

        cohort = {"version": "world-model-replay-sampler-v1", "partition": partition,
                  "ordered_datasets": [{"manifest_sha256": dataset.manifest_sha256,
                                        "dataset_identity_sha256": dataset.manifest["dataset_identity_sha256"]}
                                       for dataset in self.datasets],
                  "burn_in": self.burn_in, "loss_steps": self.loss_steps,
                  "recent_fraction": self.recent_fraction, "recent_transitions": self.recent_transitions}
        self.cohort_identity = canonical_hash(cohort)
        total = sum(dataset.manifest["transition_count"] for dataset in self.datasets)
        cutoff, offset = max(0, total - self.recent_transitions), 0
        anchors, recent = [], []
        self._valid_anchors: dict[tuple[int, str], np.ndarray] = {}
        for dataset_index, dataset in enumerate(self.datasets):
            for episode in dataset.episodes:
                key = dataset_index, episode["episode_id"]
                if key not in self._episodes:
                    continue
                valid = []
                for chunk in episode["chunks"]:
                    arrays = dataset._chunk(chunk)
                    indices = chunk["state_start"] + np.flatnonzero(arrays["image_valid"].all(axis=-1))
                    valid.extend(int(index) for index in indices if index < episode["valid_length"])
                self._valid_anchors[key] = np.asarray(valid, dtype=np.int64)
                for anchor in valid:
                    record = dataset_index, episode["episode_id"], anchor
                    anchors.append(record)
                    position = offset + episode["transition_id_start"] + anchor
                    if position >= cutoff:
                        recent.append(record)
            offset += dataset.manifest["transition_count"]
        self._anchors, self._recent = tuple(anchors), tuple(recent)
        if not anchors:
            raise ValueError(f"no valid source-image anchors in {partition} replay partition")
        if self.recent_fraction and not recent:
            raise ValueError("recent physical-transition suffix contains no eligible anchors")

    @property
    def anchor_count(self) -> int:
        return len(self._anchors)

    @property
    def recent_anchor_count(self) -> int:
        return len(self._recent)

    def state_dict(self) -> dict:
        """Checkpoint only the private RNG and hash-bound ordered sampling cohort."""
        return {"version": "world-model-replay-sampler-v1", "cohort_identity": self.cohort_identity,
                "rng_state": deepcopy(self._rng.bit_generator.state)}

    def load_state_dict(self, state: dict) -> None:
        if (not isinstance(state, dict) or set(state) != {"version", "cohort_identity", "rng_state"}
                or state["version"] != "world-model-replay-sampler-v1"
                or state["cohort_identity"] != self.cohort_identity):
            raise ValueError("sampler checkpoint cohort or sampling configuration differs")
        # Validate on a fresh generator; malformed checkpoints leave this sampler unchanged.
        generator = np.random.default_rng()
        try:
            generator.bit_generator.state = deepcopy(state["rng_state"])
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("invalid sampler RNG checkpoint") from error
        self._rng = generator

    def sample(self, batch_size: int) -> list[ReplayWindow]:
        _integer(batch_size, "batch_size", 1)
        windows = []
        for _ in range(batch_size):
            use_recent = self.recent_fraction > 0 and self._rng.random() < self.recent_fraction
            population = self._recent if use_recent else self._anchors
            source = population[int(self._rng.integers(len(population)))]
            windows.append(self.window(*source))
        return windows

    def window(self, dataset_index: int, episode_id: str, anchor: int, *,
               burn_in: int | None = None, loss_steps: int | None = None,
               full_prefix: bool = False) -> ReplayWindow:
        """Read one explicit source anchor without consuming any sampling RNG."""
        _integer(dataset_index, "dataset_index", 0)
        _integer(anchor, "anchor", 0)
        burn_in = self.burn_in if burn_in is None else _integer(burn_in, "burn_in", 0)
        loss_steps = self.loss_steps if loss_steps is None else _integer(loss_steps, "loss_steps", 1)
        if type(full_prefix) is not bool:
            raise ValueError("full_prefix must be boolean")
        key = dataset_index, episode_id
        if key not in self._episodes:
            raise ValueError("episode is outside this sampler's guarded partition/cohort")
        valid = self._valid_anchors[key]
        location = np.searchsorted(valid, anchor)
        if location == len(valid) or valid[location] != anchor:
            raise ValueError("anchor must be a nonterminal source with both valid images")
        dataset = self.datasets[dataset_index]
        episode = dataset.load_episode(episode_id, purpose=self._purpose).metadata
        start, stop = (0 if full_prefix else max(0, anchor - burn_in)), min(episode["valid_length"], anchor + loss_steps)
        parts = {name: [] for name in (*STATE_ARRAYS, *TRANSITION_ARRAYS)}
        # Chunk state and transition offsets differ at boundaries. Intersect each
        # independently, decoding every intersecting chunk at most once.
        for chunk in episode["chunks"]:
            state_low, state_high = max(start, chunk["state_start"]), min(stop+1, chunk["state_start"]+chunk["state_count"])
            trans_low, trans_high = max(start, chunk["transition_start"]), min(stop, chunk["transition_start"]+chunk["transition_count"])
            if state_low >= state_high and trans_low >= trans_high:
                continue
            arrays = dataset._chunk(chunk)
            if state_low < state_high:
                for name in STATE_ARRAYS:
                    parts[name].append(arrays[name][state_low-chunk["state_start"]:state_high-chunk["state_start"]])
            if trans_low < trans_high:
                for name in TRANSITION_ARRAYS:
                    parts[name].append(arrays[name][trans_low-chunk["transition_start"]:trans_high-chunk["transition_start"]])
        values = {name: np.concatenate(pieces, axis=0) for name, pieces in parts.items()}
        for name, array in values.items():
            expected = stop-start+1 if name in STATE_ARRAYS else stop-start
            if len(array) != expected:
                raise ValueError("replay manifest has a hole or overlap in the requested window")
            array.flags.writeable = False
        return ReplayWindow(dataset_index, episode_id, start, anchor, stop,
                            {name: values[name] for name in STATE_ARRAYS},
                            {name: values[name] for name in TRANSITION_ARRAYS},
                            tuple(dataset.manifest["schema"]["event_fields"]))
