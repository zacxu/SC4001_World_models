"""P2.3 continuous-label statistics fitted solely on real fitting episodes.

These statistics standardize the replay's geometry-normalized continuous
targets. They are labels-only metadata: model inputs never include them. One
saved object is shared by B and D for a task-family/training-seed dataset pair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor

from world_models.data.replay import ReplayDataset, canonical_hash
from world_models.models.heads import TaskSpec, auxiliary_fields, encode_transition_targets
from world_models.data.schema import REASONS


SCHEMA_VERSION = "fitting-normalization-v1"
CONSTANT_THRESHOLD = 1e-6
INPUT_SPACE = "replay_geometry_normalized_continuous_targets"


def continuous_field_names(task_family: str) -> tuple[str, ...]:
    return tuple(field.name for field in auxiliary_fields(task_family) if field.kind == "continuous")


@dataclass(frozen=True)
class FieldStatistics:
    count: int
    mean: float | None
    variance: float | None
    std: float | None
    scale: float
    minimum: float | None
    maximum: float | None

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 0:
            raise ValueError("normalization count must be a nonnegative integer")
        if isinstance(self.scale, bool) or not isinstance(self.scale, (int, float)) or not math.isfinite(self.scale):
            raise ValueError("normalization scale must be a finite number")
        moments = (self.mean, self.variance, self.std, self.minimum, self.maximum)
        if self.count == 0:
            if any(value is not None for value in moments) or self.scale != 1.0:
                raise ValueError("an unobserved field must have null moments and unit scale")
            return
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (*moments, self.scale)):
            raise ValueError("normalization moments must be finite numbers")
        assert self.variance is not None and self.std is not None
        assert self.minimum is not None and self.maximum is not None and self.mean is not None
        if self.variance < 0 or self.std < 0 or self.minimum > self.maximum:
            raise ValueError("normalization moments are out of range")
        if not math.isclose(self.std**2, self.variance, rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError("standard deviation does not match the population variance")
        if not self.minimum-1e-10 <= self.mean <= self.maximum+1e-10:
            raise ValueError("normalization mean is outside the observed range")
        expected = self.std if self.std >= CONSTANT_THRESHOLD else 1.0
        if self.scale <= 0 or not math.isclose(self.scale, expected, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("normalization scale does not match the constant-field rule")

    @property
    def available(self) -> bool:
        return self.count > 0

    @property
    def constant(self) -> bool:
        return self.available and self.std < CONSTANT_THRESHOLD


@dataclass(frozen=True)
class NormalizationStats:
    task_family: str
    fields: dict[str, FieldStatistics]
    provenance: dict

    def __post_init__(self) -> None:
        if not isinstance(self.fields, dict) or not isinstance(self.provenance, dict):
            raise ValueError("statistics fields and provenance must be mappings")
        if set(self.fields) != set(continuous_field_names(self.task_family)):
            raise ValueError("statistics must declare every continuous field for the task")
        if any(not isinstance(value, FieldStatistics) for value in self.fields.values()):
            raise ValueError("statistics fields must contain FieldStatistics objects")
        # Reject non-JSON values and NaNs in provenance before it becomes a receipt.
        json.dumps(self.provenance, allow_nan=False, sort_keys=True)

    def normalize(self, name: str, values: Tensor) -> Tensor:
        """Standardize selected valid labels, keeping tensor gradients if present.

        Mask unavailable labels before calling this method. A count-zero field
        cannot normalize nonempty supervision, rather than inventing statistics.
        Empty selection is allowed for uniformly masked fields.
        """
        if name not in self.fields:
            raise ValueError(f"unknown continuous normalization field: {name}")
        if not isinstance(values, Tensor) or not values.is_floating_point():
            raise ValueError("continuous labels must be floating-point tensors")
        stats = self.fields[name]
        if values.numel() == 0:
            return values.float()
        if not stats.available:
            raise ValueError(f"no fitting-data statistics are available for {name}")
        if not torch.isfinite(values).all():
            raise ValueError(f"nonfinite selected normalization target: {name}")
        return (values.float() - stats.mean) / stats.scale

    def denormalize(self, name: str, values: Tensor) -> Tensor:
        """Map predictions back to the replay's geometry-normalized target units."""
        if name not in self.fields or not self.fields[name].available:
            raise ValueError(f"no fitting-data statistics are available for {name}")
        if not isinstance(values, Tensor) or not values.is_floating_point():
            raise ValueError("normalized predictions must be floating-point tensors")
        stats = self.fields[name]
        return values.float() * stats.scale + stats.mean

    def as_dict(self) -> dict:
        payload = {"schema_version": SCHEMA_VERSION, "task_family": self.task_family,
                   "input_space": INPUT_SPACE, "variance_reduction": "population",
                   "constant_threshold": CONSTANT_THRESHOLD,
                   "constant_scale_rule": "std if std >= threshold, otherwise 1.0; unobserved fields remain unavailable",
                   "fields": {name: asdict(self.fields[name]) for name in sorted(self.fields)},
                   "provenance": deepcopy(self.provenance)}
        return {**payload, "identity_sha256": canonical_hash(payload)}

    @classmethod
    def from_dict(cls, payload: dict) -> "NormalizationStats":
        if not isinstance(payload, dict):
            raise ValueError("normalization metadata must be a JSON object")
        expected = {"schema_version", "task_family", "input_space", "variance_reduction", "constant_threshold",
                    "constant_scale_rule", "fields", "provenance", "identity_sha256"}
        if set(payload) != expected:
            raise ValueError("normalization metadata has missing or unexpected fields")
        if (payload["schema_version"] != SCHEMA_VERSION or payload["input_space"] != INPUT_SPACE
                or payload["variance_reduction"] != "population" or payload["constant_threshold"] != CONSTANT_THRESHOLD):
            raise ValueError("normalization metadata uses unsupported semantics")
        without_identity = {name: value for name, value in payload.items() if name != "identity_sha256"}
        if canonical_hash(without_identity) != payload["identity_sha256"]:
            raise ValueError("normalization identity hash mismatch")
        try:
            fields = {name: FieldStatistics(**value) for name, value in payload["fields"].items()}
            result = cls(payload["task_family"], fields, payload["provenance"])
        except (TypeError, AttributeError) as error:
            raise ValueError("invalid normalization field metadata") from error
        if result.as_dict() != payload:
            raise ValueError("normalization metadata differs from the canonical schema")
        return result

    def save(self, path: str | Path) -> dict:
        """Write a fresh immutable JSON and exact-byte SHA receipt; never overwrite."""
        path = Path(path)
        receipt = path.with_suffix(".sha256")
        if path.exists() or receipt.exists():
            raise FileExistsError("normalization metadata is immutable; use a new path")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (json.dumps(self.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        with path.open("xb") as stream:
            stream.write(content)
        digest = hashlib.sha256(content).hexdigest()
        with receipt.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(f"{digest}  {path.name}\n")
        return {"path": str(path), "sha256": digest, "identity_sha256": self.as_dict()["identity_sha256"]}

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        path = Path(path)
        receipt = path.with_suffix(".sha256")
        if not path.is_file() or not receipt.is_file():
            raise ValueError("normalization JSON and its SHA receipt are both required")
        content = path.read_bytes()
        parts = receipt.read_text(encoding="ascii").strip().split(maxsplit=1)
        if len(parts) != 2 or parts[0] != hashlib.sha256(content).hexdigest() or parts[1] != path.name:
            raise ValueError("normalization JSON changed or its receipt is invalid")
        return cls.from_dict(json.loads(content))

    def validate_for_dataset(self, dataset: ReplayDataset) -> None:
        """Fail if statistics from another task, seed/store, or partition are used."""
        if not isinstance(dataset, ReplayDataset):
            raise ValueError("normalization provenance requires a sealed ReplayDataset")
        metadata = dataset.manifest["metadata"]
        expected = {
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "dataset_identity_sha256": dataset.manifest["dataset_identity_sha256"],
            "task_family": metadata["task_family"], "run_seed": metadata["run_seed"],
            "data_source": "real_environment", "partition": "fit", "geometry_split": "train",
        }
        if self.task_family != metadata["task_family"] or any(self.provenance.get(name) != value for name, value in expected.items()):
            raise ValueError("normalization statistics do not belong to this fitting replay")
        identifiers = [episode.metadata["episode_id"] for episode in dataset.iter_episodes(purpose="world_model_fit")]
        if self.provenance.get("fitting_episode_ids") != identifiers or self.provenance.get("fitting_episode_ids_sha256") != canonical_hash(identifiers):
            raise ValueError("normalization fitting-episode membership differs from replay")


class _Moments:
    """Parallel-Welford merge in float64, retaining bounded memory per field."""

    def __init__(self) -> None:
        self.count, self.mean, self.m2 = 0, 0.0, 0.0
        self.minimum, self.maximum = math.inf, -math.inf

    def update(self, values: np.ndarray, valid: np.ndarray) -> None:
        values, valid = np.asarray(values), np.asarray(valid)
        if values.shape != valid.shape or valid.dtype != np.bool_:
            raise ValueError("normalization values require a matching boolean validity mask")
        selected = np.asarray(values[valid], dtype=np.float64)
        if not np.isfinite(selected).all():
            raise ValueError("nonfinite valid normalization sample")
        count = selected.size
        if not count:
            return
        mean = float(selected.mean(dtype=np.float64))
        centered = selected-mean
        m2 = float(np.sum(centered*centered, dtype=np.float64))
        total = self.count+count
        delta = mean-self.mean
        self.m2 += m2 + delta*delta*self.count*count/total
        self.mean += delta*count/total
        self.count = total
        self.minimum = min(self.minimum, float(selected.min()))
        self.maximum = max(self.maximum, float(selected.max()))

    def finish(self) -> FieldStatistics:
        if not self.count:
            return FieldStatistics(0, None, None, None, 1.0, None, None)
        variance = max(0.0, self.m2/self.count)
        std = math.sqrt(variance)
        return FieldStatistics(self.count, self.mean, variance, std,
                               std if std >= CONSTANT_THRESHOLD else 1.0, self.minimum, self.maximum)


def fit_field_statistics(samples: Iterable[Mapping[str, tuple[np.ndarray, np.ndarray]]], task_family: str,
                         *, provenance: dict | None = None) -> NormalizationStats:
    """Pure bounded-memory reducer for controlled fixtures and production chunks.

    Each sample maps a declared continuous field to (values, validity). Omitted
    fields remain unavailable. Production callers should use fit_normalization,
    which enforces source/partition membership and records replay provenance.
    """
    accumulators = {name: _Moments() for name in continuous_field_names(task_family)}
    for sample in samples:
        if set(sample)-set(accumulators):
            raise ValueError("normalization sample contains undeclared/noncontinuous fields")
        for name, (values, valid) in sample.items():
            accumulators[name].update(values, valid)
    return NormalizationStats(task_family, {name: value.finish() for name, value in accumulators.items()},
                              {"source": "explicit_field_fixture"} if provenance is None else dict(provenance))


def fit_normalization(dataset: ReplayDataset, *, task: TaskSpec | None = None,
                      progress: Callable[[dict], None] | None = None) -> NormalizationStats:
    """Fit population statistics from unique real fitting states, including reset.

    Episode chunks partition the stored state sequence, so no observation is
    duplicated as both source and destination. Field masks are independent of
    RGB validity; a missing terminal image does not erase a valid state label.
    """
    if not isinstance(dataset, ReplayDataset):
        raise ValueError("fit_normalization requires a sealed ReplayDataset")
    metadata = dataset.manifest["metadata"]
    if metadata.get("data_source") != "real_environment" or metadata.get("learning_eligible") is not True:
        raise ValueError("normalization requires real learning-eligible replay, never preflight or synthetic data")
    family = metadata["task_family"]
    task = TaskSpec.for_family(family) if task is None else task
    if task.task_family != family:
        raise ValueError("normalization label-profile task must match the replay family")
    declared_fields = auxiliary_fields(family)
    fields = tuple(field for field in declared_fields if field.kind == "continuous")
    binary_fields = tuple(field for field in declared_fields if field.kind == "binary")
    stage_field = next(field for field in declared_fields if field.name == "task/stage")
    profile = {
        "binary_fields": {field.name: {"alignment": field.alignment, "valid": 0, "positive": 0}
                          for field in binary_fields},
        "task_stage": {"valid": 0, "counts": [0]*stage_field.classes},
        "learned_outcome": {"valid": 0, "counts": [0, 0, 0], "class_names": ["continue", "success", "failure"]},
        "deadline_transitions": 0, "nonzero_bonus_transitions": 0,
    }
    manifest_before = dataset.manifest_sha256
    episodes, state_count, transition_count, chunk_count = [], 0, 0, 0

    def chunks():
        nonlocal state_count, transition_count, chunk_count
        for view in dataset.iter_episodes(purpose="world_model_fit"):
            episode = view.metadata
            if episode["partition"] != "fit" or episode["split"] != "train":
                raise ValueError("normalization attempted to read a non-fitting episode")
            expected_state = 0
            for chunk, arrays in view.chunks():
                if chunk["state_start"] != expected_state:
                    raise ValueError("fitting state chunks overlap or omit observations")
                rows = int(chunk["state_count"])
                expected_state += rows
                sample = {}
                for field in fields:
                    index = (slice(None), *field.index)
                    value, valid = arrays[field.array][index], arrays[field.validity_array][index]
                    if value.shape != (rows,) or valid.shape != (rows,):
                        raise ValueError(f"normalization target rows do not match the state sequence: {field.name}")
                    sample[field.name] = (value, valid)
                # Label prevalence is diagnostic only: no class weights are
                # changed. State and transition arrays keep independent axes.
                for field in binary_fields:
                    index = (slice(None), *field.index)
                    value, valid = arrays[field.array][index], arrays[field.validity_array][index]
                    selected = value[valid]
                    if not np.isfinite(selected).all():
                        raise ValueError(f"nonfinite valid binary profile target: {field.name}")
                    if field.alignment == "state" and np.any((selected != 0) & (selected != 1)):
                        raise ValueError(f"invalid binary state profile target: {field.name}")
                    if field.alignment == "transition" and (np.any(selected < 0) or np.any(selected != np.round(selected))):
                        raise ValueError(f"invalid binary event profile target: {field.name}")
                    profile["binary_fields"][field.name]["valid"] += int(selected.size)
                    profile["binary_fields"][field.name]["positive"] += int(np.count_nonzero(selected > 0))
                index = (slice(None), *stage_field.index)
                stages = arrays[stage_field.array][index][arrays[stage_field.validity_array][index]]
                if (not np.isfinite(stages).all() or np.any(stages != np.round(stages))
                        or np.any((stages < 0) | (stages >= stage_field.classes))):
                    raise ValueError("invalid categorical task-stage profile target")
                counts = np.bincount(stages.astype(np.int64), minlength=stage_field.classes)
                profile["task_stage"]["valid"] += int(stages.size)
                profile["task_stage"]["counts"] = [old+int(new) for old, new in zip(profile["task_stage"]["counts"], counts)]
                transition_targets = encode_transition_targets(arrays, task)
                outcomes = transition_targets.outcome[transition_targets.outcome_valid]
                counts = torch.bincount(outcomes, minlength=3).tolist()
                profile["learned_outcome"]["valid"] += int(outcomes.numel())
                profile["learned_outcome"]["counts"] = [old+int(new) for old, new in zip(profile["learned_outcome"]["counts"], counts)]
                profile["deadline_transitions"] += int(np.count_nonzero(arrays["termination_reason"] == REASONS.index("deadline")))
                profile["nonzero_bonus_transitions"] += int(torch.count_nonzero(transition_targets.bonus[transition_targets.bonus_valid] > 1e-6))
                state_count += rows
                chunk_count += 1
                yield sample
            if expected_state != episode["valid_length"]+1:
                raise ValueError("normalization episode did not contain exactly L+1 unique states")
            episodes.append(episode["episode_id"])
            transition_count += episode["valid_length"]
            if progress is not None:
                progress({"fitting_episodes": len(episodes), "unique_states": state_count,
                          "fitting_transitions": transition_count, "chunks": chunk_count})

    provisional = fit_field_statistics(chunks(), family)
    if not episodes:
        raise ValueError("normalization requires at least one fitting episode")
    if dataset.manifest_sha256 != manifest_before:
        raise ValueError("replay manifest changed while fitting normalization")
    all_episodes = dataset.episodes
    for item in profile["binary_fields"].values():
        item["positive_fraction"] = item["positive"]/item["valid"] if item["valid"] else None
    provenance = {
        "data_source": "real_environment", "partition": "fit", "geometry_split": "train",
        "task_family": family, "run_seed": metadata["run_seed"],
        "dataset_manifest_sha256": manifest_before,
        "dataset_identity_sha256": dataset.manifest["dataset_identity_sha256"],
        "fitting_episode_ids": episodes, "fitting_episode_ids_sha256": canonical_hash(episodes),
        "fitting_episode_count": len(episodes), "fitting_transition_count": transition_count,
        "unique_state_count": state_count, "chunk_count": chunk_count,
        "excluded_model_diagnostic_episode_count": sum(episode["partition"] == "diagnostic" for episode in all_episodes),
        "shared_by_variants": ["B", "D"], "accumulation": "chunked_float64_welford_population",
        "image_masks_do_not_gate_state_targets": True,
        "label_profile": profile, "profile_task_spec": asdict(task),
    }
    result = NormalizationStats(family, provisional.fields, provenance)
    result.validate_for_dataset(dataset)
    return result
