"""P2.2 supervised heads and exact task reward/clock semantics.

Privileged values are labels only: neural heads accept learned features, never
engine arrays. State labels describe o_t; event labels describe t -> t+1.
Loss reduction and fitting-data normalization are deliberately left to P2.3.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from world_models.contracts import BoundaryReason
from world_models.data.schema import ENEMY_FIELDS, PLAYER_FIELDS, REASONS
from world_models.env.targets import EVENT_FIELDS, NATIVE_COUNTERS, TASK_FIELDS


CONTINUE, SUCCESS, FAILURE = 0, 1, 2


@dataclass(frozen=True)
class TaskSpec:
    task_family: str = "relay"
    deadline: int = 400
    step_cost: float = -0.001
    success_reward: float = 1.0
    failure_reward: float = -1.0
    milestone_bonus: float = 0.1
    milestone_count: int = 1

    def __post_init__(self) -> None:
        if self.task_family not in ("relay", "cover"):
            raise ValueError("task_family must be relay or cover")
        for name in ("deadline", "milestone_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.milestone_count != (1 if self.task_family == "relay" else 2):
            raise ValueError("milestone_count must match the authored family")
        for name in ("step_cost", "success_reward", "failure_reward", "milestone_bonus"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.step_cost > 0 or self.milestone_bonus < 0 or self.success_reward <= 0 or self.failure_reward >= 0:
            raise ValueError("step cost must be nonpositive, milestone bonus nonnegative, success positive, and failure negative")

    @classmethod
    def for_family(cls, task_family: str) -> TaskSpec:
        if task_family == "relay":
            return cls()
        if task_family == "cover":
            return cls(task_family="cover", deadline=600, milestone_bonus=0.05, milestone_count=2)
        raise ValueError("task_family must be relay or cover")

    @classmethod
    def from_experiment(cls, experiment: Mapping[str, object]) -> TaskSpec:
        """Preserve the resolved, validated experiment's task reward and clock."""
        task = experiment["task"]
        if not isinstance(task, Mapping):
            raise ValueError("experiment task must be a mapping")
        return cls(task_family=task["name"], **{name: task[name] for name in (
            "deadline", "step_cost", "success_reward", "failure_reward", "milestone_bonus", "milestone_count")})

    @property
    def max_bonus(self) -> float:
        # The approved environment RewardRule uses this conservative bound.
        return self.milestone_bonus * self.milestone_count

    @property
    def stage_classes(self) -> int:
        # Success advances stage once beyond the final completed milestone.
        return self.milestone_count + 2


def _features(features: Tensor, width: int, name: str) -> None:
    if not isinstance(features, Tensor) or features.ndim != 2 or features.shape[0] == 0 or features.shape[-1] != width:
        raise ValueError(f"{name} must have shape [batch,{width}]")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError(f"{name} must contain finite floating-point values")


@dataclass(frozen=True)
class TransitionPredictions:
    bonus: Tensor
    outcome_logits: Tensor


class TransitionHeads(nn.Module):
    """One joint milestone expectation and one joint categorical outcome."""

    def __init__(self, input_width: int = 768, hidden_width: int = 256, *, max_bonus: float = 0.1):
        super().__init__()
        if type(input_width) is not int or input_width < 1 or type(hidden_width) is not int or hidden_width < 1:
            raise ValueError("head widths must be positive integers")
        if isinstance(max_bonus, bool) or not isinstance(max_bonus, (int, float)) or not math.isfinite(max_bonus) or max_bonus < 0:
            raise ValueError("max_bonus must be finite and nonnegative")
        self.input_width, self.max_bonus = input_width, float(max_bonus)
        self.trunk = nn.Sequential(nn.Linear(input_width, hidden_width), nn.SiLU(),
                                   nn.Linear(hidden_width, hidden_width), nn.SiLU())
        self.bonus_head = nn.Linear(hidden_width, 1)
        self.outcome_head = nn.Linear(hidden_width, 3)

    def forward(self, features: Tensor) -> TransitionPredictions:
        _features(features, self.input_width, "transition features")
        hidden = self.trunk(features)
        # Probability-sensitive values stay float32 under mixed precision.
        bonus = self.max_bonus * torch.sigmoid(self.bonus_head(hidden).float()).squeeze(-1)
        return TransitionPredictions(bonus, self.outcome_head(hidden).float())


@dataclass(frozen=True)
class ResolvedOutcomes:
    outcome: Tensor
    terminated: Tensor
    deadline_failure: Tensor
    terminal_reward: Tensor


def _integer_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must be an integer tensor")


def resolve_outcomes(sampled_outcome: Tensor, next_decision: Tensor, task: TaskSpec) -> ResolvedOutcomes:
    """Sampled failure, then success, then known clock expiry, then continue.

The input is one categorical sample per joint transition, so failure and success
cannot both be sampled. A success exactly at the deadline remains a success.
"""
    _integer_tensor(sampled_outcome, "sampled_outcome")
    _integer_tensor(next_decision, "next_decision")
    if sampled_outcome.shape != next_decision.shape or sampled_outcome.device != next_decision.device:
        raise ValueError("outcome and next clock must have matching shapes and devices")
    if torch.any((sampled_outcome < CONTINUE) | (sampled_outcome > FAILURE)):
        raise ValueError("outcome must be continue, success, or failure")
    if torch.any((next_decision < 1) | (next_decision > task.deadline)):
        raise ValueError("next clock must be between one and the task deadline")
    deadline_failure = (sampled_outcome == CONTINUE) & (next_decision >= task.deadline)
    outcome = torch.where(deadline_failure, FAILURE, sampled_outcome).long()
    terminal = torch.where(outcome == SUCCESS, task.success_reward,
                           torch.where(outcome == FAILURE, task.failure_reward, 0.0)).float()
    return ResolvedOutcomes(outcome, outcome != CONTINUE, deadline_failure, terminal)


def total_reward(bonus: Tensor, resolved: ResolvedOutcomes, task: TaskSpec) -> Tensor:
    if not isinstance(bonus, Tensor) or not bonus.is_floating_point() or bonus.shape != resolved.outcome.shape:
        raise ValueError("bonus must be a floating-point tensor matching the joint outcome")
    if bonus.device != resolved.outcome.device or not torch.isfinite(bonus).all():
        raise ValueError("bonus must be finite and on the outcome device")
    if torch.any((bonus < 0) | (bonus > task.max_bonus)):
        raise ValueError("bonus exceeds the declared per-transition bound")
    return bonus.float() + task.step_cost + resolved.terminal_reward


@dataclass(frozen=True)
class TransitionTargets:
    bonus: Tensor
    outcome: Tensor
    bonus_valid: Tensor
    outcome_valid: Tensor
    total_reward: Tensor


def _as_tensor(value: object, *, device: torch.device | str | None = None) -> Tensor:
    # Replay readers deliberately expose read-only NumPy arrays. Copy explicitly
    # rather than building a writeable torch view over immutable stored data.
    if isinstance(value, Tensor):
        return value.to(device=device) if device is not None else value
    return torch.tensor(value, device=device)


def encode_transition_targets(arrays: Mapping[str, object], task: TaskSpec, *,
                              device: torch.device | str | None = None) -> TransitionTargets:
    """Decode transition labels independently of terminal-image availability."""
    reward = _as_tensor(arrays["team_reward"], device=device).double()
    reason = _as_tensor(arrays["termination_reason"], device=device)
    decision = _as_tensor(arrays["decision_index"], device=device)
    _integer_tensor(reason, "termination_reason")
    _integer_tensor(decision, "decision_index")
    if reward.shape != reason.shape or reward.shape != decision.shape:
        raise ValueError("reward, reason, and clock target shapes differ")
    if not torch.isfinite(reward).all():
        raise ValueError("nonfinite transition reward target")
    if torch.any((reason < 0) | (reason >= len(REASONS))):
        raise ValueError("unknown replay termination reason")
    if torch.any((decision < 0) | (decision >= task.deadline)):
        raise ValueError("invalid source clock target")
    is_success = reason == REASONS.index(BoundaryReason.SUCCESS.value)
    is_failure = ((reason == REASONS.index(BoundaryReason.FAILURE.value)) |
                  (reason == REASONS.index(BoundaryReason.DEATH.value)))
    is_deadline = reason == REASONS.index(BoundaryReason.DEADLINE.value)
    at_deadline = decision + 1 == task.deadline
    if torch.any(is_deadline & ~at_deadline) or torch.any(at_deadline & ~(is_deadline | is_success | is_failure)):
        raise ValueError("replay boundary disagrees with the exact deadline")
    is_cutoff = reason == REASONS.index(BoundaryReason.COLLECTION_CUTOFF.value)
    for name, expected in (("terminated", is_success | is_failure | is_deadline), ("truncated", is_cutoff)):
        if name in arrays:
            actual = _as_tensor(arrays[name], device=device)
            if actual.dtype != torch.bool or actual.shape != reward.shape or torch.any(actual != expected):
                raise ValueError(f"{name} disagrees with the replay boundary")
    outcome = torch.where(is_failure, FAILURE, torch.where(is_success, SUCCESS, CONTINUE)).long()
    terminal = torch.where(is_success, task.success_reward,
                           torch.where(is_failure | is_deadline, task.failure_reward, 0.0)).double()
    bonus = reward - task.step_cost - terminal
    # Only roundoff-sized excursions are clamped. Bad reward labels are errors.
    tolerance = 1e-6
    if torch.any((bonus < -tolerance) | (bonus > task.max_bonus + tolerance)):
        raise ValueError("reward does not decompose into the declared bonus and outcome")
    valid = torch.ones_like(reason, dtype=torch.bool)
    return TransitionTargets(bonus.clamp(0.0, task.max_bonus).float(), outcome, valid, valid.clone(), reward.float())


@dataclass(frozen=True)
class AuxiliaryField:
    name: str
    kind: str
    alignment: str
    array: str
    validity_array: str
    index: tuple[int, ...]
    classes: int = 1


def _task_applicable(name: str, family: str) -> bool:
    return not ((family == "relay" and name.startswith(("progress_", "console_"))) or
                (family == "cover" and (name in ("plate_active", "latch_active") or name.startswith("gate_"))))


def _event_applicable(name: str, family: str) -> bool:
    return not ((family == "relay" and name.startswith(("console_", "enemy_"))) or
                (family == "cover" and name in ("plate_entries", "plate_exits", "gate_opens", "gate_closes", "latch_activations")))


def declared_event_fields() -> tuple[str, ...]:
    return tuple(sorted((*EVENT_FIELDS, *(f"player_{player}_{name.lower()}"
                                          for player in (0, 1) for name in NATIVE_COUNTERS))))


def auxiliary_fields(task_family: str, event_fields: Sequence[str] | None = None) -> tuple[AuxiliaryField, ...]:
    task = TaskSpec.for_family(task_family)
    events = declared_event_fields() if event_fields is None else tuple(event_fields)
    if len(events) != len(set(events)) or set(events) != set(declared_event_fields()):
        raise ValueError("event schema must contain exactly the declared registry and native events")
    fields = []
    for player in (0, 1):
        for i, name in enumerate(PLAYER_FIELDS):
            fields.append(AuxiliaryField(f"player_{player}/{name}", "binary" if name == "alive" else "continuous",
                                         "state", "normalized_player_state", "privileged_validity_mask", (player, i)))
    binary_task = {"plate_active", "latch_active", "gate_1_open", "gate_2_open", "console_1_complete",
                   "console_2_complete", "extraction_unlocked", "success", "failure"}
    for i, name in enumerate(TASK_FIELDS):
        if not _task_applicable(name, task_family):
            continue
        fields.append(AuxiliaryField(f"task/{name}", "categorical" if name == "stage" else
                                     "binary" if name in binary_task else "continuous", "state",
                                     "task_state" if name == "stage" else "normalized_task_state",
                                     "task_validity_mask", (i,), task.stage_classes if name == "stage" else 1))
    if task_family == "cover":
        for enemy in (0, 1):
            for i, name in enumerate(ENEMY_FIELDS):
                fields.append(AuxiliaryField(f"enemy_{enemy+1}/{name}", "binary" if name in ("alive", "spawned") else "continuous",
                                             "state", "normalized_enemy_state", "enemy_validity_mask", (enemy, i)))
    for i, name in enumerate(events):
        if _event_applicable(name, task_family):
            fields.append(AuxiliaryField(f"event/{name}", "binary", "transition", "event_deltas",
                                         "event_delta_validity_mask", (i,)))
    return tuple(fields)


class AuxiliaryHeads(nn.Module):
    """Observation-aligned states and transition-aligned event-presence logits."""

    def __init__(self, width: int = 256, task_family: str = "relay", transition_width: int = 768):
        super().__init__()
        if type(width) is not int or width < 1 or type(transition_width) is not int or transition_width < 1:
            raise ValueError("auxiliary head widths must be positive integers")
        self.width, self.transition_width = width, transition_width
        self.fields = auxiliary_fields(task_family)
        self.state_trunk = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.transition_trunk = nn.Sequential(nn.Linear(transition_width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.heads = nn.ModuleDict({field.name: nn.Linear(width, field.classes) for field in self.fields})

    def forward(self, state_features: Tensor, transition_features: Tensor) -> dict[str, Tensor]:
        if state_features.shape[0] != transition_features.shape[0]:
            raise ValueError("state and transition features need the same batch")
        return {**self.state_predictions(state_features), **self.transition_predictions(transition_features)}

    def state_predictions(self, state_features: Tensor) -> dict[str, Tensor]:
        """Predict o_t-aligned targets, including at reset without a past action."""
        _features(state_features, self.width, "state features")
        return self._predict_aligned(self.state_trunk(state_features), "state")

    def transition_predictions(self, transition_features: Tensor) -> dict[str, Tensor]:
        """Predict t->t+1 event presence from the transition representation."""
        _features(transition_features, self.transition_width, "transition features")
        return self._predict_aligned(self.transition_trunk(transition_features), "transition")

    def _predict_aligned(self, hidden: Tensor, alignment: str) -> dict[str, Tensor]:
        result = {}
        for field in self.fields:
            if field.alignment != alignment:
                continue
            prediction = self.heads[field.name](hidden)
            result[field.name] = prediction.float() if field.kind == "categorical" else prediction.squeeze(-1).float()
        return result


@dataclass(frozen=True)
class AuxiliaryTarget:
    value: Tensor
    valid: Tensor


def encode_auxiliary_targets(state_arrays: Mapping[str, object], transition_arrays: Mapping[str, object],
                             task: TaskSpec, event_fields: Sequence[str] | None = None, *,
                             device: torch.device | str | None = None) -> dict[str, AuxiliaryTarget]:
    """Extract each declared field without replacing missing targets by labels.

    Invalid storage padding is returned with a false mask, not supervised as a
    zero. Caller selects the observation index for state_arrays explicitly.
    """
    result = {}
    for field in auxiliary_fields(task.task_family, event_fields):
        arrays = state_arrays if field.alignment == "state" else transition_arrays
        index = (Ellipsis, *field.index)
        value = _as_tensor(arrays[field.array], device=device)[index]
        valid = _as_tensor(arrays[field.validity_array], device=device)[index]
        if valid.dtype != torch.bool or value.shape != valid.shape:
            raise ValueError(f"invalid target mask for {field.name}")
        if not torch.isfinite(value[valid]).all():
            raise ValueError(f"nonfinite valid target for {field.name}")
        if field.alignment == "transition":
            if torch.any(value[valid] < 0) or torch.any(value[valid] != value[valid].round()):
                raise ValueError(f"event counters must be nonnegative integers: {field.name}")
            value = (value > 0).float()
        elif field.kind == "categorical":
            if torch.any((value[valid] < 0) | (value[valid] >= field.classes) | (value[valid] != value[valid].round())):
                raise ValueError(f"invalid categorical target for {field.name}")
            value = value.long()
        else:
            if field.kind == "binary" and torch.any((value[valid] != 0) & (value[valid] != 1)):
                raise ValueError(f"invalid binary target for {field.name}")
            value = value.float()
        result[field.name] = AuxiliaryTarget(value, valid)
    return result
