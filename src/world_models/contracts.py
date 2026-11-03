"""Dependency-free research interfaces shared by collection and learning.

These types enforce the declared *shape* of a local policy call. They do not prove
that image pixels were rendered locally, or that a future policy implementation
does not retain external state; those require the Phase 1/3 locality checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
import math
from typing import Mapping, Protocol, Sequence


LOCAL_RGB_SHAPE = (3, 96, 128)
LOCAL_RGB_BYTES = math.prod(LOCAL_RGB_SHAPE)
LOCAL_HIDDEN_SIZE = 256
ACTION_COUNT = 12
PREVIOUS_ACTION_START = ACTION_COUNT


class Action(IntEnum):
    NOOP = 0
    FORWARD = 1
    BACKWARD = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6
    ATTACK = 7
    USE = 8
    FORWARD_ATTACK = 9
    STRAFE_LEFT_ATTACK = 10
    STRAFE_RIGHT_ATTACK = 11


ACTION_BUTTONS: tuple[tuple[str, ...], ...] = (
    (),
    ("MOVE_FORWARD",),
    ("MOVE_BACKWARD",),
    ("MOVE_LEFT",),
    ("MOVE_RIGHT",),
    ("TURN_LEFT",),
    ("TURN_RIGHT",),
    ("ATTACK",),
    ("USE",),
    ("MOVE_FORWARD", "ATTACK"),
    ("MOVE_LEFT", "ATTACK"),
    ("MOVE_RIGHT", "ATTACK"),
)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _hidden(values: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError("local_hidden must contain a nonempty sequence of numbers")
    return tuple(_finite(value, "local_hidden entry") for value in values)


def validate_hidden_size(values: Sequence[float], expected_size: int) -> tuple[float, ...]:
    """Apply the configured actor size without hard-coding a tuning choice."""
    _integer(expected_size, "expected_size", 1)
    hidden = _hidden(values)
    if len(hidden) != expected_size:
        raise ValueError(f"local_hidden must contain the configured {expected_size} entries")
    return hidden


def validate_action(action: int, *, previous: bool = False) -> int:
    """Validate executable actions, optionally permitting the reset START symbol."""
    validated = _integer(action, "action")
    maximum = PREVIOUS_ACTION_START if previous else ACTION_COUNT - 1
    if validated > maximum:
        raise ValueError(f"action must be between 0 and {maximum}")
    return validated


@dataclass(frozen=True, slots=True)
class RGBFrame:
    """An immutable CHW RGB uint8 frame at the agreed local resolution.

    A bytes payload represents unsigned 8-bit values without importing an array
    library. Runtime tensor adapters will convert this contract explicitly.
    """

    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or len(self.data) != LOCAL_RGB_BYTES:
            raise ValueError(f"RGBFrame needs exactly {LOCAL_RGB_BYTES} immutable bytes")

    @classmethod
    def copy_from_buffer(cls, buffer: bytes | bytearray | memoryview) -> RGBFrame:
        """Detach an engine buffer so later simulator calls cannot mutate it."""
        if not isinstance(buffer, (bytes, bytearray, memoryview)):
            raise ValueError("RGB buffer must be bytes, bytearray, or memoryview")
        return cls(bytes(buffer))

    @property
    def shape(self) -> tuple[int, int, int]:
        return LOCAL_RGB_SHAPE

    @property
    def dtype(self) -> str:
        return "uint8"


@dataclass(frozen=True, slots=True)
class LocalActorInput:
    """Only one actor's RGB, executed previous action, and local memory.

    There is deliberately no agent ID, reward, teammate, clock scalar, or global
    state field. Time can be visible in the locally rendered HUD.
    """

    local_rgb: RGBFrame
    previous_action: int
    local_hidden: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.local_rgb, RGBFrame):
            raise ValueError("local_rgb must be an RGBFrame")
        validate_action(self.previous_action, previous=True)
        object.__setattr__(self, "local_hidden", _hidden(self.local_hidden))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> LocalActorInput:
        allowed = {"local_rgb", "previous_action", "local_hidden"}
        if not isinstance(payload, Mapping) or set(payload) != allowed:
            raise ValueError(f"local actor input keys must be exactly {sorted(allowed)}")
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LocalActorOutput:
    action_probabilities: tuple[float, ...]
    next_local_hidden: tuple[float, ...]

    def __post_init__(self) -> None:
        probabilities = self.action_probabilities
        if not isinstance(probabilities, (tuple, list)) or len(probabilities) != ACTION_COUNT:
            raise ValueError(f"action_probabilities must have {ACTION_COUNT} entries")
        probabilities = tuple(_finite(value, "action probability") for value in probabilities)
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("action probabilities must lie in [0,1]")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("action probabilities must sum to one")
        object.__setattr__(self, "action_probabilities", probabilities)
        object.__setattr__(self, "next_local_hidden", _hidden(self.next_local_hidden))


class LocalActor(Protocol):
    def step(self, inputs: LocalActorInput) -> LocalActorOutput:
        """Return a distribution and new memory without implicit global inputs."""
        ...


@dataclass(frozen=True, slots=True)
class EpisodeClock:
    """Observation-aligned decision index; advancing models t -> t+1."""

    decision_index: int
    decision_limit: int

    def __post_init__(self) -> None:
        _integer(self.decision_index, "decision_index")
        _integer(self.decision_limit, "decision_limit", 1)
        if self.decision_index > self.decision_limit:
            raise ValueError("decision_index cannot exceed the task deadline")

    @property
    def expired(self) -> bool:
        return self.decision_index == self.decision_limit

    @property
    def normalized_remaining(self) -> float:
        return (self.decision_limit - self.decision_index) / self.decision_limit

    def advance(self) -> EpisodeClock:
        if self.expired:
            raise ValueError("cannot execute another decision after the deadline")
        return EpisodeClock(self.decision_index + 1, self.decision_limit)


class BoundaryReason(str, Enum):
    CONTINUE = "continue"
    SUCCESS = "success"
    DEATH = "death"
    FAILURE = "failure"
    DEADLINE = "deadline"
    COLLECTION_CUTOFF = "collection_cutoff"
    IMAGINATION_HORIZON = "imagination_horizon"


_TERMINAL_REASONS = frozenset(
    {BoundaryReason.SUCCESS, BoundaryReason.DEATH, BoundaryReason.FAILURE, BoundaryReason.DEADLINE}
)


@dataclass(frozen=True, slots=True)
class EpisodeBoundary:
    reason: BoundaryReason = BoundaryReason.CONTINUE
    next_observation_valid: tuple[bool, bool] = (True, True)

    def __post_init__(self) -> None:
        if not isinstance(self.reason, BoundaryReason):
            raise ValueError("reason must be a declared BoundaryReason; infrastructure errors are not transitions")
        validity = self.next_observation_valid
        if not isinstance(validity, (tuple, list)) or len(validity) != 2 or any(type(v) is not bool for v in validity):
            raise ValueError("next_observation_valid must contain exactly two booleans")
        object.__setattr__(self, "next_observation_valid", tuple(validity))
        if not self.terminated and not all(validity):
            raise ValueError("a nonterminal transition needs valid next observations for both players")

    @property
    def terminated(self) -> bool:
        return self.reason in _TERMINAL_REASONS

    @property
    def truncated(self) -> bool:
        """External collection truncation; imagination cutoffs are kept distinct."""
        return self.reason is BoundaryReason.COLLECTION_CUTOFF

    @property
    def rollout_cutoff(self) -> bool:
        return self.reason in {BoundaryReason.COLLECTION_CUTOFF, BoundaryReason.IMAGINATION_HORIZON}

    @property
    def bootstrap_mask(self) -> float:
        return 0.0 if self.terminated else 1.0

    @property
    def advantage_recursion_mask(self) -> float:
        return 0.0 if self.terminated or self.rollout_cutoff else 1.0


def resolve_boundary(
    next_clock: EpisodeClock,
    *,
    success: bool = False,
    death: bool = False,
    failure: bool = False,
    collection_cutoff: bool = False,
    next_observation_valid: tuple[bool, bool] = (True, True),
) -> EpisodeBoundary:
    """Resolve authoritative events after a decision, using the resulting clock.

    Death/failure wins over success; success attained at the deadline wins over
    deadline failure. A collection cutoff never overrides a true task terminal.
    Infrastructure failures must be logged separately and cannot be passed here.
    """
    if not isinstance(next_clock, EpisodeClock):
        raise ValueError("next_clock must be an EpisodeClock")
    if any(type(value) is not bool for value in (success, death, failure, collection_cutoff)):
        raise ValueError("event flags must be booleans")
    if death:
        reason = BoundaryReason.DEATH
    elif failure:
        reason = BoundaryReason.FAILURE
    elif success:
        reason = BoundaryReason.SUCCESS
    elif next_clock.expired:
        reason = BoundaryReason.DEADLINE
    elif collection_cutoff:
        reason = BoundaryReason.COLLECTION_CUTOFF
    else:
        reason = BoundaryReason.CONTINUE
    return EpisodeBoundary(reason, next_observation_valid)


@dataclass(frozen=True, slots=True)
class RewardRule:
    step_cost: float = -0.001
    success_reward: float = 1.0
    failure_reward: float = -1.0
    max_milestone_bonus: float = 0.1

    def __post_init__(self) -> None:
        for name in ("step_cost", "success_reward", "failure_reward", "max_milestone_bonus"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.step_cost > 0 or self.max_milestone_bonus < 0:
            raise ValueError("step cost must be <= 0 and maximum milestone bonus >= 0")

    def terminal_component(self, boundary: EpisodeBoundary) -> float:
        if not isinstance(boundary, EpisodeBoundary):
            raise ValueError("boundary must be an EpisodeBoundary")
        if boundary.reason is BoundaryReason.SUCCESS:
            return self.success_reward
        if boundary.terminated:
            return self.failure_reward
        return 0.0

    def reward(self, boundary: EpisodeBoundary, milestone_bonus: float = 0.0) -> float:
        bonus = _finite(milestone_bonus, "milestone_bonus")
        if not 0 <= bonus <= self.max_milestone_bonus:
            raise ValueError("milestone bonus exceeds the declared per-transition range")
        return self.step_cost + self.terminal_component(boundary) + bonus

    def milestone_target(self, total_reward: float, boundary: EpisodeBoundary) -> float:
        """Recover the learned bonus target; reject mislabeled reward/outcome pairs."""
        bonus = _finite(total_reward, "total_reward") - self.step_cost - self.terminal_component(boundary)
        tolerance = 1e-9
        if bonus < -tolerance or bonus > self.max_milestone_bonus + tolerance:
            raise ValueError("reward does not decompose under the declared boundary and rule")
        return min(self.max_milestone_bonus, max(0.0, bonus))


@dataclass(frozen=True, slots=True)
class JointTransitionMetadata:
    """Metadata for one real synchronized two-player action, without images.

    Observation/state targets at decision_index describe t. Reward, boundary and
    event deltas describe t -> t+1; next_decision_index identifies the latter.
    """

    decision_index: int
    start_tic: int
    end_tic: int
    actions: tuple[int, int]
    boundary: EpisodeBoundary
    action_repeat: int = 4

    def __post_init__(self) -> None:
        _integer(self.decision_index, "decision_index")
        _integer(self.start_tic, "start_tic")
        _integer(self.end_tic, "end_tic")
        _integer(self.action_repeat, "action_repeat", 1)
        if not isinstance(self.actions, (tuple, list)) or len(self.actions) != 2:
            raise ValueError("a joint transition requires exactly two actions")
        object.__setattr__(self, "actions", tuple(validate_action(a) for a in self.actions))
        if not isinstance(self.boundary, EpisodeBoundary):
            raise ValueError("boundary must be an EpisodeBoundary")
        if self.boundary.reason is BoundaryReason.IMAGINATION_HORIZON:
            raise ValueError("an imagined horizon cannot label a real replay transition")
        actual_tics = self.actual_simulator_tics
        if actual_tics <= 0 or actual_tics > self.action_repeat:
            raise ValueError("actual tics must lie between one and action_repeat")
        if actual_tics < self.action_repeat and not self.boundary.terminated:
            raise ValueError("only a task terminal may shorten a real action repeat")

    @property
    def next_decision_index(self) -> int:
        return self.decision_index + 1

    @property
    def actual_simulator_tics(self) -> int:
        return self.end_tic - self.start_tic

    @property
    def joint_transition_count(self) -> int:
        return 1
