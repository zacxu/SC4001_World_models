"""Task-facing two-player API with a strict local-image policy boundary."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any

from world_models.contracts import (
    EpisodeBoundary, EpisodeClock, JointTransitionMetadata, LocalActorInput,
    RewardRule, RGBFrame, resolve_boundary,
    validate_action,
)
from world_models.env.coordinator import CoordinatorConfig, CoordinatorError, JointRawResult, TwoPlayerCoordinator
from world_models.env.observations import adapt_rgb
from world_models.env.targets import (
    EVENT_FIELDS, build_privileged_targets, decode_user_field, event_snapshot,
    native_counter_baseline, reward_components,
)
from world_models.infrastructure.config import resolve_config


class LifecycleState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    READY = "READY"
    STEPPING = "STEPPING"
    EPISODE_ENDED = "EPISODE_ENDED"
    NEEDS_RECOVERY = "NEEDS_RECOVERY"
    CLOSED = "CLOSED"


class EnvironmentIntegrityError(CoordinatorError):
    """A consumed interaction could not be published as a valid transition."""


@dataclass(frozen=True)
class ResetResult:
    observations: tuple[RGBFrame, RGBFrame]
    training_only_info: dict[str, Any]

    @property
    def image_reconstruction_mask(self) -> tuple[float, float]:
        """Reset validates two actual observations before publishing either."""
        return (1.0, 1.0)

    def local_input(self, player: int, previous_action: int, local_hidden: tuple[float, ...]) -> LocalActorInput:
        if type(player) is not int or player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        return LocalActorInput(self.observations[player], previous_action, local_hidden)


@dataclass(frozen=True)
class StepResult:
    observations: tuple[RGBFrame | None, RGBFrame | None]
    team_reward: float
    boundary: EpisodeBoundary
    metadata: JointTransitionMetadata
    training_only_info: dict[str, Any]

    @property
    def image_reconstruction_mask(self) -> tuple[float, float]:
        """Mask next-image losses without replacing an unavailable RGB frame."""
        return tuple(float(valid) for valid in self.boundary.next_observation_valid)

    @property
    def rewards(self) -> tuple[float, float]:
        return self.team_reward, self.team_reward

    @property
    def terminated(self) -> bool:
        return self.boundary.terminated

    @property
    def truncated(self) -> bool:
        return self.boundary.truncated


class TwoPlayerCoopEnv:
    """Run an actual generated cooperative scenario, not a Python game substitute.

    A constructor seed chooses the layout/role assignment. ``reset(seed=...)``
    reseeds the engine on that fixed layout. Create another environment to change
    the layout. Raw state is available only under ``training_only_info``.
    """

    def __init__(self, task_family: str, *, seed: int = 0, work_root: str | Path = "artifacts/phase1_steps1_4/environments",
                 frame_skip: int | None = None, scenario_manifest: dict | None = None,
                 coordinator: Any | None = None, config_dir: str | Path = "configs"):
        settings = resolve_config(task_family, "B", "pilot", config_dir).settings
        self.task_family, self.layout_seed = task_family, seed
        self.frame_skip = settings["environment"]["frame_skip"] if frame_skip is None else frame_skip
        if type(self.frame_skip) is not int or self.frame_skip < 1:
            raise ValueError("frame_skip must be a positive integer")
        if scenario_manifest is None:
            from world_models.env.scenario_builder import build_scenario
            scenario_manifest = build_scenario(task_family, seed=seed, frame_skip=self.frame_skip)
        self.scenario = deepcopy(scenario_manifest)
        if scenario_manifest["task_family"] != task_family or scenario_manifest["frame_skip"] != self.frame_skip:
            raise ValueError("scenario does not match selected family/frame skip")
        task = settings["task"]
        if scenario_manifest["deadline"] != task["deadline"]:
            raise ValueError("scenario and shared task deadline disagree")
        if task_family == "cover" and scenario_manifest.get("dwell_decisions") != task["dwell_decisions"]:
            raise ValueError("scenario and selected task console dwell disagree; rebuild using the matching shared task config")
        if "seed" in scenario_manifest and scenario_manifest["seed"] != seed:
            raise ValueError("scenario layout seed differs from the declared layout seed")
        self.clock = EpisodeClock(0, task["deadline"])
        self.reward_rule = RewardRule(step_cost=task["step_cost"], success_reward=task["success_reward"],
                                      failure_reward=task["failure_reward"], max_milestone_bonus=task["milestone_bonus"] * task["milestone_count"])
        self._milestone_bonus, self._max_milestones = task["milestone_bonus"], task["milestone_count"]
        self.coordinator = coordinator or TwoPlayerCoordinator(CoordinatorConfig(
            scenario_path=scenario_manifest["wad_path"], work_root=str(work_root), seed=seed,
            frame_skip=self.frame_skip, episode_timeout_tics=(task["deadline"] + 100) * self.frame_skip,
            user_variable_count=max((int(str(index).removeprefix("USER")) for index in self.scenario["user_variables"].values()), default=0),
        ))
        self._started = False
        self._terminal = False
        self._previous_raw: JointRawResult | None = None
        self._milestones = 0
        self.initialization_joint_decisions = 0
        self.state = LifecycleState.UNINITIALIZED
        self.last_integrity_error: EnvironmentIntegrityError | None = None
        self._previous_flags: dict[str, int] | None = None
        self._previous_events: dict[str, int] | None = None
        self._native_counter_baseline: dict[str, int] = {}

    def _invalidate(self, exc: BaseException, operation: str) -> EnvironmentIntegrityError:
        self.state = LifecycleState.NEEDS_RECOVERY
        details = {"operation": operation, "error_type": type(exc).__name__, "error": str(exc),
                   "wrapper_decision_index": self.clock.decision_index,
                   "initialization_joint_decisions": self.initialization_joint_decisions}
        # The public coordinator interface owns durable accounting and evidence.
        reported = self.coordinator.invalidate(f"environment integrity failure during {operation}: {exc}", details)
        error = EnvironmentIntegrityError(str(reported), reported.record)
        self.last_integrity_error = error
        return error

    def _validate_raw(self, raw: JointRawResult) -> None:
        if not isinstance(raw, JointRawResult) or len(raw.participants) != 2:
            raise ValueError("invalid joint result")
        for name in ("episode_id", "decision_index", "dispatch_id", "consumed_joint_decisions"):
            value = getattr(raw, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid joint result {name}")
        if [p.player_index for p in raw.participants] != [0, 1]:
            raise ValueError("invalid participant identities")
        if raw.participants[0].tic != raw.participants[1].tic:
            raise ValueError("participant tic mismatch")
        for participant in raw.participants:
            if type(participant.tic) is not int or participant.tic < 0:
                raise ValueError("invalid simulator tic")
            if type(participant.dead) is not bool or type(participant.finished) is not bool:
                raise ValueError("invalid engine termination flags")
            if not math.isfinite(participant.engine_reward):
                raise ValueError("non-finite engine reward")
            if self.scenario.get("privileged_registry_version") and participant.engine_reward != 0.0:
                raise ValueError("reserved engine reward is nonzero; shared reward must be computed once centrally")
            if not isinstance(participant.variables, dict) or any(
                    not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for name, value in participant.variables.items()):
                raise ValueError("malformed or non-finite player variables")

    def _flags(self, raw: JointRawResult) -> dict[str, int | float]:
        flags = {}
        for name, index in self.scenario["user_variables"].items():
            key = index if isinstance(index, str) and index.startswith("USER") else f"USER{index}"
            values = [p.variables[key] for p in raw.participants]
            if not all(math.isfinite(value) for value in values) or abs(values[0] - values[1]) > 1e-6:
                raise ValueError(f"authoritative task state differs between players: {name}={values}")
            flags[name] = decode_user_field(name, values[0], self.scenario)
        for name in ("success", "failure", "plate_active", "latch_active", "gate_1_open", "gate_2_open",
                     "console_1_complete", "console_2_complete", "player_0_alive", "player_1_alive", "extraction_unlocked"):
            if name in flags and flags[name] not in (0, 1):
                raise ValueError(f"invalid task flag {name}={flags[name]}")
        for name in ("enemy_1_alive", "enemy_2_alive"):
            if name in flags and flags[name] not in (0, 1):
                raise ValueError(f"invalid task flag {name}={flags[name]}")
        if flags.get("success", 0) and flags.get("failure", 0):
            raise ValueError("task simultaneously reports success and failure")
        if not 0 <= flags.get("stage", 0) <= self._max_milestones + 1:
            raise ValueError("task stage out of range")
        if not 0 <= flags.get("registered_players", 2) <= 2:
            raise ValueError("registered player counter out of range")
        for name in ("progress_1", "progress_2"):
            if name in flags and not 0 <= flags[name] <= self.scenario.get("dwell_decisions", 0):
                raise ValueError(f"task progress counter out of range: {name}")
        return flags

    def _validate_progression(self, before: JointRawResult, after: JointRawResult) -> None:
        if (after.episode_id != before.episode_id or after.decision_index != before.decision_index + 1
                or after.consumed_joint_decisions != before.consumed_joint_decisions + 1
                or after.dispatch_id - before.dispatch_id != after.tic - before.tic
                or not 0 < after.tic - before.tic <= self.frame_skip):
            raise ValueError("joint transition identifiers or consumed-interaction counter are inconsistent")

    def _info(self, raw: JointRawResult, flags: dict, clock: EpisodeClock, *, reset: bool = False,
              events: dict | None = None, components: dict | None = None) -> dict[str, Any]:
        info = {"task_family": self.task_family, "layout_seed": self.layout_seed,
                "engine_seed": getattr(getattr(self.coordinator, "config", None), "seed", None),
                "coordinator_reset_strategy": getattr(getattr(self.coordinator, "config", None), "reset_strategy", None),
                "episode_id": raw.episode_id, "decision_index": clock.decision_index,
                "engine_tic": raw.tic, "dispatch_id": raw.dispatch_id,
                "consumed_joint_decisions": raw.consumed_joint_decisions,
                "initialization_joint_decisions": self.initialization_joint_decisions,
                "task_flags": dict(flags),
                "players": [dict(p.variables) for p in raw.participants],
                "players_selected_iwads": [deepcopy(p.iwad_provenance) for p in raw.participants],
                "reset_actor_memories": reset,
                "image_reconstruction_valid": [p.screen_buffer is not None for p in raw.participants],
                "privileged_targets": build_privileged_targets(raw, flags, self.scenario),
                "events": events,
                "reward_components": components,
                "scenario_manifest": deepcopy(self.scenario)}
        json.dumps(info, allow_nan=False)
        return info

    def _prepare(self, raw: JointRawResult) -> ResetResult:
        self._validate_raw(raw)
        if raw.decision_index != 0:
            raise ValueError("fresh episode did not reset the coordinator decision index")
        if self._previous_raw is not None and (
                raw.episode_id <= self._previous_raw.episode_id
                or raw.dispatch_id <= self._previous_raw.dispatch_id
                or raw.consumed_joint_decisions < self._previous_raw.consumed_joint_decisions):
            raise ValueError("fresh episode reused old identifiers or lost cumulative accounting")
        self._reset_observations(raw)
        # ACS globals can survive new_episode until OPEN executes. Always cross
        # the first initialization boundary before trusting ready/completion flags.
        self.initialization_joint_decisions += 1
        before, raw = raw, self.coordinator.step((0, 0))
        self._validate_raw(raw)
        self._validate_progression(before, raw)
        observations = self._reset_observations(raw)
        flags = self._flags(raw)
        for _ in range(16):
            if flags.get("registered_players", 2) >= 2:
                break
            self.initialization_joint_decisions += 1
            before, raw = raw, self.coordinator.step((0, 0))
            self._validate_raw(raw)
            self._validate_progression(before, raw)
            observations = self._reset_observations(raw)
            flags = self._flags(raw)
        else:
            raise RuntimeError("task scripts did not register both players during initialization")
        if any(flags.get(name, 0) for name in ("stage", "success", "failure", "milestone_count", "progress_1", "progress_2",
                                                "elapsed_decisions", "plate_active", "latch_active")):
            raise RuntimeError("scenario reset retained completed task state")
        if any(flags.get(name,0) for name in EVENT_FIELDS.values()):
            raise RuntimeError("scenario reset retained cumulative task events")
        next_clock = EpisodeClock(0, self.clock.decision_limit)
        next_native_baseline = native_counter_baseline(raw)
        events = event_snapshot(raw,flags,self.scenario,native_baseline=next_native_baseline)
        result = ResetResult(observations, self._info(raw, flags, next_clock, reset=True, events=events))
        self.clock = next_clock
        self._previous_raw, self._milestones, self._terminal = raw, 0, False
        self._previous_flags = dict(flags)
        self._previous_events = dict(events["cumulative"])
        self._native_counter_baseline = next_native_baseline
        self._started = True
        self.state = LifecycleState.READY
        self.last_integrity_error = None
        return result

    @staticmethod
    def _reset_observations(raw: JointRawResult) -> tuple[RGBFrame, RGBFrame]:
        if any(p.dead or p.finished or p.screen_buffer is None for p in raw.participants):
            raise RuntimeError("scenario reset did not produce two living local observations")
        return tuple(adapt_rgb(p.screen_buffer) for p in raw.participants)

    def reset(self, *, seed: int | None = None) -> ResetResult:
        if self.state is LifecycleState.CLOSED:
            raise RuntimeError("environment is closed")
        if self.state in (LifecycleState.NEEDS_RECOVERY, LifecycleState.STEPPING):
            raise RuntimeError("environment requires recover() before reset")
        if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**32 - 1):
            raise ValueError("seed must be a uint32 integer")
        # Coordinator.reset handles both first start and later synchronized resets.
        self.state = LifecycleState.STEPPING
        try:
            return self._prepare(self.coordinator.reset(seed=seed))
        except BaseException as exc:
            raise self._invalidate(exc, "reset") from exc

    def step(self, action_player_1: int, action_player_2: int, *, collection_cutoff: bool = False) -> StepResult:
        if self.state is LifecycleState.CLOSED:
            raise RuntimeError("environment is closed")
        if self.state is LifecycleState.NEEDS_RECOVERY:
            raise RuntimeError("environment requires recover() before stepping")
        if self.state is LifecycleState.STEPPING:
            raise RuntimeError("an environment operation is already in progress")
        if not self._started or self._previous_raw is None:
            raise RuntimeError("call reset() before step()")
        if self._terminal:
            raise RuntimeError("episode has ended; call reset()")
        if type(collection_cutoff) is not bool:
            raise ValueError("collection_cutoff must be a boolean")
        actions = (validate_action(action_player_1), validate_action(action_player_2))
        # No caller validation below this point may consume an engine decision.
        self.clock.advance()
        self.state = LifecycleState.STEPPING
        try:
            return self._step(actions, collection_cutoff)
        except BaseException as exc:
            raise self._invalidate(exc, "step") from exc

    def _step(self, actions: tuple[int, int], collection_cutoff: bool) -> StepResult:
        before = self._previous_raw
        assert before is not None
        raw = self.coordinator.step(actions)
        self._validate_raw(raw)
        self._validate_progression(before, raw)
        flags = self._flags(raw)
        for name in ("stage", "milestone_count", "progress_1", "progress_2", "elapsed_decisions", "latch_active"):
            if self._previous_flags is not None and name in flags and flags[name] < self._previous_flags.get(name, 0):
                raise ValueError(f"task counter regressed: {name}")
        if flags.get("registered_players", 2) != 2:
            raise ValueError("task lost a registered participant")
        if self._previous_flags is not None and "start_tic" in flags and flags["start_tic"] != self._previous_flags.get("start_tic"):
            raise ValueError("task start tic changed within an episode")
        next_clock = self.clock.advance()
        validity = tuple(p.screen_buffer is not None for p in raw.participants)
        boundary = resolve_boundary(next_clock, success=bool(flags.get("success", 0)),
                                    death=any(p.dead for p in raw.participants), failure=bool(flags.get("failure", 0)) and not next_clock.expired,
                                    collection_cutoff=collection_cutoff, next_observation_valid=validity)
        if any(p.finished for p in raw.participants) and not boundary.terminated:
            raise RuntimeError("engine ended without an authoritative task termination; no task-failure reward fabricated")
        if "elapsed_decisions" in flags:
            # ACS stops updating elapsed immediately on task termination. Such
            # an event can happen inside a repeated action, before its last tic.
            minimum = self.clock.decision_index if boundary.terminated else next_clock.decision_index
            if not minimum <= flags["elapsed_decisions"] <= next_clock.decision_index:
                raise ValueError("task elapsed counter disagrees with the wrapper clock")
        milestones = flags.get("milestone_count", 0)
        if not self._milestones <= milestones <= self._max_milestones:
            raise RuntimeError("milestone counter regressed or exceeded the declared one-time events")
        bonus = (milestones - self._milestones) * self._milestone_bonus
        events = event_snapshot(raw,flags,self.scenario,self._previous_events,boundary=boundary,native_baseline=self._native_counter_baseline)
        if self.scenario.get("privileged_registry_version"):
            reward_events = (events["deltas"]["latch_activations"] if self.task_family == "relay" else
                             events["deltas"]["console_1_completions"] + events["deltas"]["console_2_completions"])
            if not math.isclose(bonus,reward_events*self._milestone_bonus,abs_tol=1e-12):
                raise ValueError("reward milestone disagrees with authoritative event deltas")
        reward = self.reward_rule.reward(boundary, bonus)
        components = reward_components(self.reward_rule,boundary,bonus,reward)
        metadata = JointTransitionMetadata(decision_index=self.clock.decision_index, start_tic=before.tic, end_tic=raw.tic,
                                           actions=actions, boundary=boundary, action_repeat=self.frame_skip)
        observations = tuple(adapt_rgb(p.screen_buffer) if p.screen_buffer is not None else None for p in raw.participants)
        if not math.isfinite(reward):
            raise ValueError("non-finite team reward")
        result = StepResult(observations, reward, boundary, metadata, self._info(raw, flags, next_clock,events=events,components=components))
        # Publish atomically only after every potentially failing conversion.
        self.clock, self._previous_raw, self._milestones = next_clock, raw, milestones
        self._previous_flags = dict(flags)
        self._previous_events = dict(events["cumulative"])
        self._terminal = boundary.terminated or boundary.truncated
        self.state = LifecycleState.EPISODE_ENDED if self._terminal else LifecycleState.READY
        return result

    def recover(self) -> ResetResult:
        if self.state is LifecycleState.CLOSED:
            raise RuntimeError("environment is closed")
        if self.state is LifecycleState.STEPPING:
            raise RuntimeError("an environment operation is already in progress")
        self.state = LifecycleState.STEPPING
        try:
            return self._prepare(self.coordinator.recover())
        except BaseException as exc:
            raise self._invalidate(exc, "recovery") from exc

    def close(self) -> None:
        try:
            self.coordinator.close()
        finally:
            self._started = False
            self.state = LifecycleState.CLOSED

    def __enter__(self) -> TwoPlayerCoopEnv:
        self.reset()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
