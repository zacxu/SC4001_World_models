"""Typed, observation-aligned real replay schema for P2.1.

An invalid fixed-shape slot contains storage padding only. Its mask is false;
readers must not present that padding as an observation or supervision target.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from world_models.contracts import BoundaryReason, LOCAL_RGB_SHAPE, RGBFrame
from world_models.env.targets import PLAYER_SOURCES, TASK_FIELDS, registry_manifest


DATASET_VERSION = "real-replay-v1"
PLAYER_FIELDS = tuple(PLAYER_SOURCES) + ("heading_sin", "heading_cos", "alive")
ENEMY_FIELDS = ("position_x", "position_y", "health", "alive", "spawned")
REASONS = tuple(reason.value for reason in BoundaryReason if reason is not BoundaryReason.IMAGINATION_HORIZON)
STATE_ARRAYS = (
    "rgb", "image_valid", "privileged_player_state", "normalized_player_state",
    "privileged_validity_mask", "task_state", "normalized_task_state", "task_validity_mask",
    "enemy_state", "normalized_enemy_state", "enemy_validity_mask",
    "cumulative_event_counters", "event_validity_mask", "state_engine_tic",
)
TRANSITION_ARRAYS = (
    "transition_id", "episode_id", "decision_index", "start_tic", "end_tic", "actual_tics",
    "observation_index", "next_observation_index", "next_obs_valid", "actions", "team_reward",
    "terminated", "truncated", "termination_reason", "event_deltas", "event_delta_validity_mask",
    "collector_policy_version", "collection_block",
)


def episode_partition(episode_id: str) -> str:
    """Stable 90/10 episode assignment, independent of enumeration or RNG state."""
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("a stable nonempty episode ID is required")
    value = int.from_bytes(hashlib.sha256((DATASET_VERSION + "|" + episode_id).encode()).digest()[:8], "big")
    return "diagnostic" if value % 10 == 0 else "fit"


def field_manifest(event_fields: tuple[str, ...]) -> dict:
    return {
        "version": DATASET_VERSION, "player_fields": list(PLAYER_FIELDS), "task_fields": list(TASK_FIELDS),
        "enemy_fields": list(ENEMY_FIELDS), "event_fields": list(event_fields), "termination_reasons": list(REASONS),
        "state_arrays": list(STATE_ARRAYS), "transition_arrays": list(TRANSITION_ARRAYS),
        "image_shape": [2, *LOCAL_RGB_SHAPE], "image_dtype": "uint8",
        "raw_state_dtype": "float64", "normalized_state_dtype": "float32",
        "state_alignment": "state[index=t] describes observation o_t; transition[index=t] describes t->t+1",
        "observation_layout": "one episode observation sequence; transition indices never duplicate obs/next_obs",
        "invalid_slots": "zero padding with false masks; not a fabricated observation or valid target",
        "privileged_registry": registry_manifest(), "array_specs": array_specs(event_fields),
    }


def array_specs(event_fields: tuple[str, ...]) -> dict:
    """Per-row dtype/shape contract, with independent state/transition axes."""
    specifications = {
        "rgb": ("uint8", (2, 3, 96, 128)), "image_valid": ("bool", (2,)),
        "privileged_player_state": ("float64", (2, len(PLAYER_FIELDS))),
        "normalized_player_state": ("float32", (2, len(PLAYER_FIELDS))),
        "privileged_validity_mask": ("bool", (2, len(PLAYER_FIELDS))),
        "task_state": ("float64", (len(TASK_FIELDS),)), "normalized_task_state": ("float32", (len(TASK_FIELDS),)),
        "task_validity_mask": ("bool", (len(TASK_FIELDS),)),
        "enemy_state": ("float64", (2, len(ENEMY_FIELDS))), "normalized_enemy_state": ("float32", (2, len(ENEMY_FIELDS))),
        "enemy_validity_mask": ("bool", (2, len(ENEMY_FIELDS))),
        "cumulative_event_counters": ("int64", (len(event_fields),)), "event_validity_mask": ("bool", (len(event_fields),)),
        "state_engine_tic": ("int64", ()), "transition_id": ("int64", ()), "episode_id": ("U128", ()),
        "decision_index": ("int64", ()), "start_tic": ("int64", ()), "end_tic": ("int64", ()), "actual_tics": ("uint16", ()),
        "observation_index": ("int64", ()), "next_observation_index": ("int64", ()), "next_obs_valid": ("bool", (2,)),
        "actions": ("uint8", (2,)), "team_reward": ("float64", ()), "terminated": ("bool", ()),
        "truncated": ("bool", ()), "termination_reason": ("uint8", ()),
        "event_deltas": ("int64", (len(event_fields),)), "event_delta_validity_mask": ("bool", (len(event_fields),)),
        "collector_policy_version": ("U128", ()), "collection_block": ("int64", ()),
    }
    return {name: {"dtype": np.dtype(dtype).str, "row_shape": list(shape),
                   "alignment": "state" if name in STATE_ARRAYS else "transition"}
            for name, (dtype, shape) in specifications.items()}


def _values(record: dict | None, fields: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.zeros(len(fields), dtype=np.float64)
    normalized = np.zeros(len(fields), dtype=np.float32)
    validity = np.zeros(len(fields), dtype=np.bool_)
    if record is None:
        return raw, normalized, validity
    for index, name in enumerate(fields):
        is_valid = record.get("valid", {}).get(name, False)
        if type(is_valid) is not bool:
            raise ValueError(f"nonboolean target mask for {name}")
        value, norm = record.get("raw", {}).get(name), record.get("normalized", {}).get(name)
        for candidate in (value, norm):
            if candidate is not None and (isinstance(candidate, bool) or not isinstance(candidate, (int, float)) or not math.isfinite(candidate)):
                raise ValueError(f"nonfinite/nonnumeric state target {name}")
        if is_valid and (value is None or norm is None):
            raise ValueError(f"a valid target is missing its value: {name}")
        if value is not None:
            raw[index] = value
        if norm is not None:
            normalized[index] = norm
            if not np.isfinite(normalized[index]):
                raise ValueError(f"normalized target overflows storage dtype: {name}")
        validity[index] = is_valid
    return raw, normalized, validity


def _event_values(info: dict, event_fields: tuple[str, ...], *, delta: bool) -> tuple[np.ndarray, np.ndarray]:
    events = info.get("events") or {}
    values = events.get("deltas" if delta else "cumulative", {})
    if set(values) != set(event_fields):
        raise ValueError("event field schema changed or is missing")
    valid = events.get("valid", {})
    output = np.empty(len(event_fields), dtype=np.int64)
    masks = np.empty(len(event_fields), dtype=np.bool_)
    for index, name in enumerate(event_fields):
        value = values[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid event counter {name}")
        mask = valid.get(name, True)
        if type(mask) is not bool:
            raise ValueError(f"invalid event validity {name}")
        output[index], masks[index] = value, mask
    return output, masks


def event_field_names(result: Any) -> tuple[str, ...]:
    values = (result.training_only_info.get("events") or {}).get("cumulative")
    if not isinstance(values, dict) or not values:
        raise ValueError("real replay requires authoritative cumulative events")
    return tuple(sorted(values))


def extract_state(result: Any, event_fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    info = result.training_only_info
    frames = result.observations
    if len(frames) != 2:
        raise ValueError("two local observation slots are required")
    rgb = np.zeros((2, *LOCAL_RGB_SHAPE), dtype=np.uint8)
    image_valid = np.array([frame is not None for frame in frames], dtype=np.bool_)
    for player, frame in enumerate(frames):
        if frame is not None:
            if not isinstance(frame, RGBFrame):
                raise TypeError("real replay accepts RGBFrame or an explicitly missing terminal frame")
            rgb[player] = np.frombuffer(frame.data, dtype=np.uint8).reshape(LOCAL_RGB_SHAPE)
    if hasattr(result, "boundary") and tuple(bool(v) for v in image_valid) != result.boundary.next_observation_valid:
        raise ValueError("terminal RGB slots disagree with boundary masks")
    targets = info.get("privileged_targets")
    if targets is not None and (targets.get("schema_version") != registry_manifest()["schema_version"] or targets.get("registry_sha256") != registry_manifest()["sha256"]):
        raise ValueError("privileged target registry differs from the replay schema")
    players = targets.get("players") if targets else None
    if players is not None and (len(players) != 2 or [p.get("object_id") for p in players] != ["player_0", "player_1"]):
        raise ValueError("player targets are not in stable player order")
    player_values = [_values(players[player] if players else None, PLAYER_FIELDS) for player in (0, 1)]
    task_values = _values(targets.get("task") if targets else None, tuple(TASK_FIELDS))
    objects = {obj["object_id"]: obj for obj in targets.get("objects", [])} if targets else {}
    enemy_values = [_values(objects.get(f"enemy_{enemy}", {}).get("state"), ENEMY_FIELDS) for enemy in (1, 2)]
    cumulative, event_valid = _event_values(info, event_fields, delta=False)
    tic = info.get("engine_tic")
    if type(tic) is not int or tic < 0:
        raise ValueError("state has no valid engine tic")
    return {
        "rgb": rgb, "image_valid": image_valid,
        "privileged_player_state": np.stack([value[0] for value in player_values]),
        "normalized_player_state": np.stack([value[1] for value in player_values]),
        "privileged_validity_mask": np.stack([value[2] for value in player_values]),
        "task_state": task_values[0], "normalized_task_state": task_values[1], "task_validity_mask": task_values[2],
        "enemy_state": np.stack([value[0] for value in enemy_values]),
        "normalized_enemy_state": np.stack([value[1] for value in enemy_values]),
        "enemy_validity_mask": np.stack([value[2] for value in enemy_values]),
        "cumulative_event_counters": cumulative, "event_validity_mask": event_valid,
        "state_engine_tic": np.array(tic, dtype=np.int64),
    }


def extract_transition(step: Any, event_fields: tuple[str, ...], *, transition_id: int,
                       observation_index: int, episode_metadata: dict) -> dict[str, np.ndarray]:
    metadata = step.metadata
    reward = step.team_reward
    if not math.isfinite(reward):
        raise ValueError("nonfinite transition reward")
    if metadata.decision_index != observation_index:
        raise ValueError("transition decision index does not match the current observation")
    if step.boundary.reason.value not in REASONS:
        raise ValueError("synthetic/imagination boundaries may not enter real replay")
    delta, valid = _event_values(step.training_only_info, event_fields, delta=True)
    return {
        "transition_id": np.array(transition_id, dtype=np.int64),
        "episode_id": np.array(episode_metadata["episode_id"], dtype="U128"),
        "decision_index": np.array(metadata.decision_index, dtype=np.int64),
        "start_tic": np.array(metadata.start_tic, dtype=np.int64), "end_tic": np.array(metadata.end_tic, dtype=np.int64),
        "actual_tics": np.array(metadata.actual_simulator_tics, dtype=np.uint16),
        "observation_index": np.array(observation_index, dtype=np.int64),
        "next_observation_index": np.array(observation_index + 1, dtype=np.int64),
        "next_obs_valid": np.array(step.boundary.next_observation_valid, dtype=np.bool_),
        "actions": np.array(metadata.actions, dtype=np.uint8), "team_reward": np.array(reward, dtype=np.float64),
        "terminated": np.array(step.terminated, dtype=np.bool_), "truncated": np.array(step.truncated, dtype=np.bool_),
        "termination_reason": np.array(REASONS.index(step.boundary.reason.value), dtype=np.uint8),
        "event_deltas": delta, "event_delta_validity_mask": valid,
        "collector_policy_version": np.array(episode_metadata["policy_version"], dtype="U128"),
        "collection_block": np.array(episode_metadata.get("collection_block", 0), dtype=np.int64),
    }
