"""Detached on-policy synthetic trajectories, with explicit boundary semantics.

States include the final destination: their leading axis is T+1, whereas
actions and rewards have T entries. Local memories have already consumed the
image at their state index. Real replay is referenced, never embedded as a
future observation source.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from enum import IntEnum
import math
import json
import re
from typing import ClassVar

import torch
from torch import Tensor

from world_models.imagination.replay import AnchorRef
from world_models.models.core import WorldState


SCHEMA_VERSION = "imagined-on-policy-v1"
WORLD_FIELDS = ("memory", "stochastic", "logits", "decision_index")


class EndReason(IntEnum):
    PADDING = -1
    CONTINUE = 0
    SUCCESS = 1
    FAILURE = 2
    DEADLINE = 3
    HORIZON = 4


def _tree_copy(value):
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _tree_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_copy(item) for item in value)
    return deepcopy(value)


def _stream_tree(value, name):
    if isinstance(value, Tensor):
        if value.requires_grad or value.grad_fn is not None or value.is_inference():
            raise ValueError(f"{name} must contain detached ordinary tensors")
        if value.layout != torch.strided or value.device.type == "meta":
            raise ValueError(f"{name} must contain materialized dense tensors")
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains nonfinite state")
        return
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError(f"{name} keys must be strings")
        for key, item in value.items():
            _stream_tree(item, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _stream_tree(item, f"{name}[{index}]")
        return
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    raise ValueError(f"{name} must contain tensors or finite JSON-compatible values")


@dataclass(frozen=True)
class ImaginedRollout:
    schema_version: ClassVar[str] = SCHEMA_VERSION
    world: WorldState
    local_rgb: Tensor
    previous_actions: Tensor
    actor_hidden: Tensor
    actor_logits: Tensor
    shared_tokens: Tensor
    values: Tensor
    actions: Tensor
    log_probs: Tensor
    rewards: Tensor
    bonus: Tensor
    valid: Tensor
    terminated: Tensor
    cutoff: Tensor
    sampled_outcomes: Tensor
    outcomes: Tensor
    outcome_probabilities: Tensor
    deadline_failure: Tensor
    end_reason: Tensor
    metadata: dict
    streams_before: dict
    streams_after: dict

    def __post_init__(self):
        self.validate()

    @property
    def steps(self) -> int:
        return self.valid.shape[0]

    @property
    def batch_size(self) -> int:
        return self.valid.shape[1]

    @property
    def state_valid(self) -> Tensor:
        return torch.cat((torch.ones_like(self.valid[:1]), self.valid), dim=0)

    @property
    def old_values(self) -> Tensor:
        return self.values[:-1]

    @property
    def next_values(self) -> Tensor:
        return self.values[1:]

    @property
    def bootstrap_mask(self) -> Tensor:
        return (self.valid & ~self.terminated).to(torch.float32)

    @property
    def recursion_mask(self) -> Tensor:
        return (self.valid & ~self.terminated & ~self.cutoff).to(torch.float32)

    @property
    def bootstrap_values(self) -> Tensor:
        return self.next_values * self.bootstrap_mask

    @property
    def lengths(self) -> Tensor:
        return self.valid.sum(dim=0)

    def _metadata(self) -> tuple[int, int, dict]:
        metadata = self.metadata
        if type(metadata) is not dict:
            raise ValueError("metadata must be a dictionary")
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must contain finite JSON-compatible values") from error
        if metadata.get("schema_version") != self.schema_version:
            raise ValueError("metadata schema_version does not identify synthetic on-policy trajectories")
        for name in ("horizon", "deadline"):
            if type(metadata.get(name)) is not int or metadata[name] < 1:
                raise ValueError(f"metadata {name} must be a positive integer")
        if metadata.get("task_family") not in ("relay", "cover"):
            raise ValueError("metadata task_family must be relay or cover")
        if metadata.get("precision") not in ("float32", "bfloat16"):
            raise ValueError("metadata precision must be float32 or bfloat16")
        actor_config = metadata.get("actor_config")
        if actor_config is not None:
            if (type(actor_config) is not dict or type(actor_config.get("hidden_size")) is not int
                    or actor_config["hidden_size"] != self.actor_hidden.shape[-1]):
                raise ValueError("metadata actor_config hidden_size must match stored actor memories")
        for name in ("world_model_version", "actor_version", "critic_version", "cohort_identity"):
            if type(metadata.get(name)) is not str or re.fullmatch(r"[a-f0-9]{64}", metadata[name]) is None:
                raise ValueError(f"metadata {name} must be a lowercase SHA256 identity")
        if metadata.get("data_source") != "world_model_imagination":
            raise ValueError("synthetic rollout data_source must be world_model_imagination")
        if type(metadata.get("real_interactions")) is not int or metadata["real_interactions"] != 0:
            raise ValueError("synthetic trajectories cannot claim real interactions")
        refs = metadata.get("anchor_references")
        if type(refs) is not list or len(refs) != self.batch_size:
            raise ValueError("metadata anchor_references must contain one prefix reference per batch row")
        for index, reference in enumerate(refs):
            if type(reference) is not dict:
                raise ValueError("anchor_references must contain dictionaries")
            try:
                ref = AnchorRef(**reference)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid anchor/prefix reference") from error
            if ref.cohort_identity != metadata["cohort_identity"]:
                raise ValueError("anchor reference and rollout replay cohorts differ")
            if ref.decision_index != self.world.decision_index[0, index].item():
                raise ValueError("anchor reference clock differs from the first rollout state")
        rule = metadata.get("task_rule")
        required = ("step_cost", "success_reward", "failure_reward", "max_bonus")
        if type(rule) is not dict or any(name not in rule for name in required):
            raise ValueError("metadata task_rule must specify the full reward decomposition")
        for name in required:
            value = rule[name]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"task_rule {name} must be finite")
        if rule["step_cost"] > 0 or rule["success_reward"] <= 0 or rule["failure_reward"] >= 0 or rule["max_bonus"] < 0:
            raise ValueError("task_rule reward signs or bonus bound are invalid")
        for name in ("streams_before", "streams_after"):
            value = getattr(self, name)
            if type(value) is not dict:
                raise ValueError(f"{name} must be a dictionary")
            _stream_tree(value, name)
        return metadata["horizon"], metadata["deadline"], rule

    def validate(self) -> None:
        """Reject misaligned data before it can enter recurrent PPO or storage."""
        if (not isinstance(self.valid, Tensor) or self.valid.dtype != torch.bool or self.valid.ndim != 2
                or min(self.valid.shape) <= 0):
            raise ValueError("valid must be a nonempty boolean [T,B] tensor")
        if not isinstance(self.world, WorldState):
            raise ValueError("world must be a WorldState with a T+1 state axis")
        t, b = self.valid.shape
        device = self.valid.device
        if (not isinstance(self.actor_hidden, Tensor) or self.actor_hidden.ndim != 4
                or self.actor_hidden.shape[-1] < 1):
            raise ValueError("actor_hidden must have shape [T+1,B,2,H] with H>0")
        hidden_size = self.actor_hidden.shape[-1]
        specifications = {
            "local_rgb": ((t + 1, b, 2, 3, 96, 128), torch.uint8),
            "previous_actions": ((t + 1, b, 2), torch.long),
            "actor_hidden": ((t + 1, b, 2, hidden_size), torch.float32),
            "actor_logits": ((t + 1, b, 2, 12), torch.float32),
            "shared_tokens": ((t + 1, b, 48, 256), None),
            "values": ((t + 1, b), torch.float32),
            "actions": ((t, b, 2), torch.long),
            "log_probs": ((t, b, 2), torch.float32),
            "rewards": ((t, b), torch.float32), "bonus": ((t, b), torch.float32),
            "valid": ((t, b), torch.bool), "terminated": ((t, b), torch.bool),
            "cutoff": ((t, b), torch.bool), "sampled_outcomes": ((t, b), torch.long),
            "outcomes": ((t, b), torch.long), "outcome_probabilities": ((t, b, 3), torch.float32),
            "deadline_failure": ((t, b), torch.bool), "end_reason": ((t, b), torch.long),
        }
        states = {
            "world.memory": (self.world.memory, (t + 1, b, 16, 256), None),
            "world.stochastic": (self.world.stochastic, (t + 1, b, 32, 32), None),
            "world.logits": (self.world.logits, (t + 1, b, 32, 32), None),
            "world.decision_index": (self.world.decision_index, (t + 1, b), torch.long),
        }
        tensors = {name: (getattr(self, name), shape, dtype) for name, (shape, dtype) in specifications.items()}
        tensors.update(states)
        for name, (value, shape, dtype) in tensors.items():
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.device != device or value.layout != torch.strided or value.device.type == "meta":
                raise ValueError(f"{name} must be a dense materialized tensor on the rollout device")
            if dtype is None:
                if not value.is_floating_point():
                    raise ValueError(f"{name} must be floating point")
            elif value.dtype != dtype:
                raise ValueError(f"{name} must have dtype {dtype}")
            if value.requires_grad or value.grad_fn is not None or value.is_inference():
                raise ValueError(f"{name} must be detached ordinary tensors")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite, including padding")
        horizon, deadline, rule = self._metadata()
        if t > horizon:
            raise ValueError("stored trajectory exceeds its configured horizon")
        valid = self.valid
        if not bool(valid[0].all()) or not bool(valid[-1].any()):
            raise ValueError("each row must begin with a valid transition and T must contain a valid final transition")
        if bool((valid[1:] & ~valid[:-1]).any()):
            raise ValueError("valid transitions must form an uninterrupted prefix per row")
        if bool((self.terminated & self.cutoff).any()):
            raise ValueError("terminal and horizon cutoff masks must be disjoint")
        ended = self.terminated | self.cutoff
        last = valid & ~torch.cat((valid[1:], torch.zeros_like(valid[:1])), dim=0)
        if not torch.equal(ended, last):
            raise ValueError("each row must end exactly once at its last valid transition")
        if bool(self.cutoff.any()) and (t != horizon or bool(self.cutoff[:-1].any())):
            raise ValueError("a nonterminal cutoff belongs only at the configured imagination horizon")
        # A terminal destination is a genuine state. Everything after that
        # destination is padding, including RGB, memories and raw critic values.
        for name, (value, _, _) in tensors.items():
            mask = ~self.state_valid if name in states or name in (
                "local_rgb", "previous_actions", "actor_hidden", "actor_logits", "shared_tokens", "values") else ~valid
            expected = EndReason.PADDING if name == "end_reason" else 0
            if bool((value[mask] != expected).any()):
                raise ValueError(f"{name} padding must be {int(expected)}")
        clocks = self.world.decision_index
        if bool(((clocks[self.state_valid] < 0) | (clocks[self.state_valid] > deadline)).any()):
            raise ValueError("valid state clocks must lie within the task deadline")
        if bool((clocks[:-1][valid] >= deadline).any()) or not torch.equal(
                clocks[1:][valid], clocks[:-1][valid] + 1):
            raise ValueError("active transitions must advance the source clock by exactly one before its deadline")
        if bool(((self.actions[valid] < 0) | (self.actions[valid] >= 12)).any()):
            raise ValueError("actions must be executable indices 0..11")
        previous = self.previous_actions
        if bool(((previous[self.state_valid] < 0) | (previous[self.state_valid] > 12)).any()):
            raise ValueError("previous_actions must be executable indices or START=12")
        if not torch.equal(previous[1:][valid], self.actions[valid]):
            raise ValueError("destination previous_actions must be the actions actually executed")
        reset = clocks[0] == 0
        if bool((previous[0][reset] != 12).any()) or bool((previous[0][~reset] == 12).any()):
            raise ValueError("START previous_actions must occur exactly at reset anchors")
        probabilities = self.outcome_probabilities[valid]
        if bool(((probabilities < 0) | (probabilities > 1)).any()) or not torch.allclose(
                probabilities.sum(-1), torch.ones_like(probabilities[:, 0]), rtol=0, atol=2e-6):
            raise ValueError("valid outcome probabilities must be normalized categorical distributions")
        sampled = self.sampled_outcomes[valid]
        if bool(((sampled < 0) | (sampled > 2)).any()):
            raise ValueError("sampled outcomes must be continue, success or failure")
        deadline_failure = (sampled == 0) & (clocks[1:][valid] == deadline)
        resolved = torch.where(deadline_failure, 2, sampled)
        if not torch.equal(self.deadline_failure[valid], deadline_failure) or not torch.equal(self.outcomes[valid], resolved):
            raise ValueError("resolved outcomes and deadline precedence must agree with the sampled outcome and clock")
        if not torch.equal(self.terminated[valid], resolved != 0):
            raise ValueError("termination must agree with the resolved outcome")
        expected_reason = torch.where(deadline_failure, int(EndReason.DEADLINE), resolved)
        expected_reason = torch.where(self.cutoff[valid], int(EndReason.HORIZON), expected_reason)
        if not torch.equal(self.end_reason[valid], expected_reason):
            raise ValueError("end_reason must agree with outcome, deadline and cutoff masks")
        if bool(((self.bonus[valid] < 0) | (self.bonus[valid] > rule["max_bonus"])).any()):
            raise ValueError("imagined bonus exceeds the task reward bound")
        terminal_reward = torch.where(resolved == 1, rule["success_reward"],
                                      torch.where(resolved == 2, rule["failure_reward"], 0.0)).float()
        expected_rewards = self.bonus[valid] + rule["step_cost"] + terminal_reward
        if not torch.allclose(self.rewards[valid], expected_rewards, rtol=1e-6, atol=2e-6):
            raise ValueError("shared rewards must obey the sampled outcome/clock decomposition")
        expected_log_probs = self.actor_logits[:-1].log_softmax(-1).gather(-1, self.actions.unsqueeze(-1)).squeeze(-1)
        if not torch.allclose(self.log_probs[valid], expected_log_probs[valid], rtol=1e-5, atol=2e-5):
            raise ValueError("behavior log_probs must match source actor logits and executed actions")

    def tensor_payload(self) -> dict:
        """Plain tensor tree for restricted loading; metadata/RNG are separate."""
        return {"world": {name: getattr(self.world, name) for name in WORLD_FIELDS},
                **{field.name: getattr(self, field.name) for field in fields(self)
                   if field.name not in ("world", "metadata", "streams_before", "streams_after")}}

    @classmethod
    def from_tensor_payload(cls, payload: dict, *, metadata: dict,
                            streams_before: dict, streams_after: dict) -> ImaginedRollout:
        if type(payload) is not dict:
            raise ValueError("rollout tensor payload must be a dictionary")
        expected = {field.name for field in fields(cls)} - {"metadata", "streams_before", "streams_after"}
        if set(payload) != expected or type(payload.get("world")) is not dict or set(payload["world"]) != set(WORLD_FIELDS):
            raise ValueError("rollout tensor payload fields do not match the schema")
        return cls(world=WorldState(**payload["world"]),
                   **{name: value for name, value in payload.items() if name != "world"},
                   metadata=metadata, streams_before=streams_before, streams_after=streams_after)

    @torch.inference_mode(False)
    def to(self, device: str | torch.device) -> ImaginedRollout:
        """Own detached tensors on a new device; keep RNG states on their devices."""
        def move(value):
            if isinstance(value, Tensor):
                return value.detach().to(device).clone()
            return {name: move(item) for name, item in value.items()}
        return type(self).from_tensor_payload(move(self.tensor_payload()), metadata=deepcopy(self.metadata),
                                             streams_before=_tree_copy(self.streams_before),
                                             streams_after=_tree_copy(self.streams_after))
