"""Current-policy local history replay for whole-sequence MAPPO minibatches.

The real prefix ends immediately before the loss-bearing anchor observation.
It is reconstructed without gradients on every call, using current parameters.
The anchor image and imagined source images are then consumed once with BPTT.
Stored behavior memories are deliberately never used as actor initial states.
"""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
import json

import numpy as np
import torch
from torch import Tensor

from world_models.contracts import PREVIOUS_ACTION_START
from world_models.imagination.replay import AnchorPrefix, AnchorRef, AnchorSampler
from world_models.policies.actor import LocalActor
from world_models.rollouts.schema import ImaginedRollout


def _ordinary_replay(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        # inference_mode(False) by itself also overrides explicit no_grad.
        # Initial behavior audits request no_grad; normal optimization needs
        # BPTT and ordinary tensors even inside an inference-only outer scope.
        gradients = torch.is_grad_enabled() or torch.is_inference_mode_enabled()
        with torch.inference_mode(False), torch.set_grad_enabled(gradients):
            return function(*args, **kwargs)
    return wrapped


@contextmanager
def _replay_mode(actor: LocalActor, *, precision: str, device: torch.device):
    modes = tuple((module, module.training) for module in actor.modules())
    try:
        actor.eval()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=precision == "bfloat16"):
            yield
    finally:
        for module, training in modes:
            module.training = training


def _indices(indices, count: int) -> tuple[int, ...]:
    if isinstance(indices, Tensor):
        if indices.ndim != 1 or indices.dtype != torch.long:
            raise ValueError("minibatch indices must be a one-dimensional long tensor or integer sequence")
        indices = indices.detach().cpu().tolist()
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
        raise ValueError("minibatch indices must be a nonempty integer sequence")
    selected = tuple(indices)
    if not selected or any(type(index) is not int or index < 0 or index >= count for index in selected):
        raise ValueError("minibatch indices must identify nonempty whole trajectories")
    if len(set(selected)) != len(selected):
        raise ValueError("a recurrent minibatch cannot duplicate a trajectory")
    return selected


@dataclass(frozen=True)
class PrefixBank:
    """Owned reset-to-anchor RGB and executed-action histories for one rollout.

    Duplicate sampled references share an owned read-only prefix; tensors and
    hidden states are never cached. Keep this bank for the PPO epochs, then
    construct a new one for the next collection's references.
    """

    prefixes: tuple[AnchorPrefix, ...]
    cohort_identity: str
    task_family: str
    deadline: int

    def __post_init__(self):
        if (type(self.prefixes) is not tuple or not self.prefixes
                or any(type(prefix) is not AnchorPrefix for prefix in self.prefixes)):
            raise ValueError("a prefix bank requires a nonempty tuple of safe AnchorPrefixes")
        if self.task_family not in ("relay", "cover") or type(self.deadline) is not int or self.deadline < 1:
            raise ValueError("prefix bank requires a task family and positive deadline")
        owned = {}
        result = []
        for prefix in self.prefixes:
            if prefix.ref.cohort_identity != self.cohort_identity or prefix.ref.decision_index >= self.deadline:
                raise ValueError("prefix reference differs from the bank cohort or deadline")
            if prefix.ref in owned:
                other = owned[prefix.ref]
                if not all(np.array_equal(getattr(prefix, name), getattr(other, name))
                           for name in ("rgb", "actions", "decision_indices")):
                    raise ValueError("duplicate prefix references contain conflicting histories")
            else:
                owned[prefix.ref] = AnchorPrefix(prefix.ref, prefix.rgb, prefix.actions, prefix.decision_indices)
            result.append(owned[prefix.ref])
        object.__setattr__(self, "prefixes", tuple(result))

    @property
    def references(self) -> tuple[AnchorRef, ...]:
        return tuple(prefix.ref for prefix in self.prefixes)

    @classmethod
    def from_rollout(cls, rollout: ImaginedRollout, sampler: AnchorSampler) -> PrefixBank:
        if not isinstance(rollout, ImaginedRollout) or not isinstance(sampler, AnchorSampler):
            raise ValueError("provide a synthetic ImaginedRollout and guarded AnchorSampler")
        rollout.validate()
        if (rollout.metadata["cohort_identity"] != sampler.cohort_identity
                or rollout.metadata["task_family"] != sampler.task_family
                or rollout.metadata["deadline"] != sampler.deadline):
            raise ValueError("rollout and prefix sampler must share the exact replay cohort and task")
        references = tuple(AnchorRef(**value) for value in rollout.metadata["anchor_references"])
        unique = {}
        for reference in references:
            sampler.validate_reference(reference)
            if reference not in unique:
                unique[reference] = sampler.read_prefix(reference)
        result = cls(tuple(unique[reference] for reference in references), sampler.cohort_identity,
                     sampler.task_family, sampler.deadline)
        result.validate_rollout(rollout)
        return result

    def validate_rollout(self, rollout: ImaginedRollout) -> None:
        if not isinstance(rollout, ImaginedRollout):
            raise ValueError("prefix replay requires a synthetic ImaginedRollout")
        if (len(self.prefixes) != rollout.batch_size
                or rollout.metadata.get("cohort_identity") != self.cohort_identity
                or rollout.metadata.get("task_family") != self.task_family
                or rollout.metadata.get("deadline") != self.deadline):
            raise ValueError("prefix bank and rollout must share the exact replay cohort, task and row count")
        try:
            references = tuple(AnchorRef(**value) for value in rollout.metadata["anchor_references"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("rollout contains invalid prefix references") from error
        if references != self.references:
            raise ValueError("prefix bank references differ from the rollout's ordered anchor references")
        images = rollout.local_rgb[0].detach().cpu().numpy()
        previous = rollout.previous_actions[0].detach().cpu().numpy()
        clocks = rollout.world.decision_index[0].detach().cpu().tolist()
        for row, prefix in enumerate(self.prefixes):
            if prefix.ref.decision_index != clocks[row]:
                raise ValueError("rollout anchor clock differs from its prefix")
            expected_previous = (prefix.actions[-1] if len(prefix.actions)
                                 else np.full(2, PREVIOUS_ACTION_START, dtype=np.int64))
            if not np.array_equal(images[row], prefix.rgb[-1]):
                raise ValueError("rollout anchor RGB differs from the safe real prefix")
            if not np.array_equal(previous[row], expected_previous):
                raise ValueError("rollout anchor previous actions differ from its executed real prefix")

    @_ordinary_replay
    def replay(self, actor: LocalActor, rollout: ImaginedRollout, indices: Sequence[int], *,
               precision: str = "float32") -> Tensor:
        """Return current-policy FP32 logits [T,M,2,12] with sequence gradients.

        Prefix work has no gradient path. Its final hidden state is recomputed
        with current actor weights every call; no behavior hidden state, world
        token, reward, other player's image, or critic feature enters the actor.
        Padded source states are skipped and have exactly zero logits.
        """
        self.validate_rollout(rollout)
        if type(rollout) is not ImaginedRollout or rollout.metadata.get("policy_role", "local") != "local":
            raise ValueError("local replay requires a local-policy rollout")
        selected = _indices(indices, len(self.prefixes))
        if not isinstance(actor, LocalActor):
            raise ValueError("current-policy replay requires a LocalActor")
        if precision not in ("float32", "bfloat16") or precision != rollout.metadata["precision"]:
            raise ValueError("recurrent replay precision must match behavior collection precision")
        device = actor.gru.weight_hh.device
        if device.type not in ("cpu", "cuda") or any(
                parameter.device != device or parameter.dtype != torch.float32 for parameter in actor.parameters()):
            raise ValueError("recurrent actor parameters must be FP32 on one CPU or CUDA device")
        if rollout.local_rgb.device != device:
            raise ValueError("actor and rollout must use the same device")
        recorded_config = rollout.metadata.get("actor_config")
        if recorded_config is not None and recorded_config != json.loads(json.dumps(asdict(actor.config))):
            raise ValueError("current actor architecture differs from the behavior actor")
        prefixes = tuple(self.prefixes[index] for index in selected)
        with _replay_mode(actor, precision=precision, device=device):
            # Real histories differ in length. At each reset-relative index,
            # batch only still-needed player rows, without padding RGB banks.
            with torch.no_grad():
                hidden = actor.initial_hidden(len(selected) * 2).reshape(len(selected), 2, -1)
                for step in range(max(prefix.ref.decision_index for prefix in prefixes)):
                    active = [row for row, prefix in enumerate(prefixes) if step < prefix.ref.decision_index]
                    rgb = torch.from_numpy(np.stack([prefixes[row].rgb[step] for row in active])).to(device)
                    previous = torch.from_numpy(np.stack([
                        prefixes[row].actions[step - 1] if step else
                        np.full(2, PREVIOUS_ACTION_START, dtype=np.int64) for row in active])).to(device)
                    rows = torch.tensor(active, dtype=torch.long, device=device)
                    output = actor(rgb.flatten(0, 1), previous.flatten(), hidden[rows].flatten(0, 1))
                    hidden[rows] = output.next_local_hidden.reshape(len(active), 2, -1)
                initial = hidden.detach().reshape(len(selected) * 2, -1)
            rows = torch.tensor(selected, dtype=torch.long, device=device)
            images = rollout.local_rgb[:-1, rows].flatten(1, 2)
            previous = rollout.previous_actions[:-1, rows].flatten(1, 2)
            valid = rollout.valid[:, rows, None].expand(-1, -1, 2).flatten(1, 2)
            sequence = actor.forward_sequence(images, previous, initial, valid=valid)
            return sequence.logits.reshape(rollout.steps, len(selected), 2, actor.config.action_count)


def replay_actor(actor: LocalActor, rollout: ImaginedRollout, prefixes: PrefixBank,
                 indices: Sequence[int], *, precision: str = "float32") -> Tensor:
    """Functional equivalent of :meth:`PrefixBank.replay`."""
    if not isinstance(prefixes, PrefixBank):
        raise ValueError("provide a PrefixBank built from guarded real replay")
    return prefixes.replay(actor, rollout, indices, precision=precision)
