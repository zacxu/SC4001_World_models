"""Frozen on-policy imagination from fully conditioned, real-prefix anchors."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json

import torch
from torch import Tensor

from world_models.imagination.anchors import AnchorBatch, AnchorBuilder
from world_models.imagination.versions import module_version
from world_models.models.core import WorldState
from world_models.models.world_model import SharedWorldModel
from world_models.policies import ActorOutput, CentralizedCritic, LocalActor, imagined_rgb_to_uint8
from .schema import EndReason, ImaginedRollout
from .streams import RolloutStreams


@contextmanager
def _collection(model, actor, critic, precision, device):
    modes = [(module, module.training) for network in (model, actor, critic) for module in network.modules()]
    try:
        for network in (model, actor, critic):
            network.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                              enabled=precision == "bfloat16"):
            yield
    finally:
        for module, mode in modes:
            module.training = mode


def _check_tensor(name: str, value: Tensor, shape: tuple, device, *, dtype=None):
    if not isinstance(value, Tensor) or value.shape != shape or value.device != device:
        raise ValueError(f"anchor {name} must have shape {shape} on the network device")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"anchor {name} must have dtype {dtype}")
    if value.is_floating_point() and not torch.isfinite(value).all().item():
        raise ValueError(f"anchor {name} must be finite")


class RolloutGenerator:
    """Generate a single collection batch without optimizers or real collection.

    Network objects are borrowed and checked by content hash before and after
    collection. No weights, requires_grad flags, preexisting gradients or global
    RNG streams are modified. Callers keep these network versions frozen for
    the following PPO update. Failed collection restores the explicit RNG state.
    """
    actor_type = LocalActor
    anchor_type = AnchorBatch
    builder_type = AnchorBuilder
    rollout_type = ImaginedRollout

    def __init__(self, model: SharedWorldModel, actor: LocalActor, critic: CentralizedCritic, *,
                 horizon: int = 15, precision: str = "float32"):
        if not isinstance(model, SharedWorldModel) or not isinstance(actor, self.actor_type) or not isinstance(critic, CentralizedCritic):
            raise ValueError(f"provide SharedWorldModel, {self.actor_type.__name__} and CentralizedCritic instances")
        self.model, self.actor, self.critic = model, actor, critic
        self.horizon, self.precision = horizon, precision
        self._device()

    @classmethod
    def from_experiment(cls, model, actor, critic, experiment, *, precision=None):
        from world_models.infrastructure.config import validate_config
        from world_models.models.core import ModelConfig
        from world_models.models.heads import TaskSpec
        from world_models.policies.config import PolicyConfig

        validate_config(experiment)
        if (model.task != TaskSpec.from_experiment(experiment) or model.config != ModelConfig.from_experiment(experiment)
                or actor.config != PolicyConfig.from_experiment(experiment)):
            raise ValueError("networks must match the resolved task and local-policy configuration")
        return cls(model, actor, critic, horizon=experiment["mappo"]["horizon"],
                   precision=experiment["resources"]["mixed_precision"] if precision is None else precision)

    def _device(self):
        if type(self.horizon) is not int or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")
        # Reuse construction guards without reconstructing any prefixes or
        # consuming RNG. No previous P3.1/P3.2 implementation is modified.
        device = self.builder_type(self.model, self.actor, seed=0, precision=self.precision, cache_capacity=0)._device()
        if self.critic.config != self.actor.config:
            raise ValueError("critic and actor must share the same local-policy configuration")
        if any(parameter.device != device or parameter.dtype != torch.float32 for parameter in self.critic.parameters()):
            raise ValueError("critic parameters must be float32 on the world-model/actor device")
        networks = [self.model, self.actor, self.critic]
        seen = set()
        for network in networks:
            owned = {parameter.data_ptr() for parameter in network.parameters()}
            if seen & owned:
                raise ValueError("world model, actor and critic must own separate parameters")
            seen.update(owned)
        return device

    def _anchors(self, anchors, device):
        if type(anchors) is not self.anchor_type or not anchors.references:
            raise ValueError(f"provide a nonempty {self.anchor_type.__name__}")
        batch = len(anchors.references)
        if anchors.precision != self.precision:
            raise ValueError("anchor and rollout precision must match for behavior likelihood replay")
        config = self.model.config
        for name, value, shape, dtype in (
            ("local_rgb", anchors.local_rgb, (batch, 2, *config.image_shape), torch.uint8),
            ("previous_actions", anchors.previous_actions, (batch, 2), torch.long),
            ("actor_hidden", anchors.actor_hidden, (batch, 2, self.actor.config.hidden_size), torch.float32),
            ("actor_logits", anchors.actor_logits, (batch, 2, config.action_count), torch.float32),
            ("shared_tokens", anchors.shared_tokens, (batch, 48, config.width), None),
            ("remaining_time", anchors.remaining_time, (batch,), torch.float32),
            ("world.memory", anchors.world.memory, (batch, 16, config.width), None),
            ("world.stochastic", anchors.world.stochastic, (batch, 32, 32), None),
            ("world.logits", anchors.world.logits, (batch, 32, 32), None),
            ("world.decision_index", anchors.world.decision_index, (batch,), torch.long),
        ):
            _check_tensor(name, value, shape, device, dtype=dtype)
            if dtype is None and not value.is_floating_point():
                raise ValueError(f"anchor {name} must be floating point")
        if ((anchors.previous_actions < 0) | (anchors.previous_actions > config.action_count)).any().item():
            raise ValueError("anchor previous actions must be executable indices or START")
        clocks = torch.tensor([ref.decision_index for ref in anchors.references], device=device)
        if not torch.equal(anchors.world.decision_index, clocks) or ((clocks < 0) | (clocks >= self.model.task.deadline)).any().item():
            raise ValueError("anchor clocks must match recorded source indices before the task deadline")
        if any(ref.cohort_identity != anchors.cohort_identity for ref in anchors.references):
            raise ValueError("anchor references must share the recorded replay cohort")
        expected_remaining = (self.model.task.deadline - clocks).float() / self.model.task.deadline
        if not torch.equal(anchors.remaining_time, expected_remaining):
            raise ValueError("anchor remaining time differs from its recorded decision clock")
        return batch

    @torch.inference_mode(False)
    def collect(self, anchors: AnchorBatch, *, streams: RolloutStreams) -> ImaginedRollout:
        """Collect up to horizon valid steps; terminal states keep a final image.

        State tensors have T+1 entries, transitions T. Source action logits and
        memories already include each source observation. A terminal and horizon
        on the same decision records only the terminal. No real replay suffix is
        read, and no posterior observation occurs during imagination.
        """
        device = self._device()
        batch = self._anchors(anchors, device)
        if not isinstance(streams, RolloutStreams):
            raise ValueError("provide dedicated RolloutStreams")
        streams.validate(device=device)
        versions = tuple(module_version(network) for network in (self.model, self.actor, self.critic))
        if versions[:2] != (anchors.world_model_version, anchors.actor_version):
            raise ValueError("stale anchors: reconstruct with the current world model and behavior actor")
        if streams.identity.get("task_family", self.model.task.task_family) != self.model.task.task_family:
            raise ValueError("rollout RNG task differs from the world model")
        before = streams.state_dict()
        try:
            with _collection(self.model, self.actor, self.critic, self.precision, device):
                result = self._collect(anchors, streams, before, versions, device, batch)
            if tuple(module_version(network) for network in (self.model, self.actor, self.critic)) != versions:
                raise RuntimeError("network weights changed during collection; retry with frozen snapshots")
            return result
        except BaseException:
            streams.load_state_dict(before)
            raise

    def _actor_step(self, rgb, previous, hidden, tokens):
        return self.actor(rgb, previous, hidden)

    def _role_metadata(self, joint_steps):
        return {}

    def _collect(self, anchors, streams, before, versions, device, batch):
        horizon = self.horizon

        def zeros(shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, device=device)

        world = WorldState(zeros((horizon + 1, batch, 16, 256)),
                           zeros((horizon + 1, batch, 32, 32)), zeros((horizon + 1, batch, 32, 32)),
                           zeros((horizon + 1, batch), torch.long))
        rgb = zeros((horizon + 1, batch, 2, 3, 96, 128), torch.uint8)
        previous = zeros((horizon + 1, batch, 2), torch.long)
        hidden = zeros((horizon + 1, batch, 2, self.actor.config.hidden_size))
        logits = zeros((horizon + 1, batch, 2, 12))
        tokens = zeros((horizon + 1, batch, 48, 256))
        values = zeros((horizon + 1, batch))
        actions = zeros((horizon, batch, 2), torch.long)
        log_probs = zeros((horizon, batch, 2))
        rewards, bonus = zeros((horizon, batch)), zeros((horizon, batch))
        valid, terminated, cutoff = (zeros((horizon, batch), torch.bool) for _ in range(3))
        sampled, outcomes = (zeros((horizon, batch), torch.long) for _ in range(2))
        probabilities = zeros((horizon, batch, 3))
        deadline_failure = zeros((horizon, batch), torch.bool)
        reasons = torch.full((horizon, batch), int(EndReason.PADDING), dtype=torch.long, device=device)

        def record_state(index, current, current_rgb, current_previous, current_hidden,
                         current_logits, current_tokens, active):
            for field in ("memory", "stochastic", "logits", "decision_index"):
                getattr(world, field)[index, active] = getattr(current, field)[active]
            rgb[index, active] = current_rgb[active]
            previous[index, active] = current_previous[active]
            hidden[index, active] = current_hidden[active]
            logits[index, active] = current_logits[active]
            tokens[index, active] = current_tokens[active]
            remaining = (self.model.task.deadline - current.decision_index[active]).float() / self.model.task.deadline
            values[index, active] = self.critic(current_tokens[active], current_hidden[active], remaining)

        current = self.model.start_imagination(anchors.world)
        local_hidden, local_logits = anchors.actor_hidden.detach().clone(), anchors.actor_logits.detach().clone()
        record_state(0, current.world, anchors.local_rgb, anchors.previous_actions,
                     local_hidden, local_logits, anchors.shared_tokens, ~current.ended)
        length = 0
        for step in range(horizon):
            active = ~current.ended
            joint_actions = zeros((batch, 2), torch.long)
            # Exactly one stream per player's active batch; inactive rows draw
            # no actions. The shared world transition is advanced only once.
            for player in (0, 1):
                output = ActorOutput(local_logits[active, player], local_hidden[active, player])
                selected = output.sample(generator=streams.actors[player])
                joint_actions[active, player] = selected
                log_probs[step, active, player] = output.log_prob(selected)
            transition = self.model.imagine_step(current, joint_actions, streams=streams.world)
            if not torch.equal(transition.valid, active):
                raise RuntimeError("world-model validity differs from active rollout rows")
            actions[step, active] = joint_actions[active]
            valid[step] = active
            rewards[step] = transition.reward
            bonus[step] = transition.transition.predictions.bonus
            terminated[step] = transition.terminated
            cutoff[step] = active & ~transition.terminated & (step + 1 == horizon)
            sampled[step] = transition.sampled_outcome
            outcomes[step] = transition.outcome
            probabilities[step, active] = torch.softmax(transition.transition.predictions.outcome_logits[active].float(), dim=-1)
            deadline_failure[step] = transition.deadline_failure
            reasons[step, active] = int(EndReason.CONTINUE)
            reasons[step, active & (transition.outcome == 1)] = int(EndReason.SUCCESS)
            reasons[step, active & (transition.sampled_outcome == 2)] = int(EndReason.FAILURE)
            reasons[step, transition.deadline_failure] = int(EndReason.DEADLINE)
            reasons[step, cutoff[step]] = int(EndReason.HORIZON)

            next_rgb = zeros((batch, 2, 3, 96, 128), torch.uint8)
            next_rgb[active] = imagined_rgb_to_uint8(transition.transition.rgb[active])
            output = self._actor_step(next_rgb[active].flatten(0, 1), joint_actions[active].flatten(),
                                      local_hidden[active].flatten(0, 1), transition.transition.representation[active])
            local_hidden = local_hidden.clone()
            local_logits = local_logits.clone()
            local_hidden[active] = output.next_local_hidden.reshape(-1, 2, self.actor.config.hidden_size)
            local_logits[active] = output.logits.reshape(-1, 2, 12)
            record_state(step + 1, transition.next.world, next_rgb, joint_actions, local_hidden,
                         local_logits, transition.transition.representation, active)
            current = transition.next
            length = step + 1
            if current.ended.all().item():
                break

        def trim(value, state=False):
            return value[:length + int(state)].detach().clone()

        trace = WorldState(*(trim(getattr(world, field), True) for field in
                             ("memory", "stochastic", "logits", "decision_index")))
        task = self.model.task
        metadata = {"schema_version": self.rollout_type.schema_version, "data_source": "world_model_imagination",
                    "real_interactions": 0, "horizon": horizon, "deadline": task.deadline,
                    "task_family": task.task_family, "world_model_version": versions[0],
                    "actor_version": versions[1], "critic_version": versions[2],
                    "cohort_identity": anchors.cohort_identity,
                    "anchor_references": [asdict(ref) for ref in anchors.references],
                    "precision": self.precision, "anchor_seed": anchors.seed,
                    "anchor_observation_consumed": True, "prefix_policy": "full_episode_from_reset",
                    "actor_config": asdict(self.actor.config), "world_model_config": asdict(self.model.config),
                    "task_rule": {"step_cost": task.step_cost, "success_reward": task.success_reward,
                                  "failure_reward": task.failure_reward, "max_bonus": task.max_bonus},
                    "terminal_cause_scope": "sampled success, combined sampled failure, or exact deadline; death not separately identified",
                    "imagined_joint_steps": int(valid[:length].sum().item()),
                    "local_player_decisions": int(valid[:length].sum().item()) * 2,
                    "anchor_metadata": anchors.metadata()}
        metadata.update(self._role_metadata(metadata["imagined_joint_steps"]))
        # Dataclass configurations contain tuples; persist one canonical JSON
        # representation so in-memory and saved provenance compare identically.
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
        return self.rollout_type(world=trace, local_rgb=trim(rgb, True), previous_actions=trim(previous, True),
                               actor_hidden=trim(hidden, True), actor_logits=trim(logits, True),
                               shared_tokens=trim(tokens, True), values=trim(values, True),
                               actions=trim(actions), log_probs=trim(log_probs), rewards=trim(rewards), bonus=trim(bonus),
                               valid=trim(valid), terminated=trim(terminated), cutoff=trim(cutoff),
                               sampled_outcomes=trim(sampled), outcomes=trim(outcomes), outcome_probabilities=trim(probabilities),
                               deadline_failure=trim(deadline_failure), end_reason=trim(reasons), metadata=metadata,
                               streams_before=before, streams_after=streams.state_dict())
