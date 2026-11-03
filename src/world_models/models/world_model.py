"""P2.2 joint stochastic world model; fitting objectives live in later phases.

Only synchronized RGB, actions, and the exact decision clock enter this model.
Privileged replay arrays are accepted by target adapters, never by this class.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import Tensor, nn

from world_models.infrastructure.reproducibility import derive_seed
from .core import ModelConfig, ObservationDecoder, StateCore, WorldState
from .heads import AuxiliaryHeads, TaskSpec, TransitionHeads, TransitionPredictions, resolve_outcomes, total_reward


@dataclass
class SamplingStreams:
    """Separate serializable streams; no model operation samples global RNG."""

    prior: torch.Generator
    posterior: torch.Generator
    outcome: torch.Generator

    @classmethod
    def from_seed(cls, seed: int, device: str | torch.device = "cpu") -> "SamplingStreams":
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be a nonnegative signed 64-bit integer")
        streams = []
        for name in ("prior", "posterior", "outcome"):
            digest = hashlib.sha256(f"p22-sampling-v1|{seed}|{name}".encode()).digest()
            stream_seed = int.from_bytes(digest[:8], "big") % (2**63)
            streams.append(torch.Generator(device=device).manual_seed(stream_seed))
        return cls(*streams)

    def state_dict(self) -> dict[str, Tensor]:
        return {name: getattr(self, name).get_state().clone() for name in ("prior", "posterior", "outcome")}

    def load_state_dict(self, states: dict[str, Tensor]) -> None:
        if set(states) != {"prior", "posterior", "outcome"}:
            raise ValueError("sampling state must contain exactly prior/posterior/outcome")
        for name, state in states.items():
            getattr(self, name).set_state(state)


@dataclass
class StatePrediction:
    state: WorldState
    representation: Tensor
    rgb: Tensor
    auxiliary: dict[str, Tensor]


@dataclass
class ModelTransition:
    prior: WorldState
    state: WorldState
    representation: Tensor
    rgb: Tensor
    predictions: TransitionPredictions
    auxiliary: dict[str, Tensor]
    posterior_valid: Tensor


@dataclass
class ImaginationState:
    world: WorldState
    ended: Tensor

    def detach(self) -> "ImaginationState":
        return ImaginationState(self.world.detach(), self.ended.detach())


@dataclass
class ImaginedTransition:
    transition: ModelTransition
    next: ImaginationState
    valid: Tensor
    sampled_outcome: Tensor
    outcome: Tensor
    reward: Tensor
    terminated: Tensor
    deadline_failure: Tensor


def _select(state: WorldState, mask: Tensor) -> WorldState:
    return WorldState(*(getattr(state, field)[mask] for field in ("memory", "stochastic", "logits", "decision_index")))


def _scatter(value: Tensor, mask: Tensor, base: Tensor | None = None) -> Tensor:
    output = value.new_zeros((mask.shape[0], *value.shape[1:])) if base is None else base.clone()
    return output.index_copy(0, mask.nonzero(as_tuple=True)[0], value)


def _scatter_state(state: WorldState, mask: Tensor, base: WorldState) -> WorldState:
    return WorldState(*(_scatter(getattr(state, field), mask, getattr(base, field))
                        for field in ("memory", "stochastic", "logits", "decision_index")))


class SharedWorldModel(nn.Module):
    """One shared world sample supplies both view decoders and transition heads.

    Public image inputs are uint8 or finite floating [0,1] BCHW joint images.
    ``observe_step`` is for real synchronized evidence. ``imagine_step`` accepts
    no image or privileged-state argument and consults only its prior.
    """

    def __init__(self, config: ModelConfig | None = None, task: str | TaskSpec = "relay"):
        super().__init__()
        self.task = TaskSpec.for_family(task) if isinstance(task, str) else task
        self.config = config if config is not None else ModelConfig(deadline=self.task.deadline)
        if self.config.deadline != self.task.deadline:
            raise ValueError("model progress clock must agree with the task deadline")
        self.core = StateCore(self.config)
        self.decoder = ObservationDecoder(self.config)
        # Concatenation retains which player executed each action. A mean of
        # additive action/identity embeddings would discard that association.
        self.joint_action_projection = nn.Sequential(nn.Linear(2*self.config.width, self.config.width), nn.SiLU())
        self.transition_heads = TransitionHeads(input_width=3*self.config.width,
                                                hidden_width=self.config.width,
                                                max_bonus=self.task.max_bonus)
        self.auxiliary_heads = AuxiliaryHeads(width=self.config.width,
                                              task_family=self.task.task_family,
                                              transition_width=3*self.config.width)

    def _images(self, images: Tensor) -> Tensor:
        if not isinstance(images, Tensor):
            raise TypeError("images must be tensors; convert replay arrays explicitly")
        if images.dtype == torch.uint8:
            return images.to(dtype=torch.float32).div(255)
        if not images.is_floating_point():
            raise ValueError("images must be uint8 or floating [0,1]")
        return images.float()

    def initialize(self, observations: Tensor, *, streams: SamplingStreams) -> WorldState:
        return self.core.reset(self._images(observations), generator=streams.posterior)

    def reconstruct(self, state: WorldState) -> StatePrediction:
        """Initial observation/state heads need no preceding transition label."""
        representation = self.core.representation(state)
        auxiliary = self.auxiliary_heads.state_predictions(representation.mean(dim=1))
        return StatePrediction(state, representation, self.decoder(representation), auxiliary)

    def _predict(self, source: WorldState, actions: Tensor, prior: WorldState,
                 destination: WorldState, posterior_valid: Tensor) -> ModelTransition:
        # z was sampled in the core. Neither decoder nor heads draw new z.
        source_features = self.core.representation(source).mean(dim=1)
        representation = self.core.representation(destination)
        destination_features = representation.mean(dim=1)
        action_features = self.joint_action_projection(self.core.action_tokens(actions, batch=actions.shape[0]).flatten(1))
        features = torch.cat((source_features, action_features, destination_features), dim=-1)
        predictions = self.transition_heads(features)
        auxiliary = self.auxiliary_heads(destination_features, features)
        return ModelTransition(prior, destination, representation, self.decoder(representation),
                               predictions, auxiliary, posterior_valid)

    def observe_step(self, source: WorldState, actions: Tensor, next_observations: Tensor,
                     *, streams: SamplingStreams, next_image_valid: Tensor | None = None) -> ModelTransition:
        """t action -> t+1 prior, optionally corrected by both real local views.

        Rows missing either view retain the prior. Their posterior validity is
        false even when reward/outcome labels remain usable for later fitting.
        Invalid image slots are never passed through the CNN.
        """
        prior = self.core.prior_step(source, actions, generator=streams.prior)
        batch = source.memory.shape[0]
        if not isinstance(next_observations, Tensor) or next_observations.shape != (batch, 2, 3, 96, 128):
            raise ValueError("next observations must have shape [B,2,3,96,128]")
        if next_image_valid is None:
            next_image_valid = torch.ones((batch, 2), dtype=torch.bool, device=source.memory.device)
        if (next_image_valid.shape != (batch, 2) or next_image_valid.dtype != torch.bool
                or next_image_valid.device != source.memory.device):
            raise ValueError("next_image_valid must be boolean [B,2] on the model device")
        valid = next_image_valid.all(dim=-1)
        destination = prior
        if bool(valid.any()):
            observed = self.core.observe(_select(prior, valid), self._images(next_observations[valid]),
                                         generator=streams.posterior)
            destination = _scatter_state(observed, valid, prior)
        return self._predict(source, actions, prior, destination, valid)

    def start_imagination(self, anchor: WorldState) -> ImaginationState:
        """Caller supplies a nonterminal fitting-replay anchor, later reconstructed
        from its full prefix. Detach here to separate future policy gradients.
        """
        if bool((anchor.decision_index >= self.task.deadline).any()):
            raise ValueError("imagination anchors must precede the task deadline")
        return ImaginationState(anchor.detach(), torch.zeros_like(anchor.decision_index, dtype=torch.bool))

    def imagine_step(self, current: ImaginationState, actions: Tensor,
                     *, streams: SamplingStreams) -> ImaginedTransition:
        """Advance active rows only. Finished rows freeze; valid=False masks outputs.

        Inactive-row RGB/head values are zero padding and never observations or
        rewards. The returned next.ended is cumulative; terminated marks only
        this decision's valid terminal transitions.
        """
        source, ended = current.world, current.ended
        batch = source.memory.shape[0]
        if ended.shape != (batch,) or ended.dtype != torch.bool or ended.device != source.memory.device:
            raise ValueError("ended must be boolean [B] on the model device")
        if actions.shape != (batch, 2) or actions.dtype != torch.long or actions.device != source.memory.device:
            raise ValueError("joint actions must be long [B,2] on the model device")
        if bool(((actions < 0) | (actions >= self.config.action_count)).any()):
            raise ValueError("joint actions must be executable action indices")
        active = ~ended
        if not bool(active.any()):
            raise ValueError("all imagined episodes ended; reset or select a new anchor")
        if bool((source.decision_index[active] >= self.task.deadline).any()):
            raise ValueError("cannot advance an active state beyond its deadline")
        selected, selected_actions = _select(source, active), actions[active]
        prior = self.core.prior_step(selected, selected_actions, generator=streams.prior)
        valid_posterior = torch.zeros_like(prior.decision_index, dtype=torch.bool)
        prediction = self._predict(selected, selected_actions, prior, prior, valid_posterior)
        # Outcome is one categorical draw for the entire joint transition,
        # separate from the shared latent draw. Probability math stays float32.
        probabilities = torch.softmax(prediction.predictions.outcome_logits.float(), dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=streams.outcome).squeeze(-1)
        resolved = resolve_outcomes(sampled, prior.decision_index, self.task)
        reward = total_reward(prediction.predictions.bonus, resolved, self.task)
        next_world = _scatter_state(prior, active, source)
        termination = _scatter(resolved.terminated, active)
        full_prediction = ModelTransition(
            next_world, next_world, _scatter(prediction.representation, active),
            _scatter(prediction.rgb, active),
            TransitionPredictions(_scatter(prediction.predictions.bonus, active),
                                  _scatter(prediction.predictions.outcome_logits, active)),
            {name: _scatter(value, active) for name, value in prediction.auxiliary.items()},
            torch.zeros_like(ended))
        return ImaginedTransition(full_prediction, ImaginationState(next_world, ended | termination),
                                  active, _scatter(sampled, active), _scatter(resolved.outcome, active),
                                  _scatter(reward, active), termination, _scatter(resolved.deadline_failure, active))

    def imagine(self, anchor: WorldState, actions: Tensor, *, streams: SamplingStreams) -> tuple[ImaginedTransition, ...]:
        """Fixed-action prior-only diagnostic rollout; no real future input exists.

        Actions have shape [T,B,2]. Returns at most T steps, stopping when every
        batch row ends. Mixed-length rows carry explicit validity masks.
        """
        if actions.ndim != 3 or actions.shape[1:] != (anchor.memory.shape[0], 2):
            raise ValueError("rollout actions must have shape [T,B,2]")
        current = self.start_imagination(anchor)
        transitions = []
        for step_actions in actions:
            transition = self.imagine_step(current, step_actions, streams=streams)
            transitions.append(transition)
            current = transition.next
            if bool(current.ended.all()):
                break
        return tuple(transitions)


def build_model(experiment: dict, *, master_seed: int, variant: str) -> SharedWorldModel:
    """Same model initialization for paired B/D without perturbing global RNG.

    CUDA allocation is explicit (model.to(device)); construction is on CPU.
    Sampling stream seeds are managed separately from initialization.
    """
    task = experiment["task"]["name"]
    seed = derive_seed(task, master_seed, "world_model_initialization", variant)
    with torch.random.fork_rng(devices=[]):
        # CPU-only construction must not reseed unrelated CUDA streams.
        torch.random.default_generator.manual_seed(seed)
        return SharedWorldModel(ModelConfig.from_experiment(experiment), TaskSpec.from_experiment(experiment))
