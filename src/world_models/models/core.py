"""Shared categorical recurrent state and visual modules for the P2.2 pilot.

The only stochastic draws are explicit-generator joint categorical samples. A
posterior call changes stochastic evidence, never the deterministic memory or
clock. No module accepts privileged simulator inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    """Validated architecture dimensions; changes require a documented pilot."""

    agents: int = 2
    image_shape: tuple[int, int, int] = (3, 96, 128)
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    memory_tokens: int = 16
    width: int = 256
    transformer_blocks: int = 4
    attention_heads: int = 8
    ffn_width: int = 1024
    latent_variables: int = 32
    latent_categories: int = 32
    action_count: int = 12
    uniform_mixture: float = 0.01
    deadline: int = 400

    def __post_init__(self) -> None:
        expected = {
            "agents": 2, "image_shape": (3, 96, 128),
            "encoder_channels": (32, 64, 128, 256), "memory_tokens": 16,
            "width": 256, "transformer_blocks": 4, "attention_heads": 8,
            "ffn_width": 1024, "latent_variables": 32,
            "latent_categories": 32, "action_count": 12,
        }
        for name, value in expected.items():
            actual = getattr(self, name)
            if actual != value or (type(value) is int and type(actual) is not int):
                raise ValueError(f"P2.2 pilot requires {name}={value!r}")
        if type(self.deadline) is not int or self.deadline <= 0:
            raise ValueError("deadline must be a positive integer")
        _validate_mixture(self.uniform_mixture)

    @classmethod
    def from_experiment(cls, config: Mapping) -> "ModelConfig":
        """Read architectural fields from a resolved B/D experiment config."""
        environment, model, task = (config[name] for name in ("environment", "world_model", "task"))
        names = {item.name for item in fields(cls)}
        values = {name: model[name] for name in names if name in model}
        for name in ("agents", "image_shape", "action_count"):
            values[name] = environment[name]
        values["image_shape"] = tuple(values["image_shape"])
        values["encoder_channels"] = tuple(values["encoder_channels"])
        values["deadline"] = task["deadline"]
        return cls(**values)


@dataclass(frozen=True)
class WorldState:
    """One joint world state, with raw categorical logits and an exact clock."""

    memory: Tensor
    stochastic: Tensor
    logits: Tensor
    decision_index: Tensor

    def detach(self) -> "WorldState":
        return WorldState(*(getattr(self, name).detach() for name in (
            "memory", "stochastic", "logits", "decision_index")))


def _validate_mixture(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("uniform_mixture must be finite and in [0,1)")


def _floating(name: str, value: Tensor, shape: tuple[int, ...] | None = None) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating point tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, received {tuple(value.shape)}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")


def categorical_probs(logits: Tensor, uniform_mixture: float = 0.01) -> Tensor:
    """Mixed probabilities in float32, including under mixed-precision autocast.

    The same function is intended for subsequent prior/posterior KL objectives.
    Mixing is performed on probabilities rather than on raw categorical logits.
    """
    _validate_mixture(uniform_mixture)
    _floating("categorical logits", logits)
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("categorical logits need at least two categories")
    probabilities = F.softmax(logits.float(), dim=-1)
    return probabilities * (1.0 - uniform_mixture) + uniform_mixture / logits.shape[-1]


def sample_categorical(logits: Tensor, *, generator: torch.Generator,
                       uniform_mixture: float = 0.01, deterministic: bool = False) -> Tensor:
    """One-hot sample with straight-through probability gradients.

    ``deterministic`` is a diagnostic argmax mode and consumes no RNG draws.
    Sampling uses the supplied stream; it never consults the global torch RNG.
    """
    if not isinstance(generator, torch.Generator):
        raise ValueError("an explicit torch.Generator is required")
    if type(deterministic) is not bool:
        raise ValueError("deterministic must be boolean")
    probabilities = categorical_probs(logits, uniform_mixture)
    if torch.device(generator.device).type != logits.device.type:
        raise ValueError("categorical generator and logits must use the same device type")
    if logits.device.type == "cuda" and torch.device(generator.device).index not in (None, logits.device.index):
        raise ValueError("categorical generator and logits must use the same CUDA device")
    if deterministic:
        indices = probabilities.argmax(dim=-1)
    else:
        indices = torch.multinomial(probabilities.reshape(-1, logits.shape[-1]), 1,
                                    replacement=True, generator=generator).reshape(logits.shape[:-1])
    hard = F.one_hot(indices, logits.shape[-1]).to(probabilities.dtype)
    return hard + (probabilities - probabilities.detach())


class ImageEncoder(nn.Module):
    """Shared four-block visual encoder; the batch/agent axes share all weights."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        blocks = []
        source = 3
        for target in config.encoder_channels:
            blocks.extend((nn.Conv2d(source, target, 4, stride=2, padding=1),
                           nn.GroupNorm(8, target), nn.SiLU()))
            source = target
        self.cnn = nn.Sequential(*blocks)
        self.spatial_embedding = nn.Parameter(torch.empty(48, config.width))
        self.agent_embedding = nn.Parameter(torch.empty(config.agents, config.width))
        nn.init.normal_(self.spatial_embedding, std=0.02)
        nn.init.normal_(self.agent_embedding, std=0.02)

    def forward(self, observations: Tensor) -> Tensor:
        if not isinstance(observations, Tensor) or observations.ndim != 5 or observations.shape[0] <= 0:
            raise ValueError("observations must have shape [B,2,3,96,128] with B>0")
        batch = observations.shape[0]
        _floating("observations", observations, (batch, self.config.agents, *self.config.image_shape))
        if torch.any((observations < 0) | (observations > 1)).item():
            raise ValueError("observations must be normalized to [0,1]")
        if observations.device != self.spatial_embedding.device:
            raise ValueError("observations must use the encoder device")
        feature = self.cnn(observations.flatten(0, 1))
        tokens = feature.flatten(2).transpose(1, 2).reshape(batch, self.config.agents, 48, self.config.width)
        return tokens + self.spatial_embedding[None, None] + self.agent_embedding[None, :, None]


class StateCore(nn.Module):
    """Shared Transformer recurrence with one categorical sample for both views."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        width = config.width
        self.encoder = ImageEncoder(config)
        self.reset_memory = nn.Parameter(torch.empty(config.memory_tokens, width))
        nn.init.normal_(self.reset_memory, std=0.02)
        self.latent_projection = nn.Linear(config.latent_variables * config.latent_categories,
                                           config.memory_tokens * width)
        self.progress_embedding = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.action_embedding = nn.Embedding(config.action_count, width)
        self.action_agent_embedding = nn.Embedding(config.agents, width)
        self.action_query_norm = nn.LayerNorm(width)
        self.action_key_norm = nn.LayerNorm(width)
        self.action_attention = nn.MultiheadAttention(width, config.attention_heads, dropout=0.0, batch_first=True)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(
            width, config.attention_heads, config.ffn_width, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
            for _ in range(config.transformer_blocks))
        self.dynamics_norm = nn.LayerNorm(width)
        self.prior_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                        nn.Linear(width, config.latent_variables * config.latent_categories))
        self.posterior_query_norm = nn.LayerNorm(width)
        self.posterior_key_norm = nn.LayerNorm(width)
        self.posterior_attention = nn.MultiheadAttention(width, config.attention_heads, dropout=0.0, batch_first=True)
        self.posterior_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                            nn.Linear(width, config.latent_variables * config.latent_categories))
        self.latent_embedding = nn.Linear(config.latent_categories, width, bias=False)
        self.variable_embedding = nn.Parameter(torch.empty(config.latent_variables, width))
        nn.init.normal_(self.variable_embedding, std=0.02)

    def _validate_state(self, state: WorldState) -> int:
        if (not isinstance(state, WorldState) or not isinstance(state.memory, Tensor)
                or state.memory.ndim != 3 or state.memory.shape[0] <= 0):
            raise ValueError("state must be a nonempty WorldState")
        batch = state.memory.shape[0]
        _floating("memory", state.memory, (batch, self.config.memory_tokens, self.config.width))
        expected = (batch, self.config.latent_variables, self.config.latent_categories)
        _floating("stochastic sample", state.stochastic, expected)
        _floating("state logits", state.logits, expected)
        if (not isinstance(state.decision_index, Tensor) or state.decision_index.dtype != torch.long
                or tuple(state.decision_index.shape) != (batch,)):
            raise ValueError("decision_index must be a long tensor of shape [B]")
        if torch.any((state.decision_index < 0) | (state.decision_index > self.config.deadline)).item():
            raise ValueError("decision_index lies outside the episode clock")
        tensors = (state.memory, state.stochastic, state.logits, state.decision_index)
        if len({tensor.device for tensor in tensors}) != 1 or state.memory.device != self.reset_memory.device:
            raise ValueError("all state tensors must use the model device")
        # Checking discrete forward values catches accidental probability-state use.
        detached = state.stochastic.detach()
        if not torch.all((detached == 0) | (detached == 1)).item() or not torch.all(detached.sum(-1) == 1).item():
            raise ValueError("stochastic state must contain one-hot categorical forward samples")
        return batch

    def _progress(self, decisions: Tensor) -> Tensor:
        dtype = self.progress_embedding[0].weight.dtype
        return self.progress_embedding((decisions.to(dtype) / self.config.deadline).unsqueeze(-1))

    def action_tokens(self, actions: Tensor, batch: int | None = None) -> Tensor:
        """Embed executed joint actions, rejecting the non-executable START token."""
        if not isinstance(actions, Tensor) or actions.ndim != 2 or actions.shape[1] != self.config.agents or actions.shape[0] <= 0:
            raise ValueError("actions must have shape [B,2]")
        if batch is not None and actions.shape[0] != batch:
            raise ValueError("action and state batch sizes differ")
        if actions.dtype != torch.long or torch.any((actions < 0) | (actions >= self.config.action_count)).item():
            raise ValueError("actions must contain long executable indices in [0,11]")
        if actions.device != self.reset_memory.device:
            raise ValueError("actions must use the model device")
        ids = torch.arange(self.config.agents, device=actions.device)
        return self.action_embedding(actions) + self.action_agent_embedding(ids)[None]

    def _logits(self, head: nn.Module, memory: Tensor) -> Tensor:
        output = head(memory.mean(dim=1))
        return output.float().reshape(-1, self.config.latent_variables, self.config.latent_categories)

    def _posterior_logits(self, memory: Tensor, observations: Tensor, decisions: Tensor) -> Tensor:
        views = self.encoder(observations).flatten(1, 2)
        if views.shape[0] != memory.shape[0]:
            raise ValueError("observation and memory batch sizes differ")
        context = memory + self._progress(decisions).unsqueeze(1)
        query, keys = self.posterior_query_norm(context), self.posterior_key_norm(views)
        evidence, _ = self.posterior_attention(query, keys, keys, need_weights=False)
        return self._logits(self.posterior_head, context + evidence)

    def reset(self, observations: Tensor, *, generator: torch.Generator) -> WorldState:
        if not isinstance(observations, Tensor) or observations.ndim != 5 or observations.shape[0] <= 0:
            raise ValueError("reset observations must have shape [B,2,3,96,128]")
        if observations.device != self.reset_memory.device:
            raise ValueError("observations must use the model device")
        batch = observations.shape[0]
        memory = self.reset_memory.unsqueeze(0).expand(batch, -1, -1)
        decisions = torch.zeros(batch, dtype=torch.long, device=memory.device)
        logits = self._posterior_logits(memory, observations, decisions)
        stochastic = sample_categorical(logits, generator=generator, uniform_mixture=self.config.uniform_mixture)
        return WorldState(memory, stochastic, logits, decisions)

    def prior_step(self, state: WorldState, actions: Tensor, *, generator: torch.Generator) -> WorldState:
        batch = self._validate_state(state)
        if torch.any(state.decision_index >= self.config.deadline).item():
            raise ValueError("cannot advance an expired world state; reset the episode")
        action_tokens = self.action_tokens(actions, batch)
        projected = self.latent_projection(state.stochastic.flatten(1).to(self.latent_projection.weight.dtype))
        memory_bar = state.memory + projected.reshape(batch, self.config.memory_tokens, self.config.width)
        memory_bar = memory_bar + self._progress(state.decision_index).unsqueeze(1)
        query, keys = self.action_query_norm(memory_bar), self.action_key_norm(action_tokens)
        update, _ = self.action_attention(query, keys, keys, need_weights=False)
        memory = memory_bar + update
        for block in self.blocks:
            memory = block(memory)
        memory = self.dynamics_norm(memory)
        logits = self._logits(self.prior_head, memory)
        stochastic = sample_categorical(logits, generator=generator, uniform_mixture=self.config.uniform_mixture)
        return WorldState(memory, stochastic, logits, state.decision_index + 1)

    def observe(self, state: WorldState, observations: Tensor, *, generator: torch.Generator) -> WorldState:
        self._validate_state(state)
        if not isinstance(observations, Tensor) or observations.device != state.memory.device:
            raise ValueError("observations must use the state device")
        logits = self._posterior_logits(state.memory, observations, state.decision_index)
        stochastic = sample_categorical(logits, generator=generator, uniform_mixture=self.config.uniform_mixture)
        return WorldState(state.memory, stochastic, logits, state.decision_index)

    def representation(self, state: WorldState) -> Tensor:
        """Return the one shared Z: 16 memory plus 32 stochastic-variable tokens."""
        self._validate_state(state)
        tokens = self.latent_embedding(state.stochastic.to(self.latent_embedding.weight.dtype))
        tokens = tokens + self.variable_embedding[None]
        return torch.cat((state.memory, tokens), dim=1)


class ObservationDecoder(nn.Module):
    """View-specific spatial attention queries with one shared upsampling CNN."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.spatial_queries = nn.Parameter(torch.empty(48, config.width))
        self.agent_queries = nn.Parameter(torch.empty(config.agents, config.width))
        nn.init.normal_(self.spatial_queries, std=0.02)
        nn.init.normal_(self.agent_queries, std=0.02)
        self.query_norm = nn.LayerNorm(config.width)
        self.state_norm = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(config.width, config.attention_heads, dropout=0.0, batch_first=True)
        blocks = []
        widths = (config.width, 128, 64, 32, 3)
        for index, (source, target) in enumerate(zip(widths[:-1], widths[1:])):
            blocks.append(nn.ConvTranspose2d(source, target, 4, stride=2, padding=1))
            if index != 3:
                blocks.extend((nn.GroupNorm(8, target), nn.SiLU()))
        self.cnn = nn.Sequential(*blocks)

    def forward(self, representation: Tensor) -> Tensor:
        if not isinstance(representation, Tensor) or representation.ndim != 3 or representation.shape[0] <= 0:
            raise ValueError("shared representation must have shape [B,48,256]")
        batch = representation.shape[0]
        _floating("shared representation", representation,
                  (batch, self.config.memory_tokens + self.config.latent_variables, self.config.width))
        if representation.device != self.spatial_queries.device:
            raise ValueError("representation must use the decoder device")
        queries = self.spatial_queries[None, None] + self.agent_queries[None, :, None]
        queries = queries.expand(batch, -1, -1, -1).reshape(batch * self.config.agents, 48, self.config.width)
        shared = representation[:, None].expand(-1, self.config.agents, -1, -1).reshape(
            batch * self.config.agents, representation.shape[1], self.config.width)
        keys = self.state_norm(shared)
        decoded, _ = self.attention(self.query_norm(queries), keys, keys, need_weights=False)
        spatial = (queries + decoded).transpose(1, 2).reshape(batch * self.config.agents, self.config.width, 6, 8)
        return self.cnn(spatial).sigmoid().reshape(batch, self.config.agents, *self.config.image_shape)
