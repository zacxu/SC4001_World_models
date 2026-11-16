"""Training-only global teacher with strictly local recurrent memory.

Shared model tokens and player identity enter the current action head only.
The teacher is deliberately independent of LocalActor and has no simulator
``step(LocalActorInput)`` or local deployment interface. Its CNN/GRU parameters
are independently initialized, never aliases of a student's weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .actor import ActorOutput, _actor_fp32
from .config import PolicyConfig


@dataclass(frozen=True)
class TeacherAttentionConfig:
    """Bind attention semantics into recursive module_version identities."""
    heads: int
    width: int
    dropout: float = 0.0


class GlobalTeacher(nn.Module):
    """Local history plus current-state cross-attention for teacher actions.

    ``memory_step`` consumes only the agent's image and previously executed
    action. ``action_from_memory`` consumes its resulting memory exactly once
    alongside the current shared tokens. This split lets prefix reconstruction
    avoid any global input and prevents processing the anchor image twice.
    """

    def __init__(self, config: PolicyConfig = PolicyConfig(), attention_heads: int = 4):
        super().__init__()
        if not isinstance(config, PolicyConfig):
            raise ValueError("config must be a PolicyConfig")
        if (type(attention_heads) is not int or attention_heads <= 0
                or config.world_width % attention_heads):
            raise ValueError("attention_heads must be positive and divide world_width")
        self.config = config
        self.attention_heads = attention_heads
        blocks = []
        source = 3
        for target in config.encoder_channels:
            blocks.extend((nn.Conv2d(source, target, 4, stride=2, padding=1),
                           nn.GroupNorm(8, target), nn.SiLU()))
            source = target
        self.encoder = nn.Sequential(*blocks)
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(source * 6 * 8, config.embedding_size), nn.SiLU())
        self.gru = nn.GRUCell(config.embedding_size + config.action_count + 1,
                              config.hidden_size)
        self.query = nn.Linear(config.hidden_size + 2, config.world_width)
        self.cross_attention = nn.MultiheadAttention(
            config.world_width, attention_heads, dropout=0.0, batch_first=True)
        # Head count changes behavior without changing parameter tensor shapes.
        # module_version includes dataclass config on every nested module.
        self.cross_attention.config = TeacherAttentionConfig(attention_heads, config.world_width)
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_size + config.world_width, config.hidden_size),
            nn.SiLU(), nn.Linear(config.hidden_size, config.action_count))

    def initial_hidden(self, batch_size: int) -> Tensor:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return self.gru.weight_hh.new_zeros((batch_size, self.config.hidden_size))

    def _fp32_parameters(self) -> None:
        expected = TeacherAttentionConfig(self.attention_heads, self.config.world_width)
        if (self.cross_attention.config != expected
                or self.cross_attention.num_heads != expected.heads
                or self.cross_attention.embed_dim != expected.width
                or self.cross_attention.dropout != expected.dropout
                or not self.cross_attention.batch_first):
            raise ValueError("teacher attention configuration changed after construction")
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ValueError("teacher parameters must remain FP32; its forward disables autocast")

    def _hidden(self, local_hidden: Tensor, batch: int) -> None:
        if (not isinstance(local_hidden, Tensor)
                or local_hidden.shape != (batch, self.config.hidden_size)
                or local_hidden.device != self.gru.weight_hh.device
                or not local_hidden.is_floating_point()
                or not torch.isfinite(local_hidden).all().item()):
            raise ValueError("local_hidden must be finite [B,hidden_size] on the teacher device")

    def memory_step(self, local_rgb: Tensor, previous_action: Tensor,
                    local_hidden: Tensor) -> Tensor:
        """Advance only local history; START resets its row before consuming RGB."""
        if (not isinstance(local_rgb, Tensor) or local_rgb.ndim != 4
                or local_rgb.shape[0] <= 0
                or tuple(local_rgb.shape[1:]) != self.config.image_shape
                or local_rgb.dtype != torch.uint8
                or local_rgb.device != self.gru.weight_hh.device):
            raise ValueError("local_rgb must be uint8 [B,3,96,128] on the teacher device")
        batch = local_rgb.shape[0]
        if (not isinstance(previous_action, Tensor) or previous_action.shape != (batch,)
                or previous_action.dtype != torch.long
                or previous_action.device != local_rgb.device
                or torch.any((previous_action < 0)
                             | (previous_action > self.config.action_count)).item()):
            raise ValueError("previous_action must be long [B], executable 0..11 or START=12")
        self._hidden(local_hidden, batch)
        self._fp32_parameters()
        with _actor_fp32(local_rgb.device):
            embedding = self.projection(self.encoder(local_rgb.float().div(255)))
            previous = F.one_hot(previous_action, self.config.action_count + 1).float()
            hidden = local_hidden.float()
            hidden = torch.where((previous_action == self.config.action_count)[:, None],
                                 torch.zeros_like(hidden), hidden)
            return self.gru(torch.cat((embedding, previous), dim=-1), hidden)

    def action_from_memory(self, local_hidden: Tensor, shared_tokens: Tensor,
                           player_ids: Tensor) -> ActorOutput:
        """Read current frozen-model tokens without changing local memory.

        Tokens are detached at this boundary even when the caller passes a
        world-model graph. Policy gradients train the attention/head and local
        history branch, never the frozen world model.
        """
        if (not isinstance(local_hidden, Tensor) or local_hidden.ndim != 2
                or local_hidden.shape[0] <= 0):
            raise ValueError("local_hidden must be nonempty [B,hidden_size]")
        batch = local_hidden.shape[0]
        self._hidden(local_hidden, batch)
        if (not isinstance(shared_tokens, Tensor)
                or shared_tokens.shape != (batch, self.config.world_tokens,
                                           self.config.world_width)
                or shared_tokens.device != local_hidden.device
                or not shared_tokens.is_floating_point()
                or not torch.isfinite(shared_tokens).all().item()):
            raise ValueError("shared_tokens must be finite [B,world_tokens,world_width] on the teacher device")
        if (not isinstance(player_ids, Tensor) or player_ids.shape != (batch,)
                or player_ids.dtype != torch.long or player_ids.device != local_hidden.device
                or torch.any((player_ids < 0) | (player_ids > 1)).item()):
            raise ValueError("player_ids must be long [B] containing only 0 or 1")
        self._fp32_parameters()
        with _actor_fp32(local_hidden.device):
            hidden = local_hidden.float()
            identity = F.one_hot(player_ids, 2).float()
            query = self.query(torch.cat((hidden, identity), dim=-1)).unsqueeze(1)
            tokens = shared_tokens.detach().float()
            context, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
            logits = self.action_head(torch.cat((hidden, context.squeeze(1)), dim=-1))
            return ActorOutput(logits, hidden)

    def forward(self, local_rgb: Tensor, previous_action: Tensor, local_hidden: Tensor,
                shared_tokens: Tensor, player_ids: Tensor) -> ActorOutput:
        hidden = self.memory_step(local_rgb, previous_action, local_hidden)
        return self.action_from_memory(hidden, shared_tokens, player_ids)
