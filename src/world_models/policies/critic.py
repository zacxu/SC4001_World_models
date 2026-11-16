"""Memory-aware shared team values for the paired B/D local students."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import PolicyConfig


class CentralizedCritic(nn.Module):
    """One raw team value from shared state, both local memories and the clock.

    Inputs are fixed rollout features: the value objective never backpropagates
    into either actor or the world model. The critic owns only its MLP weights.
    Actor order is retained when concatenating memories; world tokens are
    mean-pooled. Remaining time is the known task clock normalized to [0, 1].
    """

    def __init__(self, config: PolicyConfig = PolicyConfig()):
        super().__init__()
        if not isinstance(config, PolicyConfig):
            raise ValueError("config must be a PolicyConfig")
        self.config = config
        widths = (config.world_width + 2 * config.hidden_size + 1,
                  *config.critic_hidden_sizes, 1)
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(source, target))
            if index < len(widths) - 2:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, shared_tokens: Tensor, actor_memories: Tensor,
                remaining_time: Tensor) -> Tensor:
        """Return values [B] without target or reward normalization."""
        if (not isinstance(shared_tokens, Tensor) or shared_tokens.ndim != 3
                or shared_tokens.shape[0] <= 0):
            raise ValueError("shared_tokens must have shape [B,48,256] with B>0")
        batch = shared_tokens.shape[0]
        reference = self.mlp[0].weight
        expected = (
            ("shared_tokens", shared_tokens,
             (batch, self.config.world_tokens, self.config.world_width)),
            ("actor_memories", actor_memories, (batch, 2, self.config.hidden_size)),
            ("remaining_time", remaining_time, (batch,)),
        )
        for name, value, shape in expected:
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating point tensor")
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.device != reference.device:
                raise ValueError(f"{name} must use the critic device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if bool(((remaining_time < 0) | (remaining_time > 1)).any()):
            raise ValueError("remaining_time must be normalized to [0,1]")
        # Explicit conversion supports bf16 world features alongside fp32 local
        # memories. Detach before pooling or concatenation, including the clock.
        features = torch.cat((
            shared_tokens.detach().to(reference.dtype).mean(dim=1),
            actor_memories.detach().to(reference.dtype).flatten(1),
            remaining_time.detach().to(reference.dtype).unsqueeze(-1),
        ), dim=-1)
        values = self.mlp(features).squeeze(-1)
        return values.float() if values.dtype in (torch.float16, torch.bfloat16) else values


def raw_value_loss(values: Tensor, targets: Tensor, *, valid: Tensor | None = None,
                   delta: float = 1.0) -> Tensor:
    """Mean raw Huber loss, counted once per valid joint sample.

    Values/targets may have batch or sequence/batch axes but must have exactly
    the same nonempty shape. Targets are fixed data, detached even if supplied
    with a gradient graph. Select valid entries before arithmetic so NaN padding
    is harmless. Empty masks are rejected because no value update is defined.
    """
    if (isinstance(delta, bool) or not isinstance(delta, (int, float))
            or not math.isfinite(delta) or delta <= 0):
        raise ValueError("delta must be a finite positive number")
    for name, value in (("values", values), ("targets", targets)):
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating point tensor")
    if values.ndim < 1 or values.numel() == 0 or values.shape != targets.shape:
        raise ValueError("values and targets must have the same nonempty joint-sample shape")
    if targets.device != values.device:
        raise ValueError("values and targets must use the same device")
    if valid is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    elif (not isinstance(valid, Tensor) or valid.dtype != torch.bool
          or valid.shape != values.shape or valid.device != values.device):
        raise ValueError("valid must be boolean with the same shape and device as values")
    if not bool(valid.any()):
        raise ValueError("at least one valid joint sample is required")
    predictions, reference = values[valid], targets.detach()[valid]
    if not bool(torch.isfinite(predictions).all()) or not bool(torch.isfinite(reference).all()):
        raise ValueError("valid values and targets must be finite")
    compute_dtype = (torch.float32 if values.dtype in (torch.float16, torch.bfloat16)
                     else values.dtype)
    return F.huber_loss(predictions.to(compute_dtype), reference.to(compute_dtype),
                        reduction="mean", delta=float(delta))
