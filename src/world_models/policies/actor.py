"""Shared local actor weights with explicit, independently owned agent memories.

The tensor boundary accepts only uint8 RGB, executed previous actions, and local
memory. START resets a row before consuming its first image. No memory is stored
on the module, and no world model or agent identity enters the local network.

All local actor arithmetic is FP32, including under an outer world-model BF16
autocast context. Collection and PPO replay use different batch shapes; casting
only their final logits to FP32 cannot undo BF16 recurrent rounding differences.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from world_models.contracts import LocalActorInput, LocalActorOutput, validate_hidden_size
from .config import PolicyConfig


@contextmanager
def _actor_fp32(device: torch.device):
    """Use IEEE FP32 actor forwards and restore the surrounding precision.

    CUDA TF32 also rounds FP32 convolution/matmul inputs, so disable it within
    this scope. The pinned PyTorch exposes per-operation precision controls.
    These backend flags are process-wide; project workers own separate
    processes and do not concurrently execute different networks in threads.
    """
    settings = ()
    if device.type == "cuda":
        settings = tuple((backend, backend.fp32_precision) for backend in
                         (torch.backends.cuda.matmul, torch.backends.cudnn.conv))
    try:
        for backend, _ in settings:
            backend.fp32_precision = "ieee"
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        for backend, precision in settings:
            backend.fp32_precision = precision


def imagined_rgb_to_uint8(rgb: Tensor) -> Tensor:
    """Detach decoder RGB and round to the real actor's uint8 pixel interface.

    Any leading batch/agent/time axes are retained. Nonfinite or out-of-range
    decoder outputs are errors, rather than silently changing the observation.
    """
    if (not isinstance(rgb, Tensor) or rgb.ndim < 3 or tuple(rgb.shape[-3:]) != (3, 96, 128)
            or rgb.numel() == 0 or not rgb.is_floating_point()):
        raise ValueError("decoded RGB must be floating [...,3,96,128]")
    if not torch.isfinite(rgb).all().item() or torch.any((rgb < 0) | (rgb > 1)).item():
        raise ValueError("decoded RGB must be finite and in [0,1]")
    return rgb.detach().float().mul(255).round().to(torch.uint8)


@dataclass(frozen=True)
class ActorOutput:
    logits: Tensor
    next_local_hidden: Tensor

    @property
    def distribution(self) -> Categorical:
        """Categorical probability/log-probability/entropy arithmetic in FP32."""
        return Categorical(logits=self.logits.float())

    def sample(self, *, generator: torch.Generator) -> Tensor:
        """Draw only from the caller's agent-specific stream, never global RNG."""
        if not isinstance(generator, torch.Generator):
            raise ValueError("provide an explicit per-agent torch.Generator")
        device = torch.device(generator.device)
        if device.type != self.logits.device.type or (
                device.type == "cuda" and device.index not in (None, self.logits.device.index)):
            raise ValueError("actor generator must use the logits device")
        probabilities = self.distribution.probs
        return torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1,
                                 generator=generator).reshape(probabilities.shape[:-1])

    def log_prob(self, actions: Tensor) -> Tensor:
        if (not isinstance(actions, Tensor) or actions.dtype != torch.long
                or actions.shape != self.logits.shape[:-1] or actions.device != self.logits.device
                or torch.any((actions < 0) | (actions >= self.logits.shape[-1])).item()):
            raise ValueError("actions must be long executable indices matching the logits batch")
        return self.distribution.log_prob(actions)


@dataclass(frozen=True)
class ActorSequence:
    """Time-major outputs; padded logits are zero and must be masked in losses."""
    logits: Tensor
    local_hidden: Tensor
    final_local_hidden: Tensor
    valid: Tensor


class LocalActor(nn.Module):
    def __init__(self, config: PolicyConfig = PolicyConfig()):
        super().__init__()
        if not isinstance(config, PolicyConfig):
            raise ValueError("config must be a PolicyConfig")
        self.config = config
        blocks = []
        source = 3
        for target in config.encoder_channels:
            blocks.extend((nn.Conv2d(source, target, 4, stride=2, padding=1),
                           nn.GroupNorm(8, target), nn.SiLU()))
            source = target
        self.encoder = nn.Sequential(*blocks)
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(source * 6 * 8, config.embedding_size), nn.SiLU())
        self.gru = nn.GRUCell(config.embedding_size + config.action_count + 1, config.hidden_size)
        self.action_head = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.SiLU(),
                                         nn.Linear(config.hidden_size, config.action_count))

    def initial_hidden(self, batch_size: int) -> Tensor:
        """Fresh zero memory on the actor device; each row owns its storage."""
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return self.gru.weight_hh.new_zeros((batch_size, self.config.hidden_size))

    def _hidden(self, local_hidden: Tensor, batch: int) -> None:
        if (not isinstance(local_hidden, Tensor) or local_hidden.shape != (batch, self.config.hidden_size)
                or local_hidden.device != self.gru.weight_hh.device
                or not local_hidden.is_floating_point() or not torch.isfinite(local_hidden).all().item()):
            raise ValueError("local_hidden must be finite [B,hidden_size] on the actor device")

    def forward(self, local_rgb: Tensor, previous_action: Tensor, local_hidden: Tensor) -> ActorOutput:
        if (not isinstance(local_rgb, Tensor) or local_rgb.ndim != 4 or local_rgb.shape[0] <= 0
                or tuple(local_rgb.shape[1:]) != self.config.image_shape or local_rgb.dtype != torch.uint8
                or local_rgb.device != self.gru.weight_hh.device):
            raise ValueError("local_rgb must be uint8 [B,3,96,128] on the actor device")
        batch = local_rgb.shape[0]
        if (not isinstance(previous_action, Tensor) or previous_action.shape != (batch,)
                or previous_action.dtype != torch.long or previous_action.device != local_rgb.device
                or torch.any((previous_action < 0) | (previous_action > self.config.action_count)).item()):
            raise ValueError("previous_action must be long [B], executable 0..11 or START=12")
        self._hidden(local_hidden, batch)
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ValueError("local actor parameters must remain FP32; its forward disables autocast")
        with _actor_fp32(local_rgb.device):
            rgb = local_rgb.float().div(255)
            embedding = self.projection(self.encoder(rgb))
            previous = F.one_hot(previous_action, self.config.action_count + 1).float()
            hidden = local_hidden.float()
            hidden = torch.where((previous_action == self.config.action_count)[:, None],
                                 torch.zeros_like(hidden), hidden)
            next_hidden = self.gru(torch.cat((embedding, previous), dim=-1), hidden)
            return ActorOutput(self.action_head(next_hidden), next_hidden)

    def forward_sequence(self, local_rgb: Tensor, previous_action: Tensor, local_hidden: Tensor,
                         *, valid: Tensor | None = None) -> ActorSequence:
        """Replay local histories with BPTT; caller handles no-grad real prefixes.

        Shapes are [T,B,3,96,128], [T,B], [B,H], and optional bool [T,B].
        Padding is not encoded, does not consume START, and leaves memory intact.
        START on an active row blocks gradients to the preceding episode memory.
        """
        if (not isinstance(local_rgb, Tensor) or local_rgb.ndim != 5 or min(local_rgb.shape[:2]) <= 0
                or tuple(local_rgb.shape[2:]) != self.config.image_shape or local_rgb.dtype != torch.uint8
                or local_rgb.device != self.gru.weight_hh.device):
            raise ValueError("sequence RGB must be uint8 [T,B,3,96,128] on the actor device")
        shape = local_rgb.shape[:2]
        if (not isinstance(previous_action, Tensor) or previous_action.shape != shape
                or previous_action.dtype != torch.long or previous_action.device != local_rgb.device):
            raise ValueError("sequence previous_action must be long [T,B] on the actor device")
        self._hidden(local_hidden, shape[1])
        if valid is None:
            valid = torch.ones(shape, dtype=torch.bool, device=local_rgb.device)
        if (not isinstance(valid, Tensor) or valid.shape != shape or valid.dtype != torch.bool
                or valid.device != local_rgb.device):
            raise ValueError("valid must be bool [T,B] on the actor device")
        # Single-step forward returns FP32 memory, including under autocast.
        # Preserve that convention rather than rounding each step back to the
        # caller's initial (possibly bf16) dtype during recurrent replay.
        hidden = local_hidden.float()
        logits, memories = [], []
        for images, actions, active in zip(local_rgb, previous_action, valid):
            step_logits = self.gru.weight_hh.new_zeros((shape[1], self.config.action_count), dtype=torch.float32)
            if active.any().item():
                output = self(images[active], actions[active], hidden[active])
                step_logits = step_logits.index_copy(0, active.nonzero().flatten(), output.logits)
                hidden = hidden.index_copy(0, active.nonzero().flatten(), output.next_local_hidden.to(hidden.dtype))
            logits.append(step_logits)
            memories.append(hidden)
        return ActorSequence(torch.stack(logits), torch.stack(memories), hidden, valid.clone())

    @torch.no_grad()
    def step(self, inputs: LocalActorInput) -> LocalActorOutput:
        """Implement the existing simulator's strict, single-player contract."""
        if not isinstance(inputs, LocalActorInput):
            raise ValueError("step requires LocalActorInput with only local fields")
        hidden = validate_hidden_size(inputs.local_hidden, self.config.hidden_size)
        device = self.gru.weight_hh.device
        # Own the byte storage; no read-only buffer or simulator alias is retained.
        rgb = torch.frombuffer(bytearray(inputs.local_rgb.data), dtype=torch.uint8).reshape(1, *self.config.image_shape)
        output = self(rgb.to(device), torch.tensor([inputs.previous_action], dtype=torch.long, device=device),
                      torch.tensor([hidden], dtype=self.gru.weight_hh.dtype, device=device))
        return LocalActorOutput(tuple(output.distribution.probs[0].cpu().tolist()),
                                tuple(output.next_local_hidden[0].cpu().tolist()))
