"""Mask-safe loss numerators and counts for the P2.3 world-model objective.

Every primitive selects valid entries *before* arithmetic, so NaN padding never
becomes a target or a gradient. Keeping numerator/count pairs permits exact
global reductions across time, uneven batches, and later gradient accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from world_models.models.core import categorical_probs


@dataclass(frozen=True)
class LossTerm:
    numerator: Tensor
    count: Tensor

    def __post_init__(self) -> None:
        if (not isinstance(self.numerator, Tensor) or self.numerator.ndim != 0
                or not self.numerator.is_floating_point()):
            raise ValueError("loss numerator must be a scalar floating-point tensor")
        if not torch.isfinite(self.numerator).item():
            raise ValueError("loss numerator must be finite")
        if (not isinstance(self.count, Tensor) or self.count.ndim != 0
                or self.count.dtype != torch.int64 or self.count.device != self.numerator.device
                or self.count.item() < 0):
            raise ValueError("loss count must be a nonnegative int64 scalar on the numerator device")

    @property
    def mean(self) -> Tensor:
        """Empty terms reduce to a graph-connected zero rather than NaN."""
        return self.numerator / self.count.clamp_min(1).to(self.numerator.dtype)

    def detached(self) -> dict[str, float | int]:
        return {"numerator": float(self.numerator.detach()), "count": int(self.count.detach()),
                "mean": float(self.mean.detach())}


def merge_terms(terms: Iterable[LossTerm]) -> LossTerm:
    """Merge sufficient statistics; never average already reduced means."""
    terms = tuple(terms)
    if not terms:
        raise ValueError("at least one LossTerm is required")
    if any(not isinstance(term, LossTerm) for term in terms):
        raise ValueError("only LossTerm values can be merged")
    if len({term.numerator.device for term in terms}) != 1:
        raise ValueError("merged terms must use one device")
    return LossTerm(torch.stack([term.numerator.float() for term in terms]).sum(),
                    torch.stack([term.count for term in terms]).sum())


def broadcast_mask(valid: Tensor, shape: tuple[int, ...] | torch.Size) -> Tensor:
    """Broadcast boolean validity to value shape.

    A leading-axis mask such as [T,B,V] for [T,B,V,C,H,W] gets appended
    singleton axes. Explicitly broadcastable masks are also supported.
    """
    if not isinstance(valid, Tensor) or valid.dtype != torch.bool:
        raise ValueError("validity must be a boolean tensor")
    shape = tuple(shape)
    if valid.ndim <= len(shape) and tuple(valid.shape) == shape[:valid.ndim]:
        valid = valid.reshape(*valid.shape, *((1,) * (len(shape)-valid.ndim))) if shape else valid
    try:
        return torch.broadcast_to(valid, shape)
    except RuntimeError as error:
        raise ValueError(f"validity shape {tuple(valid.shape)} cannot broadcast to {shape}") from error


def _prediction(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")


def _selected_pair(prediction: Tensor, target: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    _prediction(prediction, "prediction")
    if (not isinstance(target, Tensor) or target.is_complex() or target.shape != prediction.shape
            or target.device != prediction.device):
        raise ValueError("prediction and real-valued target must have identical shapes and devices")
    if not isinstance(valid, Tensor) or valid.device != prediction.device:
        raise ValueError("validity must be on the prediction device")
    mask = broadcast_mask(valid, prediction.shape)
    selected_prediction, selected_target = prediction[mask].float(), target[mask].float()
    if not torch.isfinite(selected_prediction).all() or not torch.isfinite(selected_target).all():
        raise ValueError("valid predictions and targets must be finite in float32")
    return selected_prediction, selected_target


def _term(values: Tensor) -> LossTerm:
    return LossTerm(values.float().sum(), torch.tensor(values.numel(), dtype=torch.int64, device=values.device))


def masked_mse(prediction: Tensor, target: Tensor, valid: Tensor) -> LossTerm:
    prediction, target = _selected_pair(prediction, target, valid)
    return _term((prediction-target).square())


def masked_huber(prediction: Tensor, target: Tensor, valid: Tensor, *, delta: float = 1.0) -> LossTerm:
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("Huber delta must be finite and positive")
    prediction, target = _selected_pair(prediction, target, valid)
    return _term(F.huber_loss(prediction, target, reduction="none", delta=delta))


def masked_bce_logits(prediction: Tensor, target: Tensor, valid: Tensor) -> LossTerm:
    prediction, target = _selected_pair(prediction, target, valid)
    if torch.any((target < 0) | (target > 1)):
        raise ValueError("valid BCE targets must lie in [0,1]")
    return _term(F.binary_cross_entropy_with_logits(prediction, target, reduction="none"))


def masked_cross_entropy(logits: Tensor, target: Tensor, valid: Tensor) -> LossTerm:
    _prediction(logits, "categorical logits")
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("categorical logits need a class axis with at least two classes")
    if (not isinstance(target, Tensor) or target.dtype != torch.long or target.shape != logits.shape[:-1]
            or target.device != logits.device):
        raise ValueError("categorical targets must be long tensors matching the sample shape and device")
    if not isinstance(valid, Tensor) or valid.device != logits.device:
        raise ValueError("validity must be on the logits device")
    mask = broadcast_mask(valid, target.shape)
    selected_logits, selected_target = logits[mask].float(), target[mask]
    if not torch.isfinite(selected_logits).all():
        raise ValueError("valid categorical logits must be finite in float32")
    if torch.any((selected_target < 0) | (selected_target >= logits.shape[-1])):
        raise ValueError("valid categorical target lies outside the declared classes")
    return _term(F.cross_entropy(selected_logits.reshape(-1, logits.shape[-1]), selected_target.reshape(-1),
                                 reduction="none"))


@dataclass(frozen=True)
class KLTerms:
    dynamics: LossTerm
    representation: LossTerm
    raw_kl: LossTerm
    posterior_entropy: LossTerm
    prior_entropy: LossTerm


def _probability_and_log(logits: Tensor, uniform_mixture: float) -> tuple[Tensor, Tensor]:
    probabilities = categorical_probs(logits, uniform_mixture)
    # Without a uniform mixture, log_softmax preserves finite log-probabilities
    # even when a very small softmax probability underflows to zero.
    logarithms = probabilities.log() if uniform_mixture > 0 else F.log_softmax(logits.float(), dim=-1)
    return probabilities, logarithms


def balanced_categorical_kl(posterior_logits: Tensor, prior_logits: Tensor, valid: Tensor, *,
                            free_nats: float = 1.0, uniform_mixture: float = 0.01) -> KLTerms:
    """KL summed over 32 variables, then floored per sample, then averaged.

    Dynamics gradients reach p only; representation gradients reach q only.
    Raw KL and summed entropies are unfloored diagnostics. Empty selections
    produce connected zero terms and do not charge a free-nat constant.
    """
    _prediction(posterior_logits, "posterior logits")
    _prediction(prior_logits, "prior logits")
    if (posterior_logits.shape != prior_logits.shape or posterior_logits.device != prior_logits.device
            or posterior_logits.ndim < 2 or posterior_logits.shape[-2:] != (32, 32)):
        raise ValueError("prior/posterior logits must have matching [...,32,32] shape and device")
    if isinstance(free_nats, bool) or not isinstance(free_nats, (int, float)) or not math.isfinite(free_nats) or free_nats < 0:
        raise ValueError("free_nats must be finite and nonnegative")
    if (not isinstance(valid, Tensor) or valid.dtype != torch.bool
            or valid.shape != posterior_logits.shape[:-2] or valid.device != prior_logits.device):
        raise ValueError("KL validity must be boolean and match the exact sample shape/device")
    q_logits, p_logits = posterior_logits[valid].float(), prior_logits[valid].float()
    if not torch.isfinite(q_logits).all() or not torch.isfinite(p_logits).all():
        raise ValueError("valid categorical logits must be finite in float32")
    # The helper validates mixture settings even for an empty selection.
    q, log_q = _probability_and_log(q_logits, uniform_mixture)
    p, log_p = _probability_and_log(p_logits, uniform_mixture)
    summed_axes = (-1, -2)
    dynamics_kl = (q.detach() * (log_q.detach()-log_p)).sum(dim=summed_axes)
    representation_kl = (q * (log_q-log_p.detach())).sum(dim=summed_axes)
    raw_kl = (q * (log_q-log_p)).sum(dim=summed_axes)
    q_entropy = -(q * log_q).sum(dim=summed_axes)
    p_entropy = -(p * log_p).sum(dim=summed_axes)
    return KLTerms(_term(dynamics_kl.clamp_min(free_nats)),
                   _term(representation_kl.clamp_min(free_nats)), _term(raw_kl),
                   _term(q_entropy), _term(p_entropy))
