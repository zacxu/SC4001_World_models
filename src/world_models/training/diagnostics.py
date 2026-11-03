"""Read-only gradient and class diagnostics for P2.3 loss inspection.

These diagnostics neither change weights/gradients nor alter the objective's
class weights. Null metrics represent empty or undefined measurements, never
invented quality scores for absent events.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .losses import broadcast_mask


def _finite_norm(values: Tensor) -> float:
    """Scaled float64 L2 reduction avoids float32 square/accumulation overflow."""
    values = values.detach()
    if values.is_complex():
        magnitudes = values.to(dtype=torch.complex128).abs()
    else:
        magnitudes = values.to(dtype=torch.float64).abs()
    if magnitudes.numel() == 0:
        return 0.0
    scale = float(magnitudes.max())
    if scale == 0.0:
        return 0.0
    return scale * math.sqrt(float((magnitudes / scale).square().sum()))


@torch.no_grad()
def gradient_diagnostics(model: nn.Module, *, error_if_nonfinite: bool = True) -> dict:
    """Inspect gradients after backward and before any clipping or zeroing.

    Parameter counts refer to parameter tensors, rather than scalar entries.
    ``nonfinite_count`` counts individual nonfinite gradient entries. Top-level
    module names group nested parameters; a group without gradients has null
    norm. With strict checking disabled, invalid norms are null JSON values.
    """
    if not isinstance(model, nn.Module) or type(error_if_nonfinite) is not bool:
        raise ValueError("model must be an nn.Module and error_if_nonfinite must be boolean")
    global_l2 = 0.0
    groups: dict[str, float] = {}
    group_seen: set[str] = set()
    group_invalid: set[str] = set()
    with_grad = without_grad = nonfinite_count = 0
    nonfinite_parameters = []
    norm_overflow = False
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        groups.setdefault(group, 0.0)
        gradient = parameter.grad
        if gradient is None:
            without_grad += 1
            continue
        with_grad += 1
        group_seen.add(group)
        # Coalescing a detached sparse view does not mutate the stored gradient.
        values = gradient.detach().coalesce().values() if gradient.is_sparse else gradient.detach()
        invalid = int((~torch.isfinite(values)).sum())
        if invalid:
            nonfinite_count += invalid
            nonfinite_parameters.append(name)
            group_invalid.add(group)
            continue
        norm = _finite_norm(values)
        if not math.isfinite(norm):
            norm_overflow = True
            group_invalid.add(group)
            continue
        groups[group] = math.hypot(groups[group], norm)
        global_l2 = math.hypot(global_l2, norm)
        if not math.isfinite(groups[group]) or not math.isfinite(global_l2):
            norm_overflow = True
            if not math.isfinite(groups[group]):
                group_invalid.add(group)
    if error_if_nonfinite and (nonfinite_count or norm_overflow):
        raise FloatingPointError(f"invalid gradient diagnostics: {nonfinite_count} nonfinite entries; norm overflow={norm_overflow}")
    return {
        "global_l2": None if nonfinite_count or norm_overflow else global_l2,
        "by_module_l2": {name: None if name not in group_seen or name in group_invalid else norm
                         for name, norm in sorted(groups.items())},
        "parameters_with_grad": with_grad,
        "parameters_without_grad": without_grad,
        "nonfinite_count": nonfinite_count,
        "nonfinite_parameters": nonfinite_parameters,
        "norm_overflow": norm_overflow,
    }


def _selected_binary(logits: Tensor, targets: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise ValueError("binary logits must be floating-point tensors")
    if (not isinstance(targets, Tensor) or targets.is_complex() or targets.shape != logits.shape
            or targets.device != logits.device):
        raise ValueError("binary targets must have matching shape and device")
    if not isinstance(valid, Tensor) or valid.device != logits.device:
        raise ValueError("validity must be a boolean tensor on the logits device")
    mask = broadcast_mask(valid, logits.shape)
    selected_logits, selected_targets = logits[mask].float(), targets[mask].float()
    if not torch.isfinite(selected_logits).all() or not torch.isfinite(selected_targets).all():
        raise ValueError("valid binary logits and targets must be finite in float32")
    if torch.any((selected_targets != 0) & (selected_targets != 1)):
        raise ValueError("binary diagnostic targets must be zero or one")
    return selected_logits, selected_targets


@torch.no_grad()
def binary_diagnostics(logits: Tensor, targets: Tensor, valid: Tensor) -> dict:
    """Describe event frequency and unweighted probability predictions.

    The decision threshold is probability >= 0.5. Precision/recall are null
    when their denominator is zero; all frequency/error means are null for an
    empty valid selection.
    """
    logits, targets = _selected_binary(logits, targets, valid)
    probabilities = logits.sigmoid()
    support = targets.numel()
    positives = int((targets == 1).sum())
    negatives = support - positives
    predicted = probabilities >= 0.5
    predicted_positives = int(predicted.sum())
    true_positives = int((predicted & (targets == 1)).sum())
    return {
        "support": support,
        "positives": positives,
        "negatives": negatives,
        "positive_fraction": positives / support if support else None,
        "mean_probability": float(probabilities.mean()) if support else None,
        "predicted_positives": predicted_positives,
        "brier": float((probabilities-targets).square().mean()) if support else None,
        "precision": true_positives / predicted_positives if predicted_positives else None,
        "recall": true_positives / positives if positives else None,
    }


@torch.no_grad()
def categorical_diagnostics(logits: Tensor, targets: Tensor, valid: Tensor) -> dict:
    """Counts, accuracy, and confusion matrix with target rows/prediction columns."""
    if (not isinstance(logits, Tensor) or not logits.is_floating_point()
            or logits.ndim < 1 or logits.shape[-1] < 2):
        raise ValueError("categorical logits need a floating-point class axis")
    if (not isinstance(targets, Tensor) or targets.dtype != torch.long or targets.shape != logits.shape[:-1]
            or targets.device != logits.device):
        raise ValueError("categorical targets must be long and match the sample shape/device")
    if not isinstance(valid, Tensor) or valid.device != logits.device:
        raise ValueError("validity must be a boolean tensor on the logits device")
    mask = broadcast_mask(valid, targets.shape)
    selected_logits, selected_targets = logits[mask].float(), targets[mask]
    classes = logits.shape[-1]
    if not torch.isfinite(selected_logits).all():
        raise ValueError("valid categorical logits must be finite in float32")
    if torch.any((selected_targets < 0) | (selected_targets >= classes)):
        raise ValueError("valid categorical targets must lie inside the declared classes")
    predicted = selected_logits.argmax(dim=-1)
    support = selected_targets.numel()
    confusion = torch.bincount(selected_targets.reshape(-1)*classes + predicted.reshape(-1),
                               minlength=classes*classes).reshape(classes, classes)
    return {
        "support": support,
        "target_counts": torch.bincount(selected_targets.reshape(-1), minlength=classes).tolist(),
        "predicted_counts": torch.bincount(predicted.reshape(-1), minlength=classes).tolist(),
        "accuracy": float((predicted == selected_targets).float().mean()) if support else None,
        "confusion_matrix": confusion.tolist(),
    }
