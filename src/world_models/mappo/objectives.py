"""Raw team GAE and individual-ratio PPO objectives in FP32."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from world_models.rollouts.schema import ImaginedRollout


def _number(value: float, name: str, low: float, high: float | None = None,
            *, strict_low: bool = False, strict_high: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (value <= low if strict_low else value < low):
        raise ValueError(f"{name} is below its allowed range")
    if high is not None and (value >= high if strict_high else value > high):
        raise ValueError(f"{name} is above its allowed range")


@dataclass(frozen=True)
class GAETargets:
    raw_advantages: Tensor
    returns: Tensor
    actor_advantages: Tensor


@torch.inference_mode(False)
@torch.no_grad()
def compute_gae(rollout: ImaginedRollout, *, gamma: float, gae_lambda: float,
                normalize_advantages: bool = True) -> GAETargets:
    """Compute one team target per joint sample, with zero padded targets.

    A true terminal removes the final value bootstrap. An alive imagination
    cutoff keeps that bootstrap in its TD residual but ends the recursion.
    Normalize only a separate actor copy over all valid joint samples.
    """
    _number(gamma, "gamma", 0, 1)
    _number(gae_lambda, "gae_lambda", 0, 1)
    if type(normalize_advantages) is not bool:
        raise ValueError("normalize_advantages must be boolean")
    if not isinstance(rollout, ImaginedRollout):
        raise ValueError("GAE requires a detached ImaginedRollout")
    rollout.validate()
    valid = rollout.valid
    raw = torch.zeros_like(rollout.rewards, dtype=torch.float32)
    carry = torch.zeros_like(raw[0])
    # Values belong to the frozen behavior critic, including the T+1 state.
    values = rollout.values.detach().float()
    rewards = rollout.rewards.detach().float()
    for step in reversed(range(rollout.steps)):
        active = valid[step]
        bootstrap = torch.where(rollout.terminated[step], 0., values[step + 1])
        delta = rewards[step] + float(gamma) * bootstrap - values[step]
        recurse = active & ~rollout.terminated[step] & ~rollout.cutoff[step]
        carry = torch.where(active, delta + float(gamma * gae_lambda)
                            * torch.where(recurse, carry, 0.), 0.)
        raw[step] = carry
    returns = torch.zeros_like(raw)
    returns[valid] = raw[valid] + values[:-1][valid]
    actor = raw.clone()
    if normalize_advantages:
        selected = raw[valid]
        actor[valid] = (selected - selected.mean()) / selected.std(correction=0).clamp_min(1e-8)
    if any(not bool(torch.isfinite(value).all()) for value in (raw, returns, actor)):
        raise ValueError("GAE produced nonfinite targets; diagnose numerical divergence")
    return GAETargets(raw, returns, actor)


@dataclass(frozen=True)
class ActorObjective:
    loss: Tensor
    policy_loss: Tensor
    entropy: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor
    ratios: Tensor


def actor_objective(logits: Tensor, actions: Tensor, old_log_probs: Tensor,
                    advantages: Tensor, valid: Tensor, *, clip_epsilon: float,
                    entropy_coefficient: float) -> ActorObjective:
    """Average individual player decisions with the same raw/normalized team A.

    Select valid entries before any probability arithmetic, so arbitrary padded
    inputs cannot poison the loss or its gradients. Stored behavior likelihoods
    and advantages are fixed targets. Probability and diagnostic arithmetic use
    FP32 even when the actor forward pass used bfloat16 autocast.
    """
    _number(clip_epsilon, "clip_epsilon", 0, 1, strict_low=True, strict_high=True)
    _number(entropy_coefficient, "entropy_coefficient", 0)
    if (not isinstance(logits, Tensor) or not logits.is_floating_point()
            or logits.ndim != 4 or min(logits.shape[:2]) <= 0
            or logits.shape[2] != 2 or logits.shape[3] < 2):
        raise ValueError("logits must be floating point [T,B,2,A] with T,B>0 and A>=2")
    shape = logits.shape[:2]
    if (not isinstance(valid, Tensor) or valid.dtype != torch.bool
            or valid.shape != shape or valid.device != logits.device):
        raise ValueError("valid must be boolean [T,B] on the logits device")
    if not bool(valid.any()):
        raise ValueError("at least one valid joint sample is required")
    for name, value, expected, integer in (
        ("actions", actions, logits.shape[:3], True),
        ("old_log_probs", old_log_probs, logits.shape[:3], False),
        ("advantages", advantages, shape, False),
    ):
        if (not isinstance(value, Tensor) or value.shape != expected
                or value.device != logits.device
                or (value.dtype != torch.long if integer else not value.is_floating_point())):
            raise ValueError(f"{name} has invalid shape, dtype or device")
    selected_logits = logits[valid].float()
    selected_actions = actions[valid]
    old = old_log_probs.detach()[valid].float()
    team_advantage = advantages.detach()[valid].float().unsqueeze(-1)
    if any(not bool(torch.isfinite(value).all()) for value in (selected_logits, old, team_advantage)):
        raise ValueError("valid actor inputs must be finite")
    if bool(((selected_actions < 0) | (selected_actions >= logits.shape[-1])).any()):
        raise ValueError("valid actions must be executable categorical indices")
    if bool((old > 1e-5).any()):
        raise ValueError("old_log_probs cannot exceed log probability zero")
    log_distribution = selected_logits.log_softmax(-1)
    log_probability = log_distribution.gather(-1, selected_actions.unsqueeze(-1)).squeeze(-1)
    log_ratio = log_probability - old
    ratio = log_ratio.exp()
    if not bool(torch.isfinite(log_distribution).all()) or not bool(torch.isfinite(ratio).all()):
        raise ValueError("nonfinite policy likelihood ratio; diagnose numerical divergence")
    unclipped = ratio * team_advantage
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * team_advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    entropy = -(log_distribution.exp() * log_distribution).sum(-1).mean()
    loss = policy_loss - float(entropy_coefficient) * entropy
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite PPO objective; diagnose numerical divergence")
    with torch.no_grad():
        # Nonnegative k3 estimator of KL(old||current) under old-policy actions.
        approx_kl = ((ratio.detach() - 1) - log_ratio.detach()).mean().clamp_min(0)
        clip_fraction = ((ratio.detach() - 1).abs() > clip_epsilon).float().mean()
        ratios = torch.zeros(logits.shape[:3], device=logits.device, dtype=torch.float32)
        ratios[valid] = ratio.detach()
        if not bool(torch.isfinite(approx_kl)):
            raise ValueError("nonfinite approximate KL; diagnose numerical divergence")
    return ActorObjective(loss, policy_loss, entropy, approx_kl, clip_fraction, ratios)


@torch.inference_mode(False)
@torch.no_grad()
def explained_variance(values: Tensor, targets: Tensor, *, valid: Tensor | None = None) -> Tensor:
    """Detached population explained variance, NaN when target variance is zero."""
    if (not isinstance(values, Tensor) or not values.is_floating_point()
            or not isinstance(targets, Tensor) or not targets.is_floating_point()
            or values.ndim < 1 or values.numel() == 0
            or values.shape != targets.shape or values.device != targets.device):
        raise ValueError("values and targets require matching nonempty floating point shapes/devices")
    if valid is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    if (not isinstance(valid, Tensor) or valid.dtype != torch.bool
            or valid.shape != values.shape or valid.device != values.device or not bool(valid.any())):
        raise ValueError("valid must select at least one matching joint sample")
    prediction, target = values.detach()[valid].float(), targets.detach()[valid].float()
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("valid predictions and targets must be finite")
    variance = target.var(correction=0)
    if variance.item() == 0:
        return torch.full((), float("nan"), device=values.device, dtype=torch.float32)
    result = 1 - (target - prediction).var(correction=0) / variance
    if not bool(torch.isfinite(result)):
        raise ValueError("nonfinite explained variance; diagnose numerical divergence")
    return result
