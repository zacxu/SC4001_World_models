"""P2.3 explicitly reduced model objective, independent of optimizer or loader.

State predictions/labels and transition predictions/labels have separate axes.
All reductions keep sufficient statistics; padded labels are selected away
before any loss, normalization, or probability calculation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from world_models.models.heads import AuxiliaryTarget, TaskSpec, TransitionTargets, auxiliary_fields
from .losses import (LossTerm, balanced_categorical_kl, broadcast_mask, masked_bce_logits,
                     masked_cross_entropy, masked_huber, masked_mse, merge_terms)
from .normalization import NormalizationStats


COMPONENTS = ("rgb", "reward", "outcome", "auxiliary", "dynamics_kl", "representation_kl")


@dataclass(frozen=True)
class ObjectiveConfig:
    rgb: float = 100.0
    reward: float = 1.0
    outcome: float = 1.0
    auxiliary: float = 1.0
    dynamics_kl: float = 0.5
    representation_kl: float = 0.1
    free_nats: float = 1.0
    uniform_mixture: float = 0.01
    huber_delta: float = 1.0

    def __post_init__(self):
        for name in (*COMPONENTS, "free_nats", "uniform_mixture", "huber_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.uniform_mixture >= 1 or self.huber_delta <= 0:
            raise ValueError("mixture must be below one and Huber delta positive")

    @classmethod
    def from_experiment(cls, experiment: Mapping) -> "ObjectiveConfig":
        model = experiment["world_model"]
        if set(model["loss_weights"]) != set(COMPONENTS):
            raise ValueError("loss weight names differ from the P2.3 specification")
        return cls(**model["loss_weights"], free_nats=model["free_nats"], uniform_mixture=model["uniform_mixture"])


@dataclass
class StateLossInput:
    rgb: Tensor                         # [...,2,3,96,128], [0,1]
    auxiliary: dict[str, Tensor]         # state-aligned outputs
    target_rgb: Tensor                   # same shape, [0,1]
    image_valid: Tensor                  # [...,2]
    targets: dict[str, AuxiliaryTarget]  # state-aligned labels
    valid: Tensor                        # [...] loss-bearing state mask
    observed: Tensor                     # [...] both required images conditioned posterior


@dataclass
class TransitionLossInput:
    bonus: Tensor                        # [...] bounded bonus expectation
    outcome_logits: Tensor               # [...,3]
    auxiliary: dict[str, Tensor]          # transition-aligned event outputs
    targets: TransitionTargets
    auxiliary_targets: dict[str, AuxiliaryTarget]
    prior_logits: Tensor                 # [...,32,32]
    posterior_logits: Tensor             # [...,32,32]
    posterior_valid: Tensor               # [...] actual destination posterior available
    next_decision: Tensor                 # [...] source clock +1
    valid: Tensor                         # [...] loss-bearing transition mask


def _mask(mask: Tensor, shape: torch.Size | tuple, device: torch.device, name: str) -> Tensor:
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != shape or mask.device != device:
        raise ValueError(f"{name} must be boolean with exact sample shape {tuple(shape)} on the prediction device")
    return mask


def _empty(reference: Tensor) -> LossTerm:
    zero = reference.reshape(-1)[:0].float().sum()
    return LossTerm(zero, torch.zeros((), dtype=torch.long, device=zero.device))


def _field_average(terms: Mapping[str, LossTerm], reference: Tensor) -> LossTerm:
    nonempty = [term.mean for term in terms.values() if int(term.count) > 0]
    if not nonempty:
        return _empty(reference)
    return LossTerm(torch.stack(nonempty).sum(), torch.tensor(len(nonempty), dtype=torch.long, device=reference.device))


@dataclass
class ObjectiveResult:
    total: Tensor
    components: dict[str, LossTerm]
    auxiliary_fields: dict[str, LossTerm]
    diagnostics: dict[str, LossTerm]
    weights: ObjectiveConfig
    extra_metrics: dict = field(default_factory=dict)

    def metrics(self) -> dict:
        return {"total": float(self.total.detach()),
                "components": {name: term.detached() for name, term in self.components.items()},
                "weighted_components": {name: float((getattr(self.weights, name)*term.mean).detach())
                                        for name, term in self.components.items()},
                "auxiliary_fields": {name: term.detached() for name, term in self.auxiliary_fields.items()},
                "diagnostics": {name: term.detached() for name, term in self.diagnostics.items()},
                **self.extra_metrics}


def merge_objective_results(results: list[ObjectiveResult]) -> ObjectiveResult:
    """Exact global reduction for retained computation graphs across microbatches.

    This does not implement the memory-saving accumulation loop. P2.4 must use
    global component/field denominators when backpropagating microbatch sums.
    """
    if not results or any(result.weights != results[0].weights for result in results):
        raise ValueError("merge nonempty objective results with identical weights")
    config = results[0].weights
    fields = {name: merge_terms(result.auxiliary_fields[name] for result in results if name in result.auxiliary_fields)
              for name in sorted(set().union(*(result.auxiliary_fields for result in results)))}
    components = {name: merge_terms(result.components[name] for result in results) for name in COMPONENTS if name != "auxiliary"}
    components["auxiliary"] = _field_average(fields, results[0].total)
    diagnostic_names = set().union(*(result.diagnostics for result in results))
    diagnostics = {name: merge_terms(result.diagnostics[name] for result in results if name in result.diagnostics)
                   for name in diagnostic_names}
    total = sum(getattr(config, name)*components[name].mean for name in COMPONENTS)
    return ObjectiveResult(total, components, fields, diagnostics, config,
                           {"microbatches_merged": len(results), "classification_metrics": "recompute from globally collected predictions/labels"})


class WorldModelObjective(nn.Module):
    def __init__(self, task: TaskSpec, normalization: NormalizationStats, config: ObjectiveConfig = ObjectiveConfig()):
        super().__init__()
        if normalization.task_family != task.task_family:
            raise ValueError("normalization and task families differ")
        profiled_task = normalization.provenance.get("profile_task_spec")
        if profiled_task is not None and profiled_task != asdict(task):
            raise ValueError("normalization label profile and configured task constants differ")
        self.task, self.normalization, self.config = task, normalization, config
        self.fields = auxiliary_fields(task.task_family)

    def forward(self, states: StateLossInput | None, transitions: TransitionLossInput | None) -> ObjectiveResult:
        if states is None and transitions is None:
            raise ValueError("at least one state or transition input is required")
        reference = states.rgb if states is not None else transitions.bonus
        components = {name: _empty(reference) for name in COMPONENTS}
        field_terms: dict[str, LossTerm] = {}
        diagnostics: dict[str, LossTerm] = {}
        classification: dict = {}
        if states is not None:
            sample = states.rgb.shape[:-4]
            if states.rgb.shape[-4:] != (2, 3, 96, 128) or states.target_rgb.shape != states.rgb.shape:
                raise ValueError("state RGB prediction and target must be [...,2,3,96,128]")
            state_mask = _mask(states.valid, sample, states.rgb.device, "state validity")
            observed = _mask(states.observed, sample, states.rgb.device, "posterior observation validity")
            image_valid = _mask(states.image_valid, (*sample, 2), states.rgb.device, "image validity")
            # Required evidence is the synchronized pair. No next-image loss
            # on a row that fell back to the prior because either view is absent.
            rgb_valid = state_mask & observed & image_valid.all(-1)
            full_rgb_mask = broadcast_mask(rgb_valid, states.rgb.shape)
            for name, value in (("RGB prediction", states.rgb), ("RGB target", states.target_rgb)):
                if not value.is_floating_point() or torch.any((value[full_rgb_mask] < 0) | (value[full_rgb_mask] > 1)):
                    raise ValueError(f"valid {name} must be floating [0,1]")
            components["rgb"] = masked_mse(states.rgb, states.target_rgb, rgb_valid)
            self._auxiliary(states.auxiliary, states.targets, state_mask, "state", field_terms, diagnostics, classification)
        if transitions is not None:
            sample = transitions.bonus.shape
            mask = _mask(transitions.valid, sample, transitions.bonus.device, "transition validity")
            posterior = _mask(transitions.posterior_valid, sample, transitions.bonus.device, "posterior validity")
            targets = transitions.targets
            bonus_valid = _mask(targets.bonus_valid, sample, transitions.bonus.device, "bonus label validity") & mask
            outcome_valid = _mask(targets.outcome_valid, sample, transitions.bonus.device, "outcome label validity") & mask
            for value in (transitions.bonus[bonus_valid], targets.bonus[bonus_valid]):
                if torch.any((value < -1e-6) | (value > self.task.max_bonus+1e-6)):
                    raise ValueError("valid bonus prediction/label exceeds the configured task range")
            components["reward"] = masked_huber(transitions.bonus, targets.bonus, bonus_valid, delta=self.config.huber_delta)
            if transitions.outcome_logits.shape != (*sample, 3):
                raise ValueError("outcome logits must have exactly three classes")
            components["outcome"] = masked_cross_entropy(transitions.outcome_logits, targets.outcome, outcome_valid)
            kl = balanced_categorical_kl(transitions.posterior_logits, transitions.prior_logits, mask & posterior,
                                         free_nats=self.config.free_nats, uniform_mixture=self.config.uniform_mixture)
            components["dynamics_kl"], components["representation_kl"] = kl.dynamics, kl.representation
            diagnostics.update(raw_kl=kl.raw_kl, posterior_entropy=kl.posterior_entropy, prior_entropy=kl.prior_entropy)
            self._auxiliary(transitions.auxiliary, transitions.auxiliary_targets, mask, "transition", field_terms, diagnostics, classification)
            from .diagnostics import categorical_diagnostics
            classification["outcome"] = categorical_diagnostics(transitions.outcome_logits, targets.outcome, outcome_valid)
            diagnostics.update(self._reward_diagnostics(transitions, mask & bonus_valid & outcome_valid))
        components["auxiliary"] = _field_average(field_terms, reference)
        total = sum(getattr(self.config, name)*components[name].mean for name in COMPONENTS)
        extra = {"classification": classification,
                 "auxiliary_fields_with_support": int(components["auxiliary"].count),
                 "auxiliary_field_weight_rule": "equal mean of nonempty per-field means"}
        if "posterior_entropy" in diagnostics:
            support = int(diagnostics["raw_kl"].count)
            extra["latent_entropy_nats_per_variable"] = {
                "posterior": float(diagnostics["posterior_entropy"].mean.detach())/32 if support else None,
                "prior": float(diagnostics["prior_entropy"].mean.detach())/32 if support else None,
                "valid_samples": support}
        return ObjectiveResult(total, components, field_terms, diagnostics, self.config, extra)

    def _auxiliary(self, predictions, targets, mask, alignment, terms, diagnostics, classification):
        from .diagnostics import binary_diagnostics, categorical_diagnostics
        fields = [field for field in self.fields if field.alignment == alignment]
        expected = {field.name for field in fields}
        if set(predictions) != expected or set(targets) != expected:
            raise ValueError(f"{alignment} auxiliary predictions/targets must match the declared field schema")
        for field in fields:
            prediction, target = predictions[field.name], targets[field.name]
            valid = _mask(target.valid, mask.shape, mask.device, f"{field.name} validity") & mask
            if field.kind == "continuous":
                if prediction.shape != mask.shape or target.value.shape != mask.shape:
                    raise ValueError(f"{field.name} must have the sample shape")
                selected_prediction = prediction[valid]
                normalized = self.normalization.normalize(field.name, target.value[valid])
                terms[field.name] = masked_huber(selected_prediction, normalized, torch.ones_like(normalized, dtype=torch.bool),
                                                delta=self.config.huber_delta)
                if normalized.numel():
                    original = self.normalization.denormalize(field.name, selected_prediction)
                    errors = (original.float()-target.value[valid].float()).abs()
                    diagnostics[f"aux_mae/{field.name}"] = LossTerm(errors.sum(), valid.sum())
                else:
                    diagnostics[f"aux_mae/{field.name}"] = _empty(selected_prediction)
            elif field.kind == "binary":
                if torch.any((target.value[valid] != 0) & (target.value[valid] != 1)):
                    raise ValueError(f"binary label must be zero or one: {field.name}")
                terms[field.name] = masked_bce_logits(prediction, target.value, valid)
                classification[field.name] = binary_diagnostics(prediction, target.value, valid)
            else:
                if prediction.shape != (*mask.shape, field.classes):
                    raise ValueError(f"categorical head class count differs: {field.name}")
                terms[field.name] = masked_cross_entropy(prediction, target.value, valid)
                classification[field.name] = categorical_diagnostics(prediction, target.value, valid)

    def _reward_diagnostics(self, inputs: TransitionLossInput, valid: Tensor) -> dict[str, LossTerm]:
        # Deterministic expectation, so diagnostics do not consume outcome RNG.
        # A CONTINUE sample exactly at the deadline is resolved as failure.
        clock = inputs.next_decision
        if clock.shape != valid.shape or clock.dtype != torch.long or clock.device != valid.device:
            raise ValueError("next_decision must be a long tensor matching transitions")
        selected_clock = clock[valid]
        if torch.any((selected_clock < 1) | (selected_clock > self.task.deadline)):
            raise ValueError("valid transition clocks lie outside the exact task deadline")
        with torch.no_grad():
            probability = inputs.outcome_logits[valid].float().softmax(-1)
            bonus = inputs.bonus[valid].float()
            terminal = probability[..., 1]*self.task.success_reward + probability[..., 2]*self.task.failure_reward
            terminal += probability[..., 0]*(selected_clock == self.task.deadline)*self.task.failure_reward
            expected = bonus + self.task.step_cost + terminal
            truth = inputs.targets.total_reward[valid].float()
            difference = expected-truth
            if not torch.isfinite(difference).all():
                raise ValueError("nonfinite valid total-reward diagnostic")
            count = valid.sum()
            return {"total_reward_mae": LossTerm(difference.abs().sum(), count),
                    "total_reward_mse": LossTerm(difference.square().sum(), count),
                    "total_reward_bias": LossTerm(difference.sum(), count),
                    "bonus_mae": LossTerm((bonus-inputs.targets.bonus[valid]).abs().sum(), count)}
