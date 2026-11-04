"""P2.4 recurrent fitting, exact accumulation, and isolated model diagnostics.

No simulator collection occurs here. Windows are real, episode-bounded replay;
the initial normalizer is frozen throughout initial and branch refresh fits.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np
import torch

from world_models.models.core import WorldState
from world_models.models.heads import declared_event_fields, encode_auxiliary_targets, encode_transition_targets
from world_models.models.world_model import SamplingStreams
from .objective import COMPONENTS, ObjectiveResult, WorldModelObjective
from .objective_batch import objective_inputs
from .sampling import ReplayWindow


@dataclass(frozen=True)
class FitSettings:
    effective_batch: int = 32
    microbatch: int = 4
    burn_in: int = 16
    loss_steps: int = 48
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 100.
    precision: str = "bfloat16"
    checkpoint_every: int = 250
    validate_every: int = 250
    validation_windows: int = 16
    log_every: int = 10

    def __post_init__(self):
        for name in ("effective_batch", "microbatch", "loss_steps", "checkpoint_every",
                     "validate_every", "validation_windows", "log_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.burn_in) is not int or self.burn_in < 0:
            raise ValueError("burn_in must be a nonnegative integer")
        if self.microbatch > self.effective_batch:
            raise ValueError("microbatch cannot exceed effective_batch")
        for name in ("learning_rate", "weight_decay", "gradient_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("invalid optimizer settings")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")

    @classmethod
    def from_experiment(cls, experiment, **overrides):
        cfg = experiment["world_model"]
        values = {name: cfg[name] for name in
                  ("effective_batch", "burn_in", "learning_rate", "weight_decay", "gradient_clip")}
        values["loss_steps"] = cfg["sequence_length"] - cfg["burn_in"]
        return cls(**(values | overrides))


def autocast_context(device, precision):
    return (torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
            if precision == "bfloat16" else nullcontext())


@dataclass
class LossSupport:
    components: dict[str, int]
    fields: dict[str, int]

    @classmethod
    def from_results(cls, results: Sequence[ObjectiveResult]):
        """Reference/test reducer; production support is calculated from labels."""
        components = {name: sum(int(r.components[name].count) for r in results)
                      for name in COMPONENTS if name != "auxiliary"}
        fields = {name: sum(int(r.auxiliary_fields[name].count) for r in results if name in r.auxiliary_fields)
                  for name in set().union(*(r.auxiliary_fields for r in results))}
        return cls(components, fields)


def loss_support(windows: Sequence[ReplayWindow], task, event_fields) -> LossSupport:
    """Global valid-label denominators, without model forwards or GPU graphs."""
    components = {name: 0 for name in COMPONENTS if name != "auxiliary"}
    fields = defaultdict(int)
    for window in windows:
        b = window.burn_in
        states = {key: torch.tensor(value[b:]) for key, value in window.states.items()}
        transitions = {key: torch.tensor(value[b:]) for key, value in window.transitions.items()
                       if value.dtype.kind not in ("U", "S", "O")}
        targets = encode_transition_targets(transitions, task)
        components["rgb"] += int(states["image_valid"].all(-1).sum()) * 2*3*96*128
        components["reward"] += int(targets.bonus_valid.sum())
        components["outcome"] += int(targets.outcome_valid.sum())
        kl_count = int(states["image_valid"][1:].all(-1).sum())
        components["dynamics_kl"] += kl_count
        components["representation_kl"] += kl_count
        auxiliary = encode_auxiliary_targets(states, transitions, task, event_fields, device="cpu")
        for name, target in auxiliary.items():
            fields[name] += int(target.valid.sum())
    return LossSupport(components, dict(fields))


def accumulation_loss(result: ObjectiveResult, support: LossSupport):
    """One microbatch contribution to the *global* objective, not mean of means."""
    loss = result.total * 0
    for name, count in support.components.items():
        if count:
            loss = loss + getattr(result.weights, name)*result.components[name].numerator/count
    supported = sum(count > 0 for count in support.fields.values())
    if supported:
        for name, term in result.auxiliary_fields.items():
            count = support.fields.get(name, 0)
            if count:
                loss = loss + result.weights.auxiliary*term.numerator/(count*supported)
    return loss


def microbatches(windows, size):
    """Batch equal-length windows; short episodes stay unpadded and isolated."""
    buckets = defaultdict(list)
    for window in windows:
        key = (window.burn_in, window.loss_steps)
        buckets[key].append(window)
        if len(buckets[key]) == size:
            yield buckets.pop(key)
    yield from buckets.values()


def _tensors(windows, kind, device, offset=0):
    sources = [getattr(window, kind) for window in windows]
    return {name: torch.tensor(np.stack([source[name][offset:] for source in sources], axis=1), device=device)
            for name in sources[0] if sources[0][name].dtype.kind not in ("U", "S", "O")}


def _condition(model, state, images, valid, actions, streams):
    """Burn-in without decoding heads; all context source images are valid."""
    prior = model.core.prior_step(state, actions.long(), generator=streams.prior)
    if not bool(valid.all()):
        raise ValueError("missing image inside burn-in context")
    return model.core.observe(prior, model._images(images), generator=streams.posterior)


def initial_window_state(model, images, decisions, streams):
    """Condition the reset memory at the *absolute* context clock.

    A mid-episode reset is the documented truncated-context approximation. The
    posterior progress embedding must see that clock before its latent draw.
    """
    batch = len(images)
    memory = model.core.reset_memory.unsqueeze(0).expand(batch, -1, -1)
    shape = (batch, model.config.latent_variables, model.config.latent_categories)
    blank = memory.new_zeros(shape)
    placeholder = blank.clone()
    placeholder[..., 0] = 1
    base = WorldState(memory, placeholder, blank, decisions.long())
    return model.core.observe(base, model._images(images), generator=streams.posterior)


def forward_windows(model, windows, streams, objective, precision="float32"):
    if not windows or len({(w.burn_in, w.loss_steps) for w in windows}) != 1:
        raise ValueError("forward batch needs equal context and loss lengths")
    device = next(model.parameters()).device
    burn = windows[0].burn_in
    states = _tensors(windows, "states", device)
    transitions = _tensors(windows, "transitions", device)
    if not bool(states["image_valid"][0].all()):
        raise ValueError("context initialization requires both real views")
    with autocast_context(device, precision):
        # Keep reset-memory gradients for a real reset/zero-context example.
        with torch.no_grad() if burn else nullcontext():
            state = initial_window_state(model, states["rgb"][0], transitions["decision_index"][0], streams)
            for t in range(burn):
                state = _condition(model, state, states["rgb"][t+1], states["image_valid"][t+1],
                                   transitions["actions"][t], streams)
        if burn:
            state = state.detach()
        initial = model.reconstruct(state)
        predictions = []
        for t in range(burn, len(transitions["actions"])):
            prediction = model.observe_step(state, transitions["actions"][t].long(), states["rgb"][t+1],
                streams=streams, next_image_valid=states["image_valid"][t+1])
            predictions.append(prediction)
            state = prediction.state
        state_input, transition_input = objective_inputs(initial, predictions,
            {name: value[burn:] for name, value in states.items()},
            {name: value[burn:] for name, value in transitions.items()}, model.task,
            event_fields=windows[0].event_fields or None)
        return objective(state_input, transition_input)


class MetricAccumulator:
    """Detach immediately so completed microbatch graphs can be released."""
    def __init__(self):
        self.components, self.fields, self.diagnostics = {}, {}, {}
        self.classification = {}
        self.config = None

    @staticmethod
    def _add(target, terms):
        for name, term in terms.items():
            row = target.setdefault(name, [0., 0])
            row[0] += float(term.numerator.detach())
            row[1] += int(term.count)

    def add(self, result):
        self.config = result.weights
        self._add(self.components, {k: v for k, v in result.components.items() if k != "auxiliary"})
        self._add(self.fields, result.auxiliary_fields)
        self._add(self.diagnostics, result.diagnostics)
        for name, row in result.extra_metrics.get("classification", {}).items():
            bucket = self.classification.setdefault(name, [])
            bucket.append(row)

    def metrics(self):
        def reduce(rows):
            return {name: {"numerator": value[0], "count": value[1],
                           "mean": value[0]/value[1] if value[1] else None}
                    for name, value in rows.items()}
        components, fields = reduce(self.components), reduce(self.fields)
        supported = [row["mean"] for row in fields.values() if row["count"]]
        aux = sum(supported)/len(supported) if supported else 0.
        components["auxiliary"] = {"mean": aux, "count": len(supported), "numerator": sum(supported)}
        total = sum(getattr(self.config, name)*(components[name]["mean"] or 0.) for name in COMPONENTS)
        classification = {}
        for name, rows in self.classification.items():
            support = sum(row["support"] for row in rows)
            if "confusion_matrix" in rows[0]:
                matrix = np.sum([row["confusion_matrix"] for row in rows], axis=0)
                classification[name] = {"support": support, "confusion_matrix": matrix.tolist(),
                    "target_counts": matrix.sum(axis=1).tolist(), "predicted_counts": matrix.sum(axis=0).tolist(),
                    "accuracy": float(np.trace(matrix))/support if support else None}
            else:
                positives = sum(row["positives"] for row in rows)
                predicted = sum(row["predicted_positives"] for row in rows)
                tp = sum(round((row["precision"] or 0.)*row["predicted_positives"]) for row in rows)
                classification[name] = {"support": support, "positives": positives,
                    "negatives": support-positives, "predicted_positives": predicted,
                    "positive_fraction": positives/support if support else None,
                    "mean_probability": sum((row["mean_probability"] or 0.)*row["support"] for row in rows)/support if support else None,
                    "brier": sum((row["brier"] or 0.)*row["support"] for row in rows)/support if support else None,
                    "precision": tp/predicted if predicted else None, "recall": tp/positives if positives else None,
                    "f1": 2*tp/(predicted+positives) if predicted+positives else None}
        return {"total": total, "components": components, "auxiliary_fields": fields,
                "diagnostics": reduce(self.diagnostics), "classification": classification,
                "latent_entropy_nats_per_variable": {name: (self.diagnostics[name][0]/self.diagnostics[name][1]/32
                    if self.diagnostics.get(name, [0, 0])[1] else None) for name in ("posterior_entropy", "prior_entropy")}}


def optimizer_update(model, optimizer, windows, streams, objective, settings):
    model.train()
    if len(windows) != settings.effective_batch:
        raise ValueError("every update must contain the configured effective sequence batch")
    event_fields = windows[0].event_fields or declared_event_fields()
    if any((window.event_fields or declared_event_fields()) != event_fields for window in windows):
        raise ValueError("effective batch contains incompatible event columns")
    support = loss_support(windows, model.task, event_fields)
    metrics = MetricAccumulator()
    optimizer.zero_grad(set_to_none=True)
    for batch in microbatches(windows, settings.microbatch):
        result = forward_windows(model, batch, streams, objective, settings.precision)
        contribution = accumulation_loss(result, support)
        if not bool(torch.isfinite(contribution)):
            raise FloatingPointError("nonfinite effective-batch objective; optimizer not advanced")
        contribution.backward()
        metrics.add(result)
        del result, contribution
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = metrics.metrics()
    output.update(gradient_norm_before_clip=float(norm), sequences=len(windows),
                  loss_transitions=sum(window.loss_steps for window in windows),
                  burn_in_transitions=sum(window.burn_in for window in windows))
    return output


def evaluate_windows(model, windows, objective, *, seed, microbatch=1, precision="float32"):
    """Fixed diagnostic bank; independent RNG and no changes to model/optimizer."""
    if not windows:
        raise ValueError("evaluation needs nonempty held-out windows")
    was_training = model.training
    streams = SamplingStreams.from_seed(seed, next(model.parameters()).device)
    metrics = MetricAccumulator()
    try:
        model.eval()
        with torch.no_grad():
            for batch in microbatches(windows, microbatch):
                result = forward_windows(model, batch, streams, objective, precision)
                metrics.add(result)
        return metrics.metrics()
    finally:
        model.train(was_training)


def compare_burn_in(model, bank, windows, objective, *, seed, contexts=(16, 32, 64),
                    microbatch=1, precision="float32"):
    """Same held-out anchors/targets, varied context, plus full-prefix reference."""
    results = {}
    for context in (*contexts, "full_prefix"):
        selected = [bank.window(w.dataset_index, w.episode_id, w.anchor,
                    burn_in=0 if context == "full_prefix" else context, loss_steps=w.loss_steps,
                    full_prefix=context == "full_prefix") for w in windows]
        results[str(context)] = evaluate_windows(model, selected, objective, seed=seed,
                                                 microbatch=microbatch, precision=precision)
    return {"anchors": [{"episode_id": w.episode_id, "anchor": w.anchor, "loss_steps": w.loss_steps}
                         for w in windows], "context_results": results,
            "selection": "validation only; no automatic protocol change", "optimizer_updates": 0}
