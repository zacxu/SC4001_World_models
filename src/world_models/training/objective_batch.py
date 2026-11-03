"""Align P2.2 model outputs with P2.1 replay labels for the P2.3 objective.

This stacks already-computed predictions. It does not sample windows, run
burn-in, cache recurrent states, collect data, or implement an optimizer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from world_models.models.heads import (TaskSpec, TransitionTargets, auxiliary_fields, declared_event_fields,
                                      encode_auxiliary_targets, encode_transition_targets)
from world_models.models.world_model import ModelTransition, StatePrediction
from .losses import broadcast_mask
from .objective import StateLossInput, TransitionLossInput


def _tensor(value, device) -> Tensor:
    # Replay NumPy buffers are immutable. Never create a writable tensor view.
    return value.to(device=device) if isinstance(value, Tensor) else torch.tensor(value, device=device)


def _valid_mask(value, shape, device, name) -> Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    value = _tensor(value, device)
    if value.dtype != torch.bool or value.shape != shape:
        raise ValueError(f"{name} must be boolean with shape {shape}")
    return value


def _transition_targets(arrays: Mapping, valid: Tensor, task: TaskSpec) -> TransitionTargets:
    selected = {}
    for name in ("team_reward", "termination_reason", "decision_index", "terminated", "truncated"):
        if name not in arrays and name in ("terminated", "truncated"):
            continue
        value = _tensor(arrays[name], valid.device)
        if value.shape != valid.shape:
            raise ValueError(f"transition target {name} has incorrect shape")
        selected[name] = value[valid]
    parsed = encode_transition_targets(selected, task)
    def scatter(value):
        output = value.new_zeros(valid.shape)
        return output.masked_scatter(valid, value)
    return TransitionTargets(*(scatter(getattr(parsed, name)) for name in
                               ("bonus", "outcome", "bonus_valid", "outcome_valid", "total_reward")))


def objective_inputs(initial: StatePrediction, transitions: Sequence[ModelTransition],
                     state_arrays: Mapping, transition_arrays: Mapping, task: TaskSpec, *,
                     state_loss_mask: Tensor | None = None, transition_loss_mask: Tensor | None = None,
                     event_fields: Sequence[str] | None = None) -> tuple[StateLossInput, TransitionLossInput | None]:
    """Build loss inputs with S=T+1 states and T transitions, ordered [time,batch].

    Real-data state index t is o_t; transition index t is t->t+1. ``initial``
    predicts the first supplied state, which may be a reconstructed prefix
    anchor. Each destination image/state is supervised once. Loss masks can
    exclude padding/burn-in independently; no 64/16 sampling policy is imposed.
    """
    transitions = tuple(transitions)
    device, batch, steps = initial.rgb.device, initial.rgb.shape[0], len(transitions)
    state_shape, transition_shape = (steps+1, batch), (steps, batch)
    state_mask = _valid_mask(state_loss_mask, state_shape, device, "state loss mask")
    transition_mask = _valid_mask(transition_loss_mask, transition_shape, device, "transition loss mask")
    state_keys = ("rgb", "image_valid", "normalized_player_state", "privileged_validity_mask", "task_state",
                  "normalized_task_state", "task_validity_mask", "normalized_enemy_state", "enemy_validity_mask")
    raw_state = {name: _tensor(state_arrays[name], device) for name in state_keys}
    if raw_state["rgb"].shape != (*state_shape, 2, 3, 96, 128):
        raise ValueError("replay RGB must have shape [T+1,B,2,3,96,128]")
    rgb = raw_state["rgb"]
    if rgb.dtype == torch.uint8:
        rgb = rgb.float()/255
    elif rgb.is_floating_point():
        rgb = rgb.float()
    else:
        raise ValueError("replay RGB must be uint8 or floating [0,1]")
    image_valid = raw_state["image_valid"]
    if image_valid.shape != (*state_shape, 2) or image_valid.dtype != torch.bool:
        raise ValueError("replay image validity must be boolean [T+1,B,2]")
    for name in ("privileged_validity_mask", "task_validity_mask", "enemy_validity_mask"):
        value = raw_state[name]
        if value.dtype != torch.bool or value.shape[:2] != state_shape:
            raise ValueError(f"{name} must have state-leading axes and boolean dtype")
        raw_state[name] = value & broadcast_mask(state_mask, value.shape)
    events = tuple(declared_event_fields() if event_fields is None else event_fields)
    raw_transition = dict(transition_arrays)
    if not steps:
        raw_transition.setdefault("event_deltas", torch.empty((*transition_shape, len(events)), dtype=torch.long, device=device))
        raw_transition.setdefault("event_delta_validity_mask", torch.empty((*transition_shape, len(events)), dtype=torch.bool, device=device))
    for name in ("event_deltas", "event_delta_validity_mask"):
        value = _tensor(raw_transition[name], device)
        if value.shape != (*transition_shape, len(events)):
            raise ValueError(f"{name} must have shape [T,B,event_fields]")
        if name.endswith("validity_mask"):
            if value.dtype != torch.bool:
                raise ValueError("event validity must be boolean")
            value = value & broadcast_mask(transition_mask, value.shape)
        raw_transition[name] = value
    auxiliary = encode_auxiliary_targets(raw_state, raw_transition, task, events, device=device)
    fields = auxiliary_fields(task.task_family, events)
    state_names = [field.name for field in fields if field.alignment == "state"]
    event_names = [field.name for field in fields if field.alignment == "transition"]
    state_predictions = {name: torch.stack([initial.auxiliary[name], *(step.auxiliary[name] for step in transitions)])
                         for name in state_names}
    observed = torch.stack([torch.ones(batch, dtype=torch.bool, device=device), *(step.posterior_valid for step in transitions)])
    if steps:
        required_views = image_valid[1:].all(-1)
        if torch.any(observed[1:][transition_mask] != required_views[transition_mask]):
            raise ValueError("destination posterior validity disagrees with required replay images")
        if "next_obs_valid" in raw_transition:
            next_valid = _tensor(raw_transition["next_obs_valid"], device)
            if next_valid.dtype != torch.bool or next_valid.shape != image_valid[1:].shape:
                raise ValueError("next_obs_valid must be boolean [T,B,2]")
            if torch.any(next_valid[transition_mask] != image_valid[1:][transition_mask]):
                raise ValueError("transition next_obs_valid disagrees with destination state images")
    state_input = StateLossInput(torch.stack([initial.rgb, *(step.rgb for step in transitions)]), state_predictions,
                                 rgb, image_valid, {name: auxiliary[name] for name in state_names}, state_mask, observed)
    if not steps:
        return state_input, None
    target = _transition_targets(raw_transition, transition_mask, task)
    clocks = _tensor(raw_transition["decision_index"], device)
    if clocks.dtype != torch.long:
        raise ValueError("source decision indices must be long")
    predicted_clocks = torch.stack([step.state.decision_index for step in transitions])
    prior_clocks = torch.stack([step.prior.decision_index for step in transitions])
    if torch.any((predicted_clocks[transition_mask] != clocks[transition_mask]+1) |
                 (prior_clocks[transition_mask] != clocks[transition_mask]+1)):
        raise ValueError("model destination clock must be source transition clock +1")
    # Check that valid transitions really follow their preceding state clock.
    source_clocks = torch.stack([initial.state.decision_index, *(step.state.decision_index for step in transitions[:-1])])
    if torch.any(source_clocks[transition_mask] != clocks[transition_mask]):
        raise ValueError("replay source clock does not align with the predicted source state")
    next_clocks = torch.zeros_like(clocks)
    next_clocks[transition_mask] = clocks[transition_mask]+1
    transition_input = TransitionLossInput(
        torch.stack([step.predictions.bonus for step in transitions]),
        torch.stack([step.predictions.outcome_logits for step in transitions]),
        {name: torch.stack([step.auxiliary[name] for step in transitions]) for name in event_names},
        target, {name: auxiliary[name] for name in event_names},
        torch.stack([step.prior.logits for step in transitions]),
        torch.stack([step.state.logits for step in transitions]),
        torch.stack([step.posterior_valid for step in transitions]), next_clocks, transition_mask)
    return state_input, transition_input
