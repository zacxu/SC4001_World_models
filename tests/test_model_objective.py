"""Analytical reductions, mask/gradient behavior, and state/transition alignment."""
from dataclasses import replace
from pathlib import Path
import math
import unittest

import numpy as np
import torch

from world_models.infrastructure.config import resolve_config
from world_models.models.heads import AuxiliaryTarget, TaskSpec, TransitionTargets, auxiliary_fields, declared_event_fields
from world_models.models.world_model import SamplingStreams, SharedWorldModel
from world_models.training.losses import LossTerm
from world_models.training.normalization import continuous_field_names, fit_field_statistics
from world_models.training.objective import (ObjectiveConfig, StateLossInput, TransitionLossInput,
                                            WorldModelObjective, merge_objective_results)
from world_models.training.objective_batch import objective_inputs


def normalization(family="relay", mean=0., scale=1.):
    samples = {name: (np.array([mean-scale, mean+scale]), np.array([True, True]))
               for name in continuous_field_names(family)}
    return fit_field_statistics([samples], family)


def auxiliary(leading, alignment, family="relay"):
    predictions, targets = {}, {}
    for field in auxiliary_fields(family):
        if field.alignment != alignment:
            continue
        predictions[field.name] = torch.zeros((*leading, field.classes) if field.kind == "categorical" else leading,
                                              requires_grad=True)
        targets[field.name] = AuxiliaryTarget(torch.zeros(leading, dtype=torch.long if field.kind == "categorical" else torch.float32),
                                              torch.zeros(leading, dtype=torch.bool))
    return predictions, targets


def inputs(batch=2, family="relay"):
    sp, st = auxiliary((1, batch), "state", family)
    ep, et = auxiliary((1, batch), "transition", family)
    states = StateLossInput(torch.zeros((1, batch, 2, 3, 96, 128), requires_grad=True), sp,
        torch.zeros((1, batch, 2, 3, 96, 128)), torch.ones((1, batch, 2), dtype=torch.bool), st,
        torch.ones((1, batch), dtype=torch.bool), torch.ones((1, batch), dtype=torch.bool))
    target = TransitionTargets(torch.zeros((1, batch)), torch.zeros((1, batch), dtype=torch.long),
                              torch.ones((1, batch), dtype=torch.bool), torch.ones((1, batch), dtype=torch.bool),
                              torch.full((1, batch), -0.001))
    transitions = TransitionLossInput(torch.zeros((1, batch), requires_grad=True), torch.zeros((1, batch, 3), requires_grad=True),
        ep, target, et, torch.zeros((1, batch, 32, 32), requires_grad=True), torch.zeros((1, batch, 32, 32), requires_grad=True),
        torch.ones((1, batch), dtype=torch.bool), torch.ones((1, batch), dtype=torch.long), torch.ones((1, batch), dtype=torch.bool))
    return states, transitions


def sequence_fixture(steps=2):
    from tests.test_model_heads import auxiliary_arrays
    state, event = auxiliary_arrays(batch=1)
    states = {name: value.unsqueeze(0).expand(steps+1, *value.shape).clone() for name, value in state.items()}
    states["rgb"] = torch.randint(0, 256, (steps+1, 1, 2, 3, 96, 128), dtype=torch.uint8,
                                  generator=torch.Generator().manual_seed(110))
    states["image_valid"] = torch.ones((steps+1, 1, 2), dtype=torch.bool)
    events = {name: value.unsqueeze(0).expand(steps, *value.shape).clone() for name, value in event.items()}
    events.update(team_reward=torch.full((steps, 1), -0.001, dtype=torch.float64),
                  termination_reason=torch.zeros((steps, 1), dtype=torch.uint8),
                  decision_index=torch.arange(steps).reshape(steps, 1),
                  actions=torch.zeros((steps, 1, 2), dtype=torch.uint8))
    # REASONS order is executable schema, not hard-coded enum assumptions.
    from world_models.data.schema import REASONS
    events["termination_reason"].fill_(REASONS.index("continue"))
    return states, events


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hand_computed_six_component_weighted_total(self):
        states, transitions = inputs()
        states.rgb = torch.ones_like(states.rgb, requires_grad=True)
        transitions.bonus = torch.full_like(transitions.bonus, .1, requires_grad=True)
        name = "player_0/health"
        states.auxiliary[name] = torch.full((1, 2), 2., requires_grad=True)
        states.targets[name] = AuxiliaryTarget(torch.zeros((1, 2)), torch.ones((1, 2), dtype=torch.bool))
        objective = WorldModelObjective(TaskSpec(), normalization())
        result = objective(states, transitions)
        expected = 100 + .005 + math.log(3) + 1.5 + .5 + .1
        self.assertAlmostEqual(float(result.total.detach()), expected, delta=1e-5)
        self.assertEqual(int(result.components["rgb"].count), 2*2*3*96*128)
        self.assertEqual(int(result.components["auxiliary"].count), 1)
        self.assertAlmostEqual(float(result.components["dynamics_kl"].mean.detach()), 1.)

    def test_auxiliary_fields_have_equal_weight_despite_unequal_support(self):
        states, _ = inputs(batch=4)
        a, b = "player_0/health", "player_1/health"
        states.auxiliary[a] = torch.ones((1, 4), requires_grad=True)
        states.auxiliary[b] = torch.full((1, 4), 3., requires_grad=True)
        states.targets[a] = AuxiliaryTarget(torch.zeros((1, 4)), torch.ones((1, 4), dtype=torch.bool))
        states.targets[b] = AuxiliaryTarget(torch.zeros((1, 4)), torch.tensor([[True, False, False, False]]))
        result = WorldModelObjective(TaskSpec(), normalization())(states, None)
        self.assertAlmostEqual(float(result.components["auxiliary"].mean.detach()), (0.5+2.5)/2)
        self.assertEqual(int(result.auxiliary_fields[b].count), 1)

    def test_normalize_targets_only_and_report_original_units(self):
        states, _ = inputs()
        name = "player_0/health"
        states.auxiliary[name] = torch.ones((1, 2), requires_grad=True)
        states.targets[name] = AuxiliaryTarget(torch.full((1, 2), 14.), torch.ones((1, 2), dtype=torch.bool))
        result = WorldModelObjective(TaskSpec(), normalization(mean=10, scale=2))(states, None)
        self.assertAlmostEqual(float(result.auxiliary_fields[name].mean.detach()), .5)
        self.assertAlmostEqual(float(result.diagnostics[f"aux_mae/{name}"].mean.detach()), 2.)
        result.total.backward()
        self.assertTrue(torch.all(states.auxiliary[name].grad < 0))

    def test_missing_required_images_skip_both_rgb_and_kl_keep_transition_labels(self):
        states, transitions = inputs()
        states.image_valid[0, 1, 1] = False
        states.observed[0, 1] = False
        states.rgb = states.rgb.detach().clone().requires_grad_()
        with torch.no_grad():
            states.rgb[0, 1] = float("nan")
        states.target_rgb[0, 1] = float("nan")
        transitions.posterior_valid[0, 1] = False
        with torch.no_grad():
            transitions.posterior_logits[0, 1] = float("nan")
        result = WorldModelObjective(TaskSpec(), normalization())(states, transitions)
        self.assertEqual(int(result.components["rgb"].count), 2*3*96*128)
        self.assertEqual(int(result.components["dynamics_kl"].count), 1)
        self.assertEqual(int(result.components["reward"].count), 2)
        self.assertEqual(int(result.components["outcome"].count), 2)
        result.total.backward()
        self.assertTrue(torch.isfinite(states.rgb.grad).all())
        self.assertEqual(float(states.rgb.grad[0, 1].abs().sum()), 0.)

    def test_all_masked_is_finite_connected_zero_without_free_nat_charge(self):
        states, transitions = inputs()
        states.valid.zero_()
        transitions.valid.zero_()
        with torch.no_grad():
            states.rgb.fill_(float("nan"))
            transitions.bonus.fill_(float("nan"))
            transitions.outcome_logits.fill_(float("nan"))
            transitions.prior_logits.fill_(float("nan"))
            transitions.posterior_logits.fill_(float("nan"))
        result = WorldModelObjective(TaskSpec(), normalization())(states, transitions)
        self.assertEqual(float(result.total.detach()), 0.)
        self.assertTrue(all(int(term.count) == 0 for term in result.components.values()))
        self.assertIsNone(result.metrics()["latent_entropy_nats_per_variable"]["posterior"])
        result.total.backward()
        self.assertTrue(torch.isfinite(states.rgb.grad).all())
        self.assertEqual(float(states.rgb.grad.abs().sum()), 0.)

    def test_bonus_and_outcome_masks_are_independent(self):
        _, transitions = inputs()
        transitions.targets = replace(transitions.targets, bonus_valid=torch.tensor([[True, False]]),
                                      outcome_valid=torch.tensor([[False, True]]))
        result = WorldModelObjective(TaskSpec(), normalization())(None, transitions)
        self.assertEqual(int(result.components["reward"].count), 1)
        self.assertEqual(int(result.components["outcome"].count), 1)
        self.assertEqual(int(result.diagnostics["total_reward_mae"].count), 0)

    def test_total_reward_expectation_handles_deadline_continue_mass(self):
        _, transitions = inputs()
        transitions.outcome_logits = torch.tensor([[[100., -100., -100.], [100., -100., -100.]]], requires_grad=True)
        transitions.next_decision = torch.tensor([[400, 1]])
        transitions.targets = replace(transitions.targets, total_reward=torch.tensor([[-1.001, -.001]]))
        result = WorldModelObjective(TaskSpec(), normalization())(None, transitions)
        self.assertAlmostEqual(float(result.diagnostics["total_reward_mae"].mean.detach()), 0., places=6)

    def test_configured_weights_and_task_profile_validation(self):
        root = Path(__file__).resolve().parents[1]
        config = resolve_config("relay", "B", "pilot", root / "configs").settings
        self.assertEqual(ObjectiveConfig.from_experiment(config), ObjectiveConfig())
        with self.assertRaises(ValueError):
            ObjectiveConfig(rgb=float("nan"))
        stats = normalization()
        with self.assertRaises(ValueError):
            WorldModelObjective(TaskSpec.for_family("cover"), stats)
        stats = replace(stats, provenance={"profile_task_spec": {"wrong": True}})
        with self.assertRaises(ValueError):
            WorldModelObjective(TaskSpec(), stats)

    def test_global_merge_uses_field_and_component_denominators(self):
        objective = WorldModelObjective(TaskSpec(), normalization())
        left, _ = inputs(batch=1)
        right, _ = inputs(batch=3)
        left.rgb = torch.ones_like(left.rgb, requires_grad=True)
        right.rgb = torch.zeros_like(right.rgb, requires_grad=True)
        a, b = "player_0/health", "player_1/health"
        left.auxiliary[a] = torch.full((1, 1), 3., requires_grad=True)
        right.auxiliary[a] = torch.ones((1, 3), requires_grad=True)
        left.targets[a] = AuxiliaryTarget(torch.zeros((1, 1)), torch.ones((1, 1), dtype=torch.bool))
        right.targets[a] = AuxiliaryTarget(torch.zeros((1, 3)), torch.ones((1, 3), dtype=torch.bool))
        right.targets[b] = AuxiliaryTarget(torch.zeros((1, 3)), torch.ones((1, 3), dtype=torch.bool))
        merged = merge_objective_results([objective(left, None), objective(right, None)])
        self.assertAlmostEqual(float(merged.components["rgb"].mean.detach()), .25)
        self.assertAlmostEqual(float(merged.auxiliary_fields[a].mean.detach()), (2.5+3*.5)/4)
        self.assertAlmostEqual(float(merged.components["auxiliary"].mean.detach()), .5)
        merged.total.backward()
        self.assertGreater(float(left.rgb.grad.abs().sum()), 0.)


class BatchIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def predictions(self, steps=2, *, missing_last=False, autocast=False):
        model = SharedWorldModel()
        streams = SamplingStreams.from_seed(41)
        states, events = sequence_fixture(steps)
        if missing_last:
            states["image_valid"][-1] = False
        predictions = []
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            state = model.initialize(states["rgb"][0], streams=streams)
            initial = model.reconstruct(state)
            for index in range(steps):
                step = model.observe_step(state, events["actions"][index].long(), states["rgb"][index+1],
                    streams=streams, next_image_valid=states["image_valid"][index+1])
                state = step.state
                predictions.append(step)
        return model, initial, predictions, states, events

    def test_sequence_initial_and_destinations_count_once_with_separate_axes(self):
        model, initial, predictions, states, events = self.predictions()
        state_input, transition_input = objective_inputs(initial, predictions, states, events, model.task)
        result = WorldModelObjective(model.task, normalization())(state_input, transition_input)
        self.assertEqual(int(result.components["rgb"].count), 3*2*3*96*128)
        self.assertEqual(int(result.components["reward"].count), 2)
        self.assertEqual(int(result.components["dynamics_kl"].count), 2)
        self.assertEqual(state_input.targets["player_0/health"].value.shape, (3, 1))
        self.assertEqual(transition_input.auxiliary_targets["event/plate_entries"].value.shape, (2, 1))

    def test_missing_images_keep_prior_reward_and_state_gradients(self):
        model, initial, predictions, states, events = self.predictions(steps=1, missing_last=True)
        state_input, transition_input = objective_inputs(initial, predictions, states, events, model.task)
        state_input.valid[0] = False  # Test terminal state supervision on its prior alone.
        result = WorldModelObjective(model.task, normalization())(state_input, transition_input)
        self.assertEqual(int(result.components["rgb"].count), 0)
        self.assertEqual(int(result.components["dynamics_kl"].count), 0)
        self.assertEqual(int(result.components["outcome"].count), 1)
        self.assertEqual(int(result.auxiliary_fields["player_0/health"].count), 1)
        result.total.backward()
        self.assertGreater(float(model.core.prior_head[-1].weight.grad.abs().sum()), 0.)

    def test_initial_only_needs_no_transition_reward_or_kl(self):
        model, initial, predictions, states, events = self.predictions(steps=0)
        state_input, transition_input = objective_inputs(initial, predictions, states, {}, model.task)
        self.assertIsNone(transition_input)
        result = WorldModelObjective(model.task, normalization())(state_input, None)
        self.assertEqual(int(result.components["rgb"].count), 2*3*96*128)
        self.assertEqual(int(result.components["reward"].count), 0)
        result.total.backward()
        self.assertGreater(float(model.core.encoder.cnn[0].weight.grad.abs().sum()), 0.)

    def test_padding_is_removed_before_target_decoding(self):
        model, initial, predictions, states, events = self.predictions()
        events["team_reward"][1] = float("nan")
        events["termination_reason"][1] = 255
        events["decision_index"][1] = 2**63-1
        state_mask, transition_mask = torch.tensor([[True], [True], [False]]), torch.tensor([[True], [False]])
        state_input, transition_input = objective_inputs(initial, predictions, states, events, model.task,
            state_loss_mask=state_mask, transition_loss_mask=transition_mask)
        result = WorldModelObjective(model.task, normalization())(state_input, transition_input)
        self.assertEqual(int(result.components["reward"].count), 1)
        self.assertEqual(int(result.components["rgb"].count), 2*2*3*96*128)

    def test_reject_misaligned_clock_posterior_and_next_view_masks(self):
        model, initial, predictions, states, events = self.predictions()
        events["decision_index"][0] = 5
        with self.assertRaisesRegex(ValueError, "clock"):
            objective_inputs(initial, predictions, states, events, model.task)
        events["decision_index"][0] = 0
        states["image_valid"][1, 0, 1] = False
        with self.assertRaisesRegex(ValueError, "posterior validity"):
            objective_inputs(initial, predictions, states, events, model.task)
        states["image_valid"][1] = True
        events["next_obs_valid"] = torch.zeros((2, 1, 2), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "next_obs_valid"):
            objective_inputs(initial, predictions, states, events, model.task)

    def test_objective_backprop_under_cpu_bfloat16(self):
        model, initial, predictions, states, events = self.predictions(autocast=True)
        state_input, transition_input = objective_inputs(initial, predictions, states, events, model.task)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = WorldModelObjective(model.task, normalization())(state_input, transition_input)
        self.assertEqual(result.total.dtype, torch.float32)
        result.total.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))


if __name__ == "__main__":
    unittest.main()
