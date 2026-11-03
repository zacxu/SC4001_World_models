"""Independent numerical checks of global loss and gradient accumulation.

Uneven microbatches must reproduce a single full batch despite different
component masks and auxiliary supports. The fixture includes above-floor KL.
"""

from dataclasses import fields, is_dataclass
import unittest

import numpy as np
import torch

from world_models.models.heads import AuxiliaryTarget, TaskSpec, TransitionTargets, auxiliary_fields
from world_models.training.normalization import continuous_field_names, fit_field_statistics
from world_models.training.objective import (
    StateLossInput, TransitionLossInput, WorldModelObjective, merge_objective_results,
)


def _fixture():
    shape = (1, 4)
    state_predictions, state_targets, event_predictions, event_targets = {}, {}, {}, {}
    for field in auxiliary_fields("relay"):
        prediction = torch.zeros((*shape, field.classes) if field.kind == "categorical" else shape,
                                 requires_grad=True)
        target = AuxiliaryTarget(torch.zeros(shape, dtype=torch.long if field.kind == "categorical" else torch.float32),
                                 torch.zeros(shape, dtype=torch.bool))
        predictions = state_predictions if field.alignment == "state" else event_predictions
        targets = state_targets if field.alignment == "state" else event_targets
        predictions[field.name], targets[field.name] = prediction, target

    rgb = torch.zeros((1, 4, 2, 3, 96, 128), requires_grad=True)
    with torch.no_grad():
        rgb[0, 0] = 1.0
        rgb[0, 1] = 0.3
    state_predictions["player_0/health"] = torch.tensor([[3., 1., 1., 1.]], requires_grad=True)
    state_targets["player_0/health"] = AuxiliaryTarget(torch.zeros(shape), torch.ones(shape, dtype=torch.bool))
    state_targets["player_1/health"] = AuxiliaryTarget(torch.zeros(shape), torch.tensor([[False, True, True, True]]))
    states = StateLossInput(rgb, state_predictions, torch.zeros_like(rgb),
                            torch.ones((*shape, 2), dtype=torch.bool), state_targets,
                            torch.ones(shape, dtype=torch.bool), torch.ones(shape, dtype=torch.bool))

    bonus_valid = torch.tensor([[True, True, False, True]])
    outcome_valid = torch.tensor([[False, True, True, True]])
    targets = TransitionTargets(torch.zeros(shape), torch.zeros(shape, dtype=torch.long),
                                bonus_valid, outcome_valid, torch.full(shape, -0.001))
    q_logits = torch.zeros((*shape, 32, 32), requires_grad=True)
    p_logits = torch.zeros((*shape, 32, 32), requires_grad=True)
    with torch.no_grad():
        q_logits[0, 0, :, 0] = 4.0
        p_logits[0, 1, :, 1] = 3.0
    transitions = TransitionLossInput(
        torch.tensor([[.1, .02, .03, .04]], requires_grad=True),
        torch.tensor([[[0., 2., -1.], [1., 0., -1.], [-2., 1., 2.], [0., 0., 0.]]], requires_grad=True),
        event_predictions, targets, event_targets, p_logits, q_logits,
        torch.ones(shape, dtype=torch.bool), torch.ones(shape, dtype=torch.long), torch.ones(shape, dtype=torch.bool))
    samples = {name: (np.array([-1., 1.]), np.array([True, True]))
               for name in continuous_field_names("relay")}
    normalization = fit_field_statistics([samples], "relay")
    return states, transitions, WorldModelObjective(TaskSpec(), normalization)


def _slice_batch(value, left, right):
    """Build independent leaf tensors; preserve no original computation graph."""
    if isinstance(value, torch.Tensor):
        return value[:, left:right].detach().clone().requires_grad_(value.requires_grad)
    if isinstance(value, dict):
        return {name: _slice_batch(item, left, right) for name, item in value.items()}
    if is_dataclass(value):
        return type(value)(**{field.name: _slice_batch(getattr(value, field.name), left, right)
                              for field in fields(value)})
    raise TypeError(f"unsupported fixture value {type(value)}")


class IndependentAccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_uneven_microbatches_match_full_batch_total_and_six_gradient_paths(self):
        states, transitions, objective = _fixture()
        left_states, left_transitions = _slice_batch(states, 0, 1), _slice_batch(transitions, 0, 1)
        right_states, right_transitions = _slice_batch(states, 1, 4), _slice_batch(transitions, 1, 4)
        whole = objective(states, transitions)
        merged = merge_objective_results([objective(left_states, left_transitions),
                                          objective(right_states, right_transitions)])

        # Hand-computed support and loss values prevent a shared reducer bug
        # from passing solely because both code paths call the same objective.
        self.assertEqual(int(whole.components["rgb"].count), 4*2*3*96*128)
        self.assertAlmostEqual(float(whole.components["rgb"].mean.detach()), (1+.3**2)/4, places=6)
        self.assertEqual(int(whole.components["reward"].count), 3)
        self.assertEqual(int(whole.components["outcome"].count), 3)
        self.assertAlmostEqual(float(whole.components["reward"].mean.detach()), (.1**2+.02**2+.04**2)/6, places=8)
        self.assertEqual(int(whole.auxiliary_fields["player_0/health"].count), 4)
        self.assertEqual(int(whole.auxiliary_fields["player_1/health"].count), 3)
        self.assertAlmostEqual(float(whole.components["auxiliary"].mean.detach()), .5)
        self.assertGreater(float(whole.components["dynamics_kl"].mean.detach()), 1.)
        self.assertGreater(float(whole.components["representation_kl"].mean.detach()), 1.)
        torch.testing.assert_close(whole.total, merged.total, rtol=1e-6, atol=1e-6)
        for name in whole.components:
            self.assertEqual(int(whole.components[name].count), int(merged.components[name].count))
            torch.testing.assert_close(whole.components[name].mean, merged.components[name].mean, rtol=1e-6, atol=1e-6)

        whole.total.backward()
        merged.total.backward()
        paths = [
            ("rgb", states.rgb, left_states.rgb, right_states.rgb),
            ("bonus", transitions.bonus, left_transitions.bonus, right_transitions.bonus),
            ("outcome", transitions.outcome_logits, left_transitions.outcome_logits, right_transitions.outcome_logits),
            ("prior", transitions.prior_logits, left_transitions.prior_logits, right_transitions.prior_logits),
            ("posterior", transitions.posterior_logits, left_transitions.posterior_logits, right_transitions.posterior_logits),
            ("auxiliary", states.auxiliary["player_0/health"], left_states.auxiliary["player_0/health"], right_states.auxiliary["player_0/health"]),
        ]
        for name, full, left, right in paths:
            with self.subTest(path=name):
                self.assertIsNotNone(full.grad)
                self.assertTrue(torch.isfinite(full.grad).all())
                self.assertGreater(float(full.grad.abs().sum()), 0.)
                torch.testing.assert_close(full.grad, torch.cat([left.grad, right.grad], dim=1), rtol=1e-6, atol=1e-6)
        self.assertEqual(float(transitions.bonus.grad[0, 2]), 0.)
        self.assertEqual(float(transitions.outcome_logits.grad[0, 0].abs().sum()), 0.)

    def test_state_only_and_transition_only_parts_preserve_diagnostic_support(self):
        states, transitions, objective = _fixture()
        together = objective(states, transitions)
        merged = merge_objective_results([objective(states, None), objective(None, transitions)])
        torch.testing.assert_close(together.total, merged.total, rtol=1e-6, atol=1e-6)
        self.assertIn("aux_mae/player_0/health", merged.diagnostics)
        self.assertIn("raw_kl", merged.diagnostics)
        self.assertIn("total_reward_mae", merged.diagnostics)
        for name, expected in together.diagnostics.items():
            self.assertEqual(int(expected.count), int(merged.diagnostics[name].count))
            torch.testing.assert_close(expected.mean, merged.diagnostics[name].mean, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
