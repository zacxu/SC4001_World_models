"""Hand-calculated team GAE and individual-decision clipped PPO regression."""
from copy import deepcopy
from dataclasses import replace
import math
import unittest

import torch

from tests.test_imagined_rollout_schema import make_rollout
from world_models.mappo.objectives import actor_objective, compute_gae, explained_variance


def arithmetic_rollout():
    rollout = make_rollout()
    rewards = torch.zeros_like(rollout.rewards)
    rewards[1, 0] = 1
    metadata = {**rollout.metadata, "task_rule": {**rollout.metadata["task_rule"], "step_cost": 0.}}
    return replace(rollout, rewards=rewards, metadata=metadata)


class MAPPOObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_hand_calculated_multistep_terminal_and_alive_cutoff(self):
        rollout = arithmetic_rollout()
        result = compute_gae(rollout, gamma=.5, gae_lambda=.5, normalize_advantages=False)
        # First row deltas: 0,-3; last terminal ignores destination value 6.
        # Second row deltas: -.5,-1.5,-1.5; last cutoff bootstraps value 11.
        expected = torch.tensor([[-.75, -.96875], [-3., -1.875], [0., -1.5]])
        expected_returns = torch.tensor([[1.25, 2.03125], [1., 3.125], [0., 5.5]])
        torch.testing.assert_close(result.raw_advantages, expected, rtol=0, atol=0)
        torch.testing.assert_close(result.returns, expected_returns, rtol=0, atol=0)
        torch.testing.assert_close(result.actor_advantages, expected, rtol=0, atol=0)
        self.assertNotEqual(result.actor_advantages.data_ptr(), result.raw_advantages.data_ptr())
        self.assertEqual(rollout.values[2, 0].item(), 6)

    def test_raw_returns_precede_global_joint_sample_advantage_normalization(self):
        rollout = arithmetic_rollout()
        raw = compute_gae(rollout, gamma=.5, gae_lambda=.5, normalize_advantages=False)
        normalized = compute_gae(rollout, gamma=.5, gae_lambda=.5)
        torch.testing.assert_close(normalized.returns, raw.returns, rtol=0, atol=0)
        torch.testing.assert_close(normalized.raw_advantages, raw.raw_advantages, rtol=0, atol=0)
        # Five joint samples, not six padded rows and not a per-trajectory norm.
        selected = normalized.actor_advantages[rollout.valid]
        self.assertAlmostEqual(selected.mean().item(), 0., places=6)
        self.assertAlmostEqual(selected.std(correction=0).item(), 1., places=6)
        mean = -1.61875
        variance = sum((value - mean) ** 2 for value in (-.75, -.96875, -3., -1.875, -1.5)) / 5
        expected = (raw.raw_advantages[rollout.valid] - mean) / math.sqrt(variance)
        torch.testing.assert_close(selected, expected)
        self.assertEqual(normalized.actor_advantages[2, 0].item(), 0.)

    def test_lambda_zero_and_discount_zero_isolate_td_residual(self):
        rollout = arithmetic_rollout()
        one_step = compute_gae(rollout, gamma=.5, gae_lambda=0, normalize_advantages=False)
        torch.testing.assert_close(one_step.raw_advantages,
                                   torch.tensor([[0., -.5], [-3., -1.5], [0., -1.5]]))
        immediate = compute_gae(rollout, gamma=0, gae_lambda=1, normalize_advantages=False)
        torch.testing.assert_close(immediate.returns, rollout.rewards)

    def test_constant_advantages_are_zero_after_normalizing(self):
        rollout = arithmetic_rollout()
        values = torch.zeros_like(rollout.values)
        values[:-1][rollout.valid] = rollout.rewards[rollout.valid]
        result = compute_gae(replace(rollout, values=values), gamma=0, gae_lambda=0)
        self.assertTrue(torch.equal(result.actor_advantages, torch.zeros_like(result.actor_advantages)))
        self.assertTrue(torch.isfinite(result.actor_advantages).all())

    def test_single_joint_sample_normalization_is_finite_and_preserves_raw_return(self):
        rollout = arithmetic_rollout()
        payload = rollout.tensor_payload()
        for name, value in payload.items():
            if name == "world":
                payload[name] = {key: tensor[1:3, :1].clone() for key, tensor in value.items()}
            else:
                payload[name] = value[1:3 if value.shape[0] == 4 else 2, :1].clone()
        metadata = deepcopy(rollout.metadata)
        metadata["horizon"] = 1
        metadata["anchor_references"] = [metadata["anchor_references"][0]]
        metadata["anchor_references"][0]["decision_index"] = 2
        single = type(rollout).from_tensor_payload(payload, metadata=metadata,
                                                  streams_before={}, streams_after={})
        result = compute_gae(single, gamma=.99, gae_lambda=.95)
        self.assertEqual(result.raw_advantages.item(), -3.)
        self.assertEqual(result.returns.item(), 1.)
        self.assertEqual(result.actor_advantages.item(), 0.)

    def test_targets_own_storage_are_detached_fp32_and_ordinary_under_inference(self):
        rollout = arithmetic_rollout()
        before = rollout.values.clone()
        with torch.inference_mode():
            result = compute_gae(rollout, gamma=.99, gae_lambda=.95)
        for tensor in (result.raw_advantages, result.returns, result.actor_advantages):
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertFalse(tensor.requires_grad)
            self.assertFalse(tensor.is_inference())
        result.returns[0, 0] = 999
        self.assertTrue(torch.equal(before, rollout.values))

    def test_gae_rejects_invalid_coefficients_or_mutated_rollout(self):
        rollout = arithmetic_rollout()
        for kwargs in ({"gamma": float("nan"), "gae_lambda": .9},
                       {"gamma": True, "gae_lambda": .9},
                       {"gamma": .99, "gae_lambda": 1.1},
                       {"gamma": .99, "gae_lambda": .95, "normalize_advantages": 1}):
            with self.assertRaises(ValueError):
                compute_gae(rollout, **kwargs)
        rollout.values[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            compute_gae(rollout, gamma=.99, gae_lambda=.95)

    def objective(self, logits, actions, old, advantages, valid, **kwargs):
        return actor_objective(logits, actions, old, advantages, valid,
                               clip_epsilon=kwargs.get("clip_epsilon", .2),
                               entropy_coefficient=kwargs.get("entropy_coefficient", 0.))

    def test_individual_ratios_are_averaged_never_multiplied(self):
        logits = torch.tensor([[[[.75, .25], [.25, .75]]]]).log().requires_grad_()
        actions = torch.zeros(1, 1, 2, dtype=torch.long)
        old = torch.full((1, 1, 2), math.log(.5), requires_grad=True)
        advantages = torch.tensor([[2.]], requires_grad=True)
        result = self.objective(logits, actions, old, advantages, torch.ones(1, 1, dtype=torch.bool))
        torch.testing.assert_close(result.ratios, torch.tensor([[[1.5, .5]]]))
        # mean(min(1.5*2,1.2*2), min(.5*2,.8*2)) = 1.7.
        self.assertAlmostEqual(result.policy_loss.item(), -1.7, places=6)
        self.assertNotAlmostEqual(result.policy_loss.item(), -(1.5 * .5 * 2))
        self.assertEqual(result.clip_fraction.item(), 1.)
        expected_kl = ((1.5 - 1 - math.log(1.5)) + (.5 - 1 - math.log(.5))) / 2
        self.assertAlmostEqual(result.approx_kl.item(), expected_kl, places=6)
        result.loss.backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(advantages.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_negative_team_advantage_clips_the_correct_surrogate_side(self):
        probabilities = torch.tensor([[[[.75, .25], [.25, .75]]]])
        result = self.objective(probabilities.log(), torch.zeros(1, 1, 2, dtype=torch.long),
                                torch.full((1, 1, 2), math.log(.5)), torch.tensor([[-2.]]),
                                torch.ones(1, 1, dtype=torch.bool))
        # min(-3,-2.4), min(-1,-1.6), followed by negative mean.
        self.assertAlmostEqual(result.policy_loss.item(), 2.3, places=6)

    def test_same_behavior_policy_has_unit_ratios_and_entropy_regularization(self):
        logits = torch.tensor([[[[0., 0.], [0., 0.]]]], requires_grad=True)
        result = self.objective(logits, torch.zeros(1, 1, 2, dtype=torch.long),
                                torch.full((1, 1, 2), -math.log(2)), torch.tensor([[3.]]),
                                torch.ones(1, 1, dtype=torch.bool), entropy_coefficient=.01)
        self.assertTrue(torch.equal(result.ratios, torch.ones_like(result.ratios)))
        self.assertEqual(result.approx_kl.item(), 0.)
        self.assertEqual(result.clip_fraction.item(), 0.)
        self.assertAlmostEqual(result.entropy.item(), math.log(2), places=6)
        self.assertAlmostEqual(result.loss.item(), -3 - .01 * math.log(2), places=6)

    def test_nan_padding_is_excluded_before_arithmetic_and_has_zero_gradient(self):
        valid = torch.tensor([[True, False], [True, True]])
        logits = torch.zeros(2, 2, 2, 3)
        logits[~valid] = float("nan")
        logits.requires_grad_()
        old = torch.full((2, 2, 2), -math.log(3))
        old[~valid] = float("nan")
        advantages = torch.tensor([[1., float("nan")], [2., 3.]])
        actions = torch.zeros(2, 2, 2, dtype=torch.long)
        actions[~valid] = -100
        result = self.objective(logits, actions, old, advantages, valid)
        self.assertAlmostEqual(result.loss.item(), -2., places=6)
        result.loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[~valid].abs().sum().item(), 0.)
        self.assertTrue(torch.equal(result.ratios[~valid], torch.zeros(1, 2)))

    def test_bfloat16_logits_use_fp32_probability_and_objective_arithmetic(self):
        logits = torch.zeros(1, 1, 2, 12, dtype=torch.bfloat16, requires_grad=True)
        result = self.objective(logits, torch.zeros(1, 1, 2, dtype=torch.long),
                                torch.full((1, 1, 2), -math.log(12)), torch.tensor([[1.]]),
                                torch.ones(1, 1, dtype=torch.bool))
        for name in ("loss", "policy_loss", "entropy", "approx_kl", "clip_fraction", "ratios"):
            self.assertEqual(getattr(result, name).dtype, torch.float32)
        result.loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bad_inputs_and_overflow_fail_instead_of_changing_optimizer_schedule(self):
        logits = torch.zeros(1, 1, 2, 2)
        actions = torch.zeros(1, 1, 2, dtype=torch.long)
        old = torch.full((1, 1, 2), -math.log(2))
        advantage = torch.ones(1, 1)
        valid = torch.ones(1, 1, dtype=torch.bool)
        cases = ((logits, actions, old, advantage, ~valid),
                 (logits, actions.float(), old, advantage, valid),
                 (logits, actions + 2, old, advantage, valid),
                 (logits, actions, -old, advantage, valid),
                 (logits, actions, torch.full_like(old, -1000), advantage, valid),
                 (logits, actions, torch.full_like(old, float("nan")), advantage, valid),
                 (logits, actions, torch.full_like(old, -.5), advantage.repeat(1, 2), valid))
        for args in cases:
            with self.assertRaises(ValueError):
                self.objective(*args)

    def test_explained_variance_is_joint_sample_masked_and_detached(self):
        targets = torch.tensor([[1., 2.], [3., float("nan")]], requires_grad=True)
        valid = torch.tensor([[True, True], [True, False]])
        self.assertEqual(explained_variance(targets, targets, valid=valid).item(), 1.)
        constant = torch.zeros_like(targets, requires_grad=True)
        result = explained_variance(constant, targets, valid=valid)
        self.assertEqual(result.item(), 0.)
        self.assertFalse(result.requires_grad)
        self.assertTrue(torch.isnan(explained_variance(torch.ones(3), torch.ones(3))))
        with self.assertRaises(ValueError):
            explained_variance(targets, targets)


if __name__ == "__main__":
    unittest.main()
