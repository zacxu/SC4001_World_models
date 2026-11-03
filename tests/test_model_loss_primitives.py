"""Analytic reductions, masking, and gradient routing for the P2.3 objective."""

import math
import unittest

import torch
from torch.nn import functional as F

from world_models.models.core import categorical_probs
from world_models.training.losses import (
    LossTerm, balanced_categorical_kl, broadcast_mask, masked_bce_logits,
    masked_cross_entropy, masked_huber, masked_mse, merge_terms,
)


class LossPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mse_global_element_reduction_and_view_mask(self):
        prediction = torch.tensor([[[[1., 3.]], [[100., 100.]]], [[[2., 4.]], [[0., 0.]]]], requires_grad=True)
        target = torch.zeros_like(prediction)
        valid = torch.tensor([[True, False], [True, True]])
        result = masked_mse(prediction, target, valid)
        self.assertEqual(result.count.item(), 6)
        self.assertAlmostEqual(result.mean.item(), (1+9+4+16)/6)
        result.mean.backward()
        self.assertTrue(torch.equal(prediction.grad[0, 1], torch.zeros_like(prediction.grad[0, 1])))

    def test_empty_nan_padding_is_zero_with_zero_gradients(self):
        for primitive in (masked_mse, masked_huber, masked_bce_logits):
            with self.subTest(primitive=primitive.__name__):
                prediction = torch.full((2, 3), float("nan"), requires_grad=True)
                target = torch.full_like(prediction, float("nan"))
                result = primitive(prediction, target, torch.zeros(2, dtype=torch.bool))
                self.assertEqual(result.count.item(), 0)
                self.assertEqual(result.mean.item(), 0.)
                result.mean.backward()
                self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))

    def test_mixed_valid_nan_slots_do_not_contaminate_forward_or_backward(self):
        prediction = torch.tensor([2., float("nan"), 3.], requires_grad=True)
        target = torch.tensor([1., float("nan"), 5.])
        result = masked_huber(prediction, target, torch.tensor([True, False, True]))
        self.assertAlmostEqual(result.mean.item(), 1.0)
        result.mean.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([0.5, 0., -0.5]))

    def test_huber_delta_has_huber_scale_not_smooth_l1_scale(self):
        prediction = torch.tensor([0., 1., 5.])
        target = torch.zeros(3)
        result = masked_huber(prediction, target, torch.ones(3, dtype=torch.bool), delta=2.)
        self.assertAlmostEqual(result.mean.item(), (0.+0.5+8.)/3, places=6)
        self.assertNotEqual(result.mean.item(), F.smooth_l1_loss(prediction, target, beta=2.).item())

    def test_cross_entropy_ignores_invalid_classes_and_nan_logits(self):
        logits = torch.tensor([[0., 0., 0.], [float("nan")]*3], requires_grad=True)
        target = torch.tensor([1, -999])
        result = masked_cross_entropy(logits, target, torch.tensor([True, False]))
        self.assertEqual(result.count.item(), 1)
        self.assertAlmostEqual(result.mean.item(), math.log(3), places=6)
        result.mean.backward()
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros(3)))

    def test_empty_cross_entropy_has_connected_zero(self):
        logits = torch.full((2, 3), float("nan"), requires_grad=True)
        result = masked_cross_entropy(logits, torch.tensor([-1, 9]), torch.zeros(2, dtype=torch.bool))
        result.mean.backward()
        self.assertEqual(result.mean.item(), 0.)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_bce_logits_matches_analytic_balanced_labels(self):
        prediction = torch.zeros(3, requires_grad=True)
        result = masked_bce_logits(prediction, torch.tensor([0., 1., float("nan")]), torch.tensor([True, True, False]))
        self.assertAlmostEqual(result.mean.item(), math.log(2), places=6)
        result.mean.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([0.25, -0.25, 0.]))

    def test_merge_unequal_batches_reduces_global_counts(self):
        left = masked_mse(torch.tensor([2.]), torch.zeros(1), torch.ones(1, dtype=torch.bool))
        right = masked_mse(torch.zeros(3), torch.zeros(3), torch.ones(3, dtype=torch.bool))
        merged = merge_terms([left, right])
        self.assertEqual(merged.detached(), {"numerator": 4., "count": 4, "mean": 1.})
        self.assertNotEqual(merged.mean.item(), (left.mean+right.mean).item()/2)
        with self.assertRaises(ValueError):
            merge_terms([])

    def test_valid_bad_values_and_shapes_are_rejected(self):
        for primitive in (masked_mse, masked_huber, masked_bce_logits):
            with self.subTest(primitive=primitive.__name__), self.assertRaises(ValueError):
                primitive(torch.tensor([float("nan")]), torch.zeros(1), torch.ones(1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            masked_mse(torch.zeros(2), torch.zeros(2), torch.ones(2))
        with self.assertRaises(ValueError):
            masked_bce_logits(torch.zeros(1), torch.tensor([2.]), torch.ones(1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            masked_cross_entropy(torch.zeros(1, 3), torch.tensor([3]), torch.ones(1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            masked_cross_entropy(torch.zeros(1, 3), torch.tensor([1.]), torch.ones(1, dtype=torch.bool))
        for delta in (-1, 0, float("nan"), True):
            with self.assertRaises(ValueError):
                masked_huber(torch.zeros(1), torch.zeros(1), torch.ones(1, dtype=torch.bool), delta=delta)

    def test_prefix_and_explicit_mask_broadcasts(self):
        valid = torch.tensor([[True, False]])
        mask = broadcast_mask(valid, (1, 2, 3, 4, 5))
        self.assertEqual(mask.shape, (1, 2, 3, 4, 5))
        self.assertEqual(mask.sum().item(), 60)
        explicit = broadcast_mask(valid[:, :, None, None, None], mask.shape)
        self.assertTrue(torch.equal(mask, explicit))
        with self.assertRaises(ValueError):
            broadcast_mask(torch.ones(7, dtype=torch.bool), (2, 3))

    def test_bfloat16_inputs_have_float32_reductions(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            for primitive in (masked_mse, masked_huber, masked_bce_logits):
                value = primitive(torch.zeros(3, dtype=torch.bfloat16), torch.zeros(3, dtype=torch.bfloat16),
                                  torch.ones(3, dtype=torch.bool))
                self.assertEqual(value.numerator.dtype, torch.float32)
                self.assertEqual(value.mean.dtype, torch.float32)


class BalancedKLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _logits(self):
        q = torch.zeros(2, 32, 32)
        q[:, :, 0] = 0.4
        p = torch.zeros_like(q)
        return q.requires_grad_(), p.requires_grad_()

    def test_free_nats_sum_variables_before_floor(self):
        q, p = self._logits()
        terms = balanced_categorical_kl(q, p, torch.ones(2, dtype=torch.bool))
        probabilities_q, probabilities_p = categorical_probs(q), categorical_probs(p)
        per_variable = (probabilities_q*(probabilities_q.log()-probabilities_p.log())).sum(-1)
        expected = per_variable.sum(-1).clamp_min(1).mean()
        self.assertLess(per_variable.max().item(), 1.)
        self.assertAlmostEqual(terms.dynamics.mean.item(), expected.item(), places=6)
        self.assertEqual(terms.dynamics.mean.item(), 1.)
        self.assertNotEqual(terms.dynamics.mean.item(), per_variable.clamp_min(1).sum(-1).mean().item())
        self.assertEqual(terms.raw_kl.count.item(), 2)
        self.assertLess(terms.raw_kl.mean.item(), 1.)

    def test_kl_above_floor_is_averaged_over_samples_after_floor(self):
        q = torch.zeros(2, 32, 32)
        q[1, :, 0] = 5.
        p = torch.zeros_like(q)
        terms = balanced_categorical_kl(q, p, torch.ones(2, dtype=torch.bool))
        qp, pp = categorical_probs(q), categorical_probs(p)
        per_sample = (qp*(qp.log()-pp.log())).sum((-1, -2))
        self.assertGreater(per_sample[1].item(), 1.)
        self.assertAlmostEqual(terms.representation.mean.item(), per_sample.clamp_min(1.).mean().item(), places=5)
        self.assertGreater(terms.representation.mean.item(), per_sample.mean().clamp_min(1.).item())

    def test_dynamics_detaches_posterior_and_representation_detaches_prior(self):
        q, p = self._logits()
        terms = balanced_categorical_kl(q, p, torch.ones(2, dtype=torch.bool), free_nats=0.)
        terms.dynamics.mean.backward()
        self.assertIsNone(q.grad)
        self.assertGreater(p.grad.abs().sum().item(), 0.)
        q, p = self._logits()
        terms = balanced_categorical_kl(q, p, torch.ones(2, dtype=torch.bool), free_nats=0.)
        terms.representation.mean.backward()
        self.assertGreater(q.grad.abs().sum().item(), 0.)
        self.assertIsNone(p.grad)

    def test_below_floor_has_zero_kl_gradient(self):
        q, p = self._logits()
        terms = balanced_categorical_kl(q, p, torch.ones(2, dtype=torch.bool))
        (terms.dynamics.mean+terms.representation.mean).backward()
        self.assertTrue(torch.equal(q.grad, torch.zeros_like(q)))
        self.assertTrue(torch.equal(p.grad, torch.zeros_like(p)))

    def test_missing_rows_skip_nan_logits_and_count_only_valid_samples(self):
        q, p = self._logits()
        q = q.detach()
        p = p.detach()
        q[1] = float("nan")
        p[1] = float("nan")
        q.requires_grad_()
        p.requires_grad_()
        terms = balanced_categorical_kl(q, p, torch.tensor([True, False]), free_nats=0.)
        (terms.dynamics.mean+terms.representation.mean).backward()
        self.assertEqual(terms.raw_kl.count.item(), 1)
        self.assertTrue(torch.equal(q.grad[1], torch.zeros_like(q.grad[1])))
        self.assertTrue(torch.equal(p.grad[1], torch.zeros_like(p.grad[1])))

    def test_empty_kl_has_connected_zero_on_appropriate_paths(self):
        q = torch.full((2, 32, 32), float("nan"), requires_grad=True)
        p = torch.full_like(q, float("nan"), requires_grad=True)
        terms = balanced_categorical_kl(q, p, torch.zeros(2, dtype=torch.bool))
        self.assertEqual(terms.dynamics.mean.item(), 0.)
        self.assertEqual(terms.representation.mean.item(), 0.)
        self.assertEqual(terms.raw_kl.count.item(), 0)
        (terms.dynamics.mean+terms.representation.mean).backward()
        self.assertTrue(torch.equal(q.grad, torch.zeros_like(q)))
        self.assertTrue(torch.equal(p.grad, torch.zeros_like(p)))

    def test_uniform_entropy_is_sum_of_all_variables(self):
        logits = torch.zeros(3, 32, 32)
        terms = balanced_categorical_kl(logits, logits, torch.ones(3, dtype=torch.bool))
        self.assertAlmostEqual(terms.prior_entropy.mean.item(), 32*math.log(32), places=4)
        self.assertAlmostEqual(terms.posterior_entropy.mean.item()/32, math.log(32), places=6)
        self.assertAlmostEqual(terms.raw_kl.mean.item(), 0.)

    def test_zero_uniform_mixture_handles_softmax_underflow(self):
        q = torch.full((1, 32, 32), -1000.)
        p = torch.full_like(q, -1000.)
        q[:, :, 0], p[:, :, 1] = 1000., 1000.
        terms = balanced_categorical_kl(q, p, torch.ones(1, dtype=torch.bool), uniform_mixture=0.)
        self.assertTrue(torch.isfinite(terms.raw_kl.mean))
        self.assertAlmostEqual(terms.raw_kl.mean.item(), 32*2000.)

    def test_kl_float32_under_autocast_and_invalid_contracts(self):
        q, p = self._logits()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            terms = balanced_categorical_kl(q.bfloat16(), p.bfloat16(), torch.ones(2, dtype=torch.bool))
            self.assertEqual(terms.dynamics.mean.dtype, torch.float32)
        for kwargs in ({"free_nats": -1}, {"uniform_mixture": 1.}, {"uniform_mixture": float("nan")}):
            with self.assertRaises(ValueError):
                balanced_categorical_kl(q, p, torch.ones(2, dtype=torch.bool), **kwargs)
        with self.assertRaises(ValueError):
            balanced_categorical_kl(q, p, torch.ones(2))
        with self.assertRaises(ValueError):
            balanced_categorical_kl(q[:, :16], p[:, :16], torch.ones(2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
