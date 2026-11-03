"""Hand-computed, mask-safe P2.3 diagnostics without model mutation."""

import json
import math
import unittest

import torch
from torch import nn

from world_models.training.diagnostics import binary_diagnostics, categorical_diagnostics, gradient_diagnostics


class GradientDiagnosticsTests(unittest.TestCase):
    def model(self):
        return nn.ModuleDict({"first": nn.Linear(2, 1), "second": nn.Linear(1, 1)})

    def test_global_and_module_norms_and_missing_gradient_counts(self):
        model = self.model()
        model["first"].weight.grad = torch.tensor([[3., 4.]])
        model["first"].bias.grad = torch.tensor([12.])
        model["second"].weight.grad = torch.tensor([[5.]])
        report = gradient_diagnostics(model)
        self.assertAlmostEqual(report["global_l2"], math.sqrt(194))
        self.assertEqual(report["by_module_l2"], {"first": 13., "second": 5.})
        self.assertEqual(report["parameters_with_grad"], 3)
        self.assertEqual(report["parameters_without_grad"], 1)
        self.assertEqual(report["nonfinite_count"], 0)
        json.dumps(report, allow_nan=False)

    def test_no_gradients_are_missing_measurements_not_clipping_operations(self):
        model = self.model()
        report = gradient_diagnostics(model)
        self.assertEqual(report["global_l2"], 0.)
        self.assertEqual(report["by_module_l2"], {"first": None, "second": None})
        self.assertEqual(report["parameters_with_grad"], 0)
        self.assertEqual(report["parameters_without_grad"], 4)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_measurements_preserve_gradient_objects_values_weights_and_mode(self):
        model = self.model().eval()
        output = model["second"](model["first"](torch.tensor([[1., 2.]])))
        output.square().sum().backward()
        parameters = [(parameter, parameter.detach().clone(), parameter.grad, parameter.grad.clone())
                      for parameter in model.parameters()]
        gradient_diagnostics(model)
        self.assertFalse(model.training)
        for parameter, weight, gradient, value in parameters:
            self.assertTrue(torch.equal(parameter, weight))
            self.assertIs(parameter.grad, gradient)
            self.assertTrue(torch.equal(parameter.grad, value))

    def test_float32_large_gradients_do_not_overflow_norm(self):
        model = nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.full_like(model.weight, 1e30)
        norm = gradient_diagnostics(model)["global_l2"]
        self.assertTrue(math.isfinite(norm))
        self.assertAlmostEqual(norm / float(model.weight.grad[0, 0]), math.sqrt(2), places=12)

    def test_nonfinite_gradients_raise_or_return_json_null_norms(self):
        model = self.model()
        model["first"].weight.grad = torch.tensor([[float("nan"), float("inf")]])
        model["second"].bias.grad = torch.tensor([4.])
        with self.assertRaises(FloatingPointError):
            gradient_diagnostics(model)
        report = gradient_diagnostics(model, error_if_nonfinite=False)
        self.assertIsNone(report["global_l2"])
        self.assertIsNone(report["by_module_l2"]["first"])
        self.assertEqual(report["by_module_l2"]["second"], 4.)
        self.assertEqual(report["nonfinite_count"], 2)
        self.assertEqual(report["nonfinite_parameters"], ["first.weight"])
        json.dumps(report, allow_nan=False)

    def test_sparse_gradient_coalescing_does_not_modify_original(self):
        model = nn.Embedding(3, 2, sparse=True)
        model(torch.tensor([0, 0, 2])).sum().backward()
        gradient = model.weight.grad
        self.assertFalse(gradient.is_coalesced())
        report = gradient_diagnostics(model)
        self.assertAlmostEqual(report["global_l2"], math.sqrt(10))
        self.assertIs(model.weight.grad, gradient)
        self.assertFalse(gradient.is_coalesced())


class EventDiagnosticsTests(unittest.TestCase):
    def test_binary_imbalanced_counts_probability_error_precision_recall(self):
        probabilities = torch.tensor([0.1, 0.8, 0.6, 0.2])
        logits = torch.logit(probabilities).requires_grad_()
        targets = torch.tensor([0., 1., 0., 1.])
        result = binary_diagnostics(logits, targets, torch.ones(4, dtype=torch.bool))
        self.assertEqual(result["support"], 4)
        self.assertEqual(result["positives"], 2)
        self.assertEqual(result["negatives"], 2)
        self.assertEqual(result["positive_fraction"], 0.5)
        self.assertAlmostEqual(result["mean_probability"], 0.425, places=6)
        self.assertEqual(result["predicted_positives"], 2)
        self.assertAlmostEqual(result["brier"], (0.01+0.04+0.36+0.64)/4, places=6)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertIsNone(logits.grad)
        json.dumps(result, allow_nan=False)

    def test_empty_binary_metrics_and_absent_positive_class_are_explicit(self):
        result = binary_diagnostics(torch.tensor([float("nan")]), torch.tensor([float("nan")]),
                                     torch.zeros(1, dtype=torch.bool))
        for key in ("positive_fraction", "mean_probability", "brier", "precision", "recall"):
            self.assertIsNone(result[key])
        self.assertEqual(result["support"], 0)
        self.assertEqual(result["predicted_positives"], 0)
        result = binary_diagnostics(torch.tensor([-5., -3.]), torch.zeros(2), torch.ones(2, dtype=torch.bool))
        self.assertEqual(result["positive_fraction"], 0.)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])

    def test_masked_binary_nan_and_prefix_broadcast_are_ignored(self):
        logits = torch.tensor([[0., 0.], [float("nan"), float("nan")]], requires_grad=True)
        targets = torch.tensor([[0., 1.], [float("nan"), 8.]])
        report = binary_diagnostics(logits, targets, torch.tensor([True, False]))
        self.assertEqual(report["support"], 2)
        self.assertEqual(report["predicted_positives"], 2)
        self.assertEqual(report["brier"], 0.25)
        self.assertTrue(torch.isnan(logits[1]).all())
        self.assertIsNone(logits.grad)

    def test_categorical_counts_and_confusion_orientation(self):
        logits = torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 0., 5.], [0., 5., 0.]], requires_grad=True)
        result = categorical_diagnostics(logits, torch.tensor([0, 0, 2, 2]), torch.ones(4, dtype=torch.bool))
        self.assertEqual(result["support"], 4)
        self.assertEqual(result["target_counts"], [2, 0, 2])
        self.assertEqual(result["predicted_counts"], [1, 2, 1])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 0, 0], [0, 1, 1]])
        self.assertIsNone(logits.grad)
        json.dumps(result, allow_nan=False)

    def test_empty_and_masked_invalid_categorical_rows(self):
        logits = torch.tensor([[0., 1., 2.], [float("nan")]*3])
        targets = torch.tensor([2, -999])
        result = categorical_diagnostics(logits, targets, torch.tensor([True, False]))
        self.assertEqual(result["accuracy"], 1.)
        self.assertEqual(result["target_counts"], [0, 0, 1])
        empty = categorical_diagnostics(logits, targets, torch.zeros(2, dtype=torch.bool))
        self.assertIsNone(empty["accuracy"])
        self.assertEqual(empty["target_counts"], [0, 0, 0])
        self.assertEqual(empty["confusion_matrix"], [[0]*3 for _ in range(3)])

    def test_diagnostic_domains_and_nonfinite_valid_predictions_are_checked(self):
        with self.assertRaises(ValueError):
            binary_diagnostics(torch.zeros(1), torch.tensor([0.5]), torch.ones(1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            binary_diagnostics(torch.tensor([float("nan")]), torch.zeros(1), torch.ones(1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            categorical_diagnostics(torch.zeros(1, 2), torch.tensor([2]), torch.ones(1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            categorical_diagnostics(torch.full((1, 2), float("nan")), torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            categorical_diagnostics(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
