"""Analytical rollout metrics: no model, GPU, training, or native simulator."""
from copy import deepcopy
import json
import unittest

import numpy as np

from world_models.evaluation.metrics import MetricsAccumulator, score_window


EVENT = "event/latch_activations"


def fixture():
    rgb = np.zeros((3, 2, 3, 1, 1), dtype=np.float32)
    rgb[1] = 1.
    targets = {"rgb": rgb, "image_valid": np.ones((3, 2), dtype=bool),
               "reward": np.array([0., 1., -.1]), "outcome": np.array([0, 0, 1]),
               "outcome_valid": np.ones(3, dtype=bool), "terminated": np.array([False, False, True]),
               "deadline": np.zeros(3, dtype=bool), "transition_index": np.arange(10, 13),
               "milestones": {EVENT: {"value": np.array([0., 1., 0.]), "valid": np.ones(3, dtype=bool)}},
               "stage": np.array([0, 1, 1]), "stage_valid": np.ones(3, dtype=bool)}
    predictions = {"active": np.ones(3, dtype=bool), "rgb": rgb.copy(),
                   "reward": np.array([0., .5, -.1]), "outcome_probabilities": np.eye(3)[[0, 0, 1]],
                   "terminated": np.array([False, False, True]), "termination_probability": np.array([.1, .1, .9]),
                   "milestones": {EVENT: np.array([.2, .8, .4])},
                   "stage_probabilities": np.array([[1., 0.], [0., 1.], [0., 1.]])}
    baseline = {"reward": 0., "outcome_probabilities": np.array([1., 0., 0.]), "milestones": {EVENT: .1}}
    return targets, predictions, baseline


def score(targets=None, predictions=None, baseline=None, *, horizon=3):
    original = fixture()
    return score_window(targets if targets is not None else original[0],
                        predictions if predictions is not None else original[1],
                        anchor_rgb=np.zeros((2, 3, 1, 1)),
                        baseline=baseline if baseline is not None else original[2],
                        episode_id="episode-a", horizon=horizon)


class Phase25MetricsTests(unittest.TestCase):
    def test_hand_computed_pixel_reward_outcome_and_milestone_scores(self):
        report = score()
        self.assertEqual(report["coverage"], {"target_steps": 3, "predicted_steps": 3, "missing_steps": 0, "fraction": 1.})
        self.assertEqual(report["rgb"]["model_mse"], {"sum": 0., "count": 18, "mean": 0.})
        self.assertAlmostEqual(report["rgb"]["persistence_mse"]["mean"], 1/3)
        self.assertAlmostEqual(report["reward"]["model_mae"]["mean"], .5/3)
        self.assertAlmostEqual(report["reward"]["baseline_mae"]["mean"], 1.1/3)
        self.assertAlmostEqual(report["reward"]["cumulative_mae"]["mean"], .5)
        self.assertAlmostEqual(report["reward"]["baseline_cumulative_mae"]["mean"], .9)
        self.assertEqual(report["outcome"]["confusion_matrix"], [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(report["outcome"]["brier"]["mean"], 0.)
        self.assertAlmostEqual(report["outcome"]["baseline_brier"]["mean"], 2/3)
        self.assertIsNone(report["outcome"]["per_class"]["failure"]["f1"])
        event = report["milestones"][EVENT]
        self.assertEqual((event["support"], event["positives"], event["unique_positive_events"]), (3, 1, 1))
        self.assertEqual(event["f1"], 1.)
        self.assertAlmostEqual(event["brier"]["mean"], .08)
        self.assertAlmostEqual(event["baseline_brier"]["mean"], .83/3)
        self.assertEqual(event["baseline"]["per_class"]["present"]["f1"], 0.)
        self.assertAlmostEqual(report["termination"]["brier"]["mean"], .01)
        self.assertEqual(report["stage"]["accuracy"], 1.)
        self.assertIsNone(report["quality_acceptance"])
        json.dumps(report, allow_nan=False)

    def test_early_model_end_never_hides_remaining_real_suffix(self):
        targets, predictions, baseline = fixture()
        predictions["active"] = np.array([True, False, False])
        predictions["terminated"] = np.array([True, False, False])
        for name in ("rgb", "reward", "outcome_probabilities", "termination_probability", "stage_probabilities"):
            predictions[name][1:] = np.nan
        predictions["milestones"][EVENT][1:] = np.nan
        report = score(targets, predictions, baseline)
        self.assertEqual(report["coverage"]["missing_steps"], 2)
        self.assertEqual(report["rgb"]["model_mse"]["mean"], 0.)
        self.assertAlmostEqual(report["rgb"]["coverage_adjusted_mse"]["mean"], 2/3)
        self.assertEqual(report["outcome"]["confusion_matrix"], [[1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
        self.assertEqual(report["outcome"]["brier"]["mean"], 0.)
        self.assertAlmostEqual(report["outcome"]["coverage_adjusted_brier"]["mean"], 4/3)
        self.assertEqual(report["outcome"]["accuracy"], 1/3)
        self.assertEqual(report["termination"]["false_early_windows"], 1)
        self.assertEqual(report["termination"]["false_early_terminations"], 1)
        self.assertEqual(report["termination"]["missed_terminals"], 1)
        self.assertAlmostEqual(report["reward"]["cumulative_mae"]["mean"], .9)
        self.assertEqual(report["milestones"][EVENT]["f1"], 0.)
        self.assertEqual(report["milestones"][EVENT]["missing_predictions"], 2)
        self.assertEqual(report["stage"]["accuracy"], 1/3)
        json.dumps(report, allow_nan=False)

    def test_short_prediction_arrays_are_missing_not_an_implicit_shorter_target(self):
        targets, predictions, baseline = fixture()
        predictions["active"] = np.array([True])
        predictions["terminated"] = np.array([True])
        for name in ("rgb", "reward", "outcome_probabilities", "termination_probability", "stage_probabilities"):
            predictions[name] = predictions[name][:1]
        predictions["milestones"][EVENT] = predictions["milestones"][EVENT][:1]
        report = score(targets, predictions, baseline)
        self.assertEqual(report["coverage"]["target_steps"], 3)
        self.assertEqual(report["coverage"]["predicted_steps"], 1)
        self.assertEqual(report["outcome"]["missing_predictions"], 2)

    def test_joint_missing_image_does_not_mask_reward_outcome_or_state_targets(self):
        targets, predictions, baseline = fixture()
        targets["image_valid"][-1, 1] = False
        targets["rgb"][-1] = np.nan
        predictions["rgb"][-1] = np.nan
        report = score(targets, predictions, baseline)
        self.assertEqual(report["rgb"]["model_mse"]["count"], 12)
        self.assertEqual(report["reward"]["model_mae"]["count"], 3)
        self.assertEqual(report["outcome"]["support"], 3)
        self.assertEqual(report["stage"]["support"], 3)

    def test_clock_deadline_is_not_learned_nondeadline_failure_support(self):
        targets, predictions, baseline = fixture()
        targets["outcome"][-1] = 0
        targets["deadline"][-1] = True
        predictions["outcome_probabilities"][-1] = [1., 0., 0.]
        predictions["termination_probability"][-1] = 1.
        baseline["reward"] = np.array([0., .2, -.1])
        report = score(targets, predictions, baseline)
        self.assertEqual(report["outcome"]["class_support"], [3, 0, 0])
        self.assertIsNone(report["outcome"]["per_class"]["failure"]["f1"])
        self.assertEqual(report["termination"]["deadline_terminals"], 1)
        self.assertEqual(report["termination"]["nondeadline_terminals"], 0)
        self.assertAlmostEqual(report["termination"]["baseline_brier"]["mean"], 0.)
        self.assertAlmostEqual(report["reward"]["baseline_mae"]["mean"], .8/3)

    def test_zero_support_remains_null_and_masked_nan_labels_are_safe(self):
        targets, predictions, baseline = fixture()
        targets["image_valid"][:] = False
        targets["outcome_valid"][:] = False
        targets["reward_valid"] = np.zeros(3, dtype=bool)
        targets["termination_valid"] = np.zeros(3, dtype=bool)
        targets["stage_valid"][:] = False
        targets["milestones"][EVENT]["valid"][:] = False
        targets["rgb"][:] = np.nan
        targets["reward"][:] = np.nan
        targets["milestones"][EVENT]["value"][:] = np.nan
        report = score(targets, predictions, baseline)
        self.assertIsNone(report["rgb"]["model_mse"]["mean"])
        self.assertIsNone(report["reward"]["cumulative_mae"]["mean"])
        self.assertIsNone(report["outcome"]["accuracy"])
        self.assertIsNone(report["milestones"][EVENT]["f1"])
        self.assertIsNone(report["stage"]["accuracy"])
        self.assertEqual(report["milestones"][EVENT]["unique_positive_events"], 0)
        json.dumps(report, allow_nan=False)

    def test_additive_counts_and_deduplicated_events_do_not_invent_independent_samples(self):
        targets, predictions, baseline = fixture()
        accumulator = MetricsAccumulator()
        for _ in range(2):
            accumulator.add_window(targets, predictions, baseline=baseline,
                anchor_rgb=np.zeros((2, 3, 1, 1)), episode_id="same-episode", horizon=3)
        report = accumulator.report()
        self.assertEqual((report["windows"], report["unique_windows"], report["unique_episodes"]), (2, 1, 1))
        self.assertEqual(report["outcome"]["support"], 6)
        self.assertEqual(report["milestones"][EVENT]["positives"], 2)
        self.assertEqual(report["milestones"][EVENT]["unique_positive_events"], 1)
        self.assertAlmostEqual(report["reward"]["cumulative_mae"]["mean"], .5)

    def test_horizon_slices_one_rollout_and_short_real_suffix_is_identified(self):
        for horizon, steps, full in ((1, 1, 1), (5, 3, 0), (10, 3, 0), (15, 3, 0)):
            report = score(horizon=horizon)
            self.assertEqual(report["coverage"]["target_steps"], steps)
            self.assertEqual(report["full_horizon_windows"], full)
            self.assertEqual(report["short_real_suffix_windows"], 1-full)

    def test_unknown_absolute_indices_cannot_claim_unique_event_counts(self):
        targets, predictions, baseline = fixture()
        del targets["transition_index"]
        report = score(targets, predictions, baseline)
        self.assertFalse(report["absolute_event_ids_available"])
        self.assertIsNone(report["unique_windows"])
        self.assertIsNone(report["milestones"][EVENT]["unique_positive_events"])

    def test_malformed_active_predictions_probabilities_and_labels_fail_without_partial_addition(self):
        targets, predictions, baseline = fixture()
        accumulator = MetricsAccumulator()
        accumulator.add_window(targets, predictions, baseline=baseline,
                               anchor_rgb=np.zeros((2, 3, 1, 1)), episode_id="episode-a", horizon=3)
        original = accumulator.report()
        mutations = [lambda t, p, b: p.update(active=np.array([True, False, True])),
                     lambda t, p, b: p["outcome_probabilities"].__setitem__((0, 0), np.nan),
                     lambda t, p, b: p["outcome_probabilities"].__setitem__(0, [1., 1., 0.]),
                     lambda t, p, b: t["outcome"].__setitem__(0, 3),
                     lambda t, p, b: p["milestones"][EVENT].__setitem__(0, 2.),
                     lambda t, p, b: p.update(terminated=np.array([True, False, True])),
                     lambda t, p, b: t["transition_index"].__setitem__(1, 99)]
        for mutate in mutations:
            t, p, b = deepcopy((targets, predictions, baseline))
            mutate(t, p, b)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                accumulator.add_window(t, p, baseline=b, anchor_rgb=np.zeros((2, 3, 1, 1)),
                                       episode_id="episode-b", horizon=3)
            self.assertEqual(accumulator.report(), original)
        with self.assertRaisesRegex(ValueError, "different horizons"):
            accumulator.add_window(targets, predictions, baseline=baseline,
                                   anchor_rgb=np.zeros((2, 3, 1, 1)), episode_id="episode-a", horizon=1)


if __name__ == "__main__":
    unittest.main()
