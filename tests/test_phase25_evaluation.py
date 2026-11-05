"""Compact P2.5 orchestration checks using synthetic CPU fixtures only.

Saved update numbers below exercise provenance and selection; they represent
no optimizer steps, research training, simulator collection, or quality claim.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_model_sampling import identifier, make_store
from world_models.data.replay import canonical_hash
from world_models.evaluation import compact
from world_models.evaluation.metrics import MetricsAccumulator
from world_models.evaluation.verification import write_json
from world_models.infrastructure.config import resolve_config
from world_models.models.heads import TaskSpec
from world_models.models.world_model import SamplingStreams, build_model
from world_models.training.checkpoints import read_checkpoint, save_checkpoint
from world_models.training.normalization import fit_normalization
from world_models.training.sampling import ReplaySampler


@contextmanager
def forced_outcome(model, label):
    head = model.transition_heads.outcome_head
    weight, bias = head.weight.detach().clone(), head.bias.detach().clone()
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-100.)
        head.bias[label] = 100.
    try:
        yield
    finally:
        with torch.no_grad():
            head.weight.copy_(weight)
            head.bias.copy_(bias)


def ideal_metrics(*, uniform_windows=16, event_windows=8, positives=True,
                  always_continue=False, sampled_terminal=True, reward_cancellation=False):
    """Exact, small analytical windows with actual production accumulators."""
    task = TaskSpec()
    event = compact.milestone_names(task)[0]
    result = {}
    for stratum, count in (("uniform", uniform_windows), ("event", event_windows)):
        accumulator = MetricsAccumulator()
        for index in range(count):
            outcome = np.zeros(15, dtype=np.int64)
            milestone = np.zeros(15)
            terminal = np.zeros(15, dtype=bool)
            reward = np.zeros(15)
            if positives:
                outcome[-1], milestone[0], terminal[-1] = 1, 1., True
                reward[0], reward[-1] = .099, .999
            targets = {"rgb": np.zeros((15, 2, 3, 1, 1)), "image_valid": np.ones((15, 2), bool),
                       "reward": reward, "outcome": outcome, "outcome_valid": np.ones(15, bool),
                       "terminated": terminal, "deadline": np.zeros(15, bool),
                       "transition_index": np.arange(15),
                       "milestones": {event: {"value": milestone, "valid": np.ones(15, bool)}}}
            prediction_outcome = np.zeros(15, dtype=np.int64) if always_continue else outcome
            predicted_reward = reward.copy()
            if reward_cancellation:
                predicted_reward[1] += .5
                predicted_reward[2] -= .5
            predictions = {"active": np.ones(15, bool), "rgb": targets["rgb"].copy(),
                           "reward": predicted_reward, "outcome_probabilities": np.eye(3)[prediction_outcome],
                           "termination_probability": terminal.astype(float),
                           "terminated": terminal if sampled_terminal else np.zeros(15, bool),
                           "milestones": {event: milestone.copy()}}
            baseline = {"reward": 0., "outcome_probabilities": [1., 0., 0.],
                        "termination_probability": 0., "milestones": {event: 0.}}
            accumulator.add_window(targets, predictions, anchor_rgb=np.zeros((2, 3, 1, 1)),
                                   baseline=baseline, episode_id=f"{stratum}-{index}", horizon=15)
        result[stratum] = {"15": accumulator.report()}
    return result


class ReadinessTests(unittest.TestCase):
    def readiness(self, metrics):
        return compact.pilot_readiness(metrics, task=TaskSpec(),
            full_heldout_support={"transition_reasons": {"success": 8, "death": 0, "failure": 0}})

    def test_perfect_eight_window_timing_run_cannot_establish_readiness(self):
        result = self.readiness(ideal_metrics(uniform_windows=4, event_windows=4))
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(any("uniform windows" in reason for reason in result["reasons"]))
        self.assertFalse(result["P2.5_quality_acceptance"])
        self.assertFalse(result["automatic_phase3_launch"])

    def test_perfect_supported_metrics_have_only_provisional_exploratory_scope(self):
        result = self.readiness(ideal_metrics())
        self.assertEqual(result["status"], "ready_for_exploratory_phase3")
        self.assertFalse(result["P2.5_quality_acceptance"])
        self.assertTrue(any("zero positives" in name for name in result["unchecked_classes"]))
        self.assertIn("B versus D efficacy", result["unchecked_classes"])

    def test_zero_event_and_success_support_cannot_pass_even_with_zero_losses(self):
        result = self.readiness(ideal_metrics(positives=False))
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(any("distinct held-out positives" in reason for reason in result["reasons"]))
        self.assertTrue(any("distinct held-out successes" in reason for reason in result["reasons"]))

    def test_continue_forever_fails_learned_success_recall(self):
        result = self.readiness(ideal_metrics(always_continue=True))
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(any("learned success recall" in reason for reason in result["reasons"]))

    def test_correct_terminal_probabilities_do_not_replace_sampled_terminal_recall(self):
        metrics = ideal_metrics(sampled_terminal=False)
        self.assertEqual(metrics["event"]["15"]["termination"]["brier"]["mean"], 0.)
        result = self.readiness(metrics)
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(any("sampled terminal recall" in reason for reason in result["reasons"]))

    def test_reward_errors_that_cancel_still_require_cumulative_path_review(self):
        metrics = ideal_metrics(reward_cancellation=True)
        for report in (metrics["uniform"]["15"], metrics["event"]["15"]):
            self.assertAlmostEqual(report["reward"]["cumulative_mae"]["mean"], 0.)
            # Use an exact reward baseline to expose intermediate cancellation.
            report["reward"]["baseline_cumulative_path_mae"]["mean"] = 0.
            report["reward"]["cumulative_path_mae"]["mean"] = .2
        result = self.readiness(metrics)
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(any("path" in reason for reason in result["reasons"]))


class RealModelEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.fit = identifier("p25-evaluator-fit", "fit")
        cls.diagnostic = identifier("p25-evaluator-diagnostic", "diagnostic")
        cls.dataset = make_store(cls.root/"replay", [(cls.fit, 5, False), (cls.diagnostic, 20, False)])
        cls.stats = fit_normalization(cls.dataset)
        cls.sampler = ReplaySampler([cls.dataset], seed=21, partition="diagnostic", burn_in=0, loss_steps=15)
        cls.cfg = resolve_config("relay", "B", "pilot")
        cls.model = build_model(cls.cfg.settings, master_seed=0, variant="B").eval()
        cls.source = {"synthetic-fixture-trainer.py": "a"*64}
        cls.provenance = {"initial_normalization": {"identity_sha256": cls.stats.as_dict()["identity_sha256"],
                            "file_sha256": "b"*64},
                          "replays": [{"manifest_sha256": cls.dataset.manifest_sha256,
                            "dataset_identity_sha256": cls.dataset.manifest["dataset_identity_sha256"],
                            "transitions": cls.dataset.manifest["transition_count"]}]}
        common = dict(cls.cfg.settings)
        common.pop("variant")
        cls.metadata = {"profile": "pilot", "stage": "initial", "variant": "shared_initial:B,D",
                        "task_family": "relay", "run_seed": 0, "target_stage_updates": 5000,
                        "precision": "float32", "source_sha256": cls.source,
                        "common_config_sha256": canonical_hash(common), **cls.provenance}
        cls.checkpoint = cls.root/"synthetic_update_00003000.pt"
        optimizer = torch.optim.AdamW(cls.model.parameters())
        save_checkpoint(cls.checkpoint, model=cls.model, optimizer=optimizer,
                        streams=SamplingStreams.from_seed(3), sampler=cls.sampler,
                        update=3000, metadata=cls.metadata)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def window(self, anchor=3, steps=15):
        return self.sampler.window(0, self.diagnostic, anchor, full_prefix=True, burn_in=0, loss_steps=steps)

    def cache(self, *anchors):
        return compact.PrefixCache(self.model, device="cpu", seed=7, checkpoint_sha256="fixture-sha",
            records=[{"dataset_index": 0, "episode_id": self.diagnostic, "anchor": anchor} for anchor in anchors])

    def case(self):
        return self.cfg, [self.dataset], self.stats, self.provenance

    def test_full_prefix_cache_matches_uncached_named_streams_and_keeps_earlier_anchors(self):
        cache = self.cache(3, 8)
        later = self.window(8, 2)
        streams = SamplingStreams.from_seed(compact._seed(7, "fixture-sha", 0, self.diagnostic, "prefix"))
        with torch.inference_mode():
            actual = cache.get(later)
            early = cache.get(self.window(3, 2))
            state = self.model.initialize(torch.tensor(later.states["rgb"][0:1]), streams=streams)
            expected_early = None
            for index in range(8):
                prior = self.model.core.prior_step(state, torch.tensor(later.transitions["actions"][index:index+1], dtype=torch.long),
                                                   generator=streams.prior)
                state = self.model.core.observe(prior, self.model._images(torch.tensor(later.states["rgb"][index+1:index+2])),
                                                generator=streams.posterior)
                if index == 2:
                    expected_early = state
        for name in ("memory", "stochastic", "logits", "decision_index"):
            torch.testing.assert_close(getattr(actual, name), getattr(state, name), rtol=0, atol=0)
            torch.testing.assert_close(getattr(early, name), getattr(expected_early, name), rtol=0, atol=0)
        self.assertEqual(cache.observed_steps, 8)
        self.assertEqual(cache.hits, 1)

    def test_one_prior_rollout_scores_all_horizons_without_future_encoding_or_global_rng(self):
        window = self.window()
        rng = torch.get_rng_state().clone()
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        with forced_outcome(self.model, 0), torch.inference_mode():
            anchor = self.cache(window.anchor).get(window)
            with patch.object(self.model.core.encoder, "forward", side_effect=AssertionError("future encoder called")), \
                 patch.object(self.model, "imagine", wraps=self.model.imagine) as imagine, \
                 patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer called")):
                predictions = compact.predict_window(self.model, anchor, window, device="cpu", seed=7,
                                                     checkpoint_sha256="fixture-sha")
                targets = compact.targets_for_window(window, self.model.task)
                baseline = compact.window_baseline(compact.fitting_baseline(self.stats, self.model.task),
                                                    self.model.task, targets["transition_index"])
                for horizon in compact.HORIZONS:
                    accumulator = MetricsAccumulator()
                    accumulator.add_window(targets, predictions, anchor_rgb=window.states["rgb"][window.anchor],
                                           baseline=baseline, episode_id=self.diagnostic, horizon=horizon)
                    self.assertEqual(accumulator.report()["coverage"]["target_steps"], horizon)
                self.assertEqual(imagine.call_count, 1)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(predictions["active"].all())
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_early_model_termination_retains_missing_suffix_in_metrics(self):
        window = self.window()
        with forced_outcome(self.model, 1), torch.inference_mode():
            anchor = self.cache(window.anchor).get(window)
            predictions = compact.predict_window(self.model, anchor, window, device="cpu", seed=7,
                                                 checkpoint_sha256="fixture-sha")
        targets = compact.targets_for_window(window, self.model.task)
        baseline = compact.window_baseline(compact.fitting_baseline(self.stats, self.model.task),
                                           self.model.task, targets["transition_index"])
        accumulator = MetricsAccumulator()
        accumulator.add_window(targets, predictions, anchor_rgb=window.states["rgb"][window.anchor],
                               baseline=baseline, episode_id=self.diagnostic, horizon=15)
        report = accumulator.report()
        self.assertEqual(report["coverage"], {"target_steps": 15, "predicted_steps": 1,
                                              "missing_steps": 14, "fraction": 1/15})
        self.assertEqual(report["termination"]["false_early_terminations"], 1)
        self.assertEqual(report["outcome"]["missing_predictions"], 14)
        self.assertEqual(report["reward"]["missing_predictions"], 14)
        self.assertGreater(report["rgb"]["coverage_adjusted_mse"]["mean"], .9)

    def test_fitting_only_baseline_and_deadline_rule_ignore_diagnostic_frequencies(self):
        profile = self.stats.provenance["label_profile"]
        baseline = compact.fitting_baseline(self.stats, self.model.task)
        # Five fitting steps and twenty held-out steps; only fitting counts enter.
        self.assertEqual(profile["learned_outcome"]["valid"], 5)
        np.testing.assert_allclose(baseline["outcome_probabilities"], [.8, .2, 0.])
        self.assertEqual(baseline["milestones"]["event/latch_activations"], .2)
        values = compact.window_baseline(baseline, self.model.task, np.array([398, 399]))
        expected = baseline["expected_bonus"] + self.model.task.step_cost + .2
        np.testing.assert_allclose(values["reward"], [expected, expected-.8])
        np.testing.assert_allclose(values["termination_probability"], [.2, 1.])

    def test_committed_interim_checkpoint_loads_without_changing_global_rng(self):
        rng = torch.get_rng_state().clone()
        with patch.object(compact, "load_case", return_value=self.case()), \
             patch.object(compact, "source_identity", return_value=self.source):
            model, datasets, stats, loaded = compact._load_model(self.checkpoint, data_root="unused",
                statistics_root="unused", config_root=None, device="cpu", precision=None)
        self.assertEqual(loaded["update"], 3000)
        self.assertLess(loaded["update"], loaded["metadata"]["target_stage_updates"])
        self.assertFalse(model.training)
        self.assertEqual(datasets, [self.dataset])
        self.assertIs(stats, self.stats)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_overfit_refresh_source_config_and_replay_provenance_are_rejected(self):
        payload = read_checkpoint(self.checkpoint)
        for name, replacement, message in (
            ("stage", "overfit", "overfit/refresh"), ("stage", "refresh", "overfit/refresh"),
            ("source_sha256", {"changed.py": "c"*64}, "source_sha256"),
            ("common_config_sha256", "c"*64, "common_config_sha256"),
            ("replays", [], "replays"),
            ("initial_normalization", {"identity_sha256": "c"*64}, "initial_normalization")):
            with self.subTest(field=name, replacement=replacement):
                changed = {**payload, "metadata": {**payload["metadata"], name: replacement}}
                with patch.object(compact, "read_checkpoint", return_value=changed), \
                     patch.object(compact, "load_case", return_value=self.case()), \
                     patch.object(compact, "source_identity", return_value=self.source), \
                     self.assertRaisesRegex(ValueError, message):
                    compact._load_model(self.checkpoint, data_root="unused", statistics_root="unused",
                                        config_root=None, device="cpu", precision=None)

    def test_tiny_complete_evaluation_writes_finite_four_horizon_strata_without_training(self):
        output = self.root/"tiny-evaluation"
        rng = torch.get_rng_state().clone()
        with patch.object(compact, "load_case", return_value=self.case()), \
             patch.object(compact, "source_identity", return_value=self.source), \
             patch.object(compact, "evaluator_source_identity", return_value={"evaluator-fixture": "d"*64}), \
             patch.object(compact, "prepare_verification", return_value={"status": "fixture-pass"}), \
             patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer called")):
            report = compact.evaluate_checkpoint(self.checkpoint, output, data_root="unused",
                statistics_root="unused", device="cpu", windows=8, seed=7)
        self.assertEqual(report["checkpoint"]["update"], 3000)
        self.assertFalse(report["checkpoint"]["original_training_target_completed"])
        self.assertEqual(report["new_optimizer_updates"], 0)
        self.assertEqual(report["new_real_interactions"], 0)
        self.assertEqual(report["pilot_readiness"]["status"], "needs_review")
        self.assertEqual(set(report["metrics"]), {"uniform", "event"})
        for stratum in report["metrics"].values():
            self.assertEqual(set(stratum), {"1", "5", "10", "15"})
            for horizon, metrics in stratum.items():
                self.assertEqual(metrics["horizon"], int(horizon))
                self.assertIn("missing_steps", metrics["coverage"])
                self.assertIn("false_early_terminations", metrics["termination"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(json.loads((output/"report.json").read_text()), report)
        json.dumps(report, allow_nan=False)
        with self.assertRaises(FileExistsError):
            compact.evaluate_checkpoint(self.checkpoint, output, data_root="unused", statistics_root="unused", device="cpu")


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_root = self.root/"runs"
        for task, seed in compact.CASES:
            for update in (3000, 5000):
                path = self.run_root/task/f"seed_{seed}"/"checkpoints"/f"update_{update:08d}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic campaign path fixture; never loaded")
                path.with_suffix(".sha256").write_text("fixture receipt\n")

    def fake_evaluate(self, checkpoint, directory, **kwargs):
        task, seed = checkpoint.parents[2].name, int(checkpoint.parents[1].name.removeprefix("seed_"))
        report = {"checkpoint": {"path": str(checkpoint), "task_family": task, "run_seed": seed,
                                  "update": 3000, "shared_by_variants": ["B", "D"]},
                  "pilot_readiness": {"status": "needs_review", "reasons": ["fixture review"],
                                       "unchecked_classes": ["death"]}, "timing": {"total_seconds": .01}}
        write_json(directory/"report.json", report)
        return report

    def test_campaign_uses_explicit_common_interim_update_for_all_four_shared_models(self):
        output = self.root/"evaluation"
        with patch.object(compact, "prepare_verification", return_value={"status": "fixture-pass"}), \
             patch.object(compact, "_evaluate_one", side_effect=self.fake_evaluate) as evaluate, \
             patch.object(compact, "evaluator_source_identity", return_value={"fixture": "e"*64}):
            report = compact.evaluate_campaign(self.run_root, 3000, output, data_root="unused",
                statistics_root="unused", device="cpu", windows=8, seed=31)
        self.assertEqual(evaluate.call_count, 4)
        self.assertEqual([call.args[0].name for call in evaluate.call_args_list], ["update_00003000.pt"]*4)
        self.assertEqual([(row["checkpoint"]["task_family"], row["checkpoint"]["run_seed"])
                          for row in report["cases"]], list(compact.CASES))
        self.assertEqual(report["checkpoint_update"], 3000)
        self.assertEqual(report["pilot_readiness"]["status"], "needs_review")
        selection = json.loads((output/"pilot_model_selection.json").read_text())
        self.assertEqual(len(selection["models"]), 4)
        self.assertFalse(selection["P2.5_quality_acceptance"])
        self.assertTrue(all(row["checkpoint"]["shared_by_variants"] == ["B", "D"] for row in selection["models"]))
        self.assertFalse(any(self.run_root.rglob("pilot_model_selection.json")))

    def test_campaign_preflights_all_committed_pairs_before_any_model_inference(self):
        missing = self.run_root/"cover"/"seed_1"/"checkpoints"/"update_00003000.sha256"
        missing.unlink()
        output = self.root/"missing-common"
        with patch.object(compact, "prepare_verification", return_value={"status": "fixture-pass"}), \
             patch.object(compact, "_evaluate_one") as evaluate, \
             self.assertRaisesRegex(FileNotFoundError, "common-update"):
            compact.evaluate_campaign(self.run_root, 3000, output, data_root="unused", statistics_root="unused", device="cpu")
        evaluate.assert_not_called()
        report = json.loads((output/"report.json").read_text())
        self.assertEqual(report["pilot_readiness"]["status"], "failed")
        self.assertEqual(report["new_optimizer_updates"], 0)

    def test_campaign_rejects_metadata_not_matching_explicit_task_seed_update(self):
        def wrong_metadata(*args, **kwargs):
            report = self.fake_evaluate(*args, **kwargs)
            report["checkpoint"]["update"] = 5000
            return report
        with patch.object(compact, "prepare_verification", return_value={"status": "fixture-pass"}), \
             patch.object(compact, "_evaluate_one", side_effect=wrong_metadata), \
             self.assertRaisesRegex(ValueError, "explicit task/seed/update"):
            compact.evaluate_campaign(self.run_root, 3000, self.root/"wrong-metadata", data_root="unused",
                                      statistics_root="unused", device="cpu")

    def test_evaluation_failure_preserves_report_and_restores_thread_setting(self):
        output = self.root/"failed"
        threads = torch.get_num_threads()
        with patch.object(compact, "prepare_verification", side_effect=RuntimeError("fixture verification failed")), \
             self.assertRaisesRegex(RuntimeError, "verification failed"):
            compact.evaluate_checkpoint("unused.pt", output, data_root="unused", statistics_root="unused", device="cpu")
        self.assertEqual(torch.get_num_threads(), threads)
        report = json.loads((output/"report.json").read_text())
        self.assertEqual(report["pilot_readiness"]["status"], "failed")
        with self.assertRaises(FileExistsError):
            compact.evaluate_checkpoint("unused.pt", output, data_root="unused", statistics_root="unused", device="cpu")


if __name__ == "__main__":
    unittest.main()
