"""Behavioral PPO evidence on the analytic CPU task, never native research runs."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import torch

from scripts.phase36_support.evidence import file_sha256, load_report
from scripts.phase36_support.mechanics import (AnalyticCueWorld, EngineeringCuePrefixes,
    cue_images, evaluate_cues, run_mechanics)
from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.imagination.versions import module_version
from world_models.models.world_model import SamplingStreams
from world_models.policies import LocalActor
from world_models.policies.config import PolicyConfig


class Phase36MechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "mechanics"
        cls.original_threads = torch.get_num_threads()
        cls.rng_before = torch.random.get_rng_state().clone()
        cls.report = run_mechanics(cls.path)
        cls.rng_after = torch.random.get_rng_state().clone()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_actual_production_mappo_learns_heldout_local_visual_cues(self):
        report = self.report
        self.assertEqual(report["status"], "passed", report["checks"])
        self.assertTrue(all(report["checks"].values()))
        self.assertGreater(report["after"]["expected_joint_success"], report["before"]["expected_joint_success"] + .4)
        self.assertGreater(report["after"]["mean_expected_team_return"], report["before"]["mean_expected_team_return"])
        self.assertGreater(report["after"]["expected_individual_accuracy"], .7)
        self.assertEqual(report["learning_algorithm"], "production RecurrentMAPPO")
        self.assertTrue(torch.equal(self.rng_before, self.rng_after))
        self.assertEqual(torch.get_num_threads(), self.original_threads)

    def test_every_cycle_replays_behavior_and_runs_the_requested_optimizer_schedule(self):
        rows = [json.loads(line) for line in (self.path / "ppo_updates.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), self.report["cycles"])
        self.assertEqual(self.report["actor_optimizer_steps"], 2 * len(rows))
        self.assertEqual(self.report["critic_optimizer_steps"], 2 * len(rows))
        for index, row in enumerate(rows, 1):
            self.assertEqual(row["schema_version"], "p34-update-v1")
            self.assertEqual(row["cycles_completed"], index)
            self.assertEqual(row["cycle_actor_optimizer_steps"], 2)
            self.assertEqual(row["imagined_joint_steps"], 32)
            self.assertEqual(row["new_environment_interactions"], 0)
            self.assertLess(row["metrics"]["initial_ratio_max_error"], 2e-5)
            self.assertEqual(row["world_model_version"], self.report["world_model_version"])
        self.assertEqual(file_sha256(self.path / "ppo_updates.jsonl"), self.report["ppo_updates_sha256"])

    def test_exported_weights_reproduce_both_reported_heldout_scores(self):
        images, targets = cue_images(self.report["heldout_cue_seed"], self.report["heldout_episodes"])
        for label in ("before", "after"):
            record = self.report["actor_exports"][label]
            directory = self.path / record["path"]
            state, metadata = load_cycle_checkpoint(directory)
            self.assertEqual(set(state), {"actor"})
            self.assertTrue(metadata["engineering_only"])
            self.assertFalse(metadata["phase3_acceptance"])
            self.assertEqual(metadata["real_joint_transitions"], 0)
            config = dict(metadata["actor_config"])
            for key in ("image_shape", "encoder_channels", "critic_hidden_sizes"):
                config[key] = tuple(config[key])
            with torch.random.fork_rng(devices=[]):
                actor = LocalActor(PolicyConfig(**config))
            actor.load_state_dict(state["actor"])
            self.assertEqual(module_version(actor), record["actor_version"])
            actual = evaluate_cues(actor, images, targets)
            for name, value in self.report[label].items():
                self.assertAlmostEqual(actual[name], value, places=6)
            self.assertEqual(file_sha256(directory / "manifest.json"), record["manifest_sha256"])
            self.assertEqual(file_sha256(directory / "state.pt"), record["payload_sha256"])

    def test_fixture_evidence_never_claims_native_environment_or_research_acceptance(self):
        self.assertTrue(self.report["engineering_only"])
        for field in ("native_environment_evaluation", "research_training_executed", "phase3_acceptance"):
            self.assertFalse(self.report[field])
        self.assertEqual(self.report["new_environment_interactions"], 0)
        images, _ = cue_images(36, 4)
        sampler = EngineeringCuePrefixes(images)
        self.assertEqual(sampler.data_source, "analytic_engineering_fixture")
        self.assertEqual(sampler.real_environment_interactions, 0)
        self.assertFalse(hasattr(sampler, "datasets"))
        self.assertFalse((self.path / "replay").exists())

    def test_balanced_independent_cues_and_heldout_brightness_are_reproducible(self):
        images, targets = cue_images(42, 32)
        images_again, targets_again = cue_images(42, 32)
        np.testing.assert_array_equal(images, images_again)
        np.testing.assert_array_equal(targets, targets_again)
        np.testing.assert_array_equal(images.mean((-1, -2))[:, :, :2].argmax(-1), targets)
        for pair in ((0, 0), (0, 1), (1, 0), (1, 1)):
            self.assertEqual(int((targets == pair).all(-1).sum()), 8)
        other, _ = cue_images(43, 32)
        self.assertFalse(np.array_equal(images, other))

    def test_analytic_shared_reward_depends_only_on_sampled_executable_actions(self):
        images, targets = cue_images(73, 4)
        model = AnalyticCueWorld()
        before = module_version(model)
        streams = SamplingStreams.from_seed(84)
        world = model.initialize(torch.tensor(images), streams=streams)
        for correct_players, expected in ((2, 2.), (1, -.5), (0, -1.)):
            actions = torch.tensor(targets)
            if correct_players < 2:
                actions[:, 0] = 11
            if correct_players < 1:
                actions[:, 1] = 10
            transition = model.imagine_step(model.start_imagination(world), actions, streams=streams)
            self.assertEqual(transition.reward.tolist(), [expected] * 4)
            self.assertTrue(transition.terminated.all())
            self.assertEqual(transition.outcome.tolist(), [1 if correct_players == 2 else 2] * 4)
        self.assertEqual(module_version(model), before)

    def test_report_receipt_and_source_binding_are_preserved(self):
        persisted = load_report(self.path / "report.json")
        self.assertEqual(persisted["checks"], self.report["checks"])
        self.assertEqual(persisted["source_sha256"], self.report["source_sha256_after"])
        self.assertIn("scripts/phase36_support/mechanics.py", persisted["source_sha256"])
        with self.assertRaises(FileExistsError):
            run_mechanics(self.path, cycles=1)

    def test_invalid_options_fail_before_output_creation(self):
        for options in ({"seed": -1}, {"seed": True}, {"cycles": 0}, {"cycles": True},
                        {"batch_size": 3}, {"batch_size": True}):
            path = Path(self.directory.name) / "invalid"
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_mechanics(path, **options)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
