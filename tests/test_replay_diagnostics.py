"""Repeatability comparisons must not hide physical or event differences."""

from copy import deepcopy
import unittest
from pathlib import Path
import tempfile

from world_models.env.replay_diagnostics import compare_replays, interaction_counts, prediction_config_record, run_fixed_action_replay


class ReplayComparisonTests(unittest.TestCase):
    def fixture(self):
        return [{"state": {"players": [{"x": 1.0}], "reward": -.001},
                 "events_rewards": {"reward": -.001, "counter": 0},
                 "rgb_sha256": ["first", "second"]}]

    def test_tiny_physical_change_fails_exact_state_comparison(self):
        reference = self.fixture()
        replay = deepcopy(reference)
        replay[0]["state"]["players"][0]["x"] += 1 / 65536
        result = compare_replays(reference, replay)
        self.assertFalse(result["state_exact"])
        self.assertTrue(result["events_rewards_exact"])
        self.assertEqual(result["first_state_difference"]["snapshot_index"], 0)

    def test_event_difference_and_missing_final_snapshot_fail_independently(self):
        reference = self.fixture()
        replay = deepcopy(reference)
        replay[0]["events_rewards"]["counter"] = 1
        self.assertFalse(compare_replays(reference, replay)["events_rewards_exact"])
        result = compare_replays(reference, [])
        self.assertFalse(result["state_exact"])
        self.assertFalse(result["events_rewards_exact"])
        self.assertFalse(result["rgb_exact"])

    def test_rgb_difference_is_reported_separately_from_exact_physical_state(self):
        reference = self.fixture()
        replay = deepcopy(reference)
        replay[0]["rgb_sha256"][1] = "different"
        result = compare_replays(reference, replay)
        self.assertTrue(result["state_exact"])
        self.assertTrue(result["events_rewards_exact"])
        self.assertFalse(result["rgb_exact"])

    def test_signed_zero_representation_is_not_silently_normalized(self):
        reference = self.fixture()
        replay = deepcopy(reference)
        reference[0]["state"]["players"][0]["x"] = 0.0
        replay[0]["state"]["players"][0]["x"] = -0.0
        self.assertFalse(compare_replays(reference, replay)["state_exact"])

    def test_invalid_replay_arguments_are_rejected_before_any_engine_start(self):
        cases = [("other", [[0, 0]], {}), ("relay", [], {}), ("relay", [[0]], {}),
                 ("relay", [[0, 12]], {}), ("relay", [[True, 0]], {}),
                 ("relay", [[0, 0]], {"repeats": 1}),
                 ("relay", [[0, 0]], {"reset_mode": "unknown"}),
                 ("relay", [[0, 0]], {"coordinator_reset_strategy": "unknown"})]
        for task, actions, options in cases:
            with self.subTest(task=task, actions=actions, options=options), self.assertRaises(ValueError):
                run_fixed_action_replay(task, actions, **options)

    def test_prediction_readback_requires_recognized_global_cvar(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "_vizdoom.ini"
            path.write_text("[UnknownConsoleVariables]\ncl_noprediction=true\n")
            self.assertIsNone(prediction_config_record(path)["cl_noprediction"])
            path.write_text("[GlobalSettings]\ncl_noprediction=false\n")
            self.assertIs(prediction_config_record(path)["cl_noprediction"], False)
            path.write_text("[GlobalSettings]\ncl_noprediction=true\n")
            self.assertIs(prediction_config_record(path)["cl_noprediction"], True)

    def test_replay_counts_distinguish_cumulative_values_from_repeat_deltas(self):
        counts = interaction_counts(115, 1, 230, 2, 114)
        self.assertEqual(counts["consumed_joint_decisions"], 230)
        self.assertEqual(counts["consumed_joint_decisions_delta"], 115)
        self.assertEqual(counts["initialization_joint_decisions_delta"], 1)
        self.assertEqual(counts["published_transitions"], 114)
        for values in ((115, 1, 114, 2, 0), (115, 1, 230, 0, 114), (115, 1, 230, 2, 115), (True, 0, 1, 0, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                interaction_counts(*values)


if __name__ == "__main__":
    unittest.main()
