"""Exact absolute collection targets and matched student-learning schedules."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import unittest
from unittest.mock import patch

from world_models.cycle.schedule import Schedule
from world_models.infrastructure.config import resolve_config
from world_models.infrastructure.reproducibility import canonical_hash


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class CycleScheduleTests(unittest.TestCase):
    def settings(self, experiment="pilot", variant="B"):
        return resolve_config("relay", variant, experiment, CONFIG_DIR).settings

    def test_pilot_and_final_exact_schedule(self):
        for experiment, blocks, cycles, budget, updates in (
                ("pilot", 6, 220, 200_000, 8_000),
                ("final", 38, 860, 1_000_000, 24_000)):
            with self.subTest(experiment=experiment):
                schedule = Schedule.from_experiment(self.settings(experiment))
                self.assertEqual(schedule.blocks, blocks)
                self.assertEqual(schedule.total_local_cycles, cycles)
                self.assertEqual(schedule.total_model_updates, updates)
                self.assertEqual(schedule.block_target(0), 50_000)
                self.assertEqual(schedule.block_target(blocks), budget)
                self.assertEqual(schedule.local_cycles(0), 100)
                self.assertEqual(schedule.model_updates(0), 5_000)
                self.assertEqual(schedule.local_cycles(1), 20)
                self.assertEqual(schedule.model_updates(1), 500)

    def test_b_and_d_have_identical_local_schedule(self):
        self.assertEqual(Schedule.from_experiment(self.settings(variant="B")),
                         Schedule.from_experiment(self.settings(variant="D")))

    def test_absolute_targets_and_cumulative_cycles(self):
        schedule = Schedule.from_experiment(self.settings())
        self.assertEqual([schedule.block_target(i) for i in range(7)],
                         [50_000, 75_000, 100_000, 125_000, 150_000, 175_000, 200_000])
        self.assertEqual([schedule.completed_local_cycles(i) for i in range(7)],
                         [100, 120, 140, 160, 180, 200, 220])
        self.assertEqual(sum(schedule.local_cycles(i) for i in range(7)),
                         schedule.total_local_cycles)

    def test_block_indices_are_strict_and_bounded(self):
        schedule = Schedule.from_experiment(self.settings())
        for invalid in (-1, 7, True, 0.0, "0"):
            for method in (schedule.block_target, schedule.local_cycles,
                           schedule.model_updates, schedule.completed_local_cycles):
                with self.subTest(invalid=invalid, method=method.__name__), self.assertRaises(ValueError):
                    method(invalid)

    def test_schedule_is_frozen_and_has_owned_description(self):
        schedule = Schedule.from_experiment(self.settings())
        with self.assertRaises(FrozenInstanceError):
            schedule.real_budget = 7
        description = schedule.describe()
        description["block_targets"][0] = 7
        self.assertEqual(schedule.block_target(0), 50_000)
        self.assertEqual(description["online_blocks"], 6)

    def test_invalid_direct_budgets_and_full_configuration_are_rejected(self):
        schedule = Schedule.from_experiment(self.settings())
        for key, value in (("real_budget", 50_000), ("real_budget", 200_001),
                           ("collection_block", 0), ("initial_model_updates", -1),
                           ("initial_local_cycles", True), ("online_local_cycles", 20.0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                replace(schedule, **{key: value})
        settings = deepcopy(self.settings())
        settings["environment"]["actor_global_input"] = True
        with self.assertRaises(ValueError):
            Schedule.from_experiment(settings)

    def selected_row(self, settings, seed, updates):
        common = deepcopy(settings)
        common.pop("variant")
        return {"task": settings["task"]["name"], "seed": seed, "model_updates": updates,
                "original_target_stage_updates": 5000, "common_config_sha256": canonical_hash(common),
                "checkpoint_path": "checkpoint.pt", "checkpoint_sha256": "a"*64,
                "selection_path": "selection.json", "selection_sha256": "b"*64}

    def test_four_selected_cases_keep_unequal_actual_counts_and_registered_refresh_budget(self):
        for task, seed, count in (("relay", 0, 4250), ("relay", 1, 4250),
                                  ("cover", 0, 4750), ("cover", 1, 4750)):
            settings = resolve_config(task, "B", "pilot", CONFIG_DIR).settings
            before = deepcopy(settings)
            row = self.selected_row(settings, seed, count)
            with patch("world_models.training.selection.resolve_selected_case", return_value=row) as resolve:
                schedule, binding = Schedule.from_initial_selection(settings, seed, "selection.json")
            resolve.assert_called_once_with("selection.json", task, seed, checkpoint=None)
            self.assertEqual(schedule.initial_model_updates, count)
            self.assertEqual(schedule.total_model_updates, count + 3000)
            self.assertEqual(schedule.online_model_updates, 500)
            self.assertEqual(schedule.total_local_cycles, 220)
            self.assertEqual(binding["model_updates"], count)
            self.assertEqual(settings, before)

    def test_manifest_revalidates_selection_and_never_defaults_to_original_initial_count(self):
        settings = self.settings()
        row = self.selected_row(settings, 0, 4250)
        with patch("world_models.training.selection.resolve_selected_case", return_value=row):
            schedule, binding = Schedule.from_initial_selection(settings, 0, "selection.json")
            manifest = {"settings": settings, "master_seed": 0, "initial_selection": binding,
                "initial_checkpoint": row["checkpoint_path"], "initial_checkpoint_sha256": row["checkpoint_sha256"],
                "registered_schedule": Schedule.from_experiment(settings).describe(), "schedule": schedule.describe(),
                "initial_training_policy": "frozen_last_saved_checkpoints", "study_scope": "exploratory_pilot"}
            self.assertEqual(Schedule.from_manifest(manifest), schedule)
            for changes in ({"schedule": Schedule.from_experiment(settings).describe()},
                            {"initial_selection": None}, {"initial_checkpoint_sha256": "c"*64},
                            {"registered_schedule": schedule.describe()}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    Schedule.from_manifest({**manifest, **changes})
        with patch("world_models.training.selection.resolve_selected_case", side_effect=ValueError("selection receipt differs")):
            with self.assertRaisesRegex(ValueError, "receipt"):
                Schedule.from_manifest(manifest)

    def test_selection_cannot_override_configuration_or_apply_to_final_study(self):
        settings = self.settings()
        row = self.selected_row(settings, 0, 4250)
        for changes in ({"common_config_sha256": "c"*64}, {"original_target_stage_updates": 4000},
                        {"model_updates": 5001}, {"model_updates": True}, {"model_updates": 0}):
            with patch("world_models.training.selection.resolve_selected_case", return_value={**row, **changes}), \
                 self.subTest(changes=changes), self.assertRaises(ValueError):
                Schedule.from_initial_selection(settings, 0, "selection.json")
        with self.assertRaisesRegex(ValueError, "pilot only"):
            Schedule.from_initial_selection(self.settings("final"), 1001, "selection.json")


if __name__ == "__main__":
    unittest.main()
