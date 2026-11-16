"""Amended budgets retain the original P2.4 identities and matched B/D inputs."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import phase35
from world_models.cycle.controller import CycleController, CycleOptions
from world_models.cycle.schedule import Schedule
from world_models.cycle.study import apply_study_profile, parent_settings_for_study
from world_models.infrastructure.config import resolve_config, validate_pair, ResolvedConfig
from world_models.infrastructure.reproducibility import canonical_hash


class CycleStudyTests(unittest.TestCase):
    def setUp(self):
        self.parent = resolve_config("relay", "B", "pilot").settings
        self.settings, self.amendment = apply_study_profile(self.parent, "relay_week_v1", master_seed=0)

    def selected_row(self):
        common = deepcopy(self.parent)
        common.pop("variant")
        return {"task": "relay", "seed": 0, "model_updates": 4500, "original_target_stage_updates": 5000,
                "common_config_sha256": canonical_hash(common), "checkpoint_path": "initial.pt",
                "checkpoint_sha256": "a"*64, "selection_path": "selection.json", "selection_sha256": "b"*64}

    def schedule_manifest(self):
        row = self.selected_row()
        schedule, binding = Schedule.from_initial_selection(self.settings, 0, "selection.json",
                                                            study_amendment=self.amendment)
        return {"settings": self.settings, "master_seed": 0, "study_amendment": self.amendment,
                "parent_schedule": Schedule.from_experiment(self.parent).describe(),
                "registered_schedule": Schedule.from_experiment(self.settings).describe(),
                "schedule": schedule.describe(), "initial_selection": binding,
                "initial_checkpoint": row["checkpoint_path"], "initial_checkpoint_sha256": row["checkpoint_sha256"],
                "initial_training_policy": "frozen_last_saved_checkpoints", "study_scope": "exploratory_pilot"}

    def test_exact_scope_budgets_and_only_permitted_fields_change(self):
        self.assertEqual(self.parent, resolve_config("relay", "B", "pilot").settings)
        changed = {(section, key) for section, values in self.parent.items() if isinstance(values, dict)
                   for key in values if values[key] != self.settings[section][key]}
        self.assertEqual(changed, {("experiment", "real_budget"), ("experiment", "checkpoints"),
                                  ("experiment", "checkpoint_episodes"), ("experiment", "final_episodes"),
                                  ("world_model", "online_updates"), ("mappo", "initial_local_cycles")})
        schedule = Schedule.from_experiment(self.settings)
        self.assertEqual(schedule.blocks, 2)
        self.assertEqual(schedule.total_local_cycles, 80)
        self.assertEqual(schedule.describe()["block_targets"], [50000, 75000, 100000])
        self.assertEqual((schedule.initial_model_updates, schedule.online_model_updates), (5000, 250))
        self.assertEqual(self.settings["experiment"]["checkpoints"], [0, 50000, 75000, 100000])
        self.assertEqual((self.settings["mappo"]["horizon"], self.settings["mappo"]["starts"]), (15, 256))
        self.assertEqual(self.amendment["task_families"], ["relay"])
        self.assertEqual(self.amendment["master_seeds"], [0, 1])

    def test_b_and_d_share_all_runtime_budgets_but_d_treatment_is_not_claimed_implemented(self):
        other, binding = apply_study_profile(resolve_config("relay", "D", "pilot").settings, "relay_week_v1", master_seed=1)
        validate_pair(ResolvedConfig(self.settings, {}, "", {}), ResolvedConfig(other, {}, "", {}))
        self.assertEqual(Schedule.from_experiment(self.settings), Schedule.from_experiment(other))
        self.assertEqual(binding["planned_d_treatment"], self.amendment["planned_d_treatment"])
        self.assertFalse(binding["d_training_implemented"])
        treatment = binding["planned_d_treatment"]
        self.assertEqual((treatment["teacher_initial_cycles"], treatment["teacher_online_cycles"]), (50, 0))
        self.assertEqual((treatment["guided_local_cycles"], treatment["unguided_local_cycles"]), (0, 80))
        self.assertEqual(treatment["auxiliary_kl_weight"], 0.)
        self.assertEqual(treatment["trajectory_source"], "teacher")
        self.assertTrue(treatment["ppo_optimizer_reset_after_distillation"])
        self.assertEqual(binding["planned_evaluation"]["split"], "test")
        self.assertEqual(binding["planned_evaluation"]["bank_seed"], 5600)
        self.assertEqual(binding["planned_evaluation"]["implementation_status"], "pending_phase5")
        self.assertEqual(binding["planned_analysis"]["bootstrap_replicates"], 0)
        self.assertEqual(binding["planned_analysis"]["training_seed_pairs"], 2)
        self.assertFalse(binding["planned_analysis"]["confidence_intervals"])
        self.assertEqual(other["experiment"]["evaluation_split"], "validation")

    def test_unapproved_task_seed_parent_or_profile_cannot_enter_amendment(self):
        for task, profile, seed in (("cover", "pilot", 0), ("relay", "pilot", 2),
                                    ("relay", "pilot", True), ("relay", "final", 1001)):
            with self.subTest(task=task, profile=profile, seed=seed), self.assertRaises(ValueError):
                apply_study_profile(resolve_config(task, "B", profile).settings, "relay_week_v1", master_seed=seed)
        changed = deepcopy(self.parent)
        changed["mappo"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "unchanged"):
            apply_study_profile(changed, "relay_week_v1")
        with self.assertRaisesRegex(ValueError, "unknown"):
            apply_study_profile(self.parent, "unregistered")

    def test_amendment_file_edit_or_extra_architecture_override_is_rejected(self):
        from world_models.cycle import study
        original = study._read(study.ROOT / "configs/studies/relay_week_v1.yaml")
        for section, key, value in (("world_model", "online_updates", 251), ("world_model", "width", 128)):
            changed = deepcopy(original)
            changed["overrides"][section][key] = value
            with patch.object(study, "_read", return_value=changed), self.assertRaisesRegex(ValueError, "fixed"):
                apply_study_profile(self.parent, "relay_week_v1")

    def test_receipt_or_active_settings_tamper_is_rejected(self):
        self.assertEqual(parent_settings_for_study(self.settings, self.amendment, master_seed=0), self.parent)
        for key, value in (("profile_sha256", "f"*64), ("parent_settings_sha256", "e"*64),
                           ("d_training_implemented", True), ("master_seeds", [0, 1, 2])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                parent_settings_for_study(self.settings, {**self.amendment, key: value}, master_seed=0)
        changed = deepcopy(self.settings)
        changed["world_model"]["online_updates"] = 251
        with self.assertRaises(ValueError):
            parent_settings_for_study(changed, self.amendment, master_seed=0)

    def test_selection_compares_original_parent_and_preserves_actual_saved_count(self):
        row = self.selected_row()
        with patch("world_models.training.selection.resolve_selected_case", return_value=row):
            schedule, binding = Schedule.from_initial_selection(self.settings, 0, "selection.json",
                                                                study_amendment=self.amendment)
            self.assertEqual((schedule.initial_model_updates, schedule.total_model_updates), (4500, 5000))
            self.assertEqual(binding["model_updates"], 4500)
            with self.assertRaisesRegex(ValueError, "configuration"):
                Schedule.from_initial_selection(self.settings, 0, "selection.json")

    def test_resume_revalidates_parent_selection_and_amendment_without_resetting_schedule(self):
        with patch("world_models.training.selection.resolve_selected_case", return_value=self.selected_row()):
            manifest = self.schedule_manifest()
            resumed = Schedule.from_manifest(manifest)
            self.assertEqual(resumed.total_local_cycles, 80)
            self.assertEqual(resumed.initial_model_updates, 4500)
            for changes in ({"study_amendment": None}, {"initial_selection": None},
                            {"parent_schedule": manifest["registered_schedule"]},
                            {"schedule": Schedule.from_experiment(self.parent).describe()},
                            {"master_seed": 2}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    Schedule.from_manifest({**manifest, **changes})

    def test_cli_plan_is_read_only_and_reports_parent_separately(self):
        output = io.StringIO()
        with patch("world_models.training.selection.resolve_selected_case", return_value=self.selected_row()), \
             patch.object(phase35, "_runtime_setup", side_effect=AssertionError("allocated runtime")), \
             patch.object(phase35, "load_case", side_effect=AssertionError("loaded replay")), redirect_stdout(output):
            code = phase35.main(["plan", "--task", "relay", "--seed", "0", "--study-profile", "relay_week_v1",
                                 "--initial-selection", "selection.json"])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(report["real_budget"], 100000)
        self.assertEqual(report["parent_schedule"]["real_budget"], 200000)
        self.assertEqual(report["total_local_cycles"], 80)
        self.assertFalse(report["training_executed"])

    def test_cli_prepare_loads_parent_replay_then_passes_explicit_amended_settings(self):
        prepared = SimpleNamespace(directory=Path("new-run"), schedule=Schedule.from_experiment(self.settings))
        data, stats = object(), object()
        with patch.object(phase35, "_runtime_setup"), \
             patch.object(phase35, "load_case", return_value=(SimpleNamespace(settings=self.parent), [data], stats, {})) as load, \
             patch.object(phase35.CycleController, "prepare", return_value=prepared) as prepare, redirect_stdout(io.StringIO()):
            phase35.main(["prepare", "--task", "relay", "--seed", "0", "--study-profile", "relay_week_v1",
                          "--initial-selection", "selection.json", "--run-dir", "new-run", "--device", "cpu"])
        self.assertEqual(load.call_args.kwargs["profile"], "pilot")
        self.assertEqual(prepare.call_args.kwargs["settings"], self.settings)
        self.assertEqual(prepare.call_args.kwargs["study_amendment"], self.amendment)
        self.assertIs(prepare.call_args.kwargs["initial_dataset"], data)

    def test_cli_rejects_missing_selection_cover_final_or_seed_before_runtime(self):
        cases = (("plan", "relay", "0", "pilot", []),
                 ("prepare", "relay", "0", "pilot", ["--initial-checkpoint", "initial.pt", "--run-dir", "run"]),
                 ("plan", "cover", "0", "pilot", ["--initial-selection", "selection.json"]),
                 ("plan", "relay", "2", "pilot", ["--initial-selection", "selection.json"]),
                 ("plan", "relay", "0", "final", ["--initial-selection", "selection.json"]))
        for command, task, seed, profile, extra in cases:
            with self.subTest(command=command, task=task, seed=seed, profile=profile), redirect_stderr(io.StringIO()), \
                 patch.object(phase35, "_runtime_setup", side_effect=AssertionError("allocated runtime")), \
                 patch.object(phase35, "load_case", side_effect=AssertionError("loaded replay")), self.assertRaises(SystemExit):
                phase35.main([command, "--task", task, "--seed", seed, "--profile", profile,
                              "--study-profile", "relay_week_v1", *extra])

    def test_controller_refuses_amendment_with_unselected_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "frozen initial selection"):
            CycleController.prepare("unused", settings=self.settings, master_seed=0, initial_dataset=object(),
                                    normalization=object(), initial_checkpoint="initial.pt", device="cpu",
                                    study_amendment=self.amendment, options=CycleOptions(precision="float32"))


if __name__ == "__main__":
    unittest.main()
