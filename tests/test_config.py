"""Research-contract tests for configuration composition and B/D fairness."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import yaml

from world_models.infrastructure.config import (
    ConfigError,
    resolve_config,
    validate_config,
    validate_pair,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class ConfigurationTests(unittest.TestCase):
    def resolve(self, task="relay", variant="B", experiment="final", config_dir=None):
        return resolve_config(task, variant, experiment, config_dir or CONFIG_DIR)

    def altered(self, section, key, value):
        settings = deepcopy(self.resolve().settings)
        settings[section][key] = value
        return settings

    def test_all_shipped_task_variant_experiment_combinations_validate(self):
        for task in ("relay", "cover"):
            for experiment in ("pilot", "final"):
                with self.subTest(task=task, experiment=experiment):
                    b = self.resolve(task, "B", experiment)
                    d = self.resolve(task, "D", experiment)
                    validate_pair(b, d)
                    validate_pair(d, b)

    def test_final_schedule_matches_approved_interaction_and_local_learning_budget(self):
        for variant in ("B", "D"):
            with self.subTest(variant=variant):
                cfg = self.resolve(variant=variant)
                self.assertEqual(cfg.derived["online_blocks"], 38)
                self.assertEqual(cfg.derived["local_ppo_cycles"], 860)
                self.assertEqual(cfg.derived["comparison_run_count"], 40)
                self.assertEqual(cfg.derived["checkpoint_count"], 12)
                self.assertEqual(cfg.derived["nominal_local_imagined_joint_steps"], 3_302_400)
        b = self.resolve()
        d = self.resolve(variant="D")
        self.assertEqual(b.derived["teacher_ppo_cycles"], 0)
        self.assertEqual(b.derived["guided_local_cycles"], 0)
        self.assertEqual(d.derived["guided_local_cycles"], 645)
        self.assertEqual(d.derived["unguided_local_cycles"], 215)

    def test_pilot_schedule_scales_guidance_to_its_own_local_budget(self):
        for variant in ("B", "D"):
            with self.subTest(variant=variant):
                cfg = self.resolve(variant=variant, experiment="pilot")
                self.assertEqual(cfg.derived["online_blocks"], 6)
                self.assertEqual(cfg.derived["local_ppo_cycles"], 220)
                self.assertEqual(cfg.derived["comparison_run_count"], 8)
                self.assertEqual(cfg.settings["experiment"]["evaluation_split"], "validation")
        d = self.resolve(variant="D", experiment="pilot")
        self.assertEqual(d.derived["guided_local_cycles"], 165)
        self.assertEqual(d.derived["unguided_local_cycles"], 55)

    def test_pair_rejects_legitimate_but_different_common_settings(self):
        b = self.resolve()
        d = self.resolve(variant="D")
        mutations = (
            ("actor", "hidden_size", 512),
            ("world_model", "memory_tokens", 32),
            ("critic", "hidden_sizes", [1024, 512]),
            ("mappo", "online_local_cycles", 40),
        )
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                settings = deepcopy(d.settings)
                settings[section][key] = value
                validate_config(settings)  # The defect is the unequal pair, not this value.
                with self.assertRaises(ConfigError):
                    validate_pair(b, replace(d, settings=settings))
        settings = deepcopy(d.settings)
        settings["experiment"]["real_budget"] = 1_025_000
        settings["experiment"]["checkpoints"].append(1_025_000)
        validate_config(settings)
        with self.assertRaises(ConfigError):
            validate_pair(b, replace(d, settings=settings))

    def test_pair_requires_both_distinct_variants(self):
        b = self.resolve()
        with self.assertRaises(ConfigError):
            validate_pair(b, b)

    def test_overlay_cannot_override_common_settings(self):
        cases = (
            ("variants/D.yaml", "actor", {"hidden_size": 512}),
            ("tasks/relay.yaml", "world_model", {"memory_tokens": 32}),
            ("experiments/final.yaml", "mappo", {"online_local_cycles": 40}),
        )
        for relative, section, settings in cases:
            with self.subTest(profile=relative), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "configs"
                shutil.copytree(CONFIG_DIR, root)
                path = root / relative
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                payload[section] = settings
                path.write_text(yaml.safe_dump(payload), encoding="utf-8")
                with self.assertRaises(ConfigError):
                    self.resolve(variant="D", config_dir=root)

    def test_unknown_and_duplicate_yaml_keys_are_not_silently_accepted(self):
        for suffix in ("\nunknown_section: true\n", "\nschema_version: 1\n"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "configs"
                shutil.copytree(CONFIG_DIR, root)
                path = root / "base.yaml"
                path.write_text(path.read_text(encoding="utf-8") + suffix, encoding="utf-8")
                with self.assertRaises(ConfigError):
                    self.resolve(config_dir=root)

    def test_duplicate_nested_yaml_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "configs"
            shutil.copytree(CONFIG_DIR, root)
            path = root / "variants" / "D.yaml"
            path.write_text(
                "variant:\n  name: D\n  teacher_enabled: true\n"
                "  teacher_enabled: false\n  distillation_enabled: true\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                self.resolve(variant="D", config_dir=root)

    def test_profile_names_must_agree_with_selected_filenames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "configs"
            shutil.copytree(CONFIG_DIR, root)
            path = root / "variants" / "D.yaml"
            path.write_text(
                "variant:\n  name: B\n  teacher_enabled: false\n"
                "  distillation_enabled: false\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                self.resolve(variant="D", config_dir=root)

    def test_unknown_config_fields_and_missing_fields_are_rejected(self):
        settings = deepcopy(self.resolve().settings)
        settings["environment"]["hidden_global_features"] = True
        with self.assertRaises(ConfigError):
            validate_config(settings)
        settings = deepcopy(self.resolve().settings)
        del settings["environment"]["separate_actor_memory"]
        with self.assertRaises(ConfigError):
            validate_config(settings)

    def test_locality_privilege_and_snapshot_rules_are_contractual(self):
        cases = (
            ("environment", "actor_global_input", True),
            ("environment", "actor_reward_input", True),
            ("environment", "actor_agent_identity", True),
            ("environment", "communication", True),
            ("environment", "prior_future_truth", True),
            ("environment", "separate_actor_memory", False),
            ("environment", "privileged_mode", "encoder_input"),
            ("environment", "local_actor_inputs", ["local_rgb", "global_state"]),
            ("critic", "actor_memories", False),
            ("critic", "remaining_time", False),
            ("mappo", "differentiate_world_model", True),
            ("mappo", "frozen_rollout_snapshots", False),
            ("teacher", "recurrent_input", "global_history"),
            ("distillation", "trajectory_source", "teacher"),
            ("study", "evaluation_enters_replay", True),
        )
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                with self.assertRaises(ConfigError):
                    validate_config(self.altered(section, key, value))

    def test_b_and_d_switches_must_match_the_selected_variant(self):
        for variant, flag, value in (
            ("B", "teacher_enabled", True),
            ("B", "distillation_enabled", True),
            ("D", "teacher_enabled", False),
            ("D", "distillation_enabled", False),
        ):
            with self.subTest(variant=variant, flag=flag):
                settings = deepcopy(self.resolve(variant=variant).settings)
                settings["variant"][flag] = value
                with self.assertRaises(ConfigError):
                    validate_config(settings)

    def test_invalid_budgets_and_checkpoint_grids_are_rejected(self):
        cases = (
            ("real_budget", 50_000),
            ("real_budget", 1_000_001),
            ("collection_block", 0),
            ("checkpoints", [0, 50_000, 50_000, 1_000_000]),
            ("checkpoints", [50_000, 1_000_000]),
            ("checkpoints", [0, 100_000, 1_000_000]),
            ("checkpoints", [0, 50_000, 60_000, 1_000_000]),
            ("checkpoints", [0, 50_000, 500_000]),
            ("master_seeds", [1001, 1001]),
            ("final_episodes", 1),
            ("evaluation_split", "validation"),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ConfigError):
                    validate_config(self.altered("experiment", key, value))

    def test_numeric_types_and_nonfinite_settings_are_rejected(self):
        cases = (
            ("environment", "agents", True),
            ("mappo", "horizon", 15.0),
            ("mappo", "gamma", True),
            ("mappo", "learning_rate", float("nan")),
            ("world_model", "gradient_clip", float("inf")),
            ("actor", "hidden_size", "256"),
            ("experiment", "master_seeds", [True]),
        )
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                with self.assertRaises(ConfigError):
                    validate_config(self.altered(section, key, value))

    def test_cross_field_model_and_sequence_shapes_are_validated(self):
        cases = (
            ("world_model", "width", 257),
            ("world_model", "burn_in", 64),
            ("world_model", "sequence_length", 0),
            ("mappo", "recurrent_minibatches", 3),
            ("environment", "image_shape", [3, 128, 96]),
            ("environment", "previous_action_start", 0),
            ("environment", "frame_skip", 0),
        )
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                with self.assertRaises(ConfigError):
                    validate_config(self.altered(section, key, value))

    def test_input_names_are_guarded_before_reading_paths(self):
        cases = (("../relay", "B", "final"), ("relay", "A", "final"),
                 ("relay", "D", "../../other"))
        for task, variant, experiment in cases:
            with self.subTest(task=task, variant=variant, experiment=experiment):
                with patch("world_models.infrastructure.config._read") as read:
                    with self.assertRaises(ConfigError):
                        resolve_config(task, variant, experiment, CONFIG_DIR)
                    read.assert_not_called()

    def test_hashes_are_stable_and_source_hashes_preserve_provenance(self):
        original = self.resolve()
        repeated = self.resolve()
        self.assertEqual(original.config_hash, repeated.config_hash)
        self.assertEqual(original.source_hashes, repeated.source_hashes)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "configs"
            shutil.copytree(CONFIG_DIR, root)
            path = root / "base.yaml"
            path.write_text(path.read_text(encoding="utf-8") + "\n# provenance-only edit\n", encoding="utf-8")
            edited = self.resolve(config_dir=root)
            self.assertEqual(original.config_hash, edited.config_hash)
            self.assertNotEqual(original.source_hashes["base.yaml"], edited.source_hashes["base.yaml"])

    def test_serialized_manifest_is_a_defensive_copy(self):
        cfg = self.resolve()
        exported = cfg.as_dict()
        exported["settings"]["actor"]["hidden_size"] = -1
        exported["derived"]["local_ppo_cycles"] = -1
        self.assertGreater(cfg.settings["actor"]["hidden_size"], 0)
        self.assertEqual(cfg.derived["local_ppo_cycles"], 860)


if __name__ == "__main__":
    unittest.main()
