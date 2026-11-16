"""P5.1 exact static choices, accounting and existing RNG compatibility."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from world_models.infrastructure.reproducibility import canonical_hash, derive_seed
from world_models_study import protocol


class StudyProtocolTests(unittest.TestCase):
    def test_exact_protocol_preserves_parent_and_returns_owned_values(self):
        p = protocol.load_protocol()
        self.assertEqual(p["study_profile"], "relay_endpoint_protocol_v1")
        self.assertEqual(p["distillation_trajectory_source"], "teacher")
        self.assertEqual(p["local_actor_compute_precision"], "float32")
        self.assertEqual(p["teacher_policy_compute_precision"], "float32")
        self.assertEqual(p["world_model_compute_precision"], "bfloat16")
        parent = protocol.ROOT / p["parent_profile_path"]
        self.assertEqual(hashlib.sha256(parent.read_bytes()).hexdigest(), p["parent_profile_sha256"])
        self.assertEqual(protocol.protocol_identity(), canonical_hash(p))
        p["master_seeds"].append(2)
        p["analysis"]["confidence_intervals"] = True
        self.assertEqual(protocol.load_protocol()["master_seeds"], [0, 1])
        self.assertFalse(protocol.load_protocol()["analysis"]["confidence_intervals"])

    def test_changed_unknown_mistyped_or_duplicate_yaml_keys_are_rejected(self):
        original = protocol.PROFILE.read_text(encoding="utf-8")
        documents = []
        for mutate in (lambda p: p.update(local_cycles=19), lambda p: p.update(initial_transitions=True),
                       lambda p: p.update(unknown=True), lambda p: p["analysis"].update(p_values=True),
                       lambda p: p["capabilities"].update(execution_ready=True)):
            p = protocol.load_protocol()
            mutate(p)
            documents.append(yaml.safe_dump(p))
        documents.append(original + "\nlocal_cycles: 21\n")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.yaml"
            for index, document in enumerate(documents):
                path.write_text(document, encoding="utf-8")
                with self.subTest(index=index), patch.object(protocol, "PROFILE", path):
                    with self.assertRaises(ValueError):
                        protocol.load_protocol()

    def test_parent_profile_byte_change_cannot_be_silently_rebased(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "configs/studies/relay_four_day_v1.yaml"
            parent.parent.mkdir(parents=True)
            parent.write_bytes((protocol.ROOT / "configs/studies/relay_four_day_v1.yaml").read_bytes() + b"\n")
            with patch.object(protocol, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "parent profile bytes"):
                    protocol.load_protocol()

    def test_plan_counts_joint_transitions_once_and_separates_attribution(self):
        result = protocol.plan()
        matrix, budget = result["run_matrix"], result["budgets"]
        self.assertEqual([(row["master_seed"], row["variant"]) for row in matrix],
                         [(0, "B"), (0, "D"), (1, "B"), (1, "D")])
        self.assertTrue(all(row["local_actor_optimizer_steps"] == row["local_critic_optimizer_steps"] == 320 for row in matrix))
        self.assertTrue(all(row["initial_transitions"] == 50000 and row["initial_model_updates"] == 4500 for row in matrix))
        self.assertEqual(budget["physical_existing_initial_joint_transitions"], 100000)
        self.assertEqual(budget["attributed_initial_joint_transitions"], sum(row["initial_transitions"] for row in matrix))
        self.assertEqual(budget["physical_selected_initial_model_updates"], 9000)
        self.assertEqual(budget["attributed_selected_initial_model_updates"], 18000)
        for key in ("local_actor_optimizer_steps", "local_critic_optimizer_steps", "teacher_actor_optimizer_steps",
                    "teacher_critic_optimizer_steps", "distillation_optimizer_steps"):
            self.assertEqual(budget[key], sum(row[key] for row in matrix))
        self.assertEqual((budget["total_local_cycles"], budget["total_teacher_cycles"]), (80, 50))
        self.assertEqual(budget["combined_imagined_joint_step_cap_excluding_prefix_reconstruction"], 503040)
        self.assertEqual(budget["final_evaluation_episodes"], 80)
        self.assertEqual(budget["final_evaluation_joint_decision_cap_excluding_resets"], 32000)
        for key in ("execution_ready", "cohort_frozen", "training_executed", "native_evaluation_executed", "phase5_acceptance"):
            self.assertIs(result[key], False)

    def test_analysis_retry_and_development_disclosures_remain_bounded(self):
        p = protocol.load_protocol()
        self.assertEqual(p["analysis"]["primary_effect"], "final_success_rate_D_minus_B")
        for key in ("confidence_intervals", "p_values", "bootstrap", "auc", "superiority_claim", "equivalence_claim"):
            self.assertIs(p["analysis"][key], False)
        self.assertEqual(p["retry_policy"]["infrastructure_retries_max"], 2)
        for key in ("retry_valid_poor_results", "retry_numerical_failures", "seed_replacement", "checkpoint_selection_from_test_results"):
            self.assertIs(p["retry_policy"][key], False)
        self.assertEqual(p["fallback_policy"]["single_pair_seed"], 0)
        self.assertTrue(p["fallback_policy"]["decision_before_test_outcomes"])
        self.assertTrue(p["fallback_policy"]["decision_before_comparative_training_and_outcome_inspection"])
        self.assertFalse(p["fallback_policy"]["automatic_fallback"])
        self.assertTrue(p["disclosures"]["adopted_development_work_is_not_a_fresh_cohort"])
        self.assertTrue(p["disclosures"]["D_extra_compute_is_not_compute_matched"])

    def test_paired_initialization_and_distinct_branch_namespaces(self):
        for seed in (0, 1):
            b, d = (protocol.training_seed_manifest(seed, branch) for branch in ("B", "D"))
            self.assertEqual(b["paired_initialization"], d["paired_initialization"])
            self.assertNotEqual(b["local"], d["local"])
            self.assertIsNone(b["teacher"])
            self.assertIsNone(b["distillation"])
            for stage in ("teacher", "distillation"):
                for component, value in d[stage]["rng_namespaces"].items():
                    self.assertEqual(value, derive_seed("relay", seed, component, "D"))
            seeds = list(d["local"]["rollout_seeds"].values())
            self.assertEqual(len(seeds), len(set(seeds)))
        for seed, variant in ((True, "B"), (-1, "D"), (2, "D"), (0, "A"), (0, None)):
            with self.assertRaises(ValueError):
                protocol.training_seed_manifest(seed, variant)

    def test_declared_rollout_components_match_existing_real_stream_factories(self):
        from world_models.infrastructure.config import resolve_config
        from world_models.rollouts.streams import RolloutStreams
        for seed in (0, 1):
            for variant in ("B", "D"):
                manifest = protocol.training_seed_manifest(seed, variant)
                settings = resolve_config("relay", variant, "pilot").settings
                current = RolloutStreams.from_experiment(settings, master_seed=seed)
                self.assertEqual(manifest["local"]["rollout_seeds"], current.identity["seeds"])
                if variant == "D":
                    for stage, component in (("teacher", "teacher_rollouts"), ("distillation", "distillation_rollouts")):
                        real = RolloutStreams.from_seed(manifest[stage]["rng_namespaces"][component])
                        self.assertEqual(manifest[stage]["rollout_seeds"], real.identity["seeds"])

    def test_plan_import_and_execution_do_not_load_torch_or_simulator(self):
        code = ("import sys; from world_models_study.protocol import plan; p=plan(); "
                "assert not p['execution_ready']; assert 'torch' not in sys.modules; "
                "assert 'vizdoom' not in sys.modules; assert 'world_models.env.coop' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
