"""Full P5.3-P5.5 handoff with tiny saved actors and synthetic engine states.

No learner performs optimization and no native environment is opened. This
tests the actual cohort receipts, final actor loader, fixed-bank evaluator,
scientific rejection boundary and paired aggregation together.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tests.test_endpoint_cohort import CohortFixture, rewrite
from tests.test_endpoint_evaluation import final_scenario
from tests.test_replay_collection import environment_fixture
from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_endpoint import analysis, cohort, evaluation, ledger


class EndpointPipelineTests(CohortFixture):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.object(evaluation, "source_identity", return_value=self.sources))
        self.enterContext(patch.object(analysis, "source_identity", return_value=self.sources))
        # Bind small seed-specific placeholder replay manifests. No dataset is
        # read by these mocked learners and no 50k research samples are created.
        for seed in (0, 1):
            replay = self.root / f"initial_seed_{seed}" / "replay"
            replay.mkdir(parents=True)
            manifest = replay / "manifest.json"
            manifest.write_text(f'{{"synthetic_seed": {seed}}}', encoding="utf-8")
            row = self.inputs["per_seed"][str(seed)]
            row["initial_binding"]["replay_path"] = str(replay)
            row["initial_binding"]["provenance"]["replays"][0]["manifest_sha256"] = file_sha256(manifest)
            row["initial_binding_sha256"] = canonical_hash(row["initial_binding"])
            self.inputs["bound_files"].append({"path": str(manifest), "sha256": file_sha256(manifest)})
        self.inputs["physical_initial_joint_transitions"] = 100000
        self.inputs["physical_selected_model_optimizer_updates"] = 9000
        self.parent["inputs"] = self.inputs
        rewrite(self.parent_path, self.parent)
        self.matrix["registration"]["sha256"] = file_sha256(self.parent_path)
        self.matrix["inputs_sha256"] = canonical_hash(self.inputs)
        rewrite(self.matrix_path, self.matrix)

    def test_frozen_four_run_actor_evaluation_accounting_and_paired_analysis_pipeline(self):
        frozen = self.freeze()
        self.fake_learners()
        for row in frozen["run_matrix"]:
            cohort.prepare_run(frozen["report_path"], variant=row["variant"], seed=row["master_seed"], device="cpu")
        for row in frozen["run_matrix"]:
            result = cohort.resume_run(row["run_dir"], device="cpu").run()
            self.assertEqual(result["status"], "complete")
        snapshots = ledger.summarize_cohort(frozen["report_path"])
        self.assertEqual(snapshots["status"], "complete")
        self.assertEqual(snapshots["shared_initial_costs"]["physical_joint_transitions"], 100000)
        self.assertEqual(snapshots["shared_initial_costs"]["configured_four_run_attributed_joint_transitions"], 200000)
        self.assertEqual(snapshots["sum_run_logical_counters"]["local_PPO_cycles"], 80)
        self.assertEqual(snapshots["sum_run_logical_counters"]["local_actor_steps"], 1280)
        self.assertEqual(snapshots["sum_run_logical_counters"]["teacher_PPO_cycles"], 50)
        self.assertEqual(snapshots["sum_run_logical_counters"]["distillation_optimizer_steps"], 200)
        self.assertEqual(snapshots["sum_run_logical_counters"]["new_environment_interactions"], 0)
        self.assertEqual(snapshots["sum_run_logical_counters"]["new_model_updates"], 0)
        reports, paths = [], []
        for row in frozen["run_matrix"]:
            bound = cohort.final_actor(row["run_dir"])
            output = self.root / "fixture_evaluations" / row["variant"] / f"seed_{row['master_seed']}"
            report = evaluation.evaluate_endpoint(row["run_dir"], output, device="cpu",
                final_binding=bound, env_factory=environment_fixture, scenario_builder=final_scenario)
            self.assertFalse(report["native_engine"])
            self.assertEqual(report["final_actor_identity"], canonical_hash(bound["binding"]))
            self.assertEqual(report["cohort_identity"], frozen["cohort_identity"])
            self.assertEqual(report, evaluation.load_evaluation(output / "evaluation.json",
                final_binding=bound, require_native=False))
            with self.assertRaisesRegex(ValueError, "synthetic fixture"):
                evaluation.load_evaluation(output / "evaluation.json", require_native=True)
            reports.append(report)
            paths.append(output / "evaluation.json")
        result = analysis.aggregate_records(reports, bank=frozen["bank"],
            cohort_identity=frozen["cohort_identity"], protocol_sha256=frozen["protocol_sha256"], require_native=False)
        self.assertEqual(result["evidence_kind"], "synthetic_fixture")
        self.assertEqual(result["total_logical_episodes"], 80)
        self.assertEqual(result["evaluation_device_backend"], "cpu")
        self.assertEqual([p["success_rate_difference_D_minus_B"] for p in result["paired_differences"]], [0.0, 0.0])
        self.assertEqual([p["mean_return_difference_D_minus_B"] for p in result["paired_differences"]], [0.0, 0.0])
        self.assertEqual([p["latch_episode_rate_difference_D_minus_B"] for p in result["paired_differences"]], [0.0, 0.0])
        # The public scientific reader always requires authentic native flags,
        # even though all four cohort/actor/trace receipt chains are otherwise valid.
        science = analysis.analyze_cohort(frozen["report_path"], paths, self.root / "science_rejection")
        self.assertEqual(science["status"], "needs_review")
        self.assertFalse(science["cohort_acceptance"])
        self.assertIsNone(science["analysis"])
        self.assertTrue(any("synthetic fixture" in reason for reason in science["reasons"]))
        # Synthetic counted physics is diagnostic test work, never imported as
        # native evaluation consumption or learning data in the research ledger.
        after = ledger.summarize_cohort(frozen["report_path"])
        self.assertEqual(after["evaluation_joint_transitions"], 0)
        self.assertFalse(after["evaluation_enters_training"])
        self.assertFalse(any((Path(row["run_dir"]) / "endpoint_test_claim.json").exists()
                             for row in frozen["run_matrix"]))
