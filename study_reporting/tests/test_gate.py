"""CPU-only, metadata fixtures for the P5.8 completion gate."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from study_reporting.gate import AUTHENTICATION, CASES, audit_retries, build_acceptance


def identity(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


class AcceptanceFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.cohort_id = identity("cohort")
        self.protocol = identity("protocol")
        self.bank = identity("bank")
        self.source = {"study_reporting/gate.py": identity("gate")}
        self.frozen = {"old.py": identity("old")}
        self.context = {
            "authentication": {name: True for name in AUTHENTICATION},
            "source_identity": self.source,
            "figure_evidence": {"fixtures_only": False, "actual_graphics_rendered": True, "matplotlib_version": "3.11.2"},
            "references": {"cohort": {"path": str(self.root / "cohort.json"), "sha256": identity("cohortfile")}},
            "deliverables": {"secondary": {"path": "secondary.json", "sha256": identity("secondary")},
                             "resources": {"path": "resources.json", "sha256": identity("resources")},
                             "figures": [{"path": "endpoints.svg", "sha256": identity("figure1")},
                                         {"path": "effect_compute.svg", "sha256": identity("figure2")}]},
            "cohort": {"cohort_identity": self.cohort_id, "protocol_sha256": self.protocol,
                       "bank_sha256": self.bank, "source_sha256": self.frozen,
                       "status": "cohort_frozen", "execution_ready": True, "candidate_only": False,
                       "run_matrix": [{"variant": v, "master_seed": s} for v, s in CASES]},
            "ledger": {"cohort_identity": self.cohort_id, "status": "complete", "missing_run_slots": [],
                       "evaluation_enters_training": False, "runs": {},
                       "shared_initial_costs": {"physical_joint_transitions": 100000,
                           "configured_four_run_attributed_joint_transitions": 200000,
                           "physical_selected_model_prefix_optimizer_steps": 9000,
                           "configured_four_run_attributed_selected_model_steps": 18000}},
            "runs": [], "evaluation_reports": []}
        for variant, seed in CASES:
            directory = self.root / "runs" / variant / f"seed_{seed}"
            directory.mkdir(parents=True)
            pairing = {name: identity(f"{seed}/{name}") for name in ("actor_cpu", "critic_cpu", "input_binding_sha256",
                "world_model_checkpoint_sha256", "normalization_identity_sha256", "replay_manifest_sha256")}
            run = {"variant": variant, "master_seed": seed, "run_dir": str(directory),
                   "run_identity": identity(f"{variant}/{seed}"), "last_checkpoint": str(directory / "checkpoint"),
                   "status": "complete", "phase": "complete", "training_complete": True,
                   "cohort_identity": self.cohort_id, "actor_export": str(directory / "actor_export"),
                   "local_cycles_completed": 20, "local_actor_optimizer_steps": 320, "local_critic_optimizer_steps": 320,
                   "real_joint_transitions": 50000, "model_updates": 4500, "new_environment_interactions": 0,
                   "new_model_updates": 0, "teacher_cycles_completed": 25 if variant == "D" else 0,
                   "teacher_actor_optimizer_steps": 400 if variant == "D" else 0,
                   "teacher_critic_optimizer_steps": 400 if variant == "D" else 0,
                   "trajectories_completed": 128 if variant == "D" else 0,
                   "distillation_optimizer_steps": 100 if variant == "D" else 0,
                   "imagined_joint_steps_local": 70000, "imagined_joint_steps_teacher": 80000 if variant == "D" else 0,
                   "imagined_joint_steps_buffer": 1700 if variant == "D" else 0,
                   "run_binding": {"run_kind": "prospective_paired_cohort", "historical_adoption": False,
                                   "cohort_identity": self.cohort_id, "variant": variant, "master_seed": seed,
                                   "pairing": pairing}}
            self.context["runs"].append(run)
            accounting = {"infrastructure_retries": 0, "training_interactions_added": 0, "learning_eligible": False}
            report = {"variant": variant, "master_seed": seed, "status": "complete",
                      "cohort_identity": self.cohort_id, "protocol_sha256": self.protocol, "bank_sha256": self.bank,
                      "fixture_only": False, "final_local_cycles": 20,
                      "final_actor_identity": identity(f"final/{variant}/{seed}"),
                      "episodes": [{"episode_index": index} for index in range(20)],
                      "attempts": [{"path": f"episodes/{index:06d}/attempt_00/attempt.json", "sha256": identity(index)} for index in range(20)],
                      "interaction_accounting": accounting,
                      **{name: True for name in ("native_backend", "native_engine", "native_environment_evaluation_executed", "research_evaluation_executed")},
                      **{name: False for name in ("world_model_loaded", "critic_loaded", "evaluation_data_enters_training", "learning_eligible")},
                      **{name: 0 for name in ("world_model_updates", "policy_optimizer_updates", "real_ppo_updates", "training_interactions_added")}}
            self.context["evaluation_reports"].append(report)
            physical = {}
            for stage in (("local_cycle",) if variant == "B" else ("teacher", "buffer", "distillation", "local_cycle", "controller")):
                physical[stage] = self.journal(directory, stage, [
                    {"event": "unit_started", "attempt": f"{stage}/one", "invocation": "first"},
                    {"event": "unit_committed", "attempt": f"{stage}/one", "invocation": "first"}])
            counters = {"local_PPO_cycles": 20, "local_actor_steps": 320, "local_critic_steps": 320,
                        "teacher_PPO_cycles": run["teacher_cycles_completed"], "teacher_actor_steps": run["teacher_actor_optimizer_steps"],
                        "teacher_critic_steps": run["teacher_critic_optimizer_steps"], "warm_start_buffer_trajectories": run["trajectories_completed"],
                        "distillation_optimizer_steps": run["distillation_optimizer_steps"], "world_model_optimizer_steps_initial": 4500,
                        "new_environment_interactions": 0, "new_model_updates": 0}
            self.context["ledger"]["runs"][f"{variant}/seed_{seed}"] = {
                "status": "complete", "run_identity": run["run_identity"], "last_checkpoint": run["last_checkpoint"],
                "run_dir": str(directory), "counters": counters, "physical_work_by_stage": physical,
                "evaluation": {"status": "complete", "evaluation_enters_training": False, "accounting": deepcopy(accounting)}}
        metrics = {"evidence_kind": "native_final_endpoint", "training_seeds": [0, 1], "runs": 4,
                   "episodes_per_run": 20, "total_logical_episodes": 80, "paired_seed_count": 2,
                   "arms": [{"variant": v, "master_seed": s, "successes": 0} for v, s in CASES],
                   "descriptive_summary": {"success": {"mean": 0}},
                   **{name: False for name in ("confidence_intervals", "p_values", "bootstrap", "auc", "superiority_claim", "equivalence_claim")}}
        self.context["analysis"] = {"status": "complete", "source_unchanged": True, "missing_evidence_imputed": False,
            "cohort_identity": self.cohort_id, "protocol_sha256": self.protocol, "bank_sha256": self.bank, "analysis": metrics}
        self.secondary = {"status": "complete", "cohort_identity": self.cohort_id, "warnings": []}
        self.resources = {"status": "complete", "cohort_identity": self.cohort_id, "warnings": []}
        self.proof = {"schema_version": "p565758-implementation-v1", "status": "complete_p565758_implementation",
            "tests_run": 30, "failures": 0, "errors": 0, "skipped": 0, "source_sha256": self.source,
            "frozen_source_sha256": self.frozen, "research_training_executed": False,
            "research_evaluation_executed": False, "gpu_runtime_measured": False}

    def journal(self, directory, stage, rows):
        folder = directory if stage == "controller" else directory / "stages" / stage
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "attempts.jsonl"
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
        path.write_bytes(raw)
        return {"source_prefix": {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
                "additional_partial_work_unknown": False, "additional_partial_optimizer_work": 0,
                "recorded_elapsed_seconds": 10.0}

    def result(self):
        return build_acceptance(self.context, self.secondary, self.resources, self.proof)

    def retry_events(self, variant, seed, errors):
        run = next(row for row in self.context["runs"] if (row["variant"], row["master_seed"]) == (variant, seed))
        stage = "controller" if variant == "D" else "local_cycle"
        rows = []
        for index, (kind, message) in enumerate(errors):
            rows.extend([{"event": "unit_started", "attempt": f"bad{index}", "invocation": f"run{index}"},
                         {"event": "campaign_invocation_failed", "attempt": f"bad{index}",
                          "invocation": f"run{index}", "error_type": kind, "error": message}])
        rows.extend([{"event": "unit_started", "attempt": "ok", "invocation": "last"},
                     {"event": "unit_committed", "attempt": "ok", "invocation": "last"}])
        physical = self.context["ledger"]["runs"][f"{variant}/seed_{seed}"]["physical_work_by_stage"]
        physical[stage] = self.journal(Path(run["run_dir"]), stage, rows)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = AcceptanceFixture(self.temporary.name)

    def rejected(self, fragment=None):
        result = self.fixture.result()
        self.assertEqual(result["status"], "needs_review", result)
        self.assertFalse(result["phase5_acceptance"])
        if fragment:
            self.assertIn(fragment, " ".join(result["reasons"]))
        return result

    def test_complete_zero_success_result_passes_execution_not_efficacy(self):
        result = self.fixture.result()
        self.assertEqual(result["status"], "passed", result["reasons"])
        self.assertTrue(result["execution_acceptance"])
        self.assertEqual(result["successful_final_episodes_observed"], 0)
        self.assertFalse(result["successful_transfer_established"])
        self.assertFalse(result["D_wins_required_for_acceptance"])
        self.assertFalse(result["research_training_executed_by_gate"])
        self.assertFalse(result["research_evaluation_executed_by_gate"])

    def test_negative_effects_remain_valid(self):
        self.fixture.context["analysis"]["analysis"]["descriptive_summary"] = {"success": {"mean": -0.5}}
        self.assertTrue(self.fixture.result()["phase5_acceptance"])

    def test_authentication_missing_or_false_rejected(self):
        for flag in AUTHENTICATION:
            with self.subTest(flag=flag):
                self.fixture.context["authentication"][flag] = False
                self.rejected("authenticated")
                self.fixture.context["authentication"][flag] = True

    def test_candidate_cohort_rejected(self):
        self.fixture.context["cohort"]["candidate_only"] = True
        self.rejected("candidate")

    def test_missing_arm_not_zero_imputed(self):
        self.fixture.context["runs"].pop()
        result = self.rejected("all four")
        self.assertFalse(result["missing_evidence_imputed"])

    def test_duplicate_arm_rejected(self):
        self.fixture.context["runs"][-1] = deepcopy(self.fixture.context["runs"][0])
        self.rejected("duplicate")

    def test_reduced_local_budget_rejected(self):
        self.fixture.context["runs"][0]["local_cycles_completed"] = 19
        self.rejected("registered value 20")

    def test_incomplete_teacher_and_distillation_rejected(self):
        run = self.fixture.context["runs"][1]
        for field, bad in (("teacher_cycles_completed", 24), ("trajectories_completed", 127),
                           ("distillation_optimizer_steps", 99), ("teacher_actor_optimizer_steps", 399)):
            old = run[field]
            run[field] = bad
            self.rejected("registered value")
            run[field] = old

    def test_shared_inputs_not_doublecounted(self):
        self.fixture.context["ledger"]["shared_initial_costs"]["physical_joint_transitions"] = 200000
        self.rejected("100000")

    def test_new_collection_and_model_updates_rejected(self):
        run = self.fixture.context["runs"][0]
        for field in ("new_environment_interactions", "new_model_updates"):
            run[field] = 1
            self.rejected("registered value 0")
            run[field] = 0

    def test_original_model_update_count_must_be_4500(self):
        self.fixture.context["runs"][0]["model_updates"] = 5000
        self.rejected("4500")

    def test_initial_pairing_mutation_rejected(self):
        self.fixture.context["runs"][1]["run_binding"]["pairing"]["actor_cpu"] = identity("different")
        self.rejected("no longer share")

    def test_historical_adoption_rejected(self):
        self.fixture.context["runs"][1]["run_binding"]["historical_adoption"] = True
        self.rejected("fresh-run binding")

    def test_missing_actor_and_stale_checkpoint_rejected(self):
        run = self.fixture.context["runs"][0]
        run["actor_export"] = None
        self.rejected("actor export")
        run["actor_export"] = "actor"
        run["last_checkpoint"] = "newer-checkpoint"
        self.rejected("checkpoint or ledger")

    def test_missing_episode_rejected(self):
        self.fixture.context["evaluation_reports"][0]["episodes"].pop()
        self.rejected("20 native bank")

    def test_fixture_native_mix_rejected(self):
        self.fixture.context["evaluation_reports"][0]["fixture_only"] = True
        self.rejected("native final")

    def test_wrong_bank_rejected(self):
        self.fixture.context["evaluation_reports"][0]["bank_sha256"] = identity("wrongbank")
        self.rejected("binding")

    def test_evaluation_retry_limit_rejected(self):
        report = self.fixture.context["evaluation_reports"][0]
        report["interaction_accounting"]["infrastructure_retries"] = 3
        for index in range(1, 4):
            report["attempts"].append({"path": f"episodes/000000/attempt_{index:02d}/attempt.json", "sha256": identity(index)})
        self.rejected("episode retries exceed two")

    def test_more_than_two_retries_across_distinct_episodes_is_valid(self):
        report = self.fixture.context["evaluation_reports"][0]
        report["interaction_accounting"]["infrastructure_retries"] = 4
        for episode in (0, 1):
            for attempt in (1, 2):
                report["attempts"].append({"path": f"episodes/{episode:06d}/attempt_{attempt:02d}/attempt.json", "sha256": identity((episode, attempt))})
        self.fixture.context["ledger"]["runs"]["B/seed_0"]["evaluation"]["accounting"] = deepcopy(report["interaction_accounting"])
        self.assertTrue(self.fixture.result()["phase5_acceptance"])

    def test_final_ledger_must_include_completed_evaluation(self):
        self.fixture.context["ledger"]["runs"]["B/seed_0"]["evaluation"]["status"] = "not_started"
        self.rejected("ledger predates")

    def test_two_clear_infrastructure_retries_allowed(self):
        self.fixture.retry_events("D", 0, [("OutOfMemoryError", "CUDA out of memory"), ("OSError", "No space left on device")])
        result = self.fixture.result()
        self.assertTrue(result["phase5_acceptance"], result["reasons"])
        self.assertEqual(result["retry_audit"]["runs"][1]["infrastructure_retries"], 2)

    def test_third_infrastructure_retry_rejected(self):
        self.fixture.retry_events("B", 0, [("TimeoutError", "timed out")] * 3)
        self.rejected("more than two")

    def test_numerical_failure_cannot_be_retried(self):
        self.fixture.retry_events("D", 1, [("ValueError", "non-finite actor gradient")])
        self.rejected("numerical failure was resumed")

    def test_unclassified_runtime_failure_not_assumed_infrastructure(self):
        self.fixture.retry_events("B", 1, [("RuntimeError", "unexpected state shape")])
        self.rejected("no established infrastructure")

    def test_missing_input_oserror_not_infrastructure(self):
        self.fixture.retry_events("B", 0, [("FileNotFoundError", "missing checkpoint")])
        self.rejected("no established infrastructure")

    def test_parent_child_failure_not_counted_twice(self):
        self.fixture.retry_events("D", 0, [("OutOfMemoryError", "CUDA out of memory")] * 2)
        run = self.fixture.context["runs"][1]
        rows = [{"event": "teacher_invocation_failed", "attempt": "child", "invocation": "child",
                 "error_type": "OutOfMemoryError", "error": "CUDA out of memory"}]
        self.fixture.context["ledger"]["runs"]["D/seed_0"]["physical_work_by_stage"]["teacher"] = self.fixture.journal(Path(run["run_dir"]), "teacher", rows)
        result = self.fixture.result()
        self.assertTrue(result["phase5_acceptance"], result["reasons"])
        self.assertEqual(result["retry_audit"]["runs"][1]["infrastructure_retries"], 2)

    def test_child_numerical_failure_cannot_hide_in_parent(self):
        run = self.fixture.context["runs"][1]
        rows = [{"event": "teacher_invocation_failed", "attempt": "child", "invocation": "child",
                 "error_type": "FloatingPointError", "error": "invalid loss"}]
        self.fixture.context["ledger"]["runs"]["D/seed_0"]["physical_work_by_stage"]["teacher"] = self.fixture.journal(Path(run["run_dir"]), "teacher", rows)
        self.rejected("numerical failure")

    def test_child_failure_without_parent_failure_is_unresolved(self):
        run = self.fixture.context["runs"][1]
        rows = [{"event": "teacher_invocation_failed", "attempt": "child", "invocation": "child",
                 "error_type": "OutOfMemoryError", "error": "CUDA out of memory"}]
        self.fixture.context["ledger"]["runs"]["D/seed_0"]["physical_work_by_stage"]["teacher"] = self.fixture.journal(Path(run["run_dir"]), "teacher", rows)
        self.rejected("not completely represented")

    def test_child_unclassified_failure_remains_unresolved(self):
        run = self.fixture.context["runs"][1]
        self.fixture.retry_events("D", 0, [("RuntimeError", "unexpected state")])
        rows = [{"event": "teacher_invocation_failed", "attempt": "child", "invocation": "child",
                 "error_type": "RuntimeError", "error": "unexpected state"}]
        self.fixture.context["ledger"]["runs"]["D/seed_0"]["physical_work_by_stage"]["teacher"] = self.fixture.journal(Path(run["run_dir"]), "teacher", rows)
        self.rejected("child failure has no established infrastructure")

    def test_cancellation_before_training_is_not_a_retry(self):
        run = self.fixture.context["runs"][1]
        rows = [{"event": "unit_started", "attempt": "cancelled", "invocation": "first"},
                {"event": "unit_cancelled_before_training", "attempt": "cancelled", "invocation": "first",
                 "research_optimization_started": False}]
        self.fixture.context["ledger"]["runs"]["D/seed_0"]["physical_work_by_stage"]["controller"] = self.fixture.journal(Path(run["run_dir"]), "controller", rows)
        self.assertTrue(self.fixture.result()["phase5_acceptance"])

    def test_journal_tamper_and_late_append_rejected(self):
        stage = self.fixture.context["ledger"]["runs"]["B/seed_0"]["physical_work_by_stage"]["local_cycle"]
        path = Path(stage["source_prefix"]["path"])
        path.write_bytes(path.read_bytes().replace(b"first", b"wrong"))
        self.rejected("prefix changed")
        with path.open("a") as stream:
            stream.write("{}\n")
        self.rejected("advanced beyond")

    def test_unmatched_hard_interruption_needs_review(self):
        run = self.fixture.context["runs"][0]
        rows = [{"event": "unit_started", "attempt": "unknown", "invocation": "old"}]
        self.fixture.context["ledger"]["runs"]["B/seed_0"]["physical_work_by_stage"]["local_cycle"] = self.fixture.journal(Path(run["run_dir"]), "local_cycle", rows)
        self.rejected("unmatched attempts")

    def test_unknown_physical_work_remains_unknown(self):
        stage = self.fixture.context["ledger"]["runs"]["B/seed_0"]["physical_work_by_stage"]["local_cycle"]
        stage["additional_partial_work_unknown"] = True
        stage["additional_partial_optimizer_work"] = None
        result = self.fixture.result()
        self.assertTrue(result["phase5_acceptance"], result["reasons"])
        self.assertTrue(result["warnings"])
        stage["additional_partial_optimizer_work"] = 0
        self.rejected("unknown interrupted optimizer work was imputed")

    def test_missing_resource_measurements_do_not_become_zero_or_block(self):
        self.fixture.resources["warnings"] = ["GPU occupancy and peak memory unmeasured"]
        self.fixture.resources["gpu_seconds"] = None
        result = self.fixture.result()
        self.assertTrue(result["phase5_acceptance"], result["reasons"])
        self.assertIn("GPU occupancy and peak memory unmeasured", result["warnings"])
        self.assertFalse(result["unmeasured_costs_assumed_zero"])

    def test_missing_figures_or_reporting_rejected(self):
        self.fixture.context["deliverables"]["figures"].pop()
        self.rejected("two deterministic endpoint figures")

    def test_mock_figures_cannot_establish_real_study_acceptance(self):
        self.fixture.context["figure_evidence"]["fixtures_only"] = True
        self.rejected("actual rendered graphics")

    def test_missing_measurements_propagate_as_explicit_warnings(self):
        self.fixture.resources["missing_measurements"] = ["gpu_seconds", "peak_memory"]
        result = self.fixture.result()
        self.assertTrue(result["phase5_acceptance"])
        self.assertTrue(any("missing measurements" in warning for warning in result["warnings"]))

    def test_implementation_proof_source_and_tests_rejected(self):
        self.fixture.proof["source_sha256"] = {"other.py": identity("other")}
        self.rejected("bind current")
        self.fixture.proof["source_sha256"] = self.fixture.source
        self.fixture.proof["skipped"] = 1
        self.rejected("registered value 0")

    def test_statistical_claims_rejected(self):
        self.fixture.context["analysis"]["analysis"]["confidence_intervals"] = True
        self.rejected("statistical scope")

    def test_input_objects_not_mutated(self):
        before = deepcopy((self.fixture.context, self.fixture.secondary, self.fixture.resources, self.fixture.proof))
        self.fixture.result()
        self.assertEqual(before, (self.fixture.context, self.fixture.secondary, self.fixture.resources, self.fixture.proof))

    def test_json_roundtrip_preserves_canonical_acceptance(self):
        original = self.fixture.result()
        self.assertTrue(original["phase5_acceptance"])
        persisted = json.loads(json.dumps({"context": self.fixture.context,
            "secondary": self.fixture.secondary, "resources": self.fixture.resources,
            "proof": self.fixture.proof}, sort_keys=True))
        reloaded = build_acceptance(persisted["context"], persisted["secondary"],
                                    persisted["resources"], persisted["proof"])
        self.assertEqual(json.dumps(original, sort_keys=True), json.dumps(reloaded, sort_keys=True))

    def test_json_roundtrip_preserves_failed_retry_audit_order(self):
        self.fixture.retry_events("D", 0, [("RuntimeError", "unclassified failure")])
        original = self.fixture.result()
        reloaded = build_acceptance(json.loads(json.dumps(self.fixture.context, sort_keys=True)),
            self.fixture.secondary, self.fixture.resources, self.fixture.proof)
        self.assertEqual(json.dumps(original, sort_keys=True), json.dumps(reloaded, sort_keys=True))

    def test_missing_context_or_malformed_summary_fails_closed(self):
        result = build_acceptance({}, {}, {}, {})
        self.assertEqual(result["status"], "needs_review")
        self.fixture.context["cohort"] = None
        self.assertEqual(self.fixture.result()["status"], "needs_review")

    def test_missing_analysis_success_count_not_zero_imputed(self):
        self.fixture.context["analysis"]["analysis"]["arms"][0].pop("successes")
        result = self.rejected("explicit nonnegative integer")
        self.assertIsNone(result["successful_final_episodes_observed"])


if __name__ == "__main__":
    unittest.main()
