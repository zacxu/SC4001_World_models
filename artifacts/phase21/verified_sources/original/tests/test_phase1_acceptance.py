"""Acceptance evidence integrity and the non-learning diagnostic boundary."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil

from scripts.phase1_acceptance import run_stage
from world_models.contracts import EpisodeBoundary, JointTransitionMetadata, LocalActorOutput, RGBFrame
from world_models.env.acceptance_metrics import choose_collectors, cpu_budget, percentile
from world_models.env.coop import ResetResult, StepResult
from world_models.env.locality import actor_step, audit_locality
from world_models.env.validation_data import ValidationLedger


class LocalityAcceptanceTests(unittest.TestCase):
    def test_both_local_histories_ignore_changed_global_and_teammate_data(self):
        history = [ResetResult((RGBFrame(bytes([index])*36864),RGBFrame(bytes([index+20])*36864)),
                               {"global_tokens":[index],"reward":999}) for index in range(3)]
        result = audit_locality(history,hidden_size=8)
        self.assertEqual(result["status"],"passed")
        for player in result["players"]:
            self.assertTrue(player["outputs_equal_after_global_and_teammate_changes"])
            self.assertTrue(player["own_image_change_changes_output"])
            self.assertTrue(player["final_memories_equal"])
        with self.assertRaises(TypeError):
            actor_step(None,history[0].observations[0],12,(0.,)*8,global_tokens=[1])

    def test_supplied_constant_actor_is_local_even_if_one_pixel_has_no_effect(self):
        class Constant:
            def step(self,inputs):
                return LocalActorOutput((1.,)+(0.,)*11,inputs.local_hidden)
        frame = RGBFrame(bytes(36864))
        result = audit_locality([ResetResult((frame,frame),{})],hidden_size=8,actor=Constant())
        self.assertEqual(result["status"],"passed")
        self.assertFalse(result["players"][0]["own_image_change_changes_output"])

    def test_local_history_uses_the_actual_executed_previous_action(self):
        frame = RGBFrame(bytes(36864))
        boundary = EpisodeBoundary()
        step = StepResult((frame,frame),-.001,boundary,JointTransitionMetadata(0,5,9,(5,8),boundary),{})
        result = audit_locality([ResetResult((frame,frame),{}),step],hidden_size=8)
        self.assertEqual(result["players"][0]["executed_previous_action_history"],[12,5])
        self.assertEqual(result["players"][1]["executed_previous_action_history"],[12,8])


class AcceptanceAccountingTests(unittest.TestCase):
    def test_validation_counts_reconcile_and_cannot_be_double_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            env = SimpleNamespace(task_family="relay",scenario={"hashes":{"wad_sha256":"abc"}},
                                  coordinator=SimpleNamespace(consumed_joint_decisions=10,run_dir=Path(directory)/"env"),
                                  initialization_joint_decisions=2)
            ledger = ValidationLedger(directory)
            ledger.record("worker_error",env,7,expected_error=True)
            report = ledger.manifest()
            self.assertEqual(report["consumed_joint_decisions"],10)
            self.assertEqual(report["published_transitions"],7)
            self.assertEqual(report["failed_dispatch_attempts"],1)
            self.assertFalse(report["learning_eligible"])
            self.assertFalse(report["enters_training_replay"])
            ledger.record("worker_error",env,7,expected_error=True)
            with self.assertRaisesRegex(ValueError,"twice"):
                ledger.manifest()

    def test_invalid_interaction_accounting_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            env = SimpleNamespace(coordinator=SimpleNamespace(consumed_joint_decisions=2),initialization_joint_decisions=1)
            ledger = ValidationLedger(directory)
            with self.assertRaises(ValueError):
                ledger.record("stress",env,2)
            self.assertFalse(ledger.path.exists())

    def test_source_drift_prevents_reuse_of_old_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("scripts.phase1_acceptance.execution_sources",return_value={"code.py":"old"}), patch("world_models.env.provenance.runtime_manifest",return_value={"version":1}):
                run_stage("probe",output,{},lambda:{"status":"passed"})
                result = run_stage("probe",output,{},lambda:self.fail("cached stage reran"),reuse=True)
                self.assertEqual(result["status"],"passed")
            with patch("scripts.phase1_acceptance.execution_sources",return_value={"code.py":"new"}), patch("world_models.env.provenance.runtime_manifest",return_value={"version":1}):
                with self.assertRaisesRegex(RuntimeError,"stale"):
                    run_stage("probe",output,{},lambda:{"status":"passed"},reuse=True)

    def test_full_cli_cannot_claim_completion_from_reduced_counts(self):
        completed = subprocess.run([sys.executable,"-m","scripts.phase1_acceptance","verify","--decisions","100","--resets","2"],
                                   capture_output=True,text=True,timeout=20,check=False)
        self.assertEqual(completed.returncode,2)
        self.assertIn(">=10000",completed.stderr)

    def test_cached_evidence_detects_changed_trace_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            evidence = output/"probe"/"trace.json"
            def callback():
                evidence.parent.mkdir()
                evidence.write_text('{"result":1}')
                return {"status":"passed"}
            with patch("scripts.phase1_acceptance.execution_sources",return_value={"code":"same"}), patch("world_models.env.provenance.runtime_manifest",return_value={"version":1}):
                run_stage("probe",output,{},callback)
                evidence.write_text('{"result":2}')
                with self.assertRaisesRegex(RuntimeError,"bytes"):
                    run_stage("probe",output,{},lambda:self.fail("tampered evidence reused"),reuse=True)

    def test_mismatched_runtime_pin_blocks_execution(self):
        with tempfile.TemporaryDirectory() as directory, patch("world_models.env.provenance.runtime_manifest",return_value={"packages":{"simulator":{"matches":False}}}):
            with self.assertRaisesRegex(RuntimeError,"requirements"):
                run_stage("probe",Path(directory),{},lambda:self.fail("wrong runtime executed"))

    def test_cache_report_receipt_detects_modified_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("scripts.phase1_acceptance.execution_sources",return_value={"code":"same"}), patch("world_models.env.provenance.runtime_manifest",return_value={"version":1}):
                run_stage("probe",output,{},lambda:{"status":"passed","measured":10})
                path = output/"probe_report.json"
                result = json.loads(path.read_text())
                result["measured"] = 999
                path.write_text(json.dumps(result))
                with self.assertRaisesRegex(RuntimeError,"receipt"):
                    run_stage("probe",output,{},lambda:self.fail("modified result reused"),reuse=True)


class ResourceMetricsTests(unittest.TestCase):
    def test_collector_selection_rejects_unstable_baseline_and_slow_family(self):
        from copy import deepcopy
        pair = {"1":{"median_transitions_per_second":10,"throughput_coefficient_of_variation":.1,"median_p95_decision_latency_seconds":.1},
                "2":{"median_transitions_per_second":18,"throughput_coefficient_of_variation":.1,"median_p95_decision_latency_seconds":.15}}
        aggregates = {"relay":deepcopy(pair),"cover":deepcopy(pair)}
        self.assertEqual(choose_collectors(aggregates)["selected_collectors"],2)
        aggregates["cover"]["2"]["median_transitions_per_second"] = 9
        self.assertEqual(choose_collectors(aggregates)["selected_collectors"],1)
        aggregates["cover"]["1"]["throughput_coefficient_of_variation"] = .5
        self.assertIsNone(choose_collectors(aggregates)["selected_collectors"])
        self.assertNotEqual(choose_collectors(aggregates)["status"],"passed")

    def test_latency_percentiles_and_cpu_affinity_restore(self):
        self.assertEqual(percentile([4,1,3,2],.5),2.5)
        self.assertAlmostEqual(percentile([0,10],.95),9.5)
        process = psutil.Process()
        original = process.cpu_affinity()
        with self.assertRaisesRegex(RuntimeError,"fixture"):
            with cpu_budget(min(2,len(original))) as budget:
                self.assertEqual(process.cpu_affinity(),budget["cpu_ids"])
                raise RuntimeError("fixture")
        self.assertEqual(process.cpu_affinity(),original)


if __name__ == "__main__":
    unittest.main()
