"""CPU accounting fixtures: no scientific models, optimizers or native engine."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from world_models.data.ledger import file_sha256
from world_models_endpoint import ledger
from world_models_study.evidence import read_report, write_report


class LedgerFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "run"
        self.run.mkdir()
        self.replay = self.root / "collection/replay"
        self.replay.mkdir(parents=True)
        (self.replay / "manifest.json").write_text("{}", encoding="utf-8")
        self.binding = {"replay_path": str(self.replay), "provenance": {"replays": [
            {"manifest_sha256": file_sha256(self.replay / "manifest.json")} ]}}
        self.info = {"run_dir": str(self.run), "variant": "B", "master_seed": 0,
            "run_identity": "a" * 64, "cohort_identity": "b" * 64, "status": "prepared", "phase": "local",
            "last_checkpoint": str(self.run / "checkpoints/state_00000001_fixture"),
            "initial_binding": deepcopy(self.binding), "local_cycles_completed": 0,
            "local_actor_optimizer_steps": 0, "local_critic_optimizer_steps": 0,
            "teacher_cycles_completed": 0, "teacher_actor_optimizer_steps": 0, "teacher_critic_optimizer_steps": 0,
            "trajectories_completed": 0, "distillation_optimizer_steps": 0,
            "imagined_joint_steps_local": 0, "imagined_joint_steps_teacher": 0, "imagined_joint_steps_buffer": 0,
            "new_environment_interactions": 0, "new_model_updates": 0}
        self.inspect = self.enterContext(patch.object(ledger, "_inspect_run", side_effect=lambda directory: deepcopy(self.info)))

    def advance(self, cycles=1):
        self.info.update(local_cycles_completed=cycles, local_actor_optimizer_steps=cycles * 16,
            local_critic_optimizer_steps=cycles * 16, imagined_joint_steps_local=cycles * 256,
            status="incomplete", phase="local")

    def journal(self, path, rows, *, append=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if append else "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def pair(self, attempt="one"):
        return [{"event": "local_cycle_started", "attempt": attempt},
                {"event": "local_cycle_committed", "attempt": attempt, "elapsed_seconds": 2.5}]


class RunLedgerTests(LedgerFixture):
    def test_prepared_run_reuses_initial_data_and_keeps_missing_history_unknown(self):
        result = ledger.snapshot_run(self.run)
        self.assertEqual(result["initial_collection"]["valid_for_replay"], 50000)
        self.assertIsNone(result["initial_collection"]["real_simulator_tics_total"])
        self.assertIsNone(result["historical_and_development_costs"]["original_fitting_beyond_selected_prefix_optimizer_steps"])
        self.assertFalse(result["historical_and_development_costs"]["selected_prefix_is_total_historical_fit"])
        self.assertEqual(result["counters"]["world_model_optimizer_steps_initial"], 4500)
        self.assertEqual(result["counters"]["world_model_optimizer_steps_online"], 0)
        self.assertFalse(result["research_training_executed_by_ledger"])

    def test_single_joint_decision_is_not_doubled_by_two_players(self):
        self.advance()
        self.journal(self.run / "attempts.jsonl", self.pair())
        result = ledger.snapshot_run(self.run)
        self.assertEqual(result["counters"]["imagined_joint_transitions_local"], 256)
        self.assertEqual(result["counters"]["local_player_decisions"], 512)
        self.assertEqual(result["counters"]["local_actor_steps"], 16)
        self.assertEqual(result["physical_work_by_stage"]["local_cycle"]["recorded_elapsed_seconds"], 2.5)
        self.assertEqual(result["physical_work_by_stage"]["local_cycle"]["physical_actor_optimizer_steps_lower_bound"], 16)

    def test_missing_or_torn_attempt_cost_is_unknown_not_zero(self):
        self.advance()
        missing = ledger.snapshot_run(self.run)["physical_work_by_stage"]["local_cycle"]
        self.assertIsNone(missing["additional_partial_optimizer_work"])
        path = self.run / "attempts.jsonl"
        self.journal(path, self.pair())
        with path.open("a", encoding="utf-8") as stream:
            stream.write('{"event":"local_cycle_started",')
        damaged = ledger.snapshot_run(self.run)["physical_work_by_stage"]["local_cycle"]
        self.assertTrue(damaged["additional_partial_work_unknown"])
        self.assertEqual(damaged["malformed_rows"], 1)

    def test_unmatched_attempt_survives_recovery_as_additional_unknown_compute(self):
        self.advance()
        self.journal(self.run / "attempts.jsonl", [{"event": "local_cycle_started", "attempt": "interrupted"}] + self.pair("retry"))
        result = ledger.snapshot_run(self.run)["physical_work_by_stage"]["local_cycle"]
        self.assertEqual(result["started_units"], 2)
        self.assertEqual(result["logical_completed_units"], 1)
        self.assertEqual(result["unmatched_attempts"], ["interrupted"])
        self.assertIsNone(result["additional_partial_optimizer_work"])

    def test_physical_duplicate_attempt_ids_are_rejected(self):
        self.journal(self.run / "attempts.jsonl", self.pair() + self.pair())
        with self.assertRaisesRegex(ValueError, "attempt identities"):
            ledger.snapshot_run(self.run)

    def test_d_buffer_is_counted_once_even_when_shared_with_distillation(self):
        self.info.update(variant="D", teacher_cycles_completed=25, teacher_actor_optimizer_steps=400,
            teacher_critic_optimizer_steps=400, trajectories_completed=128, distillation_optimizer_steps=100,
            imagined_joint_steps_teacher=6400, imagined_joint_steps_buffer=1280,
            stage_directories={name: str(self.run / ("distillation" if name in ("buffer", "distillation") else name))
                for name in ("teacher", "buffer", "distillation", "local_cycle")})
        result = ledger.snapshot_run(self.run)
        self.assertEqual(result["counters"]["imagined_joint_transitions_total"], 7680)
        self.assertEqual(result["counters"]["warm_start_buffer_trajectories"], 128)
        self.assertFalse(result["physical_work_by_stage"]["controller"]["scientific_compute_debit"])

    def test_wrong_ppo_debit_and_nonzero_environment_work_fail_closed(self):
        self.info["local_actor_optimizer_steps"] = 1
        with self.assertRaisesRegex(ValueError, "fixed B/D schedule"):
            ledger.snapshot_run(self.run)
        self.info["local_actor_optimizer_steps"] = 0
        self.info["new_environment_interactions"] = 1
        with self.assertRaisesRegex(ValueError, "fixed B/D schedule"):
            ledger.snapshot_run(self.run)

    def test_initial_collection_retains_reset_overhead_and_exact_published_tics(self):
        rows = [{"sequence": 0, "task_family": "relay", "run_seed": 0, "event": "created"},
            {"sequence": 1, "task_family": "relay", "run_seed": 0, "event": "reset", "physical_delta": 3, "initialization_delta": 3}]
        rows.extend({"sequence": n + 2, "task_family": "relay", "run_seed": 0, "event": "transition",
            "physical_delta": 1, "published_delta": 1, "actual_tics": 4} for n in range(50000))
        rows.append({"sequence": 50002, "task_family": "relay", "run_seed": 0,
                     "event": "episode_committed", "valid_length": 50000})
        path = self.replay.parent / "training_interactions.jsonl"
        self.journal(path, rows)
        report = {"status": "complete_initial_replay", "task_family": "relay", "run_seed": 0,
            "target_transitions": 50000, "dataset_manifest_sha256": file_sha256(self.replay / "manifest.json"),
            "interaction_accounting": {"ledger_path": path.name, "ledger_sha256": file_sha256(path),
                "physical_joint_decisions_including_initialization": 50003, "initialization_joint_decisions": 3,
                "published_real_joint_transitions": 50000, "failed_dispatch_attempts": 0,
                "published_real_simulator_tics_total": 200000, "replay_committed_joint_transitions": 50000}}
        (self.replay.parent / "collection_report.json").write_text(json.dumps(report), encoding="utf-8")
        result = ledger.snapshot_run(self.run)["initial_collection"]
        self.assertEqual(result["real_simulator_tics_total"], 200000)
        self.assertEqual(result["physical_joint_decisions_including_initialization"], 50003)
        self.assertIsNone(result["initialization_simulator_tics"])
        self.assertFalse(result["accounting_exact"])

    def test_evaluation_consumed_work_stays_separate_from_training_axis(self):
        output = self.root / "evaluation"
        output.mkdir()
        write_report(self.run / "endpoint_test_claim.json", {"schema_version": "p54-single-final-test-claim-v1",
            "run_dir": str(self.run), "output_dir": str(output), "cohort_identity": "b" * 64,
            "bank_sha256": "d" * 64, "final_actor_identity": "e" * 64})
        write_report(output / "evaluation.json", {"fixture": True})
        accounting = {"published_real_joint_transitions": 8040, "known_simulator_tics_total": 32170,
            "completed_episode_joint_transitions": 8000, "unknown_dispatch_attempts": 1, "unknown_simulator_tics": True}
        fake = SimpleNamespace(load_evaluation=lambda *a, **kw: {"status": "complete", "interaction_accounting": accounting},
            load_partial_accounting=lambda *a: self.fail("complete report should use authenticated loader"))
        with patch.dict(sys.modules, {"world_models_endpoint.evaluation": fake}):
            result = ledger.snapshot_run(self.run)
        self.assertEqual(result["evaluation"]["evaluation_joint_transitions"], 8040)
        self.assertTrue(result["evaluation"]["unknown_simulator_tics"])
        self.assertEqual(result["initial_collection"]["real_joint_transitions_total"], 50000)
        self.assertEqual(result["counters"]["new_environment_interactions"], 0)


class SnapshotTests(LedgerFixture):
    def test_receiptless_publication_is_preserved_and_rebuilt_without_double_debit(self):
        from world_models.data.ledger import atomic_json
        output = self.root / "ledger"
        first = ledger.snapshot_run(self.run, output)
        self.advance()
        self.journal(self.run / "attempts.jsonl", self.pair())
        captured = []
        def interrupted(path, document):
            atomic_json(path, document)
            captured.append(Path(path).read_bytes())
            raise OSError("fixture crash before receipt publication")
        with patch.object(ledger, "write_report", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "before receipt"):
                ledger.snapshot_run(self.run, output)
        unfinished = output / "snapshot_00000002.json"
        self.assertTrue(unfinished.is_file())
        self.assertFalse(unfinished.with_suffix(".sha256").exists())
        recovered = ledger.snapshot_run(self.run, output)
        self.assertEqual(recovered["snapshot_sequence"], 2)
        self.assertEqual(recovered["counters"]["local_PPO_cycles"], 1)
        row = read_report(recovered["snapshot_path"])
        self.assertEqual(row["previous"]["sha256"], file_sha256(first["snapshot_path"]))
        self.assertEqual(len(row["preserved_uncommitted_publications"]), 1)
        archived = Path(row["preserved_uncommitted_publications"][0]["path"])
        self.assertEqual(archived.read_bytes(), captured[0])
        self.assertFalse(ledger.snapshot_run(self.run, output)["new_snapshot_written"])

    def test_present_invalid_or_orphan_receipt_is_never_repaired(self):
        output = self.root / "ledger"
        result = ledger.snapshot_run(self.run, output)
        marker = Path(result["snapshot_path"]).with_suffix(".sha256")
        marker.write_text("invalid receipt\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "valid receipt"):
            ledger.snapshot_run(self.run, output)
        self.assertFalse((output / "uncommitted_publications").exists())
        Path(result["snapshot_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "orphan snapshot receipt"):
            ledger.snapshot_run(self.run, output)

    def test_append_receipts_chain_and_duplicate_observations_do_not_double_debit(self):
        output = self.root / "ledger"
        first = ledger.snapshot_run(self.run, output)
        again = ledger.snapshot_run(self.run, output)
        self.assertEqual(first["snapshot_path"], again["snapshot_path"])
        self.assertFalse(again["new_snapshot_written"])
        self.advance()
        self.journal(self.run / "attempts.jsonl", self.pair())
        later = ledger.snapshot_run(self.run, output)
        self.assertEqual(later["snapshot_sequence"], 2)
        row = read_report(later["snapshot_path"])
        self.assertEqual(row["previous"]["sha256"], file_sha256(first["snapshot_path"]))

    def test_checkpoint_counter_regression_is_rejected(self):
        output = self.root / "ledger"
        self.advance(2)
        self.journal(self.run / "attempts.jsonl", self.pair("one") + self.pair("two"))
        ledger.snapshot_run(self.run, output)
        self.advance(1)
        with self.assertRaisesRegex(ValueError, "counter regressed"):
            ledger.snapshot_run(self.run, output)

    def test_append_prefix_rejects_truncated_or_rewritten_physical_history(self):
        output = self.root / "ledger"
        self.journal(self.run / "attempts.jsonl", self.pair())
        ledger.snapshot_run(self.run, output)
        self.journal(self.run / "attempts.jsonl", self.pair("replaced"))
        with self.assertRaisesRegex(ValueError, "truncated or rewritten"):
            ledger.snapshot_run(self.run, output)

    def test_resolving_missing_commit_journal_row_is_not_counter_regression(self):
        output = self.root / "ledger"
        self.advance()
        self.journal(self.run / "attempts.jsonl", self.pair()[:1])
        first = ledger.snapshot_run(self.run, output)
        self.journal(self.run / "attempts.jsonl", self.pair()[1:], append=True)
        second = ledger.snapshot_run(self.run, output)
        self.assertEqual(second["snapshot_sequence"], first["snapshot_sequence"] + 1)
        self.assertEqual(second["physical_work_by_stage"]["local_cycle"]["logical_work_missing_commit_journal_rows"], 0)

    def test_changed_snapshot_receipt_and_reuse_for_another_subject_fail(self):
        output = self.root / "ledger"
        result = ledger.snapshot_run(self.run, output)
        self.info["run_identity"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "chain was altered"):
            ledger.snapshot_run(self.run, output)
        self.info["run_identity"] = "a" * 64
        Path(result["snapshot_path"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "valid receipt"):
            ledger.snapshot_run(self.run, output)

    def test_output_cannot_write_inside_sources_or_training_data(self):
        with self.assertRaisesRegex(ValueError, "separate from sources"):
            ledger.snapshot_run(self.run, ledger.ROOT / "src/ledger-fixture-do-not-create")


class CohortLedgerTests(LedgerFixture):
    def test_shared_inputs_are_not_multiplied_as_physical_collection_or_training(self):
        slots = [{"case": f"relay/{variant}/seed_{seed}", "run_dir": str(self.run if (variant, seed) == ("B", 0)
            else self.root / f"absent/{variant}/seed_{seed}")} for seed in (0, 1) for variant in ("B", "D")]
        cohort = {"cohort_identity": "b" * 64, "report_path": str(self.root / "cohort.json"), "run_matrix": slots,
            "inputs": {"physical_initial_joint_transitions": 100000, "physical_selected_model_optimizer_updates": 9000}}
        with patch.object(ledger, "_inspect_cohort", return_value=cohort):
            result = ledger.summarize_cohort(self.root / "cohort.json")
        shared = result["shared_initial_costs"]
        self.assertEqual(shared["physical_joint_transitions"], 100000)
        self.assertEqual(shared["configured_four_run_attributed_joint_transitions"], 200000)
        self.assertEqual(shared["prepared_run_attributed_joint_transitions"], 50000)
        self.assertEqual(shared["physical_selected_model_prefix_optimizer_steps"], 9000)
        self.assertEqual(shared["configured_four_run_attributed_selected_model_steps"], 18000)
        self.assertEqual(len(result["missing_run_slots"]), 3)

    def test_no_prepared_runs_is_pending_and_never_claimed_zero_total_campaign_cost(self):
        cohort = {"cohort_identity": "b" * 64, "report_path": str(self.root / "cohort.json"),
            "run_matrix": [{"case": "relay/B/seed_0", "run_dir": str(self.root / "missing")}],
            "inputs": {"physical_initial_joint_transitions": 100000, "physical_selected_model_optimizer_updates": 9000}}
        with patch.object(ledger, "_inspect_cohort", return_value=cohort):
            result = ledger.summarize_cohort(self.root / "cohort.json")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["sum_run_logical_counters"], {})
        self.assertIsNone(result["historical_and_development_costs"]["original_fit_beyond_selected_prefix"])


class LedgerCLITests(unittest.TestCase):
    def test_cli_ledger_is_read_only_and_prints_compact_counters(self):
        from scripts.phase53 import main
        with (patch.object(ledger, "summarize_cohort", return_value={"status": "incomplete", "snapshot_path": "proof.json"}) as call,
                patch("sys.stdout", new_callable=io.StringIO) as stream):
            self.assertEqual(main(["ledger", "--cohort", "cohort.json", "--output-dir", "ledger"]), 0)
        call.assert_called_once_with("cohort.json", "ledger")
        self.assertEqual(json.loads(stream.getvalue())["snapshot_path"], "proof.json")

    def test_cli_refuses_invalid_receipts_without_faking_success(self):
        from scripts.phase53 import main
        with patch.object(ledger, "summarize_cohort", side_effect=ValueError("changed receipt")), patch("sys.stderr", new_callable=io.StringIO) as stream:
            self.assertEqual(main(["ledger", "--cohort", "cohort.json", "--output-dir", "ledger"]), 1)
        self.assertIn("changed receipt", stream.getvalue())
