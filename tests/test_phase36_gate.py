"""Acceptance accounting tests using explicitly constructed temporary evidence.

These fixtures do not execute or claim to have executed ViZDoom. They exercise
the aggregator's acceptance/rejection logic; all files disappear after each test.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.phase36_support import gate
from scripts.phase36_support.evidence import INVARIANT_TESTS, ROOT, file_sha256, load_report, write_report
from scripts.phase36_support.evaluation import actor_version, load_actor_export, validation_bank
from world_models.cycle.checkpoints import save_cycle_checkpoint
from world_models.cycle.provenance import source_identity as training_source_identity
from world_models.cycle.schedule import Schedule
from world_models.infrastructure.config import resolve_config
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.policies import LocalActor, PolicyConfig


class Phase36GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sources = {"fixture_source.py": "0"*64}
        self.source_patch = patch.object(gate, "source_identity", return_value=self.sources)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.serial = 0
        self.run = self.root / "run"
        self.run.mkdir()
        settings = deepcopy(resolve_config("relay", "B", "pilot", ROOT / "configs").settings)
        settings["actor"].update(embedding_size=8, hidden_size=8)
        settings["critic"]["hidden_sizes"] = [16, 16]
        self.config = PolicyConfig.from_experiment(settings)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(36)
            self.actor = LocalActor(self.config)
        run_manifest = {"schema_version": "p35-run-v1", "master_seed": 0, "device": "cpu", "settings": settings,
                        "source_sha256": training_source_identity(), "schedule": Schedule.from_experiment(settings).describe()}
        (self.run / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
        self.before_export = self._actor_export("real_000000000", baseline=True)
        with torch.no_grad():
            next(self.actor.parameters()).add_(.01)
        self.after_export = self._actor_export("real_000050000", baseline=False)
        self.invariants = self._invariants()
        self.mechanics = self._mechanics()
        self.baseline = self._evaluation("baseline", self.before_export)
        self.trained = self._evaluation("trained", self.after_export, successes=2, progress=3)

    def _actor_export(self, name, *, baseline, model_updates=5000, cycles=100):
        metadata = {"schema_version": "p35-local-actor-v1", "task_family": "relay", "variant": "B", "run_seed": 0,
                    "real_joint_transitions": 0 if baseline else 50000, "actual_initial_replay_interactions": 50000,
                    "untrained_zero_interaction_baseline": baseline, "actor_version": actor_version(self.actor),
                    "actor_config": json.loads(json.dumps(asdict(self.config))), "world_model_version": "a"*64,
                    "model_updates": model_updates, "local_ppo_cycles": 0 if baseline else cycles,
                    "actor_optimizer_steps": 0 if baseline else cycles*16, "critic_optimizer_steps": 0 if baseline else cycles*16,
                    "evaluation_split": "validation", "on_registered_evaluation_grid": True,
                    "evaluation_executed": False, "phase3_acceptance": False}
        path = self.run / "evaluation" / name
        save_cycle_checkpoint(path, state={"actor": self.actor.state_dict()}, metadata=metadata)
        return path

    def _invariants(self):
        folder = self.root / "invariants"
        folder.mkdir()
        log = folder / "tests.txt"
        log.write_text("Constructed acceptance unit-test fixture, not an executed test log.\n", encoding="utf-8")
        passed = sorted({item for ids in INVARIANT_TESTS.values() for item in ids})
        sources = {path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").glob("*.py")}
        return folder, {"schema_version": "p36-implementation-v1", "status": "complete_p36_implementation",
                        "failures": 0, "errors": 0, "skipped": 0, "tests_run": len(passed), "passed_tests": passed,
                        "source_unchanged_during_verification": True, "source_sha256": self.sources,
                        "invariants": {name: {"status": "passed", "test_ids": list(ids)} for name, ids in INVARIANT_TESTS.items()},
                        "test_source_sha256": {name: file_sha256(ROOT / name) for name in sources},
                        "tests_log": {"path": "tests.txt", "sha256": file_sha256(log)}}

    def _mechanics(self):
        folder = self.root / "mechanics"
        folder.mkdir()
        rows = [{"cycles_completed": index, "actor_optimizer_steps": index*2, "critic_optimizer_steps": index*2,
                 "metrics": {"initial_ratio_max_error": 0., "actor_loss": -.1}} for index in (1, 2)]
        trace = folder / "ppo_updates.jsonl"
        trace.write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")
        exports = {}
        for index, label in enumerate(("before", "after")):
            path = folder / label
            metadata = {"schema_version": "p36-mechanics-actor-v1", "engineering_only": True,
                        "phase3_acceptance": False, "actor_version": str(index)*64, "world_model_version": "a"*64,
                        "label": label}
            save_cycle_checkpoint(path, state={"actor": {"weight": torch.tensor(float(index))}}, metadata=metadata)
            exports[label] = {"path": label, "actor_version": metadata["actor_version"],
                              "manifest_sha256": file_sha256(path / "manifest.json"),
                              "payload_sha256": file_sha256(path / "state.pt")}
        checks = {name: True for name in ("world_unchanged", "actor_changed", "critic_changed", "improvement_over_before",
                                          "above_chance", "finite_ppo_diagnostics", "initial_ratios_replayed", "independent_actor_streams")}
        return folder, {"schema_version": "p36-mechanics-v1", "status": "passed", "engineering_only": True,
                        "native_environment_evaluation": False, "phase3_acceptance": False,
                        "learning_algorithm": "production RecurrentMAPPO", "checks": checks,
                        "before": {"expected_joint_success": .01, "mean_expected_team_return": -.7},
                        "after": {"expected_joint_success": .8, "mean_expected_team_return": .9},
                        "chance": {"expected_joint_success": 1/144}, "cycles": 2,
                        "actor_optimizer_steps": 4, "critic_optimizer_steps": 4, "actor_exports": exports,
                        "source_sha256": self.sources, "source_sha256_after": self.sources,
                        "world_model_version": "a"*64, "ppo_updates_path": trace.name,
                        "ppo_updates_sha256": file_sha256(trace)}

    def _evaluation(self, name, export_path, *, successes=0, progress=0, episodes=8):
        folder = self.root / name
        folder.mkdir()
        _, export = load_actor_export(export_path)
        export["path"] = os.path.relpath(export_path, folder)
        export["run_manifest"]["path"] = os.path.relpath(self.run / "run_manifest.json", folder)
        bank = validation_bank("relay", episodes=episodes)
        records, transitions = [], []
        ledger = [{"event": "created", "learning_eligible": False, "data_source": "evaluation_environment"}]
        for index, entry in enumerate(bank["episodes"]):
            success = index < successes
            episode_id = f"episode_{index}"
            reward = float(success) + .1*progress
            counts = {"latch_activations": progress}
            records.append({**entry, "episode_id": episode_id, "length": 1, "return": reward, "success": success,
                            "terminal_reason": "success" if success else "failure", "event_counts": counts,
                            "mean_action_entropy": 1.})
            transitions.append({"episode_id": episode_id, "decision_index": 0, "terminated": True,
                                "team_reward": reward, "event_deltas": counts, "actions": [0, 1], "previous_actions": [12, 12],
                                "terminal_reason": records[-1]["terminal_reason"]})
            ledger.extend([{"event": "reset", "episode_id": episode_id, "physical_delta": 0, "initialization_delta": 0},
                           {"event": "transition", "episode_id": episode_id, "decision_index": 0,
                            "collector_kind": "local_actor_evaluation", "physical_delta": 1, "initialization_delta": 0,
                            "published_delta": 1, "failed_delta": 0, "actual_tics": 4},
                           {"event": "evaluation_episode_completed", "episode_id": episode_id, "valid_length": 1}])
        ledger = [{**row, "schema_version": "real-replay-ledger-v1", "sequence": index, "task_family": "relay", "run_seed": 0}
                  for index, row in enumerate(ledger)]
        (folder / "bank.json").write_text(json.dumps(bank), encoding="utf-8")
        for filename, rows in (("episodes.jsonl", records), ("transitions.jsonl", transitions),
                               ("evaluation_interactions.jsonl", ledger)):
            (folder / filename).write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")
        files = {path.name: {"path": path.name, "sha256": file_sha256(path)} for path in folder.iterdir()}
        accounting = {"learning_eligible": False, "training_interactions_added": 0, "data_source": "evaluation_environment",
                      "ledger_sha256": files["evaluation_interactions.jsonl"]["sha256"], "failed_dispatch_attempts": 0,
                      "real_joint_transitions_total": episodes, "published_real_joint_transitions": episodes,
                      "completed_episode_joint_transitions": episodes, "physical_joint_decisions_including_initialization": episodes,
                      "initialization_joint_decisions": 0, "published_real_simulator_tics_total": episodes*4}
        metrics = {"episodes": episodes, "successes": successes, "success_rate": successes/episodes,
                   "mean_return": sum(row["return"] for row in records)/episodes, "mean_length": 1.,
                   "mean_event_progress": float(progress)}
        return folder, {"schema_version": "p36-real-evaluation-v1", "status": "complete", "task_family": "relay",
                        "run_seed": 0, "variant": "B", "native_backend": True, "native_engine": True,
                        "training_interactions_added": 0, "real_ppo_updates": 0, "policy_optimizer_updates": 0,
                        "actor_export": export, "actor_version_before": export["metadata"]["actor_version"],
                        "actor_version_after": export["metadata"]["actor_version"], "source_unchanged": True,
                        "world_model_loaded": False, "critic_loaded": False, "learning_eligible": False,
                        "runtime": {"python": "3.12", "torch": "fixture", "vizdoom": "fixture", "device": "cpu",
                                    "precision": "float32", "deterministic_algorithms": True, "cudnn_benchmark": False,
                                    "cpu_threads": 1},
                        "protocol": {"split": "validation", "episodes": episodes, "bank_seed": 3600,
                                     "bank_sha256": canonical_hash(bank), "mode": "stochastic", "temperature": 1.,
                                     "precision": "float32", "frame_skip": 4, "deadline": 400,
                                     "evaluation_data_enters_training": False},
                        "episodes": records, "metrics": metrics, "files": files,
                        "interaction_accounting": accounting, "source_sha256": self.sources}

    def _persist(self, value):
        if value is None:
            return None
        folder, report = value
        self.serial += 1
        return write_report(folder / f"fixture_{self.serial}.json", report)

    def _gate(self, *, invariants=True, mechanics=True, native=True):
        path = self.root / f"acceptance_{self.serial}.json"
        result = gate.aggregate_acceptance(
            invariant_report=self._persist(self.invariants) if invariants else None,
            mechanics_report=self._persist(self.mechanics) if mechanics else None,
            baseline_report=self._persist(self.baseline) if native else None,
            trained_report=self._persist(self.trained) if native else None, output_path=path)
        self.assertEqual(result, load_report(path))
        self.assertFalse(result["automatic_campaign_launch"])
        self.assertFalse(result["P2.5_quality_acceptance"])
        self.assertEqual(result["new_environment_interactions"], 0)
        return result

    def test_complete_paired_evidence_accepts_only_one_exploratory_case(self):
        report = self._gate()
        self.assertTrue(report["phase3_acceptance"], report["reasons"])
        self.assertEqual(report["status"], "accepted_exploratory_pilot")
        self.assertEqual(report["accepted_scope"], {"task_family": "relay", "run_seed": 0, "variant": "B"})
        self.assertEqual(report["native_learning"]["success_pair_wins"], 2)
        self.assertFalse(report["native_learning"]["statistical_significance_established"])

    def selected_initial_run(self, *, export_updates=4250):
        """Construct a reduced-count fixture; the selection reader is unit mocked."""
        run_path = self.run / "run_manifest.json"
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
        settings = manifest["settings"]
        common = deepcopy(settings)
        common.pop("variant")
        row = {"task": "relay", "seed": 0, "model_updates": 4250, "original_target_stage_updates": 5000,
            "common_config_sha256": canonical_hash(common), "checkpoint_path": str(self.root / "p24.pt"),
            "checkpoint_sha256": "a"*64, "selection_path": str(self.root / "selection.json"),
            "selection_sha256": "b"*64}
        with patch("world_models.training.selection.resolve_selected_case", return_value=row):
            schedule, binding = Schedule.from_initial_selection(settings, 0, row["selection_path"])
        manifest.update(schedule=schedule.describe(), registered_schedule=Schedule.from_experiment(settings).describe(),
            initial_checkpoint=row["checkpoint_path"], initial_checkpoint_sha256=row["checkpoint_sha256"],
            initial_selection=binding, initial_training_policy="frozen_last_saved_checkpoints", study_scope="exploratory_pilot")
        run_path.write_text(json.dumps(manifest), encoding="utf-8")
        before = self._actor_export("selected_before", baseline=True, model_updates=export_updates)
        with torch.no_grad():
            next(self.actor.parameters()).add_(.01)
        after = self._actor_export("selected_after", baseline=False, model_updates=export_updates)
        self.baseline = self._evaluation("selected_baseline", before)
        self.trained = self._evaluation("selected_trained", after, successes=2, progress=3)
        return row

    def test_selected_initial_count_uses_real_count_in_native_evidence_gate(self):
        row = self.selected_initial_run()
        with patch("world_models.training.selection.resolve_selected_case", return_value=row):
            report = self._gate()
        self.assertTrue(report["phase3_acceptance"], report["reasons"])
        self.assertEqual(report["status"], "accepted_exploratory_pilot")

    def test_amended_budget_acceptance_uses_40_cycles_and_preserved_initial_count(self):
        from world_models.cycle.study import apply_study_profile
        parent = resolve_config("relay", "B", "pilot", ROOT / "configs").settings
        settings, amendment = apply_study_profile(parent, "relay_week_v1", master_seed=0)
        self.config = PolicyConfig.from_experiment(settings)
        self.actor = LocalActor(self.config)
        common = deepcopy(parent)
        common.pop("variant")
        row = {"task": "relay", "seed": 0, "model_updates": 4500, "original_target_stage_updates": 5000,
            "common_config_sha256": canonical_hash(common), "checkpoint_path": str(self.root / "p24.pt"),
            "checkpoint_sha256": "a"*64, "selection_path": str(self.root / "selection.json"), "selection_sha256": "b"*64}
        with patch("world_models.training.selection.resolve_selected_case", return_value=row):
            schedule, binding = Schedule.from_initial_selection(settings, 0, row["selection_path"], study_amendment=amendment)
            manifest = {"schema_version": "p35-run-v1", "master_seed": 0, "device": "cpu", "settings": settings,
                "source_sha256": training_source_identity(), "schedule": schedule.describe(),
                "study_amendment": amendment, "parent_schedule": Schedule.from_experiment(parent).describe(),
                "registered_schedule": Schedule.from_experiment(settings).describe(), "initial_selection": binding,
                "initial_checkpoint": row["checkpoint_path"], "initial_checkpoint_sha256": row["checkpoint_sha256"],
                "initial_training_policy": "frozen_last_saved_checkpoints", "study_scope": "exploratory_pilot"}
            (self.run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            before = self._actor_export("amended_before", baseline=True, model_updates=4500, cycles=40)
            with torch.no_grad():
                next(self.actor.parameters()).add_(.01)
            after = self._actor_export("amended_after", baseline=False, model_updates=4500, cycles=40)
            self.baseline = self._evaluation("amended_baseline", before)
            self.trained = self._evaluation("amended_trained", after, successes=2, progress=3)
            report = self._gate()
        self.assertTrue(report["phase3_acceptance"], report["reasons"])
        self.assertEqual(report["accepted_scope"], {"task_family": "relay", "run_seed": 0, "variant": "B"})

    def test_selected_run_cannot_report_registered_count_as_actual_learning(self):
        row = self.selected_initial_run(export_updates=5000)
        with patch("world_models.training.selection.resolve_selected_case", return_value=row):
            report = self._gate()
        self.assertEqual(report["status"], "invalid_evidence")

    def test_selected_run_requires_unchanged_selection_receipt_for_native_acceptance(self):
        self.selected_initial_run()
        with patch("world_models.training.selection.resolve_selected_case", side_effect=ValueError("selection receipt differs")):
            report = self._gate()
        self.assertEqual(report["status"], "invalid_evidence")

    def test_synthetic_mechanics_and_invariants_alone_never_accept(self):
        result = self._gate(native=False)
        self.assertFalse(result["phase3_acceptance"])
        self.assertEqual(result["status"], "pending_real_evidence")
        self.assertEqual(result["mechanics"]["status"], "passed")

    def test_missing_mechanics_or_invariant_evidence_never_accepts(self):
        self.assertFalse(self._gate(mechanics=False)["phase3_acceptance"])
        self.assertFalse(self._gate(invariants=False)["phase3_acceptance"])

    def test_failed_mechanics_behavior_blocks_native_success(self):
        self.mechanics[1]["checks"]["above_chance"] = False
        report = self._gate()
        self.assertFalse(report["phase3_acceptance"])
        self.assertEqual(report["status"], "needs_review")

    def test_parameter_change_without_behavioral_improvement_needs_review(self):
        self.trained = self._evaluation("trained_no_improvement", self.after_export)
        report = self._gate()
        self.assertEqual(report["status"], "needs_review")
        self.assertFalse(report["native_learning"]["observed_behavioral_improvement"])

    def test_event_progress_and_return_improvement_can_support_exploratory_pilot(self):
        self.trained = self._evaluation("trained_event_progress", self.after_export, progress=1)
        report = self._gate()
        self.assertTrue(report["phase3_acceptance"], report["reasons"])
        self.assertEqual(report["native_learning"]["trained_minus_baseline"]["success_rate"], 0.)

    def test_short_positive_bank_is_not_sufficient(self):
        self.baseline = self._evaluation("baseline_short", self.before_export, episodes=2)
        self.trained = self._evaluation("trained_short", self.after_export, episodes=2, successes=1)
        report = self._gate()
        self.assertEqual(report["status"], "needs_review")
        self.assertFalse(report["native_learning"]["sufficient_diagnostic_count"])

    def test_mock_engine_flags_are_rejected(self):
        self.trained[1]["native_engine"] = False
        report = self._gate()
        self.assertEqual(report["status"], "invalid_evidence")

    def test_stale_production_source_is_rejected(self):
        self.baseline[1]["source_sha256"] = {"different": "1"*64}
        self.assertEqual(self._gate()["status"], "invalid_evidence")

    def test_skipped_or_missing_actual_passed_test_ids_are_rejected(self):
        self.invariants[1]["passed_tests"].pop()
        report = self._gate()
        self.assertEqual(report["status"], "invalid_evidence")
        self.assertIn("count", report["components"]["invariants"]["reason"])

    def test_stale_invariant_test_source_is_rejected(self):
        name = next(iter(self.invariants[1]["test_source_sha256"]))
        self.invariants[1]["test_source_sha256"][name] = "f"*64
        self.assertEqual(self._gate()["status"], "invalid_evidence")

    def test_report_edit_after_receipt_fails_closed(self):
        path = self._persist(self.invariants)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(" ")
        report = gate.aggregate_acceptance(invariant_report=path, output_path=self.root / "invalid.json")
        self.assertEqual(report["status"], "invalid_evidence")

    def test_native_metrics_must_recompute_from_episode_evidence(self):
        self.trained[1]["metrics"]["success_rate"] = .9
        self.assertEqual(self._gate()["status"], "invalid_evidence")

    def test_trace_edits_or_missing_artifacts_fail_closed(self):
        (self.trained[0] / "transitions.jsonl").write_text("{}\n", encoding="utf-8")
        self.assertEqual(self._gate()["status"], "invalid_evidence")

    def test_mechanics_export_corruption_fails_closed(self):
        (self.mechanics[0] / "after/state.pt").write_bytes(b"broken")
        self.assertEqual(self._gate()["status"], "invalid_evidence")

    def test_actor_export_corruption_fails_before_acceptance(self):
        (self.after_export / "state.pt").write_bytes(b"broken")
        self.assertEqual(self._gate()["status"], "invalid_evidence")

    def test_mismatched_precision_or_bank_seed_cannot_be_paired(self):
        self.trained[1]["protocol"]["precision"] = "bfloat16"
        self.trained[1]["runtime"]["precision"] = "bfloat16"
        report = self._gate()
        self.assertEqual(report["status"], "invalid_evidence")
        self.assertIn("protocols differ", report["components"]["pairing"]["reason"])

    def test_untouched_test_bank_cannot_be_used_for_acceptance(self):
        self.trained[1]["protocol"]["split"] = "test"
        self.assertEqual(self._gate()["status"], "invalid_evidence")

    def test_source_change_during_aggregation_fails_closed(self):
        with patch.object(gate, "source_identity", side_effect=[self.sources, {"changed": "f"*64}]):
            self.assertEqual(self._gate()["status"], "invalid_evidence")

    def test_incorrect_evaluation_cost_is_rejected(self):
        self.trained[1]["interaction_accounting"]["published_real_joint_transitions"] += 1
        self.assertEqual(self._gate()["status"], "invalid_evidence")

    def test_relative_evidence_and_actor_paths_remain_portable(self):
        report = self._gate()
        self.assertTrue(report["phase3_acceptance"], report["reasons"])
        self.assertIn("..", Path(self.trained[1]["actor_export"]["path"]).parts)

    def test_greedy_pair_is_diagnostic_only_even_when_behavior_improves(self):
        for _, report in (self.baseline, self.trained):
            report["protocol"].update(mode="greedy", temperature=None)
        result = self._gate()
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("greedy", result["components"]["trained"]["reason"])

    def test_matching_changed_task_deadline_still_fails_protocol(self):
        for _, report in (self.baseline, self.trained):
            report["protocol"]["deadline"] += 1
        self.assertEqual(self._gate()["status"], "invalid_evidence")

    def test_rehashed_ledger_with_false_debit_is_not_trusted(self):
        path = self.trained[0] / "evaluation_interactions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        next(row for row in rows if row["event"] == "transition")["physical_delta"] = 2
        path.write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")
        digest = file_sha256(path)
        self.trained[1]["files"][path.name]["sha256"] = digest
        self.trained[1]["interaction_accounting"]["ledger_sha256"] = digest
        result = self._gate()
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("ledger debit", result["components"]["trained"]["reason"])

    def test_rehashed_ledger_cannot_omit_creation_record(self):
        path = self.trained[0] / "evaluation_interactions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()][1:]
        for index, row in enumerate(rows):
            row["sequence"] = index
        path.write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")
        digest = file_sha256(path)
        self.trained[1]["files"][path.name]["sha256"] = digest
        self.trained[1]["interaction_accounting"]["ledger_sha256"] = digest
        result = self._gate()
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("creation record", result["components"]["trained"]["reason"])

    def test_rehashed_trace_with_invented_event_progress_is_not_trusted(self):
        path = self.trained[0] / "transitions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["event_deltas"] = {"latch_activations": 0}
        path.write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")
        self.trained[1]["files"][path.name]["sha256"] = file_sha256(path)
        result = self._gate()
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("task-event counts", result["components"]["trained"]["reason"])

    def test_different_evaluation_runtime_is_not_a_paired_learning_check(self):
        self.trained[1]["runtime"]["device"] = "cuda:0"
        result = self._gate()
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIn("runtimes differ", result["components"]["pairing"]["reason"])


if __name__ == "__main__":
    unittest.main()
