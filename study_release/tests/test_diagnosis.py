"""Synthetic saved-trace fixtures; no scientific episode is executed here."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from study_release import diagnosis


def _trace(index, outcome):
    success, latch = outcome == "success", outcome in ("success", "latch_failure")
    n = 400 if outcome == "deadline" else 2
    rows = []
    for decision in range(n):
        last = decision == n - 1
        terminal = 1.0 if success and last else (-1.0 if last else 0.0)
        total = -0.001 + terminal
        rows.append({"decision_index": decision, "actions": [1, 2],
            "previous_actions": [12, 12] if decision == 0 else [1, 2],
            "team_reward": total, "reward_components": {"step": -0.001, "terminal": terminal,
                "milestone": 0.0, "total": total, "shared_rewards": [total, total]},
            "event_deltas": {"latch_activations": int(latch and last), "successes": int(success and last)},
            "action_entropy": [1.2, 1.4], "terminated": last, "truncated": False,
            "terminal_reason": ("success" if success else ("deadline" if outcome == "deadline" else "failure")) if last else "none",
            "start_tic": decision * 4, "end_tic": (decision + 1) * 4, "actual_tics": 4})
    record = {"episode_index": index, "episode_id": f"relay-test-{index:06d}",
        "success": success, "latch_activated": latch, "joint_transitions": n,
        "simulator_tics": 4 * n, "termination_cause": rows[-1]["terminal_reason"],
        "episode_return": sum(row["team_reward"] for row in rows),
        "mean_action_entropy": sum(sum(row["action_entropy"]) / 2 for row in rows) / n,
        "event_counts": {"latch_activations": int(latch), "successes": int(success)}}
    pointers = [{"path": "/synthetic-only/events.jsonl", "line": i + 1, "sequence": i,
                 "row_sha256": "e" * 64} for i in range(n)]
    return record, {"transitions": rows, "pointers": pointers, "sources": []}


def context_fixture(outcomes=None):
    outcomes = outcomes or {}
    bank = [{"episode_index": i, "template_id": f"relay_test_{i % 2:02d}"} for i in range(20)]
    cohort = {"cohort_identity": "c" * 64, "protocol_sha256": "p" * 64,
        "bank_sha256": "b" * 64, "bank": {"episodes": bank}}
    context = {"cohort": cohort, "authentication": {k: True for k in diagnosis._AUTH},
        "evaluation_reports": [], "evaluation_paths": []}
    traces = {}
    for seed in (0, 1):
        for variant in ("B", "D"):
            episodes = []
            for i, entry in enumerate(bank):
                record, loaded = _trace(i, outcomes.get((seed, variant, i), "failure"))
                record.update(template_id=entry["template_id"], bank_entry=deepcopy(entry))
                episodes.append(record)
                traces[seed, variant, i] = loaded
            report = {"schema_version": "p54-final-test-evaluation-v1", "status": "complete",
                "task_family": "relay", "variant": variant, "master_seed": seed,
                "cohort_identity": cohort["cohort_identity"], "protocol_sha256": cohort["protocol_sha256"],
                "bank_sha256": cohort["bank_sha256"], "final_local_cycles": 20,
                "fixture_only": False, "native_backend": True, "native_engine": True,
                "native_environment_evaluation_executed": True, "research_evaluation_executed": True,
                "episodes": episodes, "attempts": []}
            context["evaluation_reports"].append(report)
            context["evaluation_paths"].append(f"/synthetic-only/{variant}/seed_{seed}/evaluation.json")
    def loader(evaluation, path, record):
        return deepcopy(traces[evaluation["master_seed"], evaluation["variant"], record["episode_index"]])
    return context, traces, loader


class DiagnosisTests(unittest.TestCase):
    def test_no_success_selects_first_failure_for_all_arms_and_exposes_missing(self):
        context, _, loader = context_fixture()
        result = diagnosis.build_diagnosis(context, trace_loader=loader)
        self.assertEqual(result["selected_count"], 4)
        self.assertEqual([r["episode_index"] for r in result["selected_traces"]], [0, 0, 0, 0])
        self.assertEqual([r["category"] for r in result["missing_categories"]], ["successful"] * 4)
        self.assertEqual(result["endpoint_interpretation"]["code"], "both_arms_unsuccessful")
        self.assertTrue(result["fixtures_only"])
        self.assertTrue(result["outcome_categories_complete"])
        self.assertFalse(result["new_training_or_test_execution"])

    def test_first_matching_bank_order_is_independent_of_input_order(self):
        changes = {(seed, variant, i): "success" for seed in (0, 1) for variant in ("B", "D") for i in (3, 8)}
        context, _, loader = context_fixture(changes)
        for report in context["evaluation_reports"]:
            report["episodes"].reverse()
        context["evaluation_reports"].reverse()
        context["evaluation_paths"].reverse()
        result = diagnosis.build_diagnosis(context, trace_loader=loader)
        self.assertEqual(result["selected_count"], 8)
        self.assertEqual([r["episode_index"] for r in result["selected_traces"]], [3, 0] * 4)
        self.assertFalse(result["missing_categories"])

    def test_every_episode_success_has_no_failed_category(self):
        context, _, loader = context_fixture({(s, v, i): "success" for s in (0, 1) for v in ("B", "D") for i in range(20)})
        result = diagnosis.build_diagnosis(context, trace_loader=loader)
        self.assertEqual(len(result["missing_categories"]), 4)
        self.assertTrue(all(row["category"] == "failed" for row in result["missing_categories"]))
        self.assertEqual(result["endpoint_interpretation"]["code"], "inconsistent_or_unhelpful_transfer")

    def test_trace_observations_keep_unsupported_causal_explanations_undetermined(self):
        context, _, loader = context_fixture({(0, "D", 0): "latch_failure"})
        result = diagnosis.build_diagnosis(context, trace_loader=loader)
        selected = next(r for r in result["selected_traces"] if r["master_seed"] == 0 and r["variant"] == "D")
        classes = selected["classifications"]
        self.assertEqual(classes["incomplete_coordination"]["status"], "observed_incomplete_after_latch")
        for key in ("mechanics_local_ambiguity", "model_reward_termination_mismatch", "unstable_optimization"):
            self.assertEqual(classes[key]["status"], "undetermined")
            self.assertFalse(classes[key]["causal_effect_established"])
        first = selected["summary"]["first_positive_event"]["latch_activations"]
        self.assertEqual(first["decision_index"], 1)
        self.assertEqual(first["pointer"]["line"], 2)

    def test_deadline_episode_must_have_400_decisions(self):
        context, traces, loader = context_fixture({(0, "B", 0): "deadline"})
        result = diagnosis.build_diagnosis(context, trace_loader=loader)
        self.assertEqual(result["selected_traces"][0]["summary"]["joint_transitions"], 400)
        traces[0, "B", 0]["transitions"].pop()
        with self.assertRaises(ValueError):
            diagnosis.build_diagnosis(context, trace_loader=loader)

    def test_fixed_cohort_and_native_authentication_are_required(self):
        for key in diagnosis._AUTH:
            with self.subTest(key=key):
                context, _, loader = context_fixture()
                context["authentication"][key] = False
                with self.assertRaises(ValueError):
                    diagnosis.build_diagnosis(context, trace_loader=loader)
        for field, value in (("status", "failed"), ("cohort_identity", "wrong"), ("protocol_sha256", "wrong"),
                ("bank_sha256", "wrong"), ("fixture_only", True), ("native_engine", False),
                ("final_local_cycles", 80), ("task_family", "cover")):
            with self.subTest(field=field):
                context, _, loader = context_fixture()
                context["evaluation_reports"][0][field] = value
                with self.assertRaises(ValueError):
                    diagnosis.build_diagnosis(context, trace_loader=loader)

    def test_missing_duplicate_runs_or_bank_episodes_are_rejected(self):
        for kind in ("run", "duplicate_run", "bank", "episode", "duplicate_episode", "path"):
            with self.subTest(kind=kind):
                context, _, loader = context_fixture()
                if kind == "run":
                    context["evaluation_reports"].pop()
                elif kind == "duplicate_run":
                    context["evaluation_reports"][0] = deepcopy(context["evaluation_reports"][1])
                elif kind == "bank":
                    context["cohort"]["bank"]["episodes"].reverse()
                elif kind == "episode":
                    context["evaluation_reports"][0]["episodes"].pop()
                elif kind == "duplicate_episode":
                    context["evaluation_reports"][0]["episodes"][1] = deepcopy(context["evaluation_reports"][0]["episodes"][0])
                else:
                    context["evaluation_paths"].pop()
                with self.assertRaises(ValueError):
                    diagnosis.build_diagnosis(context, trace_loader=loader)

    def test_event_reward_terminal_and_entropy_summary_tampering_is_rejected(self):
        for field, value in (("event_counts", {}), ("episode_return", 99), ("simulator_tics", 99),
                             ("mean_action_entropy", 0.0), ("bank_entry", {})):
            with self.subTest(field=field):
                context, _, loader = context_fixture()
                context["evaluation_reports"][0]["episodes"][0][field] = value
                with self.assertRaises(ValueError):
                    diagnosis.build_diagnosis(context, trace_loader=loader)

    def test_action_alignment_numeric_and_decomposition_corruption_are_rejected(self):
        for field, value in (("decision_index", 9), ("previous_actions", [1, 2]), ("actions", [12, 0]),
                ("terminated", True), ("truncated", True), ("event_deltas", {"latch_activations": -1}),
                ("team_reward", math.nan), ("action_entropy", [math.inf, 0]),
                ("actual_tics", 5), ("end_tic", 99), ("reward_components", {})):
            with self.subTest(field=field):
                context, traces, loader = context_fixture()
                traces[0, "B", 0]["transitions"][0][field] = value
                with self.assertRaises(ValueError):
                    diagnosis.build_diagnosis(context, trace_loader=loader)

    def test_sample_never_exceeds_eight_and_is_deterministic(self):
        context, _, loader = context_fixture({(s, v, i): "success" for s in (0, 1) for v in ("B", "D") for i in range(0, 20, 2)})
        first = diagnosis.build_diagnosis(context, trace_loader=loader)
        second = diagnosis.build_diagnosis(deepcopy(context), trace_loader=loader)
        self.assertEqual(first, second)
        self.assertLessEqual(first["selected_count"], 8)

    def test_positive_paired_signs_are_exploratory_not_superiority(self):
        context, _, loader = context_fixture({(s, "D", 0): "success" for s in (0, 1)})
        result = diagnosis.build_diagnosis(context, trace_loader=loader)["endpoint_interpretation"]
        self.assertEqual(result["code"], "D_higher_success_in_both_pairs")
        self.assertFalse(result["statistical_superiority_established"])
        self.assertFalse(result["causal_component_effect_established"])
        self.assertFalse(result["successful_transfer_established"])

    def test_mixed_signs_and_secondary_progress_keep_per_seed_values(self):
        context, _, loader = context_fixture({(0, "D", 0): "success", (1, "B", 0): "success"})
        result = diagnosis.build_diagnosis(context, trace_loader=loader)["endpoint_interpretation"]
        self.assertEqual(result["code"], "inconsistent_or_unhelpful_transfer")
        self.assertEqual([row["success_rate_difference_D_minus_B"] for row in result["paired_differences"]], [0.05, -0.05])
        context, _, loader = context_fixture({(s, v, 1): "success" for s in (0, 1) for v in ("B", "D")}
            | {(0, "D", 0): "latch_failure"})
        result = diagnosis.build_diagnosis(context, trace_loader=loader)["endpoint_interpretation"]
        self.assertEqual(result["code"], "secondary_progress_without_success_benefit")
        self.assertEqual(result["paired_differences"][0]["latch_episode_rate_difference_D_minus_B"], 0.05)

    def test_history_is_separate_and_no_unavailable_numbers_are_invented(self):
        context, _, loader = context_fixture()
        result = diagnosis.build_diagnosis(context, trace_loader=loader)
        self.assertEqual([row["status"] for row in result["separate_history"]], ["not_packaged"] * 2)
        self.assertTrue(all(row["included_in_paired_endpoint_metrics"] is False for row in result["separate_history"]))
        self.assertEqual(sum(row["episodes"] for row in result["full_bank_arm_outcomes"]), 80)

    def test_historical_archives_are_hash_bound_and_never_pooled(self):
        context, _, loader = context_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cover.json"
            path.write_text('{"quality_acceptance": false}', encoding="utf-8")
            context["historical_evidence"] = [{"kind": "cover_model_validation", "report_path": str(path),
                "sha256": diagnosis._sha(path), "summary": "separate old model screen"}]
            result = diagnosis.build_diagnosis(context, trace_loader=loader)
            self.assertEqual(result["separate_history"][0]["status"], "archived_separately")
            self.assertFalse(result["separate_history"][0]["included_in_paired_endpoint_metrics"])
            self.assertEqual(len(result["diagnostic_sources"]), 1)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                diagnosis.validate_diagnosis_sources(result)
            with self.assertRaises(ValueError):
                diagnosis.build_diagnosis(context, trace_loader=loader)

    def test_human_report_exposes_missing_successes_and_causal_limits(self):
        context, _, loader = context_fixture()
        result = diagnosis.build_diagnosis(context, trace_loader=loader)
        text = diagnosis.render_markdown(result)
        self.assertIn("Selected 4 of at most eight", text)
        self.assertEqual(text.count("no successful episode exists"), 4)
        self.assertIn("undetermined", text)
        self.assertIn("No simulator execution", text)
        self.assertIn("not pooled", text)


class NativeSavedTraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.context, self.traces, self.loader = context_fixture()
        self.evaluation = self.context["evaluation_reports"][0]
        self.record = self.evaluation["episodes"][0]
        self.evaluation_path = self.root / "evaluation.json"
        self.evaluation_path.write_text(json.dumps(self.evaluation), encoding="utf-8")
        self.attempt_path = self.root / "episodes/000000/attempt_00/attempt.json"
        self.attempt_path.parent.mkdir(parents=True)
        self.journal = self.attempt_path.parent / "events.jsonl"
        previous, rows = "0" * 64, []
        for index, transition in enumerate(self.traces[0, "B", 0]["transitions"]):
            body = {"sequence": index, "previous_sha256": previous,
                    "event": "transition", "transition": transition}
            previous = hashlib.sha256(diagnosis._canonical(body).encode("utf-8")).hexdigest()
            rows.append({**body, "sha256": previous})
        body = {"sequence": len(rows), "previous_sha256": previous,
                "event": "episode_complete", "record": self.record}
        rows.append({**body, "sha256": hashlib.sha256(diagnosis._canonical(body).encode("utf-8")).hexdigest()})
        self.journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
        self.attempt = {"schema_version": "p54-episode-attempt-v1", "episode_index": 0, "attempt": 0,
            "native_backend": True, "native_engine": True, "status": "complete",
            "record": deepcopy(self.record), "journal_incomplete": False,
            "journal": {"path": "events.jsonl", "sha256": diagnosis._sha(self.journal), "bytes": self.journal.stat().st_size}}
        self.publish()

    def publish(self):
        self.attempt_path.write_text(json.dumps(self.attempt), encoding="utf-8")
        self.attempt_path.with_suffix(".sha256").write_text(f"{diagnosis._sha(self.attempt_path)}  {self.attempt_path.name}\n", encoding="ascii")
        self.evaluation["attempts"] = [{"path": self.attempt_path.relative_to(self.root).as_posix(),
            "sha256": diagnosis._sha(self.attempt_path), "bytes": self.attempt_path.stat().st_size}]

    def load(self):
        return diagnosis._native_trace(self.evaluation, self.evaluation_path, self.record)

    def test_saved_native_schema_loads_verified_rows_with_exact_line_pointers(self):
        loaded = self.load()
        self.assertEqual(len(loaded["transitions"]), 2)
        self.assertEqual(loaded["pointers"][0]["line"], 1)
        self.assertEqual(len(loaded["sources"]), 3)
        summary = diagnosis._trace_summary(loaded, self.record)
        self.assertEqual(summary["termination_cause"], "failure")

    def test_receipt_and_native_identity_changes_fail_closed(self):
        self.attempt_path.with_suffix(".sha256").write_text("invalid", encoding="ascii")
        with self.assertRaises(ValueError):
            self.load()
        self.attempt["native_engine"] = False
        self.publish()
        with self.assertRaises(ValueError):
            self.load()

    def test_refreshed_journal_hash_cannot_hide_hashchain_corruption(self):
        rows = [json.loads(line) for line in self.journal.read_text(encoding="utf-8").splitlines()]
        rows[0]["transition"]["actions"] = [0, 0]
        self.journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
        self.attempt["journal"].update(sha256=diagnosis._sha(self.journal), bytes=self.journal.stat().st_size)
        self.publish()
        with self.assertRaisesRegex(ValueError, "hash chain"):
            self.load()

    def test_evidence_paths_cannot_escape_or_hide_changed_bytes(self):
        for reference in ({"path": "../outside.json", "sha256": "x", "bytes": 0},
                {"path": str(self.attempt_path), "sha256": "x", "bytes": 0},
                {**self.evaluation["attempts"][0], "sha256": "x"}):
            with self.subTest(reference=reference):
                self.evaluation["attempts"] = [reference]
                with self.assertRaises(ValueError):
                    self.load()

    def test_native_trace_requires_one_completed_attempt_and_final_record_agreement(self):
        for field, value in (("status", "failed"), ("record", {}), ("journal_incomplete", True)):
            with self.subTest(field=field):
                prior = deepcopy(self.attempt)
                self.attempt[field] = value
                self.publish()
                with self.assertRaises(ValueError):
                    self.load()
                self.attempt = prior

    def test_mutated_journal_after_loading_is_detected_by_final_barrier(self):
        loaded = self.load()
        report = {"diagnostic_sources": loaded["sources"]}
        diagnosis.validate_diagnosis_sources(report)
        self.journal.write_text("changed\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            diagnosis.validate_diagnosis_sources(report)

    def test_native_trace_rejects_duplicate_or_unordered_attempt_numbers(self):
        self.evaluation["attempts"].append(deepcopy(self.evaluation["attempts"][0]))
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()
