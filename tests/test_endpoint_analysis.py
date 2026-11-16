"""Exact descriptive endpoint statistics and fail-closed saved-evidence reading.

All examples below are artificial episode dictionaries. No policy checkpoint,
world model, research replay or native environment is opened by this suite.
"""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_study.bank import build_test_bank
from world_models_study.evidence import read_report, write_report
from world_models_endpoint import analysis


class EndpointAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.bank = build_test_bank()
        self.cohort_identity = "c" * 64
        self.protocol = self.bank["protocol_sha256"]
        self.reports = []
        for seed in (0, 1):
            for variant in ("B", "D"):
                rows = []
                for entry in self.bank["episodes"]:
                    index = entry["episode_index"]
                    success = index < (2 + seed + (2 if variant == "D" else 0))
                    length = 10 + index if success else 400
                    rows.append({"episode_index": index, "template_id": entry["template_id"],
                        "success": success, "episode_return": float(index + seed + (3 if variant == "D" else 0)),
                        "latch_activated": index < (6 + seed + (4 if variant == "D" else 0)),
                        "joint_transitions": length, "simulator_tics": length * 4,
                        "termination_cause": "success" if success else "deadline"})
                self.reports.append({"schema_version": analysis.EVALUATION_SCHEMA, "status": "complete",
                    "task_family": "relay", "variant": variant, "master_seed": seed,
                    "cohort_identity": self.cohort_identity, "protocol_sha256": self.protocol,
                    "bank_sha256": canonical_hash(self.bank), "final_local_cycles": 20,
                    "final_actor_identity": canonical_hash({"fixture": True, "seed": seed, "variant": variant}),
                    "runtime": {"device": "cpu", "action_rng_backend": "cpu", "torch": "synthetic-pytorch-version",
                                "action_sampler": "torch.multinomial_per_player"},
                    **{key: False for key in analysis._NATIVE_FLAGS}, "episodes": rows})

    def aggregate(self, reports=None, **kwargs):
        return analysis.aggregate_records(self.reports if reports is None else reports, bank=self.bank,
            cohort_identity=self.cohort_identity, protocol_sha256=self.protocol, require_native=False, **kwargs)

    def test_exact_raw_counts_rates_secondary_differences_and_two_seed_summary(self):
        report = self.aggregate()
        self.assertEqual(report["evidence_kind"], "synthetic_fixture")
        self.assertEqual(report["total_logical_episodes"], 80)
        self.assertEqual([row["successes"] for row in report["arms"]], [2, 4, 3, 5])
        self.assertEqual([row["latch_episodes"] for row in report["arms"]], [6, 10, 7, 11])
        self.assertEqual([row["mean_return"] for row in report["arms"]], [9.5, 12.5, 10.5, 13.5])
        for row in report["paired_differences"]:
            self.assertEqual(row["success_count_difference_D_minus_B"], 2)
            self.assertAlmostEqual(row["success_rate_difference_D_minus_B"], .1)
            self.assertEqual(row["mean_return_difference_D_minus_B"], 3)
            self.assertEqual(row["latch_count_difference_D_minus_B"], 4)
            self.assertAlmostEqual(row["latch_episode_rate_difference_D_minus_B"], .2)
        summary = report["descriptive_summary"]["success_rate_difference_D_minus_B"]
        self.assertAlmostEqual(summary["descriptive_mean"], .1)
        self.assertEqual(summary["paired_seeds"], 2)
        self.assertEqual(summary["observed_min"], min(summary["values_by_master_seed"].values()))
        self.assertEqual(summary["observed_max"], max(summary["values_by_master_seed"].values()))
        self.assertEqual(len(report["episode_records"]), 80)

    def test_identical_arms_give_exact_zero_and_swapping_arms_negates_all_effects(self):
        identical = deepcopy(self.reports)
        for seed in (0, 1):
            identical[seed * 2 + 1]["episodes"] = deepcopy(identical[seed * 2]["episodes"])
        for row in self.aggregate(identical)["paired_differences"]:
            for key, value in row.items():
                if key.endswith("_D_minus_B"):
                    self.assertEqual(value, 0)
        swapped = deepcopy(self.reports)
        for row in swapped:
            row["variant"] = "D" if row["variant"] == "B" else "B"
        before, after = self.aggregate(), self.aggregate(swapped)
        for original, reversed_row in zip(before["paired_differences"], after["paired_differences"]):
            for key, value in original.items():
                if key.endswith("_D_minus_B"):
                    self.assertEqual(reversed_row[key], -value)

    def test_negative_and_inconsistent_seed_effects_remain_visible(self):
        reports = deepcopy(self.reports)
        for row in reports[3]["episodes"]:
            row.update(success=False, termination_cause="deadline")
        pairs = self.aggregate(reports)["paired_differences"]
        self.assertAlmostEqual(pairs[0]["success_rate_difference_D_minus_B"], .1)
        self.assertAlmostEqual(pairs[1]["success_rate_difference_D_minus_B"], -.15)
        summary = self.aggregate(reports)["descriptive_summary"]["success_rate_difference_D_minus_B"]
        self.assertAlmostEqual(summary["descriptive_mean"], -.025)
        self.assertAlmostEqual(summary["observed_min"], -.15)
        self.assertAlmostEqual(summary["observed_max"], .1)

    def test_completion_lengths_are_conditional_on_success_with_null_zero_denominator(self):
        report = self.aggregate()
        self.assertEqual(report["arms"][0]["successful_completion_length_denominator"], 2)
        self.assertEqual(report["arms"][0]["mean_successful_completion_length"], 10.5)
        self.assertEqual(report["arms"][0]["mean_successful_completion_tics"], 42)
        empty = deepcopy(self.reports)
        for arm in empty:
            for row in arm["episodes"]:
                row.update(success=False, termination_cause="deadline")
        for row in self.aggregate(empty)["arms"]:
            self.assertEqual(row["successes"], 0)
            self.assertEqual(row["successful_completion_length_denominator"], 0)
            self.assertIsNone(row["mean_successful_completion_length"])
            self.assertEqual(row["successful_completion_tics_denominator"], 0)
            self.assertIsNone(row["mean_successful_completion_tics"])

    def test_layouts_keep_ten_episodes_and_terminal_counts_retain_zero_classes(self):
        result = self.aggregate()
        self.assertEqual(len(result["per_layout"]), 8)
        self.assertTrue(all(row["episodes"] == 10 for row in result["per_layout"]))
        for row in result["arms"] + result["per_layout"]:
            self.assertEqual(sum(row["termination_counts"].values()), row["episodes"])
            self.assertEqual(set(row["termination_counts"]), {"success", "deadline", "failure", "death"})
            self.assertEqual(row["termination_counts"]["death"], 0)

    def test_absent_extra_duplicate_actors_and_unpaired_seeds_fail_closed(self):
        bad = [self.reports[:3], self.reports + [self.reports[0]],
               [self.reports[0], self.reports[0], *self.reports[2:]]]
        wrong = deepcopy(self.reports)
        wrong[3]["master_seed"] = 2
        bad.append(wrong)
        for reports in bad:
            with self.subTest(reports=len(reports)), self.assertRaises(ValueError):
                self.aggregate(reports)

    def test_missing_duplicate_and_extra_episodes_are_never_imputed(self):
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.append(deepcopy(rows[0])),
                         lambda rows: rows.__setitem__(19, deepcopy(rows[0]))):
            reports = deepcopy(self.reports)
            mutation(reports[0]["episodes"])
            with self.assertRaises(ValueError):
                self.aggregate(reports)

    def test_wrong_bank_layout_variant_seed_task_cohort_amendment_and_final_stage_rejected(self):
        changes = (("bank_sha256", "0" * 64), ("cohort_identity", "0" * 64),
            ("protocol_sha256", "0" * 64), ("variant", "A"), ("master_seed", True),
            ("task_family", "cover"), ("status", "incomplete"), ("schema_version", "p36-real-evaluation-v1"),
            ("final_local_cycles", 19), ("final_local_cycles", True), ("final_actor_identity", "invalid"))
        for key, value in changes:
            reports = deepcopy(self.reports)
            reports[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.aggregate(reports)
        reports = deepcopy(self.reports)
        reports[0]["episodes"][0]["template_id"] = "relay_validation_00"
        with self.assertRaises(ValueError):
            self.aggregate(reports)
        reports = deepcopy(self.reports)
        reports[1]["final_actor_identity"] = reports[0]["final_actor_identity"]
        with self.assertRaises(ValueError):
            self.aggregate(reports)

    def test_native_and_fixture_results_cannot_be_mixed_or_promoted(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            analysis.aggregate_records(self.reports, bank=self.bank, cohort_identity=self.cohort_identity,
                                       protocol_sha256=self.protocol)
        mixed = deepcopy(self.reports)
        mixed[0].update({key: True for key in analysis._NATIVE_FLAGS})
        with self.assertRaisesRegex(ValueError, "mix"):
            self.aggregate(mixed)
        mixed = deepcopy(self.reports)
        mixed[0]["native_engine"] = True
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.aggregate(mixed)

    def test_all_four_actor_evaluations_require_one_consistent_cpu_or_cuda_rng_backend(self):
        self.assertEqual(self.aggregate()["evaluation_device_backend"], "cpu")
        mixed = deepcopy(self.reports)
        mixed[1]["runtime"].update(device="cuda:0", action_rng_backend="cuda")
        with self.assertRaisesRegex(ValueError, "same evaluation action RNG backend"):
            self.aggregate(mixed)
        for row in mixed:
            row["runtime"].update(device=f"cuda:{row['master_seed']}", action_rng_backend="cuda")
        self.assertEqual(self.aggregate(mixed)["evaluation_device_backend"], "cuda")
        for runtime in ({"device": "cpu", "action_rng_backend": "cuda"},
                        {"device": "cuda:0", "action_rng_backend": "cpu"},
                        {"device": "cpu"}, {"device": "mps", "action_rng_backend": "mps"}):
            invalid = deepcopy(self.reports)
            invalid[0]["runtime"] = runtime
            with self.subTest(runtime=runtime), self.assertRaises(ValueError):
                self.aggregate(invalid)
        for changes in ({"torch": "different-pytorch-version"}, {"action_sampler": "shared_player_stream"},
                        {"torch": ""}):
            invalid = deepcopy(self.reports)
            invalid[0]["runtime"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.aggregate(invalid)

    def test_invalid_numeric_booleans_nonfinite_and_inconsistent_terminal_records_rejected(self):
        changes = (("success", 1), ("latch_activated", 0), ("episode_return", float("nan")),
            ("episode_return", float("inf")), ("episode_return", True), ("joint_transitions", 0),
            ("joint_transitions", 401), ("joint_transitions", True), ("simulator_tics", -1),
            ("termination_cause", "collection_cutoff"), ("termination_cause", "deadline"),
            ("episode_index", True), ("episode_index", 20))
        for key, value in changes:
            reports = deepcopy(self.reports)
            reports[0]["episodes"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.aggregate(reports)

    def test_finite_values_cannot_overflow_aggregate_return_statistics(self):
        reports = deepcopy(self.reports)
        for row in reports[0]["episodes"]:
            row["episode_return"] = 1e308
        with self.assertRaisesRegex(ValueError, "overflow"):
            self.aggregate(reports)

    def test_order_of_complete_report_and_episode_inputs_does_not_change_statistics(self):
        original = deepcopy(self.reports)
        reversed_reports = deepcopy(self.reports[::-1])
        for report in reversed_reports:
            report["episodes"].reverse()
        self.assertEqual(self.aggregate(original), self.aggregate(reversed_reports))
        self.assertEqual(original, self.reports)

    def test_analysis_has_only_descriptive_claims_and_treats_seed_as_replication_unit(self):
        report = self.aggregate()
        self.assertEqual(report["replication_unit"], "training_master_seed")
        for key in ("confidence_intervals", "p_values", "bootstrap", "auc", "superiority_claim", "equivalence_claim"):
            self.assertIs(report[key], False)
        self.assertEqual(report["paired_seed_count"], 2)
        self.assertIn("partial progress", report["interpretation"])


class SavedEndpointReaderTests(unittest.TestCase):
    def setUp(self):
        fixture = EndpointAnalysisTests()
        fixture.setUp()
        self.bank, self.reports = fixture.bank, fixture.reports
        self.cohort_identity, self.protocol = fixture.cohort_identity, fixture.protocol
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cohort_path = self.root / "cohort/cohort_manifest.json"
        self.cohort_path.parent.mkdir()
        self.manifest = {"cohort_identity": self.cohort_identity, "bank": self.bank,
            "bank_sha256": canonical_hash(self.bank), "protocol_sha256": self.protocol,
            "run_matrix": [{"master_seed": seed, "variant": variant} for seed in (0, 1) for variant in ("B", "D")]}
        write_report(self.cohort_path, self.manifest)
        self.sources = {"synthetic_analysis_source.py": "a" * 64}
        self.enterContext(patch.object(analysis, "source_identity", return_value=self.sources))
        # Replace only authentication seams with strict receipt readers for
        # synthetic JSON. Production readers authenticate all linked trace and
        # final-actor provenance, which are exercised in the evaluator suite.
        self.enterContext(patch.object(analysis, "_inspect_cohort", side_effect=read_report))
        self.loader = self.enterContext(patch.object(analysis, "_load_evaluation",
            side_effect=lambda path, **kwargs: read_report(path)))
        self.paths = []
        for index, report in enumerate(self.reports):
            declared = deepcopy(report)
            declared.update({key: True for key in analysis._NATIVE_FLAGS})
            path = self.root / f"evaluation_{index}/evaluation.json"
            path.parent.mkdir()
            write_report(path, declared)
            self.paths.append(path)

    def analyze(self, *, paths=None, name="analysis"):
        return analysis.analyze_cohort(self.cohort_path, self.paths if paths is None else paths, self.root / name)

    def test_saved_reader_publishes_receipt_bound_json_and_four_exact_csv_tables(self):
        report = self.analyze()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["reasons"], [])
        self.assertEqual(self.loader.call_count, 8)
        self.assertEqual(read_report(report["report_path"]), report)
        self.assertEqual(set(report["files"]), {"arms.csv", "paired_differences.csv", "per_layout.csv", "episodes.csv"})
        for name, expected in (("arms.csv", 4), ("paired_differences.csv", 2), ("per_layout.csv", 8), ("episodes.csv", 80)):
            path = self.root / "analysis" / name
            self.assertEqual(file_sha256(path), report["files"][name]["sha256"])
            self.assertEqual(len(list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))), expected)
        for key in ("research_training_executed", "research_evaluation_executed", "native_environment_opened",
                    "missing_evidence_imputed", "statistical_superiority_established"):
            self.assertIs(report[key], False)
        self.assertEqual(report["new_environment_interactions"], 0)

    def test_missing_or_duplicate_report_paths_emit_needs_review_without_any_metrics(self):
        for index, paths in enumerate((self.paths[:3], [self.paths[0], self.paths[0], *self.paths[2:]])):
            report = self.analyze(paths=paths, name=f"missing_{index}")
            self.assertEqual(report["status"], "needs_review")
            self.assertIsNone(report["analysis"])
            self.assertEqual(report["files"], {})
            self.assertTrue(report["reasons"])
            self.assertIs(report["missing_evidence_imputed"], False)
            self.assertEqual(read_report(report["report_path"]), report)

    def test_altered_or_absent_receipt_is_not_accepted_or_imputed(self):
        self.paths[0].with_suffix(".sha256").unlink()
        report = self.analyze()
        self.assertEqual(report["status"], "needs_review")
        self.assertIn("receipt", report["reasons"][0])
        self.assertIsNone(report["analysis"])
        self.assertEqual(report["files"], {})

    def test_changed_report_between_authentication_and_publication_fails_closed(self):
        initial = read_report(self.paths[0])
        later = deepcopy(initial)
        later["episodes"][0]["episode_return"] += 1
        calls = [0]
        def moving(path, **kwargs):
            calls[0] += 1
            if calls[0] == 5:
                return later
            return read_report(path)
        self.loader.side_effect = moving
        report = self.analyze()
        self.assertEqual(report["status"], "needs_review")
        self.assertIsNone(report["analysis"])
        self.assertIn("changed", report["reasons"][0])

    def test_fixture_report_cannot_enter_scientific_reader(self):
        self.loader.side_effect = lambda path, **kwargs: deepcopy(self.reports[self.paths.index(Path(path))])
        report = self.analyze()
        self.assertEqual(report["status"], "needs_review")
        self.assertIn("synthetic", report["reasons"][0])
        self.assertIsNone(report["analysis"])

    def test_invalid_cohort_bank_or_run_matrix_is_not_analyzed(self):
        for index, change in enumerate((lambda manifest: manifest.update(bank_sha256="0" * 64),
            lambda manifest: manifest["run_matrix"].pop(),
            lambda manifest: manifest["run_matrix"].__setitem__(3, deepcopy(manifest["run_matrix"][0])))):
            manifest = deepcopy(self.manifest)
            change(manifest)
            with patch.object(analysis, "_inspect_cohort", return_value=manifest):
                report = self.analyze(name=f"bad_cohort_{index}")
            self.assertEqual(report["status"], "needs_review")
            self.assertIsNone(report["analysis"])

    def test_fresh_output_required_and_source_or_evaluation_overlap_rejected(self):
        self.analyze()
        with self.assertRaises(FileExistsError):
            self.analyze()
        with self.assertRaises(ValueError):
            analysis.analyze_cohort(self.cohort_path, self.paths, self.paths[0].parent / "analysis")
        with self.assertRaises(ValueError):
            analysis.analyze_cohort(self.cohort_path, self.paths, analysis.ROOT / "src/p55_forbidden_output")
        self.assertFalse((analysis.ROOT / "src/p55_forbidden_output").exists())

    def test_cli_help_has_saved_evidence_only_and_analysis_import_has_no_torch_or_native_backend(self):
        process = subprocess.run([sys.executable, "-m", "scripts.phase55", "analyze", "--help"],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        for flag in ("--cohort", "--evaluation", "--output-dir"):
            self.assertIn(flag, process.stdout)
        code = ("import sys; import world_models_endpoint.analysis; "
            "assert 'torch' not in sys.modules; assert 'vizdoom' not in sys.modules; "
            "assert 'world_models.env.coop' not in sys.modules")
        process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_cli_reports_completion_review_and_error_with_correct_exit_codes(self):
        from scripts.phase55 import main
        arguments = ["analyze", "--cohort", str(self.cohort_path), "--output-dir", str(self.root / "cli")]
        for path in self.paths:
            arguments.extend(["--evaluation", str(path)])
        for status, expected in (("complete", 0), ("needs_review", 1)):
            result = {"status": status, "reasons": [], "cohort_identity": self.cohort_identity,
                "report_path": "fixture.json", "research_training_executed": False,
                "research_evaluation_executed": False, "missing_evidence_imputed": False}
            stream = io.StringIO()
            with patch.object(analysis, "analyze_cohort", return_value=result) as call, redirect_stdout(stream):
                self.assertEqual(main(arguments), expected)
            self.assertEqual(json.loads(stream.getvalue())["status"], status)
            self.assertEqual(call.call_count, 1)
        errors = io.StringIO()
        with patch.object(analysis, "analyze_cohort", side_effect=ValueError("fixture failure")), redirect_stderr(errors):
            self.assertEqual(main(arguments), 1)
        self.assertIn("P5.5 error (ValueError)", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
