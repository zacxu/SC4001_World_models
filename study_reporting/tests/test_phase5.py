"""Reporting orchestration with synthetic files and an explicitly mocked renderer."""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from study_reporting import common, evidence, phase5
from study_reporting.gate import build_acceptance
from study_reporting.tests.test_gate import AcceptanceFixture


class CompletionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        fixture = self.fixture = AcceptanceFixture(self.root)
        self.output = self.root / "completion"
        self.cohort = self.root / "cohort.json"
        self.proof = self.root / "verification.json"
        fixture.context["cohort"]["report_path"] = str(self.cohort)
        evidence.write_report(self.cohort, fixture.context["cohort"])
        evidence.write_report(self.proof, fixture.proof)
        fixture.context["references"] = [common.reference(self.cohort)]
        fixture.resources["per_run"] = []
        fixture.secondary["training_diagnostics"] = []
        for run in fixture.context["runs"]:
            counts = deepcopy(fixture.context["ledger"]["runs"][f"{run['variant']}/seed_{run['master_seed']}"]["counters"])
            counts["imagined_joint_transitions_total"] = sum(run[f"imagined_joint_steps_{stage}"] for stage in ("local", "teacher", "buffer"))
            fixture.resources["per_run"].append({"case": f"{run['variant']}/seed_{run['master_seed']}",
                "variant": run["variant"], "master_seed": run["master_seed"], "logical_counters": counts,
                "recorded_scientific_unit_elapsed_seconds": 10.0,
                "actual_GPU_occupancy_seconds": None, "current_retained_file_bytes": 100})
            fixture.secondary["training_diagnostics"].append({"variant": run["variant"], "master_seed": run["master_seed"],
                "teacher_PPO": None, "distillation": None, "local_PPO": {"entropy": 0.4}})
        self.api = SimpleNamespace(inspect_cohort=Mock(return_value=fixture.context["cohort"]))
        self.loader = Mock(side_effect=lambda *args, **kwargs: deepcopy(fixture.context))
        self.secondary_check, self.resources_check = Mock(), Mock()
        self.renderer = Mock(side_effect=self.fake_figures)
        self.components = (lambda context: deepcopy(fixture.secondary), self.secondary_check,
            lambda context: deepcopy(fixture.resources), self.resources_check, self.renderer, build_acceptance)
        for owner, name, kwargs in (
                (common, "require_server", {"return_value": None}),
                (common, "apis", {"return_value": self.api}),
                (common, "guard_output", {"side_effect": lambda output, *_: Path(output).resolve()}),
                (common, "load_context", {"side_effect": self.loader}),
                (common, "validate_context", {"side_effect": lambda context, **kwargs: common.validate_references(context["references"])}),
                (evidence, "validate_verification", {"return_value": fixture.proof}),
                (evidence, "source_identity", {"return_value": fixture.source}),
                (phase5, "_components", {"return_value": self.components})):
            self.enterContext(patch.object(owner, name, **kwargs))

    def fake_figures(self, context, resources, output):
        # This tests publication plumbing only. Figure tests independently ensure
        # an injected renderer cannot represent production scientific graphics.
        output.mkdir()
        files = []
        for stem in ("endpoint_results", "paired_effects_compute"):
            for extension in ("svg", "png"):
                path = output / f"{stem}.{extension}"
                path.write_bytes(b"synthetic graphic")
                files.append(common.reference(path, receipt=False))
        data = output / "fixture_data.json"
        evidence.write_report(data, {"synthetic": True})
        return {"status": "complete", "fixtures_only": False, "actual_graphics_rendered": True,
            "matplotlib_version": "3.11.2", "files": files, "data": [common.reference(data)]}

    def finish(self):
        return phase5.complete(self.cohort, self.output, self.proof)

    def test_zero_success_pipeline_and_repeat_inspection(self):
        report = self.finish()
        self.assertEqual(report["status"], "passed", report.get("reasons"))
        self.assertEqual(report["successful_final_episodes_observed"], 0)
        self.assertFalse(report["successful_transfer_established"])
        self.assertEqual(phase5.inspect_completion(self.output / "phase5_acceptance.json"), report)
        self.loader.assert_called_once()
        self.renderer.assert_called_once()
        self.assertEqual(self.secondary_check.call_count, 2)
        self.assertEqual(self.resources_check.call_count, 2)
        self.assertFalse(report["native_environment_opened"])
        context = evidence.read_report(self.output / "evidence_context.json")
        self.assertEqual(len(context["deliverables"]["figures"]), 2)
        self.assertTrue(any(row["path"].endswith("fixture_data.json") for row in context["references"]))

    def test_missing_saved_evidence_records_failure_without_running_learners(self):
        self.loader.side_effect = ValueError("missing native saved report")
        report = self.finish()
        self.assertFalse(report["phase5_acceptance"])
        self.assertIn("missing native", report["reasons"][0])
        self.renderer.assert_not_called()
        self.assertFalse(report["research_evaluation_executed_by_gate"])
        self.assertEqual(phase5.inspect_completion(self.output / "phase5_acceptance.json"), report)

    def test_fixture_graphics_cannot_complete_scientific_report(self):
        self.renderer.side_effect = lambda *args: {"status": "complete", "fixtures_only": True,
            "actual_graphics_rendered": False, "matplotlib_version": None}
        report = self.finish()
        self.assertFalse(report["phase5_acceptance"])
        self.assertIn("actual pinned-renderer", report["reasons"][0])

    def test_missing_plot_dependency_leaves_needs_review(self):
        self.renderer.side_effect = ImportError("install reporting requirements on the server")
        report = self.finish()
        self.assertEqual(report["status"], "needs_review")
        self.assertIn("ImportError", report["reasons"][0])
        self.assertTrue((self.output / "secondary_evidence.json").is_file())

    def test_changed_reporting_or_data_source_rejected_on_inspection(self):
        self.finish()
        with patch.object(evidence, "source_identity", return_value={"changed.py": "0" * 64}):
            with self.assertRaisesRegex(ValueError, "sources changed"):
                phase5.inspect_completion(self.output / "phase5_acceptance.json")
        (self.output / "figures/fixture_data.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            phase5.inspect_completion(self.output / "phase5_acceptance.json")

    def test_same_sources_proof_cannot_validate_different_frozen_cohort(self):
        self.fixture.proof["frozen_source_sha256"] = {"different.py": "f" * 64}
        with self.assertRaisesRegex(ValueError, "frozen scientific sources"):
            self.finish()
        self.assertFalse(self.output.exists())

    def test_missing_budget_is_not_promoted_by_successful_report_generation(self):
        self.fixture.context["runs"][0]["local_cycles_completed"] = 19
        report = self.finish()
        self.assertEqual(report["status"], "needs_review")
        self.assertIn("registered value 20", " ".join(report["reasons"]))

    def test_resource_csv_keeps_unknown_time_blank(self):
        self.finish()
        import csv
        with (self.output / "resources.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["actual_GPU_occupancy_seconds"] == "" for row in rows))
        self.assertTrue(all(row["recorded_scientific_unit_elapsed_seconds"] == "10.0" for row in rows))

    def test_tampered_gate_cannot_pass_recomputed_acceptance(self):
        self.finish()
        path = self.output / "phase5_acceptance.json"
        report = evidence.read_report(path)
        report["successful_final_episodes_observed"] = 1
        path.unlink()
        path.with_suffix(".sha256").unlink()
        evidence.write_report(path, report)
        with self.assertRaisesRegex(ValueError, "no longer reproduces"):
            phase5.inspect_completion(path)


class CompletionCliTests(unittest.TestCase):
    def invoke(self, args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return phase5.main(args)

    def test_cli_dispatch_and_exit_status(self):
        with patch.object(phase5, "complete", return_value={"status": "passed"}) as complete:
            self.assertEqual(self.invoke(["complete", "--cohort", "cohort", "--output-dir", "out", "--verification-report", "proof"]), 0)
            complete.assert_called_once_with("cohort", "out", "proof")
        with patch.object(phase5, "inspect_completion", return_value={"status": "needs_review"}):
            self.assertEqual(self.invoke(["inspect", "--report", "report"]), 1)
        with patch.object(evidence, "verify", return_value={"status": evidence.STATUS}):
            self.assertEqual(self.invoke(["verify", "--output-dir", "check"]), 0)
        with patch.object(evidence, "verify", return_value={"status": "failed"}):
            self.assertEqual(self.invoke(["verify", "--output-dir", "check"]), 1)

    def test_errors_cannot_be_reported_as_success(self):
        with patch.object(phase5, "complete", side_effect=ValueError("invalid source")):
            self.assertEqual(self.invoke(["complete", "--cohort", "c", "--output-dir", "o", "--verification-report", "v"]), 1)

    def test_cli_requires_explicit_evidence_and_has_no_training_device_option(self):
        for args in (["complete", "--cohort", "c"], ["complete", "--cohort", "c", "--output-dir", "o", "--verification-report", "v", "--device", "cuda:0"]):
            with self.subTest(args=args), self.assertRaises(SystemExit) as caught:
                self.invoke(args)
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
