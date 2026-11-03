"""Failed futures retain resource evidence without hiding the original error."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import phase21
import tests.test_phase21_cli as cli_fixture


class ObservedResources(cli_fixture.FakeResources):
    def report(self):
        return {"cpu_ids": self.cpu_ids, "logical_cpu_limit": len(self.cpu_ids),
                "cpu_seconds_after_first_sample": 12.5, "peak_sum_rss_bytes": 424242,
                "sample_count": 99}


class CollectionResourceReportingTests(unittest.TestCase):
    def test_failed_main_future_retains_observed_cpu_ram_and_original_error(self):
        fixture = cli_fixture.Phase21CLITests("test_default_case_matrix_and_profile_seeds_are_exact")
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            mocks = fixture.mocks(stack)
            stack.enter_context(patch.object(phase21, "CollectionResources", ObservedResources))
            mocks["collect_cases"].side_effect = RuntimeError("injected worker startup failure")
            self.assertEqual(phase21.main(fixture.arguments(directory, "collect", "--task", "relay", "--seeds", "0")), 1)
            report = json.loads((Path(directory) / "reports/collect_report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("injected worker startup failure", report["error"])
            self.assertEqual(report["collection_resources"]["cpu_seconds_after_first_sample"], 12.5)
            self.assertEqual(report["collection_resources"]["peak_sum_rss_bytes"], 424242)
            mocks["audit_initial_dataset"].assert_not_called()

    def test_failed_complete_preflight_retains_resources_before_stage_report_exists(self):
        fixture = cli_fixture.Phase21CLITests("test_default_case_matrix_and_profile_seeds_are_exact")
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            mocks = fixture.mocks(stack)
            stack.enter_context(patch.object(phase21, "CollectionResources", ObservedResources))
            mocks["collect_cases"].side_effect = RuntimeError("injected diagnostic future failure")
            self.assertEqual(phase21.main(fixture.arguments(directory)), 1)
            report = json.loads((Path(directory) / "reports/phase21_acceptance.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("injected diagnostic future failure", report["error"])
            self.assertEqual(report["preflight_resources"]["sample_count"], 99)
            self.assertNotIn("preflight", report["stages"])

    def test_secondary_resource_export_error_does_not_mask_original_failure(self):
        class BrokenExport(ObservedResources):
            def report(self):
                raise AttributeError("injected resource export failure")
        original = RuntimeError("original native failure")
        report = {}
        def fail():
            raise original
        with patch.object(phase21, "CollectionResources", BrokenExport):
            with self.assertRaises(RuntimeError) as caught:
                phase21.collect_with_resource_report(report, "collection_resources", [0,1,2,3], fail)
        self.assertIs(caught.exception, original)
        self.assertEqual(report["collection_resources"]["status"], "unavailable")
        self.assertIn("resource export failure", report["collection_resources"]["error"])

    def test_missing_resource_export_still_fails_an_otherwise_successful_stage(self):
        class BrokenExport(ObservedResources):
            def report(self):
                raise AttributeError("injected resource export failure")
        report = {}
        with patch.object(phase21, "CollectionResources", BrokenExport):
            with self.assertRaises(AttributeError):
                phase21.collect_with_resource_report(report, "collection_resources", [0,1,2,3], lambda: "collected")
        self.assertEqual(report["collection_resources"]["status"], "unavailable")

    def test_successful_stage_preserves_result_and_resource_schema(self):
        report = {}
        with patch.object(phase21, "CollectionResources", ObservedResources):
            result = phase21.collect_with_resource_report(report, "collection_resources", [0,1,2,3], lambda: ["result"])
        self.assertEqual(result, ["result"])
        self.assertEqual(report["collection_resources"]["logical_cpu_limit"], 4)
        self.assertEqual(report["collection_resources"]["sample_count"], 99)


if __name__ == "__main__":
    unittest.main()
