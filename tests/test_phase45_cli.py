"""Acceptance commands never launch research optimization or native evaluation."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import phase45 as cli


class Phase45CliTests(unittest.TestCase):
    def call(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(list(args))
        return code, json.loads(output.getvalue())

    @staticmethod
    def coverage(passed=True):
        return {f"requirement_{index}": {"status": "passed" if passed else "missing",
                                        "test_ids": ["fixture"],
                                        "missing_test_ids": [] if passed else ["fixture"]}
                for index in range(8)}

    def test_accept_uses_completed_campaign_and_explicit_verification_report_without_training(self):
        report = {"status": "passed", "phase4_engineering_acceptance": True,
                  "phase4_acceptance": True, "research_training_executed": False,
                  "bound_files": [{"path": f"large-evidence-{index}"} for index in range(2000)]}
        with patch.object(cli, "_runtime_setup") as runtime, \
                patch.object(cli, "_accept", return_value=report) as accept:
            self.assertEqual(self.call("accept", "--run-dir", "completed", "--verification-report", "checks.json",
                                       "--output-dir", "audit"), (0, {
                "status": "passed", "phase4_engineering_acceptance": True, "phase4_acceptance": True,
                "reasons": [], "report_path": str(Path("audit") / "phase4_acceptance.json"),
                "learned_student_quality_established": False, "cohort_acceptance": False}))
        runtime.assert_called_once_with()
        accept.assert_called_once_with("completed", "audit", verification_report="checks.json", device="cpu")

    def test_accept_retains_explicit_device_without_changing_visible_gpu_environment(self):
        report = {"phase4_engineering_acceptance": True}
        with patch.dict(cli.os.environ, {"CUDA_VISIBLE_DEVICES": "3"}), \
                patch.object(cli, "_runtime_setup"), patch.object(cli, "_accept", return_value=report) as accept:
            code, summary = self.call("accept", "--run-dir", "finished", "--verification-report", "v",
                                      "--output-dir", "o", "--device", "cuda:0")
            self.assertEqual((code, summary["phase4_engineering_acceptance"]), (0, True))
            self.assertEqual(cli.os.environ["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(accept.call_args.kwargs["device"], "cuda:0")

    def test_accept_failure_is_nonzero_and_does_not_relabel_quality(self):
        for report in ({"status": "needs_review", "phase4_engineering_acceptance": False,
                        "reasons": ["incomplete campaign"]},
                       {"status": "passed", "phase4_acceptance": True}):
            with patch.object(cli, "_runtime_setup"), patch.object(cli, "_accept", return_value=report):
                code, summary = self.call("accept", "--run-dir", "r", "--verification-report", "v",
                                          "--output-dir", "o")
                self.assertEqual((code, summary["phase4_engineering_acceptance"]), (1, False))
                self.assertEqual(summary["reasons"], report.get("reasons", []))
                self.assertFalse(summary["learned_student_quality_established"])
                self.assertFalse(summary["cohort_acceptance"])

    def test_accept_propagates_failure_without_retrying_or_training(self):
        with patch.object(cli, "_runtime_setup"), \
                patch.object(cli, "_accept", side_effect=ValueError("campaign is incomplete")) as accept:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self.call("accept", "--run-dir", "r", "--verification-report", "v", "--output-dir", "o")
        self.assertEqual(accept.call_count, 1)

    def test_missing_inputs_and_training_commands_are_rejected(self):
        for arguments in (("accept", "--run-dir", "r", "--output-dir", "o"),
                          ("accept", "--run-dir", "r", "--verification-report", "v"),
                          ("verify",), ("run", "--run-dir", "r"), ("prepare", "--run-dir", "r")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(list(arguments))
            self.assertEqual(error.exception.code, 2)

    def test_import_and_help_do_not_load_torch_or_research_learner(self):
        source = (
            "import sys; from scripts import phase45; "
            "assert 'torch' not in sys.modules; "
            "assert 'world_models_campaign.controller' not in sys.modules; "
            "assert 'world_models.teacher.runner' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", source], cwd=cli.ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_setup_preserves_cublas_override_and_uses_deterministic_kernels(self):
        torch = Mock()
        torch.set_num_interop_threads.side_effect = RuntimeError("already initialized")
        torch.backends.cudnn = SimpleNamespace(benchmark=True)
        with patch.dict(cli.os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}), \
                patch.object(cli, "_torch", return_value=torch):
            cli._runtime_setup()
            self.assertEqual(cli.os.environ["CUBLAS_WORKSPACE_CONFIG"], ":16:8")
        torch.set_num_threads.assert_called_once_with(1)
        torch.use_deterministic_algorithms.assert_called_once_with(True)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_verify_dispatch_does_not_audit_campaign_or_launch_training(self):
        report = {"status": "complete_p45_implementation", "tests_run": 8, "failures": 0,
                  "errors": 0, "skipped": 0, "all_requirements_passed": True,
                  "research_training_executed": False, "phase4_acceptance": False}
        with patch.object(cli, "verify", return_value=report) as verify, \
                patch.object(cli, "_runtime_setup", side_effect=AssertionError("accept runtime")), \
                patch.object(cli, "_accept", side_effect=AssertionError("campaign")):
            self.assertEqual(self.call("verify", "--output-dir", "fixture-check"), (0, report))
        verify.assert_called_once_with("fixture-check")
        report["status"] = "failed"
        with patch.object(cli, "verify", return_value=report):
            self.assertEqual(self.call("verify", "--output-dir", "other")[0], 1)

    def test_fixture_report_binds_sources_coverage_and_log_and_keeps_research_acceptance_pending(self):
        class Fixture(unittest.TestCase):
            def runTest(self):
                pass

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(cli, "source_identity", return_value={"fixture.py": "a" * 64}), \
                patch.object(cli, "requirement_coverage", return_value=self.coverage()) as coverage, \
                patch.object(cli, "verification_contract_identity", return_value="b" * 64), \
                patch.object(cli.unittest.defaultTestLoader, "loadTestsFromNames",
                             return_value=unittest.TestSuite([Fixture()])):
            output = Path(folder) / "verify"
            report = cli.verify(output)
            self.assertEqual((report["status"], report["tests_run"]), ("complete_p45_implementation", 1))
            self.assertEqual(report["passed_tests"], sorted(coverage.call_args.args[0]))
            self.assertTrue(report["all_requirements_passed"])
            self.assertEqual(report["verification_contract_sha256"], "b" * 64)
            for key in ("research_training_executed", "research_replay_used", "gpu_runtime_measured",
                        "learned_student_quality_measured", "phase4_acceptance"):
                self.assertFalse(report[key], key)
            path = output / "implementation_checks.json"
            self.assertEqual(path.with_suffix(".sha256").read_text().strip(),
                             f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
            self.assertEqual(report["tests_log"], {"path": "tests.txt", "sha256":
                             hashlib.sha256((output / "tests.txt").read_bytes()).hexdigest()})
            with self.assertRaises(FileExistsError):
                cli.verify(output)

    def test_verification_rejects_changed_sources_skipped_or_missing_requirement_evidence(self):
        class Fixture(unittest.TestCase):
            def runTest(self):
                pass

        class SkippedFixture(unittest.TestCase):
            @unittest.skip("fixture")
            def runTest(self):
                pass

        cases = ((Fixture(), [{"a": "old"}, {"a": "new"}], self.coverage()),
                 (SkippedFixture(), [{"a": "same"}, {"a": "same"}], self.coverage()),
                 (Fixture(), [{"a": "same"}, {"a": "same"}], self.coverage(False)),
                 (Fixture(), [{"a": "same"}, {"a": "same"}], {}))
        for case, identities, coverage in cases:
            with tempfile.TemporaryDirectory() as folder, \
                    patch.object(cli, "source_identity", side_effect=identities), \
                    patch.object(cli, "requirement_coverage", return_value=coverage), \
                    patch.object(cli, "verification_contract_identity", return_value="c" * 64), \
                    patch.object(cli.unittest.defaultTestLoader, "loadTestsFromNames",
                                 return_value=unittest.TestSuite([case])):
                self.assertEqual(cli.verify(Path(folder) / "verify")["status"], "failed")

    def test_zero_executed_tests_cannot_publish_a_passing_implementation_report(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(cli, "source_identity", return_value={"a": "same"}), \
                patch.object(cli, "requirement_coverage", return_value=self.coverage()), \
                patch.object(cli, "verification_contract_identity", return_value="d" * 64), \
                patch.object(cli.unittest.defaultTestLoader, "loadTestsFromNames", return_value=unittest.TestSuite()):
            self.assertEqual(cli.verify(Path(folder) / "verify")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
