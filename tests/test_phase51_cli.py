"""Static protocol commands cannot launch training or final test evaluation."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import phase51 as cli


class Phase51CliTests(unittest.TestCase):
    def call(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(list(args))
        return code, json.loads(output.getvalue())

    @staticmethod
    def candidate():
        return {"status": "protocol_registered", "protocol_frozen": True,
                "candidate_only": True, "execution_ready": False, "cohort_frozen": False,
                "pending": ["P5.2 matched reduced B runner", "P5.4 endpoint evaluator",
                            "P5.5 paired endpoint reader", "prospective cohort eligibility and source freeze"]}

    def test_plan_calls_only_static_protocol_preview(self):
        report = {"execution_ready": False, "training_executed": False}
        with patch.object(cli, "plan", return_value=report) as plan, \
                patch.object(cli, "register_protocol", side_effect=AssertionError("registration")), \
                patch.object(cli, "inspect_registration", side_effect=AssertionError("inspection")):
            self.assertEqual(self.call("plan"), (0, report))
        plan.assert_called_once_with()

    def test_import_and_real_plan_do_not_load_torch_vizdoom_or_learner_modules(self):
        source = (
            "import sys; from scripts.phase51 import main; main(['plan']); "
            "assert 'torch' not in sys.modules; assert 'vizdoom' not in sys.modules; "
            "assert 'world_models_campaign.controller' not in sys.modules; "
            "assert 'world_models.teacher.runner' not in sys.modules; "
            "assert 'world_models_study.registration' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", source], cwd=cli.ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_register_requires_explicit_selection_and_verification_and_keeps_execution_blocked(self):
        report = self.candidate()
        report["source_sha256"] = {f"large/source_{index}.py": "a" * 64 for index in range(2000)}
        with patch.object(cli, "register_protocol", return_value=report) as register:
            code, summary = self.call("register", "--output-dir", "fresh",
                                      "--initial-selection", "selection.json", "--verification-report", "checks.json")
        register.assert_called_once_with("fresh", initial_selection="selection.json",
            data_root=str(cli.ROOT / "data/initial_replay/pilot"),
            statistics_root=str(cli.ROOT / "artifacts/phase23/normalization"),
            config_root=str(cli.ROOT / "configs"), verification_report="checks.json")
        self.assertEqual(code, 0)
        self.assertEqual(summary, {**self.candidate(),
                                 "report_path": str(Path("fresh") / "protocol_registration.json")})
        self.assertNotIn("source_sha256", summary)

    def test_register_preserves_explicit_metadata_roots_and_report_path(self):
        report = {**self.candidate(), "report_path": "/actual/protocol_registration.json"}
        with patch.object(cli, "register_protocol", return_value=report) as register:
            code, summary = self.call("register", "--output-dir", "fresh", "--initial-selection", "s",
                "--verification-report", "v", "--data-root", "r", "--statistics-root", "n", "--config-root", "c")
        register.assert_called_once_with("fresh", initial_selection="s", data_root="r", statistics_root="n",
                                         config_root="c", verification_report="v")
        self.assertEqual((code, summary["report_path"]), (0, "/actual/protocol_registration.json"))
        self.assertFalse(summary["execution_ready"])

    def test_inspect_validates_exact_registration_without_rewriting_or_promoting(self):
        report = {**self.candidate(), "registration_identity": "b" * 64}
        with patch.object(cli, "inspect_registration", return_value=report) as inspect, \
                patch.object(cli, "register_protocol", side_effect=AssertionError("rewrite")):
            code, summary = self.call("inspect", "--registration", "saved/protocol_registration.json")
        inspect.assert_called_once_with("saved/protocol_registration.json")
        self.assertEqual((code, summary["registration_identity"]), (0, "b" * 64))
        self.assertTrue(summary["protocol_frozen"])
        self.assertFalse(summary["execution_ready"])
        self.assertFalse(summary["cohort_frozen"])

    def test_failed_registration_or_inspection_propagates_without_retry(self):
        cases = (("register_protocol", ("register", "--output-dir", "f", "--initial-selection", "s",
                                       "--verification-report", "v")),
                 ("inspect_registration", ("inspect", "--registration", "old")))
        for helper, arguments in cases:
            with patch.object(cli, helper, side_effect=ValueError("invalid immutable input")) as operation:
                with self.assertRaisesRegex(ValueError, "immutable input"):
                    self.call(*arguments)
            self.assertEqual(operation.call_count, 1)

    def test_missing_inputs_and_training_evaluation_commands_are_rejected(self):
        for arguments in (("register", "--output-dir", "f", "--initial-selection", "s"),
                          ("register", "--output-dir", "f", "--verification-report", "v"),
                          ("inspect",), ("verify",), ("run",), ("train",), ("evaluate",)):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(list(arguments))
            self.assertEqual(error.exception.code, 2)

    def test_verify_dispatch_keeps_registration_and_research_separate(self):
        report = {"status": "complete_p51_implementation", "tests_run": 3, "failures": 0,
                  "errors": 0, "skipped": 0, "research_training_executed": False,
                  "research_evaluation_executed": False, "execution_ready": False, "cohort_frozen": False}
        with patch.object(cli, "verify", return_value=report) as verify, \
                patch.object(cli, "register_protocol", side_effect=AssertionError("registration")):
            code, summary = self.call("verify", "--output-dir", "check")
        verify.assert_called_once_with("check")
        self.assertEqual(code, 0)
        self.assertFalse(summary["execution_ready"])
        self.assertEqual(summary["report_path"], str(Path("check") / "implementation_checks.json"))
        report["status"] = "failed"
        with patch.object(cli, "verify", return_value=report):
            self.assertEqual(self.call("verify", "--output-dir", "failed")[0], 1)

    def test_static_fixture_report_binds_sources_log_and_test_ids_without_numerical_runtime(self):
        class Fixture(unittest.TestCase):
            def runTest(self):
                pass

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(cli, "source_identity", return_value={"fixture.py": "c" * 64}), \
                patch.object(cli.unittest.defaultTestLoader, "loadTestsFromNames",
                             return_value=unittest.TestSuite([Fixture()])):
            output = Path(folder) / "verify"
            report = cli.verify(output)
            self.assertEqual((report["status"], report["tests_run"]), ("complete_p51_implementation", 1))
            self.assertEqual(report["source_sha256"], {"fixture.py": "c" * 64})
            self.assertEqual(report["test_modules"], list(cli.TEST_MODULES))
            self.assertEqual(len(report["passed_tests"]), 1)
            for key in ("research_training_executed", "research_evaluation_executed",
                        "native_environment_evaluation_executed", "research_replay_used",
                        "fixture_optimizer_updates_executed", "gpu_runtime_measured", "protocol_frozen",
                        "execution_ready", "cohort_frozen"):
                self.assertFalse(report[key], key)
            path = output / "implementation_checks.json"
            self.assertEqual(path.with_suffix(".sha256").read_text().strip(),
                             f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
            self.assertEqual(report["tests_log"], {"path": "tests.txt", "sha256":
                             hashlib.sha256((output / "tests.txt").read_bytes()).hexdigest()})
            with self.assertRaises(FileExistsError):
                cli.verify(output)

    def test_changed_sources_skips_or_zero_executed_tests_cannot_pass_verification(self):
        class Fixture(unittest.TestCase):
            def runTest(self):
                pass

        class SkippedFixture(unittest.TestCase):
            @unittest.skip("fixture")
            def runTest(self):
                pass

        cases = (([Fixture()], [{"a": "old"}, {"a": "new"}]),
                 ([SkippedFixture()], [{"a": "same"}, {"a": "same"}]),
                 ([], [{"a": "same"}, {"a": "same"}]))
        for fixtures, sources in cases:
            with tempfile.TemporaryDirectory() as folder, \
                    patch.object(cli, "source_identity", side_effect=sources), \
                    patch.object(cli.unittest.defaultTestLoader, "loadTestsFromNames",
                                 return_value=unittest.TestSuite(fixtures)):
                self.assertEqual(cli.verify(Path(folder) / "verify")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
