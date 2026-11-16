"""Explicit operator commands retain training and prospective-cohort boundaries."""
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

from scripts import phase52 as cli


class Phase52CliTests(unittest.TestCase):
    def call(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(list(args))
        return code, json.loads(output.getvalue())

    @staticmethod
    def candidate():
        return {"status": "paired_training_registered", "candidate_only": True,
                "training_tools_ready": True, "execution_ready": False, "cohort_frozen": False,
                "cohort_acceptance": False, "pending": ["P5.4 endpoint evaluator", "P5.5 paired reader"]}

    def test_plan_status_and_stop_never_initialize_training_runtime(self):
        operations = (("plan", (), {"training_executed": False}),
                      ("read_status", ("existing",), {"status": "running", "local_cycles_completed": 7}),
                      ("request_stop", ("existing",), {"status": "stop_requested"}))
        for method, arguments, report in operations:
            cohort = Mock()
            getattr(cohort, method).return_value = report
            command = "plan" if method == "plan" else "status" if method == "read_status" else "stop"
            args = (command,) if command == "plan" else (command, "--run-dir", "existing")
            with patch.object(cli, "_cohort", return_value=cohort), \
                    patch.object(cli, "_runtime_setup", side_effect=AssertionError("runtime")), \
                    patch.object(cli, "_torch", side_effect=AssertionError("torch")):
                self.assertEqual(self.call(*args), (0, report))
            getattr(cohort, method).assert_called_once_with(*arguments)
            cohort.prepare_run.assert_not_called()
            cohort.resume_run.assert_not_called()

    def test_import_and_static_plan_do_not_import_torch_vizdoom_or_learners(self):
        code = (
            "import sys; from scripts.phase52 import main; main(['plan']); "
            "assert 'torch' not in sys.modules; assert 'vizdoom' not in sys.modules; "
            "assert 'world_models_paired.runner' not in sys.modules; "
            "assert 'world_models_campaign.controller' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=cli.ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["execution_ready"])

    def test_register_binds_explicit_candidate_and_verification_without_training(self):
        cohort = Mock()
        report = {**self.candidate(), "source_sha256": {"large.py": "a" * 64},
                  "report_path": "/saved/cohort_manifest.json"}
        cohort.register_cohort.return_value = report
        with patch.object(cli, "_cohort", return_value=cohort), \
                patch.object(cli, "_runtime_setup", side_effect=AssertionError("runtime")):
            code, summary = self.call("register", "--output-dir", "fresh", "--registration", "p51.json",
                                      "--verification-report", "checks.json")
        cohort.register_cohort.assert_called_once_with("fresh", registration="p51.json",
                                                       verification_report="checks.json")
        self.assertEqual(code, 0)
        self.assertEqual(summary, {**self.candidate(), "report_path": "/saved/cohort_manifest.json"})
        cohort.prepare_run.assert_not_called()
        cohort.resume_run.assert_not_called()

    def test_inspect_preserves_candidate_only_flags_and_exact_path(self):
        cohort = Mock()
        cohort.inspect_cohort.return_value = self.candidate()
        with patch.object(cli, "_cohort", return_value=cohort), \
                patch.object(cli, "_runtime_setup", side_effect=AssertionError("runtime")):
            code, report = self.call("inspect", "--cohort", "saved/cohort_manifest.json")
        cohort.inspect_cohort.assert_called_once_with("saved/cohort_manifest.json")
        self.assertEqual(code, 0)
        self.assertFalse(report["execution_ready"])
        self.assertFalse(report["cohort_frozen"])
        self.assertTrue(report["training_tools_ready"])

    def test_prepare_requires_explicit_development_opt_in_and_never_runs_controller(self):
        for opted_in in (False, True):
            cohort = Mock()
            cohort.prepare_run.return_value = {"status": "prepared", "variant": "B"}
            if not opted_in:
                cohort.prepare_run.side_effect = ValueError("P5.4 and P5.5 remain pending")
            args = ["prepare", "--cohort", "c", "--run-dir", "r", "--variant", "B", "--seed", "0",
                    "--device", "cpu"]
            if opted_in:
                args.append("--development-only")
            with patch.object(cli, "_cohort", return_value=cohort), patch.object(cli, "_runtime_setup"):
                if opted_in:
                    code, report = self.call(*args)
                    self.assertEqual(code, 0)
                    self.assertFalse(report["training_executed"])
                else:
                    with self.assertRaisesRegex(ValueError, "pending"):
                        self.call(*args)
            cohort.prepare_run.assert_called_once_with("r", cohort="c", variant="B", seed=0, device="cpu",
                                                       development_only=opted_in)
            cohort.resume_run.assert_not_called()

    def test_D_prepare_keeps_requested_seed_device_and_no_adoption_option(self):
        cohort = Mock()
        cohort.prepare_run.return_value = {"status": "prepared", "variant": "D", "stage": "teacher"}
        with patch.object(cli, "_cohort", return_value=cohort), patch.object(cli, "_runtime_setup"):
            code, report = self.call("prepare", "--cohort", "candidate", "--run-dir", "D1", "--variant", "D",
                "--seed", "1", "--device", "cuda:0", "--development-only")
        cohort.prepare_run.assert_called_once_with("D1", cohort="candidate", variant="D", seed=1,
                                                   device="cuda:0", development_only=True)
        self.assertEqual((code, report["stage"]), (0, "teacher"))

    def test_run_resumes_only_exact_branch_and_restores_handlers(self):
        cohort, controller = Mock(), Mock()
        cohort.resume_run.return_value = controller
        controller.run.return_value = {"status": "stopped", "local_cycles_completed": 6}
        with patch.object(cli, "_cohort", return_value=cohort), patch.object(cli, "_runtime_setup"), \
                patch.object(cli.signal, "signal", return_value=cli.signal.SIG_DFL) as signals:
            code, report = self.call("run", "--run-dir", "existing", "--device", "cuda:0", "--max-units", "2")
        cohort.resume_run.assert_called_once_with("existing", device="cuda:0")
        controller.run.assert_called_once_with(max_units=2)
        cohort.prepare_run.assert_not_called()
        self.assertEqual((code, report["status"]), (0, "stopped"))
        self.assertEqual(signals.call_count, 4)
        with redirect_stderr(io.StringIO()):
            signals.call_args_list[0].args[1](None, None)
        controller.request_stop.assert_called_once_with()

    def test_unlimited_run_and_failure_both_restore_signal_handlers(self):
        for fail in (False, True):
            cohort, controller = Mock(), Mock()
            cohort.resume_run.return_value = controller
            controller.run.return_value = {"status": "complete"}
            if fail:
                controller.run.side_effect = RuntimeError("checkpoint rejected")
            with patch.object(cli, "_cohort", return_value=cohort), patch.object(cli, "_runtime_setup"), \
                    patch.object(cli.signal, "signal", return_value=cli.signal.SIG_DFL) as signals:
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "checkpoint rejected"):
                        self.call("run", "--run-dir", "r")
                else:
                    self.assertEqual(self.call("run", "--run-dir", "r"), (0, {"status": "complete"}))
            controller.run.assert_called_once_with(max_units=None)
            self.assertEqual(signals.call_count, 4)

    def test_registration_failure_does_not_retry_or_launch_training(self):
        cohort = Mock()
        cohort.register_cohort.side_effect = ValueError("stale receipt")
        with patch.object(cli, "_cohort", return_value=cohort), \
                patch.object(cli, "_runtime_setup", side_effect=AssertionError("runtime")):
            with self.assertRaisesRegex(ValueError, "stale receipt"):
                self.call("register", "--output-dir", "f", "--registration", "r", "--verification-report", "v")
        self.assertEqual(cohort.register_cohort.call_count, 1)
        cohort.prepare_run.assert_not_called()

    def test_invalid_inputs_and_evaluation_or_adoption_commands_rejected(self):
        cases = (("prepare", "--run-dir", "x", "--variant", "B", "--seed", "0"),
                 ("prepare", "--cohort", "c", "--run-dir", "r", "--variant", "A", "--seed", "0"),
                 ("prepare", "--cohort", "c", "--run-dir", "r", "--variant", "D", "--seed", "2"),
                 ("register", "--output-dir", "f", "--registration", "r"), ("inspect",),
                 ("run", "--run-dir", "r", "--max-units", "0"),
                 ("run", "--run-dir", "r", "--max-units", "-1"), ("stop",),
                 ("evaluate",), ("adopt",), ("train-all",))
        for arguments in cases:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(list(arguments))
            self.assertEqual(error.exception.code, 2)

    def test_runtime_setup_preserves_operator_cublas_and_restores_torch_thread_count(self):
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

    def test_verify_status_is_nonzero_on_failure_and_keeps_cohort_pending(self):
        report = {"status": "complete_p52_implementation", "tests_run": 4, "failures": 0, "errors": 0,
                  "skipped": 0, "research_training_executed": False, "research_evaluation_executed": False,
                  "execution_ready": False}
        for failed in (False, True):
            if failed:
                report["status"] = "failed"
            with patch.object(cli, "verify", return_value=report), \
                    patch.object(cli, "_cohort", side_effect=AssertionError("cohort")):
                code, summary = self.call("verify", "--output-dir", "proof")
            self.assertEqual(code, int(failed))
            self.assertFalse(summary["execution_ready"])
            self.assertEqual(summary["report_path"], str(Path("proof") / "implementation_checks.json"))

    def test_verification_writes_receipt_source_test_ids_log_and_research_boundaries(self):
        class Fixture(unittest.TestCase):
            def runTest(self):
                pass

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(cli, "source_identity", return_value={"fixture.py": "a" * 64}), \
                patch.object(cli.unittest.defaultTestLoader, "loadTestsFromNames",
                             return_value=unittest.TestSuite([Fixture()])):
            output = Path(folder) / "proof"
            report = cli.verify(output)
            self.assertEqual((report["status"], report["tests_run"]), ("complete_p52_implementation", 1))
            self.assertEqual(report["test_modules"], list(cli.TEST_MODULES))
            self.assertEqual(len(report["passed_tests"]), 1)
            for key in ("research_training_executed", "research_evaluation_executed", "research_replay_used",
                        "native_environment_evaluation_executed", "gpu_runtime_measured", "cohort_acceptance",
                        "execution_ready", "learned_student_quality_established"):
                self.assertFalse(report[key], key)
            path = output / "implementation_checks.json"
            self.assertEqual(path.with_suffix(".sha256").read_text().strip(),
                             f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
            self.assertEqual(report["tests_log"], {"path": "tests.txt", "sha256":
                hashlib.sha256((output / "tests.txt").read_bytes()).hexdigest()})
            with self.assertRaises(FileExistsError):
                cli.verify(output)

    def test_zero_tests_skips_changed_sources_or_failed_test_cannot_pass_verification(self):
        class Fixture(unittest.TestCase):
            def runTest(self):
                pass

        class SkippedFixture(unittest.TestCase):
            @unittest.skip("fixture")
            def runTest(self):
                pass

        class FailedFixture(unittest.TestCase):
            def runTest(self):
                self.fail("fixture")

        cases = (([Fixture()], [{"a": "old"}, {"a": "new"}]),
                 ([SkippedFixture()], [{"a": "same"}, {"a": "same"}]),
                 ([FailedFixture()], [{"a": "same"}, {"a": "same"}]),
                 ([], [{"a": "same"}, {"a": "same"}]))
        for fixtures, identities in cases:
            with tempfile.TemporaryDirectory() as folder, \
                    patch.object(cli, "source_identity", side_effect=identities), \
                    patch.object(cli.unittest.defaultTestLoader, "loadTestsFromNames",
                                 return_value=unittest.TestSuite(fixtures)):
                self.assertEqual(cli.verify(Path(folder) / "proof")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
