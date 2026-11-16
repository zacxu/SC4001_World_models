"""Operator commands retain source-freeze, training and evaluation boundaries."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import phase5 as cli


class TinyPass(unittest.TestCase):
    def runTest(self):
        self.assertTrue(True)


class Phase5CliTests(unittest.TestCase):
    def call(self, *args):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = cli.main(list(args))
        return code, json.loads(stream.getvalue())

    def test_freeze_requires_explicit_matrix_current_proof_and_nonuse_attestation(self):
        backend = Mock()
        backend.freeze_cohort.return_value = {"status": "cohort_frozen", "execution_ready": True,
                                               "cohort_frozen": True, "cohort_identity": "a" * 64}
        with patch.object(cli, "_cohort", return_value=backend):
            code, result = self.call("freeze", "--matrix", "candidate.json", "--verification-report", "proof.json",
                                     "--output-dir", "fresh", "--confirm-test-bank-unused")
        self.assertEqual(code, 0)
        self.assertTrue(result["execution_ready"])
        backend.freeze_cohort.assert_called_once_with("fresh", matrix="candidate.json", verification_report="proof.json",
                                                       confirm_test_bank_unused=True)
        backend.prepare_run.assert_not_called()
        backend.resume_run.assert_not_called()

    def test_inspect_and_status_do_not_initialize_gpu_training_runtime(self):
        backend = Mock()
        backend.inspect_cohort.return_value = {"status": "cohort_frozen", "cohort_identity": "a" * 64}
        backend.audit_run.return_value = {"status": "incomplete", "local_cycles_completed": 7}
        with patch.object(cli, "_cohort", return_value=backend), \
             patch("scripts.phase52._runtime_setup", side_effect=AssertionError("training setup forbidden")):
            self.assertEqual(self.call("inspect", "--cohort", "c.json")[0], 0)
            self.assertEqual(self.call("status", "--run-dir", "run")[1]["local_cycles_completed"], 7)
        backend.audit_run.assert_called_once_with("run")

    def test_prepare_is_fresh_and_never_runs_optimization(self):
        backend = Mock()
        backend.prepare_run.return_value = {"status": "prepared", "variant": "B", "local_cycles_completed": 0}
        with patch.object(cli, "_cohort", return_value=backend), patch("scripts.phase52._runtime_setup"):
            code, result = self.call("prepare", "--cohort", "c.json", "--variant", "B", "--seed", "1", "--device", "cpu")
        self.assertEqual(code, 0)
        self.assertEqual(result["local_cycles_completed"], 0)
        backend.prepare_run.assert_called_once_with("c.json", variant="B", seed=1, device="cpu")
        backend.resume_run.assert_not_called()

    def test_run_uses_only_selected_run_and_restores_signal_handlers(self):
        backend, worker = Mock(), Mock()
        worker.run.return_value = {"status": "incomplete", "local_cycles_completed": 2}
        backend.resume_run.return_value = worker
        before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        with patch.object(cli, "_cohort", return_value=backend), patch("scripts.phase52._runtime_setup"):
            result = self.call("run", "--run-dir", "selected", "--device", "cpu", "--max-units", "2")
        self.assertEqual(result[0], 0)
        backend.resume_run.assert_called_once_with("selected", device="cpu")
        worker.run.assert_called_once_with(max_units=2)
        self.assertEqual({s: signal.getsignal(s) for s in before}, before)

    def test_run_failure_preserves_failure_and_restores_signal_handlers(self):
        backend, worker = Mock(), Mock()
        worker.run.side_effect = RuntimeError("fixture error")
        backend.resume_run.return_value = worker
        before = signal.getsignal(signal.SIGINT)
        with patch.object(cli, "_cohort", return_value=backend), patch("scripts.phase52._runtime_setup"):
            with self.assertRaisesRegex(RuntimeError, "fixture error"):
                self.call("run", "--run-dir", "selected", "--device", "cpu")
        self.assertEqual(signal.getsignal(signal.SIGINT), before)

    def test_stop_only_writes_requested_pause_through_api(self):
        backend = Mock()
        backend.request_stop.return_value = {"status": "stop_requested", "run_dir": "run"}
        with patch.object(cli, "_cohort", return_value=backend):
            self.assertEqual(self.call("stop", "--run-dir", "run")[1]["status"], "stop_requested")
        backend.request_stop.assert_called_once_with("run")
        backend.resume_run.assert_not_called()

    def test_zero_work_limit_invalid_variant_or_missing_required_args_fail_parser(self):
        for args in (("run", "--run-dir", "run", "--max-units", "0"),
                     ("prepare", "--cohort", "c", "--variant", "A", "--seed", "0"),
                     ("status",), ("evaluate",)):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.call(*args)

    def test_verify_uses_only_fixtures_and_publishes_receipted_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = {"synthetic.py": "a" * 64}
            suite = unittest.TestSuite([TinyPass()])
            with patch.object(cli, "source_identity", return_value=sources), \
                 patch.object(cli.unittest.defaultTestLoader, "loadTestsFromNames", return_value=suite):
                r = cli.verify(Path(directory) / "check")
            self.assertEqual(r["status"], "complete_p535455_implementation")
            self.assertEqual(r["tests_run"], 1)
            self.assertFalse(r["research_training_executed"])
            self.assertFalse(r["research_evaluation_executed"])
            self.assertTrue((Path(directory) / "check/implementation_checks.sha256").is_file())

    def test_verify_failure_is_nonzero_and_never_freezes_or_trains(self):
        failed = {"status": "failed", "tests_run": 0, "failures": 0, "errors": 1, "skipped": 0,
                  "research_training_executed": False, "research_evaluation_executed": False}
        with patch.object(cli, "verify", return_value=failed), patch.object(cli, "_cohort") as backend:
            self.assertEqual(self.call("verify", "--output-dir", "fresh")[0], 1)
        backend.assert_not_called()
