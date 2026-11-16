"""The acceptance CLI keeps native execution and training opt-in and explicit."""
from contextlib import redirect_stdout, redirect_stderr
import io
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.phase36 import main


class Phase36CLITests(unittest.TestCase):
    def invoke(self, argv):
        with redirect_stdout(io.StringIO()) as output:
            code = main(argv)
        return code, output.getvalue()

    def test_import_does_not_load_training_or_world_model(self):
        command = ("import sys; import scripts.phase36; "
                   "assert not any(x.startswith('world_models.models') for x in sys.modules); "
                   "assert 'world_models.cycle.controller' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_verify_routes_to_cpu_checks_and_reports_failure(self):
        result = {"status": "failed", "tests_run": 1, "failures": 1, "errors": 0,
                  "skipped": 0, "phase3_acceptance": False}
        with patch("scripts.phase36.verify", return_value=result) as verify, patch("scripts.phase36._runtime_setup") as setup:
            code, output = self.invoke(["verify", "--output-dir", "fresh"])
        verify.assert_called_once_with("fresh")
        setup.assert_not_called()
        self.assertEqual(code, 1)
        self.assertIn('"phase3_acceptance": false', output)

    def test_accept_missing_native_inputs_keeps_incomplete_exit_status(self):
        aggregate = Mock(return_value={"status": "pending_real_evidence", "phase3_acceptance": False})
        with patch.dict(sys.modules, {"scripts.phase36_support.gate": SimpleNamespace(aggregate_acceptance=aggregate)}):
            with patch("scripts.phase36._runtime_setup") as setup:
                code, _ = self.invoke(["accept", "--checks", "checks.json", "--output", "phase3_acceptance.json"])
        setup.assert_not_called()
        self.assertEqual(code, 2)
        aggregate.assert_called_once_with(invariant_report="checks.json", mechanics_report=None,
            baseline_report=None, trained_report=None, output_path="phase3_acceptance.json")

    def test_accept_full_evidence_returns_zero_only_when_accepted(self):
        aggregate = Mock(return_value={"status": "passed", "phase3_acceptance": True})
        with patch.dict(sys.modules, {"scripts.phase36_support.gate": SimpleNamespace(aggregate_acceptance=aggregate)}):
            code, _ = self.invoke(["accept", "--checks", "c", "--mechanics", "m", "--baseline", "b",
                                   "--trained", "t", "--output", "o.json"])
        self.assertEqual(code, 0)
        self.assertEqual(aggregate.call_args.kwargs["trained_report"], "t")

    def test_native_evaluation_has_explicit_actor_and_paired_bank(self):
        evaluate = Mock(return_value={"status": "complete", "phase3_acceptance": False})
        module = SimpleNamespace(evaluate_actor_export=evaluate)
        with patch.dict(sys.modules, {"scripts.phase36_support.evaluation": module}), patch("scripts.phase36._runtime_setup") as setup:
            code, _ = self.invoke(["evaluate", "--actor-export", "actor", "--output-dir", "eval",
                                   "--device", "cuda:2", "--source-device", "cuda:1"])
        self.assertEqual(code, 0)
        setup.assert_called_once()
        evaluate.assert_called_once_with("actor", "eval", episodes=8, bank_seed=3600, device="cuda:2",
                                          mode="stochastic", precision="float32", source_device="cuda:1")

    def test_mechanics_is_a_separate_explicit_command(self):
        mechanics = Mock(return_value={"status": "passed", "phase3_acceptance": False})
        module = SimpleNamespace(run_mechanics=mechanics)
        with patch.dict(sys.modules, {"scripts.phase36_support.mechanics": module}), patch("scripts.phase36._runtime_setup"):
            code, _ = self.invoke(["mechanics", "--output-dir", "mechanics"])
        self.assertEqual(code, 0)
        mechanics.assert_called_once_with("mechanics", seed=3606, cycles=40, batch_size=32)

    def test_test_bank_and_implicit_training_are_not_cli_options(self):
        for args in (["evaluate", "--actor-export", "a", "--output-dir", "o", "--split", "test"],
                     ["train", "--output-dir", "o"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)

    def test_runtime_sets_cuda_determinism_before_any_allocation(self):
        from scripts.phase36 import _runtime_setup
        fake_torch = SimpleNamespace(set_num_threads=Mock(), set_num_interop_threads=Mock(),
                                     use_deterministic_algorithms=Mock(),
                                     backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True)))
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.dict(os.environ, {}, clear=True):
            _runtime_setup()
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertFalse(fake_torch.backends.cudnn.benchmark)
            fake_torch.use_deterministic_algorithms.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()
