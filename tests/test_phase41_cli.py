"""Teacher CLI side-effect boundaries and the fixed P4.1 budget."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import phase41 as cli
from world_models.teacher.protocol import load_protocol, plan, protocol_identity


class Phase41CliTests(unittest.TestCase):
    def call(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(list(args))
        return code, json.loads(output.getvalue())

    def test_plan_is_fixed_and_never_constructs_training_runtime(self):
        with patch.object(cli, "_runtime_setup", side_effect=AssertionError("runtime")), \
                patch.object(cli, "_runner", side_effect=AssertionError("runner")):
            for seed in (0, 1):
                code, report = self.call("plan", "--seed", str(seed))
                self.assertEqual(code, 0)
                self.assertEqual(report["teacher_actor_optimizer_steps"], 400)
                self.assertEqual(report["teacher_critic_optimizer_steps"], 400)
                self.assertEqual(report["imagined_joint_step_cap"], 96000)
                self.assertFalse(report["training_executed"])
                self.assertFalse(report["full_d_pipeline_implemented"])
                p = report["protocol"]
                self.assertEqual((p["teacher_cycles"], p["starts"], p["horizon"]), (25, 256, 15))
                self.assertEqual((p["initial_transitions"], p["initial_model_updates"]), (50000, 4500))
                self.assertEqual((p["new_environment_interactions"], p["new_model_updates"]), (0, 0))

    def test_protocol_copies_cannot_mutate_registration(self):
        before = protocol_identity()
        p = load_protocol()
        p["teacher_cycles"] = 999
        p["master_seeds"].append(99)
        self.assertEqual(protocol_identity(), before)
        self.assertEqual(plan(0)["protocol"]["teacher_cycles"], 25)
        for seed in (True, -1, 2, "0"):
            with self.assertRaises(ValueError):
                plan(seed)

    def test_status_and_stop_do_not_load_weights_or_runtime(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(cli, "_runtime_setup", side_effect=AssertionError("runtime")), \
                patch.object(cli, "_runner", side_effect=AssertionError("runner")):
            root = Path(folder)
            (root / "run_manifest.json").write_text(json.dumps({"schema_version": "p41-teacher-run-v1"}))
            status = {"status": "running", "teacher_cycles_completed": 7}
            (root / "status.json").write_text(json.dumps(status))
            self.assertEqual(self.call("status", "--run-dir", folder), (0, status))
            code, response = self.call("stop", "--run-dir", folder)
            self.assertEqual((code, response["status"]), (0, "stop_requested"))
            self.assertIn("requested_at", json.loads((root / "stop.request").read_text()))

    def test_stop_rejects_a_local_student_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "run_manifest.json").write_text(json.dumps({"schema_version": "p35-run-v1"}))
            with self.assertRaisesRegex(ValueError, "P4.1"):
                self.call("stop", "--run-dir", folder)
            self.assertFalse((root / "stop.request").exists())

    def test_prepare_initializes_only_and_uses_explicit_selected_checkpoint(self):
        controller = Mock()
        controller.status.return_value = {"status": "prepared", "teacher_cycles_completed": 0}
        runner = Mock()
        runner.prepare.return_value = controller
        with patch.object(cli, "_runtime_setup") as runtime, patch.object(cli, "_runner", return_value=runner):
            code, report = self.call("prepare", "--seed", "1", "--run-dir", "fresh",
                                     "--initial-selection", "selected.json", "--device", "cpu",
                                     "--precision", "float32", "--data-root", "data", "--statistics-root", "stats")
        runtime.assert_called_once_with()
        runner.prepare.assert_called_once_with("fresh", initial_selection="selected.json", master_seed=1,
            data_root="data", statistics_root="stats", device="cpu", precision="float32")
        runner.resume.assert_not_called()
        controller.run.assert_not_called()
        self.assertEqual(code, 0)
        self.assertFalse(report["training_executed"])

    def test_run_resumes_and_restores_signal_handlers(self):
        controller = Mock()
        controller.run.return_value = {"status": "stopped", "teacher_cycles_completed": 8}
        runner = Mock()
        runner.resume.return_value = controller
        with patch.object(cli, "_runtime_setup") as runtime, patch.object(cli, "_runner", return_value=runner), \
                patch.object(cli.signal, "signal", return_value=cli.signal.SIG_DFL) as signals:
            code, report = self.call("run", "--run-dir", "existing", "--device", "cuda:0", "--max-cycles", "2")
        runtime.assert_called_once_with()
        runner.resume.assert_called_once_with("existing", device="cuda:0")
        controller.run.assert_called_once_with(max_cycles=2)
        runner.prepare.assert_not_called()
        self.assertEqual((code, report["status"]), (0, "stopped"))
        self.assertEqual(signals.call_count, 4)
        callback = signals.call_args_list[0].args[1]
        with redirect_stderr(io.StringIO()):
            callback(None, None)
        controller.request_stop.assert_called_once_with()

    def test_negative_cycle_limit_and_unregistered_seed_are_rejected(self):
        for args in (("run", "--run-dir", "x", "--max-cycles", "0"),
                     ("plan", "--seed", "2")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(list(args))
            self.assertEqual(error.exception.code, 2)

    def test_runtime_setup_preserves_explicit_cublas_configuration(self):
        with patch.dict(cli.os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}), \
                patch.object(cli.torch, "set_num_threads") as threads, \
                patch.object(cli.torch, "set_num_interop_threads", side_effect=RuntimeError), \
                patch.object(cli.torch, "use_deterministic_algorithms") as deterministic, \
                patch.object(cli.torch.backends, "cudnn", SimpleNamespace(benchmark=True)):
            cli._runtime_setup()
            self.assertEqual(cli.os.environ["CUBLAS_WORKSPACE_CONFIG"], ":16:8")
            threads.assert_called_once_with(1)
            deterministic.assert_called_once_with(True)
            self.assertFalse(cli.torch.backends.cudnn.benchmark)

    def test_verification_failure_is_not_success(self):
        report = {"status": "failed", "tests_run": 3, "failures": 1, "errors": 0, "skipped": 0,
                  "research_training_executed": False, "phase4_acceptance": False}
        with patch.object(cli, "verify", return_value=report):
            code, result = self.call("verify", "--output-dir", "unused")
        self.assertEqual(code, 1)
        self.assertEqual(result, report)
