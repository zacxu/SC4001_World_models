"""Distillation command boundaries, receipts and explicit workload dispatch."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import phase42 as cli
from world_models.distillation.protocol import load_protocol, plan, protocol_identity


class Phase42CliTests(unittest.TestCase):
    def call(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(list(args))
        return code, json.loads(output.getvalue())

    @staticmethod
    def manifest(root, *, schema="p42-distillation-run-v1"):
        path = root / "run_manifest.json"
        content = json.dumps({"schema_version": schema}).encode()
        path.write_bytes(content)
        path.with_suffix(".sha256").write_text(f"{hashlib.sha256(content).hexdigest()}  {path.name}\n")

    def test_plan_is_read_only_and_uses_registered_seed(self):
        report = {"training_executed": False, "master_seed": 1}
        with patch.object(cli, "_runtime_setup", side_effect=AssertionError("runtime")), \
                patch.object(cli, "_runner", side_effect=AssertionError("runner")), \
                patch.object(cli, "plan", return_value=report) as plan:
            self.assertEqual(self.call("plan", "--seed", "1"), (0, report))
        plan.assert_called_once_with(1)

    def test_registered_plan_has_fixed_budget_and_owned_protocol(self):
        identity = protocol_identity()
        with patch.object(cli, "_runtime_setup", side_effect=AssertionError("runtime")), \
                patch.object(cli, "_runner", side_effect=AssertionError("runner")):
            for seed in (0, 1):
                code, report = self.call("plan", "--seed", str(seed))
                self.assertEqual((code, report["student_optimizer_steps"], report["imagined_joint_step_cap"]),
                                 (0, 100, 1920))
                self.assertEqual(report["expected_buffer_passes"], 25)
                self.assertFalse(report["training_executed"])
                self.assertFalse(report["full_d_pipeline_implemented"])
                recipe = report["protocol"]
                self.assertEqual((recipe["trajectories"], recipe["sequences_per_update"], recipe["horizon"]),
                                 (128, 32, 15))
                self.assertEqual((recipe["initial_transitions"], recipe["initial_model_updates"]), (50000, 4500))
                self.assertEqual((recipe["new_environment_interactions"], recipe["new_model_updates"],
                                  recipe["critic_updates"], recipe["teacher_updates"]), (0, 0, 0, 0))
        recipe = load_protocol()
        recipe["master_seeds"].append(2)
        recipe["updates"] = 999
        self.assertEqual(protocol_identity(), identity)
        for seed in (True, -1, 2, "0"):
            with self.assertRaises(ValueError):
                plan(seed)

    def test_status_and_stop_need_no_training_runtime(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(cli, "_runtime_setup", side_effect=AssertionError("runtime")), \
                patch.object(cli, "_runner", side_effect=AssertionError("runner")):
            root = Path(folder)
            self.manifest(root)
            status = {"status": "running", "distillation_updates": 7}
            (root / "status.json").write_text(json.dumps(status))
            self.assertEqual(self.call("status", "--run-dir", folder), (0, status))
            code, response = self.call("stop", "--run-dir", folder)
            self.assertEqual((code, response["status"]), (0, "stop_requested"))
            self.assertIn("requested_at", json.loads((root / "stop.request").read_text()))

    def test_stop_rejects_teacher_and_unreceipted_or_tampered_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.manifest(root, schema="p41-teacher-run-v1")
            with self.assertRaisesRegex(ValueError, "P4.2"):
                self.call("stop", "--run-dir", folder)
            self.assertFalse((root / "stop.request").exists())
            self.manifest(root)
            (root / "run_manifest.sha256").unlink()
            with self.assertRaisesRegex(ValueError, "receipt"):
                self.call("stop", "--run-dir", folder)
            self.manifest(root)
            (root / "run_manifest.sha256").write_text("0" * 64 + "  run_manifest.json\n")
            with self.assertRaisesRegex(ValueError, "receipt"):
                self.call("stop", "--run-dir", folder)
            self.assertFalse((root / "stop.request").exists())

    def test_prepare_initializes_from_explicit_teacher_without_running(self):
        controller = Mock()
        controller.status.return_value = {"status": "prepared", "distillation_updates": 0}
        runner = Mock()
        runner.prepare.return_value = controller
        with patch.object(cli, "_runtime_setup") as runtime, patch.object(cli, "_runner", return_value=runner):
            code, report = self.call("prepare", "--run-dir", "fresh", "--teacher-run", "teacher", "--device", "cpu")
        runtime.assert_called_once_with()
        runner.prepare.assert_called_once_with("fresh", teacher_run="teacher", device="cpu")
        runner.resume.assert_not_called()
        controller.run.assert_not_called()
        self.assertEqual(code, 0)
        self.assertFalse(report["training_executed"])

    def test_run_resumes_additional_units_and_restores_signal_handlers(self):
        controller = Mock()
        controller.run.return_value = {"status": "stopped", "distillation_updates": 8}
        runner = Mock()
        runner.resume.return_value = controller
        with patch.object(cli, "_runtime_setup") as runtime, patch.object(cli, "_runner", return_value=runner), \
                patch.object(cli.signal, "signal", return_value=cli.signal.SIG_DFL) as signals:
            code, report = self.call("run", "--run-dir", "existing", "--device", "cuda:0", "--max-units", "2")
        runtime.assert_called_once_with()
        runner.resume.assert_called_once_with("existing", device="cuda:0")
        controller.run.assert_called_once_with(max_units=2)
        runner.prepare.assert_not_called()
        self.assertEqual((code, report["status"]), (0, "stopped"))
        self.assertEqual(signals.call_count, 4)
        callback = signals.call_args_list[0].args[1]
        with redirect_stderr(io.StringIO()):
            callback(None, None)
        controller.request_stop.assert_called_once_with()

    def test_run_restores_signal_handlers_when_training_raises(self):
        controller = Mock()
        controller.run.side_effect = RuntimeError("failure")
        runner = Mock()
        runner.resume.return_value = controller
        with patch.object(cli, "_runtime_setup"), patch.object(cli, "_runner", return_value=runner), \
                patch.object(cli.signal, "signal", return_value=cli.signal.SIG_DFL) as signals:
            with self.assertRaisesRegex(RuntimeError, "failure"):
                self.call("run", "--run-dir", "existing")
        self.assertEqual(signals.call_count, 4)

    def test_invalid_unit_limit_seed_and_missing_teacher_are_rejected(self):
        for args in (("run", "--run-dir", "x", "--max-units", "0"),
                     ("run", "--run-dir", "x", "--max-units", "-1"),
                     ("plan", "--seed", "2"), ("prepare", "--run-dir", "x")):
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

    def test_verify_uses_fresh_output_and_binds_report_to_sources(self):
        class Fixture(unittest.TestCase):
            def runTest(self):
                self.assertTrue(True)

        suite = unittest.TestSuite([Fixture()])
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(cli, "source_identity", return_value={"fixture.py": "a" * 64}), \
                patch.object(cli.unittest.defaultTestLoader, "loadTestsFromNames", return_value=suite):
            output = Path(folder) / "verify"
            report = cli.verify(output)
            self.assertEqual(report["status"], "complete_p42_implementation")
            self.assertEqual(report["tests_run"], 1)
            self.assertFalse(report["research_training_executed"])
            self.assertFalse(report["learned_student_quality_measured"])
            path = output / "implementation_checks.json"
            self.assertEqual(path.with_suffix(".sha256").read_text().strip(),
                             f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
            with self.assertRaises(FileExistsError):
                cli.verify(output)

    def test_verify_source_change_or_skip_is_not_success(self):
        class Fixture(unittest.TestCase):
            def runTest(self):
                pass

        class SkippedFixture(unittest.TestCase):
            @unittest.skip("fixture")
            def runTest(self):
                pass

        for case, identities in ((Fixture(), [{"a": "old"}, {"a": "new"}]),
                                 (SkippedFixture(), [{"a": "same"}, {"a": "same"}])):
            with self.subTest(case=type(case).__name__), tempfile.TemporaryDirectory() as folder, \
                    patch.object(cli, "source_identity", side_effect=identities), \
                    patch.object(cli.unittest.defaultTestLoader, "loadTestsFromNames", return_value=unittest.TestSuite([case])):
                self.assertEqual(cli.verify(Path(folder) / "verify")["status"], "failed")
