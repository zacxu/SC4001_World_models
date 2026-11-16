"""P4.4 operator commands preserve explicit training and checkpoint boundaries."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import phase44 as cli


class Phase44CliTests(unittest.TestCase):
    def call(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(list(args))
        return code, json.loads(output.getvalue())

    @staticmethod
    def manifest(root, *, schema="p44-d-campaign-run-v1"):
        path = root / "run_manifest.json"
        content = json.dumps({"schema_version": schema}).encode()
        path.write_bytes(content)
        path.with_suffix(".sha256").write_text(f"{hashlib.sha256(content).hexdigest()}  {path.name}\n")

    def test_plan_does_not_initialize_or_load_training_runtime(self):
        report = {"training_executed": False, "master_seed": 1}
        with patch.object(cli, "_runtime_setup", side_effect=AssertionError("runtime")), \
                patch.object(cli, "_controller", side_effect=AssertionError("controller")), \
                patch.object(cli, "_torch", side_effect=AssertionError("torch")), \
                patch.object(cli, "plan", return_value=report) as preview:
            self.assertEqual(self.call("plan", "--seed", "1"), (0, report))
        preview.assert_called_once_with(1)

    def test_status_and_stop_use_no_training_runtime(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(cli, "_runtime_setup", side_effect=AssertionError("runtime")), \
                patch.object(cli, "_controller", side_effect=AssertionError("controller")), \
                patch.object(cli, "_torch", side_effect=AssertionError("torch")):
            root = Path(folder)
            self.manifest(root)
            status = {"status": "running", "stage": "local", "local_cycles_completed": 7}
            (root / "status.json").write_text(json.dumps(status))
            self.assertEqual(self.call("status", "--run-dir", folder), (0, status))
            code, report = self.call("stop", "--run-dir", folder)
            self.assertEqual((code, report["status"]), (0, "stop_requested"))
            self.assertIn("requested_at", json.loads((root / "stop.request").read_text()))
            self.assertEqual(json.loads((root / "status.json").read_text()), status)

    def test_fresh_process_status_stop_and_preview_do_not_import_torch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.manifest(root)
            (root / "status.json").write_text(json.dumps({"status": "running"}))
            code = (
                "import sys; from scripts.phase44 import main; "
                "assert 'torch' not in sys.modules; "
                "main(['status','--run-dir',sys.argv[1]]); "
                "main(['stop','--run-dir',sys.argv[1]]); "
                "main(['plan','--seed','0']); "
                "assert 'torch' not in sys.modules"
            )
            result = subprocess.run([sys.executable, "-c", code, folder], cwd=cli.ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_stop_rejects_wrong_stage_and_unreceipted_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for schema in ("p41-teacher-run-v1", "p42-distillation-run-v1", "p43-local-refinement-run-v1"):
                self.manifest(root, schema=schema)
                with self.assertRaisesRegex(ValueError, "P4.4"):
                    self.call("stop", "--run-dir", folder)
            self.manifest(root)
            (root / "run_manifest.sha256").unlink()
            with self.assertRaisesRegex(ValueError, "receipt"):
                self.call("stop", "--run-dir", folder)
            self.manifest(root)
            (root / "run_manifest.sha256").write_text("0" * 64 + "  run_manifest.json\n")
            with self.assertRaisesRegex(ValueError, "receipt"):
                self.call("stop", "--run-dir", folder)
            self.assertFalse((root / "stop.request").exists())

    def test_prepare_only_binds_explicit_initial_selection(self):
        controller = Mock()
        controller.status.return_value = {"status": "prepared", "stage": "teacher"}
        kind = Mock()
        kind.prepare.return_value = controller
        with patch.object(cli, "_runtime_setup") as runtime, patch.object(cli, "_controller", return_value=kind):
            code, report = self.call("prepare", "--run-dir", "fresh", "--seed", "1",
                                     "--initial-selection", "selection.json", "--config-root", "c",
                                     "--replay-root", "r", "--normalization-dir", "n", "--device", "cpu")
        runtime.assert_called_once_with()
        kind.prepare.assert_called_once_with("fresh", seed=1, initial_selection="selection.json",
                                             config_root="c", replay_root="r", normalization_dir="n", device="cpu",
                                             precision="bfloat16")
        kind.resume.assert_not_called()
        controller.run.assert_not_called()
        self.assertEqual(code, 0)
        self.assertFalse(report["training_executed"])

    def test_adopt_validates_completed_refinement_without_training_or_restart(self):
        controller = Mock()
        controller.status.return_value = {"status": "complete"}
        kind = Mock()
        kind.adopt.return_value = controller
        with patch.object(cli, "_runtime_setup"), patch.object(cli, "_controller", return_value=kind):
            code, report = self.call("adopt", "--run-dir", "fresh", "--refinement-run", "finished",
                                     "--device", "cpu")
        kind.adopt.assert_called_once_with("fresh", refinement_run="finished", device="cpu")
        kind.prepare.assert_not_called()
        controller.run.assert_not_called()
        self.assertEqual(code, 0)
        self.assertFalse(report["training_executed"])

    def test_run_resumes_units_and_restores_handlers(self):
        controller = Mock()
        controller.run.return_value = {"status": "stopped", "stage": "distillation"}
        kind = Mock()
        kind.resume.return_value = controller
        with patch.object(cli, "_runtime_setup"), patch.object(cli, "_controller", return_value=kind), \
                patch.object(cli.signal, "signal", return_value=cli.signal.SIG_DFL) as signals:
            code, report = self.call("run", "--run-dir", "existing", "--device", "cuda:0", "--max-units", "2")
        kind.resume.assert_called_once_with("existing", device="cuda:0")
        controller.run.assert_called_once_with(max_units=2)
        kind.prepare.assert_not_called()
        self.assertEqual((code, report["status"]), (0, "stopped"))
        self.assertEqual(signals.call_count, 4)
        with redirect_stderr(io.StringIO()):
            signals.call_args_list[0].args[1](None, None)
        controller.request_stop.assert_called_once_with()

    def test_failure_also_restores_handlers_and_unlimited_run_passes_none(self):
        for failure in (False, True):
            controller = Mock()
            controller.run.return_value = {"status": "complete"}
            if failure:
                controller.run.side_effect = RuntimeError("failure")
            kind = Mock()
            kind.resume.return_value = controller
            with patch.object(cli, "_runtime_setup"), patch.object(cli, "_controller", return_value=kind), \
                    patch.object(cli.signal, "signal", return_value=cli.signal.SIG_DFL) as signals:
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "failure"):
                        self.call("run", "--run-dir", "existing")
                else:
                    self.assertEqual(self.call("run", "--run-dir", "existing"), (0, {"status": "complete"}))
            controller.run.assert_called_once_with(max_units=None)
            self.assertEqual(signals.call_count, 4)

    def test_missing_inputs_invalid_seed_and_nonpositive_limits_rejected(self):
        for args in (("run", "--run-dir", "x", "--max-units", "0"),
                     ("run", "--run-dir", "x", "--max-units", "-1"),
                     ("plan", "--seed", "2"), ("prepare", "--run-dir", "x", "--seed", "0"),
                     ("adopt", "--run-dir", "x")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(list(args))
            self.assertEqual(error.exception.code, 2)

    def test_runtime_setup_preserves_explicit_cublas_and_replay_precision(self):
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

    def test_verification_failure_is_nonzero(self):
        report = {"status": "failed", "tests_run": 3, "failures": 1, "errors": 0, "skipped": 0,
                  "research_training_executed": False, "phase4_acceptance": False}
        with patch.object(cli, "verify", return_value=report):
            self.assertEqual(self.call("verify", "--output-dir", "unused"), (1, report))

    def test_fixture_report_binds_sources_and_keeps_research_acceptance_pending(self):
        class Fixture(unittest.TestCase):
            def runTest(self):
                pass

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(cli, "source_identity", return_value={"fixture.py": "a" * 64}), \
                patch.object(cli.unittest.defaultTestLoader, "loadTestsFromNames", return_value=unittest.TestSuite([Fixture()])):
            output = Path(folder) / "verify"
            report = cli.verify(output)
            self.assertEqual((report["status"], report["tests_run"]), ("complete_p44_implementation", 1))
            for key in ("research_training_executed", "research_replay_used", "gpu_runtime_measured",
                        "learned_student_quality_measured", "phase4_acceptance"):
                self.assertFalse(report[key], key)
            path = output / "implementation_checks.json"
            self.assertEqual(path.with_suffix(".sha256").read_text().strip(),
                             f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
            with self.assertRaises(FileExistsError):
                cli.verify(output)

    def test_verification_rejects_changed_sources_and_skips(self):
        class Fixture(unittest.TestCase):
            def runTest(self):
                pass

        class SkippedFixture(unittest.TestCase):
            @unittest.skip("fixture")
            def runTest(self):
                pass

        for case, identities in ((Fixture(), [{"a": "old"}, {"a": "new"}]),
                                 (SkippedFixture(), [{"a": "same"}, {"a": "same"}])):
            with tempfile.TemporaryDirectory() as folder, \
                    patch.object(cli, "source_identity", side_effect=identities), \
                    patch.object(cli.unittest.defaultTestLoader, "loadTestsFromNames", return_value=unittest.TestSuite([case])):
                self.assertEqual(cli.verify(Path(folder) / "verify")["status"], "failed")
