"""P3.5 CLI routing without allocating a GPU or starting a research run."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import phase35 as cli
from world_models.cycle.controller import CycleOptions
from world_models.cycle.schedule import Schedule
from world_models.infrastructure.config import resolve_config


class Phase35CliTests(unittest.TestCase):
    def test_recover_routes_explicit_fresh_destination_without_training(self):
        result = {"status": "recovered", "local_cycles_completed": 30,
                  "remaining_initial_units": 11, "training_executed": False}
        with patch.object(cli, "_runtime_setup") as runtime, \
             patch("world_models.cycle.recovery.recover_run", return_value=result) as recover, \
             patch.object(cli.CycleController, "prepare", side_effect=AssertionError("must not restart")), \
             patch.object(cli.CycleController, "resume", side_effect=AssertionError("must not train")):
            code, report = self.call("recover", "--source-run", "old-run", "--run-dir", "fresh-run", "--device", "cuda:0")
        runtime.assert_called_once_with()
        recover.assert_called_once_with("old-run", "fresh-run", device="cuda:0")
        self.assertEqual((code, report), (0, result))

    def test_training_provenance_binds_shortened_selection_reader_and_freezer(self):
        identity = cli.production_source_identity()
        for key in ("scripts/phase24_freeze.py", "src/world_models/training/selection.py"):
            self.assertIn(key, identity)
            self.assertEqual(len(identity[key]), 64)

    def call(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(list(arguments))
        return code, json.loads(output.getvalue())

    def no_training(self, stack):
        for owner, name in ((cli.CycleController, "prepare"), (cli.CycleController, "resume"),
                            (cli, "load_case"), (cli, "_runtime_setup")):
            stack.enter_context(patch.object(owner, name, side_effect=AssertionError("read-only CLI loaded training runtime")))

    def test_plan_pilot_and_final_budgets_are_read_only(self):
        with ExitStack() as stack:
            self.no_training(stack)
            for profile, seed, blocks, cycles, budget in (("pilot", 0, 6, 220, 200000),
                                                        ("final", 1001, 38, 860, 1000000)):
                with self.subTest(profile=profile):
                    code, report = self.call("plan", "--task", "relay", "--seed", str(seed), "--profile", profile)
                    self.assertEqual(code, 0)
                    self.assertEqual(report["variant"], "B")
                    self.assertEqual(report["online_blocks"], blocks)
                    self.assertEqual(report["total_local_cycles"], cycles)
                    self.assertEqual(report["real_budget"], budget)
                    self.assertEqual(report["block_targets"][0], 50000)
                    self.assertEqual(report["block_targets"][-1], budget)
                    self.assertEqual(report["ppo_optimizer_steps_per_cycle"], 16)
                    self.assertTrue(report["native_environment_required_for_run"])
                    self.assertFalse(report["training_executed"])

    def test_status_only_reads_saved_json_without_weights_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.no_training(stack)
            saved = {"status": "stopped", "real_joint_transitions": 75000, "model_updates": 5250}
            Path(directory, "status.json").write_text(json.dumps(saved), encoding="utf-8")
            code, report = self.call("status", "--run-dir", directory)
            self.assertEqual(code, 0)
            self.assertEqual(report, saved)

    def test_stop_writes_durable_request_without_loading_runtime_or_weights(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.no_training(stack)
            root = Path(directory)
            (root / "run_manifest.json").write_text("{}", encoding="utf-8")
            code, report = self.call("stop", "--run-dir", directory)
            self.assertEqual(code, 0)
            self.assertEqual(report, {"status": "stop_requested", "run_dir": directory})
            request = json.loads((root / "stop.request").read_text(encoding="utf-8"))
            self.assertEqual(set(request), {"requested_at"})
            self.assertTrue(request["requested_at"])

    def test_stop_rejects_unprepared_run_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.no_training(stack)
            missing = Path(directory) / "missing"
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main(["stop", "--run-dir", str(missing)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(missing.exists())

    def test_cublas_determinism_is_configured_before_prepare_and_resume(self):
        settings = resolve_config("relay", "B", "pilot").settings
        prepared = SimpleNamespace(directory=Path("fixture-run"), schedule=Schedule.from_experiment(settings))
        controller = Mock()
        controller.run.return_value = {"status": "stopped"}

        def prepare(*_args, **_kwargs):
            self.assertEqual(cli.os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")
            return prepared

        def resume(*_args, **_kwargs):
            self.assertEqual(cli.os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")
            return controller

        for command in ("prepare", "run"):
            with self.subTest(command=command), patch.dict(cli.os.environ), \
                 patch.object(cli.torch, "set_num_threads"), patch.object(cli.torch, "set_num_interop_threads"), \
                 patch.object(cli.torch, "use_deterministic_algorithms") as deterministic, \
                 patch.object(cli.torch.backends, "cudnn", SimpleNamespace(benchmark=True)), \
                 patch.object(cli, "load_case", return_value=(SimpleNamespace(settings=settings), [object()], object(), {})), \
                 patch.object(cli.CycleController, "prepare", side_effect=prepare), \
                 patch.object(cli.CycleController, "resume", side_effect=resume), \
                 patch.object(cli.signal, "signal", return_value=cli.signal.SIG_DFL):
                cli.os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
                arguments = [command, "--run-dir", "fixture-run"]
                if command == "prepare":
                    arguments += ["--task", "relay", "--seed", "0", "--initial-checkpoint", "initial.pt"]
                code, _ = self.call(*arguments)
                self.assertEqual(code, 0)
                deterministic.assert_called_once_with(True)
                self.assertFalse(cli.torch.backends.cudnn.benchmark)

    def test_runtime_setup_preserves_explicit_cublas_configuration(self):
        with patch.dict(cli.os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}), \
             patch.object(cli.torch, "set_num_threads"), patch.object(cli.torch, "set_num_interop_threads", side_effect=RuntimeError), \
             patch.object(cli.torch, "use_deterministic_algorithms"), \
             patch.object(cli.torch.backends, "cudnn", SimpleNamespace(benchmark=True)):
            cli._runtime_setup()
            self.assertEqual(cli.os.environ["CUBLAS_WORKSPACE_CONFIG"], ":16:8")
            self.assertFalse(cli.torch.backends.cudnn.benchmark)

    def test_prepare_forwards_explicit_sources_device_and_checkpoint_options(self):
        settings = resolve_config("cover", "B", "pilot").settings
        initial, stats, ignored_second = object(), object(), object()
        prepared = SimpleNamespace(directory=Path("fixture-run"), schedule=Schedule.from_experiment(settings))
        with patch.object(cli, "_runtime_setup") as runtime, \
             patch.object(cli, "load_case", return_value=(SimpleNamespace(settings=settings), [initial, ignored_second], stats, {})) as load, \
             patch.object(cli.CycleController, "prepare", return_value=prepared) as prepare, \
             patch.object(cli.CycleController, "resume", side_effect=AssertionError("prepare resumed a run")):
            code, report = self.call(
                "prepare", "--task", "cover", "--seed", "1", "--profile", "pilot",
                "--run-dir", "fixture-run", "--initial-checkpoint", "initial.pt",
                "--data-root", "fixture-data", "--statistics-root", "fixture-stats",
                "--config-root", "fixture-configs", "--device", "cpu", "--precision", "float32",
                "--checkpoint-cycles", "3", "--checkpoint-model-updates", "7", "--model-microbatch", "2")
        runtime.assert_called_once_with()
        load.assert_called_once_with("cover", 1, data_root="fixture-data", statistics_root="fixture-stats",
                                     config_root="fixture-configs", profile="pilot")
        prepare.assert_called_once_with("fixture-run", settings=settings, master_seed=1,
            initial_dataset=initial, normalization=stats, initial_checkpoint="initial.pt", device="cpu",
            options=CycleOptions(precision="float32", checkpoint_cycles=3, checkpoint_model_updates=7, model_microbatch=2))
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "prepared")
        self.assertEqual(report["run_dir"], "fixture-run")
        self.assertFalse(report["training_executed"])

    def test_prepare_defaults_bind_one_current_visible_gpu_and_no_training(self):
        settings = resolve_config("relay", "B", "pilot").settings
        prepared = SimpleNamespace(directory=Path("fixture-run"), schedule=Schedule.from_experiment(settings))
        with patch.object(cli, "_runtime_setup"), \
             patch.object(cli, "load_case", return_value=(SimpleNamespace(settings=settings), [object()], object(), {})), \
             patch.object(cli.CycleController, "prepare", return_value=prepared) as prepare:
            code, report = self.call("prepare", "--task", "relay", "--seed", "0",
                                     "--run-dir", "fixture-run", "--initial-checkpoint", "initial.pt")
        self.assertEqual(code, 0)
        self.assertEqual(prepare.call_args.kwargs["device"], "cuda:0")
        self.assertEqual(prepare.call_args.kwargs["options"], CycleOptions())
        self.assertFalse(report["training_executed"])

    def test_prepare_selection_routes_explicit_pilot_authorization_without_checkpoint_override(self):
        settings = resolve_config("relay", "B", "pilot").settings
        prepared = SimpleNamespace(directory=Path("fixture-run"), schedule=Schedule.from_experiment(settings))
        with patch.object(cli, "_runtime_setup"), \
             patch.object(cli, "load_case", return_value=(SimpleNamespace(settings=settings), [object()], object(), {})), \
             patch.object(cli.CycleController, "prepare", return_value=prepared) as prepare:
            code, report = self.call("prepare", "--task", "relay", "--seed", "0",
                "--run-dir", "fixture-run", "--initial-selection", "frozen/selection.json")
        self.assertEqual(code, 0)
        self.assertEqual(prepare.call_args.kwargs["initial_selection"], "frozen/selection.json")
        self.assertNotIn("initial_checkpoint", prepare.call_args.kwargs)
        self.assertFalse(report["training_executed"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["prepare", "--task", "relay", "--seed", "0", "--run-dir", "fixture-run",
                      "--initial-selection", "frozen/selection.json", "--initial-checkpoint", "override.pt"])

    def test_plan_selected_checkpoint_uses_actual_count_and_keeps_original_schedule(self):
        from dataclasses import replace
        settings = resolve_config("relay", "B", "pilot").settings
        selected = replace(Schedule.from_experiment(settings), initial_model_updates=4250)
        with ExitStack() as stack:
            self.no_training(stack)
            selection = stack.enter_context(patch.object(cli.Schedule, "from_initial_selection",
                return_value=(selected, {"model_updates": 4250})))
            code, report = self.call("plan", "--task", "relay", "--seed", "0", "--initial-selection", "selection.json")
        selection.assert_called_once_with(settings, 0, "selection.json")
        self.assertEqual(code, 0)
        self.assertEqual(report["initial_model_updates"], 4250)
        self.assertEqual(report["total_model_updates"], 7250)
        self.assertEqual(report["registered_schedule"]["initial_model_updates"], 5000)
        self.assertEqual(report["study_scope"], "exploratory_pilot")
        self.assertFalse(report["training_executed"])

    def test_run_resumes_passes_unit_limit_and_signal_stop_restores_handlers(self):
        controller = Mock()
        handlers, originals, changes = {}, {}, []

        def install(signum, handler):
            changes.append((signum, handler))
            if signum not in originals:
                originals[signum] = object()
                handlers[signum] = handler
                return originals[signum]
            return handlers[signum]

        def execute(*, max_units):
            self.assertEqual(max_units, 2)
            for signum, handler in handlers.items():
                handler(signum, None)
            return {"status": "stopped", "units_executed": 2}

        controller.run.side_effect = execute
        output = io.StringIO()
        with patch.object(cli, "_runtime_setup") as runtime, \
             patch.object(cli.CycleController, "resume", return_value=controller) as resume, \
             patch.object(cli.signal, "signal", side_effect=install), redirect_stdout(output):
            code = cli.main(["run", "--run-dir", "fixture-run", "--device", "cpu", "--max-units", "2"])
        runtime.assert_called_once_with()
        resume.assert_called_once_with("fixture-run", device="cpu")
        controller.run.assert_called_once_with(max_units=2)
        self.assertEqual(controller.request_stop.call_count, len(originals))
        self.assertEqual(code, 0)
        self.assertIn('"units_executed": 2', output.getvalue())
        self.assertIn("Stop requested", output.getvalue())
        for signum, handler in originals.items():
            self.assertIn((signum, handler), changes[len(originals):])

    def test_run_without_unit_limit_and_restore_after_runtime_error(self):
        controller = Mock()
        controller.run.side_effect = RuntimeError("fixture controller failed")
        changes = []
        previous = object()
        def install(signum, handler):
            changes.append((signum, handler))
            return previous
        with patch.object(cli, "_runtime_setup"), \
             patch.object(cli.CycleController, "resume", return_value=controller), \
             patch.object(cli.signal, "signal", side_effect=install):
            with self.assertRaisesRegex(RuntimeError, "fixture controller failed"):
                cli.main(["run", "--run-dir", "fixture-run"])
        controller.run.assert_called_once_with(max_units=None)
        count = sum(hasattr(cli.signal, name) for name in ("SIGINT", "SIGTERM"))
        self.assertEqual(len(changes), 2*count)
        self.assertTrue(all(handler is previous for _, handler in changes[count:]))

    def test_verify_delegates_without_recursively_running_the_suite(self):
        for status, expected_code in (("complete_p35_implementation", 0), ("failed", 1)):
            result = {"status": status, "tests_run": 42, "failures": int(expected_code), "errors": 0,
                      "skipped": 0, "research_training_executed": False, "phase3_acceptance": False}
            with self.subTest(status=status), \
                 patch.object(cli, "verify", return_value=result) as verify, \
                 patch.object(cli, "_runtime_setup", side_effect=AssertionError("verify started runtime")):
                code, report = self.call("verify", "--output-dir", "fixture-checks")
            verify.assert_called_once_with("fixture-checks")
            self.assertEqual(code, expected_code)
            self.assertEqual(report, result)

    def test_variant_d_has_no_cli_training_option(self):
        with redirect_stderr(io.StringIO()), patch.object(cli, "_runtime_setup") as runtime:
            with self.assertRaises(SystemExit) as error:
                cli.main(["plan", "--task", "relay", "--seed", "0", "--variant", "D"])
        self.assertEqual(error.exception.code, 2)
        runtime.assert_not_called()

    def test_plan_rejects_seed_outside_selected_profile(self):
        for profile, seed in (("pilot", 1001), ("final", 0), ("pilot", -1)):
            with self.subTest(profile=profile), redirect_stderr(io.StringIO()), \
                 patch.object(cli, "_runtime_setup") as runtime:
                with self.assertRaises(SystemExit) as error:
                    cli.main(["plan", "--task", "relay", "--seed", str(seed), "--profile", profile])
            self.assertEqual(error.exception.code, 2)
            runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
