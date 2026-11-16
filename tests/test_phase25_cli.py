"""Bounded P2.5 dispatch, structural evidence reuse and source-transfer errors."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import phase25 as cli
from world_models.evaluation import verification as checks
from world_models.evaluation import compact


class CommandTests(unittest.TestCase):
    def call(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = cli.main(arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_verify_dispatches_fixtures_without_evaluation(self):
        with patch.object(cli, "verify_implementation", return_value={"status": "passed"}) as verify, \
             patch.object(cli, "evaluate_checkpoint") as evaluate, patch.object(cli, "evaluate_campaign") as campaign:
            code, output, _ = self.call(["verify", "--output-dir", "evidence"])
        self.assertEqual(code, 0)
        verify.assert_called_once_with(Path("evidence"))
        evaluate.assert_not_called()
        campaign.assert_not_called()
        self.assertEqual(json.loads(output)["status"], "passed")

    def test_single_model_defaults_are_eight_windows_seed_zero_and_checkpoint_precision(self):
        with patch.object(cli, "evaluate_checkpoint", return_value={"pilot_readiness": {"status": "needs_review"}}) as evaluate:
            code, output, _ = self.call(["evaluate", "--checkpoint", "update_00003000.pt", "--output-dir", "screen"])
        self.assertEqual(code, 0)
        args, kwargs = evaluate.call_args
        self.assertEqual(args, (Path("update_00003000.pt"), Path("screen")))
        self.assertEqual(kwargs["windows"], 8)
        self.assertEqual(kwargs["seed"], 0)
        self.assertEqual(kwargs["device"], "cuda:0")
        self.assertIsNone(kwargs["precision"])
        self.assertFalse(json.loads(output)["P2.5_quality_acceptance"])

    def test_campaign_requires_explicit_update_and_preserves_paths_limits_and_receipt(self):
        with patch.object(cli, "evaluate_campaign", return_value={"pilot_readiness": {"status": "ready_for_exploratory_phase3"}}) as evaluate:
            code, output, _ = self.call(["campaign", "--run-root", "fits", "--checkpoint-update", "3000",
                "--output-dir", "campaign", "--windows", "64", "--seed", "17", "--device", "cpu",
                "--precision", "float32", "--data-dir", "replay", "--statistics-dir", "stats",
                "--verification-report", "checks/implementation_checks.json"])
        self.assertEqual(code, 0)
        args, kwargs = evaluate.call_args
        self.assertEqual(args, (Path("fits"), 3000, Path("campaign")))
        self.assertEqual(kwargs["windows"], 64)
        self.assertEqual(kwargs["seed"], 17)
        self.assertEqual(kwargs["data_root"], Path("replay"))
        self.assertEqual(kwargs["statistics_root"], Path("stats"))
        self.assertEqual(kwargs["verification_report"], Path("checks/implementation_checks.json"))
        self.assertEqual(json.loads(output)["pilot_readiness"]["status"], "ready_for_exploratory_phase3")

    def test_invalid_windows_seed_and_missing_checkpoint_update_fail_before_dispatch(self):
        base = ["evaluate", "--checkpoint", "model.pt", "--output-dir", "screen"]
        with patch.object(cli, "evaluate_checkpoint") as evaluate:
            for option, value in (("--windows", "0"), ("--windows", "65"), ("--windows", "1.5"), ("--seed", "-1")):
                with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main([*base, option, value])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(["campaign", "--run-root", "fits", "--output-dir", "screen"])
        evaluate.assert_not_called()

    def test_failed_readiness_and_errors_return_one_without_phase3_dispatch(self):
        arguments = ["evaluate", "--checkpoint", "model.pt", "--output-dir", "screen"]
        with patch.object(cli, "evaluate_checkpoint", return_value={"pilot_readiness": {"status": "failed", "reasons": ["fixture"]}}):
            self.assertEqual(self.call(arguments)[0], 1)
        with patch.object(cli, "evaluate_checkpoint", side_effect=ValueError("source mismatch")):
            code, _, error = self.call(arguments)
        self.assertEqual(code, 1)
        self.assertIn("source mismatch", error)

    def test_parallel_campaign_dispatches_two_gpus_and_four_independent_workers(self):
        with patch.object(cli, "evaluate_parallel_campaign", return_value={"pilot_readiness": {"status": "needs_review"}}) as parallel, \
             patch.object(cli, "evaluate_campaign") as sequential:
            code, _, _ = self.call(["campaign", "--run-root", "fits", "--checkpoint-update", "5000",
                "--output-dir", "screen", "--gpus", "0,1", "--workers-per-gpu", "2", "--windows", "64"])
        self.assertEqual(code, 0)
        sequential.assert_not_called()
        self.assertEqual(parallel.call_args.kwargs["gpus"], [0, 1])
        self.assertEqual(parallel.call_args.kwargs["workers_per_gpu"], 2)
        self.assertNotIn("device", parallel.call_args.kwargs)

    def test_parallel_flags_reject_ambiguous_devices_and_invalid_workers(self):
        base = ["campaign", "--run-root", "fits", "--checkpoint-update", "5000", "--output-dir", "screen"]
        for options in (("--gpus", "0,0"), ("--gpus", "-1"), ("--gpus", "0,"),
                        ("--gpus", "0,1", "--device", "cpu"), ("--workers-per-gpu", "0"),
                        ("--workers-per-gpu", "5")):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main([*base, *options])
        with patch.object(cli, "evaluate_campaign") as sequential:
            code, _, error = self.call([*base, "--workers-per-gpu", "2"])
        self.assertEqual(code, 1)
        self.assertIn("requires --gpus", error)
        sequential.assert_not_called()

    def test_frozen_selection_dispatches_both_cpu_and_two_gpu_modes(self):
        for options in (("--device", "cpu"), ("--gpus", "0,1", "--workers-per-gpu", "2")):
            with self.subTest(options=options), \
                 patch.object(cli, "evaluate_selected_campaign", return_value={"pilot_readiness": {"status": "needs_review"}}) as selected, \
                 patch.object(cli, "evaluate_campaign") as common:
                code, _, _ = self.call(["campaign", "--selection", "frozen/selection.json",
                    "--output-dir", "screen", "--windows", "64", "--seed", "0", *options])
            self.assertEqual(code, 0)
            common.assert_not_called()
            self.assertEqual(selected.call_args.args, (Path("frozen/selection.json"), Path("screen")))
            self.assertEqual(selected.call_args.kwargs["windows"], 64)
            self.assertEqual(selected.call_args.kwargs["gpus"], [0, 1] if "--gpus" in options else None)

    def test_frozen_selection_rejects_ambiguous_and_missing_checkpoint_modes(self):
        base = ["campaign", "--output-dir", "screen"]
        for options in ((), ("--selection", "selection.json", "--run-root", "fits"),
                        ("--selection", "selection.json", "--checkpoint-update", "5000"),
                        ("--checkpoint-update", "5000")):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main([*base, *options])


class ParallelCampaignTests(unittest.TestCase):
    """Exercise parent orchestration with synthetic reports and process boundaries."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run_root, self.output = self.root/"runs", self.root/"campaign"
        self.source = {"scripts/phase25.py": "fixture-source"}
        self.evidence = {"status": "passed", "report_sha256": "fixture-verification",
                         "report_path": str(self.output/"verification/implementation_checks.json")}
        self.paths = []
        for task, seed in compact.CASES:
            path = self.run_root/task/f"seed_{seed}"/"checkpoints/update_00005000.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(f"synthetic checkpoint path {task}/{seed}".encode())
            path.with_suffix(".sha256").write_text("fixture receipt; read_checkpoint is mocked\n")
            self.paths.append(path)
        self.started, self.processes = [], []
        self.change_report = None

    def checkpoint(self, path):
        return {"update": 5000, "metadata": {"task_family": path.parents[2].name,
                "run_seed": int(path.parents[1].name.removeprefix("seed_")), "profile": "pilot",
                "stage": "initial", "variant": "shared_initial:B,D"}}

    def evaluate(self):
        return cli.evaluate_parallel_campaign(self.run_root, 5000, self.output, gpus=[0, 1],
            workers_per_gpu=2, data_root=self.root/"replay", statistics_root=self.root/"stats",
            windows=64, seed=17, precision="bfloat16")

    def write_report(self, args):
        cp = self.checkpoint(args.checkpoint)
        report = {"schema_version": compact.VERSION, "P2.5_quality_acceptance": False,
                  "checkpoint": {"path": str(args.checkpoint), "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      "task_family": cp["metadata"]["task_family"], "run_seed": cp["metadata"]["run_seed"],
                      "update": cp["update"], "shared_by_variants": ["B", "D"],
                      "original_target_stage_updates": 5000,
                      "original_training_target_completed": cp["update"] >= 5000},
                  "pilot_readiness": {"status": "needs_review", "reasons": ["fixture review"], "unchecked_classes": ["death"]},
                  "source_sha256": self.source, "seed": args.seed, "verification": self.evidence,
                  "windows": [{}]*args.windows, "timing": {"total_seconds": .1},
                  "new_optimizer_updates": 0, "new_real_interactions": 0}
        if self.change_report is not None:
            self.change_report(report)
        checks.write_json(args.output_dir/"report.json", report)
        return report

    def fake_process(self, command, **kwargs):
        self.started.append((command, kwargs))
        args = cli.parser().parse_args(command[3:])
        self.write_report(args)
        process = Mock()
        def poll():
            self.assertEqual(len(self.started), 4, "every worker must start before waiting on the first")
            return 0
        process.poll.side_effect = poll
        process.wait.return_value = 0
        self.processes.append(process)
        return process

    def patches(self):
        from contextlib import ExitStack
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.verify = stack.enter_context(patch.object(compact, "prepare_verification", return_value=self.evidence))
        stack.enter_context(patch.object(compact, "evaluator_source_identity", return_value=self.source))
        self.read = stack.enter_context(patch("world_models.training.checkpoints.read_checkpoint", side_effect=self.checkpoint))
        stack.enter_context(patch.object(cli, "_gpu_environments", return_value=[
            {"CUDA_VISIBLE_DEVICES": "GPU-allocated-a", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
            {"CUDA_VISIBLE_DEVICES": "GPU-allocated-b", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}]))
        self.spawn = stack.enter_context(patch.object(cli.subprocess, "Popen", side_effect=self.fake_process))
        stack.enter_context(redirect_stderr(io.StringIO()))
        return stack

    def test_all_four_launch_round_robin_and_commit_same_campaign_contract(self):
        with self.patches():
            def checked_start(command, **kwargs):
                self.assertEqual(self.read.call_count, 4)
                return self.fake_process(command, **kwargs)
            self.spawn.side_effect = checked_start
            report = self.evaluate()
        self.verify.assert_called_once()
        self.assertEqual([kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs in self.started],
                         ["GPU-allocated-a", "GPU-allocated-b"]*2)
        for command, kwargs in self.started:
            args = cli.parser().parse_args(command[3:])
            self.assertEqual(args.device, "cuda:0")
            self.assertEqual(args.windows, 64)
            self.assertEqual(args.seed, 17)
            self.assertEqual(args.precision, "bfloat16")
            self.assertEqual(args.verification_report, Path(self.evidence["report_path"]))
            self.assertEqual(kwargs["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertEqual(kwargs["cwd"], cli.ROOT)
        self.assertEqual([(row["checkpoint"]["task_family"], row["checkpoint"]["run_seed"])
                          for row in report["cases"]], list(compact.CASES))
        self.assertEqual(report["checkpoint_update"], 5000)
        self.assertEqual(report["pilot_readiness"]["status"], "needs_review")
        self.assertFalse(report["P2.5_quality_acceptance"])
        self.assertTrue((self.output/"report.sha256").is_file())
        selection = json.loads((self.output/"pilot_model_selection.json").read_text())
        self.assertEqual(len(selection["models"]), 4)
        self.assertTrue(all(row["checkpoint"]["shared_by_variants"] == ["B", "D"] for row in selection["models"]))
        self.assertEqual(len(json.loads((self.output/"launch_plan.json").read_text())["jobs"]), 4)
        self.assertTrue(all(kwargs["stdout"].closed for _, kwargs in self.started))

    def test_gpu_mapping_preserves_scheduler_uuid_mask_and_rejects_outside_allocation(self):
        import torch
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-one,GPU-two", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=2):
            environments = cli._gpu_environments([1, 0])
            self.assertEqual([env["CUDA_VISIBLE_DEVICES"] for env in environments], ["GPU-two", "GPU-one"])
            self.assertTrue(all(env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" for env in environments))
            with self.assertRaisesRegex(ValueError, "unavailable"):
                cli._gpu_environments([0, 2])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-one,GPU-two")

    def test_missing_last_receipt_prevents_all_worker_launches(self):
        self.paths[-1].with_suffix(".sha256").unlink()
        with self.patches(), self.assertRaisesRegex(FileNotFoundError, "common-update"):
            self.evaluate()
        self.spawn.assert_not_called()
        self.read.assert_not_called()
        self.assertEqual(json.loads((self.output/"report.json").read_text())["pilot_readiness"]["status"], "failed")
        self.assertFalse((self.output/"pilot_model_selection.json").exists())

    def test_wrong_last_checkpoint_metadata_prevents_all_worker_launches(self):
        with self.patches():
            def read(path):
                result = self.checkpoint(path)
                if path == self.paths[-1]:
                    result["update"] = 4000
                return result
            self.read.side_effect = read
            with self.assertRaisesRegex(ValueError, "explicit task/seed/update"):
                self.evaluate()
        self.spawn.assert_not_called()
        self.assertFalse((self.output/"pilot_model_selection.json").exists())

    def test_failed_incomplete_or_changed_worker_evidence_never_creates_selection(self):
        for field, value in (("source_sha256", {}), ("windows", []), ("seed", 99),
                             ("pilot_readiness", {"status": "failed"})):
            with self.subTest(field=field):
                self.output = self.root/f"bad-{field}"
                self.evidence["report_path"] = str(self.output/"verification/implementation_checks.json")
                self.started.clear()
                self.change_report = lambda report: report.update({field: value})
                with self.patches(), self.assertRaisesRegex(ValueError, "incomplete or differs"):
                    self.evaluate()
                self.assertFalse((self.output/"pilot_model_selection.json").exists())
                self.assertEqual(json.loads((self.output/"report.json").read_text())["pilot_readiness"]["status"], "failed")

    def test_changed_worker_receipt_is_rejected(self):
        with self.patches():
            def tampered(command, **kwargs):
                process = self.fake_process(command, **kwargs)
                args = cli.parser().parse_args(command[3:])
                path = args.output_dir/"report.json"
                path.write_bytes(path.read_bytes()+b" ")
                return process
            self.spawn.side_effect = tampered
            with self.assertRaisesRegex(ValueError, "SHA receipt mismatch"):
                self.evaluate()
        self.assertFalse((self.output/"pilot_model_selection.json").exists())

    def test_missing_worker_receipt_cannot_commit_campaign(self):
        with self.patches():
            def incomplete(command, **kwargs):
                process = self.fake_process(command, **kwargs)
                args = cli.parser().parse_args(command[3:])
                (args.output_dir/"report.sha256").unlink()
                return process
            self.spawn.side_effect = incomplete
            with self.assertRaisesRegex(ValueError, "did not commit"):
                self.evaluate()
        self.assertFalse((self.output/"pilot_model_selection.json").exists())

    def test_worker_failure_stops_every_running_worker_and_records_failed_campaign(self):
        with self.patches():
            def failed(command, **kwargs):
                process = self.fake_process(command, **kwargs)
                process.poll.side_effect = None
                process.poll.return_value = 17 if len(self.started) == 2 else None
                return process
            self.spawn.side_effect = failed
            with self.assertRaisesRegex(RuntimeError, "exited 17"):
                self.evaluate()
        self.assertEqual(len(self.processes), 4)
        for index, process in enumerate(self.processes):
            if index != 1:
                process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=5)
        self.assertFalse((self.output/"pilot_model_selection.json").exists())
        report = json.loads((self.output/"report.json").read_text())
        self.assertEqual(report["pilot_readiness"]["status"], "failed")
        self.assertEqual(len(report["worker_jobs"]), 4)

    def test_launch_failure_stops_previously_started_worker_including_kill_fallback(self):
        with self.patches():
            process = Mock()
            process.poll.return_value = None
            process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 5), 0]
            self.spawn.side_effect = [process, OSError("fixture launch failure")]
            with self.assertRaisesRegex(OSError, "fixture launch failure"):
                self.evaluate()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
        self.assertFalse((self.output/"pilot_model_selection.json").exists())

    def test_termination_uses_cleanup_exception_and_restores_original_handler(self):
        original = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(KeyboardInterrupt):
            with cli._interrupt_on_termination():
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        self.assertEqual(signal.getsignal(signal.SIGTERM), original)

    def test_keyboard_interrupt_stops_launched_workers_and_preserves_failure_evidence(self):
        import torch
        threads = torch.get_num_threads()
        with self.patches():
            def interrupted(command, **kwargs):
                process = self.fake_process(command, **kwargs)
                process.poll.side_effect = [KeyboardInterrupt("fixture interrupt"), None] if len(self.started) == 1 else None
                process.poll.return_value = None
                return process
            self.spawn.side_effect = interrupted
            with self.assertRaises(KeyboardInterrupt):
                self.evaluate()
        for process in self.processes:
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=5)
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertFalse((self.output/"pilot_model_selection.json").exists())
        self.assertEqual(json.loads((self.output/"report.json").read_text())["pilot_readiness"]["status"], "failed")


class FrozenCampaignTests(unittest.TestCase):
    """Frozen unequal budgets must remain fixed through sequential/parallel screens."""
    write_report = ParallelCampaignTests.write_report
    fake_process = ParallelCampaignTests.fake_process
    patches = ParallelCampaignTests.patches

    def setUp(self):
        ParallelCampaignTests.setUp(self)
        self.updates = [4250, 4250, 4750, 4750]
        for index, (path, update) in enumerate(zip(self.paths, self.updates)):
            target = path.with_name(f"update_{update:08d}.pt")
            path.rename(target)
            path.with_suffix(".sha256").rename(target.with_suffix(".sha256"))
            self.paths[index] = target
        self.selection = self.root/"frozen/selection.json"
        self.manifest = {"schema_version": "p24-initial-selection-v1", "selection_rule": "latest_committed_per_case",
            "reason": "user time budget", "user_authorized_shortened_pilot": True,
            "full_schedule_completion_claim": False,
            "cases": [{"task": task, "seed": seed, "model_updates": update,
                       "checkpoint_path": Path(os.path.relpath(path, self.selection.parent)).as_posix(),
                       "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "original_target_stage_updates": 5000}
                      for (task, seed), path, update in zip(compact.CASES, self.paths, self.updates)]}
        checks.write_json(self.selection, self.manifest)

    def checkpoint(self, path):
        result = ParallelCampaignTests.checkpoint(self, path)
        result["update"] = int(path.stem.removeprefix("update_"))
        return result

    def evaluate(self, *, parallel=True):
        return cli.evaluate_selected_campaign(self.selection, self.output,
            gpus=[0, 1] if parallel else None, workers_per_gpu=2 if parallel else 1, device="cpu",
            data_root=self.root/"replay", statistics_root=self.root/"stats", windows=64, seed=17)

    def selected(self):
        return patch("world_models.training.selection.load_selection", side_effect=lambda _: deepcopy(self.manifest))

    def test_parallel_uses_true_unequal_updates_and_binds_both_evidence_products(self):
        with self.patches(), self.selected() as loaded:
            report = self.evaluate()
        self.assertEqual(loaded.call_count, 2)
        self.assertEqual([row["checkpoint"]["update"] for row in report["cases"]], self.updates)
        self.assertIsNone(report["checkpoint_update"])
        self.assertEqual(report["checkpoint_updates"], {f"{task}/seed_{seed}": update
            for (task, seed), update in zip(compact.CASES, self.updates)})
        self.assertTrue(report["protocol_deviation"]["unequal_updates_allowed"])
        self.assertFalse(report["protocol_deviation"]["full_schedule_completion_claim"])
        self.assertFalse(report["P2.5_quality_acceptance"])
        evidence = report["initial_checkpoint_selection"]
        self.assertEqual(evidence["sha256"], hashlib.sha256(self.selection.read_bytes()).hexdigest())
        self.assertEqual(evidence["manifest"], self.manifest)
        models = json.loads((self.output/"pilot_model_selection.json").read_text())
        self.assertEqual(models["initial_checkpoint_selection"], evidence)
        self.assertEqual([job["logical_gpu"] for job in report["launch"]["jobs"]], [0, 1, 0, 1])
        self.assertTrue(all(row["checkpoint"]["shared_by_variants"] == ["B", "D"] for row in report["cases"]))
        self.assertTrue(all(row["checkpoint"]["original_training_target_completed"] is False for row in report["cases"]))

    def test_sequential_cpu_uses_same_frozen_paths_metrics_and_one_verification(self):
        observed = []
        def evaluate(path, directory, **kwargs):
            observed.append((path, kwargs))
            return self.write_report(SimpleNamespace(checkpoint=path, output_dir=directory,
                                                    windows=kwargs["windows"], seed=kwargs["seed"]))
        with self.patches(), self.selected(), patch.object(compact, "_evaluate_one", side_effect=evaluate):
            report = self.evaluate(parallel=False)
        self.verify.assert_called_once()
        self.spawn.assert_not_called()
        self.assertEqual([path for path, _ in observed], self.paths)
        self.assertTrue(all(args["device"] == "cpu" and args["windows"] == 64 and args["seed"] == 17
                            and args["verification"] == self.evidence for _, args in observed))
        self.assertEqual(report["launch"]["mode"], "sequential")
        self.assertEqual([row["checkpoint"]["update"] for row in report["cases"]], self.updates)

    def test_bad_selection_receipt_is_rejected_before_any_evaluation(self):
        self.selection.write_bytes(self.selection.read_bytes()+b" ")
        with self.patches(), patch.object(compact, "_evaluate_one") as evaluate, \
             self.assertRaisesRegex(ValueError, "selection SHA256 receipt mismatch"):
            self.evaluate(parallel=False)
        self.spawn.assert_not_called()
        evaluate.assert_not_called()
        self.assertFalse((self.output/"pilot_model_selection.json").exists())

    def test_selected_checkpoint_replacement_cannot_launch_workers(self):
        self.paths[-1].write_bytes(b"different checkpoint")
        with self.patches(), self.selected(), self.assertRaisesRegex(ValueError, "selected checkpoint changed"):
            self.evaluate()
        self.spawn.assert_not_called()

    def test_changed_selection_during_workers_never_commits_campaign(self):
        with self.patches(), self.selected():
            def replaced(command, **kwargs):
                process = self.fake_process(command, **kwargs)
                self.selection.write_bytes(self.selection.read_bytes()+b" ")
                return process
            self.spawn.side_effect = replaced
            with self.assertRaisesRegex(ValueError, "selection changed during evaluation"):
                self.evaluate()
        self.assertFalse((self.output/"pilot_model_selection.json").exists())
        self.assertEqual(json.loads((self.output/"report.json").read_text())["pilot_readiness"]["status"], "failed")

    def test_worker_cannot_claim_original_training_target_was_completed(self):
        self.change_report = lambda report: report["checkpoint"].update(original_training_target_completed=True)
        with self.patches(), self.selected(), self.assertRaisesRegex(ValueError, "original training target"):
            self.evaluate()
        self.assertFalse((self.output/"pilot_model_selection.json").exists())

    def test_newer_checkpoint_does_not_retarget_frozen_selection(self):
        later = self.paths[0].with_name("update_00004500.pt")
        later.write_bytes(b"newer but not selected")
        later.with_suffix(".sha256").write_text("not the frozen receipt")
        with self.patches(), self.selected():
            report = self.evaluate()
        self.assertEqual(report["cases"][0]["checkpoint"]["update"], 4250)
        self.assertEqual(Path(report["cases"][0]["checkpoint"]["path"]), self.paths[0])


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.identity = {"scripts/phase25.py": "source", "src/world_models/models/core.py": "model"}
        self.p24 = {"src/world_models/models/core.py": "model"}

    def report(self):
        return {"schema_version": checks.SCHEMA_VERSION, "status": "passed", "scope": "required_structural_checks",
                "source_sha256": deepcopy(self.identity), "p24_source_sha256": deepcopy(self.p24),
                "source_unchanged": True, "structural_tests": list(checks.STRUCTURAL_TESTS),
                "structural_checks_passed": True, "passed_test_ids": list(checks.STRUCTURAL_TESTS),
                "tests_run": len(checks.STRUCTURAL_TESTS), "failures": 0, "errors": 0,
                "research_training_executed": False, "research_checkpoint_evaluated": False,
                "new_environment_interactions": 0, "P2.5_quality_acceptance": False}

    def save(self, value=None, name="implementation_checks.json"):
        path = self.root / name
        checks.write_json(path, self.report() if value is None else value)
        return path

    def reuse(self, path, output="copy"):
        with patch.object(checks, "source_identity", return_value=self.identity), \
             patch.object(checks, "p24_source_identity", return_value=self.p24):
            return checks.prepare_verification(self.root / output, reuse_path=path)

    def test_fresh_json_receipts_are_hash_bound_and_immutable(self):
        path = self.save()
        self.assertEqual(path.with_suffix(".sha256").read_text().strip(),
                         f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        with self.assertRaises(FileExistsError):
            checks.write_json(path, {})

    def test_valid_reuse_copies_evidence_and_never_runs_tests(self):
        path = self.save()
        with patch.object(checks, "_run") as run:
            result = self.reuse(path)
        run.assert_not_called()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["reused"])
        self.assertEqual(Path(result["report_path"]).read_bytes(), path.read_bytes())
        self.assertEqual(result["report_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            self.reuse(path)

    def test_receipt_tampering_and_missing_receipt_fail_before_output_creation(self):
        path = self.save()
        original = path.read_bytes()
        path.write_bytes(original+b"changed")
        with self.assertRaisesRegex(ValueError, "receipt mismatch"):
            self.reuse(path)
        self.assertFalse((self.root / "copy").exists())
        path.with_suffix(".sha256").unlink()
        with self.assertRaisesRegex(ValueError, "both required"):
            self.reuse(path)

    def test_p24_wrong_schema_failed_status_missing_test_and_changed_sources_are_rejected(self):
        mutations = [lambda value: value.update(schema_version="p24-implementation-v1"),
                     lambda value: value.update(status="failed"),
                     lambda value: value.update(structural_checks_passed=False),
                     lambda value: value.update(passed_test_ids=list(checks.STRUCTURAL_TESTS[:-1])),
                     lambda value: value.update(source_sha256={"changed": "source"}),
                     lambda value: value.update(p24_source_sha256={"changed": "source"}),
                     lambda value: value.update(research_training_executed=True),
                     lambda value: value.update({"P2.5_quality_acceptance": True})]
        for index, mutate in enumerate(mutations):
            value = self.report()
            mutate(value)
            path = self.save(value, name=f"invalid_{index}.json")
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.reuse(path, output=f"copy_{index}")

    def test_lightweight_prepare_runs_only_required_structural_suite(self):
        with patch.object(checks, "_run", return_value={"status": "passed"}) as run:
            result = checks.prepare_verification(self.root / "fresh")
        self.assertEqual(result["status"], "passed")
        run.assert_called_once_with(self.root / "fresh", full=False)
        with patch.object(checks, "_run", return_value={"status": "passed"}) as run:
            checks.verify_implementation(self.root / "full")
        run.assert_called_once_with(self.root / "full", full=True)

    def test_missing_tests_initializer_or_shared_fixture_has_clear_preflight(self):
        package = self.root / "tests"
        package.mkdir()
        with patch.object(checks, "ROOT", self.root), self.assertRaisesRegex(FileNotFoundError, "tests/__init__.py"):
            checks._preflight(checks.STRUCTURAL_TESTS)
        for name in ("__init__", "test_world_model", "test_model_heads", "test_model_sampling", "test_targets"):
            (package / f"{name}.py").write_text("")
        with patch.object(checks, "ROOT", self.root), self.assertRaisesRegex(FileNotFoundError, "test_replay_storage"):
            checks._preflight(checks.STRUCTURAL_TESTS)
        (package / "test_replay_storage.py").write_text("")
        with patch.object(checks, "ROOT", self.root), \
             patch.object(checks.importlib.util, "find_spec", return_value=SimpleNamespace(origin="/foreign/tests/__init__.py")), \
             self.assertRaisesRegex(ImportError, "resolved tests"):
            checks._preflight(checks.STRUCTURAL_TESTS)

    def test_failed_structural_test_writes_failed_evidence_and_cannot_be_reused(self):
        class Failing(unittest.TestCase):
            def runTest(self):
                self.fail("fixture structural failure")
        with patch.object(checks, "_preflight"), \
             patch.object(checks, "source_identity", return_value=self.identity), \
             patch.object(checks, "p24_source_identity", return_value=self.p24), \
             patch.object(checks.unittest.defaultTestLoader, "loadTestsFromName",
                          side_effect=lambda _: unittest.TestSuite([Failing()])):
            with self.assertRaisesRegex(RuntimeError, "implementation checks failed"):
                checks.prepare_verification(self.root / "failed")
        path = self.root / "failed/implementation_checks.json"
        self.assertEqual(json.loads(path.read_text())["status"], "failed")
        self.assertIn("fixture structural failure", (self.root / "failed/tests.txt").read_text())
        with self.assertRaises(ValueError):
            self.reuse(path)


if __name__ == "__main__":
    unittest.main()
