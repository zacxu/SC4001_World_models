"""P2.4 process orchestration and crash recovery without research-model fits."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from scripts import phase24 as cli
from world_models.infrastructure.config import resolve_config
from world_models.models.heads import TaskSpec
from world_models.training.checkpoints import read_checkpoint
from world_models.training.fitting import FitSettings
from world_models.training import runner


def case_fixture(task="relay", seed=0):
    config = resolve_config(task, "B", "pilot")
    provenance = {"initial_normalization": {"identity_sha256": "normalizer", "file_sha256": "normalizer-file"},
                  "replays": [{"manifest_sha256": f"{task}-{seed}-manifest",
                               "dataset_identity_sha256": f"{task}-{seed}-identity", "transitions": 50000}]}
    return config, [SimpleNamespace(manifest={"metadata": {}})], object(), provenance


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "campaign"

    def args(self, command="plan", *extra):
        return cli.parser().parse_args([command, "--output-dir", str(self.output), *extra])

    def load(self, task, seed, **kwargs):
        return case_fixture(task, seed)

    def test_default_plan_maps_four_independent_fits_to_one_visible_gpu(self):
        with patch.object(cli, "load_case", side_effect=self.load) as load, \
             patch.object(cli, "fit_case") as fit, patch.object(cli.subprocess, "Popen") as popen:
            plan = cli.launch_plan(self.args())
        self.assertEqual(plan["worker_count"], 4)
        self.assertEqual(plan["workers_per_gpu"], 4)
        self.assertEqual(plan["gpus"], [0])
        self.assertEqual([(job["task"], job["seed"]) for job in plan["jobs"]], list(runner.CASES))
        self.assertEqual([job["logical_gpu"] for job in plan["jobs"]], [0]*4)
        self.assertFalse(plan["training_executed"])
        self.assertEqual(plan["optimizer_updates_executed"], 0)
        self.assertEqual(load.call_count, 4)
        fit.assert_not_called()
        popen.assert_not_called()
        self.assertFalse(self.output.exists())
        for job in plan["jobs"]:
            self.assertEqual(job["command"][job["command"].index("--device")+1], "cuda:0")
            self.assertEqual(job["data"]["replays"][0]["manifest_sha256"], f"{job['task']}-{job['seed']}-manifest")

    def test_plan_propagates_resume_settings_and_round_robin_logical_slots(self):
        # Parent recovery resumes started cases and starts never-launched cases.
        for task, seed in runner.CASES[:2]:
            case = self.output / task / f"seed_{seed}"
            case.mkdir(parents=True)
            (case / "run_manifest.json").write_text("{}")
        args = self.args("plan", "--gpus", "0,1", "--workers-per-gpu", "2", "--resume",
                         "--updates", "7", "--microbatch", "2", "--precision", "float32")
        with patch.object(cli, "load_case", side_effect=self.load):
            plan = cli.launch_plan(args)
        self.assertEqual([job["logical_gpu"] for job in plan["jobs"]], [0, 1, 0, 1])
        for index, job in enumerate(plan["jobs"]):
            command = job["command"]
            self.assertEqual("--resume" in command, index < 2)
            self.assertEqual(command[command.index("--updates")+1], "7")
            self.assertEqual(command[command.index("--microbatch")+1], "2")
            self.assertEqual(command[command.index("--precision")+1], "float32")

    def test_invalid_gpu_indices_slot_capacity_and_final_launcher_rejected(self):
        for value in ("", "-1", "0,0", "0,", "a", "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cli.gpu_indices(value)
        for arguments in (("--workers-per-gpu", "0"), ("--workers-per-gpu", "3"),
                          ("--profile", "final")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError), \
                 patch.object(cli, "load_case") as load:
                cli.launch_plan(self.args("plan", *arguments))
            load.assert_not_called()

    def test_nonpositive_plan_updates_fail_before_replay_reads(self):
        for value in (0, -1, True):
            args = self.args()
            args.updates = value
            with self.subTest(value=value), patch.object(cli, "load_case") as load, \
                 self.assertRaisesRegex(ValueError, "updates must be positive"):
                cli.launch_plan(args)
            load.assert_not_called()

    def test_overfit_launcher_uses_explicit_small_stage_budget(self):
        with patch.object(cli, "load_case", side_effect=self.load):
            plan = cli.launch_plan(self.args("plan", "--stage", "overfit"))
        self.assertEqual(plan["stage"], "overfit")
        for job in plan["jobs"]:
            self.assertEqual(job["target_stage_updates"], 50)
            self.assertEqual(job["command"][3], "overfit")
            self.assertEqual(job["command"][job["command"].index("--updates")+1], "50")

    def test_inherited_cuda_mapping_supports_scheduler_indices_uuid_and_mig(self):
        inherited = {"CUDA_VISIBLE_DEVICES": "7,GPU-abc,MIG-GPU-def/1/0", "OTHER": "retained"}
        for index, expected in enumerate(("7", "GPU-abc", "MIG-GPU-def/1/0")):
            actual = cli.gpu_environment(index, inherited)
            self.assertEqual(actual["CUDA_VISIBLE_DEVICES"], expected)
            self.assertEqual(actual["OTHER"], "retained")
            self.assertEqual(actual["OMP_NUM_THREADS"], "1")
            self.assertEqual(actual["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(inherited["CUDA_VISIBLE_DEVICES"], "7,GPU-abc,MIG-GPU-def/1/0")
        self.assertEqual(cli.gpu_environment(2, {})["CUDA_VISIBLE_DEVICES"], "2")
        with self.assertRaises(ValueError):
            cli.gpu_environment(3, inherited)
        with self.assertRaises(ValueError):
            cli.gpu_environment(0, {"CUDA_VISIBLE_DEVICES": ""})

    def test_launch_starts_all_four_before_polling_or_waiting(self):
        events, environments = [], []
        def spawn(command, **kwargs):
            events.append("spawn")
            environments.append(kwargs["env"])
            process = Mock(pid=100+len(events), returncode=0)
            def poll():
                self.assertEqual(events.count("spawn"), 4)
                events.append("poll")
                return 0
            process.poll.side_effect = poll
            process.wait.side_effect = lambda: events.append("wait")
            return process
        with patch.object(cli, "load_case", side_effect=self.load), \
             patch.object(cli.torch.cuda, "is_available", return_value=True), \
             patch.object(cli.torch.cuda, "device_count", return_value=1), \
             patch.object(cli.subprocess, "Popen", side_effect=spawn) as popen, \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-assigned"}), \
             patch.object(cli.time, "sleep") as sleep:
            result = cli.launch(self.args("launch"))
        self.assertEqual(events[:4], ["spawn"]*4)
        self.assertEqual(popen.call_count, 4)
        self.assertEqual([env["CUDA_VISIBLE_DEVICES"] for env in environments], ["GPU-assigned"]*4)
        self.assertEqual(result["status"], "complete_fit_stages")
        self.assertFalse(result["P2.5_quality_acceptance"])
        sleep.assert_not_called()

    def test_launch_failure_terminates_and_waits_for_started_children(self):
        started = []
        def spawn(*args, **kwargs):
            if len(started) == 2:
                raise OSError("fixture spawn failure")
            process = Mock(pid=100+len(started), returncode=None)
            process.poll.return_value = None
            started.append(process)
            return process
        with patch.object(cli, "load_case", side_effect=self.load), \
             patch.object(cli.torch.cuda, "is_available", return_value=True), \
             patch.object(cli.torch.cuda, "device_count", return_value=1), \
             patch.object(cli.subprocess, "Popen", side_effect=spawn), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(OSError, "spawn failure"):
                cli.launch(self.args("launch"))
        self.assertEqual(len(started), 2)
        for process in started:
            process.terminate.assert_called_once()
            process.wait.assert_called_once()

    def test_launch_resume_preserves_original_plan_and_rejects_changed_provenance(self):
        def process(*args, **kwargs):
            child = Mock(pid=123, returncode=0)
            child.poll.return_value = 0
            return child
        with patch.object(cli, "load_case", side_effect=self.load), \
             patch.object(cli.torch.cuda, "is_available", return_value=True), \
             patch.object(cli.torch.cuda, "device_count", return_value=1), \
             patch.object(cli.subprocess, "Popen", side_effect=process) as popen, \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            cli.launch(self.args("launch"))
            original = (self.output / "launch_plan.json").read_bytes()
            for task, seed in runner.CASES[:2]:
                case = self.output / task / f"seed_{seed}"
                case.mkdir(parents=True)
                (case / "run_manifest.json").write_text("{}")
            cli.launch(self.args("launch", "--resume"))
            self.assertEqual(popen.call_count, 8)
            self.assertEqual((self.output / "launch_plan.json").read_bytes(), original)
            receipts = list(self.output.glob("launch_plan_resume_*.json"))
            self.assertEqual(len(receipts), 1)
            resumed = json.loads(receipts[0].read_text())
            self.assertEqual(["--resume" in job["command"] for job in resumed["jobs"]],
                             [True, True, False, False])
            popen.reset_mock()
            with self.assertRaisesRegex(ValueError, "launch plan changed"):
                cli.launch(self.args("launch", "--resume", "--updates", "9"))
            popen.assert_not_called()
            def changed(task, seed, **kwargs):
                cfg, datasets, stats, provenance = self.load(task, seed, **kwargs)
                provenance["replays"][0]["manifest_sha256"] = "changed-replay"
                return cfg, datasets, stats, provenance
            with patch.object(cli, "load_case", side_effect=changed), \
                 self.assertRaisesRegex(ValueError, "launch plan changed"):
                cli.launch(self.args("launch", "--resume"))
            popen.assert_not_called()
            self.assertEqual((self.output / "launch_plan.json").read_bytes(), original)
            self.assertEqual(list(self.output.glob("launch_plan_resume_*.json")), receipts)

    def test_plan_verify_status_commands_never_dispatch_research_training(self):
        with patch.object(cli, "fit_case") as fit, patch.object(cli, "evaluate_checkpoint") as evaluate, \
             patch.object(cli.subprocess, "Popen") as popen, \
             patch.object(cli, "load_case", side_effect=self.load), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["plan", "--output-dir", str(self.output)]), 0)
            self.assertEqual(cli.main(["status", "--output-dir", str(self.output)]), 0)
            # Do not recursively execute this suite through the verify command.
            with patch.object(cli.unittest.defaultTestLoader, "loadTestsFromName", side_effect=lambda _: unittest.TestSuite()):
                self.assertEqual(cli.main(["verify", "--output-dir", str(self.output / "verify")]), 0)
        fit.assert_not_called()
        evaluate.assert_not_called()
        popen.assert_not_called()
        report = json.loads((self.output / "verify/phase24_acceptance.json").read_text())
        self.assertFalse(report["research_training_executed"])
        self.assertFalse(report["gpu_training_measured"])


class VerificationSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "evidence"
        self.test_names = ("test_model_sampling", "test_model_checkpoints", "test_model_fitting",
                           "test_model_validation_bank", "test_phase24_cli", "test_replay_storage",
                           "test_targets", "test_model_objective_accumulation")
        (self.root / "tests").mkdir()
        for name in ("__init__", *self.test_names):
            (self.root / "tests" / f"{name}.py").write_text("")

    def test_missing_transferred_test_or_shared_fixture_fails_before_loading(self):
        for name in ("test_model_sampling", "test_replay_storage"):
            with self.subTest(name=name):
                path = self.root / "tests" / f"{name}.py"
                path.unlink()
                try:
                    with patch.object(cli, "ROOT", self.root), \
                         patch.object(cli.unittest.defaultTestLoader, "loadTestsFromName") as load, \
                         self.assertRaisesRegex(FileNotFoundError, name):
                        cli.verify(self.output)
                    load.assert_not_called()
                    self.assertFalse(self.output.exists())
                finally:
                    path.write_text("")

    def test_missing_package_initializer_reports_complete_tests_transfer(self):
        (self.root / "tests/__init__.py").unlink()
        with patch.object(cli, "ROOT", self.root), \
             self.assertRaisesRegex(FileNotFoundError, "including tests/__init__.py"):
            cli.verify(self.output)
        self.assertFalse(self.output.exists())

    def test_foreign_tests_package_reports_origin_instead_of_import_placeholders(self):
        origin = self.root / "site-packages/tests/__init__.py"
        with patch.object(cli, "ROOT", self.root), \
             patch.object(cli.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(origin))), \
             patch.object(cli.unittest.defaultTestLoader, "loadTestsFromName") as load, \
             self.assertRaisesRegex(ImportError, "resolved 'tests'.*site-packages"):
            cli.verify(self.output)
        load.assert_not_called()
        self.assertFalse(self.output.exists())


class ToyModel(nn.Linear):
    def __init__(self):
        super().__init__(2, 1)
        self.task = TaskSpec()


class ToySampler:
    def __init__(self, datasets, *, seed, **kwargs):
        self.generator = torch.Generator().manual_seed(seed)
        self.samples = 0

    def sample(self, size):
        self.samples += size
        return torch.randn(size, 2, generator=self.generator)

    def state_dict(self):
        return {"rng": self.generator.get_state(), "samples": self.samples}

    def load_state_dict(self, value):
        self.generator.set_state(value["rng"])
        self.samples = value["samples"]


def toy_update(model, optimizer, windows, streams, objective, settings):
    noise = random.random() + float(np.random.random()) + torch.rand(())
    for name in ("prior", "posterior", "outcome"):
        noise += torch.rand((), generator=getattr(streams, name))
    loss = (model(windows)-noise).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"total": float(loss.detach())}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.settings = FitSettings(effective_batch=2, microbatch=1, burn_in=0, loss_steps=1,
                                    precision="float32", checkpoint_every=1, validate_every=1,
                                    validation_windows=1, log_every=1)
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                 torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())
        self.addCleanup(random.setstate, state[0])
        self.addCleanup(np.random.set_state, state[1])
        self.addCleanup(torch.set_rng_state, state[2])
        self.addCleanup(torch.use_deterministic_algorithms, state[3])
        self.addCleanup(torch.set_num_threads, state[4])

    def patches(self, update=toy_update, provenance=None):
        stack = ExitStack()
        def load(task, seed, **kwargs):
            cfg, datasets, stats, base = case_fixture(task, seed)
            if kwargs.get("additional_replays"):
                base["replays"].append({"manifest_sha256": "online-manifest", "dataset_identity_sha256": "online-identity",
                                        "transitions": 25000})
                datasets.append(SimpleNamespace(manifest={"metadata": {"training_variant": "B"}}))
            return cfg, datasets, stats, deepcopy(base if provenance is None else provenance)
        stack.enter_context(patch.object(runner, "load_case", side_effect=load))
        stack.enter_context(patch.object(runner, "build_model", side_effect=lambda *args, **kwargs: ToyModel()))
        stack.enter_context(patch.object(runner, "ReplaySampler", ToySampler))
        stack.enter_context(patch.object(runner, "WorldModelObjective", return_value=object()))
        stack.enter_context(patch.object(runner, "source_identity", return_value={"fixture.py": "fixed-source"}))
        stack.enter_context(patch.object(runner, "runtime_inventory", return_value={"device": "cpu", "fixture": True}))
        stack.enter_context(patch.object(runner, "fixed_diagnostic_windows", return_value=["held-out-fixture"]))
        stack.enter_context(patch.object(runner, "evaluate_windows", return_value={"total": 1.}))
        stack.enter_context(patch.object(runner, "optimizer_update", side_effect=update))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def fit(self, output, **kwargs):
        return runner.fit_case("relay", 0, output, data_root=self.root / "data", statistics_root=self.root / "stats",
                               settings=self.settings, device="cpu", updates=2, **kwargs)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_run_lock_excludes_second_writer_and_releases_after_exception(self):
        output = self.root / "locked"
        with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
            with runner.run_lock(output):
                with self.assertRaisesRegex(RuntimeError, "another worker"):
                    with runner.run_lock(output):
                        self.fail("second writer acquired run lock")
                raise RuntimeError("fixture interruption")
        with runner.run_lock(output):
            self.assertTrue((output / ".worker.lock").is_file())

    def test_quarantine_preserves_debris_and_never_moves_corrupt_committed_pair(self):
        output, directory = self.root / "run", self.root / "run/checkpoints"
        directory.mkdir(parents=True)
        files = {"update_00000001.pt": b"corrupted committed payload",
                 "update_00000001.sha256": b"bad hash but committed pair",
                 "update_00000001.pt.lock": b"stale lock",
                 "update_00000002.pt": b"orphan payload", "update_00000003.sha256": b"orphan receipt",
                 "update_00000004.pt.tmp-interrupted": b"temporary", "unrelated.txt": b"keep"}
        for name, content in files.items():
            (directory / name).write_bytes(content)
        with runner.run_lock(output):
            moved = runner.quarantine_uncommitted_checkpoints(output)
        self.assertEqual({row["original"] for row in moved}, set(files)-{
            "update_00000001.pt", "update_00000001.sha256", "unrelated.txt"})
        for row in moved:
            self.assertEqual((directory / row["preserved"]).read_bytes(), files[row["original"]])
        self.assertEqual((directory / "update_00000001.pt").read_bytes(), files["update_00000001.pt"])
        self.assertEqual(runner.checkpoint_candidates(output), [directory / "update_00000001.pt"])
        with self.assertRaises(ValueError):
            read_checkpoint(directory / "update_00000001.pt")
        self.assertEqual(runner.quarantine_uncommitted_checkpoints(output), [])

    def test_checkpoint_candidates_require_pair_and_sort_absolute_update_descending(self):
        directory = self.root / "checkpoints"
        directory.mkdir()
        for index in (0, 9, 10):
            (directory / f"update_{index:08d}.pt").touch()
            (directory / f"update_{index:08d}.sha256").touch()
        (directory / "update_99999999.pt").touch()
        self.assertEqual([path.stem for path in runner.checkpoint_candidates(self.root)],
                         ["update_00000010", "update_00000009", "update_00000000"])

    def test_source_identity_covers_execution_modules_and_changes_with_bytes(self):
        expected = ["src/world_models/models/core.py", "src/world_models/training/runner.py",
                    "src/world_models/training/checkpoints.py", "scripts/phase24.py",
                    "src/world_models/data/replay.py", "src/world_models/data/schema.py",
                    "src/world_models/infrastructure/config.py", "src/world_models/infrastructure/reproducibility.py"]
        expected += [f"src/world_models/training/{name}.py" for name in
                     ("fitting", "sampling", "validation", "objective", "objective_batch", "losses",
                      "normalization", "diagnostics")]
        for relative in expected:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
        before = runner.source_identity(self.root)
        self.assertEqual(set(before), set(expected))
        path = self.root / expected[0]
        path.write_bytes(b"changed")
        after = runner.source_identity(self.root)
        self.assertNotEqual(before[expected[0]], after[expected[0]])
        self.assertEqual(after[expected[0]], hashlib.sha256(b"changed").hexdigest())

    def test_completed_stage_binds_same_checkpoint_and_completed_resume_is_idempotent(self):
        output = self.root / "complete"
        with self.patches():
            result = self.fit(output)
        checkpoint = Path(result["checkpoint"])
        before = checkpoint.read_bytes()
        self.assertEqual(result["model_updates"], 2)
        self.assertEqual(result["stage_updates"], 2)
        self.assertEqual(result["new_environment_interactions"], 0)
        self.assertEqual(len(list((output / "checkpoints").glob("*.pt"))), 3)
        left = json.loads((output / "bindings/B.json").read_text())
        right = json.loads((output / "bindings/D.json").read_text())
        self.assertEqual(left["checkpoint"], right["checkpoint"])
        self.assertEqual(left["checkpoint_sha256"], right["checkpoint_sha256"])
        with self.patches(update=Mock(side_effect=AssertionError("completed schedule repeated"))):
            resumed = self.fit(output, resume=True)
        self.assertEqual(resumed["model_updates"], 2)
        self.assertEqual(checkpoint.read_bytes(), before)

    def test_resume_after_failure_matches_uninterrupted_model_optimizer_and_every_rng(self):
        baseline, failed = self.root / "baseline", self.root / "failed"
        with self.patches():
            expected = self.fit(baseline)
        calls = 0
        def interrupted(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("fixture failure after committed update")
            return toy_update(*args)
        with self.patches(update=interrupted), self.assertRaisesRegex(RuntimeError, "fixture failure"):
            self.fit(failed)
        status = json.loads((failed / "status.json").read_text())
        self.assertEqual(status["last_committed_update"], 1)
        self.assertEqual(status["status"], "failed")
        # Receipt-less crash debris at the next logical update is preserved,
        # allowing that update number to be regenerated from committed state.
        orphan = failed / "checkpoints/update_00000002.pt"
        orphan.write_bytes(b"uncommitted checkpoint publication")
        with self.patches():
            actual = self.fit(failed, resume=True)
        self.assertEqual(actual["model_updates"], 2)
        self.assertEqual(actual["stage_updates"], 2)
        expected_payload, actual_payload = read_checkpoint(expected["checkpoint"]), read_checkpoint(actual["checkpoint"])
        for name in ("model", "optimizer", "streams", "sampler", "rng"):
            self.assert_tree_equal(expected_payload[name], actual_payload[name])
        preserved = list((failed / "checkpoints/uncommitted").glob("*"))
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_bytes(), b"uncommitted checkpoint publication")

    def test_resume_rejects_changed_provenance_and_missing_run_manifest(self):
        output = self.root / "complete"
        with self.patches():
            self.fit(output)
        _, _, _, changed = case_fixture()
        changed["initial_normalization"]["identity_sha256"] = "different-normalizer"
        with self.patches(provenance=changed), self.assertRaisesRegex(ValueError, "provenance changed"):
            self.fit(output, resume=True)
        with self.patches(), self.assertRaisesRegex(ValueError, "existing run manifest"):
            self.fit(self.root / "never-started", resume=True)

    def test_resume_rejects_changed_runtime_contract_before_optimizer_update(self):
        output = self.root / "runtime-contract"
        with self.patches():
            result = self.fit(output)
        payload = read_checkpoint(result["checkpoint"])
        self.assertEqual(payload["metadata"]["runtime_contract"], runner.runtime_contract(torch.device("cpu")))
        changed = deepcopy(payload["metadata"]["runtime_contract"])
        changed["torch"] = "different-runtime"
        update = Mock(side_effect=AssertionError("runtime mismatch reached optimizer"))
        with self.patches(update=update), patch.object(runner, "runtime_contract", return_value=changed), \
             self.assertRaisesRegex(ValueError, "provenance changed"):
            self.fit(output, resume=True)
        update.assert_not_called()

    def test_refresh_uses_absolute_parent_count_and_rejects_missing_or_partial_parent(self):
        for arguments in ({"stage": "refresh"}, {"stage": "initial", "variant": "B"}):
            with self.subTest(arguments=arguments), patch.object(runner, "load_case") as load, self.assertRaises(ValueError):
                self.fit(self.root / "invalid", **arguments)
            load.assert_not_called()
        initial, refreshed = self.root / "initial", self.root / "refresh"
        with self.patches():
            parent = self.fit(initial)
            result = self.fit(refreshed, stage="refresh", variant="B", parent_checkpoint=parent["checkpoint"],
                              additional_replays=["online-shard"])
        self.assertEqual(result["model_updates"], 4)
        self.assertEqual(result["stage_updates"], 2)
        self.assertFalse((refreshed / "bindings").exists())
        incomplete = initial / "checkpoints/update_00000001.pt"
        with self.patches(), self.assertRaisesRegex(ValueError, "completed initial/refresh"):
            self.fit(self.root / "partial-parent", stage="refresh", variant="B", parent_checkpoint=incomplete,
                     additional_replays=["online-shard"])
        with self.patches(), self.assertRaisesRegex(ValueError, "training_variant"):
            self.fit(self.root / "wrong-branch", stage="refresh", variant="D", parent_checkpoint=parent["checkpoint"],
                     additional_replays=["online-shard"])


if __name__ == "__main__":
    unittest.main()
