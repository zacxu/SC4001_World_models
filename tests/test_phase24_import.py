"""Cross-server continuation uses tiny saved states, never research-model fits."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import phase24 as cli
from scripts import phase24_import as importer
from tests import test_phase24_relocate as baseline
from world_models.training import runner
from world_models.training.checkpoints import read_checkpoint


NEW_RUNTIME = {"python": "3.12.11", "torch": "2.5.1+cu121", "numpy": "2.0.0",
               "cuda_runtime": "12.1", "cudnn": 8907, "gpu_name": "NVIDIA A100-SXM4-40GB",
               "gpu_capability": [8, 0]}


class GPUInventoryTests(unittest.TestCase):
    def test_two_visible_devices_keep_scheduler_masks_and_distinct_physical_identities(self):
        tokens = ["GPU-pbs-5", "GPU-pbs-2"]
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ",".join(tokens)}), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=2), \
             patch.object(torch.cuda, "get_device_properties", side_effect=lambda device:
                 SimpleNamespace(uuid=tokens[device.index])), \
             patch.object(runner, "runtime_contract", return_value=deepcopy(NEW_RUNTIME)):
            inventory = importer._inventory([0, 1])
        self.assertEqual([row["logical_gpu"] for row in inventory], [0, 1])
        self.assertEqual([row["mask_token"] for row in inventory], tokens)
        self.assertEqual([row["uuid"] for row in inventory], tokens)
        self.assertEqual([row["runtime_contract"] for row in inventory], [NEW_RUNTIME] * 2)


class CrossServerImportTests(unittest.TestCase):
    # Reuse realistic model, AdamW moments, four named RNG/sampler states and
    # unequal saved update counts from the existing relocation fixture.
    load_case = staticmethod(baseline.RelocationTests.load_case)
    assert_tree_equal = baseline.RelocationTests.assert_tree_equal
    snapshot = baseline.RelocationTests.snapshot

    def setUp(self):
        baseline.RelocationTests.setUp(self)
        self.data = self.root / "aspire_replays"
        self.statistics = self.root / "aspire_statistics"
        self.config = self.root / "aspire_configs"
        shutil.copytree(runner.ROOT / "configs", self.config)
        self.inventory = [{"logical_gpu": index, "mask_token": str(index),
                           "uuid": f"GPU-aspire-{index}", "runtime_contract": deepcopy(NEW_RUNTIME)}
                          for index in range(4)]
        # The copied plan names paths and a Python interpreter which cannot
        # exist on this machine. Import must parse history without using them.
        path = self.source / "launch_plan.json"
        plan = json.loads(path.read_text())
        for job in plan["jobs"]:
            old_root = f"/retired/H200/project/artifacts/pilot_v1/{job['task']}/seed_{job['seed']}"
            command = job["command"]
            command[0] = "/retired/H200/project/.venv/bin/python"
            job["output_dir"] = old_root
            for option, value in (("--output-dir", old_root), ("--data-dir", "/retired/H200/replays"),
                                  ("--statistics-dir", "/retired/H200/statistics"),
                                  ("--config-dir", "/retired/H200/configs")):
                command[command.index(option) + 1] = value
        baseline.json_write(path, plan)

    def patches(self, inventory=None, load_case=None, visible="0,1,2,3"):
        stack = ExitStack()
        stack.enter_context(patch.object(importer, "_gpu_inventory", return_value=deepcopy(
            self.inventory if inventory is None else inventory)))
        stack.enter_context(patch.object(importer, "_validate_cuda_state", return_value=None))
        stack.enter_context(patch.object(cli, "load_case", side_effect=load_case or self.load_case))
        if hasattr(importer, "load_case"):
            stack.enter_context(patch.object(importer, "load_case", side_effect=load_case or self.load_case))
        stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}))
        return stack

    def prepare(self):
        with self.patches():
            return importer.prepare(self.source, self.output, gpus="0,1,2,3", data_dir=self.data,
                statistics_dir=self.statistics, config_dir=self.config)

    def assert_not_prepared(self):
        self.assertFalse(self.output.exists())

    def test_import_preserves_all_training_state_counts_target_source_and_parent_rng(self):
        source = self.snapshot()
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        report = self.prepare()
        self.assertEqual(self.snapshot(), source)
        self.assertEqual(random.getstate(), before[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertEqual(runner.source_identity(), self.source_hashes)
        self.assertTrue(report["training_state_preserved"])
        self.assertFalse(report["gpu_numerical_equivalence_measured"])
        self.assertEqual(report["optimizer_updates_executed"], 0)
        self.assertEqual(set(report["changed_checkpoint_metadata_fields"]),
                         {"cuda_visible_devices", "runtime_contract"})
        for index, ((task, seed), count, checkpoint) in enumerate(zip(
                runner.CASES, self.counts, self.source_checkpoints)):
            case = self.output / task / f"seed_{seed}"
            old, new = read_checkpoint(checkpoint), read_checkpoint(case / "checkpoints" / checkpoint.name)
            old["metadata"]["cuda_visible_devices"] = str(index)
            old["metadata"]["runtime_contract"] = deepcopy(NEW_RUNTIME)
            self.assert_tree_equal(old, new)
            self.assertEqual(new["update"], count)
            self.assertEqual(new["metadata"]["target_stage_updates"], 5000)
            self.assertEqual(json.loads((case / "run_manifest.json").read_text())["metadata"], new["metadata"])

    def test_original_paths_are_historical_and_new_plan_has_four_parallel_single_gpu_workers(self):
        with self.patches(), patch.object(cli.subprocess, "Popen") as spawn, \
             patch.object(importer, "_validate_cuda_state", return_value=None) as validate:
            importer.prepare(self.source, self.output, gpus="0,1,2,3", data_dir=self.data,
                statistics_dir=self.statistics, config_dir=self.config)
        spawn.assert_not_called()
        self.assertEqual(validate.call_count, 4)
        plan = json.loads((self.output / "launch_plan.json").read_text())
        self.assertEqual(plan["gpus"], [0, 1, 2, 3])
        self.assertEqual(plan["workers_per_gpu"], 1)
        self.assertEqual([job["logical_gpu"] for job in plan["jobs"]], [0, 1, 2, 3])
        for job in plan["jobs"]:
            self.assertNotIn("/retired/", " ".join(job["command"]))
            parsed = cli.parser().parse_args(job["command"][3:])
            self.assertEqual(parsed.data_dir.resolve(), self.data.resolve())
            self.assertEqual(parsed.statistics_dir.resolve(), self.statistics.resolve())
            self.assertEqual(parsed.config_dir.resolve(), self.config.resolve())
            self.assertTrue(parsed.resume)
            self.assertEqual(parsed.updates, 5000)

    def test_two_gpu_import_preserves_all_four_states_and_records_round_robin_slots(self):
        inventory = deepcopy(self.inventory[:2])
        for row, token in zip(inventory, ("GPU-allocation-5", "GPU-allocation-2")):
            row["mask_token"] = token
        visible = ",".join(row["mask_token"] for row in inventory)
        source = self.snapshot()
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with self.patches(inventory, visible=visible), patch.object(cli.subprocess, "Popen") as spawn:
            report = importer.prepare(self.source, self.output, gpus="0,1", workers_per_gpu=2,
                data_dir=self.data, statistics_dir=self.statistics, config_dir=self.config)
        spawn.assert_not_called()
        self.assertEqual(self.snapshot(), source)
        self.assertEqual(random.getstate(), before[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertEqual(report["gpus"], inventory)
        self.assertEqual(report["workers_per_gpu"], 2)
        self.assertEqual(report["optimizer_updates_executed"], 0)
        self.assertTrue(report["training_state_preserved"])
        plan = json.loads((self.output / "launch_plan.json").read_text())
        self.assertEqual(plan["gpus"], [0, 1])
        self.assertEqual(plan["workers_per_gpu"], 2)
        self.assertEqual(plan["worker_count"], 4)
        self.assertEqual([job["logical_gpu"] for job in plan["jobs"]], [0, 1, 0, 1])
        for index, ((task, seed), count, checkpoint, row) in enumerate(zip(
                runner.CASES, self.counts, self.source_checkpoints, report["cases"])):
            expected = read_checkpoint(checkpoint)
            expected["metadata"]["cuda_visible_devices"] = inventory[index % 2]["mask_token"]
            expected["metadata"]["runtime_contract"] = deepcopy(NEW_RUNTIME)
            actual = read_checkpoint(row["checkpoint"])
            self.assert_tree_equal(expected, actual)
            self.assertEqual(actual["update"], count)
            self.assertEqual(actual["metadata"]["target_stage_updates"], 5000)
            self.assertEqual((row["task"], row["seed"]), (task, seed))
            self.assertEqual(row["gpu"], inventory[index % 2])
            self.assertEqual(json.loads(Path(row["manifest"]).read_text())["metadata"], actual["metadata"])

    def test_two_gpu_resume_launches_four_workers_before_waiting_with_single_device_masks(self):
        inventory = deepcopy(self.inventory[:2])
        for row, token in zip(inventory, ("GPU-pbs-5", "GPU-pbs-2")):
            row["mask_token"] = token
        visible = ",".join(row["mask_token"] for row in inventory)
        spawned = []

        def spawn(command, **kwargs):
            args = cli.parser().parse_args(command[3:])
            index = len(spawned)
            self.assertEqual((args.task, args.seed), runner.CASES[index])
            self.assertTrue(args.resume)
            self.assertEqual(args.device, "cuda:0")
            self.assertEqual(args.updates, 5000)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], inventory[index % 2]["mask_token"])
            spawned.append(command)

            def poll():
                self.assertEqual(len(spawned), 4, "all workers must start before waiting")
                return 0

            return SimpleNamespace(pid=100 + index, returncode=0, poll=poll)

        with self.patches(inventory, visible=visible):
            importer.prepare(self.source, self.output, gpus="0,1", workers_per_gpu=2,
                data_dir=self.data, statistics_dir=self.statistics, config_dir=self.config)
            with patch.object(torch.cuda, "is_available", return_value=True), \
                 patch.object(torch.cuda, "device_count", return_value=2), \
                 patch.object(cli.subprocess, "Popen", side_effect=spawn) as launch, \
                 patch.object(cli.time, "sleep", side_effect=AssertionError("unexpected wait")):
                self.assertEqual(importer.resume(self.output), 0)
        self.assertEqual(launch.call_count, 4)
        status = json.loads((self.output / "launch_status.json").read_text())
        self.assertEqual(status["status"], "complete_fit_stages")

    def test_import_prepared_by_an_older_helper_can_be_source_for_fresh_two_gpu_import(self):
        old_report = self.prepare()
        old_report["helper_sha256"] = "legacy-helper-sha256"
        baseline.json_write(self.output / "import.json", old_report)
        (self.output / "import.sha256").write_text(
            f"{importer.sha256_file(self.output / 'import.json')}  import.json\n", encoding="ascii")
        destination = self.root / "two_gpus"
        with self.patches(self.inventory[:2], visible="0,1"):
            report = importer.prepare(self.output, destination, gpus="0,1", workers_per_gpu=2,
                data_dir=self.data, statistics_dir=self.statistics, config_dir=self.config)
        self.assertEqual([row["model_updates"] for row in report["cases"]], self.counts)
        for index, (old, new) in enumerate(zip(old_report["cases"], report["cases"])):
            expected = read_checkpoint(old["checkpoint"])
            expected["metadata"]["cuda_visible_devices"] = str(index % 2)
            self.assert_tree_equal(expected, read_checkpoint(new["checkpoint"]))

    def test_incompatible_python_minor_numpy_or_torch_base_fails_before_destination(self):
        for field, value in (("python", "3.13.0"), ("numpy", "2.1.0"), ("torch", "2.6.0+cu124")):
            inventory = deepcopy(self.inventory)
            inventory[1]["runtime_contract"][field] = value
            with self.subTest(field=field), self.patches(inventory), self.assertRaises(ValueError):
                importer.prepare(self.source, self.output, data_dir=self.data,
                    statistics_dir=self.statistics, config_dir=self.config)
            self.assert_not_prepared()

    def test_cuda_rng_incompatibility_fails_without_any_destination(self):
        with self.patches(), patch.object(importer, "_validate_cuda_state", side_effect=ValueError("incompatible RNG")), \
             self.assertRaisesRegex(ValueError, "RNG"):
            importer.prepare(self.source, self.output, data_dir=self.data,
                statistics_dir=self.statistics, config_dir=self.config)
        self.assert_not_prepared()

    def test_duplicate_gpu_uuid_or_mask_is_rejected_even_with_multiple_workers(self):
        for field in ("uuid", "mask_token"):
            inventory = deepcopy(self.inventory[:2])
            inventory[1][field] = inventory[0][field]
            with self.subTest(field=field), self.patches(inventory), self.assertRaisesRegex(ValueError, "distinct"):
                importer.prepare(self.source, self.output, gpus="0,1", workers_per_gpu=2,
                    data_dir=self.data, statistics_dir=self.statistics, config_dir=self.config)
            self.assert_not_prepared()

    def test_insufficient_capacity_invalid_indices_or_worker_counts_are_rejected(self):
        for gpus in ("0", "0,1,2", "0,1,2,3,4", "0,0,1,2"):
            with self.subTest(gpus=gpus), self.patches(), self.assertRaises(ValueError):
                importer.prepare(self.source, self.output, gpus=gpus)
            self.assert_not_prepared()
        for workers in (0, -1, True, 1.5, "2", 1):
            with self.subTest(workers=workers), self.patches(), self.assertRaises(ValueError):
                importer.prepare(self.source, self.output, gpus="0,1", workers_per_gpu=workers)
            self.assert_not_prepared()

    def test_import_requires_fresh_nonoverlapping_destination(self):
        for destination in (self.source, self.source / "inside", self.root):
            with self.subTest(destination=destination), self.patches(), \
                 self.assertRaises((ValueError, FileExistsError)):
                importer.prepare(self.source, destination)
        self.output.mkdir()
        with self.patches(), self.assertRaises(FileExistsError):
            importer.prepare(self.source, self.output)

    def test_active_copied_source_launcher_or_case_lock_prevents_import(self):
        for directory in (self.source, self.source / "relay/seed_0"):
            with self.subTest(directory=directory), runner.run_lock(directory), self.patches(), \
                 self.assertRaisesRegex(RuntimeError, "worker|lock"):
                importer.prepare(self.source, self.output)
            self.assert_not_prepared()

    def test_corrupt_committed_latest_or_missing_payload_never_falls_back(self):
        checkpoint = self.source_checkpoints[0]
        original = checkpoint.read_bytes()
        receipt = checkpoint.with_suffix(".sha256")
        original_receipt = receipt.read_bytes()
        lock = checkpoint.with_name(checkpoint.name + ".lock")
        for fault in ("truncated", "receipt", "missing", "lock"):
            with self.subTest(fault=fault):
                if fault == "truncated":
                    checkpoint.write_bytes(original[:32])
                elif fault == "receipt":
                    receipt.write_text("malformed receipt", encoding="ascii")
                elif fault == "missing":
                    checkpoint.unlink()
                else:
                    lock.touch()
                try:
                    with self.patches(), self.assertRaises((ValueError, FileExistsError, RuntimeError)):
                        importer.prepare(self.source, self.output)
                    self.assert_not_prepared()
                finally:
                    checkpoint.write_bytes(original)
                    receipt.write_bytes(original_receipt)
                    lock.unlink(missing_ok=True)

    def test_uncommitted_later_payload_is_preserved_and_last_committed_selected(self):
        checkpoint = self.source_checkpoints[0]
        checkpoint.with_suffix(".sha256").unlink()
        before = self.snapshot()
        report = self.prepare()
        self.assertEqual(self.snapshot(), before)
        files = list((self.output / "relay/seed_0/checkpoints").glob("*.pt"))
        self.assertEqual([path.name for path in files], ["update_00000000.pt"])
        self.assertEqual(read_checkpoint(files[0])["update"], 0)
        self.assertIn(checkpoint.name, json.dumps(report))

    def test_checkpoint_metadata_source_settings_target_or_config_changes_reject(self):
        checkpoint = self.source_checkpoints[0]
        original = checkpoint.read_bytes()
        receipt = checkpoint.with_suffix(".sha256")
        original_receipt = receipt.read_bytes()
        changes = {"source_sha256": {}, "common_config_sha256": "changed-config", "settings": {},
                   "initial_normalization": {}, "replays": [], "target_stage_updates": 2000,
                   "cuda_visible_devices": "7"}
        for key, value in changes.items():
            with self.subTest(field=key):
                baseline.rewrite_checkpoint(checkpoint, lambda payload: payload["metadata"].update({key: value}))
                try:
                    with self.patches(), self.assertRaises(ValueError):
                        importer.prepare(self.source, self.output)
                    self.assert_not_prepared()
                finally:
                    checkpoint.write_bytes(original)
                    receipt.write_bytes(original_receipt)

    def test_changed_current_replay_normalizer_or_config_rejects_import(self):
        for fault in ("replay", "normalizer", "config"):
            def changed(task, seed, **kwargs):
                cfg, datasets, stats, provenance = self.load_case(task, seed, **kwargs)
                if fault == "replay":
                    provenance["replays"][0]["manifest_sha256"] = "changed-replay"
                elif fault == "normalizer":
                    provenance["initial_normalization"]["file_sha256"] = "changed-normalizer"
                else:
                    cfg.settings["world_model"]["initial_updates"] = 4999
                return cfg, datasets, stats, provenance
            with self.subTest(fault=fault), self.patches(load_case=changed), self.assertRaises(ValueError):
                importer.prepare(self.source, self.output)
            self.assert_not_prepared()

    def test_changed_original_plan_settings_case_command_or_provenance_rejects_import(self):
        path = self.source / "launch_plan.json"
        original = path.read_bytes()
        for fault in ("settings", "case_command", "provenance", "target"):
            plan = json.loads(original)
            if fault == "settings":
                plan["settings"]["microbatch"] = 2
            elif fault == "case_command":
                command = plan["jobs"][1]["command"]
                command[command.index("--microbatch") + 1] = "2"
            elif fault == "provenance":
                plan["jobs"][1]["data"]["replays"][0]["manifest_sha256"] = "changed"
            else:
                plan["jobs"][1]["target_stage_updates"] = 2000
            baseline.json_write(path, plan)
            try:
                with self.subTest(fault=fault), self.patches(), self.assertRaises(ValueError):
                    importer.prepare(self.source, self.output)
                self.assert_not_prepared()
            finally:
                path.write_bytes(original)

    def test_update_beyond_original_target_is_rejected(self):
        baseline.rewrite_checkpoint(self.source_checkpoints[0], lambda payload: payload.update(update=5001))
        with self.patches(), self.assertRaises(ValueError):
            importer.prepare(self.source, self.output)
        self.assert_not_prepared()

    def test_resume_calls_original_launcher_using_current_paths_four_gpu_plan_and_original_target(self):
        self.prepare()
        with self.patches(), patch.object(cli, "main", return_value=0) as launch:
            result = importer.resume(self.output)
        launch.assert_called_once()
        args = cli.parser().parse_args(launch.call_args.args[0])
        self.assertTrue(args.resume)
        self.assertEqual(args.gpus, "0,1,2,3")
        self.assertEqual(args.workers_per_gpu, 1)
        self.assertEqual(args.updates, 5000)
        self.assertEqual(args.data_dir.resolve(), self.data.resolve())
        self.assertEqual(args.output_dir.resolve(), self.output.resolve())
        self.assertEqual(result, 0)

    def test_new_pbs_job_can_resume_with_different_physical_gpu_uuids(self):
        self.prepare()
        inventory = deepcopy(self.inventory)
        for index, row in enumerate(inventory):
            row["uuid"] = f"GPU-next-PBS-allocation-{index}"
        with self.patches(inventory), patch.object(cli, "main", return_value=0) as launch:
            self.assertEqual(importer.resume(self.output), 0)
        launch.assert_called_once()

    def test_resume_rejects_aliased_physical_gpus_before_launch(self):
        self.prepare()
        inventory = deepcopy(self.inventory)
        inventory[1]["uuid"] = inventory[0]["uuid"]
        with self.patches(inventory), patch.object(cli, "main") as launch, \
             self.assertRaisesRegex(ValueError, "distinct"):
            importer.resume(self.output)
        launch.assert_not_called()

    def test_source_and_imported_checkpoint_receipt_corruption_prevents_resume(self):
        self.prepare()
        imported = self.output / "relay/seed_0/checkpoints" / self.source_checkpoints[0].name
        for checkpoint in (self.source_checkpoints[0], imported):
            receipt = checkpoint.with_suffix(".sha256")
            original = receipt.read_bytes()
            receipt.write_text("malformed checkpoint receipt", encoding="ascii")
            try:
                with self.subTest(checkpoint=checkpoint), self.patches(), patch.object(cli, "main") as launch, \
                     self.assertRaises(ValueError):
                    importer.resume(self.output)
                launch.assert_not_called()
            finally:
                receipt.write_bytes(original)

    def test_resume_rejects_changed_runtime_mask_plan_fitting_source_or_receipt_before_launch(self):
        self.prepare()
        plan_path = self.output / "launch_plan.json"
        original_plan = plan_path.read_bytes()
        receipt = self.output / "import.sha256"
        original_receipt = receipt.read_bytes()
        for fault in ("runtime", "mask", "plan", "source", "receipt", "missing_receipt"):
            inventory = deepcopy(self.inventory)
            if fault == "runtime":
                inventory[0]["runtime_contract"]["gpu_name"] = "Different GPU"
            elif fault == "mask":
                inventory[0]["mask_token"] = "9"
            elif fault == "plan":
                plan = json.loads(original_plan)
                plan["workers_per_gpu"] = 4
                baseline.json_write(plan_path, plan)
            elif fault == "receipt":
                receipt.write_text("malformed", encoding="ascii")
            elif fault == "missing_receipt":
                receipt.unlink()
            try:
                with self.subTest(fault=fault), self.patches(inventory), ExitStack() as stack:
                    if fault == "source":
                        stack.enter_context(patch.object(runner, "source_identity", return_value={"changed.py": "changed"}))
                        stack.enter_context(patch.object(cli, "source_identity", return_value={"changed.py": "changed"}))
                    launch = stack.enter_context(patch.object(cli, "main"))
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        importer.resume(self.output)
                    launch.assert_not_called()
            finally:
                plan_path.write_bytes(original_plan)
                receipt.write_bytes(original_receipt)


class CUDARandomStateValidationTests(unittest.TestCase):
    def payload(self):
        return {"rng": {"torch_cuda": {"0": torch.tensor([1, 2], dtype=torch.uint8)}},
                "streams": {name: torch.tensor([index, 4], dtype=torch.uint8)
                            for index, name in enumerate(("prior", "posterior", "outcome"))}}

    def test_rng_compatibility_checks_use_four_isolated_generators_without_global_restore(self):
        payload, states = self.payload(), []
        before = torch.get_rng_state().clone()

        def generator(*, device):
            self.assertEqual(device, "cuda:3")
            local = {}

            def set_state(state):
                local["state"] = state.clone()
                states.append(state.clone())

            return SimpleNamespace(set_state=set_state, get_state=lambda: local["state"].clone())

        with patch.object(importer.torch, "Generator", side_effect=generator) as make, \
             patch.object(torch.cuda, "set_rng_state", side_effect=AssertionError("global CUDA RNG restore")) as restore, \
             patch.object(torch, "set_rng_state", side_effect=AssertionError("global CPU RNG restore")) as cpu_restore:
            importer._validate_cuda_state(payload, 3)
        self.assertEqual(make.call_count, 4)
        restore.assert_not_called()
        cpu_restore.assert_not_called()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        expected = [payload["rng"]["torch_cuda"]["0"], *payload["streams"].values()]
        for old, actual in zip(expected, states):
            self.assertTrue(torch.equal(old, actual))

    def test_rng_restore_failure_is_actionable_and_multidevice_source_is_rejected(self):
        payload = self.payload()
        with patch.object(importer.torch, "Generator", side_effect=RuntimeError("unsupported state")), \
             self.assertRaisesRegex(ValueError, "incompatible.*CUDA"):
            importer._validate_cuda_state(payload, 0)
        generator = SimpleNamespace(set_state=lambda state: None,
                                    get_state=lambda: torch.tensor([9, 9], dtype=torch.uint8))
        with patch.object(importer.torch, "Generator", return_value=generator), \
             self.assertRaisesRegex(ValueError, "RNG state changed"):
            importer._validate_cuda_state(payload, 0)
        payload["rng"]["torch_cuda"]["1"] = payload["rng"]["torch_cuda"]["0"]
        with patch.object(importer.torch, "Generator") as make, self.assertRaisesRegex(ValueError, "single-GPU"):
            importer._validate_cuda_state(payload, 0)
        make.assert_not_called()


class ImportSignalTests(unittest.TestCase):
    def test_pbs_termination_interrupts_launcher_and_restores_original_signal_handlers(self):
        handlers = {signal.SIGINT: signal.SIG_IGN, signal.SIGTERM: signal.SIG_DFL}
        active = dict(handlers)
        registrations = []

        def install(name, handler):
            registrations.append((name, handler))
            active[name] = handler

        def resume(_):
            self.assertIs(active[signal.SIGINT], signal.default_int_handler)
            # Raising KeyboardInterrupt unwinds the original launcher's existing
            # cleanup, which terminates and waits for all four child processes.
            active[signal.SIGTERM](signal.SIGTERM, None)

        with patch.object(importer.signal, "getsignal", side_effect=handlers.__getitem__), \
             patch.object(importer.signal, "signal", side_effect=install), \
             patch.object(importer, "resume", side_effect=resume) as launch, \
             self.assertRaises(KeyboardInterrupt):
            importer.main(["resume", "--output-dir", "saved_import"])
        launch.assert_called_once()
        self.assertEqual(active, handlers)
        self.assertEqual([name for name, _ in registrations],
                         [signal.SIGINT, signal.SIGTERM, signal.SIGINT, signal.SIGTERM])


if __name__ == "__main__":
    unittest.main()
