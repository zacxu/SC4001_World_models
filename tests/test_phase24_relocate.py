"""Four-worker GPU relocation checks using tiny CPU checkpoints, never research fits."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import phase24 as cli
from scripts import phase24_relocate as relocate
from tests.test_model_checkpoints import fixture, update
from tests.test_phase24_cli import case_fixture
from world_models.data.replay import canonical_hash
from world_models.infrastructure.reproducibility import derive_seed
from world_models.training import runner
from world_models.training.checkpoints import read_checkpoint, save_checkpoint


RUNTIME = {"python": "3.12.0", "torch": "2.5.1+cu124", "numpy": "2.0.0",
           "cuda_runtime": "12.4", "cudnn": 90100, "gpu_name": "NVIDIA H200",
           "gpu_capability": [9, 0]}


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def rewrite_checkpoint(path, mutate):
    payload = read_checkpoint(path)
    mutate(payload)
    torch.save(payload, path)
    Path(path).with_suffix(".sha256").write_text(
        f"{hashlib.sha256(Path(path).read_bytes()).hexdigest()}  {Path(path).name}\n", encoding="ascii")


class GPUInventoryTests(unittest.TestCase):
    def test_visible_logical_devices_map_to_inherited_physical_masks_and_distinct_uuids(self):
        for mask in ("5,3,7,1", "GPU-assigned-5,GPU-assigned-3,GPU-assigned-7,GPU-assigned-1"):
            tokens = mask.split(",")
            uuids = [f"GPU-physical-{number}" for number in (5, 3, 7, 1)]
            devices = [torch.device(f"cuda:{index}") for index in range(4)]
            with self.subTest(mask=mask), \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mask}), \
                 patch.object(torch.cuda, "is_available", return_value=True), \
                 patch.object(torch.cuda, "device_count", return_value=4), \
                 patch.object(torch.cuda, "get_device_properties", side_effect=lambda device:
                     SimpleNamespace(uuid=uuids[device.index])) as properties, \
                 patch.object(runner, "runtime_contract", return_value=deepcopy(RUNTIME)) as runtime:
                actual = relocate._inventory([0, 1, 2, 3])
                self.assertEqual([row["logical_gpu"] for row in actual], [0, 1, 2, 3])
                self.assertEqual([row["mask_token"] for row in actual], tokens)
                self.assertEqual([row["uuid"] for row in actual], uuids)
                self.assertEqual(len({row["uuid"] for row in actual}), 4)
                self.assertEqual([row["runtime_contract"] for row in actual], [RUNTIME] * 4)
                self.assertEqual([call.args[0] for call in properties.call_args_list], devices)
                self.assertEqual([call.args[0] for call in runtime.call_args_list], devices)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], mask)


class RelocationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        self.output = self.root / "four_gpus"
        self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.addCleanup(random.setstate, self.rng[0])
        self.addCleanup(np.random.set_state, self.rng[1])
        self.addCleanup(torch.set_rng_state, self.rng[2])
        self.source_hashes = runner.source_identity()
        self.counts = [1000, 1250, 1500, 1750]
        self.inventory = [{"logical_gpu": index, "mask_token": str(index),
                           "uuid": f"GPU-fixture-{index}", "runtime_contract": deepcopy(RUNTIME)}
                          for index in range(4)]
        args = cli.parser().parse_args(["plan", "--output-dir", str(self.source),
            "--data-dir", str(self.root / "data"), "--statistics-dir", str(self.root / "statistics"),
            "--gpus", "0", "--workers-per-gpu", "4"])
        with patch.object(cli, "load_case", side_effect=self.load_case):
            plan = cli.launch_plan(args)
        json_write(self.source / "launch_plan.json", plan)
        self.source_checkpoints = []
        for (task, seed), count in zip(runner.CASES, self.counts):
            cfg, _, _, provenance = self.load_case(task, seed)
            settings = cli.settings_from_args(args, task)
            config = deepcopy(cfg.settings)
            config.pop("variant")
            metadata = {"task_family": task, "run_seed": seed, "profile": "pilot", "stage": "initial",
                "variant": "shared_initial:B,D", "common_config_sha256": canonical_hash(config),
                "settings": asdict(settings), "precision": settings.precision, "device_type": "cuda",
                "device": "cuda:0", "cuda_visible_devices": "0", "target_stage_updates": 5000,
                "source_sha256": self.source_hashes, **provenance,
                "fitting_seed": derive_seed(task, seed, "world_model_fitting_initial", "B"),
                "runtime_contract": deepcopy(RUNTIME),
                "normalization_policy": "frozen initial fitting statistics; no updates from diagnostics or refresh"}
            case = self.source / task / f"seed_{seed}"
            json_write(case / "run_manifest.json", {"schema_version": "p24-run-v1", "metadata": metadata,
                "resolved_config": cfg.as_dict(), "runtime": {"device": "cuda:0", "fixture": True},
                "created_at_utc": "2026-11-05T00:00:00+00:00", "training_budget_interactions_added": 0})
            model, optimizer, streams, sampler = fixture()
            first = case / "checkpoints" / "update_00000000.pt"
            save_checkpoint(first, model=model, optimizer=optimizer, streams=streams,
                            sampler=sampler, update=0, metadata=metadata)
            rewrite_checkpoint(first, lambda payload: payload["rng"].update(
                torch_cuda={"0": torch.tensor([1, 2, 3, 4], dtype=torch.uint8)}))
            # Populate real AdamW moments with one tiny fixture update. The large
            # update label represents existing server work only in this fixture.
            update(model, optimizer, streams, sampler)
            checkpoint = case / "checkpoints" / f"update_{count:08d}.pt"
            save_checkpoint(checkpoint, model=model, optimizer=optimizer, streams=streams,
                            sampler=sampler, update=count, metadata=metadata)
            rewrite_checkpoint(checkpoint, lambda payload: payload["rng"].update(
                torch_cuda={"0": torch.tensor([1, 2, 3, 4], dtype=torch.uint8)}))
            self.source_checkpoints.append(checkpoint)
            json_write(case / "status.json", {"status": "interrupted", "model_updates": count + 3,
                "last_committed_update": count, "target_stage_updates": 5000})
        json_write(self.source / "launch_status.json", {"status": "failed", "workers": []})

    @staticmethod
    def load_case(task, seed, **kwargs):
        return case_fixture(task, seed)

    def patches(self, inventory=None):
        stack = ExitStack()
        stack.enter_context(patch.object(relocate, "_gpu_inventory", return_value=deepcopy(
            self.inventory if inventory is None else inventory)))
        stack.enter_context(patch.object(cli, "load_case", side_effect=self.load_case))
        if hasattr(relocate, "load_case"):
            stack.enter_context(patch.object(relocate, "load_case", side_effect=self.load_case))
        stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}))
        return stack

    def prepare(self):
        with self.patches():
            return relocate.prepare(self.source, self.output, gpus="0,1,2,3")

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            self.assertEqual(left.dtype, right.dtype)
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for old, new in zip(left, right):
                self.assert_tree_equal(old, new)
        else:
            self.assertEqual(left, right)

    def snapshot(self):
        # Advisory lock files are synchronization bookkeeping, not run evidence.
        return {path.relative_to(self.source).as_posix(): path.read_bytes()
                for path in self.source.rglob("*") if path.is_file() and path.name != ".worker.lock"}

    def test_payload_changes_only_cuda_mask_and_is_detached_from_source(self):
        original = read_checkpoint(self.source_checkpoints[0])
        snapshot = deepcopy(original)
        actual = relocate.relocate_payload(original, "GPU-destination")
        self.assert_tree_equal(original, snapshot)
        expected = deepcopy(original)
        expected["metadata"]["cuda_visible_devices"] = "GPU-destination"
        self.assert_tree_equal(actual, expected)
        actual["model"]["weight"].fill_(900)
        actual["metadata"]["settings"]["microbatch"] = 900
        self.assert_tree_equal(original, snapshot)

    def test_checkpoint_writer_preserves_all_training_state_and_is_immutable(self):
        payload = relocate.relocate_payload(read_checkpoint(self.source_checkpoints[0]), "3")
        destination = self.root / "copy" / self.source_checkpoints[0].name
        relocate.write_relocated_checkpoint(destination, payload)
        self.assert_tree_equal(read_checkpoint(destination), payload)
        before = destination.read_bytes()
        with self.assertRaises(FileExistsError):
            relocate.write_relocated_checkpoint(destination, payload)
        self.assertEqual(destination.read_bytes(), before)

    def test_checkpoint_writer_rejects_invalid_state_without_publishing_receipt(self):
        payload = relocate.relocate_payload(read_checkpoint(self.source_checkpoints[0]), "3")
        payload["model"]["weight"] = payload["model"]["weight"].double()
        destination = self.root / "invalid" / self.source_checkpoints[0].name
        with self.assertRaises(ValueError):
            relocate.write_relocated_checkpoint(destination, payload)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_suffix(".sha256").exists())

    def test_prepare_keeps_source_all_four_counts_target_and_cpu_rng(self):
        source = self.snapshot()
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.prepare()
        self.assertEqual(self.snapshot(), source)
        self.assertEqual(random.getstate(), before[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertEqual(runner.source_identity(), self.source_hashes)
        plan = json.loads((self.output / "launch_plan.json").read_text())
        self.assertEqual(plan["gpus"], [0, 1, 2, 3])
        self.assertEqual(plan["workers_per_gpu"], 1)
        self.assertEqual([job["logical_gpu"] for job in plan["jobs"]], [0, 1, 2, 3])
        for index, ((task, seed), count, old_checkpoint) in enumerate(zip(
                runner.CASES, self.counts, self.source_checkpoints)):
            case = self.output / task / f"seed_{seed}"
            new_checkpoint = case / "checkpoints" / old_checkpoint.name
            old, new = read_checkpoint(old_checkpoint), read_checkpoint(new_checkpoint)
            old["metadata"]["cuda_visible_devices"] = str(index)
            self.assert_tree_equal(new, old)
            self.assertEqual(new["update"], count)
            self.assertEqual(new["metadata"]["target_stage_updates"], 5000)
            self.assertEqual(json.loads((case / "run_manifest.json").read_text())["metadata"], new["metadata"])

    def test_destination_cannot_be_source_child_or_reused(self):
        for destination in (self.source, self.source / "inside"):
            with self.subTest(destination=destination), self.patches(), self.assertRaises((ValueError, FileExistsError)):
                relocate.prepare(self.source, destination, gpus="0,1,2,3")
        self.output.mkdir()
        with self.patches(), self.assertRaises(FileExistsError):
            relocate.prepare(self.source, self.output, gpus="0,1,2,3")

    def test_active_source_worker_or_launcher_prevents_preparation(self):
        for directory in (self.source, self.source / "relay/seed_0"):
            with self.subTest(directory=directory), runner.run_lock(directory), self.patches(), \
                 self.assertRaisesRegex(RuntimeError, "worker|lock"):
                relocate.prepare(self.source, self.output, gpus="0,1,2,3")
            self.assertFalse(self.output.exists())

    def test_corrupt_committed_receipt_and_locked_checkpoint_fail_closed(self):
        checkpoint = self.source_checkpoints[0]
        original = checkpoint.read_bytes()
        receipt = checkpoint.with_suffix(".sha256")
        original_receipt = receipt.read_bytes()
        lock = checkpoint.with_name(checkpoint.name + ".lock")
        for fault in ("truncated", "receipt", "lock"):
            with self.subTest(fault=fault):
                if fault == "truncated":
                    checkpoint.write_bytes(original[:32])
                elif fault == "receipt":
                    receipt.write_text("malformed receipt", encoding="ascii")
                else:
                    lock.touch()
                try:
                    with self.patches(), self.assertRaises((ValueError, FileExistsError, RuntimeError)):
                        relocate.prepare(self.source, self.output, gpus="0,1,2,3")
                    self.assertFalse(self.output.exists())
                finally:
                    checkpoint.write_bytes(original)
                    receipt.write_bytes(original_receipt)
                    lock.unlink(missing_ok=True)

    def test_uncommitted_orphan_is_preserved_and_previous_committed_checkpoint_selected(self):
        checkpoint = self.source_checkpoints[0]
        checkpoint.with_suffix(".sha256").unlink()
        before = self.snapshot()
        report = self.prepare()
        self.assertEqual(self.snapshot(), before)
        paths = list((self.output / "relay/seed_0/checkpoints").glob("*.pt"))
        self.assertEqual([path.name for path in paths], ["update_00000000.pt"])
        self.assertEqual(read_checkpoint(paths[0])["update"], 0)
        self.assertIn(checkpoint.name, json.dumps(report))

    def test_training_metadata_mismatches_fail_before_destination_publication(self):
        checkpoint = self.source_checkpoints[0]
        original = checkpoint.read_bytes()
        original_receipt = checkpoint.with_suffix(".sha256").read_bytes()
        changes = {"source_sha256": {}, "common_config_sha256": "changed-config",
                   "initial_normalization": {"identity_sha256": "changed-normalizer"},
                   "replays": [], "settings": {}, "runtime_contract": {"torch": "changed"},
                   "target_stage_updates": 2000, "cuda_visible_devices": "7"}
        for key, value in changes.items():
            with self.subTest(field=key):
                rewrite_checkpoint(checkpoint, lambda payload: payload["metadata"].update({key: value}))
                try:
                    with self.patches(), self.assertRaises(ValueError):
                        relocate.prepare(self.source, self.output, gpus="0,1,2,3")
                    self.assertFalse(self.output.exists())
                finally:
                    checkpoint.write_bytes(original)
                    checkpoint.with_suffix(".sha256").write_bytes(original_receipt)

    def test_checkpoint_update_beyond_preserved_training_budget_is_rejected(self):
        rewrite_checkpoint(self.source_checkpoints[0], lambda payload: payload.update(update=5001))
        with self.patches(), self.assertRaises(ValueError):
            relocate.prepare(self.source, self.output, gpus="0,1,2,3")
        self.assertFalse(self.output.exists())

    def test_different_gpu_runtime_or_duplicate_device_uuid_is_rejected(self):
        for fault in ("runtime", "uuid"):
            inventory = deepcopy(self.inventory)
            if fault == "runtime":
                inventory[2]["runtime_contract"]["gpu_name"] = "NVIDIA A100"
            else:
                inventory[2]["uuid"] = inventory[0]["uuid"]
            with self.subTest(fault=fault), self.patches(inventory), self.assertRaises(ValueError):
                relocate.prepare(self.source, self.output, gpus="0,1,2,3")
            self.assertFalse(self.output.exists())

    def test_current_replay_normalizer_and_config_changes_are_rejected(self):
        for fault in ("replay", "normalizer", "config"):
            def changed(task, seed, **kwargs):
                cfg, datasets, stats, provenance = self.load_case(task, seed, **kwargs)
                if fault == "replay":
                    provenance["replays"][0]["manifest_sha256"] = "replaced-replay"
                elif fault == "normalizer":
                    provenance["initial_normalization"]["file_sha256"] = "replaced-normalizer"
                else:
                    cfg.settings["world_model"]["initial_updates"] = 4999
                return cfg, datasets, stats, provenance
            with self.subTest(fault=fault), self.patches(), ExitStack() as stack:
                stack.enter_context(patch.object(cli, "load_case", side_effect=changed))
                if hasattr(relocate, "load_case"):
                    stack.enter_context(patch.object(relocate, "load_case", side_effect=changed))
                with self.assertRaises(ValueError):
                    relocate.prepare(self.source, self.output, gpus="0,1,2,3")
                self.assertFalse(self.output.exists())

    def test_modified_original_launch_settings_are_rejected(self):
        path = self.source / "launch_plan.json"
        plan = json.loads(path.read_text())
        plan["settings"]["microbatch"] = 2
        json_write(path, plan)
        with self.patches(), self.assertRaises(ValueError):
            relocate.prepare(self.source, self.output, gpus="0,1,2,3")
        self.assertFalse(self.output.exists())

    def test_resume_calls_original_launcher_with_four_single_gpu_workers(self):
        self.prepare()
        with self.patches(), patch.object(cli, "main", return_value=0) as launch:
            result = relocate.resume(self.output)
        launch.assert_called_once()
        args = cli.parser().parse_args(launch.call_args.args[0])
        self.assertTrue(args.resume)
        self.assertEqual(args.gpus, "0,1,2,3")
        self.assertEqual(args.workers_per_gpu, 1)
        self.assertEqual(args.updates, 5000)
        self.assertEqual(args.output_dir.resolve(), self.output.resolve())
        self.assertEqual(result, 0)

    def test_changed_destination_plan_prevents_resume_launch(self):
        self.prepare()
        path = self.output / "launch_plan.json"
        plan = json.loads(path.read_text())
        plan["workers_per_gpu"] = 4
        json_write(path, plan)
        with self.patches(), patch.object(cli, "main") as launch, self.assertRaises(ValueError):
            relocate.resume(self.output)
        launch.assert_not_called()

    def test_changed_gpu_mask_prevents_resume_launch(self):
        self.prepare()
        inventory = deepcopy(self.inventory)
        inventory[1]["mask_token"] = "9"
        with self.patches(inventory), patch.object(cli, "main") as launch, self.assertRaises(ValueError):
            relocate.resume(self.output)
        launch.assert_not_called()

    def test_changed_training_source_identity_prevents_resume_launch(self):
        self.prepare()
        with self.patches(), patch.object(runner, "source_identity", return_value={"changed.py": "changed"}), \
             patch.object(cli, "source_identity", return_value={"changed.py": "changed"}), \
             patch.object(cli, "main") as launch, self.assertRaises(ValueError):
            relocate.resume(self.output)
        launch.assert_not_called()

    def test_missing_or_changed_relocation_receipt_prevents_resume_launch(self):
        self.prepare()
        receipt = self.output / "relocation.sha256"
        original = receipt.read_bytes()
        for fault in ("missing", "malformed"):
            with self.subTest(fault=fault):
                if fault == "missing":
                    receipt.unlink()
                else:
                    receipt.write_text("malformed", encoding="ascii")
                try:
                    with self.patches(), patch.object(cli, "main") as launch, self.assertRaises((ValueError, FileNotFoundError)):
                        relocate.resume(self.output)
                    launch.assert_not_called()
                finally:
                    receipt.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
