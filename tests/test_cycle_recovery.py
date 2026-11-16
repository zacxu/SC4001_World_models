"""Numerical recovery preserves committed fixtures and refuses unrelated drift."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests import test_cycle_controller as fixtures
from tests.test_imagined_rollout_generator import assert_tree_equal
from scripts.phase36_support.evaluation import load_actor_export
from scripts.phase36_support.gate import _export
from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.cycle.controller import CycleController, CycleOptions
from world_models.cycle.provenance import source_identity
from world_models.cycle.recovery import (
    ADDED_SOURCES, LEGACY_PATCH_FILES, LEGACY_SOURCE_IDENTITY, REPLAY_ERROR,
    _validate_sources, recover_run,
)
from world_models.data.ledger import atomic_json, file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.training.checkpoints import read_checkpoint
from world_models.training.runner import run_lock


def legacy_sources():
    result = source_identity()
    for name in ADDED_SOURCES:
        result.pop(name)
    result.update(LEGACY_PATCH_FILES)
    return result


class CycleRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse synthetic P2.4 input construction only, not its test suite.
        fixtures.CycleControllerTests.setUpClass.__func__(cls)
        cls.settings = deepcopy(cls.settings)
        cls.settings["mappo"]["initial_local_cycles"] = 2
        payload = read_checkpoint(cls.initial_checkpoint)
        common = deepcopy(cls.settings)
        common.pop("variant")
        payload["metadata"]["common_config_sha256"] = canonical_hash(common)
        cls.initial_checkpoint = cls.root / "recovery_initial.pt"
        torch.save(payload, cls.initial_checkpoint)
        cls.initial_checkpoint.with_suffix(".sha256").write_text(
            f"{file_sha256(cls.initial_checkpoint)}  {cls.initial_checkpoint.name}\n", encoding="ascii")
        del payload

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "failed"
        self.destination = Path(self.temp.name) / "recovered"

    def failed_fixture(self):
        controller = CycleController.prepare(self.source, settings=deepcopy(self.settings), master_seed=0,
            initial_dataset=self.dataset, normalization=self.normalization,
            initial_checkpoint=self.initial_checkpoint, device="cpu",
            options=CycleOptions(precision="float32", model_microbatch=1, checkpoint_cycles=1))
        controller.run(max_units=1)
        controller.manifest["source_sha256"] = legacy_sources()
        atomic_json(self.source / "run_manifest.json", controller.manifest)
        checkpoint = controller._checkpoint()
        controller._status("failed", error_type="ValueError", error=REPLAY_ERROR + "; check history and precision")
        del controller
        gc.collect()
        return checkpoint

    def test_known_patch_rejects_unknown_sources_and_changes_outside_allowlist(self):
        old, current = legacy_sources(), source_identity()
        self.assertEqual(canonical_hash(old), LEGACY_SOURCE_IDENTITY)
        _validate_sources(old, current)
        for altered_old, altered_new in (
            ({**old, "src/world_models/policies/actor.py": "0" * 64}, current),
            (old, {**current, "src/world_models/models/core.py": "0" * 64}),
            (old, {**current, "src/world_models/unrelated.py": "0" * 64}),
        ):
            with self.subTest(previous_changed=altered_old != old), self.assertRaises(ValueError):
                _validate_sources(altered_old, altered_new)

    def test_recovery_preserves_all_state_and_baseline_then_continues_to_p36_export(self):
        checkpoint = self.failed_fixture()
        original, _ = load_cycle_checkpoint(checkpoint)
        source_hashes = {p.relative_to(self.source).as_posix(): file_sha256(p)
                         for p in self.source.rglob("*") if p.is_file()}
        with patch.object(CycleController, "run", side_effect=AssertionError("recovery must not train")):
            report = recover_run(self.source, self.destination, device="cpu")
        self.assertEqual(report["remaining_initial_units"], 2)
        self.assertEqual(report["local_cycles_completed"], 1)
        self.assertFalse(report["training_executed"])
        recovered, metadata = load_cycle_checkpoint(self.destination / "checkpoints" / checkpoint.name)
        assert_tree_equal(self, original, recovered)
        self.assertEqual(source_hashes, {p.relative_to(self.source).as_posix(): file_sha256(p)
                                       for p in self.source.rglob("*") if p.is_file()})
        new_manifest = json.loads((self.destination / "run_manifest.json").read_text())
        self.assertEqual(metadata["run_identity"], canonical_hash(new_manifest))
        self.assertEqual(new_manifest["source_sha256"], source_identity())
        self.assertFalse(new_manifest["recovery"]["exact_original_numerical_trajectory"])
        baseline = self.destination / "evaluation/real_000000000"
        _, old_meta = load_cycle_checkpoint(self.source / "evaluation/real_000000000")
        _, new_meta = load_cycle_checkpoint(baseline)
        self.assertEqual(old_meta, new_meta)
        with self.assertRaisesRegex(ValueError, "source identity changed"):
            CycleController.resume(self.source, device="cpu")
        controller = CycleController.resume(self.destination, device="cpu")
        report = controller.run(max_units=2)
        self.assertEqual(report["local_cycles_completed"], 2)
        self.assertEqual(report["real_joint_transitions"], 6)
        self.assertEqual(report["phase"], "collect")
        identities = []
        for path in (baseline, self.destination / "evaluation/real_000000006"):
            actor, record = load_actor_export(path)
            del actor
            _export(record, self.destination)
            identities.append(record["run_identity"])
        self.assertEqual(identities[0], identities[1])

    def test_bad_receipt_and_manifest_binding_fail_before_destination_creation(self):
        checkpoint = self.failed_fixture()
        receipt = checkpoint / "manifest.sha256"
        original_receipt = receipt.read_bytes()
        receipt.write_text("0" * 64 + "  manifest.json\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "receipt mismatch"):
            recover_run(self.source, self.destination, device="cpu")
        self.assertFalse(self.destination.exists())
        receipt.write_bytes(original_receipt)
        manifest = json.loads((self.source / "run_manifest.json").read_text())
        manifest["created_at"] = "changed"
        atomic_json(self.source / "run_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "not bound"):
            recover_run(self.source, self.destination, device="cpu")
        self.assertFalse(self.destination.exists())

    def test_running_source_runtime_change_or_uncommitted_progress_are_rejected(self):
        self.failed_fixture()
        path = self.source / "status.json"
        status = json.loads(path.read_text())
        atomic_json(path, {**status, "status": "running"})
        with self.assertRaisesRegex(ValueError, "recorded initial behavior"):
            recover_run(self.source, self.destination, device="cpu")
        atomic_json(path, {**status, "local_cycles_completed": 2})
        with self.assertRaisesRegex(ValueError, "status leads"):
            recover_run(self.source, self.destination, device="cpu")
        atomic_json(path, status)
        with patch("world_models.cycle.recovery._runtime", return_value={}), \
             self.assertRaisesRegex(ValueError, "same logical device and runtime"):
            recover_run(self.source, self.destination, device="cpu")
        with run_lock(self.source), self.assertRaisesRegex(RuntimeError, "another worker"):
            recover_run(self.source, self.destination, device="cpu")
        self.assertFalse(self.destination.exists())

    def test_existing_or_nested_destination_is_not_overwritten(self):
        self.failed_fixture()
        self.destination.mkdir()
        marker = self.destination / "keep.txt"
        marker.write_text("existing result")
        with self.assertRaises(FileExistsError):
            recover_run(self.source, self.destination, device="cpu")
        self.assertEqual(marker.read_text(), "existing result")
        with self.assertRaisesRegex(ValueError, "separate"):
            recover_run(self.source, self.source / "nested", device="cpu")


if __name__ == "__main__":
    unittest.main()
