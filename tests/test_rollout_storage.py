"""Synthetic stores commit atomically, retain provenance and exclude real replay."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_imagined_rollout_schema import make_rollout
from world_models.data.replay import ReplayDataset
from world_models.rollouts.storage import load_rollout, save_rollout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_tree(case, left, right):
    if isinstance(left, torch.Tensor):
        case.assertEqual(left.dtype, right.dtype)
        case.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        case.assertEqual(set(left), set(right))
        for key in left:
            equal_tree(case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            equal_tree(case, a, b)
    else:
        case.assertEqual(left, right)


class RolloutStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "imagined"

    def save(self):
        rollout = make_rollout()
        manifest = save_rollout(rollout, self.output)
        self.assertEqual(manifest, self.output / "manifest.json")
        return rollout

    def manifest(self):
        return json.loads((self.output / "manifest.json").read_text())

    def recommit(self, manifest):
        path = self.output / "manifest.json"
        path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        (self.output / "manifest.sha256").write_text(f"{digest(path)}  manifest.json\n")

    def replace_payload(self, payload):
        path = self.output / "rollout.pt"
        torch.save(payload, path)
        manifest = self.manifest()
        manifest["payload"]["sha256"] = digest(path)
        manifest["payload"]["bytes"] = path.stat().st_size
        self.recommit(manifest)

    def test_roundtrip_retains_tensor_alignment_metadata_rng_and_synthetic_markers(self):
        rollout = make_rollout()
        before = torch.get_rng_state().clone()
        save_rollout(rollout, self.output)
        loaded = load_rollout(self.output)
        equal_tree(self, rollout.tensor_payload(), loaded.tensor_payload())
        equal_tree(self, rollout.streams_before, loaded.streams_before)
        equal_tree(self, rollout.streams_after, loaded.streams_after)
        self.assertEqual(rollout.metadata, loaded.metadata)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        manifest = self.manifest()
        self.assertEqual(manifest["schema_version"], "imagined-on-policy-v1")
        self.assertEqual(manifest["data_source"], "world_model_imagination")
        self.assertIs(manifest["learning_eligible_for_world_model"], False)
        self.assertEqual(manifest["metadata"]["anchor_references"], rollout.metadata["anchor_references"])
        self.assertEqual(manifest["tensor_specs"]["tensors/actions"]["shape"], list(rollout.actions.shape))
        payload = torch.load(self.output / "rollout.pt", weights_only=True)
        self.assertEqual(payload["tensors"]["actions"].device.type, "cpu")
        self.assertFalse(payload["tensors"]["actions"].is_inference())
        rollout.actions.fill_(11)
        self.assertFalse(torch.equal(rollout.actions, loaded.actions))

    def test_existing_output_directory_is_never_overwritten(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            save_rollout(make_rollout(), self.output)
        self.output.rmdir()
        self.save()
        original = (self.output / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            save_rollout(make_rollout(), self.output)
        self.assertEqual(original, (self.output / "manifest.json").read_bytes())

    def test_outer_inference_context_loads_ordinary_tensors_usable_for_policy_gradients(self):
        rollout = make_rollout()
        with torch.inference_mode():
            save_rollout(rollout, self.output)
            loaded = load_rollout(self.output)
        def ordinary(value):
            if isinstance(value, torch.Tensor):
                self.assertFalse(value.is_inference())
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
            elif isinstance(value, dict):
                for child in value.values():
                    ordinary(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    ordinary(child)
        ordinary(loaded.tensor_payload())
        ordinary(loaded.streams_before)
        ordinary(loaded.streams_after)
        weight = torch.nn.Parameter(torch.ones(256))
        loss = (loaded.shared_tokens * weight).sum()
        loss.backward()
        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.isfinite(weight.grad).all())

    def test_real_replay_ancestors_block_synthetic_store_creation(self):
        for index, (name, marker) in enumerate((
            ("manifest.json", {"dataset_version": "real-replay-v1"}),
            ("manifest.partial.json", {"metadata": {"data_source": "real_environment"}}),
        )):
            replay = self.root / f"real_{index}"
            replay.mkdir()
            path = replay / name
            path.write_text(json.dumps(marker))
            target = replay / "nested" / "imagined"
            with self.assertRaisesRegex(ValueError, "real replay"):
                save_rollout(make_rollout(), target)
            self.assertFalse(target.exists())
            self.assertEqual(json.loads(path.read_text()), marker)

    def test_real_replay_reader_rejects_the_synthetic_store(self):
        self.save()
        with self.assertRaises(ValueError):
            ReplayDataset(self.output)

    def test_missing_manifest_or_receipt_rejects_uncommitted_output(self):
        self.save()
        receipt = self.output / "manifest.sha256"
        saved = receipt.read_bytes()
        receipt.unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_rollout(self.output)
        receipt.write_bytes(saved)
        (self.output / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_rollout(self.output)

    def test_manifest_and_payload_tampering_is_detected_before_deserialization(self):
        self.save()
        receipt = self.output / "manifest.sha256"
        saved = receipt.read_bytes()
        receipt.write_text("0" * 64 + "  manifest.json\n")
        with patch("world_models.rollouts.storage.torch.load") as deserialize:
            with self.assertRaisesRegex(ValueError, "receipt"):
                load_rollout(self.output)
            deserialize.assert_not_called()
        receipt.write_bytes(saved)
        path = self.output / "rollout.pt"
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
        with patch("world_models.rollouts.storage.torch.load") as deserialize:
            with self.assertRaisesRegex(ValueError, "hash"):
                load_rollout(self.output)
            deserialize.assert_not_called()

    def test_rehashed_semantically_invalid_payload_is_rejected(self):
        self.save()
        payload = torch.load(self.output / "rollout.pt", weights_only=True)
        payload["tensors"]["actions"][0, 0, 0] = 99
        self.replace_payload(payload)
        with self.assertRaises(ValueError):
            load_rollout(self.output)

    def test_rehashed_payload_envelope_and_marker_changes_are_rejected(self):
        self.save()
        original = torch.load(self.output / "rollout.pt", weights_only=True)
        for field, value in (("learning_eligible_for_world_model", True),
                             ("data_source", "real_environment"), ("unexpected", {})):
            with self.subTest(field=field):
                payload = deepcopy(original)
                payload[field] = value
                self.replace_payload(payload)
                with self.assertRaises(ValueError):
                    load_rollout(self.output)

    def test_rehashed_manifest_metadata_tensor_sizes_and_schema_are_checked(self):
        self.save()
        original = self.manifest()
        mutations = (
            lambda item: item["metadata"].update(actor_version="a" * 64),
            lambda item: item["metadata"].update(real_interactions=False),
            lambda item: item["tensor_specs"]["tensors/actions"].update(bytes=0),
            lambda item: item.update(schema_version="real-replay-v1"),
            lambda item: item.update(learning_eligible_for_world_model=0),
            lambda item: item.update(unexpected=True),
        )
        for mutate in mutations:
            manifest = deepcopy(original)
            mutate(manifest)
            self.recommit(manifest)
            with self.assertRaises(ValueError):
                load_rollout(self.output)

    def test_manifest_payload_paths_cannot_escape_the_store(self):
        self.save()
        original = self.manifest()
        for path in ("../rollout.pt", "/tmp/rollout.pt", "subdir/rollout.pt", "C:\\rollout.pt"):
            manifest = deepcopy(original)
            manifest["payload"]["path"] = path
            self.recommit(manifest)
            with self.assertRaisesRegex(ValueError, "path"):
                load_rollout(self.output)
        with self.assertRaises(ValueError):
            load_rollout(self.output / ".." / "imagined")

    def test_symbolic_link_members_and_root_are_rejected(self):
        self.save()
        # Exercise the symlink branch without requiring Windows symlink privilege.
        original = Path.is_symlink
        def symlink_payload(path):
            return path.name == "rollout.pt" or original(path)
        with patch.object(Path, "is_symlink", symlink_payload):
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                load_rollout(self.output)
        with patch.object(Path, "is_symlink", lambda path: path == self.output or original(path)):
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                load_rollout(self.output)

    def test_invalid_rollout_is_rejected_before_output_directory_is_created(self):
        rollout = make_rollout()
        rollout.rewards.fill_(float("nan"))
        with self.assertRaises(ValueError):
            save_rollout(rollout, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
