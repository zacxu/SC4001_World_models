"""Immutable controller state, coherent optimizer/RNG resume and corruption checks."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from world_models.cycle.checkpoints import load_cycle_checkpoint, save_cycle_checkpoint


class CycleCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "checkpoint"

    def state(self):
        return {"model": OrderedDict(weight=torch.arange(6, dtype=torch.float32).reshape(2, 3)),
                "optimizer": {"state": {0: {"step": torch.tensor(2.),
                                           "exp_avg": torch.tensor([0.2, 0.3])}},
                              "param_groups": [{"params": [0], "lr": 0.001}]},
                "rng": torch.Generator().manual_seed(7).get_state(),
                "controller": {"block": 3, "done": True, "stage": "local", "tuple": (None, 7)}}

    def metadata(self):
        return {"configuration_sha256": "a" * 64, "block_target": 125000,
                "stage": "local_cycle_complete", "nested": [True, 0.1, None]}

    def save(self):
        return save_cycle_checkpoint(self.root, state=self.state(), metadata=self.metadata())

    def rewrite_manifest(self, modify):
        path = self.root / "manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        modify(document)
        path.write_text(json.dumps(document), encoding="utf-8")
        (self.root / "manifest.sha256").write_text(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  manifest.json\n", encoding="ascii")

    def test_roundtrip_owns_tensors_metadata_integer_optimizer_keys_and_rng(self):
        state, metadata = self.state(), self.metadata()
        before_rng = torch.random.get_rng_state().clone()
        path = save_cycle_checkpoint(self.root, state=state, metadata=metadata)
        self.assertEqual(path, self.root / "manifest.json")
        state["model"]["weight"].zero_()
        metadata["nested"][0] = False
        restored, restored_metadata = load_cycle_checkpoint(self.root)
        self.assertEqual(restored["model"]["weight"].sum().item(), 15.)
        self.assertEqual(restored_metadata, self.metadata())
        self.assertIn(0, restored["optimizer"]["state"])
        self.assertIsInstance(restored["controller"]["tuple"], tuple)
        generator = torch.Generator()
        generator.set_state(restored["rng"])
        self.assertTrue(torch.equal(torch.rand(4, generator=generator),
                                   torch.rand(4, generator=torch.Generator().manual_seed(7))))
        restored["model"]["weight"].zero_()
        reloaded, _ = load_cycle_checkpoint(self.root)
        self.assertEqual(reloaded["model"]["weight"].sum().item(), 15.)
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))

    def test_adam_state_resumes_identical_next_parameter_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
        optimizer = torch.optim.Adam([parameter], lr=0.03)
        for _ in range(2):
            optimizer.zero_grad()
            parameter.square().sum().backward()
            optimizer.step()
        save_cycle_checkpoint(self.root, state={"weight": parameter.detach(),
                                               "optimizer": optimizer.state_dict()}, metadata={})
        restored, _ = load_cycle_checkpoint(self.root)
        resumed_parameter = torch.nn.Parameter(restored["weight"])
        resumed_optimizer = torch.optim.Adam([resumed_parameter], lr=1.)
        resumed_optimizer.load_state_dict(restored["optimizer"])
        for value, opt in ((parameter, optimizer), (resumed_parameter, resumed_optimizer)):
            opt.zero_grad()
            value.square().sum().backward()
            opt.step()
        self.assertTrue(torch.equal(parameter, resumed_parameter))

    def test_outer_inference_context_yields_ordinary_owned_tensors(self):
        state = self.state()
        with torch.inference_mode():
            save_cycle_checkpoint(self.root, state=state, metadata={})
            restored, _ = load_cycle_checkpoint(self.root)
        self.assertFalse(torch.is_inference(restored["model"]["weight"]))
        parameter = torch.ones_like(restored["model"]["weight"], requires_grad=True)
        (parameter * restored["model"]["weight"]).sum().backward()
        self.assertEqual(parameter.grad.sum().item(), 15.)

    def test_existing_directory_is_never_overwritten(self):
        self.save()
        digest = (self.root / "state.pt").read_bytes()
        with self.assertRaises(FileExistsError):
            self.save()
        self.assertEqual((self.root / "state.pt").read_bytes(), digest)

    def test_missing_commit_receipt_is_incomplete(self):
        self.save()
        (self.root / "manifest.sha256").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_cycle_checkpoint(self.root)

    def test_manifest_and_payload_corruption_are_checked_before_torch_load(self):
        self.save()
        (self.root / "manifest.json").write_text("{}", encoding="utf-8")
        with patch("world_models.cycle.checkpoints.torch.load") as loader:
            with self.assertRaisesRegex(ValueError, "receipt"):
                load_cycle_checkpoint(self.root)
            loader.assert_not_called()
        self.rewrite_manifest(lambda document: None)
        with self.assertRaisesRegex(ValueError, "manifest"):
            load_cycle_checkpoint(self.root)

    def test_payload_hash_length_and_manifest_path_mismatch_are_rejected(self):
        for kind in ("hash", "length", "path", "manifest_path"):
            with self.subTest(kind=kind):
                root = self.root / kind
                save_cycle_checkpoint(root, state=self.state(), metadata={})
                path = root / "manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if kind == "hash":
                    payload_path = root / "state.pt"
                    contents = bytearray(payload_path.read_bytes())
                    contents[-1] ^= 1
                    payload_path.write_bytes(contents)
                elif kind == "length":
                    manifest["payload"]["bytes"] += 1
                elif kind == "path":
                    manifest["payload"]["path"] = "../state.pt"
                else:
                    manifest["payload"]["path"] = "manifest.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                (root / "manifest.sha256").write_text(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  manifest.json\n", encoding="ascii")
                with patch("world_models.cycle.checkpoints.torch.load") as loader:
                    with self.assertRaises(ValueError):
                        load_cycle_checkpoint(root)
                    loader.assert_not_called()

    def test_metadata_and_tensor_specs_must_match_payload(self):
        self.save()
        self.rewrite_manifest(lambda doc: doc["metadata"].update(stage="other"))
        with self.assertRaisesRegex(ValueError, "metadata or tensor"):
            load_cycle_checkpoint(self.root)

    def test_tensor_specification_mismatch_is_rejected(self):
        self.save()
        self.rewrite_manifest(lambda doc: doc["tensor_specs"][0].update(shape=[999]))
        with self.assertRaisesRegex(ValueError, "metadata or tensor"):
            load_cycle_checkpoint(self.root)

    def test_incorrect_synthetic_markers_cannot_make_checkpoint_replay(self):
        self.save()
        self.rewrite_manifest(lambda doc: doc.update(learning_eligible_for_world_model=True))
        with self.assertRaisesRegex(ValueError, "markers"):
            load_cycle_checkpoint(self.root)

    def test_invalid_states_fail_before_creating_directory(self):
        inference_tensor = None
        with torch.inference_mode():
            inference_tensor = torch.ones(1)
        for invalid in (float("nan"), torch.tensor(float("inf")),
                        torch.ones(1, requires_grad=True), inference_tensor, object(), {True: 1}):
            with self.subTest(type=type(invalid)), self.assertRaises(ValueError):
                save_cycle_checkpoint(self.root, state={"value": invalid}, metadata={})
            self.assertFalse(self.root.exists())

    def test_invalid_metadata_and_empty_state_fail_before_publication(self):
        for state, metadata in (({}, {}), ({"a": 1}, {1: "bad"}),
                                ({"a": 1}, {"x": (1,)}), ({"a": 1}, {"x": float("nan")})):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                save_cycle_checkpoint(self.root, state=state, metadata=metadata)
            self.assertFalse(self.root.exists())

    def test_real_replay_ancestor_and_parent_traversal_are_rejected(self):
        replay = self.root / "real"
        replay.mkdir(parents=True)
        (replay / "manifest.json").write_text(json.dumps({"dataset_version": "real-replay-v1"}))
        for path in (replay / "checkpoint", self.root / "a" / ".." / "b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                save_cycle_checkpoint(path, state=self.state(), metadata={})

    def test_publication_failure_leaves_uncommitted_directory(self):
        with patch("world_models.cycle.checkpoints._write_json", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                self.save()
        self.assertTrue((self.root / "state.pt").is_file())
        self.assertFalse((self.root / "manifest.sha256").exists())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_cycle_checkpoint(self.root)

    def test_receipt_publication_failure_never_exposes_a_partial_commit(self):
        with patch("world_models.cycle.checkpoints.os.replace", side_effect=OSError("interrupted receipt publish")):
            with self.assertRaisesRegex(OSError, "receipt publish"):
                self.save()
        self.assertTrue((self.root / "manifest.json").is_file())
        self.assertTrue((self.root / "manifest.sha256.tmp").is_file())
        self.assertFalse((self.root / "manifest.sha256").exists())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_cycle_checkpoint(self.root)


if __name__ == "__main__":
    unittest.main()
