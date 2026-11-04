"""P2.4 continuation checks on a tiny Linear model, never research training."""
from copy import deepcopy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from world_models.models.world_model import SamplingStreams
from world_models.training.checkpoints import SCHEMA_VERSION, load_checkpoint, read_checkpoint, save_checkpoint


class ToySampler:
    def __init__(self, seed=27):
        self.generator = torch.Generator().manual_seed(seed)
        self.draws = 0

    def sample(self):
        self.draws += 1
        return torch.randn(3, 2, generator=self.generator)

    def state_dict(self):
        return {"rng": self.generator.get_state(), "draws": self.draws}

    def load_state_dict(self, state):
        self.generator.set_state(state["rng"])
        self.draws = state["draws"]


def fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    return model, optimizer, SamplingStreams.from_seed(8), ToySampler()


def metadata():
    return {"task": "relay", "seed": 0, "config": {"batch": 32},
            "replay_manifests": ["manifest-sha"], "normalization_identity": "stats-sha",
            "source_hashes": {"trainer.py": "source-sha"},
            "precision": {"parameters": "float32", "autocast": "bfloat16"},
            "device": "cpu"}


def update(model, optimizer, streams, sampler):
    inputs = sampler.sample()
    stochastic = torch.rand((), generator=streams.prior) + torch.rand((), generator=streams.posterior)
    stochastic += torch.rand((), generator=streams.outcome)
    target = random.random() + float(np.random.random()) + torch.rand(()) + stochastic
    loss = (model(inputs)-target).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "update_000001.pt"
        self.model, self.optimizer, self.streams, self.sampler = fixture()
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_rng = torch.get_rng_state()
        self.addCleanup(random.setstate, self.python_rng)
        self.addCleanup(np.random.set_state, self.numpy_rng)
        self.addCleanup(torch.set_rng_state, self.torch_rng)

    def save(self, **changes):
        arguments = {"model": self.model, "optimizer": self.optimizer, "streams": self.streams,
                     "sampler": self.sampler, "update": 1, "metadata": metadata()}
        arguments.update(changes)
        return save_checkpoint(self.path, **arguments)

    def load(self, **changes):
        arguments = {"model": self.model, "optimizer": self.optimizer, "streams": self.streams,
                     "sampler": self.sampler, "expected_metadata": metadata()}
        arguments.update(changes)
        return load_checkpoint(self.path, **arguments)

    def rewrite(self, mutate):
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        mutate(payload)
        torch.save(payload, self.path)
        self.path.with_suffix(".sha256").write_text(
            f"{hashlib.sha256(self.path.read_bytes()).hexdigest()}  {self.path.name}\n", encoding="ascii")

    def test_optimizer_and_all_rng_resume_matches_uninterrupted_next_update(self):
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        update(self.model, self.optimizer, self.streams, self.sampler)
        receipt = self.save()
        expected_loss = update(self.model, self.optimizer, self.streams, self.sampler)
        expected_model = deepcopy(self.model.state_dict())
        expected_optimizer = deepcopy(self.optimizer.state_dict())
        resumed, optimizer, streams, sampler = fixture()
        payload = load_checkpoint(self.path, model=resumed, optimizer=optimizer, streams=streams,
                                  sampler=sampler, expected_metadata=metadata())
        actual_loss = update(resumed, optimizer, streams, sampler)
        self.assertEqual(expected_loss, actual_loss)
        for key, value in expected_model.items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[key]))
        for key, state in expected_optimizer["state"].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, optimizer.state_dict()["state"][key][name]))
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["update"], 1)
        self.assertEqual(sampler.draws, 2)
        self.assertEqual(receipt["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_snapshot_cpu_state_and_exact_nested_provenance_subset(self):
        self.save()
        payload = self.load(expected_metadata={"config": {"batch": 32}}, restore_rng=False)
        self.assertTrue(all(value.device.type == "cpu" for value in payload["model"].values()))
        for changed in ({"task": "cover"}, {"seed": 1}, {"config": {}},
                        {"normalization_identity": "wrong"}, {"replay_manifests": ["wrong"]},
                        {"source_hashes": {}}, {"precision": {"parameters": "float32"}},
                        {"device": "cuda:0"}, {"missing": None}, {"seed": False}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "metadata mismatch"):
                self.load(expected_metadata=changed)

    def test_restore_rng_false_preserves_global_and_unsupplied_component_states(self):
        self.save()
        random.random()
        np.random.random()
        torch.rand(1)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        load_checkpoint(self.path, model=self.model, restore_rng=False)
        self.assertEqual(random.getstate(), before[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

    def test_public_read_validates_payload_without_model_or_rng_restoration(self):
        self.save()
        random.random()
        np.random.random()
        torch.rand(1)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        payload = read_checkpoint(self.path)
        self.assertEqual(payload["metadata"], metadata())
        self.assertEqual(payload["update"], 1)
        self.assertTrue(all(value.device.type == "cpu" for value in payload["model"].values()))
        self.assertEqual(random.getstate(), before[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.rewrite(lambda item: item.update(update=-1))
        with self.assertRaisesRegex(ValueError, "update"):
            read_checkpoint(self.path)

    def test_model_weights_and_optimizer_moments_are_restored(self):
        update(self.model, self.optimizer, self.streams, self.sampler)
        self.save()
        original = deepcopy(self.model.state_dict())
        with torch.no_grad():
            self.model.weight.fill_(100.)
        self.optimizer.param_groups[0]["lr"] = 10.
        self.load()
        self.assertTrue(torch.equal(self.model.weight, original["weight"]))
        self.assertEqual(self.optimizer.param_groups[0]["lr"], .01)
        self.assertEqual(float(self.optimizer.state[self.model.weight]["step"]), 1.)

    def test_hash_tampering_and_truncation_fail_before_model_mutation(self):
        self.save()
        original = self.path.read_bytes()
        initial = self.model.weight.detach().clone()
        for content in (original + b"tamper", original[:20]):
            self.path.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.load()
            self.assertTrue(torch.equal(self.model.weight, initial))

    def test_missing_or_malformed_receipts_and_orphan_receipts_fail(self):
        self.save()
        receipt = self.path.with_suffix(".sha256")
        receipt.unlink()
        with self.assertRaisesRegex(ValueError, "orphan"):
            self.load()
        receipt.write_text("broken receipt", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "malformed"):
            self.load()
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "orphan"):
            self.load()
        with self.assertRaisesRegex(FileExistsError, "orphan"):
            self.save()

    def test_immutable_files_and_stale_writer_artifacts(self):
        self.save()
        before = self.path.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "immutable"):
            self.save()
        self.assertEqual(self.path.read_bytes(), before)
        other = self.path.with_name("update_000002.pt")
        lock = other.with_name(other.name+".lock")
        lock.touch()
        with self.assertRaisesRegex(FileExistsError, "stale"):
            save_checkpoint(other, model=self.model, optimizer=self.optimizer, streams=self.streams,
                            sampler=self.sampler, update=2, metadata=metadata())
        with self.assertRaisesRegex(ValueError, "writer lock"):
            load_checkpoint(other, model=self.model)
        lock.unlink()
        other.with_name(other.name+".tmp-crashed").touch()
        with self.assertRaisesRegex(FileExistsError, "stale"):
            save_checkpoint(other, model=self.model, optimizer=self.optimizer, streams=self.streams,
                            sampler=self.sampler, update=2, metadata=metadata())

    def test_unsupported_schema_and_invalid_update_and_precision_are_rejected(self):
        self.save()
        original = self.path.read_bytes()
        mutations = [lambda item: item.update(schema_version="future"),
                     lambda item: item.update(update=-1), lambda item: item.update(update=True),
                     lambda item: item["model"].update(weight=item["model"]["weight"].double()),
                     lambda item: item["streams"].pop("outcome")]
        for mutate in mutations:
            self.path.write_bytes(original)
            self.rewrite(mutate)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.load()

    def test_boundary_and_float32_optimizer_contracts(self):
        self.model(torch.ones(1, 2)).sum().backward()
        with self.assertRaisesRegex(ValueError, "optimizer boundary"):
            self.save()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.state[self.model.weight]["bad_moment"] = torch.ones_like(self.model.weight, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "optimizer floating tensors"):
            self.save()
        self.optimizer.state.clear()
        with self.assertRaises(ValueError):
            self.save(update=True)
        with self.assertRaises(ValueError):
            self.save(metadata={"unsafe": object()})

    def test_failed_publication_leaves_an_explicit_orphan_not_a_loadable_checkpoint(self):
        real_replace = __import__("os").replace
        def fail_receipt(source, destination):
            if Path(destination).suffix == ".sha256":
                raise OSError("simulated publication interruption")
            return real_replace(source, destination)
        with patch("world_models.training.checkpoints.os.replace", side_effect=fail_receipt):
            with self.assertRaises(OSError):
                self.save()
        self.assertTrue(self.path.is_file())
        self.assertFalse(self.path.with_suffix(".sha256").exists())
        self.assertFalse(list(self.path.parent.glob("*.tmp-*")))
        self.assertFalse(list(self.path.parent.glob("*.lock")))
        with self.assertRaisesRegex(ValueError, "orphan"):
            self.load()

    def test_restricted_loader_refuses_arbitrary_pickled_objects(self):
        # An unsupported global remains rejected even with an authentic hash.
        torch.save({"arbitrary": nn.Linear(1, 1)}, self.path)
        self.path.with_suffix(".sha256").write_text(
            f"{hashlib.sha256(self.path.read_bytes()).hexdigest()}  {self.path.name}\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "restricted weights-only"):
            self.load()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA continuation requires a GPU host")
    def test_cuda_checkpoint_uses_cpu_serialization_and_restores_cuda_state(self):
        model = nn.Linear(2, 1).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        streams = SamplingStreams.from_seed(8, device="cuda")
        sampler = ToySampler()
        model(torch.ones(1, 2, device="cuda")).square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        save_checkpoint(self.path, model=model, optimizer=optimizer, streams=streams,
                        sampler=sampler, update=1, metadata={"device": "cuda:0"})
        expected = torch.rand(3, device="cuda")
        payload = load_checkpoint(self.path, model=model, optimizer=optimizer, streams=streams, sampler=sampler)
        self.assertTrue(torch.equal(expected, torch.rand(3, device="cuda")))
        self.assertTrue(all(value.device.type == "cpu" for value in payload["model"].values()))
        self.assertEqual(optimizer.state[model.weight]["exp_avg"].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
