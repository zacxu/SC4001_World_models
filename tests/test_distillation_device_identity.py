"""Historical device labels and committed initial weights at the P4.2 handoff.

CUDA-labelled fixtures are synthetic CPU tensors with independently computed
legacy identities. No CUDA allocation or research training is required.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from tests import test_distillation_handoff as handoff_fixture
from tests.test_cycle_controller import controller_settings
from world_models.cycle.checkpoints import load_cycle_checkpoint, save_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.distillation.handoff import load_completed_teacher
from world_models.distillation.runner import _paired_students
from world_models.distillation.versions import module_version_at_device
from world_models.imagination import versions as legacy_versions
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.policies.initialization import LocalActorCritic
from world_models.teacher.runner import _policies


def legacy_hash_at_device(module, device):
    """Independent oracle: retain the original hashing routine and byte order."""
    original_json = legacy_versions._json

    def recorded_metadata(value):
        if isinstance(value, dict) and set(value) == {"kind", "name", "dtype", "shape", "device"}:
            value = {**value, "device": device}
        return original_json(value)

    with patch.object(legacy_versions, "_json", side_effect=recorded_metadata):
        return legacy_versions.module_version(module)


@dataclass
class _Configuration:
    width: int = 3
    options: tuple = ("recorded", 7)


class _ConfiguredNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Configuration()
        self.projection = nn.Linear(3, 2)
        self.projection.config = _Configuration(width=2)
        self.register_buffer("counter", torch.tensor(4, dtype=torch.int64))
        self.register_buffer("temporary", torch.tensor([1, 2], dtype=torch.bfloat16), persistent=False)


class HistoricalDeviceVersionTests(unittest.TestCase):
    def test_cpu_identity_preserves_the_original_hash_and_all_module_state(self):
        network = _ConfiguredNetwork()
        before = deepcopy(network.state_dict())
        rng = torch.get_rng_state().clone()
        self.assertEqual(module_version_at_device(network, "cpu"), legacy_versions.module_version(network))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, network.state_dict()[key]) for key, value in before.items()))
        self.assertEqual(network.temporary.device.type, "cpu")

    def test_recorded_cuda_identity_matches_independent_legacy_oracle(self):
        network = _ConfiguredNetwork()
        with patch("torch.cuda.is_available", side_effect=AssertionError("hash attempted CUDA discovery")):
            recorded = module_version_at_device(network, "cuda:0")
        self.assertEqual(recorded, legacy_hash_at_device(network, "cuda:0"))
        self.assertNotEqual(recorded, legacy_versions.module_version(network))
        self.assertNotEqual(recorded, module_version_at_device(network, "cuda:3"))
        with torch.no_grad():
            network.projection.weight[0, 0].add_(0.25)
        self.assertNotEqual(recorded, module_version_at_device(network, "cuda:0"))
        self.assertEqual(module_version_at_device(network, "cuda:0"), legacy_hash_at_device(network, "cuda:0"))

    def test_unsupported_or_noncanonical_origin_labels_are_rejected(self):
        network = _ConfiguredNetwork()
        for label in (None, "", "cuda", "cuda:-1", "cuda:01", "meta", "mps", "cpu:0"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                module_version_at_device(network, label)

    def test_paired_student_hashes_describe_the_selected_device_after_transfer(self):
        settings = controller_settings()
        settings["variant"] = {"name": "D", "teacher_enabled": True, "distillation_enabled": True}
        context = SimpleNamespace(settings=settings, master_seed=0,
            model=SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda:0"))])))

        def simulated_move(bundle, device):
            self.assertEqual(str(device), "cuda:0")
            for network in (bundle.actor, bundle.critic):
                network._fixture_device = "cuda:0"
            return bundle

        def simulated_native_version(network):
            return legacy_hash_at_device(network, getattr(network, "_fixture_device", "cpu"))

        with patch.object(LocalActorCritic, "to", new=simulated_move), \
                patch("world_models.distillation.runner.module_version", side_effect=simulated_native_version):
            student, expected = _paired_students(context)
        for name in ("actor", "critic"):
            network = getattr(student, name)
            self.assertEqual(expected[name], legacy_hash_at_device(network, "cuda:0"))
            self.assertNotEqual(expected[name], legacy_hash_at_device(network, "cpu"))


class CompletedTeacherDeviceHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        handoff_fixture.DistillationHandoffTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        handoff_fixture.DistillationHandoffTests.tearDownClass.__func__(cls)

    setUp = handoff_fixture.DistillationHandoffTests.setUp
    prepare = handoff_fixture.DistillationHandoffTests.prepare
    completed = handoff_fixture.DistillationHandoffTests.completed
    rewrite_json = handoff_fixture.DistillationHandoffTests.rewrite_json

    def gpu_labelled_teacher(self, *, corrupt_final_weight=False):
        worker = self.completed()
        manifest = deepcopy(worker.manifest)
        initial_checkpoint = next((self.directory / "checkpoints").glob("state_00000001_*"))
        initial, initial_metadata = load_cycle_checkpoint(initial_checkpoint)
        original_networks = (worker.model, worker.teacher, worker.critic)
        initial_networks = tuple(deepcopy(network).cpu() for network in original_networks)
        for network, name in zip(initial_networks, ("model", "teacher", "critic")):
            network.load_state_dict(initial[name])
        initial_versions = tuple(legacy_hash_at_device(network, "cuda:0") for network in initial_networks)
        final_versions = tuple(legacy_hash_at_device(network, "cuda:0") for network in original_networks)
        for name, version in zip(("initial_model_version", "initial_teacher_version", "initial_critic_version"),
                                 initial_versions):
            manifest[name] = version
        manifest["runtime"]["logical_device"] = "cuda:0"
        manifest["rollout_stream_identity"]["device"] = "cuda:0"
        self.rewrite_json("run_manifest.json", manifest)
        run_identity = canonical_hash(manifest)

        def relabel(saved, metadata, versions):
            saved["learner"]["network_versions"] = versions
            saved["streams"]["identity"]["device"] = "cuda:0"
            metadata["initial_model_version"] = initial_versions[0]
            metadata["run_identity"] = run_identity
            if saved["last_report"] is not None:
                for key, value in zip(("world_model_version", "actor_version_after", "critic_version_after"), versions):
                    saved["last_report"][key] = value

        relabel(initial, initial_metadata, initial_versions)
        replacement_initial = initial_checkpoint.with_name("state_00000001_abbacafe")
        save_cycle_checkpoint(replacement_initial, state=initial, metadata=initial_metadata)
        initial_checkpoint.joinpath("manifest.sha256").unlink()

        final, metadata = load_cycle_checkpoint(worker.last_checkpoint)
        relabel(final, metadata, final_versions)
        if corrupt_final_weight:
            key = next(key for key, value in final["teacher"].items() if value.is_floating_point())
            final["teacher"][key].add_(0.25)
        replacement_final = worker.last_checkpoint.with_name(
            f"state_{final['controller']['sequence']:08d}_feedcafe")
        save_cycle_checkpoint(replacement_final, state=final, metadata=metadata)
        worker.last_checkpoint.joinpath("manifest.sha256").unlink()
        handoff = json.loads((self.directory / "frozen_teacher.json").read_text())
        handoff.update({"run_identity": run_identity,
            "checkpoint": replacement_final.relative_to(self.directory).as_posix(),
            "checkpoint_manifest_sha256": file_sha256(replacement_final / "manifest.json"),
            "model_version": final_versions[0], "teacher_version": final_versions[1],
            "critic_version": final_versions[2]})
        self.rewrite_json("frozen_teacher.json", handoff)
        return final_versions

    def test_cuda_recorded_teacher_imports_on_cpu_without_modifying_original_receipts(self):
        versions = self.gpu_labelled_teacher()
        before = {path.relative_to(self.directory).as_posix(): file_sha256(path)
                  for path in self.directory.rglob("*") if path.is_file()}
        with patch("torch.cuda.is_available", side_effect=AssertionError("CPU import attempted CUDA discovery")):
            context = load_completed_teacher(self.directory, device="cpu")
        self.assertEqual(context.provenance["verified_origin_device"], "cuda:0")
        self.assertEqual(tuple(context.provenance[key] for key in
                               ("model_version", "teacher_version", "critic_version")), versions)
        for network, version in zip((context.model, context.teacher, context.critic), versions):
            self.assertEqual(next(network.parameters()).device.type, "cpu")
            self.assertEqual(module_version_at_device(network, "cuda:0"), version)
        self.assertEqual(before, {path.relative_to(self.directory).as_posix(): file_sha256(path)
                                 for path in self.directory.rglob("*") if path.is_file()})

    def test_device_relabeling_does_not_accept_changed_teacher_parameters(self):
        self.gpu_labelled_teacher(corrupt_final_weight=True)
        with self.assertRaisesRegex(ValueError, "handoff differs|bound weights"):
            load_completed_teacher(self.directory, device="cpu")

    def test_committed_initial_teacher_overrides_runtime_initializer_roundoff(self):
        self.completed()

        def altered_initializer(*args, **kwargs):
            teacher, critic = _policies(*args, **kwargs)
            with torch.no_grad():
                next(teacher.parameters()).add_(1e-4)
                next(critic.parameters()).add_(1e-4)
            return teacher, critic

        with patch("world_models.distillation.handoff._policies", side_effect=altered_initializer):
            context = load_completed_teacher(self.directory, device="cpu")
        self.assertEqual(context.provenance["verified_origin_device"], "cpu")
        self.assertEqual(legacy_versions.module_version(context.teacher), context.handoff["teacher_version"])


if __name__ == "__main__":
    unittest.main()
