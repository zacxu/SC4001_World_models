"""P4.5 independent deployment isolation and remaining cross-stage invariants."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from world_models.cycle.checkpoints import save_cycle_checkpoint
from world_models.distillation.versions import module_version_at_device
from world_models.imagination.versions import module_version
from world_models.policies.actor import LocalActor
from world_models.policies.config import PolicyConfig
from world_models_acceptance.deployment import (
    FORBIDDEN_IMPORTS, _actor_version, load_local_actor_export, probe_actor_export,
)


class Phase45DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = PolicyConfig(encoder_channels=(8, 8, 8, 8), embedding_size=16,
                              hidden_size=16, world_tokens=6, world_width=16,
                              critic_hidden_sizes=(16, 16))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4501)
            self.actor = LocalActor(config)

    def export(self, name="actor_export", *, origin="cpu", metadata=None, state=None):
        values = json.loads(json.dumps(asdict(self.actor.config)))
        document = {"schema_version": "p35-local-actor-v1", "task_family": "relay", "variant": "D",
                    "run_seed": 0, "real_joint_transitions": 50000,
                    "actual_initial_replay_interactions": 50000,
                    "untrained_zero_interaction_baseline": False,
                    "actor_version": module_version_at_device(self.actor, origin),
                    "actor_config": values, "world_model_version": "a" * 64,
                    "model_updates": 4500, "local_ppo_cycles": 20,
                    "actor_optimizer_steps": 320, "critic_optimizer_steps": 320,
                    "evaluation_split": "validation", "on_registered_evaluation_grid": False,
                    "evaluation_executed": False, "phase3_acceptance": False}
        if metadata:
            document.update(metadata)
        directory = self.root / name
        save_cycle_checkpoint(directory, state=state or {"actor": self.actor.state_dict()}, metadata=document)
        return directory

    def test_fresh_process_executes_local_actor_with_learning_imports_blocked(self):
        directory = self.export(origin="cuda:0")
        report = probe_actor_export(directory, output_dir=self.root / "probe", source_device="cuda:0")
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["passed"] and report["isolated_process"] and report["local_only"])
        self.assertEqual(report["blocked_import_attempts"], list(FORBIDDEN_IMPORTS))
        self.assertEqual(report["forbidden_modules_loaded"], [])
        self.assertEqual(report["observation_steps"], 3)
        self.assertTrue(report["weights_unchanged"])
        self.assertEqual(report["source_device"], "cuda:0")
        self.assertEqual(report["actor_version"], module_version_at_device(self.actor, "cuda:0"))
        self.assertFalse(report["training_executed"])
        self.assertFalse(report["research_evaluation_executed"])
        self.assertFalse(report["native_environment_opened"])
        path = self.root / "probe" / "deployment_probe.json"
        self.assertEqual((path.with_suffix(".sha256")).read_text().strip(),
                         f"{hashlib.sha256(path.read_bytes()).hexdigest()}  deployment_probe.json")
        actor, _ = load_local_actor_export(directory)
        for expected, actual in zip(self.actor.parameters(), actor.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(actor.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in actor.parameters()))

    def test_origin_device_hash_matches_registered_algorithm(self):
        self.assertEqual(_actor_version(self.actor, "cpu"), module_version(self.actor))
        for origin in ("cpu", "cuda:0", "cuda:3", "cuda:27"):
            self.assertEqual(_actor_version(self.actor, origin), module_version_at_device(self.actor, origin))
        directory = self.export(origin="cuda:27")
        with self.assertRaisesRegex(ValueError, "source_device"):
            load_local_actor_export(directory)
        _, provenance = load_local_actor_export(directory, source_device="cuda:27")
        self.assertEqual(provenance["verified_origin_device"], "cuda:27")

    def test_probe_rejects_full_controller_checkpoint_before_tensor_loading(self):
        directory = self.export(state={"actor": self.actor.state_dict(), "teacher": {"weight": torch.ones(1)}})
        with patch("world_models.cycle.checkpoints.load_cycle_checkpoint", side_effect=AssertionError("deserialized")):
            with self.assertRaisesRegex(ValueError, "actor-only"):
                load_local_actor_export(directory)
        with self.assertRaisesRegex(ValueError, "actor-only"):
            probe_actor_export(directory)

    def test_manifest_and_payload_tampering_rejected(self):
        directory = self.export()
        manifest = directory / "manifest.json"
        content = manifest.read_bytes()
        manifest.write_bytes(content + b" ")
        with self.assertRaisesRegex(ValueError, "receipt mismatch"):
            load_local_actor_export(directory)
        manifest.write_bytes(content)
        payload = directory / "state.pt"
        payload.write_bytes(payload.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            load_local_actor_export(directory)

    def test_export_metadata_and_actor_versions_are_checked(self):
        cases = ({"teacher_optimizer_steps": 400}, {"actor_version": "b" * 64},
                 {"actor_optimizer_steps": 0}, {"critic_optimizer_steps": 319},
                 {"evaluation_executed": True}, {"model_updates": True})
        for index, metadata in enumerate(cases):
            with self.subTest(metadata=metadata):
                directory = self.export(f"invalid_{index}", metadata=metadata)
                with self.assertRaises(ValueError):
                    load_local_actor_export(directory)
        for index, origin in enumerate(("cuda", "cpu:0", "bogus")):
            directory = self.export(f"origin_{index}")
            with self.assertRaises((ValueError, RuntimeError)):
                load_local_actor_export(directory, source_device=origin)

    def test_deployment_helper_is_importable_without_torch(self):
        code = "import sys; import world_models_acceptance.deployment; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_probe_failure_never_publishes_a_pass_receipt(self):
        directory = self.export()
        (directory / "manifest.sha256").unlink()
        output = self.root / "must_not_exist"
        with self.assertRaisesRegex(ValueError, "probe failed"):
            probe_actor_export(directory, output_dir=output)
        self.assertFalse(output.exists())


class Phase45CrossBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_privileged_sidecars_do_not_change_local_student_replay(self):
        from tests.test_distillation_learning import DistillationTests
        from world_models.distillation.learning import replay_student
        fixture = DistillationTests("test_prefix_reconstructed_current_weights_anchor_consumed_once")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        buffer = fixture.buffer(clocks=(3,), horizon=2)
        baseline = replay_student(fixture.student, buffer, [0])
        metadata = deepcopy(buffer.metadata)
        metadata["privileged_sidecar"] = {"shared_tokens": [99999], "teacher_memory": [-99],
                                          "reward": 12, "other_player_id": 1, "global_clock": 999}
        modified = replace(buffer, metadata=metadata)
        actual = replay_student(fixture.student, modified, [0])
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        metadata["privileged_sidecar"] = {"shared_tokens": [-999], "teacher_memory": [41],
                                          "reward": -20, "other_player_id": 0, "global_clock": 0}
        second = replay_student(fixture.student, replace(buffer, metadata=metadata), [0])
        torch.testing.assert_close(second, baseline, rtol=0, atol=0)

    def test_zero_kl_and_detached_teacher_targets_have_no_teacher_gradient(self):
        from world_models.distillation.learning import forward_kl
        generator = torch.Generator().manual_seed(4502)
        teacher_logits = torch.randn((2, 1, 2, 12), generator=generator, requires_grad=True)
        student_logits = teacher_logits.detach().clone().requires_grad_()
        targets = teacher_logits.softmax(-1).detach()
        valid = torch.ones((2, 1), dtype=torch.bool)
        loss, _ = forward_kl(student_logits, targets, valid)
        self.assertLess(abs(float(loss.detach())), 1e-6)
        loss.backward()
        self.assertIsNone(teacher_logits.grad)
        self.assertLess(float(student_logits.grad.abs().max()), 1e-7)
        changed = student_logits.detach().clone()
        changed[..., 0].add_(1.)
        changed.requires_grad_()
        positive, _ = forward_kl(changed, targets, valid)
        self.assertGreater(float(positive.detach()), 0.)
        positive.backward()
        self.assertGreater(float(changed.grad.abs().sum()), 0.)
        self.assertIsNone(teacher_logits.grad)
        with self.assertRaisesRegex(ValueError, "detached"):
            forward_kl(changed, teacher_logits.softmax(-1), valid)


if __name__ == "__main__":
    unittest.main()
