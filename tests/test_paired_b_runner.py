"""Matched B mechanics using temporary six-transition CPU fixtures only."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from tests import test_teacher_runner as teacher_fixture
from tests.test_imagined_rollout_generator import assert_tree_equal
from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.imagination import module_version
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.policies.initialization import build_actor_critic
from world_models.training.runner import run_lock
from world_models_paired.b_inputs import load_b_protocol
from world_models_paired.runner import BRunner


ROOT = Path(__file__).resolve().parents[1]


class PairedBRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        teacher_fixture.TeacherRunnerTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        teacher_fixture.TeacherRunnerTests.tearDownClass.__func__(cls)

    def setUp(self):
        teacher_fixture.TeacherRunnerTests.setUp(self)
        self.directory = Path(self.temp.name) / "matched_B"
        self.protocol = load_b_protocol()
        self.protocol.update(initial_transitions=6, initial_model_updates=1,
            starts=2, horizon=1, local_cycles=2, epochs=1, minibatches=1,
            world_model_compute_precision="float32")
        self.enterContext(patch("world_models_paired.runner.load_protocol", return_value=self.protocol))
        self.enterContext(patch("world_models_paired.b_inputs.load_b_protocol", return_value=self.protocol))
        self.enterContext(patch("world_models_paired.runner.source_identity", return_value={"fixture": "b" * 64}))

    def prepare(self, directory=None, **overrides):
        kwargs = {"initial_selection": self.selection["selection_path"], "master_seed": 0,
            "data_root": self.root, "statistics_root": self.root, "config_root": ROOT / "configs",
            "device": "cpu", "precision": "float32", "cohort_binding": {"synthetic": True}}
        kwargs.update(overrides)
        return BRunner.prepare(directory or self.directory, **kwargs)

    def resume(self):
        return BRunner.resume(self.directory, device="cpu")

    def test_prepare_paired_initialization_and_fresh_adams_no_training_or_global_rng(self):
        rng = torch.get_rng_state().clone()
        with patch("world_models_paired.runner.RecurrentMAPPO.update", side_effect=AssertionError("prepare trained")):
            worker = self.prepare()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        settings = deepcopy(worker.settings)
        settings["variant"] = {"name": "D", "teacher_enabled": True, "distillation_enabled": True}
        paired = build_actor_critic(settings, master_seed=0, variant="D")
        assert_tree_equal(self, paired.actor.state_dict(), worker.actor.state_dict())
        assert_tree_equal(self, paired.critic.state_dict(), worker.critic.state_dict())
        self.assertEqual(worker.pairing()["input_binding_sha256"],
                         canonical_hash(worker.context.binding["shared_initial_binding"]))
        self.assertFalse(worker.learner.actor_optimizer.state)
        self.assertFalse(worker.learner.critic_optimizer.state)
        self.assertEqual(worker.status()["local_cycles_completed"], 0)
        self.assertEqual(worker.manifest["cohort_binding"], {"synthetic": True})
        self.assertFalse(hasattr(worker, "teacher"))
        self.assertFalse(hasattr(worker, "model_optimizer"))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in worker.model.parameters()))
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_real_local_ppo_split_resume_is_exact_for_weights_adams_and_all_rng(self):
        reference = self.prepare(Path(self.temp.name) / "reference")
        reference.run()
        worker = self.prepare()
        original = worker.pairing()
        world = module_version(worker.model)
        self.assertEqual(worker.run(max_cycles=1)["status"], "stopped")
        state = {key: deepcopy(getattr(worker, key).state_dict()) for key in ("learner", "streams", "anchor_sampler")}
        restored = self.resume()
        for key, value in state.items():
            assert_tree_equal(self, value, getattr(restored, key).state_dict())
        result = restored.run()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["local_cycles_completed"], 2)
        self.assertEqual((result["actor_optimizer_steps"], result["critic_optimizer_steps"]), (2, 2))
        self.assertEqual((result["new_environment_interactions"], result["new_model_updates"]), (0, 0))
        self.assertEqual(restored.pairing(), original)
        self.assertEqual(module_version(restored.model), world)
        self.assertEqual(module_version(restored.actor), module_version(reference.actor))
        self.assertEqual(module_version(restored.critic), module_version(reference.critic))
        for key in ("learner", "streams", "anchor_sampler"):
            assert_tree_equal(self, getattr(restored, key).state_dict(), getattr(reference, key).state_dict())

    def test_stop_request_before_and_during_cycle_preserves_whole_boundary(self):
        worker = self.prepare()
        (self.directory / "stop.request").write_text("stop")
        self.assertEqual(worker.run()["cycles_executed"], 0)
        self.assertFalse((self.directory / "stop.request").exists())
        update = worker.learner.update
        def stop(*args, **kwargs):
            report = update(*args, **kwargs)
            worker.request_stop()
            return report
        with patch.object(worker.learner, "update", side_effect=stop):
            self.assertEqual(worker.run()["cycles_executed"], 1)
        self.assertEqual(self.resume().run()["status"], "complete")

    def test_failed_checkpoint_resume_rolls_back_uncommitted_optimizer_and_rng(self):
        worker = self.prepare()
        initial = {key: deepcopy(getattr(worker, key).state_dict()) for key in ("learner", "streams", "anchor_sampler")}
        with patch("world_models_paired.runner.save_cycle_checkpoint", side_effect=OSError("fixture disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                worker.run(max_cycles=1)
        with self.assertRaisesRegex(RuntimeError, "fresh instance"):
            worker.run()
        restored = self.resume()
        self.assertEqual(restored.learner.cycles_completed, 0)
        for key, value in initial.items():
            assert_tree_equal(self, value, getattr(restored, key).state_dict())
        attempts = [json.loads(x) for x in (self.directory / "attempts.jsonl").read_text().splitlines()]
        self.assertIn("unknown", attempts[-1]["partial_physical_optimizer_work"])

    def test_lock_and_stale_worker_cannot_advance_same_saved_branch(self):
        worker = self.prepare()
        stale = self.resume()
        with run_lock(self.directory):
            with self.assertRaisesRegex(RuntimeError, "another worker"):
                self.resume()
            with self.assertRaisesRegex(RuntimeError, "another worker"):
                worker.run()
        worker.run(max_cycles=1)
        with self.assertRaisesRegex(RuntimeError, "advanced"):
            stale.run()

    def test_source_runtime_frozen_input_and_world_mutation_fail_closed(self):
        worker = self.prepare()
        with patch("world_models_paired.runner.source_identity", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                worker.run()
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.resume()
        with patch("world_models_paired.runner._runtime", return_value={"changed": "runtime"}):
            with self.assertRaisesRegex(ValueError, "logical device.*runtime"):
                self.resume()
        original = deepcopy(self.selection)
        self.selected.return_value = {**original, "checkpoint_sha256": "f" * 64}
        with self.assertRaises(ValueError):
            worker.run()
        self.selected.return_value = original
        with torch.no_grad():
            next(worker.model.parameters()).view(-1)[0].add_(1)
        with self.assertRaisesRegex(ValueError, "frozen shared world model"):
            worker.run()

    def test_ahead_status_uncommitted_debris_and_corrupt_payload(self):
        worker = self.prepare()
        (self.directory / "status.json").write_text('{"status":"complete","local_cycles_completed":999}')
        orphan = self.directory / "checkpoints/state_99999999_deadbeef"
        orphan.mkdir()
        (orphan / "state.pt").write_text("uncommitted")
        restored = self.resume()
        self.assertEqual(restored.learner.cycles_completed, 0)
        restored.run(max_cycles=1)
        with (restored.last_checkpoint / "state.pt").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "payload length mismatch"):
            self.resume()

    def test_final_export_is_local_only_compatible_and_completion_idempotent(self):
        from scripts.phase36_support.evaluation import load_actor_export
        worker = self.prepare()
        worker.run()
        report = json.loads((self.directory / "final_student.json").read_text())
        actor, exported = load_actor_export(self.directory / report["actor_export"], device="cpu")
        self.assertEqual(module_version(actor), module_version(worker.actor))
        self.assertEqual(exported["metadata"]["variant"], "B")
        self.assertEqual(exported["metadata"]["evaluation_split"], "test")
        initial_files = {str(p): p.read_bytes() for p in self.directory.rglob("*")
                         if p.is_file() and p.name not in ("status.json",)}
        restored = self.resume()
        with patch.object(restored.learner, "update", side_effect=AssertionError("repeated optimization")):
            self.assertEqual(restored.run()["cycles_executed"], 0)
        self.assertEqual(initial_files, {str(p): p.read_bytes() for p in self.directory.rglob("*")
                                      if p.is_file() and p.name not in ("status.json",)})

    def test_precision_protocol_and_manifest_receipt_guards(self):
        with self.assertRaisesRegex(ValueError, "registered P5.2"):
            self.prepare(precision="bfloat16")
        self.assertFalse(self.directory.exists())
        self.prepare()
        path = self.directory / "run_manifest.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.resume()

    def test_cpu_inspection_uses_committed_progress_and_actor_publication(self):
        from world_models_paired.runner import inspect
        worker = self.prepare()
        (self.directory / "status.json").write_text('{"status":"running","local_cycles_completed":999}')
        with patch("torch.cuda.is_available", side_effect=AssertionError("inspection queried CUDA")):
            first = inspect(self.directory)
        self.assertEqual(first["local_cycles_completed"], 0)
        self.assertEqual(first["live_process_status"], "running")
        self.assertFalse(first["live_status_is_durable_evidence"])
        for torn in ('{"status":', '[]'):
            with self.subTest(live_status=torn):
                (self.directory / "status.json").write_text(torn)
                advisory = inspect(self.directory)
                self.assertEqual(advisory["local_cycles_completed"], 0)
                self.assertIsNone(advisory["live_process_status"])
                self.assertIsInstance(advisory["live_status_read_error"], str)
        worker.run()
        with patch("torch.cuda.is_available", side_effect=AssertionError("inspection queried CUDA")):
            final = inspect(self.directory)
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["local_cycles_completed"], 2)
        self.assertTrue(Path(final["actor_export"]).is_dir())


if __name__ == "__main__":
    unittest.main()
