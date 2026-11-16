"""Tiny CPU P4.3 execution, fresh optimizers, boundaries and exact resume.

Only temporary synthetic replay, teacher and distillation runs are used. The
production research protocol remains fixed and no native environment is opened.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests import test_distillation_runner as distillation_fixture
from tests import test_teacher_runner as teacher_fixture
from tests.test_imagined_rollout_generator import assert_tree_equal
from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.imagination.versions import module_version
from world_models.refinement.protocol import REFINEMENT_PROTOCOL
from world_models.refinement.runner import RefinementRunner
from world_models.rollouts.schema import ImaginedRollout
from world_models.training.runner import run_lock


class RefinementRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        teacher_fixture.TeacherRunnerTests.setUpClass.__func__(cls)
        # The imported stage is immutable. Construct its real tiny teacher and
        # completed distillation once, then reuse those read-only receipts for
        # independent refinement runs. Repeating P4.1/P4.2 per test would mostly
        # benchmark checkpoint I/O, not exercise any additional P4.3 behavior.
        fixture = cls("test_completed_handoff_export_and_no_repeated_optimization")
        distillation_fixture.DistillationRunnerTests.setUp(fixture)
        cls.addClassCleanup(fixture.doCleanups)
        distiller = distillation_fixture.DistillationRunnerTests.prepare(fixture)
        distiller.run()
        cls.distillation_directory = fixture.directory
        cls.distilled_actor = deepcopy(distiller.actor.state_dict())
        cls.original_critic = deepcopy(distiller.critic.state_dict())
        cls.teacher_buffer = distiller.buffer

    @classmethod
    def tearDownClass(cls):
        teacher_fixture.TeacherRunnerTests.tearDownClass.__func__(cls)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "refinement"
        self.enterContext(patch.dict(REFINEMENT_PROTOCOL, {
            "initial_transitions": 6, "initial_model_updates": 1,
            "local_cycles": 2, "starts": 2, "horizon": 1, "epochs": 1, "minibatches": 1}))
        # Stage-handoff tests cover source compatibility itself. The completed
        # fixture intentionally records a synthetic P4.2 identity, and this
        # independent runner suite pins its own identity while files are edited.
        self.enterContext(patch("world_models.refinement.handoff._source_compatibility"))
        self.enterContext(patch("world_models.refinement.runner.source_identity",
                                return_value={"synthetic-fixture.py": "b" * 64}))

    def prepare(self, directory=None):
        return RefinementRunner.prepare(directory or self.directory,
            distillation_run=self.distillation_directory, device="cpu")

    def resume(self, directory=None):
        return RefinementRunner.resume(directory or self.directory, device="cpu")

    def test_prepare_imports_warm_started_actor_and_original_critic_with_fresh_adams(self):
        files = {p.relative_to(self.distillation_directory).as_posix(): file_sha256(p)
                 for p in self.distillation_directory.rglob("*") if p.is_file()}
        rng = torch.get_rng_state().clone()
        with patch("world_models.refinement.runner.RecurrentMAPPO.update",
                   side_effect=AssertionError("prepare optimized PPO")):
            worker = self.prepare()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        assert_tree_equal(self, worker.actor.state_dict(), self.distilled_actor)
        assert_tree_equal(self, worker.critic.state_dict(), self.original_critic)
        self.assertEqual(worker.status()["status"], "prepared")
        self.assertEqual(worker.learner.cycles_completed, 0)
        self.assertEqual(worker.manifest["protocol"]["teacher_kl_weight"], 0)
        self.assertFalse(worker.learner.actor_optimizer.state)
        self.assertFalse(worker.learner.critic_optimizer.state)
        self.assertTrue(all(p.requires_grad for p in worker.actor.parameters()))
        self.assertTrue(all(p.requires_grad for p in worker.critic.parameters()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in worker.model.parameters()))
        self.assertFalse(hasattr(worker, "teacher"))
        self.assertFalse(hasattr(worker, "distiller"))
        self.assertFalse(hasattr(worker, "model_optimizer"))
        self.assertTrue((self.directory / "run_manifest.sha256").is_file())
        self.assertFalse((self.directory / "refined_student.json").exists())
        self.assertEqual(files, {p.relative_to(self.distillation_directory).as_posix(): file_sha256(p)
                                 for p in self.distillation_directory.rglob("*") if p.is_file()})
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_real_local_rollout_and_ppo_resume_are_exact(self):
        reference = self.prepare(Path(self.temp.name) / "uninterrupted")
        reference.run()
        worker = self.prepare()
        initial_actor = module_version(worker.actor)
        initial_critic = module_version(worker.critic)
        frozen_world = module_version(worker.model)
        first = worker.run(max_cycles=1)
        self.assertEqual(first["status"], "stopped")
        self.assertEqual(worker.learner.cycles_completed, 1)
        self.assertEqual(worker.learner.actor_optimizer_steps, 1)
        self.assertEqual(worker.learner.critic_optimizer_steps, 1)
        saved_learner, saved_streams = deepcopy(worker.learner.state_dict()), deepcopy(worker.streams.state_dict())
        saved_sampler = deepcopy(worker.anchor_sampler.state_dict())
        restored = self.resume()
        assert_tree_equal(self, saved_learner, restored.learner.state_dict())
        assert_tree_equal(self, saved_streams, restored.streams.state_dict())
        assert_tree_equal(self, saved_sampler, restored.anchor_sampler.state_dict())
        result = restored.run()
        self.assertEqual((result["status"], result["cycles_executed"]), ("complete", 1))
        self.assertEqual(module_version(restored.actor), module_version(reference.actor))
        self.assertEqual(module_version(restored.critic), module_version(reference.critic))
        self.assertNotEqual(module_version(restored.actor), initial_actor)
        self.assertNotEqual(module_version(restored.critic), initial_critic)
        self.assertEqual(module_version(restored.model), frozen_world)
        assert_tree_equal(self, restored.learner.state_dict(), reference.learner.state_dict())
        assert_tree_equal(self, restored.streams.state_dict(), reference.streams.state_dict())
        assert_tree_equal(self, restored.anchor_sampler.state_dict(), reference.anchor_sampler.state_dict())
        self.assertEqual(restored.last_report, reference.last_report)
        self.assertEqual(len(list((self.directory / "checkpoints").glob("*/manifest.sha256"))), 3)

    def test_collection_and_updater_consume_only_new_local_student_rollouts(self):
        worker = self.prepare()
        collected = []
        collect = worker.generator.collect
        update = worker.learner.update

        def collect_local(*args, **kwargs):
            rollout = collect(*args, **kwargs)
            self.assertIs(type(rollout), ImaginedRollout)
            self.assertEqual(rollout.metadata.get("policy_role", "local"), "local")
            self.assertEqual(rollout.metadata["actor_version"], module_version(worker.actor))
            collected.append(rollout)
            return rollout

        def update_local(rollout, *args, **kwargs):
            self.assertIs(rollout, collected[-1])
            self.assertNotIn("teacher_version", rollout.metadata)
            return update(rollout, *args, **kwargs)

        with self.assertRaisesRegex(ValueError, "ImaginedRollout"):
            worker.learner.update(self.teacher_buffer, worker.anchor_sampler)
        with patch.object(worker.generator, "collect", side_effect=collect_local), \
                patch.object(worker.learner, "update", side_effect=update_local), \
                patch("world_models.policies.teacher.GlobalTeacher.forward",
                      side_effect=AssertionError("refinement queried teacher")), \
                patch("world_models.policies.teacher.GlobalTeacher.memory_step",
                      side_effect=AssertionError("refinement replayed teacher history")), \
                patch("world_models.policies.teacher.GlobalTeacher.action_from_memory",
                      side_effect=AssertionError("refinement queried privileged action head")):
            worker.run()
        self.assertEqual(len(collected), 2)
        self.assertIsNot(collected[0], collected[1])
        self.assertNotEqual(collected[0].metadata["actor_version"], collected[1].metadata["actor_version"])

    def test_stop_before_or_during_cycle_commits_only_whole_boundaries(self):
        worker = self.prepare()
        (self.directory / "stop.request").write_text("stop")
        self.assertEqual(worker.run()["cycles_executed"], 0)
        self.assertFalse((self.directory / "stop.request").exists())
        update = worker.learner.update

        def stop_after_update(*args, **kwargs):
            result = update(*args, **kwargs)
            worker.request_stop()
            return result

        with patch.object(worker.learner, "update", side_effect=stop_after_update):
            result = worker.run()
        self.assertEqual((result["status"], worker.learner.cycles_completed), ("stopped", 1))
        restored = self.resume()
        self.assertEqual(restored.learner.cycles_completed, 1)
        self.assertEqual(restored.run()["status"], "complete")

    def test_failed_uncommitted_cycle_restores_last_durable_rng_and_optimizer_boundary(self):
        worker = self.prepare()
        learner = deepcopy(worker.learner.state_dict())
        streams = deepcopy(worker.streams.state_dict())
        sampler = deepcopy(worker.anchor_sampler.state_dict())
        with patch("world_models.refinement.runner.save_cycle_checkpoint", side_effect=OSError("fixture disk failure")):
            with self.assertRaisesRegex(OSError, "fixture disk failure"):
                worker.run(max_cycles=1)
        self.assertEqual(worker.status()["status"], "failed")
        with self.assertRaisesRegex(RuntimeError, "fresh instance"):
            worker.run()
        restored = self.resume()
        assert_tree_equal(self, learner, restored.learner.state_dict())
        assert_tree_equal(self, streams, restored.streams.state_dict())
        assert_tree_equal(self, sampler, restored.anchor_sampler.state_dict())
        self.assertEqual(restored.learner.cycles_completed, 0)
        attempts = [json.loads(line) for line in (self.directory / "attempts.jsonl").read_text().splitlines()]
        self.assertIn("unknown", attempts[-1]["partial_physical_optimizer_work"])

    def test_ahead_status_and_uncommitted_debris_do_not_override_committed_checkpoint(self):
        worker = self.prepare()
        orphan = self.directory / "checkpoints/state_99999999_aaaaaaaa"
        orphan.mkdir()
        (orphan / "state.pt").write_text("partial")
        status = worker.status()
        status.update(local_cycles_completed=999, status="complete", phase="complete")
        (self.directory / "status.json").write_text(json.dumps(status))
        restored = self.resume()
        self.assertEqual(restored.learner.cycles_completed, 0)
        restored.run(max_cycles=1)
        with (restored.last_checkpoint / "state.pt").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "payload length mismatch"):
            self.resume()

    def test_worker_lock_and_stale_worker_cannot_overwrite_new_checkpoint(self):
        worker = self.prepare()
        old = self.resume()
        with run_lock(self.directory):
            with self.assertRaisesRegex(RuntimeError, "another worker"):
                self.resume()
            with self.assertRaisesRegex(RuntimeError, "another worker"):
                worker.run()
        worker.run(max_cycles=1)
        with self.assertRaisesRegex(RuntimeError, "advanced"):
            old.run()

    def test_source_runtime_manifest_and_frozen_world_changes_are_rejected(self):
        worker = self.prepare()
        with patch("world_models.refinement.runner.source_identity", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.resume()
            with self.assertRaisesRegex(ValueError, "source changed"):
                worker.run()
        with patch("world_models.refinement.runner._runtime", return_value={"changed": "runtime"}):
            with self.assertRaisesRegex(ValueError, "same logical device.*runtime"):
                self.resume()
        with torch.no_grad():
            next(worker.model.parameters()).view(-1)[0].add_(1)
        with self.assertRaisesRegex(ValueError, "frozen shared world model"):
            worker.run()
        manifest = self.directory / "run_manifest.json"
        manifest.write_text(manifest.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.resume()

    def test_completed_handoff_export_and_no_repeated_optimization(self):
        from scripts.phase36_support.evaluation import load_actor_export

        worker = self.prepare()
        result = worker.run()
        self.assertEqual(worker.learner.cycles_completed, 2)
        self.assertEqual((worker.learner.actor_optimizer_steps, worker.learner.critic_optimizer_steps), (2, 2))
        self.assertEqual((result["new_environment_interactions"], result["new_model_updates"]), (0, 0))
        self.assertEqual(result["imagined_joint_steps"], 4)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in worker.model.parameters()))
        path = self.directory / "refined_student.json"
        handoff = json.loads(path.read_text())
        self.assertEqual(path.with_suffix(".sha256").read_text().strip(), f"{file_sha256(path)}  {path.name}")
        self.assertEqual(handoff["checkpoint_manifest_sha256"], file_sha256(worker.last_checkpoint / "manifest.json"))
        self.assertEqual(handoff["actor_version"], module_version(worker.actor))
        actor, provenance = load_actor_export(self.directory / "actor_export", device="cpu")
        self.assertEqual(module_version(actor), module_version(worker.actor))
        self.assertEqual(provenance["metadata"]["variant"], "D")
        self.assertEqual(provenance["metadata"]["local_ppo_cycles"], 2)
        saved, _ = load_cycle_checkpoint(worker.last_checkpoint)
        self.assertNotIn("teacher", saved)
        self.assertNotIn("distillation_optimizer", saved)
        restored = self.resume()
        with patch.object(restored.learner, "update", side_effect=AssertionError("repeated complete PPO")):
            self.assertEqual(restored.run()["cycles_executed"], 0)
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "receipt"):
            restored.run()

    def test_incomplete_final_handoff_can_be_repaired_without_repeating_training(self):
        worker = self.prepare()
        with patch.object(worker, "_publish_refined_student", side_effect=OSError("fixture publication failed")):
            with self.assertRaisesRegex(OSError, "publication failed"):
                worker.run()
        restored = self.resume()
        self.assertEqual(restored.learner.cycles_completed, 2)
        with patch.object(restored.learner, "update", side_effect=AssertionError("repeated complete PPO")):
            self.assertEqual(restored.run()["status"], "complete")
        self.assertTrue((self.directory / "refined_student.sha256").is_file())


if __name__ == "__main__":
    unittest.main()
