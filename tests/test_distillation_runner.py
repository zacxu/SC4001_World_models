"""Tiny CPU stage execution, immutable boundaries and exact distillation resume.

Only temporary synthetic replay and fixture teacher/model weights are used.
The research CLI registration is never reduced outside these tests.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from tests import test_teacher_runner as teacher_fixture
from tests.test_imagined_rollout_generator import assert_tree_equal
from world_models.cycle.checkpoints import load_cycle_checkpoint, save_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.distillation.protocol import DISTILLATION_PROTOCOL
from world_models.distillation.runner import DistillationRunner
from world_models.distillation.handoff import load_completed_teacher
from world_models.imagination.versions import module_version
from world_models.teacher.runner import source_identity as teacher_sources
from world_models.training.runner import run_lock


class DistillationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        teacher_fixture.TeacherRunnerTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        teacher_fixture.TeacherRunnerTests.tearDownClass.__func__(cls)

    def setUp(self):
        teacher_fixture.TeacherRunnerTests.setUp(self)
        self.teacher_directory = self.directory
        historical = {name: value for name, value in teacher_sources().items()
                      if not name.startswith("src/world_models/distillation/")}
        with patch("world_models.teacher.runner.source_identity", return_value=historical):
            teacher = teacher_fixture.TeacherRunnerTests.prepare(self)
            teacher.run()
        self.directory = Path(self.temp.name) / "distillation"
        self.enterContext(patch.dict(DISTILLATION_PROTOCOL, {
            "initial_transitions": 6, "initial_model_updates": 1, "teacher_cycles": 2,
            "trajectories": 2, "horizon": 1, "updates": 2, "sequences_per_update": 2}))
        # The stage identity guard is tested below. Pin the fixture identity so
        # concurrent editing of this new package cannot alter synthetic runs.
        self.enterContext(patch("world_models.distillation.runner.source_identity",
                                return_value={"synthetic-fixture.py": "a" * 64}))

    def prepare(self, directory=None):
        return DistillationRunner.prepare(directory or self.directory,
            teacher_run=self.teacher_directory, device="cpu")

    def resume(self, directory=None):
        return DistillationRunner.resume(directory or self.directory, device="cpu")

    def test_prepare_has_paired_fresh_student_and_no_generation_or_optimization(self):
        rng = torch.get_rng_state().clone()
        teacher_files = {p.relative_to(self.teacher_directory).as_posix(): file_sha256(p)
                         for p in self.teacher_directory.rglob("*") if p.is_file()}
        with patch("world_models.distillation.runner.TeacherRolloutGenerator.collect",
                   side_effect=AssertionError("prepare generated trajectories")), \
                patch("world_models.distillation.runner.StudentDistiller.step",
                      side_effect=AssertionError("prepare optimized student")):
            worker = self.prepare()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(worker.status()["status"], "prepared")
        self.assertEqual(worker.state["trajectories_completed"], 0)
        self.assertEqual(worker.learner.updates_completed, 0)
        self.assertEqual(len(worker.references), 2)
        self.assertEqual(len(worker.shards), 0)
        self.assertEqual(module_version(worker.actor), worker.paired_versions["actor"])
        self.assertEqual(module_version(worker.critic), worker.paired_versions["critic"])
        self.assertFalse(worker.learner.optimizer.state)
        self.assertFalse(hasattr(worker, "model_optimizer"))
        self.assertFalse(hasattr(worker, "ppo_optimizer"))
        self.assertFalse((self.directory / "buffer").exists())
        self.assertFalse((self.directory / "warm_started_student.json").exists())
        self.assertTrue((self.directory / "run_manifest.sha256").is_file())
        self.assertEqual(teacher_files, {p.relative_to(self.teacher_directory).as_posix(): file_sha256(p)
                                        for p in self.teacher_directory.rglob("*") if p.is_file()})
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_resumed_trajectory_and_adam_updates_match_uninterrupted_training(self):
        reference = self.prepare(Path(self.temp.name) / "uninterrupted")
        reference.run()
        worker = self.prepare()
        first = worker.run(max_units=1)
        self.assertEqual((first["status"], first["trajectories_completed"], first["distillation_optimizer_steps"]),
                         ("stopped", 1, 0))
        streams = deepcopy(worker.streams.state_dict())
        restored = self.resume()
        assert_tree_equal(self, streams, restored.streams.state_dict())
        middle = restored.run(max_units=2)
        self.assertEqual((middle["trajectories_completed"], middle["distillation_optimizer_steps"]), (2, 1))
        optimizer = deepcopy(restored.learner.state_dict())
        restored = self.resume()
        assert_tree_equal(self, optimizer, restored.learner.state_dict())
        completed = restored.run()
        self.assertEqual((completed["status"], completed["units_executed"]), ("complete", 1))
        self.assertEqual(module_version(reference.actor), module_version(restored.actor))
        self.assertNotEqual(module_version(restored.actor), restored.paired_versions["actor"])
        self.assertEqual(module_version(reference.critic), module_version(restored.critic))
        assert_tree_equal(self, reference.learner.state_dict(), restored.learner.state_dict())
        assert_tree_equal(self, reference.streams.state_dict(), restored.streams.state_dict())
        assert_tree_equal(self, reference.sampler.state_dict(), restored.sampler.state_dict())
        self.assertEqual(reference.diagnostics_before, restored.diagnostics_before)
        self.assertEqual(reference.diagnostics_after, restored.diagnostics_after)
        self.assertEqual(reference.last_report, restored.last_report)
        self.assertEqual(reference.references, restored.references)
        for name in ("local_rgb", "previous_actions", "actions", "valid", "targets"):
            self.assertTrue(torch.equal(getattr(reference.buffer, name), getattr(restored.buffer, name)))
        self.assertEqual(len(list((self.directory / "checkpoints").glob("*/manifest.sha256"))), 5)
        attempts = [json.loads(line) for line in (self.directory / "attempts.jsonl").read_text().splitlines()]
        starts = [row for row in attempts if row["event"] == "unit_started"]
        commits = [row for row in attempts if row["event"] == "unit_committed"]
        self.assertEqual([row["phase"] for row in commits], ["buffer", "buffer", "distillation", "distillation"])
        self.assertEqual([row["attempt"] for row in starts], [row["attempt"] for row in commits])

    def test_complete_handoff_binds_all_work_and_frozen_networks(self):
        worker = self.prepare()
        frozen = deepcopy(worker.frozen_versions)
        report = worker.run()
        self.assertEqual((report["trajectories_completed"], report["distillation_optimizer_steps"]), (2, 2))
        self.assertEqual(report["imagined_joint_steps"], 2)
        for key in ("new_environment_interactions", "new_model_updates", "local_ppo_optimizer_steps",
                    "teacher_optimizer_steps"):
            self.assertEqual(report[key], 0)
        for name, network in worker._frozen().items():
            self.assertEqual(module_version(network), frozen[name])
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in network.parameters()))
        for diagnostics, updates in ((worker.diagnostics_before, 0), (worker.diagnostics_after, 2)):
            self.assertEqual(diagnostics["optimizer_updates"], updates)
            self.assertEqual(diagnostics["trajectories"], 2)
            self.assertEqual(diagnostics["valid_player_decisions"], 4)
            self.assertIn("not validation", diagnostics["scope"])
        payload, metadata = load_cycle_checkpoint(worker.last_checkpoint)
        self.assertEqual(metadata["distillation_optimizer_steps"], 2)
        self.assertEqual(payload["diagnostics_before"], worker.diagnostics_before)
        self.assertEqual(payload["diagnostics_after"], worker.diagnostics_after)
        path = self.directory / "warm_started_student.json"
        handoff = json.loads(path.read_text())
        self.assertEqual(path.with_suffix(".sha256").read_text().strip(), f"{file_sha256(path)}  {path.name}")
        self.assertEqual(handoff["checkpoint_manifest_sha256"], file_sha256(worker.last_checkpoint / "manifest.json"))
        self.assertEqual(handoff["actor_version"], module_version(worker.actor))
        self.assertEqual(handoff["critic_version"], worker.paired_versions["critic"])
        self.assertTrue(handoff["fresh_local_ppo_optimizers_required"])
        self.assertFalse(handoff["full_d_pipeline_implemented"])
        self.assertFalse(handoff["phase4_acceptance"])
        restored = self.resume()
        with patch.object(restored.generator, "collect", side_effect=AssertionError("repeated buffer")), \
                patch.object(restored.learner, "step", side_effect=AssertionError("repeated optimization")):
            self.assertEqual(restored.run()["units_executed"], 0)

    def test_early_terminal_trajectories_can_be_shorter_than_requested_horizon(self):
        def terminating_fixture(*args, **kwargs):
            context = load_completed_teacher(*args, **kwargs)
            # This synthetic replacement supplies a one-step terminal model to
            # the stage. The real completed-teacher importer remains unchanged.
            with torch.no_grad():
                head = context.model.transition_heads.outcome_head
                head.weight.zero_()
                head.bias.fill_(-1000)
                head.bias[1] = 1000
            return context

        with patch.dict(DISTILLATION_PROTOCOL, {"horizon": 3}), \
                patch("world_models.distillation.runner.load_completed_teacher", side_effect=terminating_fixture):
            worker = self.prepare()
            worker.run(max_units=1)
            restored = self.resume()
            report = restored.run()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["imagined_joint_steps"], 2)
        self.assertEqual(restored.buffer.steps, 1)
        self.assertEqual(restored.buffer.batch_size, 2)
        self.assertTrue(bool(restored.buffer.terminated.all()))
        self.assertFalse(bool(restored.buffer.cutoff.any()))

    def test_pause_before_unit_and_during_trajectory_and_update_is_durable(self):
        worker = self.prepare()
        (self.directory / "stop.request").write_text("stop")
        self.assertEqual(worker.run()["units_executed"], 0)
        self.assertFalse((self.directory / "stop.request").exists())
        original = worker.generator.collect

        def stop_after_trajectory(*args, **kwargs):
            result = original(*args, **kwargs)
            worker.request_stop()
            return result

        with patch.object(worker.generator, "collect", side_effect=stop_after_trajectory):
            report = worker.run()
        self.assertEqual((report["status"], report["trajectories_completed"]), ("stopped", 1))
        restored = self.resume()
        restored.run(max_units=1)
        update = restored.learner.step

        def stop_after_update(*args, **kwargs):
            result = update(*args, **kwargs)
            restored.request_stop()
            return result

        with patch.object(restored.learner, "step", side_effect=stop_after_update):
            report = restored.run()
        self.assertEqual((report["status"], report["distillation_optimizer_steps"]), ("stopped", 1))
        latest = self.resume()
        self.assertEqual(latest.learner.updates_completed, 1)
        self.assertEqual(latest.run()["status"], "complete")

    def test_uncommitted_shards_and_ahead_status_are_ignored(self):
        worker = self.prepare()
        debris = self.directory / "buffer/trajectory_0000_aaaaaaaa"
        debris.mkdir(parents=True)
        (debris / "state.pt").write_bytes(b"incomplete fixture payload")
        uncommitted = self.directory / "checkpoints/state_99999999_aaaaaaaa"
        uncommitted.mkdir()
        (uncommitted / "state.pt").write_bytes(b"incomplete checkpoint")
        status = worker.status()
        status.update(trajectories_completed=999, distillation_optimizer_steps=999, phase="complete", status="complete")
        (self.directory / "status.json").write_text(json.dumps(status))
        restored = self.resume()
        self.assertEqual(restored.state["trajectories_completed"], 0)
        self.assertEqual(restored.learner.updates_completed, 0)
        restored.run()
        self.assertEqual(len(restored.shards), 2)
        self.assertEqual(len({row["path"] for row in restored.shards}), 2)
        self.assertNotIn(debris.relative_to(self.directory).as_posix(), [row["path"] for row in restored.shards])
        self.assertTrue(debris.exists())

    def test_corrupt_latest_committed_checkpoint_never_falls_back(self):
        worker = self.prepare()
        worker.run(max_units=1)
        self.assertEqual(len(list((self.directory / "checkpoints").glob("*/manifest.sha256"))), 2)
        with (worker.last_checkpoint / "state.pt").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "payload length mismatch"):
            self.resume()

    def test_receipt_valid_shards_cannot_swap_frozen_teacher_or_sampling_history(self):
        worker = self.prepare()
        worker.run(max_units=1)
        original_row = deepcopy(worker.shards[0])
        saved, metadata = load_cycle_checkpoint(self.directory / original_row["path"])
        variants = []
        changed_teacher = deepcopy(saved)
        changed_teacher["buffer"]["metadata"]["teacher_version"] = "a" * 64
        variants.append(changed_teacher)
        changed_stream = deepcopy(saved)
        changed_stream["buffer"]["streams"][0]["before"] = deepcopy(changed_stream["buffer"]["streams"][0]["after"])
        variants.append(changed_stream)
        for index, replacement in enumerate(variants):
            with self.subTest(payload=index):
                path = self.directory / "buffer" / f"receipt_valid_swap_{index}"
                save_cycle_checkpoint(path, state=replacement, metadata=metadata)
                worker.shards[0] = {"path": path.relative_to(self.directory).as_posix(),
                                    "manifest_sha256": file_sha256(path / "manifest.json")}
                # The outer metadata and payload receipts are valid. Reject
                # semantically mismatched frozen weights or RNG continuation.
                with self.assertRaisesRegex(ValueError, "payload anchor, frozen weights or sampling chain"):
                    worker._load_buffer()
        worker.shards[0] = original_row
        self.assertEqual(worker._load_buffer().batch_size, 1)

    def test_failed_checkpoint_after_shard_publication_replays_only_uncommitted_work(self):
        worker = self.prepare()
        before = deepcopy(worker.streams.state_dict())
        checkpoint = worker.last_checkpoint

        def fail_main_checkpoint(path, **kwargs):
            if Path(path).parent.name == "checkpoints":
                raise OSError("fixture checkpoint disk failure")
            return save_cycle_checkpoint(path, **kwargs)

        with patch("world_models.distillation.runner.save_cycle_checkpoint", side_effect=fail_main_checkpoint):
            with self.assertRaisesRegex(OSError, "fixture checkpoint disk failure"):
                worker.run(max_units=1)
        self.assertEqual(worker.status()["status"], "failed")
        with self.assertRaisesRegex(RuntimeError, "fresh instance"):
            worker.run()
        restored = self.resume()
        self.assertEqual(restored.last_checkpoint, checkpoint)
        self.assertEqual(restored.state["trajectories_completed"], 0)
        assert_tree_equal(self, before, restored.streams.state_dict())
        restored.run(max_units=1)
        self.assertEqual(len(restored.shards), 1)
        self.assertEqual(len(list((self.directory / "buffer").glob("*/manifest.sha256"))), 2)
        attempts = [json.loads(line) for line in (self.directory / "attempts.jsonl").read_text().splitlines()]
        self.assertEqual(len([row for row in attempts if row["event"] == "unit_committed"]), 1)
        self.assertIn("unknown", next(row for row in attempts if row["event"] == "invocation_failed")["partial_physical_work"])

    def test_failed_handoff_publication_recovers_without_repeating_updates(self):
        worker = self.prepare()
        with patch.object(worker, "_publish_handoff", side_effect=OSError("fixture publication failure")):
            with self.assertRaisesRegex(OSError, "fixture publication failure"):
                worker.run()
        restored = self.resume()
        self.assertEqual(restored.state["phase"], "complete")
        self.assertEqual(restored.learner.updates_completed, 2)
        with patch.object(restored.learner, "step", side_effect=AssertionError("repeated optimization")):
            self.assertEqual(restored.run()["units_executed"], 0)
        path = self.directory / "warm_started_student.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "invalid receipt"):
            restored.run()

    def test_sources_runtime_teacher_and_initial_replay_binding_changes_are_rejected(self):
        worker = self.prepare()
        with patch("world_models.distillation.runner.source_identity", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.resume()
            with self.assertRaisesRegex(ValueError, "sources changed"):
                worker.run()
        with patch("world_models.distillation.runner._runtime", return_value={"changed": "runtime"}):
            with self.assertRaisesRegex(ValueError, "same logical device and runtime"):
                self.resume()
        changed_selection = deepcopy(self.selection)
        changed_selection["selection_sha256"] = "b" * 64
        self.selected.return_value = changed_selection
        with self.assertRaisesRegex(ValueError, "frozen replay, normalization, selection"):
            self.resume()
        with self.assertRaisesRegex(ValueError, "frozen teacher weights, replay, selection or normalization"):
            worker.run()
        self.selected.return_value = deepcopy(self.selection)
        replay_chunk = next(self.dataset.path.rglob("*.npz"))
        original_replay = replay_chunk.read_bytes()
        try:
            replay_chunk.write_bytes(original_replay + b"corrupt fixture replay")
            with self.assertRaises(ValueError):
                self.resume()
            with self.assertRaises(ValueError):
                worker.run()
        finally:
            replay_chunk.write_bytes(original_replay)
        path = self.teacher_directory / "frozen_teacher.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.resume()
        with self.assertRaisesRegex(ValueError, "receipt"):
            worker.run()

    def test_frozen_network_mutation_is_detected_before_any_unit(self):
        worker = self.prepare()
        for name, network in worker._frozen().items():
            with self.subTest(network=name):
                parameter = next(network.parameters())
                before = parameter.detach().clone()
                with torch.no_grad():
                    parameter.view(-1)[0].add_(1)
                with self.assertRaisesRegex(ValueError, "frozen"):
                    worker.run()
                with torch.no_grad():
                    parameter.copy_(before)
        self.assertEqual(self.resume().state["trajectories_completed"], 0)

    def test_exclusive_lock_and_stale_worker_cannot_duplicate_committed_units(self):
        worker = self.prepare()
        old = self.resume()
        with run_lock(self.directory):
            with self.assertRaisesRegex(RuntimeError, "another worker"):
                self.resume()
            with self.assertRaisesRegex(RuntimeError, "another worker"):
                worker.run()
        worker.run(max_units=1)
        with self.assertRaisesRegex(RuntimeError, "another worker advanced"):
            old.run()
        for invalid in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                worker.run(max_units=invalid)


if __name__ == "__main__":
    unittest.main()
