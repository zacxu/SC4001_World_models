"""P4.1 CPU fixture checkpoints, exact continuation, frozen inputs and locks.

All inputs are generated in temporary directories. The tiny registration and
selection lookup are patched only inside fixtures; no research weights or
native environment are used and production CLI budgets remain fixed.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_cycle_controller import controller_settings
from tests.test_imagined_rollout_generator import assert_tree_equal
from tests.test_model_sampling import identifier, make_store
from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.imagination.versions import module_version
from world_models.infrastructure.reproducibility import canonical_hash, derive_seed
from world_models.models.world_model import SamplingStreams, build_model
from world_models.teacher.protocol import TEACHER_PROTOCOL
from world_models.teacher.runner import TeacherRunner, _latest, _publish_receipt
from world_models.training.checkpoints import save_checkpoint
from world_models.training.normalization import fit_normalization
from world_models.training.runner import run_lock, source_identity as p24_sources
from world_models.training.sampling import ReplaySampler


ROOT = Path(__file__).resolve().parents[1]


class TeacherRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.settings = controller_settings()
        cls.dataset = make_store(cls.root / "replay", [(identifier("p41-runner", "fit"), 6, False)])
        cls.stats = fit_normalization(cls.dataset)
        cls.stats_path = cls.root / "normalization.json"
        cls.stats.save(cls.stats_path)
        cls.provenance = {"initial_normalization": {"identity_sha256": cls.stats.as_dict()["identity_sha256"],
                                                   "file_sha256": file_sha256(cls.stats_path)},
            "replays": [{"manifest_sha256": cls.dataset.manifest_sha256,
                         "dataset_identity_sha256": cls.dataset.manifest["dataset_identity_sha256"],
                         "transitions": 6}]}
        model = build_model(cls.settings, master_seed=0, variant="B")
        with torch.no_grad():
            model.transition_heads.outcome_head.weight.zero_()
            model.transition_heads.outcome_head.bias.fill_(-1000)
            model.transition_heads.outcome_head.bias[0] = 1000
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        common = deepcopy(cls.settings)
        common.pop("variant")
        metadata = {"task_family": "relay", "run_seed": 0, "stage": "initial", "variant": "shared_initial:B,D",
                    "target_stage_updates": 1, "common_config_sha256": canonical_hash(common),
                    "source_sha256": p24_sources(), **cls.provenance}
        cls.checkpoint = cls.root / "synthetic_p24.pt"
        save_checkpoint(cls.checkpoint, model=model, optimizer=optimizer,
            streams=SamplingStreams.from_seed(17), sampler=ReplaySampler([cls.dataset], seed=17, burn_in=0, loss_steps=1),
            update=1, metadata=metadata)
        cls.selection = {"task": "relay", "seed": 0, "model_updates": 1,
            "checkpoint_path": str(cls.checkpoint), "checkpoint_sha256": file_sha256(cls.checkpoint),
            "selection_path": str(cls.root / "synthetic_selection.json"), "selection_sha256": "a" * 64,
            **cls.provenance}
        del model, optimizer
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "teacher"
        self.enterContext(patch.dict(TEACHER_PROTOCOL, {"initial_transitions": 6, "initial_model_updates": 1,
            "teacher_cycles": 2, "starts": 2, "horizon": 1, "epochs": 1, "minibatches": 1}))
        self.load_case = self.enterContext(patch("world_models.teacher.runner.load_case", return_value=(
            SimpleNamespace(settings=deepcopy(self.settings)), [self.dataset], self.stats, deepcopy(self.provenance))))
        self.selected = self.enterContext(patch("world_models.teacher.runner.resolve_selected_case",
            return_value=deepcopy(self.selection)))

    def prepare(self, directory=None, **overrides):
        arguments = {"initial_selection": self.selection["selection_path"], "master_seed": 0,
                     "device": "cpu", "precision": "float32", "config_root": ROOT / "configs",
                     "data_root": self.root, "statistics_root": self.root}
        arguments.update(overrides)
        return TeacherRunner.prepare(directory or self.directory, **arguments)

    def resume(self, directory=None):
        return TeacherRunner.resume(directory or self.directory, device="cpu")

    def test_prepare_does_not_train_or_consume_global_rng_and_namespaces_are_separate(self):
        rng = torch.get_rng_state().clone()
        with patch("world_models.teacher.runner.TeacherMAPPO.update", side_effect=AssertionError("prepare trained")):
            worker = self.prepare()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(worker.status()["status"], "prepared")
        self.assertEqual(worker.status()["teacher_cycles_completed"], 0)
        self.assertFalse(hasattr(worker, "student"))
        self.assertFalse(hasattr(worker, "model_optimizer"))
        self.assertTrue(all(not value.requires_grad for value in worker.model.parameters()))
        namespaces = worker.manifest["rng_namespaces"]
        self.assertEqual(len(set(namespaces.values())), len(namespaces))
        self.assertNotIn(derive_seed("relay", 0, "local_policy_updates", "D"), namespaces.values())
        self.assertNotIn(derive_seed("relay", 0, "imagination", "D"), namespaces.values())
        self.assertEqual(worker.learner.seed, namespaces["teacher_updates"])
        self.assertTrue((self.directory / "run_manifest.sha256").is_file())
        self.assertEqual(len(list((self.directory / "checkpoints").glob("*/manifest.sha256"))), 1)
        self.assertFalse((self.directory / "frozen_teacher.json").exists())
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_actual_teacher_cycle_resume_matches_next_cycle_optimizers_rng_and_frozen_weights(self):
        worker = self.prepare()
        model_before, teacher_before = module_version(worker.model), module_version(worker.teacher)
        first = worker.run(max_cycles=1)
        self.assertEqual((first["status"], first["teacher_cycles_completed"]), ("stopped", 1))
        self.assertEqual((first["teacher_actor_optimizer_steps"], first["teacher_critic_optimizer_steps"]), (1, 1))
        self.assertNotEqual(teacher_before, module_version(worker.teacher))
        saved_optimizer = deepcopy(worker.learner.state_dict())
        saved_streams = deepcopy(worker.streams.state_dict())
        reference = self.resume()
        assert_tree_equal(self, saved_optimizer, reference.learner.state_dict())
        assert_tree_equal(self, saved_streams, reference.streams.state_dict())
        # Hold an independently resumed instance at the same boundary, then
        # produce its exact next cycle without writing into the original run.
        anchors = reference.builder.build(reference.anchor_sampler)
        rollout = reference.generator.collect(anchors, streams=reference.streams)
        expected_report = reference.learner.update(rollout, reference.anchor_sampler)
        final = worker.run()
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["teacher_cycles_completed"], 2)
        self.assertEqual((final["model_updates"], final["new_model_updates"], final["real_joint_transitions"],
                          final["new_environment_interactions"], final["local_student_optimizer_steps"]), (1, 0, 6, 0, 0))
        self.assertEqual(final["teacher_player_decisions"], 2 * final["imagined_joint_steps"])
        self.assertEqual(module_version(worker.model), model_before)
        self.assertTrue(all(value.grad is None for value in worker.model.parameters()))
        self.assertEqual(module_version(worker.teacher), module_version(reference.teacher))
        self.assertEqual(module_version(worker.critic), module_version(reference.critic))
        assert_tree_equal(self, worker.learner.state_dict(), reference.learner.state_dict())
        assert_tree_equal(self, worker.streams.state_dict(), reference.streams.state_dict())
        assert_tree_equal(self, worker.anchor_sampler.state_dict(), reference.anchor_sampler.state_dict())
        self.assertEqual(worker.last_report, expected_report)
        self.assertFalse(worker.teacher.training)
        self.assertTrue(all(not value.requires_grad for value in worker.teacher.parameters()))
        payload, metadata = load_cycle_checkpoint(worker.last_checkpoint)
        self.assertTrue(metadata["frozen_for_distillation"])
        self.assertEqual(payload["last_report"], expected_report)
        handoff = json.loads((self.directory / "frozen_teacher.json").read_text())
        self.assertEqual(handoff["checkpoint_manifest_sha256"], file_sha256(worker.last_checkpoint / "manifest.json"))
        self.assertFalse(handoff["deployable_local_actor"])
        completed = self.resume()
        self.assertEqual(completed.run()["cycles_executed"], 0)
        self.assertEqual(len(list((self.directory / "checkpoints").glob("*/manifest.sha256"))), 3)
        attempts = [json.loads(line) for line in (self.directory / "attempts.jsonl").read_text().splitlines()]
        started = [row for row in attempts if row["event"] == "teacher_cycle_started"]
        committed = [row for row in attempts if row["event"] == "teacher_cycle_committed"]
        self.assertEqual([row["teacher_cycle"] for row in committed], [1, 2])
        self.assertEqual([row["attempt"] for row in started], [row["attempt"] for row in committed])

    def test_stop_before_cycle_and_during_cycle_commits_complete_boundary(self):
        worker = self.prepare()
        (self.directory / "stop.request").write_text("stop")
        report = worker.run()
        self.assertEqual(report["teacher_cycles_completed"], 0)
        self.assertFalse((self.directory / "stop.request").exists())
        original = worker.learner.update
        def stop_after_update(*args, **kwargs):
            report = original(*args, **kwargs)
            worker.request_stop()
            return report
        with patch.object(worker.learner, "update", side_effect=stop_after_update):
            report = worker.run()
        self.assertEqual((report["status"], report["teacher_cycles_completed"]), ("stopped", 1))
        restored = self.resume()
        self.assertEqual(restored.learner.cycles_completed, 1)
        self.assertEqual(restored.run()["teacher_cycles_completed"], 2)

    def test_uncommitted_debris_is_ignored_and_corrupt_latest_commit_never_falls_back(self):
        worker = self.prepare()
        orphan = self.directory / "checkpoints/state_99999999_aaaaaaaa"
        orphan.mkdir()
        (orphan / "state.pt").write_text("partial")
        self.assertEqual(_latest(self.directory), worker.last_checkpoint)
        self.assertEqual(self.resume().learner.cycles_completed, 0)
        path = worker.last_checkpoint / "state.pt"
        with path.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "payload length mismatch"):
            self.resume()

    def test_failed_uncommitted_cycle_resumes_old_rng_and_optimizers(self):
        worker = self.prepare()
        before = deepcopy(worker.learner.state_dict())
        streams = deepcopy(worker.streams.state_dict())
        with patch("world_models.teacher.runner.save_cycle_checkpoint", side_effect=OSError("fixture disk failure")):
            with self.assertRaisesRegex(OSError, "fixture disk failure"):
                worker.run(max_cycles=1)
        self.assertEqual(worker.status()["status"], "failed")
        with self.assertRaisesRegex(RuntimeError, "fresh instance"):
            worker.run()
        restored = self.resume()
        assert_tree_equal(self, before, restored.learner.state_dict())
        assert_tree_equal(self, streams, restored.streams.state_dict())
        self.assertEqual(restored.learner.cycles_completed, 0)
        attempts = [json.loads(line) for line in (self.directory / "attempts.jsonl").read_text().splitlines()]
        self.assertEqual([row["event"] for row in attempts], ["teacher_cycle_started", "teacher_invocation_failed"])
        self.assertIn("unknown", attempts[-1]["partial_physical_optimizer_work"])

    def test_worker_lock_and_stale_loaded_worker_cannot_overwrite_new_checkpoint(self):
        worker = self.prepare()
        old = self.resume()
        with run_lock(self.directory):
            with self.assertRaisesRegex(RuntimeError, "another worker"):
                self.resume()
            with self.assertRaisesRegex(RuntimeError, "another worker"):
                worker.run()
        worker.run(max_cycles=1)
        with self.assertRaisesRegex(RuntimeError, "advanced this teacher"):
            old.run()

    def test_manifest_source_runtime_and_input_tampering_are_rejected(self):
        worker = self.prepare()
        with patch("world_models.teacher.runner.source_identity", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.resume()
            with self.assertRaisesRegex(ValueError, "source changed"):
                worker.run()
        with patch("world_models.teacher.runner._runtime", return_value={"changed": "runtime"}):
            with self.assertRaisesRegex(ValueError, "same logical device and runtime"):
                self.resume()
        self.selected.return_value["selection_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "input identities changed"):
            self.resume()
        self.selected.return_value = deepcopy(self.selection)
        path = self.directory / "run_manifest.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "manifest receipt mismatch"):
            self.resume()

    def test_production_input_guards_reject_bad_seed_update_normalization_and_recipe(self):
        for seed in (-1, 2, True):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seeds 0 and 1"):
                self.prepare(master_seed=seed)
        self.selected.return_value["model_updates"] = 2
        with self.assertRaisesRegex(ValueError, "exact frozen"):
            self.prepare()
        self.selected.return_value = deepcopy(self.selection)
        self.selected.return_value["initial_normalization"]["file_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "exact frozen"):
            self.prepare()
        self.selected.return_value = deepcopy(self.selection)
        with patch.dict(TEACHER_PROTOCOL, {"starts": 4}):
            with self.assertRaisesRegex(ValueError, "registered PPO"):
                self.prepare()
        self.assertFalse(self.directory.exists())

    def test_frozen_world_mutation_rejected_before_training(self):
        worker = self.prepare()
        with torch.no_grad():
            next(worker.model.parameters()).view(-1)[0].add_(1)
        with self.assertRaisesRegex(ValueError, "frozen shared world model"):
            worker.run()
        self.assertEqual(self.resume().learner.cycles_completed, 0)

    def test_incomplete_handoff_publication_repairs_without_repeating_training(self):
        worker = self.prepare()
        with patch.object(worker, "_publish_frozen_teacher", side_effect=OSError("fixture handoff disk failure")):
            with self.assertRaisesRegex(OSError, "handoff disk failure"):
                worker.run()
        completed = self.resume()
        self.assertEqual(completed.learner.cycles_completed, 2)
        with patch.object(completed.learner, "update", side_effect=AssertionError("repeated completed training")):
            report = completed.run()
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["frozen_teacher_available"])
        handoff = self.directory / "frozen_teacher.json"
        handoff.write_text(handoff.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "handoff receipt mismatch"):
            completed.run()


if __name__ == "__main__":
    unittest.main()
