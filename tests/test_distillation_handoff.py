"""CPU fixture audits for completed-teacher input; no research runs or weights."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from tests import test_teacher_runner as teacher_fixture
from world_models.cycle.checkpoints import load_cycle_checkpoint, save_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.distillation.handoff import load_completed_teacher
from world_models.imagination.versions import module_version
from world_models.teacher.runner import source_identity


class DistillationHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        teacher_fixture.TeacherRunnerTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        teacher_fixture.TeacherRunnerTests.tearDownClass.__func__(cls)

    def setUp(self):
        teacher_fixture.TeacherRunnerTests.setUp(self)

    prepare = teacher_fixture.TeacherRunnerTests.prepare

    def completed(self):
        # Simulate an already running P4.1 registration made before P4.2 was
        # added. Import accepts new distillation modules, exact resume does not.
        historical = {name: value for name, value in source_identity().items()
                      if not name.startswith("src/world_models/distillation/")}
        with patch("world_models.teacher.runner.source_identity", return_value=historical):
            worker = self.prepare()
            worker.run()
        return worker

    def rewrite_json(self, name, value):
        path = self.directory / name
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        path.with_suffix(".sha256").write_text(f"{file_sha256(path)}  {path.name}\n", encoding="ascii")

    def replacement(self, worker, saved, metadata):
        original = worker.last_checkpoint
        replacement = original.with_name(f"state_{saved['controller']['sequence']:08d}_aabbccdd")
        save_cycle_checkpoint(replacement, state=saved, metadata=metadata)
        original.joinpath("manifest.sha256").unlink()
        handoff = json.loads((self.directory / "frozen_teacher.json").read_text())
        handoff["checkpoint"] = replacement.relative_to(self.directory).as_posix()
        handoff["checkpoint_manifest_sha256"] = file_sha256(replacement / "manifest.json")
        self.rewrite_json("frozen_teacher.json", handoff)

    def test_completed_handoff_is_frozen_read_only_and_does_not_construct_optimizers(self):
        worker = self.completed()
        before = {path.relative_to(self.directory).as_posix(): file_sha256(path)
                  for path in self.directory.rglob("*") if path.is_file()}
        rng = torch.get_rng_state().clone()
        with patch("torch.optim.Adam", side_effect=AssertionError("handoff constructed an optimizer")):
            context = load_completed_teacher(self.directory, device="cpu")
        self.assertEqual(before, {path.relative_to(self.directory).as_posix(): file_sha256(path)
                                 for path in self.directory.rglob("*") if path.is_file()})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(context.master_seed, 0)
        self.assertEqual(context.precision, "float32")
        self.assertEqual(context.binding, worker.manifest["input_binding"])
        self.assertEqual(context.provenance["teacher_cycles"], 2)
        self.assertFalse(context.provenance["cross_runtime_numerical_equivalence_measured"])
        for network, original in ((context.model, worker.model), (context.teacher, worker.teacher),
                                  (context.critic, worker.critic)):
            self.assertEqual(module_version(network), module_version(original))
            self.assertFalse(network.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in network.parameters()))
        self.assertFalse(hasattr(context, "optimizer"))

    def test_incomplete_run_cannot_be_loaded(self):
        self.prepare().run(max_cycles=1)
        with self.assertRaisesRegex(ValueError, "frozen_teacher.json receipt"):
            load_completed_teacher(self.directory)

    def test_both_publication_receipts_are_required(self):
        self.completed()
        for name in ("run_manifest", "frozen_teacher"):
            path = self.directory / f"{name}.sha256"
            original = path.read_text()
            path.write_text("0" * 64 + "  wrong.json\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "valid.*receipt"):
                load_completed_teacher(self.directory)
            path.write_text(original)

    def test_changed_recorded_sources_and_omitted_dependencies_are_rejected(self):
        worker = self.completed()
        name = "src/world_models/teacher/runner.py"
        changed = deepcopy(worker.manifest)
        changed["source_sha256"][name] = "0" * 64
        self.rewrite_json("run_manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "teacher source changed"):
            load_completed_teacher(self.directory)
        changed = deepcopy(worker.manifest)
        del changed["source_sha256"][name]
        self.rewrite_json("run_manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "omits an existing dependency"):
            load_completed_teacher(self.directory)

    def test_handoff_must_reference_latest_committed_checkpoint(self):
        self.completed()
        handoff = json.loads((self.directory / "frozen_teacher.json").read_text())
        handoff["checkpoint"] = "../../unrelated"
        self.rewrite_json("frozen_teacher.json", handoff)
        with self.assertRaisesRegex(ValueError, "latest committed checkpoint"):
            load_completed_teacher(self.directory)

    def test_corrupted_checkpoint_payload_fails_before_weights_are_used(self):
        worker = self.completed()
        path = worker.last_checkpoint / "state.pt"
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"x")
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            load_completed_teacher(self.directory)

    def test_valid_receipts_do_not_hide_incomplete_logical_counters(self):
        worker = self.completed()
        saved, metadata = load_cycle_checkpoint(worker.last_checkpoint)
        saved["learner"]["cycles_completed"] -= 1
        self.replacement(worker, saved, metadata)
        with self.assertRaisesRegex(ValueError, "optimizer state does not describe completed"):
            load_completed_teacher(self.directory)

    def test_recorded_optimizer_slots_must_match_completed_step_counter(self):
        worker = self.completed()
        saved, metadata = load_cycle_checkpoint(worker.last_checkpoint)
        saved["learner"]["actor_optimizer"]["state"][0]["step"].sub_(1)
        self.replacement(worker, saved, metadata)
        with self.assertRaisesRegex(ValueError, "Adam step differs"):
            load_completed_teacher(self.directory)

    def test_handoff_teacher_identity_is_rejected(self):
        worker = self.completed()
        handoff = json.loads((self.directory / "frozen_teacher.json").read_text())
        handoff["teacher_version"] = "0" * 64
        self.rewrite_json("frozen_teacher.json", handoff)
        with self.assertRaisesRegex(ValueError, "handoff differs"):
            load_completed_teacher(self.directory)

    def test_changed_frozen_model_is_rejected(self):
        worker = self.completed()
        saved, metadata = load_cycle_checkpoint(worker.last_checkpoint)
        key = next(key for key, value in saved["model"].items() if value.is_floating_point())
        saved["model"][key].add_(0.1)
        self.replacement(worker, saved, metadata)
        with self.assertRaisesRegex(ValueError, "changed its frozen initial world model"):
            load_completed_teacher(self.directory)

    def test_initialization_and_replay_binding_are_revalidated(self):
        worker = self.completed()
        changed = deepcopy(worker.manifest)
        changed["initial_teacher_version"] = "0" * 64
        self.rewrite_json("run_manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "initial model, policy or critic identity"):
            load_completed_teacher(self.directory)
        changed = deepcopy(worker.manifest)
        changed["input_binding"]["normalization"]["identity_sha256"] = "0" * 64
        self.rewrite_json("run_manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "frozen replay, normalization, selection"):
            load_completed_teacher(self.directory)


if __name__ == "__main__":
    unittest.main()
