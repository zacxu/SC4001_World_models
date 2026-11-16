"""Read-only completed P4.2 import; no research training or native environment."""
from copy import copy, deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests import test_distillation_runner as fixture
from tests.test_distillation_device_identity import legacy_hash_at_device
from world_models.cycle.checkpoints import load_cycle_checkpoint, save_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.distillation.versions import module_version_at_device
from world_models.imagination.versions import module_version
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.refinement.handoff import (ROOT, _source_compatibility,
    load_warm_started_student, verify_frozen_inputs)
from world_models.teacher.runner import _publish_receipt


class RefinementHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.DistillationRunnerTests.setUpClass.__func__(cls)
        owner = cls("test_import_is_read_only_local_and_never_restores_a_teacher_or_optimizer")
        owner.enterContext(patch("world_models.distillation.handoff._source_compatibility"))
        fixture.DistillationRunnerTests.setUp(owner)
        cls.addClassCleanup(owner.doCleanups)
        cls.template_worker = fixture.DistillationRunnerTests.prepare(owner)
        cls.template_worker.run()
        cls.template_directory = owner.directory
        cls.shared_teacher_directory = owner.teacher_directory

    @classmethod
    def tearDownClass(cls):
        fixture.DistillationRunnerTests.tearDownClass.__func__(cls)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "distillation"
        self.teacher_directory = self.shared_teacher_directory
        # Only the fixture's deliberately fake source registration is exempt.
        # Dedicated tests below exercise the actual historical source guard.
        self.compatibility = self.enterContext(patch("world_models.refinement.handoff._source_compatibility"))

    def completed(self):
        shutil.copytree(self.template_directory, self.directory)
        worker = copy(self.template_worker)
        worker.directory = self.directory
        worker.last_checkpoint = self.directory / self.template_worker.last_checkpoint.relative_to(self.template_directory)
        return worker

    def rewrite_json(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        receipt = path.with_suffix(".sha256")
        if receipt.exists():
            receipt.unlink()
        _publish_receipt(path)

    def replacement(self, worker, saved, metadata):
        original = worker.last_checkpoint
        path = original.with_name(f"state_{saved['controller']['sequence']:08d}_abababab")
        save_cycle_checkpoint(path, state=saved, metadata=metadata)
        original.joinpath("manifest.sha256").unlink()
        handoff_path = self.directory / "warm_started_student.json"
        handoff = json.loads(handoff_path.read_text())
        handoff.update(checkpoint=path.relative_to(self.directory).as_posix(),
                       checkpoint_manifest_sha256=file_sha256(path / "manifest.json"))
        self.rewrite_json(handoff_path, handoff)
        return path

    def test_import_is_read_only_local_and_never_restores_a_teacher_or_optimizer(self):
        worker = self.completed()
        before = {str(path): file_sha256(path) for root in (self.directory, self.teacher_directory)
                  for path in root.rglob("*") if path.is_file()}
        rng = torch.get_rng_state().clone()
        with patch("torch.optim.Adam", side_effect=AssertionError("restored optimizer")), \
                patch("world_models.teacher.runner._policies", side_effect=AssertionError("constructed teacher")), \
                patch("world_models.policies.teacher.GlobalTeacher.forward", side_effect=AssertionError("queried teacher")):
            context = load_warm_started_student(self.directory, device="cpu")
            verify_frozen_inputs(context)
        self.assertEqual(module_version(context.actor), module_version(worker.actor))
        self.assertEqual(module_version(context.critic), worker.paired_versions["critic"])
        self.assertEqual(module_version(context.model), worker.frozen_versions["model"])
        self.assertEqual(context.binding, worker.context.binding)
        self.assertTrue(all(parameter.requires_grad for parameter in context.actor.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in context.critic.parameters()))
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in context.model.parameters()))
        self.assertFalse(any(hasattr(context, name) for name in ("teacher", "optimizer", "buffer", "learner")))
        self.assertFalse(context.provenance["upstream_optimizers_restored"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, {str(path): file_sha256(path) for root in (self.directory, self.teacher_directory)
                                  for path in root.rglob("*") if path.is_file()})

    def test_initial_checkpoint_overrides_changed_random_initializer(self):
        worker = self.completed()
        from world_models.policies.initialization import build_actor_critic

        def changed_initialization(*args, **kwargs):
            result = build_actor_critic(*args, **kwargs)
            with torch.no_grad():
                next(result.actor.parameters()).add_(0.1)
                next(result.critic.parameters()).add_(0.1)
            return result

        with patch("world_models.refinement.handoff.build_actor_critic", side_effect=changed_initialization):
            context = load_warm_started_student(self.directory)
        self.assertEqual(module_version(context.actor), module_version(worker.actor))
        self.assertEqual(module_version(context.critic), module_version(worker.critic))

    def test_status_is_informational_and_incomplete_checkpoints_are_ignored(self):
        self.completed()
        (self.directory / "status.json").write_text('{"status":"failed","distillation_optimizer_steps":999}')
        incomplete = self.directory / "checkpoints/state_99999999_abababab"
        incomplete.mkdir()
        (incomplete / "state.pt").write_bytes(b"incomplete")
        context = load_warm_started_student(self.directory)
        self.assertEqual(context.handoff["distillation_optimizer_steps"], 2)

    def test_missing_completion_handoff_is_rejected(self):
        worker = fixture.DistillationRunnerTests.prepare(self)
        worker.run(max_units=1)
        with self.assertRaisesRegex(ValueError, "receipt"):
            load_warm_started_student(self.directory)

    def test_all_upstream_receipts_are_required(self):
        self.completed()
        for path in (self.directory / "run_manifest.sha256", self.directory / "warm_started_student.sha256",
                     self.teacher_directory / "frozen_teacher.sha256"):
            original = path.read_bytes()
            path.write_text("0" * 64 + "  changed.json\n")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "receipt"):
                load_warm_started_student(self.directory)
            path.write_bytes(original)

    def test_latest_committed_checkpoint_corruption_cannot_fall_back(self):
        worker = self.completed()
        with (worker.last_checkpoint / "state.pt").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "payload length mismatch"):
            load_warm_started_student(self.directory)

    def test_valid_receipts_do_not_allow_changed_critic_or_incomplete_adam(self):
        for damage in ("critic", "adam", "rng", "counter"):
            with self.subTest(damage=damage):
                worker = self.completed()
                saved, metadata = load_cycle_checkpoint(worker.last_checkpoint)
                if damage == "critic":
                    next(iter(saved["critic"].values())).add_(0.1)
                    expected = "paired initial student critic"
                elif damage == "adam":
                    saved["learner"]["optimizer"]["state"][0]["step"].sub_(1)
                    expected = "Adam step"
                elif damage == "rng":
                    saved["learner"]["sampling_rng"] = torch.Generator().manual_seed(77).get_state()
                    expected = "sampling RNG"
                else:
                    saved["controller"]["trajectories_completed"] -= 1
                    expected = "completed fixed-schedule"
                self.replacement(worker, saved, metadata)
                with self.assertRaisesRegex(ValueError, expected):
                    load_warm_started_student(self.directory)
                # Every subcase owns a fresh output folder, retaining damaged
                # publications so no test accidentally repairs historical bytes.
                self.directory = Path(self.temp.name) / f"distillation_{damage}"

    def test_buffer_and_frozen_inputs_remain_bound_after_import(self):
        worker = self.completed()
        context = load_warm_started_student(self.directory)
        shard = self.directory / worker.shards[0]["path"] / "state.pt"
        with shard.open("ab") as stream:
            stream.write(b"changed after import")
        with self.assertRaisesRegex(ValueError, "input changed"):
            verify_frozen_inputs(context)
        with self.assertRaisesRegex(ValueError, "payload length mismatch"):
            load_warm_started_student(self.directory)

    def test_exact_recorded_sources_required_but_new_refinement_files_are_allowed(self):
        self.compatibility.stop()
        recorded = {path.relative_to(ROOT).as_posix(): file_sha256(path)
                    for path in (ROOT / "src/world_models").rglob("*.py")
                    if not path.relative_to(ROOT).as_posix().startswith("src/world_models/refinement/")}
        recorded.update({name: file_sha256(ROOT / name) for name in ("scripts/phase41.py", "scripts/phase42.py")})
        _source_compatibility(recorded, stage="distillation")
        altered = deepcopy(recorded)
        altered["src/world_models/distillation/runner.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source changed"):
            _source_compatibility(altered, stage="distillation")
        omitted = deepcopy(recorded)
        del omitted["src/world_models/policies/actor.py"]
        with self.assertRaisesRegex(ValueError, "omits an existing dependency"):
            _source_compatibility(omitted, stage="distillation")

    def test_cuda_labelled_saved_student_imports_without_cuda_discovery(self):
        worker = self.completed()
        manifest = deepcopy(worker.manifest)
        initial_path = next((self.directory / "checkpoints").glob("state_00000001_*"))
        initial, initial_metadata = load_cycle_checkpoint(initial_path)
        saved, metadata = load_cycle_checkpoint(worker.last_checkpoint)
        label = "cuda:0"
        initial_actor, initial_critic = deepcopy(worker.actor), deepcopy(worker.critic)
        initial_actor.load_state_dict(initial["actor"])
        initial_critic.load_state_dict(initial["critic"])
        paired = {name: legacy_hash_at_device(network, label) for name, network in
                  (("actor", initial_actor), ("critic", initial_critic))}
        frozen = {name: legacy_hash_at_device(network, label) for name, network in worker._frozen().items()}
        manifest.update(paired_initial_versions=paired, frozen_versions=frozen)
        manifest["runtime"]["logical_device"] = label
        manifest["rollout_stream_identity"]["device"] = label
        self.rewrite_json(self.directory / "run_manifest.json", manifest)
        identity = canonical_hash(manifest)
        for index, row in enumerate(saved["shards"]):
            path = self.directory / row["path"]
            payload, shard_metadata = load_cycle_checkpoint(path)
            shard_metadata.update(run_identity=identity, frozen_versions=frozen)
            payload["buffer"]["metadata"].update(teacher_version=frozen["teacher"], world_model_version=frozen["model"])
            for side in ("before", "after"):
                payload["buffer"]["streams"][0][side]["identity"]["device"] = label
            replacement = path.with_name(f"cuda_trajectory_{index}")
            save_cycle_checkpoint(replacement, state=payload, metadata=shard_metadata)
            row.update(path=replacement.relative_to(self.directory).as_posix(),
                       manifest_sha256=file_sha256(replacement / "manifest.json"))
        final_versions = {"actor": legacy_hash_at_device(worker.actor, label), "critic": paired["critic"]}
        for state, meta, versions in ((initial, initial_metadata, paired), (saved, metadata, final_versions)):
            state["learner"].update(paired_actor_version=paired["actor"], paired_critic_version=paired["critic"],
                                    current_actor_version=versions["actor"])
            state["streams"]["identity"]["device"] = label
            meta.update(run_identity=identity, actor_version=versions["actor"], critic_version=versions["critic"])
        new_initial = initial_path.with_name("state_00000001_abcdabcd")
        save_cycle_checkpoint(new_initial, state=initial, metadata=initial_metadata)
        initial_path.joinpath("manifest.sha256").unlink()
        final_path = self.replacement(worker, saved, metadata)
        handoff_path = self.directory / "warm_started_student.json"
        handoff = json.loads(handoff_path.read_text())
        handoff.update(run_identity=identity, actor_version=final_versions["actor"], critic_version=paired["critic"],
                       paired_initial_versions=paired, frozen_versions=frozen)
        self.rewrite_json(handoff_path, handoff)
        with patch("torch.cuda.is_available", side_effect=AssertionError("CPU handoff discovered CUDA")):
            context = load_warm_started_student(self.directory, device="cpu")
        self.assertEqual(context.checkpoint, final_path)
        self.assertEqual(context.provenance["verified_origin_device"], label)
        self.assertEqual(module_version_at_device(context.actor, label), final_versions["actor"])
        self.assertEqual(next(context.actor.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
