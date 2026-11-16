"""D adaptation against the real parent controller and tiny durable child stores.

No world model, optimizer, research replay or simulator is constructed.  The
parent's actual receipt/checkpoint/reconciliation code remains under test.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch

from world_models.cycle.checkpoints import load_cycle_checkpoint, save_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_paired import d_runner as adapter
from world_models_study.protocol import training_seed_manifest


class _Stages:
    def __init__(self, binding):
        self.binding = binding
        self.interrupt_after_commit = False
        self.research_calls = []

    def save_initial(self, directory, *, inputs, mode="fresh", distillation_run=None):
        directory = Path(directory)
        seed = inputs["master_seed"]
        seeds = training_seed_manifest(seed, "D")["paired_initialization"]
        metadata = {"schema_version": "p44-paired-initial-student-v1", "master_seed": seed,
            "inputs": deepcopy(inputs), "input_binding": deepcopy(self.binding),
            "paired_versions_cpu": {"actor": "a" * 64, "critic": "b" * 64},
            "provenance": {"mode": mode, "saved_before_teacher_training": True,
                           "chronology": "fresh_before_teacher"},
            "initialization_seeds": {name: seeds[name] for name in
                ("local_actor_initialization", "local_critic_initialization")},
            "actor_optimizer_steps": 0, "critic_optimizer_steps": 0,
            "new_environment_interactions": 0, "new_model_updates": 0}
        save_cycle_checkpoint(directory, state={"actor": {"weight": torch.tensor([1.0])},
                                               "critic": {"weight": torch.tensor([2.0])}}, metadata=metadata)
        return self.verify_initial(directory)

    def verify_initial(self, directory, *, distillation_run=None):
        directory = Path(directory)
        _, metadata = load_cycle_checkpoint(directory)
        return {"path": str(directory), "manifest_sha256": file_sha256(directory / "manifest.json"),
                "payload_sha256": file_sha256(directory / "state.pt"), "metadata": metadata}

    def prepare(self, name, directory, *, inputs, device, precision, **kwargs):
        directory = Path(directory)
        adapter.campaign._publish(directory / "run_manifest.json", {"inputs": inputs, "fixture": name})
        self._save(directory, {"completed": 0, "rng": 17})
        return {"directory": directory, "name": name}

    def resume(self, name, directory, *, device):
        return {"directory": Path(directory), "name": name}

    def _save(self, directory, state):
        checkpoint = directory / "checkpoints" / f"state_{state['completed'] + 1:08d}_{uuid.uuid4().hex[:8]}"
        save_cycle_checkpoint(checkpoint, state={"fixture": state}, metadata={"fixture": True})

    def run(self, name, worker):
        path = adapter._latest(worker["directory"])
        saved, _ = load_cycle_checkpoint(path)
        state = saved["fixture"]
        state["completed"] += 1
        state["rng"] = (state["rng"] * 1103515245 + 12345) % 2**31
        self.research_calls.append((name, state["completed"], state["rng"]))
        self._save(worker["directory"], state)
        if self.interrupt_after_commit:
            self.interrupt_after_commit = False
            raise OSError("fixture infrastructure failure after durable child commit")

    def inspect(self, name, directory):
        directory = Path(directory)
        checkpoint = adapter._latest(directory)
        saved, _ = load_cycle_checkpoint(checkpoint)
        completed = saved["fixture"]["completed"]
        manifest = adapter.campaign._read(directory / "run_manifest.json")
        return {"stage": name, "path": str(directory), "completed": completed, "target": 25,
            "checkpoint": str(checkpoint), "checkpoint_manifest_sha256": file_sha256(checkpoint / "manifest.json"),
            "manifest_sha256": file_sha256(directory / "run_manifest.json"), "manifest_identity": canonical_hash(manifest),
            "publications": [], "physical_work": {},
            "counters": {"model_updates": 4500, "real_joint_transitions": 50000,
                         "new_model_updates": 0, "new_environment_interactions": 0,
                         "teacher_actor_optimizer_steps": completed * 16,
                         "teacher_critic_optimizer_steps": completed * 16,
                         "imagined_joint_steps": completed}}


class PairedDRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.directory = self.root / "fresh_d"
        self.selection = self.root / "selection.json"
        normalization = {"identity_sha256": "c" * 64}
        replay = {"manifest_sha256": "d" * 64, "dataset_identity_sha256": "e" * 64, "transitions": 50000}
        self.binding = {"selection": {"task": "relay", "seed": 0, "model_updates": 4500,
            "shared_initial_variants": ["B", "D"], "checkpoint_sha256": "f" * 64,
            "selection_path": str(self.selection), "replays": [replay], "initial_normalization": normalization},
            "normalization": normalization, "provenance": {"replays": [replay], "initial_normalization": normalization},
            "replay_path": str(self.root / "replay"),
            "settings": {"task": {"name": "relay"}, "variant": {"name": "D"}}}
        self.stages = _Stages(self.binding)
        self.cohort = "1" * 64
        mocks = [patch.object(adapter.campaign, "source_identity", return_value={"fixture": "2" * 64}),
            patch.object(adapter.campaign, "selected_device", side_effect=lambda device, precision: torch.device("cpu")),
            patch.object(adapter.campaign, "_runtime", side_effect=lambda device: {"logical_device": str(device), "fixture": True}),
            patch.object(adapter.campaign.stages, "save_initial_student", side_effect=self.stages.save_initial),
            patch.object(adapter.campaign.stages, "verify_initial_student", side_effect=self.stages.verify_initial),
            patch.object(adapter.campaign.stages, "prepare_stage", side_effect=self.stages.prepare),
            patch.object(adapter.campaign.stages, "resume_stage", side_effect=self.stages.resume),
            patch.object(adapter.campaign.stages, "run_one", side_effect=self.stages.run),
            patch.object(adapter.campaign.stages, "inspect_stage", side_effect=self.stages.inspect)]
        for item in mocks:
            item.start()
            self.addCleanup(item.stop)

    def prepare(self, **kwargs):
        options = {"seed": 0, "initial_selection": self.selection, "cohort_identity": self.cohort,
                   "input_binding": self.binding, "device": "cpu"}
        options.update(kwargs)
        return adapter.PairedDRunner.prepare(self.directory, **options)

    def rewrite(self, name, value):
        path = self.directory / name
        path.unlink()
        path.with_suffix(".sha256").unlink()
        adapter.campaign._publish(path, value)

    def test_prepare_is_fresh_receipted_and_bound_to_paired_inputs_without_training(self):
        worker = self.prepare()
        self.assertEqual(self.stages.research_calls, [])
        self.assertEqual(worker.worker.manifest["mode"], "fresh")
        self.assertEqual(worker.worker.manifest["precision"], "bfloat16")
        self.assertFalse(worker.manifest["fresh_cohort_eligible"])
        self.assertEqual(worker.pairing(), {"actor_cpu": "a" * 64, "critic_cpu": "b" * 64,
            "input_binding_sha256": canonical_hash(self.binding), "world_model_checkpoint_sha256": "f" * 64,
            "normalization_identity_sha256": "c" * 64, "replay_manifest_sha256": "d" * 64})
        status = adapter.inspect(self.directory, cohort_identity=self.cohort, input_binding=self.binding)
        self.assertEqual((status["units_completed"], status["target_units"]), (0, 274))
        self.assertEqual(status["target_teacher_cycles"], 25)
        self.assertEqual(status["target_trajectories"], 128)
        self.assertEqual(status["target_distillation_optimizer_steps"], 100)
        self.assertEqual(status["target_local_cycles"], 20)
        self.assertEqual(status["new_environment_interactions"], 0)

    def test_resume_preserves_committed_child_rng_and_never_repeats_interrupted_work(self):
        worker = self.prepare()
        self.stages.interrupt_after_commit = True
        with self.assertRaisesRegex(OSError, "durable child"):
            worker.run(max_units=1)
        self.assertEqual(len(self.stages.research_calls), 1)
        before = adapter.inspect(self.directory)
        self.assertEqual(before["units_completed"], 0)
        resumed = adapter.PairedDRunner.resume(self.directory, device="cpu", cohort_identity=self.cohort,
                                               input_binding=self.binding)
        resumed.run(max_units=1)
        self.assertEqual(len(self.stages.research_calls), 1)
        self.assertEqual(adapter.inspect(self.directory)["teacher_cycles_completed"], 1)
        resumed.run(max_units=1)
        self.assertEqual([row[1] for row in self.stages.research_calls], [1, 2])
        saved, _ = load_cycle_checkpoint(adapter._latest(self.directory / "stages/teacher"))
        self.assertEqual(saved["fixture"]["rng"], self.stages.research_calls[-1][2])

    def test_stop_delegates_to_parent_without_any_work(self):
        worker = self.prepare()
        worker.request_stop()
        status = worker.run(max_units=1)
        self.assertEqual(status["status"], "stopped")
        self.assertEqual(self.stages.research_calls, [])
        self.assertEqual(status["cohort_identity"], self.cohort)

    def test_prepare_rejects_wrong_seed_precision_and_input_before_parent_creation(self):
        for changes in ({"seed": True}, {"seed": 2}, {"precision": "float32"}, {"cohort_identity": "bad"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.prepare(**changes)
            self.assertFalse(self.directory.exists())
        binding = deepcopy(self.binding)
        binding["selection"]["model_updates"] = 4000
        with self.assertRaises(ValueError):
            self.prepare(input_binding=binding)
        self.assertFalse(self.directory.exists())

    def test_no_historical_adoption_or_changed_manifest_can_be_promoted(self):
        worker = self.prepare()
        manifest = deepcopy(worker.worker.manifest)
        manifest["mode"] = "adopted_development"
        self.rewrite("run_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "fresh"):
            adapter.PairedDRunner.resume(self.directory, device="cpu")
        self.assertEqual(self.stages.research_calls, [])

    def test_resume_rejects_changed_pairing_rng_and_input_even_with_new_binding_receipt(self):
        worker = self.prepare()
        original = deepcopy(worker.manifest)
        for field, changed in (("paired_initialization", {"actor_cpu": "0" * 64}),
                               ("training_seeds", {"master_seed": 0}),
                               ("input_binding", {"replaced": True})):
            with self.subTest(field=field):
                modified = deepcopy(original)
                modified[field] = changed
                self.rewrite(adapter.BINDING_NAME, modified)
                with self.assertRaises(ValueError):
                    adapter.PairedDRunner.resume(self.directory, device="cpu")
        self.assertEqual(self.stages.research_calls, [])

    def test_resume_rejects_other_cohort_or_missing_receipt_and_run_checks_again(self):
        worker = self.prepare()
        with self.assertRaisesRegex(ValueError, "another P5.2 cohort"):
            adapter.PairedDRunner.resume(self.directory, device="cpu", cohort_identity="9" * 64)
        (self.directory / "d_binding.sha256").unlink()
        with self.assertRaisesRegex(ValueError, "receipt"):
            worker.run(max_units=1)
        self.assertEqual(self.stages.research_calls, [])

    def test_fresh_campaign_cannot_be_attached_to_a_preexisting_directory(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(self.stages.research_calls, [])

    def test_inspection_uses_committed_state_and_does_not_trust_mutable_status(self):
        self.prepare()
        (self.directory / "status.json").write_text(json.dumps({"status": "complete", "units_completed": 274}))
        with patch.object(adapter.campaign, "selected_device", side_effect=AssertionError("GPU probe forbidden")):
            result = adapter.inspect(self.directory)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["units_completed"], 0)
        self.assertEqual(self.stages.research_calls, [])

    def test_final_parent_commit_requires_valid_actor_only_and_completion_publications(self):
        prepared = self.prepare()
        exported = self.directory / "actor_export"
        save_cycle_checkpoint(exported, state={"actor": {"weight": torch.tensor([3.0])}}, metadata={"fixture": True})
        export = {"path": str(exported), "manifest_sha256": file_sha256(exported / "manifest.json"),
                  "payload_sha256": file_sha256(exported / "state.pt")}
        counts = {"model_updates": 4500, "real_joint_transitions": 50000,
                  "new_model_updates": 0, "new_environment_interactions": 0}
        worker = SimpleNamespace(directory=self.directory, manifest=prepared.worker.manifest,
            state={"export": export, "units_completed": 274}, last_checkpoint=prepared.worker.last_checkpoint,
            _counts=lambda: deepcopy(counts))
        self.assertFalse(adapter._verify_completion(worker))
        accounting = {"schema_version": "p44-d-accounting-v1", "run_identity": canonical_hash(worker.manifest),
                      "mode": "fresh", "logical_units_completed": 274, **counts}
        adapter.campaign._publish(self.directory / "accounting.json", accounting)
        completion = {"schema_version": "p44-completed-d-student-v1", "run_identity": canonical_hash(worker.manifest),
            "mode": "fresh", "actor_export": export, "paired_initial_student": worker.manifest["initial_student"],
            "accounting_sha256": file_sha256(self.directory / "accounting.json"),
            "checkpoint": str(worker.last_checkpoint),
            "checkpoint_manifest_sha256": file_sha256(worker.last_checkpoint / "manifest.json"),
            "fresh_cohort_eligible": False, **counts}
        adapter.campaign._publish(self.directory / "completed_student.json", completion)
        self.assertTrue(adapter._verify_completion(worker))
        completion["new_model_updates"] = 1
        self.rewrite("completed_student.json", completion)
        with self.assertRaisesRegex(ValueError, "accounting"):
            adapter._verify_completion(worker)

    def test_recover_prepared_finishes_missing_external_receipt_and_preserves_partial_bytes(self):
        publish = adapter.campaign._publish

        def fail_external(path, content):
            if Path(path).name == adapter.BINDING_NAME:
                raise OSError("fixture interrupted external binding publication")
            return publish(path, content)

        with patch.object(adapter.campaign, "_publish", side_effect=fail_external):
            with self.assertRaisesRegex(OSError, "external binding"):
                self.prepare()
        checkpoint = adapter._latest(self.directory)
        checkpoint_hash = file_sha256(checkpoint / "state.pt")
        partial = b'{"interrupted":'
        (self.directory / adapter.BINDING_NAME).write_bytes(partial)
        recovered = adapter.PairedDRunner.recover_prepared(self.directory, device="cpu",
            cohort_identity=self.cohort, input_binding=self.binding)
        self.assertEqual(recovered.worker.state["sequence"], 1)
        self.assertEqual(adapter._latest(self.directory), checkpoint)
        self.assertEqual(file_sha256(checkpoint / "state.pt"), checkpoint_hash)
        preserved = list((self.directory / "uncommitted").glob("d_binding_*.json"))
        self.assertEqual([path.read_bytes() for path in preserved], [partial])
        self.assertEqual(recovered.pairing()["input_binding_sha256"], canonical_hash(self.binding))
        self.assertEqual(self.stages.research_calls, [])
        again = adapter.PairedDRunner.recover_prepared(self.directory, device="cpu",
            cohort_identity=self.cohort, input_binding=self.binding)
        self.assertEqual(again.manifest, recovered.manifest)
        self.assertEqual(len(list((self.directory / "uncommitted").iterdir())), 1)

    def test_recover_prepared_rejects_any_research_progress_child_store_or_attempt(self):
        for kind in ("trained", "child_store", "attempted"):
            with self.subTest(kind=kind):
                self.directory = self.root / kind
                worker = self.prepare()
                if kind == "trained":
                    worker.run(max_units=1)
                elif kind == "child_store":
                    (self.directory / "stages/teacher").mkdir(parents=True)
                else:
                    (self.directory / "attempts.jsonl").write_text('{"event":"unit_started"}\n')
                before = list(self.stages.research_calls)
                with self.assertRaisesRegex(ValueError, "zero-work"):
                    adapter.PairedDRunner.recover_prepared(self.directory, device="cpu",
                        cohort_identity=self.cohort, input_binding=self.binding)
                self.assertEqual(self.stages.research_calls, before)

    def test_recover_prepared_rejects_wrong_initial_inputs_adoption_and_invalid_committed_receipt(self):
        worker = self.prepare()
        receipt = self.directory / "d_binding.sha256"
        receipt.unlink()
        original_bytes = (self.directory / adapter.BINDING_NAME).read_bytes()
        binding = deepcopy(self.binding)
        binding["selection"]["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "input binding"):
            adapter.PairedDRunner.recover_prepared(self.directory, device="cpu",
                cohort_identity=self.cohort, input_binding=binding)
        self.assertEqual((self.directory / adapter.BINDING_NAME).read_bytes(), original_bytes)
        receipt.write_text("invalid committed receipt\n")
        with self.assertRaisesRegex(ValueError, "receipt"):
            adapter.PairedDRunner.recover_prepared(self.directory, device="cpu",
                cohort_identity=self.cohort, input_binding=self.binding)
        receipt.unlink()
        manifest = deepcopy(worker.worker.manifest)
        manifest["mode"] = "adopted_development"
        self.rewrite("run_manifest.json", manifest)
        with self.assertRaises(ValueError):
            adapter.PairedDRunner.recover_prepared(self.directory, device="cpu",
                cohort_identity=self.cohort, input_binding=self.binding)
        self.assertEqual(self.stages.research_calls, [])


if __name__ == "__main__":
    unittest.main()
