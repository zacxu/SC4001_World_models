"""P4.4 orchestration invariants using tiny, durable stage-adapter fixtures.

The fake stages commit deterministic counters and private RNG state to disk;
they never construct a model, generate research imagination or open ViZDoom.
The separate adapter suite covers real numerical stage interoperability.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from world_models.cycle.checkpoints import save_cycle_checkpoint
from world_models.data.ledger import atomic_json, file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.training.runner import run_lock
from world_models_campaign import controller as campaign
from world_models_campaign.protocol import CAMPAIGN_PROTOCOL


def receipt(path, content):
    atomic_json(path, content)
    path.with_suffix(".sha256").write_text(f"{file_sha256(path)}  {path.name}\n", encoding="ascii")


class DurableStages:
    """Small adapter stand-in with real durable, receipt-bound progress."""

    def __init__(self, case):
        self.case = case
        self.calls = []
        self.input_binding = {"selected_model": "a" * 64, "replay": "b" * 64,
                              "normalization": "c" * 64}
        self.targets = {"teacher": 2, "buffer": 2, "distillation": 2, "local_cycle": 2}
        self.initial_record = None
        self.interrupt_after_commit = False
        self.adopted_paths = None
        self.frozen_changed = False

    def validate_initial_inputs(self, inputs):
        if self.frozen_changed:
            return {}, {"changed": "binding"}
        return {"task": {"name": "relay"}, "variant": {"name": "D"}}, deepcopy(self.input_binding)

    def save_initial_student(self, directory, *, inputs, mode="fresh", distillation_run=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.calls.append(("initial_snapshot", mode))
        record = {"schema_version": "fixture-paired-initial-v1", "seed": inputs["master_seed"],
                  "actor_version": "d" * 64, "critic_version": "e" * 64,
                  "mode": mode, "optimizer_steps": 0, "teacher_work_preceded_snapshot": mode != "fresh"}
        receipt(directory / "paired_initial_student.json", record)
        self.initial_record = record
        return deepcopy(record)

    def verify_initial_student(self, directory, *, distillation_run=None):
        if self.frozen_changed:
            raise ValueError("fixture frozen input binding changed")
        path = Path(directory) / "paired_initial_student.json"
        self.case.assertEqual(path.with_suffix(".sha256").read_text().strip(),
                              f"{file_sha256(path)}  {path.name}")
        return json.loads(path.read_text())

    def _state_path(self, directory):
        return Path(directory) / "fixture_state.json"

    def _checkpoint(self, directory, state):
        sequence = 1 + sum(state[key] for key in self.targets)
        checkpoint = Path(directory) / "checkpoints" / f"state_{sequence:08d}_fixture"
        if not checkpoint.exists():
            save_cycle_checkpoint(checkpoint, state={"fixture": state}, metadata={"fixture": True})
        return checkpoint

    def prepare_stage(self, name, directory, *, inputs, device, precision="bfloat16",
                      teacher_run=None, distillation_run=None):
        self.case.assertIsNotNone(self.initial_record, "teacher started before paired student snapshot")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._state_path(directory).exists():
            raise FileExistsError("fixture stage already exists")
        self.calls.append(("prepare", name))
        receipt(directory / "run_manifest.json", {"fixture": name, "inputs": inputs, "precision": precision,
            "master_seed": inputs["master_seed"],
            "teacher_run": str(teacher_run) if teacher_run is not None else None,
            "distillation_run": str(distillation_run) if distillation_run is not None else None})
        receipt(self._state_path(directory), {"teacher": 0, "buffer": 0, "distillation": 0,
                                              "local_cycle": 0, "rng": 17, "attempts": []})
        self._checkpoint(directory, json.loads(self._state_path(directory).read_text()))
        return {"name": name, "directory": directory}

    def resume_stage(self, name, directory, *, device):
        self.calls.append(("resume", name))
        return {"name": name, "directory": Path(directory)}

    def run_one(self, name, worker):
        self.calls.append(("run", name))
        path = self._state_path(worker["directory"])
        state = json.loads(path.read_text())
        self.case.assertLess(state[name], self.targets[name], "completed logical unit repeated")
        state[name] += 1
        state["rng"] = (state["rng"] * 1103515245 + 12345) % (2 ** 31)
        state["attempts"].append({"stage": name, "ordinal": state[name], "rng": state["rng"]})
        receipt(path, state)
        self._checkpoint(worker["directory"], state)
        if self.interrupt_after_commit:
            self.interrupt_after_commit = False
            raise OSError("fixture controller lost after durable child commit")
        return self.inspect_stage(name, worker["directory"])

    def inspect_stage(self, name, directory):
        directory = Path(directory)
        path = self._state_path(directory)
        state = json.loads(path.read_text())
        self.case.assertEqual(path.with_suffix(".sha256").read_text().strip(),
                              f"{file_sha256(path)}  {path.name}")
        count = state[name]
        checkpoint = self._checkpoint(directory, state)
        counters = {"real_joint_transitions": 50000, "model_updates": 4500,
                    "new_environment_interactions": 0, "new_model_updates": 0,
                    "teacher_actor_optimizer_steps": state["teacher"],
                    "teacher_critic_optimizer_steps": state["teacher"],
                    "distillation_optimizer_steps": state["distillation"],
                    "local_actor_optimizer_steps": state["local_cycle"],
                    "local_critic_optimizer_steps": state["local_cycle"],
                    "imagined_joint_steps": state["teacher"] + state["buffer"] + state["local_cycle"]}
        publications = []
        handoff_name = {"teacher": "frozen_teacher.json", "distillation": "warm_started_student.json"}.get(name)
        if handoff_name and count == self.targets[name]:
            handoff_path = directory / handoff_name
            if not handoff_path.exists():
                receipt(handoff_path, {"fixture": name, "complete": True})
            publications.append({"path": str(handoff_path), "sha256": file_sha256(handoff_path)})
        if name == "local_cycle" and count == self.targets[name]:
            export = directory / "actor_export"
            if not (export / "manifest.json").exists():
                save_cycle_checkpoint(export, state={"actor": {"fixture.weight": torch.tensor([float(state["rng"])])}},
                                      metadata={"fixture": "local-actor-only"})
            handoff = directory / "refined_student.json"
            if not handoff.exists():
                receipt(handoff, {"actor_export": "actor_export", "actor_export_manifest_sha256":
                                  file_sha256(export / "manifest.json")})
            publications = [{"path": str(handoff), "sha256": file_sha256(handoff)}]
        manifest = json.loads((directory / "run_manifest.json").read_text())
        return {"stage": name, "path": str(directory), "completed": count, "target": self.targets[name],
                "checkpoint": str(checkpoint),
                "checkpoint_manifest_sha256": file_sha256(checkpoint / "manifest.json"),
                "manifest_sha256": file_sha256(directory / "run_manifest.json"),
                "manifest_identity": canonical_hash(manifest), "counters": counters,
                "publications": publications,
                "physical_work": {"journal": str(directory / "attempts.jsonl"),
                                  "started_units": count, "committed_units": count,
                                  "recorded_elapsed_seconds": float(count), "unmatched_attempts": [],
                                  "partial_interrupted_compute": "none recorded"}}

    def adoption_lineage(self, refinement_run, *, device="cpu"):
        if self.adopted_paths is None:
            raise ValueError("fixture has no completed refinement")
        return {"inputs": deepcopy(self.inputs), "paths": deepcopy(self.adopted_paths),
                "provenance": {"validated_completed_lineage": True, "development": True}}


class CampaignControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "campaign"
        self.enterContext(patch.dict(CAMPAIGN_PROTOCOL, {"teacher_cycles": 2, "trajectories": 2,
            "distillation_updates": 2, "local_cycles": 2, "epochs": 1, "minibatches": 1}))
        self.backend = DurableStages(self)
        self.inputs = {"initial_selection": str(self.root / "selection.json"), "master_seed": 0,
                       "config_root": str(self.root / "configs"), "data_root": str(self.root / "replay"),
                       "statistics_root": str(self.root / "normalization")}
        self.backend.inputs = self.inputs
        for name in ("validate_initial_inputs", "save_initial_student", "verify_initial_student",
                     "prepare_stage", "resume_stage", "run_one", "inspect_stage", "adoption_lineage"):
            self.enterContext(patch.object(campaign.stages, name, getattr(self.backend, name)))
        self.sources = {"fixture-source.py": "f" * 64}
        self.enterContext(patch.object(campaign, "source_identity", return_value=self.sources))
        self.enterContext(patch.object(campaign, "_runtime", return_value={"logical_device": "cpu", "fixture": 1}))

    def prepare(self, directory=None):
        return campaign.CampaignController.prepare(directory or self.directory, seed=0,
            initial_selection=self.inputs["initial_selection"], config_root=self.inputs["config_root"],
            replay_root=self.inputs["data_root"], normalization_dir=self.inputs["statistics_root"],
            device="cpu", precision="float32")

    def resume(self, directory=None):
        return campaign.CampaignController.resume(directory or self.directory, device="cpu")

    def events(self):
        path = self.directory / "attempts.jsonl"
        return [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines()]

    def test_prepare_saves_paired_student_before_any_teacher_work(self):
        worker = self.prepare()
        self.assertEqual(self.backend.calls[0], ("initial_snapshot", "fresh"))
        self.assertFalse(any(call[0] == "run" for call in self.backend.calls))
        self.assertEqual(worker.status()["status"], "prepared")
        self.assertEqual(worker.state["units_completed"], 0)
        self.assertEqual(worker.state["phase"], "teacher")
        self.assertTrue((self.directory / "run_manifest.sha256").is_file())
        self.assertFalse((self.directory / "completed_student.json").exists())
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_full_fixed_schedule_runs_ordered_units_once_and_publishes_local_export(self):
        worker = self.prepare()
        result = worker.run()
        self.assertEqual(result["status"], "complete")
        calls = [name for action, name in self.backend.calls if action == "run"]
        self.assertEqual(calls, ["teacher"] * 2 + ["buffer"] * 2 + ["distillation"] * 2 + ["local_cycle"] * 2)
        self.assertEqual(worker.state["units_completed"], 9)
        self.assertEqual(worker.state["phase"], "complete")
        self.assertIsNotNone(worker.state["export"])
        self.assertEqual(result["new_environment_interactions"], 0)
        self.assertEqual(result["new_model_updates"], 0)
        self.assertEqual(result["teacher_actor_optimizer_steps"], 2)
        self.assertEqual(result["teacher_critic_optimizer_steps"], 2)
        self.assertEqual(result["local_actor_optimizer_steps"], 2)
        self.assertEqual(result["local_critic_optimizer_steps"], 2)
        self.assertEqual(result["distillation_optimizer_steps"], 2)
        self.assertEqual(result["imagined_joint_steps"], 2 + 2 + 2)
        before = deepcopy(self.backend.calls)
        repeated = self.resume().run()
        self.assertEqual(repeated["status"], "complete")
        self.assertEqual([call for call in self.backend.calls[len(before):] if call[0] == "run"], [])

    def test_limited_invocations_resume_exact_private_stage_state_without_reset(self):
        reference = self.prepare(self.root / "reference")
        reference.run()
        worker = self.prepare()
        first = worker.run(max_units=1)
        self.assertEqual(first["status"], "stopped")
        self.assertEqual(worker.state["units_completed"], 1)
        second = self.resume()
        second.run(max_units=2)
        self.assertEqual(second.state["units_completed"], 3)
        self.assertEqual(second.state["phase"], "buffer")
        final = self.resume()
        final.run()
        for name in ("teacher", "buffer", "distillation", "local_cycle"):
            left = final.state["stage_evidence"][name]
            right = reference.state["stage_evidence"][name]
            self.assertEqual(left["completed"], right["completed"])
            self.assertEqual(left["counters"], right["counters"])
            self.assertEqual(json.loads((Path(left["path"]) / "fixture_state.json").read_text()),
                             json.loads((Path(right["path"]) / "fixture_state.json").read_text()))

    def test_safe_stop_consumes_request_at_completed_unit_boundary(self):
        worker = self.prepare()
        (self.directory / "stop.request").write_text("stop")
        worker.run()
        self.assertEqual(worker.state["units_completed"], 0)
        self.assertFalse((self.directory / "stop.request").exists())
        run_one = self.backend.run_one

        def stop_during_unit(name, stage_worker):
            result = run_one(name, stage_worker)
            worker.request_stop()
            return result

        with patch.object(campaign.stages, "run_one", side_effect=stop_during_unit):
            result = worker.run()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(worker.state["units_completed"], 1)
        self.assertEqual(self.resume().state["units_completed"], 1)

    def test_child_commit_survives_controller_failure_without_repeating_optimizer_work(self):
        worker = self.prepare()
        self.backend.interrupt_after_commit = True
        with self.assertRaisesRegex(OSError, "durable child commit"):
            worker.run(max_units=1)
        with self.assertRaisesRegex(RuntimeError, "fresh|resume|failed"):
            worker.run()
        self.assertEqual([call for call in self.backend.calls if call[0] == "run"], [("run", "teacher")])
        restored = self.resume()
        self.assertEqual(restored.state["units_completed"], 0)
        restored.run(max_units=1)
        self.assertEqual(restored.state["units_completed"], 1)
        self.assertEqual([call for call in self.backend.calls if call[0] == "run"], [("run", "teacher")])
        restored.run(max_units=1)
        self.assertEqual(restored.state["units_completed"], 2)
        count = restored.state["stage_evidence"]["teacher"]["completed"]
        self.assertEqual(count, 2)
        journal = self.events()
        self.assertTrue(any("fail" in item["event"] for item in journal))

    def test_lock_and_stale_worker_prevent_duplicate_stage_work(self):
        worker = self.prepare()
        stale = self.resume()
        with run_lock(self.directory):
            with self.assertRaisesRegex(RuntimeError, "another worker"):
                worker.run(max_units=1)
        worker.run(max_units=1)
        before = deepcopy(self.backend.calls)
        with self.assertRaisesRegex(RuntimeError, "advanced|latest|stale"):
            stale.run(max_units=1)
        self.assertFalse(any(call[0] == "run" for call in self.backend.calls[len(before):]))

    def test_source_runtime_frozen_binding_and_manifest_receipt_changes_fail_closed(self):
        self.prepare()
        with patch.object(campaign, "source_identity", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "source"):
                self.resume()
        with patch.object(campaign, "_runtime", return_value={"logical_device": "cpu", "fixture": 2}):
            with self.assertRaisesRegex(ValueError, "runtime"):
                self.resume()
        self.backend.frozen_changed = True
        with self.assertRaisesRegex(ValueError, "input|frozen|binding"):
            self.resume()
        self.backend.frozen_changed = False
        path = self.directory / "run_manifest.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.resume()

    def test_bad_unit_limits_do_not_advance_or_poison_worker(self):
        worker = self.prepare()
        for limit in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                worker.run(max_units=limit)
        self.assertEqual(worker.state["units_completed"], 0)
        self.assertEqual(worker.run(max_units=1)["status"], "stopped")

    def test_completed_standalone_work_is_adopted_without_retraining_or_relabeling(self):
        source = self.prepare(self.root / "source_development")
        source.run()
        self.backend.adopted_paths = deepcopy(source.manifest["paths"])
        source_files = {path.relative_to(source.directory).as_posix(): file_sha256(path)
                        for path in source.directory.rglob("*") if path.is_file()}
        before = len([call for call in self.backend.calls if call[0] == "run"])
        adopted = campaign.CampaignController.adopt(self.directory,
            refinement_run=self.backend.adopted_paths["local_cycle"], device="cpu")
        self.assertEqual(adopted.status()["status"], "complete")
        self.assertEqual(adopted.manifest["mode"], "adopted_development")
        self.assertIs(adopted.manifest["fresh_cohort_eligible"], False)
        self.assertEqual(adopted.manifest["starting_units"], 8)
        self.assertEqual(adopted.state["units_completed"], 9)
        self.assertEqual(len([call for call in self.backend.calls if call[0] == "run"]), before)
        self.assertEqual(source_files, {path.relative_to(source.directory).as_posix(): file_sha256(path)
                                       for path in source.directory.rglob("*") if path.is_file()})
        report = json.loads((self.directory / "completed_student.json").read_text())
        self.assertFalse(report["phase4_acceptance"])
        self.assertEqual(self.resume().run()["status"], "complete")

    def test_adoption_rejects_incomplete_source_without_advancing_it(self):
        source = self.prepare(self.root / "source_incomplete")
        source.run(max_units=1)
        self.backend.adopted_paths = deepcopy(source.manifest["paths"])
        # Provide all expected paths as structurally present, while retaining
        # the first teacher's incomplete committed progress.
        for name in ("buffer", "local_cycle"):
            path = Path(self.backend.adopted_paths[name])
            self.backend.prepare_stage(name, path, inputs=self.inputs, device="cpu", precision="float32")
        before = deepcopy(self.backend.calls)
        with self.assertRaisesRegex(ValueError, "complete"):
            campaign.CampaignController.adopt(self.directory,
                refinement_run=self.backend.adopted_paths["local_cycle"], device="cpu")
        self.assertFalse(any(call[0] == "run" for call in self.backend.calls[len(before):]))
        self.assertFalse((self.directory / "actor_export").exists())

    def test_final_export_publication_retries_without_repeating_any_learner_units(self):
        worker = self.prepare()
        publish = worker._publish_export

        def export_committed_then_process_lost():
            publish()
            raise OSError("fixture stopped after actor export publication")

        with patch.object(worker, "_publish_export", side_effect=export_committed_then_process_lost):
            with self.assertRaisesRegex(OSError, "after actor export"):
                worker.run()
        self.assertEqual(worker.state["units_completed"], 8)
        before = len([call for call in self.backend.calls if call[0] == "run"])
        recovered = self.resume()
        self.assertEqual(recovered.run()["status"], "complete")
        self.assertEqual(len([call for call in self.backend.calls if call[0] == "run"]), before)
        exports = [path for path in self.directory.glob("actor_export*") if path.is_dir()]
        self.assertEqual(len(exports), 1)

    def test_child_checkpoint_tampering_cannot_fall_back_to_earlier_progress(self):
        worker = self.prepare()
        worker.run(max_units=2)
        evidence = worker.state["stage_evidence"]["teacher"]
        checkpoint = Path(evidence["checkpoint"])
        payload = checkpoint / "state.pt"
        original = payload.read_bytes()
        payload.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "hash|payload|checksum"):
            self.resume()

    def test_foreign_prepared_child_is_rejected_before_any_training(self):
        worker = self.prepare()
        foreign = deepcopy(self.inputs)
        foreign["master_seed"] = 1
        self.backend.prepare_stage("teacher", worker.paths["teacher"], inputs=foreign,
                                   device="cpu", precision="float32")
        before = deepcopy(self.backend.calls)
        with self.assertRaisesRegex(ValueError, "input|seed|lineage"):
            worker.run(max_units=1)
        self.assertFalse(any(call[0] == "run" for call in self.backend.calls[len(before):]))

    def test_empty_partial_child_preparation_is_preserved_then_retried_once(self):
        for with_manifest in (False, True):
            with self.subTest(with_manifest=with_manifest):
                directory = self.root / f"partial_{int(with_manifest)}"
                worker = self.prepare(directory)
                child = worker.paths["teacher"]
                child.mkdir(parents=True)
                debris = child / "interrupted_prepare.txt"
                debris.write_text("partial untrained initialization")
                if with_manifest:
                    receipt(child / "run_manifest.json", {"fixture": "interrupted-before-checkpoint"})
                result = worker.run(max_units=1)
                self.assertEqual(result["status"], "stopped")
                self.assertEqual(worker.state["units_completed"], 1)
                archives = list(child.parent.glob("teacher_incomplete_prepare_*"))
                self.assertEqual(len(archives), 1)
                self.assertEqual((archives[0] / debris.name).read_text(), "partial untrained initialization")
                self.assertFalse((child / debris.name).exists())
                self.assertEqual(self.resume(directory).state["units_completed"], 1)

    def test_partial_child_with_attempted_research_work_is_never_restarted(self):
        worker = self.prepare()
        child = worker.paths["teacher"]
        child.mkdir(parents=True)
        (child / "attempts.jsonl").write_text('{"event":"unit_started","attempt":"uncommitted"}\n')
        before = deepcopy(self.backend.calls)
        with self.assertRaisesRegex(ValueError, "possible research work|recovery"):
            worker.run(max_units=1)
        self.assertFalse(any(call[0] in ("prepare", "run") for call in self.backend.calls[len(before):]))
        self.assertFalse(list(child.parent.glob("teacher_incomplete_prepare_*")))

    def test_stop_during_slow_child_preparation_prevents_first_optimizer_unit(self):
        worker = self.prepare()
        prepare_stage = self.backend.prepare_stage

        def prepare_then_stop(*args, **kwargs):
            child = prepare_stage(*args, **kwargs)
            worker.request_stop()
            return child

        with patch.object(campaign.stages, "prepare_stage", side_effect=prepare_then_stop):
            result = worker.run(max_units=1)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(worker.state["units_completed"], 0)
        self.assertFalse(any(call[0] == "run" for call in self.backend.calls))
        self.assertTrue((worker.paths["teacher"] / "run_manifest.sha256").is_file())
        restored = self.resume()
        self.assertEqual(restored.run(max_units=1)["status"], "stopped")
        self.assertEqual(restored.state["units_completed"], 1)
        self.assertEqual([call for call in self.backend.calls if call[0] == "run"], [("run", "teacher")])


if __name__ == "__main__":
    unittest.main()
