"""Tiny CPU adapter integration; no research weights or native environments."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests import test_teacher_runner as fixture
from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.distillation.protocol import DISTILLATION_PROTOCOL
from world_models.refinement.protocol import REFINEMENT_PROTOCOL
from world_models_campaign import stages


class CampaignStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.TeacherRunnerTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        fixture.TeacherRunnerTests.tearDownClass.__func__(cls)

    def setUp(self):
        fixture.TeacherRunnerTests.setUp(self)
        self.enterContext(patch.dict(DISTILLATION_PROTOCOL, {
            "initial_transitions": 6, "initial_model_updates": 1, "teacher_cycles": 2,
            "trajectories": 2, "horizon": 1, "updates": 2, "sequences_per_update": 2}))
        self.enterContext(patch.dict(REFINEMENT_PROTOCOL, {
            "initial_transitions": 6, "initial_model_updates": 1,
            "local_cycles": 2, "starts": 2, "horizon": 1, "epochs": 1, "minibatches": 1}))
        self.inputs = {"initial_selection": self.selection["selection_path"], "master_seed": 0,
            "data_root": str(self.root), "statistics_root": str(self.root),
            "config_root": str(fixture.ROOT / "configs")}

    def test_tiny_real_stage_chain_snapshots_resume_and_honest_readonly_adoption(self):
        root = Path(self.temp.name)
        rng = torch.get_rng_state().clone()
        initial = stages.save_initial_student(root / "initial_student", inputs=self.inputs)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(initial["metadata"]["provenance"]["saved_before_teacher_training"])
        self.assertFalse(self.directory.exists())
        self.assertEqual(initial, stages.verify_initial_student(root / "initial_student"))
        with self.assertRaises(FileExistsError):
            stages.save_initial_student(root / "initial_student", inputs=self.inputs)
        teacher = stages.prepare_stage("teacher", self.directory, inputs=self.inputs,
                                        device="cpu", precision="float32")
        self.assertEqual(stages.inspect_stage("teacher", self.directory)["completed"], 0)
        stages.run_one("teacher", teacher)
        teacher = stages.resume_stage("teacher", self.directory, device="cpu")
        stages.run_one("teacher", teacher)
        evidence = stages.inspect_stage("teacher", self.directory)
        self.assertEqual((evidence["completed"], evidence["target"]), (2, 2))
        self.assertEqual(evidence["counters"]["teacher_actor_optimizer_steps"], 2)
        self.assertEqual(evidence["physical_work"]["started_units"], 2)
        p42 = root / "distillation"
        student = stages.prepare_stage("buffer", p42, inputs=self.inputs,
                                       teacher_run=self.directory, device="cpu")
        self.assertEqual(initial, stages.verify_initial_student(root / "initial_student", distillation_run=p42))
        for index in range(2):
            stages.run_one("buffer", student)
            self.assertEqual(stages.inspect_stage("buffer", p42)["completed"], index + 1)
            student = stages.resume_stage("buffer", p42, device="cpu")
        with self.assertRaisesRegex(ValueError, "wrong stage"):
            stages.run_one("buffer", student)
        for index in range(2):
            stages.run_one("distillation", student)
            self.assertEqual(stages.inspect_stage("distillation", p42)["completed"], index + 1)
            student = stages.resume_stage("distillation", p42, device="cpu")
        p43 = root / "refinement"
        local = stages.prepare_stage("local_cycle", p43, inputs=self.inputs,
                                     distillation_run=p42, device="cpu")
        for index in range(2):
            stages.run_one("local_cycle", local)
            progress = stages.inspect_stage("local_cycle", p43)
            self.assertEqual(progress["completed"], index + 1)
            self.assertEqual(progress["counters"]["local_actor_optimizer_steps"], index + 1)
            local = stages.resume_stage("local_cycle", p43, device="cpu")
        self.assertEqual(progress["counters"]["new_environment_interactions"], 0)
        self.assertEqual(progress["counters"]["new_model_updates"], 0)
        self.assertTrue(any(row["path"].endswith("refined_student.json") for row in progress["publications"]))
        before = {str(path): file_sha256(path) for path in root.rglob("*") if path.is_file()}
        lineage = stages.adoption_lineage(p43, device="cpu")
        after = {str(path): file_sha256(path) for path in root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(lineage["paths"]["buffer"], lineage["paths"]["distillation"])
        self.assertFalse(lineage["provenance"]["fresh_cohort_claim"])
        adopted = stages.save_initial_student(root / "adopted_initial", inputs=self.inputs,
                                              mode="adopted", distillation_run=p42)
        self.assertFalse(adopted["metadata"]["provenance"]["saved_before_teacher_training"])
        stages.verify_initial_student(root / "adopted_initial", distillation_run=p42)
        original_state, _ = load_cycle_checkpoint(root / "initial_student")
        adopted_state, _ = load_cycle_checkpoint(root / "adopted_initial")
        for name in original_state:
            for key in original_state[name]:
                self.assertTrue(torch.equal(original_state[name][key], adopted_state[name][key]))
        # Informational status cannot advance durable stage evidence.
        (p43 / "status.json").write_text('{"local_cycles_completed":999}')
        self.assertEqual(stages.inspect_stage("local_cycle", p43)["completed"], 2)
        # A hard interruption between final checkpoint and publication needs no
        # repeated PPO; completed-run publication repair is pure checkpoint I/O.
        (p43 / "refined_student.sha256").unlink()
        (p43 / "refined_student.json").unlink()
        repaired = stages.resume_stage("local_cycle", p43, device="cpu")
        stages.run_one("local_cycle", repaired)
        self.assertEqual(repaired.learner.cycles_completed, 2)
        self.assertEqual(stages.inspect_stage("local_cycle", p43)["completed"], 2)
        # Every buffer payload stays bound, even in a pure progress inspection.
        payload = next((p42 / "buffer").glob("*/state.pt"))
        with payload.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "payload changed"):
            stages.inspect_stage("buffer", p42)

    def test_real_campaign_controller_runs_all_stages_and_resumes_master(self):
        from world_models_campaign.controller import CampaignController
        from world_models_campaign.protocol import CAMPAIGN_PROTOCOL
        self.enterContext(patch.dict(CAMPAIGN_PROTOCOL, {
            "initial_transitions": 6, "initial_model_updates": 1,
            "teacher_cycles": 2, "trajectories": 2, "distillation_updates": 2,
            "distillation_sequences_per_update": 2, "local_cycles": 2,
            "starts": 2, "horizon": 1, "epochs": 1, "minibatches": 1}))
        # Other controller tests exercise source-tampering rejection. Pin this
        # fixture's outer identity while unrelated new controller files settle.
        self.enterContext(patch("world_models_campaign.controller.source_identity",
                                return_value={"tiny-campaign-fixture.py": "c" * 64}))
        root = Path(self.temp.name)
        directory = root / "campaign"
        controller = CampaignController.prepare(directory, seed=0,
            initial_selection=self.inputs["initial_selection"], config_root=self.inputs["config_root"],
            replay_root=self.inputs["data_root"], normalization_dir=self.inputs["statistics_root"],
            device="cpu", precision="float32")
        self.assertEqual(controller.state["units_completed"], 0)
        self.assertTrue((directory / "initial_student/manifest.sha256").is_file())
        self.assertFalse((directory / "stages/teacher").exists())
        phases = []
        for units in (1, 2, 2, 2, 2):
            result = controller.run(max_units=units)
            phases.append(result["phase"])
            controller = CampaignController.resume(directory, device="cpu")
        self.assertEqual(phases, ["teacher", "buffer", "distillation", "local_cycle", "complete"])
        final = controller.status()
        self.assertEqual(final["status"], "complete")
        self.assertEqual(final["units_completed"], 9)
        self.assertEqual(final["teacher_cycles_completed"], 2)
        self.assertEqual(final["trajectories_completed"], 2)
        self.assertEqual(final["distillation_optimizer_steps"], 2)
        self.assertEqual(final["local_cycles_completed"], 2)
        self.assertEqual(final["local_actor_optimizer_steps"], 2)
        self.assertEqual(final["local_critic_optimizer_steps"], 2)
        self.assertEqual(final["new_environment_interactions"], 0)
        self.assertEqual(final["new_model_updates"], 0)
        for name in ("run_manifest", "accounting", "completed_student"):
            self.assertTrue((directory / f"{name}.sha256").is_file())
        payload, metadata = load_cycle_checkpoint(controller.state["export"]["path"])
        self.assertEqual(set(payload), {"actor"})
        self.assertEqual(metadata["variant"], "D")
        self.assertEqual(metadata["local_ppo_cycles"], 2)
        self.assertEqual(controller.run()["units_executed"], 0)
        checkpoints = list((directory / "checkpoints").glob("*/manifest.sha256"))
        self.assertEqual(len(checkpoints), 10)
        adopted = CampaignController.adopt(root / "adopted_campaign",
            refinement_run=controller.paths["local_cycle"], device="cpu")
        self.assertEqual(adopted.status()["status"], "complete")
        self.assertEqual(adopted.status()["units_completed"], 9)
        self.assertEqual(adopted.manifest["mode"], "adopted_development")
        self.assertFalse(adopted.manifest["fresh_cohort_eligible"])
        CampaignController.resume(root / "adopted_campaign", device="cpu")


class CampaignPhysicalWorkTests(unittest.TestCase):
    def test_shared_journal_counts_each_stage_once_and_preserves_interrupted_work(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows = [{"event": "unit_started", "phase": "buffer", "attempt": "buffer-ok"},
                    {"event": "unit_committed", "phase": "buffer", "attempt": "buffer-ok", "elapsed_seconds": 2.0},
                    {"event": "unit_started", "phase": "distillation", "attempt": "distill-fail"},
                    {"event": "invocation_failed", "attempt": "distill-fail"},
                    {"event": "unit_started", "phase": "distillation", "attempt": "distill-ok"},
                    {"event": "unit_committed", "phase": "distillation", "attempt": "distill-ok", "elapsed_seconds": 3.0}]
            (directory / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            buffer = stages._physical(directory, "buffer")
            distillation = stages._physical(directory, "distillation")
            self.assertEqual(buffer["recorded_elapsed_seconds"], 2)
            self.assertEqual(distillation["recorded_elapsed_seconds"], 3)
            self.assertEqual(distillation["unmatched_attempts"], ["distill-fail"])
            self.assertIn("unknown", distillation["partial_interrupted_compute"])

    def test_duplicate_or_orphan_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "attempts.jsonl"
            path.write_text(json.dumps({"event": "teacher_cycle_committed", "attempt": "orphan"}) + "\n")
            with self.assertRaisesRegex(ValueError, "identities"):
                stages._physical(directory, "teacher")

    def test_truncated_last_row_reports_unknown_compute_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "attempts.jsonl"
            valid = (json.dumps({"event": "teacher_cycle_started", "attempt": "ok"}) + "\n"
                + json.dumps({"event": "teacher_cycle_committed", "attempt": "ok",
                              "elapsed_seconds_including_checkpoint": 2.5}) + "\n").encode()
            torn = b'{"event": "teacher_cycle_started", "attempt": "torn'
            original = valid + torn
            path.write_bytes(original)
            digest = file_sha256(path)
            physical = stages._physical(directory, "teacher")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(physical["journal_sha256"], digest)
            self.assertEqual((physical["started_units"], physical["committed_units"]), (1, 1))
            self.assertEqual(physical["recorded_elapsed_seconds"], 2.5)
            self.assertEqual(physical["malformed_rows"], 1)
            self.assertEqual(physical["malformed_locations"], [
                {"line": 3, "byte_offset": len(valid), "bytes": len(torn)}])
            self.assertIn("unknown", physical["partial_interrupted_compute"])

    def test_malformed_earlier_row_can_explain_unattributed_commit_but_duplicates_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "attempts.jsonl"
            commit = json.dumps({"event": "teacher_cycle_committed", "attempt": "lost-start",
                                 "elapsed_seconds": 1.5}) + "\n"
            path.write_bytes(b'{"event": "teacher_cycle_sta\n' + commit.encode())
            original = path.read_bytes()
            physical = stages._physical(directory, "teacher")
            self.assertEqual(physical["unattributed_orphan_commits"], ["lost-start"])
            self.assertEqual(physical["committed_units"], 1)
            self.assertIn("unknown", physical["partial_interrupted_compute"])
            self.assertEqual(path.read_bytes(), original)
            # Later damage cannot account for an orphan that preceded it.
            path.write_bytes(commit.encode() + b'{"event": "later-torn')
            with self.assertRaisesRegex(ValueError, "identities"):
                stages._physical(directory, "teacher")
            start = json.dumps({"event": "teacher_cycle_started", "attempt": "duplicate"}) + "\n"
            path.write_bytes(b'{"event": "torn\n' + start.encode() + start.encode())
            with self.assertRaisesRegex(ValueError, "identities"):
                stages._physical(directory, "teacher")


if __name__ == "__main__":
    unittest.main()
