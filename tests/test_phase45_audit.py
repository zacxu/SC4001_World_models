"""One shared tiny numerical campaign, then read-only acceptance/tamper checks."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from tests import test_teacher_runner as fixture
from world_models.cycle.checkpoints import load_cycle_checkpoint, save_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.distillation.protocol import DISTILLATION_PROTOCOL
from world_models.refinement.protocol import REFINEMENT_PROTOCOL
from world_models_campaign.controller import CampaignController
from world_models_campaign.protocol import CAMPAIGN_PROTOCOL
from world_models_acceptance import audit as audit_module
from world_models_acceptance.audit import audit_campaign


class Phase45AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.TeacherRunnerTests.setUpClass.__func__(cls)
        owner = cls("test_complete_campaign_readonly_without_optimizers_resume_or_queries")
        cls.addClassCleanup(owner.doCleanups)
        fixture.TeacherRunnerTests.setUp(owner)
        owner.enterContext(patch.dict(DISTILLATION_PROTOCOL, {
            "initial_transitions": 6, "initial_model_updates": 1, "teacher_cycles": 2,
            "trajectories": 2, "horizon": 1, "updates": 2, "sequences_per_update": 2}))
        owner.enterContext(patch.dict(REFINEMENT_PROTOCOL, {
            "initial_transitions": 6, "initial_model_updates": 1,
            "local_cycles": 2, "starts": 2, "horizon": 1, "epochs": 1, "minibatches": 1}))
        owner.enterContext(patch.dict(CAMPAIGN_PROTOCOL, {
            "initial_transitions": 6, "initial_model_updates": 1,
            "teacher_cycles": 2, "trajectories": 2, "distillation_updates": 2,
            "distillation_sequences_per_update": 2, "local_cycles": 2,
            "starts": 2, "horizon": 1, "epochs": 1, "minibatches": 1}))
        cls.owner = owner
        cls.directory = Path(owner.temp.name) / "campaign"
        worker = CampaignController.prepare(cls.directory, seed=0,
            initial_selection=owner.selection["selection_path"], config_root=fixture.ROOT / "configs",
            replay_root=owner.root, normalization_dir=owner.root, device="cpu", precision="float32")
        worker.run()
        cls.worker = worker

    @classmethod
    def tearDownClass(cls):
        fixture.TeacherRunnerTests.tearDownClass.__func__(cls)

    def test_complete_campaign_readonly_without_optimizers_resume_or_queries(self):
        before = {str(path): file_sha256(path) for path in Path(self.owner.temp.name).rglob("*") if path.is_file()}
        rng = torch.get_rng_state().clone()
        with patch("torch.optim.Adam", side_effect=AssertionError("audit constructed optimizer")), \
             patch.object(CampaignController, "resume", side_effect=AssertionError("audit resumed campaign")), \
             patch("world_models.refinement.runner.RefinementRunner.resume", side_effect=AssertionError("audit resumed GPU worker")), \
             patch("world_models.policies.teacher.GlobalTeacher.forward", side_effect=AssertionError("audit queried teacher")):
            result = audit_campaign(self.directory)
        self.assertTrue(result["engineering_passed"])
        self.assertFalse(result["cohort_ready"])
        self.assertFalse(result["learned_quality_established"])
        self.assertEqual(result["new_environment_interactions"], 0)
        self.assertEqual(len(result["diagnostics"]["teacher_progression"]), 2)
        self.assertEqual(len(result["diagnostics"]["local_progression"]), 2)
        self.assertEqual(result["diagnostics"]["buffer_masks"]["trajectories"], 2)
        self.assertEqual(result["diagnostics"]["buffer_masks"]["valid_player_decisions"], 4)
        self.assertEqual(result["actor_export"]["source_device"], "cpu")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        after = {str(path): file_sha256(path) for path in Path(self.owner.temp.name).rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        for row in result["bound_files"]:
            self.assertEqual(file_sha256(row["path"]), row["sha256"])

    def test_complete_gate_audit_and_isolated_deployment_produce_engineering_only_acceptance(self):
        from world_models_acceptance import evidence, gate
        verification = Path(self.owner.temp.name) / "gate_fixture_verification" / "implementation_checks.json"
        verification.parent.mkdir()
        verification.write_text('{"fixture":"trusted coverage injected only in this integration test"}')
        passed = {test for group in evidence.REQUIRED_CHECKS.values() for test in group}
        trusted = ({"fixture": True}, evidence.requirement_coverage(passed),
                   {"path": str(verification), "sha256": file_sha256(verification)})
        before = {str(path): file_sha256(path) for path in self.directory.rglob("*") if path.is_file()}
        output = Path(self.owner.temp.name) / "gate_acceptance"
        # Evidence-producer authentication has its own strict tests. Here the
        # actual auditor and fresh deployment subprocess must interoperate with
        # a genuinely trained, exported tiny student's saved stage history.
        with patch.object(gate, "load_valid_verification", return_value=trusted):
            report = gate.accept_campaign(self.directory, output, verification_report=verification)
        self.assertEqual(report["status"], "passed", report["reasons"])
        self.assertTrue(report["phase4_acceptance"])
        self.assertTrue(report["phase4_engineering_acceptance"])
        for key in ("research_training_executed", "native_environment_evaluation_executed",
                    "learned_student_quality_established", "b_versus_d_efficacy_established",
                    "cohort_acceptance", "fresh_cohort_eligible", "automatic_phase5_launch"):
            self.assertFalse(report[key])
        self.assertTrue(report["deployment"]["local_only"])
        self.assertEqual(report["deployment"]["forbidden_modules_loaded"], [])
        published = evidence.read_report(output / "phase4_acceptance.json")
        self.assertEqual(published, report)
        after = {str(path): file_sha256(path) for path in self.directory.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_ahead_informational_status_does_not_change_durable_audit(self):
        path = self.directory / "status.json"
        previous = path.read_bytes()
        try:
            path.write_text('{"phase":"failed","local_cycles_completed":999}')
            self.assertTrue(audit_campaign(self.directory)["engineering_passed"])
        finally:
            path.write_bytes(previous)

    def test_export_tampering_is_rejected(self):
        path = Path(self.worker.state["export"]["path"]) / "state.pt"
        previous = path.read_bytes()
        try:
            path.write_bytes(previous + b"tampered")
            with self.assertRaises(ValueError):
                audit_campaign(self.directory)
        finally:
            path.write_bytes(previous)

    def test_buffer_tampering_is_rejected(self):
        path = next((self.worker.paths["buffer"] / "buffer").glob("*/state.pt"))
        previous = path.read_bytes()
        try:
            path.write_bytes(previous + b"tampered")
            with self.assertRaises(ValueError):
                audit_campaign(self.directory)
        finally:
            path.write_bytes(previous)

    def test_accounting_receipt_tampering_is_rejected(self):
        path = self.directory / "accounting.json"
        previous = path.read_bytes()
        try:
            path.write_bytes(previous + b" ")
            with self.assertRaises(ValueError):
                audit_campaign(self.directory)
        finally:
            path.write_bytes(previous)

    def test_source_tampering_and_non_cpu_target_rejected(self):
        with patch("world_models_campaign.controller.source_identity", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                audit_campaign(self.directory)
        with self.assertRaisesRegex(ValueError, "CPU only"):
            audit_campaign(self.directory, device="cuda:0")

    def test_semantic_optimizer_rng_and_guidance_checks_are_independent_of_receipts(self):
        final = self.worker.paths["local_cycle"] / "checkpoints"
        final = sorted(final.glob("state_*"))[-1]
        original = audit_module._bound_checkpoint
        mutations = ("moments", "shuffle", "guidance", "capturable")
        for mutation in mutations:
            def altered(path, bound):
                saved, metadata = original(path, bound)
                if Path(path) == final:
                    if mutation == "moments":
                        saved["learner"]["actor_optimizer"]["state"][0]["exp_avg_sq"].view(-1)[0] = -1
                    elif mutation == "shuffle":
                        saved["learner"]["shuffle_rng"] = torch.Generator().manual_seed(123).get_state()
                    elif mutation == "capturable":
                        saved["learner"]["actor_optimizer"]["param_groups"][0]["capturable"] = True
                    else:
                        saved["last_report"]["teacher_kl_weight"] = .1
                return saved, metadata
            with self.subTest(mutation=mutation), patch.object(audit_module, "_bound_checkpoint", side_effect=altered):
                with self.assertRaisesRegex(ValueError, "second moments|shuffle RNG|teacher guidance|Adam execution flags"):
                    audit_campaign(self.directory)

    def test_master_stage_order_counters_checked_for_each_historical_commit(self):
        historical = sorted((self.directory / "checkpoints").glob("state_*"))[1]
        original = audit_module._bound_checkpoint
        def altered(path, bound):
            saved, metadata = original(path, bound)
            if Path(path) == historical:
                saved["controller"]["units_completed"] = 0
            return saved, metadata
        with patch.object(audit_module, "_bound_checkpoint", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "unit accounting|master history"):
                audit_campaign(self.directory)

    def test_second_committed_parent_export_is_rejected(self):
        path = self.directory / "actor_export_duplicate"
        self.assertEqual(path.resolve().parent, self.directory.resolve())
        saved, metadata = load_cycle_checkpoint(self.worker.state["export"]["path"])
        try:
            save_cycle_checkpoint(path, state=saved, metadata=metadata)
            with self.assertRaisesRegex(ValueError, "exactly one committed"):
                audit_campaign(self.directory)
        finally:
            if path.exists():
                for item in path.iterdir():
                    item.unlink()
                path.rmdir()

    def test_incomplete_parent_never_accepted(self):
        path = self.directory / "completed_student.sha256"
        previous = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaisesRegex(ValueError, "receipt"):
                audit_campaign(self.directory)
        finally:
            path.write_bytes(previous)

    def test_adoption_keeps_historical_provenance_and_is_not_fresh_cohort(self):
        adopted = Path(self.owner.temp.name) / "adopted"
        CampaignController.adopt(adopted, refinement_run=self.worker.paths["local_cycle"], device="cpu")
        result = audit_campaign(adopted)
        self.assertEqual(result["mode"], "adopted_development")
        self.assertFalse(result["cohort_ready"])
        self.assertFalse(result["lineage"]["paired_initial_student"]["metadata"]["provenance"]["saved_before_teacher_training"])


if __name__ == "__main__":
    unittest.main()
