"""Prospective cohort and training-to-test handoff integration, synthetic only."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from world_models.cycle.checkpoints import save_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_endpoint import cohort, evidence
from world_models_endpoint.evidence import read_report, write_report
from world_models_study.bank import build_test_bank
from world_models_study.protocol import load_protocol
from world_models_paired.cohort import plan as paired_plan
from world_models.policies.actor import LocalActor
from world_models.policies.config import PolicyConfig
from scripts.phase36_support.evaluation import actor_version


def rewrite(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    path.with_suffix(".sha256").write_text(f"{file_sha256(path)}  {path.name}\n", encoding="ascii")


class CohortFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.sources = {"synthetic_original.py": "a" * 64, "synthetic_endpoint.py": "b" * 64}
        self.enterContext(patch.object(cohort, "ROOT", self.root))
        self.enterContext(patch.object(cohort, "source_identity", return_value=self.sources))
        self.enterContext(patch.object(evidence, "source_identity", return_value=self.sources))
        self.proof = self.root / "verify/implementation_checks.json"
        self.proof.parent.mkdir()
        self.combined = {"source_sha256": self.sources, "passed_tests": []}
        self.enterContext(patch.object(cohort, "validate_verification", return_value=self.combined))
        write_report(self.proof, {"synthetic_only": True, "source_sha256": self.sources})
        inputs_file = self.root / "inputs/selected.json"
        inputs_file.parent.mkdir()
        inputs_file.write_text("synthetic frozen input", encoding="utf-8")
        per_seed = {}
        for seed in (0, 1):
            initial = {"selection": {"checkpoint_sha256": str(seed+1) * 64},
                "normalization": {"identity_sha256": str(seed+3) * 64},
                "provenance": {"replays": [{"manifest_sha256": str(seed+5) * 64}]}}
            per_seed[str(seed)] = {"initial_binding": initial, "initial_binding_sha256": canonical_hash(initial)}
        self.inputs = {"per_seed": per_seed, "bound_files": [{"path": str(inputs_file), "sha256": file_sha256(inputs_file)}]}
        self.parent_path = self.root / "registration/protocol_registration.json"
        self.parent_path.parent.mkdir()
        bank = build_test_bank()
        write_report(self.parent_path.parent / "test_bank.json", bank)
        self.parent = {"schema_version": "p51-protocol-registration-v1", "status": "protocol_registered",
            "source_sha256": {"synthetic_original.py": "a" * 64}, "protocol": load_protocol(),
            "protocol_sha256": cohort.protocol_identity(), "inputs": self.inputs,
            "input_roots": {"initial_selection": str(inputs_file), "data_root": str(self.root / "inputs/data"),
                "statistics_root": str(self.root / "inputs/stats"), "config_root": str(self.root / "inputs/config")},
            "test_bank_sha256": canonical_hash(bank)}
        self.parent["verification"] = self.historical("p51", evidence.TEST_MODULES[:5])
        write_report(self.parent_path, self.parent)
        self.matrix_path = self.root / "candidate/cohort_manifest.json"
        self.matrix_path.parent.mkdir()
        self.matrix = {"schema_version": "p52-paired-matrix-v1", "status": "training_matrix_registered",
            "candidate_only": True, "cohort_frozen": False, "plan": paired_plan(),
            "source_sha256": self.parent["source_sha256"],
            "registration": {"path": str(self.parent_path), "sha256": file_sha256(self.parent_path)},
            "inputs_sha256": canonical_hash(self.inputs), "verification": self.historical("p52", evidence.TEST_MODULES[5:9])}
        write_report(self.matrix_path, self.matrix)
        self.directory = self.root / "fresh"
        self.workers = {}

    def historical(self, prefix, modules):
        path = self.root / prefix / "implementation_checks.json"
        path.parent.mkdir()
        name = modules[0] + ".Fixture.test_ok"
        self.combined["passed_tests"].append(name)
        log = path.parent / "tests.txt"
        log.write_text(f"test_ok ({name}) ... ok\n\nRan 1 test in 0.001s\nOK\n", encoding="utf-8")
        report = {"schema_version": prefix + "-implementation-v1", "test_modules": list(modules),
            "status": "complete_" + prefix + "_implementation",
            "source_sha256": {"synthetic_original.py": "a" * 64}, "source_unchanged_during_verification": True,
            "passed_tests": [name], "tests_run": 1, "errors": 0, "failures": 0, "skipped": 0,
            "research_training_executed": False, "research_evaluation_executed": False,
            "native_environment_evaluation_executed": False, "gpu_runtime_measured": False,
            "tests_log": {"path": "tests.txt", "sha256": file_sha256(log)}}
        write_report(path, report)
        return {"path": str(path), "sha256": file_sha256(path), "tests_log_sha256": file_sha256(log)}

    def freeze(self):
        return cohort.freeze_cohort(self.directory, matrix=self.matrix_path,
                                   verification_report=self.proof, confirm_test_bank_unused=True)

    def backend(self, variant):
        owner = self
        class Worker:
            def __init__(self, directory, seed, kwargs):
                self.directory, self.seed, self.variant, self.cycles = directory, seed, variant, 0
                self.stopped = False
                self.manifest = {"variant": variant, "master_seed": seed, "cohort_binding": kwargs.get("cohort_binding"),
                    "paths": {name: str(directory / f"stages/{name}") for name in ("teacher", "buffer", "distillation", "local_cycle")}}
                if variant == "D":
                    initial = directory / "initial_student"
                    save_cycle_checkpoint(initial, state={"fixture": torch.zeros(1)}, metadata={"fixture_only": True})
                    self.manifest["initial_student"] = {"path": str(initial)}
                    write_report(directory / "d_binding.json", {"fixture_only": True})
                write_report(directory / "run_manifest.json", self.manifest)
                self.checkpoint()
            def checkpoint(self):
                state = {"stage_evidence": {}}
                path = self.directory / f"checkpoints/state_{self.cycles+1:08d}_aaaaaaaa"
                save_cycle_checkpoint(path, state={"controller": state}, metadata={"fixture_only": True})
            def pairing(self):
                initial = owner.inputs["per_seed"][str(self.seed)]["initial_binding"]
                return {"actor_cpu": str(self.seed+7) * 64, "critic_cpu": str(self.seed+8) * 64,
                    "input_binding_sha256": canonical_hash(initial),
                    "world_model_checkpoint_sha256": initial["selection"]["checkpoint_sha256"],
                    "normalization_identity_sha256": initial["normalization"]["identity_sha256"],
                    "replay_manifest_sha256": initial["provenance"]["replays"][0]["manifest_sha256"]}
            def request_stop(self):
                self.stopped = True
            def run(self, **kwargs):
                if self.stopped:
                    return
                self.cycles = min(20, self.cycles + (kwargs.get("max_cycles") or kwargs.get("max_units") or 20))
                self.checkpoint()
                if self.cycles == 20:
                    self.export()
            def export(self):
                with torch.random.fork_rng(devices=[]):
                    actor = LocalActor(PolicyConfig(encoder_channels=(8, 8, 8, 8), embedding_size=8,
                        hidden_size=8, critic_hidden_sizes=(16, 16)))
                metadata = {"schema_version": "p35-local-actor-v1", "task_family": "relay", "variant": variant,
                    "run_seed": self.seed, "real_joint_transitions": 50000, "actual_initial_replay_interactions": 50000,
                    "untrained_zero_interaction_baseline": False, "actor_version": actor_version(actor),
                    "actor_config": json.loads(json.dumps(asdict(actor.config))), "world_model_version": "f" * 64,
                    "model_updates": 4500, "local_ppo_cycles": 20, "actor_optimizer_steps": 320,
                    "critic_optimizer_steps": 320, "evaluation_split": "test" if variant == "B" else "validation",
                    "on_registered_evaluation_grid": False, "evaluation_executed": False, "phase3_acceptance": False}
                export = self.directory / "actor_export"
                save_cycle_checkpoint(export, state={"actor": actor.state_dict()}, metadata=metadata)
                final = {"actor_export": "actor_export"} if variant == "B" else {"actor_export": {"path": str(export)}}
                write_report(self.directory / ("final_student.json" if variant == "B" else "completed_student.json"), final)
                if variant == "D":
                    write_report(self.directory / "accounting.json", {"fixture": True})
        def prepare(path, **kwargs):
            path = Path(path)
            path.mkdir(parents=True)
            seed = kwargs.get("master_seed", kwargs.get("seed"))
            w = Worker(path, seed, kwargs)
            owner.workers[str(path)] = w
            return w
        def inspect(path):
            w = owner.workers[str(path)]
            return {"status": "complete" if w.cycles == 20 else "prepared", "phase": "complete" if w.cycles == 20 else "local",
                "local_cycles_completed": w.cycles, "imagined_joint_steps": w.cycles * 256,
                "teacher_cycles_completed": 25 if variant == "D" and w.cycles == 20 else 0,
                "teacher_actor_optimizer_steps": 400 if variant == "D" and w.cycles == 20 else 0,
                "teacher_critic_optimizer_steps": 400 if variant == "D" and w.cycles == 20 else 0,
                "trajectories_completed": 128 if variant == "D" and w.cycles == 20 else 0,
                "distillation_optimizer_steps": 100 if variant == "D" and w.cycles == 20 else 0}
        return SimpleNamespace(prepare=prepare, inspect=inspect,
            resume=lambda directory, **_: owner.workers[str(directory)])

    def fake_learners(self):
        backends = {v: self.backend(v) for v in ("B", "D")}
        self.enterContext(patch.object(cohort, "_backend", side_effect=lambda v: backends[v]))


class EndpointCohortTests(CohortFixture):
    def test_freeze_preserves_candidate_and_creates_four_fresh_slots_without_training(self):
        original = self.matrix_path.read_bytes()
        c = self.freeze()
        self.assertTrue(c["execution_ready"])
        self.assertTrue(c["cohort_frozen"])
        self.assertFalse(c["research_training_executed_at_freeze"])
        self.assertEqual(self.matrix_path.read_bytes(), original)
        self.assertEqual(cohort.inspect_cohort(c["report_path"]), c)
        self.assertEqual(len(c["run_matrix"]), 4)
        self.assertFalse(any(Path(row["run_dir"]).exists() for row in c["run_matrix"]))

    def test_freeze_rejects_missing_nonuse_attestation_changed_sources_or_development_slots(self):
        with self.assertRaisesRegex(ValueError, "unused"):
            cohort.freeze_cohort(self.directory, matrix=self.matrix_path, verification_report=self.proof)
        self.matrix["source_sha256"]["synthetic_original.py"] = "f" * 64
        rewrite(self.matrix_path, self.matrix)
        with self.assertRaisesRegex(ValueError, "unchanged"):
            self.freeze()

    def test_existing_output_prior_test_claim_and_changed_proof_fail_closed(self):
        c = self.freeze()
        with self.assertRaises(FileExistsError):
            self.freeze()
        claim = self.root / "artifacts/old/endpoint_test_claim.json"
        claim.parent.mkdir(parents=True)
        write_report(claim, {"bank_sha256": c["bank_sha256"]})
        self.directory = self.root / "another"
        with self.assertRaisesRegex(ValueError, "execution claim"):
            self.freeze()
        self.proof.write_text("changed")
        with self.assertRaisesRegex(ValueError, "receipt"):
            cohort.inspect_cohort(c["report_path"])

    def test_preparation_pairing_checkpointed_training_and_final_handoff(self):
        c = self.freeze()
        self.fake_learners()
        first = cohort.prepare_run(c["report_path"], variant="B", seed=0, device="cpu")
        self.assertEqual(first["local_cycles_completed"], 0)
        with self.assertRaises((ValueError, OSError)):
            cohort.resume_run(first["run_dir"], device="cpu").run()
        for v, seed in (("D", 0), ("B", 1), ("D", 1)):
            cohort.prepare_run(c["report_path"], variant=v, seed=seed, device="cpu")
        with self.assertRaisesRegex(ValueError, "finish"):
            cohort.final_actor(first["run_dir"])
        for row in c["run_matrix"]:
            result = cohort.resume_run(row["run_dir"], device="cpu").run()
            self.assertEqual(result["status"], "complete")
        final = cohort.final_actor(first["run_dir"])
        self.assertEqual(final["variant"], "B")
        self.assertTrue(final["binding"]["all_four_training_complete"])
        self.assertEqual(final["bank_sha256"], c["bank_sha256"])
        # Metadata/receipt reads authenticate evaluation without a controller.
        with patch.object(cohort, "_backend", side_effect=AssertionError("actor path imported a learner")):
            self.assertEqual(cohort.final_actor(first["run_dir"]), final)
        Path(self.inputs["bound_files"][0]["path"]).write_text("changed frozen input")
        with self.assertRaisesRegex(ValueError, "audited bytes"):
            cohort.final_actor(first["run_dir"])

    def test_interrupted_audit_publication_is_preserved_and_recovered(self):
        c = self.freeze()
        self.fake_learners()
        first = cohort.prepare_run(c["report_path"], variant="B", seed=0, device="cpu")
        directory = Path(first["run_dir"])
        self.workers[str(directory)].run(max_cycles=1)
        partial = directory / "endpoint_audits/audit_00000002.json"
        partial.write_text('{"interrupted": true}', encoding="utf-8")
        result = cohort.audit_run(directory)
        self.assertEqual(result["local_cycles_completed"], 1)
        preserved = list((partial.parent / "uncommitted").glob("*.json"))
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_text(), '{"interrupted": true}')
        partial.with_suffix(".sha256").write_text("invalid receipt", encoding="ascii")
        with self.assertRaises(ValueError):
            cohort.audit_run(directory)

    def test_d_initial_snapshot_and_binding_remain_bound_after_audit(self):
        c = self.freeze()
        self.fake_learners()
        prepared = cohort.prepare_run(c["report_path"], variant="D", seed=0, device="cpu")
        directory = Path(prepared["run_dir"])
        files = {r["path"] for r in read_report(prepared["audit_path"])["audit"]["files"]}
        self.assertIn(str(directory / "initial_student/state.pt"), files)
        self.assertIn(str(directory / "d_binding.sha256"), files)
        (directory / "initial_student/state.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "audited bytes"):
            cohort.inspect_run(directory)

    def test_ahead_checkpoint_requires_audit_refresh_and_snapshot_tamper_is_rejected(self):
        c = self.freeze()
        self.fake_learners()
        first = cohort.prepare_run(c["report_path"], variant="B", seed=0, device="cpu")
        directory = Path(first["run_dir"])
        self.workers[str(directory)].run(max_cycles=1)
        with self.assertRaisesRegex(ValueError, "behind"):
            cohort.inspect_run(directory)
        result = cohort.audit_run(directory)
        self.assertEqual(result["local_cycles_completed"], 1)
        pub = read_report(result["audit_path"])
        pub["audit"]["run_identity"] = "f" * 64
        rewrite(Path(result["audit_path"]), pub)
        with self.assertRaisesRegex(ValueError, "behind"):
            cohort.inspect_run(directory)

    def test_slot_reuse_wrong_seed_and_immutable_input_mutation_are_rejected(self):
        c = self.freeze()
        self.fake_learners()
        first = cohort.prepare_run(c["report_path"], variant="B", seed=0, device="cpu")
        with self.assertRaises(FileExistsError):
            cohort.prepare_run(c["report_path"], variant="B", seed=0, device="cpu")
        with self.assertRaises(ValueError):
            cohort.prepare_run(c["report_path"], variant="B", seed=True, device="cpu")
        Path(self.inputs["bound_files"][0]["path"]).write_text("changed input")
        with self.assertRaisesRegex(ValueError, "audited bytes"):
            cohort.inspect_cohort(c["report_path"], check_inputs=True)

    def test_stop_is_a_request_and_cannot_advance_training(self):
        c = self.freeze()
        self.fake_learners()
        run = cohort.prepare_run(c["report_path"], variant="B", seed=0, device="cpu")
        self.assertEqual(cohort.request_stop(run["run_dir"])["status"], "stop_requested")
        self.assertEqual(cohort.inspect_run(run["run_dir"])["local_cycles_completed"], 0)


class EndpointProofTests(unittest.TestCase):
    def test_required_id_discovery_has_no_duplicates_or_missing_modules(self):
        ids = evidence.expected_ids()
        self.assertTrue(ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(any("_FailedTest" in name for name in ids))
