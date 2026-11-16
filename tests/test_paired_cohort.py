"""Synthetic matrix provenance and fail-closed candidate scheduling checks."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_paired import cohort, evidence
from world_models_study.evidence import read_report, write_report


class MatrixFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.sources = {"synthetic.py": "a" * 64}
        self.enterContext(patch.object(cohort, "source_identity", return_value=self.sources))
        self.enterContext(patch.object(evidence, "source_identity", return_value=self.sources))
        self.proof = self.root / "checks/implementation_checks.json"
        self.proof.parent.mkdir()
        log = self.proof.parent / "tests.txt"
        self.ids = {"tests.fixture.Check.test_ok"}
        log.write_text("test_ok (tests.fixture.Check.test_ok) ... ok\n\nRan 1 test in 0.001s\n\nOK\n", encoding="utf-8")
        self.enterContext(patch.object(evidence, "_expected_ids", return_value=self.ids))
        report = {"schema_version": "p52-implementation-v1", "status": "complete_p52_implementation",
            "test_modules": list(evidence.TEST_MODULES), "passed_tests": sorted(self.ids), "tests_run": 1,
            "failures": 0, "errors": 0, "skipped": 0, "source_sha256": self.sources,
            "source_unchanged_during_verification": True, "research_training_executed": False,
            "research_evaluation_executed": False, "gpu_runtime_measured": False,
            "tests_log": {"path": "tests.txt", "sha256": file_sha256(log)}}
        write_report(self.proof, report)
        self.parent = self.root / "protocol/protocol_registration.json"
        self.parent.parent.mkdir()
        inputs = self.root / "inputs"
        inputs.mkdir()
        selected = inputs / "selection.json"
        selected.write_text("synthetic only", encoding="utf-8")
        self.initial = {"seed": 0, "synthetic_only": True}
        self.parent_doc = {"input_roots": {"initial_selection": str(selected),
            "data_root": str(inputs / "data"), "statistics_root": str(inputs / "stats"),
            "config_root": str(inputs / "configs")},
            "inputs": {"bound_files": [{"path": str(selected), "sha256": file_sha256(selected)}],
                "per_seed": {str(s): {"initial_binding": {"seed": s, "synthetic_only": True}} for s in (0, 1)}},
            "test_bank_sha256": "f" * 64, "candidate_only": True}
        write_report(self.parent, self.parent_doc)
        self.parent_inspect = self.enterContext(patch.object(cohort, "inspect_registration",
            side_effect=lambda p: read_report(p)))

    def register(self):
        return cohort.register_cohort(self.root / "matrix", registration=self.parent, verification_report=self.proof)

    @staticmethod
    def rewrite(path, value):
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        path.with_suffix(".sha256").write_text(f"{file_sha256(path)}  {path.name}\n", encoding="ascii")

    def fake_backend(self, *, pairing=None):
        default = {"actor_cpu": "a" * 64, "critic_cpu": "b" * 64,
                   "input_binding_sha256": canonical_hash(self.initial),
                   "world_model_checkpoint_sha256": "c" * 64,
                   "normalization_identity_sha256": "d" * 64,
                   "replay_manifest_sha256": "e" * 64}
        instance = SimpleNamespace(pairing=lambda: deepcopy(pairing or default),
            status=lambda: {"status": "prepared", "local_cycles_completed": 0},
            run=lambda **kw: kw, request_stop=lambda: None, manifest={})
        def prepare(path, **kwargs):
            Path(path).mkdir(parents=True)
            instance.manifest["cohort_binding"] = kwargs.get("cohort_binding")
            write_report(Path(path) / "run_manifest.json", instance.manifest)
            return instance
        backend = SimpleNamespace(prepare=Mock(side_effect=prepare), resume=lambda *a, **kw: instance,
            inspect=lambda directory: {"local_cycles_completed": 0, "target_local_cycles": 20,
                                        "phase": "local", "live_status_is_authoritative": False})
        self.enterContext(patch.object(cohort, "_backend", return_value=backend))
        return backend, instance


class MatrixTests(MatrixFixture):
    def test_plan_has_four_fixed_slots_and_exact_workload(self):
        p = cohort.plan()
        self.assertEqual(len(p["run_matrix"]), 4)
        self.assertEqual({r["case"] for r in p["run_matrix"]},
                         {f"relay/{v}/seed_{s}" for v in ("B", "D") for s in (0, 1)})
        for row in p["run_matrix"]:
            self.assertEqual((row["local_cycles"], row["local_actor_optimizer_steps"], row["initial_model_updates"]),
                             (20, 320, 4500))
            self.assertEqual(row["new_model_updates"], 0)
        self.assertFalse(p["execution_ready"])
        self.assertTrue(p["training_tools_ready"])

    def test_registration_authenticates_inputs_proof_and_does_no_training(self):
        with patch.object(cohort, "_backend") as backend:
            r = self.register()
            self.assertEqual(cohort.inspect_cohort(r["report_path"]), r)
            backend.assert_not_called()
        self.assertTrue(r["candidate_only"])
        self.assertFalse(r["cohort_frozen"])
        self.assertFalse(r["automatic_training_launch"])

    def test_modified_slots_readiness_or_current_sources_are_rejected(self):
        r = self.register()
        p = Path(r["report_path"])
        for field, value in (("execution_ready", True), ("fresh_cohort_eligible", True),
                             ("plan", {}), ("source_sha256", {})):
            bad = {**r, field: value}
            self.rewrite(p, bad)
            with self.subTest(field=field), self.assertRaises(ValueError):
                cohort.inspect_cohort(p)
        self.rewrite(p, r)
        self.assertEqual(cohort.inspect_cohort(p), r)

    def test_stale_proof_fails_before_parent_input_validation(self):
        r = read_report(self.proof)
        r["source_sha256"] = {}
        self.rewrite(self.proof, r)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.register()
        self.parent_inspect.assert_not_called()
        self.assertFalse((self.root / "matrix").exists())

    def test_input_mutation_during_registration_is_detected_before_publication(self):
        validate = cohort.validate_verification
        calls = []
        def moving(path):
            result = validate(path)
            calls.append(path)
            if len(calls) == 2:
                Path(self.parent_doc["input_roots"]["initial_selection"]).write_text("moved inputs")
            return result
        with patch.object(cohort, "validate_verification", side_effect=moving):
            with self.assertRaisesRegex(ValueError, "immutable input changed"):
                self.register()
        self.assertFalse((self.root / "matrix").exists())

    def test_existing_or_input_nested_outputs_are_rejected(self):
        self.register()
        with self.assertRaises(FileExistsError):
            self.register()
        with self.assertRaisesRegex(ValueError, "separate"):
            cohort.register_cohort(self.root / "inputs/nested", registration=self.parent,
                                    verification_report=self.proof)

    def test_prospective_training_is_blocked_before_any_network_preparation(self):
        r = self.register()
        with patch.object(cohort, "_backend") as backend, self.assertRaisesRegex(ValueError, "P5.4/P5.5"):
            cohort.prepare_run(self.root / "matrix/runs/B/seed_0", cohort=r["report_path"],
                               variant="B", seed=0, device="cpu")
        backend.assert_not_called()

    def test_explicit_development_slot_can_prepare_resume_and_stop(self):
        r = self.register()
        _, instance = self.fake_backend()
        directory = self.root / "matrix/runs/B/seed_0"
        result = cohort.prepare_run(directory, cohort=r["report_path"], variant="B", seed=0,
                                    device="cpu", development_only=True)
        self.assertEqual(result["status"], "prepared")
        binding = read_report(directory / "matrix_run.json")
        self.assertFalse(binding["fresh_cohort_eligible"])
        with self.assertRaises(FileExistsError):
            cohort.prepare_run(directory, cohort=r["report_path"], variant="B", seed=0,
                               device="cpu", development_only=True)
        self.assertEqual(cohort.resume_run(directory, device="cpu").run(max_units=2), {"max_cycles": 2})
        self.assertEqual(cohort.request_stop(directory)["status"], "stop_requested")
        self.assertTrue((directory / "stop.request").is_file())
        self.assertEqual(cohort.read_status(directory)["local_cycles_completed"], 0)

    def test_wrong_seed_variant_slot_and_frozen_input_mismatch_fail(self):
        r = self.register()
        for seed, variant, directory in ((True, "B", "B/seed_0"), (2, "B", "B/seed_2"),
                                         (0, "A", "A/seed_0"), (0, "B", "D/seed_0")):
            with self.subTest(seed=seed, variant=variant), self.assertRaises(ValueError):
                cohort.prepare_run(self.root / "matrix/runs" / directory, cohort=r["report_path"],
                                   variant=variant, seed=seed, device="cpu", development_only=True)
        self.fake_backend(pairing={"input_binding_sha256": "f" * 64})
        with self.assertRaisesRegex(ValueError, "registered seed"):
            cohort.prepare_run(self.root / "matrix/runs/B/seed_0", cohort=r["report_path"],
                               variant="B", seed=0, device="cpu", development_only=True)

    def test_interrupted_zero_work_matrix_publication_recovers_without_repreparing(self):
        r = self.register()
        backend, _ = self.fake_backend()
        directory = self.root / "matrix/runs/B/seed_0"
        def interrupted(path, report):
            path.write_text("partial unpublished metadata", encoding="utf-8")
            raise OSError("fixture publication interrupted")
        with patch.object(cohort, "write_report", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "interrupted"):
                cohort.prepare_run(directory, cohort=r["report_path"], variant="B", seed=0,
                                   device="cpu", development_only=True)
        self.assertEqual(backend.prepare.call_count, 1)
        result = cohort.prepare_run(directory, cohort=r["report_path"], variant="B", seed=0,
                                    device="cpu", development_only=True)
        self.assertEqual(backend.prepare.call_count, 1)
        self.assertEqual(result["local_cycles_completed"], 0)
        self.assertEqual(len(list((directory / "uncommitted").glob("matrix_run_*.json"))), 1)
        self.assertTrue(read_report(directory / "matrix_run.json")["case"].endswith("B/seed_0"))

    def test_missing_matrix_binding_cannot_adopt_trained_or_other_candidate(self):
        r = self.register()
        backend, instance = self.fake_backend()
        directory = self.root / "matrix/runs/B/seed_0"
        with patch.object(cohort, "write_report", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                cohort.prepare_run(directory, cohort=r["report_path"], variant="B", seed=0,
                                   device="cpu", development_only=True)
        backend.inspect = lambda path: {"local_cycles_completed": 1, "phase": "local"}
        with self.assertRaisesRegex(ValueError, "zero-work"):
            cohort.prepare_run(directory, cohort=r["report_path"], variant="B", seed=0,
                               device="cpu", development_only=True)
        instance.manifest["cohort_binding"] = {"different": "matrix"}
        with self.assertRaisesRegex(ValueError, "different matrix"):
            cohort.prepare_run(directory, cohort=r["report_path"], variant="B", seed=0,
                               device="cpu", development_only=True)

    def test_pairing_mismatch_between_arms_is_rejected(self):
        r = self.register()
        self.fake_backend()
        for variant in ("B", "D"):
            directory = self.root / f"matrix/runs/{variant}/seed_0"
            cohort.prepare_run(directory, cohort=r["report_path"], variant=variant, seed=0,
                               device="cpu", development_only=True)
        p = self.root / "matrix/runs/D/seed_0/matrix_run.json"
        saved = read_report(p)
        saved["pairing"]["actor_cpu"] = "f" * 64
        self.rewrite(p, saved)
        with self.assertRaisesRegex(ValueError, "initial actor"):
            cohort.resume_run(self.root / "matrix/runs/B/seed_0", device="cpu")

    def test_replacing_learner_manifest_or_changing_inner_matrix_binding_is_rejected(self):
        r = self.register()
        _, instance = self.fake_backend()
        directory = self.root / "matrix/runs/B/seed_0"
        cohort.prepare_run(directory, cohort=r["report_path"], variant="B", seed=0,
                           device="cpu", development_only=True)
        manifest = directory / "run_manifest.json"
        original = read_report(manifest)
        original_bytes = manifest.read_bytes()
        original_receipt = manifest.with_suffix(".sha256").read_bytes()
        self.rewrite(manifest, {**original, "different_run": True})
        with self.assertRaisesRegex(ValueError, "learner manifest differs"):
            cohort.resume_run(directory, device="cpu")
        manifest.write_bytes(original_bytes)
        manifest.with_suffix(".sha256").write_bytes(original_receipt)
        instance.manifest["cohort_binding"] = {"different_matrix": True}
        with self.assertRaisesRegex(ValueError, "different matrix"):
            cohort.resume_run(directory, device="cpu")


class ProofTests(MatrixFixture):
    def test_required_test_id_log_receipt_and_research_markers(self):
        self.assertEqual(evidence.validate_verification(self.proof)["tests_run"], 1)
        original = read_report(self.proof)
        for key, value in (("passed_tests", []), ("skipped", 1), ("research_training_executed", True)):
            self.rewrite(self.proof, {**original, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                evidence.validate_verification(self.proof)
        self.rewrite(self.proof, original)
        self.proof.write_text(self.proof.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "receipt"):
            evidence.validate_verification(self.proof)

    def test_truncated_test_log_is_rejected_even_with_matching_hashes(self):
        original = read_report(self.proof)
        log = self.proof.parent / "tests.txt"
        log.write_text("Ran 1 test in 0.01s\nOK\n", encoding="utf-8")
        original["tests_log"]["sha256"] = file_sha256(log)
        self.rewrite(self.proof, original)
        with self.assertRaisesRegex(ValueError, "test-log"):
            evidence.validate_verification(self.proof)
