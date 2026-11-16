"""Temporary static registration and fail-closed evidence; no research inputs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from world_models.data.ledger import file_sha256
from world_models_study import evidence, registration


class StaticFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.sources = {"synthetic_fixture.py": "a" * 64}
        self.enterContext(patch.object(evidence, "source_identity", return_value=self.sources))
        self.enterContext(patch.object(registration, "source_identity", return_value=self.sources))
        self.expected = {"tests.fixture.StaticCheck.test_receipt"}
        self.enterContext(patch.object(evidence, "_expected_ids", return_value=self.expected))
        self.verify = self.root / "verification" / "implementation_checks.json"
        self.verify.parent.mkdir()
        self.log = self.verify.parent / "tests.txt"
        self.log.write_text("test_receipt (tests.fixture.StaticCheck.test_receipt) ... ok\n\n"
                            "----------------------------------------------------------------------\n"
                            "Ran 1 test in 0.001s\n\nOK\n", encoding="utf-8")
        self.report = {"schema_version": "p51-implementation-v1", "status": "complete_p51_implementation",
            "test_modules": list(evidence.TEST_MODULES), "passed_tests": sorted(self.expected), "tests_run": 1,
            "failures": 0, "errors": 0, "skipped": 0, "source_sha256": self.sources,
            "source_unchanged_during_verification": True, "research_training_executed": False,
            "research_evaluation_executed": False, "gpu_runtime_measured": False,
            "tests_log": {"path": "tests.txt", "sha256": file_sha256(self.log)}}
        evidence.write_report(self.verify, self.report)
        store = self.root / "immutable_inputs"
        store.mkdir()
        selected = store / "selection.json"
        selected.write_text('{"data_source":"synthetic_unit_test_only"}', encoding="utf-8")
        self.roots = {"initial_selection": str(selected), "data_root": str(store / "replay"),
                      "statistics_root": str(store / "normalization"), "config_root": str(store / "config")}
        self.inputs = {"schema_version": "synthetic-binding-for-registration-unit-tests-only",
            "bound_files": [{"path": str(selected), "sha256": file_sha256(selected)}],
            "per_seed": {"0": {"test_only": True}, "1": {"test_only": True}}}
        self.binder = self.enterContext(patch.object(registration, "bind_initial_inputs",
                                                    side_effect=lambda **_: deepcopy(self.inputs)))

    def rewrite(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
        path.with_suffix(".sha256").write_text(f"{file_sha256(path)}  {path.name}\n", encoding="ascii")

    def register(self, output=None):
        return registration.register_protocol(output or self.root / "registration",
            verification_report=self.verify, **self.roots)


class StudyEvidenceTests(StaticFixture):
    def test_exact_test_ids_sources_and_receipt_are_required(self):
        self.assertEqual(evidence.validate_verification(self.verify), self.report)
        for key, value in (("source_sha256", {}), ("skipped", 1), ("passed_tests", []),
                           ("research_training_executed", True), ("research_evaluation_executed", True)):
            invalid = deepcopy(self.report)
            invalid[key] = value
            self.rewrite(self.verify, invalid)
            with self.subTest(field=key), self.assertRaises(ValueError):
                evidence.validate_verification(self.verify)
        self.rewrite(self.verify, self.report)
        self.verify.write_text(self.verify.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "receipt"):
            evidence.validate_verification(self.verify)

    def test_matching_checksums_do_not_hide_truncated_log(self):
        self.log.write_text("Ran 1 test in 0.001s\nOK\n", encoding="utf-8")
        changed = deepcopy(self.report)
        changed["tests_log"]["sha256"] = file_sha256(self.log)
        self.rewrite(self.verify, changed)
        with self.assertRaisesRegex(ValueError, "test-log"):
            evidence.validate_verification(self.verify)

    def test_legacy_unittest_log_format_is_authenticated(self):
        self.log.write_text(self.log.read_text().replace("StaticCheck.test_receipt)", "StaticCheck)"), encoding="utf-8")
        self.report["tests_log"]["sha256"] = file_sha256(self.log)
        self.rewrite(self.verify, self.report)
        self.assertEqual(evidence.validate_verification(self.verify)["tests_run"], 1)

    def test_report_publication_refuses_overwrite(self):
        previous = self.verify.read_bytes()
        with self.assertRaises(FileExistsError):
            evidence.write_report(self.verify, {"replace": True})
        self.assertEqual(self.verify.read_bytes(), previous)


class StudyRegistrationTests(StaticFixture):
    def test_real_binder_static_bank_and_registration_interoperate_without_research_weights(self):
        from tests.test_study_inputs import StudyInputTests
        from world_models_study.inputs import bind_initial_inputs
        owner = StudyInputTests("test_two_seed_inputs_are_paired_receipt_bound_json_and_readonly")
        owner.setUp()
        self.addCleanup(owner.doCleanups)
        roots = {"initial_selection": owner.selection, "data_root": owner.data,
                 "statistics_root": owner.norm, "config_root": owner.config}
        with patch.object(registration, "bind_initial_inputs", side_effect=bind_initial_inputs):
            result = registration.register_protocol(self.root / "integrated_registration",
                verification_report=self.verify, **roots)
            loaded = registration.inspect_registration(result["report_path"])
        self.assertEqual(loaded["inputs"]["physical_initial_joint_transitions"], 100000)
        self.assertEqual(set(loaded["inputs"]["per_seed"]), {"0", "1"})
        self.assertFalse(loaded["execution_ready"])
        self.assertEqual(owner.loader.call_count, 4)

    def test_registration_round_trip_preserves_inputs_and_never_promotes_execution(self):
        before = {p: p.read_bytes() for p in (self.verify, self.log, Path(self.roots["initial_selection"]))}
        result = self.register()
        self.assertTrue(result["protocol_frozen"])
        self.assertTrue(result["candidate_only"])
        for key in ("execution_ready", "cohort_frozen", "cohort_acceptance", "fresh_cohort_eligible",
                    "research_training_executed", "research_evaluation_executed", "test_bank_executed",
                    "test_bank_prior_use_confirmed_absent", "automatic_training_launch"):
            self.assertIs(result[key], False)
        self.assertEqual(registration.inspect_registration(result["report_path"]), result)
        self.assertEqual(len(result["cohort_policy"]["required_runs"]), 4)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_invalid_verification_stops_before_inputs_or_output_creation(self):
        self.verify.with_suffix(".sha256").unlink()
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.register()
        self.binder.assert_not_called()
        self.assertFalse((self.root / "registration").exists())

    def test_registration_rejects_existing_output_and_input_overlap(self):
        existing = self.root / "already_exists"
        existing.mkdir()
        for output in (existing, Path(self.roots["initial_selection"]).parent / "nested"):
            with self.subTest(output=str(output)), self.assertRaises((ValueError, FileExistsError)):
                self.register(output)
        self.binder.assert_not_called()

    def test_changed_bound_input_is_rejected_before_publication(self):
        Path(self.roots["initial_selection"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "input changed"):
            self.register()
        self.assertFalse((self.root / "registration").exists())

    def test_wrong_bank_and_self_asserted_readiness_are_rejected_even_with_new_receipt(self):
        result = self.register()
        path = Path(result["report_path"])
        for key, value in (("execution_ready", True), ("cohort_frozen", True),
                           ("test_bank_prior_use_confirmed_absent", True), ("test_bank_file", "../elsewhere.json")):
            changed = deepcopy(result)
            changed[key] = value
            self.rewrite(path, changed)
            with self.subTest(field=key), self.assertRaises(ValueError):
                registration.inspect_registration(path)
        self.rewrite(path, result)
        bank_path = path.parent / "test_bank.json"
        bank = evidence.read_report(bank_path)
        bank["bank_seed"] = 999
        self.rewrite(bank_path, bank)
        with self.assertRaises(ValueError):
            registration.inspect_registration(path)

    def test_input_metadata_is_rebound_and_cannot_relabel_development(self):
        result = self.register()
        path = Path(result["report_path"])
        changed = deepcopy(result)
        changed["inputs"]["per_seed"]["0"]["fresh_cohort_eligible"] = True
        self.rewrite(path, changed)
        with self.assertRaisesRegex(ValueError, "input metadata"):
            registration.inspect_registration(path)
        changed = deepcopy(result)
        changed["cohort_policy"]["adopted_development_eligible_automatically"] = True
        self.rewrite(path, changed)
        with self.assertRaisesRegex(ValueError, "candidate-only"):
            registration.inspect_registration(path)

    def test_source_change_during_registration_never_publishes(self):
        with patch.object(registration, "source_identity", side_effect=[self.sources, {"moving": "source"}]):
            with self.assertRaisesRegex(ValueError, "sources or verification"):
                self.register()
        self.assertFalse((self.root / "registration").exists())


if __name__ == "__main__":
    unittest.main()
