"""Synthetic source-receipt tests; no scientific jobs or real artifacts."""
from __future__ import annotations

from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from study_release import evidence


class ReleaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.identities = {"study_release/evidence.py": "a" * 64}
        self.prior = {"frozen_source_sha256": {"src/fixture.py": "b" * 64},
            "reporting_source_sha256": {"study_reporting/fixture.py": "c" * 64}}
        self.ids = {"fixtures.Case.test_first", "fixtures.Case.test_second"}
        self.stack = self.enterContext(ExitStack())
        for name, value in (("source_identity", self.identities), ("prior_identities", self.prior),
                ("expected_ids", self.ids)):
            self.stack.enter_context(patch.object(evidence, name, return_value=value))

    def proof(self, name="proof", **updates):
        directory = self.root / name
        directory.mkdir()
        log = directory / "tests.txt"
        log.write_text("test_first (fixtures.Case) ... ok\n"
            "test_second (fixtures.Case.test_second) ... ok\n\n"
            "----------------------------------------------------------------------\n"
            "Ran 2 tests in 0.001s\n\nOK\n", encoding="utf-8")
        report = {"schema_version": evidence.SCHEMA, "status": evidence.STATUS,
            "source_sha256": self.identities, **self.prior,
            "test_modules": list(evidence.TEST_MODULES),
            "source_unchanged_during_verification": True,
            "tests_run": 2, "passed_tests": sorted(self.ids),
            "failures": 0, "errors": 0, "skipped": 0,
            "tests_log": {"path": "tests.txt", "sha256": evidence.file_sha256(log)},
            "research_training_executed": False, "research_evaluation_executed": False,
            "actual_scientific_diagnosis_executed": False,
            "actual_scientific_release_generated": False,
            "actual_scientific_graphics_rendered": False, "gpu_runtime_measured": False,
            "phase6_acceptance": False, "new_environment_interactions": 0,
            "new_model_updates": 0, "new_policy_optimizer_updates": 0}
        report.update(copy.deepcopy(updates))
        path = directory / "implementation_checks.json"
        evidence.write_report(path, report)
        return path, report

    def test_complete_synthetic_proof_is_accepted(self):
        path, report = self.proof()
        self.assertEqual(evidence.validate_verification(path), report)
        self.assertFalse(report["actual_scientific_release_generated"])

    def test_schema_status_maps_modules_and_unchanged_are_strict(self):
        mutations = [{"schema_version": "old"}, {"status": "failed"},
            {"source_sha256": {}}, {"frozen_source_sha256": {}},
            {"reporting_source_sha256": {}}, {"test_modules": []},
            {"source_unchanged_during_verification": 1}]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=mutation):
                path, _ = self.proof(f"invalid_{index}", **mutation)
                with self.assertRaises(ValueError):
                    evidence.validate_verification(path)

    def test_failures_errors_and_skips_require_integer_zero(self):
        for field in ("failures", "errors", "skipped"):
            for index, value in enumerate((1, -1, False, 0.0, "0", None)):
                with self.subTest(field=field, value=value):
                    path, _ = self.proof(f"{field}_{index}", **{field: value})
                    with self.assertRaises(ValueError):
                        evidence.validate_verification(path)

    def test_scientific_execution_flags_require_false(self):
        fields = ("research_training_executed", "research_evaluation_executed",
            "actual_scientific_diagnosis_executed", "actual_scientific_release_generated",
            "actual_scientific_graphics_rendered", "gpu_runtime_measured")
        for field in fields:
            for index, value in enumerate((True, 0, None)):
                with self.subTest(field=field, value=value):
                    path, _ = self.proof(f"{field}_{index}", **{field: value})
                    with self.assertRaises(ValueError):
                        evidence.validate_verification(path)

    def test_phase_acceptance_and_new_work_counters_cannot_claim_science(self):
        for index, value in enumerate((True, 0, None)):
            with self.subTest(field="phase6_acceptance", value=value):
                path, _ = self.proof(f"acceptance_{index}", phase6_acceptance=value)
                with self.assertRaises(ValueError):
                    evidence.validate_verification(path)
        for field in ("new_environment_interactions", "new_model_updates", "new_policy_optimizer_updates"):
            for index, value in enumerate((1, -1, False, 0.0, None, "0")):
                with self.subTest(field=field, value=value):
                    path, _ = self.proof(f"{field}_{index}", **{field: value})
                    with self.assertRaises(ValueError):
                        evidence.validate_verification(path)

    def test_missing_duplicate_substituted_test_ids_and_typed_count_fail(self):
        mutations = [{"passed_tests": []}, {"passed_tests": ["fixtures.Case.test_first"]},
            {"passed_tests": ["fixtures.Case.test_first"] * 2},
            {"passed_tests": ["fixtures.Case.test_first", "fixtures.Case.test_other"]},
            {"passed_tests": "fixtures.Case.test_first"},
            {"tests_run": True}, {"tests_run": 2.0}, {"tests_run": 0}]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=mutation):
                path, _ = self.proof(f"ids_{index}", **mutation)
                with self.assertRaises(ValueError):
                    evidence.validate_verification(path)

    def test_changed_or_missing_proof_receipt_fails(self):
        path, _ = self.proof()
        path.with_suffix(".sha256").write_text("0" * 64 + "  implementation_checks.json\n", encoding="ascii")
        with self.assertRaises(ValueError):
            evidence.validate_verification(path)
        path.with_suffix(".sha256").unlink()
        with self.assertRaises(ValueError):
            evidence.validate_verification(path)

    def test_changed_log_and_resealed_truncated_log_fail(self):
        path, report = self.proof()
        log = path.parent / "tests.txt"
        log.write_text("Ran 2 tests in 0.001s\nOK\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            evidence.validate_verification(path)
        report["tests_log"]["sha256"] = evidence.file_sha256(log)
        replacement = self.root / "resealed" / "implementation_checks.json"
        replacement.parent.mkdir()
        (replacement.parent / "tests.txt").write_bytes(log.read_bytes())
        evidence.write_report(replacement, report)
        with self.assertRaises(ValueError):
            evidence.validate_verification(replacement)

    def test_log_must_have_each_success_once_correct_count_and_ok(self):
        bodies = ["test_first (fixtures.Case) ... ok\n" * 2 + "Ran 2 tests in 0s\nOK\n",
            "test_first (fixtures.Case) ... ok\ntest_second (fixtures.Case) ... ok\nRan 1 test in 0s\nOK\n",
            "test_first (fixtures.Case) ... ok\ntest_second (fixtures.Case) ... ok\nRan 2 tests in 0s\nFAILED\n"]
        for index, body in enumerate(bodies):
            path, report = self.proof(f"logbase_{index}")
            out = self.root / f"logchanged_{index}"
            out.mkdir()
            log = out / "tests.txt"
            log.write_text(body, encoding="utf-8")
            report["tests_log"]["sha256"] = evidence.file_sha256(log)
            path = evidence.write_report(out / path.name, report)
            with self.assertRaises(ValueError):
                evidence.validate_verification(path)

    def test_report_reference_bytes_and_fresh_destination_are_checked(self):
        path = evidence.write_report(self.root / "report.json", {"fixture": True})
        ref = evidence.reference(path)
        evidence.validate_references([ref])
        with self.assertRaises(FileExistsError):
            evidence.write_report(path, {"fixture": False})
        with self.assertRaises(ValueError):
            evidence.validate_references([{**ref, "bytes": ref["bytes"] + 1}])
        path.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            evidence.validate_references([ref])
        with self.assertRaises(ValueError):
            evidence.read_report(path)

    def test_non_object_json_and_unnormalized_paths_fail(self):
        path = evidence.write_report(self.root / "array.json", [1, 2])
        with self.assertRaises(ValueError):
            evidence.read_report(path)
        with self.assertRaises(ValueError):
            evidence.safe_path(self.root / "child" / ".." / "report.json")
        with self.assertRaises(ValueError):
            evidence.validate_references({})


if __name__ == "__main__":
    unittest.main()
