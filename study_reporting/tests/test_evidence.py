"""Source identity and complete test-log authentication, using tiny fixtures."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from study_reporting import evidence


class ReportingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.proof = self.root / "implementation_checks.json"
        self.log = self.root / "tests.txt"
        self.name = "study_reporting.tests.Fixture.test_ok"
        self.log.write_text(f"test_ok ({self.name}) ... ok\n\nRan 1 test in 0.1s\n\nOK\n", encoding="utf-8")
        self.source = {"study_reporting/fake.py": "a" * 64}
        self.frozen = {"src/original.py": "b" * 64}
        self.enterContext(patch.object(evidence, "source_identity", return_value=self.source))
        self.enterContext(patch.object(evidence, "frozen_source_identity", return_value=self.frozen))
        self.enterContext(patch.object(evidence, "expected_ids", return_value={self.name}))
        self.report = {"schema_version": evidence.SCHEMA, "status": evidence.STATUS,
            "test_modules": list(evidence.TEST_MODULES), "source_sha256": self.source,
            "frozen_source_sha256": self.frozen, "source_unchanged_during_verification": True,
            "tests_run": 1, "passed_tests": [self.name], "failures": 0, "errors": 0, "skipped": 0,
            "research_training_executed": False, "research_evaluation_executed": False,
            "gpu_runtime_measured": False,
            "tests_log": {"path": "tests.txt", "sha256": evidence.file_sha256(self.log)}}

    def publish(self, report=None):
        if self.proof.exists():
            self.proof.unlink()
            self.proof.with_suffix(".sha256").unlink()
        evidence.write_report(self.proof, report or self.report)

    def test_complete_exact_fixture_proof_is_authenticated(self):
        self.publish()
        self.assertEqual(evidence.validate_verification(self.proof), self.report)

    def test_changed_frozen_or_extension_source_is_rejected(self):
        for key in ("source_sha256", "frozen_source_sha256"):
            report = deepcopy(self.report)
            report[key] = {"changed": "f" * 64}
            self.publish(report)
            with self.assertRaisesRegex(ValueError, "stale"):
                evidence.validate_verification(self.proof)

    def test_empty_skipped_duplicate_or_wrong_ids_cannot_pass(self):
        for updates in ({"skipped": 1}, {"tests_run": 0, "passed_tests": []},
                        {"passed_tests": [self.name, self.name], "tests_run": 2},
                        {"passed_tests": ["other"], "tests_run": 1}, {"errors": False}):
            self.publish({**self.report, **updates})
            with self.assertRaises(ValueError):
                evidence.validate_verification(self.proof)

    def test_truncated_log_and_invalid_receipt_are_rejected(self):
        self.log.write_text("incomplete", encoding="utf-8")
        self.report["tests_log"]["sha256"] = evidence.file_sha256(self.log)
        self.publish()
        with self.assertRaisesRegex(ValueError, "truncated"):
            evidence.validate_verification(self.proof)
        self.proof.with_suffix(".sha256").write_text("bad receipt")
        with self.assertRaisesRegex(ValueError, "receipt"):
            evidence.validate_verification(self.proof)

    def test_verification_cannot_claim_research_execution(self):
        for name in ("research_training_executed", "research_evaluation_executed", "gpu_runtime_measured"):
            self.publish({**self.report, name: True})
            with self.assertRaisesRegex(ValueError, "scientific execution"):
                evidence.validate_verification(self.proof)

    def test_reporting_is_outside_frozen_training_source_namespace(self):
        with patch.object(evidence, "ROOT", self.root):
            (self.root / "study_reporting/tests").mkdir(parents=True)
            (self.root / "study_reporting/requirements.txt").write_text("fixture")
            (self.root / "study_reporting/tool.py").write_text("# fixture")
            # Bypass the instance mock for this one actual identity operation.
            from importlib import util
            spec = util.spec_from_file_location("standalone_reporting_evidence", Path(evidence.__file__))
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.ROOT = self.root
            names = set(module.source_identity())
        self.assertEqual(names, {"study_reporting/tool.py", "study_reporting/requirements.txt"})
        self.assertFalse(any(name.startswith(("src/", "scripts/", "tests/")) for name in names))
