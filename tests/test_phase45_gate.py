"""Fail-closed engineering decisions with temporary synthetic evidence only.

Numerical stage correctness is exercised by the separate real tiny audit suite.
These tests focus on evidence authentication and avoiding accidental promotion.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from world_models.data.ledger import file_sha256
from world_models_acceptance import evidence, gate


class _EvidenceFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = {"synthetic-test-only.py": "a" * 64}
        self.enterContext(patch.object(evidence, "source_identity", return_value=self.sources))

    def verification(self, label="verify", *, rendering="full"):
        directory = self.root / label
        directory.mkdir()
        passed = sorted({test for group in evidence.REQUIRED_CHECKS.values() for test in group})
        lines = []
        for test in passed:
            parenthetical = test if rendering == "full" else test.rsplit(".", 1)[0]
            lines.append(f"{test.rsplit('.', 1)[-1]} ({parenthetical}) ... ok\n")
        lines += [f"\n----------------------------------------------------------------------\nRan {len(passed)} tests in 0.001s\n\nOK\n"]
        log = directory / "tests.txt"
        log.write_text("".join(lines), encoding="utf-8")
        report = {"schema_version": "p45-implementation-v1", "status": "complete_p45_implementation",
            "source_unchanged_during_verification": True, "source_sha256": deepcopy(self.sources),
            "verification_contract_sha256": evidence.verification_contract_identity(),
            "test_modules": list(evidence.TEST_MODULES), "tests_run": len(passed), "passed_tests": passed,
            "failures": 0, "errors": 0, "skipped": 0, "requirement_coverage": evidence.requirement_coverage(passed),
            "all_requirements_passed": True, "research_training_executed": False, "research_replay_used": False,
            "gpu_runtime_measured": False, "phase4_acceptance": False,
            "tests_log": {"path": "tests.txt", "sha256": file_sha256(log)}}
        path = directory / "implementation_checks.json"
        evidence.write_report(path, report)
        return path, report

    def replace_report(self, path, report):
        path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        path.with_suffix(".sha256").write_text(f"{file_sha256(path)}  {path.name}\n", encoding="ascii")


class Phase45EvidenceTests(_EvidenceFixture, unittest.TestCase):
    def test_verification_requires_exact_named_coverage_and_both_unittest_log_formats(self):
        for rendering in ("full", "class"):
            path, _ = self.verification(rendering, rendering=rendering)
            report, coverage, bound = evidence.load_valid_verification(path)
            self.assertEqual(len(coverage), 8)
            self.assertTrue(all(row["status"] == "passed" for row in coverage.values()))
            self.assertEqual(bound["sha256"], file_sha256(path))
            self.assertFalse(report["phase4_acceptance"])

    def test_one_missing_required_test_cannot_be_replaced_by_a_large_pass_count(self):
        path, report = self.verification()
        report["passed_tests"][0] = "tests.fake.ManyPassingTests.test_unrelated"
        report["requirement_coverage"] = evidence.requirement_coverage(report["passed_tests"])
        self.replace_report(path, report)
        with self.assertRaisesRegex(ValueError, "missing required"):
            evidence.load_valid_verification(path)

    def test_stale_sources_contract_skips_or_research_claims_do_not_pass(self):
        for index, (key, value) in enumerate((("source_sha256", {"changed": "source"}),
                ("verification_contract_sha256", "0" * 64), ("test_modules", []),
                ("skipped", 1), ("research_training_executed", True), ("phase4_acceptance", True))):
            with self.subTest(field=key):
                path, report = self.verification(str(index))
                report[key] = value
                self.replace_report(path, report)
                with self.assertRaises(ValueError):
                    evidence.load_valid_verification(path)

    def test_log_and_receipt_tampering_and_duplicate_pass_ids_are_rejected(self):
        path, report = self.verification("log")
        (path.parent / "tests.txt").write_text("OK\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            evidence.load_valid_verification(path)
        path, report = self.verification("receipt")
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "receipt"):
            evidence.load_valid_verification(path)
        path, report = self.verification("duplicate")
        report["passed_tests"].append(report["passed_tests"][0])
        report["tests_run"] += 1
        self.replace_report(path, report)
        with self.assertRaisesRegex(ValueError, "identities"):
            evidence.load_valid_verification(path)

    def test_matching_checksum_cannot_hide_a_truncated_or_falsified_test_log(self):
        path, report = self.verification()
        log = path.parent / "tests.txt"
        log.write_text("\n".join(log.read_text().splitlines()[1:]) + "\n")
        report["tests_log"]["sha256"] = file_sha256(log)
        self.replace_report(path, report)
        with self.assertRaisesRegex(ValueError, "test-log identities"):
            evidence.load_valid_verification(path)


class Phase45GateTests(_EvidenceFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.object(gate, "source_identity", return_value=self.sources))
        self.run = self.root / "completed_campaign"
        self.run.mkdir()
        self.bound = self.run / "immutable.txt"
        self.bound.write_text("saved research bytes")
        self.export = self.run / "actor_export"
        self.export.mkdir()
        (self.export / "manifest.json").write_text("actor metadata")
        (self.export / "state.pt").write_bytes(b"actor-only-state")
        self.audit = {"schema_version": "p45-campaign-audit-v1", "engineering_passed": True,
            "actor_export": {"path": str(self.export), "manifest_sha256": file_sha256(self.export / "manifest.json"),
                "payload_sha256": file_sha256(self.export / "state.pt"), "source_device": "cuda:0"},
            "mode": "adopted_development", "fresh_cohort_eligible": False,
            "bound_files": [{"path": str(self.bound), "sha256": file_sha256(self.bound)}]}
        self.audit_mock = self.enterContext(patch.object(gate, "audit_campaign", return_value=self.audit))
        self.deployment = {"status": "passed", "passed": True, "local_only": True, "forbidden_modules_loaded": [],
            "manifest_sha256": self.audit["actor_export"]["manifest_sha256"],
            "payload_sha256": self.audit["actor_export"]["payload_sha256"]}
        self.probe = self.enterContext(patch.object(gate, "probe_actor_export", return_value=self.deployment))

    def accept(self, name="accept"):
        path, _ = self.verification(f"{name}_verify")
        return gate.accept_campaign(self.run, self.root / name, verification_report=path)

    def test_pass_is_single_run_engineering_only_and_leaves_input_bytes_unchanged(self):
        before = {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        report = self.accept()
        self.assertTrue(report["phase4_acceptance"])
        self.assertTrue(report["phase4_engineering_acceptance"])
        for key in ("research_training_executed", "native_environment_evaluation_executed",
                    "learned_student_quality_established", "b_versus_d_efficacy_established",
                    "cohort_acceptance", "fresh_cohort_eligible", "automatic_phase5_launch"):
            self.assertIs(report[key], False, key)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.run.rglob("*") if p.is_file()})
        published = self.root / "accept/phase4_acceptance.json"
        self.assertEqual(evidence.read_report(published), report)
        self.assertTrue((self.root / "accept/campaign_audit.sha256").is_file())

    def test_incomplete_or_invalid_campaign_writes_needs_review_without_deployment(self):
        self.audit_mock.side_effect = ValueError("completed campaign receipt missing")
        report = self.accept()
        self.assertFalse(report["phase4_acceptance"])
        self.assertEqual(report["status"], "needs_review")
        self.assertIn("receipt missing", report["reasons"][0])
        self.probe.assert_not_called()

    def test_invalid_verification_prevents_any_campaign_or_deployment_work(self):
        path, _ = self.verification()
        path.with_suffix(".sha256").write_text("invalid")
        report = gate.accept_campaign(self.run, self.root / "accept", verification_report=path)
        self.assertFalse(report["phase4_engineering_acceptance"])
        self.audit_mock.assert_not_called()
        self.probe.assert_not_called()

    def test_deployment_failure_or_different_export_cannot_accept(self):
        for index, field in enumerate(("passed", "local_only", "manifest_sha256", "forbidden_modules_loaded")):
            self.deployment[field] = {"passed": False, "local_only": False, "manifest_sha256": "0" * 64,
                                      "forbidden_modules_loaded": ["world_models.models"]}[field]
            report = self.accept(str(index))
            self.assertFalse(report["phase4_acceptance"])
            self.deployment.update(passed=True, local_only=True, forbidden_modules_loaded=[],
                                   manifest_sha256=self.audit["actor_export"]["manifest_sha256"])

    def test_input_changed_during_deployment_rejects_the_decision(self):
        def mutate(*args, **kwargs):
            self.bound.write_text("changed")
            return self.deployment
        self.probe.side_effect = mutate
        report = self.accept()
        self.assertFalse(report["phase4_engineering_acceptance"])
        self.assertIn("changed during acceptance", report["reasons"][0])

    def test_source_changed_during_acceptance_rejects_even_passing_artifacts(self):
        with patch.object(gate, "source_identity", side_effect=[self.sources, {"changed": "source"}, {"changed": "source"}]):
            report = self.accept()
        self.assertFalse(report["phase4_acceptance"])
        self.assertFalse(report["source_unchanged_during_acceptance"])

    def test_output_cannot_overwrite_or_enter_training_and_evidence_directories(self):
        path, _ = self.verification()
        for destination in (self.run / "report", path.parent / "report", self.run):
            with self.subTest(destination=destination), self.assertRaises((ValueError, FileExistsError)):
                gate.accept_campaign(self.run, destination, verification_report=path)
        report = gate.accept_campaign(self.run, self.root / "fresh", verification_report=path)
        self.assertTrue(report["phase4_acceptance"])
        with self.assertRaises(FileExistsError):
            gate.accept_campaign(self.run, self.root / "fresh", verification_report=path)


if __name__ == "__main__":
    unittest.main()
