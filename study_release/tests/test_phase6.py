"""Synthetic Phase 6 CLI/publication tests, with no scientific execution."""
from __future__ import annotations

from copy import deepcopy
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from study_release import evidence, packaging, phase6
from study_release.tests.test_packaging import Fixture


class Phase6PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.fixture = Fixture(self.root)
        self.context = self.fixture.context
        self.context["release_inputs"] = []
        self.output = self.root / "artifacts/final_phase6"
        self.sources = {"study_release/saved.py": packaging._sha(self.root / "study_release/saved.py")}
        self.proof = {"schema_version": evidence.SCHEMA, "status": evidence.STATUS,
            "source_sha256": self.sources,
            "frozen_source_sha256": self.context["cohort"]["source_sha256"],
            "reporting_source_sha256": self.context["source_identity"]}
        self.phase5_path = Path(self.fixture.report("artifacts/phase5_acceptance.json",
            {"status": "passed", "phase5_acceptance": True})["path"])
        self.proof_path = Path(self.fixture.report("artifacts/proof/implementation_checks.json", self.proof)["path"])
        self.fixture.file("artifacts/proof/tests.txt", b"synthetic checks only\n")
        self.diagnosis = {"schema_version": "p61-fixed-trace-diagnosis-v1", "status": "complete",
            "fixtures_only": False, "cohort_identity": self.context["cohort"]["cohort_identity"],
            "selected_count": 1, "missing_categories": [{"master_seed": 0, "variant": "B", "category": "successful"}],
            "selected_traces": [{"master_seed": 0, "variant": "B", "category": "failed", "episode_index": 0,
                "template_id": "relay_test_00", "record": {"termination_cause": "failure", "episode_return": -1.002,
                    "latch_activated": False, "joint_transitions": 2},
                "summary": {"last_transition": {"path": "/synthetic-only/events.jsonl", "line": 8,
                    "row_sha256": "a" * 64}}}]}
        self.require_server = self.enterContext(patch.object(phase6.common, "require_server"))
        self.load = self.enterContext(patch.object(phase6.common, "load_completed", return_value=self.context))
        self.history = self.enterContext(patch.object(phase6.common, "attach_history"))
        self.validate_context = self.enterContext(patch.object(phase6.common, "validate_completed", return_value=True))
        self.validate_proof = self.enterContext(patch.object(phase6.evidence, "validate_verification", return_value=self.proof))
        self.source_identity = self.enterContext(patch.object(phase6.evidence, "source_identity", return_value=self.sources))
        self.guard = self.enterContext(patch.object(phase6.common, "guard_output", side_effect=self._guard))
        self.build_diagnosis = Mock(side_effect=lambda context: deepcopy(self.diagnosis))
        self.check_diagnosis = Mock()
        self.render = Mock(return_value="Synthetic fixed trace diagnosis\n")
        self.build_release = Mock(side_effect=self._minimal_release)
        self.inspect_release = Mock(wraps=packaging.inspect_release)
        self.components = self.enterContext(patch.object(phase6, "_components", return_value=(
            self.build_diagnosis, self.check_diagnosis, self.render, self.build_release, self.inspect_release)))

    def _guard(self, value, context, proof):
        output = Path(value).resolve()
        if not output.is_relative_to(self.root) or output.exists():
            raise FileExistsError("synthetic test requires a fresh confined output")
        return output

    def complete(self):
        return phase6.complete(self.phase5_path, self.output, self.proof_path)

    def _minimal_release(self, context, diagnosis, output):
        """Small byte-verified package for orchestration tests, not native data.

        One integration test below uses the complete real packager. Its own
        dedicated suite covers the full archive allowlist and provenance graph.
        """
        output = Path(output)
        output.mkdir(parents=True)
        payload = output / "synthetic_unit_payload.txt"
        payload.write_text("Synthetic orchestrator fixture only\n", encoding="utf-8")
        manifest = {"schema_version": packaging.SCHEMA, "status": "complete",
            "cohort_identity": context["cohort"]["cohort_identity"],
            "files": [{"path": payload.name, "sha256": packaging._sha(payload),
                "bytes": payload.stat().st_size, "category": "evidence"}],
            "local_actor_exports": [{"variant": variant, "master_seed": seed}
                for seed in (0, 1) for variant in ("B", "D")],
            "fixture_only": True}
        evidence.write_report(output / "release_manifest.json", manifest)
        files = sorted(path for path in output.rglob("*") if path.is_file())
        (output / "SHA256SUMS").write_text("".join(
            f"{packaging._sha(path)}  {path.relative_to(output).as_posix()}\n" for path in files), encoding="ascii")
        return packaging.inspect_release(output)

    def test_saved_evidence_pipeline_packages_bytes_without_finishing_paper(self):
        self.build_release.side_effect = lambda context, diagnosis, output: packaging.build_release(
            context, diagnosis, output, root=self.root)
        report = self.complete()
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["P6.1_complete"])
        self.assertTrue(report["P6.2_complete"])
        self.assertFalse(report["phase6_acceptance"])
        self.assertFalse(report["P6.3_paper_review_complete"])
        self.assertFalse(report["saved_record_reproduction_executed"])
        self.assertFalse(report["successful_transfer_established"])
        self.assertTrue(report["release_byte_integrity_checked"])
        self.assertGreater(report["release_files"], 0)
        manifest = packaging.inspect_release(self.output / "release")
        self.assertEqual(len(manifest["local_actor_exports"]), 4)
        self.assertEqual(evidence.read_report(self.output / "phase61_62_completion.json"), report)
        self.require_server.assert_called_once()
        self.load.assert_called_once_with(self.phase5_path)
        self.history.assert_called_once_with(self.context)
        self.validate_context.assert_called_once_with(self.context)
        self.assertEqual(self.validate_proof.call_count, 2)
        self.assertEqual(self.check_diagnosis.call_count, 2)

    def test_diagnosis_json_prose_and_csv_have_publication_provenance(self):
        self.complete()
        directory = self.output / "diagnosis"
        report = evidence.read_report(directory / "diagnosis.json")
        publication = report["publication"]
        self.assertEqual(publication["implementation_verification"], evidence.reference(self.proof_path))
        self.assertEqual(publication["phase5_acceptance"], evidence.reference(self.phase5_path))
        self.assertEqual(publication["source_sha256"], self.sources)
        self.assertEqual((directory / "diagnosis.md").read_text(encoding="utf-8"), "Synthetic fixed trace diagnosis\n")
        with (directory / "selected_traces.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["episode_index"], "0")
        self.assertEqual(rows[0]["termination_cause"], "failure")
        self.assertEqual(rows[0]["terminal_line"], "8")
        self.assertEqual(len(self.context["diagnosis_artifacts"]), 3)
        evidence.validate_references(self.context["diagnosis_artifacts"])

    def test_fixture_diagnosis_is_rejected_before_packaging(self):
        self.diagnosis["fixtures_only"] = True
        report = self.complete()
        self.assertEqual(report["status"], "needs_review")
        self.assertFalse(report["P6.1_complete"])
        self.assertFalse(report["P6.2_complete"])
        self.assertIn("not fixture outputs", report["reasons"][0])
        self.build_release.assert_not_called()
        self.assertFalse((self.output / "diagnosis").exists())

    def test_missing_or_corrupt_saved_trace_cannot_publish_acceptance(self):
        self.build_diagnosis.side_effect = ValueError("selected native trace bytes changed")
        report = self.complete()
        self.assertEqual(report["status"], "needs_review")
        self.assertIn("trace bytes changed", report["reasons"][0])
        self.assertFalse(report["phase6_acceptance"])
        self.build_release.assert_not_called()
        self.assertEqual(evidence.read_report(self.output / "phase61_62_completion.json"), report)

    def test_changed_diagnosis_source_fails_before_publication(self):
        self.check_diagnosis.side_effect = ValueError("saved diagnostic evidence changed")
        report = self.complete()
        self.assertEqual(report["status"], "needs_review")
        self.assertFalse((self.output / "diagnosis").exists())
        self.build_release.assert_not_called()

    def test_package_inventory_corruption_leaves_needs_review(self):
        def corrupt(path):
            (Path(path) / "synthetic_unit_payload.txt").write_text("changed bytes\n", encoding="utf-8")
            return packaging.inspect_release(path)
        self.inspect_release.side_effect = corrupt
        report = self.complete()
        self.assertEqual(report["status"], "needs_review")
        self.assertIn("file inventory", report["reasons"][0])
        self.assertFalse(report["P6.2_complete"])
        self.assertFalse(report["research_training_executed"])

    def test_release_substitution_during_final_inspection_is_rejected(self):
        def substitute(path):
            manifest = packaging.inspect_release(path)
            return {**manifest, "cohort_identity": "changed"}
        self.inspect_release.side_effect = substitute
        report = self.complete()
        self.assertEqual(report["status"], "needs_review")
        self.assertIn("changed during final publication", report["reasons"][0])

    def test_incomplete_or_wrong_cohort_diagnosis_cannot_be_packaged(self):
        for field, value in (("status", "needs_review"), ("cohort_identity", "wrong")):
            with self.subTest(field=field):
                # Each subcase needs its own fresh publication directory.
                self.output = self.root / "artifacts" / f"bad_{field}"
                self.diagnosis[field] = value
                report = self.complete()
                self.assertEqual(report["status"], "needs_review")
                self.assertFalse(report["P6.1_complete"])
                self.diagnosis[field] = "complete" if field == "status" else self.context["cohort"]["cohort_identity"]
        self.build_release.assert_not_called()

    def test_wrong_prior_source_proof_fails_before_any_output(self):
        for key in ("frozen_source_sha256", "reporting_source_sha256"):
            with self.subTest(key=key):
                original = self.proof[key]
                self.proof[key] = {"changed.py": "0" * 64}
                with self.assertRaisesRegex(ValueError, "proof differs"):
                    self.complete()
                self.assertFalse(self.output.exists())
                self.proof[key] = original
        self.build_diagnosis.assert_not_called()
        self.guard.assert_not_called()

    def test_invalid_or_missing_phase5_completion_fails_before_any_output(self):
        self.load.side_effect = ValueError("P6.1-P6.2 require a passed, authenticated Phase 5 completion")
        with self.assertRaisesRegex(ValueError, "authenticated Phase 5"):
            self.complete()
        self.assertFalse(self.output.exists())
        self.build_diagnosis.assert_not_called()

    def test_source_mutation_and_new_context_tampering_block_final_gate(self):
        self.validate_context.side_effect = ValueError("Phase 5 completion evidence changed")
        report = self.complete()
        self.assertEqual(report["status"], "needs_review")
        self.assertIn("Phase 5 completion evidence changed", report["reasons"][0])
        self.output = self.root / "artifacts/changed_extension"
        self.validate_context.side_effect = None
        self.source_identity.side_effect = [self.sources, {"study_release/saved.py": "f" * 64}]
        report = self.complete()
        self.assertEqual(report["status"], "needs_review")
        self.assertIn("changed during final publication", report["reasons"][0])

    def test_reusing_destination_preserves_first_publication(self):
        first = self.complete()
        original = (self.output / "phase61_62_completion.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.complete()
        self.assertEqual((self.output / "phase61_62_completion.json").read_bytes(), original)
        self.assertEqual(evidence.read_report(self.output / "phase61_62_completion.json"), first)

    def test_no_subprocess_job_training_or_native_evaluation_is_dispatched(self):
        with patch("subprocess.run") as run, patch("subprocess.Popen") as popen:
            report = self.complete()
        run.assert_not_called()
        popen.assert_not_called()
        for field in ("research_training_executed", "research_evaluation_executed", "native_environment_opened",
                      "scientific_protocol_amended", "saved_record_reproduction_executed"):
            self.assertFalse(report[field])
        for field in ("new_environment_interactions", "new_model_updates", "new_policy_optimizer_updates"):
            self.assertEqual(report[field], 0)
        processing = evidence.read_report(self.output / "processing_manifest.json")
        self.assertFalse(processing["phase6_argv_is_historical_training_command"])


class Phase6CliTests(unittest.TestCase):
    def main(self, argv):
        with patch("sys.stdout", new_callable=io.StringIO) as stdout, patch("sys.stderr", new_callable=io.StringIO) as stderr:
            result = phase6.main(argv)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_verify_dispatch_and_failed_checks_exit_nonzero(self):
        for status, expected in ((evidence.STATUS, 0), ("failed", 1)):
            with self.subTest(status=status), patch.object(phase6.evidence, "verify", return_value={
                    "status": status, "tests_run": 22, "failures": int(expected != 0)}) as verify:
                code, stdout, stderr = self.main(["verify", "--output-dir", "synthetic-check"])
                self.assertEqual(code, expected)
                self.assertIn(status, stdout)
                self.assertEqual(stderr, "")
                verify.assert_called_once_with("synthetic-check")

    def test_complete_dispatch_and_needs_review_exit_nonzero(self):
        for status, expected in (("complete", 0), ("needs_review", 1)):
            with self.subTest(status=status), patch.object(phase6, "complete", return_value={"status": status}) as complete:
                code, stdout, stderr = self.main(["complete", "--phase5-acceptance", "prior.json",
                    "--verification-report", "proof.json", "--output-dir", "fresh-output"])
                self.assertEqual(code, expected)
                self.assertIn(status, stdout)
                self.assertEqual(stderr, "")
                complete.assert_called_once_with("prior.json", "fresh-output", "proof.json")

    def test_expected_read_and_validation_errors_are_printed_and_exit_one(self):
        with patch.object(phase6, "complete", side_effect=ValueError("missing saved evidence")):
            code, stdout, stderr = self.main(["complete", "--phase5-acceptance", "prior.json",
                "--verification-report", "proof.json", "--output-dir", "fresh-output"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("P6.1-P6.2 error (ValueError): missing saved evidence", stderr)

    def test_cli_requires_explicit_sources_and_does_not_expose_training_or_gpu_options(self):
        invalid = ([], ["complete"], ["complete", "--phase5-acceptance", "prior.json", "--output-dir", "fresh"],
            ["verify"], ["verify", "--output-dir", "fresh", "--device", "cuda:0"],
            ["run"], ["train"], ["evaluate"])
        with patch("sys.stderr", new_callable=io.StringIO):
            for argv in invalid:
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as error:
                    phase6.main(argv)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
