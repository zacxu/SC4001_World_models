"""Synthetic original-server handoff tests without scientific execution."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from study_release import common, evidence


class CompletedEvidenceFixture:
    def __init__(self, root):
        self.root = root
        self.cohort = root / "cohort"
        self.previous = self.cohort / "completion"
        self.previous.mkdir(parents=True)
        self.proof = root / "checks" / "implementation_checks.json"
        evidence.write_report(self.proof, {"fixture_only": True})
        (self.proof.parent / "tests.txt").write_text("fixture implementation log\n", encoding="utf-8")
        self.secondary = evidence.write_report(self.previous / "secondary.json", {"status": "complete"})
        self.resources = evidence.write_report(self.previous / "resources.json", {"status": "complete"})
        self.context = {"cohort": {"report_path": str(self.cohort / "cohort_manifest.json"),
            "run_matrix": [{"run_dir": str(self.cohort / "runs" / variant / f"seed_{seed}")}
                for variant in ("B", "D") for seed in (0, 1)],
            "inputs": {"bound_files": [{"path": str(root / "inputs" / "world_model.pt")}]}},
            "deliverables": {"secondary": evidence.reference(self.secondary),
                "resources": evidence.reference(self.resources)},
            "evaluation_paths": [str(self.cohort / "evaluation" / variant / f"seed_{seed}" / "evaluation.json")
                for variant in ("B", "D") for seed in (0, 1)]}
        self.context_path = evidence.write_report(self.previous / "context.json", self.context)
        self.manifest = evidence.write_report(self.previous / "completion_manifest.json",
            {"verification": evidence.reference(self.proof)})
        self.completion = {"status": "passed", "phase5_acceptance": True,
            "context": evidence.reference(self.context_path),
            "reporting_manifest": evidence.reference(self.manifest)}
        self.path = evidence.write_report(self.previous / "phase5_acceptance.json", self.completion)


class CompletedEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.fixture = CompletedEvidenceFixture(self.root)
        self.inspect = self.enterContext(patch.object(common, "inspect_phase5",
            return_value=copy.deepcopy(self.fixture.completion)))
        self.enterContext(patch.object(common.platform, "system", return_value="Linux"))
        self.enterContext(patch.object(common, "ROOT", self.root))

    def test_authenticated_completion_loads_reports_and_bound_release_inputs(self):
        result = common.load_completed(self.fixture.path)
        self.inspect.assert_called_once_with(self.fixture.path.resolve())
        self.assertEqual(result["completion"], self.fixture.completion)
        self.assertEqual(result["secondary"], {"status": "complete"})
        self.assertEqual(result["resources"], {"status": "complete"})
        self.assertEqual(len(result["release_inputs"]), 5)
        evidence.validate_references(result["release_inputs"])
        self.assertFalse(result["commands_provenance"]["historical_shell_commands_recorded"])
        self.assertFalse(result["commands_provenance"]["physical_CUDA_VISIBLE_DEVICES_recorded"])
        self.assertFalse(result["commands_provenance"]["reproduction_recipes_are_historical_command_transcripts"])
        self.assertEqual(set(result) - set(self.fixture.context),
            {"completion", "completion_path", "secondary", "resources", "release_inputs", "commands_provenance"})

    def test_windows_is_rejected_before_any_phase5_inspection(self):
        with patch.object(common.platform, "system", return_value="Windows"):
            with self.assertRaisesRegex(ValueError, "original Linux server"):
                common.load_completed(self.fixture.path)
        self.inspect.assert_not_called()

    def test_only_explicit_passed_and_true_phase5_acceptance_are_allowed(self):
        for index, delta in enumerate(({"status": "needs_review"}, {"status": "failed"},
                {"phase5_acceptance": False}, {"phase5_acceptance": 1})):
            with self.subTest(index=index):
                self.inspect.return_value = {**self.fixture.completion, **delta}
                with self.assertRaisesRegex(ValueError, "passed, authenticated"):
                    common.load_completed(self.fixture.path)

    def test_missing_or_mutated_linked_reports_fail_closed(self):
        self.fixture.resources.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "valid receipt"):
            common.load_completed(self.fixture.path)

    def test_revalidation_reinspects_phase5_and_detects_completion_change(self):
        result = common.load_completed(self.fixture.path)
        self.assertTrue(common.validate_completed(result))
        self.inspect.return_value = {**self.fixture.completion, "new_field": "changed"}
        with self.assertRaisesRegex(ValueError, "changed during Phase 6"):
            common.validate_completed(result)

    def test_revalidation_detects_input_log_changes(self):
        result = common.load_completed(self.fixture.path)
        (self.fixture.proof.parent / "tests.txt").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            common.validate_completed(result)

    def test_guard_accepts_separate_fresh_output_and_rejects_existing(self):
        context = common.load_completed(self.fixture.path)
        destination = self.root / "releases" / "fresh"
        self.assertEqual(common.guard_output(destination, context, self.fixture.proof), destination)
        self.assertFalse(destination.exists(), "guard must not execute or create a release")
        destination.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            common.guard_output(destination, context, self.fixture.proof)

    def test_guard_protects_sources_runs_inputs_completion_evaluation_and_proof(self):
        context = common.load_completed(self.fixture.path)
        protected = [self.root / name / "future" for name in
            ("src", "scripts", "tests", "configs", "scenarios", "study_reporting", "study_release")]
        protected += [Path(context["cohort"]["run_matrix"][0]["run_dir"]) / "future",
            self.root / "inputs" / "future", self.fixture.previous / "future",
            self.fixture.proof.parent / "future",
            Path(context["evaluation_paths"][0]).parent / "future"]
        for destination in protected:
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ValueError, "separate from sources"):
                    common.guard_output(destination, context, self.fixture.proof)

    def test_guard_rejects_fresh_ancestor_that_would_contain_bound_inputs(self):
        context = common.load_completed(self.fixture.path)
        destination = self.root / "future_inputs"
        context["cohort"]["inputs"]["bound_files"] = [{"path": str(destination / "data" / "input.pt")}]
        with self.assertRaisesRegex(ValueError, "separate from sources"):
            common.guard_output(destination, context, self.fixture.proof)


if __name__ == "__main__":
    unittest.main()
