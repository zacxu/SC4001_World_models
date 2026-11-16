from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from study_paper import audit
from study_paper.tables import replace_block


class ClaimAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "data.json"
        self.path.write_text(json.dumps({"episodes": [{"successes": 0}], "false": False, "a/b": {"~": 2}}))
        self.digest = audit.file_sha256(self.path)
        self.manifest = {"cohort_identity": "cohort", "files": [{"path": "data.json", "sha256": self.digest}]}
        self.claims = {"schema_version": "p63-claim-evidence-v1", "cohort_identity": "cohort", "claims": [
            {"id": "zero", "manuscript_heading": "## Results", "claim": "Zero observed successes.",
             "scope": "one synthetic test", "limitation": "not a scientific result", "evidence": [
                 {"package_path": "data.json", "sha256": self.digest, "pointer": "/episodes/0/successes", "expected": 0}]}]}

    def check(self):
        return audit.audit_claims(self.claims, "## Results\nSynthetic", self.root, self.manifest)

    def test_valid_claim_has_exact_assertion(self):
        self.assertEqual(self.check(), {"claims": 1, "exact_json_assertions": 1})

    def test_claim_tampered_value_and_type_rejected(self):
        for value in (1, False, 0.0):
            with self.subTest(value=value):
                self.claims["claims"][0]["evidence"][0]["expected"] = value
                with self.assertRaisesRegex(ValueError, "assertion differs"):
                    self.check()

    def test_tampered_bytes_rejected_even_if_valid_json(self):
        self.path.write_text('{"episodes":[{"successes":1}]}')
        with self.assertRaisesRegex(ValueError, "not sealed"):
            self.check()

    def test_missing_pointer_rejected(self):
        self.claims["claims"][0]["evidence"][0]["pointer"] = "/missing"
        with self.assertRaises(KeyError):
            self.check()

    def test_duplicate_or_empty_claims_rejected(self):
        self.claims["claims"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check()
        self.claims["claims"] = []
        with self.assertRaisesRegex(ValueError, "empty claims"):
            self.check()

    def test_empty_evidence_and_unlocated_claim_rejected(self):
        self.claims["claims"][0]["evidence"] = []
        with self.assertRaisesRegex(ValueError, "no evidence"):
            self.check()
        self.claims["claims"][0]["manuscript_heading"] = "## Absent"
        with self.assertRaisesRegex(ValueError, "unlocated"):
            self.check()

    def test_wrong_cohort_and_unindexed_file_rejected(self):
        self.claims["cohort_identity"] = "other"
        with self.assertRaisesRegex(ValueError, "cohort"):
            self.check()
        self.claims["cohort_identity"] = "cohort"
        self.manifest["files"] = []
        with self.assertRaisesRegex(ValueError, "not sealed"):
            self.check()

    def test_path_escape_forms_rejected(self):
        for value in ("../data.json", "/data.json", "C:/data.json", "a\\data.json", "a//data.json", "./data.json"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.relative_file(self.root, value)

    def test_link_evidence_rejected(self):
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "links"):
                audit.relative_file(self.root, "data.json")

    def test_json_pointer_escapes_and_array_indices(self):
        value = json.loads(self.path.read_text())
        self.assertEqual(audit.json_pointer(value, "/a~1b/~0"), 2)
        for pointer in ("not/a/pointer", "/a~2b", "/episodes/-1", "/episodes/00"):
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                audit.json_pointer(value, pointer)

    def test_concurrent_mutation_rejected(self):
        pins = {self.path: self.digest}
        audit.verify_pins(pins)
        self.path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed during"):
            audit.verify_pins(pins)


class ReviewAndStageTests(unittest.TestCase):
    def reports(self):
        c = {"schema_version": "p6162-completion-v1", "status": "complete", "cohort_identity": "x",
            "P6.1_complete": True, "P6.2_complete": True, "release_byte_integrity_checked": True,
            "saved_record_reproduction_executed": False, "research_training_executed": False, "research_evaluation_executed": False}
        r = {"schema_version": "p62-saved-reproduction-v1", "status": "complete", "cohort_identity": "x",
            "numerical_reproduction": True, "six_tables_byte_identical": True, "two_figures_rendered": True,
            "research_training_executed": False, "research_evaluation_executed": False}
        return c, r, {"cohort_identity": "x"}

    def test_later_reproduction_accepts_historical_false_flag(self):
        audit.validate_stage_reports(*self.reports())

    def test_stage_mismatch_missing_and_nonboolean_rejected(self):
        for report_index, key, value in ((1, "cohort_identity", "other"), (0, "P6.1_complete", 1),
            (1, "numerical_reproduction", False), (1, "research_training_executed", True)):
            c, r, m = self.reports()
            (c, r)[report_index][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.validate_stage_reports(c, r, m)

    def test_marker_missing_duplicate_reverse_and_changed_table(self):
        valid = "<!-- P63:X:BEGIN -->\n\nold\n\n<!-- P63:X:END -->"
        changed = replace_block(valid, "X", "new")
        self.assertNotEqual(valid, changed)
        self.assertEqual(replace_block(changed, "X", "new"), changed)
        for bad in ("", valid + valid, "<!-- P63:X:END --><!-- P63:X:BEGIN -->"):
            with self.assertRaises(ValueError):
                replace_block(bad, "X", "new")

    def test_review_binds_prose_and_claim_bytes(self):
        review = {"schema_version": "p63-substantive-review-v1", "status": "complete", "manuscript_sha256": "m",
            "claims_sha256": "c", "independent_human_peer_review": False,
            "checks": [{"id": name, "status": "reviewed", "finding": "synthetic finding"} for name in
                ("method_and_protocol", "numerical_claims", "causal_limits", "literature", "scope_and_deviations", "reproducibility")]}
        audit.validate_review(review, "m", "c")
        for mh, ch in (("edited", "c"), ("m", "edited")):
            with self.assertRaisesRegex(ValueError, "stale"):
                audit.validate_review(review, mh, ch)
        review["checks"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            audit.validate_review(review, "m", "c")

    def test_imports_need_no_torch_simulator_or_matplotlib(self):
        code = "import study_paper.phase63; import sys; assert not {'torch','vizdoom','matplotlib'} & sys.modules.keys()"
        r = subprocess.run([sys.executable, "-B", "-c", code], cwd=audit.ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_source_set_added_removed_or_changed_rejected(self):
        m = {"frozen_source_sha256": {"a.py": "a"}, "reporting_source_sha256": {"b.py": "b"}}
        c = {"source_sha256": {"c.py": "c"}}
        actual = deepcopy(m)
        with patch("study_release.evidence.prior_identities", return_value=actual), patch("study_release.evidence.source_identity", return_value=c["source_sha256"]):
            audit.verify_sources(m, c)
            for bad in ({}, {"a.py": "changed"}, {"a.py": "a", "new.py": "n"}):
                actual["frozen_source_sha256"] = bad
                with self.assertRaisesRegex(ValueError, "source set changed"):
                    audit.verify_sources(m, c)


if __name__ == "__main__":
    unittest.main()
