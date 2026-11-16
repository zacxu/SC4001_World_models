"""Acceptance reports fail closed when receipts or referenced sources change."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.phase36_support.evidence import INVARIANT_TESTS, load_report, source_identity, write_report


class Phase36EvidenceTests(unittest.TestCase):
    def test_round_trip_and_immutable_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            report = {"value": [False, None, 1, 0.5], "phase3_acceptance": False}
            self.assertEqual(write_report(path, report), path.resolve())
            self.assertEqual(load_report(path), report)
            with self.assertRaises(FileExistsError):
                write_report(path, {"new": True})
            self.assertEqual(load_report(path), report)

    def test_changed_bytes_and_missing_receipt_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_report(path, {"passed": False})
            path.write_text('{"passed": true}\n')
            with self.assertRaisesRegex(ValueError, "receipt"):
                load_report(path)
            path.with_suffix(".sha256").unlink()
            with self.assertRaises(FileNotFoundError):
                load_report(path)

    def test_interrupted_receipt_publication_never_looks_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with patch("scripts.phase36_support.evidence.os.replace", side_effect=OSError("disk failed")):
                with self.assertRaises(OSError):
                    write_report(path, {"passed": True})
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".sha256").exists())
            with self.assertRaises(FileNotFoundError):
                load_report(path)
            with self.assertRaises(FileExistsError):
                write_report(path, {"passed": True})

    def test_duplicate_keys_nonfinite_and_nonobjects_are_not_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            for content in (b'{"x":1,"x":2}', b'{"x":NaN}', b'[]'):
                path.write_bytes(content)
                path.with_suffix(".sha256").write_text(f"{hashlib.sha256(content).hexdigest()}  {path.name}\n")
                with self.assertRaises(ValueError):
                    load_report(path)
            with self.assertRaises(ValueError):
                write_report(Path(directory) / "nan.json", {"x": float("nan")})
            with self.assertRaises(ValueError):
                write_report(Path(directory) / "array.json", [])

    def test_receipt_cannot_name_another_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_report(path, {"x": 1})
            receipt = path.with_suffix(".sha256")
            receipt.write_text(receipt.read_text().replace("result.json", "other.json"))
            with self.assertRaises(ValueError):
                load_report(path)

    def test_source_identity_binds_acceptance_training_configs_and_native_scenarios(self):
        identity = source_identity()
        for key in ("scripts/phase36.py", "scripts/phase36_support/evidence.py", "scripts/phase24_freeze.py",
                    "src/world_models/training/selection.py",
                    "src/world_models/cycle/controller.py", "src/world_models/mappo/updater.py",
                    "configs/environment_splits.yaml", "scenarios/relay/task.acs",
                    "requirements-phase1.lock"):
            self.assertIn(key, identity)
            self.assertEqual(len(identity[key]), 64)
        self.assertFalse(any(name.startswith("tests/") for name in identity))
        self.assertEqual(len(INVARIANT_TESTS), 6)

    def test_all_declared_invariants_resolve_to_real_tests(self):
        ids = [name for values in INVARIANT_TESTS.values() for name in values]
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(ids)
        self.assertEqual(loader.errors, [])
        self.assertEqual(suite.countTestCases(), len(ids))


if __name__ == "__main__":
    unittest.main()
