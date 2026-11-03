"""Closed journal migration retains evidence until durable exact archival succeeds."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from world_models.data.spool import (
    ArchiveCleanupError, archive_closed_file, archive_closed_tree, verify_archive_tree,
)
from world_models.data import spool as spool_module


class ReplaySpoolTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory)
        spool = base/"spool"
        source = spool/"runtime/episode-0"
        source.mkdir(parents=True)
        (source/"worker0").mkdir()
        (source/"empty").mkdir()
        (source/"dispatch_ledger.jsonl").write_bytes(b'{"dispatch":1}\n{"dispatch":2}\n')
        (source/"worker0/state.json").write_bytes(b'{"tic":8}\n')
        return spool, source, base/"archive/episode-0"

    def test_exact_archive_receipt_precedes_source_deletion_and_is_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            receipt = archive_closed_tree(source, destination, spool_root=spool)
            self.assertTrue(receipt["source_removed"])
            self.assertFalse(source.exists())
            self.assertTrue(spool.exists())
            self.assertEqual(receipt["file_count"], 2)
            self.assertEqual({row["path"] for row in receipt["files"]}, {"dispatch_ledger.jsonl", "worker0/state.json"})
            self.assertEqual((destination/"worker0/state.json").read_bytes(), b'{"tic":8}\n')
            self.assertIn("empty", receipt["directories"])
            moved = Path(directory)/"relocated"
            shutil.copytree(destination, moved)
            verified = verify_archive_tree(moved, expected_receipt_sha256=receipt["receipt_sha256"])
            self.assertEqual(verified["content_sha256"], receipt["content_sha256"])

    def test_copy_failure_preserves_all_source_bytes_and_partial_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            before = {path.relative_to(source).as_posix():path.read_bytes() for path in source.rglob("*") if path.is_file()}
            with patch("world_models.data.spool._copy_closed_file", side_effect=OSError("injected full archive disk")):
                with self.assertRaises(OSError): archive_closed_tree(source, destination, spool_root=spool)
            self.assertTrue(destination.exists())
            self.assertFalse((destination/"archive_receipt.json").exists())
            self.assertEqual(before, {path.relative_to(source).as_posix():path.read_bytes() for path in source.rglob("*") if path.is_file()})
            with self.assertRaises(FileExistsError): archive_closed_tree(source, destination, spool_root=spool)

    def test_receipt_fsync_failure_retains_entire_source(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            with patch("world_models.data.spool._write_receipt", side_effect=OSError("injected receipt failure")):
                with self.assertRaises(OSError): archive_closed_tree(source, destination, spool_root=spool)
            self.assertTrue((source/"dispatch_ledger.jsonl").exists())
            self.assertTrue((source/"worker0/state.json").exists())

    def test_cleanup_failure_reports_verified_archive_and_retains_hot_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            with patch("world_models.data.spool._remove_verified_tree", side_effect=PermissionError("still open")):
                with self.assertRaises(ArchiveCleanupError) as error:
                    archive_closed_tree(source, destination, spool_root=spool)
            self.assertFalse(error.exception.receipt["source_removed"])
            self.assertTrue(source.exists())
            self.assertEqual(verify_archive_tree(destination)["file_count"], 2)

    def test_source_change_during_copy_prevents_any_cleanup(self):
        from world_models.data import spool as module
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            original = module._copy_closed_file
            def changed(reader, writer, expected):
                original(reader, writer, expected)
                if reader.name == "state.json":
                    reader.write_bytes(b'{"tic":999}\n')
            with patch.object(module, "_copy_closed_file", side_effect=changed):
                with self.assertRaisesRegex(RuntimeError, "inventory changed"):
                    archive_closed_tree(source, destination, spool_root=spool)
            self.assertTrue((source/"dispatch_ledger.jsonl").exists())
            self.assertFalse((destination/"archive_receipt.json").exists())

    def test_refuses_root_escape_overlap_and_existing_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            outside = Path(directory)/"outside"
            outside.mkdir()
            for invalid_source, invalid_destination in ((spool, destination), (outside, destination), (source, source/"archive"), (source, spool/"other")):
                with self.subTest(source=str(invalid_source), destination=str(invalid_destination)), self.assertRaises(ValueError):
                    archive_closed_tree(invalid_source, invalid_destination, spool_root=spool)
            destination.mkdir(parents=True)
            with self.assertRaises(FileExistsError): archive_closed_tree(source, destination, spool_root=spool)
            self.assertTrue(source.exists())

    def test_rejects_symlink_without_deleting_target_when_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            external = Path(directory)/"external.txt"
            external.write_bytes(b"must survive")
            try:
                (source/"link").symlink_to(external)
            except OSError:
                self.skipTest("Windows symlink privilege/developer mode unavailable")
            with self.assertRaises(ValueError): archive_closed_tree(source, destination, spool_root=spool)
            self.assertEqual(external.read_bytes(), b"must survive")
            self.assertTrue(source.exists())

    def test_windows_reparse_attribute_is_rejected_without_symlink_privileges(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            real_lstat = Path.lstat
            def metadata(path, *args, **kwargs):
                if path == source:
                    return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
                return real_lstat(path, *args, **kwargs)
            with patch.object(Path, "lstat", autospec=True, side_effect=metadata):
                with self.assertRaisesRegex(ValueError, "reparse points"):
                    archive_closed_tree(source, destination, spool_root=spool)
            self.assertTrue((source/"dispatch_ledger.jsonl").exists())
            self.assertFalse(destination.exists())

    def test_archive_verification_detects_changes_missing_and_extra_files(self):
        for damage in ("change", "missing", "extra", "receipt"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                spool, source, destination = self.fixture(directory)
                receipt = archive_closed_tree(source, destination, spool_root=spool)
                if damage == "change": (destination/"worker0/state.json").write_bytes(b"bad")
                elif damage == "missing": (destination/"worker0/state.json").unlink()
                elif damage == "extra": (destination/"untracked.txt").write_bytes(b"bad")
                else:
                    path = destination/"archive_receipt.json"
                    data = json.loads(path.read_text())
                    data["total_bytes"] += 1
                    path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    verify_archive_tree(destination, expected_receipt_sha256=receipt["receipt_sha256"])

    def test_platform_enumeration_order_does_not_change_original_receipt_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            (source/"ZUPPER.txt").write_bytes(b"uppercase")
            (source/"lower.txt").write_bytes(b"lowercase")
            receipt = archive_closed_tree(source, destination, spool_root=spool)
            original_bytes = (destination/"archive_receipt.json").read_bytes()
            real_inventory = spool_module._inventory
            def opposite_order(root):
                inventory = real_inventory(root)
                return {"files": list(reversed(inventory["files"])), "directories": list(reversed(inventory["directories"]))}
            with patch.object(spool_module, "_inventory", side_effect=opposite_order):
                verified = verify_archive_tree(destination, expected_receipt_sha256=receipt["receipt_sha256"])
            self.assertEqual(verified["content_sha256"], receipt["content_sha256"])
            self.assertEqual((destination/"archive_receipt.json").read_bytes(), original_bytes)

    def test_shuffled_receipt_order_keeps_its_original_ordered_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            (source/"ZUPPER.txt").write_bytes(b"uppercase")
            archive_closed_tree(source, destination, spool_root=spool)
            path = destination/"archive_receipt.json"
            receipt = json.loads(path.read_text())
            receipt["files"].reverse()
            receipt["directories"].reverse()
            original_ordered_members = {key: receipt[key] for key in ("directories", "files")}
            receipt["content_sha256"] = spool_module._canonical(original_ordered_members)
            path.write_text(json.dumps(receipt), encoding="utf-8")
            before = path.read_bytes()
            receipt_sha = spool_module._sha(path)
            verified = verify_archive_tree(destination, expected_receipt_sha256=receipt_sha)
            self.assertEqual(verified["content_sha256"], spool_module._canonical(original_ordered_members))
            self.assertEqual(path.read_bytes(), before)
            # Sorting is a comparison-only operation; a different ordered hash
            # must not be accepted as the identity of these existing bytes.
            receipt["content_sha256"] = "f"*64
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "member bytes or inventory changed"):
                verify_archive_tree(destination)

    def test_order_independent_archive_comparison_still_rejects_payload_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            archive_closed_tree(source, destination, spool_root=spool)
            (destination/"worker0/state.json").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "member bytes or inventory changed"):
                verify_archive_tree(destination)

    def test_closed_ledger_copy_preserves_source_by_default_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            ledger = source/"dispatch_ledger.jsonl"
            target = Path(directory)/"training_interactions.jsonl"
            receipt = archive_closed_file(ledger, target, spool_root=spool)
            self.assertEqual(ledger.read_bytes(), target.read_bytes())
            self.assertFalse(receipt["source_removed"])
            self.assertTrue(ledger.exists())
            with self.assertRaises(FileExistsError): archive_closed_file(ledger, target, spool_root=spool)

    def test_closed_ledger_can_be_removed_only_after_verified_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            ledger = source/"dispatch_ledger.jsonl"
            expected = ledger.read_bytes()
            target = Path(directory)/"training_interactions.jsonl"
            receipt = archive_closed_file(ledger, target, spool_root=spool, remove_source=True)
            self.assertFalse(ledger.exists())
            self.assertTrue(receipt["source_removed"])
            self.assertEqual(target.read_bytes(), expected)

    def test_closed_ledger_fsync_failure_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            spool, source, destination = self.fixture(directory)
            ledger = source/"dispatch_ledger.jsonl"
            with patch("world_models.data.spool.os.fsync", side_effect=OSError("injected fsync failure")):
                with self.assertRaises(OSError):
                    archive_closed_file(ledger, Path(directory)/"ledger.jsonl", spool_root=spool, remove_source=True)
            self.assertTrue(ledger.exists())


if __name__ == "__main__":
    unittest.main()
