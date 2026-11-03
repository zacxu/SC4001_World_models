"""Durably archive closed hot journals before freeing their bounded scratch space.

The caller must first close both simulator participants. A failed copy retains
the source and the partial destination. Nothing is automatically garbage-collected.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat


ARCHIVE_VERSION = "closed-journal-archive-v1"
RECEIPT_NAME = "archive_receipt.json"


class ArchiveCleanupError(RuntimeError):
    """A verified archive exists, but scratch cleanup did not fully finish."""

    def __init__(self, message: str, receipt: dict):
        super().__init__(message)
        self.receipt = receipt


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _reject_links(path: Path) -> None:
    """Check every existing component, including Windows junction/reparse points."""
    path = path.absolute()
    for component in (*reversed(path.parents), path):
        if not component.exists() and not component.is_symlink():
            continue
        attributes = component.lstat()
        if (stat.S_ISLNK(attributes.st_mode)
                or getattr(attributes, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError(f"journal archive rejects symbolic links and reparse points: {component}")


def _checked_source(source: str | Path, spool_root: str | Path, *, directory: bool) -> tuple[Path, Path]:
    source, root = Path(source).absolute(), Path(spool_root).absolute()
    _reject_links(root)
    _reject_links(source)
    root, source = root.resolve(strict=True), source.resolve(strict=True)
    if not root.is_dir() or source == root or not source.is_relative_to(root):
        raise ValueError("archive source must be a strict descendant of the explicitly named spool root")
    if directory and not source.is_dir() or not directory and not source.is_file():
        raise ValueError("archive source has the wrong file/directory type")
    return source, root


def _checked_destination(destination: str | Path, spool_root: Path) -> Path:
    destination = Path(destination).absolute()
    _reject_links(destination)
    destination = destination.resolve(strict=False)
    if destination == spool_root or destination.is_relative_to(spool_root):
        raise ValueError("archive destination must be outside the hot spool root")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite an existing or partial journal archive: {destination}")
    return destination


def _inventory(root: Path) -> dict:
    directories, files = [], []
    for path in sorted(root.rglob("*")):
        _reject_links(path)
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("archive member escaped its closed source tree")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.append(relative)
        elif path.is_file():
            before = path.stat()
            digest = _sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError("a supposedly closed journal changed while hashing")
            files.append({"path": relative, "bytes": after.st_size, "sha256": digest})
        else:
            raise ValueError("journal trees may contain only ordinary files and directories")
    return {"directories": directories, "files": files}


def _copy_closed_file(source: Path, destination: Path, expected: dict) -> None:
    """Exclusive-create preserves conflicts; fsync is included in the measured cost."""
    _reject_links(source)
    _reject_links(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(block)
            digest.update(block)
            count += len(block)
        writer.flush()
        os.fsync(writer.fileno())
    if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise RuntimeError("closed journal changed during the archival copy")
    if destination.stat().st_size != expected["bytes"] or _sha(destination) != expected["sha256"]:
        raise RuntimeError("durable archived journal bytes do not match the source")


def _write_receipt(destination: Path, receipt: dict) -> str:
    path = destination / RECEIPT_NAME
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return _sha(path)


def _remove_verified_tree(source: Path, spool_root: Path, expected: dict) -> None:
    """Delete only the verified strict descendant, with checks before every removal."""
    checked, root = _checked_source(source, spool_root, directory=True)
    if checked != source or root != spool_root or _inventory(source) != expected:
        raise RuntimeError("closed source changed before scratch cleanup")
    # Avoid any shell or recursive-delete expansion. Every target was inventoried
    # and is checked again against the intended absolute source/spool boundaries.
    for record in reversed(expected["files"]):
        path = source / record["path"]
        _reject_links(path)
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(source) or not resolved.is_relative_to(spool_root):
            raise ValueError("scratch deletion target escaped its verified source/spool root")
        path.unlink()
    for relative in sorted(expected["directories"], key=lambda value: len(Path(value).parts), reverse=True):
        path = source / relative
        _reject_links(path)
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(source) or not resolved.is_relative_to(spool_root):
            raise ValueError("scratch directory deletion target escaped its verified source/spool root")
        path.rmdir()
    checked, root = _checked_source(source, spool_root, directory=True)
    if checked != source or checked == root:
        raise ValueError("refusing to remove a changed source or spool root")
    source.rmdir()


def archive_closed_tree(source_dir: str | Path, destination_dir: str | Path, *, spool_root: str | Path) -> dict:
    """Copy/fsync/hash/receipt a closed tree, and only then remove its hot copy.

    On archive failure no source member is removed. A partial destination is
    deliberately retained and may never be silently overwritten on retry.
    """
    source, root = _checked_source(source_dir, spool_root, directory=True)
    destination = _checked_destination(destination_dir, root)
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("source and archive trees must not overlap")
    expected = _inventory(source)
    if any(record["path"] == RECEIPT_NAME for record in expected["files"]):
        raise ValueError("source tree contains the reserved archive receipt filename")
    destination.mkdir(parents=True, exist_ok=False)
    for relative in expected["directories"]:
        (destination / relative).mkdir(parents=True, exist_ok=False)
    for record in expected["files"]:
        _copy_closed_file(source / record["path"], destination / record["path"], record)
    if _inventory(source) != expected or _inventory(destination) != expected:
        raise RuntimeError("source or archived member inventory changed before verification")
    receipt = {
        "schema_version": ARCHIVE_VERSION, "status": "verified_closed_archive",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path_at_collection": str(source), "spool_root_at_collection": str(root),
        "member_paths": "relative to the archive directory; original source paths are historical provenance only",
        **expected, "file_count": len(expected["files"]),
        "total_bytes": sum(record["bytes"] for record in expected["files"]),
        "content_sha256": _canonical(expected),
        "durability": "every archived file and the receipt are flushed and fsynced before any source deletion",
    }
    receipt_sha256 = _write_receipt(destination, receipt)
    verified = verify_archive_tree(destination, expected_receipt_sha256=receipt_sha256)
    try:
        _remove_verified_tree(source, root, expected)
    except BaseException as exc:
        raise ArchiveCleanupError("archive verified; hot journal cleanup failed and requires inspection", {
            **verified, "receipt_sha256": receipt_sha256, "source_removed": False}) from exc
    return {**verified, "receipt_sha256": receipt_sha256, "source_removed": True}


def verify_archive_tree(destination_dir: str | Path, *, expected_receipt_sha256: str | None = None) -> dict:
    """Validate a relocated archive using relative members, without its old C: tree."""
    destination = Path(destination_dir).absolute()
    _reject_links(destination)
    destination = destination.resolve(strict=True)
    path = destination / RECEIPT_NAME
    if expected_receipt_sha256 is not None and _sha(path) != expected_receipt_sha256:
        raise ValueError("closed journal archive receipt changed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != ARCHIVE_VERSION or receipt.get("status") != "verified_closed_archive":
        raise ValueError("unsupported or incomplete closed journal archive")
    inventory = _inventory(destination)
    inventory["files"] = [record for record in inventory["files"] if record["path"] != RECEIPT_NAME]
    expected = {key: receipt[key] for key in ("directories", "files")}
    for relative in [*expected["directories"], *(record["path"] for record in expected["files"])]:
        member = Path(relative)
        if member.is_absolute() or member.drive or ".." in member.parts:
            raise ValueError("journal archive receipt contains an escaping member path")
    # WindowsPath and PosixPath enumerate mixed-case members in different
    # orders. Compare exact names/bytes/hashes independently of that ordering,
    # while checking the ORIGINAL ordered receipt for its historical identity.
    canonical_members = lambda value: {
        "directories": sorted(value["directories"]),
        "files": sorted(value["files"], key=lambda record: record["path"]),
    }
    if canonical_members(inventory) != canonical_members(expected) or receipt["content_sha256"] != _canonical(expected):
        raise ValueError("closed journal archive member bytes or inventory changed")
    if receipt["file_count"] != len(expected["files"]) or receipt["total_bytes"] != sum(record["bytes"] for record in expected["files"]):
        raise ValueError("closed journal archive totals disagree")
    return {**receipt, "receipt_sha256": _sha(path)}


def archive_closed_file(source_file: str | Path, destination_file: str | Path, *,
                        spool_root: str | Path, remove_source: bool = False) -> dict:
    """Durably copy a closed final/failure ledger, retaining its hot copy by default."""
    if type(remove_source) is not bool:
        raise ValueError("remove_source must be a boolean")
    source, root = _checked_source(source_file, spool_root, directory=False)
    destination = _checked_destination(destination_file, root)
    before = source.stat()
    expected = {"path": destination.name, "bytes": before.st_size, "sha256": _sha(source)}
    _copy_closed_file(source, destination, expected)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or _sha(source) != expected["sha256"]:
        raise RuntimeError("closed ledger changed during its archival copy")
    if remove_source:
        checked, checked_root = _checked_source(source, root, directory=False)
        if checked != source or checked_root != root:
            raise ValueError("ledger deletion target changed")
        source.unlink()
    return {"schema_version": ARCHIVE_VERSION, "status": "verified_closed_file", **expected,
            "source_path_at_collection": str(source), "source_removed": remove_source,
            "durability": "exclusive destination creation, flush/fsync, and source/destination SHA256 equality"}
