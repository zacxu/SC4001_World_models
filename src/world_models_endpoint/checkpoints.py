"""Structural checkpoint authentication without importing networks or Torch.

Training audits deserialize on CPU through the original controllers separately.
The actor evaluator authenticates those audited bytes using only this reader.
"""
from __future__ import annotations

from pathlib import Path
import re

from world_models.data.ledger import file_sha256
from .evidence import read_report, safe_path


_NUMBERED = re.compile(r"state_([0-9]{8})_[0-9a-f]{8}\Z")


def latest(directory):
    directory = safe_path(directory)
    rows = []
    for p in (directory / "checkpoints").glob("state_*"):
        if not (p / "manifest.sha256").is_file():
            continue
        match = _NUMBERED.fullmatch(p.name)
        if match is None:
            raise ValueError("malformed committed checkpoint name")
        rows.append((int(match[1]), p))
    if not rows or len({n for n, _ in rows}) != len(rows):
        raise ValueError("missing or ambiguous committed checkpoint")
    return max(rows)[1]


def authenticate(directory, *, actor_only=False):
    directory = safe_path(directory)
    manifest = read_report(directory / "manifest.json")
    if (manifest.get("schema_version") != "real-model-policy-cycle-checkpoint-v1"
            or manifest.get("data_source") != "training_controller_state"
            or manifest.get("learning_eligible_for_world_model") is not False
            or manifest.get("status") != "complete" or type(manifest.get("metadata")) is not dict):
        raise ValueError("invalid committed checkpoint schema")
    payload = manifest.get("payload", {})
    path = safe_path(directory / "state.pt")
    if (payload.get("path") != "state.pt" or not path.is_file()
            or type(payload.get("bytes")) is not int or payload["bytes"] != path.stat().st_size
            or payload.get("sha256") != file_sha256(path)):
        raise ValueError("committed checkpoint payload bytes differ")
    if actor_only:
        specs = manifest.get("tensor_specs")
        if (not specs or any(not row.get("path") or row["path"][0] != ["key", "actor"] for row in specs)
                or manifest["metadata"].get("schema_version") != "p35-local-actor-v1"):
            raise ValueError("deployment requires an actor-only checkpoint")
    return manifest


def files(directory):
    directory = safe_path(directory)
    authenticate(directory)
    return [{"path": str(directory / name), "sha256": file_sha256(directory / name)}
            for name in ("manifest.json", "manifest.sha256", "state.pt")]


def validate_files(rows):
    if type(rows) is not list or not rows:
        raise ValueError("audit has no immutable file bindings")
    seen = set()
    for row in rows:
        path = safe_path(row["path"])
        if str(path) != row["path"] or str(path) in seen or not path.is_file() or file_sha256(path) != row["sha256"]:
            raise ValueError(f"audited bytes changed: {path}")
        seen.add(str(path))
