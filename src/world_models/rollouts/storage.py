"""Immutable, receipt-bound storage for imagined policy trajectories only."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from world_models.rollouts.schema import ImaginedRollout


SCHEMA_VERSION = "imagined-on-policy-v1"
DATA_SOURCE = "world_model_imagination"
_MARKERS = {"schema_version": SCHEMA_VERSION, "data_source": DATA_SOURCE,
            "learning_eligible_for_world_model": False}
_PAYLOAD_NAME = "rollout.pt"
_PAYLOAD_KEYS = {*_MARKERS, "metadata", "streams_before", "streams_after", "tensors"}
_MANIFEST_KEYS = {*_MARKERS, "status", "metadata", "payload", "tensor_specs"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_copy(value: Any) -> Any:
    """Normalize metadata to portable JSON without silently stringifying keys."""
    def check(item: Any) -> None:
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("rollout metadata keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is None or type(item) in (str, bool, int):
            pass
        elif type(item) is float and math.isfinite(item):
            pass
        else:
            raise ValueError("rollout metadata must contain finite JSON values")
    check(value)
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _tensor_tree(value: Any, *, copy_cpu: bool) -> Any:
    """Permit weights-only compatible tensors and finite primitive containers."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.requires_grad or value.grad_fn is not None:
            raise ValueError("rollout storage requires detached dense tensors")
        if copy_cpu:
            with torch.inference_mode(False), torch.no_grad():
                return value.detach().to("cpu").clone()
        return value
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("rollout tensor-tree keys must be strings")
        return {key: _tensor_tree(child, copy_cpu=copy_cpu) for key, child in value.items()}
    if isinstance(value, list):
        return [_tensor_tree(child, copy_cpu=copy_cpu) for child in value]
    if isinstance(value, tuple):
        return tuple(_tensor_tree(child, copy_cpu=copy_cpu) for child in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("unsupported rollout tensor-tree value")


def _specifications(value: Any, prefix: str = "") -> dict:
    if isinstance(value, torch.Tensor):
        return {prefix: {"shape": list(value.shape), "dtype": str(value.dtype),
                         "elements": value.numel(), "bytes": value.numel() * value.element_size()}}
    result = {}
    children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else ()
    for key, child in children:
        result.update(_specifications(child, f"{prefix}/{key}" if prefix else str(key)))
    return result


def _root(path: str | Path) -> Path:
    candidate = Path(path)
    if ".." in candidate.parts or candidate.is_symlink():
        raise ValueError("rollout directory may not traverse parents or be a symbolic link")
    return candidate.resolve()


def _guard_real_replay(root: Path) -> None:
    for ancestor in (root, *root.parents):
        for name in ("manifest.json", "manifest.partial.json"):
            candidate = ancestor / name
            if not candidate.is_file():
                continue
            try:
                document = json.loads(candidate.read_text(encoding="utf-8"))
            except (ValueError, OSError) as error:
                raise ValueError("cannot verify separation from an existing replay manifest") from error
            if (isinstance(document, dict) and
                    (document.get("dataset_version") == "real-replay-v1" or
                     isinstance(document.get("metadata"), dict) and
                     document["metadata"].get("data_source") == "real_environment")):
                raise ValueError("imagined trajectories cannot be stored inside real replay")


def _member(root: Path, name: str) -> Path:
    if not isinstance(name, str) or name not in ("manifest.json", "manifest.sha256", _PAYLOAD_NAME):
        raise ValueError("unsupported or escaping rollout storage member")
    candidate = root / name
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
        raise ValueError("rollout storage member escapes through a symbolic link")
    if not candidate.is_file():
        raise ValueError(f"incomplete imagined rollout: missing {name}")
    return candidate


def _validate_markers(document: dict) -> None:
    if any(type(document.get(key)) is not type(value) or document[key] != value
           for key, value in _MARKERS.items()):
        raise ValueError("rollout schema or synthetic-data markers differ")


def _decode(payload: dict) -> ImaginedRollout:
    if type(payload) is not dict or set(payload) != _PAYLOAD_KEYS:
        raise ValueError("imagined rollout payload has unsupported keys")
    _validate_markers(payload)
    metadata = _json_copy(payload["metadata"])
    if type(metadata) is not dict:
        raise ValueError("rollout metadata must be a JSON object")
    _tensor_tree(payload, copy_cpu=False)
    return ImaginedRollout.from_tensor_payload(payload["tensors"], metadata=metadata,
        streams_before=payload["streams_before"], streams_after=payload["streams_after"])


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@torch.inference_mode(False)
def save_rollout(rollout: ImaginedRollout, output_dir: str | Path) -> Path:
    """Publish a fresh synthetic store; its receipt is the final commit marker.

    No real replay writer is used, and an existing directory is never reused.
    Failed publication intentionally leaves an uncommitted directory for diagnosis.
    """
    if not isinstance(rollout, ImaginedRollout):
        raise TypeError("save_rollout requires an ImaginedRollout")
    root = _root(output_dir)
    _guard_real_replay(root)
    if root.exists():
        raise FileExistsError("imagined rollout output must be a fresh directory")
    payload = {**_MARKERS, "metadata": _json_copy(rollout.metadata),
               "streams_before": _tensor_tree(rollout.streams_before, copy_cpu=True),
               "streams_after": _tensor_tree(rollout.streams_after, copy_cpu=True),
               "tensors": _tensor_tree(rollout.tensor_payload(), copy_cpu=True)}
    _decode(payload)
    root.mkdir(parents=True, exist_ok=False)
    payload_path = root / _PAYLOAD_NAME
    with payload_path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Validate the exact persisted bytes before publishing either manifest marker.
    with payload_path.open("rb") as stream:
        saved = torch.load(stream, map_location="cpu", weights_only=True)
    _decode(saved)
    manifest = {**_MARKERS, "status": "complete", "metadata": deepcopy(payload["metadata"]),
                "payload": {"path": _PAYLOAD_NAME, "bytes": payload_path.stat().st_size,
                            "sha256": _sha256(payload_path)},
                "tensor_specs": _specifications(saved)}
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    with (root / "manifest.sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{_sha256(manifest_path)}  manifest.json\n")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest_path


@torch.inference_mode(False)
def load_rollout(output_dir: str | Path, *, device: str | torch.device = "cpu") -> ImaginedRollout:
    """Load only a committed, hash-verified synthetic rollout without changing RNG."""
    root = _root(output_dir)
    _guard_real_replay(root)
    manifest_path, receipt = _member(root, "manifest.json"), _member(root, "manifest.sha256")
    if receipt.read_text(encoding="ascii").strip() != f"{_sha256(manifest_path)}  manifest.json":
        raise ValueError("imagined rollout manifest receipt mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if type(manifest) is not dict or set(manifest) != _MANIFEST_KEYS or manifest.get("status") != "complete":
        raise ValueError("unsupported or incomplete imagined rollout manifest")
    _validate_markers(manifest)
    specification = manifest["payload"]
    if type(specification) is not dict or set(specification) != {"path", "bytes", "sha256"}:
        raise ValueError("unsupported rollout payload specification")
    if specification["path"] != _PAYLOAD_NAME:
        raise ValueError("unsupported or escaping rollout payload path")
    payload_path = _member(root, specification["path"])
    if type(specification["bytes"]) is not int or payload_path.stat().st_size != specification["bytes"]:
        raise ValueError("imagined rollout payload length mismatch")
    if _sha256(payload_path) != specification["sha256"]:
        raise ValueError("imagined rollout payload hash mismatch")
    with payload_path.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    rollout = _decode(payload)
    expected = {"metadata": payload["metadata"], "tensor_specs": _specifications(payload)}
    recorded = {key: manifest[key] for key in expected}
    if (json.dumps(recorded, sort_keys=True, allow_nan=False) !=
            json.dumps(expected, sort_keys=True, allow_nan=False)):
        raise ValueError("imagined rollout manifest metadata or tensor specifications differ")
    return rollout.to(device)
