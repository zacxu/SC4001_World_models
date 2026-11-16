"""Immutable, receipt-bound state for completed controller stages.

This store knows no model or optimizer classes. The controller validates its own
state schema, logical stage identifiers, replay identity and configuration hash.
The final manifest receipt marks a complete publication; partial output is never
silently reused. Checkpoint tensors cannot become real replay observations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = "real-model-policy-cycle-checkpoint-v1"
_PAYLOAD_NAME = "state.pt"
_MARKERS = {"schema_version": SCHEMA_VERSION, "data_source": "training_controller_state",
            "learning_eligible_for_world_model": False}
_MANIFEST_KEYS = {*_MARKERS, "status", "metadata", "payload", "tensor_specs"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _json_copy(value: Any) -> Any:
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("checkpoint metadata keys must be strings")
        return {key: _json_copy(child) for key, child in value.items()}
    if type(value) is list:
        return [_json_copy(child) for child in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("checkpoint metadata must contain finite JSON values")


def _tree(value: Any, *, device: str | torch.device = "cpu") -> Any:
    if isinstance(value, torch.Tensor):
        if (value.layout != torch.strided or value.requires_grad or value.grad_fn is not None
                or torch.is_inference(value) or not torch.isfinite(value).all().item()):
            raise ValueError("checkpoint tensors must be ordinary detached dense finite tensors")
        with torch.no_grad(), torch.inference_mode(False):
            return value.detach().to(device).clone()
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise ValueError("checkpoint state keys must be strings or integers")
        return {key: _tree(child, device=device) for key, child in value.items()}
    if type(value) in (tuple, list):
        return type(value)(_tree(child, device=device) for child in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("checkpoint state contains an unsupported or nonfinite value")


def _specifications(value: Any, path: tuple = ()) -> list[dict]:
    if isinstance(value, torch.Tensor):
        return [{"path": list(path), "shape": list(value.shape), "dtype": str(value.dtype),
                 "elements": value.numel(), "bytes": value.numel() * value.element_size()}]
    result = []
    children = value.items() if isinstance(value, dict) else enumerate(value) if type(value) in (tuple, list) else ()
    for key, child in children:
        # Preserve key type: optimizer's integer key 0 differs from string "0".
        step = ["key", key] if isinstance(value, dict) else ["index", key]
        result.extend(_specifications(child, (*path, step)))
    return result


def _root(path: str | Path) -> Path:
    candidate = Path(path)
    if ".." in candidate.parts or any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError("checkpoint directory may not traverse parents or symbolic links")
    return candidate.resolve()


def _guard_real_replay(root: Path) -> None:
    for ancestor in (root, *root.parents):
        for name in ("manifest.json", "manifest.partial.json"):
            candidate = ancestor / name
            if not candidate.is_file():
                continue
            try:
                document = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ValueError("cannot verify checkpoint separation from real replay") from error
            if isinstance(document, dict) and (document.get("dataset_version") == "real-replay-v1" or
                    isinstance(document.get("metadata"), dict) and
                    document["metadata"].get("data_source") == "real_environment"):
                raise ValueError("controller checkpoints cannot be stored inside real replay")


def _member(root: Path, name: str) -> Path:
    if type(name) is not str or name not in ("manifest.json", "manifest.sha256", _PAYLOAD_NAME):
        raise ValueError("unsupported or escaping checkpoint member")
    candidate = root / name
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
        raise ValueError("checkpoint member escapes through a symbolic link")
    if not candidate.is_file():
        raise ValueError(f"incomplete checkpoint: missing {name}")
    return candidate


def _markers(document: dict) -> None:
    if any(type(document.get(key)) is not type(value) or document[key] != value
           for key, value in _MARKERS.items()):
        raise ValueError("unsupported checkpoint schema or data markers")


def _decode(payload: Any) -> tuple[dict, dict]:
    if type(payload) is not dict or set(payload) != {*_MARKERS, "state", "metadata"}:
        raise ValueError("unsupported checkpoint payload keys")
    _markers(payload)
    if type(payload["state"]) is not dict or not payload["state"]:
        raise ValueError("checkpoint state must be a nonempty dictionary")
    if type(payload["metadata"]) is not dict:
        raise ValueError("checkpoint metadata must be a JSON object")
    return _tree(payload["state"]), _json_copy(payload["metadata"])


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@torch.inference_mode(False)
def save_cycle_checkpoint(output_dir: str | Path, *, state: dict, metadata: dict) -> Path:
    """Publish detached CPU-owned state into a fresh directory, receipt last."""
    root = _root(output_dir)
    _guard_real_replay(root)
    if root.exists():
        raise FileExistsError("cycle checkpoint output must be a fresh directory")
    if type(state) is not dict or not state or type(metadata) is not dict:
        raise ValueError("checkpoint requires nonempty state and dictionary metadata")
    owned_state, owned_metadata = _tree(state), _json_copy(metadata)
    payload = {**_MARKERS, "state": owned_state, "metadata": owned_metadata}
    root.mkdir(parents=True, exist_ok=False)
    payload_path = root / _PAYLOAD_NAME
    with payload_path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    with payload_path.open("rb") as stream:
        saved = torch.load(stream, map_location="cpu", weights_only=True)
    _decode(saved)
    manifest = {**_MARKERS, "status": "complete", "metadata": _json_copy(owned_metadata),
                "payload": {"path": _PAYLOAD_NAME, "bytes": payload_path.stat().st_size,
                            "sha256": _sha256(payload_path)},
                "tensor_specs": _specifications(saved["state"])}
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    temporary_receipt = root / "manifest.sha256.tmp"
    with temporary_receipt.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{_sha256(manifest_path)}  manifest.json\n")
        stream.flush()
        os.fsync(stream.fileno())
    # The final name becomes visible atomically after its complete bytes are
    # durable. A process killed during writing leaves only an uncommitted tmp.
    os.replace(temporary_receipt, root / "manifest.sha256")
    if os.name == "posix":
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return manifest_path


@torch.inference_mode(False)
def load_cycle_checkpoint(output_dir: str | Path, *, device: str | torch.device = "cpu") -> tuple[dict, dict]:
    """Verify the immutable publication before loading weights-only owned state."""
    root = _root(output_dir)
    _guard_real_replay(root)
    manifest_path, receipt = _member(root, "manifest.json"), _member(root, "manifest.sha256")
    if receipt.read_text(encoding="ascii").strip() != f"{_sha256(manifest_path)}  manifest.json":
        raise ValueError("cycle checkpoint manifest receipt mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if type(manifest) is not dict or set(manifest) != _MANIFEST_KEYS or manifest.get("status") != "complete":
        raise ValueError("unsupported or incomplete cycle checkpoint manifest")
    _markers(manifest)
    specification = manifest["payload"]
    if type(specification) is not dict or set(specification) != {"path", "bytes", "sha256"}:
        raise ValueError("unsupported checkpoint payload specification")
    if specification["path"] != _PAYLOAD_NAME:
        raise ValueError("unsupported or escaping checkpoint payload path")
    payload_path = _member(root, specification["path"])
    if type(specification["bytes"]) is not int or specification["bytes"] <= 0 or payload_path.stat().st_size != specification["bytes"]:
        raise ValueError("cycle checkpoint payload length mismatch")
    if (type(specification["sha256"]) is not str or len(specification["sha256"]) != 64 or
            any(char not in "0123456789abcdef" for char in specification["sha256"]) or
            _sha256(payload_path) != specification["sha256"]):
        raise ValueError("cycle checkpoint payload hash mismatch")
    with payload_path.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    state, metadata = _decode(payload)
    recorded = {"metadata": manifest["metadata"], "tensor_specs": manifest["tensor_specs"]}
    expected = {"metadata": metadata, "tensor_specs": _specifications(state)}
    if json.dumps(recorded, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("cycle checkpoint metadata or tensor specifications differ")
    return _tree(state, device=device), _json_copy(metadata)
