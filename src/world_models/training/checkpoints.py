"""Verified, immutable P2.4 optimizer-boundary checkpoints.

Call ``optimizer.zero_grad(set_to_none=True)`` before saving. The receipt is the
commit marker: a checkpoint without its sibling SHA256 receipt is incomplete.
Only state dictionaries, primitive containers and CPU tensors are serialized;
loading always uses PyTorch's restricted ``weights_only`` loader.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
from torch import Tensor, nn


SCHEMA_VERSION = "p24-checkpoint-v1"
_KEYS = {"schema_version", "update", "metadata", "model", "optimizer", "streams", "sampler", "rng"}


def _json_metadata(value: dict, name: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    def validate(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for child in item.values():
                validate(child)
            return
        raise ValueError(f"{name} contains a non-JSON or nonfinite value")
    validate(value)
    # Detach the metadata from mutable caller-owned containers.
    return json.loads(json.dumps(value, allow_nan=False))


def _same_json(left, right) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _cpu_state(value, *, location: str):
    """Copy only safe state-dictionary types, never whole Python objects."""
    if isinstance(value, Tensor):
        if value.layout != torch.strided:
            raise ValueError(f"{location} contains a non-strided tensor")
        return value.detach().to(device="cpu", copy=True)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a nonfinite scalar")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _cpu_state(float(value), location=location)
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError(f"{location} contains an unsupported NumPy array")
        return torch.tensor(value.copy())
    if isinstance(value, Mapping):
        if any(type(key) not in (str, int) for key in value):
            raise ValueError(f"{location} state keys must be strings or integers")
        return {key: _cpu_state(item, location=f"{location}/{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_cpu_state(item, location=location) for item in value]
        return tuple(converted) if isinstance(value, tuple) else converted
    raise ValueError(f"{location} contains unsupported state type {type(value).__name__}")


def _fp32_tensors(value, name: str) -> None:
    if isinstance(value, Tensor):
        if value.is_complex() or (value.is_floating_point() and value.dtype != torch.float32):
            raise ValueError(f"{name} floating tensors must be float32")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{name} contains a nonfinite tensor")
    elif isinstance(value, Mapping):
        for item in value.values():
            _fp32_tensors(item, name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _fp32_tensors(item, name)


def _rng_tensor(value, name: str) -> Tensor:
    if (not isinstance(value, Tensor) or value.dtype != torch.uint8 or value.ndim != 1
            or value.device.type != "cpu" or not value.numel()):
        raise ValueError(f"{name} must be a nonempty CPU uint8 RNG state")
    return value


def _cuda_devices(model: nn.Module) -> list[int]:
    return sorted({tensor.device.index for tensor in (*model.parameters(), *model.buffers())
                   if tensor.device.type == "cuda"})


def _capture_rng(model: nn.Module) -> dict:
    algorithm, keys, position, has_gaussian, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": algorithm, "keys": torch.tensor(keys.astype(np.int64)),
                  "position": int(position), "has_gaussian": int(has_gaussian),
                  "cached_gaussian": float(cached_gaussian)},
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": {str(device): torch.cuda.get_rng_state(device).cpu().clone()
                       for device in _cuda_devices(model)},
    }


def _validate_rng(rng: dict, *, restoring: bool) -> None:
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("checkpoint RNG schema is invalid")
    try:
        random.Random().setstate(rng["python"])
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint Python RNG state is invalid") from error
    state = rng["numpy"]
    if not isinstance(state, dict) or set(state) != {"algorithm", "keys", "position", "has_gaussian", "cached_gaussian"}:
        raise ValueError("checkpoint NumPy RNG schema is invalid")
    keys = state["keys"]
    if (state["algorithm"] != "MT19937" or not isinstance(keys, Tensor) or keys.dtype != torch.int64
            or keys.shape != (624,) or keys.device.type != "cpu"
            or bool(((keys < 0) | (keys > 2**32-1)).any())
            or type(state["position"]) is not int or not 0 <= state["position"] <= 624
            or type(state["has_gaussian"]) is not int or state["has_gaussian"] not in (0, 1)
            or type(state["cached_gaussian"]) is not float or not math.isfinite(state["cached_gaussian"])):
        raise ValueError("checkpoint NumPy RNG state is invalid")
    _rng_tensor(rng["torch_cpu"], "torch CPU")
    try:
        torch.Generator(device="cpu").set_state(rng["torch_cpu"])
    except RuntimeError as error:
        raise ValueError("checkpoint torch CPU RNG state is invalid") from error
    cuda = rng["torch_cuda"]
    if not isinstance(cuda, dict):
        raise ValueError("checkpoint CUDA RNG states must be a mapping")
    for device, value in cuda.items():
        if not isinstance(device, str) or not device.isdecimal() or str(int(device)) != device:
            raise ValueError("checkpoint CUDA RNG device index is invalid")
        _rng_tensor(value, f"CUDA {device}")
        if restoring and (not torch.cuda.is_available() or int(device) >= torch.cuda.device_count()):
            raise ValueError(f"checkpoint CUDA RNG device {device} is unavailable")


def _restore_rng(rng: dict) -> None:
    state = rng["numpy"]
    random.setstate(rng["python"])
    np.random.set_state((state["algorithm"], state["keys"].numpy().astype(np.uint32),
                         state["position"], state["has_gaussian"], state["cached_gaussian"]))
    torch.set_rng_state(rng["torch_cpu"])
    for device, value in rng["torch_cuda"].items():
        torch.cuda.set_rng_state(value, int(device))


def _validate_payload(payload: dict, *, restore_rng: bool) -> None:
    if not isinstance(payload, dict) or set(payload) != _KEYS or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported or incomplete checkpoint schema")
    if type(payload["update"]) is not int or payload["update"] < 0:
        raise ValueError("checkpoint update must be a nonnegative integer")
    _json_metadata(payload["metadata"], "checkpoint metadata")
    for name in ("model", "optimizer", "streams", "sampler"):
        if not isinstance(payload[name], dict):
            raise ValueError(f"checkpoint {name} must be a state dictionary")
    if not payload["model"] or any(not isinstance(value, Tensor) for value in payload["model"].values()):
        raise ValueError("checkpoint model must contain named state tensors")
    if set(payload["optimizer"]) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer schema is invalid")
    if not isinstance(payload["optimizer"]["state"], dict) or not isinstance(payload["optimizer"]["param_groups"], list):
        raise ValueError("checkpoint optimizer state/groups are invalid")
    groups = payload["optimizer"]["param_groups"]
    if any(not isinstance(group, dict) or not isinstance(group.get("params"), list)
           or any(type(index) is not int or index < 0 for index in group["params"])
           for group in groups):
        raise ValueError("checkpoint optimizer parameter groups are invalid")
    parameters = [index for group in groups for index in group["params"]]
    if len(parameters) != len(set(parameters)) or any(type(index) is not int or index not in parameters
                                                     for index in payload["optimizer"]["state"]):
        raise ValueError("checkpoint optimizer parameter identifiers are invalid")
    _fp32_tensors(payload["model"], "model")
    _fp32_tensors(payload["optimizer"], "optimizer")
    # Sampler states are small; this additionally rejects otherwise allowlisted
    # torch objects (for example torch.device) outside the primitive contract.
    _cpu_state(payload["sampler"], location="sampler")
    if set(payload["streams"]) != {"prior", "posterior", "outcome"}:
        raise ValueError("checkpoint streams must contain prior/posterior/outcome")
    for name, value in payload["streams"].items():
        _rng_tensor(value, f"sampling stream {name}")
    _validate_rng(payload["rng"], restoring=restore_rng)


def _digest(stream) -> str:
    hasher = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        hasher.update(block)
    return hasher.hexdigest()


def _fsync_directory(directory: Path) -> None:
    # Windows does not expose directory fsync through os.open; file data is
    # still flushed before replacement on every platform.
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _paths(path: str | Path) -> tuple[Path, Path, Path]:
    path = Path(path)
    return path, path.with_suffix(".sha256"), path.with_name(path.name + ".lock")


def save_checkpoint(path, *, model, optimizer, streams, sampler, update: int, metadata: dict) -> dict:
    """Publish a new immutable checkpoint after clearing optimizer gradients.

    The caller records task/seed, resolved config, replay manifest identities,
    normalization identity, source hashes, precision and device in ``metadata``.
    ``update`` counts completed optimizer updates, independently of interactions.
    A stale lock/temp file or payload/receipt orphan requires explicit recovery;
    this function never overwrites existing evidence.
    """
    if not isinstance(model, nn.Module) or not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("checkpoint requires a model and optimizer")
    if type(update) is not int or update < 0:
        raise ValueError("update must be a nonnegative integer")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("save only at an optimizer boundary after zero_grad(set_to_none=True)")
    metadata = _json_metadata(metadata, "metadata")
    payload = {"schema_version": SCHEMA_VERSION, "update": update, "metadata": metadata,
               "model": _cpu_state(model.state_dict(), location="model"),
               "optimizer": _cpu_state(optimizer.state_dict(), location="optimizer"),
               "streams": _cpu_state(streams.state_dict(), location="streams"),
               "sampler": _cpu_state(sampler.state_dict(), location="sampler"),
               "rng": _capture_rng(model)}
    _validate_payload(payload, restore_rng=False)
    path, receipt, lock = _paths(path)
    if path == receipt:
        raise ValueError("checkpoint path must not use the reserved .sha256 extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() != receipt.exists():
        raise FileExistsError("incomplete checkpoint/receipt pair; recover the orphan before saving")
    if path.exists() or receipt.exists():
        raise FileExistsError("checkpoint is immutable; choose a new numbered path")
    stale = [*path.parent.glob(path.name + ".tmp-*"), *path.parent.glob(receipt.name + ".tmp-*")]
    if stale or lock.exists():
        raise FileExistsError("stale checkpoint lock/temporary artifacts; inspect and recover them before saving")
    try:
        lock_descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError("checkpoint is being written or has a stale lock") from error
    suffix = ".tmp-" + uuid.uuid4().hex
    temporary, temporary_receipt = path.with_name(path.name + suffix), receipt.with_name(receipt.name + suffix)
    try:
        os.close(lock_descriptor)
        # Recheck after acquiring the exclusive writer lock.
        if path.exists() or receipt.exists():
            raise FileExistsError("checkpoint is immutable; choose a new numbered path")
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        with temporary.open("rb") as stream:
            digest = _digest(stream)
        with temporary_receipt.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(f"{digest}  {path.name}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        os.replace(temporary_receipt, receipt)
        _fsync_directory(path.parent)
    finally:
        for owned in (temporary, temporary_receipt, lock):
            owned.unlink(missing_ok=True)
    return {"path": str(path), "receipt_path": str(receipt), "sha256": digest,
            "update": update, "schema_version": SCHEMA_VERSION}


def read_checkpoint(path) -> dict:
    """Return a hash/schema-validated CPU payload without restoring any state.

    This supports a metadata peek before model construction and works on CPU
    hosts for checkpoints written on CUDA. It never alters any RNG stream.
    """
    path, receipt, lock = _paths(path)
    if lock.exists():
        raise ValueError("checkpoint has an active/stale writer lock; inspect it before loading")
    if not path.is_file() or not receipt.is_file():
        raise ValueError("checkpoint and its SHA256 receipt are both required; incomplete/orphan checkpoint")
    try:
        parts = receipt.read_text(encoding="ascii").strip().split(maxsplit=1)
    except (UnicodeError, OSError) as error:
        raise ValueError("checkpoint SHA256 receipt cannot be read") from error
    if len(parts) != 2 or parts[1] != path.name or len(parts[0]) != 64:
        raise ValueError("checkpoint SHA256 receipt is malformed")
    with path.open("rb") as stream:
        if _digest(stream) != parts[0]:
            raise ValueError("checkpoint SHA256 mismatch; checkpoint is truncated or changed")
        stream.seek(0)
        try:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError("checkpoint cannot be decoded by the restricted weights-only loader") from error
    _validate_payload(payload, restore_rng=False)
    return payload


def load_checkpoint(path, *, model, optimizer=None, streams=None, sampler=None,
                    expected_metadata: dict | None = None, restore_rng: bool = True) -> dict:
    """Verify bytes and provenance before restoring state dictionaries.

    Expected top-level metadata keys may be a subset; each requested value must
    match deeply and exactly, including all nested keys and scalar types. No
    fallback to unrestricted pickle is allowed. Set ``restore_rng=False`` for
    inspection or device migration where exact continuation is not claimed.
    """
    if not isinstance(model, nn.Module) or type(restore_rng) is not bool:
        raise ValueError("model must be an nn.Module and restore_rng must be boolean")
    expected = None if expected_metadata is None else _json_metadata(expected_metadata, "expected metadata")
    payload = read_checkpoint(path)
    if restore_rng:
        _validate_rng(payload["rng"], restoring=True)
    if expected is not None:
        for key, value in expected.items():
            if key not in payload["metadata"] or not _same_json(value, payload["metadata"][key]):
                raise ValueError(f"checkpoint metadata mismatch: {key}")
    current = model.state_dict()
    if set(current) != set(payload["model"]):
        raise ValueError("checkpoint model state keys differ from the destination model")
    if any(current[key].shape != value.shape or current[key].dtype != value.dtype
           for key, value in payload["model"].items()):
        raise ValueError("checkpoint model tensor shape/dtype differs from the destination model")
    if optimizer is not None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("optimizer must be a torch optimizer")
        groups = optimizer.state_dict()["param_groups"]
        saved = payload["optimizer"]["param_groups"]
        if len(groups) != len(saved) or any(len(left["params"]) != len(right["params"])
                                            for left, right in zip(groups, saved)):
            raise ValueError("checkpoint optimizer parameter groups differ")
    # PyTorch maps optimizer moments to each parameter's device. Its scalar
    # Adam step deliberately remains on CPU unless capturable/fused is enabled.
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if streams is not None:
        streams.load_state_dict(payload["streams"])
    if sampler is not None:
        sampler.load_state_dict(payload["sampler"])
    if restore_rng:
        _restore_rng(payload["rng"])
    return payload
