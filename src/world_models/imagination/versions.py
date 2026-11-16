"""Exact, side-effect-free parameter identities for full-prefix reconstruction.

Hash values at public reconstruction boundaries rather than relying on a caller's
version label or Tensor._version: writes through ``parameter.data`` can bypass
the latter. These identities deliberately exclude train/eval and requires_grad;
the prefix builder supplies its own evaluation and no-gradient context.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json

import torch
from torch import nn


def _json(value) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False, ensure_ascii=True).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("module version metadata must be finite JSON-compatible values") from error


def _configuration(module: nn.Module, name: str):
    value = getattr(module, name, None)
    if value is None:
        return None
    if not is_dataclass(value) or isinstance(value, type):
        raise ValueError(f"module {name} must be a dataclass instance when present")
    return {"class": f"{type(value).__module__}.{type(value).__qualname__}",
            "values": asdict(value)}


@torch.no_grad()
def module_version(module: nn.Module) -> str:
    """Return a SHA256 of exact parameters, buffers, and nested configuration.

    Includes each named submodule's fully qualified class and optional dataclass
    ``config`` and ``task``, and each state tensor's name, dtype, shape and device.
    Nonpersistent buffers are included because they can influence inference even
    when they are omitted from checkpoints. Equal values in separate storage have
    equal identities; device migration invalidates a cache. Floating and complex
    values must be finite. State must use ordinary dense materialized tensors;
    custom extra state is deliberately unsupported.

    No tensors, gradients, modes or RNG streams are modified. Byte extraction
    supports bfloat16 and scalar tensors without converting their numeric type.
    """
    if not isinstance(module, nn.Module):
        raise ValueError("module_version requires a torch.nn.Module")
    digest = hashlib.sha256()

    def append(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    descriptions = []
    for name, child in sorted(module.named_modules(remove_duplicate=False)):
        descriptions.append({"name": name,
                             "class": f"{type(child).__module__}.{type(child).__qualname__}",
                             "config": _configuration(child, "config"),
                             "task": _configuration(child, "task")})
    append(_json({"schema_version": "imagination-module-version-v1", "modules": descriptions}))
    state = module.state_dict()
    tensors = [("state_dict", name, tensor) for name, tensor in state.items()]
    tensors.extend(("nonpersistent_buffer", name, tensor)
                   for name, tensor in module.named_buffers(remove_duplicate=False) if name not in state)
    for kind, name, tensor in sorted(tensors, key=lambda entry: entry[:2]):
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.device.type == "meta" or tensor.is_quantized):
            raise ValueError(f"module state {name!r} must be a dense materialized tensor")
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all().item():
            raise ValueError(f"module state {name!r} contains nonfinite values")
        append(_json({"kind": kind, "name": name, "dtype": str(tensor.dtype),
                      "shape": list(tensor.shape), "device": str(tensor.device)}))
        storage = tensor.detach().cpu().resolve_conj().resolve_neg().contiguous().reshape(-1)
        append(storage.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
