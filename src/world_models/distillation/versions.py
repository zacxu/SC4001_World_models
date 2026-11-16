"""Verify historical device-labelled identities without allocating that device.

The original imagination hash deliberately includes tensor device names for
cache invalidation. A stage handoff must verify those exact historical labels
before moving saved values to its own inference device. Never change or rewrite
the historical hash algorithm or the saved registration to perform a transfer.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn

from world_models.imagination.versions import _configuration, _json


def origin_device(value):
    """Accept only canonical CPU or explicit logical CUDA device labels."""
    if type(value) is not str:
        raise ValueError("recorded device must be cpu or an explicit cuda:N")
    try:
        device = torch.device(value)
    except (ValueError, RuntimeError) as error:
        raise ValueError("recorded device must be cpu or an explicit cuda:N") from error
    if (str(device) != value or device.type not in ("cpu", "cuda")
            or device.type == "cpu" and device.index is not None
            or device.type == "cuda" and device.index is None):
        raise ValueError("recorded device must be cpu or an explicit cuda:N")
    return value


@torch.no_grad()
def module_version_at_device(module: nn.Module, recorded_device: str) -> str:
    """Hash exact bytes/configuration with the recorded device metadata.

    This is byte verification, not a numerical inference-equivalence claim.
    It neither allocates CUDA tensors nor mutates weights, modes or RNG.
    """
    label = origin_device(recorded_device)
    if not isinstance(module, nn.Module):
        raise ValueError("module version verification requires a torch.nn.Module")
    digest = hashlib.sha256()

    def append(value):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    descriptions = [{"name": name, "class": f"{type(child).__module__}.{type(child).__qualname__}",
                     "config": _configuration(child, "config"), "task": _configuration(child, "task")}
                    for name, child in sorted(module.named_modules(remove_duplicate=False))]
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
                      "shape": list(tensor.shape), "device": label}))
        storage = tensor.detach().cpu().resolve_conj().resolve_neg().contiguous().reshape(-1)
        append(storage.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
