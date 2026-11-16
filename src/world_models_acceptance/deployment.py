"""Actor-only deployment proof in a fresh, deliberately restricted process.

The parent module is standard-library only. The child verifies the immutable
export and performs three synthetic local observations, with teacher, world
model, imagination, learning and simulator imports unavailable. This is an
engineering check; it produces no scientific performance measurements.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
import hashlib
import importlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import subprocess
import sys


SCHEMA_VERSION = "p45-actor-deployment-probe-v1"
FORBIDDEN_IMPORTS = (
    "world_models.models", "world_models.teacher", "world_models.policies.teacher",
    "world_models.distillation", "world_models.refinement", "world_models_campaign",
    "world_models.imagination", "world_models.rollouts", "world_models.mappo",
    "world_models.training", "world_models.env", "vizdoom",
)
_EXPORT_KEYS = {
    "schema_version", "task_family", "variant", "run_seed", "real_joint_transitions",
    "actual_initial_replay_interactions", "untrained_zero_interaction_baseline",
    "actor_version", "actor_config", "world_model_version", "model_updates",
    "local_ppo_cycles", "actor_optimizer_steps", "critic_optimizer_steps",
    "evaluation_split", "on_registered_evaluation_grid", "evaluation_executed", "phase3_acceptance",
}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actor_version(actor, origin_device):
    """Reproduce the registered actor fingerprint without learning imports.

    The historical version includes tensor device labels. CPU-loaded tensor
    bytes are checked with the recorded origin label; CUDA is never allocated.
    """
    import torch
    digest = hashlib.sha256()
    def append(value):
        digest.update(len(value).to_bytes(8, "big")); digest.update(value)
    def configuration(module, name):
        value = getattr(module, name, None)
        if value is None:
            return None
        if not is_dataclass(value) or isinstance(value, type):
            raise ValueError("unsupported actor configuration")
        return {"class": f"{type(value).__module__}.{type(value).__qualname__}", "values": asdict(value)}
    descriptions = [{"name": name, "class": f"{type(child).__module__}.{type(child).__qualname__}",
                     "config": configuration(child, "config"), "task": configuration(child, "task")}
                    for name, child in sorted(actor.named_modules(remove_duplicate=False))]
    append(_json_bytes({"schema_version": "imagination-module-version-v1", "modules": descriptions}))
    state = actor.state_dict()
    tensors = [("state_dict", name, value) for name, value in state.items()]
    tensors.extend(("nonpersistent_buffer", name, value) for name, value in
                   actor.named_buffers(remove_duplicate=False) if name not in state)
    for kind, name, value in sorted(tensors, key=lambda row: row[:2]):
        if (value.layout != torch.strided or value.device.type == "meta" or value.is_quantized
                or not bool(torch.isfinite(value).all())):
            raise ValueError("actor state must be materialized, dense, and finite")
        append(_json_bytes({"kind": kind, "name": name, "dtype": str(value.dtype),
                            "shape": list(value.shape), "device": origin_device}))
        append(value.detach().cpu().resolve_conj().resolve_neg().contiguous().reshape(-1)
               .view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_local_actor_export(export_dir, *, source_device=None):
    """Verify an actor-only receipt and return an inference-only CPU actor.

    This path imports only the generic tensor-checkpoint reader and local policy
    code. Its exact state schema rejects a controller checkpoint before tensors
    are loaded, and the generic reader verifies payload hashes/specifications.
    """
    import torch
    from world_models.cycle.checkpoints import load_cycle_checkpoint
    from world_models.policies.actor import LocalActor
    from world_models.policies.config import PolicyConfig
    candidate = Path(export_dir)
    if ".." in candidate.parts or any(p.is_symlink() for p in (candidate, *candidate.parents)):
        raise ValueError("actor export cannot traverse parents or symbolic links")
    root = candidate.resolve()
    manifest_path, receipt = root / "manifest.json", root / "manifest.sha256"
    if not manifest_path.is_file() or not receipt.is_file() or any(
            p.is_symlink() for p in (manifest_path, receipt)):
        raise ValueError("actor export requires an immutable manifest receipt")
    if receipt.read_text(encoding="ascii").strip() != f"{_sha256(manifest_path)}  manifest.json":
        raise ValueError("actor export manifest receipt mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specifications = manifest.get("tensor_specs")
    if (manifest.get("metadata", {}).get("schema_version") != "p35-local-actor-v1"
            or not isinstance(specifications, list) or not specifications or any(
                type(entry) is not dict or not entry.get("path") or
                entry["path"][0] != ["key", "actor"] for entry in specifications)):
        raise ValueError("deployment requires an actor-only export")
    state, metadata = load_cycle_checkpoint(root, device="cpu")
    if set(state) != {"actor"} or type(state["actor"]) is not dict or set(metadata) != _EXPORT_KEYS:
        raise ValueError("deployment requires exactly actor weights and local export metadata")
    if (metadata["task_family"] not in ("relay", "cover") or metadata["variant"] not in ("B", "D")
            or metadata["evaluation_split"] not in ("validation", "test")
            or metadata["evaluation_executed"] is not False or metadata["phase3_acceptance"] is not False):
        raise ValueError("invalid actor-only export identity or execution markers")
    for name in ("run_seed", "real_joint_transitions", "actual_initial_replay_interactions", "model_updates",
                 "local_ppo_cycles", "actor_optimizer_steps", "critic_optimizer_steps"):
        if type(metadata[name]) is not int or metadata[name] < 0:
            raise ValueError("actor export counters must be nonnegative integers")
    for name in ("untrained_zero_interaction_baseline", "on_registered_evaluation_grid"):
        if type(metadata[name]) is not bool:
            raise ValueError("actor export flags must be boolean")
    if (metadata["untrained_zero_interaction_baseline"] and any(metadata[key] for key in (
            "real_joint_transitions", "local_ppo_cycles", "actor_optimizer_steps", "critic_optimizer_steps"))
            or (metadata["local_ppo_cycles"] == 0) != (metadata["actor_optimizer_steps"] == 0)
            or metadata["actor_optimizer_steps"] != metadata["critic_optimizer_steps"]):
        raise ValueError("actor export learning counters disagree")
    for name in ("actor_version", "world_model_version"):
        if (type(metadata[name]) is not str or len(metadata[name]) != 64 or
                any(ch not in "0123456789abcdef" for ch in metadata[name])):
            raise ValueError("actor export version must be a SHA256 identifier")
    values = metadata["actor_config"]
    if type(values) is not dict or set(values) != {field.name for field in fields(PolicyConfig)}:
        raise ValueError("actor export requires its exact local configuration")
    values = dict(values)
    for name in ("image_shape", "encoder_channels", "critic_hidden_sizes"):
        if type(values[name]) is not list:
            raise ValueError("actor configuration dimensions must use JSON arrays")
        values[name] = tuple(values[name])
    config = PolicyConfig(**values)
    with torch.random.fork_rng(devices=[]):
        actor = LocalActor(config)
    expected = actor.state_dict()
    if set(state["actor"]) != set(expected) or any(
            not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
            tuple(value.shape) != tuple(expected[name].shape) for name, value in state["actor"].items()):
        raise ValueError("actor tensor names, shapes, or FP32 dtypes disagree")
    actor.load_state_dict(state["actor"], strict=True)
    candidates = ["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    if source_device is not None:
        label = torch.device(source_device)
        if (str(label) != source_device or label.type not in ("cpu", "cuda")
                or label.type == "cpu" and label.index is not None
                or label.type == "cuda" and label.index is None):
            raise ValueError("source_device must be cpu or an explicit cuda:N")
        candidates = [source_device]
    origin = next((label for label in candidates if _actor_version(actor, label) == metadata["actor_version"]), None)
    if origin is None:
        raise ValueError("actor tensor version mismatch; supply the original source_device for other GPU indices")
    actor.eval().requires_grad_(False)
    return actor, {"path": str(root), "manifest_sha256": _sha256(manifest_path),
                   "payload_sha256": manifest["payload"]["sha256"], "metadata": metadata,
                   "verified_origin_device": origin,
                   "portable_weights_sha256": _actor_version(actor, "cpu")}


def _forbidden(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORTS)


class _BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if _forbidden(fullname):
            raise ImportError(f"P4.5 deployment isolation forbids {fullname}")
        return None


def _child_probe(export_dir, *, source_device=None):
    present = [name for name in sys.modules if _forbidden(name)]
    if present:
        raise ValueError(f"deployment worker already imported forbidden modules: {present}")
    blocker = _BlockTrainingImports()
    sys.meta_path.insert(0, blocker)
    import torch
    from world_models.contracts import LocalActorInput, RGBFrame
    torch.set_num_threads(1)
    actor, provenance = load_local_actor_export(export_dir, source_device=source_device)
    blocked_attempts = []
    for name in FORBIDDEN_IMPORTS:
        try:
            importlib.import_module(name)
        except ImportError as error:
            if "P4.5 deployment isolation forbids" not in str(error):
                raise ValueError(f"deployment import blocker did not prove isolation for {name}") from error
            blocked_attempts.append(name)
        else:
            raise ValueError(f"deployment unexpectedly imported {name}")
    before = _actor_version(actor, "cpu")
    hidden = tuple(0.0 for _ in range(actor.config.hidden_size))
    outputs, actions = [], []
    for step in range(3):
        # Distinct fixed current images and actually selected preceding actions.
        image = RGBFrame(bytes([step * 71]) * (3 * 96 * 128))
        previous = 12 if step == 0 else actions[-1]
        output = actor.step(LocalActorInput(image, previous, hidden))
        probabilities, hidden = output.action_probabilities, output.next_local_hidden
        if (len(probabilities) != 12 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)
                or abs(sum(probabilities) - 1) > 2e-6 or len(hidden) != actor.config.hidden_size
                or any(not math.isfinite(v) for v in hidden)):
            raise ValueError("actor deployment produced invalid probabilities or local memory")
        actions.append(max(range(12), key=lambda index: probabilities[index]))
        outputs.append(list(probabilities))
    if _actor_version(actor, "cpu") != before:
        raise ValueError("actor deployment mutated its weights")
    imported = [name for name in sys.modules if _forbidden(name)]
    if imported:
        raise ValueError(f"deployment imported forbidden modules: {imported}")
    return {"schema_version": SCHEMA_VERSION, "status": "passed", "passed": True,
            "local_only": True, "isolated_process": True,
            "data_source": "synthetic_local_actor_fixture", "native_environment_opened": False,
            "research_evaluation_executed": False, "training_executed": False,
            "new_environment_interactions": 0, "observation_steps": 3,
            "action_probabilities": outputs, "selected_actions": actions,
            "forbidden_imports": list(FORBIDDEN_IMPORTS), "imported_forbidden_modules": imported,
            "forbidden_modules_loaded": imported, "blocked_import_attempts": blocked_attempts,
            "manifest_sha256": provenance["manifest_sha256"],
            "payload_sha256": provenance["payload_sha256"],
            "actor_version": provenance["metadata"]["actor_version"],
            "source_device": provenance["verified_origin_device"],
            "weights_unchanged": True, "provenance": provenance}


def probe_actor_export(export_dir, *, output_dir=None, source_device=None):
    """Run the isolated proof; optionally publish a new receipt-bound result."""
    root = Path(export_dir).resolve()
    command = [sys.executable, "-m", "world_models_acceptance.deployment", "--worker", str(root)]
    if source_device is not None:
        command.extend(["--source-device", source_device])
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(command, capture_output=True, text=True, env=environment,
                            timeout=180, check=False)
    if result.returncode:
        raise ValueError("actor-only deployment probe failed: " + result.stderr.strip()[-4000:])
    try:
        report = json.loads(result.stdout)
    except ValueError as error:
        raise ValueError("actor-only deployment worker did not return JSON") from error
    if (report.get("schema_version") != SCHEMA_VERSION or report.get("status") != "passed"
            or report.get("isolated_process") is not True or report.get("imported_forbidden_modules") != []):
        raise ValueError("actor-only deployment worker returned invalid evidence")
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        path = destination / "deployment_probe.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (destination / "deployment_probe.sha256").write_text(f"{_sha256(path)}  {path.name}\n", encoding="ascii")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--source-device")
    arguments = parser.parse_args(argv)
    report = _child_probe(arguments.worker, source_device=arguments.source_device)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
