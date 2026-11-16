"""Local-actor-only, counted native validation; fixtures cannot claim native evidence.

There is deliberately no import of an imagination builder, world model, critic
instance, controller, optimizer, or replay writer. P3.5's generic tensor store
is used only to read an export whose complete state is exactly ``{'actor'}``.
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

import torch

from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.data.ledger import TrainingInteractionLedger, atomic_json, file_sha256
from world_models.env.coop import TwoPlayerCoopEnv
from world_models.env.coordinator import TwoPlayerCoordinator
from world_models.env.splits import build_split_scenario, list_templates, load_split_spec
from world_models.infrastructure.config import resolve_config
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.policies.actor import LocalActor
from world_models.policies.config import PolicyConfig

from .evidence import source_identity, write_report


SCHEMA_VERSION = "p36-real-evaluation-v1"
_NATIVE_ENV, _NATIVE_COORDINATOR = TwoPlayerCoopEnv, TwoPlayerCoordinator
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


@torch.no_grad()
def actor_version(actor: LocalActor, *, origin_device: str | None = None) -> str:
    """P3.2-compatible version without importing its world-model package.

The P3.5 version includes device names. An explicit origin device verifies a
GPU export on CPU using the same bytes; no CUDA allocation is needed.
    """
    if type(actor) is not LocalActor:
        raise ValueError("only the declared local actor can be evaluated")
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
                or not torch.isfinite(value).all().item()):
            raise ValueError("actor state must be materialized, dense, and finite")
        append(_json_bytes({"kind": kind, "name": name, "dtype": str(value.dtype),
                            "shape": list(value.shape), "device": origin_device or str(value.device)}))
        append(value.detach().cpu().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _device(value):
    result = torch.device(value)
    if result.type not in ("cpu", "cuda") or result.type == "cpu" and result.index is not None:
        raise ValueError("evaluation device must be cpu or a logical CUDA device")
    if result.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA evaluation requested but CUDA is unavailable")
        result = torch.device("cuda", torch.cuda.current_device() if result.index is None else result.index)
        if result.index >= torch.cuda.device_count():
            raise ValueError("evaluation CUDA device is outside the visible allocation")
    return result


def _origin_device(value):
    device = torch.device(value)
    if str(device) != value or device.type not in ("cpu", "cuda") or (
            device.type == "cpu" and device.index is not None) or (
            device.type == "cuda" and device.index is None):
        raise ValueError("source_device must be cpu or an explicit cuda:N")
    return str(device)


@torch.inference_mode(False)
def load_actor_export(path, *, device="cpu", source_device=None):
    """Return ``(LocalActor, provenance)`` after receipt and exact-state checks."""
    candidate = Path(path)
    if any(item.is_symlink() for item in (candidate, *candidate.parents)):
        raise ValueError("actor export cannot traverse symbolic links")
    root = candidate.resolve()
    pending = (" Actor publication may still be pending even when PPO cycles have finished; "
               "inspect the run's status and evaluation_checkpoints.json."
               if root.parent.name == "evaluation" and root.name.startswith("real_") else "")
    if not root.is_dir():
        raise ValueError(f"local actor export directory does not exist: {root}.{pending}")
    # Refuse a full controller/model checkpoint before deserializing its tensors.
    # The generic loader verifies the same receipt and complete specifications
    # again, then checks the payload; no non-actor tensor is expected here.
    manifest_path = root / "manifest.json"
    receipt = root / "manifest.sha256"
    if not manifest_path.is_file():
        raise ValueError(f"local actor export is incomplete: missing manifest.json in {root}.{pending}")
    if not receipt.is_file():
        raise ValueError(f"local actor export is incomplete: missing manifest.sha256 in {root}.{pending}")
    if receipt.read_text(encoding="ascii").strip() != f"{file_sha256(manifest_path)}  manifest.json":
        raise ValueError(f"local actor export manifest receipt mismatch: {root}; preserve the export and verify its transfer")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("metadata", {}).get("schema_version") != "p35-local-actor-v1"
            or not manifest.get("tensor_specs") or any(
                not entry.get("path") or entry["path"][0] != ["key", "actor"]
                for entry in manifest["tensor_specs"])):
        raise ValueError("evaluation requires an actor-only export, never a full training checkpoint")
    state, metadata = load_cycle_checkpoint(root, device="cpu")
    if set(state) != {"actor"} or type(state["actor"]) is not dict:
        raise ValueError("evaluation requires an actor-only export, never a full training checkpoint")
    if set(metadata) != _EXPORT_KEYS or metadata.get("schema_version") != "p35-local-actor-v1":
        raise ValueError("unsupported local actor export metadata")
    if (metadata["task_family"] not in ("relay", "cover") or metadata["variant"] not in ("B", "D")
            or metadata["evaluation_split"] not in ("validation", "test")
            or metadata["evaluation_executed"] is not False or metadata["phase3_acceptance"] is not False):
        raise ValueError("invalid local actor export identity or execution markers")
    for name in ("run_seed", "real_joint_transitions", "actual_initial_replay_interactions", "model_updates",
                 "local_ppo_cycles", "actor_optimizer_steps", "critic_optimizer_steps"):
        if type(metadata[name]) is not int or metadata[name] < 0:
            raise ValueError("actor export counters must be nonnegative integers")
    for name in ("untrained_zero_interaction_baseline", "on_registered_evaluation_grid"):
        if type(metadata[name]) is not bool:
            raise ValueError("actor export flags must be boolean")
    if metadata["untrained_zero_interaction_baseline"] and any(metadata[key] for key in (
            "real_joint_transitions", "local_ppo_cycles", "actor_optimizer_steps", "critic_optimizer_steps")):
        raise ValueError("untrained baseline export contains learned policy counters")
    if (metadata["local_ppo_cycles"] == 0) != (metadata["actor_optimizer_steps"] == 0):
        raise ValueError("actor export policy counters disagree")
    if metadata["actor_optimizer_steps"] != metadata["critic_optimizer_steps"]:
        raise ValueError("actor export actor/critic optimizer counters disagree")
    for name in ("actor_version", "world_model_version"):
        if (type(metadata[name]) is not str or len(metadata[name]) != 64
                or any(ch not in "0123456789abcdef" for ch in metadata[name])):
            raise ValueError("actor export versions must be SHA256 identifiers")
    values = metadata["actor_config"]
    if type(values) is not dict or set(values) != {field.name for field in fields(PolicyConfig)}:
        raise ValueError("actor export must contain its complete policy configuration")
    values = dict(values)
    for name in ("image_shape", "encoder_channels", "critic_hidden_sizes"):
        if type(values[name]) is not list:
            raise ValueError("actor config dimensions must use JSON arrays")
        values[name] = tuple(values[name])
    config = PolicyConfig(**values)
    with torch.random.fork_rng(devices=[]):
        actor = LocalActor(config)
    if set(state["actor"]) != set(actor.state_dict()) or any(
            not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or tuple(value.shape) != tuple(actor.state_dict()[name].shape)
            for name, value in state["actor"].items()):
        raise ValueError("actor export tensor names, shapes, or FP32 dtypes do not match its architecture")
    actor.load_state_dict(state["actor"], strict=True)
    run_ref = None
    candidates = []
    if source_device is not None:
        candidates.append(_origin_device(source_device))
    run_path = root.parent.parent / "run_manifest.json"
    if root.parent.name == "evaluation" and run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        settings = run.get("settings", {})
        expected_config = PolicyConfig(
            image_shape=tuple(settings["environment"]["image_shape"]),
            encoder_channels=tuple(settings["world_model"]["encoder_channels"]),
            embedding_size=settings["actor"]["embedding_size"], hidden_size=settings["actor"]["hidden_size"],
            action_count=settings["environment"]["action_count"],
            world_tokens=settings["world_model"]["memory_tokens"] + settings["world_model"]["latent_variables"],
            world_width=settings["world_model"]["width"], critic_hidden_sizes=tuple(settings["critic"]["hidden_sizes"]),
            value_huber_delta=settings["mappo"]["value_huber_delta"])
        if (run.get("schema_version") != "p35-run-v1" or run.get("master_seed") != metadata["run_seed"]
                or run["settings"]["task"]["name"] != metadata["task_family"]
                or run["settings"]["variant"]["name"] != metadata["variant"]
                or expected_config != config
                or settings["experiment"]["initial_transitions"] != metadata["actual_initial_replay_interactions"]):
            raise ValueError("adjacent run manifest disagrees with the actor export")
        current = resolve_config(metadata["task_family"], metadata["variant"], "pilot").settings
        if settings["task"] != current["task"] or settings["environment"] != current["environment"]:
            raise ValueError("evaluation task rules or actor observation protocol changed since training")
        candidates.append(_origin_device(run["device"]))
        run_ref = {"path": str(run_path), "sha256": file_sha256(run_path), "run_identity": canonical_hash(run)}
    candidates.extend(["cpu", str(device), "cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    verified_device = None
    for candidate in dict.fromkeys(candidates):
        if actor_version(actor, origin_device=candidate) == metadata["actor_version"]:
            verified_device = candidate
            break
    if verified_device is None or source_device is not None and verified_device != source_device:
        raise ValueError("actor tensors do not match their recorded version; provide the original --source-device for other GPU indices")
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["payload"]
    provenance = {"path": str(root), "manifest_sha256": file_sha256(root / "manifest.json"),
                  "payload_sha256": payload["sha256"], "metadata": metadata,
                  "verified_origin_device": verified_device, "portable_weights_sha256": actor_version(actor, origin_device="cpu"),
                  "run_manifest": run_ref, "run_identity": None if run_ref is None else run_ref["run_identity"]}
    return actor.to(_device(device)), provenance


def validation_bank(task_family, *, episodes=8, bank_seed=3600):
    """Balanced held-out geometry; every method/training seed gets this same bank."""
    if type(episodes) is not int or episodes < 2 or episodes % 2:
        raise ValueError("validation needs an even positive episode count covering both layouts")
    if type(bank_seed) is not int or bank_seed < 0:
        raise ValueError("bank seed must be a nonnegative integer")
    templates = list_templates(task_family, "validation")
    if len(templates) != 2:
        raise ValueError("the frozen validation bank must have two layouts")
    rows = []
    for index in range(episodes):
        def seed(component):
            return int(canonical_hash({"namespace": "p36-validation-bank-v1", "task": task_family,
                                       "bank_seed": bank_seed, "episode": index, "component": component})[:8], 16)
        rows.append({"index": index, "template_index": index % 2, "template_id": templates[index % 2]["id"],
                     "layout_seed": seed("layout"), "engine_seed": seed("engine"),
                     "action_seeds": [seed("actor0"), seed("actor1")]})
    return {"schema_version": "p36-validation-bank-v1", "task_family": task_family,
            "split": "validation", "bank_seed": bank_seed, "split_manifest_sha256": canonical_hash(load_split_spec()),
            "episodes": rows}


class EvaluationLedger(TrainingInteractionLedger):
    """The existing fsynced physics accounting, explicitly outside learning data."""
    def __init__(self, directory, *, task_family, run_seed):
        self.directory, self.path = Path(directory), Path(directory) / "evaluation_interactions.jsonl"
        if self.path.exists():
            raise FileExistsError("evaluation ledger already exists")
        self.learning_eligible = False
        self.identity = {"task_family": task_family, "run_seed": run_seed}
        self._physical = self._initialization = self._published = self._published_tics = 0
        self._failed = self._committed = self._sequence = 0
        self._kinds, self._seen = {"local_actor_evaluation": 0}, {}
        self._append("created", learning_eligible=False, data_source="evaluation_environment", purpose="local_actor_validation")

    def _append(self, event, **fields):
        if "replay_status" in fields:
            fields["evaluation_status"] = fields.pop("replay_status")
        if event == "episode_committed":
            event = "evaluation_episode_completed"
        return super()._append(event, **fields)

    def manifest(self):
        result = super().manifest()
        result.update(purpose="local_actor_validation", ledger_path=self.path.name,
                      training_interactions_added=0, data_source="evaluation_environment")
        result["completed_episode_joint_transitions"] = result.pop("replay_committed_joint_transitions")
        result["accounting_rule"] = "Evaluation only: physical dispatches = published decisions + reset synchronization + failed dispatches; none enters training replay."
        return result


def _append_jsonl(path, row):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _metrics(rows, family):
    count = len(rows)
    events = Counter()
    for row in rows:
        events.update(row["event_counts"])
    progress = (events["latch_activations"] if family == "relay" else
                events["console_1_completions"] + events["console_2_completions"])
    decisions = sum(row["length"] for row in rows)
    return {"episodes": count, "successes": sum(row["success"] for row in rows),
            "success_rate": sum(row["success"] for row in rows) / count if count else None,
            "mean_return": sum(row["return"] for row in rows) / count if count else None,
            "mean_length": decisions / count if count else None,
            "mean_action_entropy": sum(row["mean_action_entropy"] * row["length"] for row in rows) / decisions if decisions else None,
            "event_totals": dict(events), "mean_event_progress": progress / count if count else None}


@torch.inference_mode(False)
def evaluate_actor_export(actor_export, output_dir, *, episodes=8, bank_seed=3600, device="cpu",
                          mode="stochastic", precision="float32", source_device=None,
                          env_factory=None, scenario_builder=None):
    """Run complete validation episodes. Native engine execution is opt-in by call.

Eight episodes are a small diagnostic, not the registered 50-episode checkpoint
or 200-episode final score. Test layouts are unavailable through this API.
    """
    if mode not in ("stochastic", "greedy") or precision not in ("float32", "bfloat16"):
        raise ValueError("evaluation needs stochastic or greedy mode and float32 or bfloat16 precision")
    actor, export = load_actor_export(actor_export, device=device, source_device=source_device)
    meta, device = export["metadata"], next(actor.parameters()).device
    if precision == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("selected GPU does not support bfloat16 evaluation")
    bank = validation_bank(meta["task_family"], episodes=episodes, bank_seed=bank_seed)
    settings = resolve_config(meta["task_family"], meta["variant"], "pilot").settings
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError("evaluation output must be fresh; never overwrite consumed interactions")
    export["path_at_evaluation"] = export["path"]
    export["path"] = os.path.relpath(export["path"], root).replace(os.sep, "/")
    if export["run_manifest"] is not None:
        reference = export["run_manifest"]
        reference["path_at_evaluation"] = reference["path"]
        reference["path"] = os.path.relpath(reference["path"], root).replace(os.sep, "/")
    root.mkdir(parents=True)
    sources = source_identity()
    protocol = {"split": "validation", "episodes": episodes, "bank_seed": bank_seed,
                "bank_sha256": canonical_hash(bank), "mode": mode, "temperature": 1.0 if mode == "stochastic" else None,
                "precision": precision, "frame_skip": settings["environment"]["frame_skip"],
                "deadline": settings["task"]["deadline"], "evaluation_data_enters_training": False}
    atomic_json(root / "bank.json", bank)
    ledger = EvaluationLedger(root, task_family=meta["task_family"], run_seed=meta["run_seed"])
    native = env_factory is None and scenario_builder is None
    create_env, build_scenario = env_factory or _NATIVE_ENV, scenario_builder or build_split_scenario
    env, current_episode, operation, records = None, "", "setup", []
    before = actor_version(actor)
    modes = [(module, module.training) for module in actor.modules()]
    started = time.perf_counter()
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "device": str(device),
               "vizdoom": None, "precision": precision,
               "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
               "cudnn_benchmark": torch.backends.cudnn.benchmark,
               "cpu_threads": torch.get_num_threads(), "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}

    def report(status, **extra):
        result = {"schema_version": SCHEMA_VERSION, "status": status, "task_family": meta["task_family"],
                  "run_seed": meta["run_seed"], "variant": meta["variant"], "protocol": protocol,
                  "native_backend": native, "native_engine": native, "runtime": runtime,
                  "actor_export": export, "actor_version_before": before, "actor_version_after": actor_version(actor),
                  "source_sha256": sources, "source_unchanged": source_identity() == sources,
                  "training_interactions_added": 0, "policy_optimizer_updates": 0, "real_ppo_updates": 0,
                  "world_model_loaded": False, "critic_loaded": False, "learning_eligible": False,
                  "phase3_acceptance": False, "episodes": records, "metrics": _metrics(records, meta["task_family"]),
                  "interaction_accounting": ledger.manifest(), "elapsed_seconds": time.perf_counter() - started,
                  "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "files": {name: {"path": name, "sha256": file_sha256(root / name)}
                            for name in ("bank.json", "episodes.jsonl", "transitions.jsonl", "evaluation_interactions.jsonl")
                            if (root / name).is_file()}, **extra}
        destination = root / "evaluation.json"
        if status == "failed" and destination.exists():
            destination = root / "evaluation_failure.json"
        write_report(destination, result)
        return result

    try:
        runtime["vizdoom"] = importlib.metadata.version("vizdoom")
        actor.eval()
        with torch.no_grad():
            for entry in bank["episodes"]:
                current_episode = f"{meta['task_family']}-validation-{entry['index']:06d}"
                operation = "scenario_build"
                scenario = build_scenario(meta["task_family"], "validation", entry["template_index"],
                                          seed=entry["layout_seed"], purpose="validation", frame_skip=protocol["frame_skip"],
                                          output_dir=root / "runtime" / "scenarios" / current_episode)
                provenance = scenario.get("split_provenance", {})
                if (provenance.get("split") != "validation" or provenance.get("purpose") != "validation"
                        or provenance.get("template_id") != entry["template_id"]
                        or provenance.get("learning_eligible") is not False):
                    raise ValueError("evaluation requires the selected held-out validation geometry")
                operation = "environment_create"
                env = create_env(meta["task_family"], seed=entry["layout_seed"], scenario_manifest=scenario,
                                 work_root=root / "runtime" / "environments" / current_episode,
                                 frame_skip=protocol["frame_skip"])
                native = native and type(env) is _NATIVE_ENV and type(env.coordinator) is _NATIVE_COORDINATOR
                operation = "reset"
                result = env.reset(seed=entry["engine_seed"])
                ledger.record_reset(env, current_episode)
                hidden, previous = actor.initial_hidden(2), torch.full((2,), 12, dtype=torch.long, device=device)
                streams = [torch.Generator(device=device).manual_seed(seed) for seed in entry["action_seeds"]]
                total_return, entropy_sum, counts, length = 0.0, 0.0, Counter(), 0
                while True:
                    operation = "actor_sampling"
                    images = torch.stack([torch.frombuffer(bytearray(image.data), dtype=torch.uint8).reshape(3, 96, 128)
                                          for image in result.observations]).to(device)
                    with torch.autocast(device.type, dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext():
                        local = actor(images, previous, hidden)
                    distribution = local.distribution
                    actions = tuple(int(distribution.probs[player].argmax().item()) if mode == "greedy" else
                                    int(torch.multinomial(distribution.probs[player], 1, generator=streams[player]).item())
                                    for player in (0, 1))
                    entropies = [float(value) for value in distribution.entropy().tolist()]
                    operation = "step"
                    step = env.step(*actions)
                    ledger.record_transition(env, current_episode, "local_actor_evaluation", step)
                    length += 1
                    if tuple(step.metadata.actions) != actions or step.metadata.decision_index != length - 1:
                        raise ValueError("evaluation executed-action or decision alignment changed")
                    if not math.isfinite(step.team_reward):
                        raise ValueError("evaluation reward is nonfinite")
                    if step.truncated or length > protocol["deadline"]:
                        raise ValueError("evaluation must finish whole native episodes, without collection cutoffs")
                    deltas = step.training_only_info["events"]["deltas"]
                    if any(type(value) is not int or value < 0 for value in deltas.values()):
                        raise ValueError("evaluation event deltas must be nonnegative integers")
                    counts.update(deltas)
                    total_return += step.team_reward
                    entropy_sum += sum(entropies) / 2
                    operation = "trace_write"
                    _append_jsonl(root / "transitions.jsonl", {
                        "episode_id": current_episode, "decision_index": length - 1, "actions": list(actions),
                        "previous_actions": previous.tolist(), "team_reward": step.team_reward,
                        "reward_components": step.training_only_info["reward_components"],
                        "event_deltas": deltas, "action_entropy": entropies, "terminated": step.terminated,
                        "terminal_reason": step.boundary.reason.value, "actual_tics": step.metadata.actual_simulator_tics})
                    hidden, previous, result = local.next_local_hidden, torch.tensor(actions, dtype=torch.long, device=device), step
                    if step.terminated:
                        break
                    if length == protocol["deadline"]:
                        raise ValueError("environment failed to terminate at its declared deadline")
                record = {**entry, "episode_id": current_episode, "length": length, "return": total_return,
                          "success": step.boundary.reason.value == "success", "terminal_reason": step.boundary.reason.value,
                          "event_counts": dict(counts), "mean_action_entropy": entropy_sum / length,
                          "scenario_sha256": scenario["hashes"]["wad_sha256"], "split_provenance": provenance}
                operation = "episode_publish"
                _append_jsonl(root / "episodes.jsonl", record)
                ledger.commit_episode(current_episode, length)
                records.append(record)
                operation = "environment_close"
                env.close(); env = None
        operation = "integrity_check"
        if actor_version(actor) != before or source_identity() != sources:
            raise RuntimeError("evaluation changed actor weights or implementation sources")
        if len(records) != episodes or ledger.manifest()["failed_dispatch_attempts"]:
            raise ValueError("evaluation did not complete its exact episode bank")
        return report("complete")
    except BaseException as error:
        try:
            ledger.record_failure(env, current_episode, operation, error)
            failure = {"unreconciled": False}
        except BaseException as ledger_error:
            failure = {"unreconciled": True, "ledger_error": repr(ledger_error)}
        if env is not None:
            try:
                env.close()
            except BaseException as close_error:
                failure["close_error"] = repr(close_error)
        report("failed", operation=operation, error=repr(error), traceback=traceback.format_exc(), **failure)
        raise
    finally:
        for module, mode in modes:
            module.training = mode
