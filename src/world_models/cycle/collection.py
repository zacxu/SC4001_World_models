"""Counted online real collection using only each player's local actor inputs.

The P2 replay codec predates branch collection and fixes its dataset ``variant``
field to ``shared_initial:B,D``. ``training_variant`` is the authoritative online
owner, as required by the existing refresh runner. This adapter changes neither
that frozen codec nor the durable ledger's conservation mechanics.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
from typing import Callable

import torch

from world_models.data.ledger import TrainingInteractionLedger, atomic_json, file_sha256
from world_models.data.replay import ReplayDataset, ReplayDatasetWriter
from world_models.env.coop import TwoPlayerCoopEnv
from world_models.env.splits import build_split_scenario, list_templates
from world_models.imagination.versions import module_version
from world_models.infrastructure.config import resolve_config
from world_models.infrastructure.reproducibility import canonical_hash, derive_seed
from world_models.policies import LocalActor, PolicyConfig


COLLECTION_VERSION = "online-local-real-replay-v1"
COLLECTOR_KIND = "online_local_policy"


class OnlineInteractionLedger(TrainingInteractionLedger):
    """Reuse fsynced dispatch accounting with an explicit online policy kind."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kinds = {COLLECTOR_KIND: 0}

    def _append(self, event, **fields):
        if event == "created":
            fields["purpose"] = "online_local_learning_replay"
        return super()._append(event, **fields)

    def manifest(self):
        result = super().manifest()
        result["purpose"] = "online_local_learning_replay"
        result["accounting_rule"] = (
            "Each online local-policy joint decision is charged once before replay storage. "
            "Failed non-initialization dispatches remain charged; reset synchronization "
            "dispatches are reported separately as physical overhead.")
        return result


class OnlineReplayDatasetWriter(ReplayDatasetWriter):
    """Preserve P2 typed storage, with correctly labelled online episode metadata.

    The inherited start validator accepts two initial collector labels. A local
    compatibility argument passes that historical gate; before any partial
    manifest or transition is written, ``_checkpoint`` restores the real online
    collector and branch. All other provenance/state validations are inherited.
    """

    def __init__(self, path, metadata, **kwargs):
        if metadata.get("training_variant") not in ("B", "D"):
            raise ValueError("online replay needs its explicit training variant")
        self.training_variant = metadata["training_variant"]
        super().__init__(path, metadata, **kwargs)

    def _checkpoint(self):
        if self._active is not None:
            self._active["metadata"].update(
                collector_kind=COLLECTOR_KIND, variant=self.training_variant,
                training_variant=self.training_variant, branch_data_shared=False)
        super()._checkpoint()

    def start_episode(self, metadata, reset):
        if metadata.get("collector_kind") != COLLECTOR_KIND:
            raise ValueError("online writer accepts only local-policy real episodes")
        compatibility = deepcopy(metadata)
        compatibility["collector_kind"] = "random"
        super().start_episode(compatibility, reset)


def _episode_seed(root: int, episode: int, component: str) -> int:
    return int(canonical_hash({"root": root, "episode": episode, "component": component})[:8], 16)


@torch.inference_mode(False)
def collect_policy_block(
    task_family: str, run_seed: int, variant: str, output_dir: str | Path, *,
    actor: LocalActor, settings: dict, block_index: int,
    total_transitions: int | None = None,
    env_factory: Callable | None = None, scenario_builder: Callable | None = None,
    writer_factory: Callable | None = None, chunk_transitions: int = 128,
    compression: str = "deflate", compression_level: int = 1,
    resource_guard: Callable | None = None, progress_callback: Callable | None = None,
) -> dict:
    """Seal one exact configured online block; interrupted outputs are retained.

    ``block_index`` is zero based. Collection uses stochastic temperature-one
    local policies; real replay supplies model fitting and anchors, never a PPO
    loss. Engine calls occur only after the output and fsynced ledger exist.
    """
    expected_policy = PolicyConfig.from_experiment(settings)
    if (not isinstance(actor, LocalActor) or actor.config != expected_policy
            or settings["task"]["name"] != task_family
            or settings["variant"]["name"] != variant or variant not in ("B", "D")):
        raise ValueError("task, branch, and local actor must match the collection settings")
    experiment = settings["experiment"]
    # TwoPlayerCoopEnv resolves its task rules from the repository configuration.
    # Do not let a caller label transitions with a different reward or deadline.
    if settings["task"] != resolve_config(task_family, variant, experiment["name"]).settings["task"]:
        raise ValueError("collection task rules differ from the actual environment configuration")
    if type(run_seed) is not int or run_seed not in experiment["master_seeds"]:
        raise ValueError("run seed must belong to the selected experiment")
    total = experiment["collection_block"] if total_transitions is None else total_transitions
    if type(total) is not int or total != experiment["collection_block"]:
        raise ValueError("online collection must consume exactly one configured collection block")
    blocks = (experiment["real_budget"] - experiment["initial_transitions"]) // total
    if type(block_index) is not int or not 0 <= block_index < blocks:
        raise ValueError("online block index is outside the configured zero-based schedule")
    if resource_guard is not None and not callable(resource_guard):
        raise ValueError("resource guard must be callable")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress callback must be callable")
    if type(chunk_transitions) is not int or chunk_transitions < 1:
        raise ValueError("chunk_transitions must be positive")
    if compression not in ("stored", "deflate") or type(compression_level) is not int or not 0 <= compression_level <= 9:
        raise ValueError("invalid replay compression")
    tensors = tuple(actor.parameters()) + tuple(actor.buffers())
    device = next(actor.parameters()).device
    if (device.type not in ("cpu", "cuda") or any(t.device != device for t in tensors)
            or any(t.is_floating_point() and t.dtype != torch.float32 for t in tensors)):
        raise ValueError("online actor must use FP32 weights on one CPU or CUDA device")
    templates = list_templates(task_family, "train")
    if len(templates) != 6:
        raise ValueError("online collection requires the six approved training templates")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("preserve prior block accounting; use a fresh collection directory")
    actor_version = module_version(actor)
    root_seed = derive_seed(task_family, run_seed, f"real_collection:block:{block_index}", variant)
    action_seeds = tuple(derive_seed(task_family, run_seed,
                                    f"local_action_sampling:real_block:{block_index}", variant, player)
                         for player in (0, 1))
    if len(set((root_seed, *action_seeds))) != 3:
        raise ValueError("online collection seed namespace collision")
    streams = tuple(torch.Generator(device=device).manual_seed(seed) for seed in action_seeds)
    metadata = {
        "collector_version": COLLECTION_VERSION, "task_family": task_family,
        "run_seed": run_seed, "profile": experiment["name"], "training_variant": variant,
        "collection_scope": COLLECTOR_KIND, "branch_data_shared": False,
        "data_source": "real_environment", "learning_eligible": True,
        "budget_scope": "approved_online_block", "target_transitions": total,
        "collection_block": block_index, "collection_seed": root_seed,
        "actor_version": actor_version, "actor_action_seeds": list(action_seeds),
        "action_sampling": "independent per-player categorical; temperature 1.0",
        "actor_inputs": ["own_uint8_rgb", "own_executed_previous_action", "own_recurrent_memory"],
        "actor_memory_at_reset": "zero; previous_action=START(12)",
        "setting_sha256": canonical_hash(settings),
        "partition_rule": "episode-id deterministic 90% fit / 10% model-diagnostic",
        "evaluation_layouts_used": False, "synthetic_ppo_data_imported": False,
        "environment_validation_data_imported": False, "real_ppo_updates": 0,
        "codec_variant_note": "P2 codec fixes dataset variant/paired_variants; training_variant identifies the sole online owner",
    }
    output.mkdir(parents=True)
    atomic_json(output / "collection_spec.json", metadata)
    ledger = OnlineInteractionLedger(output, task_family=task_family, run_seed=run_seed, learning_eligible=True)
    writer, env = None, None
    create_env = env_factory or TwoPlayerCoopEnv
    build_scenario = scenario_builder or build_split_scenario
    published = episodes = 0
    current_episode, operation = "", "setup"
    template_counts, terminals = Counter(), Counter()
    started = time.perf_counter()
    modes = [(module, module.training) for module in actor.modules()]

    def progress(status="collecting"):
        record = {
            "status": status, "task_family": task_family, "run_seed": run_seed,
            "variant": variant, "profile": experiment["name"], "block_index": block_index,
            "target_transitions": total, "published_joint_transitions": published,
            "completed_episodes": episodes, "interaction_accounting": ledger.manifest(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(output / "collection_progress.json", record)
        if progress_callback is not None:
            progress_callback(deepcopy(record))
        return record

    try:
        operation = "storage_create"
        writer = (writer_factory or OnlineReplayDatasetWriter)(
            output / "replay", metadata, chunk_transitions=chunk_transitions,
            compression=compression, compression_level=compression_level)
        engine_version = importlib.metadata.version("vizdoom")
        actor.eval()
        with torch.no_grad():
            while published < total:
                current_episode = f"{task_family}-{variant}-s{run_seed}-b{block_index:06d}-e{episodes:06d}"
                layout_seed = _episode_seed(root_seed, episodes, "layout")
                engine_seed = _episode_seed(root_seed, episodes, "engine")
                operation = "scenario_build"
                scenario = build_scenario(
                    task_family, "train", (block_index + episodes) % 6, seed=layout_seed,
                    purpose="training", frame_skip=settings["environment"]["frame_skip"],
                    output_dir=output / "runtime" / "scenarios" / current_episode)
                provenance = scenario.get("split_provenance", {})
                if (provenance.get("split") != "train" or provenance.get("purpose") != "training"
                        or provenance.get("template_id") != templates[(block_index + episodes) % 6]["id"]
                        or provenance.get("learning_eligible") is not True):
                    raise ValueError("online learning collection requires approved training geometry")
                operation = "environment_create"
                env = create_env(task_family, seed=layout_seed, scenario_manifest=scenario,
                                 work_root=output / "runtime" / "environments" / current_episode,
                                 frame_skip=settings["environment"]["frame_skip"])
                operation = "resource_enforcement"
                if resource_guard is not None:
                    resource_guard()
                operation = "reset"
                result = env.reset(seed=engine_seed)
                ledger.record_reset(env, current_episode)
                operation = "storage_start_episode"
                writer.start_episode({
                    "episode_id": current_episode, "task_family": task_family, "run_seed": run_seed,
                    "variant": variant, "training_variant": variant, "split": "train",
                    "layout_id": provenance["template_id"], "collector_kind": COLLECTOR_KIND,
                    "policy_version": actor_version, "collector_seed": root_seed,
                    "collection_block": block_index, "environment_seed": engine_seed,
                    "engine_version": engine_version, "scenario_hash": scenario["hashes"]["wad_sha256"],
                    "initial_tic": result.training_only_info["engine_tic"],
                }, result)
                hidden = actor.initial_hidden(2)
                previous = torch.full((2,), 12, dtype=torch.long, device=device)
                episode_length = 0
                while published < total:
                    # No clock, reward, privileged target, or other player's RGB
                    # enters this call. Separate rows contain independent memory.
                    operation = "actor_sampling"
                    images = torch.stack([
                        torch.frombuffer(bytearray(image.data), dtype=torch.uint8).reshape(3, 96, 128)
                        for image in result.observations]).to(device)
                    local = actor(images, previous, hidden)
                    actions = tuple(int(torch.multinomial(local.distribution.probs[player], 1,
                                                        generator=streams[player]).item()) for player in (0, 1))
                    operation = "resource_enforcement"
                    if resource_guard is not None:
                        resource_guard()
                    operation = "step"
                    step = env.step(*actions, collection_cutoff=published + 1 == total)
                    ledger.record_transition(env, current_episode, COLLECTOR_KIND, step)
                    published += 1
                    episode_length += 1
                    operation = "storage_append"
                    writer.append(step)
                    if tuple(step.metadata.actions) != actions:
                        raise ValueError("published executed actions differ from local policy samples")
                    hidden = local.next_local_hidden
                    previous = torch.tensor(step.metadata.actions, dtype=torch.long, device=device)
                    result = step
                    if published % 1000 == 0:
                        progress()
                    if step.terminated or step.truncated:
                        break
                if not result.terminated and not result.truncated:
                    raise AssertionError("online block ended without a real terminal or external cutoff")
                operation = "storage_finish_episode"
                episode = writer.finish_episode()
                if episode["valid_length"] != episode_length:
                    raise ValueError("replay episode length differs from charged interactions")
                ledger.commit_episode(current_episode, episode_length)
                terminals[result.boundary.reason.value] += 1
                template_counts[provenance["template_id"]] += episode_length
                episodes += 1
                operation = "environment_close"
                env.close()
                env = None
                progress()
        operation = "actor_version_check"
        actor_after = module_version(actor)
        if actor_after != actor_version:
            raise RuntimeError("local actor changed during frozen real collection")
        accounting = ledger.manifest()
        if (published != total or accounting["published_real_joint_transitions"] != total
                or accounting["replay_committed_joint_transitions"] != total
                or accounting["failed_dispatch_attempts"]):
            raise ValueError("online replay does not conserve the exact physical block budget")
        operation = "seal"
        manifest = writer.seal()
        reader = ReplayDataset(output / "replay", verify=True)
        if (reader.manifest["metadata"].get("training_variant") != variant
                or reader.manifest["transition_count"] != total):
            raise ValueError("sealed online replay branch or count differs from its approved block")
        report = progress("complete_online_replay")
        report.update(
            dataset_path="replay", dataset_manifest_sha256=file_sha256(output / "replay" / "manifest.json"),
            dataset_identity_sha256=manifest["dataset_identity_sha256"],
            collection_spec_sha256=file_sha256(output / "collection_spec.json"),
            actor_version_before=actor_version, actor_version_after=actor_after,
            data_source="real_environment", learning_eligible=True, training_variant=variant,
            collector_counts={COLLECTOR_KIND: published}, terminal_counts=dict(terminals),
            template_transitions=dict(template_counts), real_ppo_updates=0,
            synthetic_imported=False, diagnostic_imported=False,
            policy_optimizer_updates=0, world_model_optimizer_updates=0)
        atomic_json(output / "collection_report.json", report)
        return report
    except BaseException as exc:
        failure = {"status": "failed", "task_family": task_family, "run_seed": run_seed,
                   "variant": variant, "block_index": block_index, "operation": operation,
                   "episode_id": current_episode, "error": repr(exc),
                   "published_joint_transitions": published, "completed_episodes": episodes,
                   "traceback": traceback.format_exc(),
                   "retry_rule": "No automatic retry or overwrite; preserve all consumed-interaction accounting and partial replay."}
        try:
            ledger.record_failure(env, current_episode, operation, exc)
            failure.update(interaction_accounting=ledger.manifest(), unreconciled=False)
        except BaseException as ledger_error:
            failure.update(ledger_error=repr(ledger_error), unreconciled=True)
        if env is not None:
            try:
                env.close()
            except BaseException as close_error:
                failure["runtime_close_error"] = repr(close_error)
        atomic_json(output / "collection_error.json", failure)
        raise
    finally:
        for module, mode in modes:
            module.training = mode
