"""Explicit continuation after a fully committed, zero-physics startup failure.

The caller names the expected failure hash and a fresh recovery ID/hot root.
All original payloads/metadata are verified and retained; physics failures and
active partial episodes require a separate budget/recovery decision.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import ast
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback
from typing import Callable

if __name__ == "__main__":
    for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_thread_variable] = "1"

import numpy as np

from world_models.data.collection import (
    CorrelatedRandomPolicy, ScriptedExplorationPolicy, _stream_seed, mixture_segments,
)
from world_models.data.ledger import TrainingInteractionLedger, atomic_json, file_sha256
from world_models.data.replay import ReplayDataset, ReplayDatasetWriter
from world_models.data.spool import archive_closed_file, archive_closed_tree, verify_archive_tree
from world_models.env.coop import TwoPlayerCoopEnv
from world_models.env.coordinator import CoordinatorError
from world_models.env.splits import build_split_scenario, list_templates
from world_models.infrastructure.config import resolve_config, validate_pair
from world_models.infrastructure.reproducibility import canonical_hash, derive_seed


ROOT = Path(__file__).resolve().parents[3]
RECOVERY_VERSION = "verified-initial-replay-continuation-v1"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _snapshot_file(source, destination):
    """Exclusive durable evidence copy within the dataset; never deletes source."""
    source, destination = Path(source), Path(destination)
    before = source.stat()
    expected = file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    after = source.stat()
    if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or destination.stat().st_size != after.st_size
            or file_sha256(destination) != expected or file_sha256(source) != expected):
        raise ValueError("original recovery evidence changed during durable snapshot")
    return {"sha256": expected, "bytes": after.st_size}


def _inside(root, relative):
    encoded = str(relative).replace("\\", "/")
    path = Path(encoded)
    result = (Path(root) / path).resolve()
    if path.is_absolute() or ".." in path.parts or not result.is_relative_to(Path(root).resolve()):
        raise ValueError("recovery evidence path escapes its original dataset")
    return result


def _remaining_segments(total, scripted_fraction, block, published):
    result, expected, cursor = [], Counter(), 0
    for index, (kind, length) in enumerate(mixture_segments(total, scripted_fraction, block)):
        used = max(0, min(length, published - cursor))
        expected[kind] += used
        if used < length:
            result.append((index, kind, length - used))
        cursor += length
    if not 0 <= published < total:
        raise ValueError("recovery needs a nonempty remaining initial budget")
    return result, dict(expected)


def _origin_guard(metadata, origin_project_root):
    origin = metadata.get("origin_provenance", {})
    sources = origin.get("execution_source_sha256", {})
    if not sources:
        if metadata.get("learning_eligible"):
            raise ValueError("learning-eligible continuation needs original execution provenance")
        return {"origin_source_check": "injected preflight fixture"}
    if origin_project_root is None:
        raise ValueError("name the untouched original project root for continuation provenance")
    original = Path(origin_project_root).resolve()
    for relative, expected in sources.items():
        if file_sha256(_inside(original, relative)) != expected:
            raise ValueError(f"original execution source changed: {relative}")
    from world_models.data import collection
    tree = ast.parse((original / "src/world_models/data/collection.py").read_text(encoding="utf-8"))
    definitions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    names = ("_stream_seed", "mixture_segments", "CorrelatedRandomPolicy", "ScriptedExplorationPolicy")
    for name in names:
        loaded = ast.parse(inspect.getsource(getattr(collection, name))).body[0]
        if ast.dump(loaded, include_attributes=False) != ast.dump(definitions[name], include_attributes=False):
            raise ValueError("continuation changed a collection RNG/action policy")
    # The named original root is evidence, not permission to run different
    # currently loaded native/game code. Verify the actual loaded generators and
    # their authored maps/configs too, while allowing only declared data-module
    # recovery/reader overlays.
    critical = {
        "world_models.contracts": "src/world_models/contracts.py",
        "world_models.infrastructure.config": "src/world_models/infrastructure/config.py",
        "world_models.infrastructure.reproducibility": "src/world_models/infrastructure/reproducibility.py",
        "world_models.data.schema": "src/world_models/data/schema.py",
    }
    critical.update({"world_models.env." + name: "src/world_models/env/" + name + ".py"
        for name in ("actions", "observations", "coop", "coordinator", "workers", "scripted", "splits", "scenario_builder", "targets", "provenance")})
    for module_name, relative in critical.items():
        loaded = importlib.import_module(module_name)
        if file_sha256(loaded.__file__) != sources.get(relative):
            raise ValueError("loaded native generation module differs from original: " + module_name)
    from world_models.env.scenario_builder import ROOT as generation_root
    authored = [relative for relative in sources if relative.startswith(("configs/", "scenarios/"))]
    for relative in authored:
        if file_sha256(_inside(generation_root, relative)) != sources[relative]:
            raise ValueError("currently loaded authored game/config inputs differ: " + relative)
    from world_models.env.provenance import runtime_manifest
    actual = runtime_manifest(original)
    actual.pop("recorded_at_utc", None)
    expected = origin.get("runtime")
    relocation = None
    if expected:
        # An isolated recovery checkout can locate the byte-identical ACC
        # executable at another path. Native SDK/runtime/assets stay identical.
        if actual["compiler"]["path"] != expected["compiler"]["path"]:
            if (actual["compiler"]["sha256"], actual["compiler"]["version"]) != (
                    expected["compiler"]["sha256"], expected["compiler"]["version"]):
                raise ValueError("continuation changed the pinned compiler bytes/version")
            relocation = {"original_path": expected["compiler"]["path"],
                          "recovery_path": actual["compiler"]["path"],
                          "same_sha256": actual["compiler"]["sha256"]}
            actual["compiler"]["path"] = expected["compiler"]["path"]
        if actual != expected:
            raise ValueError("continuation changed the original native SDK/runtime/assets")
    return {"origin_project_root": str(original), "original_execution_sources_verified": len(sources),
            "actual_loaded_generation_modules_verified": len(critical),
            "actual_authored_generation_inputs_verified": len(authored),
            "unchanged_generation_definitions": list(names), "identical_compiler_relocation": relocation}


def inspect_recoverable_prefix(output_dir, *, expected_failure_sha256,
                               failure_report=None, origin_project_root=None,
                               require_native_failure_journal=True):
    """Read/verify only; no prefix bytes or timestamps are rewritten."""
    output = Path(output_dir).resolve()
    failure_path = Path(failure_report).resolve() if failure_report else output / "collection_error.json"
    if not failure_path.is_relative_to(output):
        raise ValueError("failure report must be retained inside the dataset")
    if file_sha256(failure_path) != expected_failure_sha256:
        raise ValueError("failure report differs from the explicitly requested continuation")
    failure = _read(failure_path)
    manifest = _read(output / "replay/manifest.partial.json")
    metadata = manifest["metadata"]
    if (failure.get("status") != "failed" or failure.get("operation") != "reset"
            or failure.get("unreconciled") is not False or manifest.get("active_episode") is not None
            or (output / "replay/manifest.json").exists() or (output / "collection_report.json").exists()):
        raise ValueError("only an unsealed fully committed reset failure can be continued")
    original = _origin_guard(metadata, origin_project_root)
    accounting = failure["interaction_accounting"]
    count = manifest["transition_count"]
    if (failure["published_joint_transitions"] != count or accounting["published_real_joint_transitions"] != count
            or accounting["replay_committed_joint_transitions"] != count or accounting["failed_dispatch_attempts"]):
        raise ValueError("failed collection and committed prefix counts disagree")
    codec = manifest["storage"]
    writer = ReplayDatasetWriter(output / "replay", metadata, chunk_transitions=codec["chunk_transitions"],
        compression=codec["compression"], compression_level=codec["compression_level"], resume=True)
    if writer.transition_count != count:
        raise ValueError("verified writer prefix length changed")
    episodes = writer.manifest["episodes"]
    expected_id = f"{metadata['task_family']}-s{metadata['run_seed']}-e{len(episodes):06d}"
    if failure["completed_episodes"] != len(episodes) or failure["episode_id"] != expected_id:
        raise ValueError("failed startup would change the original episode/RNG sequence")
    ledger_path = output / "training_interactions.jsonl"
    if file_sha256(ledger_path) != accounting["ledger_sha256"]:
        raise ValueError("original cold interaction ledger changed")
    journal = failure["journal_storage"]
    hot_ledger = Path(journal["ledger_hot_path_at_collection"])
    if not hot_ledger.is_file() or file_sha256(hot_ledger) != accounting["ledger_sha256"]:
        raise ValueError("original hot/cold ledger mirrors must match")
    first = last = None
    with ledger_path.open(encoding="utf-8") as stream:
        for line in stream:
            last = json.loads(line)
            if first is None:
                first = last
    if (not last or last["event"] != "infrastructure_or_storage_error"
            or last["episode_id"] != expected_id or last.get("operation") != "reset"
            or any(last.get(key, 0) for key in ("physical_delta", "initialization_delta", "published_delta",
                "failed_delta", "physical_consumed_at_source", "initialization_consumed_at_source"))):
        raise ValueError("failed startup consumed physics or its failure ledger is incomplete")
    complete_ids = {episode["episode_id"] for episode in episodes}
    successful, failed_archives = [], []
    failed_wad_sha = None
    native_failure_checked = False
    for archive in journal["runtime_archives"]:
        destination = _inside(output, archive["archive_path"])
        verified = verify_archive_tree(destination, expected_receipt_sha256=archive["receipt_sha256"])
        if archive["episode_id"] in complete_ids:
            successful.append(deepcopy(archive))
        elif archive["episode_id"] == expected_id:
            failed_archives.append(deepcopy(archive))
            if archive["category"] == "scenarios":
                wads = [member["sha256"] for member in verified["files"] if member["path"].endswith(".wad")]
                if len(wads) == 1:
                    failed_wad_sha = wads[0]
            elif archive["category"] == "environments":
                paths = list(destination.rglob("dispatch_ledger.jsonl"))
                if require_native_failure_journal and len(paths) != 1:
                    raise ValueError("zero-physics native startup journal is missing")
                if paths:
                    rows = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
                    if (any(row.get("consumed_joint_decisions", 0) for row in rows)
                            or any(row.get("command") == "step" for row in rows)):
                        raise ValueError("failed native startup contains an unaccounted physics action")
                    native_failure_checked = True
        else:
            raise ValueError("failure archive contains unexplained episode identities")
    expected_categories = {(episode_id, category) for episode_id in complete_ids
                           for category in ("scenarios", "environments")}
    if (len(successful) != 2 * len(episodes)
            or {(entry["episode_id"], entry["category"]) for entry in successful} != expected_categories
            or len(failed_archives) != 2
            or {archive["category"] for archive in failed_archives} != {"scenarios", "environments"}):
        raise ValueError("completed and failed startup runtime archive inventories differ")
    if require_native_failure_journal and (not native_failure_checked or not failed_wad_sha):
        raise ValueError("zero-physics failed startup proof is incomplete")
    for index, episode in enumerate(episodes):
        if episode["episode_id"] != f"{metadata['task_family']}-s{metadata['run_seed']}-e{index:06d}":
            raise ValueError("original episode IDs changed the stable collector sequence")
    remaining, expected_mix = _remaining_segments(metadata["target_transitions"],
        metadata["collector_mixture"]["scripted"], metadata["mixture_block"], count)
    actual_mix = {kind: sum(ep["valid_length"] for ep in episodes if ep["collector_kind"] == kind)
                  for kind in ("random", "scripted")}
    if actual_mix != expected_mix or actual_mix != accounting["collector_counts"]:
        raise ValueError("prefix collector counts differ from the original mixture timeline")
    return {"output": output, "failure_path": failure_path, "failure": failure, "writer": writer,
        "manifest": manifest, "episodes": episodes, "metadata": metadata, "count": count,
        "remaining_segments": remaining, "successful_archives": successful,
        "failed_runtime_archives": failed_archives, "failed_wad_sha256": failed_wad_sha,
        "native_zero_physics_failure_checked": native_failure_checked, "original_generation_proof": original,
        "original_session_elapsed_seconds": (datetime.fromisoformat(last["recorded_at_utc"])
                                            - datetime.fromisoformat(first["recorded_at_utc"])).total_seconds(),
        "expected_next_episode_id": expected_id}


def _prefix_statistics(prefix):
    """Recompute coverage from stored targets rather than a progress snapshot."""
    root = prefix["output"] / "replay"
    fields = prefix["manifest"]["schema"]["event_fields"]
    stage_index = prefix["manifest"]["schema"]["task_fields"].index("stage")
    result = {name: Counter() for name in ("counts", "kind_episodes", "events", "terminals",
        "template_episodes", "template_transitions", "stages", "scripted_profiles")}
    for episode in prefix["episodes"]:
        kind, length, layout = episode["collector_kind"], episode["valid_length"], episode["layout_id"]
        result["counts"][kind] += length
        result["kind_episodes"][kind] += 1
        result["terminals"][episode["end_reason"]] += 1
        result["template_episodes"][layout] += 1
        result["template_transitions"][layout] += length
        if kind == "scripted":
            result["scripted_profiles"][episode["scripted_profile"]] += 1
        for chunk in episode["chunks"]:
            with np.load(_inside(root, chunk["path"]), allow_pickle=False) as arrays:
                result["events"].update(dict(zip(fields, arrays["event_deltas"].sum(axis=0).tolist())))
                local = arrays["next_observation_index"] - chunk["state_start"]
                if not arrays["task_validity_mask"][local, stage_index].all():
                    raise ValueError("prefix stage coverage has missing state targets")
                result["stages"].update(str(int(value)) for value in arrays["task_state"][local, stage_index])
    return result


def resume_initial_dataset(output_dir, *, journal_root, recovery_id, expected_failure_sha256,
                           failure_report=None, origin_project_root=None, max_startup_attempts=2,
                           journal_spool_max_bytes=512 * 1024 * 1024, progress_callback=None,
                           env_factory=None, scenario_builder=None, config_dir=None,
                           recovery_origin_provenance=None,
                           resource_guard: Callable[[], None] | None = None):
    """Continue the same seed/config sequence, retrying only zero-physics startups.

    Every prefix payload is verified before mutation. Original error, manifest,
    progress, and exact ledger snapshots survive any subsequent continuation.
    A physics-step failure never receives an automatic retry.
    A resource-guard failure is recorded separately and never retried as startup.
    """
    continuation_started = time.perf_counter()
    if (not isinstance(recovery_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", recovery_id)
            or type(max_startup_attempts) is not int or not 1 <= max_startup_attempts <= 2):
        raise ValueError("name a safe recovery ID and at most two fresh startup attempts")
    if type(journal_spool_max_bytes) is not int or journal_spool_max_bytes <= 0:
        raise ValueError("hot journal spool needs a positive finite limit")
    if resource_guard is not None and not callable(resource_guard):
        raise ValueError("resource guard must be callable")
    if recovery_origin_provenance is not None and not isinstance(recovery_origin_provenance, dict):
        raise ValueError("recovery implementation provenance must be a JSON dictionary")
    json.dumps(recovery_origin_provenance or {}, allow_nan=False)
    prefix = inspect_recoverable_prefix(output_dir,
        expected_failure_sha256=expected_failure_sha256, failure_report=failure_report,
        origin_project_root=origin_project_root, require_native_failure_journal=env_factory is None)
    output, metadata, writer = prefix["output"], prefix["metadata"], prefix["writer"]
    family, run_seed = metadata["task_family"], metadata["run_seed"]
    config = Path(config_dir or ROOT / "configs")
    left = resolve_config(family, "B", metadata["profile"], config)
    right = resolve_config(family, "D", metadata["profile"], config)
    validate_pair(left, right)
    settings = left.settings
    common = deepcopy(settings)
    common.pop("variant")
    if canonical_hash(common) != metadata["shared_configuration_sha256"]:
        raise ValueError("continuation configuration differs from the original shared protocol")
    initial_seed = derive_seed(family, run_seed, "initial_collection", "B")
    if (initial_seed != metadata["initial_collection_seed"] or not prefix["episodes"]
            or importlib.metadata.version("vizdoom") != prefix["episodes"][-1]["engine_version"]):
        raise ValueError("continuation changed the original seeds or native engine version")
    hot = Path(journal_root).resolve()
    if (hot == output or hot.is_relative_to(output) or output.is_relative_to(hot)
            or hot.exists() and any(hot.iterdir())):
        raise ValueError("continuation requires a fresh separate hot directory")
    probe = hot
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < journal_spool_max_bytes:
        raise OSError("insufficient free space for the bounded continuation spool")
    recovery_dir = output / "recovery" / recovery_id
    if recovery_dir.exists():
        raise FileExistsError("recovery ID already has retained evidence")
    recovery_dir.mkdir(parents=True)
    before = recovery_dir / "before"
    before.mkdir()
    snapshots = {}
    for source, name in (
            (output / "replay/manifest.partial.json", "manifest.partial.json"),
            (prefix["failure_path"], "collection_error.json"),
            (output / "collection_spec.json", "collection_spec.json"),
            (output / "collection_progress.json", "collection_progress.json"),
            (output / "training_interactions.jsonl", "training_interactions.jsonl")):
        if not source.is_file():
            continue
        receipt = _snapshot_file(source, before / name)
        snapshots[name] = {"path": (before / name).relative_to(output).as_posix(),
                           "sha256": receipt["sha256"], "bytes": receipt["bytes"]}
    prefix_files = []
    for episode in prefix["episodes"]:
        prefix_files.append({"path": "replay/" + episode["manifest_path"], "sha256": episode["manifest_sha256"]})
        prefix_files.extend({"path": "replay/" + chunk["path"], "sha256": chunk["sha256"]}
                            for chunk in episode["chunks"])
    lineage = {
        "schema_version": RECOVERY_VERSION, "recovery_id": recovery_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "explicit verified continuation after zero-physics native startup failure",
        "original_failure_sha256": expected_failure_sha256,
        "original_failure_report_path": prefix["failure_path"].relative_to(output).as_posix(),
        "prefix_joint_transitions": prefix["count"], "prefix_episodes": len(prefix["episodes"]),
        "original_metadata_sha256": canonical_hash(metadata), "snapshots": snapshots,
        "original_prefix_files": prefix_files, "original_generation_proof": prefix["original_generation_proof"],
        "recovery_module_sha256": file_sha256(__file__), "maximum_fresh_startup_attempts": max_startup_attempts,
        "recovery_origin_provenance": deepcopy(recovery_origin_provenance or {}),
        "step_failures_retried": False, "new_random_or_scripted_interactions_discarded": 0,
    }
    atomic_json(recovery_dir / "lineage.json", lineage)
    hot.mkdir(parents=True, exist_ok=True)
    ledger = TrainingInteractionLedger.resume_from_verified(hot / "ledger",
        source_path=output / "training_interactions.jsonl",
        expected_sha256=prefix["failure"]["interaction_accounting"]["ledger_sha256"],
        task_family=family, run_seed=run_seed, expected_committed=prefix["count"],
        learning_eligible=metadata["learning_eligible"],
        lineage={"path": (recovery_dir / "lineage.json").relative_to(output).as_posix(),
                 "sha256": file_sha256(recovery_dir / "lineage.json"), "recovery_id": recovery_id})
    stats = _prefix_statistics(prefix)
    for name in ("counts", "kind_episodes"):
        for kind in ("random", "scripted"):
            stats[name].setdefault(kind, 0)
    published, episodes = prefix["count"], len(prefix["episodes"])
    runtime_archives = deepcopy(prefix["successful_archives"])
    failed_archives = deepcopy(prefix["failed_runtime_archives"])
    peak_hot = 0
    ledger_archive = None
    env = None
    current_episode = ""
    operation = "continuation_setup"
    active_attempt = None
    started = continuation_started
    templates = list_templates(family, "train")
    create_env = env_factory or TwoPlayerCoopEnv
    build = scenario_builder or build_split_scenario
    old_progress = _read(before / "collection_progress.json")
    elapsed_before = max(old_progress.get("elapsed_seconds", 0.0), prefix["original_session_elapsed_seconds"])

    def measure():
        nonlocal peak_hot
        count = sum(path.stat().st_size for path in hot.rglob("*") if path.is_file())
        peak_hot = max(peak_hot, count)
        if count > journal_spool_max_bytes:
            raise OSError("bounded continuation journal spool exceeded")
        return count

    def archive_attempt(episode_id, attempt, *, failed=False):
        measure()
        destination_list = failed_archives if failed else runtime_archives
        for category in ("scenarios", "environments"):
            source = hot / "runtime" / f"attempt-{attempt:03d}" / category / episode_id
            if source.exists():
                destination = output / "runtime" / "recovery" / recovery_id / f"attempt-{attempt:03d}" / category / episode_id
                receipt = archive_closed_tree(source, destination, spool_root=hot)
                destination_list.append({"episode_id": episode_id, "category": category,
                    "archive_path": destination.relative_to(output).as_posix(),
                    "recovery_id": recovery_id, "startup_attempt": attempt, **receipt})

    def storage_info():
        previous = prefix["failure"]["journal_storage"]
        return {"spooled": True, "spool_root_at_collection": str(hot),
            "runtime_hot_root_at_collection": str(hot / "runtime"), "runtime_archive_root": "runtime",
            "ledger_hot_path_at_collection": str(ledger.path), "ledger_archive_path": "training_interactions.jsonl",
            "ledger_archive": deepcopy(ledger_archive), "runtime_archives": deepcopy(runtime_archives),
            "failed_runtime_archives": deepcopy(failed_archives),
            "maximum_hot_bytes": journal_spool_max_bytes,
            "peak_hot_bytes": max(peak_hot, previous["peak_hot_bytes"]), "archive_policy": previous["archive_policy"],
            "retained_prior_hot_ledger_copies": [{"source_path_at_collection": previous["ledger_hot_path_at_collection"],
                "matching_archive_path": snapshots["training_interactions.jsonl"]["path"],
                "sha256": snapshots["training_interactions.jsonl"]["sha256"],
                "bytes": snapshots["training_interactions.jsonl"]["bytes"]}]}

    def progress(status="recovering"):
        measure()
        continuation_elapsed = time.perf_counter() - started
        total_attempt_elapsed = elapsed_before + continuation_elapsed
        report = {"status": status, "task_family": family, "run_seed": run_seed, "profile": metadata["profile"],
            "target_transitions": metadata["target_transitions"], "budget_scope": metadata["budget_scope"],
            "learning_eligible": metadata["learning_eligible"], "published_joint_transitions": published,
            "completed_episodes": episodes, "collector_counts": dict(stats["counts"]),
            "event_counts": dict(stats["events"]), "terminal_counts": dict(stats["terminals"]),
            "elapsed_seconds": total_attempt_elapsed,
            "original_prefix_elapsed_seconds": old_progress.get("elapsed_seconds", 0.0),
            "original_failed_collection_attempt_elapsed_seconds": elapsed_before,
            "original_failed_startup_elapsed_seconds": max(0.0, elapsed_before - old_progress.get("elapsed_seconds", 0.0)),
            "continuation_elapsed_seconds": continuation_elapsed,
            "total_collection_attempt_elapsed_seconds": total_attempt_elapsed,
            "elapsed_accounting": "sum of original failed collection attempt and explicit continuation; waiting/debugging between attempts is excluded, reset/startup work within attempts is included",
            "interaction_accounting": ledger.manifest(), "journal_storage": storage_info(),
            "recovery_id": recovery_id,
            "recorded_zero_physics_startup_failures": len(failed_archives) // 2,
            "recovery_lineage": {"path": (recovery_dir / "lineage.json").relative_to(output).as_posix(),
                "sha256": file_sha256(recovery_dir / "lineage.json"), "recovery_id": recovery_id,
                "prefix_joint_transitions": prefix["count"], "prefix_episodes": len(prefix["episodes"])}}
        atomic_json(output / "collection_progress.json", report)
        if progress_callback:
            progress_callback(deepcopy(report))
        return report

    try:
        progress()
        for segment_index, kind, length in prefix["remaining_segments"]:
            segment_remaining = length
            while segment_remaining:
                current_episode = f"{family}-s{run_seed}-e{episodes:06d}"
                template_index = stats["kind_episodes"][kind] % len(templates)
                layout_seed = _stream_seed(initial_seed, current_episode, "layout-nuisance")
                engine_seed = _stream_seed(initial_seed, current_episode, "engine")
                policy_seed = _stream_seed(initial_seed, current_episode, "policy")
                for attempt in range(1, max_startup_attempts + 1):
                    active_attempt = attempt
                    ledger.note("startup_attempt", episode_id=current_episode, recovery_id=recovery_id,
                        startup_attempt=attempt, environment_seed=engine_seed, layout_seed=layout_seed,
                        collector_seed=policy_seed, template_index=template_index)
                    operation = "scenario_build"
                    scenario = build(family, "train", template_index, seed=layout_seed, purpose="training",
                        frame_skip=settings["environment"]["frame_skip"],
                        output_dir=hot / "runtime" / f"attempt-{attempt:03d}" / "scenarios" / current_episode)
                    provenance = scenario["split_provenance"]
                    if (provenance["split"] != "train" or provenance["purpose"] != "training"
                            or provenance["learning_eligible"] is not True):
                        raise ValueError("continuation selected a nontraining scenario")
                    if current_episode == prefix["expected_next_episode_id"] and prefix["failed_wad_sha256"]:
                        if scenario["hashes"]["wad_sha256"] != prefix["failed_wad_sha256"]:
                            raise ValueError("fresh startup changed the original failed scenario bytes")
                    operation = "environment_create"
                    env = create_env(family, seed=layout_seed, scenario_manifest=scenario,
                        work_root=hot / "runtime" / f"attempt-{attempt:03d}" / "environments" / current_episode,
                        frame_skip=settings["environment"]["frame_skip"])
                    operation = "resource_enforcement"
                    if resource_guard is not None:
                        resource_guard()
                    operation = "reset"
                    try:
                        reset = env.reset(seed=engine_seed)
                    except BaseException as startup_error:
                        zero = (env.coordinator.consumed_joint_decisions == 0
                                and env.initialization_joint_decisions == 0)
                        if not (zero and isinstance(startup_error, CoordinatorError)
                                and attempt < max_startup_attempts):
                            raise
                        ledger.record_failure(env, current_episode, "reset", startup_error)
                        env.close()
                        operation = "runtime_archive"
                        archive_attempt(current_episode, attempt, failed=True)
                        env = None
                        continue
                    ledger.record_reset(env, current_episode)
                    break
                policy = (CorrelatedRandomPolicy(policy_seed) if kind == "random"
                          else ScriptedExplorationPolicy(scenario, stats["kind_episodes"][kind] // len(templates)))
                policy_version = metadata["random_policy"]["version"] if kind == "random" else policy.version
                episode_metadata = {"episode_id": current_episode, "task_family": family,
                    "layout_id": provenance["template_id"], "split": "train",
                    "scenario_hash": scenario["hashes"]["wad_sha256"],
                    "engine_version": prefix["episodes"][-1]["engine_version"],
                    "environment_seed": engine_seed, "layout_seed": layout_seed, "collector_seed": policy_seed,
                    "run_seed": run_seed, "variant": "shared_initial:B,D", "collector_kind": kind,
                    "policy_version": policy_version,
                    "collection_block": published // settings["experiment"]["collection_block"],
                    "mixture_segment": segment_index, "initial_tic": reset.training_only_info["engine_tic"],
                    "recovery_id": recovery_id, "startup_attempt": active_attempt}
                if kind == "scripted":
                    episode_metadata["scripted_profile"] = policy.profile
                operation = "storage_start_episode"
                writer.start_episode(episode_metadata, reset)
                result, episode_length = reset, 0
                limit = min(segment_remaining,
                    settings["experiment"]["collection_block"] - published % settings["experiment"]["collection_block"])
                try:
                    while episode_length < limit:
                        actions = policy.actions() if kind == "random" else policy.actions(result.training_only_info)
                        operation = "resource_enforcement"
                        if resource_guard is not None:
                            resource_guard()
                        operation = "step"
                        step = env.step(*actions, collection_cutoff=episode_length + 1 == limit)
                        ledger.record_transition(env, current_episode, kind, step)
                        published += 1
                        episode_length += 1
                        segment_remaining -= 1
                        stats["counts"][kind] += 1
                        operation = "storage_append"
                        writer.append(step)
                        info = step.training_only_info
                        stats["events"].update(info["events"]["deltas"])
                        stats["stages"][str(info["task_flags"]["stage"])] += 1
                        stats["template_transitions"][provenance["template_id"]] += 1
                        result = step
                        if published % 128 == 0:
                            measure()
                        if published % 1000 == 0:
                            progress()
                        if step.terminated or step.truncated:
                            break
                    if not result.terminated and not result.truncated:
                        raise AssertionError("continuation stopped without a declared boundary")
                    operation = "storage_finish_episode"
                    episode = writer.finish_episode()
                    if episode["valid_length"] != episode_length:
                        raise ValueError("continuation committed length differs from charged work")
                    ledger.commit_episode(current_episode, episode_length)
                    stats["terminals"][result.boundary.reason.value] += 1
                    stats["template_episodes"][provenance["template_id"]] += 1
                    if kind == "scripted":
                        stats["scripted_profiles"][policy.profile] += 1
                    stats["kind_episodes"][kind] += 1
                    episodes += 1
                finally:
                    env.close()
                operation = "runtime_archive"
                archive_attempt(current_episode, active_attempt)
                env = None
                progress()
        total = metadata["target_transitions"]
        scripted = round(total * metadata["collector_mixture"]["scripted"])
        if published != total or stats["counts"] != {"random": total - scripted, "scripted": scripted}:
            raise AssertionError("continuation did not finish the original exact budget/mixture")
        for record in prefix_files:
            if file_sha256(_inside(output, record["path"])) != record["sha256"]:
                raise ValueError("an original committed replay file changed during continuation")
        operation = "seal"
        sealed = writer.seal()
        accounting = ledger.manifest()
        if (accounting["replay_committed_joint_transitions"] != total
                or accounting["failed_dispatch_attempts"] or sealed["metadata"] != metadata):
            raise ValueError("continued replay accounting or original metadata changed")
        reader = ReplayDataset(output / "replay", verify=True)
        bindings = {}
        for variant in ("B", "D"):
            binding = output / "bindings" / f"{variant}.json"
            reader.bind_variant(binding, variant)
            bindings[variant] = {"path": binding.relative_to(output).as_posix(),
                "sha256": file_sha256(binding), "attributed_initial_joint_transitions": total}
        report = progress("complete_initial_replay")
        operation = "ledger_archive"
        ledger_archive = ledger.archive_to(output / "training_interactions.jsonl", spool_root=hot)
        report["journal_storage"] = storage_info()
        report.update(dataset_path="replay",
            dataset_manifest_sha256=file_sha256(output / "replay/manifest.json"),
            dataset_identity_sha256=sealed["dataset_identity_sha256"], bindings=bindings,
            physical_initial_dataset_copies=1, template_episodes=dict(stats["template_episodes"]),
            template_transitions=dict(stats["template_transitions"]), stage_counts=dict(stats["stages"]),
            scripted_profile_episodes=dict(stats["scripted_profiles"]),
            collection_spec_sha256=file_sha256(output / "collection_spec.json"),
            data_source="real_environment", diagnostic_imported=False, synthetic_imported=False,
            script_actions_are_imitation_targets=False, no_world_model_or_policy_training_started=True,
            recovery_history=[report["recovery_lineage"]])
        atomic_json(output / "collection_report.json", report)
        atomic_json(recovery_dir / "continuation_result.json", {"status": "passed", "recovery_id": recovery_id,
            "prefix_joint_transitions": prefix["count"], "added_joint_transitions": total - prefix["count"],
            "final_joint_transitions": total, "original_prefix_files_unchanged": len(prefix_files),
            "original_metadata_unchanged": True,
            "collection_report_sha256": file_sha256(output / "collection_report.json")})
        return report
    except BaseException as error:
        details = {"status": "failed", "task_family": family, "run_seed": run_seed,
            "episode_id": current_episode, "operation": operation, "error": repr(error),
            "published_joint_transitions": published, "completed_episodes": episodes,
            "recovery_id": recovery_id, "traceback": traceback.format_exc(),
            "retry_rule": "No physics-step retry; another explicit continuation requires a verified boundary/budget decision."}
        if hasattr(error, "collection_resource_report"):
            details["collection_resource_report"] = error.collection_resource_report
        try:
            ledger.record_failure(env, current_episode, operation, error)
            details["interaction_accounting"] = ledger.manifest()
            details["unreconciled"] = False
        except BaseException as ledger_error:
            details.update(unreconciled=True, ledger_error=repr(ledger_error))
        closed = env is None
        if env is not None:
            try:
                env.close()
                closed = True
            except BaseException as close_error:
                details["close_error"] = repr(close_error)
        if closed and active_attempt is not None and operation != "runtime_archive":
            try:
                archive_attempt(current_episode, active_attempt, failed=True)
            except BaseException as archive_error:
                details["runtime_archive_error"] = repr(archive_error)
        try:
            ledger_archive = ledger.archive_to(output / "training_interactions.jsonl", spool_root=hot)
        except BaseException as archive_error:
            details["ledger_archive_error"] = repr(archive_error)
        details["journal_storage"] = storage_info()
        atomic_json(recovery_dir / "continuation_error.json", details)
        raise


def main(argv=None):
    """Installed public CLI; no source overlays or dependency upgrades occur.

    Active collection time is reported as original prefix/failed-startup time,
    continuation time, and their explicit sum. The resource monitor covers this
    continuation process and its observed descendants only.
    """
    import argparse
    from world_models.data.resources import CollectionResources
    from world_models.env.acceptance_metrics import cpu_budget

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--journal-root", type=Path)
    parser.add_argument("--recovery-id", required=True)
    parser.add_argument("--failure-sha256", required=True)
    parser.add_argument("--failure-report", type=Path)
    parser.add_argument("--origin-project", "--original-project", dest="origin_project", type=Path, default=ROOT)
    parser.add_argument("--max-startup-attempts", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.inspect_only and args.journal_root is None:
        parser.error("native continuation requires a fresh --journal-root")
    if args.output.resolve().is_relative_to(args.dataset.resolve()):
        parser.error("CLI report must be outside the dataset's original/recovery evidence")
    report = {"status": "running", "inspect_only": args.inspect_only,
        "dataset": str(args.dataset.resolve()), "recovery_id": args.recovery_id,
        "failure_sha256": args.failure_sha256, "entrypoint": "python -m world_models.data.recovery",
        "implementation_sha256": file_sha256(__file__),
        "source_overlays_or_dependency_upgrades": False}
    started = time.perf_counter()
    resources = None

    def progress(record):
        if record["published_joint_transitions"] % 1000 == 0 or record["status"] == "complete_initial_replay":
            print(json.dumps({"status": record["status"], "recovery_id": args.recovery_id,
                "published": record["published_joint_transitions"], "target": record["target_transitions"],
                "original_prefix_elapsed_seconds": record["original_prefix_elapsed_seconds"],
                "continuation_elapsed_seconds": record["continuation_elapsed_seconds"],
                "total_collection_attempt_elapsed_seconds": record["total_collection_attempt_elapsed_seconds"]}), flush=True)

    try:
        with cpu_budget(4) as budget:
            report["cpu_budget"] = budget
            resources = CollectionResources(budget["cpu_ids"])
            with resources:
                if args.inspect_only:
                    prefix = inspect_recoverable_prefix(args.dataset,
                        expected_failure_sha256=args.failure_sha256, failure_report=args.failure_report,
                        origin_project_root=args.origin_project)
                    report["prefix"] = {"published": prefix["count"], "episodes": len(prefix["episodes"]),
                        "next_episode": prefix["expected_next_episode_id"],
                        "next_segment": prefix["remaining_segments"][0],
                        "remaining_transitions": sum(length for _, _, length in prefix["remaining_segments"]),
                        "native_zero_physics_failure_checked": prefix["native_zero_physics_failure_checked"],
                        "original_generation_proof": prefix["original_generation_proof"]}
                else:
                    result = resume_initial_dataset(args.dataset, journal_root=args.journal_root,
                        recovery_id=args.recovery_id, expected_failure_sha256=args.failure_sha256,
                        failure_report=args.failure_report, origin_project_root=args.origin_project,
                        config_dir=args.origin_project / "configs", max_startup_attempts=args.max_startup_attempts,
                        progress_callback=progress, resource_guard=resources.check_health,
                        recovery_origin_provenance={
                            "entrypoint": report["entrypoint"], "implementation_sha256": report["implementation_sha256"]})
                    report["interaction_accounting"] = result["interaction_accounting"]
                    report["published_joint_transitions"] = result["published_joint_transitions"]
                    for field in ("original_prefix_elapsed_seconds", "original_failed_collection_attempt_elapsed_seconds",
                                  "original_failed_startup_elapsed_seconds", "continuation_elapsed_seconds",
                                  "total_collection_attempt_elapsed_seconds"):
                        report[field] = result[field]
            report["continuation_resource_measurements"] = resources.report()
        report["status"] = "passed_inspection" if args.inspect_only else "complete_initial_replay"
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        if hasattr(error, "collection_resource_report"):
            report["collection_resource_report"] = error.collection_resource_report
    if resources is not None:
        try:
            report["continuation_resource_measurements"] = resources.report()
        except BaseException as resource_error:
            report["continuation_resource_report_error"] = repr(resource_error)
    report["cli_elapsed_seconds"] = time.perf_counter() - started
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output), "error": report.get("error")}), flush=True)
    return 0 if report["status"] in ("passed_inspection", "complete_initial_replay") else 1


if __name__ == "__main__":
    raise SystemExit(main())
