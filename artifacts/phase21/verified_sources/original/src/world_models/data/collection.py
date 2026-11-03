"""Training-only, paired initial replay collection for P2.1.

Only this collection pipeline writes learning-eligible initial data. Privileged
waypoint scripts generate transitions, not imitation targets. Random and scripted
quotas count actual jointly executed decisions rather than episodes or actors.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time
import traceback
from typing import Any, Callable

from world_models.data.ledger import TrainingInteractionLedger, atomic_json, file_sha256
from world_models.data.replay import ReplayDataset, ReplayDatasetWriter
from world_models.env.coop import TwoPlayerCoopEnv
from world_models.env.scripted import _Controller
from world_models.env.splits import build_split_scenario, list_templates
from world_models.infrastructure.config import resolve_config, validate_pair
from world_models.infrastructure.reproducibility import canonical_hash, derive_seed


ROOT = Path(__file__).resolve().parents[3]
COLLECTION_VERSION = "paired-initial-replay-v1"
RANDOM_POLICY_VERSION = "correlated-uniform-actions-v1"
SCRIPTED_POLICY_VERSION = "privileged-mechanics-exploration-v1"


def _stream_seed(initial_seed: int, episode_id: str, stream: str) -> int:
    return int(canonical_hash({"initial_seed": initial_seed, "episode_id": episode_id, "stream": stream})[:8], 16)


def mixture_segments(total: int, scripted_fraction: float = .3, block: int = 1000) -> list[tuple[str, int]]:
    """Exact transition quotas, with bounded alternation rather than long phases."""
    if type(total) is not int or total < 1 or type(block) is not int or block < 2:
        raise ValueError("positive integer total and mixture block >=2 are required")
    if type(scripted_fraction) not in (int, float) or not 0 < scripted_fraction < 1:
        raise ValueError("scripted fraction must be strictly between zero and one")
    scripted = total * scripted_fraction
    if abs(scripted - round(scripted)) > 1e-8:
        raise ValueError("total must permit exact integral random/scripted transition quotas")
    remaining = {"random": total - round(scripted), "scripted": round(scripted)}
    per_block = {"scripted": max(1, round(block * scripted_fraction))}
    per_block["random"] = max(1, block - per_block["scripted"])
    segments: list[tuple[str, int]] = []
    while any(remaining.values()):
        for kind in ("random", "scripted"):
            count = min(per_block[kind], remaining[kind])
            if count:
                segments.append((kind, count))
                remaining[kind] -= count
    return segments


class CorrelatedRandomPolicy:
    """Independent uniform categorical actions, held 2--8 decisions per player."""

    def __init__(self, seed: int):
        self.rng = [random.Random(_stream_seed(seed, "random-policy", f"player-{i}")) for i in (0, 1)]
        self.remaining = [0, 0]
        self.current = [0, 0]

    def actions(self) -> tuple[int, int]:
        for player in (0, 1):
            if self.remaining[player] == 0:
                self.current[player] = self.rng[player].randrange(12)
                self.remaining[player] = self.rng[player].randint(2, 8)
            self.remaining[player] -= 1
        return tuple(self.current)


class ScriptedExplorationPolicy:
    """Use stock mechanics controllers with documented variation in behavior.

    All profiles obey the same simulator rules. No task state, health, objects,
    geometry, or rewards are changed. No script action is an imitation label.
    """

    PROFILES = ("standard", "no_cover_fire", "same_operator", "dwell_pause")

    def __init__(self, scenario: dict, index: int):
        self.scenario = scenario
        self.profile = self.PROFILES[index % len(self.PROFILES)] if scenario["task_family"] == "cover" else "standard"
        self.controller = _Controller(scenario, operator_policy="same" if self.profile == "same_operator" else "swap")
        self.pause_left = 0
        self.pause_started = False

    @property
    def version(self) -> str:
        return SCRIPTED_POLICY_VERSION + ":" + self.profile

    def actions(self, info: dict) -> tuple[int, int]:
        flags = info["task_flags"]
        actions = self.controller.actions(info["players"], flags, covering_fire=self.profile != "no_cover_fire")
        if (self.profile == "dwell_pause" and flags["stage"] == 0
                and not self.pause_started and flags.get("progress_1", 0) >= 5):
            self.pause_left, self.pause_started = 3, True
        if self.pause_left:
            actions[self.controller.stage_operator(int(flags["stage"]))] = 0
            self.pause_left -= 1
        return tuple(actions)


def _initial_manifest(task_family: str, run_seed: int, profile: str,
                      total: int, scripted_fraction: float, settings: dict,
                      initial_seed: int) -> dict:
    shared = deepcopy(settings)
    shared.pop("variant")
    return {
        "dataset_version": COLLECTION_VERSION,
        "task_family": task_family, "run_seed": run_seed,
        "profile": profile, "data_source": "real_environment",
        "variant": "shared_initial:B,D", "paired_variants": ["B", "D"],
        "target_transitions": total,
        "configured_initial_transitions": settings["experiment"]["initial_transitions"],
        "budget_scope": "approved_initial" if total == settings["experiment"]["initial_transitions"] else "collection_preflight",
        "learning_eligible": total == settings["experiment"]["initial_transitions"],
        "initial_collection_seed": initial_seed,
        "shared_configuration_sha256": canonical_hash(shared),
        "collector_mixture": {"random": 1-scripted_fraction, "scripted": scripted_fraction},
        "random_policy": {"version": RANDOM_POLICY_VERSION, "action_count": 12,
                          "hold_decisions_inclusive": [2, 8], "agent_streams": "independent"},
        "scripted_policy": {"version": SCRIPTED_POLICY_VERSION,
                            "profiles": list(ScriptedExplorationPolicy.PROFILES),
                            "uses_privileged_state": True, "provides_imitation_targets": False},
        "partition_rule": "episode-id deterministic 90% fitting / 10% model-diagnostic",
        "evaluation_layouts_used": False,
        "environment_validation_data_imported": False,
        "synthetic_ppo_data_imported": False,
        "runtime_scope": "real replay pipeline only; no world-model or policy training",
    }


def collect_initial_dataset(
    task_family: str,
    run_seed: int,
    output_dir: str | Path,
    *,
    profile: str = "pilot",
    total_transitions: int | None = None,
    chunk_transitions: int = 128,
    compression: str = "deflate",
    compression_level: int = 1,
    mixture_block: int = 1000,
    env_factory: Callable[..., Any] | None = None,
    scenario_builder: Callable[..., dict] | None = None,
    writer_factory: Callable[..., Any] | None = None,
    progress_callback: Callable[[dict], Any] | None = None,
    config_dir: str | Path | None = None,
    journal_root: str | Path | None = None,
    journal_spool_max_bytes: int = 512 * 1024 * 1024,
    origin_provenance: dict | None = None,
) -> dict:
    """Collect and seal one physically shared B/D dataset, or fail closed.

    Existing output directories are rejected. An interrupted/failed dataset keeps
    its ledger, partial chunks, coordinator journals, and error record; restarting
    it must not silently erase already consumed interactions.
    """
    config_dir = Path(config_dir or ROOT / "configs")
    left = resolve_config(task_family, "B", profile, config_dir)
    right = resolve_config(task_family, "D", profile, config_dir)
    validate_pair(left, right)
    settings = left.settings
    if type(run_seed) is not int or run_seed not in settings["experiment"]["master_seeds"]:
        raise ValueError("run seed must belong to the selected pilot/final profile")
    total = settings["experiment"]["initial_transitions"] if total_transitions is None else total_transitions
    scripted_fraction = settings["experiment"]["initial_scripted_fraction"]
    segments = mixture_segments(total, scripted_fraction, mixture_block)
    templates = list_templates(task_family, "train")
    if len(templates) != 6:
        raise ValueError("initial replay requires the six approved training templates")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("use a new dataset directory; existing collection accounting must be preserved")
    if origin_provenance is not None and not isinstance(origin_provenance, dict):
        raise ValueError("origin provenance must be a JSON-compatible dictionary")
    json.dumps(origin_provenance or {}, allow_nan=False)
    if type(journal_spool_max_bytes) is not int or journal_spool_max_bytes <= 0:
        raise ValueError("journal spool limit must be a positive finite byte count")
    hot_root = Path(journal_root).resolve() if journal_root is not None else output
    spooled = journal_root is not None
    if spooled:
        if hot_root == output or hot_root.is_relative_to(output) or output.is_relative_to(hot_root):
            raise ValueError("hot journal root and archived dataset directory must not overlap")
        if hot_root.exists() and any(hot_root.iterdir()):
            raise FileExistsError("hot journal directory is not empty; preserve earlier collection evidence")
        probe = hot_root
        while not probe.exists():
            probe = probe.parent
        if shutil.disk_usage(probe).free < journal_spool_max_bytes:
            raise OSError("insufficient free space for the declared bounded journal spool")
    output.mkdir(parents=True)
    if spooled:
        hot_root.mkdir(parents=True, exist_ok=True)
    runtime_root = hot_root / "runtime"
    ledger_directory = hot_root / "ledger" if spooled else output
    initial_seed = derive_seed(task_family, run_seed, "initial_collection", "B")
    if initial_seed != derive_seed(task_family, run_seed, "initial_collection", "D"):
        raise AssertionError("initial collection RNG must be paired between B and D")
    metadata = _initial_manifest(task_family, run_seed, profile, total, scripted_fraction, settings, initial_seed)
    metadata["mixture_block"] = mixture_block
    metadata["origin_provenance"] = deepcopy(origin_provenance or {})
    metadata["journal_spooling"] = {
        "enabled": spooled, "maximum_hot_bytes": journal_spool_max_bytes if spooled else None,
        "durability": "per-dispatch/per-transition fsync is retained; closed journals are durably archived before hot cleanup",
        "rgb_storage": "replay chunks always reside in the dataset output directory",
    }
    atomic_json(output / "collection_spec.json", metadata)
    ledger = TrainingInteractionLedger(ledger_directory, task_family=task_family, run_seed=run_seed,
                                       learning_eligible=metadata["learning_eligible"])
    writer_class = writer_factory or ReplayDatasetWriter
    writer = writer_class(output / "replay", metadata, chunk_transitions=chunk_transitions,
                          compression=compression, compression_level=compression_level)
    create_env = env_factory or TwoPlayerCoopEnv
    build_scenario = scenario_builder or build_split_scenario
    engine_version = importlib.metadata.version("vizdoom")
    published = 0
    episodes = 0
    counts = {"random": 0, "scripted": 0}
    kind_episodes = {"random": 0, "scripted": 0}
    event_counts: Counter = Counter()
    terminal_counts: Counter = Counter()
    template_episodes: Counter = Counter()
    template_transitions: Counter = Counter()
    stage_counts: Counter = Counter()
    scripted_profiles: Counter = Counter()
    started = time.perf_counter()
    env = None
    current_episode = ""
    operation = "setup"
    runtime_archives = []
    ledger_archive = None
    peak_spool_bytes = 0

    def measure_spool():
        nonlocal peak_spool_bytes
        if not spooled:
            return 0
        current = sum(path.stat().st_size for path in hot_root.rglob("*") if path.is_file())
        peak_spool_bytes = max(peak_spool_bytes, current)
        if current > journal_spool_max_bytes:
            raise OSError("bounded hot journal spool exceeded its declared size; source files retained")
        return current

    def journal_storage():
        return {
            "spooled": spooled,
            "spool_root_at_collection": str(hot_root) if spooled else None,
            "runtime_hot_root_at_collection": str(runtime_root),
            "runtime_archive_root": "runtime",
            "ledger_hot_path_at_collection": str(ledger.path),
            "ledger_archive_path": "training_interactions.jsonl",
            "ledger_archive": deepcopy(ledger_archive),
            "runtime_archives": deepcopy(runtime_archives),
            "maximum_hot_bytes": journal_spool_max_bytes if spooled else None,
            "peak_hot_bytes": peak_spool_bytes,
            "archive_policy": "close both participants; copy/flush/fsync/hash every member and receipt; delete only verified closed hot runtime trees",
        }

    def archive_episode(episode_id):
        if not spooled or not episode_id:
            return
        from world_models.data.spool import archive_closed_tree
        measure_spool()
        for category in ("scenarios", "environments"):
            source = runtime_root / category / episode_id
            if source.exists():
                destination = output / "runtime" / category / episode_id
                receipt = archive_closed_tree(source, destination, spool_root=hot_root)
                runtime_archives.append({
                    "episode_id": episode_id, "category": category,
                    "archive_path": destination.relative_to(output).as_posix(), **receipt,
                })

    def archive_ledger():
        nonlocal ledger_archive
        if spooled:
            ledger_archive = ledger.archive_to(output / "training_interactions.jsonl", spool_root=hot_root)
        return ledger_archive

    def progress(status="collecting"):
        measure_spool()
        record = {
            "status": status, "task_family": task_family, "run_seed": run_seed,
            "profile": profile, "target_transitions": total,
            "budget_scope": metadata["budget_scope"], "learning_eligible": metadata["learning_eligible"],
            "published_joint_transitions": published,
            "completed_episodes": episodes, "collector_counts": dict(counts),
            "event_counts": dict(event_counts), "terminal_counts": dict(terminal_counts),
            "elapsed_seconds": time.perf_counter()-started,
            "interaction_accounting": ledger.manifest(),
            "journal_storage": journal_storage(),
        }
        atomic_json(output / "collection_progress.json", record)
        if progress_callback:
            progress_callback(deepcopy(record))
        return record

    try:
        measure_spool()
        for segment_index, (kind, segment_length) in enumerate(segments):
            segment_remaining = segment_length
            while segment_remaining:
                current_episode = f"{task_family}-s{run_seed}-e{episodes:06d}"
                template_index = kind_episodes[kind] % len(templates)
                layout_seed = _stream_seed(initial_seed, current_episode, "layout-nuisance")
                engine_seed = _stream_seed(initial_seed, current_episode, "engine")
                policy_seed = _stream_seed(initial_seed, current_episode, "policy")
                operation = "scenario_build"
                scenario = build_scenario(
                    task_family, "train", template_index, seed=layout_seed,
                    purpose="training", frame_skip=settings["environment"]["frame_skip"],
                    output_dir=runtime_root / "scenarios" / current_episode,
                )
                provenance = scenario.get("split_provenance", {})
                if (provenance.get("split") != "train" or provenance.get("purpose") != "training"
                        or provenance.get("learning_eligible") is not True):
                    raise ValueError("initial collection may only use learning-eligible training geometry")
                operation = "environment_create"
                env = create_env(task_family, seed=layout_seed, scenario_manifest=scenario,
                                 work_root=runtime_root / "environments" / current_episode,
                                 frame_skip=settings["environment"]["frame_skip"])
                operation = "reset"
                reset = env.reset(seed=engine_seed)
                ledger.record_reset(env, current_episode)
                policy = CorrelatedRandomPolicy(policy_seed) if kind == "random" else ScriptedExplorationPolicy(scenario, kind_episodes[kind] // len(templates))
                policy_version = RANDOM_POLICY_VERSION if kind == "random" else policy.version
                collection_block = published // settings["experiment"]["collection_block"]
                episode_metadata = {
                    "episode_id": current_episode, "task_family": task_family,
                    "layout_id": provenance["template_id"], "split": "train",
                    "scenario_hash": scenario["hashes"]["wad_sha256"],
                    "engine_version": engine_version,
                    "environment_seed": engine_seed, "layout_seed": layout_seed,
                    "collector_seed": policy_seed, "run_seed": run_seed,
                    "variant": "shared_initial:B,D", "collector_kind": kind,
                    "policy_version": policy_version, "collection_block": collection_block,
                    "mixture_segment": segment_index,
                    "initial_tic": reset.training_only_info["engine_tic"],
                }
                if kind == "scripted":
                    episode_metadata["scripted_profile"] = policy.profile
                    scripted_profiles[policy.profile] += 1
                operation = "storage_start_episode"
                writer.start_episode(episode_metadata, reset)
                result = reset
                episode_length = 0
                # Stop at both mixture and external collection block boundaries.
                boundary_limit = settings["experiment"]["collection_block"] - published % settings["experiment"]["collection_block"]
                episode_limit = min(segment_remaining, boundary_limit)
                try:
                    while episode_length < episode_limit:
                        actions = policy.actions() if kind == "random" else policy.actions(result.training_only_info)
                        operation = "step"
                        step = env.step(*actions, collection_cutoff=episode_length + 1 == episode_limit)
                        # Debit real work before any storage/metadata operation.
                        ledger.record_transition(env, current_episode, kind, step)
                        published += 1
                        counts[kind] += 1
                        episode_length += 1
                        segment_remaining -= 1
                        operation = "storage_append"
                        writer.append(step)
                        info = step.training_only_info
                        event_counts.update(info["events"]["deltas"])
                        stage_counts[str(info["task_flags"]["stage"])] += 1
                        template_transitions[provenance["template_id"]] += 1
                        result = step
                        if spooled and published % 128 == 0:
                            measure_spool()
                        if published % 1000 == 0:
                            progress()
                        if step.terminated or step.truncated:
                            break
                    if not result.terminated and not result.truncated:
                        raise AssertionError("episode segment ended without a declared boundary")
                    operation = "storage_finish_episode"
                    episode = writer.finish_episode()
                    if episode.get("valid_length") != episode_length:
                        raise ValueError("committed replay episode length differs from charged interactions")
                    ledger.commit_episode(current_episode, episode_length)
                    terminal_counts[result.boundary.reason.value] += 1
                    template_episodes[provenance["template_id"]] += 1
                    episodes += 1
                    kind_episodes[kind] += 1
                finally:
                    env.close()
                operation = "runtime_archive"
                archive_episode(current_episode)
                env = None
                progress()
        expected_scripted = round(total * scripted_fraction)
        if counts != {"random": total - expected_scripted, "scripted": expected_scripted}:
            raise AssertionError("collection did not meet exact shared random/scripted transition quotas")
        if published != total:
            raise AssertionError("initial collection budget not met")
        operation = "seal"
        manifest = writer.seal()
        accounting = ledger.manifest()
        if (accounting["published_real_joint_transitions"] != total
                or accounting["replay_committed_joint_transitions"] != total
                or accounting["failed_dispatch_attempts"]):
            raise AssertionError("sealed replay and interaction accounting disagree")
        operation = "paired_bindings"
        # The bindings point to one immutable store; no images are copied.
        reader = ReplayDataset(output / "replay", verify=True)
        binding_dir = output / "bindings"
        binding_dir.mkdir()
        bindings = {}
        for variant in ("B", "D"):
            binding_path = binding_dir / f"{variant}.json"
            reader.bind_variant(binding_path, variant)
            bindings[variant] = {
                "path": str(binding_path.relative_to(output)),
                "sha256": file_sha256(binding_path),
                "attributed_initial_joint_transitions": total,
            }
        report = progress("complete_initial_replay")
        operation = "ledger_archive"
        archive_ledger()
        report["journal_storage"] = journal_storage()
        report.update(
            dataset_path="replay", dataset_manifest_sha256=file_sha256(output / "replay" / "manifest.json"),
            dataset_identity_sha256=manifest["dataset_identity_sha256"],
            bindings=bindings, physical_initial_dataset_copies=1,
            template_episodes=dict(template_episodes), template_transitions=dict(template_transitions),
            stage_counts=dict(stage_counts), scripted_profile_episodes=dict(scripted_profiles),
            collection_spec_sha256=file_sha256(output / "collection_spec.json"),
            data_source="real_environment", diagnostic_imported=False, synthetic_imported=False,
            script_actions_are_imitation_targets=False,
            no_world_model_or_policy_training_started=True,
        )
        atomic_json(output / "collection_report.json", report)
        return report
    except BaseException as exc:
        # Retain consumed work before closing engines or archiving closed files.
        try:
            ledger.record_failure(env, current_episode, operation, exc)
        except BaseException as ledger_error:
            failure = {"error": repr(exc), "ledger_error": repr(ledger_error),
                       "episode_id": current_episode, "operation": operation,
                       "unreconciled": True}
        else:
            failure = {"error": repr(exc), "episode_id": current_episode,
                       "operation": operation, "unreconciled": False,
                       "interaction_accounting": ledger.manifest()}
        closed = env is None
        if env is not None:
            try:
                env.close()
                closed = True
            except BaseException as close_error:
                failure["runtime_close_error"] = repr(close_error)
        if spooled:
            # A failed archival copy is never retried/overwritten automatically;
            # preserve its hot source and partial cold destination for inspection.
            if closed and operation != "runtime_archive":
                try:
                    archive_episode(current_episode)
                except BaseException as archive_error:
                    failure["runtime_archive_error"] = repr(archive_error)
            try:
                archive_ledger()
            except BaseException as archive_error:
                failure["ledger_archive_error"] = repr(archive_error)
        failure.update(
            status="failed", task_family=task_family, run_seed=run_seed,
            published_joint_transitions=published, completed_episodes=episodes,
            traceback=traceback.format_exc(),
            journal_storage=journal_storage(),
            retry_rule="No automatic retries or overwrite; retain consumed-interaction ledger and any unarchived hot files.")
        atomic_json(output / "collection_error.json", failure)
        raise


def verify_initial_collection(output_dir: str | Path) -> dict:
    """Verify sealed data, ledger conservation, mixture, and both branch bindings."""
    output = Path(output_dir)
    report = json.loads((output / "collection_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete_initial_replay":
        raise ValueError("initial replay collection is not complete")
    reader = ReplayDataset(output / report["dataset_path"], verify=True)
    if file_sha256(output / "replay" / "manifest.json") != report["dataset_manifest_sha256"]:
        raise ValueError("shared initial dataset manifest changed")
    if file_sha256(output / "collection_spec.json") != report["collection_spec_sha256"]:
        raise ValueError("initial collection specification changed")
    accounting = report["interaction_accounting"]
    if file_sha256(output / accounting["ledger_path"]) != accounting["ledger_sha256"]:
        raise ValueError("real-interaction ledger changed")
    rows = [json.loads(line) for line in (output / accounting["ledger_path"]).read_text(encoding="utf-8").splitlines()]
    if [row["sequence"] for row in rows] != list(range(len(rows))):
        raise ValueError("noncontiguous real-interaction ledger")
    physical = sum(row.get("physical_delta", 0) for row in rows)
    initialization = sum(row.get("initialization_delta", 0) for row in rows)
    published = sum(row.get("published_delta", 0) for row in rows)
    failed = sum(row.get("failed_delta", 0) for row in rows)
    published_tics = sum(row["actual_tics"] for row in rows if row["event"] == "transition")
    committed = sum(row.get("valid_length", 0) for row in rows if row["event"] == "episode_committed")
    counts = dict(Counter(row["collector_kind"] for row in rows if row["event"] == "transition"))
    if physical != initialization + published + failed:
        raise ValueError("physical real-interaction ledger does not reconcile")
    if (physical != accounting["physical_joint_decisions_including_initialization"]
            or initialization != accounting["initialization_joint_decisions"]
            or published != accounting["published_real_joint_transitions"]
            or published_tics != accounting["published_real_simulator_tics_total"]
            or committed != accounting["replay_committed_joint_transitions"]
            or failed != accounting["failed_dispatch_attempts"]
            or counts != accounting["collector_counts"]):
        raise ValueError("collection report disagrees with durable real-interaction ledger")
    if published != report["target_transitions"] or committed != published or failed:
        raise ValueError("complete replay contains missing or uncharged real transitions")
    journal = report.get("journal_storage", {})
    if journal.get("spooled"):
        from world_models.data.spool import verify_archive_tree
        receipt = journal.get("ledger_archive")
        if (not isinstance(receipt, dict) or receipt.get("sha256") != accounting["ledger_sha256"]
                or receipt.get("bytes") != (output / accounting["ledger_path"]).stat().st_size):
            raise ValueError("hot-to-cold ledger archive does not match final interaction accounting")
        maximum, peak = journal.get("maximum_hot_bytes"), journal.get("peak_hot_bytes")
        if type(maximum) is not int or type(peak) is not int or not 0 <= peak <= maximum:
            raise ValueError("completed collection exceeded its bounded hot journal spool")
        seen_archives = set()
        for archive in journal.get("runtime_archives", []):
            relative = archive["archive_path"]
            path = (output / relative).resolve()
            if not path.is_relative_to(output.resolve()) or relative in seen_archives:
                raise ValueError("runtime archive path escapes or repeats in its collection report")
            verify_archive_tree(path, expected_receipt_sha256=archive["receipt_sha256"])
            seen_archives.add(relative)
    for variant in ("B", "D"):
        binding = output / report["bindings"][variant]["path"]
        if file_sha256(binding) != report["bindings"][variant]["sha256"]:
            raise ValueError("variant initialization binding changed")
        linked = ReplayDataset.from_binding(binding, variant=variant, verify=True)
        if linked.manifest["dataset_identity_sha256"] != reader.manifest["dataset_identity_sha256"]:
            raise ValueError("B/D binding does not share the same immutable data identity")
    return {"status": "passed", "task_family": report["task_family"], "run_seed": report["run_seed"],
            "joint_transitions": published, "episodes": report["completed_episodes"],
            "physical_joint_decisions": physical, "initialization_joint_decisions": initialization,
            "published_real_simulator_tics_total": published_tics,
            "collector_counts": counts, "dataset_identity_sha256": report["dataset_identity_sha256"]}
