"""Independent, bounded-memory verification of sealed P2.1 replay."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from world_models.data.replay import ReplayDataset
from world_models.data.schema import REASONS, episode_partition
from world_models.infrastructure.config import resolve_config


def audit_initial_dataset(directory: str | Path, *, expected_transitions: int | None = None,
                          require_full_coverage: bool = False) -> dict:
    """Verify bytes and label alignment; load at most one episode at a time."""
    directory = Path(directory).resolve()
    # Verify the charged source interactions as well as the replay payload.
    # Imported lazily to avoid a collector/storage module import cycle.
    from world_models.data.collection import verify_initial_collection
    accounting_verification = verify_initial_collection(directory)
    dataset = ReplayDataset(directory / "replay", verify=True, cache_chunks=2)
    manifest = dataset.manifest
    family, seed = manifest["metadata"]["task_family"], manifest["metadata"]["run_seed"]
    profile = manifest["metadata"].get("profile", "pilot")
    learning_eligible = manifest["metadata"]["learning_eligible"]
    config = resolve_config(family, "B", profile).settings
    if expected_transitions is not None and manifest["transition_count"] != expected_transitions:
        raise ValueError("sealed dataset does not have the requested transition count")
    kinds, partitions, partition_episodes, endings = Counter(), Counter(), Counter(), Counter()
    layouts, events, layout_events = {}, Counter(), {}
    event_fields = manifest["schema"]["event_fields"]
    checks = 0
    for episode in dataset.episodes:
        identifier, length = episode["episode_id"], episode["valid_length"]
        view = dataset.load_episode(identifier, purpose="diagnostic")
        arrays = view.arrays()
        if episode["partition"] != episode_partition(identifier):
            raise ValueError("episode partition is not derived from its stable ID")
        if len(arrays["rgb"]) != length + 1:
            raise ValueError("observations were duplicated or omitted at chunk boundaries")
        if not np.array_equal(arrays["state_engine_tic"][:-1], arrays["start_tic"]) or not np.array_equal(arrays["state_engine_tic"][1:], arrays["end_tic"]):
            raise ValueError("state at t does not align with transition t->t+1")
        actual_tics = arrays["actual_tics"]
        frame_skip = config["environment"]["frame_skip"]
        if (np.any(actual_tics < 1) or np.any(actual_tics > frame_skip)
                or np.any((actual_tics < frame_skip) & ~arrays["terminated"])):
            raise ValueError("a published decision has an unexpected native tic count")
        if not np.array_equal(np.diff(arrays["cumulative_event_counters"], axis=0), arrays["event_deltas"]):
            raise ValueError("transition events are not cumulative-counter differences")
        if not np.array_equal(arrays["next_obs_valid"], arrays["image_valid"][1:]):
            raise ValueError("transition image validity is shifted")
        if np.any(arrays["actions"] >= config["environment"]["action_count"]):
            raise ValueError("undeclared action entered real replay")
        labels = [REASONS[int(value)] for value in arrays["termination_reason"]]
        milestone_names = ("latch_activations",) if family == "relay" else ("console_1_completions", "console_2_completions")
        bonus = np.zeros(length, dtype=np.float64)
        for name in milestone_names:
            if name not in event_fields:
                raise ValueError("required authoritative milestone field is missing")
            bonus += arrays["event_deltas"][:, event_fields.index(name)] * config["task"]["milestone_bonus"]
        outcomes = np.array([config["task"]["success_reward"] if label == "success" else
                             config["task"]["failure_reward"] if label in ("death", "failure", "deadline") else 0.
                             for label in labels])
        if not np.allclose(arrays["team_reward"], config["task"]["step_cost"] + bonus + outcomes, rtol=0, atol=1e-9):
            raise ValueError("stored reward differs from the shared authoritative reward rule")
        for purpose in ("world_model_fit", "imagination"):
            if not learning_eligible or episode["partition"] == "diagnostic":
                try:
                    dataset.load_episode(identifier, purpose=purpose)
                except ValueError:
                    pass
                else:
                    raise ValueError("a diagnostic/preflight episode is accessible to fitting/imagination")
            else:
                dataset.load_episode(identifier, purpose=purpose)
        kind, layout, partition = episode["collector_kind"], episode["layout_id"], episode["partition"]
        kinds[kind] += length
        partitions[partition] += length
        partition_episodes[partition] += 1
        endings[episode["end_reason"]] += 1
        layouts.setdefault(layout, Counter())[kind] += length
        event_counts = {name: int(value) for name, value in zip(event_fields, arrays["event_deltas"].sum(axis=0))}
        events.update(event_counts)
        layout_events.setdefault(layout, Counter()).update(event_counts)
        checks += length
    total = manifest["transition_count"]
    expected_scripted = int(round(total * config["experiment"]["initial_scripted_fraction"]))
    if kinds != Counter(random=total - expected_scripted, scripted=expected_scripted):
        raise ValueError("stored random/scripted interaction counts differ from the agreed mixture")
    if require_full_coverage:
        if len(layouts) != config["task"]["train_layouts"] or any(not all(counts[kind] > 0 for kind in ("random", "scripted")) for counts in layouts.values()):
            raise ValueError("both collectors must visit every training geometry")
        if not all(partition_episodes[part] > 0 for part in ("fit", "diagnostic")):
            raise ValueError("full initialization must contain fitting and diagnostic episodes")
        required_events = ("latch_activations", "successes", "failures") if family == "relay" else ("console_1_completions", "console_2_completions", "successes", "failures")
        # Event field names are authoritative registry names, not image guesses.
        for name in required_events:
            if name not in event_fields or events[name] == 0:
                raise ValueError(f"initial data did not cover required task event: {name}")
        if family == "cover":
            combat_fields = [name for name in event_fields if name.endswith(("_hitcount", "_damagecount", "_damage_taken"))]
            if not any(events[name] > 0 for name in combat_fields):
                raise ValueError("cover initialization did not contain authoritative combat events")
    bindings = {}
    for variant in ("B", "D"):
        path = directory / "bindings" / f"{variant}.json"
        binding = json.loads(path.read_text(encoding="utf-8"))
        if binding["variant"] != variant or binding["attributed_initial_joint_transitions"] != total:
            raise ValueError("B/D initialization attribution is not equal and complete")
        shared = ReplayDataset.from_binding(path, verify=False, cache_chunks=1)
        if shared.path != dataset.path or shared.manifest_sha256 != dataset.manifest_sha256:
            raise ValueError("B/D are not bound to the same immutable physical replay")
        bindings[variant] = binding
    return {"status": "passed", "task_family": family, "run_seed": seed,
            "transition_count": total, "observation_count": sum(ep["observation_count"] for ep in dataset.episodes),
            "episode_count": len(dataset.episodes), "partition_transition_counts": dict(partitions),
            "partition_episode_counts": dict(partition_episodes), "collector_transition_counts": dict(kinds),
            "end_reason_counts": dict(endings), "layout_collector_transition_counts": {key: dict(value) for key, value in layouts.items()},
            "event_counts": dict(events), "layout_event_counts": {key: dict(value) for key, value in layout_events.items()},
            "aligned_transitions_checked": checks, "manifest_sha256": dataset.manifest_sha256,
            "dataset_identity_sha256": manifest["dataset_identity_sha256"],
            "raw_rgb_array_bytes": sum(ep["observation_count"] for ep in dataset.episodes) * 73728,
            "chunk_bytes": sum(chunk["bytes"] for ep in dataset.episodes for chunk in ep["chunks"]),
            "bindings": bindings,
            "interaction_ledger_verification": accounting_verification,
            "scope": "P2.1 real replay integrity and coverage only; no learned world model or policy claim."}
