"""Fixed open-loop joint-action replay with independent state and RGB checks.

This compares exactly the same externally supplied action sequence. Closed-loop
controllers can amplify one difference into a different future action sequence;
their histories alone cannot establish fixed-action simulator repeatability.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from world_models.contracts import validate_action
from world_models.env.coop import TwoPlayerCoopEnv
from world_models.env.observations import frame_array, save_rgb


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    files = sorted(package.glob("*.py")) + [package.parent / "contracts.py"]
    return {str(path.relative_to(package.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def prediction_config_record(path: str | Path) -> dict:
    """Read back a recognized engine CVAR, not an unknown-option declaration."""
    path = Path(path)
    section, observed = None, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        if section == "GlobalSettings":
            match = re.fullmatch(r"cl_noprediction=(true|false)", line.strip())
            if match:
                observed = match.group(1) == "true"
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cl_noprediction": observed, "recognized_section": "GlobalSettings" if observed is not None else None}


def interaction_counts(before_count: int, before_initialization: int, after_count: int,
                       after_initialization: int, published: int) -> dict:
    values = (before_count, before_initialization, after_count, after_initialization, published)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("interaction counters must be nonnegative integers")
    consumed = after_count - before_count
    initialization = after_initialization - before_initialization
    if consumed < 0 or initialization < 0 or consumed != published + initialization:
        raise ValueError("replay interaction accounting does not reconcile")
    return {"consumed_joint_decisions": after_count, "consumed_joint_decisions_delta": consumed,
            "initialization_joint_decisions": after_initialization,
            "initialization_joint_decisions_delta": initialization, "published_transitions": published}


def replay_snapshot(result: Any, *, reset: bool = False) -> dict:
    """Exclude run IDs/accounting/wall time, retaining all physical variables."""
    info = result.training_only_info
    state = {"engine_tic": info["engine_tic"], "decision_index": info["decision_index"],
             "task_flags": deepcopy(info["task_flags"]), "players": deepcopy(info["players"])}
    # P1.5 enrichments remain training-only but are part of repeatability.
    for name in ("privileged_targets", "event_counters", "event_deltas", "events", "reward_components"):
        if name in info:
            state[name] = deepcopy(info[name])
    events_rewards = {"task_flags": deepcopy(info["task_flags"])}
    for name in ("event_counters", "event_deltas", "events", "reward_components"):
        if name in info:
            events_rewards[name] = deepcopy(info[name])
    if not reset:
        state.update(reward=result.team_reward, terminated=result.terminated, truncated=result.truncated,
                     boundary_reason=result.boundary.reason.value,
                     actual_simulator_tics=result.metadata.actual_simulator_tics)
        events_rewards.update(reward=result.team_reward, terminated=result.terminated,
                              truncated=result.truncated, boundary_reason=result.boundary.reason.value)
    rgb = [None if frame is None else hashlib.sha256(frame.data).hexdigest() for frame in result.observations]
    return {"state": state, "events_rewards": events_rewards, "rgb_sha256": rgb,
            "state_sha256": _digest(state), "events_rewards_sha256": _digest(events_rewards)}


def _first_difference(reference: list[dict], replay: list[dict], key: str) -> dict | None:
    for index, (left, right) in enumerate(zip(reference, replay)):
        if _canonical(left[key]) != _canonical(right[key]):
            return {"snapshot_index": index, "reference": left[key], "replay": right[key]}
    if len(reference) != len(replay):
        return {"snapshot_index": min(len(reference), len(replay)),
                "reference_snapshots": len(reference), "replay_snapshots": len(replay)}
    return None


def compare_replays(reference: list[dict], replay: list[dict]) -> dict:
    state = _first_difference(reference, replay, "state")
    events = _first_difference(reference, replay, "events_rewards")
    rgb = _first_difference(reference, replay, "rgb_sha256")
    return {"state_exact": state is None, "events_rewards_exact": events is None, "rgb_exact": rgb is None,
            "first_state_difference": state, "first_events_rewards_difference": events,
            "first_rgb_difference": rgb}


def run_fixed_action_replay(task_family: str, actions: Sequence[Sequence[int]], *, seed: int = 0,
                            repeats: int = 3, output_dir: str | Path = "artifacts/phase1/replay_diagnostics",
                            scenario_manifest: dict | None = None, reset_mode: str = "fresh_process",
                            coordinator_reset_strategy: str = "fresh_process") -> dict:
    if task_family not in ("relay", "cover") or type(seed) is not int or seed < 0:
        raise ValueError("task family and nonnegative seed required")
    if type(repeats) is not int or repeats < 2:
        raise ValueError("repeats must be at least two")
    if reset_mode not in ("fresh_process", "coordinated_reset"):
        raise ValueError("reset_mode must be fresh_process or coordinated_reset")
    if coordinator_reset_strategy not in ("fresh_process", "native_map"):
        raise ValueError("invalid coordinator reset strategy")
    sequence = []
    for joint in actions:
        if len(joint) != 2:
            raise ValueError("each joint action must have two participants")
        sequence.append([validate_action(action) for action in joint])
    if not sequence:
        raise ValueError("replay action sequence cannot be empty")
    folder = Path(output_dir) / f"{task_family}_{seed}_{reset_mode}"
    folder.mkdir(parents=True, exist_ok=True)
    traces, cases, env, reference_frames = [], [], None, []
    initial_sources = _source_hashes()
    try:
        for repeat in range(repeats):
            if env is None:
                env = TwoPlayerCoopEnv(task_family, seed=seed, scenario_manifest=scenario_manifest,
                                       work_root=folder / "environments")
                env.coordinator.config = replace(env.coordinator.config, reset_strategy=coordinator_reset_strategy)
            before_count = env.coordinator.consumed_joint_decisions
            before_initialization = env.initialization_joint_decisions
            result = env.reset(seed=seed)
            snapshots = [replay_snapshot(result, reset=True)]
            frames = [result.observations]
            selected_iwads = deepcopy(result.training_only_info["players_selected_iwads"])
            for joint in sequence:
                step = env.step(*joint)
                snapshots.append(replay_snapshot(step))
                frames.append(step.observations)
                if step.terminated or step.truncated:
                    break
            path = folder / f"repeat_{repeat}.json"
            path.write_text(json.dumps(snapshots, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            traces.append(snapshots)
            rgb_differences = []
            if repeat == 0:
                reference_frames = frames
            else:
                import numpy as np
                for index, (reference, current) in enumerate(zip(reference_frames, frames)):
                    if snapshots[index]["rgb_sha256"] == traces[0][index]["rgb_sha256"]:
                        continue
                    for player, (left, right) in enumerate(zip(reference, current)):
                        if left is None or right is None or left.data == right.data:
                            continue
                        left_path = folder / f"rgb_difference_repeat{repeat}_snapshot{index}_p{player}_reference.png"
                        right_path = folder / f"rgb_difference_repeat{repeat}_snapshot{index}_p{player}_replay.png"
                        save_rgb(left, left_path)
                        save_rgb(right, right_path)
                        left_array, right_array = frame_array(left), frame_array(right)
                        delta = np.abs(left_array.astype(np.int16)-right_array.astype(np.int16))
                        rgb_differences.append({"snapshot_index": index, "player": player,
                                                "changed_channels": int(np.count_nonzero(delta)),
                                                "maximum_channel_difference": int(delta.max()),
                                                "mean_absolute_channel_difference": float(delta.mean()),
                                                "reference_image": str(left_path.resolve()), "replay_image": str(right_path.resolve())})
                    # Save the first actual differing snapshot, not every frame.
                    break
            cases.append({"repeat": repeat, "snapshots": len(snapshots), "decisions": len(snapshots)-1,
                          "trace_path": str(path.resolve()), "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "state_trace_sha256": _digest([item["state_sha256"] for item in snapshots]),
                          "events_rewards_trace_sha256": _digest([item["events_rewards_sha256"] for item in snapshots]),
                          "rgb_trace_sha256": _digest([item["rgb_sha256"] for item in snapshots]),
                          "scenario_sha256": env.scenario["hashes"]["wad_sha256"],
                          "source_sha256": _source_hashes(),
                          "episode_id": result.training_only_info["episode_id"],
                          "native_work_root": str(env.coordinator.run_dir),
                          "coordinator_ledger_path": str(env.coordinator.run_dir / "dispatch_ledger.jsonl"),
                          "engine_config_paths": [str(Path(directory) / "_vizdoom.ini") for directory in env.coordinator.worker_work_dirs],
                          "worker_instances": [participant.instance_id for participant in env.coordinator.start().participants],
                          "rgb_difference_evidence": rgb_differences,
                          "players_selected_iwads": selected_iwads,
                          **interaction_counts(before_count, before_initialization,
                                               env.coordinator.consumed_joint_decisions,
                                               env.initialization_joint_decisions, len(snapshots)-1)})
            if reset_mode == "fresh_process":
                env.close()
                env = None
    finally:
        if env is not None:
            env.close()
    comparisons = [{"repeat": index, **compare_replays(traces[0], trace)} for index, trace in enumerate(traces[1:], 1)]
    state_exact = all(item["state_exact"] for item in comparisons)
    events_exact = all(item["events_rewards_exact"] for item in comparisons)
    rgb_exact = all(item["rgb_exact"] for item in comparisons)
    same_scenario = len({case["scenario_sha256"] for case in cases}) == 1
    same_iwads = all(case["players_selected_iwads"] == cases[0]["players_selected_iwads"] for case in cases)
    sources_unchanged = all(case["source_sha256"] == initial_sources for case in cases) and _source_hashes() == initial_sources
    for case in cases:
        case["engine_config_readback"] = [prediction_config_record(path) for path in case["engine_config_paths"]]
    no_prediction = all(item["cl_noprediction"] is True for case in cases for item in case["engine_config_readback"])
    report = {"schema_version": 1, "kind": "fixed_open_loop_joint_action_replay", "task_family": task_family,
              "seed": seed, "reset_mode": reset_mode, "repeats": repeats, "actions": sequence,
              "coordinator_reset_strategy": coordinator_reset_strategy,
              "actions_sha256": _digest(sequence), "cases": cases, "comparisons": comparisons,
              "state_exact": state_exact, "events_rewards_exact": events_exact, "rgb_exact": rgb_exact,
              "setup_consistency": {"scenario_bytes_match": same_scenario, "selected_iwads_match": same_iwads,
                                    "sources_unchanged": sources_unchanged, "client_prediction_disabled_readback": no_prediction},
              "status": "passed" if state_exact and events_exact and same_scenario and same_iwads and sources_unchanged and no_prediction else "failed",
              "achieved_determinism_level": "state_events_rewards_and_rgb_exact" if state_exact and events_exact and rgb_exact
                  else "state_events_rewards_exact_rgb_differs" if state_exact and events_exact else "state_or_events_reward_divergence",
              "comparison_rules": "Exact numeric/state/event/reward equality; RGB byte hashes compared separately. No tolerance or rounding hides differences.",
              "sources": ["https://vizdoom.farama.org/api/python/doom_game/",
                          "https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/1.3.2/examples/python/multiple_instances.py"]}
    path = folder / "replay_report.json"
    report["report_path"] = str(path.resolve())
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("relay", "cover"))
    parser.add_argument("--trace", type=Path, required=True, help="JSON mechanics trace containing records[].actions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reset-mode", choices=("fresh_process", "coordinated_reset"), default="fresh_process")
    parser.add_argument("--reset-strategy", choices=("fresh_process", "native_map"), default="fresh_process")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase1/replay_diagnostics"))
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    report = run_fixed_action_replay(args.task, [row["actions"] for row in trace["records"]], seed=args.seed,
                                     repeats=args.repeats, output_dir=args.output_dir, reset_mode=args.reset_mode,
                                     coordinator_reset_strategy=args.reset_strategy)
    print(json.dumps({key: report[key] for key in ("status", "state_exact", "events_rewards_exact", "rgb_exact", "report_path")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
