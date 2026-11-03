"""Independent P2.1 payload audit; reads one episode, never starts a simulator.

This artifact intentionally lives outside execution sources frozen for the
collection campaign. It independently decodes NPZ arrays and recomputes labels;
it does not call the production collection/audit verifier. Production reader
imports are used only to exercise the fitting/imagination access guards.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import psutil


PLAYERS = ("position_x", "position_y", "position_z", "velocity_x", "velocity_y", "health", "armor", "ammo", "heading_sin", "heading_cos", "alive")
TASK = ("stage", "plate_active", "latch_active", "gate_1_open", "gate_2_open", "progress_1", "progress_2", "console_1_complete", "console_2_complete", "extraction_unlocked", "success", "failure", "milestone_count", "elapsed_decisions")
ENEMIES = ("position_x", "position_y", "health", "alive", "spawned")
TERMINAL = ("success", "death", "failure", "deadline")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_archive_paths(base, report, episodes, check):
    """Resolve successful original/recovery scenarios without assuming a tree layout."""
    expected = {episode["episode_id"] for episode in episodes}
    if not report["journal_storage"]["spooled"]:
        return {identifier: base / "runtime/scenarios" / identifier / "manifest.json" for identifier in expected}
    result = {}
    for archive in report["journal_storage"]["runtime_archives"]:
        if archive["category"] != "scenarios":
            continue
        identifier = archive["episode_id"]
        check(identifier in expected and identifier not in result, "unique successful scenario archive per committed episode")
        relative = Path(archive["archive_path"].replace("\\", "/"))
        path = (base / relative).resolve()
        check(not relative.is_absolute() and ".." not in relative.parts and path.is_relative_to(base.resolve()), "scenario archive remains inside dataset")
        result[identifier] = path / "manifest.json"
    check(set(result) == expected, "successful scenario archive coverage matches all committed episode IDs")
    return result


class Checks:
    def __init__(self):
        self.scalar_conditions = 0
        self.assertion_groups = 0

    def __call__(self, condition, label):
        values = np.asarray(condition)
        self.scalar_conditions += values.size
        self.assertion_groups += 1
        if not bool(values.all()):
            raise AssertionError(label)


def target_checks(arrays, scenario, schema, check):
    family = scenario["task_family"]
    width = float(scenario["common_x"] + 128) if family == "relay" else float(scenario["zones"]["extraction"][2])
    height = 448. if family == "relay" else 768.
    raw, norm, valid = (arrays[name] for name in ("privileged_player_state", "normalized_player_state", "privileged_validity_mask"))
    expected = np.empty_like(raw)
    for j, name in enumerate(PLAYERS):
        value = raw[:, :, j]
        expected[:, :, j] = (2*value/width-1 if name == "position_x" else 2*value/height-1 if name == "position_y"
                            else value/128 if name == "position_z" else value/32 if name.startswith("velocity")
                            else np.maximum(value, 0)/100 if name in ("health", "armor") else value/200 if name == "ammo" else value)
    check(np.isclose(norm[valid], expected[valid].astype(np.float32), rtol=0, atol=1e-7), "player target normalization")
    check(norm[~valid] == 0, "missing player target normalized slots are padding")
    angles_valid = valid[:, :, PLAYERS.index("heading_sin")] & valid[:, :, PLAYERS.index("heading_cos")]
    circle = raw[:, :, PLAYERS.index("heading_sin")]**2 + raw[:, :, PLAYERS.index("heading_cos")]**2
    check(np.isclose(circle[angles_valid], 1., rtol=0, atol=1e-10), "heading sine/cosine pair")
    raw, norm, valid = (arrays[name] for name in ("task_state", "normalized_task_state", "task_validity_mask"))
    expected = raw.copy()
    for j, name in enumerate(TASK):
        divisor = (scenario.get("dwell_decisions", 20) if name.startswith("progress_") else scenario["deadline"] if name == "elapsed_decisions"
                   else (2 if family == "relay" else 3) if name == "stage" else (1 if family == "relay" else 2) if name == "milestone_count" else 1)
        if divisor == 0:
            check(~valid[:, j], "zero-dwell relay progress must be inapplicable")
        else:
            expected[:, j] /= divisor
    check(np.isclose(norm[valid], expected[valid].astype(np.float32), rtol=0, atol=1e-7), "task target normalization")
    check(norm[~valid] == 0, "missing task target normalized slots are padding")
    for j, name in enumerate(TASK):
        inapplicable = ((family == "relay" and (name.startswith("progress_") or name.startswith("console_")))
                        or (family == "cover" and (name in ("plate_active", "latch_active") or name.startswith("gate_"))))
        if inapplicable:
            check(~valid[:, j], "inapplicable task-field mask")
    raw, norm, valid = (arrays[name] for name in ("enemy_state", "normalized_enemy_state", "enemy_validity_mask"))
    if family == "relay":
        check(~valid, "relay enemy state is inapplicable")
    else:
        expected = raw.copy()
        expected[:, :, ENEMIES.index("position_x")] = 2*raw[:, :, ENEMIES.index("position_x")]/width-1
        expected[:, :, ENEMIES.index("position_y")] = 2*raw[:, :, ENEMIES.index("position_y")]/height-1
        expected[:, :, ENEMIES.index("health")] = np.maximum(raw[:, :, ENEMIES.index("health")], 0)/20
        check(np.isclose(norm[valid], expected[valid].astype(np.float32), rtol=0, atol=1e-7), "enemy target normalization")
        for name in ("position_x", "position_y"):
            position_valid = valid[:, :, ENEMIES.index(name)]
            check(~position_valid | ((raw[:, :, ENEMIES.index("alive")] == 1) & (raw[:, :, ENEMIES.index("spawned")] == 1)), "enemy position mask needs a live spawned enemy")
    check(norm[~valid] == 0, "missing enemy target normalized slots are padding")


def archive_checks(base, report, check):
    trees = files = 0
    storage = report["journal_storage"]
    if not storage["spooled"]:
        return {"spooled": False, "scope": "dataset-local journals were not copied through a hot spool"}
    for archive in storage["runtime_archives"]:
        directory = base / archive["archive_path"]
        receipt_path = directory / "archive_receipt.json"
        check(sha(receipt_path) == archive["receipt_sha256"], "archive receipt hash")
        receipt = load(receipt_path)
        expected = {key: receipt[key] for key in ("directories", "files")}
        check(canonical(expected) == receipt["content_sha256"], "archive content identity")
        actual_files = sorted((path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file() and path != receipt_path), key=str.casefold)
        check(actual_files == sorted((row["path"] for row in receipt["files"]), key=str.casefold), "archive exact file membership")
        actual_directories = sorted((path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_dir()), key=str.casefold)
        check(actual_directories == sorted(receipt["directories"], key=str.casefold), "archive exact directory membership")
        for record in receipt["files"]:
            member = Path(record["path"])
            check(not member.is_absolute() and not member.drive and ".." not in member.parts, "portable archive member path")
            path = directory / member
            check(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "archive member bytes")
            files += 1
        check(receipt["file_count"] == len(receipt["files"]) and receipt["total_bytes"] == sum(row["bytes"] for row in receipt["files"]), "archive totals")
        check(archive["source_removed"] is True and not Path(receipt["source_path_at_collection"]).exists(), "hot runtime source removed after archive")
        trees += 1
    ledger = storage["ledger_archive"]
    check(sha(base / "training_interactions.jsonl") == ledger["sha256"], "durable ledger archive hash")
    check((base / "training_interactions.jsonl").stat().st_size == ledger["bytes"], "durable ledger archive size")
    return {"spooled": True, "verified_runtime_trees": trees, "verified_runtime_files": files, "ledger_archive_verified": True}


def audit(base: Path, expected_transitions: int | None, *, process):
    check = Checks()
    base = base.resolve(strict=True)
    report = load(base / "collection_report.json")
    manifest_path = base / "replay/manifest.json"
    manifest = load(manifest_path)
    specification = load(base / "collection_spec.json")
    metadata, schema = manifest["metadata"], manifest["schema"]
    family, seed = metadata["task_family"], metadata["run_seed"]
    check(family in ("relay", "cover"), "task family")
    check(report["status"] == "complete_initial_replay" and manifest["status"] == "sealed", "complete sealed replay")
    check(report["task_family"] == family and report["run_seed"] == seed, "dataset case identity")
    check(sha(manifest_path) == report["dataset_manifest_sha256"], "dataset manifest hash")
    check((base / "replay/manifest.sha256").read_text().split()[0] == sha(manifest_path), "dataset manifest receipt")
    check(sha(base / "collection_spec.json") == report["collection_spec_sha256"], "collection specification hash")
    check(manifest["dataset_identity_sha256"] == canonical({key: value for key, value in manifest.items() if key not in ("sealed_at_utc", "dataset_identity_sha256")}), "dataset logical identity")
    check(manifest["dataset_identity_sha256"] == report["dataset_identity_sha256"], "reported dataset identity")
    check(tuple(schema["player_fields"]) == PLAYERS and tuple(schema["task_fields"]) == TASK and tuple(schema["enemy_fields"]) == ENEMIES, "independently fixed target field orders")
    check(metadata["data_source"] == "real_environment" and metadata["paired_variants"] == ["B", "D"], "real shared initialization")
    check(metadata["origin_provenance"] == specification["origin_provenance"], "frozen collection procedure identity")
    eligible = metadata["budget_scope"] == "approved_initial"
    check(metadata["learning_eligible"] is eligible, "preflight cannot be learning data")
    check(metadata["scripted_policy"]["provides_imitation_targets"] is False, "script actions are not imitation targets")
    event_fields, reasons = schema["event_fields"], schema["termination_reasons"]
    scenario_paths = scenario_archive_paths(base, report, manifest["episodes"], check)
    kinds, endings, partitions, partition_episodes, events, layouts = (Counter() for _ in range(6))
    nstates = ntransitions = nchunks = 0
    seen = set()
    peak_rss = process.memory_info().rss
    for episode in manifest["episodes"]:
        identifier, length = episode["episode_id"], episode["valid_length"]
        check(identifier not in seen, "unique episode ID")
        seen.add(identifier)
        check(sha(base / "replay" / episode["manifest_path"]) == episode["manifest_sha256"], "episode manifest hash")
        check(episode["split"] == "train" and episode["split_provenance"]["purpose"] == "training" and episode["split_provenance"]["learning_eligible"] is True, "training geometry only")
        check(episode["learning_eligible"] is eligible, "episode learning eligibility agrees with store")
        assignment = "diagnostic" if int.from_bytes(hashlib.sha256(("real-replay-v1|" + identifier).encode()).digest()[:8], "big") % 10 == 0 else "fit"
        check(episode["partition"] == assignment, "deterministic episode-ID split")
        state_offset = transition_offset = 0
        members = []
        for chunk in episode["chunks"]:
            path = base / "replay" / chunk["path"]
            check(path.stat().st_size == chunk["bytes"] and sha(path) == chunk["sha256"], "chunk size/hash")
            check(chunk["state_start"] == state_offset and chunk["transition_start"] == transition_offset, "no duplicated or missing chunk rows")
            with np.load(path, allow_pickle=False) as content:
                arrays = {key: content[key] for key in content.files}
            check(set(arrays) == set(schema["state_arrays"] + schema["transition_arrays"]), "chunk array membership")
            for name, value in arrays.items():
                specification_row = schema["array_specs"][name]
                count = chunk["state_count"] if specification_row["alignment"] == "state" else chunk["transition_count"]
                check(value.dtype.str == specification_row["dtype"] and value.shape == (count, *specification_row["row_shape"]), "typed array shape: " + name)
                if value.dtype.kind == "f": check(np.isfinite(value), "finite targets/rewards: " + name)
            members.append(arrays)
            state_offset += chunk["state_count"]
            transition_offset += chunk["transition_count"]
            nchunks += 1
        arrays = {key: np.concatenate([part[key] for part in members], axis=0) for key in members[0]}
        peak_rss = max(peak_rss, process.memory_info().rss)
        check(len(arrays["rgb"]) == length + 1 == episode["observation_count"], "one observation sequence N+1")
        check(arrays["rgb"].dtype == np.uint8 and arrays["rgb"].shape == (length+1, 2, 3, 96, 128), "RGB storage contract")
        check(arrays["rgb"][~arrays["image_valid"]] == 0, "unavailable images contain masked storage padding only")
        check(arrays["observation_index"] == np.arange(length), "source observation indices")
        check(arrays["next_observation_index"] == np.arange(1, length+1), "next observation indices")
        check(arrays["transition_id"] == ntransitions + np.arange(length), "global transition IDs")
        check(arrays["decision_index"] == np.arange(length), "decision indices")
        check(arrays["episode_id"] == identifier, "per-transition episode identity")
        check(arrays["collector_policy_version"] == episode["policy_version"], "per-transition collector version")
        check(arrays["state_engine_tic"][0] == episode["initial_tic"], "initial state tic")
        check(arrays["state_engine_tic"][:-1] == arrays["start_tic"], "state o_t corresponds to source tic")
        check(arrays["state_engine_tic"][1:] == arrays["end_tic"], "state o_(t+1) corresponds to end tic")
        check(arrays["end_tic"] - arrays["start_tic"] == arrays["actual_tics"], "actual native tics")
        check((arrays["actual_tics"] >= 1) & (arrays["actual_tics"] <= episode["action_repeat"]), "native tic range")
        check(arrays["actual_tics"][~arrays["terminated"]] == episode["action_repeat"], "only terminals may shorten action repeat")
        check(arrays["cumulative_event_counters"] >= 0, "nonnegative event counters")
        check(arrays["event_deltas"] >= 0, "nonnegative event deltas")
        check(np.diff(arrays["cumulative_event_counters"], axis=0) == arrays["event_deltas"], "authoritative cumulative event differences")
        check(arrays["next_obs_valid"] == arrays["image_valid"][1:], "next-image mask alignment")
        check(arrays["actions"] < 12, "declared primitive actions")
        check(arrays["termination_reason"] < len(reasons), "known boundary reasons")
        labels = [reasons[int(code)] for code in arrays["termination_reason"]]
        terminal = np.array([label in TERMINAL for label in labels])
        cutoff = np.array([label == "collection_cutoff" for label in labels])
        check(arrays["terminated"] == terminal, "true terminals")
        check(arrays["truncated"] == cutoff, "external truncations")
        check(not np.any(terminal[:-1] | cutoff[:-1]) and bool(terminal[-1] | cutoff[-1]), "only final transition ends episode")
        check(labels[-1] == episode["end_reason"], "episode terminal metadata")
        check(arrays["image_valid"][1:][~terminal], "nonterminal next images are valid")
        scenario = load(scenario_paths[identifier])
        check(scenario["hashes"]["wad_sha256"] == episode["scenario_hash"] and scenario["task_family"] == family, "scenario identity")
        milestone = ("latch_activations",) if family == "relay" else ("console_1_completions", "console_2_completions")
        bonus = sum(arrays["event_deltas"][:, event_fields.index(name)] for name in milestone) * (.1 if family == "relay" else .05)
        outcomes = np.array([1. if label == "success" else -1. if label in TERMINAL else 0. for label in labels])
        check(np.isclose(arrays["team_reward"], -.001 + bonus + outcomes, rtol=0, atol=1e-12), "shared reward from authoritative events/outcomes")
        target_checks(arrays, scenario, schema, check)
        kinds[episode["collector_kind"]] += length
        endings[episode["end_reason"]] += 1
        partitions[assignment] += length
        partition_episodes[assignment] += 1
        layouts[episode["layout_id"]] += length
        events.update({name: int(value) for name, value in zip(event_fields, arrays["event_deltas"].sum(axis=0))})
        nstates += length+1
        ntransitions += length
        del arrays, members
    check(ntransitions == manifest["transition_count"] == report["target_transitions"], "dataset totals")
    if expected_transitions is not None: check(ntransitions == expected_transitions, "requested dataset size")
    check(nstates == ntransitions + len(manifest["episodes"]), "one reset observation per episode")
    check(kinds == Counter(random=round(ntransitions*.7), scripted=round(ntransitions*.3)), "exact transition-level 70/30 mixture")
    check(dict(kinds) == report["collector_counts"], "reported collector mixture")
    check(dict(events) == report["event_counts"], "reported authoritative event totals")
    if eligible:
        check(partition_episodes["fit"] > 0 and partition_episodes["diagnostic"] > 0, "full initialization includes both model partitions")
        check(len(layouts) == 6, "all six training geometries visited")
    for variant in ("B", "D"):
        path = base / "bindings" / f"{variant}.json"
        pointer = load(path)
        check(sha(path) == report["bindings"][variant]["sha256"], "branch pointer hash")
        check(pointer["variant"] == variant and not Path(pointer["dataset_path"]).is_absolute(), "portable paired pointer")
        check((path.parent / pointer["dataset_path"].replace("\\", "/")).resolve() == (base / "replay").resolve(), "same physical initialization")
        check(pointer["manifest_sha256"] == sha(manifest_path) and pointer["dataset_identity_sha256"] == manifest["dataset_identity_sha256"], "paired immutable identity")
        check(pointer["attributed_initial_joint_transitions"] == ntransitions and pointer["new_physical_transitions"] == 0, "full per-arm charge without repeated collection")
    from world_models.data.replay import ReplayDataset
    reader = ReplayDataset(base / "replay", verify=False, cache_chunks=1)
    rejected = 0
    for episode in reader.episodes:
        for purpose in ("world_model_fit", "imagination"):
            excluded = not eligible or episode["partition"] == "diagnostic"
            try:
                reader.load_episode(episode["episode_id"], purpose=purpose)
            except ValueError:
                check(excluded, "unexpected fitting/imagination exclusion")
                rejected += 1
            else:
                check(not excluded, "model-diagnostic/preflight data leaked to fitting/imagination")
    archives = archive_checks(base, report, check)
    return {"status": "passed", "dataset_path": str(base), "task_family": family, "run_seed": seed,
            "transition_count": ntransitions, "observation_count": nstates, "episode_count": len(seen), "chunk_count": nchunks,
            "collector_counts": dict(kinds), "end_reason_counts": dict(endings), "event_counts": dict(events),
            "partition_transition_counts": dict(partitions), "partition_episode_counts": dict(partition_episodes),
            "layout_transitions": dict(layouts), "learning_eligible": eligible, "learning_access_rejections_checked": rejected,
            "archives": archives, "assertion_groups": check.assertion_groups, "scalar_conditions_checked": int(check.scalar_conditions),
            "peak_process_rss_bytes_observed_after_episode_load": peak_rss,
            "scope": "Direct typed NPZ/hash/index/normalization/reward/mask/partition/guard/archive audit. Native executed-action/ACK audit is separate."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-transitions", type=int)
    parser.add_argument("--cpu-ids", type=int, nargs=4, default=[0, 1, 2, 3])
    args = parser.parse_args(argv)
    if args.output.exists(): parser.error("use a new output report; preserve previous evidence")
    process = psutil.Process()
    original_affinity = process.cpu_affinity()
    if len(set(args.cpu_ids)) != 4 or not set(args.cpu_ids) <= set(original_affinity):
        parser.error("four unique allowed logical CPU IDs are required")
    process.cpu_affinity(args.cpu_ids)
    started = time.perf_counter()
    report = {"status": "running", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "audit_script_sha256": sha(Path(__file__)), "cpu_ids": process.cpu_affinity(),
              "thread_environment": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
              "python": platform.python_version(), "numpy": np.__version__, "datasets": []}
    try:
        for path in args.datasets:
            report["datasets"].append(audit(path, args.expected_transitions, process=process))
        report["status"] = "passed"
        report["total_transitions_checked"] = sum(row["transition_count"] for row in report["datasets"])
        report["total_scalar_conditions_checked"] = sum(row["scalar_conditions_checked"] for row in report["datasets"])
        code = 0
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        code = 1
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        process.cpu_affinity(original_affinity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "output": str(args.output), "total_transitions_checked": report.get("total_transitions_checked"), "error": report.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
