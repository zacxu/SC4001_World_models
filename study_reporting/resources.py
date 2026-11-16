"""Metadata-only compute accounting for the fixed-data Relay endpoint study.

Recorded unit durations are elapsed host wall times, not GPU occupancy.  The
ledger remains authoritative for logical/physical work and interruption gaps.
This reader never constructs learners, restores optimizers or runs a simulator.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA = "p57-resource-summary-v1"
CAPS = {"physical_initial_joint_transitions": 100000,
    "new_real_training_joint_transitions": 0, "new_model_optimizer_updates": 0,
    "local_PPO_cycles": 80, "local_imagined_joint_steps": 307200,
    "teacher_PPO_cycles": 50, "teacher_imagined_joint_steps": 192000,
    "buffer_imagined_joint_steps": 3840, "combined_imagined_joint_steps": 503040,
    "distillation_optimizer_steps": 200, "final_test_episodes": 80,
    "final_test_decision_cap": 32000}


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"resource {name} must be finite and nonnegative")
    return value


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _path(value):
    path = Path(value).absolute()
    if path.is_symlink():
        raise ValueError("resource evidence cannot use symbolic links")
    resolved = path.resolve()
    if path != resolved:
        raise ValueError("resource evidence path contains a symbolic link or unnormalized component")
    return resolved


def _read(path, references):
    path = _path(path)
    data = path.read_bytes()
    record = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    previous = references.setdefault(str(path), record)
    if previous != record:
        raise RuntimeError("resource source changed during inspection")
    return data


def _stats(values):
    return {"samples": len(values), "sum_seconds": sum(values) if values else None,
        "mean_seconds": statistics.mean(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "minimum_seconds": min(values) if values else None,
        "maximum_seconds": max(values) if values else None}


def _date(value):
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError("resource timestamp must be an ISO date string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("resource timestamp must be an ISO date string") from error
    if parsed.tzinfo is None:
        raise ValueError("resource timestamp must carry its time zone")
    return parsed


def _journal(stage, debit, directory, references):
    path = _path(debit["journal"])
    if not path.is_relative_to(directory):
        raise ValueError("resource journal is outside its authenticated run")
    history, malformed, starts, ends = [], 0, [], []
    if path.is_file():
        raw = _read(path, references)
        expected = debit.get("journal_sha256")
        if expected is None or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("resource attempt journal differs from the final ledger")
        for line in raw.splitlines():
            try:
                row = json.loads(line)
                if type(row) is not dict or type(row.get("event")) is not str:
                    raise ValueError("invalid journal row")
            except (ValueError, UnicodeError):
                malformed += 1
                continue
            if stage in ("buffer", "distillation") and row.get("phase") != stage:
                continue
            stamp = row.get("created_at")
            if row["event"].endswith("_started") and stamp is not None:
                starts.append((stamp, _date(stamp)))
            if not row["event"].endswith("_committed"):
                continue
            if stamp is not None:
                ends.append((stamp, _date(stamp)))
            value = row.get("elapsed_seconds", row.get("elapsed_seconds_including_checkpoint"))
            duration = None
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                duration = value
            history.append({"attempt": row.get("attempt"), "elapsed_seconds": duration,
                "unit": row.get("teacher_cycle", row.get("local_cycle", row.get("unit"))),
                "phase": row.get("phase", stage), "committed_at": row.get("created_at")})
        if malformed != debit.get("malformed_rows", 0):
            raise ValueError("resource malformed journal count differs from the final ledger")
    elif debit.get("committed_units", 0):
        # The logical checkpoint can survive a missing attempt journal.  Its
        # physical duration is unknown, rather than zero or an invented rate.
        history = []
    durations = [row["elapsed_seconds"] for row in history if row["elapsed_seconds"] is not None]
    expected_elapsed = _number(debit.get("recorded_elapsed_seconds", 0), "recorded elapsed time")
    if not math.isclose(sum(durations), expected_elapsed, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("resource recorded times do not reconcile with the ledger")
    if len(history) != debit.get("committed_units", 0) and path.is_file():
        raise ValueError("resource committed unit count differs from the final ledger")
    unknown = bool(debit.get("additional_partial_work_unknown") or not path.is_file()
        or debit.get("unmeasured_elapsed_commits", 0)
        or len(history) != debit.get("logical_completed_units", 0))
    first = min(starts, key=lambda item: item[1]) if starts else None
    last = max(ends, key=lambda item: item[1]) if ends else None
    span = (last[1] - first[1]).total_seconds() if first and last else None
    if span is not None and span < 0:
        raise ValueError("resource journal timestamps run backwards")
    return {"stage": stage, "logical_completed_units": debit.get("logical_completed_units", 0),
        "recorded_committed_units": debit.get("committed_units", 0),
        "recorded_elapsed_seconds": expected_elapsed,
        "recorded_unit_times": _stats(durations), "unit_time_history": history,
        "complete_elapsed_cost_known": not unknown,
        "recorded_first_start_at": None if first is None else first[0],
        "recorded_last_commit_at": None if last is None else last[0],
        "recorded_stage_wall_span_seconds": span,
        "stage_wall_span_scope": "first recorded start to last recorded commit; includes pauses/gaps and is not active GPU time",
        "additional_partial_work_unknown": unknown,
        "scientific_compute_debit": stage != "controller",
        "duration_scope": "completed recorded units, including checkpoint time where the original journal records it",
        "actual_GPU_occupancy_seconds": None, "process_CPU_seconds": None}


def _runtime(path, references):
    data = json.loads(_read(path, references))
    runtime = data.get("runtime")
    if type(runtime) is not dict:
        raise ValueError("training runtime manifest is missing its runtime metadata")
    return {"path": str(_path(path)), "prepared_at": data.get("created_at"), "runtime": runtime}


def _metrics(path, references):
    if not path.is_file():
        return {"path": str(path), "available": False, "valid_rows": 0, "malformed_rows": 0,
            "controller_unit_times": _stats([]), "complete_elapsed_cost_known": False}
    raw = _read(path, references)
    count, malformed, values = 0, 0, []
    by_stage = {}
    for line in raw.splitlines():
        try:
            row = json.loads(line)
            if type(row) is not dict:
                raise ValueError("not a metric record")
        except (ValueError, UnicodeError):
            malformed += 1
            continue
        count += 1
        duration = row.get("elapsed_controller_unit_seconds")
        if duration is not None:
            _number(duration, "controller unit time")
            values.append(duration)
            by_stage.setdefault(row.get("stage", "unknown"), []).append(duration)
    return {"path": str(path), "available": True, "valid_rows": count,
        "malformed_rows": malformed, "controller_unit_times": _stats(values),
        "controller_unit_times_by_stage": {k: _stats(v) for k, v in sorted(by_stage.items())},
        "complete_elapsed_cost_known": False,
        "scope": "controller timing includes child work; never add it to child-stage elapsed time"}


def _file_stat(path):
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev, "inode": stat.st_ino}


def _inventory(paths):
    """Measure retained regular-file sizes once per inode within this scope."""
    all_files = {}
    for encoded in paths:
        path = _path(encoded)
        if not path.exists():
            raise ValueError(f"resource storage source is missing: {path}")
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        for member in candidates:
            checked = _path(member)
            if checked.is_file():
                all_files[str(checked)] = _file_stat(checked)
    files = sorted(all_files.values(), key=lambda row: row["path"])
    distinct = {}
    for row in files:
        identity = (row["device"], row["inode"]) if row["inode"] else row["path"]
        distinct.setdefault(identity, row["bytes"])
    return {"files": files, "file_count": len(files), "distinct_file_count": len(distinct),
        "retained_file_bytes": sum(distinct.values()),
        "logical_path_bytes": sum(row["bytes"] for row in files),
        "inventory_sha256": _canonical(files), "historical_peak_storage_bytes": None,
        "scope": "current retained regular-file lengths; hard links counted once; filesystem allocation and historical peak are unmeasured"}


def _evaluation(info, references):
    # P5.4 records its last invocation only. Resuming after partial evaluation
    # does not recover previous invocation time; do not divide all 20 episodes
    # by a possibly partial denominator. Simulator tics are not seconds.
    result = {"completed_episodes": None, "joint_transitions": info.get("evaluation_joint_transitions"),
        "simulator_tics": info.get("evaluation_simulator_tics"),
        "elapsed_seconds": None, "episodes_per_second": None,
        "joint_transitions_per_second": None, "process_CPU_seconds": None,
        "last_invocation_elapsed_seconds": None, "completed_at_utc": None,
        "duration_scope": "only final invocation elapsed time was recorded; complete multi-invocation wall time and per-invocation episode counts are unavailable"}
    for ref in info.get("sources", []):
        path = Path(ref["path"])
        if path.name == "evaluation.json":
            report = json.loads(_read(path, references))
            if references[str(_path(path))]["sha256"] != ref["sha256"]:
                raise ValueError("evaluation changed after final ledger publication")
            episodes = report.get("episodes", [])
            result["completed_episodes"] = len(episodes)
            measured = report.get("elapsed_seconds_this_invocation")
            if measured is not None:
                result["last_invocation_elapsed_seconds"] = _number(measured, "evaluation invocation elapsed")
            stamp = report.get("completed_at_utc")
            _date(stamp)
            result["completed_at_utc"] = stamp
    return result


def validate_resource_sources(report):
    """Final publication barrier; detect changed evidence or retained file sizes."""
    for ref in report.get("evidence_references", []):
        path = _path(ref["path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref["sha256"] or len(raw) != ref["bytes"]:
            raise RuntimeError("resource evidence changed before publication")
    for inventory in report.get("storage_inventories", []):
        files = inventory["files"]
        if _canonical(files) != inventory["inventory_sha256"]:
            raise ValueError("resource storage inventory identity differs")
        if any(_file_stat(_path(row["path"])) != row for row in files):
            raise RuntimeError("retained storage changed before resource publication")
        if "roots" in inventory:
            current = _inventory(inventory["roots"])
            if current["inventory_sha256"] != inventory["inventory_sha256"]:
                raise RuntimeError("retained storage membership changed before publication")


def build_resources(context):
    """Produce P5.7 tables from already authenticated complete cohort evidence."""
    cohort, ledger, runs = (context[key] for key in ("cohort", "ledger", "runs"))
    identity = cohort["cohort_identity"]
    if ledger.get("cohort_identity") != identity or ledger.get("status") != "complete":
        raise ValueError("resource reporting requires the final complete cohort ledger")
    if len(runs) != 4 or {(r["variant"], r["master_seed"]) for r in runs} != {
            (variant, seed) for variant in ("B", "D") for seed in (0, 1)}:
        raise ValueError("resource reporting requires four distinct paired run records")
    refs, inventories, result = {}, [], []
    for run in sorted(runs, key=lambda row: (row["master_seed"], row["variant"])):
        if run.get("cohort_identity") != identity or not run.get("training_complete"):
            raise ValueError("resource run differs from the completed cohort")
        case = f"{run['variant']}/seed_{run['master_seed']}"
        entry = ledger["runs"][case]
        if entry.get("run_dir") != run["run_dir"]:
            raise ValueError("resource run and final ledger directory differ")
        directory = _path(run["run_dir"])
        stages = [_journal(name, debit, directory, refs) for name, debit in
            sorted(entry["physical_work_by_stage"].items())]
        metadata_paths = [directory / "run_manifest.json"]
        if run["variant"] == "D":
            metadata_paths += [Path(path) / "run_manifest.json" for path in
                sorted(set(run.get("stage_directories", {}).values()))]
        runtimes = [_runtime(path, refs) for path in metadata_paths]
        scientific = [s for s in stages if s["scientific_compute_debit"]]
        elapsed = sum(s["recorded_elapsed_seconds"] for s in scientific)
        known = all(s["complete_elapsed_cost_known"] for s in scientific)
        controller_row = next((s for s in stages if s["stage"] == "controller"), None)
        controller = None if controller_row is None else controller_row["recorded_elapsed_seconds"]
        if controller_row and not controller_row["complete_elapsed_cost_known"] and not controller:
            controller = None
        inventory = _inventory([directory]); inventory["roots"] = [str(directory)]
        inventories.append(inventory)
        counters = entry["counters"]
        imagined = counters["imagined_joint_transitions_total"]
        local_seconds = sum(s["recorded_elapsed_seconds"] for s in scientific if s["stage"] == "local_cycle")
        local_steps = counters["imagined_joint_transitions_local"]
        result.append({"case": case, "variant": run["variant"], "master_seed": run["master_seed"],
            "run_dir": str(directory), "runtime_manifests": runtimes,
            "physical_GPU_assignment": None,
            "physical_GPU_assignment_scope": "logical cuda:0 and GPU model are recorded; physical UUID and CUDA_VISIBLE_DEVICES were not recorded",
            "prepared_at": run["run_manifest"].get("created_at"),
            "training_started_at": None, "training_finished_at": None,
            "queue_wait_seconds": None, "end_to_end_elapsed_seconds": None,
            "recorded_scientific_unit_elapsed_seconds": elapsed,
            "complete_scientific_elapsed_cost_known": known,
            "D_controller_elapsed_seconds_informational": controller,
            "controller_elapsed_added_to_child_time": False,
            "actual_GPU_occupancy_seconds": None, "process_CPU_seconds": None,
            "peak_GPU_memory_bytes": None, "peak_CPU_memory_bytes": None,
            "historical_peak_storage_bytes": None,
            "current_retained_file_bytes": inventory["retained_file_bytes"],
            "storage_inventory_sha256": inventory["inventory_sha256"],
            "stages": stages, "metrics_history": _metrics(directory / "metrics.jsonl", refs),
            "logical_counters": counters,
            "recorded_imagined_joint_steps_per_elapsed_second": imagined / elapsed if elapsed > 0 and known else None,
            "local_recorded_imagined_joint_steps_per_elapsed_second": local_steps / local_seconds
                if local_seconds > 0 and known else None,
            "throughput_scope": "logical imagined work divided by recorded unit elapsed time; excludes preparation, prefix/I/O overhead outside timed units and unknown interrupted work",
            "evaluation": _evaluation(entry["evaluation"], refs)})
    roots = [str(_path(r["run_dir"])) for r in runs]
    union = _inventory(roots); union["roots"] = roots; inventories.append(union)
    shared_paths = [row["path"] for row in cohort.get("inputs", {}).get("bound_files", [])]
    shared = _inventory(shared_paths) if shared_paths else None
    if shared is not None:
        shared["roots"] = sorted(set(shared_paths)); inventories.append(shared)
    # Shared replay/model storage is measured once; per-arm attribution is an
    # accounting view, never four separately collected/fitted physical inputs.
    total_elapsed = sum(row["recorded_scientific_unit_elapsed_seconds"] for row in result)
    totals = {"recorded_scientific_unit_elapsed_seconds": total_elapsed,
        "sum_is_parallel_learner_elapsed_not_campaign_wall_time": True,
        "campaign_start_at": None, "campaign_end_at": None, "campaign_wall_seconds": None,
        "queue_wait_seconds": None, "actual_GPU_occupancy_seconds": None,
        "process_CPU_seconds": None, "peak_GPU_memory_bytes": None, "peak_CPU_memory_bytes": None,
        "historical_peak_storage_bytes": None,
        "current_all_run_retained_file_bytes": union["retained_file_bytes"],
        "current_shared_input_retained_file_bytes": None if shared is None else shared["retained_file_bytes"],
        "sum_run_logical_counters": ledger["sum_run_logical_counters"],
        "evaluation_joint_transitions": ledger["evaluation_joint_transitions"],
        "evaluation_known_simulator_tics": ledger["evaluation_known_simulator_tics"],
        "evaluation_unknown_dispatch_attempts": ledger["evaluation_unknown_dispatch_attempts"]}
    report = {"schema_version": SCHEMA, "status": "complete",
        "cohort_identity": identity, "per_run": result, "cohort_totals": totals,
        "shared_initial_costs": ledger["shared_initial_costs"],
        "workload_caps": CAPS.copy(), "caps_are_measured_volumes": False,
        "prefix_reconstruction_steps_measured": None,
        "compute_matched": False,
        "scheduling_policy": {"max_independent_learners": 4, "workers_per_run": 1,
            "device_selection": "user-selected physical GPU via CUDA_VISIBLE_DEVICES; logical cuda:0 per worker",
            "single_learner_distributed": False, "online_native_collectors": 0,
            "native_evaluation": "serialize or stagger if CPU capacity is limited",
            "additional_GPU_experiment_required": False},
        "conditional_planning_scenario": {"cycle_minutes": [5, 15], "D_PPO_cycles": 45,
            "D_PPO_hours": [3.75, 11.25], "distillation_and_overhead_included": False,
            "measured_runtime_guarantee": False},
        "workload_reduction_vs_week_plan": {"PPO_cycles": [420, 130],
            "online_model_updates": [2000, 0], "new_real_training_transitions": [200000, 0],
            "test_episodes": [320, 80], "measured_wall_clock_speedup": None},
        "missing_measurements": ["historical physical GPU assignment/UUID and competing GPU workloads",
            "GPU occupancy and process CPU seconds", "training/campaign start-end wall clock and queue wait",
            "peak GPU/CPU memory and historical peak storage", "native evaluation total wall time across invocations and episode throughput; final-invocation elapsed is retained when present",
            "original fitting beyond the selected 4500-update prefixes and earlier development cost",
            "full-prefix reconstruction volume"],
        "evidence_references": sorted(refs.values(), key=lambda row: row["path"]),
        "storage_inventories": inventories, "research_training_executed_by_reporter": False,
        "research_evaluation_executed_by_reporter": False,
        "measured_gpu_occupancy_inferred_from_elapsed": False}
    validate_resource_sources(report)
    return report
