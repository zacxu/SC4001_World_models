"""Read-only scientific accounting and append-only, receipt-bound snapshots.

Committed checkpoints establish logical work. Attempt journals establish only
recorded physical work: interrupted/unrecorded optimization remains unknown.
Snapshots contain cumulative state and must never be summed as extra work.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_study.evidence import ROOT, read_report, safe_path, timestamp, write_report


SCHEMA = "p53-complete-ledger-v1"
_SNAPSHOT = re.compile(r"snapshot_([0-9]{8})\.json")


def _inspect_run(directory):
    from .cohort import audit_run, inspect_run
    audit_run(directory)
    return inspect_run(directory)


def _inspect_cohort(path):
    from .cohort import inspect_cohort
    return inspect_cohort(path)


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"ledger {name} must be a nonnegative integer")
    return value


def _file(path):
    path = safe_path(path)
    if not path.is_file():
        raise ValueError(f"ledger source is missing: {path}")
    return {"path": str(path), "sha256": file_sha256(path)}


def _member(directory, encoded):
    if type(encoded) is not str or not encoded or "\0" in encoded:
        raise ValueError("ledger member path is invalid")
    path = safe_path(directory / encoded.replace("\\", "/"))
    if not path.is_relative_to(directory):
        raise ValueError("ledger source member escapes its directory")
    return path


def _collection(binding, seed):
    """Verify available original collection costs; missing history stays null.

    The sealed replay proves 50k valid transitions even if its separate original
    collection journal was not transferred. Published-step tics exclude reset
    synchronization; reset/failed-dispatch tic cost is not invented.
    """
    result = {"real_joint_transitions_total": 50000, "valid_for_replay": 50000,
        "real_initial": 50000, "real_local_collection": 0, "real_teacher_collection": 0,
        "real_simulator_tics_total": None, "real_simulator_tics_scope": "published initial replay steps only",
        "physical_joint_decisions_including_initialization": None,
        "physical_joint_decisions_lower_bound": 50000, "initialization_joint_decisions": None,
        "initialization_simulator_tics": None, "failed_dispatch_attempts": None,
        "accounting_exact": False, "sources": [],
        "unavailable_costs": ["original collection reset/failed physical work and simulator tics are unmeasured"]}
    replay = safe_path(binding["replay_path"])
    report_path = replay.parent / "collection_report.json"
    if not report_path.exists():
        return result
    source = _file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    accounting = report.get("interaction_accounting", {})
    expected_manifest = binding["provenance"]["replays"][0]["manifest_sha256"]
    if (report.get("status") != "complete_initial_replay" or report.get("task_family") != "relay"
            or type(report.get("run_seed")) is not int or report["run_seed"] != seed
            or report.get("target_transitions") != 50000
            or report.get("dataset_manifest_sha256") != expected_manifest
            or file_sha256(replay / "manifest.json") != expected_manifest):
        raise ValueError("initial collection report differs from the registered replay")
    path = _member(replay.parent, accounting.get("ledger_path"))
    journal = _file(path)
    if journal["sha256"] != accounting.get("ledger_sha256"):
        raise ValueError("original real interaction ledger receipt differs")
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    if (not rows or [row.get("sequence") for row in rows] != list(range(len(rows)))
            or any(type(row.get("sequence")) is not int or row.get("task_family") != "relay"
                   or type(row.get("run_seed")) is not int or row["run_seed"] != seed for row in rows)):
        raise ValueError("original real interaction ledger identity or sequence differs")
    totals = {name: 0 for name in ("physical_delta", "initialization_delta", "published_delta", "failed_delta")}
    tics, committed = 0, 0
    for row in rows:
        values = {name: _integer(row.get(name, 0), name) for name in totals}
        if values["physical_delta"] != sum(values[name] for name in
                ("initialization_delta", "published_delta", "failed_delta")):
            raise ValueError("original physical interaction debit does not reconcile")
        for name in totals:
            totals[name] += values[name]
        if row.get("event") == "transition":
            if values["published_delta"] != 1 or _integer(row.get("actual_tics"), "initial actual tics") < 1:
                raise ValueError("initial joint transition was not charged once")
            tics += _integer(row.get("actual_tics"), "initial actual tics")
        elif values["published_delta"]:
            raise ValueError("published initial work has no transition record")
        if row.get("event") == "episode_committed":
            committed += _integer(row.get("valid_length"), "initial committed length")
    expected = {"physical_joint_decisions_including_initialization": totals["physical_delta"],
        "initialization_joint_decisions": totals["initialization_delta"],
        "published_real_joint_transitions": totals["published_delta"],
        "failed_dispatch_attempts": totals["failed_delta"],
        "published_real_simulator_tics_total": tics, "replay_committed_joint_transitions": committed}
    if (any(type(accounting.get(name)) is not int or accounting[name] != value for name, value in expected.items())
            or totals["published_delta"] != 50000 or committed != 50000 or totals["failed_delta"]):
        raise ValueError("initial collection physical counters do not conserve replay work")
    if file_sha256(path) != journal["sha256"] or file_sha256(report_path) != source["sha256"]:
        raise RuntimeError("initial accounting changed while reading; retry inspection")
    return {**result, "real_joint_transitions_total": totals["published_delta"] + totals["failed_delta"],
        "real_simulator_tics_total": tics,
        "physical_joint_decisions_including_initialization": totals["physical_delta"],
        "physical_joint_decisions_lower_bound": totals["physical_delta"],
        "initialization_joint_decisions": totals["initialization_delta"], "failed_dispatch_attempts": totals["failed_delta"],
        "sources": [source, journal], "published_step_accounting_verified": True,
        "unavailable_costs": ["reset synchronization simulator tics are not recorded by the original collection ledger"]}


def _journal(directory, stage, logical):
    # Reuse the existing tolerant parser. It preserves torn rows/orphan commits
    # as unknown physical costs without repairing scientific history.
    from world_models_campaign.stages import _physical
    result = _physical(safe_path(directory), stage)
    path = safe_path(result["journal"])
    prefix = None
    if path.is_file():
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != result.get("journal_sha256"):
            raise RuntimeError("physical attempt journal advanced while reading; retry inspection")
        prefix = {"path": str(path), "bytes": len(raw), "sha256": digest}
    elif logical:
        result["partial_interrupted_compute"] = "unknown; logical checkpoint exists without its physical attempt journal"
    committed = result["committed_units"]
    unknown = bool(result["unmatched_attempts"] or result["malformed_rows"]
                   or result["unattributed_orphan_commits"] or result["unmeasured_elapsed_commits"]
                   or committed != logical or (logical and prefix is None))
    actor_per_unit = 16 if stage in ("teacher", "local_cycle") else 0
    distillation_per_unit = 1 if stage == "distillation" else 0
    return {**result, "logical_completed_units": logical,
        "recorded_committed_actor_optimizer_steps": actor_per_unit * committed,
        "recorded_committed_critic_optimizer_steps": actor_per_unit * committed,
        "recorded_committed_distillation_optimizer_steps": distillation_per_unit * committed,
        "physical_actor_optimizer_steps_lower_bound": actor_per_unit * max(logical, committed),
        "physical_critic_optimizer_steps_lower_bound": actor_per_unit * max(logical, committed),
        "physical_distillation_optimizer_steps_lower_bound": distillation_per_unit * max(logical, committed),
        "additional_replayed_optimizer_steps": None if unknown else 0,
        "additional_completed_physical_units_lower_bound": 0,
        "recorded_commits_not_yet_logically_attributed": max(0, committed - logical),
        "commit_mismatch_policy": "a child may lead its parent; unmatched commits are not automatically replayed compute",
        "additional_partial_optimizer_work": None if unknown else 0,
        "additional_partial_work_unknown": unknown,
        "logical_work_missing_commit_journal_rows": max(0, logical - committed),
        "source_prefix": prefix}


def _physical_work(info):
    if info["variant"] == "B":
        return {"local_cycle": _journal(info["run_dir"], "local_cycle", info["local_cycles_completed"])}
    targets = {"teacher": info["teacher_cycles_completed"], "buffer": info["trajectories_completed"],
        "distillation": info["distillation_optimizer_steps"], "local_cycle": info["local_cycles_completed"]}
    rows = {}
    for name, logical in targets.items():
        directory = info["stage_directories"][name]
        rows[name] = _journal(directory, name, logical)
    # Parent orchestration attempts reconcile children and are not added as
    # another set of scientific optimizer updates.
    logical_parent_units = sum(targets.values()) + int(info["phase"] == "complete")
    if "units_completed" in info and info["units_completed"] != logical_parent_units:
        raise ValueError("D controller units disagree with its durable scientific stages")
    rows["controller"] = _journal(info["run_dir"], "controller", logical_parent_units)
    rows["controller"]["scientific_compute_debit"] = False
    return rows


def _evaluation(directory, cohort_identity):
    path = directory / "endpoint_test_claim.json"
    if not path.exists() and not path.with_suffix(".sha256").exists():
        return {"status": "not_started", "evaluation_joint_transitions": 0,
            "evaluation_simulator_tics": 0, "completed_episode_joint_transitions": 0,
            "unknown_dispatch_attempts": 0, "unknown_simulator_tics": False,
            "evaluation_enters_training": False, "sources": []}
    claim = read_report(path)
    if (claim.get("schema_version") != "p54-single-final-test-claim-v1"
            or safe_path(claim.get("run_dir", "")) != directory
            or claim.get("cohort_identity") != cohort_identity
            or any(type(claim.get(key)) is not str or re.fullmatch(r"[a-f0-9]{64}", claim[key]) is None
                   for key in ("bank_sha256", "final_actor_identity"))):
        raise ValueError("endpoint evaluation claim belongs to a different training run")
    output = safe_path(claim["output_dir"])
    report_path = output / "evaluation.json"
    from .evaluation import load_evaluation, load_partial_accounting
    if report_path.with_suffix(".sha256").is_file():
        report = load_evaluation(report_path, require_native=True)
        accounting, status, sources = report["interaction_accounting"], report["status"], [_file(path), _file(report_path)]
    else:
        manifest_path = output / "evaluation_manifest.json"
        manifest = read_report(manifest_path)
        if (manifest.get("run_dir") != str(directory) or manifest.get("native_backend") is not True
                or manifest.get("native_engine") is not True or any(manifest.get(key) != claim[key]
                    for key in ("cohort_identity", "bank_sha256", "final_actor_identity"))):
            raise ValueError("unfinished evaluation manifest differs from its registered final actor claim")
        report = load_partial_accounting(output)
        accounting = report.get("interaction_accounting", report)
        status, sources = "incomplete", [_file(path), _file(manifest_path)]
    if type(accounting) is not dict:
        raise ValueError("evaluation accounting is not an authenticated object")
    known_transitions = _integer(accounting.get("published_real_joint_transitions"), "evaluation published transitions")
    known_tics = _integer(accounting.get("known_simulator_tics_total"), "evaluation known tics")
    unknown = _integer(accounting.get("unknown_dispatch_attempts"), "evaluation unknown dispatch attempts")
    if type(accounting.get("unknown_simulator_tics")) is not bool:
        raise ValueError("evaluation uncertainty must be explicit")
    prefixes = []
    for record in accounting.get("journal_prefixes", []):
        journal = _member(output, record["path"])
        length = _integer(record.get("prefix_bytes"), "evaluation journal prefix length")
        with journal.open("rb") as stream:
            raw = stream.read(length)
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != length or digest != record.get("sha256"):
            raise RuntimeError("evaluation attempt journal changed while accounting; retry inspection")
        prefixes.append({"path": str(journal), "bytes": length, "sha256": digest})
    return {"status": status, "evaluation_joint_transitions": known_transitions,
        "evaluation_simulator_tics": known_tics,
        "completed_episode_joint_transitions": _integer(accounting.get("completed_episode_joint_transitions", 0), "completed evaluation steps"),
        "unknown_dispatch_attempts": unknown, "unknown_simulator_tics": accounting["unknown_simulator_tics"],
        "evaluation_enters_training": False, "accounting": deepcopy(accounting), "sources": sources,
        "source_prefixes": prefixes}


def _snapshot(info):
    seed = info.get("master_seed", info.get("run_seed"))
    if type(seed) is not int or seed not in (0, 1) or info.get("variant") not in ("B", "D"):
        raise ValueError("ledger only accepts authenticated Relay B/D seeds 0 and 1")
    directory = safe_path(info["run_dir"])
    local, teacher, buffer = (_integer(info[name], name) for name in
        ("imagined_joint_steps_local", "imagined_joint_steps_teacher", "imagined_joint_steps_buffer"))
    counters = {"world_model_optimizer_steps_initial": 4500, "world_model_optimizer_steps_online": 0,
        "local_PPO_cycles": _integer(info["local_cycles_completed"], "local cycles"),
        "local_actor_steps": _integer(info["local_actor_optimizer_steps"], "local actor steps"),
        "local_critic_steps": _integer(info["local_critic_optimizer_steps"], "local critic steps"),
        "teacher_PPO_cycles": _integer(info["teacher_cycles_completed"], "teacher cycles"),
        "teacher_actor_steps": _integer(info["teacher_actor_optimizer_steps"], "teacher actor steps"),
        "teacher_critic_steps": _integer(info["teacher_critic_optimizer_steps"], "teacher critic steps"),
        "distillation_optimizer_steps": _integer(info["distillation_optimizer_steps"], "distillation steps"),
        "distillation_sequences_processed": 32 * _integer(info["distillation_optimizer_steps"], "distillation steps"),
        "warm_start_buffer_trajectories": _integer(info["trajectories_completed"], "buffer trajectories"),
        "imagined_joint_transitions_local": local, "imagined_joint_transitions_teacher": teacher,
        "imagined_joint_transitions_buffer_generation": buffer,
        "imagined_joint_transitions_total": local + teacher + buffer,
        "local_player_decisions": 2 * local, "new_environment_interactions": 0, "new_model_updates": 0}
    if (counters["local_actor_steps"] != 16 * counters["local_PPO_cycles"]
            or counters["local_critic_steps"] != counters["local_actor_steps"]
            or counters["teacher_actor_steps"] != 16 * counters["teacher_PPO_cycles"]
            or counters["teacher_critic_steps"] != counters["teacher_actor_steps"]
            or counters["local_PPO_cycles"] > 20 or counters["teacher_PPO_cycles"] > 25
            or counters["distillation_optimizer_steps"] > 100 or counters["warm_start_buffer_trajectories"] > 128
            or any(_integer(info.get(key, 0), key) != 0 for key in ("new_environment_interactions", "new_model_updates"))
            or (info["variant"] == "B" and (teacher or buffer or counters["teacher_PPO_cycles"]
                or counters["distillation_optimizer_steps"] or counters["warm_start_buffer_trajectories"]))):
        raise ValueError("ledger scientific counters differ from the fixed B/D schedule")
    binding = info["initial_binding"]
    initial = _collection(binding, seed)
    return {"schema_version": SCHEMA, "scope": "one prospectively registered cumulative run boundary",
        "subject_identity": info["run_identity"], "run_identity": info["run_identity"],
        "cohort_identity": info["cohort_identity"], "run_dir": str(directory), "variant": info["variant"],
        "master_seed": seed, "status": info["status"], "phase": info["phase"],
        "last_checkpoint": info.get("last_checkpoint"), "initial_collection": initial,
        "counters": counters, "physical_work_by_stage": _physical_work(info),
        "evaluation": _evaluation(directory, info["cohort_identity"]),
        "historical_and_development_costs": {"selected_model_prefix_optimizer_steps": 4500,
            "selected_prefix_is_total_historical_fit": False,
            "original_fitting_beyond_selected_prefix_optimizer_steps": None,
            "original_fitting_replayed_physical_work": None, "other_cover_training_and_collection": None,
            "prior_policy_development_training": None,
            "status": "unmeasured here; preserve original fitting, Cover and development artifacts separately",
            "included_in_comparative_policy_axes": False},
        "counting_rules": {"joint_transition": "one synchronized two-player decision counted once",
            "replay_reuse_adds_new_real_data": False, "buffer_generation_counted_once": True,
            "evaluation_separate_from_training": True, "reset_overhead_separate": True,
            "snapshot_is_cumulative_not_an_incremental_debit": True,
            "unknown_physical_work_is_zero": False},
        "research_training_executed_by_ledger": False, "research_evaluation_executed_by_ledger": False}


def _numeric(value, prefix=""):
    if type(value) is dict:
        for key, member in value.items():
            yield from _numeric(member, f"{prefix}/{key}")
    elif type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError("ledger numeric counters must be finite")
        yield prefix, value


def _monotonic(document):
    """Only cumulative debits increase; pending gaps can legitimately resolve."""
    cumulative = {"evaluation_joint_transitions", "evaluation_simulator_tics", "evaluation_known_simulator_tics",
        "completed_episode_joint_transitions", "real_joint_transitions_total", "valid_for_replay", "real_initial",
        "real_local_collection", "real_teacher_collection", "real_simulator_tics_total",
        "physical_joint_decisions_including_initialization", "physical_joint_decisions_lower_bound",
        "initialization_joint_decisions", "initialization_simulator_tics", "failed_dispatch_attempts",
        "started_units", "committed_units", "recorded_elapsed_seconds", "prepared_run_attributed_joint_transitions",
        "recorded_committed_actor_optimizer_steps", "recorded_committed_critic_optimizer_steps",
        "recorded_committed_distillation_optimizer_steps", "physical_actor_optimizer_steps_lower_bound",
        "physical_critic_optimizer_steps_lower_bound", "physical_distillation_optimizer_steps_lower_bound",
        "prepared_run_attributed_selected_model_steps"}
    for name, number in _numeric(document):
        parts = name.split("/")
        if "counters" in parts or "sum_run_logical_counters" in parts or parts[-1] in cumulative:
            yield name, number


def _validate_static_sources(value):
    if type(value) is dict:
        if set(value) == {"path", "sha256"}:
            path = safe_path(value["path"])
            if not path.is_file() or file_sha256(path) != value["sha256"]:
                raise ValueError("previous immutable accounting source changed")
        for member in value.values():
            _validate_static_sources(member)
    elif type(value) is list:
        for member in value:
            _validate_static_sources(member)


def _prefixes(value):
    if type(value) is dict:
        if set(value) == {"path", "bytes", "sha256"}:
            yield value
        for member in value.values():
            yield from _prefixes(member)
    elif type(value) is list:
        for member in value:
            yield from _prefixes(member)


def _validate_prefixes(previous):
    for record in _prefixes(previous):
        path, size = safe_path(record["path"]), _integer(record["bytes"], "journal byte prefix")
        if not path.is_file():
            raise ValueError("previous physical journal disappeared")
        with path.open("rb") as stream:
            raw = stream.read(size)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("append-only physical journal was truncated or rewritten")


@contextmanager
def _lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / ".ledger.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0"); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError("another process is writing this scientific ledger") from error
    try:
        yield
    finally:
        if os.name == "nt":
            import msvcrt
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _append(document, output_dir):
    output = safe_path(output_dir)
    protected = [ROOT / name for name in ("src", "scripts", "tests", "configs", "scenarios", "data")]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError("ledger snapshots must be separate from sources and training inputs")
    with _lock(output):
        # A receipt is the commit marker. A killed publisher can leave only its
        # JSON body; preserve those bytes before reconstructing a fresh snapshot
        # from the authenticated scientific inputs. Never repair a present bad
        # receipt, or treat an orphan receipt as permission to replace history.
        for marker in output.glob("snapshot_*.sha256"):
            body = marker.with_suffix(".json")
            if _SNAPSHOT.fullmatch(body.name) and not body.is_file():
                raise ValueError("scientific ledger has an orphan snapshot receipt")
        for path in sorted(output.glob("snapshot_*.json")):
            if not _SNAPSHOT.fullmatch(path.name):
                continue
            safe_path(path)
            marker = path.with_suffix(".sha256")
            if marker.exists() or marker.is_symlink():
                read_report(path)
                continue
            archive = safe_path(output / "uncommitted_publications")
            if not archive.is_relative_to(output):
                raise ValueError("uncommitted ledger archive escaped its output")
            archive.mkdir(exist_ok=True)
            target = archive / f"{path.stem}_{uuid.uuid4().hex}.json"
            if target.exists():
                raise FileExistsError("uncommitted ledger archive needs a fresh path")
            path.rename(target)
        archives = [_file(path) for path in sorted((output / "uncommitted_publications").glob("*.json"))]
        candidates = sorted(path for path in output.glob("snapshot_*.json") if _SNAPSHOT.fullmatch(path.name))
        previous, previous_path = None, None
        for sequence, path in enumerate(candidates, start=1):
            row = read_report(path)
            if (int(_SNAPSHOT.fullmatch(path.name)[1]) != sequence or row.get("sequence") != sequence
                    or row.get("schema_version") != "p53-durable-snapshot-v1"
                    or row.get("subject_identity") != document["subject_identity"]
                    or row.get("previous") != (None if previous_path is None else
                        {"path": previous_path.name, "sha256": file_sha256(previous_path)})):
                raise ValueError("scientific ledger snapshot chain was altered or omitted")
            _validate_prefixes(row["snapshot"])
            _validate_static_sources(row["snapshot"])
            _validate_static_sources(row.get("preserved_uncommitted_publications", []))
            previous, previous_path = row, path
        if previous:
            old, new = dict(_monotonic(previous["snapshot"])), dict(_monotonic(document))
            # Timestamps/sequence are outside snapshot data. Scientific counters,
            # physical elapsed work and known evaluation costs may only increase.
            for name, value in old.items():
                if name not in new or new[name] < value:
                    raise ValueError(f"scientific ledger counter regressed: {name}")
            if canonical_hash(previous["snapshot"]) == canonical_hash(document):
                return {**document, "snapshot_path": str(previous_path), "snapshot_sequence": previous["sequence"],
                    "new_snapshot_written": False}
        sequence = len(candidates) + 1
        path = output / f"snapshot_{sequence:08d}.json"
        write_report(path, {"schema_version": "p53-durable-snapshot-v1", "sequence": sequence,
            "recorded_at": timestamp(), "subject_identity": document["subject_identity"],
            "previous": None if previous_path is None else {"path": previous_path.name, "sha256": file_sha256(previous_path)},
            "preserved_uncommitted_publications": archives,
            "snapshot": document})
    return {**document, "snapshot_path": str(path), "snapshot_sequence": sequence, "new_snapshot_written": True}


def snapshot_run(run_dir, output_dir=None):
    """Inspect committed scientific work; optionally append a durable snapshot."""
    info = _inspect_run(safe_path(run_dir))
    result = _snapshot(info)
    return result if output_dir is None else _append(result, output_dir)


def summarize_cohort(cohort, output_dir=None):
    """Keep shared physical inputs distinct from four-run attribution totals."""
    cohort = _inspect_cohort(safe_path(cohort))
    runs, missing = [], []
    for slot in cohort["run_matrix"]:
        directory = safe_path(slot["run_dir"])
        if not directory.exists():
            missing.append(slot["case"])
        else:
            runs.append(snapshot_run(directory))
    inputs = cohort["inputs"]
    for name, target in (("physical_initial_joint_transitions", 100000),
                         ("physical_selected_model_optimizer_updates", 9000)):
        if type(inputs.get(name)) is not int or inputs[name] != target:
            raise ValueError("cohort shared initial work differs from the registered two-seed inputs")
    counters = {name: sum(run["counters"][name] for run in runs) for name in runs[0]["counters"]} if runs else {}
    initial_by_seed = {}
    for run in runs:
        key, value = str(run["master_seed"]), run["initial_collection"]
        if key in initial_by_seed and canonical_hash(initial_by_seed[key]) != canonical_hash(value):
            raise ValueError("paired runs disagree about their one shared physical initial collection")
        initial_by_seed[key] = value
    all_initial_known = len(initial_by_seed) == 2
    def initial_sum(name):
        values = [value[name] for value in initial_by_seed.values()]
        return sum(values) if all_initial_known and all(type(value) is int for value in values) else None
    result = {"schema_version": SCHEMA, "scope": "four-run cohort cumulative boundary",
        "subject_identity": cohort["cohort_identity"], "cohort_identity": cohort["cohort_identity"],
        "cohort_path": cohort["report_path"], "runs": {f"{run['variant']}/seed_{run['master_seed']}": run for run in runs},
        "missing_run_slots": missing, "status": "complete" if len(runs) == 4 and all(run["status"] == "complete" for run in runs) else "incomplete",
        "shared_initial_costs": {"physical_joint_transitions": 100000,
            "physical_replay_joint_transitions": 100000,
            "configured_four_run_attributed_joint_transitions": 200000,
            "prepared_run_attributed_joint_transitions": 50000 * len(runs),
            "physical_selected_model_prefix_optimizer_steps": 9000,
            "configured_four_run_attributed_selected_model_steps": 18000,
            "prepared_run_attributed_selected_model_steps": 4500 * len(runs),
            "selected_prefix_is_total_historical_fit": False},
        "initial_collection_by_seed": initial_by_seed,
        "known_initial_physical_joint_decisions_including_resets": initial_sum("physical_joint_decisions_including_initialization"),
        "known_initial_published_step_simulator_tics": initial_sum("real_simulator_tics_total"),
        "sum_run_logical_counters": counters,
        "evaluation_joint_transitions": sum(run["evaluation"]["evaluation_joint_transitions"] for run in runs),
        "evaluation_known_simulator_tics": sum(run["evaluation"]["evaluation_simulator_tics"] for run in runs),
        "evaluation_unknown_dispatch_attempts": sum(run["evaluation"]["unknown_dispatch_attempts"] for run in runs),
        "evaluation_unknown_simulator_tics": any(run["evaluation"]["unknown_simulator_tics"] for run in runs),
        "historical_and_development_costs": {"cover_work": None, "original_fit_beyond_selected_prefix": None,
            "earlier_policy_development": None, "status": "preserved separately; absent measurements remain unknown"},
        "evaluation_enters_training": False, "snapshot_is_cumulative_not_an_incremental_debit": True,
        "research_training_executed_by_ledger": False, "research_evaluation_executed_by_ledger": False}
    return result if output_dir is None else _append(result, output_dir)
