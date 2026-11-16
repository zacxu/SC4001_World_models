"""Frozen P5.4 final-test evaluation, with actor-only loading and durable costs.

The native API has no episode/seed/checkpoint tuning knobs. Fixture hooks are
explicitly labelled and can never produce native scientific evidence. Every
dispatch is preceded by a durable intent, so an interrupted worker cannot turn
consumed or unknown simulator work into zero-cost work on resume.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import torch

from scripts.phase36_support.evaluation import actor_version, load_actor_export
from world_models.data.ledger import file_sha256
from world_models.env.coop import TwoPlayerCoopEnv
from world_models.env.coordinator import CoordinatorError, TwoPlayerCoordinator
from world_models.env.splits import build_split_scenario
from world_models.infrastructure.reproducibility import canonical_hash, canonical_json
from world_models_study.bank import validate_test_bank

from .evidence import ROOT, read_report, safe_path, source_identity, write_report


SCHEMA_VERSION = "p54-final-test-evaluation-v1"
_NATIVE_ENV, _NATIVE_COORDINATOR = TwoPlayerCoopEnv, TwoPlayerCoordinator
_TERMINALS = {"success", "deadline", "failure", "death"}
_ZERO_FLAGS = {"training_interactions_added": 0, "policy_optimizer_updates": 0,
               "real_ppo_updates": 0, "world_model_updates": 0,
               "world_model_loaded": False, "critic_loaded": False,
               "learning_eligible": False, "evaluation_data_enters_training": False}
_PROTOCOL = {"split": "test", "purpose": "test", "episodes": 20, "bank_seed": 5600,
             "mode": "stochastic", "temperature": 1.0, "actor_precision": "float32",
             "frame_skip": 4, "deadline": 400, "final_local_cycles": 20}


def _sampling_runtime(actor):
    return {"action_rng_backend": next(actor.parameters()).device.type,
            "action_sampler": "torch.multinomial_per_player", "torch": str(torch.__version__),
            "precision": "float32"}


def _valid_sampling_runtime(runtime):
    return (type(runtime) is dict and set(runtime) == {"action_rng_backend", "action_sampler", "torch", "precision"}
            and runtime.get("action_rng_backend") in ("cpu", "cuda")
            and runtime.get("action_sampler") == "torch.multinomial_per_player"
            and type(runtime.get("torch")) is str and bool(runtime["torch"])
            and runtime.get("precision") == "float32")


def _final_actor(run_dir):
    # Importing this module or --help must not import a training runtime.
    from .cohort import final_actor
    return final_actor(run_dir)


def _same(a, b):
    return canonical_json(a) == canonical_json(b)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _reference(path, base):
    path = safe_path(path)
    return {"path": path.relative_to(base).as_posix(), "sha256": file_sha256(path),
            "bytes": path.stat().st_size}


def _referenced(base, reference):
    if (type(reference) is not dict or set(reference) != {"path", "sha256", "bytes"}
            or type(reference["path"]) is not str or Path(reference["path"]).is_absolute()
            or "\\" in reference["path"]):
        raise ValueError("invalid endpoint evidence reference")
    path = safe_path(base / reference["path"])
    if (not path.is_relative_to(base) or not path.is_file()
            or _reference(path, base) != reference):
        raise ValueError("endpoint evidence file changed or escaped its evaluation directory")
    return path


def _binding(value):
    required = {"actor_export", "variant", "master_seed", "cohort_identity", "protocol_sha256",
                "bank", "binding", "source_sha256", "run_dir"}
    if type(value) is not dict or not required <= set(value):
        raise ValueError("P5.4 requires a complete authenticated final-actor binding")
    if (value["variant"] not in ("B", "D") or type(value["master_seed"]) is not int
            or value["master_seed"] not in (0, 1) or type(value["binding"]) is not dict):
        raise ValueError("invalid final B/D actor or paired master seed")
    for key in ("cohort_identity", "protocol_sha256"):
        if type(value[key]) is not str or len(value[key]) != 64 or any(c not in "0123456789abcdef" for c in value[key]):
            raise ValueError("final actor binding identities must be SHA256 strings")
    bank = validate_test_bank(value["bank"])
    if value["protocol_sha256"] != bank["protocol_sha256"]:
        raise ValueError("final actor protocol differs from its exact test bank")
    if value["source_sha256"] != source_identity():
        raise ValueError("frozen cohort sources differ from the current endpoint implementation")
    if "bank_sha256" in value and value["bank_sha256"] != canonical_hash(bank):
        raise ValueError("frozen cohort bank hash differs from its manifest")
    return deepcopy(value)


def _claim_body(binding, output):
    return {"schema_version": "p54-single-final-test-claim-v1", "run_dir": binding["run_dir"],
             "output_dir": str(output), "cohort_identity": binding["cohort_identity"],
             "bank_sha256": canonical_hash(binding["bank"]),
             "final_actor_identity": canonical_hash(binding["binding"])}


def _preserve_uncommitted(path):
    """Keep an atomic JSON payload whose publication lost only its receipt."""
    if not path.is_file() or path.with_suffix(".sha256").exists():
        raise ValueError("only an unreceipted endpoint payload can be preserved for recovery")
    digest = file_sha256(path)
    for index in range(10000):
        destination = path.with_name(f"{path.stem}.uncommitted_{index:04d}_{digest[:12]}{path.suffix}")
        if not destination.exists():
            path.rename(destination)
            if file_sha256(destination) != digest:
                raise RuntimeError("uncommitted endpoint payload changed during preservation")
            return destination
    raise ValueError("too many uncommitted endpoint payload publications")


def _recover_exact_payload(path, expected):
    if path.exists() and not path.with_suffix(".sha256").exists():
        if not _same(json.loads(path.read_text(encoding="utf-8")), expected):
            raise ValueError("uncommitted endpoint payload differs from the authenticated resume identity")
        _preserve_uncommitted(path)
        write_report(path, expected)


def _native_claim(binding, output, *, resume=False):
    claim = _claim_body(binding, output)
    path = safe_path(binding["run_dir"]) / "endpoint_test_claim.json"
    if resume:
        _recover_exact_payload(path, claim)
    if path.exists() or path.with_suffix(".sha256").exists():
        if not _same(read_report(path), claim):
            raise ValueError("this final actor already claimed another test output; preserve the first attempt")
    else:
        write_report(path, claim)
    return claim


class _Lease:
    """One evaluation writer per actor/output; OS releases this on process death."""
    def __init__(self, path):
        self.path, self.stream = path, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                if not self.stream.read(1):
                    self.stream.write(b"0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RuntimeError("another process owns this final actor's endpoint evaluation") from error
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()


class _Journal:
    def __init__(self, path):
        self.path = path
        self.sequence = 0
        self.previous = "0" * 64
        with path.open("x", encoding="utf-8", newline="\n"):
            pass

    def append(self, event, **values):
        body = {"sequence": self.sequence, "previous_sha256": self.previous, "event": event, **values}
        row = {**body, "sha256": canonical_hash(body)}
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.sequence += 1
        self.previous = row["sha256"]
        return row


def _rows(path, *, allow_incomplete=False):
    raw = path.read_bytes()
    rows, previous, incomplete = [], "0" * 64, False
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if allow_incomplete and index == len(lines) - 1:
                incomplete = True
                break
            raise ValueError("evaluation journal has an incomplete durable row")
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("evaluation journal is malformed") from error
        if (type(row) is not dict or row.get("sequence") != len(rows)
                or type(row.get("sequence")) is not int or row.get("previous_sha256") != previous):
            raise ValueError("evaluation journal sequence or hash chain changed")
        body = {key: value for key, value in row.items() if key != "sha256"}
        if row.get("sha256") != canonical_hash(body):
            raise ValueError("evaluation journal row hash changed")
        rows.append(row)
        previous = row["sha256"]
    return rows, incomplete


def _snapshot(env):
    physical = env.coordinator.consumed_joint_decisions
    initialization = env.initialization_joint_decisions
    _integer(physical, "consumed physical decisions")
    _integer(initialization, "initialization decisions")
    # A failed reset may count an initialization attempt before it is consumed.
    return {"physical": physical, "initialization": min(initialization, physical)}


def _trace_record(trace, entry, episode_id, scenario, action_seeds):
    if not trace or len(trace) > 400:
        raise ValueError("evaluation episode must have 1..400 whole native decisions")
    events, entropy, total_return, tics = Counter(), 0.0, 0.0, 0
    previous, last_tic = [12, 12], None
    for index, row in enumerate(trace):
        if (row.get("decision_index") != index or type(row.get("decision_index")) is not int
                or row.get("previous_actions") != previous
                or type(row.get("actions")) is not list or len(row["actions"]) != 2
                or any(type(a) is not int or not 0 <= a < 12 for a in row["actions"])):
            raise ValueError("final-test action or decision alignment is invalid")
        if type(row.get("terminated")) is not bool or type(row.get("truncated")) is not bool or row["truncated"]:
            raise ValueError("final test must contain full episodes without collection cutoffs")
        if row["terminated"] != (index == len(trace) - 1):
            raise ValueError("final test trace ended early or continued beyond termination")
        _integer(row["start_tic"], "transition start tic")
        _integer(row["end_tic"], "transition end tic", 1)
        _integer(row["actual_tics"], "actual simulator tics", 1)
        if (row["end_tic"] - row["start_tic"] != row["actual_tics"]
                or not 1 <= row["actual_tics"] <= 4
                or last_tic is not None and row["start_tic"] != last_tic):
            raise ValueError("final-test simulator tics are inconsistent")
        if (type(row.get("event_deltas")) is not dict or any(
                type(v) is not int or v < 0 for v in row["event_deltas"].values())):
            raise ValueError("final-test event increments must be nonnegative integers")
        if type(row.get("action_entropy")) is not list or len(row["action_entropy"]) != 2:
            raise ValueError("final test requires both local actor entropies")
        entropies = [_finite(v, "local action entropy") for v in row["action_entropy"]]
        if any(v < 0 or v > math.log(12) + 1e-5 for v in entropies):
            raise ValueError("local action entropy is outside the 12-action distribution")
        total_return += _finite(row["team_reward"], "team reward")
        tics += row["actual_tics"]
        entropy += sum(entropies) / 2
        events.update(row["event_deltas"])
        previous, last_tic = row["actions"], row["end_tic"]
    reason = trace[-1]["terminal_reason"]
    if reason not in _TERMINALS or reason == "deadline" and len(trace) != 400:
        raise ValueError("final test has an invalid authoritative termination cause")
    return {"episode_index": entry["episode_index"], "episode_id": episode_id,
            "template_id": entry["template_id"], "bank_entry": entry, "action_seeds": action_seeds,
            "success": reason == "success", "episode_return": total_return,
            "latch_activated": events["latch_activations"] > 0, "joint_transitions": len(trace),
            "simulator_tics": tics, "termination_cause": reason, "event_counts": dict(events),
            "mean_action_entropy": entropy / len(trace), "scenario": scenario,
            "split_provenance": scenario["split_provenance"]}


def _scenario(scenario, entry, *, native):
    if (type(scenario) is not dict or not _same(scenario.get("split_provenance"), entry["split_provenance"])
            or not _same(scenario.get("geometry_template"), entry["geometry"])
            or not _same(scenario.get("randomization_draw"), entry["nuisance_draw"]["indices"])
            or scenario.get("seed") != entry["layout_seed"] or type(scenario.get("seed")) is not int
            or scenario.get("task_family") != "relay" or scenario.get("frame_skip") != 4
            or scenario.get("deadline") != 400):
        raise ValueError("scenario does not realize its exact registered held-out test geometry and nuisance draw")
    hashes = scenario.get("hashes", {})
    if type(hashes.get("wad_sha256")) is not str or len(hashes["wad_sha256"]) != 64:
        raise ValueError("test scenario has no compiled WAD identity")
    if native:
        path = safe_path(scenario["wad_path"])
        if file_sha256(path) != hashes["wad_sha256"]:
            raise ValueError("compiled native test scenario changed before environment creation")
    return deepcopy(scenario)


def _attempt_accounting(rows, incomplete=False):
    physical = initialization = published = published_tics = reset_tics = 0
    pending, unknown, unknown_tics = None, 0, incomplete
    trace, completions, scenarios = [], [], []
    for row in rows:
        event = row["event"]
        if event == "scenario_built":
            if scenarios or trace or physical:
                raise ValueError("scenario publication must precede all native episode interactions")
            scenarios.append(row)
        elif event == "intent":
            if pending is not None:
                raise ValueError("evaluation journal contains overlapping physical operations")
            if row["operation"] not in ("environment_create", "reset", "step"):
                raise ValueError("unrecognized simulator operation")
            pending = row["operation"]
        elif event == "observed":
            if pending != row["operation"]:
                raise ValueError("evaluation journal observation has no matching dispatch intent")
            p, i = row["physical"], row["initialization"]
            _integer(p, "physical decisions")
            _integer(i, "initialization decisions")
            if p < physical or i < initialization or i > p:
                raise ValueError("evaluation simulator counters regressed")
            if row["operation"] == "reset":
                if p != i:
                    raise ValueError("reset hid a non-initialization decision")
                reset_tics = _integer(row["reset_tics"], "initial/reset simulator tics")
            elif row["operation"] == "step" and (p != physical + 1 or i != initialization):
                raise ValueError("published native step must debit exactly one joint decision")
            physical, initialization, pending = p, i, None
        elif event == "transition":
            if pending is not None or physical != initialization + published + 1:
                raise ValueError("evaluation trace has no exactly once charged native step")
            trace.append(row["transition"])
            published += 1
            published_tics += _integer(row["transition"]["actual_tics"], "published tics", 1)
        elif event == "failure_snapshot":
            p, i = row["physical"], row["initialization"]
            _integer(p, "failed physical decisions")
            _integer(i, "failed initialization decisions")
            if p < physical or i < initialization or i > p:
                raise ValueError("failed attempt lost consumed physical decisions")
            # Failed acknowledgements do not establish tics, even when decision
            # debits are known. No guessed frame_skip conversion is performed.
            unknown_tics = unknown_tics or p > physical
            physical, initialization, pending = p, i, None
        elif event == "episode_complete":
            completions.append(row["record"])
        else:
            raise ValueError("unrecognized evaluation journal record")
    # A terminal transition is already a valid measured outcome. Losing only
    # its summary publication cannot authorize another draw from the test bank.
    if completions or trace and trace[-1].get("terminated") is True:
        if len(scenarios) != 1:
            raise ValueError("terminal trace has no durable scenario identity")
        saved = scenarios[0]
        derived = _trace_record(trace, saved["entry"], saved["episode_id"], saved["scenario"], saved["action_seeds"])
        if completions and (len(completions) != 1 or not _same(completions[0], derived)):
            raise ValueError("durable episode summary differs from its terminal action/physics trace")
        completions = [derived]
    if pending is not None:
        unknown += 1
        unknown_tics = True
    if incomplete:
        unknown += 1
    failed = physical - initialization - published
    unknown_tics = unknown_tics or failed > 0
    if failed < 0 or len(completions) > 1:
        raise ValueError("evaluation accounting or episode publication does not reconcile")
    return {"published_real_joint_transitions": published,
            "completed_episode_joint_transitions": sum(r["joint_transitions"] for r in completions),
            "initialization_joint_decisions": initialization, "failed_dispatch_attempts": failed,
            "physical_joint_decisions_including_initialization": physical,
            "published_real_simulator_tics_total": published_tics,
            "initialization_simulator_tics": reset_tics,
            "known_simulator_tics_total": published_tics + reset_tics,
            "unknown_dispatch_attempts": unknown, "unknown_simulator_tics": bool(unknown_tics),
            "accounting_exact": not unknown and not unknown_tics}, trace, completions


def _metrics(records, *, per_layout=True):
    if not records:
        raise ValueError("final-test metrics require completed episodes")
    n, successes = len(records), sum(row["success"] for row in records)
    successful = [row for row in records if row["success"]]
    result = {"episodes": n, "successes": successes, "success_rate": successes / n,
        "mean_return": sum(row["episode_return"] for row in records) / n,
        "latch_episodes": sum(row["latch_activated"] for row in records),
        "latch_episode_rate": sum(row["latch_activated"] for row in records) / n,
        "mean_latch_progress": sum(row["latch_activated"] for row in records) / n,
        "mean_length": sum(row["joint_transitions"] for row in records) / n,
        "mean_successful_completion_length": (sum(row["joint_transitions"] for row in successful) / successes
                                               if successes else None),
        "successful_completion_length_denominator": successes,
        "mean_successful_completion_simulator_tics": (sum(row["simulator_tics"] for row in successful) / successes
                                                      if successes else None),
        "termination_counts": dict(Counter(row["termination_cause"] for row in records))}
    if per_layout:
        result["per_layout"] = {layout: _metrics([r for r in records if r["template_id"] == layout], per_layout=False)
                                for layout in sorted({r["template_id"] for r in records})}
    return result


def _infrastructure(error, operation):
    if operation not in ("scenario_build", "environment_create", "reset", "step"):
        return False
    chain, current = [], error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__
    # Numerical/data-integrity failures cannot become retryable by being
    # wrapped in CoordinatorError by the environment boundary.
    if any(isinstance(e, (ValueError, ArithmeticError, AssertionError)) for e in chain):
        return False
    if any(isinstance(e, (OSError, TimeoutError, ConnectionError)) for e in chain):
        return True
    return any(isinstance(e, CoordinatorError) and any(word in str(e).lower() for word in (
        "timeout", "timed out", "watchdog", "worker exited", "worker died", "connection", "broken pipe",
        "startup failed", "failed to start")) for e in chain)


def _attempt(path, entry, binding, actor, *, env_factory, scenario_builder, native):
    path.mkdir(parents=True, exist_ok=False)
    journal = _Journal(path / "events.jsonl")
    index, seed = entry["episode_index"], binding["master_seed"]
    episode_id = f"relay-test-{index:06d}"
    action_seeds = entry["action_seeds_by_master_seed"][str(seed)]
    env, operation, scenario = None, "scenario_build", None
    device = next(actor.parameters()).device
    outcome, record, close_error = None, None, None
    try:
        scenario = _scenario(scenario_builder("relay", "test", entry["template_index"],
            seed=entry["layout_seed"], purpose="test", frame_skip=4, output_dir=path / "scenario"), entry, native=native)
        journal.append("scenario_built", scenario=scenario, entry=entry,
                       episode_id=episode_id, action_seeds=action_seeds)
        operation = "environment_create"
        journal.append("intent", operation=operation)
        env = env_factory("relay", seed=entry["layout_seed"], scenario_manifest=scenario,
                          work_root=path / "environment", frame_skip=4)
        if native and (type(env) is not _NATIVE_ENV or type(env.coordinator) is not _NATIVE_COORDINATOR):
            raise ValueError("native evaluation cannot use a replacement environment or coordinator")
        journal.append("observed", operation=operation, **_snapshot(env))
        operation = "reset"
        journal.append("intent", operation=operation)
        result = env.reset(seed=entry["engine_seed"])
        journal.append("observed", operation=operation, reset_tics=result.training_only_info["engine_tic"], **_snapshot(env))
        hidden = actor.initial_hidden(2)
        previous = torch.full((2,), 12, dtype=torch.long, device=device)
        streams = [torch.Generator(device=device).manual_seed(s) for s in action_seeds]
        trace = []
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            for decision in range(400):
                operation = "actor_sampling"
                images = torch.stack([torch.frombuffer(bytearray(frame.data), dtype=torch.uint8).reshape(3, 96, 128)
                                      for frame in result.observations]).to(device)
                local = actor(images, previous, hidden)
                distribution = local.distribution
                if distribution.probs.dtype != torch.float32 or not torch.isfinite(distribution.probs).all().item():
                    raise ValueError("final test requires finite FP32 local action probabilities")
                actions = [int(torch.multinomial(distribution.probs[player], 1, generator=streams[player]).item())
                           for player in (0, 1)]
                operation = "step"
                journal.append("intent", operation=operation, decision_index=decision, actions=actions)
                step = env.step(*actions)
                journal.append("observed", operation=operation, **_snapshot(env))
                operation = "transition_validation"
                if list(step.metadata.actions) != actions or step.metadata.decision_index != decision:
                    raise ValueError("native final-test executed action or decision index changed")
                transition = {"decision_index": decision, "actions": actions,
                    "previous_actions": previous.tolist(), "team_reward": step.team_reward,
                    "reward_components": step.training_only_info["reward_components"],
                    "event_deltas": step.training_only_info["events"]["deltas"],
                    "action_entropy": [float(v) for v in distribution.entropy().tolist()],
                    "terminated": step.terminated, "truncated": step.truncated,
                    "terminal_reason": step.boundary.reason.value, "start_tic": step.metadata.start_tic,
                    "end_tic": step.metadata.end_tic, "actual_tics": step.metadata.actual_simulator_tics}
                if step.truncated or not math.isfinite(step.team_reward):
                    raise ValueError("final test cannot truncate an episode or accept nonfinite reward")
                operation = "trace_write"
                journal.append("transition", transition=transition)
                trace.append(transition)
                if step.terminated:
                    operation = "episode_validation"
                    record = _trace_record(trace, entry, episode_id, scenario, action_seeds)
                    journal.append("episode_complete", record=record)
                    break
                hidden, previous, result = local.next_local_hidden, torch.tensor(actions, dtype=torch.long, device=device), step
            else:
                raise ValueError("native final-test episode exceeded the fixed 400-decision deadline")
        outcome = {"status": "complete", "retryable_infrastructure": False}
    except BaseException as error:
        snapshot_error = None
        if env is not None:
            try:
                journal.append("failure_snapshot", **_snapshot(env))
            except BaseException as snapshot_failure:
                snapshot_error = repr(snapshot_failure)
        outcome = {"status": "failed", "operation": operation, "error_type": type(error).__name__,
                   "error": str(error), "retryable_infrastructure": _infrastructure(error, operation),
                   "snapshot_error": snapshot_error}
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            outcome["retryable_infrastructure"] = True
            outcome["interruption"] = True
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as error:
                close_error = repr(error)
    rows, incomplete = _rows(journal.path)
    accounting, trace, records = _attempt_accounting(rows, incomplete)
    # A first valid terminal episode is immutable even if environment cleanup
    # later reports an infrastructure error; it must never be resampled.
    if records:
        outcome = {"status": "complete", "retryable_infrastructure": False}
        record = records[0]
    report = {"schema_version": "p54-episode-attempt-v1", "episode_index": index,
              "entry": entry, "attempt": int(path.name.split("_")[-1]),
              "native_backend": native, "native_engine": native,
              "record": record, "accounting": accounting,
              "journal": _reference(journal.path, path), "journal_incomplete": incomplete,
              "close_error": close_error, **outcome}
    write_report(path / "attempt.json", report)
    return report


def _read_attempt(path, entry, binding, *, allow_interruption=False):
    report_path, journal = path / "attempt.json", path / "events.jsonl"
    if report_path.exists() and not report_path.with_suffix(".sha256").exists() and allow_interruption:
        _preserve_uncommitted(report_path)
    if not report_path.exists():
        if not allow_interruption:
            raise ValueError("unsealed endpoint attempt requires explicit --resume")
        rows, incomplete = _rows(journal, allow_incomplete=True)
        accounting, trace, records = _attempt_accounting(rows, incomplete)
        # A completed immutable journal wins over a lost final receipt. This
        # seals the existing episode, without executing another simulator step.
        report = {"schema_version": "p54-episode-attempt-v1", "episode_index": entry["episode_index"],
            "entry": entry, "attempt": int(path.name.split("_")[-1]), "native_backend": binding["native_backend"],
            "native_engine": binding["native_backend"], "record": records[0] if records else None,
            "accounting": accounting, "journal": _reference(journal, path), "journal_incomplete": incomplete,
            "status": "complete" if records else "failed", "retryable_infrastructure": not bool(records),
            "operation": "uncommitted_process_interruption", "close_error": None}
        write_report(report_path, report)
    report = read_report(report_path)
    if (report.get("schema_version") != "p54-episode-attempt-v1"
            or not _same(report.get("entry"), entry)
            or report.get("episode_index") != entry["episode_index"]
            or report.get("attempt") != int(path.name.split("_")[-1])
            or report.get("native_backend") is not binding["native_backend"]
            or report.get("native_engine") is not binding["native_backend"]):
        raise ValueError("endpoint attempt identity changed")
    journal = _referenced(path, report["journal"])
    rows, incomplete = _rows(journal, allow_incomplete=report.get("journal_incomplete") is True)
    accounting, trace, records = _attempt_accounting(rows, incomplete)
    if not _same(accounting, report.get("accounting")) or incomplete is not report.get("journal_incomplete"):
        raise ValueError("endpoint attempt accounting does not reproduce its durable journal")
    if report.get("status") == "complete":
        if len(records) != 1 or not trace or report.get("retryable_infrastructure") is not False:
            raise ValueError("complete endpoint attempt lacks its first valid terminal episode")
        record = records[0]
        scenario = _scenario(record["scenario"], entry, native=binding["native_backend"])
        expected = _trace_record(trace, entry, f"relay-test-{entry['episode_index']:06d}", scenario,
                                 entry["action_seeds_by_master_seed"][str(binding["master_seed"])])
        if not _same(expected, record) or not _same(record, report.get("record")):
            raise ValueError("endpoint episode record does not reproduce its action/physics trace")
    elif (report.get("status") != "failed" or records or report.get("record") is not None
            or type(report.get("retryable_infrastructure")) is not bool):
        raise ValueError("failed endpoint attempt contains an invalid scientific outcome")
    return report


def _total_accounting(attempts):
    names = ("published_real_joint_transitions", "completed_episode_joint_transitions",
             "initialization_joint_decisions", "failed_dispatch_attempts",
             "physical_joint_decisions_including_initialization", "published_real_simulator_tics_total",
             "initialization_simulator_tics", "known_simulator_tics_total", "unknown_dispatch_attempts")
    result = {name: sum(a["accounting"][name] for a in attempts) for name in names}
    result.update(unknown_simulator_tics=any(a["accounting"]["unknown_simulator_tics"] for a in attempts),
                  accounting_exact=all(a["accounting"]["accounting_exact"] for a in attempts),
                  attempts_total=len(attempts),
                  infrastructure_retries=sum(a["attempt"] > 0 for a in attempts),
                  training_interactions_added=0, learning_eligible=False,
                  purpose="final_test_endpoint_outside_training",
                  physical_counts_are_lower_bounds_when_unknown=True,
                  simulator_tics_total=(None if any(a["accounting"]["unknown_simulator_tics"] for a in attempts)
                                        else sum(a["accounting"]["known_simulator_tics_total"] for a in attempts)))
    return result


def _scan(output, manifest, *, allow_interruption=False):
    attempts, records, refs = [], [], []
    for entry in manifest["bank"]["episodes"]:
        directory = output / "episodes" / f"{entry['episode_index']:06d}"
        paths = sorted(directory.glob("attempt_*")) if directory.exists() else []
        if [p.name for p in paths] != [f"attempt_{i:02d}" for i in range(len(paths))] or len(paths) > 3:
            raise ValueError("endpoint retries changed order or exceeded the two-infrastructure-retry limit")
        for i, path in enumerate(paths):
            if i and (attempts[-1]["status"] == "complete" or not attempts[-1]["retryable_infrastructure"]):
                raise ValueError("endpoint attempt retried a valid outcome or non-infrastructure failure")
            attempt = _read_attempt(path, entry, manifest, allow_interruption=allow_interruption)
            attempts.append(attempt)
            refs.append(_reference(path / "attempt.json", output))
            if attempt["status"] == "complete":
                records.append(attempt["record"])
    return attempts, records, refs


def load_partial_accounting(output_dir, *, final_binding=None, require_native=True):
    """Read-only costs for finished/failed or live evaluation prefixes.

    Live unsealed attempts retain their journal byte hash and unknown dispatch
    intent; this reader neither seals nor retries them.
    """
    output = safe_path(output_dir)
    manifest = read_report(output / "evaluation_manifest.json")
    bound = _binding(_final_actor(manifest["run_dir"]) if final_binding is None else final_binding)
    native = manifest.get("native_backend")
    if (type(require_native) is not bool or type(native) is not bool
            or require_native and not native or manifest.get("native_engine") is not native
            or manifest.get("schema_version") != "p54-final-test-manifest-v1"
            or not _valid_sampling_runtime(manifest.get("sampling_runtime"))
            or not _same(manifest.get("bank"), bound["bank"])
            or manifest.get("bank_sha256") != canonical_hash(bound["bank"])
            or not _same(manifest.get("protocol"), _PROTOCOL)
            or manifest.get("source_sha256") != bound["source_sha256"]
            or not _same(manifest.get("final_actor_binding"), bound["binding"])
            or manifest.get("final_actor_identity") != canonical_hash(bound["binding"])
            or any(not _same(manifest.get(k), bound[k]) for k in (
                "run_dir", "variant", "master_seed", "cohort_identity", "protocol_sha256"))
            or any(not _same(manifest.get(k), v) for k, v in _ZERO_FLAGS.items())):
        raise ValueError("partial endpoint accounting lacks its exact native cohort/test provenance")
    if native:
        claim = read_report(safe_path(bound["run_dir"]) / "endpoint_test_claim.json")
        if not _same(claim, _claim_body(bound, output)):
            raise ValueError("partial evaluation differs from its unique native test claim")
    attempts, snapshots = [], []
    for entry in manifest["bank"]["episodes"]:
        directory = output / "episodes" / f"{entry['episode_index']:06d}"
        paths = sorted(directory.glob("attempt_*")) if directory.exists() else []
        if len(paths) > 3 or [p.name for p in paths] != [f"attempt_{i:02d}" for i in range(len(paths))]:
            raise ValueError("invalid endpoint attempt history")
        for index, path in enumerate(paths):
            if index and (attempts[-1].get("status") == "complete" or attempts[-1].get("retryable_infrastructure") is False):
                raise ValueError("endpoint retried a valid or non-infrastructure outcome")
            if (path / "attempt.json").exists() and (path / "attempt.sha256").exists():
                attempts.append(_read_attempt(path, entry, manifest))
            else:
                before = _reference(path / "events.jsonl", output)
                rows, incomplete = _rows(path / "events.jsonl", allow_incomplete=True)
                accounting, _, _ = _attempt_accounting(rows, incomplete)
                attempts.append({"accounting": accounting, "attempt": index, "status": "unsealed"})
                if before != _reference(path / "events.jsonl", output):
                    raise RuntimeError("live evaluation journal advanced while accounting; retry inspection")
                snapshots.append(before)
    result = _total_accounting(attempts)
    references = []
    for journal in sorted((output / "episodes").glob("*/attempt_*/events.jsonl")):
        reference = _reference(journal, output)
        original = next((row for row in snapshots if row["path"] == reference["path"]), reference)
        if reference != original:
            raise RuntimeError("live evaluation journal advanced while publishing costs; retry inspection")
        references.append(dict(original, prefix_bytes=original["bytes"]))
    result["journal_prefixes"] = references
    return result


@torch.inference_mode(False)
def _evaluate_endpoint(run_dir, output_dir, *, device="cpu", resume=False,
                      env_factory=None, scenario_builder=None, final_binding=None, actor_loader=None):
    """Evaluate one final actor after the complete prospective cohort is trained.

    Native execution is only possible with the authenticated cohort API and
    original environment/builder/actor loader. Synthetic hooks always produce
    fixture-only reports, rejected by the scientific reader's default policy.
    """
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    native = all(v is None for v in (env_factory, scenario_builder, final_binding, actor_loader))
    bound = _binding(_final_actor(run_dir) if final_binding is None else final_binding)
    if safe_path(bound["run_dir"]) != safe_path(run_dir):
        raise ValueError("final-actor proof belongs to another training run")
    output = safe_path(output_dir)
    protected = [safe_path(bound["run_dir"]), safe_path(bound["actor_export"])]
    protected.extend(ROOT / name for name in ("src", "scripts", "tests", "configs", "scenarios"))
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("evaluation output cannot replace its training run or actor export")
    if output.exists() and not resume:
        raise FileExistsError("evaluation output exists; use --resume to retain first valid episodes and all costs")
    actor, export = (actor_loader or load_actor_export)(bound["actor_export"], device=device)
    metadata = export["metadata"]
    if (metadata["task_family"] != "relay" or metadata["variant"] != bound["variant"]
            or metadata["run_seed"] != bound["master_seed"] or metadata["local_ppo_cycles"] != 20
            or metadata["actor_optimizer_steps"] != 320 or metadata["critic_optimizer_steps"] != 320
            or metadata["model_updates"] != 4500 or metadata["actual_initial_replay_interactions"] != 50000
            or metadata["untrained_zero_interaction_baseline"] is not False):
        raise ValueError("final evaluation requires the registered completed 20-cycle FP32 local actor")
    if any(p.dtype != torch.float32 for p in actor.parameters()):
        raise ValueError("final endpoint actor parameters must use FP32")
    before = actor_version(actor)
    modes = [(module, module.training) for module in actor.modules()]
    sources = source_identity()
    manifest = {"schema_version": "p54-final-test-manifest-v1", "run_dir": str(safe_path(run_dir)),
        "cohort_identity": bound["cohort_identity"], "protocol_sha256": bound["protocol_sha256"],
        "variant": bound["variant"], "master_seed": bound["master_seed"], "bank": bound["bank"],
        "bank_sha256": canonical_hash(bound["bank"]), "final_actor_binding": bound["binding"],
        "final_actor_identity": canonical_hash(bound["binding"]), "actor_export": export,
        "source_sha256": sources, "native_backend": native, "native_engine": native,
        "sampling_runtime": _sampling_runtime(actor),
        "protocol": deepcopy(_PROTOCOL), **_ZERO_FLAGS}
    if native:
        _native_claim(bound, output, resume=resume)
    if output.exists():
        if (not (output / "evaluation_manifest.json").exists()
                and not (output / "evaluation_manifest.sha256").exists()
                and not (output / "episodes").exists()):
            write_report(output / "evaluation_manifest.json", manifest)
        _recover_exact_payload(output / "evaluation_manifest.json", manifest)
        if not _same(read_report(output / "evaluation_manifest.json"), manifest):
            raise ValueError("resume changed actor, cohort, protocol, native mode or implementation")
        _recover_exact_payload(output / "bank.json", bound["bank"])
        if (not (output / "bank.json").exists() and not (output / "bank.sha256").exists()
                and not (output / "episodes").exists()):
            write_report(output / "bank.json", bound["bank"])
        if not _same(read_report(output / "bank.json"), bound["bank"]):
            raise ValueError("resumed endpoint bank differs from the immutable final-test bank")
        if (output / "evaluation.json").exists():
            if (output / "evaluation.sha256").exists():
                return load_evaluation(output / "evaluation.json", require_native=native, final_binding=bound)
            _preserve_uncommitted(output / "evaluation.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_report(output / "evaluation_manifest.json", manifest)
        write_report(output / "bank.json", bound["bank"])
    started = time.perf_counter()
    actor.eval()
    try:
        attempts, records, _ = _scan(output, manifest, allow_interruption=resume)
        for entry in bound["bank"]["episodes"]:
            existing = [a for a in attempts if a["episode_index"] == entry["episode_index"]]
            if existing and existing[-1]["status"] == "complete":
                continue
            if existing and not existing[-1]["retryable_infrastructure"]:
                raise ValueError("episode failed outside the infrastructure retry policy; preserve and report it")
            for index in range(len(existing), 3):
                attempt_path = output / "episodes" / f"{entry['episode_index']:06d}" / f"attempt_{index:02d}"
                attempt = _attempt(attempt_path, entry, bound, actor, env_factory=env_factory or _NATIVE_ENV,
                    scenario_builder=scenario_builder or build_split_scenario, native=native)
                attempts.append(attempt)
                if attempt["status"] == "complete":
                    break
                if not attempt["retryable_infrastructure"]:
                    raise RuntimeError(f"final-test episode {entry['episode_index']} failed: {attempt.get('error')}")
                if attempt.get("interruption"):
                    raise KeyboardInterrupt("evaluation interrupted; resume the same output to preserve consumed work")
            else:
                raise RuntimeError("endpoint episode exhausted its two infrastructure retries")
            print(f"P5.4 {bound['variant']}/seed_{bound['master_seed']}: {entry['episode_index'] + 1}/20 test episodes", flush=True)
        attempts, records, references = _scan(output, manifest)
        if len(records) != 20 or actor_version(actor) != before or source_identity() != sources:
            raise ValueError("endpoint did not complete the fixed bank with unchanged actor and sources")
        # Reauthenticate all completed runs/inputs after evaluation. Fixture
        # callers explicitly supply synthetic evidence and cannot claim native.
        if native and not _same(_final_actor(run_dir), bound):
            raise ValueError("cohort or final-actor binding changed during test evaluation")
        runtime = {"python": platform.python_version(), "device": str(next(actor.parameters()).device),
            **_sampling_runtime(actor),
            "vizdoom": importlib.metadata.version("vizdoom") if native else None,
            "cpu_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
        report = {"schema_version": SCHEMA_VERSION, "status": "complete", "task_family": "relay",
            "run_dir": manifest["run_dir"], "variant": bound["variant"], "master_seed": bound["master_seed"],
            "cohort_identity": bound["cohort_identity"], "protocol_sha256": bound["protocol_sha256"],
            "bank_sha256": manifest["bank_sha256"], "final_actor_identity": manifest["final_actor_identity"],
            "final_local_cycles": 20, "final_actor_binding": bound["binding"],
            "actor_export": export, "source_sha256": sources, "source_unchanged": True,
            "actor_version_before": before, "actor_version_after": actor_version(actor),
            "native_backend": native, "native_engine": native,
            "native_environment_evaluation_executed": native, "research_evaluation_executed": native,
            "fixture_only": not native, "protocol": manifest["protocol"], "runtime": runtime,
            "episodes": records, "metrics": _metrics(records), "interaction_accounting": _total_accounting(attempts),
            "attempts": references, "manifest": _reference(output / "evaluation_manifest.json", output),
            "bank": _reference(output / "bank.json", output),
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(), **_ZERO_FLAGS}
        write_report(output / "evaluation.json", report)
        return load_evaluation(output / "evaluation.json", require_native=native, final_binding=bound)
    finally:
        for module, mode in modes:
            module.training = mode


def evaluate_endpoint(run_dir, output_dir, *, device="cpu", resume=False,
                      env_factory=None, scenario_builder=None, final_binding=None, actor_loader=None):
    """Public serialized evaluator; explicit fixture hooks cannot claim native evidence."""
    native = all(v is None for v in (env_factory, scenario_builder, final_binding, actor_loader))
    bound = _binding(_final_actor(run_dir) if final_binding is None else final_binding)
    output = safe_path(output_dir)
    protected = [safe_path(bound["run_dir"]), safe_path(bound["actor_export"])]
    protected.extend(ROOT / name for name in ("src", "scripts", "tests", "configs", "scenarios"))
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("endpoint output must be outside training exports and project sources")
    lock = safe_path(run_dir) / "endpoint_test.lock" if native else output.parent / f".{output.name}.endpoint.lock"
    with _Lease(lock):
        return _evaluate_endpoint(run_dir, output, device=device, resume=resume,
            env_factory=env_factory, scenario_builder=scenario_builder, final_binding=final_binding,
            actor_loader=actor_loader)


def load_evaluation(path, *, require_native=True, final_binding=None):
    """Authenticate a final endpoint, every episode/attempt/trace and its cohort."""
    if type(require_native) is not bool:
        raise ValueError("require_native must be boolean")
    path = safe_path(path)
    output, report = path.parent, read_report(path)
    if path.name != "evaluation.json" or report.get("schema_version") != SCHEMA_VERSION or report.get("status") != "complete":
        raise ValueError("P5.4 requires a complete receipt-bound final evaluation")
    bound = _binding(_final_actor(report["run_dir"]) if final_binding is None else final_binding)
    native = report.get("native_backend") is True
    if require_native and not native:
        raise ValueError("synthetic fixture evaluation cannot establish native research outcomes")
    for key in ("native_backend", "native_engine", "native_environment_evaluation_executed", "research_evaluation_executed"):
        if type(report.get(key)) is not bool or report[key] is not native:
            raise ValueError("endpoint native execution markers disagree")
    if report.get("fixture_only") is not (not native):
        raise ValueError("endpoint fixture marker disagrees with native execution")
    runtime = report.get("runtime")
    if (type(runtime) is not dict or runtime.get("action_rng_backend") not in ("cpu", "cuda")
            or type(runtime.get("device")) is not str or type(runtime.get("torch")) is not str
            or not runtime["torch"] or runtime.get("action_sampler") != "torch.multinomial_per_player"
            or runtime.get("precision") != "float32"):
        raise ValueError("final test has no explicit FP32 local action-generator backend identity")
    try:
        runtime_backend = torch.device(runtime["device"]).type
    except (ValueError, RuntimeError) as error:
        raise ValueError("final test recorded an invalid action-generator device") from error
    if runtime_backend != runtime["action_rng_backend"]:
        raise ValueError("final test action RNG backend disagrees with its recorded device")
    manifest_path = _referenced(output, report["manifest"])
    if manifest_path.name != "evaluation_manifest.json":
        raise ValueError("evaluation has no canonical manifest reference")
    manifest = read_report(manifest_path)
    if (manifest.get("schema_version") != "p54-final-test-manifest-v1"
            or manifest.get("native_backend") is not native or manifest.get("native_engine") is not native
            or not _same(manifest.get("protocol"), _PROTOCOL)
            or not _valid_sampling_runtime(manifest.get("sampling_runtime"))
            or not _same(manifest["sampling_runtime"], {key: runtime[key] for key in manifest["sampling_runtime"]})):
        raise ValueError("final-test manifest changed its native mode or fixed evaluation protocol")
    bank_path = _referenced(output, report["bank"])
    if bank_path.name != "bank.json" or not _same(read_report(bank_path), bound["bank"]):
        raise ValueError("evaluation bank differs from the frozen exact test bank")
    for key in ("run_dir", "variant", "master_seed", "cohort_identity", "protocol_sha256"):
        if not _same(report.get(key), bound[key]) or not _same(manifest.get(key), bound[key]):
            raise ValueError("final evaluation identity differs from its authenticated run")
    if (report.get("task_family") != "relay" or report.get("final_local_cycles") != 20
            or type(report.get("final_local_cycles")) is not int
            or report.get("source_sha256") != bound["source_sha256"]
            or manifest.get("source_sha256") != bound["source_sha256"]
            or report.get("source_unchanged") is not True
            or report.get("actor_version_before") != report.get("actor_version_after")
            or report.get("bank_sha256") != canonical_hash(bound["bank"])
            or manifest.get("bank_sha256") != canonical_hash(bound["bank"])
            or not _same(manifest.get("bank"), bound["bank"])
            or not _same(report.get("actor_export"), manifest.get("actor_export"))
            or not _same(report.get("protocol"), manifest.get("protocol"))
            or not _same(report.get("final_actor_binding"), bound["binding"])
            or not _same(manifest.get("final_actor_binding"), bound["binding"])
            or report.get("final_actor_identity") != canonical_hash(bound["binding"])
            or manifest.get("final_actor_identity") != canonical_hash(bound["binding"])):
        raise ValueError("final endpoint provenance changed")
    for source in (manifest, report):
        if any(not _same(source.get(key), value) for key, value in _ZERO_FLAGS.items()):
            raise ValueError("test evaluation imported learning data, critic/model state or optimization work")
    export = report["actor_export"]
    if (type(export) is not dict or safe_path(export.get("path", "")) != safe_path(bound["actor_export"])
            or export.get("manifest_sha256") != file_sha256(safe_path(bound["actor_export"]) / "manifest.json")):
        raise ValueError("evaluation actor export differs from its authenticated final actor")
    actor_manifest = json.loads((safe_path(bound["actor_export"]) / "manifest.json").read_text(encoding="utf-8"))
    _referenced(safe_path(bound["actor_export"]), actor_manifest["payload"])
    if (not _same(actor_manifest.get("metadata"), export.get("metadata"))
            or export.get("payload_sha256") != actor_manifest["payload"]["sha256"]
            or file_sha256(safe_path(bound["actor_export"]) / actor_manifest["payload"]["path"]) != export.get("payload_sha256")):
        raise ValueError("evaluated actor tensors or metadata no longer match their committed receipt")
    if native:
        claim = read_report(safe_path(bound["run_dir"]) / "endpoint_test_claim.json")
        if not _same(claim, _claim_body(bound, output)):
            raise ValueError("native test result differs from its unique first-output claim")
    attempts, records, refs = _scan(output, manifest)
    if (len(records) != 20 or [r["episode_index"] for r in records] != list(range(20))
            or not _same(report.get("episodes"), records) or not _same(report.get("attempts"), refs)
            or not _same(report.get("metrics"), _metrics(records))
            or not _same(report.get("interaction_accounting"), _total_accounting(attempts))):
        raise ValueError("endpoint metrics or costs do not reproduce the complete first-valid episode bank")
    return deepcopy(report)


def validate_evaluation_report(path, *, cohort=None, require_native=True):
    report = load_evaluation(path, require_native=require_native)
    if cohort is not None:
        if type(cohort) is dict:
            identity = cohort.get("cohort_identity")
        elif type(cohort) is str and len(cohort) == 64 and all(c in "0123456789abcdef" for c in cohort):
            identity = cohort
        else:
            from .cohort import inspect_cohort
            identity = inspect_cohort(cohort)["cohort_identity"]
        if report["cohort_identity"] != identity:
            raise ValueError("evaluation belongs to another prospective cohort")
    return report


inspect_evaluation = load_evaluation
