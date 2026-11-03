"""Native Phase 1 acceptance fixtures; all trajectories are validation-only."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import threading
import time

import psutil

from world_models.contracts import BoundaryReason
from world_models.env.coop import EnvironmentIntegrityError, LifecycleState, TwoPlayerCoopEnv
from world_models.env.locality import audit_locality
from world_models.env.mechanics_video import MechanicsVideo
from world_models.env.scripted import _Controller, _arrived, _move
from world_models.env.splits import build_split_scenario, list_templates
from world_models.env.targets import FIXED_POSITION_FIELDS, USER_FIELDS, registry_manifest
from world_models.env.validation_data import ValidationLedger
from world_models.infrastructure.config import resolve_config


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def validate_transition(result, previous_events: dict, scenario: dict) -> dict:
    """Cross-check published timing, rewards, units, targets and event deltas."""
    info = result.training_only_info
    if result.rewards != (result.team_reward, result.team_reward):
        raise AssertionError("participant rewards differ")
    components = info["reward_components"]
    if (components["shared_rewards"] != list(result.rewards)
            or components["total"] != result.team_reward
            or not math.isclose(sum(components[key] for key in ("step", "terminal", "milestone")), result.team_reward, abs_tol=1e-12)):
        raise AssertionError("shared reward components disagree")
    if result.metadata.actual_simulator_tics != scenario["frame_skip"] and not result.terminated:
        raise AssertionError("nonterminal action repeat did not advance the declared tics")
    if info["decision_index"] != result.metadata.next_decision_index:
        raise AssertionError("wrapper and transition indexes disagree")
    events = info["events"]
    if set(events["cumulative"]) != set(previous_events):
        raise AssertionError("event schema changed within episode")
    for name, value in events["cumulative"].items():
        if value < previous_events[name] or events["deltas"][name] != value - previous_events[name]:
            raise AssertionError(f"invalid cumulative event/delta: {name}")
    targets = info["privileged_targets"]
    if targets["registry_sha256"] != registry_manifest()["sha256"]:
        raise AssertionError("privileged target registry differs from exported registry")
    if len(targets["players"]) != 2:
        raise AssertionError("privileged targets must contain both player slots")
    for raw, target in zip(info["players"], targets["players"]):
        angle = math.radians(raw["ANGLE"])
        if not math.isclose(target["raw"]["heading_sin"], math.sin(angle), abs_tol=1e-12):
            raise AssertionError("heading sine conversion differs")
        if not math.isclose(target["raw"]["heading_cos"], math.cos(angle), abs_tol=1e-12):
            raise AssertionError("heading cosine conversion differs")
    fractional = False
    for name in FIXED_POSITION_FIELDS:
        raw = info["players"][0][f"USER{USER_FIELDS[name]}"]
        decoded = info["task_flags"][name]
        if decoded != raw / 65536:
            raise AssertionError("ACS signed 16.16 coordinate conversion differs")
        fractional |= bool(raw % 65536)
    validity = tuple(frame is not None for frame in result.observations)
    if validity != result.boundary.next_observation_valid:
        raise AssertionError("final image validity differs from boundary mask")
    if info["image_reconstruction_valid"] != list(validity) or result.image_reconstruction_mask != tuple(float(value) for value in validity):
        raise AssertionError("image reconstruction masks differ from actual observations")
    return {"fractional_fixed_point_seen": fractional}


def transition_record(result, actions: list[int]) -> dict:
    return {"actions": actions, "reward": result.team_reward, "boundary": result.boundary.reason.value,
            "bootstrap_mask": result.boundary.bootstrap_mask,
            "next_observation_valid": list(result.boundary.next_observation_valid),
            "rgb_sha256": [None if frame is None else hashlib.sha256(frame.data).hexdigest() for frame in result.observations],
            "metadata": {"decision_index": result.metadata.decision_index,
                         "start_tic": result.metadata.start_tic, "end_tic": result.metadata.end_tic},
            "training_only_info": result.training_only_info}


def run_mechanics_case(scenario: dict, directory: Path, ledger: ValidationLedger, *, inactive_player=None,
                       video=False) -> dict:
    family, seed = scenario["task_family"], scenario["seed"]
    directory.mkdir(parents=True, exist_ok=True)
    env = TwoPlayerCoopEnv(family, seed=seed, scenario_manifest=scenario, work_root=directory/"environments")
    records, history, total, published, fractional = [], [], 0.0, 0, False
    result = None
    movie = MechanicsVideo(directory/"video", family, env.frame_skip,
                           frame_provenance="Native ViZDoom local observations collected through TwoPlayerCoopEnv; trace and exact PNGs accompany this video") if video else None
    try:
        result = env.reset(seed=seed)
        history.append(result)
        initial_events = deepcopy(result.training_only_info["events"]["cumulative"])
        previous = initial_events
        if movie:
            movie.append(result)
        controller = _Controller(scenario)
        for _ in range(scenario["deadline"]):
            info = result.training_only_info
            actions = controller.actions(info["players"], info["task_flags"])
            if inactive_player is not None:
                actions[inactive_player] = 0
            result = env.step(*actions)
            published += 1
            checks = validate_transition(result, previous, scenario)
            fractional |= checks["fractional_fixed_point_seen"]
            previous = deepcopy(result.training_only_info["events"]["cumulative"])
            total += result.team_reward
            records.append(transition_record(result, actions))
            if len(history) < 6 and all(frame is not None for frame in result.observations):
                history.append(result)
            if movie:
                movie.append(result, actions)
            if result.terminated or result.truncated:
                break
        succeeded = result.boundary.reason is BoundaryReason.SUCCESS
        expected = inactive_player is None
        if succeeded != expected:
            raise RuntimeError(f"mechanics {scenario.get('split_provenance',{}).get('template_id')} inactive={inactive_player} ended {result.boundary.reason}")
        events = result.training_only_info["events"]["cumulative"]
        if succeeded:
            if events["successes"] != 1:
                raise AssertionError("team success cumulative event was not captured")
            expected_milestones = 1 if family == "relay" else 2
            if result.training_only_info["task_flags"]["milestone_count"] != expected_milestones:
                raise AssertionError("successful mechanics fixture missed a milestone")
        case = {"status": "passed", "task_family": family, "template_id": scenario["split_provenance"]["template_id"],
                "split": scenario["split_provenance"]["split"], "purpose": "diagnostic", "learning_eligible": False,
                "seed": seed, "inactive_player": inactive_player, "success": succeeded,
                "decisions": published, "return": total, "boundary": result.boundary.reason.value,
                "scenario_manifest": scenario, "initial_events": initial_events, "final_events": events,
                "fixed_point_fractional_seen": fractional,
                "locality": audit_locality(history,hidden_size=resolve_config(family,"B","pilot").settings["actor"]["hidden_size"]) if video else None,
                "video": movie.finish() if movie else None,
                "records": records}
        path = directory/"mechanics_trace.json"
        save_json(path, case)
        return {**{key:value for key,value in case.items() if key != "records"},
                "trace_path": str(path.resolve()), "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    except BaseException as exc:
        save_json(directory/"mechanics_failure.json",{"status":"failed","error":repr(exc),
                  "scenario_manifest":scenario,"records":records,
                  "last_valid_info":None if result is None else result.training_only_info})
        raise
    finally:
        try:
            ledger.record("mechanics", env, published, expected_error=env.state is LifecycleState.NEEDS_RECOVERY)
        finally:
            env.close()


def mechanics_campaign(output: Path, *, template_limit: int | None = None) -> dict:
    ledger = ValidationLedger(output.parent)
    print_lock = threading.Lock()
    def run_family(family):
        cases, inactive = [], []
        templates = list_templates(family)
        if template_limit is not None:
            templates = templates[:template_limit]
        split_indices = Counter()
        first = None
        for template in templates:
            split = template["split"]
            index = split_indices[split]
            split_indices[split] += 1
            scenario = build_split_scenario(family, split, index, seed=0, purpose="diagnostic",
                                            output_dir=output/"scenarios"/template["id"])
            first = scenario if first is None else first
            case = run_mechanics_case(scenario, output/template["id"], ledger, video=template == templates[0])
            cases.append(case)
            with print_lock:
                print(json.dumps({"progress":"template_mechanics", "template":template["id"], "decisions":case["decisions"]}), flush=True)
            save_json(output/f"{family}_mechanics_progress.json", {"cases": cases, "inactive": inactive})
        for player in (0, 1):
            inactive.append(run_mechanics_case(first, output/f"{family}_inactive{player}", ledger, inactive_player=player))
        return cases,inactive
    with ThreadPoolExecutor(max_workers=2) as pool:
        family_results = list(pool.map(run_family,("relay","cover")))
    cases = [case for result,_ in family_results for case in result]
    inactive = [case for _,result in family_results for case in result]
    complete = len(cases) == 20
    return {"status":"passed" if complete else "passed_preflight", "cases":cases,
            "inactive_checks":inactive, "all_template_mechanics_passed":complete,
            "fixed_point_fractional_observed":any(case["fixed_point_fractional_seen"] for case in cases),
            "scope":"Native mechanics at seed0 for every template; full bank reachability/initial-state invariants are checked separately in the split manifest."}


def stress_campaign(output: Path, *, decisions: int = 10000, coordinated_resets: int = 100) -> dict:
    if decisions < coordinated_resets or coordinated_resets < 2 or coordinated_resets % 2 or decisions % 2:
        raise ValueError("stress requires even counts, at least two resets and >=one decision per reset")
    output.mkdir(parents=True, exist_ok=True)
    ledger = ValidationLedger(output.parent)
    print_lock = threading.Lock()
    def run_family(family):
        scenario = build_split_scenario(family, "train", 0, seed=0, purpose="diagnostic", output_dir=output/"scenarios"/family)
        env = TwoPlayerCoopEnv(family, seed=0, scenario_manifest=scenario, work_root=output/"environments")
        published, resets, reset_rows = 0, 0, []
        action_counts = Counter()
        rng = random.Random(492731 if family == "relay" else 971443)
        raw_path = output/f"{family}_decisions.jsonl"
        raw_path.write_text("", encoding="utf-8")
        begun = time.perf_counter()
        try:
            current = env.reset(seed=19000)
            first_reset = current
            previous_episode = current.training_only_info["episode_id"]
            previous_events = current.training_only_info["events"]["cumulative"]
            initial_image = tuple(frame.data for frame in current.observations)
            target = decisions//2
            batches = coordinated_resets//2
            for batch in range(batches):
                count_before = env.coordinator.consumed_joint_decisions
                reset_started = time.perf_counter()
                current = env.reset(seed=19001+batch)
                elapsed_reset = time.perf_counter()-reset_started
                info = current.training_only_info
                if (info["episode_id"] <= previous_episode or info["decision_index"] != 0
                        or not info["reset_actor_memories"]
                        or info["consumed_joint_decisions"] <= count_before
                        or any(info["events"]["cumulative"].values())):
                    raise AssertionError("coordinated reset leaked episode state/accounting")
                previous_episode = info["episode_id"]
                previous_events = deepcopy(info["events"]["cumulative"])
                resets += 1
                reset_rows.append({"episode_id":previous_episode,"engine_seed":19001+batch,
                                   "reset_seconds":elapsed_reset,"consumed_before":count_before,
                                   "consumed_after_initialization":info["consumed_joint_decisions"],
                                   "reset_actor_memories":True,"zero_events":True})
                chunk = target//batches + int(batch < target % batches)
                actions = [0, 0]
                with raw_path.open("a", encoding="utf-8") as stream:
                    for index in range(chunk):
                        if index % 4 == 0:
                            actions = [rng.randrange(12),rng.randrange(12)]
                        result = env.step(*actions,collection_cutoff=index == chunk-1)
                        action_counts.update(actions)
                        validate_transition(result, previous_events, scenario)
                        previous_events = deepcopy(result.training_only_info["events"]["cumulative"])
                        published += 1
                        row = transition_record(result, list(actions))
                        stream.write(json.dumps(row,allow_nan=False)+"\n")
                        if (result.terminated or result.truncated) and index < chunk-1:
                            current = env.reset(seed=200000+published)
                            if current.training_only_info["episode_id"] <= previous_episode:
                                raise AssertionError("terminal reset reused episode ID")
                            previous_episode = current.training_only_info["episode_id"]
                            previous_events = deepcopy(current.training_only_info["events"]["cumulative"])
                            resets += 1
                            if (current.training_only_info["decision_index"] != 0
                                    or not current.training_only_info["reset_actor_memories"]
                                    or any(previous_events.values())):
                                raise AssertionError("early-terminal reset retained task state")
                            reset_rows.append({"episode_id":previous_episode,"engine_seed":200000+published,
                                               "reason":"early_terminal_within_validation_block",
                                               "consumed_after_initialization":current.training_only_info["consumed_joint_decisions"],
                                               "reset_actor_memories":True,"zero_events":True})
                    stream.flush()
                # Saved immutable renderer buffers must survive many episodes.
                if tuple(frame.data for frame in first_reset.observations) != initial_image:
                    raise AssertionError("saved initial actor buffers mutated during the campaign")
                with print_lock:
                    print(json.dumps({"progress":"stress","task":family,"decisions":published,"coordinated_resets":resets}),flush=True)
            if published != target:
                raise AssertionError("stress count differs from requested actual joint transitions")
            dispatches = acknowledgements = 0
            with (env.coordinator.run_dir/"dispatch_ledger.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["kind"] == "infrastructure_error":
                        raise AssertionError("stress produced an unexplained infrastructure error")
                    if row["kind"] == "dispatch":
                        dispatches += 1
                        if row["dispatch_id"] != dispatches:
                            raise AssertionError("stress dispatch journal is not contiguous")
                    if row["kind"] == "acknowledged":
                        acknowledgements += 1
                        if row["dispatch_id"] != acknowledgements or len(set(row["tics"])) != 1:
                            raise AssertionError("stress ACK journal contains a stale ID or tic mismatch")
            if dispatches != acknowledgements:
                raise AssertionError("stress journal has an unacknowledged dispatch")
            family_report = {"task_family":family,"published_joint_decisions":published,"coordinated_resets":resets,
                             "unexplained_tic_mismatches":0,"stale_responses":0,"infrastructure_errors":0,
                             "wall_seconds":time.perf_counter()-begun,"reset_records":reset_rows,
                             "action_counts":dict(action_counts),"native_dispatches":dispatches,"native_acknowledgements":acknowledgements,
                             "trace_path":str(raw_path.resolve()),"trace_sha256":hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                             "coordinator_ledger":str(env.coordinator.run_dir/"dispatch_ledger.jsonl"),
                             "consumed_joint_decisions":env.coordinator.consumed_joint_decisions,
                             "initialization_joint_decisions":env.initialization_joint_decisions}
        finally:
            try:
                ledger.record("stress", env, published, expected_error=env.state is LifecycleState.NEEDS_RECOVERY)
            finally:
                env.close()
        return family_report
    with ThreadPoolExecutor(max_workers=2) as pool:
        families = list(pool.map(run_family,("relay","cover")))
    total_published = sum(case["published_joint_decisions"] for case in families)
    total_resets = sum(case["coordinated_resets"] for case in families)
    reached = total_published >= 10000 and total_resets >= 100
    return {"status":"passed" if reached else "passed_preflight", "families":families,
            "published_joint_decisions":total_published,"coordinated_resets":total_resets,
            "acceptance_threshold_met":reached,"purpose":"validation_only; excluded from training replay",
            "scheduler":"Two independent joint environments, one per family, collected concurrently under the shared four-logical-CPU acceptance budget",
            "policy":"Fixed diagnostic RNG seeds; each independently sampled joint action is held for four decisions. All 12 primitives remain available."}


def boundary_campaign(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    ledger = ValidationLedger(output.parent)
    cases = []
    for mode, family in (("collection_cutoff","relay"),("deadline","relay"),("death","cover")):
        scenario = build_split_scenario(family,"train",0,seed=0,purpose="diagnostic",output_dir=output/"scenarios"/mode)
        env = TwoPlayerCoopEnv(family,seed=0,scenario_manifest=scenario,work_root=output/"environments")
        published = 0
        try:
            reset = env.reset(seed=401)
            previous = reset.training_only_info["events"]["cumulative"]
            for decision in range(scenario["deadline"]):
                result = env.step(0,0,collection_cutoff=mode == "collection_cutoff" and decision == 4)
                published += 1
                validate_transition(result,previous,scenario)
                previous = deepcopy(result.training_only_info["events"]["cumulative"])
                if result.terminated or result.truncated:
                    break
            expected = BoundaryReason(mode)
            if result.boundary.reason is not expected:
                raise AssertionError(f"{mode} native fixture ended through {result.boundary.reason}")
            if result.boundary.bootstrap_mask != (1.0 if mode == "collection_cutoff" else 0.0):
                raise AssertionError("native terminal/truncation bootstrap mask differs")
            expected_reward = -.001 if mode == "collection_cutoff" else -1.001
            if not math.isclose(result.team_reward,expected_reward,abs_tol=1e-12):
                raise AssertionError("boundary was assigned the wrong reward")
            frozen = deepcopy(transition_record(result,[0,0]))
            count = env.coordinator.consumed_joint_decisions
            try:
                env.step(0,0)
            except RuntimeError:
                pass
            else:
                raise AssertionError("ended boundary accepted another action")
            if env.coordinator.consumed_joint_decisions != count:
                raise AssertionError("rejected ended-episode call consumed an action")
            new = env.reset(seed=402)
            if result.training_only_info != frozen["training_only_info"] or new.training_only_info["episode_id"] == reset.training_only_info["episode_id"]:
                raise AssertionError("final reward/events were overwritten by reset")
            cases.append({"path":mode,"decisions":published,"final_transition":frozen,
                          "captured_before_reset":True,"new_episode_id":new.training_only_info["episode_id"]})
            print(json.dumps({"progress":"boundary","path":mode,"decisions":published}),flush=True)
        finally:
            try:
                ledger.record("boundaries",env,published,expected_error=env.state is LifecycleState.NEEDS_RECOVERY)
            finally:
                env.close()
    scenario = build_split_scenario("relay","train",0,seed=0,purpose="diagnostic",output_dir=output/"scenarios"/"worker_error")
    for fault in ("native_engine_exit","participant_process_exit"):
        env = TwoPlayerCoopEnv("relay",seed=0,scenario_manifest=scenario,work_root=output/"environments")
        published = 0
        try:
            env.reset()
            guest = env.coordinator._processes[1]
            owned = env.coordinator.native_processes
            guest_engine = next(identity for identity in owned if identity.player_index == 1)
            if fault == "native_engine_exit":
                process = psutil.Process(guest_engine.pid)
                if process.create_time() != guest_engine.create_time:
                    raise RuntimeError("owned native process identity changed before the fixture")
                process.kill()
                process.wait(timeout=10)
            else:
                # Deliberately bypass the worker finally block to expose orphan
                # cleanup; signal only this coordinator's own spawned worker.
                guest.terminate()
                guest.join(timeout=10)
                if guest.is_alive():
                    raise RuntimeError("controlled participant termination did not finish")
            before = env.coordinator.consumed_joint_decisions
            try:
                env.step(0,0)
            except EnvironmentIntegrityError as exc:
                error = exc.record
            else:
                raise AssertionError("participant failure produced a reward-bearing transition")
            consumed = env.coordinator.consumed_joint_decisions
            if env.state is not LifecycleState.NEEDS_RECOVERY or consumed != before+1:
                raise AssertionError("worker error lifecycle/accounting differs")
            try:
                env.step(0,0)
            except RuntimeError:
                pass
            else:
                raise AssertionError("worker-error retry bypassed recovery")
            if env.coordinator.consumed_joint_decisions != consumed:
                raise AssertionError("worker-error retry consumed another decision")
            fresh = env.recover()
            for identity in owned:
                try:
                    old = psutil.Process(identity.pid)
                    if old.create_time() == identity.create_time and old.is_running():
                        raise AssertionError("native engine from a failed participant survived recovery")
                except psutil.NoSuchProcess:
                    pass
            result = env.step(0,0)
            published += 1
            if not fresh.training_only_info["reset_actor_memories"] or result.terminated:
                raise AssertionError("worker-error recovery did not produce a fresh live episode")
            cases.append({"path":"worker_error","fault":fault,
                          "source":"Terminated exactly an owned guest engine or its participant; PID/creation-time guarded cleanup checked after recovery",
                          "native_pid":guest_engine.pid,"participant_pid":guest.pid,
                          "error_record":error,"reward_bearing_transition_returned":False,
                          "consumed_failed_attempts":1,"rejected_retry_consumed":0,"fresh_recovery":True,
                          "old_native_engines_gone":True})
            print(json.dumps({"progress":"boundary","path":"worker_error","fault":fault}),flush=True)
        finally:
            try:
                ledger.record("worker_error",env,published,expected_error=True)
            finally:
                env.close()
    return {"status":"passed","cases":cases,
            "success_path":"Covered by every native template mechanics completion.",
            "missing_terminal_images":"Covered by contract/wrapper tests: missing images stay None, validity/reconstruction masks false; no fabricated frame."}


def authoritative_event_probe(output: Path) -> dict:
    scenario = build_split_scenario("relay","train",0,seed=0,purpose="diagnostic",output_dir=output/"scenario")
    env = TwoPlayerCoopEnv("relay",seed=0,scenario_manifest=scenario,work_root=output/"environments")
    ledger = ValidationLedger(output.parent)
    published = 0
    try:
        initial = env.reset()
        operator = scenario["role_assignment"]["operator"]
        prior = initial.training_only_info["events"]["cumulative"]
        result = initial
        for target in ((144,96),(64,96)):
            for _ in range(100):
                player = result.training_only_info["players"][operator]
                if _arrived(player,target):
                    break
                actions = [0,0]
                actions[operator] = _move(player,target)
                result = env.step(*actions)
                published += 1
                validate_transition(result,prior,scenario)
                prior = deepcopy(result.training_only_info["events"]["cumulative"])
            else:
                raise AssertionError("event probe navigation did not converge")
            # Allow physical ceiling-open/close arrival events to complete.
            for _ in range(8):
                result = env.step(0,0)
                published += 1
                validate_transition(result,prior,scenario)
                prior = deepcopy(result.training_only_info["events"]["cumulative"])
        first, last = initial.training_only_info, result.training_only_info
        deltas = {name:last["events"]["cumulative"][name]-first["events"]["cumulative"][name]
                  for name in first["events"]["cumulative"]}
        if first["task_flags"]["plate_active"] or last["task_flags"]["plate_active"]:
            raise AssertionError("event probe endpoint plate flags should both be inactive")
        if not all(deltas[name] >= 1 for name in ("plate_entries","plate_exits","gate_opens","gate_closes")):
            raise AssertionError("cumulative counters lost the intermediate plate/gate episode")
        return {"status":"passed","initial_plate_flag":0,"final_plate_flag":0,"cumulative_differences":deltas,
                "published_joint_decisions":published,
                "scope":"Native events retained between deliberately selected observation endpoints across several decisions; single-decision hidden pulses are also tested at the registry unit boundary."}
    finally:
        try:
            ledger.record("event_probe",env,published,expected_error=env.state is LifecycleState.NEEDS_RECOVERY)
        finally:
            env.close()
