"""Build and verify P1.1-P1.4; do not mark P1.5-P1.7 complete."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from world_models.env.coop import EnvironmentIntegrityError, LifecycleState, TwoPlayerCoopEnv
from world_models.env.observations import save_rgb
from world_models.env.provenance import runtime_manifest
from world_models.env.scenario_builder import build_scenario


ROOT = Path(__file__).resolve().parents[1]


def source_hashes() -> dict:
    paths = set()
    for folder, pattern in (("src", "*.py"),("scripts","*.py"),("tests","*.py"),
                            ("configs","*.yaml"),("scenarios","*.acs"),("scenarios","*.md"),("docs","*.md")):
        paths.update((ROOT/folder).rglob(pattern))
    paths.update(ROOT/name for name in ("requirements-phase1.lock","requirements-phase0.lock","pyproject.toml","README.md"))
    return {str(path.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def smoke_task(task: str, seed: int, output: Path) -> dict:
    env = TwoPlayerCoopEnv(task,seed=seed,work_root=output/"environments")
    samples = []
    try:
        reset = env.reset(seed=seed + 1000)
        selected_iwads = validate_loaded_iwads(env, reset.training_only_info)
        original = [frame.data for frame in reset.observations]
        for player, frame in enumerate(reset.observations):
            save_rgb(frame,output/f"{task}_initial_player{player}.png")
        for index in range(20):
            result = env.step(0,0)
            info = result.training_only_info
            if result.metadata.actual_simulator_tics != env.frame_skip:
                raise RuntimeError("logical decision did not advance four synchronized tics")
            if info["task_flags"]["elapsed_decisions"] != env.clock.decision_index:
                raise RuntimeError("ACS and wrapper episode clocks disagree")
            samples.append({"decision":env.clock.decision_index,"tic":info["engine_tic"],
                            "reward":result.team_reward,"task_flags":info["task_flags"]})
            if result.terminated:
                raise RuntimeError("task terminated during the short no-op smoke fixture")
        if [frame.data for frame in reset.observations] != original:
            raise RuntimeError("saved actor images changed after engine advancement")
        resets = []
        for _ in range(2):
            next_reset = env.reset(seed=seed + 1001)
            flags = next_reset.training_only_info["task_flags"]
            if flags["milestone_count"] or flags["success"] or flags["failure"] or flags["elapsed_decisions"]:
                raise RuntimeError("reset retained stale task state")
            resets.append({"engine_tic":next_reset.training_only_info["engine_tic"],"flags":flags})
        return {"status":"passed","task_family":task,"seed":seed,"first_engine_seed":seed+1000,"samples":samples,"resets":resets,
                "scenario_manifest":env.scenario,"images_immutable":True,
                "players_selected_iwads":selected_iwads,
                "consumed_joint_decisions":env.coordinator.consumed_joint_decisions,
                "initialization_joint_decisions":env.initialization_joint_decisions}
    finally:
        env.close()


def validate_loaded_iwads(env: TwoPlayerCoopEnv, info: dict) -> list[dict]:
    selected = info["players_selected_iwads"]
    expected_path = Path(env.coordinator.config.iwad_path).resolve()
    expected_hash = env.coordinator.config.iwad_sha256
    if len(selected) != 2 or any(not item or Path(item["path"]).resolve() != expected_path
                                 or item["sha256"] != expected_hash
                                 or Path(item["configured_path"]).resolve() != expected_path for item in selected):
        raise RuntimeError("workers did not report the explicitly selected IWAD path/hash")
    return selected


def recovery_smoke(output: Path) -> dict:
    """Inject a wrapper conversion error after one actual joint dispatch."""
    from unittest.mock import patch
    env = TwoPlayerCoopEnv("relay", seed=0, work_root=output/"environments")
    try:
        reset = env.reset()
        selected = validate_loaded_iwads(env, reset.training_only_info)
        old_instances = [participant.instance_id for participant in env.coordinator.start().participants]
        env.step(0, 0)
        before_clock = env.clock.decision_index
        before_count = env.coordinator.consumed_joint_decisions
        before_episode = reset.training_only_info["episode_id"]
        with patch("world_models.env.coop.adapt_rgb", side_effect=ValueError("injected native RGB conversion fault")):
            try:
                env.step(0, 0)
            except EnvironmentIntegrityError as exc:
                record = exc.record
            else:
                raise AssertionError("injected conversion failure was published")
        fault_count = env.coordinator.consumed_joint_decisions
        if env.state is not LifecycleState.NEEDS_RECOVERY or env.clock.decision_index != before_clock or fault_count != before_count + 1:
            raise AssertionError("native fault did not preserve atomic clock and consumed dispatch")
        error_path = env.coordinator.run_dir/"last_error.json"
        if not error_path.is_file() or json.loads(error_path.read_text(encoding="utf-8")) != record:
            raise AssertionError("native integrity error was not durable")
        for retry in (lambda: env.step(0, 0), lambda: env.reset()):
            try:
                retry()
            except RuntimeError:
                pass
            else:
                raise AssertionError("interaction continued without recovery")
            if env.coordinator.consumed_joint_decisions != fault_count:
                raise AssertionError("blocked retry consumed a decision")
        fresh = env.recover()
        new_instances = [participant.instance_id for participant in env.coordinator.start().participants]
        recovered_count = env.coordinator.consumed_joint_decisions
        if (fresh.training_only_info["episode_id"] <= before_episode or not fresh.training_only_info["reset_actor_memories"]
                or env.clock.decision_index != 0 or env.state is not LifecycleState.READY
                or recovered_count <= fault_count):
            raise AssertionError("fresh recovery lost cumulative accounting or reset signals")
        if any(not current or current == previous for previous, current in zip(old_instances, new_instances)):
            raise AssertionError("recovery did not restart both native participants")
        validate_loaded_iwads(env, fresh.training_only_info)
        following = env.step(0, 0)
        if env.clock.decision_index != 1 or env.coordinator.consumed_joint_decisions != recovered_count + 1:
            raise AssertionError("recovered environment did not accept a valid next step")
        return {"status":"passed", "injected_fault":"RGB conversion after real engine dispatch",
                "wrapper_clock_before_and_after_fault":before_clock, "count_before_fault":before_count,
                "count_after_fault":fault_count, "blocked_retry_additional_decisions":0,
                "count_after_recovery_initialization":recovered_count,
                "count_after_valid_step":following.training_only_info["consumed_joint_decisions"],
                "old_episode_id":before_episode, "new_episode_id":fresh.training_only_info["episode_id"],
                "old_worker_instances":old_instances,"new_worker_instances":new_instances,
                "reset_actor_memories":True, "error_record":record,"error_path":str(error_path),
                "players_selected_iwads":selected}
    finally:
        env.close()


def combat_rule_checks(paired: dict, same_operator: dict) -> dict:
    """Inspect live traces rather than inferring combat rules from success."""
    cases = [json.loads(Path(pair["no_fire"]["record_path"]).read_text(encoding="utf-8")) for pair in paired["pairs"]]
    movement = {}
    for enemy in (1, 2):
        movement[str(enemy)] = all(len({(row["task_flags"][f"enemy_{enemy}_x"],row["task_flags"][f"enemy_{enemy}_y"])
                                        for row in case["records"] if row["task_flags"][f"enemy_{enemy}_alive"]}) > 1
                                  for case in cases)
    surviving = all(any(row["task_flags"]["stage"] >= 1 and row["task_flags"]["enemy_1_alive"]
                        and row["task_flags"]["enemy_2_alive"] for row in case["records"]) for case in cases)
    same = json.loads(Path(same_operator["record_path"]).read_text(encoding="utf-8"))
    operators = {row["operator_activating"] for row in same["records"] if row["operator_activating"] is not None}
    checks = {"both_enemies_moved_on_each_no_fire_layout":all(movement.values()),
              "enemy_1_survived_console_1_with_distinct_enemy_2":surviving,
              "same_player_operated_both_consoles_and_extracted":same["success"] and len(operators) == 1
                 and same["metrics"]["console_1_attempted"] and same["metrics"]["console_2_attempted"]}
    return {"checks":checks,"enemy_movement_by_id":movement,
            "same_operator_player_indices":sorted(operators)}


def verify(output: Path, seeds: list[int]) -> tuple[dict,bool]:
    from world_models.env.scripted import run_cover_diagnostics, run_scripted_task
    from world_models.env.marker_diagnostics import run_marker_audit
    initial_sources = source_hashes()
    report = {"schema_version":2,"generated_at_utc":datetime.now(timezone.utc).isoformat(),
              "scope":["P1.1","P1.2","P1.3","P1.4"],"status":"failed",
              "later_steps":{"P1.5":"not_started","P1.6":"not_completed","P1.7":"not_started"},
              "runtime":runtime_manifest(ROOT),"smoke":[],"scripted_completions":[],"inactive_partner_checks":[],
              "source_sha256_at_start":initial_sources}
    def checkpoint():
        (output/"verification_progress.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    checkpoint()
    if not all(pin["matches"] for pin in report["runtime"]["packages"].values()):
        raise RuntimeError("runtime does not match requirements-phase1.lock")
    tests = subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],
                           cwd=ROOT,capture_output=True,text=True,timeout=240,check=False)
    report["unit_tests"] = {"exit_code":tests.returncode,"stdout":tests.stdout,"stderr":tests.stderr}
    checkpoint()
    if tests.returncode:
        report["error"] = "unit tests failed"
        return report,False
    report["native_recovery"] = recovery_smoke(output/"recovery")
    checkpoint()
    print(json.dumps({"progress":"native_recovery","status":"passed"}),flush=True)
    for task in ("relay","cover"):
        report["smoke"].append(smoke_task(task,seeds[0],output))
        checkpoint()
        for seed in seeds:
            case = run_scripted_task(task,seed,output_dir=output/"mechanics")
            if task == "cover" and case["success"]:
                checks = case["mechanic_checks"]
                if checks["pause_without_use_decisions"] != 3 or checks["repeated_completed_use_decisions"] != 3:
                    raise RuntimeError("successful cover fixture did not exercise pause and repeated-USE checks")
            report["scripted_completions"].append(case)
            checkpoint()
            print(json.dumps({"progress":"mechanics","task":task,"seed":seed,"success":case["success"]}),flush=True)
            if not case["success"]:
                report["error"] = f"scripted {task} seed {seed} did not complete; remaining cases not run"
                report["source_sha256"] = source_hashes()
                return report,False
        for player in (0,1):
            case = run_scripted_task(task,seeds[0],inactive_player=player,output_dir=output/"mechanics")
            report["inactive_partner_checks"].append(case)
            checkpoint()
            print(json.dumps({"progress":"inactive_partner","task":task,"inactive":player,"success":case["success"]}),flush=True)
            if case["success"]:
                report["error"] = f"inactive-player {task} fixture unexpectedly completed"
                report["source_sha256"] = source_hashes()
                return report,False
    report["covering_fire_diagnostics"] = run_cover_diagnostics(seeds,output_dir=output/"cover_diagnostics")
    checkpoint()
    print(json.dumps({"progress":"paired_cover_diagnostics","seeds":seeds}),flush=True)
    report["same_operator_check"] = run_scripted_task("cover",seeds[0],output_dir=output/"same_operator",operator_policy="same",mechanics_probes=False)
    report["combat_rule_checks"] = combat_rule_checks(report["covering_fire_diagnostics"],report["same_operator_check"])
    report["marker_audits"] = []
    checkpoint()
    for task in ("relay","cover"):
        for seed in seeds:
            report["marker_audits"].append(run_marker_audit(task,seed,output_dir=output/"markers"))
            checkpoint()
            print(json.dumps({"progress":"marker_audit","task":task,"seed":seed}),flush=True)
    diagnostics_valid = all(all(pair["diagnostic_checks"].values()) for pair in report["covering_fire_diagnostics"]["pairs"])
    native_cases = [*report["smoke"],*report["scripted_completions"],*report["inactive_partner_checks"],
                    report["native_recovery"],report["same_operator_check"],*report["marker_audits"]]
    native_cases.extend(pair[condition] for pair in report["covering_fire_diagnostics"]["pairs"] for condition in ("fire","no_fire"))
    baseline = report["runtime"]["selected_iwad"]
    report["selected_iwad_consistent_across_native_cases"] = all(
        len(case.get("players_selected_iwads", [])) == 2
        and all(item and Path(item["path"]).resolve() == Path(baseline["path"]).resolve()
                and item["sha256"] == baseline["sha256"] for item in case["players_selected_iwads"])
        for case in native_cases)
    passed = (all(case["success"] for case in report["scripted_completions"])
              and all(not case["success"] for case in report["inactive_partner_checks"])
              and report["same_operator_check"]["success"] and diagnostics_valid
              and all(report["combat_rule_checks"]["checks"].values())
              and report["selected_iwad_consistent_across_native_cases"]
              and all(case["status"] == "passed" for case in report["marker_audits"]))
    report["status"] = "complete_P1.1_to_P1.4" if passed else "failed"
    report["source_sha256"] = source_hashes()
    report["sources_unchanged_during_verification"] = report["source_sha256"] == initial_sources
    report["runtime_at_end"] = runtime_manifest(ROOT)
    report["runtime_unchanged_during_verification"] = (
        {key:value for key,value in report["runtime"].items() if key != "recorded_at_utc"}
        == {key:value for key,value in report["runtime_at_end"].items() if key != "recorded_at_utc"})
    if not report["sources_unchanged_during_verification"]:
        report["status"], passed = "failed_source_drift", False
    if not report["runtime_unchanged_during_verification"]:
        report["status"], passed = "failed_runtime_drift", False
    report["interpretation"] = "Scripted controllers use privileged positions for mechanics validation; these are not learned-policy results or a P1.7 acceptance campaign."
    return report,passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("build","smoke","mechanics","cover-diagnostics","recovery-smoke","marker-audit","verify"))
    parser.add_argument("--task",choices=("relay","cover","both"),default="both")
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--inactive-player",type=int,choices=(0,1))
    parser.add_argument("--operator-policy",choices=("swap","same"),default="swap")
    parser.add_argument("--no-covering-fire",action="store_true")
    parser.add_argument("--no-mechanics-probes",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"artifacts"/"phase1_steps1_4")
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct nonnegative integers")
    if args.command == "verify" and args.task != "both":
        parser.error("verify covers both task families; use smoke/mechanics for a single task")
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    passed = True
    try:
        tasks = ("relay","cover") if args.task == "both" else (args.task,)
        if args.command == "build":
            report = {"scenarios":[build_scenario(task,seed) for task in tasks for seed in args.seeds]}
        elif args.command == "smoke":
            report = {"smoke":[smoke_task(task,args.seeds[0],output) for task in tasks]}
        elif args.command == "mechanics":
            from world_models.env.scripted import run_scripted_task
            cases = [run_scripted_task(task,seed,inactive_player=args.inactive_player,output_dir=output/"mechanics",
                                       covering_fire=not args.no_covering_fire, mechanics_probes=not args.no_mechanics_probes,
                                       operator_policy=args.operator_policy) for task in tasks for seed in args.seeds]
            passed = all(case["success"] == (args.inactive_player is None) for case in cases)
            report = {"status":"passed" if passed else "failed","cases":cases}
        elif args.command == "cover-diagnostics":
            from world_models.env.scripted import run_cover_diagnostics
            report = run_cover_diagnostics(args.seeds,output_dir=output,operator_policy=args.operator_policy)
            passed = all(all(pair["diagnostic_checks"].values()) for pair in report["pairs"])
            report["status"] = "passed" if passed else "diagnostic_incomplete"
        elif args.command == "recovery-smoke":
            report = recovery_smoke(output)
        elif args.command == "marker-audit":
            from world_models.env.marker_diagnostics import run_marker_audit
            cases = [run_marker_audit(task,seed,output_dir=output) for task in tasks for seed in args.seeds]
            passed = all(case["status"] == "passed" for case in cases)
            report = {"status":"passed" if passed else "failed","cases":cases}
        else:
            report,passed = verify(output,args.seeds)
    except Exception as exc:
        progress = output/"verification_progress.json"
        report = json.loads(progress.read_text(encoding="utf-8")) if args.command == "verify" and progress.is_file() else {}
        report.update(status="failed",scope=["P1.1","P1.2","P1.3","P1.4"],error=repr(exc),details=getattr(exc,"record",None))
        passed = False
    path = output/("acceptance.json" if args.command == "verify" else f"{args.command}_report.json")
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":report.get("status","passed"),"report":str(path)}),flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
