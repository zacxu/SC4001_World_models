"""Privileged environment diagnostics using only the twelve declared actions.

Training-only positions guide this controller; it is not a learned-policy
baseline. No actor is teleported, rotated or killed by an engine command.
Withholding fire changes only attack actions, not navigation/use/turn rules.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import mean

from world_models.contracts import BoundaryReason
from world_models.env.coop import TwoPlayerCoopEnv


def _arrived(player: dict, target: tuple[float, float]) -> bool:
    return (abs(target[0] - player["POSITION_X"]) <= 12
            and abs(target[1] - player["POSITION_Y"]) <= 12
            and abs(player.get("VELOCITY_X", 0)) <= 2.5
            and abs(player.get("VELOCITY_Y", 0)) <= 2.5)


def _move(player: dict, target: tuple[float, float], *, y_first: bool = False) -> int:
    """Brake toward a world-axis waypoint in the actor's actual reference frame."""
    dx, dy = target[0] - player["POSITION_X"], target[1] - player["POSITION_Y"]
    vx, vy = player.get("VELOCITY_X", 0), player.get("VELOCITY_Y", 0)
    checks = ((dy, vy, 0, 1), (dx, vx, 1, 0)) if y_first else ((dx, vx, 1, 0), (dy, vy, 0, 1))
    angle = math.radians(player.get("ANGLE", 0))
    for delta, velocity, axis_x, axis_y in checks:
        if abs(delta) > 12 or abs(velocity) > 2.5:
            error = delta - 4 * velocity
            if abs(error) <= 4:
                return 0
            direction = 1 if error > 0 else -1
            world_x, world_y = direction * axis_x, direction * axis_y
            forward = world_x * math.cos(angle) + world_y * math.sin(angle)
            left = -world_x * math.sin(angle) + world_y * math.cos(angle)
            if abs(forward) >= abs(left):
                return 1 if forward > 0 else 2
            return 3 if left > 0 else 4
    return 0


def _enemy_targets(flags: dict) -> list[tuple[float, float]]:
    return [(flags[f"enemy_{index}_x"], flags[f"enemy_{index}_y"])
            for index in (1, 2)
            if flags.get(f"enemy_{index}_alive", 0) and flags.get(f"enemy_{index}_health", 0) > 0]


def _aim(player: dict, targets: list[tuple[float, float]]) -> int:
    """Turn toward the nearest moving target, then fire the stock pistol.

    TURN is coarse at four tics per decision; the 8-degree window avoids repeated
overshoot. ATTACK commands are not counted as hits or actual shots.
    """
    if not targets:
        return 0
    x, y = player["POSITION_X"], player["POSITION_Y"]
    target = min(targets, key=lambda point: (point[0] - x) ** 2 + (point[1] - y) ** 2)
    desired = math.degrees(math.atan2(target[1] - y, target[0] - x))
    delta = (desired - player.get("ANGLE", 0) + 180) % 360 - 180
    return (5 if delta > 0 else 6) if abs(delta) > 8 else 7


def _without_attack(action: int) -> int:
    return {7: 0, 9: 1, 10: 3, 11: 4}.get(action, action)


class _Controller:
    """Waypoint script with no guard-readiness or enemy-death task gates."""

    def __init__(self, scenario: dict, *, operator_policy: str = "swap"):
        if operator_policy not in ("swap", "same"):
            raise ValueError("operator_policy must be 'swap' or 'same'")
        self.scenario = scenario
        self.operator_policy = operator_policy
        self.operator = scenario["role_assignment"]["operator"]
        self.runner = 1 - self.operator
        self.guard_indices = [0, 0]
        # Routes pass west of the pillar cells without occupying a console.
        # Short approaches let the fire treatment actually fire during a dwell;
        # neither console waits for these routes to finish.
        self.guard_paths = [[(384, 352), (384, 192), (448, 192)],
                            [(384, 192), (384, 576), (448, 576)]]
        if operator_policy == "same":
            # Vacate the first-console lane before moving west. Otherwise a
            # same-operator diagnostic can create a controller-only head-on
            # collision between players traversing y=192 in opposite directions.
            self.guard_paths[1] = [(448, 96), (128, 96), (128, 576)]

    def stage_operator(self, stage: int) -> int:
        return self.runner if stage == 1 and self.operator_policy == "swap" else self.operator

    def _guard(self, stage: int, player: dict, flags: dict) -> int:
        path = self.guard_paths[stage]
        index = self.guard_indices[stage]
        while index < len(path) and _arrived(player, path[index]):
            index += 1
        self.guard_indices[stage] = index
        if index < len(path):
            return _move(player, path[index])
        if not _arrived(player, path[-1]):
            return _move(player, path[-1])
        return _aim(player, _enemy_targets(flags))

    def actions(self, players: list[dict], flags: dict, *, covering_fire: bool = True) -> list[int]:
        actions = [0, 0]
        if self.scenario["task_family"] == "relay":
            if not flags["latch_active"]:
                actions[self.operator] = _move(players[self.operator], (144, 96))
                latch = tuple(self.scenario["scripted_waypoints"]["runner"][0])
                actions[self.runner] = 8 if _arrived(players[self.runner], latch) else _move(players[self.runner], latch)
            else:
                common = self.scenario["common_x"]
                actions[self.operator] = _move(players[self.operator], (common + 72, 312))
                actions[self.runner] = _move(players[self.runner], (common + 72, 392))
            return actions
        stage = int(flags["stage"])
        if stage < 2:
            operator = self.stage_operator(stage)
            guard = 1 - operator
            console = (320, 192) if stage == 0 else (704, 576)
            # Only this operator's position/momentum determines readiness.
            actions[operator] = 8 if _arrived(players[operator], console) else _move(players[operator], console)
            actions[guard] = self._guard(stage, players[guard], flags)
        else:
            bounds = self.scenario["zones"]["extraction"]
            exit_x, height = (bounds[0] + bounds[2]) / 2, bounds[3] - bounds[1]
            for index in (0, 1):
                target = (exit_x, bounds[1] + height * (0.3 if index == 0 else 0.7))
                actions[index] = _move(players[index], target, y_first=True)
        return actions if covering_fire else [_without_attack(action) for action in actions]


def _inside(player: dict, bounds: list[float]) -> bool:
    return bounds[0] <= player["POSITION_X"] < bounds[2] and bounds[1] <= player["POSITION_Y"] < bounds[3]


def run_scripted_task(task_family: str, seed: int = 0, *, inactive_player: int | None = None,
                      output_dir: str | Path = "artifacts/phase1/scenario_diagnostics",
                      capture_images: bool = True, covering_fire: bool = True,
                      mechanics_probes: bool = True, operator_policy: str = "swap") -> dict:
    if inactive_player not in (None, 0, 1) or isinstance(inactive_player, bool):
        raise ValueError("inactive_player must be None, 0 or 1")
    if not isinstance(covering_fire, bool) or not isinstance(mechanics_probes, bool):
        raise ValueError("covering_fire and mechanics_probes must be booleans")
    if operator_policy not in ("same", "swap"):
        raise ValueError("operator_policy must be 'swap' or 'same'")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    case_name = f"{task_family}_{seed}_inactive{inactive_player}_fire{int(covering_fire)}_{operator_policy}"
    env = TwoPlayerCoopEnv(task_family, seed=seed, work_root=output / "environments")
    records, image_paths = [], []
    pause_remaining, pause_progress, pause_verified = 0, None, 0
    replay_use_remaining, replay_use_verified = (3 if mechanics_probes else 0), 0
    maximum_clock_error = 0
    metrics = {"operator_activation_damage": 0.0, "operator_activation_health_loss": 0.0,
               "operator_activation_armor_loss": 0.0, "total_health_loss": 0.0, "total_armor_loss": 0.0,
               "deaths": 0, "ammunition_used": 0.0, "shots_fired": 0,
               "observed_ammunition_gains": 0.0, "ammunition_gain_intervals": 0,
               "fire_action_decisions": 0, "fire_during_activation_decisions": 0,
               "shots_during_activation": 0, "console_1_use_decisions": 0,
               "console_2_use_decisions": 0, "extraction_attempt_decisions": 0}

    def save_images(observations, decision):
        if not capture_images:
            return
        from world_models.env.observations import save_rgb
        for player, frame in enumerate(observations):
            if frame is not None:
                path = output / f"{case_name}_step{decision}_p{player}.png"
                save_rgb(frame, path)
                image_paths.append(str(path.resolve()))

    try:
        result = env.reset()
        save_images(result.observations, 0)
        info = result.training_only_info
        scenario = {**info["scenario_manifest"], "task_family": task_family}
        selected_iwads = deepcopy(info["players_selected_iwads"])
        initial_state = {"engine_tic": info["engine_tic"],
                         "players": [{name: p.get(name, 0) for name in ("POSITION_X", "POSITION_Y", "ANGLE", "HEALTH", "ARMOR", "SELECTED_WEAPON_AMMO")}
                                     for p in info["players"]],
                         "task_flags": info["task_flags"]}
        controller = _Controller(scenario, operator_policy=operator_policy)
        total_reward = 0.0
        final_boundary = BoundaryReason.CONTINUE
        for decision in range(scenario["deadline"]):
            players, flags = info["players"], info["task_flags"]
            actions = controller.actions(players, flags, covering_fire=covering_fire)
            is_pause = is_repeated_use = False
            if task_family == "cover" and mechanics_probes and flags["stage"] == 0:
                if pause_progress is None and flags["progress_1"] >= 5:
                    pause_progress, pause_remaining = flags["progress_1"], 3
                if pause_remaining:
                    actions[controller.operator] = 0
                    pause_remaining -= 1
                    is_pause = True
            if task_family == "cover" and flags["stage"] == 1 and replay_use_remaining:
                actions[controller.operator] = 8
                replay_use_remaining -= 1
                is_repeated_use = True
            if inactive_player is not None:
                actions[inactive_player] = 0
            active_operator = None
            if task_family == "cover":
                stage = int(flags["stage"])
                if stage < 2:
                    operator = controller.stage_operator(stage)
                    if actions[operator] == 8 and _inside(players[operator], scenario["zones"][f"console_{stage + 1}"]):
                        active_operator = operator
                        metrics[f"console_{stage + 1}_use_decisions"] += 1
                elif any(player["HEALTH"] > 0 for player in players):
                    metrics["extraction_attempt_decisions"] += 1
            step = env.step(*actions)
            final_boundary = step.boundary.reason
            total_reward += step.team_reward
            info = step.training_only_info
            if info["players_selected_iwads"] != selected_iwads:
                raise AssertionError("loaded IWAD provenance changed within the diagnostic episode")
            after = info["players"]
            health_loss = [max(0.0, max(0, before["HEALTH"]) - max(0, current["HEALTH"])) for before, current in zip(players, after)]
            armor_loss = [max(0.0, before.get("ARMOR", 0) - current.get("ARMOR", 0)) for before, current in zip(players, after)]
            rounds = [max(0.0, before.get("SELECTED_WEAPON_AMMO", 0) - current.get("SELECTED_WEAPON_AMMO", 0))
                      for before, current in zip(players, after)]
            ammo_gains = [max(0.0, current.get("SELECTED_WEAPON_AMMO", 0) - before.get("SELECTED_WEAPON_AMMO", 0))
                          for before, current in zip(players, after)]
            spent = sum(rounds)
            metrics["total_health_loss"] += sum(health_loss)
            metrics["total_armor_loss"] += sum(armor_loss)
            if active_operator is not None:
                metrics["operator_activation_health_loss"] += health_loss[active_operator]
                metrics["operator_activation_armor_loss"] += armor_loss[active_operator]
                metrics["operator_activation_damage"] += health_loss[active_operator] + armor_loss[active_operator]
                metrics["fire_during_activation_decisions"] += actions[1 - active_operator] in (7, 9, 10, 11)
                metrics["shots_during_activation"] += int(rounds[1 - active_operator])
            metrics["deaths"] += sum(before["HEALTH"] > 0 >= current["HEALTH"] for before, current in zip(players, after))
            metrics["ammunition_used"] += spent
            metrics["shots_fired"] += int(spent)  # Lower bound; a clip pickup can mask a shot.
            metrics["observed_ammunition_gains"] += sum(ammo_gains)
            metrics["ammunition_gain_intervals"] += any(gain > 0 for gain in ammo_gains)
            metrics["fire_action_decisions"] += sum(action in (7, 9, 10, 11) for action in actions)
            if is_pause:
                if info["task_flags"]["progress_1"] != pause_progress:
                    raise AssertionError("console progress advanced without USE")
                pause_verified += 1
            if is_repeated_use:
                if info["task_flags"]["milestone_count"] != 1 or abs(step.team_reward + 0.001) > 1e-9:
                    raise AssertionError("completed console rewarded repeated USE")
                replay_use_verified += 1
            if info["task_flags"]["milestone_count"] < flags["milestone_count"]:
                raise AssertionError("irreversible milestone counter regressed")
            clock_error = abs(info["task_flags"]["elapsed_decisions"] - (decision + 1))
            maximum_clock_error = max(maximum_clock_error, clock_error)
            if not step.terminated and clock_error:
                raise AssertionError("task and coordinator decision clocks diverged")
            records.append({"decision": decision + 1, "actions": actions, "reward": step.team_reward,
                            "boundary_reason": final_boundary.value,
                            "engine_tic": info["engine_tic"], "actual_tics": step.metadata.actual_simulator_tics,
                            "task_flags": info["task_flags"], "positions": [[p["POSITION_X"], p["POSITION_Y"]] for p in after],
                            "angles": [p.get("ANGLE", 0) for p in after], "health": [p["HEALTH"] for p in after],
                            "ammunition": [p.get("SELECTED_WEAPON_AMMO", 0) for p in after],
                            "armor": [p.get("ARMOR", 0) for p in after],
                            "operator_activating": active_operator, "health_loss": health_loss, "armor_loss": armor_loss,
                            "observed_ammunition_decreases": rounds, "observed_ammunition_gains": ammo_gains})
            if step.team_reward > 0 or step.terminated:
                save_images(step.observations, decision + 1)
            if step.terminated:
                break
        # The wrapper applies death/failure precedence when events share a
        # repeated action. A raw ACS success flag alone is not the outcome.
        success = final_boundary is BoundaryReason.SUCCESS
        metrics.update({"team_success": success, "elapsed_decisions": len(records),
                        "boundary_reason": final_boundary.value,
                        "completion_decisions": len(records) if success else None,
                        "completion_seconds": len(records) * scenario.get("frame_skip", 4) / 35 if success else None,
                        "console_1_attempted": metrics["console_1_use_decisions"] > 0,
                        "console_2_attempted": metrics["console_2_use_decisions"] > 0,
                        "extraction_attempted": metrics["extraction_attempt_decisions"] > 0})
        summary = {"task_family": task_family, "seed": seed, "inactive_player": inactive_player,
                   "layout_seed": info.get("layout_seed", seed), "engine_seed": info.get("engine_seed", seed),
                   "initial_state": initial_state,
                   "players_selected_iwads": selected_iwads,
                   "success": success, "decisions": len(records), "return": total_reward,
                   "boundary_reason": final_boundary.value,
                   "final_flags": info["task_flags"], "final_players": info["players"],
                   "scenario_hash": scenario["hashes"]["wad_sha256"],
                   "controller_kind": "privileged_mechanics_diagnostic_12_actions",
                   "controller_config": {"covering_fire": covering_fire, "mechanics_probes": mechanics_probes,
                                         "operator_policy": operator_policy,
                                         "attack_withholding": "7->0,9->1,10->3,11->4; navigation/use/turn rules unchanged"},
                   "metrics": metrics, "image_paths": image_paths,
                   "metric_definitions": {"operator_activation_damage": "Observed health plus armor loss during a USE decision inside the active console; components also reported separately.",
                                          "ammunition_used": "Lower-bound observed ammunition decreases between decisions. Stock clip pickups can mask expenditure within the same four-tic interval; observed increases are reported separately.",
                                          "shots_fired": "Compatibility key for a lower-bound fixed-pistol shot estimate from observed ammunition decreases, not exact engine shot events or ATTACK-command count.",
                                          "shots_during_activation": "The same lower-bound ammunition-decrease estimate restricted to the covering player during active-console USE.",
                                          "completion_seconds": "Successful episode decisions * action-repeat / 35 engine tics per second; excludes reset warm-up."},
                   "mechanic_checks": {"pause_without_use_decisions": pause_verified,
                                       "repeated_completed_use_decisions": replay_use_verified,
                                       "milestones_monotonic": True, "maximum_clock_error": maximum_clock_error,
                                       "both_alive_at_end": all(p["HEALTH"] > 0 for p in info["players"])},
                   "records": records}
        path = output / f"{case_name}.json"
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return {**{key: value for key, value in summary.items() if key not in ("records", "final_players")},
                "record_path": str(path.resolve())}
    finally:
        env.close()


def run_cover_diagnostics(seeds=(0, 1, 2), *, output_dir: str | Path = "artifacts/phase1/cover_diagnostics",
                          capture_images: bool = True, operator_policy: str = "swap") -> dict:
    """Paired development diagnostics; no positive covering effect is required."""
    seeds = tuple(seeds)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("paired diagnostic seeds must be unique")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pairs = []
    for seed in seeds:
        enabled = run_scripted_task("cover", seed, output_dir=output / "fire", capture_images=capture_images,
                                    covering_fire=True, mechanics_probes=False, operator_policy=operator_policy)
        withheld = run_scripted_task("cover", seed, output_dir=output / "no_fire", capture_images=capture_images,
                                     covering_fire=False, mechanics_probes=False, operator_policy=operator_policy)
        if enabled["scenario_hash"] != withheld["scenario_hash"]:
            raise AssertionError("paired controllers did not use identical scenario bytes")
        if any(enabled[name] != withheld[name] for name in ("layout_seed", "engine_seed", "initial_state")):
            raise AssertionError("paired diagnostics did not begin from identical seeded engine states")
        if enabled["players_selected_iwads"] != withheld["players_selected_iwads"]:
            raise AssertionError("paired diagnostics did not load identical IWAD paths and hashes")
        fire, no_fire = enabled["metrics"], withheld["metrics"]
        if no_fire["ammunition_used"] or no_fire["fire_action_decisions"]:
            raise AssertionError("the no-fire diagnostic fired")
        pairs.append({"seed": seed, "scenario_hash": enabled["scenario_hash"], "fire": enabled, "no_fire": withheld,
                      "diagnostic_checks": {"fire_treatment_exercised": fire["shots_fired"] > 0,
                                            "shots_during_operator_activation": fire["shots_during_activation"] > 0,
                                            "withheld_zero_shots": no_fire["shots_fired"] == 0,
                                            "withheld_attempted_both_consoles_and_extraction": all(no_fire[key] for key in ("console_1_attempted", "console_2_attempted", "extraction_attempted"))},
                      "differences": {"success_fire_minus_no_fire": int(fire["team_success"]) - int(no_fire["team_success"]),
                                      "operator_damage_reduction": no_fire["operator_activation_damage"] - fire["operator_activation_damage"],
                                      "total_health_loss_reduction": no_fire["total_health_loss"] - fire["total_health_loss"],
                                      "deaths_reduction": no_fire["deaths"] - fire["deaths"],
                                      "completion_decisions_reduction": (no_fire["completion_decisions"] - fire["completion_decisions"])
                                      if fire["team_success"] and no_fire["team_success"] else None,
                                      "ammunition_fire_minus_no_fire": fire["ammunition_used"] - no_fire["ammunition_used"]}})
    names = ("operator_activation_damage", "operator_activation_health_loss", "operator_activation_armor_loss",
             "total_health_loss", "total_armor_loss", "deaths", "ammunition_used", "shots_fired",
             "observed_ammunition_gains", "ammunition_gain_intervals", "shots_during_activation",
             "fire_action_decisions", "elapsed_decisions")
    aggregates = {condition: {**{name: mean(pair[condition]["metrics"][name] for pair in pairs) for name in names},
                              "team_success_rate": mean(pair[condition]["success"] for pair in pairs),
                              "all_attempt_console_1": all(pair[condition]["metrics"]["console_1_attempted"] for pair in pairs),
                              "all_attempt_console_2": all(pair[condition]["metrics"]["console_2_attempted"] for pair in pairs),
                              "all_attempt_extraction": all(pair[condition]["metrics"]["extraction_attempted"] for pair in pairs)}
                  for condition in ("fire", "no_fire")}
    for condition in ("fire", "no_fire"):
        completed = [pair[condition]["metrics"]["completion_decisions"] for pair in pairs if pair[condition]["success"]]
        aggregates[condition]["mean_completion_decisions_successes_only"] = mean(completed) if completed else None
    report = {"kind": "paired_development_environment_diagnostic", "research_comparison": ["B", "D"],
              "seeds": list(seeds), "operator_policy": operator_policy, "pairs": pairs, "aggregate": aggregates,
              "interpretation": "Paired descriptive diagnostics only; a null or negative covering effect is valid. Damage reduction measures protection; success improvement is stronger task evidence. No-fire failures before extraction do not demonstrate protection by themselves."}
    path = output / "paired_cover_diagnostics.json"
    report["report_path"] = str(path.resolve())
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("relay", "cover"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inactive-player", type=int, choices=(0, 1))
    parser.add_argument("--no-covering-fire", action="store_true")
    parser.add_argument("--no-mechanics-probes", action="store_true")
    parser.add_argument("--operator-policy", choices=("same", "swap"), default="swap")
    args = parser.parse_args()
    print(json.dumps(run_scripted_task(args.task, args.seed, inactive_player=args.inactive_player,
                                       covering_fire=not args.no_covering_fire,
                                       mechanics_probes=not args.no_mechanics_probes,
                                       operator_policy=args.operator_policy), indent=2))
