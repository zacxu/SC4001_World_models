"""Action-only native camera fixtures for extraction-marker visual review.

Privileged positions guide the fixture's navigation and camera orientation.
Within each fresh episode, every change of position, heading, lock state and
outcome uses the declared twelve actions through the public environment. This is not an image navigator
or evidence of learned visual competence. Saved PNGs are the exact 128x96 local
actor observations, including the normal HUD.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

from world_models.env.coop import TwoPlayerCoopEnv
from world_models.env.observations import save_rgb
from world_models.env.scenario_builder import _parameters
from world_models.env.scripted import _Controller, _aim, _arrived, _enemy_targets, _inside, _move


def _face(player: dict, target: tuple[float, float], *, tolerance: float = 8) -> int:
    desired = math.degrees(math.atan2(target[1] - player["POSITION_Y"],
                                     target[0] - player["POSITION_X"]))
    delta = (desired - player.get("ANGLE", 0) + 180) % 360 - 180
    return (5 if delta > 0 else 6) if abs(delta) > tolerance else 0


def _static_boundary_check(task: str, seed: int) -> dict:
    scenario, cells, _ = _parameters(task, seed)
    marker = scenario["extraction_marker"]
    x1, y1, x2, y2 = scenario["zones"]["extraction"]
    expected = {(x, y) for x in range(x1, x2, 64) for y in range(y1, y2, 64)}
    actual = {point for point, props in cells.items() if props.get("tag") == marker["sector_tag"]}
    matched = expected == actual and all(cells[point]["floor"] == marker["locked_floor"] for point in expected)
    if not matched:
        raise AssertionError("visible extraction footprint differs from success rectangle")
    return {"passed": matched, "bounds": [x1, y1, x2, y2],
            "marked_sector_count": len(actual), "containment": marker["containment"],
            "source": "same scenario parameters generate both marked sector union and ACS InExtraction bounds"}


def run_marker_audit(task: str, seed: int = 0, *, output_dir: str | Path = "artifacts/phase1/marker_audit") -> dict:
    """Save locked/open frontal and oblique views reached with ordinary actions.

    Relay's plate holder cannot reach locked extraction before the latch opens.
    Its runner supplies the locked views; both players supply open views. Role
    swaps across seeds cover both player indices without bypassing that mechanic.
    Cover supplies both players in both states. Finite enemies are first cleared
    with ordinary aiming/firing to make the visual fixture reproducible; these
    episodes are separate from the paired covering-fire diagnostic.
    """
    if task not in ("relay", "cover") or type(seed) is not int or seed < 0:
        raise ValueError("task must be relay/cover and seed a nonnegative integer")
    destination = Path(output_dir) / f"{task}_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    report = {"kind": "native_action_only_extraction_marker_fixture", "task_family": task,
              "seed": seed, "status": "running", "actor_image_size": [128, 96],
              "boundary_check": _static_boundary_check(task, seed), "captures": [],
              "records": [], "episode_resets": 0, "limitations": [
                  "Privileged positions guide fixture camera placement. Images require human review; this is not a learned image navigator.",
                  "Finite enemies may be cleared before cover images; this fixture is separate from the paired fire/no-fire diagnostic."]}
    env = TwoPlayerCoopEnv(task, seed=seed, work_root=destination / "environment")
    current = None
    scenario = None
    phase = "reset"

    def verify_iwads(info):
        selected = info["players_selected_iwads"]
        if "players_selected_iwads" not in report:
            report["players_selected_iwads"] = deepcopy(selected)
        elif selected != report["players_selected_iwads"]:
            raise AssertionError("native marker fixture participants changed the selected IWAD")

    def advance(actions, *, allow_success=False):
        nonlocal current
        current = env.step(*actions)
        info = current.training_only_info
        verify_iwads(info)
        flags, players = info["task_flags"], info["players"]
        report["records"].append({"episode_id": info["episode_id"], "decision": info["decision_index"], "phase": phase,
                                  "actions": list(actions), "engine_tic": info["engine_tic"],
                                  "positions": [[p["POSITION_X"], p["POSITION_Y"]] for p in players],
                                  "angles": [p.get("ANGLE", 0) for p in players],
                                  "task_flags": flags})
        if current.terminated:
            if not allow_success or not flags["success"]:
                raise RuntimeError(f"marker fixture ended during {phase}: {current.boundary}")
            if not all(_inside(player, scenario["zones"]["extraction"]) and player["HEALTH"] > 0 for player in players):
                raise AssertionError("native success disagrees with the visible extraction rectangle")
        return info

    def follow(paths, *, hold=None, allow_success=False, limit=150):
        # Native TURN has a coarse increment. Return to a near-cardinal heading
        # before translating; arbitrary oblique view headings can otherwise
        # make a sequential world-axis waypoint controller oscillate.
        for _ in range(32):
            players = current.training_only_info["players"]
            actions = [0, 0] if hold is None else list(hold)
            turning = False
            for player in paths:
                actor = players[player]
                actions[player] = _face(actor, (actor["POSITION_X"] + 1024, actor["POSITION_Y"]), tolerance=8)
                turning |= actions[player] != 0
            if not turning:
                break
            advance(actions, allow_success=allow_success)
            if current.terminated:
                return
        else:
            raise RuntimeError("cardinal camera navigation alignment did not converge")
        indices = {player: 0 for player in paths}
        for _ in range(limit):
            players = current.training_only_info["players"]
            actions = [0, 0] if hold is None else list(hold)
            pending = False
            for player, points in paths.items():
                while indices[player] < len(points) and _arrived(players[player], points[indices[player]]):
                    indices[player] += 1
                if indices[player] < len(points):
                    pending = True
                    actions[player] = _move(players[player], points[indices[player]])
            if not pending:
                return
            advance(actions, allow_success=allow_success)
            if current.terminated:
                return
        raise RuntimeError(f"marker route did not converge during {phase}")

    def face_and_capture(player, target, name, state):
        for _ in range(32):
            actor = current.training_only_info["players"][player]
            action = _face(actor, target)
            if action == 0:
                break
            actions = [0, 0]
            actions[player] = action
            advance(actions)
        else:
            raise RuntimeError("marker camera turn failed to converge")
        # A real noop removes movement bob/turn transients without fabricating pixels.
        advance([0, 0])
        info = current.training_only_info
        flags = info["task_flags"]
        opened = bool(flags["latch_active"] if task == "relay" else flags["stage"] >= 2)
        if opened != (state == "unlocked"):
            raise AssertionError("camera fixture has the wrong extraction lock state")
        images = []
        for index, frame in enumerate(current.observations):
            if frame is None:
                raise AssertionError("marker fixture received no actor image")
            path = destination / f"{state}_camera{player}_{name}_p{index}.png"
            save_rgb(frame, path)
            images.append(str(path.resolve()))
        report["captures"].append({"state": state, "camera_player": player, "view": name,
                                   "episode_id": info["episode_id"], "decision": info["decision_index"], "image_paths": images,
                                   "positions": [[p["POSITION_X"], p["POSITION_Y"]] for p in info["players"]],
                                   "angles": [p.get("ANGLE", 0) for p in info["players"]],
                                   "expected_floor": "EXFOPEN" if opened else "EXFLOCK",
                                   "expected_sign": "EXSOPEN" if opened else "EXSLOCK"})

    def clear_threats():
        if task != "cover":
            return
        for _ in range(100):
            info = current.training_only_info
            targets = _enemy_targets(info["task_flags"])
            if not targets:
                return
            advance([_aim(player, targets) for player in info["players"]])
        raise RuntimeError("ordinary-action marker fixture could not clear finite enemies")

    def relay_views(player, state):
        common = scenario["common_x"]
        views = [
            ("west", [(common + 48, 352), (common - 48, 352)]),
            ("south", [(common + 48, 352), (common + 48, 240)]),
            ("northwest", [(common + 48, 352), (common - 48, 352), (common - 48, 416)]),
        ]
        for name, path in views:
            follow({player: path})
            face_and_capture(player, (common + 128, 384), name, state)

    def cover_views(player, state):
        width = scenario["zones"]["extraction"][2]
        # Every movement around the central pillars uses the clear y=608 lane.
        follow({player: [(384, 576), (384, 608), (width - 192, 608)]})
        views = [
            ("west", [(width - 192, 704)]),
            ("south", [(width - 64, 704), (width - 64, 608)]),
            ("southwest", [(width - 64, 576), (width - 192, 576)]),
        ]
        for name, path in views:
            follow({player: path})
            face_and_capture(player, (width, 704), name, state)

    try:
        current = env.reset()
        verify_iwads(current.training_only_info)
        report["episode_resets"] += 1
        scenario = current.training_only_info["scenario_manifest"]
        report["scenario_hash"] = scenario["hashes"]["wad_sha256"]
        report["scenario_version"] = scenario["scenario_version"]
        operator = scenario["role_assignment"]["operator"]
        runner = 1 - operator
        controller = _Controller(scenario)
        if task == "relay":
            common = scenario["common_x"]
            phase = "locked_runner_approach"
            # The plate stays held until the runner has crossed its real gate.
            for _ in range(100):
                players = current.training_only_info["players"]
                if _arrived(players[operator], (144, 96)) and _arrived(players[runner], (common - 48, 352)):
                    break
                actions = [0, 0]
                actions[operator] = _move(players[operator], (144, 96))
                actions[runner] = _move(players[runner], (common - 48, 352))
                advance(actions)
            else:
                raise RuntimeError("relay locked camera could not pass plate-controlled gate")
            phase = "locked_views"
            relay_views(runner, "locked")
            report["limitations"].append(f"Player {operator} must hold the relay plate and cannot reach locked extraction before latch release; player {runner} supplies locked views on this layout.")
            # A separate standard episode keeps visual travel out of the fixed
            # 400-decision task budget used for the open-marker views.
            current = env.reset()
            verify_iwads(current.training_only_info)
            report["episode_resets"] += 1
            controller = _Controller(scenario)
            phase = "ordinary_latch_use"
            for _ in range(100):
                info = current.training_only_info
                if info["task_flags"]["latch_active"]:
                    break
                advance(controller.actions(info["players"], info["task_flags"]))
            else:
                raise AssertionError("ordinary USE failed to unlock relay extraction")
            follow({runner: [(common - 48, 288)]})
            phase = "unlocked_operator_approach"
            follow({operator: [(common + 48, 96), (common + 48, 352)]})
            phase = "unlocked_views"
            relay_views(operator, "unlocked")
            follow({operator: [(common - 128, 416)]})
            relay_views(runner, "unlocked")
            phase = "visible_region_success"
            follow({operator: [(common - 128, 312), (common + 72, 312)],
                    runner: [(common - 48, 392), (common + 72, 392)]}, allow_success=True)
        else:
            phase = "visual_fixture_clear_initial_enemy"
            clear_threats()
            width = scenario["zones"]["extraction"][2]
            # Keep the other player outside the success footprint and every camera route.
            park = (width - 256, 736)
            for state in ("locked", "unlocked"):
                phase = state + "_views"
                for camera in (0, 1):
                    if camera == 1:
                        # Vacate the shared parking point before swapping cameras.
                        # Otherwise both solid player bodies would target one point.
                        follow({camera: [(384, 736), (384, 608)]})
                    other = 1 - camera
                    current_x = current.training_only_info["players"][other]["POSITION_X"]
                    if state == "locked" and camera == 0:
                        # Initial starts share x=192; take a parallel lane so
                        # the parked player's route cannot hit the camera body.
                        current_y = current.training_only_info["players"][other]["POSITION_Y"]
                        follow({other: [(128, current_y), (128, 736), park]})
                    else:
                        follow({other: [(current_x, 736), park]})
                    cover_views(camera, state)
                if state == "locked":
                    # Separate visual states into fresh standard episodes, so
                    # camera travel does not consume the task's fixed deadline.
                    # No camera coordinates or engine commands are used by reset.
                    current = env.reset()
                    verify_iwads(current.training_only_info)
                    report["episode_resets"] += 1
                    controller = _Controller(scenario)
                    phase = "ordinary_console_use"
                    for _ in range(250):
                        info = current.training_only_info
                        if info["task_flags"]["stage"] >= 2:
                            break
                        advance(controller.actions(info["players"], info["task_flags"]))
                    else:
                        raise RuntimeError("ordinary console controller did not unlock extraction")
                    phase = "visual_fixture_clear_remaining_enemies"
                    clear_threats()
            phase = "visible_region_success"
            follow({0: [(width - 256, 672), (width - 64, 672)],
                    1: [(width - 192, 736), (width - 64, 736)]}, allow_success=True)
        flags = current.training_only_info["task_flags"]
        if not current.terminated or not flags["success"]:
            # Success may become authoritative on the next native tic after settling.
            advance([0, 0], allow_success=True)
        if not current.terminated or not current.training_only_info["task_flags"]["success"]:
            raise AssertionError("both living players entered marked extraction without success")
        report.update(status="passed", final_flags=current.training_only_info["task_flags"],
                      decisions=len(report["records"]), final_episode_decisions=current.training_only_info["decision_index"],
                      success_inside_visible_region=True,
                      human_visual_review="pending; inspect saved actor frames for sign, floor, lock state and approach visibility")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), failed_phase=phase)
        raise
    finally:
        env.close()
        path = destination / "marker_audit.json"
        report["report_path"] = str(path.resolve())
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
