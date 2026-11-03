"""Behavioral regression tests for privileged environment diagnostics."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from world_models.contracts import BoundaryReason, EpisodeBoundary
from world_models.env.scripted import (_Controller, _aim, _enemy_targets, _move,
                                       _without_attack, run_cover_diagnostics,
                                       run_scripted_task)


def player(x=0, y=0, angle=0, health=100, ammo=200, armor=100):
    return {"POSITION_X": x, "POSITION_Y": y, "ANGLE": angle,
            "VELOCITY_X": 0, "VELOCITY_Y": 0, "HEALTH": health,
            "SELECTED_WEAPON_AMMO": ammo, "ARMOR": armor}


def scenario():
    return {"task_family": "cover", "role_assignment": {"operator": 0},
            "zones": {"console_1": [280, 152, 360, 232],
                      "console_2": [664, 536, 744, 616], "extraction": [896, 640, 1024, 768]},
            "deadline": 600, "frame_skip": 4, "hashes": {"wad_sha256": "same-map"}}


def flags(stage=0, elapsed=0, success=False):
    return {"stage": stage, "success": success, "failure": False,
            "progress_1": 20 if stage else 0, "progress_2": 20 if stage >= 2 else 0,
            "milestone_count": min(stage, 2), "elapsed_decisions": elapsed,
            "enemy_1_x": 704, "enemy_1_y": 192, "enemy_1_health": 20, "enemy_1_alive": 1,
            "enemy_2_x": 256, "enemy_2_y": 576, "enemy_2_health": 20 if stage else 0,
            "enemy_2_alive": int(stage > 0), "enemy_health": 40}


class ControllerTests(unittest.TestCase):
    def test_navigation_uses_observed_heading(self):
        for heading, expected in ((0, 1), (90, 4), (180, 2), (270, 3)):
            with self.subTest(heading=heading):
                self.assertEqual(_move(player(angle=heading), (100, 0)), expected)

    def test_operator_uses_console_before_guard_reaches_any_waypoint(self):
        controller = _Controller(scenario())
        actions = controller.actions([player(320, 192), player(192, 352)], flags())
        self.assertEqual(controller.guard_indices[0], 0)
        self.assertEqual(actions[0], 8)

    def test_extraction_begins_with_both_enemies_alive(self):
        controller = _Controller(scenario())
        actions = controller.actions([player(320, 192), player(704, 576)], flags(stage=2))
        self.assertTrue(all(action in (1, 2, 3, 4) for action in actions))
        self.assertEqual(controller.guard_indices, [0, 0])

    def test_fire_withholding_keeps_all_navigation_use_and_turn_actions(self):
        for stage in (0, 1, 2):
            for angle in (0, 90, 180, 270):
                first, second = _Controller(scenario()), _Controller(scenario())
                first.guard_indices = second.guard_indices = [3, 3]
                positions = [player(320, 192), player(448, 192, angle)]
                enabled = first.actions(copy.deepcopy(positions), flags(stage))
                withheld = second.actions(copy.deepcopy(positions), flags(stage), covering_fire=False)
                self.assertEqual(withheld, [_without_attack(action) for action in enabled])
        self.assertEqual([_without_attack(i) for i in (7, 9, 10, 11)], [0, 1, 3, 4])

    def test_stationary_guard_turns_and_aims_at_current_enemy_coordinates(self):
        self.assertEqual(_aim(player(), [(100, 0)]), 7)
        self.assertEqual(_aim(player(), [(0, 100)]), 5)
        self.assertEqual(_aim(player(), [(0, -100)]), 6)
        self.assertEqual(_aim(player(angle=359), [(100, 0)]), 7)
        state = flags(stage=1)
        state.update(enemy_1_alive=0, enemy_2_x=111, enemy_2_y=222)
        self.assertEqual(_enemy_targets(state), [(111, 222)])

    def test_same_operator_policy_operates_both_consoles(self):
        controller = _Controller(scenario(), operator_policy="same")
        self.assertEqual(controller.actions([player(320, 192), player()], flags())[0], 8)
        self.assertEqual(controller.actions([player(704, 576), player()], flags(stage=1))[0], 8)
        self.assertEqual(_Controller(scenario()).stage_operator(1), 1)


class _FakeEnv:
    """Three-decision engine fixture: both consoles then live-enemy extraction."""

    def __init__(self, *args, **kwargs):
        self.index = 0
        self.actions = []
        self.closed = False

    def _result(self, stage, success=False):
        positions = ([player(320, 192), player(192, 352)] if self.index == 0
                     else [player(320, 192, health=97, armor=98), player(704, 576)])
        info = {"players": positions, "task_flags": flags(stage, self.index, success),
                "scenario_manifest": scenario(), "engine_tic": 5 + 4 * self.index,
                "players_selected_iwads": [{"path": "fixed/freedoom2.wad", "configured_path": "fixed/freedoom2.wad", "sha256": "a" * 64}
                                           for _ in (0, 1)]}
        return SimpleNamespace(observations=(None, None), training_only_info=info,
                               terminated=success, team_reward=0.999 if success else 0.049,
                               boundary=EpisodeBoundary(BoundaryReason.SUCCESS if success else BoundaryReason.CONTINUE),
                               metadata=SimpleNamespace(actual_simulator_tics=4))

    def reset(self):
        return self._result(0)

    def step(self, *actions):
        self.actions.append(actions)
        self.index += 1
        return self._result(min(self.index, 2), self.index == 3)

    def close(self):
        self.closed = True


class DiagnosticReportTests(unittest.TestCase):
    def test_published_death_overrides_raw_success_flag(self):
        class DeathAtGoalEnv(_FakeEnv):
            def _result(self, stage, success=False):
                result = super()._result(stage, success)
                if success:
                    result.boundary = EpisodeBoundary(BoundaryReason.DEATH)
                    result.training_only_info["players"][0]["HEALTH"] = 0
                    result.team_reward = -1.001
                return result

        with tempfile.TemporaryDirectory() as directory, patch("world_models.env.scripted.TwoPlayerCoopEnv", DeathAtGoalEnv):
            result = run_scripted_task("cover", output_dir=directory, capture_images=False, mechanics_probes=False)
            self.assertEqual(result["final_flags"]["success"], True)
            self.assertFalse(result["success"])
            self.assertEqual(result["boundary_reason"], "death")
            self.assertIsNone(result["metrics"]["completion_decisions"])
            self.assertEqual(result["metrics"]["deaths"], 1)

    def test_zero_success_flag_and_published_deadline_report_failure(self):
        class DeadlineEnv(_FakeEnv):
            def _result(self, stage, success=False):
                result = super()._result(stage, success)
                if success:
                    result.boundary = EpisodeBoundary(BoundaryReason.DEADLINE)
                    result.training_only_info["task_flags"]["success"] = 0
                    result.team_reward = -1.001
                return result

        with tempfile.TemporaryDirectory() as directory, patch("world_models.env.scripted.TwoPlayerCoopEnv", DeadlineEnv):
            result = run_scripted_task("cover", output_dir=directory, capture_images=False, mechanics_probes=False)
            self.assertEqual(result["final_flags"]["success"], 0)
            self.assertFalse(result["success"])
            self.assertEqual(result["boundary_reason"], "deadline")
            self.assertIsNone(result["metrics"]["completion_seconds"])

    def test_reports_console_and_extraction_attempts_damage_and_no_fire(self):
        with tempfile.TemporaryDirectory() as directory, patch("world_models.env.scripted.TwoPlayerCoopEnv", _FakeEnv):
            result = run_scripted_task("cover", output_dir=directory, capture_images=False,
                                       covering_fire=False, mechanics_probes=False)
            metrics = result["metrics"]
            self.assertTrue(result["success"])
            self.assertEqual(metrics["operator_activation_damage"], 5)
            self.assertEqual(metrics["operator_activation_health_loss"], 3)
            self.assertEqual(metrics["operator_activation_armor_loss"], 2)
            self.assertEqual(metrics["total_health_loss"], 3)
            self.assertEqual(metrics["shots_fired"], 0)
            self.assertTrue(metrics["console_1_attempted"])
            self.assertTrue(metrics["console_2_attempted"])
            self.assertTrue(metrics["extraction_attempted"])
            trace = json.loads(Path(result["record_path"]).read_text())
            self.assertEqual(trace["records"][0]["actions"][0], 8)
            self.assertEqual(trace["records"][1]["actions"][1], 8)
            self.assertTrue(all(action in (1, 2, 3, 4) for action in trace["records"][2]["actions"]))

    def test_paired_reports_accept_a_null_covering_effect(self):
        with tempfile.TemporaryDirectory() as directory, patch("world_models.env.scripted.TwoPlayerCoopEnv", _FakeEnv):
            report = run_cover_diagnostics([0, 1], output_dir=directory, capture_images=False)
            self.assertEqual(report["research_comparison"], ["B", "D"])
            self.assertEqual(len(report["pairs"]), 2)
            self.assertEqual(report["aggregate"]["no_fire"]["team_success_rate"], 1)
            for pair in report["pairs"]:
                self.assertEqual(pair["differences"]["operator_damage_reduction"], 0)
                self.assertFalse(pair["fire"]["controller_config"]["mechanics_probes"])
                self.assertTrue(pair["no_fire"]["metrics"]["extraction_attempted"])

    def test_ammo_pickup_is_reported_without_erasing_observed_expenditure(self):
        class AmmoPickupEnv(_FakeEnv):
            def _result(self, stage, success=False):
                result = super()._result(stage, success)
                # A two-round decrease, a pickup restoring both rounds, then
                # one more decrease: net inventory loss would report only one.
                result.training_only_info["players"][0]["SELECTED_WEAPON_AMMO"] = (200, 198, 200, 199)[self.index]
                return result

        with tempfile.TemporaryDirectory() as directory, patch("world_models.env.scripted.TwoPlayerCoopEnv", AmmoPickupEnv):
            result = run_scripted_task("cover", output_dir=directory, capture_images=False, mechanics_probes=False)
            self.assertEqual(result["metrics"]["ammunition_used"], 3)
            self.assertEqual(result["metrics"]["observed_ammunition_gains"], 2)
            self.assertEqual(result["metrics"]["ammunition_gain_intervals"], 1)
            self.assertIn("lower-bound", result["metric_definitions"]["shots_fired"])
            self.assertIn("mask", result["metric_definitions"]["ammunition_used"])

    def test_changed_worker_iwad_provenance_rejects_episode(self):
        class ChangedIwadEnv(_FakeEnv):
            def _result(self, stage, success=False):
                result = super()._result(stage, success)
                if self.index:
                    result.training_only_info["players_selected_iwads"][0]["sha256"] = "b" * 64
                return result

        with tempfile.TemporaryDirectory() as directory, patch("world_models.env.scripted.TwoPlayerCoopEnv", ChangedIwadEnv):
            with self.assertRaisesRegex(AssertionError, "IWAD provenance changed"):
                run_scripted_task("cover", output_dir=directory, capture_images=False, mechanics_probes=False)

    def test_paired_diagnostics_reject_different_loaded_iwads(self):
        class DifferentIwadEnv(_FakeEnv):
            created = 0

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                type(self).created += 1
                self.digest = ("a" if type(self).created == 1 else "b") * 64

            def _result(self, stage, success=False):
                result = super()._result(stage, success)
                for item in result.training_only_info["players_selected_iwads"]:
                    item["sha256"] = self.digest
                return result

        with tempfile.TemporaryDirectory() as directory, patch("world_models.env.scripted.TwoPlayerCoopEnv", DifferentIwadEnv):
            with self.assertRaisesRegex(AssertionError, "identical IWAD"):
                run_cover_diagnostics([0], output_dir=directory, capture_images=False)

    def test_invalid_settings_rejected_before_engine_construction(self):
        with patch("world_models.env.scripted.TwoPlayerCoopEnv") as engine:
            for kwargs in ({"inactive_player": True}, {"operator_policy": "forced"}, {"covering_fire": 1}):
                with self.assertRaises(ValueError):
                    run_scripted_task("cover", **kwargs)
            for seeds in ([], [True], [-1], [0, 0]):
                with self.assertRaises(ValueError):
                    run_cover_diagnostics(seeds)
            engine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
