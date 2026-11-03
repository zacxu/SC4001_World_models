"""P2.2 architectural heads and exact reward/clock label contracts."""

from dataclasses import replace
import unittest

import torch

from world_models.data.schema import ENEMY_FIELDS, PLAYER_FIELDS, REASONS
from world_models.env.targets import TASK_FIELDS
from world_models.models.heads import (
    AuxiliaryHeads, CONTINUE, FAILURE, SUCCESS, TaskSpec, TransitionHeads,
    auxiliary_fields, declared_event_fields, encode_auxiliary_targets,
    encode_transition_targets, resolve_outcomes, total_reward,
)


def transitions(reasons, *, family="relay", decisions=None, bonuses=None):
    task = TaskSpec.for_family(family)
    bonus = [0.0] * len(reasons) if bonuses is None else bonuses
    reward = [task.step_cost + b + (task.success_reward if reason == "success" else
                                   task.failure_reward if reason in ("death", "failure", "deadline") else 0.0)
              for reason, b in zip(reasons, bonus)]
    return {"termination_reason": torch.tensor([REASONS.index(reason) for reason in reasons], dtype=torch.uint8),
            "decision_index": torch.tensor(decisions or [0] * len(reasons)),
            "team_reward": torch.tensor(reward, dtype=torch.float64),
            "terminated": torch.tensor([r in ("success", "death", "failure", "deadline") for r in reasons]),
            "truncated": torch.tensor([r == "collection_cutoff" for r in reasons])}


def auxiliary_arrays(batch=2):
    state = {
        "normalized_player_state": torch.zeros(batch, 2, len(PLAYER_FIELDS)),
        "privileged_validity_mask": torch.ones(batch, 2, len(PLAYER_FIELDS), dtype=torch.bool),
        "normalized_task_state": torch.zeros(batch, len(TASK_FIELDS)),
        "task_state": torch.zeros(batch, len(TASK_FIELDS)),
        "task_validity_mask": torch.ones(batch, len(TASK_FIELDS), dtype=torch.bool),
        "normalized_enemy_state": torch.zeros(batch, 2, len(ENEMY_FIELDS)),
        "enemy_validity_mask": torch.ones(batch, 2, len(ENEMY_FIELDS), dtype=torch.bool),
    }
    transition = {"event_deltas": torch.zeros(batch, len(declared_event_fields()), dtype=torch.int64),
                  "event_delta_validity_mask": torch.ones(batch, len(declared_event_fields()), dtype=torch.bool)}
    return state, transition


class TaskHeadsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_task_specs_match_declared_family_and_terminal_stage(self):
        for family, deadline, classes in (("relay", 400, 3), ("cover", 600, 4)):
            task = TaskSpec.for_family(family)
            self.assertEqual(task.deadline, deadline)
            self.assertEqual(task.max_bonus, 0.1)
            self.assertEqual(task.stage_classes, classes)
        with self.assertRaises(ValueError):
            TaskSpec.for_family("other")
        with self.assertRaises(ValueError):
            replace(TaskSpec(), deadline=True)

    def test_task_spec_preserves_resolved_experiment_reward_and_clock(self):
        resolved = {"task": {"name": "cover", "deadline": 500, "step_cost": -0.002,
                             "success_reward": 2.0, "failure_reward": -2.0,
                             "milestone_bonus": 0.02, "milestone_count": 2}}
        task = TaskSpec.from_experiment(resolved)
        self.assertEqual(task.task_family, "cover")
        self.assertEqual(task.deadline, 500)
        self.assertEqual(task.step_cost, -0.002)
        self.assertEqual(task.success_reward, 2.0)
        self.assertEqual(task.failure_reward, -2.0)
        self.assertEqual(task.max_bonus, 0.04)
        self.assertEqual(task.stage_classes, 4)

    def test_joint_outcome_truth_table_and_deadline_precedence(self):
        task = TaskSpec.for_family("relay")
        resolved = resolve_outcomes(torch.tensor([CONTINUE, SUCCESS, FAILURE, CONTINUE, SUCCESS, FAILURE]),
                                    torch.tensor([1, 1, 1, 400, 400, 400]), task)
        self.assertEqual(resolved.outcome.tolist(), [CONTINUE, SUCCESS, FAILURE, FAILURE, SUCCESS, FAILURE])
        self.assertEqual(resolved.deadline_failure.tolist(), [False, False, False, True, False, False])
        self.assertEqual(resolved.terminated.tolist(), [False, True, True, True, True, True])
        torch.testing.assert_close(total_reward(torch.zeros(6), resolved, task),
                                   torch.tensor([-0.001, 0.999, -1.001, -1.001, 0.999, -1.001]))

    def test_outcome_resolution_rejects_unknown_codes_and_expired_source_steps(self):
        for outcome, clock in ((3, 1), (-1, 1), (0, 0), (0, 401)):
            with self.subTest(outcome=outcome, clock=clock), self.assertRaises(ValueError):
                resolve_outcomes(torch.tensor([outcome]), torch.tensor([clock]), TaskSpec())
        with self.assertRaises(ValueError):
            resolve_outcomes(torch.tensor([0.0]), torch.tensor([1]), TaskSpec())
        with self.assertRaises(ValueError):
            resolve_outcomes(torch.tensor([0]), torch.tensor([1, 2]), TaskSpec())

    def test_reward_labels_separate_non_deadline_outcome_from_clock(self):
        for family in ("relay", "cover"):
            task = TaskSpec.for_family(family)
            arrays = transitions(["continue", "success", "death", "failure", "deadline", "collection_cutoff"],
                                 family=family, decisions=[1, 2, 3, 4, task.deadline - 1, 5],
                                 bonuses=[0, task.max_bonus, 0, 0.05, 0, 0.05])
            # Missing both final images does not erase reward/outcome supervision.
            arrays["next_obs_valid"] = torch.zeros(6, 2, dtype=torch.bool)
            target = encode_transition_targets(arrays, task)
            self.assertEqual(target.outcome.tolist(), [CONTINUE, SUCCESS, FAILURE, FAILURE, CONTINUE, CONTINUE])
            self.assertTrue(target.bonus_valid.all())
            self.assertTrue(target.outcome_valid.all())
            torch.testing.assert_close(target.bonus, torch.tensor([0, 0.1, 0, 0.05, 0, 0.05]))

    def test_bad_reward_and_boundary_labels_raise_instead_of_clipping(self):
        task = TaskSpec()
        for reward in (-0.2, 0.2, float("nan")):
            arrays = transitions(["continue"])
            arrays["team_reward"][0] = reward
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                encode_transition_targets(arrays, task)
        for reason, clock in (("deadline", 4), ("continue", 399), ("collection_cutoff", 399)):
            with self.subTest(reason=reason, clock=clock), self.assertRaises(ValueError):
                encode_transition_targets(transitions([reason], decisions=[clock]), task)
        arrays = transitions(["success"])
        arrays["terminated"][0] = False
        with self.assertRaises(ValueError):
            encode_transition_targets(arrays, task)

    def test_roundoff_only_clipping_and_success_at_deadline(self):
        arrays = transitions(["success"], decisions=[399], bonuses=[0.1000000001])
        target = encode_transition_targets(arrays, TaskSpec())
        self.assertAlmostEqual(target.bonus.item(), 0.1, places=7)
        self.assertEqual(target.outcome.item(), SUCCESS)

    def test_transition_heads_bounded_float32_outputs_and_both_gradients(self):
        torch.manual_seed(87)
        head = TransitionHeads()
        features = torch.randn(3, 768, requires_grad=True)
        result = head(features)
        self.assertEqual(result.bonus.shape, (3,))
        self.assertEqual(result.outcome_logits.shape, (3, 3))
        self.assertEqual(result.bonus.dtype, torch.float32)
        self.assertTrue(torch.all((result.bonus >= 0) & (result.bonus <= 0.1)))
        (result.bonus.sum() + result.outcome_logits.square().sum()).backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertGreater(head.bonus_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.outcome_head.weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            head(torch.zeros(3, 256))
        with self.assertRaises(ValueError):
            head(torch.full((3, 768), float("nan")))

    def test_auxiliary_schema_covers_registry_fields_and_family_masks(self):
        for family in ("relay", "cover"):
            fields = auxiliary_fields(family)
            names = {f.name for f in fields}
            for player in (0, 1):
                self.assertTrue({f"player_{player}/{name}" for name in PLAYER_FIELDS}.issubset(names))
            self.assertIn("task/stage", names)
            self.assertIn("event/player_0_damage_taken", names)
            self.assertIn("event/player_1_deathcount", names)
            self.assertEqual(next(f.classes for f in fields if f.name == "task/stage"), 3 if family == "relay" else 4)
            self.assertEqual("task/latch_active" in names, family == "relay")
            self.assertEqual("event/latch_activations" in names, family == "relay")
            self.assertEqual("task/console_1_complete" in names, family == "cover")
            self.assertEqual("enemy_1/health" in names, family == "cover")
            self.assertEqual("event/enemy_1_deaths" in names, family == "cover")

    def test_auxiliary_heads_route_source_state_and_transition_features(self):
        torch.manual_seed(923)
        head = AuxiliaryHeads(task_family="cover")
        state = torch.randn(2, 256, requires_grad=True)
        transition = torch.randn(2, 768, requires_grad=True)
        result = head(state, transition)
        changed = head(state, transition + 1)
        torch.testing.assert_close(result["player_0/health"], changed["player_0/health"], rtol=0, atol=0)
        self.assertFalse(torch.equal(result["event/player_0_hitcount"], changed["event/player_0_hitcount"]))
        self.assertEqual(result["task/stage"].shape, (2, 4))
        self.assertEqual(result["enemy_1/alive"].shape, (2,))
        sum(value.square().sum() for value in result.values()).backward()
        self.assertGreater(state.grad.abs().sum().item(), 0)
        self.assertGreater(transition.grad.abs().sum().item(), 0)

    def test_auxiliary_target_alignment_masks_binary_event_presence_and_stage(self):
        state, transition = auxiliary_arrays()
        health = PLAYER_FIELDS.index("health")
        state["normalized_player_state"][:, 0, health] = torch.tensor([0.4, 0.9])
        state["privileged_validity_mask"][1, 0, health] = False
        state["normalized_player_state"][1, 0, health] = float("nan")
        state["task_state"][:, TASK_FIELDS.index("stage")] = torch.tensor([0, 3])
        event_index = declared_event_fields().index("player_0_hitcount")
        transition["event_deltas"][:, event_index] = torch.tensor([0, 7])
        labels = encode_auxiliary_targets(state, transition, TaskSpec.for_family("cover"))
        self.assertEqual(labels["player_0/health"].value[0].item(), state["normalized_player_state"][0, 0, health].item())
        self.assertEqual(labels["player_0/health"].valid.tolist(), [True, False])
        self.assertEqual(labels["task/stage"].value.tolist(), [0, 3])
        self.assertEqual(labels["event/player_0_hitcount"].value.tolist(), [0, 1])
        self.assertEqual(labels["event/player_0_hitcount"].valid.tolist(), [True, True])

    def test_auxiliary_bad_valid_targets_raise_and_event_schema_is_explicit(self):
        for bad_stage in (3, -1, 0.5):
            state, transition = auxiliary_arrays()
            state["task_state"][0, TASK_FIELDS.index("stage")] = bad_stage
            with self.subTest(stage=bad_stage), self.assertRaises(ValueError):
                encode_auxiliary_targets(state, transition, TaskSpec())
        state, transition = auxiliary_arrays()
        state["normalized_player_state"][0, 0, PLAYER_FIELDS.index("alive")] = 0.3
        with self.assertRaises(ValueError):
            encode_auxiliary_targets(state, transition, TaskSpec())
        with self.assertRaises(ValueError):
            auxiliary_fields("relay", ("missing",))


if __name__ == "__main__":
    unittest.main()
