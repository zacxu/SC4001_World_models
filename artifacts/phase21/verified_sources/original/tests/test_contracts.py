"""Phase 0 semantic checks; no simulator, tensor library, or GPU is needed."""

import unittest

from world_models.contracts import (
    ACTION_BUTTONS,
    ACTION_COUNT,
    LOCAL_HIDDEN_SIZE,
    LOCAL_RGB_BYTES,
    PREVIOUS_ACTION_START,
    Action,
    BoundaryReason,
    EpisodeBoundary,
    EpisodeClock,
    JointTransitionMetadata,
    LocalActorInput,
    LocalActorOutput,
    RGBFrame,
    RewardRule,
    resolve_boundary,
    validate_action,
    validate_hidden_size,
)


class LocalInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.frame = RGBFrame(bytes(LOCAL_RGB_BYTES))
        self.hidden = (0.0,) * LOCAL_HIDDEN_SIZE

    def test_action_mapping_and_reset_symbol(self):
        self.assertEqual(len(ACTION_BUTTONS), ACTION_COUNT)
        self.assertEqual(ACTION_BUTTONS[Action.USE], ("USE",))
        self.assertEqual(ACTION_BUTTONS[Action.STRAFE_RIGHT_ATTACK], ("MOVE_RIGHT", "ATTACK"))
        self.assertEqual(validate_action(PREVIOUS_ACTION_START, previous=True), 12)
        for invalid in (PREVIOUS_ACTION_START, -1, True, 1.0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_action(invalid)

    def test_renderer_buffer_is_detached(self):
        buffer = bytearray(LOCAL_RGB_BYTES)
        frame = RGBFrame.copy_from_buffer(buffer)
        buffer[0] = 255
        self.assertEqual(frame.data[0], 0)
        self.assertEqual(frame.shape, (3, 96, 128))
        self.assertEqual(frame.dtype, "uint8")
        for invalid in (bytes(3), bytearray(LOCAL_RGB_BYTES)):
            with self.assertRaises(ValueError):
                RGBFrame(invalid)

    def test_only_local_actor_fields_are_accepted(self):
        payload = dict(local_rgb=self.frame, previous_action=PREVIOUS_ACTION_START, local_hidden=self.hidden)
        accepted = LocalActorInput.from_mapping(payload)
        self.assertEqual(accepted.local_rgb, self.frame)
        for hidden_field in ("global_state", "team_reward", "teammate_rgb", "agent_id", "episode_clock"):
            with self.subTest(field=hidden_field), self.assertRaises(ValueError):
                LocalActorInput.from_mapping({**payload, hidden_field: object()})
        del payload["local_hidden"]
        with self.assertRaises(ValueError):
            LocalActorInput.from_mapping(payload)

    def test_local_memory_is_immutable_and_finite(self):
        mutable_memory = [0.0] * LOCAL_HIDDEN_SIZE
        inputs = LocalActorInput(self.frame, 0, mutable_memory)
        mutable_memory[0] = 99
        self.assertEqual(inputs.local_hidden[0], 0)
        for invalid in ((), (float("nan"),) * LOCAL_HIDDEN_SIZE, ("0",) * LOCAL_HIDDEN_SIZE):
            with self.subTest(invalid=invalid[:1]), self.assertRaises(ValueError):
                LocalActorInput(self.frame, 0, invalid)

    def test_memory_dimension_follows_explicit_policy_configuration(self):
        small_actor_input = LocalActorInput(self.frame, 0, (0.0,) * 64)
        self.assertEqual(len(validate_hidden_size(small_actor_input.local_hidden, 64)), 64)
        with self.assertRaises(ValueError):
            validate_hidden_size(small_actor_input.local_hidden, 256)

    def test_output_requires_a_real_distribution_and_local_memory(self):
        output = LocalActorOutput((1 / ACTION_COUNT,) * ACTION_COUNT, self.hidden)
        self.assertAlmostEqual(sum(output.action_probabilities), 1)
        for invalid in ((0.0,) * ACTION_COUNT, (1.0,) * ACTION_COUNT, (-1.0,) + (2 / 11,) * 11):
            with self.assertRaises(ValueError):
                LocalActorOutput(invalid, self.hidden)


class ClockAndBoundaryTests(unittest.TestCase):
    def test_clock_cannot_advance_past_the_task_deadline(self):
        clock = EpisodeClock(399, 400)
        self.assertAlmostEqual(clock.normalized_remaining, 1 / 400)
        deadline = clock.advance()
        self.assertTrue(deadline.expired)
        self.assertEqual(deadline.normalized_remaining, 0)
        with self.assertRaises(ValueError):
            deadline.advance()
        for index, limit in ((401, 400), (0, 0), (-1, 10), (True, 10)):
            with self.assertRaises(ValueError):
                EpisodeClock(index, limit)

    def test_death_success_deadline_and_collection_precedence(self):
        clock = EpisodeClock(400, 400)
        self.assertEqual(resolve_boundary(clock).reason, BoundaryReason.DEADLINE)
        self.assertEqual(resolve_boundary(clock, success=True).reason, BoundaryReason.SUCCESS)
        self.assertEqual(resolve_boundary(clock, success=True, death=True).reason, BoundaryReason.DEATH)
        self.assertEqual(resolve_boundary(clock, collection_cutoff=True).reason, BoundaryReason.DEADLINE)
        self.assertEqual(resolve_boundary(EpisodeClock(20, 400), collection_cutoff=True).reason,
                         BoundaryReason.COLLECTION_CUTOFF)

    def test_bootstrap_and_advantage_recursion_are_different_at_cutoffs(self):
        for reason in (BoundaryReason.SUCCESS, BoundaryReason.DEATH, BoundaryReason.DEADLINE):
            boundary = EpisodeBoundary(reason, (False, False))
            self.assertTrue(boundary.terminated)
            self.assertEqual(boundary.bootstrap_mask, 0)
            self.assertEqual(boundary.advantage_recursion_mask, 0)
        for reason in (BoundaryReason.COLLECTION_CUTOFF, BoundaryReason.IMAGINATION_HORIZON):
            boundary = EpisodeBoundary(reason)
            self.assertFalse(boundary.terminated)
            self.assertEqual(boundary.bootstrap_mask, 1)
            self.assertEqual(boundary.advantage_recursion_mask, 0)
        self.assertTrue(EpisodeBoundary(BoundaryReason.COLLECTION_CUTOFF).truncated)
        self.assertFalse(EpisodeBoundary(BoundaryReason.IMAGINATION_HORIZON).truncated)
        self.assertEqual(EpisodeBoundary().advantage_recursion_mask, 1)

    def test_missing_observations_are_not_fabricated_nonterminal_states(self):
        with self.assertRaises(ValueError):
            EpisodeBoundary(BoundaryReason.COLLECTION_CUTOFF, (True, False))
        terminal = EpisodeBoundary(BoundaryReason.DEATH, (True, False))
        self.assertEqual(terminal.next_observation_valid, (True, False))
        with self.assertRaises(ValueError):
            EpisodeBoundary("infrastructure_error")


class RewardAndTransitionTests(unittest.TestCase):
    def test_reward_decomposes_for_success_death_and_deadline(self):
        rule = RewardRule()
        for reason, terminal_component in ((BoundaryReason.SUCCESS, 1), (BoundaryReason.DEATH, -1),
                                           (BoundaryReason.DEADLINE, -1), (BoundaryReason.CONTINUE, 0)):
            boundary = EpisodeBoundary(reason)
            reward = rule.reward(boundary, 0.1)
            self.assertAlmostEqual(reward, terminal_component + 0.1 - 0.001)
            self.assertAlmostEqual(rule.milestone_target(reward, boundary), 0.1)
        with self.assertRaises(ValueError):
            rule.milestone_target(1, EpisodeBoundary(BoundaryReason.DEADLINE))
        with self.assertRaises(ValueError):
            rule.reward(EpisodeBoundary(), 0.2)

    def test_joint_transition_counts_once_and_preserves_timestep_alignment(self):
        transition = JointTransitionMetadata(5, 20, 24, (Action.FORWARD, Action.USE), EpisodeBoundary())
        self.assertEqual(transition.next_decision_index, 6)
        self.assertEqual(transition.actual_simulator_tics, 4)
        self.assertEqual(transition.joint_transition_count, 1)

    def test_shortened_repeat_requires_true_terminal(self):
        terminal = JointTransitionMetadata(5, 20, 22, (0, 0), EpisodeBoundary(BoundaryReason.DEATH))
        self.assertEqual(terminal.actual_simulator_tics, 2)
        for boundary in (EpisodeBoundary(), EpisodeBoundary(BoundaryReason.COLLECTION_CUTOFF)):
            with self.assertRaises(ValueError):
                JointTransitionMetadata(5, 20, 22, (0, 0), boundary)

    def test_bad_or_synthetic_steps_are_rejected_from_real_metadata(self):
        for end_tic, actions, boundary in (
            (20, (0, 0), EpisodeBoundary()),
            (25, (0, 0), EpisodeBoundary()),
            (24, (0,), EpisodeBoundary()),
            (24, (0, PREVIOUS_ACTION_START), EpisodeBoundary()),
            (24, (0, 0), EpisodeBoundary(BoundaryReason.IMAGINATION_HORIZON)),
        ):
            with self.subTest(end_tic=end_tic, actions=actions), self.assertRaises(ValueError):
                JointTransitionMetadata(0, 20, end_tic, actions, boundary)


if __name__ == "__main__":
    unittest.main()
