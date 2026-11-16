"""Boundary math and recurrent alignment for detached synthetic rollouts."""
from dataclasses import asdict, replace
import math
import unittest

import torch

from world_models.imagination.replay import AnchorRef
from world_models.models.core import WorldState
from world_models.rollouts.schema import EndReason, ImaginedRollout, SCHEMA_VERSION


def make_rollout() -> ImaginedRollout:
    """Two starts: success after two actions, and a three-action alive cutoff."""
    t, b = 3, 2
    valid = torch.tensor([[True, True], [True, True], [False, True]])
    state_valid = torch.cat((torch.ones_like(valid[:1]), valid))
    clocks = torch.tensor([[1, 1], [2, 2], [3, 3], [0, 4]])
    actions = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[0, 0], [9, 10]]])
    previous = torch.zeros(t + 1, b, 2, dtype=torch.long)
    previous[1:][valid] = actions[valid]
    terminated = torch.tensor([[False, False], [True, False], [False, False]])
    cutoff = torch.tensor([[False, False], [False, False], [False, True]])
    outcomes = torch.zeros(t, b, dtype=torch.long)
    outcomes[1, 0] = 1
    reasons = outcomes.clone()
    reasons[~valid] = int(EndReason.PADDING)
    reasons[cutoff] = int(EndReason.HORIZON)
    rewards = torch.zeros(t, b)
    rewards[valid] = -.001
    rewards[1, 0] += 1
    probabilities = torch.zeros(t, b, 3)
    probabilities[valid] = torch.tensor([.6, .3, .1])
    rgb = torch.zeros(t + 1, b, 2, 3, 96, 128, dtype=torch.uint8)
    rgb[state_valid] = 100
    logits = torch.zeros(t + 1, b, 2, 12)
    log_probs = torch.zeros(t, b, 2)
    log_probs[valid] = -math.log(12)
    identity = "a" * 64
    metadata = {
        "schema_version": SCHEMA_VERSION, "horizon": 3, "deadline": 5,
        "task_family": "relay", "precision": "float32", "real_interactions": 0,
        "data_source": "world_model_imagination", "world_model_version": "b" * 64,
        "actor_version": "c" * 64, "critic_version": "d" * 64,
        "cohort_identity": identity,
        "anchor_references": [asdict(AnchorRef(identity, 0, "e" * 64, f"fit-{index}", 1))
                              for index in range(b)],
        "task_rule": {"step_cost": -.001, "success_reward": 1., "failure_reward": -1., "max_bonus": .1},
    }
    return ImaginedRollout(
        world=WorldState(torch.zeros(t + 1, b, 16, 256), torch.zeros(t + 1, b, 32, 32),
                         torch.zeros(t + 1, b, 32, 32), clocks),
        local_rgb=rgb, previous_actions=previous, actor_hidden=torch.zeros(t + 1, b, 2, 256),
        actor_logits=logits, shared_tokens=torch.zeros(t + 1, b, 48, 256),
        values=torch.tensor([[2., 3.], [4., 5.], [6., 7.], [0., 11.]]),
        actions=actions, log_probs=log_probs, rewards=rewards, bonus=torch.zeros(t, b),
        valid=valid, terminated=terminated, cutoff=cutoff,
        sampled_outcomes=outcomes.clone(), outcomes=outcomes, outcome_probabilities=probabilities,
        deadline_failure=torch.zeros(t, b, dtype=torch.bool), end_reason=reasons,
        metadata=metadata, streams_before={"state": torch.tensor([1, 2], dtype=torch.uint8)},
        streams_after={"state": torch.tensor([3, 4], dtype=torch.uint8)})


class ImaginedRolloutSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_terminal_zero_bootstrap_but_horizon_keeps_raw_destination_value(self):
        rollout = make_rollout()
        self.assertEqual((rollout.steps, rollout.batch_size), (3, 2))
        self.assertEqual(rollout.lengths.tolist(), [2, 3])
        self.assertEqual(rollout.state_valid.tolist(), [[True, True], [True, True], [True, True], [False, True]])
        self.assertEqual(rollout.next_values[1, 0].item(), 6)
        self.assertEqual(rollout.bootstrap_values.tolist(), [[4., 5.], [0., 7.], [0., 11.]])
        self.assertEqual(rollout.recursion_mask.tolist(), [[1., 1.], [0., 1.], [0., 0.]])
        # The last cutoff delta uses value 11; a true terminal ignores value 6.
        delta = rollout.rewards + .99 * rollout.bootstrap_values - rollout.old_values
        self.assertAlmostEqual(delta[2, 1].item(), -.001 + .99 * 11 - 7, places=5)
        self.assertAlmostEqual(delta[1, 0].item(), .999 - 4, places=5)

    def test_terminal_at_horizon_takes_precedence_over_cutoff(self):
        rollout = make_rollout()
        outcomes = rollout.outcomes.clone()
        outcomes[-1, 1] = 2
        rewards = rollout.rewards.clone()
        rewards[-1, 1] -= 1
        terminated = rollout.terminated.clone()
        terminated[-1, 1] = True
        reasons = rollout.end_reason.clone()
        reasons[-1, 1] = int(EndReason.FAILURE)
        terminal = replace(rollout, sampled_outcomes=outcomes.clone(), outcomes=outcomes, rewards=rewards,
                           terminated=terminated, cutoff=torch.zeros_like(rollout.cutoff), end_reason=reasons)
        self.assertEqual(terminal.bootstrap_values[-1, 1].item(), 0)
        self.assertEqual(terminal.recursion_mask[-1, 1].item(), 0)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            replace(terminal, cutoff=rollout.cutoff)

    def test_exact_deadline_requires_failure_without_overriding_sampled_success(self):
        rollout = make_rollout()
        metadata = {**rollout.metadata, "deadline": 4}
        outcomes = rollout.outcomes.clone()
        outcomes[-1, 1] = 2
        terminated = rollout.terminated.clone()
        terminated[-1, 1] = True
        deadline = rollout.deadline_failure.clone()
        deadline[-1, 1] = True
        reasons = rollout.end_reason.clone()
        reasons[-1, 1] = int(EndReason.DEADLINE)
        rewards = rollout.rewards.clone()
        rewards[-1, 1] -= 1
        deadline_rollout = replace(rollout, metadata=metadata, outcomes=outcomes, terminated=terminated,
                                   cutoff=torch.zeros_like(rollout.cutoff), deadline_failure=deadline,
                                   end_reason=reasons, rewards=rewards)
        self.assertEqual(deadline_rollout.bootstrap_values[-1, 1].item(), 0)
        sampled = rollout.sampled_outcomes.clone()
        sampled[-1, 1] = 1
        outcomes[-1, 1] = 1
        rewards[-1, 1] += 2
        reasons[-1, 1] = int(EndReason.SUCCESS)
        success = replace(deadline_rollout, sampled_outcomes=sampled, outcomes=outcomes,
                          deadline_failure=torch.zeros_like(deadline), end_reason=reasons, rewards=rewards)
        self.assertEqual(success.end_reason[-1, 1].item(), EndReason.SUCCESS)

    def test_rejects_recurrent_action_logprob_and_clock_misalignment(self):
        rollout = make_rollout()
        previous = rollout.previous_actions.clone()
        previous[1, 0, 0] = 3
        with self.assertRaisesRegex(ValueError, "actually executed"):
            replace(rollout, previous_actions=previous)
        log_probs = rollout.log_probs.clone()
        log_probs[0, 0, 0] += .1
        with self.assertRaisesRegex(ValueError, "log_probs"):
            replace(rollout, log_probs=log_probs)
        clocks = rollout.world.decision_index.clone()
        clocks[1, 0] += 1
        with self.assertRaisesRegex(ValueError, "advance"):
            replace(rollout, world=replace(rollout.world, decision_index=clocks))

    def test_rejects_padding_reactivation_and_missing_boundaries(self):
        rollout = make_rollout()
        rewards = rollout.rewards.clone()
        rewards[-1, 0] = 1
        with self.assertRaisesRegex(ValueError, "padding"):
            replace(rollout, rewards=rewards)
        rgb = rollout.local_rgb.clone()
        rgb[-1, 0] = 255
        with self.assertRaisesRegex(ValueError, "padding"):
            replace(rollout, local_rgb=rgb)
        valid = rollout.valid.clone()
        valid[1, 1] = False
        with self.assertRaisesRegex(ValueError, "prefix"):
            replace(rollout, valid=valid)
        with self.assertRaisesRegex(ValueError, "end exactly once"):
            replace(rollout, cutoff=torch.zeros_like(rollout.cutoff))
        with self.assertRaisesRegex(ValueError, "configured imagination horizon"):
            replace(rollout, metadata={**rollout.metadata, "horizon": 4})

    def test_rejects_gradient_and_inference_tensors_but_accepts_bf16_features(self):
        rollout = make_rollout()
        with self.assertRaisesRegex(ValueError, "detached ordinary"):
            replace(rollout, values=rollout.values.clone().requires_grad_())
        with torch.inference_mode():
            inference = rollout.values.clone()
        with self.assertRaisesRegex(ValueError, "detached ordinary"):
            replace(rollout, values=inference)
        bf16 = replace(rollout, shared_tokens=rollout.shared_tokens.to(torch.bfloat16))
        self.assertEqual(bf16.shared_tokens.dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "dtype"):
            replace(rollout, actor_hidden=rollout.actor_hidden.to(torch.bfloat16))

    def test_rejects_inconsistent_probabilities_reward_causes_and_source_metadata(self):
        rollout = make_rollout()
        probability = rollout.outcome_probabilities.clone()
        probability[0, 0] *= .5
        with self.assertRaisesRegex(ValueError, "normalized"):
            replace(rollout, outcome_probabilities=probability)
        reward = rollout.rewards.clone()
        reward[0, 0] = 10
        with self.assertRaisesRegex(ValueError, "decomposition"):
            replace(rollout, rewards=reward)
        reason = rollout.end_reason.clone()
        reason[1, 0] = int(EndReason.FAILURE)
        with self.assertRaisesRegex(ValueError, "end_reason"):
            replace(rollout, end_reason=reason)
        for key, value in (("actor_version", "unversioned"), ("real_interactions", 1),
                           ("data_source", "real_environment")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                replace(rollout, metadata={**rollout.metadata, key: value})

    def test_payload_roundtrip_and_to_own_tensor_metadata_and_rng_storage(self):
        rollout = make_rollout()
        restored = ImaginedRollout.from_tensor_payload(rollout.tensor_payload(), metadata=rollout.metadata,
                                                      streams_before=rollout.streams_before,
                                                      streams_after=rollout.streams_after)
        torch.testing.assert_close(restored.log_probs, rollout.log_probs)
        copied = rollout.to("cpu")
        self.assertNotEqual(copied.values.data_ptr(), rollout.values.data_ptr())
        self.assertNotEqual(copied.world.memory.data_ptr(), rollout.world.memory.data_ptr())
        self.assertNotEqual(copied.streams_before["state"].data_ptr(), rollout.streams_before["state"].data_ptr())
        copied.metadata["task_rule"]["step_cost"] = -10
        self.assertEqual(rollout.metadata["task_rule"]["step_cost"], -.001)
        with self.assertRaisesRegex(ValueError, "fields"):
            ImaginedRollout.from_tensor_payload({**rollout.tensor_payload(), "unknown": torch.zeros(1)},
                                                metadata=rollout.metadata, streams_before={}, streams_after={})


if __name__ == "__main__":
    unittest.main()
