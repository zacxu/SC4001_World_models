"""P4.1 CPU synthetic fixtures: genuine teacher collection and recurrent PPO."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from tests import test_mappo_updater as base
from tests.test_imagined_rollout_generator import assert_tree_equal
from world_models.imagination.anchors import AnchorBuilder
from world_models.imagination.versions import module_version
from world_models.mappo.config import MAPPOConfig
from world_models.mappo.recurrent import PrefixBank
from world_models.mappo.updater import RecurrentMAPPO
from world_models.policies import LocalActor
from world_models.policies.teacher import GlobalTeacher
from world_models.rollouts.generator import RolloutGenerator
from world_models.rollouts.streams import RolloutStreams
from world_models.teacher.learning import (
    ROLE, TeacherAnchorBuilder, TeacherImaginedRollout, TeacherMAPPO,
    TeacherPrefixBank, TeacherRolloutGenerator,
)


class TeacherLearningTests(base.RecurrentMAPPOTests):
    """Run the common PPO contract suite against independent teacher adapters."""

    def make_networks(self, *, deadline=400):
        super().make_networks(deadline=deadline)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(4104)
            self.actor = GlobalTeacher(self.actor.config)

    def collect(self, *, clocks=(0, 1, 2, 3), horizon=2, seed=31, precision="float32"):
        refs = tuple(self.sampler.reference(0, self.episode, clock) for clock in clocks)
        anchors = TeacherAnchorBuilder(self.model, self.actor, seed=17, precision=precision).build(self.sampler, refs)
        return TeacherRolloutGenerator(self.model, self.actor, self.critic, horizon=horizon,
                                       precision=precision).collect(anchors, streams=RolloutStreams.from_seed(seed))

    def learner(self, *, epochs=2, minibatches=2, seed=51, precision="float32"):
        return TeacherMAPPO(self.model, self.actor, self.critic,
                            config=MAPPOConfig(epochs=epochs, minibatches=minibatches),
                            seed=seed, precision=precision)

    def test_explicit_teacher_roles_prevent_local_anchor_rollout_and_ppo_reuse(self):
        rollout = self.collect()
        self.assertIs(type(rollout), TeacherImaginedRollout)
        self.assertEqual(rollout.metadata["policy_role"], ROLE)
        self.assertEqual(rollout.metadata["local_player_decisions"], 0)
        self.assertEqual(rollout.metadata["teacher_player_decisions"], 2 * int(rollout.valid.sum()))
        self.assertEqual(rollout.metadata["schema_version"], "p41-teacher-rollout-v1")
        self.assertEqual(rollout.metadata["anchor_metadata"]["schema_version"], "p41-teacher-anchor-v1")
        with torch.random.fork_rng(devices=[]):
            local = LocalActor(self.actor.config)
        with self.assertRaises(ValueError):
            AnchorBuilder(self.model, self.actor, seed=1)
        with self.assertRaises(ValueError):
            TeacherAnchorBuilder(self.model, local, seed=1)
        local_updater = RecurrentMAPPO(self.model, local, self.critic, config=MAPPOConfig(minibatches=2))
        with self.assertRaisesRegex(ValueError, "ImaginedRollout"):
            local_updater.update(rollout, self.sampler)
        with self.assertRaisesRegex(ValueError, "local-policy rollout"):
            PrefixBank.from_rollout(rollout, self.sampler).replay(local, rollout, [0])
        with self.assertRaises(ValueError):
            replace(rollout, metadata={**rollout.metadata, "policy_role": "local"})
        refs = tuple(self.sampler.reference(0, self.episode, clock) for clock in (0, 1))
        local_anchors = AnchorBuilder(self.model, local, seed=2).build(self.sampler, refs)
        teacher_generator = TeacherRolloutGenerator(self.model, self.actor, self.critic, horizon=1)
        with self.assertRaisesRegex(ValueError, "TeacherAnchorBatch"):
            teacher_generator.collect(local_anchors, streams=RolloutStreams.from_seed(1))
        local_rollout = RolloutGenerator(self.model, local, self.critic, horizon=1).collect(
            local_anchors, streams=RolloutStreams.from_seed(2))
        with self.assertRaisesRegex(ValueError, "TeacherImaginedRollout"):
            self.learner().update(local_rollout, self.sampler)
        with self.assertRaises(ValueError):
            self.learner().load_state_dict(local_updater.state_dict())

    def test_teacher_head_changes_invalidate_anchor_cache_and_alter_current_actions_only(self):
        ref = self.sampler.reference(0, self.episode, 3)
        builder = TeacherAnchorBuilder(self.model, self.actor, seed=17)
        old = builder.build(self.sampler, [ref])
        cached = builder.build(self.sampler, [ref])
        self.assertEqual(cached.stats["cache_hits"], 1)
        with torch.no_grad():
            self.actor.action_head[-1].bias[0].add_(1)
        new = builder.build(self.sampler, [ref])
        self.assertNotEqual(old.actor_version, new.actor_version)
        self.assertEqual(new.stats["cache_hits"], 0)
        self.assertEqual(new.stats["cache_invalidations"], 1)
        torch.testing.assert_close(old.actor_hidden, new.actor_hidden, rtol=0, atol=0)
        torch.testing.assert_close(old.world.memory, new.world.memory, rtol=0, atol=0)
        self.assertFalse(torch.equal(old.actor_logits, new.actor_logits))

    def test_full_prefix_memory_consumes_anchor_once_and_matches_independent_player_histories(self):
        ref = self.sampler.reference(0, self.episode, 3)
        prefix = self.sampler.read_prefix(ref)
        calls = []
        memory_step = self.actor.memory_step

        def record(rgb, previous, hidden):
            calls.append((rgb.shape[0], torch.is_grad_enabled()))
            return memory_step(rgb, previous, hidden)

        with patch.object(self.actor, "memory_step", side_effect=record):
            anchors = TeacherAnchorBuilder(self.model, self.actor, seed=17).build(self.sampler, [ref])
        self.assertEqual(calls, [(2, False)] * 4)
        with torch.no_grad():
            for player in (0, 1):
                hidden = self.actor.initial_hidden(1)
                for step in range(4):
                    rgb = torch.from_numpy(prefix.rgb[step, player:player + 1].copy())
                    previous = torch.tensor([12 if not step else int(prefix.actions[step - 1, player])])
                    hidden = self.actor.memory_step(rgb, previous, hidden)
                torch.testing.assert_close(hidden[0], anchors.actor_hidden[0, player], rtol=1e-5, atol=2e-6)
        calls.clear()
        with patch.object(self.actor, "memory_step", side_effect=record):
            rollout = TeacherRolloutGenerator(self.model, self.actor, self.critic, horizon=2).collect(
                anchors, streams=RolloutStreams.from_seed(31))
        self.assertEqual(calls, [(2, False)] * 2)  # destinations only; anchor already consumed
        calls.clear()
        bank = TeacherPrefixBank.from_rollout(rollout, self.sampler)
        with patch.object(self.actor, "memory_step", side_effect=record):
            logits = bank.replay(self.actor, rollout, [0])
        self.assertEqual(calls, [(2, False)] * 3 + [(2, True)] * 2)
        torch.testing.assert_close(logits, rollout.actor_logits[:-1], rtol=1e-5, atol=2e-6)
        logits.square().sum().backward()
        self.assertIsNotNone(self.actor.gru.weight_hh.grad)
        self.assertIsNotNone(self.actor.cross_attention.in_proj_weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_teacher_rollout_never_reads_real_suffix_or_runs_posterior_and_rng_restores_on_failure(self):
        ref = self.sampler.reference(0, self.episode, 1)
        anchors = TeacherAnchorBuilder(self.model, self.actor, seed=17).build(self.sampler, [ref])
        generator = TeacherRolloutGenerator(self.model, self.actor, self.critic, horizon=2)
        streams = RolloutStreams.from_seed(31)
        original = deepcopy(streams.state_dict())
        with patch.object(self.sampler, "read_prefix", side_effect=AssertionError("future real input")), \
                patch.object(self.model.core, "observe", side_effect=AssertionError("posterior during imagination")):
            rollout = generator.collect(anchors, streams=streams)
        self.assertEqual(rollout.steps, 2)
        streams.load_state_dict(original)
        with patch.object(self.actor, "forward", side_effect=RuntimeError("injected teacher failure")):
            with self.assertRaisesRegex(RuntimeError, "injected teacher"):
                generator.collect(anchors, streams=streams)
        assert_tree_equal(self, original, streams.state_dict())

    def test_frozen_global_tokens_are_used_in_likelihood_replay_with_no_world_execution(self):
        rollout = self.collect(horizon=1)
        bank = TeacherPrefixBank.from_rollout(rollout, self.sampler)
        with torch.no_grad():
            logits = bank.replay(self.actor, rollout, [0, 1, 2, 3])
        torch.testing.assert_close(logits, rollout.actor_logits[:-1], rtol=1e-5, atol=2e-6)
        changed_tokens = rollout.shared_tokens.clone()
        changed_tokens[0].add_(4)
        changed = replace(rollout, shared_tokens=changed_tokens)
        with torch.no_grad():
            new_logits = bank.replay(self.actor, changed, [0, 1, 2, 3])
        self.assertFalse(torch.allclose(logits, new_logits))
        before = tuple(module_version(network) for network in (self.model, self.actor, self.critic))
        with self.assertRaisesRegex(ValueError, "initial (behavior|critic) replay"):
            self.learner().update(changed, self.sampler)
        self.assertEqual(before, tuple(module_version(network) for network in (self.model, self.actor, self.critic)))

    def test_teacher_payload_reconstruction_preserves_role_and_complete_tensor_trace(self):
        rollout = self.collect()
        restored = TeacherImaginedRollout.from_tensor_payload(
            deepcopy(rollout.tensor_payload()), metadata=deepcopy(rollout.metadata),
            streams_before=deepcopy(rollout.streams_before), streams_after=deepcopy(rollout.streams_after))
        assert_tree_equal(self, rollout.tensor_payload(), restored.tensor_payload())
        self.assertEqual(restored.metadata["policy_role"], ROLE)
        self.assertIs(type(restored.to("cpu")), TeacherImaginedRollout)

    def test_teacher_reward_diagnostics_count_only_valid_imagined_joint_steps(self):
        self.make_networks(deadline=3)
        rollout = self.collect(clocks=(0, 1, 2, 2))
        self.assertFalse(rollout.valid.all())
        report = self.learner(epochs=1).update(rollout, self.sampler)
        self.assertEqual(report["schema_version"], "p41-teacher-update-v1")
        self.assertEqual(report["policy_role"], ROLE)
        self.assertAlmostEqual(report["metrics"]["imagined_reward_mean"],
                               float(rollout.rewards[rollout.valid].sum() / rollout.valid.sum()))
        self.assertAlmostEqual(report["metrics"]["mean_imagined_return"],
                               float((rollout.rewards * rollout.valid).sum() / rollout.batch_size))
        self.assertEqual(report["new_environment_interactions"], 0)

    def test_prefix_calls_global_head_only_once_after_current_anchor_memories(self):
        refs = [self.sampler.reference(0, self.episode, index) for index in (0, 2, 3)]
        with patch.object(self.actor, "action_from_memory", wraps=self.actor.action_from_memory) as head:
            anchors = TeacherAnchorBuilder(self.model, self.actor, seed=17).build(self.sampler, refs)
        self.assertEqual(head.call_count, 1)
        self.assertEqual(head.call_args.args[0].shape[0], len(refs) * 2)
        tokens, players = head.call_args.args[1:]
        torch.testing.assert_close(tokens[0::2], anchors.shared_tokens, rtol=0, atol=0)
        torch.testing.assert_close(tokens[1::2], anchors.shared_tokens, rtol=0, atol=0)
        self.assertEqual(players.tolist(), [0, 1] * len(refs))


if __name__ == "__main__":
    unittest.main()
