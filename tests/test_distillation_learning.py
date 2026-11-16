"""CPU fixtures for detached teacher targets and current-student sequence KL."""
from copy import deepcopy
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests import test_teacher_learning as teacher_tests
from tests.test_imagined_rollout_generator import assert_tree_equal
from world_models.distillation import (
    DistillationBuffer, DistillationConfig, StudentDistiller, forward_kl, replay_student,
)
from world_models.distillation.protocol import plan
from world_models.imagination.versions import module_version
from world_models.imagination.replay import AnchorPrefix
from world_models.mappo.recurrent import PrefixBank
from world_models.policies.actor import LocalActor


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.fixture = teacher_tests.TeacherLearningTests(methodName="test_explicit_teacher_roles_prevent_local_anchor_rollout_and_ppo_reuse")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(4201)
            self.student = LocalActor(self.fixture.actor.config)
        self.initial = deepcopy(self.student.state_dict())
        self.initial_version = module_version(self.student)
        self.critic_version = module_version(self.fixture.critic)
        self.fixture.critic.requires_grad_(False)

    def buffer(self, **kwargs):
        rollout = self.fixture.collect(**kwargs)
        return DistillationBuffer.from_teacher_rollout(rollout, self.fixture.sampler, self.fixture.actor)

    def learner(self, **kwargs):
        config = DistillationConfig(updates=kwargs.pop("updates", 3), sequences_per_update=kwargs.pop("sequences", 2))
        return StudentDistiller(self.student, self.fixture.critic, paired_actor_version=self.initial_version,
                                paired_critic_version=self.critic_version, seed=kwargs.pop("seed", 42), config=config)

    def test_registered_recipe_caps_and_dedicated_optimizer_defaults(self):
        recipe = plan(0)
        self.assertEqual(recipe["imagined_joint_step_cap"], 1920)
        self.assertEqual(recipe["student_optimizer_steps"], 100)
        self.assertEqual(recipe["expected_buffer_passes"], 25)
        self.assertEqual(recipe["protocol"]["temperature"], 1.)
        default = DistillationConfig()
        self.assertEqual((default.updates, default.sequences_per_update), (100, 32))
        optimizer = self.learner().optimizer
        self.assertIs(type(optimizer), torch.optim.Adam)
        self.assertEqual((optimizer.defaults["lr"], optimizer.defaults["eps"], optimizer.defaults["betas"],
                          optimizer.defaults["weight_decay"]), (3e-4, 1e-5, (.9, .999), 0.))
        self.assertFalse(optimizer.state)
        with self.assertRaises(ValueError):
            plan(2)
        with self.assertRaises(ValueError):
            DistillationConfig(weight_decay=.01)

    def test_targets_temperature_and_execution_alignment_have_no_teacher_memory(self):
        rollout = self.fixture.collect()
        buffer = DistillationBuffer.from_teacher_rollout(rollout, self.fixture.sampler, self.fixture.actor)
        torch.testing.assert_close(buffer.targets, rollout.actor_logits[:-1].softmax(-1), rtol=0, atol=0)
        self.assertEqual(buffer.targets.dtype, torch.float32)
        self.assertFalse(buffer.targets.requires_grad)
        self.assertTrue(torch.equal(buffer.actions, rollout.actions))
        self.assertTrue(torch.equal(buffer.previous_actions[1:], rollout.actions[:-1]))
        self.assertFalse(hasattr(buffer, "actor_hidden"))
        self.assertFalse(hasattr(buffer, "shared_tokens"))
        self.assertFalse(buffer.metadata["world_model_fitting_eligible"])
        self.assertEqual(buffer.metadata["real_interactions"], 0)
        poisoned = rollout.actor_hidden.clone().fill_(999)
        altered = replace(rollout, actor_hidden=poisoned)
        other = DistillationBuffer.from_teacher_rollout(altered, self.fixture.sampler, self.fixture.actor)
        torch.testing.assert_close(other.targets, buffer.targets, rtol=0, atol=0)
        with torch.no_grad():
            self.fixture.actor.action_head[-1].bias[0].add_(.1)
        with self.assertRaisesRegex(ValueError, "teacher changed"):
            DistillationBuffer.from_teacher_rollout(rollout, self.fixture.sampler, self.fixture.actor)

    def test_teacher_replay_rejects_forged_logits_even_with_matching_version(self):
        rollout = self.fixture.collect()
        logits = rollout.actor_logits.clone()
        logits[..., 0] += .1
        log_probs = logits[:-1].log_softmax(-1).gather(-1, rollout.actions[..., None]).squeeze(-1)
        altered = replace(rollout, actor_logits=logits, log_probs=log_probs)
        with self.assertRaisesRegex(ValueError, "replay disagrees"):
            DistillationBuffer.from_teacher_rollout(altered, self.fixture.sampler, self.fixture.actor)

    def test_padding_and_terminal_actions_are_masked_without_discarding_terminals(self):
        self.fixture.make_networks(deadline=3)
        buffer = self.buffer(clocks=(0, 1, 2, 2))
        self.assertEqual(buffer.valid.sum(0).tolist(), [2, 2, 1, 1])
        self.assertEqual(int(buffer.terminated.sum()), 3)
        self.assertTrue(bool((buffer.targets[~buffer.valid] == 0).all()))
        logits = torch.zeros_like(buffer.targets, requires_grad=True)
        loss, entropy = forward_kl(logits, buffer.targets, buffer.valid)
        self.assertAlmostEqual(float(entropy.detach()), math.log(12), places=6)
        loss.backward()
        self.assertTrue(bool((logits.grad[~buffer.valid] == 0).all()))
        self.assertGreater(float(logits.grad[buffer.terminated].abs().sum()), 0)

    def test_plain_forward_kl_matches_manual_valid_player_mean(self):
        logits = torch.tensor([[[[2.] + [0.] * 11, [0., 1.] + [0.] * 10]],
                               [[[0.] * 12, [0.] * 12]]], requires_grad=True)
        target = torch.zeros_like(logits)
        target[0, 0, 0, :2] = torch.tensor([.75, .25])
        target[0, 0, 1, :2] = torch.tensor([.2, .8])
        valid = torch.tensor([[True], [False]])
        loss, _ = forward_kl(logits, target, valid)
        manual = sum(float(target[0, 0, player, action]) * (
            math.log(float(target[0, 0, player, action])) - logits[0, 0, player].log_softmax(-1)[action])
            for player in (0, 1) for action in (0, 1)) / 2
        torch.testing.assert_close(loss, manual)
        loss.backward()
        self.assertTrue(bool((logits.grad[1] == 0).all()))
        with self.assertRaisesRegex(ValueError, "detached"):
            forward_kl(logits, target.requires_grad_(), valid)

    def test_buffer_owns_histories_and_roundtrips_without_real_replay_access(self):
        buffer = self.buffer()
        state = buffer.state_dict()
        with patch.object(self.fixture.sampler, "read_prefix", side_effect=AssertionError("live replay access")):
            recovered = DistillationBuffer.from_state_dict(state)
            logits = replay_student(self.student, recovered, [1, 3])
        self.assertEqual(logits.shape, (2, 2, 2, 12))
        assert_tree_equal(self, buffer.state_dict(), recovered.state_dict())
        state["tensors"]["local_rgb"].zero_()
        self.assertFalse(torch.equal(state["tensors"]["local_rgb"], recovered.local_rgb))
        self.assertFalse(recovered.prefixes.prefixes[0].rgb.flags.writeable)

    def test_buffer_rejects_invalid_targets_actions_and_prefixes(self):
        buffer = self.buffer()
        invalid = buffer.targets.clone()
        invalid[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(buffer, targets=invalid)
        with self.assertRaisesRegex(ValueError, "normalized"):
            replace(buffer, targets=buffer.targets / 2)
        with self.assertRaisesRegex(ValueError, "detached"):
            replace(buffer, targets=buffer.targets.clone().requires_grad_())
        previous = buffer.previous_actions.clone()
        previous[1, 0, 0] = (int(previous[1, 0, 0]) + 1) % 12
        with self.assertRaisesRegex(ValueError, "actually executed"):
            replace(buffer, previous_actions=previous)
        metadata = deepcopy(buffer.metadata)
        metadata["temperature"] = 2.
        with self.assertRaisesRegex(ValueError, "temperature"):
            replace(buffer, metadata=metadata)

    def test_concat_preserves_whole_trajectories_and_prefix_provenance(self):
        self.fixture.make_networks(deadline=3)
        long = self.buffer(clocks=(0,), seed=1)
        short = self.buffer(clocks=(2,), seed=2)
        buffer = DistillationBuffer.concat([long, short])
        self.assertEqual(buffer.valid.sum(0).tolist(), [2, 1])
        self.assertEqual(buffer.batch_size, 2)
        self.assertEqual(len(buffer.streams), 2)
        self.assertEqual(buffer.prefixes.references, long.prefixes.references + short.prefixes.references)
        self.assertEqual(buffer.metadata["shared_state_sha256"], long.metadata["shared_state_sha256"] + short.metadata["shared_state_sha256"])
        logits = replay_student(self.student, buffer, [0, 1])
        self.assertTrue(bool((logits[~buffer.valid] == 0).all()))

    def test_prefix_reconstructed_current_weights_anchor_consumed_once(self):
        buffer = self.buffer(clocks=(3,), horizon=2)
        calls = []
        actual = self.student.forward
        def record(rgb, previous, hidden):
            calls.append((rgb.detach().clone(), previous.detach().clone(), torch.is_grad_enabled()))
            return actual(rgb, previous, hidden)
        with patch.object(self.student, "forward", side_effect=record):
            first = replay_student(self.student, buffer, [0])
        self.assertEqual([row[2] for row in calls], [False] * 3 + [True] * 2)
        prefix = buffer.prefixes.prefixes[0]
        for step in range(3):
            np.testing.assert_array_equal(calls[step][0].numpy(), prefix.rgb[step])
            expected = prefix.actions[step - 1] if step else np.full(2, 12)
            np.testing.assert_array_equal(calls[step][1].numpy(), expected)
        torch.testing.assert_close(calls[3][0], buffer.local_rgb[0, 0])
        torch.testing.assert_close(calls[4][1], buffer.actions[0, 0])
        with torch.no_grad():
            self.student.gru.bias_hh.add_(.5)
        calls.clear()
        with patch.object(self.student, "forward", side_effect=record):
            second = replay_student(self.student, buffer, [0])
        self.assertEqual([row[2] for row in calls], [False] * 3 + [True] * 2)
        self.assertFalse(torch.equal(first, second))
        second.sum().backward()
        self.assertIsNotNone(self.student.gru.weight_hh.grad)

    def test_updates_change_only_student_and_preserve_modes_global_rng_and_buffer(self):
        buffer = self.buffer()
        learner = self.learner(updates=8, sequences=4)
        frozen = tuple(module_version(value) for value in (self.fixture.actor, self.fixture.model, self.fixture.critic))
        saved = buffer.state_dict()
        self.student.train()
        self.student.encoder.eval()
        modes = [module.training for module in self.student.modules()]
        rng = torch.get_rng_state().clone()
        with patch.object(self.fixture.actor, "forward", side_effect=AssertionError("teacher execution")), \
                patch.object(self.fixture.model, "imagine_step", side_effect=AssertionError("model execution")), \
                patch.object(self.fixture.critic, "forward", side_effect=AssertionError("critic execution")):
            before = learner.diagnostics(buffer, batch_size=2)
            for _ in range(8):
                report = learner.step(buffer)
            after = learner.diagnostics(buffer, batch_size=2)
        self.assertNotEqual(module_version(self.student), self.initial_version)
        self.assertEqual(frozen, tuple(module_version(value) for value in (self.fixture.actor, self.fixture.model, self.fixture.critic)))
        self.assertEqual(modes, [module.training for module in self.student.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        assert_tree_equal(self, saved, buffer.state_dict())
        self.assertEqual(report["optimizer_update"], 8)
        self.assertEqual(len(set(report["sequence_indices"])), 4)
        self.assertEqual(before["valid_player_decisions"], 16)
        self.assertLess(after["forward_kl"], before["forward_kl"])
        self.assertTrue(all(parameter.grad is None for parameter in self.fixture.critic.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.fixture.actor.parameters()))

    def test_student_player_zero_never_reads_other_players_real_or_imagined_images(self):
        buffer = self.buffer(clocks=(3,), horizon=2)
        baseline = replay_student(self.student, buffer, [0]).detach()
        prefix = buffer.prefixes.prefixes[0]
        prefix_rgb = prefix.rgb.copy()
        prefix_rgb[:-1, 1] = 255 - prefix_rgb[:-1, 1]
        changed_prefix = AnchorPrefix(prefix.ref, prefix_rgb, prefix.actions, prefix.decision_indices)
        bank = PrefixBank((changed_prefix,), buffer.prefixes.cohort_identity,
                          buffer.prefixes.task_family, buffer.prefixes.deadline)
        images = buffer.local_rgb.clone()
        images[1:, :, 1] = 255 - images[1:, :, 1]
        changed = replace(buffer, local_rgb=images, prefixes=bank)
        logits = replay_student(self.student, changed, [0]).detach()
        torch.testing.assert_close(logits[:, :, 0], baseline[:, :, 0], rtol=0, atol=0)
        self.assertFalse(torch.equal(logits[:, :, 1], baseline[:, :, 1]))

    def test_paired_initialization_and_frozen_critic_guards(self):
        with self.assertRaisesRegex(ValueError, "paired initial hashes"):
            StudentDistiller(self.student, self.fixture.critic, paired_actor_version="a" * 64,
                             paired_critic_version=self.critic_version, seed=1)
        learner = self.learner()
        with torch.no_grad():
            next(self.fixture.critic.parameters()).add_(.001)
        with self.assertRaisesRegex(ValueError, "critic changed"):
            learner.step(self.buffer())

    def test_resume_preserves_minibatch_rng_adam_and_result_exactly(self):
        buffer = self.buffer()
        learner = self.learner()
        learner.step(buffer)
        actor_state = deepcopy(self.student.state_dict())
        state = learner.state_dict()
        expected = learner.step(buffer)
        final = deepcopy(self.student.state_dict())
        self.student.load_state_dict(self.initial)
        resumed = self.learner(seed=42)
        self.student.load_state_dict(actor_state)
        resumed.load_state_dict(state)
        observed = resumed.step(buffer)
        self.assertEqual(expected, observed)
        assert_tree_equal(self, final, self.student.state_dict())
        resumed.step(buffer)
        with self.assertRaisesRegex(ValueError, "budget"):
            resumed.step(buffer)

    def test_failed_replay_restores_private_rng_and_modes(self):
        buffer = self.buffer()
        learner = self.learner()
        rng = learner.generator.get_state().clone()
        modes = [module.training for module in self.student.modules()]
        with patch.object(self.student, "forward", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                learner.step(buffer)
        self.assertTrue(torch.equal(rng, learner.generator.get_state()))
        self.assertEqual(modes, [module.training for module in self.student.modules()])
        self.assertEqual(learner.updates_completed, 0)
        self.assertEqual(module_version(self.student), self.initial_version)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            learner.step(buffer)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            learner.state_dict()

    def test_resume_rejects_missing_moments_invalid_shapes_negative_variance_and_maximization(self):
        buffer = self.buffer()
        learner = self.learner()
        learner.step(buffer)
        original = learner.state_dict()
        first = next(iter(original["optimizer"]["state"]))
        corrupted = []
        missing = deepcopy(original)
        del missing["optimizer"]["state"][first]
        corrupted.append(missing)
        shape = deepcopy(original)
        shape["optimizer"]["state"][first]["exp_avg"] = torch.zeros(1)
        corrupted.append(shape)
        negative = deepcopy(original)
        negative["optimizer"]["state"][first]["exp_avg_sq"].fill_(-1)
        corrupted.append(negative)
        maximize = deepcopy(original)
        maximize["optimizer"]["param_groups"][0]["maximize"] = True
        corrupted.append(maximize)
        reordered = deepcopy(original)
        reordered["optimizer"]["param_groups"][0]["params"].reverse()
        corrupted.append(reordered)
        wrong_seed = deepcopy(original)
        wrong_seed["sampling_rng"] = torch.Generator().manual_seed(999).get_state()
        corrupted.append(wrong_seed)
        wrong_dtype = deepcopy(original)
        wrong_dtype["optimizer"]["state"][first]["exp_avg"] = wrong_dtype["optimizer"]["state"][first]["exp_avg"].double()
        corrupted.append(wrong_dtype)
        for index, value in enumerate(corrupted):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    learner.load_state_dict(value)
        assert_tree_equal(self, original, learner.state_dict())

    def test_critic_must_be_frozen_and_no_gradient_state_imported(self):
        self.fixture.critic.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.learner()
        self.fixture.critic.requires_grad_(False)
        next(self.fixture.critic.parameters()).grad = torch.ones_like(next(self.fixture.critic.parameters()))
        with self.assertRaisesRegex(ValueError, "gradients"):
            self.learner()
        state = self.buffer().state_dict()
        state["tensors"]["targets"].requires_grad_()
        with self.assertRaisesRegex(ValueError, "detached"):
            DistillationBuffer.from_state_dict(state)


if __name__ == "__main__":
    unittest.main()
