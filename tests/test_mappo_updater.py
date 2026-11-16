"""P3.4 end-to-end CPU checks on newly generated synthetic fixtures only."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_imagined_rollout_generator import assert_tree_equal
from tests.test_model_sampling import identifier, make_store
from world_models.imagination.anchors import AnchorBuilder
from world_models.imagination.replay import AnchorSampler
from world_models.imagination.versions import module_version
from world_models.mappo.config import MAPPOConfig
from world_models.mappo.updater import RecurrentMAPPO
from world_models.models.core import ModelConfig
from world_models.models.heads import TaskSpec
from world_models.models.world_model import SharedWorldModel
from world_models.policies import CentralizedCritic, LocalActor
from world_models.policies.config import PolicyConfig
from world_models.rollouts.generator import RolloutGenerator
from world_models.rollouts.streams import RolloutStreams


def tensors(tree):
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from tensors(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            yield from tensors(value)


class RecurrentMAPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.episode = identifier("p34-updater", "fit")
        self.dataset = make_store(Path(self.directory.name) / "replay", [(self.episode, 4, False)])
        self.make_networks()

    def make_networks(self, *, deadline=400):
        policy = PolicyConfig(embedding_size=8, hidden_size=8, critic_hidden_sizes=(16, 16))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(3404)
            self.model = SharedWorldModel(ModelConfig(deadline=deadline), TaskSpec(deadline=deadline))
            self.actor = LocalActor(policy)
            self.critic = CentralizedCritic(policy)
        with torch.no_grad():
            self.model.transition_heads.outcome_head.weight.zero_()
            self.model.transition_heads.outcome_head.bias.fill_(-1000)
            self.model.transition_heads.outcome_head.bias[0] = 1000
        self.sampler = AnchorSampler([self.dataset], seed=34, deadline=deadline, batch_size=4)

    def collect(self, *, clocks=(0, 1, 2, 3), horizon=2, seed=31, precision="float32"):
        refs = tuple(self.sampler.reference(0, self.episode, clock) for clock in clocks)
        anchors = AnchorBuilder(self.model, self.actor, seed=17, precision=precision).build(self.sampler, refs)
        return RolloutGenerator(self.model, self.actor, self.critic, horizon=horizon,
                                precision=precision).collect(anchors, streams=RolloutStreams.from_seed(seed))

    def learner(self, *, epochs=2, minibatches=2, seed=51, precision="float32"):
        return RecurrentMAPPO(self.model, self.actor, self.critic,
                              config=MAPPOConfig(epochs=epochs, minibatches=minibatches),
                              seed=seed, precision=precision)

    def network_versions(self):
        return tuple(module_version(network) for network in (self.model, self.actor, self.critic))

    def test_update_changes_only_policy_networks_and_uses_all_whole_sequences_each_epoch(self):
        rollout = self.collect()
        old_payload, old_metadata = deepcopy(rollout.tensor_payload()), deepcopy(rollout.metadata)
        old_versions = self.network_versions()
        learner = self.learner()
        with ExitStack() as stack:
            for method in ("initialize", "observe_step", "imagine_step"):
                stack.enter_context(patch.object(self.model, method, side_effect=AssertionError("model execution during PPO")))
            report = learner.update(rollout, self.sampler)
        new_versions = self.network_versions()
        self.assertEqual(old_versions[0], new_versions[0])
        self.assertNotEqual(old_versions[1], new_versions[1])
        self.assertNotEqual(old_versions[2], new_versions[2])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["cycles_completed"], 1)
        for name in ("actor_optimizer_steps", "critic_optimizer_steps",
                     "cycle_actor_optimizer_steps", "cycle_critic_optimizer_steps"):
            self.assertEqual(report[name], 4)
        self.assertEqual(len(report["minibatches"]), 4)
        epochs = sorted({row["epoch"] for row in report["minibatches"]})
        self.assertEqual(len(epochs), 2)
        for epoch in epochs:
            rows = [row for row in report["minibatches"] if row["epoch"] == epoch]
            self.assertEqual(len(rows), 2)
            self.assertEqual(sorted(index for row in rows for index in row["sequence_indices"]), list(range(4)))
            self.assertEqual(sum(row["joint_samples"] for row in rows), int(rollout.valid.sum()))
        for name in ("initial_ratio_max_error", "initial_log_prob_max_error"):
            self.assertLess(report["metrics"][name], 2e-5)
        for name in ("approx_kl", "clip_fraction", "entropy", "value_loss", "value_error",
                     "actor_grad_norm", "critic_grad_norm", "mean_trajectory_length",
                     "predicted_termination_frequency"):
            self.assertTrue(math.isfinite(report["metrics"][name]), name)
        self.assertEqual(report["world_model_version"], old_versions[0])
        self.assertEqual(report["actor_version_before"], old_versions[1])
        self.assertEqual(report["actor_version_after"], new_versions[1])
        self.assertEqual(report["critic_version_before"], old_versions[2])
        self.assertEqual(report["critic_version_after"], new_versions[2])
        json.dumps(report, allow_nan=False)
        assert_tree_equal(self, old_payload, rollout.tensor_payload())
        self.assertEqual(old_metadata, rollout.metadata)

    def test_all_default_epochs_and_minibatches_run_with_no_early_stopping(self):
        rollout = self.collect(horizon=1)
        report = self.learner(epochs=4, minibatches=4).update(rollout, self.sampler)
        self.assertEqual(report["cycle_actor_optimizer_steps"], 16)
        self.assertEqual(report["cycle_critic_optimizer_steps"], 16)
        self.assertEqual(len(report["minibatches"]), 16)

    def test_padding_deadline_and_cutoff_rows_have_valid_finite_updates(self):
        self.make_networks(deadline=3)
        rollout = self.collect(clocks=(0, 1, 2, 2))
        self.assertEqual(rollout.lengths.tolist(), [2, 2, 1, 1])
        self.assertTrue(rollout.cutoff[1, 0])
        self.assertEqual(int(rollout.terminated.sum()), 3)
        report = self.learner().update(rollout, self.sampler)
        self.assertEqual(report["metrics"]["mean_trajectory_length"], 1.5)
        self.assertGreater(report["metrics"]["predicted_termination_frequency"], 0)
        for epoch in {row["epoch"] for row in report["minibatches"]}:
            self.assertEqual(sum(row["joint_samples"] for row in report["minibatches"]
                                 if row["epoch"] == epoch), 6)
        json.dumps(report, allow_nan=False)

    def test_stale_model_actor_or_critic_rejected_before_updates_or_rng_consumption(self):
        rollout = self.collect()
        learner = self.learner()
        for network in (self.model, self.actor, self.critic):
            with self.subTest(network=type(network).__name__):
                parameter = next(network.parameters())
                original = parameter.detach().clone()
                with torch.no_grad():
                    parameter.add_(.01)
                before = deepcopy(learner.state_dict())
                versions = self.network_versions()
                with self.assertRaises((ValueError, RuntimeError)):
                    learner.update(rollout, self.sampler)
                self.assertEqual(versions, self.network_versions())
                assert_tree_equal(self, before, learner.state_dict())
                with torch.no_grad():
                    parameter.copy_(original)

    def test_tampered_behavior_probabilities_fail_before_any_parameter_change(self):
        rollout = self.collect()
        changed_logits = rollout.actor_logits.clone()
        changed_logits[0, 0, 0, rollout.actions[0, 0, 0]] += .3
        changed_log_probs = changed_logits[:-1].log_softmax(-1).gather(
            -1, rollout.actions.unsqueeze(-1)).squeeze(-1)
        changed = replace(rollout, actor_logits=changed_logits, log_probs=changed_log_probs)
        learner = self.learner()
        before, versions = deepcopy(learner.state_dict()), self.network_versions()
        with self.assertRaisesRegex(ValueError, r"max_abs_error=.*tolerance=2e-05.*step=0, trajectory=0, player=0"):
            learner.update(changed, self.sampler)
        self.assertEqual(versions, self.network_versions())
        assert_tree_equal(self, before, learner.state_dict())

    def test_different_replay_cohort_is_rejected_before_optimization(self):
        rollout = self.collect()
        other_episode = identifier("p34-wrong-cohort", "fit")
        other_dataset = make_store(Path(self.directory.name) / "other", [(other_episode, 4, False)])
        other_sampler = AnchorSampler([other_dataset], seed=34, deadline=400, batch_size=4)
        learner = self.learner()
        before = deepcopy(learner.state_dict())
        with self.assertRaises((ValueError, RuntimeError)):
            learner.update(rollout, other_sampler)
        assert_tree_equal(self, before, learner.state_dict())

    def test_update_preserves_world_gradients_modes_global_rng_and_sampler_state(self):
        rollout = self.collect()
        for network in (self.model, self.actor, self.critic):
            network.train()
            list(network.modules())[-1].eval()
        world_parameter = next(self.model.parameters())
        world_parameter.grad = torch.full_like(world_parameter, .125)
        old_gradient = world_parameter.grad.clone()
        old_flags = [parameter.requires_grad for parameter in self.model.parameters()]
        modes = [(module, module.training) for network in (self.model, self.actor, self.critic)
                 for module in network.modules()]
        global_rng, sampler_state = torch.random.get_rng_state().clone(), deepcopy(self.sampler.state_dict())
        self.learner().update(rollout, self.sampler)
        self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(old_gradient, world_parameter.grad))
        self.assertEqual(old_flags, [parameter.requires_grad for parameter in self.model.parameters()])
        self.assertEqual([mode for _, mode in modes], [module.training for module, _ in modes])
        assert_tree_equal(self, sampler_state, self.sampler.state_dict())

    def test_critic_uses_immutable_rollout_memories_tokens_and_clock_after_actor_changes(self):
        rollout = self.collect()
        observed = []

        def record(_module, args):
            observed.append(tuple(value.detach().clone() for value in args))

        hook = self.critic.register_forward_pre_hook(record)
        try:
            report = self.learner().update(rollout, self.sampler)
        finally:
            hook.remove()
        # The exact number/order of diagnostic value forwards is intentionally
        # unconstrained; every observed feature row must be frozen rollout data.
        self.assertTrue(observed)
        expected = [(rollout.shared_tokens[step, batch], rollout.actor_hidden[step, batch],
                     (400 - rollout.world.decision_index[step, batch]).float() / 400)
                    for step, batch in rollout.valid.nonzero().tolist()]
        for features in observed:
            for actual in zip(*features):
                self.assertTrue(any(all(torch.equal(a, e) for a, e in zip(actual, candidate))
                                    for candidate in expected))
        self.assertNotEqual(report["actor_version_before"], report["actor_version_after"])

    def test_optimizer_rng_and_counters_resume_the_next_update_exactly(self):
        learner = self.learner()
        learner.update(self.collect(), self.sampler)
        actor_state, critic_state = deepcopy(self.actor.state_dict()), deepcopy(self.critic.state_dict())
        saved = deepcopy(learner.state_dict())
        rollout = self.collect(seed=32)
        first_report = learner.update(rollout, self.sampler)
        first_actor, first_critic, first_state = (deepcopy(self.actor.state_dict()),
                                                deepcopy(self.critic.state_dict()), deepcopy(learner.state_dict()))
        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(critic_state)
        restored = self.learner()
        restored.load_state_dict(saved)
        second_report = restored.update(rollout, self.sampler)
        assert_tree_equal(self, first_actor, self.actor.state_dict())
        assert_tree_equal(self, first_critic, self.critic.state_dict())
        assert_tree_equal(self, first_state, restored.state_dict())
        self.assertEqual(first_report, second_report)
        self.assertEqual(second_report["cycles_completed"], 2)
        self.assertEqual(second_report["actor_optimizer_steps"], 8)

    def test_exported_state_is_owned_and_bad_resume_is_transactional(self):
        learner = self.learner()
        learner.update(self.collect(), self.sampler)
        before = deepcopy(learner.state_dict())
        exported = learner.state_dict()
        for value in tensors(exported):
            value.zero_()
        assert_tree_equal(self, before, learner.state_dict())
        bad_actor, bad_critic, bad_rng = deepcopy(before), deepcopy(before), deepcopy(before)
        bad_actor["actor_optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
        bad_critic["critic_optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
        bad_rng["shuffle_rng"] = torch.zeros(1, dtype=torch.uint8)
        for invalid in (None, {**before, "unexpected": True}, {**before, "seed": -1},
                        bad_actor, bad_critic, bad_rng):
            with self.subTest(invalid_type=type(invalid).__name__):
                with self.assertRaises((ValueError, RuntimeError, TypeError)):
                    learner.load_state_dict(invalid)
                assert_tree_equal(self, before, learner.state_dict())

    def test_mid_cycle_failure_rolls_back_parameters_optimizers_and_rng(self):
        rollout = self.collect()
        learner = self.learner()
        old_actor, old_critic = deepcopy(self.actor.state_dict()), deepcopy(self.critic.state_dict())
        old_state = deepcopy(learner.state_dict())
        # Fail at the third optimizer step, after one actor/critic pair has
        # already mutated weights and Adam moments.
        original_step = torch.optim.Adam.step
        calls = 0

        def interrupted(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("injected mid-cycle failure")
            return original_step(optimizer, *args, **kwargs)

        with patch.object(torch.optim.Adam, "step", interrupted):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                learner.update(rollout, self.sampler)
        self.assertEqual(calls, 3)
        assert_tree_equal(self, old_actor, self.actor.state_dict())
        assert_tree_equal(self, old_critic, self.critic.state_dict())
        assert_tree_equal(self, old_state, learner.state_dict())
        # A diagnosed retry still receives the full configured schedule.
        self.assertEqual(learner.update(rollout, self.sampler)["cycle_actor_optimizer_steps"], 4)

    def test_nonfinite_gradient_aborts_without_committing_a_partial_update(self):
        rollout = self.collect()
        learner = self.learner()
        before, versions = deepcopy(learner.state_dict()), self.network_versions()
        hook = next(self.actor.parameters()).register_hook(lambda value: torch.full_like(value, float("nan")))
        try:
            with self.assertRaises((ValueError, RuntimeError)):
                learner.update(rollout, self.sampler)
        finally:
            hook.remove()
        self.assertEqual(versions, self.network_versions())
        assert_tree_equal(self, before, learner.state_dict())

    def test_bfloat16_collection_replays_and_updates_with_float32_adam_moments(self):
        rollout = self.collect(precision="bfloat16")
        learner = self.learner(epochs=1, precision="bfloat16")
        report = learner.update(rollout, self.sampler)
        self.assertLess(report["metrics"]["initial_log_prob_max_error"], 2e-5)
        self.assertEqual(report["metrics"]["behavior_log_prob_tolerance"], 2e-5)
        self.assertEqual(report["cycle_actor_optimizer_steps"], 2)
        for value in tensors(learner.state_dict()):
            if value.is_floating_point():
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(torch.isfinite(value).all())

    def test_bfloat16_context_rejects_small_behavior_corruption_before_updates(self):
        # This drift would pass the previous BF16 0.02 tolerance. The actor is
        # now FP32 even in mixed precision, so the strict check must catch it.
        rollout = self.collect(precision="bfloat16")
        logits = rollout.actor_logits.clone()
        logits[0, 2, 1, rollout.actions[0, 2, 1]] += .005
        logs = rollout.log_probs.clone()
        logs[rollout.valid] = logits[:-1][rollout.valid].log_softmax(-1).gather(
            -1, rollout.actions[rollout.valid][..., None]).squeeze(-1)
        changed = replace(rollout, actor_logits=logits, log_probs=logs)
        learner = self.learner(precision="bfloat16")
        before, versions = deepcopy(learner.state_dict()), self.network_versions()
        with self.assertRaisesRegex(ValueError, r"step=0, trajectory=2, player=1.*actor_precision=float32"):
            learner.update(changed, self.sampler)
        self.assertEqual(versions, self.network_versions())
        assert_tree_equal(self, before, learner.state_dict())

    def test_precision_mismatch_or_too_many_minibatches_is_rejected_before_updates(self):
        rollout = self.collect()
        for learner in (self.learner(precision="bfloat16"), self.learner(minibatches=5)):
            before, versions = deepcopy(learner.state_dict()), self.network_versions()
            with self.assertRaises((ValueError, RuntimeError)):
                learner.update(rollout, self.sampler)
            self.assertEqual(versions, self.network_versions())
            assert_tree_equal(self, before, learner.state_dict())

    def test_update_inside_outer_inference_mode_keeps_trainable_ordinary_parameters(self):
        rollout = self.collect()
        learner = self.learner(epochs=1)
        with torch.inference_mode():
            report = learner.update(rollout, self.sampler)
            state = learner.state_dict()
        self.assertEqual(report["cycle_actor_optimizer_steps"], 2)
        self.assertTrue(all(not value.is_inference() for value in tensors(state)))
        self.assertTrue(all(parameter.requires_grad and not parameter.is_inference()
                            for network in (self.actor, self.critic) for parameter in network.parameters()))


if __name__ == "__main__":
    unittest.main()
