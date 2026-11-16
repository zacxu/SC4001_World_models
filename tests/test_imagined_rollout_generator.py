"""P3.3 real-prefix to on-policy prior-only trajectories on CPU fixtures."""
from contextlib import ExitStack
from dataclasses import fields, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_model_sampling import identifier, make_store
from world_models.imagination.anchors import AnchorBuilder
from world_models.imagination.replay import AnchorSampler
from world_models.imagination.versions import module_version
from world_models.infrastructure.config import resolve_config
from world_models.models.core import ModelConfig
from world_models.models.heads import TaskSpec
from world_models.models.world_model import SharedWorldModel
from world_models.policies import ActorOutput, CentralizedCritic, LocalActor
from world_models.rollouts.generator import RolloutGenerator
from world_models.rollouts.schema import EndReason, ImaginedRollout
from world_models.rollouts.streams import RolloutStreams
from world_models.rollouts.storage import load_rollout, save_rollout


WORLD_FIELDS = ("memory", "stochastic", "logits", "decision_index")


def assert_tree_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for name in left:
            assert_tree_equal(test, left[name], right[name])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class ImaginedRolloutGeneratorTests(unittest.TestCase):
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
        self.episode = identifier("p33-generator", "fit")
        self.dataset = make_store(Path(self.directory.name) / "replay", [(self.episode, 4, False)])
        self.make_networks()

    def make_networks(self, *, deadline=400, outcome=0):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(3303)
            self.model = SharedWorldModel(ModelConfig(deadline=deadline), TaskSpec(deadline=deadline))
            self.actor = LocalActor()
            self.critic = CentralizedCritic()
        self.force_outcome(outcome)
        with torch.no_grad():
            for parameter in self.critic.parameters():
                parameter.zero_()
            self.critic.mlp[-1].bias.fill_(3.25)
        self.sampler = AnchorSampler([self.dataset], seed=4, deadline=deadline, batch_size=2)

    def force_outcome(self, outcome):
        with torch.no_grad():
            self.model.transition_heads.outcome_head.weight.zero_()
            self.model.transition_heads.outcome_head.bias.fill_(-1000)
            self.model.transition_heads.outcome_head.bias[outcome] = 1000
            self.model.transition_heads.bonus_head.weight.zero_()
            self.model.transition_heads.bonus_head.bias.fill_(-1000)

    def anchors(self, *clocks, precision="float32"):
        refs = tuple(self.sampler.reference(0, self.episode, clock) for clock in clocks)
        return AnchorBuilder(self.model, self.actor, seed=9, precision=precision).build(self.sampler, refs)

    def collect(self, anchors, *, horizon=3, seed=31, precision="float32"):
        return RolloutGenerator(self.model, self.actor, self.critic, horizon=horizon,
                                precision=precision).collect(anchors, streams=RolloutStreams.from_seed(seed))

    def test_one_joint_prior_step_no_real_suffix_or_posterior_and_no_anchor_reencoding(self):
        anchors = self.anchors(0, 1)
        actor_rows, prior_rows, executed = [], [], []

        def record_actor(_module, args):
            actor_rows.append(args[0].shape[0])
            executed.append(args[1].clone())

        actor_hook = self.actor.register_forward_pre_hook(record_actor)
        prior_original = self.model.core.prior_step

        def prior(*args, **kwargs):
            prior_rows.append(args[0].memory.shape[0])
            return prior_original(*args, **kwargs)

        streams = RolloutStreams.from_seed(3)
        posterior = streams.world.posterior.get_state().clone()
        try:
            with ExitStack() as stack:
                for target, method in ((self.sampler, "read_prefix"), (self.dataset, "load_episode"),
                                       (self.model, "initialize"), (self.model, "observe_step"),
                                       (self.model.core, "observe"), (self.model.core.encoder, "forward")):
                    stack.enter_context(patch.object(target, method, side_effect=AssertionError("real/posterior access")))
                joint = stack.enter_context(patch.object(self.model, "imagine_step", wraps=self.model.imagine_step))
                stack.enter_context(patch.object(self.model.core, "prior_step", side_effect=prior))
                rollout = RolloutGenerator(self.model, self.actor, self.critic, horizon=3).collect(anchors, streams=streams)
        finally:
            actor_hook.remove()
        self.assertEqual(joint.call_count, 3)
        self.assertEqual(prior_rows, [2, 2, 2])
        self.assertEqual(actor_rows, [4, 4, 4])
        self.assertEqual(sum(actor_rows), 2 * int(rollout.valid.sum()))
        self.assertTrue(torch.equal(posterior, streams.world.posterior.get_state()))
        self.assertTrue(torch.equal(rollout.actor_hidden[0], anchors.actor_hidden))
        self.assertTrue(torch.equal(rollout.actor_logits[0], anchors.actor_logits))
        for step, previous in enumerate(executed):
            self.assertTrue(torch.equal(previous, rollout.actions[step].flatten()))
            self.assertTrue(torch.equal(rollout.previous_actions[step + 1], rollout.actions[step]))

    def test_stored_behavior_likelihood_and_recurrent_replay_begin_with_ratio_one(self):
        anchors = self.anchors(0, 2)
        rollout = self.collect(anchors, horizon=3)
        self.assertEqual((rollout.steps, rollout.batch_size), (3, 2))
        for player in (0, 1):
            initial = anchors.actor_output(player).log_prob(rollout.actions[0, :, player])
            torch.testing.assert_close(initial, rollout.log_probs[0, :, player], rtol=0, atol=0)
        with torch.no_grad():
            for state in range(1, rollout.steps + 1):
                active = rollout.state_valid[state]
                output = self.actor(rollout.local_rgb[state, active].flatten(0, 1),
                                    rollout.previous_actions[state, active].flatten(),
                                    rollout.actor_hidden[state - 1, active].flatten(0, 1))
                torch.testing.assert_close(output.next_local_hidden.reshape(-1, 2, 256),
                                           rollout.actor_hidden[state, active], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(output.logits.reshape(-1, 2, 12),
                                           rollout.actor_logits[state, active], rtol=1e-5, atol=1e-6)
                if state < rollout.steps:
                    new_log = output.log_prob(rollout.actions[state, active].flatten()).reshape(-1, 2)
                    torch.testing.assert_close((new_log - rollout.log_probs[state, active]).exp(),
                                               torch.ones_like(new_log), rtol=1e-5, atol=1e-6)
        self.assertTrue(rollout.valid.all())
        self.assertFalse(rollout.terminated.any())
        self.assertTrue(rollout.cutoff[-1].all())
        self.assertTrue((rollout.end_reason[-1] == EndReason.HORIZON).all())

    def test_nonterminal_horizon_bootstraps_final_raw_value_and_retains_destination(self):
        rollout = self.collect(self.anchors(0), horizon=2)
        self.assertEqual(rollout.values.shape, (3, 1))
        self.assertTrue((rollout.values == 3.25).all())
        self.assertEqual(rollout.bootstrap_values[-1, 0].item(), 3.25)
        self.assertEqual(rollout.bootstrap_mask[-1, 0].item(), 1)
        self.assertEqual(rollout.recursion_mask[-1, 0].item(), 0)
        self.assertFalse(rollout.terminated[-1, 0])
        self.assertTrue(rollout.cutoff[-1, 0])
        self.assertEqual(rollout.world.decision_index[:, 0].tolist(), [0, 1, 2])
        self.assertGreater(rollout.local_rgb[-1].count_nonzero().item(), 0)

    def test_learned_success_and_failure_stop_early_with_zero_bootstrap(self):
        for sampled, reason, reward in ((1, EndReason.SUCCESS, .999), (2, EndReason.FAILURE, -1.001)):
            with self.subTest(sampled=sampled):
                self.force_outcome(sampled)
                rollout = self.collect(self.anchors(1), horizon=4)
                self.assertEqual(rollout.steps, 1)
                self.assertTrue(rollout.terminated[0, 0])
                self.assertFalse(rollout.cutoff.any())
                self.assertEqual(rollout.end_reason[0, 0].item(), reason)
                self.assertEqual(rollout.sampled_outcomes[0, 0].item(), sampled)
                self.assertAlmostEqual(rollout.rewards[0, 0].item(), reward, places=6)
                self.assertEqual(rollout.values[1, 0].item(), 3.25)
                self.assertEqual(rollout.bootstrap_values[0, 0].item(), 0)
                self.assertEqual(rollout.recursion_mask[0, 0].item(), 0)
                self.assertTrue(rollout.state_valid[1, 0])

    def test_exact_deadline_and_learned_terminal_take_precedence_over_horizon(self):
        self.make_networks(deadline=3)
        for sampled, reason, is_deadline in ((0, EndReason.DEADLINE, True),
                                             (1, EndReason.SUCCESS, False),
                                             (2, EndReason.FAILURE, False)):
            with self.subTest(sampled=sampled):
                self.force_outcome(sampled)
                rollout = self.collect(self.anchors(2), horizon=1)
                self.assertEqual(rollout.world.decision_index[-1, 0].item(), 3)
                self.assertEqual(rollout.end_reason[0, 0].item(), reason)
                self.assertEqual(rollout.deadline_failure[0, 0].item(), is_deadline)
                self.assertTrue(rollout.terminated[0, 0])
                self.assertFalse(rollout.cutoff.any())
                self.assertEqual(rollout.bootstrap_values[0, 0].item(), 0)
                self.assertEqual(rollout.outcomes[0, 0].item(), 1 if sampled == 1 else 2)

    def test_mixed_lengths_freeze_finished_world_rows_and_never_draw_padding_actions(self):
        self.make_networks(deadline=3)
        anchors = self.anchors(0, 2)
        streams = RolloutStreams.from_seed(44)
        initial = streams.state_dict()
        observations, snapshots, batch_sizes = [], [], []
        original = self.model.imagine_step

        def imagine(current, actions, **kwargs):
            result = original(current, actions, **kwargs)
            snapshots.append((current.world.memory.clone(), result.next.world.memory.clone(), current.ended.clone()))
            observations.append(actions.clone())
            return result

        hook = self.actor.register_forward_pre_hook(lambda _module, args: batch_sizes.append(args[0].shape[0]))
        try:
            with patch.object(self.model, "imagine_step", side_effect=imagine):
                rollout = RolloutGenerator(self.model, self.actor, self.critic, horizon=4).collect(anchors, streams=streams)
        finally:
            hook.remove()
        self.assertEqual(rollout.lengths.tolist(), [3, 1])
        self.assertEqual(batch_sizes, [4, 2, 2])
        self.assertEqual(rollout.valid.tolist(), [[True, True], [True, False], [True, False]])
        self.assertTrue((rollout.end_reason[1:, 1] == EndReason.PADDING).all())
        for source, destination, ended in snapshots[1:]:
            self.assertTrue(ended[1])
            self.assertTrue(torch.equal(source[1], destination[1]))
        self.assertTrue(all(torch.equal(actions[1], torch.zeros(2, dtype=torch.long)) for actions in observations[1:]))
        self.assertTrue((rollout.actor_hidden[2:, 1] == 0).all())
        self.assertTrue((rollout.local_rgb[2:, 1] == 0).all())
        for player in (0, 1):
            reference = torch.Generator().manual_seed(0)
            reference.set_state(initial["actors"][str(player)])
            for step, active in enumerate(rollout.valid):
                selected = ActorOutput(rollout.actor_logits[step, active, player],
                                       rollout.actor_hidden[step, active, player]).sample(generator=reference)
                self.assertTrue(torch.equal(selected, rollout.actions[step, active, player]))
            self.assertTrue(torch.equal(reference.get_state(), streams.actors[player].get_state()))

    def test_exact_resume_reproduces_every_trajectory_tensor_and_preserves_global_rng(self):
        anchors = self.anchors(1, 2)
        streams = RolloutStreams.from_seed(52)
        saved, global_rng = streams.state_dict(), torch.random.get_rng_state().clone()
        generator = RolloutGenerator(self.model, self.actor, self.critic, horizon=2)
        first = generator.collect(anchors, streams=streams)
        streams.load_state_dict(saved)
        second = generator.collect(anchors, streams=streams)
        for field in fields(ImaginedRollout):
            left, right = getattr(first, field.name), getattr(second, field.name)
            if field.name == "world":
                for name in WORLD_FIELDS:
                    self.assertTrue(torch.equal(getattr(left, name), getattr(right, name)))
            else:
                assert_tree_equal(self, left, right)
        self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(saved["world"]["posterior"], streams.world.posterior.get_state()))

    def test_stale_world_or_actor_anchor_rejected_before_any_rng_advance(self):
        streams = RolloutStreams.from_seed(61)
        initial = streams.state_dict()
        for network in (self.model, self.actor):
            anchors = self.anchors(0)
            with torch.no_grad():
                next(network.parameters()).view(-1)[0].add_(.125)
            with self.assertRaisesRegex(ValueError, "stale anchors"):
                RolloutGenerator(self.model, self.actor, self.critic).collect(anchors, streams=streams)
            assert_tree_equal(self, initial, streams.state_dict())

    def test_current_critic_version_is_stamped_without_invalidating_anchors(self):
        anchors = self.anchors(0)
        previous = module_version(self.critic)
        with torch.no_grad():
            self.critic.mlp[-1].bias.add_(1)
        rollout = self.collect(anchors, horizon=1)
        self.assertNotEqual(previous, rollout.metadata["critic_version"])
        self.assertEqual(rollout.metadata["critic_version"], module_version(self.critic))
        self.assertTrue((rollout.values == 4.25).all())

    def test_frozen_weights_modes_requires_grad_and_existing_gradients_are_unchanged(self):
        self.model.train()
        self.model.core.eval()
        self.actor.train()
        self.actor.encoder[0].eval()
        self.critic.eval()
        self.critic.mlp[0].train()
        networks = (self.model, self.actor, self.critic)
        parameters = tuple(parameter for network in networks for parameter in network.parameters())
        parameters[0].requires_grad_(False)
        for network in networks:
            parameter = tuple(network.parameters())[-1]
            parameter.grad = torch.full_like(parameter, .25)
        modes = [(module, module.training) for network in networks for module in network.modules()]
        flags = [parameter.requires_grad for parameter in parameters]
        gradients = [(parameter, parameter.grad, parameter.grad.clone() if parameter.grad is not None else None)
                     for parameter in parameters]
        versions = [module_version(network) for network in networks]
        anchors = self.anchors(1)
        rollout = self.collect(anchors, horizon=2)
        self.assertEqual(versions, [module_version(network) for network in networks])
        self.assertEqual(flags, [parameter.requires_grad for parameter in parameters])
        self.assertTrue(all(module.training == mode for module, mode in modes))
        for parameter, original, snapshot in gradients:
            self.assertIs(parameter.grad, original)
            if snapshot is not None:
                self.assertTrue(torch.equal(parameter.grad, snapshot))
        for field in fields(ImaginedRollout):
            value = getattr(rollout, field.name)
            if isinstance(value, torch.Tensor):
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
                self.assertFalse(value.is_inference())

    def test_exception_rolls_back_all_streams_and_restores_mixed_modes(self):
        anchors = self.anchors(0)
        self.actor.train()
        self.actor.encoder[0].eval()
        modes = [(module, module.training) for network in (self.model, self.actor, self.critic)
                 for module in network.modules()]
        streams = RolloutStreams.from_seed(74)
        original_state = streams.state_dict()
        with patch.object(self.actor, "forward", side_effect=RuntimeError("injected actor error")):
            with self.assertRaisesRegex(RuntimeError, "injected actor error"):
                RolloutGenerator(self.model, self.actor, self.critic).collect(anchors, streams=streams)
        assert_tree_equal(self, original_state, streams.state_dict())
        self.assertTrue(all(module.training == mode for module, mode in modes))

    def test_concurrent_weight_change_is_detected_and_rng_rolls_back(self):
        anchors = self.anchors(0)
        streams = RolloutStreams.from_seed(75)
        original_state = streams.state_dict()
        modes = [(module, module.training) for network in (self.model, self.actor, self.critic)
                 for module in network.modules()]
        original = self.model.imagine_step

        def mutate(*args, **kwargs):
            transition = original(*args, **kwargs)
            with torch.no_grad():
                self.critic.mlp[-1].bias.add_(.125)
            return transition

        with patch.object(self.model, "imagine_step", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "weights changed"):
                RolloutGenerator(self.model, self.actor, self.critic, horizon=1).collect(anchors, streams=streams)
        assert_tree_equal(self, original_state, streams.state_dict())
        self.assertTrue(all(module.training == mode for module, mode in modes))

    def test_cpu_bfloat16_collection_replays_likelihoods_at_the_same_precision(self):
        anchors = self.anchors(0, 1, precision="bfloat16")
        rollout = self.collect(anchors, horizon=2, precision="bfloat16")
        self.assertEqual(rollout.metadata["precision"], "bfloat16")
        self.assertEqual(rollout.log_probs.dtype, torch.float32)
        self.assertEqual(rollout.actor_hidden.dtype, torch.float32)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            output = self.actor(rollout.local_rgb[1].flatten(0, 1), rollout.previous_actions[1].flatten(),
                                rollout.actor_hidden[0].flatten(0, 1))
            replayed = output.log_prob(rollout.actions[1].flatten()).reshape(2, 2)
        torch.testing.assert_close((replayed - rollout.log_probs[1]).exp(), torch.ones_like(replayed),
                                   rtol=1e-5, atol=1e-6)

    def test_precision_horizon_and_task_stream_guards_reject_before_sampling(self):
        anchors = self.anchors(0)
        streams = RolloutStreams.from_seed(81)
        original_state = streams.state_dict()
        with self.assertRaisesRegex(ValueError, "precision must match"):
            RolloutGenerator(self.model, self.actor, self.critic, precision="bfloat16").collect(anchors, streams=streams)
        assert_tree_equal(self, original_state, streams.state_dict())
        for horizon in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "horizon"):
                RolloutGenerator(self.model, self.actor, self.critic, horizon=horizon)
        with self.assertRaisesRegex(ValueError, "precision"):
            RolloutGenerator(self.model, self.actor, self.critic, precision="float16")
        cover = resolve_config("cover", "B", "pilot", Path(__file__).resolve().parents[1] / "configs").settings
        other_task = RolloutStreams.from_experiment(cover, master_seed=0)
        other_state = other_task.state_dict()
        with self.assertRaisesRegex(ValueError, "RNG task differs"):
            RolloutGenerator(self.model, self.actor, self.critic).collect(anchors, streams=other_task)
        assert_tree_equal(self, other_state, other_task.state_dict())

    def test_actual_generated_rollout_storage_preserves_tensors_metadata_and_rng_resume(self):
        rollout = self.collect(self.anchors(0, 1), horizon=2)
        directory = Path(self.directory.name) / "synthetic_rollout"
        save_rollout(rollout, directory)
        restored = load_rollout(directory)
        assert_tree_equal(self, rollout.tensor_payload(), restored.tensor_payload())
        assert_tree_equal(self, rollout.metadata, restored.metadata)
        assert_tree_equal(self, rollout.streams_before, restored.streams_before)
        assert_tree_equal(self, rollout.streams_after, restored.streams_after)
        self.assertTrue((directory / "manifest.sha256").is_file())
        self.assertEqual(restored.metadata["data_source"], "world_model_imagination")
        streams = RolloutStreams.from_seed(31)
        streams.load_state_dict(restored.streams_after)
        assert_tree_equal(self, streams.state_dict(), restored.streams_after)

    def test_experiment_factory_rejects_model_settings_outside_resolved_config(self):
        config = resolve_config("relay", "B", "pilot", Path(__file__).resolve().parents[1] / "configs").settings
        with torch.random.fork_rng(devices=[]):
            mismatched = SharedWorldModel(replace(self.model.config, uniform_mixture=.02), self.model.task)
        with self.assertRaisesRegex(ValueError, "configuration"):
            RolloutGenerator.from_experiment(mismatched, self.actor, self.critic, config)


if __name__ == "__main__":
    unittest.main()
