"""Full real-prefix reconstruction, versioned caches, and frozen anchor tensors."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_model_sampling import identifier, make_store
from world_models.contracts import PREVIOUS_ACTION_START
from world_models.imagination.anchors import AnchorBuilder
from world_models.imagination.replay import AnchorSampler
from world_models.infrastructure.config import resolve_config
from world_models.models.world_model import SamplingStreams, SharedWorldModel, build_model
from world_models.policies import CentralizedCritic, LocalActor, build_actor_critic


ROOT = Path(__file__).resolve().parents[1]
WORLD_FIELDS = ("memory", "stochastic", "logits", "decision_index")
BATCH_FIELDS = ("actor_hidden", "actor_logits", "local_rgb", "previous_actions",
                "shared_tokens", "remaining_time")


class ImaginationAnchorTests(unittest.TestCase):
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
        self.first = identifier("anchor-first", "fit")
        self.second = identifier("anchor-second", "fit")
        self.dataset = make_store(Path(self.directory.name) / "real",
                                  [(self.first, 5, False), (self.second, 6, True)])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(3201)
            self.model = SharedWorldModel()
            self.actor = LocalActor()
        self.sampler = AnchorSampler([self.dataset], seed=11, deadline=self.model.config.deadline,
                                     batch_size=3)
        self.builder = AnchorBuilder(self.model, self.actor, seed=81)

    def reference(self, decision=3, episode=None):
        return self.sampler.reference(0, episode or self.first, decision)

    def assert_world_equal(self, left, right):
        for field in WORLD_FIELDS:
            with self.subTest(world_field=field):
                torch.testing.assert_close(getattr(left, field), getattr(right, field), rtol=0, atol=0)

    def assert_batch_equal(self, left, right):
        self.assertEqual(left.references, right.references)
        self.assertEqual(left.world_model_version, right.world_model_version)
        self.assertEqual(left.actor_version, right.actor_version)
        self.assertEqual(left.cohort_identity, right.cohort_identity)
        self.assert_world_equal(left.world, right.world)
        for field in BATCH_FIELDS:
            with self.subTest(batch_field=field):
                torch.testing.assert_close(getattr(left, field), getattr(right, field), rtol=0, atol=0)

    def direct(self, reference, version):
        """Independent reset-to-anchor replay using only public core/actor APIs."""
        prefix = self.sampler.read_prefix(reference)
        rgb = torch.from_numpy(prefix.rgb.copy())
        actions = torch.from_numpy(prefix.actions.copy()).long()
        streams = SamplingStreams.from_seed(self.builder.posterior_seed(reference, version))
        with torch.no_grad():
            world = self.model.core.reset(rgb[:1].float() / 255, generator=streams.posterior)
            for index, action in enumerate(actions):
                prior = self.model.core.prior_step(world, action[None], generator=streams.prior)
                world = self.model.core.observe(prior, rgb[index + 1:index + 2].float() / 255,
                                                generator=streams.posterior)
            memories, logits = [], []
            # Replay each player's own image/action history independently.
            for player in (0, 1):
                hidden = self.actor.initial_hidden(1)
                previous = torch.tensor([PREVIOUS_ACTION_START], dtype=torch.long)
                for index, images in enumerate(rgb):
                    output = self.actor(images[player:player + 1], previous, hidden)
                    hidden = output.next_local_hidden
                    if index < len(actions):
                        previous = actions[index, player:player + 1]
                memories.append(hidden[0])
                logits.append(output.logits[0])
        return world, torch.stack(memories), torch.stack(logits), rgb[-1], (
            actions[-1] if len(actions) else torch.full((2,), PREVIOUS_ACTION_START, dtype=torch.long))

    def test_full_prefix_matches_independent_world_and_each_local_actor_replay(self):
        self.model.eval()
        self.actor.eval()
        references = (self.reference(0), self.reference(3), self.reference(4, self.second))
        batch = self.builder.build(self.sampler, references)
        self.assertEqual(batch.references, references)
        self.assertEqual(batch.seed, 81)
        self.assertEqual(batch.precision, "float32")
        self.assertEqual(batch.world.memory.shape, (3, 16, 256))
        self.assertEqual(batch.actor_hidden.shape, (3, 2, 256))
        self.assertEqual(batch.actor_logits.shape, (3, 2, 12))
        self.assertEqual(batch.shared_tokens.shape, (3, 48, 256))
        self.assertEqual(batch.local_rgb.dtype, torch.uint8)
        self.assertEqual(batch.previous_actions.dtype, torch.long)
        self.assertEqual(batch.world.decision_index.tolist(), [0, 3, 4])
        torch.testing.assert_close(batch.remaining_time, 1 - torch.tensor([0., 3., 4.]) / 400)
        for index, reference in enumerate(references):
            expected, memory, logits, rgb, previous = self.direct(reference, batch.world_model_version)
            for field in WORLD_FIELDS:
                torch.testing.assert_close(getattr(batch.world, field)[index:index + 1],
                                           getattr(expected, field), rtol=0, atol=0)
            torch.testing.assert_close(batch.actor_hidden[index], memory, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(batch.actor_logits[index], logits, rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(batch.local_rgb[index], rgb))
            self.assertTrue(torch.equal(batch.previous_actions[index], previous))
        with torch.no_grad():
            torch.testing.assert_close(batch.shared_tokens, self.model.core.representation(batch.world))

    def test_anchor_image_is_consumed_once_and_actor_output_does_not_encode_again(self):
        reference = self.reference(3)
        actor_pixels, world_pixels = [], []
        actor_hook = self.actor.encoder.register_forward_pre_hook(
            lambda _module, args: actor_pixels.extend(args[0][:, 0, 0, 0].mul(255).round().tolist()))
        world_hook = self.model.core.encoder.register_forward_pre_hook(
            lambda _module, args: world_pixels.extend(args[0][:, :, 0, 0, 0].mul(255).round().tolist()))
        try:
            batch = self.builder.build(self.sampler, (reference,))
            count = len(actor_pixels)
            for player in (0, 1):
                output = batch.actor_output(player)
                self.assertTrue(torch.equal(output.logits, batch.actor_logits[:, player]))
                self.assertTrue(torch.equal(output.next_local_hidden, batch.actor_hidden[:, player]))
            self.assertEqual(len(actor_pixels), count)
        finally:
            actor_hook.remove()
            world_hook.remove()
        self.assertEqual(sorted(actor_pixels), [0., 1., 10., 11., 20., 21., 30., 31.])
        self.assertEqual(world_pixels, [[0., 1.], [10., 11.], [20., 21.], [30., 31.]])
        with self.assertRaises(ValueError):
            batch.actor_output(2)

    def test_reconstruction_uses_no_decoder_reward_outcome_or_auxiliary_heads(self):
        with ExitStack() as stack:
            for module in (self.model.decoder, self.model.transition_heads, self.model.auxiliary_heads):
                stack.enter_context(patch.object(module, "forward", side_effect=AssertionError("prediction head used")))
            stack.enter_context(patch.object(self.model, "observe_step", side_effect=AssertionError("full prediction used")))
            batch = self.builder.build(self.sampler, (self.reference(3),))
        self.assertEqual(batch.world.decision_index.item(), 3)

    def test_cache_and_uncached_results_agree_for_reverse_order_and_duplicate_references(self):
        references = (self.reference(1), self.reference(3), self.reference(2, self.second))
        first = self.builder.build(self.sampler, references)
        reordered = (references[2], references[1], references[0], references[1])
        cached = self.builder.build(self.sampler, reordered)
        fresh = AnchorBuilder(self.model, self.actor, seed=81).build(self.sampler, reordered)
        self.assert_batch_equal(cached, fresh)
        positions = [2, 1, 0, 1]
        for field in WORLD_FIELDS:
            self.assertTrue(torch.equal(getattr(cached.world, field), getattr(first.world, field)[positions]))
        for field in BATCH_FIELDS:
            self.assertTrue(torch.equal(getattr(cached, field), getattr(first, field)[positions]))

    def test_cached_episode_prefix_reuses_observations_across_anchors(self):
        encoded = []
        hook = self.actor.encoder.register_forward_pre_hook(lambda _module, args: encoded.append(len(args[0])))
        try:
            self.builder.build(self.sampler, (self.reference(1),))
            before = sum(encoded)
            self.builder.build(self.sampler, (self.reference(3),))
            self.assertEqual(sum(encoded) - before, 4)  # Two new observations for each player.
            before = sum(encoded)
            self.builder.build(self.sampler, (self.reference(1), self.reference(3)))
            self.assertEqual(sum(encoded), before)
        finally:
            hook.remove()

    def test_actor_change_invalidates_actor_cache_and_keeps_world_posterior_identical(self):
        references = (self.reference(3),)
        initial = self.builder.build(self.sampler, references)
        # .data intentionally bypasses Tensor._version: versions must identify bytes.
        self.actor.gru.bias_hh.data.add_(0.03)
        changed = self.builder.build(self.sampler, references)
        fresh = AnchorBuilder(self.model, self.actor, seed=81).build(self.sampler, references)
        self.assertNotEqual(initial.actor_version, changed.actor_version)
        self.assertEqual(initial.world_model_version, changed.world_model_version)
        self.assert_world_equal(initial.world, changed.world)
        self.assertFalse(torch.equal(initial.actor_hidden, changed.actor_hidden))
        self.assert_batch_equal(changed, fresh)

    def test_world_change_invalidates_reconstruction_even_for_data_mutations(self):
        references = (self.reference(3),)
        initial = self.builder.build(self.sampler, references)
        self.model.core.reset_memory.data[0, 0].add_(0.75)
        changed = self.builder.build(self.sampler, references)
        fresh = AnchorBuilder(self.model, self.actor, seed=81).build(self.sampler, references)
        self.assertNotEqual(initial.world_model_version, changed.world_model_version)
        self.assertEqual(initial.actor_version, changed.actor_version)
        self.assertTrue(torch.equal(initial.actor_hidden, changed.actor_hidden))
        self.assertFalse(torch.equal(initial.world.memory, changed.world.memory))
        self.assert_batch_equal(changed, fresh)
        with torch.no_grad():
            self.model.core.reset_memory[0, 1].add_(0.5)
        self.assertNotEqual(changed.world_model_version,
                            self.builder.build(self.sampler, references).world_model_version)

    def test_mutating_returned_tensors_cannot_corrupt_cached_or_duplicate_anchors(self):
        reference = self.reference(2)
        expected = AnchorBuilder(self.model, self.actor, seed=81).build(self.sampler, (reference, reference))
        returned = self.builder.build(self.sampler, (reference, reference))
        returned.actor_hidden[0].fill_(100)
        self.assertTrue(torch.equal(returned.actor_hidden[1], expected.actor_hidden[1]))
        for field in WORLD_FIELDS:
            getattr(returned.world, field).zero_()
        for field in BATCH_FIELDS:
            getattr(returned, field).zero_()
        refreshed = self.builder.build(self.sampler, (reference, reference))
        self.assert_batch_equal(refreshed, expected)

    def test_cache_capacity_zero_or_one_preserves_results(self):
        references = (self.reference(1), self.reference(3), self.reference(2, self.second), self.reference(1))
        expected = self.builder.build(self.sampler, references)
        for capacity in (0, 1):
            with self.subTest(capacity=capacity):
                builder = AnchorBuilder(self.model, self.actor, seed=81, cache_capacity=capacity)
                self.assert_batch_equal(builder.build(self.sampler, references), expected)
                self.assert_batch_equal(builder.build(self.sampler, references), expected)

    def test_preserves_rng_mixed_module_modes_parameter_flags_and_values(self):
        self.model.train()
        self.model.core.encoder.eval()
        self.model.core.blocks[0].eval()
        self.actor.eval()
        self.actor.gru.train()
        model_parameter = next(self.model.parameters())
        actor_parameter = next(self.actor.parameters())
        model_parameter.requires_grad_(False)
        actor_parameter.requires_grad_(False)
        modules = [*self.model.modules(), *self.actor.modules()]
        parameters = [*self.model.parameters(), *self.actor.parameters()]
        modes = [module.training for module in modules]
        flags = [parameter.requires_grad for parameter in parameters]
        versions = [parameter._version for parameter in parameters]
        global_torch = torch.get_rng_state().clone()
        global_numpy = np.random.get_state()
        global_python = random.getstate()
        batch = self.builder.build(self.sampler, (self.reference(3),))
        self.assertEqual(modes, [module.training for module in modules])
        self.assertEqual(flags, [parameter.requires_grad for parameter in parameters])
        self.assertEqual(versions, [parameter._version for parameter in parameters])
        self.assertTrue(all(parameter.grad is None for parameter in parameters))
        self.assertTrue(torch.equal(global_torch, torch.get_rng_state()))
        self.assertEqual(global_python, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(global_numpy[0], after_numpy[0])
        np.testing.assert_array_equal(global_numpy[1], after_numpy[1])
        self.assertEqual(global_numpy[2:], after_numpy[2:])
        for tensor in [*(getattr(batch.world, field) for field in WORLD_FIELDS),
                       *(getattr(batch, field) for field in BATCH_FIELDS)]:
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
            self.assertFalse(torch.is_inference(tensor))

    def test_detached_anchor_memory_can_start_gradient_bearing_local_sequence(self):
        batch = self.builder.build(self.sampler, (self.reference(3),))
        images = torch.full((2, 3, 96, 128), 100, dtype=torch.uint8)
        output = self.actor(images, torch.tensor([1, 4]), batch.actor_hidden[0])
        output.log_prob(torch.tensor([2, 6])).sum().backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                            for parameter in self.actor.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_outer_inference_mode_does_not_make_anchors_unusable_for_policy_gradients(self):
        with torch.inference_mode():
            batch = self.builder.build(self.sampler, (self.reference(3),))
        for tensor in [*(getattr(batch.world, field) for field in WORLD_FIELDS),
                       *(getattr(batch, field) for field in BATCH_FIELDS)]:
            self.assertFalse(torch.is_inference(tensor))
        current = batch.actor_output(0)
        actor = self.actor(batch.local_rgb[:, 0], torch.tensor([4]), current.next_local_hidden)
        critic = CentralizedCritic()
        value = critic(batch.shared_tokens, batch.actor_hidden, batch.remaining_time)
        (actor.log_prob(torch.tensor([6])).sum() + value.square().mean()).backward()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                            for parameter in self.actor.parameters()))
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                            for parameter in critic.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_inconsistent_internal_world_configuration_is_rejected(self):
        self.model.core.config = replace(self.model.config, deadline=self.model.config.deadline + 1)
        with self.assertRaises(ValueError):
            AnchorBuilder(self.model, self.actor, seed=81)

    def test_prefix_longer_than_model_fitting_burn_in_replays_from_episode_reset(self):
        episode = identifier("long-anchor-history", "fit")
        dataset = make_store(Path(self.directory.name) / "long", [(episode, 20, False)])
        self.sampler = AnchorSampler([dataset], seed=11, deadline=400)
        reference = self.sampler.reference(0, episode, 18)
        self.model.eval()
        self.actor.eval()
        encoded = []
        hook = self.model.core.encoder.register_forward_pre_hook(lambda _module, _args: encoded.append(True))
        try:
            batch = self.builder.build(self.sampler, (reference,))
        finally:
            hook.remove()
        self.assertEqual(len(encoded), 19)
        world, hidden, logits, rgb, previous = self.direct(reference, batch.world_model_version)
        self.assert_world_equal(batch.world, world)
        torch.testing.assert_close(batch.actor_hidden[0], hidden, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(batch.actor_logits[0], logits, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(batch.local_rgb[0], rgb))
        self.assertTrue(torch.equal(batch.previous_actions[0], previous))

    def test_bfloat16_prefix_cache_extension_matches_full_uncached_reconstruction(self):
        cached = AnchorBuilder(self.model, self.actor, seed=81, precision="bfloat16")
        cached.build(self.sampler, (self.reference(1),))
        references = (self.reference(3), self.reference(1))
        result = cached.build(self.sampler, references)
        fresh = AnchorBuilder(self.model, self.actor, seed=81, precision="bfloat16", cache_capacity=0)
        self.assert_batch_equal(result, fresh.build(self.sampler, references))
        self.assertEqual(result.precision, "bfloat16")
        self.assertEqual(result.actor_hidden.dtype, torch.float32)
        self.assertEqual(result.actor_logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result.shared_tokens).all())

    def test_reconstruction_failure_restores_mixed_modes_and_discards_partial_cache(self):
        self.model.train()
        self.model.core.encoder.eval()
        self.actor.eval()
        self.actor.gru.train()
        modules = [*self.model.modules(), *self.actor.modules()]
        modes = [module.training for module in modules]
        self.builder.build(self.sampler, (self.reference(0),))
        with patch.object(self.model.core, "prior_step", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                self.builder.build(self.sampler, (self.reference(3),))
        self.assertEqual(modes, [module.training for module in modules])
        self.assertEqual(self.builder.cache_entries, 0)

    def test_paired_b_and_d_initial_weights_give_identical_anchors(self):
        results = []
        references = (self.reference(2),)
        for variant in ("B", "D"):
            settings = resolve_config("relay", variant, "pilot", ROOT / "configs").settings
            model = build_model(settings, master_seed=0, variant=variant)
            policy = build_actor_critic(settings, master_seed=0, variant=variant)
            results.append(AnchorBuilder(model, policy.actor, seed=81).build(self.sampler, references))
        self.assert_batch_equal(*results)

    def test_omitting_references_uses_sampler_batch_and_explicit_references_do_not_sample(self):
        references = (self.reference(0), self.reference(1), self.reference(2))
        with patch.object(self.sampler, "sample", return_value=references) as sample:
            actual = self.builder.build(self.sampler)
            sample.assert_called_once()
        self.assertEqual(actual.references, references)
        with patch.object(self.sampler, "sample", side_effect=AssertionError("explicit refs consumed RNG")):
            self.assert_batch_equal(self.builder.build(self.sampler, references), actual)


if __name__ == "__main__":
    unittest.main()
