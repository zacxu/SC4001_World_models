"""Behavioral and gradient checks for the P2.2 recurrent visual core."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from world_models.models.core import (
    ImageEncoder, ModelConfig, ObservationDecoder, StateCore, WorldState,
    categorical_probs, sample_categorical,
)


def stream(seed=123):
    return torch.Generator(device="cpu").manual_seed(seed)


class CategoricalTests(unittest.TestCase):
    def test_mixture_float32_and_straight_through(self):
        logits = torch.linspace(-1000, 1000, 32).repeat(2, 32, 1).requires_grad_()
        probabilities = categorical_probs(logits.to(torch.bfloat16))
        self.assertEqual(probabilities.dtype, torch.float32)
        self.assertTrue(torch.all(probabilities >= 0.01 / 32))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 32))
        sample = sample_categorical(torch.zeros_like(logits, requires_grad=True), generator=stream())
        self.assertTrue(torch.all((sample == 0) | (sample == 1)))
        torch.testing.assert_close(sample.sum(-1), torch.ones(2, 32))
        differentiable_logits = torch.zeros(1, 32, 32, requires_grad=True)
        draw = sample_categorical(differentiable_logits, generator=stream())
        (draw * torch.arange(32)).sum().backward()
        self.assertGreater(differentiable_logits.grad.abs().sum().item(), 0)

    def test_explicit_stream_is_reproducible_and_isolated(self):
        logits = torch.zeros(2, 32, 32)
        global_before = torch.random.get_rng_state().clone()
        generator = stream(3)
        first = sample_categorical(logits, generator=generator)
        saved = generator.get_state()
        second = sample_categorical(logits, generator=generator)
        generator.set_state(saved)
        self.assertTrue(torch.equal(second, sample_categorical(logits, generator=generator)))
        self.assertTrue(torch.equal(first, sample_categorical(logits, generator=stream(3))))
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))

    def test_bad_categorical_inputs_rejected(self):
        for mixture in (-0.1, 1.0, float("nan"), True):
            with self.assertRaises(ValueError):
                categorical_probs(torch.zeros(32), mixture)
        with self.assertRaises(ValueError):
            sample_categorical(torch.zeros(32), generator=None)
        with self.assertRaises(ValueError):
            categorical_probs(torch.tensor([float("nan"), 0.0]))
        generator = stream()
        before = generator.get_state()
        draw = sample_categorical(torch.tensor([0.0, 1.0]), generator=generator, deterministic=True)
        self.assertTrue(torch.equal(draw, torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.equal(before, generator.get_state()))


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(12345)
        cls.config = ModelConfig()
        cls.core = StateCore(cls.config)
        cls.decoder = ObservationDecoder(cls.config)
        cls.observations = torch.rand(2, 2, 3, 96, 128, generator=stream(7))

    def state(self):
        return self.core.reset(self.observations, generator=stream())

    def test_configuration_matches_approved_dimensions(self):
        self.assertEqual(self.config.encoder_channels, (32, 64, 128, 256))
        self.assertEqual(len(self.core.blocks), 4)
        self.assertTrue(all(block.norm_first for block in self.core.blocks))
        self.assertTrue(all(block.self_attn.num_heads == 8 for block in self.core.blocks))
        with self.assertRaises(ValueError):
            ModelConfig(width=128)
        with self.assertRaises(ValueError):
            ModelConfig(deadline=True)
        config = ModelConfig.from_experiment({
            "environment": {"agents": 2, "action_count": 12, "image_shape": [3, 96, 128]},
            "world_model": {"encoder_channels": [32, 64, 128, 256], "width": 256},
            "task": {"deadline": 600},
        })
        self.assertEqual(config.deadline, 600)

    def test_spatial_shape_and_shared_independent_view_encoder(self):
        encoded = self.core.encoder(self.observations)
        self.assertEqual(tuple(encoded.shape), (2, 2, 48, 256))
        changed = self.observations.clone()
        changed[:, 1] = 1 - changed[:, 1]
        encoded_changed = self.core.encoder(changed)
        self.assertTrue(torch.equal(encoded[:, 0], encoded_changed[:, 0]))
        self.assertFalse(torch.equal(encoded[:, 1], encoded_changed[:, 1]))
        shared_views = self.observations[:, :1].expand(-1, 2, -1, -1, -1)
        shared_encoded = self.core.encoder(shared_views)
        identity_difference = self.core.encoder.agent_embedding[0] - self.core.encoder.agent_embedding[1]
        torch.testing.assert_close(shared_encoded[:, 0] - shared_encoded[:, 1],
                                   identity_difference.expand(2, 48, -1), atol=5e-7, rtol=1e-4)

    def test_reset_uses_learned_constant_memory_and_posterior_evidence(self):
        state = self.state()
        self.assertEqual(tuple(state.memory.shape), (2, 16, 256))
        self.assertEqual(tuple(state.stochastic.shape), (2, 32, 32))
        self.assertEqual(tuple(state.logits.shape), (2, 32, 32))
        self.assertEqual(state.logits.dtype, torch.float32)
        self.assertTrue(torch.equal(state.decision_index, torch.zeros(2, dtype=torch.long)))
        self.assertTrue(torch.equal(state.memory[0], state.memory[1]))
        changed = self.core.reset(1 - self.observations, generator=stream())
        self.assertTrue(torch.equal(state.memory, changed.memory))
        self.assertFalse(torch.equal(state.logits, changed.logits))

    def test_prior_then_posterior_preserves_clock_and_memory(self):
        state = self.state()
        prior = self.core.prior_step(state, torch.tensor([[0, 1], [2, 3]]), generator=stream(4))
        self.assertTrue(torch.equal(prior.decision_index, torch.ones(2, dtype=torch.long)))
        posterior = self.core.observe(prior, 1 - self.observations, generator=stream(5))
        self.assertIs(posterior.memory, prior.memory)
        self.assertIs(posterior.decision_index, prior.decision_index)
        self.assertFalse(torch.equal(posterior.logits, prior.logits))
        self.assertTrue(torch.equal(state.decision_index, torch.zeros(2, dtype=torch.long)))
        self.assertFalse(torch.equal(state.memory, prior.memory))

    def test_each_view_supplies_posterior_evidence_and_prior_needs_no_views(self):
        state = self.state()
        baseline = self.core.observe(state, self.observations, generator=stream())
        for agent in (0, 1):
            observations = self.observations.clone()
            observations[:, agent] = 1 - observations[:, agent]
            changed = self.core.observe(state, observations, generator=stream())
            self.assertTrue(torch.equal(changed.memory, baseline.memory))
            self.assertFalse(torch.equal(changed.logits, baseline.logits))
        with patch.object(self.core.encoder, "forward", side_effect=AssertionError("prior used visual evidence")):
            first = self.core.prior_step(state, torch.zeros(2, 2, dtype=torch.long), generator=stream(1))
            second = self.core.prior_step(state, torch.zeros(2, 2, dtype=torch.long), generator=stream(2))
        self.assertTrue(torch.equal(first.memory, second.memory))
        self.assertTrue(torch.equal(first.logits, second.logits))
        self.assertFalse(torch.equal(first.stochastic, second.stochastic))

    def test_actions_agent_identity_and_clock_affect_dynamics(self):
        state = self.state()
        with torch.no_grad():
            first = self.core.prior_step(state, torch.tensor([[0, 6], [0, 6]]), generator=stream(9))
            swapped = self.core.prior_step(state, torch.tensor([[6, 0], [6, 0]]), generator=stream(9))
            later = self.core.prior_step(replace(state, decision_index=torch.full((2,), 100)),
                                         torch.tensor([[0, 6], [0, 6]]), generator=stream(9))
        self.assertFalse(torch.equal(first.memory, swapped.memory))
        self.assertFalse(torch.equal(first.memory, later.memory))
        self.assertFalse(torch.equal(first.logits, swapped.logits))

    def test_shared_decoder_shape_range_and_no_hidden_rng(self):
        with torch.no_grad():
            global_before = torch.random.get_rng_state().clone()
            state = self.state()
            next_state = self.core.prior_step(state, torch.zeros(2, 2, dtype=torch.long), generator=stream(9))
            shared = self.core.representation(next_state)
            decoded = self.decoder(shared)
            repeated = self.decoder(shared)
        self.assertEqual(tuple(shared.shape), (2, 48, 256))
        self.assertEqual(tuple(decoded.shape), (2, 2, 3, 96, 128))
        self.assertTrue(torch.isfinite(decoded).all())
        self.assertTrue(torch.all((decoded >= 0) & (decoded <= 1)))
        self.assertTrue(torch.equal(decoded, repeated))
        self.assertFalse(torch.equal(decoded[:, 0], decoded[:, 1]))
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))

    def test_end_to_end_gradients_reach_transition_and_visual_paths(self):
        self.core.zero_grad(set_to_none=True)
        self.decoder.zero_grad(set_to_none=True)
        state = self.state()
        prior = self.core.prior_step(state, torch.tensor([[1, 6], [2, 7]]), generator=stream(8))
        posterior = self.core.observe(prior, 1 - self.observations, generator=stream(9))
        decoded = self.decoder(self.core.representation(posterior))
        loss = (decoded - self.observations).square().mean() + prior.logits.square().mean() + posterior.logits.square().mean()
        loss.backward()
        parameters = {
            "image convolution": self.core.encoder.cnn[0].weight,
            "learned reset": self.core.reset_memory,
            "latent projection": self.core.latent_projection.weight,
            "action embedding": self.core.action_embedding.weight,
            "action cross attention": self.core.action_attention.in_proj_weight,
            "prior": self.core.prior_head[-1].weight,
            "posterior cross attention": self.core.posterior_attention.in_proj_weight,
            "posterior": self.core.posterior_head[-1].weight,
            "decoder attention": self.decoder.attention.in_proj_weight,
            "decoder convolution": self.decoder.cnn[0].weight,
        }
        parameters.update({f"Transformer block {i}": block.linear1.weight
                           for i, block in enumerate(self.core.blocks)})
        for name, parameter in parameters.items():
            with self.subTest(path=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        detached = posterior.detach()
        self.assertFalse(detached.memory.requires_grad)
        self.assertFalse(detached.stochastic.requires_grad)
        self.assertFalse(detached.logits.requires_grad)

    def test_cpu_autocast_keeps_probabilities_float32(self):
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            state = self.state()
            state = self.core.prior_step(state, torch.zeros(2, 2, dtype=torch.long), generator=stream())
            decoded = self.decoder(self.core.representation(state))
        self.assertEqual(state.logits.dtype, torch.float32)
        self.assertEqual(state.stochastic.dtype, torch.float32)
        self.assertEqual(categorical_probs(state.logits).dtype, torch.float32)
        self.assertTrue(torch.isfinite(decoded).all())

    def test_invalid_inputs_and_expired_state_rejected(self):
        state = self.state()
        invalid_actions = (torch.full((2, 2), 12), torch.zeros(2, 2), torch.zeros(2, 3, dtype=torch.long))
        for action in invalid_actions:
            with self.assertRaises(ValueError):
                self.core.prior_step(state, action, generator=stream())
        for observation in (torch.zeros(2, 2, 3, 96, 128, dtype=torch.uint8), self.observations * 2,
                            torch.full_like(self.observations, float("nan"))):
            with self.assertRaises(ValueError):
                self.core.reset(observation, generator=stream())
        with self.assertRaises(ValueError):
            self.core.prior_step(replace(state, decision_index=torch.full((2,), 400)),
                                 torch.zeros(2, 2, dtype=torch.long), generator=stream())
        with self.assertRaises(ValueError):
            self.core.representation(replace(state, stochastic=torch.ones(2, 32, 32) / 32))
        with self.assertRaises(ValueError):
            self.core.representation(replace(state, memory=[]))
        with self.assertRaises(ValueError):
            self.core.representation(replace(state, decision_index=[0, 0]))


if __name__ == "__main__":
    unittest.main()
