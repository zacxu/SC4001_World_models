"""Integration invariants of the untrained P2.2 shared model."""
from dataclasses import replace
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from world_models.infrastructure.config import resolve_config
from world_models.models.core import WorldState
from world_models.models.world_model import SamplingStreams, SharedWorldModel, build_model


ROOT = Path(__file__).resolve().parents[1]


class WorldModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(311)
            self.model = SharedWorldModel()
        self.images = torch.randint(0, 256, (2, 2, 3, 96, 128), dtype=torch.uint8,
                                    generator=torch.Generator().manual_seed(40))
        self.actions = torch.tensor([[0, 7], [6, 2]], dtype=torch.long)
        self.streams = SamplingStreams.from_seed(42)

    def _continue_only(self):
        with torch.no_grad():
            head = self.model.transition_heads.outcome_head
            head.weight.zero_()
            head.bias.copy_(torch.tensor([100., -100., -100.]))

    def test_initial_state_reconstruction_and_state_only_auxiliary(self):
        state = self.model.initialize(self.images, streams=self.streams)
        prediction = self.model.reconstruct(state)
        self.assertEqual(tuple(prediction.representation.shape), (2, 48, 256))
        self.assertEqual(tuple(prediction.rgb.shape), (2, 2, 3, 96, 128))
        self.assertEqual(state.decision_index.tolist(), [0, 0])
        self.assertTrue(all(not key.startswith("event/") for key in prediction.auxiliary))
        self.assertIn("task/stage", prediction.auxiliary)

    def test_prior_preserved_when_either_terminal_view_missing(self):
        state = self.model.initialize(self.images, streams=self.streams)
        images = self.images.float() / 255
        images[1] = float("nan")  # Invalid slots may not enter the encoder.
        valid = torch.tensor([[True, True], [True, False]])
        prediction = self.model.observe_step(state, self.actions, images, streams=self.streams,
                                             next_image_valid=valid)
        self.assertEqual(prediction.posterior_valid.tolist(), [True, False])
        for key in ("memory", "stochastic", "logits", "decision_index"):
            self.assertTrue(torch.equal(getattr(prediction.state, key)[1], getattr(prediction.prior, key)[1]))
        self.assertFalse(torch.equal(prediction.state.logits[0], prediction.prior.logits[0]))
        self.assertEqual(prediction.state.decision_index.tolist(), [1, 1])

    def test_all_missing_skips_encoder_and_still_predicts_reward_outcome(self):
        state = self.model.initialize(self.images, streams=self.streams)
        with patch.object(self.model.core.encoder, "forward", side_effect=AssertionError("unavailable image used")):
            step = self.model.observe_step(state, self.actions, torch.full_like(self.images.float(), float("nan")),
                                           streams=self.streams, next_image_valid=torch.zeros((2, 2), dtype=torch.bool))
        self.assertIs(step.state, step.prior)
        self.assertTrue(torch.isfinite(step.predictions.bonus).all())
        self.assertEqual(tuple(step.predictions.outcome_logits.shape), (2, 3))

    def test_sampling_resume_and_global_rng_independence(self):
        self._continue_only()
        state = self.model.initialize(self.images, streams=self.streams)
        rng_checkpoint = self.streams.state_dict()
        actions = self.actions.unsqueeze(0).expand(3, -1, -1)
        with torch.no_grad():
            first = self.model.imagine(state, actions, streams=self.streams)
            self.streams.load_state_dict(rng_checkpoint)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(99999)
                global_before = torch.get_rng_state().clone()
                second = self.model.imagine(state, actions, streams=self.streams)
                self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a.transition.rgb, b.transition.rgb))
            self.assertTrue(torch.equal(a.transition.state.stochastic, b.transition.state.stochastic))
            self.assertTrue(torch.equal(a.reward, b.reward))

    def test_autonomous_rollout_has_no_future_image_or_privileged_route(self):
        self._continue_only()
        state = self.model.initialize(self.images, streams=self.streams)
        parameters = set(inspect.signature(self.model.imagine_step).parameters)
        self.assertEqual(parameters, {"current", "actions", "streams"})
        with torch.no_grad(), patch.object(self.model.core.encoder, "forward", side_effect=AssertionError("future input used")):
            result = self.model.imagine(state, self.actions.unsqueeze(0).expand(3, -1, -1), streams=self.streams)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(not step.transition.posterior_valid.any() for step in result))

    def test_shared_representation_reaches_decoder_once_for_both_views(self):
        state = self.model.initialize(self.images, streams=self.streams)
        captured = []
        handle = self.model.decoder.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        try:
            transition = self.model.observe_step(state, self.actions, self.images, streams=self.streams)
        finally:
            handle.remove()
        self.assertEqual(len(captured), 1)
        self.assertIs(captured[0], transition.representation)
        self.assertEqual(tuple(transition.state.stochastic.shape), (2, 32, 32))

    def test_independent_batch_termination_freezes_finished_rows(self):
        self._continue_only()
        state = self.model.initialize(self.images, streams=self.streams)
        state = replace(state, decision_index=torch.tensor([399, 397]))
        with torch.no_grad():
            result = self.model.imagine(state, self.actions.unsqueeze(0).expand(8, -1, -1), streams=self.streams)
        self.assertEqual(len(result), 3)
        self.assertEqual([step.valid.tolist() for step in result], [[True, True], [False, True], [False, True]])
        self.assertEqual(result[-1].next.world.decision_index.tolist(), [400, 400])
        self.assertEqual(result[-1].next.ended.tolist(), [True, True])
        self.assertTrue(result[0].deadline_failure[0])
        self.assertAlmostEqual(float(result[0].reward[0] - result[0].transition.predictions.bonus[0]), -1.001, places=6)
        self.assertEqual(float(result[1].reward[0]), 0.)
        self.assertTrue(torch.equal(result[0].next.world.memory[0], result[2].next.world.memory[0]))
        with self.assertRaisesRegex(ValueError, "all imagined episodes ended"):
            self.model.imagine_step(result[-1].next, self.actions, streams=self.streams)

    def test_gradient_paths_through_posterior_dynamics_decoder_and_heads(self):
        state = self.model.initialize(self.images, streams=self.streams)
        step = self.model.observe_step(state, self.actions, self.images.flip(-1), streams=self.streams)
        # Connectivity probe only; P2.3 defines the actual research objective.
        loss = step.rgb.square().mean() + step.predictions.bonus.mean()
        loss = loss + step.predictions.outcome_logits.square().mean() + step.prior.logits.square().mean()
        loss = loss + sum(value.square().mean() for value in step.auxiliary.values()) / len(step.auxiliary)
        loss.backward()
        parameters = dict(self.model.named_parameters())
        names = ["core.encoder.cnn.0.weight", "core.reset_memory", "core.latent_projection.weight",
                 "core.action_embedding.weight", "core.action_attention.in_proj_weight",
                 "core.blocks.0.self_attn.in_proj_weight", "core.prior_head.3.weight",
                 "core.posterior_head.3.weight", "decoder.cnn.0.weight",
                 "joint_action_projection.0.weight", "transition_heads.outcome_head.weight"]
        for name in names:
            with self.subTest(parameter=name):
                gradient = parameters[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)

    def test_paired_initialization_matches_and_preserves_global_rng(self):
        left = resolve_config("relay", "B", "pilot", ROOT / "configs").settings
        right = resolve_config("relay", "D", "pilot", ROOT / "configs").settings
        before = torch.get_rng_state().clone()
        with patch("torch.cuda.manual_seed_all") as seed_cuda:
            a = build_model(left, master_seed=0, variant="B")
            seed_cuda.assert_not_called()
        b = build_model(right, master_seed=0, variant="D")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, b.state_dict()[key]) for key, value in a.state_dict().items()))
        c = build_model(left, master_seed=1, variant="B")
        self.assertFalse(torch.equal(a.core.reset_memory, c.core.reset_memory))

    def test_model_checkpoint_roundtrip(self):
        self._continue_only()
        state = self.model.initialize(self.images, streams=self.streams)
        clone = SharedWorldModel()
        clone.load_state_dict(self.model.state_dict())
        checkpoint = self.streams.state_dict()
        a = self.model.imagine_step(self.model.start_imagination(state), self.actions, streams=self.streams)
        self.streams.load_state_dict(checkpoint)
        b = clone.imagine_step(clone.start_imagination(state), self.actions, streams=self.streams)
        self.assertTrue(torch.equal(a.transition.rgb, b.transition.rgb))
        self.assertTrue(torch.equal(a.reward, b.reward))

    def test_ordered_actions_are_preserved_in_transition_heads(self):
        embedded = self.model.core.action_tokens(self.actions)
        projection = self.model.joint_action_projection
        self.assertFalse(torch.allclose(projection(embedded.flatten(1)), projection(embedded.flip(1).flatten(1))))

    def test_cpu_bfloat16_probability_dtype_and_gradient(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            state = self.model.initialize(self.images, streams=self.streams)
            step = self.model.observe_step(state, self.actions, self.images, streams=self.streams)
            loss = step.rgb.float().mean() + step.predictions.bonus.mean()
        self.assertEqual(state.logits.dtype, torch.float32)
        self.assertEqual(state.stochastic.dtype, torch.float32)
        self.assertEqual(step.predictions.outcome_logits.dtype, torch.float32)
        self.assertEqual(step.predictions.bonus.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.core.encoder.cnn[0].weight.grad).all())

    def test_invalid_images_actions_and_expired_anchor_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model.initialize(torch.zeros_like(self.images, dtype=torch.long), streams=self.streams)
        state = self.model.initialize(self.images, streams=self.streams)
        with self.assertRaises(ValueError):
            self.model.observe_step(state, torch.full_like(self.actions, 12), self.images, streams=self.streams)
        with self.assertRaises(ValueError):
            self.model.start_imagination(replace(state, decision_index=torch.tensor([400, 400])))


if __name__ == "__main__":
    unittest.main()
