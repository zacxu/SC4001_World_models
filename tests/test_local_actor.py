"""P3.1 neural locality, recurrent replay, and real/imagined image boundaries."""
from __future__ import annotations

import inspect
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch
from torch import nn

from world_models.contracts import (
    EpisodeBoundary, JointTransitionMetadata, LocalActorInput, LocalActorOutput,
    PREVIOUS_ACTION_START, RGBFrame,
)
from world_models.env.coop import ResetResult, StepResult
from world_models.env.locality import actor_step, audit_locality
from world_models.policies import ActorOutput, LocalActor, imagined_rgb_to_uint8
from world_models.policies.actor import _actor_fp32


class LocalActorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(3101)
            self.actor = LocalActor()
        self.generator = torch.Generator().manual_seed(82)
        self.images = torch.randint(0, 256, (2, 3, 96, 128), dtype=torch.uint8,
                                    generator=self.generator)
        self.hidden = self.actor.initial_hidden(2)
        self.start = torch.full((2,), PREVIOUS_ACTION_START, dtype=torch.long)

    @staticmethod
    def frame(image):
        return RGBFrame(image.numpy().tobytes())

    def test_local_network_architecture_and_no_batch_dependent_layers(self):
        self.assertEqual(set(inspect.signature(self.actor.forward).parameters),
                         {"local_rgb", "previous_action", "local_hidden"})
        convolutions = [module for module in self.actor.encoder if isinstance(module, nn.Conv2d)]
        self.assertEqual([module.out_channels for module in convolutions], [32, 64, 128, 256])
        self.assertEqual(self.actor.projection[1].in_features, 256 * 6 * 8)
        self.assertEqual(self.actor.projection[1].out_features, 256)
        self.assertEqual((self.actor.gru.input_size, self.actor.gru.hidden_size), (269, 256))
        self.assertEqual(sum(isinstance(module, nn.Linear) for module in self.actor.action_head), 2)
        forbidden = (nn.modules.batchnorm._BatchNorm, nn.Dropout, nn.Dropout2d, nn.Dropout3d)
        self.assertFalse(any(isinstance(module, forbidden) for module in self.actor.modules()))
        output = self.actor(self.images, self.start, self.hidden)
        self.assertEqual(tuple(output.logits.shape), (2, 12))
        self.assertEqual(tuple(output.next_local_hidden.shape), (2, 256))
        self.assertEqual(output.distribution.probs.dtype, torch.float32)
        self.assertTrue(torch.all(output.distribution.probs > 0))

    def test_start_resets_memory_and_blocks_previous_episode_gradients(self):
        old = torch.randn((2, 256), generator=self.generator, requires_grad=True)
        actions = torch.tensor([PREVIOUS_ACTION_START, 4])
        result = self.actor(self.images, actions, old)
        fresh = self.actor(self.images[:1], self.start[:1], self.hidden[:1])
        torch.testing.assert_close(result.logits[:1], fresh.logits, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(result.next_local_hidden[:1], fresh.next_local_hidden, rtol=1e-5, atol=1e-7)
        result.next_local_hidden.square().sum().backward()
        self.assertTrue(torch.equal(old.grad[0], torch.zeros_like(old.grad[0])))
        self.assertGreater(old.grad[1].abs().sum().item(), 0)

    def test_previous_executed_action_and_own_history_affect_policy(self):
        actions = torch.tensor([0, 7])
        baseline = self.actor(self.images, actions, self.hidden)
        changed_action = self.actor(self.images, torch.tensor([1, 8]), self.hidden)
        changed_history = self.actor(self.images, actions, torch.ones_like(self.hidden))
        changed_image = self.actor(255 - self.images, actions, self.hidden)
        for changed in (changed_action, changed_history, changed_image):
            self.assertFalse(torch.equal(baseline.logits, changed.logits))
            self.assertFalse(torch.equal(baseline.next_local_hidden, changed.next_local_hidden))
        start = self.actor(self.images, self.start, self.hidden)
        self.assertFalse(torch.equal(start.logits, baseline.logits))

    def test_neural_actor_passes_environment_metadata_and_teammate_locality_audit(self):
        observations = tuple(self.frame(image) for image in self.images)
        reset = ResetResult(observations, {"positions": [1, 2], "role": 0})
        boundary = EpisodeBoundary()
        step = StepResult(
            tuple(self.frame(255 - image) for image in self.images), -0.001, boundary,
            JointTransitionMetadata(0, 0, 4, (1, 7), boundary), {"teacher_tokens": [3, 4]},
        )
        result = audit_locality([reset, step], actor=self.actor)
        self.assertEqual(result["status"], "passed")
        for player in result["players"]:
            self.assertTrue(player["outputs_equal_after_global_and_teammate_changes"])
            self.assertTrue(player["final_memories_equal"])

    def test_weights_are_shared_but_agent_memories_are_independent(self):
        first_memory = self.actor.initial_hidden(1)
        second_memory = self.actor.initial_hidden(1)
        self.assertNotEqual(first_memory.data_ptr(), second_memory.data_ptr())
        first_memory.fill_(2)
        self.assertTrue(torch.equal(second_memory, torch.zeros_like(second_memory)))
        before = first_memory.clone()
        first = self.actor(self.images[:1], torch.tensor([2]), first_memory)
        second = self.actor(self.images[1:], torch.tensor([9]), second_memory)
        self.assertTrue(torch.equal(before, first_memory))
        again = self.actor(self.images[:1], torch.tensor([2]), first_memory)
        self.assertTrue(torch.equal(first.logits, again.logits))
        self.assertTrue(torch.equal(first.next_local_hidden, again.next_local_hidden))
        self.assertNotEqual(first.next_local_hidden.data_ptr(), second.next_local_hidden.data_ptr())
        self.assertEqual(list(self.actor.buffers()), [])

    def test_train_eval_and_batch_composition_do_not_change_local_policy(self):
        actions = torch.tensor([2, 7])
        self.actor.train()
        together = self.actor(self.images, actions, self.hidden)
        alone = self.actor(self.images[:1], actions[:1], self.hidden[:1])
        self.actor.eval()
        evaluation = self.actor(self.images, actions, self.hidden)
        self.assertTrue(torch.equal(together.logits, evaluation.logits))
        self.assertTrue(torch.equal(together.next_local_hidden, evaluation.next_local_hidden))
        torch.testing.assert_close(together.logits[:1], alone.logits, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(together.next_local_hidden[:1], alone.next_local_hidden, rtol=1e-5, atol=1e-7)

    def test_sequence_matches_step_loop_and_replayed_likelihood_ratio_is_one(self):
        images = torch.stack((self.images, 255 - self.images, self.images.roll(5, dims=-1)))
        previous = torch.tensor([[12, 12], [1, 7], [4, 2]])
        executed = torch.tensor([[1, 7], [4, 2], [9, 6]])
        hidden, logits, memories, log_probs = self.hidden, [], [], []
        for rgb, action, taken in zip(images, previous, executed):
            output = self.actor(rgb, action, hidden)
            hidden = output.next_local_hidden
            logits.append(output.logits)
            memories.append(hidden)
            log_probs.append(output.log_prob(taken).detach())
        replay = self.actor.forward_sequence(images, previous, self.hidden)
        self.assertTrue(torch.equal(replay.logits, torch.stack(logits)))
        self.assertTrue(torch.equal(replay.local_hidden, torch.stack(memories)))
        self.assertTrue(torch.equal(replay.final_local_hidden, hidden))
        current = ActorOutput(replay.logits, replay.local_hidden).log_prob(executed)
        ratios = (current - torch.stack(log_probs)).exp()
        self.assertTrue(torch.equal(ratios, torch.ones_like(ratios)))

    def test_padding_skips_encoding_and_invalid_action_slots_without_changing_memory(self):
        images = self.images.unsqueeze(0).expand(4, -1, -1, -1, -1).clone()
        previous = torch.tensor([[12, 12], [1, 7], [999, -99], [2, 999]])
        valid = torch.tensor([[True, True], [True, True], [False, False], [True, False]])
        encoded_batches = []
        hook = self.actor.encoder.register_forward_pre_hook(
            lambda _module, args: encoded_batches.append(args[0].shape[0]))
        try:
            result = self.actor.forward_sequence(images, previous, self.hidden, valid=valid)
        finally:
            hook.remove()
        self.assertEqual(encoded_batches, [2, 2, 1])
        self.assertTrue(torch.equal(result.local_hidden[2], result.local_hidden[1]))
        self.assertTrue(torch.equal(result.local_hidden[3, 1], result.local_hidden[1, 1]))
        self.assertTrue(torch.equal(result.logits[~valid], torch.zeros_like(result.logits[~valid])))
        self.assertTrue(torch.equal(result.valid, valid))
        valid.fill_(False)
        self.assertTrue(result.valid.any())

    def test_autocast_sequence_matches_step_loop_with_bfloat16_initial_memory(self):
        images = torch.stack((self.images, 255 - self.images, self.images.roll(3, dims=-1)))
        previous = torch.tensor([[1, 7], [4, 2], [3, 8]])
        initial = torch.randn((2, 256), generator=self.generator).bfloat16()
        hidden, logits, memories = initial, [], []
        with torch.autocast("cpu", dtype=torch.bfloat16):
            for rgb, action in zip(images, previous):
                output = self.actor(rgb, action, hidden)
                hidden = output.next_local_hidden
                logits.append(output.logits)
                memories.append(hidden)
            replay = self.actor.forward_sequence(images, previous, initial)
        self.assertEqual(replay.local_hidden.dtype, torch.float32)
        self.assertEqual(replay.logits.dtype, torch.float32)
        self.assertTrue(torch.equal(replay.logits, torch.stack(logits)))
        self.assertTrue(torch.equal(replay.local_hidden, torch.stack(memories)))
        self.assertTrue(torch.equal(replay.final_local_hidden, hidden))

    def test_outer_bfloat16_does_not_round_actor_internals_or_gradients(self):
        # A peaked, nontrivial head exposes errors hidden by a nearly uniform
        # initialized policy. Include changing active batch sizes and histories.
        with torch.no_grad():
            self.actor.action_head[-1].weight.mul_(12)
            self.actor.action_head[-1].bias.copy_(torch.linspace(-3, 3, 12))
        images = torch.stack([self.images.roll(step, dims=-1) for step in range(5)])
        previous = torch.tensor([[12, 12], [1, 7], [4, 2], [3, 8], [0, 999]])
        valid = torch.tensor([[True, True]] * 4 + [[True, False]])
        initial = self.hidden.bfloat16()
        before = {name: value.clone() for name, value in self.actor.state_dict().items()}
        reference = self.actor.forward_sequence(images, previous, initial, valid=valid)
        reference.logits[valid].log_softmax(-1)[:, 0].mean().backward()
        gradients = [parameter.grad.clone() for parameter in self.actor.parameters()]
        self.actor.zero_grad(set_to_none=True)
        dtypes = []
        hooks = [module.register_forward_hook(lambda _module, _args, out: dtypes.append(out.dtype))
                 for module in (self.actor.encoder, self.actor.projection, self.actor.gru,
                                self.actor.action_head)]
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                replay = self.actor.forward_sequence(images, previous, initial, valid=valid)
                self.assertTrue(torch.is_autocast_enabled("cpu"))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertTrue(dtypes)
        self.assertEqual(set(dtypes), {torch.float32})
        torch.testing.assert_close(replay.logits, reference.logits, rtol=0, atol=0)
        torch.testing.assert_close(replay.local_hidden, reference.local_hidden, rtol=0, atol=0)
        replay.logits[valid].log_softmax(-1)[:, 0].mean().backward()
        for parameter, expected in zip(self.actor.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.actor.encoder[0].weight.grad.abs().sum().item(), 0)
        for name, value in self.actor.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_cuda_ieee_precision_scope_restores_backend_settings_on_failure(self):
        # Backend configuration is testable without CUDA execution. Mock only
        # the device autocast context, and exercise actual precision settings.
        backends = (torch.backends.cuda.matmul, torch.backends.cudnn.conv)
        original = tuple(backend.fp32_precision for backend in backends)
        try:
            for backend in backends:
                backend.fp32_precision = "tf32"
            with patch("world_models.policies.actor.torch.autocast", return_value=nullcontext()):
                with self.assertRaisesRegex(RuntimeError, "injected actor failure"):
                    with _actor_fp32(torch.device("cuda")):
                        self.assertEqual(tuple(backend.fp32_precision for backend in backends), ("ieee", "ieee"))
                        raise RuntimeError("injected actor failure")
            self.assertEqual(tuple(backend.fp32_precision for backend in backends), ("tf32", "tf32"))
        finally:
            for backend, precision in zip(backends, original):
                backend.fp32_precision = precision

    def test_actor_rejects_non_fp32_parameters_instead_of_silently_rounding(self):
        self.actor.bfloat16()
        with self.assertRaisesRegex(ValueError, "parameters must remain FP32"):
            self.actor(self.images, self.start, self.hidden)

    def test_all_padding_neither_encodes_images_nor_resets_hidden(self):
        initial = torch.randn((2, 256), generator=self.generator, requires_grad=True)
        images = self.images.unsqueeze(0).expand(2, -1, -1, -1, -1)
        starts = self.start.unsqueeze(0).expand(2, -1)
        calls = []
        hook = self.actor.encoder.register_forward_pre_hook(lambda _module, _args: calls.append(True))
        try:
            result = self.actor.forward_sequence(images, starts, initial,
                                                 valid=torch.zeros((2, 2), dtype=torch.bool))
        finally:
            hook.remove()
        self.assertEqual(calls, [])
        self.assertTrue(torch.equal(result.final_local_hidden, initial))
        result.final_local_hidden.sum().backward()
        self.assertTrue(torch.equal(initial.grad, torch.ones_like(initial)))

    def test_sequence_reset_blocks_previous_history_gradient(self):
        initial = torch.randn((2, 256), generator=self.generator, requires_grad=True)
        images = self.images.unsqueeze(0).expand(2, -1, -1, -1, -1)
        previous = torch.tensor([[3, 4], [12, 6]])
        result = self.actor.forward_sequence(images, previous, initial)
        result.final_local_hidden.square().sum().backward()
        self.assertTrue(torch.equal(initial.grad[0], torch.zeros_like(initial.grad[0])))
        self.assertGreater(initial.grad[1].abs().sum().item(), 0)
        fresh = self.actor(self.images[:1], self.start[:1], self.hidden[:1])
        torch.testing.assert_close(result.final_local_hidden[:1], fresh.next_local_hidden, rtol=1e-5, atol=1e-7)

    def test_explicit_sampling_streams_are_independent_resumable_and_global_rng_safe(self):
        output = ActorOutput(torch.zeros((64, 12)), torch.zeros((64, 256)))
        first_stream, second_stream = torch.Generator().manual_seed(32), torch.Generator().manual_seed(91)
        global_before = torch.get_rng_state().clone()
        first = output.sample(generator=first_stream)
        output.sample(generator=first_stream)
        second = output.sample(generator=second_stream)
        expected_second = output.sample(generator=torch.Generator().manual_seed(91))
        self.assertTrue(torch.equal(second, expected_second))
        self.assertFalse(torch.equal(first, second))
        saved = second_stream.get_state().clone()
        expected_next = output.sample(generator=second_stream)
        second_stream.set_state(saved)
        self.assertTrue(torch.equal(expected_next, output.sample(generator=second_stream)))
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
        self.assertTrue(torch.all((first >= 0) & (first < PREVIOUS_ACTION_START)))
        with self.assertRaises(ValueError):
            output.sample(generator=None)

    def test_log_probability_rejects_start_and_invalid_actions(self):
        output = self.actor(self.images, self.start, self.hidden)
        for actions in (self.start, torch.tensor([-1, 3]), torch.tensor([1., 2.]), torch.tensor([[1, 2]])):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                output.log_prob(actions)
        self.assertEqual(output.log_prob(torch.tensor([0, 11])).dtype, torch.float32)

    def test_uint8_imagination_conversion_detaches_and_matches_real_observations(self):
        decoded = (self.images.float() / 255).requires_grad_()
        converted = imagined_rgb_to_uint8(decoded)
        self.assertEqual(converted.dtype, torch.uint8)
        self.assertFalse(converted.requires_grad)
        self.assertIsNone(converted.grad_fn)
        self.assertTrue(torch.equal(converted, self.images))
        real = self.actor(self.images, self.start, self.hidden)
        imagined = self.actor(converted, self.start, self.hidden)
        self.assertTrue(torch.equal(real.logits, imagined.logits))
        imagined.logits.square().mean().backward()
        self.assertIsNone(decoded.grad)
        self.assertGreater(self.actor.encoder[0].weight.grad.abs().sum().item(), 0)
        decimal = torch.full((1, 2, 3, 96, 128), 0.5)
        self.assertTrue(torch.all(imagined_rgb_to_uint8(decimal) == 128))

    def test_imagination_conversion_rejects_malformed_or_nonfinite_images(self):
        invalid = (self.images, self.images.float() + 1, torch.full_like(self.images.float(), float("nan")),
                   torch.zeros((3, 96, 127)), torch.zeros((0, 3, 96, 128)))
        for images in invalid:
            with self.subTest(shape=images.shape, dtype=images.dtype), self.assertRaises(ValueError):
                imagined_rgb_to_uint8(images)

    def test_contract_step_matches_tensor_boundary_and_preserves_module_mode(self):
        inputs = LocalActorInput(self.frame(self.images[0]), 7, (0.25,) * 256)
        self.actor.train()
        actual = actor_step(self.actor, inputs.local_rgb, inputs.previous_action, inputs.local_hidden)
        expected = self.actor(self.images[:1], torch.tensor([7]), torch.full((1, 256), 0.25))
        self.assertIsInstance(actual, LocalActorOutput)
        self.assertTrue(self.actor.training)
        torch.testing.assert_close(torch.tensor(actual.action_probabilities), expected.distribution.probs[0],
                                   rtol=0, atol=0)
        torch.testing.assert_close(torch.tensor(actual.next_local_hidden), expected.next_local_hidden[0],
                                   rtol=0, atol=0)
        with self.assertRaises(ValueError):
            self.actor.step(LocalActorInput(inputs.local_rgb, 7, (0.0,) * 64))
        with self.assertRaises(ValueError):
            self.actor.step({"local_rgb": inputs.local_rgb, "previous_action": 7,
                             "local_hidden": inputs.local_hidden, "global_state": object()})

    def test_tensor_boundary_rejects_invalid_local_inputs(self):
        invalid = (
            (self.images.float() / 255, self.start, self.hidden),
            (self.images.unsqueeze(1), self.start, self.hidden),
            (self.images, self.start.float(), self.hidden),
            (self.images, torch.tensor([13, 2]), self.hidden),
            (self.images, self.start, torch.zeros((2, 64))),
            (self.images, self.start, torch.full((2, 256), float("inf"))),
            (self.images, self.start, self.hidden.to("meta")),
        )
        for rgb, action, hidden in invalid:
            with self.subTest(rgb_shape=rgb.shape, action=action), self.assertRaises(ValueError):
                self.actor(rgb, action, hidden)
        for batch in (0, -1, True, 1.0):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                self.actor.initial_hidden(batch)


if __name__ == "__main__":
    unittest.main()
