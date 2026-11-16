"""Raw team values, fixed-feature gradients, and valid joint-sample reduction."""
import unittest

import torch
from torch import nn

from world_models.models.world_model import SamplingStreams, SharedWorldModel
from world_models.policies.critic import CentralizedCritic, raw_value_loss


class LocalCriticTests(unittest.TestCase):
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
            self.critic = CentralizedCritic()
        stream = torch.Generator().manual_seed(19)
        self.tokens = torch.randn((2, 48, 256), generator=stream)
        self.memories = torch.randn((2, 2, 256), generator=stream)
        self.clock = torch.tensor([0.75, 0.25])

    def test_architecture_and_team_output(self):
        layers = [layer for layer in self.critic.modules() if isinstance(layer, nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in layers],
                         [(769, 512), (512, 512), (512, 1)])
        values = self.critic(self.tokens, self.memories, self.clock)
        self.assertEqual(values.shape, (2,))
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(all(name.startswith("mlp.") for name, _ in self.critic.named_parameters()))

    def test_pools_world_state_and_preserves_both_memory_slots_and_clock(self):
        # An explicit probe makes input dependence independent of random weights.
        with torch.no_grad():
            for parameter in self.critic.parameters():
                parameter.zero_()
            self.critic.mlp[0].weight[0, [0, 256, 512, 768]] = torch.tensor([1., 2., 3., 4.])
            self.critic.mlp[2].weight[0, 0] = 1.
            self.critic.mlp[4].weight[0, 0] = 1.
        baseline = self.critic(self.tokens, self.memories, self.clock)
        for source in ("tokens", "first_memory", "second_memory", "clock"):
            tokens, memories, clock = self.tokens.clone(), self.memories.clone(), self.clock.clone()
            if source == "tokens":
                tokens[:, :, 0] += 1
            elif source == "first_memory":
                memories[:, 0, 0] += 1
            elif source == "second_memory":
                memories[:, 1, 0] += 1
            else:
                clock += 0.1
            with self.subTest(source=source):
                self.assertFalse(torch.equal(baseline, self.critic(tokens, memories, clock)))
        permutation = torch.randperm(48, generator=torch.Generator().manual_seed(8))
        torch.testing.assert_close(baseline, self.critic(self.tokens[:, permutation], self.memories, self.clock))
        self.assertFalse(torch.equal(baseline, self.critic(self.tokens, self.memories.flip(1), self.clock)))

    def test_raw_values_are_unbounded_and_batch_independent(self):
        batch_values = self.critic(self.tokens, self.memories, self.clock)
        single_value = self.critic(self.tokens[:1], self.memories[:1], self.clock[:1])
        torch.testing.assert_close(batch_values[:1], single_value)
        with torch.no_grad():
            self.critic.mlp[-1].weight.zero_()
            self.critic.mlp[-1].bias.fill_(123.)
        expected = torch.tensor([123., 123.])
        torch.testing.assert_close(self.critic(self.tokens, self.memories, self.clock), expected)
        single = self.critic(self.tokens[:1], self.memories[:1], self.clock[:1])
        torch.testing.assert_close(single, expected[:1])

    def test_detaches_all_inputs_and_raw_targets(self):
        tokens = self.tokens.requires_grad_()
        memories = self.memories.requires_grad_()
        clock = self.clock.requires_grad_()
        targets = torch.tensor([1., -2.], requires_grad=True)
        loss = raw_value_loss(self.critic(tokens, memories, clock), targets)
        loss.backward()
        for value in (tokens, memories, clock, targets):
            self.assertIsNone(value.grad)
        gradients = [parameter.grad for parameter in self.critic.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.)

    def test_optimizer_step_cannot_update_actual_world_model_or_memory_producer(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(381)
            model = SharedWorldModel()
            memory_producer = nn.Linear(3, 512)
        images = torch.zeros((1, 2, 3, 96, 128), dtype=torch.uint8)
        state = model.initialize(images, streams=SamplingStreams.from_seed(7))
        tokens = model.core.representation(state)
        memories = memory_producer(torch.ones(1, 3)).reshape(1, 2, 256)
        self.assertTrue(tokens.requires_grad)
        self.assertTrue(memories.requires_grad)
        model_versions = [parameter._version for parameter in model.parameters()]
        memory_before = {name: value.detach().clone() for name, value in memory_producer.named_parameters()}
        critic_before = self.critic.mlp[-1].weight.detach().clone()
        optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.01)
        raw_value_loss(self.critic(tokens, memories, torch.ones(1)), torch.tensor([2.])).backward()
        optimizer.step()
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(model_versions, [parameter._version for parameter in model.parameters()])
        for name, parameter in memory_producer.named_parameters():
            self.assertIsNone(parameter.grad)
            self.assertTrue(torch.equal(parameter, memory_before[name]))
        self.assertFalse(torch.equal(critic_before, self.critic.mlp[-1].weight))

    def test_low_precision_features_and_autocast_keep_raw_values_float32(self):
        expected = self.critic(self.tokens.bfloat16(), self.memories, self.clock)
        self.assertEqual(expected.dtype, torch.float32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = self.critic(self.tokens.bfloat16(), self.memories, self.clock)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)

    def test_rejects_malformed_critic_inputs(self):
        cases = [
            (self.tokens[:, :-1], self.memories, self.clock),
            (self.tokens.long(), self.memories, self.clock),
            (self.tokens, self.memories[:, :1], self.clock),
            (self.tokens, self.memories.long(), self.clock),
            (self.tokens, self.memories, self.clock[:, None]),
            (self.tokens, self.memories, self.clock.long()),
            (self.tokens, self.memories, torch.tensor([-0.1, 1.])),
            (self.tokens, self.memories, torch.tensor([0., 1.1])),
            (self.tokens * float("nan"), self.memories, self.clock),
            (self.tokens, self.memories * float("inf"), self.clock),
            (self.tokens, self.memories, self.clock * float("nan")),
            (self.tokens[:0], self.memories[:0], self.clock[:0]),
        ]
        for index, arguments in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.critic(*arguments)
        for arguments in ((self.tokens.to("meta"), self.memories, self.clock),
                          (self.tokens, self.memories.to("meta"), self.clock),
                          (self.tokens, self.memories, self.clock.to("meta"))):
            with self.assertRaisesRegex(ValueError, "critic device"):
                self.critic(*arguments)


class RawValueLossTests(unittest.TestCase):
    def test_known_huber_values_and_gradients_are_averaged_once_per_joint_sample(self):
        values = torch.tensor([0., 0.5, 2., -2.], requires_grad=True)
        targets = torch.zeros(4, requires_grad=True)
        loss = raw_value_loss(values, targets)
        self.assertAlmostEqual(float(loss.detach()), (0. + 0.125 + 1.5 + 1.5) / 4)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([0., 0.125, 0.25, -0.25]))
        self.assertIsNone(targets.grad)
        self.assertAlmostEqual(float(raw_value_loss(torch.tensor([3.]), torch.tensor([0.]), delta=2.)), 4.)

    def test_sequence_mask_selects_before_arithmetic_and_ignores_nan_padding(self):
        values = torch.tensor([[0.5, float("nan")], [2., float("nan")]], requires_grad=True)
        targets = torch.tensor([[0., float("nan")], [0., float("nan")]], requires_grad=True)
        valid = torch.tensor([[True, False], [True, False]])
        loss = raw_value_loss(values, targets, valid=valid)
        self.assertAlmostEqual(float(loss.detach()), (0.125 + 1.5) / 2)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([[0.25, 0.], [0.5, 0.]]))
        self.assertIsNone(targets.grad)

    def test_raw_target_scale_is_preserved_and_bfloat16_math_is_promoted(self):
        loss = raw_value_loss(torch.tensor([2.], dtype=torch.bfloat16),
                              torch.tensor([102.], dtype=torch.bfloat16))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(float(loss), 99.5)
        double_loss = raw_value_loss(torch.tensor([2.], dtype=torch.float64), torch.tensor([3.]))
        self.assertEqual(double_loss.dtype, torch.float64)

    def test_invalid_shapes_masks_values_or_delta_are_rejected(self):
        values, targets = torch.zeros(2), torch.ones(2)
        for invalid_delta in (0, -1, float("inf"), float("nan"), True, "1"):
            with self.subTest(delta=invalid_delta), self.assertRaises(ValueError):
                raw_value_loss(values, targets, delta=invalid_delta)
        cases = [
            (values, targets[:, None], None),
            (values.long(), targets, None),
            (values, targets.long(), None),
            (values[:0], targets[:0], None),
            (values, targets, torch.zeros(2, dtype=torch.bool)),
            (values, targets, torch.ones(2)),
            (values, targets, torch.ones(2, 1, dtype=torch.bool)),
            (values * float("nan"), targets, None),
            (values, targets * float("inf"), None),
            (values, targets.to("meta"), None),
            (values, targets, torch.ones(2, dtype=torch.bool, device="meta")),
        ]
        for index, (predictions, reference, mask) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                raw_value_loss(predictions, reference, valid=mask)


if __name__ == "__main__":
    unittest.main()
