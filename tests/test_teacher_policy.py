"""P4.1 teacher current-global head and strictly local history invariants."""
from __future__ import annotations

import inspect
import unittest

import torch
from torch import nn

from world_models.policies.actor import LocalActor
from world_models.policies.config import PolicyConfig
from world_models.policies.teacher import GlobalTeacher
from world_models.imagination.versions import module_version


class GlobalTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.config = PolicyConfig(encoder_channels=(8, 8, 8, 8), embedding_size=16,
                                   hidden_size=16, world_tokens=6, world_width=16,
                                   critic_hidden_sizes=(16, 16))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(4101)
            self.teacher = GlobalTeacher(self.config)
        self.generator = torch.Generator().manual_seed(4102)
        self.images = torch.randint(0, 256, (2, 3, 96, 128), dtype=torch.uint8,
                                    generator=self.generator)
        self.actions = torch.tensor([12, 3])
        self.hidden = self.teacher.initial_hidden(2)
        self.tokens = torch.randn((2, 6, 16), generator=self.generator)
        self.players = torch.tensor([0, 1])

    def test_production_architecture_and_no_deployment_interface(self):
        teacher = GlobalTeacher()
        self.assertFalse(isinstance(teacher, LocalActor))
        self.assertFalse(hasattr(teacher, "step"))
        self.assertEqual(teacher.config, PolicyConfig())
        self.assertEqual(teacher.cross_attention.num_heads, 4)
        self.assertEqual(teacher.cross_attention.embed_dim, 256)
        self.assertEqual(teacher.cross_attention.dropout, 0)
        self.assertEqual(teacher.gru.input_size, 269)
        self.assertEqual(teacher.gru.hidden_size, 256)
        self.assertEqual(teacher.projection[1].out_features, 256)
        self.assertEqual([x.out_channels for x in teacher.encoder if isinstance(x, nn.Conv2d)],
                         [32, 64, 128, 256])
        self.assertEqual(set(inspect.signature(teacher.memory_step).parameters),
                         {"local_rgb", "previous_action", "local_hidden"})
        forbidden = (nn.modules.batchnorm._BatchNorm, nn.Dropout, nn.Dropout2d)
        self.assertFalse(any(isinstance(x, forbidden) for x in teacher.modules()))

    def test_global_tokens_and_identity_change_action_but_never_memory(self):
        baseline = self.teacher(self.images, self.actions, self.hidden, self.tokens, self.players)
        for tokens, players in ((self.tokens + 2, self.players),
                                (self.tokens, 1 - self.players)):
            changed = self.teacher(self.images, self.actions, self.hidden, tokens, players)
            self.assertTrue(torch.equal(baseline.next_local_hidden, changed.next_local_hidden))
            self.assertFalse(torch.equal(baseline.logits, changed.logits))
        hidden = baseline.next_local_hidden
        before = hidden.clone()
        output = self.teacher.action_from_memory(hidden, self.tokens, self.players)
        self.assertIs(output.next_local_hidden, hidden)
        self.assertTrue(torch.equal(hidden, before))
        self.assertEqual(tuple(output.logits.shape), (2, 12))
        self.assertTrue(torch.all(output.distribution.probs > 0))

    def test_shared_tokens_cannot_enter_history_or_carry_gradients_into_world(self):
        world = nn.Linear(3, 16)
        tokens = world(torch.randn((2, 6, 3), generator=self.generator))
        tokens.retain_grad()
        output = self.teacher(self.images, self.actions, self.hidden, tokens, self.players)
        output.log_prob(torch.tensor([2, 7])).mean().backward()
        self.assertIsNone(tokens.grad)
        self.assertTrue(all(p.grad is None for p in world.parameters()))
        for name, parameter in self.teacher.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for module in (self.teacher.encoder[0], self.teacher.gru,
                       self.teacher.query, self.teacher.cross_attention, self.teacher.action_head):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)

    def test_memory_prefix_and_anchor_consumed_exactly_once(self):
        images = (self.images, 255 - self.images, self.images.roll(1, -1))
        actions = (torch.tensor([12, 12]), torch.tensor([1, 7]), torch.tensor([3, 2]))
        hidden = self.hidden
        calls = []
        hook = self.teacher.encoder.register_forward_pre_hook(
            lambda _module, _args: calls.append(True))
        try:
            for rgb, previous in zip(images[:2], actions[:2]):
                hidden = self.teacher.memory_step(rgb, previous, hidden)
            before_anchor = hidden
            hidden = self.teacher.memory_step(images[2], actions[2], hidden)
            split = self.teacher.action_from_memory(hidden, self.tokens, self.players)
            self.assertEqual(len(calls), 3)
        finally:
            hook.remove()
        full = self.teacher(images[2], actions[2], before_anchor, self.tokens, self.players)
        self.assertTrue(torch.equal(split.logits, full.logits))
        self.assertTrue(torch.equal(split.next_local_hidden, full.next_local_hidden))

    def test_start_resets_only_selected_row_and_blocks_past_gradient(self):
        old = torch.randn((2, 16), generator=self.generator, requires_grad=True)
        output = self.teacher(self.images, self.actions, old, self.tokens, self.players)
        fresh = self.teacher(self.images, self.actions, self.hidden, self.tokens, self.players)
        self.assertTrue(torch.equal(output.next_local_hidden[0], fresh.next_local_hidden[0]))
        self.assertFalse(torch.equal(output.next_local_hidden[1], fresh.next_local_hidden[1]))
        output.logits.square().sum().backward()
        self.assertTrue(torch.equal(old.grad[0], torch.zeros_like(old.grad[0])))
        self.assertGreater(old.grad[1].abs().sum().item(), 0)

    def test_own_inputs_affect_memory_and_output_without_mutating_inputs(self):
        baseline = self.teacher(self.images, self.actions, self.hidden, self.tokens, self.players)
        for images, actions, hidden in ((255 - self.images, self.actions, self.hidden),
                                        (self.images, torch.tensor([1, 7]), self.hidden),
                                        (self.images, self.actions, self.hidden + 1)):
            saved = hidden.clone()
            changed = self.teacher(images, actions, hidden, self.tokens, self.players)
            self.assertFalse(torch.equal(baseline.next_local_hidden, changed.next_local_hidden))
            self.assertFalse(torch.equal(baseline.logits, changed.logits))
            self.assertTrue(torch.equal(hidden, saved))
        first, second = self.teacher.initial_hidden(1), self.teacher.initial_hidden(1)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_parameters_independent_from_student_and_other_teacher(self):
        student = LocalActor(self.config)
        other = GlobalTeacher(self.config)
        teacher_pointers = {p.data_ptr() for p in self.teacher.parameters()}
        self.assertFalse(teacher_pointers.intersection(p.data_ptr() for p in student.parameters()))
        self.assertFalse(teacher_pointers.intersection(p.data_ptr() for p in other.parameters()))
        student_before = {k: v.clone() for k, v in student.state_dict().items()}
        self.teacher(self.images, self.actions, self.hidden, self.tokens,
                     self.players).logits.square().sum().backward()
        self.assertTrue(all(p.grad is None for p in student.parameters()))
        self.assertTrue(all(torch.equal(v, student_before[k]) for k, v in student.state_dict().items()))

    def test_outer_autocast_leaves_complete_teacher_and_gradients_fp32(self):
        initial = self.hidden.bfloat16()
        tokens = self.tokens.bfloat16()
        expected = self.teacher(self.images, self.actions, initial, tokens, self.players)
        expected.log_prob(torch.tensor([2, 7])).mean().backward()
        gradients = {name: p.grad.clone() for name, p in self.teacher.named_parameters()}
        self.teacher.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = self.teacher(self.images, self.actions, initial, tokens, self.players)
            self.assertTrue(torch.is_autocast_enabled("cpu"))
        self.assertEqual(actual.logits.dtype, torch.float32)
        self.assertEqual(actual.next_local_hidden.dtype, torch.float32)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        torch.testing.assert_close(actual.next_local_hidden, expected.next_local_hidden, rtol=0, atol=0)
        actual.log_prob(torch.tensor([2, 7])).mean().backward()
        for name, p in self.teacher.named_parameters():
            torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)

    def test_train_eval_batch_composition_and_replay_likelihood(self):
        self.teacher.train()
        expected = self.teacher(self.images, self.actions, self.hidden, self.tokens, self.players)
        self.teacher.eval()
        actual = self.teacher(self.images, self.actions, self.hidden, self.tokens, self.players)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        single = self.teacher(self.images[:1], self.actions[:1], self.hidden[:1],
                              self.tokens[:1], self.players[:1])
        torch.testing.assert_close(single.logits, expected.logits[:1], rtol=1e-5, atol=1e-7)
        executed = torch.tensor([1, 10])
        ratio = (actual.log_prob(executed) - expected.log_prob(executed).detach()).exp()
        torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=0)

    def test_future_or_extra_inputs_are_not_accepted(self):
        with self.assertRaises(TypeError):
            self.teacher.memory_step(self.images, self.actions, self.hidden, self.tokens)
        with self.assertRaises(TypeError):
            self.teacher(self.images, self.actions, self.hidden, self.tokens,
                         self.players, future_tokens=self.tokens)
        with self.assertRaises(ValueError):
            self.teacher.action_from_memory(self.hidden, self.tokens.unsqueeze(0), self.players)

    def test_invalid_inputs_and_non_fp32_parameters_rejected(self):
        invalid_local = ((self.images.float(), self.actions, self.hidden),
                         (self.images[:0], self.actions[:0], self.hidden[:0]),
                         (self.images, self.actions.float(), self.hidden),
                         (self.images, torch.tensor([-1, 0]), self.hidden),
                         (self.images, torch.tensor([13, 0]), self.hidden),
                         (self.images, self.actions, self.hidden[:, :5]),
                         (self.images, self.actions, self.hidden + float("nan")))
        for args in invalid_local:
            with self.subTest(shapes=[x.shape for x in args]), self.assertRaises(ValueError):
                self.teacher.memory_step(*args)
        invalid_head = ((self.hidden[:0], self.tokens[:0], self.players[:0]),
                        (self.hidden, self.tokens.long(), self.players),
                        (self.hidden, self.tokens[:, :3], self.players),
                        (self.hidden, self.tokens + float("inf"), self.players),
                        (self.hidden, self.tokens, self.players.float()),
                        (self.hidden, self.tokens, torch.tensor([-1, 0])),
                        (self.hidden, self.tokens, torch.tensor([0, 2])),
                        (self.hidden, self.tokens, self.players[:, None]))
        for args in invalid_head:
            with self.subTest(shapes=[x.shape for x in args]), self.assertRaises(ValueError):
                self.teacher.action_from_memory(*args)
        self.teacher.bfloat16()
        with self.assertRaisesRegex(ValueError, "parameters must remain FP32"):
            self.teacher(self.images, self.actions, self.hidden, self.tokens, self.players)

    def test_invalid_configuration_and_initial_batch_rejected(self):
        with self.assertRaises(ValueError):
            GlobalTeacher({})
        for heads in (True, 0, -1, 3, 4.0):
            with self.subTest(heads=heads), self.assertRaises(ValueError):
                GlobalTeacher(self.config, heads)
        for batch in (True, 0, -1, 1.0):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                self.teacher.initial_hidden(batch)

    def test_attention_semantics_bound_into_versions_and_mutations_rejected(self):
        other = GlobalTeacher(self.config, attention_heads=2)
        other.load_state_dict(self.teacher.state_dict())
        self.assertNotEqual(module_version(self.teacher), module_version(other))
        self.teacher.cross_attention.num_heads = 2
        with self.assertRaisesRegex(ValueError, "attention configuration changed"):
            self.teacher(self.images, self.actions, self.hidden, self.tokens, self.players)


if __name__ == "__main__":
    unittest.main()
