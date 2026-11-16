"""Complete current-policy prefixes, local BPTT, and behavior replay evidence."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_imagined_rollout_schema import make_rollout
from tests.test_model_sampling import identifier, make_store
from world_models.contracts import PREVIOUS_ACTION_START
from world_models.imagination.replay import AnchorPrefix, AnchorSampler
from world_models.imagination.versions import module_version
from world_models.mappo.recurrent import PrefixBank, replay_actor
from world_models.models.core import WorldState
from world_models.policies import LocalActor, PolicyConfig


class MappoRecurrentTests(unittest.TestCase):
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
        self.episode = identifier("p34-recurrent", "fit")
        self.dataset = make_store(Path(self.directory.name) / "real", [(self.episode, 6, False)])
        self.sampler = AnchorSampler([self.dataset], seed=34, deadline=40, batch_size=2)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(3403)
            self.actor = LocalActor(PolicyConfig(encoder_channels=(8, 8, 8, 8), embedding_size=16,
                                                hidden_size=16, critic_hidden_sizes=(16, 16)))
        self.rollout = self.build_rollout()
        self.bank = PrefixBank.from_rollout(self.rollout, self.sampler)

    def direct(self, rollout, prefixes, *, precision="float32"):
        """Independent player-by-player reset-to-destination reference."""
        logits = torch.zeros_like(rollout.actor_logits)
        memories = torch.zeros_like(rollout.actor_hidden)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bfloat16"):
            for row, prefix in enumerate(prefixes):
                for player in (0, 1):
                    hidden = self.actor.initial_hidden(1)
                    for step in range(prefix.ref.decision_index):
                        previous = (int(prefix.actions[step - 1, player]) if step else PREVIOUS_ACTION_START)
                        output = self.actor(torch.from_numpy(prefix.rgb[step, player].copy())[None],
                                            torch.tensor([previous]), hidden)
                        hidden = output.next_local_hidden
                    for step in range(int(rollout.lengths[row]) + 1):
                        output = self.actor(rollout.local_rgb[step, row, player][None],
                                            rollout.previous_actions[step, row, player][None], hidden)
                        hidden = output.next_local_hidden
                        logits[step, row, player] = output.logits[0]
                        memories[step, row, player] = hidden[0]
        return logits, memories

    def build_rollout(self, *, clocks=(0, 3), precision="float32", sampler=None):
        sampler = sampler or self.sampler
        rollout = make_rollout()
        references = [sampler.reference(0, self.episode, clock) for clock in clocks]
        prefixes = [sampler.read_prefix(reference) for reference in references]
        rgb, previous = rollout.local_rgb.clone(), rollout.previous_actions.clone()
        for row, prefix in enumerate(prefixes):
            rgb[0, row] = torch.from_numpy(prefix.rgb[-1].copy())
            previous[0, row] = (torch.from_numpy(prefix.actions[-1].copy()) if len(prefix.actions)
                                else torch.full((2,), PREVIOUS_ACTION_START, dtype=torch.long))
        source = torch.tensor(clocks)[None] + torch.arange(rollout.steps + 1)[:, None]
        source = torch.where(rollout.state_valid, source, 0)
        world = WorldState(rollout.world.memory, rollout.world.stochastic, rollout.world.logits, source)
        metadata = {**rollout.metadata, "deadline": sampler.deadline, "precision": precision,
                    "cohort_identity": sampler.cohort_identity,
                    "anchor_references": [asdict(reference) for reference in references],
                    "actor_config": json.loads(json.dumps(asdict(self.actor.config))),
                    "actor_version": module_version(self.actor)}
        rollout = replace(rollout, world=world, local_rgb=rgb, previous_actions=previous, metadata=metadata,
                          actor_hidden=torch.zeros(rollout.steps + 1, 2, 2, self.actor.config.hidden_size))
        logits, memories = self.direct(rollout, prefixes, precision=precision)
        log_probs = torch.zeros_like(rollout.log_probs)
        log_probs[rollout.valid] = logits[:-1][rollout.valid].log_softmax(-1).gather(
            -1, rollout.actions[rollout.valid][..., None]).squeeze(-1)
        return replace(rollout, actor_logits=logits, actor_hidden=memories, log_probs=log_probs)

    def test_full_prefix_replays_frozen_behavior_with_individual_ratios_one(self):
        logits = self.bank.replay(self.actor, self.rollout, [0, 1])
        self.assertEqual(logits.shape, (3, 2, 2, 12))
        active = self.rollout.valid
        torch.testing.assert_close(logits[active], self.rollout.actor_logits[:-1][active], rtol=1e-5, atol=1e-6)
        log_probs = logits[active].log_softmax(-1).gather(-1, self.rollout.actions[active][..., None]).squeeze(-1)
        torch.testing.assert_close((log_probs - self.rollout.log_probs[active]).exp(),
                                   torch.ones_like(log_probs), rtol=1e-5, atol=1e-6)
        reordered = replay_actor(self.actor, self.rollout, self.bank, torch.tensor([1, 0]))
        torch.testing.assert_close(reordered, logits[:, [1, 0]], rtol=1e-5, atol=1e-6)
        for row in (0, 1):
            single = self.bank.replay(self.actor, self.rollout, [row])
            torch.testing.assert_close(single[:, 0], logits[:, row], rtol=1e-5, atol=1e-6)

    def test_prefix_has_no_graph_and_anchor_is_consumed_once_with_full_sequence_bptt(self):
        calls = []

        def record(_module, args, output):
            calls.append((torch.is_grad_enabled(), args[0].shape[0], args[2], output.next_local_hidden))
            if output.next_local_hidden.requires_grad:
                output.next_local_hidden.retain_grad()

        handle = self.actor.register_forward_hook(record)
        try:
            logits = self.bank.replay(self.actor, self.rollout, [0, 1])
            logits[2, 1, 0, 0].backward()
        finally:
            handle.remove()
        prefix_calls = [call for call in calls if not call[0]]
        active_calls = [call for call in calls if call[0]]
        self.assertEqual(sum(call[1] for call in prefix_calls), 2 * 3)
        self.assertEqual(sum(call[1] for call in active_calls), 2 * int(self.rollout.valid.sum()))
        self.assertTrue(all(not call[3].requires_grad and call[3].grad_fn is None for call in prefix_calls))
        self.assertFalse(active_calls[0][2].requires_grad)
        self.assertGreater(active_calls[0][3].grad[2].abs().sum().item(), 0)
        self.assertGreater(active_calls[1][3].grad[2].abs().sum().item(), 0)
        self.assertGreater(self.actor.encoder[0].weight.grad.abs().sum().item(), 0)

    def test_current_weights_reconstruct_prefix_and_stored_memories_are_ignored(self):
        before = self.bank.replay(self.actor, self.rollout, [0, 1]).detach()
        with torch.no_grad():
            self.actor.gru.bias_ih.add_(.5)
        self.rollout.actor_hidden[self.rollout.state_valid] = 999
        actual = self.bank.replay(self.actor, self.rollout, [0, 1])
        expected, _ = self.direct(self.rollout, self.bank.prefixes)
        torch.testing.assert_close(actual[self.rollout.valid], expected[:-1][self.rollout.valid], rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(before[self.rollout.valid], actual[self.rollout.valid]))

    def test_padding_skips_encoding_and_keeps_zero_logits(self):
        rows = []
        handle = self.actor.encoder.register_forward_pre_hook(lambda _module, args: rows.append(args[0].shape[0]))
        try:
            logits = self.bank.replay(self.actor, self.rollout, [0, 1])
        finally:
            handle.remove()
        self.assertEqual(rows, [2, 2, 2, 4, 4, 2])
        self.assertEqual(logits[2, 0].count_nonzero().item(), 0)

    def test_own_history_locality_and_reset_are_preserved(self):
        original = self.bank.replay(self.actor, self.rollout, [0, 1]).detach()
        prefix = self.bank.prefixes[1]
        changed = prefix.rgb.copy()
        changed[:-1, 1] = 255 - changed[:-1, 1]
        different = AnchorPrefix(prefix.ref, changed, prefix.actions, prefix.decision_indices)
        bank = PrefixBank((self.bank.prefixes[0], different), self.bank.cohort_identity, "relay", 40)
        output = bank.replay(self.actor, self.rollout, [0, 1])
        torch.testing.assert_close(output[:, 0], original[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(output[:, 1, 0], original[:, 1, 0], rtol=0, atol=0)
        self.assertFalse(torch.allclose(output[:, 1, 1], original[:, 1, 1], rtol=1e-6, atol=1e-7))
        with torch.no_grad():
            reset = self.actor(self.rollout.local_rgb[0, 0], torch.full((2,), PREVIOUS_ACTION_START),
                               torch.full((2, self.actor.config.hidden_size), 500.)).logits
        torch.testing.assert_close(output[0, 0], reset, rtol=1e-5, atol=1e-6)

    def test_long_prefix_is_complete_and_never_uses_a_fixed_burnin_cap(self):
        dataset = make_store(Path(self.directory.name) / "long", [(self.episode, 21, False)])
        sampler = AnchorSampler([dataset], seed=9, deadline=40)
        rollout = self.build_rollout(clocks=(18, 19), sampler=sampler)
        bank = PrefixBank.from_rollout(rollout, sampler)
        count = []
        handle = self.actor.register_forward_pre_hook(
            lambda _module, args: count.append((torch.is_grad_enabled(), args[0].shape[0])))
        try:
            logits = bank.replay(self.actor, rollout, [0, 1])
        finally:
            handle.remove()
        self.assertEqual(sum(size for grad, size in count if not grad), 2 * (18 + 19))
        torch.testing.assert_close(logits[rollout.valid], rollout.actor_logits[:-1][rollout.valid], rtol=1e-5, atol=1e-6)

    def test_prefix_bank_owns_readonly_data_and_deduplicates_references_without_rng_draws(self):
        rollout = self.build_rollout(clocks=(3, 3))
        before = deepcopy(self.sampler.state_dict())
        with patch.object(self.sampler, "read_prefix", wraps=self.sampler.read_prefix) as reader:
            bank = PrefixBank.from_rollout(rollout, self.sampler)
        self.assertEqual(reader.call_count, 1)
        self.assertIs(bank.prefixes[0], bank.prefixes[1])
        self.assertEqual(before, self.sampler.state_dict())
        external = self.sampler.read_prefix(bank.references[0])
        copied = PrefixBank((external, external), bank.cohort_identity, "relay", 40)
        external.rgb.flags.writeable = True
        external.rgb.fill(255)
        self.assertFalse(np.array_equal(copied.prefixes[0].rgb, external.rgb))
        self.assertFalse(copied.prefixes[0].rgb.flags.writeable)

    def test_bank_rejects_foreign_cohort_refs_images_and_previous_actions(self):
        other = make_store(Path(self.directory.name) / "foreign", [(identifier("foreign", "fit"), 6, False)])
        with self.assertRaisesRegex(ValueError, "cohort"):
            PrefixBank.from_rollout(self.rollout, AnchorSampler([other], seed=0, deadline=40))
        bad_rgb = self.rollout.local_rgb.clone()
        bad_rgb[0, 1, 0, 0, 0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, "anchor RGB"):
            self.bank.replay(self.actor, replace(self.rollout, local_rgb=bad_rgb), [0])
        bad_previous = self.rollout.previous_actions.clone()
        bad_previous[0, 1, 0] = (bad_previous[0, 1, 0] + 1) % 12
        with self.assertRaisesRegex(ValueError, "previous actions"):
            PrefixBank.from_rollout(replace(self.rollout, previous_actions=bad_previous), self.sampler)
        metadata = deepcopy(self.rollout.metadata)
        metadata["anchor_references"][1]["dataset_identity"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "cohort|identity"):
            PrefixBank.from_rollout(replace(self.rollout, metadata=metadata), self.sampler)

    def test_bfloat16_replay_matches_behavior_and_retains_fp32_likelihood_logits(self):
        rollout = self.build_rollout(precision="bfloat16")
        bank = PrefixBank.from_rollout(rollout, self.sampler)
        logits = bank.replay(self.actor, rollout, [0, 1], precision="bfloat16")
        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(logits[rollout.valid], rollout.actor_logits[:-1][rollout.valid], rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.isfinite(logits).all())
        logits[rollout.valid].square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in self.actor.parameters()))

    def test_peaked_actor_bfloat16_context_replays_unequal_long_prefixes_across_batches(self):
        with torch.no_grad():
            self.actor.action_head[-1].weight.mul_(12)
            self.actor.action_head[-1].bias.copy_(torch.linspace(-3, 3, 12))
        dataset = make_store(Path(self.directory.name) / "precision", [(self.episode, 24, False)])
        sampler = AnchorSampler([dataset], seed=9, deadline=40)
        rollout = self.build_rollout(clocks=(7, 22), sampler=sampler, precision="bfloat16")
        bank = PrefixBank.from_rollout(rollout, sampler)
        for indices in ([0, 1], [1, 0], [0], [1]):
            with self.subTest(indices=indices):
                logits = bank.replay(self.actor, rollout, indices, precision="bfloat16")
                valid = rollout.valid[:, indices]
                logs = logits[valid].log_softmax(-1).gather(
                    -1, rollout.actions[:, indices][valid][..., None]).squeeze(-1)
                torch.testing.assert_close(logs, rollout.log_probs[:, indices][valid], rtol=0, atol=2e-5)
                torch.testing.assert_close((logs - rollout.log_probs[:, indices][valid]).exp(),
                                           torch.ones_like(logs), rtol=0, atol=2e-5)

    def test_modes_global_rng_and_behavior_data_are_preserved_even_on_failure(self):
        self.actor.train()
        self.actor.encoder[1].eval()
        modes = tuple(module.training for module in self.actor.modules())
        rng = torch.random.get_rng_state().clone()
        stored = {name: getattr(self.rollout, name).clone()
                  for name in ("log_probs", "values", "actor_hidden", "actor_logits", "local_rgb")}
        self.bank.replay(self.actor, self.rollout, [0, 1])
        self.assertEqual(modes, tuple(module.training for module in self.actor.modules()))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, before in stored.items():
            self.assertTrue(torch.equal(before, getattr(self.rollout, name)))
        with patch.object(self.actor, "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                self.bank.replay(self.actor, self.rollout, [0, 1])
        self.assertEqual(modes, tuple(module.training for module in self.actor.modules()))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_outer_inference_mode_still_returns_ordinary_trainable_sequence(self):
        with torch.inference_mode():
            logits = self.bank.replay(self.actor, self.rollout, [0, 1])
        self.assertFalse(logits.is_inference())
        self.assertTrue(logits.requires_grad)
        logits[self.rollout.valid].square().mean().backward()
        self.assertGreater(self.actor.gru.weight_hh.grad.abs().sum().item(), 0)

    def test_explicit_no_grad_behavior_audit_has_no_sequence_graph(self):
        with torch.no_grad():
            logits = self.bank.replay(self.actor, self.rollout, [0, 1])
        self.assertFalse(logits.requires_grad)
        self.assertIsNone(logits.grad_fn)
        self.assertFalse(logits.is_inference())

    def test_invalid_minibatches_precision_and_actor_architecture_fail(self):
        for indices in ([], [0, 0], [-1], [2], [True], "0", torch.tensor([0.])):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.bank.replay(self.actor, self.rollout, indices)
        with self.assertRaisesRegex(ValueError, "precision"):
            self.bank.replay(self.actor, self.rollout, [0], precision="bfloat16")
        different = LocalActor(PolicyConfig(encoder_channels=(8, 8, 8, 8), embedding_size=16, hidden_size=24))
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.bank.replay(different, self.rollout, [0])


if __name__ == "__main__":
    unittest.main()
