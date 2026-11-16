"""Paired B/D policy construction and compatibility with the frozen model."""
from copy import deepcopy
from dataclasses import replace
import io
from pathlib import Path
import unittest

import torch

from world_models.infrastructure.config import resolve_config
from world_models.infrastructure.reproducibility import derive_seed
from world_models.models.world_model import SamplingStreams, build_model
from world_models.policies import (
    CentralizedCritic, LocalActor, PolicyConfig, build_actor_critic, imagined_rgb_to_uint8, raw_value_loss,
)


ROOT = Path(__file__).resolve().parents[1]


class LocalPolicyInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self, family="relay", variant="B", profile="pilot"):
        return resolve_config(family, variant, profile, ROOT / "configs").settings

    def test_all_profiles_and_variants_use_the_same_local_architecture(self):
        for family in ("relay", "cover"):
            for variant in ("B", "D"):
                for profile in ("pilot", "final"):
                    self.assertEqual(PolicyConfig.from_experiment(self.config(family, variant, profile)), PolicyConfig())

    def test_paired_initialization_matches_with_disjoint_parameter_storage(self):
        for family in ("relay", "cover"):
            with self.subTest(family=family):
                left = build_actor_critic(self.config(family, "B"), master_seed=0, variant="B")
                right = build_actor_critic(self.config(family, "D"), master_seed=0, variant="D")
                self.assertEqual(set(left.state_dict()), set(right.state_dict()))
                for name, value in left.state_dict().items():
                    self.assertTrue(torch.equal(value, right.state_dict()[name]), name)
                    self.assertNotEqual(value.data_ptr(), right.state_dict()[name].data_ptr(), name)
                actor_ptrs = {p.data_ptr() for p in left.actor.parameters()}
                self.assertTrue(actor_ptrs.isdisjoint(p.data_ptr() for p in left.critic.parameters()))

    def test_initialization_has_separate_component_namespaces_and_preserves_rng(self):
        before = torch.random.get_rng_state().clone()
        network = build_actor_critic(self.config(), master_seed=1, variant="B")
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derive_seed("relay", 1, "local_actor_initialization", "B"))
            actor = LocalActor()
            torch.random.default_generator.manual_seed(derive_seed("relay", 1, "local_critic_initialization", "B"))
            critic = CentralizedCritic()
        for actual, expected in ((network.actor, actor), (network.critic, critic)):
            for name, tensor in actual.state_dict().items():
                self.assertTrue(torch.equal(tensor, expected.state_dict()[name]), name)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_task_and_seed_namespaces_produce_different_weights(self):
        reference = build_actor_critic(self.config(), master_seed=0, variant="B")
        for family, seed in (("relay", 1), ("cover", 0)):
            other = build_actor_critic(self.config(family), master_seed=seed, variant="B")
            for component in ("actor", "critic"):
                self.assertFalse(torch.equal(next(getattr(reference, component).parameters()),
                                             next(getattr(other, component).parameters())))

    def test_config_changes_cannot_silently_enable_dropout_or_value_normalization(self):
        for section, field, value in (("actor", "dropout", 0.1), ("mappo", "normalize_values", True),
                                      ("critic", "actor_memories", False), ("critic", "remaining_time", False),
                                      ("environment", "actor_global_input", True),
                                      ("environment", "shared_actor_weights", False)):
            settings = deepcopy(self.config())
            settings[section][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_actor_critic(settings, master_seed=0, variant="B")
        with self.assertRaises(ValueError):
            build_actor_critic(self.config(), master_seed=0, variant="D")

    def test_invalid_dimensions_are_rejected(self):
        for change in ({"image_shape": (3, 128, 96)}, {"action_count": 13}, {"hidden_size": True},
                       {"embedding_size": 0}, {"world_tokens": -1}, {"world_width": 0},
                       {"encoder_channels": (32, 64, 128)}, {"encoder_channels": (32, 64, 128, 255)},
                       {"critic_hidden_sizes": (512,)}, {"value_huber_delta": float("nan")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(PolicyConfig(), **change)

    def test_state_dict_round_trip_preserves_recurrent_outputs_and_values(self):
        first = build_actor_critic(self.config(), master_seed=0, variant="B")
        second = build_actor_critic(self.config(), master_seed=1, variant="B")
        buffer = io.BytesIO()
        torch.save(first.state_dict(), buffer)
        buffer.seek(0)
        second.load_state_dict(torch.load(buffer, weights_only=True))
        images = torch.full((2, 3, 96, 128), 101, dtype=torch.uint8)
        previous = torch.tensor([12, 7])
        memory = first.actor.initial_hidden(2)
        with torch.no_grad():
            a, b = first.actor(images, previous, memory), second.actor(images, previous, memory)
            self.assertTrue(torch.equal(a.logits, b.logits))
            self.assertTrue(torch.equal(a.next_local_hidden, b.next_local_hidden))
            tokens = torch.ones(1, 48, 256)
            remaining = torch.tensor([0.75])
            self.assertTrue(torch.equal(first.critic(tokens, a.next_local_hidden[None], remaining),
                                        second.critic(tokens, b.next_local_hidden[None], remaining)))

    def test_actual_world_model_outputs_connect_without_parameter_or_gradient_sharing(self):
        settings = self.config()
        model = build_model(settings, master_seed=0, variant="B")
        policy = build_actor_critic(settings, master_seed=0, variant="B")
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        model_ptrs = {p.data_ptr() for p in model.parameters()}
        self.assertTrue(model_ptrs.isdisjoint(p.data_ptr() for p in policy.parameters()))
        images = torch.zeros(1, 2, 3, 96, 128, dtype=torch.uint8)
        # Intentionally keep model autograd enabled: policy boundaries must isolate it.
        state = model.initialize(images, streams=SamplingStreams.from_seed(11))
        decoded = model.reconstruct(state)
        local_rgb = imagined_rgb_to_uint8(decoded.rgb).flatten(0, 1)
        actor = policy.actor(local_rgb, torch.tensor([12, 12]), policy.actor.initial_hidden(2))
        value = policy.critic(decoded.representation, actor.next_local_hidden[None], torch.ones(1))
        loss = raw_value_loss(value, torch.tensor([0.9])) - actor.log_prob(torch.tensor([0, 1])).mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.actor.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
        optimizer.step()  # One synthetic fixture update, no research policy training.
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], tensor), name)


if __name__ == "__main__":
    unittest.main()
