"""Optimization settings stay paired and preserve the fixed study contract."""
from copy import deepcopy
from dataclasses import asdict, FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

from world_models.infrastructure.config import resolve_config
from world_models.mappo.config import MAPPOConfig


ROOT = Path(__file__).resolve().parents[1]


class MAPPOConfigTests(unittest.TestCase):
    def settings(self, family="relay", variant="B", profile="pilot"):
        return resolve_config(family, variant, profile, ROOT / "configs").settings

    def test_default_settings_match_every_registered_local_policy(self):
        for family in ("relay", "cover"):
            for variant in ("B", "D"):
                for profile in ("pilot", "final"):
                    self.assertEqual(MAPPOConfig.from_experiment(self.settings(family, variant, profile)),
                                     MAPPOConfig())

    def test_exact_registered_mapping_and_serialization(self):
        settings = deepcopy(self.settings())
        settings["mappo"].update(epochs=2, recurrent_minibatches=8, gamma=.9, gae_lambda=.8,
                                 clip_epsilon=.1, learning_rate=.0001, optimizer_epsilon=.00001,
                                 entropy_coefficient=.02, value_coefficient=.75,
                                 value_huber_delta=2., gradient_clip=.7)
        result = MAPPOConfig.from_experiment(settings)
        self.assertEqual((result.epochs, result.minibatches), (2, 8))
        self.assertEqual((result.actor_lr, result.critic_lr), (.0001, .0001))
        self.assertEqual((result.gamma, result.gae_lambda, result.clip_epsilon), (.9, .8, .1))
        self.assertEqual((result.entropy_coefficient, result.value_coefficient), (.02, .75))
        self.assertEqual((result.value_huber_delta, result.max_grad_norm), (2., .7))
        self.assertEqual(MAPPOConfig(**json.loads(json.dumps(asdict(result)))), result)
        with self.assertRaises(FrozenInstanceError):
            result.epochs = 5

    def test_rejects_bad_integers_and_numeric_settings(self):
        for field in ("epochs", "minibatches"):
            for value in (0, -1, True, 2.5):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(MAPPOConfig(), **{field: value})
        for field in ("gamma", "gae_lambda", "clip_epsilon", "actor_lr", "critic_lr", "adam_epsilon",
                      "entropy_coefficient", "value_coefficient", "value_huber_delta", "max_grad_norm"):
            for value in (float("nan"), float("inf"), True, "0.5", -1):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(MAPPOConfig(), **{field: value})
        for field in ("clip_epsilon", "actor_lr", "critic_lr", "adam_epsilon",
                      "value_coefficient", "value_huber_delta", "max_grad_norm"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(MAPPOConfig(), **{field: 0})
        for field in ("gamma", "gae_lambda", "clip_epsilon"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(MAPPOConfig(), **{field: 1.1})
        with self.assertRaises(ValueError):
            replace(MAPPOConfig(), normalize_advantages=1)

    def test_mathematical_gamma_lambda_endpoints_are_available_for_unit_fixtures(self):
        config = MAPPOConfig(gamma=1, gae_lambda=0, entropy_coefficient=0,
                             normalize_advantages=False)
        self.assertEqual((config.gamma, config.gae_lambda, config.entropy_coefficient), (1, 0, 0))

    def test_experiment_cannot_change_raw_value_or_deterministic_replay_contract(self):
        for section, key, value in (("actor", "dropout", .1), ("mappo", "normalize_values", True),
                                    ("mappo", "differentiate_world_model", True),
                                    ("mappo", "frozen_rollout_snapshots", False),
                                    ("mappo", "normalize_actor_advantages", False)):
            settings = deepcopy(self.settings())
            settings[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                MAPPOConfig.from_experiment(settings)

    def test_experiment_validates_full_config_and_minibatch_divisibility(self):
        settings = deepcopy(self.settings())
        settings["mappo"]["recurrent_minibatches"] = 3
        with self.assertRaises(ValueError):
            MAPPOConfig.from_experiment(settings)
        settings = deepcopy(self.settings())
        settings["environment"]["actor_global_input"] = True
        with self.assertRaises(ValueError):
            MAPPOConfig.from_experiment(settings)

    def test_updater_factory_resolves_task_variant_rng_and_precision_without_updates(self):
        import torch

        from world_models.imagination.versions import module_version
        from world_models.infrastructure.reproducibility import derive_seed
        from world_models.mappo.updater import RecurrentMAPPO
        from world_models.models.world_model import build_model
        from world_models.policies.initialization import build_actor_critic

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            settings = self.settings()
            global_rng = torch.random.get_rng_state().clone()
            model = build_model(settings, master_seed=7, variant="B")
            network = build_actor_critic(settings, master_seed=7, variant="B")
            versions = tuple(module_version(part) for part in (model, network.actor, network.critic))
            left = RecurrentMAPPO.from_experiment(model, network.actor, network.critic, settings, master_seed=7)
            right = RecurrentMAPPO.from_experiment(model, network.actor, network.critic, self.settings(variant="D"),
                                                   master_seed=7, precision="float32")
            self.assertEqual(left.seed, derive_seed("relay", 7, "local_policy_updates", "B"))
            self.assertEqual(right.seed, derive_seed("relay", 7, "local_policy_updates", "D"))
            self.assertNotEqual(left.seed, right.seed)
            self.assertEqual(left.precision, settings["resources"]["mixed_precision"])
            self.assertEqual(right.precision, "float32")
            for updater in (left, right):
                self.assertEqual(updater.config, MAPPOConfig())
                self.assertEqual((updater.cycles_completed, updater.actor_optimizer_steps,
                                  updater.critic_optimizer_steps), (0, 0, 0))
                self.assertEqual(updater.actor_optimizer.state_dict()["state"], {})
                self.assertEqual(updater.critic_optimizer.state_dict()["state"], {})
            self.assertEqual(versions, tuple(module_version(part) for part in (model, network.actor, network.critic)))
            self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
