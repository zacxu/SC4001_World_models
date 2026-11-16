"""Registered endpoint budgets and honest boundaries of the integrated D path."""
import unittest

from world_models.infrastructure.reproducibility import canonical_hash
from world_models_campaign.protocol import load_protocol, plan, protocol_identity, stage_targets


class CampaignProtocolTests(unittest.TestCase):
    def test_active_recipe_has_exact_one_time_transfer_and_matched_local_budget(self):
        p = load_protocol()
        self.assertEqual(p["master_seeds"], [0, 1])
        self.assertEqual(p["initial_transitions"], 50000)
        self.assertEqual(p["initial_model_updates"], 4500)
        self.assertEqual(stage_targets(), {"teacher": 25, "buffer": 128,
                                          "distillation": 100, "local_cycle": 20})
        self.assertEqual((p["starts"], p["horizon"], p["epochs"], p["minibatches"]), (256, 15, 4, 4))
        self.assertEqual(p["teacher_kl_weight"], 0.0)
        self.assertEqual(p["teacher_refresh_cycles"], 0)
        self.assertIs(p["guidance_decay_enabled"], False)
        self.assertEqual(p["local_ppo_optimizer_initialization"], "fresh_after_distillation")
        self.assertIs(p["model_and_normalization_frozen"], True)

    def test_plan_accounts_for_teacher_cost_separately_from_local_opportunities(self):
        preview = plan(0)
        self.assertEqual(preview["learner_units"], 273)
        self.assertEqual(preview["total_units"], 274)
        self.assertEqual(preview["teacher_actor_optimizer_steps"], 400)
        self.assertEqual(preview["teacher_critic_optimizer_steps"], 400)
        self.assertEqual(preview["local_actor_optimizer_steps"], 320)
        self.assertEqual(preview["local_critic_optimizer_steps"], 320)
        self.assertEqual(preview["teacher_imagined_joint_step_cap"], 96000)
        self.assertEqual(preview["local_imagined_joint_step_cap"], 76800)
        self.assertEqual(preview["distillation_imagined_joint_step_cap"], 1920)

    def test_retrieved_protocol_and_plan_do_not_mutate_the_registration(self):
        identity = protocol_identity()
        copy = load_protocol()
        copy["local_cycles"] = 999
        copy["master_seeds"].append(5)
        preview = plan(1)
        preview["protocol"]["teacher_cycles"] = 999
        self.assertEqual(load_protocol()["local_cycles"], 20)
        self.assertEqual(load_protocol()["master_seeds"], [0, 1])
        self.assertEqual(protocol_identity(), identity)
        self.assertEqual(identity, canonical_hash(load_protocol()))

    def test_unsupported_or_boolean_seeds_are_rejected(self):
        for seed in (-1, 2, True, False, 0.0, "0", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                plan(seed)
        self.assertEqual(plan(1)["master_seed"], 1)

    def test_plan_is_not_a_training_evaluation_or_acceptance_claim(self):
        preview = plan(0)
        for key in ("training_executed", "gpu_runtime_measured", "phase4_acceptance",
                    "matched_reduced_b_controller_implemented", "endpoint_evaluator_implemented"):
            self.assertIs(preview[key], False)
        self.assertEqual(preview["new_environment_interactions"], 0)
        self.assertEqual(preview["new_model_updates"], 0)


if __name__ == "__main__":
    unittest.main()
