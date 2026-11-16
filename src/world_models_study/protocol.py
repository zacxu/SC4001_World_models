"""Static P5.1 candidate protocol, independent of frozen learner sources.

Registration describes the approved experiment. It does not implement its B
runner, native endpoint evaluator or analysis reader and cannot authorize their
execution. The existing P4.4 YAML is preserved and bound by its original bytes.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import yaml

from world_models.infrastructure.config import _UniqueKeyLoader
from world_models.infrastructure.reproducibility import (
    PAIRED_INITIAL_COMPONENTS, SEED_DERIVATION_VERSION, canonical_hash, canonical_json, derive_seed,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/studies/relay_endpoint_protocol_v1.yaml"
_PARENT_SHA = "a10c1d3688943c88f34cad3a6447fcd287e68615028739db42bea3551e4e5b6a"
_EXPECTED = {
    "schema_version": "p51-exploratory-endpoint-protocol-v1", "study_profile": "relay_endpoint_protocol_v1",
    "parent_study_profile": "relay_four_day_v1", "parent_profile_path": "configs/studies/relay_four_day_v1.yaml",
    "parent_profile_sha256": _PARENT_SHA, "task_family": "relay", "master_seeds": [0, 1],
    "variants": ["B", "D"], "training_matrix_runs": 4, "initial_transitions": 50000,
    "initial_model_updates": 4500, "shared_initial_replay_per_seed": True,
    "model_and_normalization_frozen": True, "new_environment_interactions": 0, "new_model_updates": 0,
    "local_cycles": 20, "starts": 256, "horizon": 15, "epochs": 4, "minibatches": 4,
    "teacher_cycles": 25, "trajectories": 128, "distillation_updates": 100,
    "distillation_sequences_per_update": 32, "distillation_temperature": 1.0,
    "distillation_loss": "forward_kl_teacher_to_student",
    "distillation_trajectory_source": "teacher", "local_actor_compute_precision": "float32",
    "teacher_policy_compute_precision": "float32", "world_model_compute_precision": "bfloat16",
    "distillation_optimizer": {"kind": "Adam", "learning_rate": 3e-4, "epsilon": 1e-5,
        "betas": [.9, .999], "weight_decay": 0.0, "gradient_clip": .5},
    "teacher_kl_weight": 0.0, "teacher_refresh_cycles": 0, "guidance_decay_enabled": False,
    "local_ppo_optimizer_initialization": {"B": "fresh_from_paired_initialization", "D": "fresh_after_distillation"},
    "evaluation_episodes_per_actor": 20, "evaluation_bank_seed": 5600, "evaluation_split": "test",
    "evaluation_mode": "stochastic", "evaluation_temperature": 1.0, "evaluation_actor_precision": "float32",
    "evaluation_frame_skip": 4, "evaluation_deadline": 400,
    "evaluation_layouts": ["relay_test_00", "relay_test_01"], "evaluation_episodes_per_layout": 10,
    "evaluation_actor_only": True, "evaluation_final_local_cycles": 20, "evaluation_data_enters_training": False,
    "evaluation_seed_policy": "shared_scenarios_across_all_runs_actions_paired_by_master_seed_and_player_excluding_branch",
    "primary_metric": "final_success_rate",
    "analysis": {"primary_effect": "final_success_rate_D_minus_B",
        "secondary_effects": ["mean_return_D_minus_B", "latch_episode_rate_D_minus_B"],
        "latch_definition": "at_least_one_latch_activation_in_the_episode",
        "summaries": ["all_arm_success_counts_and_rates", "each_seed_paired_difference", "descriptive_mean",
            "observed_min_max", "per_layout", "termination_counts", "completion_length_given_success"],
        "no_success_completion_length": "null_with_zero_denominator", "confidence_intervals": False,
        "p_values": False, "bootstrap": False, "auc": False, "superiority_claim": False,
        "equivalence_claim": False, "seed_is_replication_unit": True, "missing_evidence_imputation": "forbidden"},
    "retry_policy": {"infrastructure_retries_max": 2, "same_seed_configuration_and_artifact_identity": True,
        "restore_exact_committed_optimizer_rng_and_stage": True, "retain_failed_attempts_and_physical_compute": True,
        "retry_valid_poor_results": False, "retry_numerical_failures": False, "seed_replacement": False,
        "checkpoint_selection_from_test_results": False, "evaluation_retry_selects_first_valid_episode_attempt": True},
    "fallback_policy": {"default_master_seeds": [0, 1], "single_pair_seed": 0,
        "requires_predeclared_cutoff_and_explicit_deviation": True, "decision_before_test_outcomes": True,
        "decision_before_comparative_training_and_outcome_inspection": True,
        "automatic_fallback": False},
    "disclosures": {"exploratory_pilot": True, "seeds_and_task_already_used_in_development": True,
        "adopted_development_work_is_not_a_fresh_cohort": True,
        "long_online_B_development_run_is_not_a_matched_B_arm": True,
        "D_extra_compute_is_not_compute_matched": True, "generalization_scope": "two_authored_unseen_relay_layouts",
        "low_distillation_kl_does_not_establish_real_cooperation": True, "test_bank_reserved_for_final_endpoint": True,
        "historical_costs_and_unsuccessful_runs_remain_visible": True},
    "capabilities": {"candidate_registration_only": True, "matched_reduced_B_runner_implemented": False,
        "endpoint_evaluator_implemented": False, "paired_endpoint_reader_implemented": False,
        "execution_ready": False, "cohort_frozen": False},
}


def load_protocol():
    """Read strict additive YAML; unknown, changed or mistyped values fail closed."""
    value = yaml.load(PROFILE.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if type(value) is not dict or canonical_json(value) != canonical_json(_EXPECTED):
        raise ValueError("P5.1 endpoint protocol differs from the approved static registration")
    parent = ROOT / value["parent_profile_path"]
    if not parent.is_file() or hashlib.sha256(parent.read_bytes()).hexdigest() != _PARENT_SHA:
        raise ValueError("the preserved P4.4 parent profile bytes changed")
    return deepcopy(value)


def protocol_identity():
    return canonical_hash(load_protocol())


def _seed(value, variant):
    if type(value) is not int or value not in (0, 1) or variant not in ("B", "D"):
        raise ValueError("the endpoint matrix permits master seeds 0/1 and variants B/D only")


def _rollout_seeds(world):
    return {"world": world, **{name: int.from_bytes(hashlib.sha256(
        f"p22-sampling-v1|{world}|{name}".encode()).digest()[:8], "big") % 2**63
        for name in ("prior", "posterior", "outcome")}}


def _from_seed_components(seed):
    def named(name):
        return int.from_bytes(hashlib.sha256(f"p33-rollout-streams-v1|{seed}|{name}".encode()).digest()[:8], "big") % 2**63
    return {**_rollout_seeds(named("world")), **{f"actor_{player}": named(f"actor_{player}") for player in (0, 1)}}


def training_seed_manifest(seed, variant):
    """Describe actual P4.1-P4.3 namespaces without importing Torch or learners."""
    _seed(seed, variant)
    load_protocol()
    derive = lambda component: derive_seed("relay", seed, component, variant)
    local = {name: derive(name) for name in ("imagination_anchors", "imagination_prefixes", "local_policy_updates")}
    result = {"schema_version": "p51-training-seed-manifest-v1", "derivation_version": SEED_DERIVATION_VERSION,
        "task_family": "relay", "master_seed": seed, "variant": variant,
        "paired_initialization": {name: derive(name) for name in sorted(PAIRED_INITIAL_COMPONENTS)},
        "local": {"rng_namespaces": local, "rollout_seeds": {**_rollout_seeds(derive("imagination")),
            **{f"actor_{player}": derive_seed("relay", seed, "local_action_sampling", variant, player) for player in (0, 1)}}},
        "teacher": None, "distillation": None,
        "independence_scope": "distinct recorded namespaces, not a statistical independence claim"}
    if variant == "D":
        for stage, names, component in (
                ("teacher", ("teacher_initialization", "teacher_critic_initialization", "teacher_anchor_sampling",
                    "teacher_anchor_prefix", "teacher_rollouts", "teacher_updates"), "teacher_rollouts"),
                ("distillation", ("distillation_anchors", "distillation_prefix", "distillation_rollouts",
                    "distillation_updates"), "distillation_rollouts")):
            result[stage] = {"rng_namespaces": {name: derive(name) for name in names},
                             "rollout_seeds": _from_seed_components(derive(component))}
    return result


def plan():
    """Return workload arithmetic and pending execution boundaries, never results."""
    p = load_protocol()
    matrix, per_cycle = [], p["epochs"] * p["minibatches"]
    for seed in p["master_seeds"]:
        for variant in p["variants"]:
            matrix.append({"task_family": "relay", "master_seed": seed, "variant": variant,
                "initial_transitions": p["initial_transitions"], "initial_model_updates": p["initial_model_updates"],
                "new_environment_interactions": 0, "new_model_updates": 0,
                "local_cycles": p["local_cycles"], "local_actor_optimizer_steps": p["local_cycles"] * per_cycle,
                "local_critic_optimizer_steps": p["local_cycles"] * per_cycle,
                "teacher_cycles": p["teacher_cycles"] if variant == "D" else 0,
                "teacher_actor_optimizer_steps": p["teacher_cycles"] * per_cycle if variant == "D" else 0,
                "teacher_critic_optimizer_steps": p["teacher_cycles"] * per_cycle if variant == "D" else 0,
                "distillation_trajectories": p["trajectories"] if variant == "D" else 0,
                "distillation_optimizer_steps": p["distillation_updates"] if variant == "D" else 0,
                "evaluation_episodes": p["evaluation_episodes_per_actor"],
                "training_seeds": training_seed_manifest(seed, variant)})
    local_cap = 4 * p["local_cycles"] * p["starts"] * p["horizon"]
    teacher_cap = 2 * p["teacher_cycles"] * p["starts"] * p["horizon"]
    buffer_cap = 2 * p["trajectories"] * p["horizon"]
    return {"schema_version": "p51-exploratory-endpoint-plan-v1", "protocol": p,
        "protocol_sha256": protocol_identity(), "run_matrix": matrix,
        "budgets": {"physical_existing_initial_joint_transitions": 100000,
            "attributed_initial_joint_transitions": 200000, "new_training_joint_transitions": 0,
            "physical_selected_initial_model_updates": 9000, "attributed_selected_initial_model_updates": 18000,
            "new_model_updates": 0, "total_local_cycles": 80, "total_teacher_cycles": 50,
            "local_actor_optimizer_steps": 1280, "local_critic_optimizer_steps": 1280,
            "teacher_actor_optimizer_steps": 800, "teacher_critic_optimizer_steps": 800,
            "distillation_optimizer_steps": 200, "local_imagined_joint_step_cap": local_cap,
            "teacher_imagined_joint_step_cap": teacher_cap, "buffer_imagined_joint_step_cap": buffer_cap,
            "combined_imagined_joint_step_cap_excluding_prefix_reconstruction": local_cap + teacher_cap + buffer_cap,
            "final_evaluation_episodes": 80, "final_evaluation_joint_decision_cap_excluding_resets": 32000},
        "execution_ready": False, "cohort_frozen": False, "training_executed": False,
        "native_evaluation_executed": False, "phase5_acceptance": False,
        "pending": ["P5.2 matched reduced B runner", "P5.4 actor-only final test evaluator",
                    "P5.5 paired endpoint reader", "prospective executable cohort freeze"]}
