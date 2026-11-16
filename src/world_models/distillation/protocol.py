"""Registered P4.2 supervised warm-start, separate from on-policy MAPPO."""
from copy import deepcopy

from world_models.infrastructure.reproducibility import canonical_hash


DISTILLATION_PROTOCOL = {
    "schema_version": "p42-distillation-protocol-v1",
    "study_profile": "relay_four_day_v1", "stage": "student_warm_start",
    "task_family": "relay", "master_seeds": [0, 1],
    "initial_transitions": 50000, "initial_model_updates": 4500,
    "teacher_cycles": 25, "trajectories": 128, "horizon": 15,
    "temperature": 1.0, "target_dtype": "float32",
    "updates": 100, "sequences_per_update": 32,
    "objective": "valid_player_mean_forward_kl", "objective_weight": 1.0,
    "learning_rate": 3e-4, "adam_epsilon": 1e-5,
    "adam_betas": [0.9, 0.999], "weight_decay": 0.0, "gradient_clip": 0.5,
    "prefix": "complete_real_history_recomputed_with_current_student_each_minibatch",
    "imagined_previous_actions": "executed_teacher_actions",
    "new_environment_interactions": 0, "new_model_updates": 0,
    "critic_updates": 0, "teacher_updates": 0,
    "next_local_ppo_optimizer": "fresh",
    "full_d_pipeline_implemented": False,
}


def load_protocol():
    return deepcopy(DISTILLATION_PROTOCOL)


def protocol_identity():
    return canonical_hash(DISTILLATION_PROTOCOL)


def plan(seed):
    if type(seed) is not int or seed not in DISTILLATION_PROTOCOL["master_seeds"]:
        raise ValueError("P4.2 permits Relay master seeds 0 and 1 only")
    return {"schema_version": "p42-plan-v1", "protocol": load_protocol(),
            "protocol_sha256": protocol_identity(), "master_seed": seed,
            "imagined_joint_step_cap": 1920, "student_optimizer_steps": 100,
            "expected_buffer_passes": 25, "training_executed": False,
            "gpu_runtime_measured": False, "full_d_pipeline_implemented": False}
