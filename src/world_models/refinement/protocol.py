"""Fixed P4.3 stage of the approved Relay warm-start pilot."""
from copy import deepcopy

from world_models.infrastructure.reproducibility import canonical_hash


REFINEMENT_PROTOCOL = {
    "schema_version": "p43-local-refinement-protocol-v1", "study_profile": "relay_four_day_v1",
    "stage": "unguided_local_refinement", "task_family": "relay", "variant": "D",
    "master_seeds": [0, 1], "initial_transitions": 50000, "initial_model_updates": 4500,
    "local_cycles": 20, "starts": 256, "horizon": 15, "epochs": 4, "minibatches": 4,
    "teacher_kl_weight": 0.0, "teacher_queries": 0, "teacher_refresh_cycles": 0,
    "guidance_decay_enabled": False, "teacher_optimizer_steps": 0,
    "distillation_optimizer_steps": 0, "new_environment_interactions": 0, "new_model_updates": 0,
    "local_ppo_optimizer_initialization": "fresh_after_distillation",
    "actor_history": "current_student_complete_own_real_prefix_then_own_imagined_actions",
    "rollout_behavior": "current_local_student_only", "evaluation_executed": False,
    "full_d_pipeline_implemented": False,
}


def load_protocol():
    return deepcopy(REFINEMENT_PROTOCOL)


def protocol_identity():
    return canonical_hash(REFINEMENT_PROTOCOL)


def plan(seed):
    if type(seed) is not int or seed not in REFINEMENT_PROTOCOL["master_seeds"]:
        raise ValueError("P4.3 permits Relay master seeds 0 and 1 only")
    p = load_protocol()
    steps = p["local_cycles"] * p["epochs"] * p["minibatches"]
    return {"schema_version": "p43-plan-v1", "protocol": p, "protocol_sha256": protocol_identity(),
            "master_seed": seed, "local_cycles": p["local_cycles"],
            "actor_optimizer_steps": steps, "critic_optimizer_steps": steps,
            "imagined_joint_step_cap": p["local_cycles"] * p["starts"] * p["horizon"],
            "prefix_reconstruction_included_in_cap": False,
            "training_executed": False, "gpu_runtime_measured": False,
            "full_d_pipeline_implemented": False, "phase4_acceptance": False}
