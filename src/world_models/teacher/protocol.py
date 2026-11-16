"""Fixed P4.1 stage of the approved four-day pilot, independent of legacy configs.

This registration implements only the initial teacher. It does not make the
reduced B controller, distillation or complete D experiment executable.
"""
from copy import deepcopy

from world_models.infrastructure.reproducibility import canonical_hash


TEACHER_PROTOCOL = {
    "schema_version": "p41-teacher-protocol-v1",
    "study_profile": "relay_four_day_v1",
    "stage": "initial_teacher",
    "task_family": "relay",
    "master_seeds": [0, 1],
    "initial_transitions": 50000,
    "initial_model_updates": 4500,
    "teacher_cycles": 25,
    "starts": 256,
    "horizon": 15,
    "epochs": 4,
    "minibatches": 4,
    "attention_heads": 4,
    "world_width": 256,
    "new_environment_interactions": 0,
    "new_model_updates": 0,
    "teacher_online_cycles": 0,
    "student_training_implemented": False,
}


def load_protocol():
    """Return owned recipe data; callers must not mutate the registration."""
    return deepcopy(TEACHER_PROTOCOL)


def protocol_identity():
    return canonical_hash(TEACHER_PROTOCOL)


def plan(seed):
    if type(seed) is not int or seed not in TEACHER_PROTOCOL["master_seeds"]:
        raise ValueError("P4.1 permits Relay master seeds 0 and 1 only")
    p = load_protocol()
    steps = p["teacher_cycles"] * p["epochs"] * p["minibatches"]
    return {
        "schema_version": "p41-plan-v1", "protocol": p,
        "protocol_sha256": protocol_identity(), "master_seed": seed,
        "teacher_actor_optimizer_steps": steps,
        "teacher_critic_optimizer_steps": steps,
        "imagined_joint_step_cap": p["teacher_cycles"] * p["starts"] * p["horizon"],
        "prefix_reconstruction_included_in_cap": False,
        "training_executed": False, "gpu_runtime_measured": False,
        "full_d_pipeline_implemented": False,
    }
