"""Separate registration of the approved fixed-data schedule; no legacy edits."""
from copy import deepcopy
from pathlib import Path

import yaml

from world_models.infrastructure.reproducibility import canonical_hash


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/studies/relay_four_day_v1.yaml"
CAMPAIGN_PROTOCOL = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def load_protocol():
    return deepcopy(CAMPAIGN_PROTOCOL)


def protocol_identity():
    return canonical_hash(CAMPAIGN_PROTOCOL)


def stage_targets():
    p = load_protocol()
    return {"teacher": p["teacher_cycles"], "buffer": p["trajectories"],
            "distillation": p["distillation_updates"], "local_cycle": p["local_cycles"]}


def plan(seed):
    p = load_protocol()
    if type(seed) is not int or seed not in p["master_seeds"]:
        raise ValueError("the fixed-data Relay pilot permits master seeds 0 and 1 only")
    per_cycle = p["epochs"] * p["minibatches"]
    targets = stage_targets()
    return {"schema_version": "p44-d-campaign-plan-v1", "protocol": p,
        "protocol_sha256": protocol_identity(), "master_seed": seed, "variant": "D",
        "stage_targets": targets, "learner_units": sum(targets.values()),
        "total_units": sum(targets.values()) + 1,
        "teacher_actor_optimizer_steps": p["teacher_cycles"] * per_cycle,
        "teacher_critic_optimizer_steps": p["teacher_cycles"] * per_cycle,
        "local_actor_optimizer_steps": p["local_cycles"] * per_cycle,
        "local_critic_optimizer_steps": p["local_cycles"] * per_cycle,
        "local_imagined_joint_step_cap": p["local_cycles"] * p["starts"] * p["horizon"],
        "teacher_imagined_joint_step_cap": p["teacher_cycles"] * p["starts"] * p["horizon"],
        "distillation_imagined_joint_step_cap": p["trajectories"] * p["horizon"],
        "training_executed": False, "new_environment_interactions": 0, "new_model_updates": 0,
        "gpu_runtime_measured": False, "phase4_acceptance": False,
        "matched_reduced_b_controller_implemented": False, "endpoint_evaluator_implemented": False}
