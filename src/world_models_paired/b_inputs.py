"""Read-only exact shared P2.4 inputs and paired B initialization.

The authoritative original loader validates the historical fitting contract.
Only the branch changes to B; no teacher, distillation input or inherited
optimizer is constructed. The reduced schedule belongs to the new run manifest.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from world_models.infrastructure.config import resolve_config, validate_config
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.models.world_model import build_model
from world_models.policies.initialization import build_actor_critic
from world_models.teacher.runner import _initial_inputs, _root
from world_models.training.runner import selected_device
from world_models_study.protocol import load_protocol as endpoint_protocol


def load_b_protocol():
    p = endpoint_protocol()
    return {"schema_version": "p52-b-local-protocol-v1", "study_profile": p["study_profile"],
        "task_family": "relay", "variant": "B", "master_seeds": p["master_seeds"],
        **{key: p[key] for key in ("initial_transitions", "initial_model_updates", "local_cycles", "starts",
                                  "horizon", "epochs", "minibatches")},
        "teacher_kl_weight": 0.0, "teacher_queries": 0, "teacher_optimizer_steps": 0,
        "distillation_optimizer_steps": 0, "new_environment_interactions": 0, "new_model_updates": 0,
        "local_ppo_optimizer_initialization": "fresh_from_paired_initialization",
        "local_actor_compute_precision": p["local_actor_compute_precision"],
        "world_model_compute_precision": p["world_model_compute_precision"]}


@dataclass
class InitialBInputs:
    settings: dict
    master_seed: int
    precision: str
    model: object
    actor: object
    critic: object
    dataset: object
    normalization: object
    binding: dict
    provenance: dict


def _binding(inputs):
    settings, dataset, stats, payload, shared = _initial_inputs(**inputs)
    settings = deepcopy(settings)
    settings["variant"] = deepcopy(resolve_config("relay", "B", "pilot", inputs["config_root"]).settings["variant"])
    validate_config(settings)
    p = load_b_protocol()
    recipe = settings["mappo"]
    if ((recipe["starts"], recipe["horizon"], recipe["epochs"], recipe["recurrent_minibatches"])
            != (p["starts"], p["horizon"], p["epochs"], p["minibatches"])
            or dataset.manifest["transition_count"] != p["initial_transitions"]
            or payload["update"] != p["initial_model_updates"]):
        raise ValueError("B inputs differ from the fixed shared model, replay or local PPO recipe")
    return settings, dataset, stats, payload, {"shared_initial_binding": shared, "B_settings": settings}


def load_initial_b(inputs, *, device, precision):
    if precision not in ("float32", "bfloat16"):
        raise ValueError("B world-model precision must be float32 or bfloat16")
    settings, dataset, stats, payload, binding = _binding(inputs)
    target = selected_device(device, precision)
    model = build_model(settings, master_seed=inputs["master_seed"], variant="B").to(target)
    model.load_state_dict(payload["model"], strict=True)
    policies = build_actor_critic(settings, master_seed=inputs["master_seed"], variant="B").to(target)
    return InitialBInputs(settings, inputs["master_seed"], precision, model, policies.actor, policies.critic,
        dataset, stats, binding, {"schema_version": "p52-frozen-B-input-v1", "inputs": deepcopy(inputs),
                                 "input_binding_sha256": canonical_hash(binding),
                                 "upstream_optimizers_restored": False, "teacher_constructed_or_queried": False})


def verify_frozen_inputs(context):
    _, _, _, _, binding = _binding(context.provenance["inputs"])
    if canonical_hash(binding) != canonical_hash(context.binding):
        raise ValueError("B frozen replay, normalization, selection or settings changed")


def input_identity(context):
    return canonical_hash({"binding": context.binding, "provenance": context.provenance})


def safe_inputs(*, initial_selection, master_seed, data_root, statistics_root, config_root):
    if type(master_seed) is not int or master_seed not in load_b_protocol()["master_seeds"]:
        raise ValueError("B permits Relay master seeds 0/1 only")
    return {"master_seed": master_seed,
        **{name: str(_root(value)) for name, value in (
            ("initial_selection", initial_selection), ("data_root", data_root),
            ("statistics_root", statistics_root), ("config_root", config_root))}}
