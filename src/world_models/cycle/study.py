"""Explicit study amendments without rewriting the frozen P2.4 registration.

Only the named, repository-owned budget amendment is supported. It supplies new
Phase 3 settings and retains the original complete settings for validating the
shared initial checkpoint. The Phase 4 treatment is a declaration, not training
code. Imports remain model-free for actor-only evaluation.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from world_models.infrastructure.config import _read, resolve_config, validate_config
from world_models.infrastructure.reproducibility import canonical_hash


ROOT = Path(__file__).resolve().parents[3]
STUDY_PROFILES = ("relay_week_v1",)
_OVERRIDES = {
    "experiment": {"real_budget": 100000, "checkpoints": [0, 50000, 75000, 100000],
                   "checkpoint_episodes": 20, "final_episodes": 20},
    "world_model": {"online_updates": 250},
    "mappo": {"initial_local_cycles": 40, "online_local_cycles": 20},
}
_TREATMENT = {
    "name": "teacher_warm_start_only", "implementation_status": "pending_phase4",
    "teacher_initial_cycles": 50, "teacher_online_cycles": 0, "initial_trajectories": 128,
    "warm_start_updates": 100, "warm_start_batch_sequences": 32, "auxiliary_kl_weight": 0.0,
    "trajectory_source": "teacher", "temperature": 1.0,
    "distillation_optimizer": {"name": "Adam", "learning_rate": 0.0003, "epsilon": 0.00001,
                               "betas": [0.9, 0.999], "weight_decay": 0.0, "gradient_clip": 0.5},
    "ppo_optimizer_reset_after_distillation": True,
    "guided_local_cycles": 0, "unguided_local_cycles": 80,
}
_EVALUATION = {
    "implementation_status": "pending_phase5", "split": "test", "bank_seed": 5600,
    "checkpoint_episodes": 20, "episodes_per_layout": 10, "checkpoints": [0, 50000, 75000, 100000],
    "mode": "stochastic", "temperature": 1.0, "evaluation_enters_replay": False,
}
_ANALYSIS = {
    "implementation_status": "pending_phase5", "primary_metric": "normalized_success_auc",
    "normalization_budget": 100000, "training_seed_pairs": 2, "summaries": ["paired_values", "mean", "range"],
    "bootstrap_replicates": 0, "confidence_intervals": False, "statistical_significance_claim": False,
}


def _profile(name):
    if name not in STUDY_PROFILES:
        raise ValueError("unknown study profile")
    path = ROOT / "configs/studies" / f"{name}.yaml"
    profile = _read(path)
    expected = {"schema_version": 1, "name": name, "parent_profile": "pilot",
                "scope": "exploratory_relay_two_seed", "task_families": ["relay"],
                "master_seeds": [0, 1], "overrides": _OVERRIDES,
                "planned_d_treatment": _TREATMENT, "planned_evaluation": _EVALUATION, "planned_analysis": _ANALYSIS}
    # The name denotes one fixed amendment. Changing budgets requires a new
    # named amendment, not changing the meaning of an existing run profile.
    if canonical_hash(profile) != canonical_hash(expected):
        raise ValueError("study amendment differs from the fixed relay_week_v1 protocol")
    return profile, hashlib.sha256(path.read_bytes()).hexdigest()


def apply_study_profile(parent_settings, name, *, master_seed=None):
    """Return owned active settings plus a binding to the complete parent.

    This never alters base/task/variant/experiment files or the selection. B and
    D receive identical common settings; only B is runnable through Phase 3.
    """
    profile, digest = _profile(name)
    validate_config(parent_settings)
    task, variant = parent_settings["task"]["name"], parent_settings["variant"]["name"]
    if task not in profile["task_families"]:
        raise ValueError("relay_week_v1 permits Relay only; Cover remains diagnostic evidence")
    if master_seed is not None and (type(master_seed) is not int or master_seed not in profile["master_seeds"]):
        raise ValueError("relay_week_v1 permits master seeds 0 and 1 only")
    parent = resolve_config(task, variant, profile["parent_profile"], ROOT / "configs").settings
    if parent_settings != parent:
        raise ValueError("study amendment requires the unchanged original pilot parent settings")
    settings = deepcopy(parent)
    for section, changes in profile["overrides"].items():
        settings[section].update(deepcopy(changes))
    validate_config(settings)
    binding = {"schema_version": "study-amendment-v1", "name": name,
               "profile_sha256": digest, "parent_profile": profile["parent_profile"],
               "parent_settings_sha256": canonical_hash(parent), "settings_sha256": canonical_hash(settings),
               "scope": profile["scope"], "task_families": profile["task_families"],
               "master_seeds": profile["master_seeds"],
               "planned_d_treatment": deepcopy(profile["planned_d_treatment"]),
               "planned_evaluation": deepcopy(profile["planned_evaluation"]),
               "planned_analysis": deepcopy(profile["planned_analysis"]),
               "d_training_implemented": False}
    return settings, binding


def parent_settings_for_study(settings, binding, *, master_seed=None):
    """Validate active settings and the exact amendment before returning parent."""
    if not isinstance(binding, dict):
        raise ValueError("study amendment binding must be an object")
    validate_config(settings)
    parent = resolve_config(settings["task"]["name"], settings["variant"]["name"],
                            "pilot", ROOT / "configs").settings
    expected_settings, expected_binding = apply_study_profile(parent, binding.get("name"), master_seed=master_seed)
    if canonical_hash(settings) != canonical_hash(expected_settings) or canonical_hash(binding) != canonical_hash(expected_binding):
        raise ValueError("active settings or study amendment identity changed")
    return parent
