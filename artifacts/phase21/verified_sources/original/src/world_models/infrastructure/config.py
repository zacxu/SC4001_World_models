"""Strict, reproducible configuration composition for the two-arm protocol."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml

from world_models.contracts import ACTION_COUNT, LOCAL_RGB_SHAPE, PREVIOUS_ACTION_START
from world_models.infrastructure.reproducibility import canonical_hash


class ConfigError(ValueError):
    """Configuration violates the declared implementation/research contract."""


def _fields(kind: Any, names: str) -> dict[str, Any]:
    return {name: kind for name in names.split()}


SCHEMA: dict[str, Any] = {
    "schema_version": int,
    "study": {
        **_fields([str], "variants task_families"),
        **_fields(bool, "separate_family_runs evaluation_enters_replay"),
        **_fields(str, "primary_metric interaction_unit"),
    },
    "environment": {
        **_fields(int, "agents frame_skip action_count previous_action_start"),
        **_fields([int], "image_shape render_shape"),
        "local_actor_inputs": [str],
        "privileged_mode": str,
        **_fields(bool, "actor_reward_input actor_global_input actor_agent_identity communication "
                  "shared_actor_weights separate_actor_memory prior_future_truth exact_episode_clock"),
    },
    "resources": {
        **_fields(int, "requested_gpu_count requested_cpu_cores initial_collectors learners_per_gpu "
                  "data_loader_workers cpu_threads"),
        **_fields(str, "requested_gpu_model mixed_precision probability_dtype optimizer_dtype"),
        "distributed_training": bool,
    },
    "world_model": {
        "encoder_channels": [int],
        **_fields(int, "memory_tokens width transformer_blocks attention_heads ffn_width "
                  "latent_variables latent_categories effective_batch sequence_length burn_in "
                  "initial_updates online_updates recent_transitions"),
        **_fields(float, "learning_rate weight_decay gradient_clip free_nats uniform_mixture "
                  "recent_fraction fitting_episode_fraction"),
        "loss_weights": _fields(float, "rgb reward outcome auxiliary dynamics_kl representation_kl"),
    },
    "actor": {**_fields(int, "embedding_size hidden_size"), "dropout": float},
    "critic": {"hidden_sizes": [int], **_fields(bool, "actor_memories remaining_time")},
    "mappo": {
        **_fields(int, "horizon starts epochs recurrent_minibatches initial_local_cycles online_local_cycles"),
        **_fields(float, "gamma gae_lambda clip_epsilon learning_rate optimizer_epsilon "
                  "entropy_coefficient value_coefficient value_huber_delta gradient_clip"),
        **_fields(bool, "normalize_actor_advantages normalize_values differentiate_world_model frozen_rollout_snapshots"),
    },
    "teacher": {**_fields(int, "initial_cycles online_cycles attention_heads head_width"), "recurrent_input": str},
    "distillation": {
        **_fields(str, "target direction trajectory_source"),
        **_fields(int, "initial_trajectories warm_start_updates warm_start_batch_sequences"),
        **_fields(float, "temperature initial_weight decay_fraction"),
    },
    "task": {
        "name": str,
        **_fields(int, "deadline dwell_decisions milestone_count train_layouts validation_layouts test_layouts"),
        **_fields(float, "success_reward failure_reward step_cost milestone_bonus"),
    },
    "variant": {"name": str, **_fields(bool, "teacher_enabled distillation_enabled")},
    "experiment": {
        **_fields(str, "name evaluation_split"),
        **_fields([int], "master_seeds checkpoints"),
        **_fields(int, "real_budget initial_transitions collection_block checkpoint_episodes final_episodes bootstrap_replicates"),
        **_fields(float, "practical_auc_effect initial_scripted_fraction"),
    },
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ConfigError(f"non-string or duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _read(path: Path) -> dict:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return value


def _shape(value: Any, schema: Any, path: str = "config") -> None:
    if isinstance(schema, dict):
        if not isinstance(value, dict) or set(value) != set(schema):
            actual = set(value) if isinstance(value, dict) else set()
            raise ConfigError(f"{path}: missing={sorted(set(schema)-actual)}, unknown={sorted(actual-set(schema))}")
        for key, child in schema.items():
            _shape(value[key], child, f"{path}.{key}")
    elif isinstance(schema, list):
        if not isinstance(value, list) or not value:
            raise ConfigError(f"{path} must be a nonempty list")
        for index, child in enumerate(value):
            _shape(child, schema[0], f"{path}[{index}]")
    elif schema is float:
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ConfigError(f"{path} must be a finite number")
    elif type(value) is not schema:
        raise ConfigError(f"{path} must have type {schema.__name__}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate_config(cfg: dict) -> None:
    """Validate the complete settings dictionary, including cross-field rules."""
    _shape(cfg, SCHEMA)
    _require(cfg["schema_version"] == 1, "unsupported schema_version")
    study, env, model = cfg["study"], cfg["environment"], cfg["world_model"]
    run, variant, task = cfg["experiment"], cfg["variant"], cfg["task"]
    ppo, distill, teacher = cfg["mappo"], cfg["distillation"], cfg["teacher"]
    _require(study["variants"] == ["B", "D"] and study["task_families"] == ["relay", "cover"], "only B/D and relay/cover are in scope")
    _require(study["separate_family_runs"] and not study["evaluation_enters_replay"], "separate families and isolated evaluation are required")
    _require(study["primary_metric"] == "normalized_success_auc" and study["interaction_unit"] == "joint_transition", "research metric/interaction contract changed")
    _require(env["agents"] == 2 and env["image_shape"] == list(LOCAL_RGB_SHAPE), "unsupported actor observation/agent contract")
    _require(env["action_count"] == ACTION_COUNT and env["previous_action_start"] == PREVIOUS_ACTION_START, "action contract does not match implementation")
    _require(env["local_actor_inputs"] == ["local_rgb", "previous_action", "local_hidden"], "local actor inputs changed")
    _require(not any(env[key] for key in ("actor_reward_input", "actor_global_input", "actor_agent_identity", "communication", "prior_future_truth")), "locality or future-information contract violated")
    _require(all(env[key] for key in ("shared_actor_weights", "separate_actor_memory", "exact_episode_clock")), "shared weights, separate memory, and exact clock are required")
    _require(env["privileged_mode"] == "auxiliary_targets_only", "privilege is auxiliary supervision only")
    _require(variant["name"] in ("B", "D"), "unknown variant")
    expected = variant["name"] == "D"
    _require(variant["teacher_enabled"] is expected and variant["distillation_enabled"] is expected, "B/D teacher/distillation flags inconsistent")
    _require(task["name"] in ("relay", "cover") and run["name"] in ("pilot", "final"), "unknown task/experiment")
    _require(run["evaluation_split"] == ("test" if run["name"] == "final" else "validation"), "pilot/final evaluation split mismatch")
    for section in (model, ppo, teacher, cfg["actor"], cfg["resources"], task, run):
        for key, value in section.items():
            if type(value) is int:
                _require(value >= 0, f"{key} must not be negative")
    for section, keys in ((model, "memory_tokens width transformer_blocks attention_heads ffn_width latent_variables latent_categories effective_batch sequence_length initial_updates online_updates recent_transitions"),
                          (ppo, "horizon starts epochs recurrent_minibatches initial_local_cycles online_local_cycles"),
                          (teacher, "initial_cycles online_cycles attention_heads head_width"),
                          (task, "deadline milestone_count train_layouts validation_layouts test_layouts"),
                          (run, "real_budget initial_transitions collection_block checkpoint_episodes final_episodes bootstrap_replicates")):
        for key in keys.split():
            _require(section[key] > 0, f"{key} must be positive")
    _require(len(model["encoder_channels"]) == 4 and all(x > 0 for x in model["encoder_channels"]), "encoder requires four positive channel widths")
    _require(all(x > 0 for x in cfg["critic"]["hidden_sizes"]), "critic widths must be positive")
    _require(all(x > 0 for x in env["render_shape"]) and len(env["render_shape"]) == 2 and env["frame_skip"] > 0, "invalid render shape/frame skip")
    _require(all(cfg["actor"][key] > 0 for key in ("hidden_size", "embedding_size")), "actor dimensions must be positive")
    _require(model["width"] % model["attention_heads"] == 0 and teacher["head_width"] % teacher["attention_heads"] == 0, "attention width must divide by head count")
    _require(0 <= model["burn_in"] < model["sequence_length"], "burn-in must leave loss-bearing steps")
    _require(ppo["starts"] % ppo["recurrent_minibatches"] == 0, "recurrent minibatches must divide rollout starts")
    _require(cfg["critic"]["actor_memories"] and cfg["critic"]["remaining_time"], "critic must include local memories and clock")
    _require(ppo["normalize_actor_advantages"] and not ppo["differentiate_world_model"] and ppo["frozen_rollout_snapshots"], "PPO gradient/snapshot contract violated")
    _require(teacher["recurrent_input"] == "local_history", "teacher memory must remain locally reconstructible")
    _require(distill["target"] == "soft_action_distribution" and distill["direction"] == "teacher_to_student_forward_kl" and distill["trajectory_source"] == "student", "distillation target/direction/rollout contract violated")
    for section, keys in ((model, "learning_rate gradient_clip free_nats"), (ppo, "learning_rate optimizer_epsilon value_huber_delta gradient_clip"), (distill, "temperature initial_weight")):
        _require(all(section[key] > 0 for key in keys.split()), f"positive optimization values required: {keys}")
    for section, key in ((ppo, "gamma"), (ppo, "gae_lambda"), (ppo, "clip_epsilon"), (distill, "decay_fraction"), (model, "fitting_episode_fraction")):
        _require(0 < section[key] < 1, f"{key} must be strictly between zero and one")
    for section, key in ((model, "uniform_mixture"), (model, "recent_fraction"), (cfg["actor"], "dropout"), (run, "initial_scripted_fraction"), (run, "practical_auc_effect")):
        _require(0 <= section[key] <= 1, f"{key} must be between zero and one")
    _require(all(weight >= 0 for weight in model["loss_weights"].values()), "negative model loss weights")
    _require(model["weight_decay"] >= 0 and ppo["entropy_coefficient"] >= 0 and ppo["value_coefficient"] > 0, "invalid regularization/value loss settings")
    _require(all(distill[key] > 0 for key in ("initial_trajectories", "warm_start_updates", "warm_start_batch_sequences")), "distillation budgets must be positive")
    _require(task["success_reward"] > 0 and task["failure_reward"] < 0 and task["step_cost"] <= 0 and task["milestone_bonus"] >= 0, "invalid task reward signs")
    _require(task["dwell_decisions"] < task["deadline"], "console dwell cannot consume entire deadline")
    _require(all(cfg["resources"][key] > 0 for key in ("requested_gpu_count", "requested_cpu_cores", "initial_collectors", "learners_per_gpu", "cpu_threads")), "resource counts must be positive")
    _require(cfg["resources"]["probability_dtype"] == "float32" and cfg["resources"]["optimizer_dtype"] == "float32", "probability/optimizer arithmetic must be float32")
    _require(len(set(run["master_seeds"])) == len(run["master_seeds"]) and all(0 <= s <= 2**63 - 1 for s in run["master_seeds"]), "invalid/duplicate master seeds")
    initial, budget, block = run["initial_transitions"], run["real_budget"], run["collection_block"]
    _require(budget > initial and (budget - initial) % block == 0, "real budget must contain whole online collection blocks")
    checkpoints = run["checkpoints"]
    _require(checkpoints == sorted(set(checkpoints)) and checkpoints[0] == 0 and checkpoints[-1] == budget and initial in checkpoints, "checkpoint grid must span zero to budget and include initial collection")
    _require(all(n == 0 or (initial <= n <= budget and (n-initial) % block == 0) for n in checkpoints), "checkpoints must fall on completed absolute collection-block boundaries")
    _require(run["final_episodes"] >= run["checkpoint_episodes"], "final evaluation cannot have fewer episodes than AUC evaluation")


@dataclass(frozen=True)
class ResolvedConfig:
    settings: dict[str, Any]
    derived: dict[str, Any]
    config_hash: str
    source_hashes: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return deepcopy({"settings": self.settings, "derived": self.derived,
                         "config_hash": self.config_hash, "source_hashes": self.source_hashes})


def resolve_config(task: str, variant: str, experiment: str, config_dir: Path | str = "configs") -> ResolvedConfig:
    if task not in ("relay", "cover") or variant not in ("B", "D") or experiment not in ("pilot", "final"):
        raise ConfigError("choose task relay/cover, variant B/D, experiment pilot/final")
    root = Path(config_dir)
    paths = [root / "base.yaml", root / "tasks" / f"{task}.yaml",
             root / "variants" / f"{variant}.yaml", root / "experiments" / f"{experiment}.yaml"]
    cfg = _read(paths[0])
    _require(set(cfg) == set(SCHEMA) - {"task", "variant", "experiment"}, "base.yaml has missing or unknown sections")
    for path, section in zip(paths[1:], ("task", "variant", "experiment")):
        overlay = _read(path)
        _require(set(overlay) == {section}, f"{path}: overlay may change only '{section}'")
        cfg[section] = overlay[section]
    validate_config(cfg)
    _require(cfg["task"]["name"] == task and cfg["variant"]["name"] == variant and cfg["experiment"]["name"] == experiment, "profile name differs from selected filename")
    run, ppo, distill = cfg["experiment"], cfg["mappo"], cfg["distillation"]
    blocks = (run["real_budget"] - run["initial_transitions"]) // run["collection_block"]
    cycles = ppo["initial_local_cycles"] + blocks * ppo["online_local_cycles"]
    guided = math.ceil(distill["decay_fraction"] * cycles) if variant == "D" else 0
    refresh_blocks = sum(ppo["initial_local_cycles"] + i * ppo["online_local_cycles"] < guided for i in range(blocks)) if variant == "D" else 0
    derived = {"online_blocks": blocks, "local_ppo_cycles": cycles,
               "nominal_local_imagined_joint_steps": cycles * ppo["starts"] * ppo["horizon"],
               "guided_local_cycles": guided, "unguided_local_cycles": cycles-guided,
               "teacher_ppo_cycles": (cfg["teacher"]["initial_cycles"] + refresh_blocks * cfg["teacher"]["online_cycles"]) if variant == "D" else 0,
               "comparison_run_count": len(cfg["study"]["task_families"]) * len(run["master_seeds"]) * 2,
               "checkpoint_count": len(run["checkpoints"])}
    import hashlib
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return ResolvedConfig(cfg, derived, canonical_hash(cfg), sources)


def validate_pair(left: ResolvedConfig, right: ResolvedConfig) -> None:
    """Both complete settings must match except the explicit variant switch."""
    validate_config(left.settings)
    validate_config(right.settings)
    a, b = deepcopy(left.settings), deepcopy(right.settings)
    _require({a.pop("variant")["name"], b.pop("variant")["name"]} == {"B", "D"}, "comparison requires a B/D pair")
    _require(a == b, "B/D common configuration mismatch (model, student, critic, schedules, task or budget)")
