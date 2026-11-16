"""Read-only binding of the two seeds' existing shared initial study inputs.

The original P2.4 registration remains authoritative, including its four-case
selection. Nothing here constructs a network, restores an optimizer, samples a
policy or writes to a dataset. The fixed-data amendment is recorded separately
from the historical full pilot configuration used to verify saved checkpoints.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

from world_models.data.ledger import file_sha256
from world_models.infrastructure.config import resolve_config, validate_pair
from world_models.infrastructure.reproducibility import canonical_hash


ROOT = Path(__file__).resolve().parents[2]


def _load_initial_inputs(**kwargs):
    """Lazy original validator; tests replace this dependency, never its rules."""
    from world_models.teacher.runner import _initial_inputs
    return _initial_inputs(**kwargs)


def _absolute(path):
    raw = Path(path)
    if any(member.is_symlink() for member in (raw, *raw.parents)):
        raise ValueError(f"study input paths cannot traverse symbolic links: {raw}")
    return raw.resolve()


class _Files:
    """Cache each binder-owned digest once, then perform one final recheck."""
    def __init__(self):
        self.digests = {}

    def add(self, path, *, expected=None, receipt=False):
        path = _absolute(path)
        if not path.is_file():
            raise ValueError(f"required immutable study input is missing: {path}")
        name = str(path)
        if name not in self.digests:
            self.digests[name] = file_sha256(path)
        digest = self.digests[name]
        if expected is not None and (type(expected) is not str or re.fullmatch(r"[a-f0-9]{64}", expected) is None
                                     or digest != expected):
            raise ValueError(f"study input does not match its recorded SHA256: {path}")
        marker = path.with_suffix(".sha256")
        if marker != path and (receipt or marker.is_file()):
            self.add(marker)
            if marker.read_text(encoding="ascii").strip() != f"{digest}  {path.name}":
                raise ValueError(f"study input receipt mismatch: {path}")
        return {"path": name, "sha256": digest}

    def finish(self):
        rows = [{"path": path, "sha256": digest} for path, digest in sorted(self.digests.items())]
        for row in rows:
            path = _absolute(row["path"])
            if not path.is_file() or file_sha256(path) != row["sha256"]:
                raise ValueError(f"study input changed while being bound: {path}")
        return rows


def _bind_legacy_selection(path, files):
    """Retain every historical dependency required by the existing loader."""
    from world_models.training.selection import _resolve
    files.add(path, receipt=True)
    document = json.loads(path.read_text(encoding="utf-8"))
    cases = document.get("cases")
    if (type(cases) is not list or [(row.get("task"), row.get("seed")) for row in cases]
            != [("relay", 0), ("relay", 1), ("cover", 0), ("cover", 1)]):
        raise ValueError("the original four-case P2.4 selection must remain intact")
    run = _resolve(path.parent, document["source_run_root"])
    files.add(run / "launch_plan.json", expected=document["source_launch_plan_sha256"])
    for row in cases:
        files.add(_resolve(path.parent, row["checkpoint_path"]),
                  expected=row["checkpoint_sha256"], receipt=True)
        files.add(_resolve(path.parent, row["run_manifest_path"]), expected=row["run_manifest_sha256"])
        for name, expected in row["source_sha256"].items():
            # The upstream loader compares this exact dependency map against
            # its executable source identity; retain those bytes as well.
            files.add(ROOT / name, expected=expected)
    return document


def _shared_binding(path, *, variant, dataset, files):
    record = files.add(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    encoded = value.get("dataset_path")
    if type(encoded) is not str or not encoded.strip() or "\0" in encoded:
        raise ValueError("shared replay binding has no valid dataset path")
    member = Path(encoded.replace("\\", "/"))
    target = _absolute(member if member.is_absolute() else path.parent / member)
    if (value.get("variant") != variant or target != _absolute(dataset.path)
            or value.get("manifest_sha256") != dataset.manifest_sha256
            or value.get("dataset_identity_sha256") != dataset.manifest["dataset_identity_sha256"]
            or value.get("attributed_initial_joint_transitions") != 50000
            or value.get("physical_collection_shared") is not True
            or type(value.get("new_physical_transitions")) is not int or value["new_physical_transitions"] != 0):
        raise ValueError("B/D replay bindings must identify the same 50000-transition physical collection")
    return {**record, "binding": value}


def bind_initial_inputs(*, initial_selection, data_root, statistics_root, config_root):
    """Bind two immutable Relay inputs; all new research work remains zero.

    This may read many existing replay/checkpoint bytes. Upstream independent
    validation is deliberately preserved, even where its legacy four-case
    selection check repeats I/O. No runtime or model numerical work is measured.
    """
    from .protocol import load_protocol

    roots = {key: str(_absolute(value)) for key, value in {
        "initial_selection": initial_selection, "data_root": data_root,
        "statistics_root": statistics_root, "config_root": config_root}.items()}
    protocol = load_protocol()
    if (protocol.get("master_seeds") != [0, 1] or protocol.get("initial_transitions") != 50000
            or protocol.get("initial_model_updates") != 4500 or protocol.get("variants") != ["B", "D"]):
        raise ValueError("the study input binder requires the fixed two-seed Relay protocol")
    files, per_seed = _Files(), {}
    path = Path(roots["initial_selection"])
    selection_document = None
    for seed in (0, 1):
        settings, dataset, stats, payload, binding = _load_initial_inputs(**roots, master_seed=seed)
        selected = binding.get("selection", {})
        normalization = stats.as_dict()
        provenance = binding.get("provenance", {})
        replay_record = {"manifest_sha256": dataset.manifest_sha256,
            "dataset_identity_sha256": dataset.manifest.get("dataset_identity_sha256"),
            "transitions": dataset.manifest.get("transition_count")}
        metadata = dataset.manifest.get("metadata", {})
        if (settings.get("task", {}).get("name") != "relay" or settings.get("variant", {}).get("name") != "D"
                or settings.get("experiment", {}).get("initial_transitions") != 50000
                or selected.get("task") != "relay" or type(selected.get("seed")) is not int or selected["seed"] != seed
                or type(selected.get("model_updates")) is not int or selected["model_updates"] != 4500
                or payload.get("update") != 4500 or selected.get("shared_initial_variants") != ["B", "D"]
                or metadata.get("task_family") != "relay" or type(metadata.get("run_seed")) is not int
                or metadata["run_seed"] != seed or metadata.get("learning_eligible") is not True
                or metadata.get("data_source") != "real_environment" or replay_record["transitions"] != 50000
                or provenance.get("replays") != [replay_record] or selected.get("replays") != [replay_record]
                or selected.get("initial_normalization") != provenance.get("initial_normalization")
                or normalization.get("identity_sha256") != provenance.get("initial_normalization", {}).get("identity_sha256")
                or _absolute(selected.get("selection_path", "")) != path
                or _absolute(binding.get("replay_path", "")) != _absolute(dataset.path)):
            raise ValueError(f"Relay seed {seed} is not the exact shared 4500-update / 50000-transition input")
        # Restricted CPU checkpoint deserialization was performed upstream.
        # Do not retain its model or Adam tensors in a protocol registration.
        del payload
        if selection_document is None:
            selection_document = _bind_legacy_selection(path, files)
        selected_file = files.add(selected["checkpoint_path"], expected=selected["checkpoint_sha256"], receipt=True)
        files.add(path, expected=selected["selection_sha256"], receipt=True)
        files.add(selected["run_manifest_path"], expected=selected["run_manifest_sha256"])
        data_path = _absolute(dataset.path)
        files.add(data_path / "manifest.json", expected=dataset.manifest_sha256, receipt=True)
        for member in sorted(data_path.rglob("*")):
            if member.is_file():
                files.add(member)
        normalization_path = Path(roots["statistics_root"]) / "relay" / f"seed_{seed}.json"
        normalization_file = files.add(normalization_path,
            expected=provenance["initial_normalization"]["file_sha256"], receipt=True)
        configs = {variant: resolve_config("relay", variant, "pilot", roots["config_root"]) for variant in ("B", "D")}
        validate_pair(configs["B"], configs["D"])
        if settings != configs["D"].settings:
            raise ValueError("loaded historical settings differ from the immutable original pilot parent")
        for config in configs.values():
            for name, expected in config.source_hashes.items():
                files.add(Path(roots["config_root"]) / name.replace("\\", "/"), expected=expected)
        branches = {variant: _shared_binding(Path(roots["data_root"]) / "relay" / f"seed_{seed}" /
                    "bindings" / f"{variant}.json", variant=variant, dataset=dataset, files=files)
                    for variant in ("B", "D")}
        scientific_recipe = {name: deepcopy(settings[name]) for name in
                             ("environment", "world_model", "actor", "critic", "mappo", "teacher", "distillation", "task")}
        fixed_data = {key: deepcopy(protocol[key]) for key in
            ("initial_transitions", "initial_model_updates", "local_cycles", "starts", "horizon", "epochs", "minibatches",
             "teacher_cycles", "trajectories", "distillation_updates", "distillation_sequences_per_update",
             "distillation_trajectory_source", "distillation_temperature", "distillation_loss", "distillation_optimizer",
             "local_ppo_optimizer_initialization", "local_actor_compute_precision", "teacher_policy_compute_precision",
             "world_model_compute_precision", "teacher_kl_weight", "teacher_refresh_cycles", "guidance_decay_enabled",
             "evaluation_split", "evaluation_episodes_per_actor", "evaluation_bank_seed", "evaluation_mode",
             "evaluation_temperature", "evaluation_actor_precision", "evaluation_final_local_cycles", "primary_metric")}
        fixed_data.update(new_environment_interactions=0, new_model_updates=0,
                          model_and_normalization_frozen=True)
        per_seed[str(seed)] = {"master_seed": seed, "task_family": "relay", "variants": ["B", "D"],
            "selected_model": {**selected_file, "model_updates": 4500,
                "historical_target_updates": selected["original_target_stage_updates"],
                "weight_identity_scope": "complete committed checkpoint bytes; no device-labelled module version constructed",
                "checkpoint_metadata_validated_without_model_construction": True,
                "original_selection_record": deepcopy(selected)},
            "replay": {"path": str(data_path), **replay_record, "physical_joint_transitions": 50000,
                       "physical_collection_shared": True, "branch_bindings": branches},
            "normalization": {**normalization_file, "identity_sha256": normalization["identity_sha256"],
                              "statistics": normalization, "frozen": True},
            "original_parent": {"profile": "pilot", "settings_role": "historical_checkpoint_validation_only",
                "resolved_by_variant": {variant: config.as_dict() for variant, config in configs.items()},
                "loaded_D_settings_sha256": canonical_hash(settings)},
            "scientific_recipe": scientific_recipe, "scientific_recipe_sha256": canonical_hash(scientific_recipe),
            "scientific_recipe_role": "historical pilot settings only; effective_fixed_data_contract supplies the endpoint overrides",
            "architecture_sha256": canonical_hash({key: scientific_recipe[key] for key in
                ("environment", "world_model", "actor", "critic", "task")}),
            "loss_and_optimizer_settings_sha256": canonical_hash({key: scientific_recipe[key] for key in
                ("world_model", "mappo", "teacher", "distillation")}),
            "effective_fixed_data_contract": fixed_data,
            "effective_contract_role": "separate endpoint amendment; original pilot schedules remain historical",
            "initial_binding_sha256": canonical_hash(binding), "initial_binding": deepcopy(binding)}
    if (per_seed["0"]["replay"]["path"] == per_seed["1"]["replay"]["path"]
            or per_seed["0"]["replay"]["dataset_identity_sha256"] == per_seed["1"]["replay"]["dataset_identity_sha256"]
            or per_seed["0"]["selected_model"]["path"] == per_seed["1"]["selected_model"]["path"]):
        raise ValueError("the two master seeds must retain their distinct physical replay and selected checkpoint")
    return {"schema_version": "p51-shared-initial-inputs-v1", "input_roots": roots, "per_seed": per_seed,
        "physical_initial_joint_transitions": 100000, "per_run_attributed_initial_joint_transitions": 50000,
        "sum_four_run_attributed_initial_joint_transitions": 200000,
        "physical_selected_model_optimizer_updates": 9000, "sum_four_run_attributed_model_optimizer_updates": 18000,
        "original_four_case_selection_preserved": True, "research_training_executed": False,
        "model_constructed": False, "model_inference_executed": False, "optimizer_constructed_or_restored": False,
        "native_environment_evaluation_executed": False, "new_environment_interactions": 0, "new_model_updates": 0,
        "bound_files": files.finish()}
