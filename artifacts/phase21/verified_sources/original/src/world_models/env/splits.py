"""Versioned geometry holdouts and explicit environment-data provenance.

The same nuisance banks are used across splits to isolate the declared geometry
holdout. Selecting an evaluation or diagnostic layout never authorizes replay
insertion. Future collectors must account for any diagnostic import separately.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import yaml

from world_models.infrastructure.config import _UniqueKeyLoader


ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "configs" / "environment_splits.yaml"
SPLITS = ("train", "validation", "test")
PURPOSES = ("training", "validation", "test", "diagnostic")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_split_spec() -> dict:
    spec = yaml.load(SPEC_PATH.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("unsupported environment split schema")
    if spec.get("split_counts") != {"train": 6, "validation": 2, "test": 2}:
        raise ValueError("environment split must allocate six/two/two templates")
    if set(spec.get("templates", {})) != {"relay", "cover"}:
        raise ValueError("split specification must cover exactly relay and cover")
    for bank in spec.get("banks", {}).values():
        if bank.get("role_assignment") != "seeded_swap" or bank.get("task_family") not in ("relay", "cover"):
            raise ValueError("unsupported randomization bank")
        for name in ("spawn_pairs", "wall_textures", "enemy_positions"):
            if not isinstance(bank.get(name), list) or not bank[name]:
                raise ValueError(f"randomization bank needs a nonempty {name}")
        if any(set(pair) != {"operator", "runner"} or any(
                len(point) != 2 or any(type(value) is not int for value in point) for point in pair.values())
               for pair in bank["spawn_pairs"]):
            raise ValueError("spawn bank needs integer operator/runner coordinate pairs")
        expected_enemies = 0 if bank["task_family"] == "relay" else 2
        if any(len(enemies) != expected_enemies or any(
                len(point) != 2 or any(type(value) is not int for value in point) for point in enemies)
               for enemies in bank["enemy_positions"]):
            raise ValueError("enemy bank must preserve the declared enemy count and integer coordinates")
        if any(not isinstance(texture, str) or not texture for texture in bank["wall_textures"]):
            raise ValueError("wall texture identifiers must be nonempty strings")
    names = set()
    for family, templates in spec["templates"].items():
        if Counter(template["split"] for template in templates) != Counter(spec["split_counts"]):
            raise ValueError(f"invalid template allocation for {family}")
        geometries = set()
        for template in templates:
            if template["id"] in names or not template["id"].startswith(family + "_"):
                raise ValueError("template identifiers must be unique and task-scoped")
            names.add(template["id"])
            bank = spec["banks"].get(template["bank"])
            if not bank or bank["task_family"] != family:
                raise ValueError("template references an incompatible randomization bank")
            signature = _digest(template["geometry"])
            if signature in geometries:
                raise ValueError("geometry duplicated across template identifiers")
            geometries.add(signature)
            geometry = template["geometry"]
            if family == "relay":
                if set(geometry) != {"common_x", "gate_x", "obstacles"}:
                    raise ValueError("invalid relay geometry fields")
                if geometry["common_x"] % 64 or geometry["gate_x"] % 64 or not 192 <= geometry["gate_x"] < geometry["common_x"] - 256:
                    raise ValueError("relay dimensions violate gate/route bounds")
                obstacles = geometry["obstacles"]
            else:
                if set(geometry) != {"width", "pillars"} or geometry["width"] < 960 or geometry["width"] % 64:
                    raise ValueError("invalid cover geometry fields")
                obstacles = geometry["pillars"]
            if len({tuple(point) for point in obstacles}) != len(obstacles) or any(
                    len(point) != 2 or any(type(value) is not int or value % 64 for value in point) for point in obstacles):
                raise ValueError("obstacles must be unique 64-unit grid cells")
    return spec


def list_templates(task_family: str, split: str | None = None) -> list[dict]:
    if task_family not in ("relay", "cover") or split is not None and split not in SPLITS:
        raise ValueError("unknown task family or split")
    spec = load_split_spec()
    return deepcopy([template for template in spec["templates"][task_family] if split is None or template["split"] == split])


def get_template(task_family: str, template_id: str, *, split: str | None = None,
                 purpose: str = "diagnostic", randomization_bank: str | None = None) -> tuple[dict, dict, dict]:
    if purpose not in PURPOSES:
        raise ValueError("unknown scenario purpose")
    if task_family not in ("relay", "cover") or split is not None and split not in SPLITS:
        raise ValueError("unknown task family or split")
    spec = load_split_spec()
    templates = [template for template in spec["templates"][task_family] if split is None or template["split"] == split]
    matches = [template for template in templates if template["id"] == template_id]
    if len(matches) != 1:
        raise ValueError("template does not belong to the declared task/split")
    template = deepcopy(matches[0])
    required_split = {"training": "train", "validation": "validation", "test": "test"}.get(purpose)
    if required_split is not None and template["split"] != required_split:
        raise ValueError(f"{purpose} may not select {template['split']} geometry")
    if randomization_bank is not None and randomization_bank != template["bank"]:
        raise ValueError("randomization bank differs from the versioned template allocation")
    bank = deepcopy(spec["banks"][template["bank"]])
    provenance = {"split_version": spec["version"], "split_manifest_sha256": _digest(spec),
                  "template_id": template_id, "split": template["split"], "purpose": purpose,
                  "randomization_version": spec["randomization_version"],
                  "randomization_bank": template["bank"], "randomization_bank_version": bank["version"],
                  "randomization_bank_sha256": _digest(bank),
                  "learning_eligible": purpose == "training", "diagnostic_data_enters_replay": False,
                  "diagnostic_import_requires_training_ledger": True,
                  "geometry_scope": spec["geometry_scope"]}
    return template, bank, provenance


def randomization_draw(task_family: str, seed: int, bank: dict) -> dict:
    if type(seed) is not int or seed < 0:
        raise ValueError("randomization seed must be a nonnegative integer")
    if bank["task_family"] != task_family:
        raise ValueError("randomization bank task mismatch")

    def choose(name, choices):
        # Deliberately omit template ID: nuisance draws can be paired across
        # geometries without changing the held-out geometry allocation.
        key = f"{bank['version']}|{task_family}|{name}|{seed}".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
        index = rng.randrange(len(choices))
        return index, deepcopy(choices[index])

    role_index, operator = choose("roles", [0, 1])
    spawn_index, pair = choose("spawns", bank["spawn_pairs"])
    texture_index, wall = choose("textures", bank["wall_textures"])
    enemy_index, enemies = choose("enemies", bank["enemy_positions"])
    return {"operator": operator, "spawn_pair": pair, "wall_texture": wall, "enemy_positions": enemies,
            "indices": {"role": role_index, "spawn": spawn_index, "texture": texture_index, "enemy": enemy_index}}


def build_split_scenario(task_family: str, split: str, template_index: int = 0, *, seed: int = 0,
                         purpose: str = "diagnostic", frame_skip: int = 4,
                         output_dir: str | Path | None = None) -> dict:
    if type(template_index) is not int or template_index < 0:
        raise ValueError("template index must be a nonnegative integer")
    templates = list_templates(task_family, split)
    if template_index >= len(templates):
        raise ValueError("template index outside the declared split")
    from world_models.env.scenario_builder import build_scenario
    return build_scenario(task_family, seed, frame_skip, output_dir,
                          template_id=templates[template_index]["id"], split=split, purpose=purpose)


def split_manifest() -> dict:
    """Publish actual geometry hashes/checks as well as the declarative banks."""
    from world_models.env.scenario_builder import _template_parameters, geometry_fingerprint, validate_geometry
    spec = load_split_spec()
    records = []
    for family in ("relay", "cover"):
        for template in spec["templates"][family]:
            params, cells, _ = _template_parameters(family, 0, template["id"])
            checks = validate_geometry(params, cells)
            bank = spec["banks"][template["bank"]]
            bank_combinations = 0
            for pair in bank["spawn_pairs"]:
                for operator in (0, 1):
                    for enemies in bank["enemy_positions"]:
                        trial = deepcopy(params)
                        trial["role_assignment"] = {"operator": operator, "runner": 1 - operator}
                        trial["spawns"] = [pair["operator"], pair["runner"]] if operator == 0 else [pair["runner"], pair["operator"]]
                        if family == "cover":
                            trial["enemy_positions"] = enemies
                        validate_geometry(trial, cells)
                        bank_combinations += 1
            records.append({"task_family": family, "template_id": template["id"], "split": template["split"],
                            "geometry_sha256": geometry_fingerprint(params, cells), "checks": checks,
                            "randomization_bank": template["bank"],
                            "all_spawn_role_enemy_bank_combinations_validated": bank_combinations})
    for family in ("relay", "cover"):
        signatures = [record["geometry_sha256"] for record in records if record["task_family"] == family]
        if len(set(signatures)) != 10:
            raise ValueError("formal templates do not have ten distinct realized geometries")
    if _digest(load_split_spec()) != _digest(spec):
        raise ValueError("split specification changed during manifest generation")
    return {"schema_version": 1, "version": spec["version"], "sha256": _digest(spec),
            "specification_path": str(SPEC_PATH.resolve()), "specification": spec, "templates": records,
            "geometry_disjoint_across_splits": True,
            "nuisance_distribution": "same versioned spawn/texture/enemy banks across geometry splits",
            "scope": spec["geometry_scope"], "evaluation_data_enters_replay": False,
            "diagnostic_import_requires_training_ledger": True}


def write_split_manifest(path: str | Path) -> dict:
    manifest = split_manifest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
