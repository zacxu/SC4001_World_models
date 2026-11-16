"""Deterministic static final-test bank, with no simulator or actor imports."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib

from world_models.env.splits import SPEC_PATH, get_template, list_templates, load_split_spec, randomization_draw
from world_models.infrastructure.reproducibility import canonical_hash, canonical_json
from .protocol import load_protocol, protocol_identity


_NAMESPACE = "p51-final-test-bank-v1"


def _component_seed(episode, component, **extra):
    return int(canonical_hash({"namespace": _NAMESPACE, "task_family": "relay", "bank_seed": 5600,
                              "episode_index": episode, "component": component, **extra})[:8], 16)


def evaluation_action_seed(master_seed, episode, player):
    """Pair B/D action randomness while retaining separate players and seeds."""
    if (type(master_seed) is not int or master_seed not in (0, 1)
            or type(episode) is not int or not 0 <= episode < 20
            or type(player) is not int or player not in (0, 1)):
        raise ValueError("test action seeds require master seed 0/1, episode 0..19 and player 0/1")
    return _component_seed(episode, "local_action_sampling", master_seed=master_seed, player=player)


def build_test_bank():
    """Describe the bank and nuisance choices without building/opening scenarios."""
    p = load_protocol()
    spec = load_split_spec()
    templates = list_templates("relay", "test")
    if [row["id"] for row in templates] != p["evaluation_layouts"]:
        raise ValueError("final test layouts differ from the fixed two-layout endpoint registration")
    episodes, all_seeds = [], []
    for index in range(20):
        template_index = index % 2
        template, nuisance_bank, provenance = get_template("relay", templates[template_index]["id"],
                                                           split="test", purpose="test")
        layout, engine = _component_seed(index, "layout"), _component_seed(index, "engine")
        action_seeds = {str(seed): [evaluation_action_seed(seed, index, player) for player in (0, 1)]
                        for seed in (0, 1)}
        all_seeds.extend([layout, engine, *action_seeds["0"], *action_seeds["1"]])
        episodes.append({"episode_index": index, "template_id": template["id"], "template_index": template_index,
            "layout_seed": layout, "engine_seed": engine, "action_seeds_by_master_seed": action_seeds,
            "geometry": deepcopy(template["geometry"]), "geometry_sha256": canonical_hash(template["geometry"]),
            "nuisance_draw": randomization_draw("relay", layout, nuisance_bank), "split_provenance": provenance})
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("a fixed final-bank RNG seed collision requires a new explicitly versioned registration")
    if Counter(row["template_id"] for row in episodes) != Counter({name: 10 for name in p["evaluation_layouts"]}):
        raise ValueError("final bank must contain exactly ten episodes per test layout")
    # Detect input changes while deriving a manifest, rather than binding a
    # mixture of two split specifications during concurrent synchronization.
    if canonical_hash(load_split_spec()) != canonical_hash(spec):
        raise ValueError("split registry changed during bank construction")
    return {"schema_version": _NAMESPACE, "task_family": "relay", "split": "test", "purpose": "test",
        "bank_seed": 5600, "episodes_per_actor": 20, "episodes_per_layout": 10,
        "master_seeds": [0, 1], "protocol_sha256": protocol_identity(),
        "split_registry_path": SPEC_PATH.relative_to(SPEC_PATH.parents[1]).as_posix(),
        "split_registry_file_sha256": hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
        "split_manifest_sha256": canonical_hash(spec), "split_version": spec["version"],
        "randomization_version": spec["randomization_version"],
        "action_seed_policy": "master_seed_episode_player_excluding_variant",
        "scenario_seed_policy": "bank_episode_shared_across_all_master_seeds_and_variants",
        "mode": "stochastic", "temperature": 1.0, "actor_precision": "float32",
        "frame_skip": 4, "deadline": 400, "learning_eligible": False,
        "scenario_built": False, "native_environment_opened": False, "native_evaluation_executed": False,
        "test_bank_exposure_claim": "reserved by this protocol; historical nonuse requires separate provenance review",
        "geometry_hash_scope": "declarative authored geometry, not a compiled scenario or native measurement",
        "episodes": episodes}


def validate_test_bank(bank):
    """Reject subsets, altered streams, layouts, metadata or input identities."""
    if type(bank) is not dict or canonical_json(bank) != canonical_json(build_test_bank()):
        raise ValueError("test bank differs from the exact registered deterministic manifest")
    return deepcopy(bank)
