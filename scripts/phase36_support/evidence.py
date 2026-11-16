"""Immutable acceptance evidence and explicit coverage of the Phase 3 invariants.

This module uses only the standard library. In particular, recording evaluation
provenance must not import a world model or construct training networks.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

INVARIANT_TESTS = {
    "gae": (
        "tests.test_mappo_objectives.MAPPOObjectiveTests.test_hand_calculated_multistep_terminal_and_alive_cutoff",
        "tests.test_mappo_objectives.MAPPOObjectiveTests.test_raw_returns_precede_global_joint_sample_advantage_normalization",
    ),
    "clock_and_reward": (
        "tests.test_contracts.ClockAndBoundaryTests.test_death_success_deadline_and_collection_precedence",
        "tests.test_contracts.RewardAndTransitionTests.test_reward_decomposes_for_success_death_and_deadline",
        "tests.test_model_heads.TaskHeadsTests.test_joint_outcome_truth_table_and_deadline_precedence",
        "tests.test_model_heads.TaskHeadsTests.test_reward_labels_separate_non_deadline_outcome_from_clock",
        "tests.test_imagined_rollout_generator.ImaginedRolloutGeneratorTests.test_exact_deadline_and_learned_terminal_take_precedence_over_horizon",
    ),
    "behavior_replay": (
        "tests.test_mappo_recurrent.MappoRecurrentTests.test_full_prefix_replays_frozen_behavior_with_individual_ratios_one",
        "tests.test_mappo_recurrent.MappoRecurrentTests.test_current_weights_reconstruct_prefix_and_stored_memories_are_ignored",
        "tests.test_imagined_rollout_generator.ImaginedRolloutGeneratorTests.test_stored_behavior_likelihood_and_recurrent_replay_begin_with_ratio_one",
    ),
    "locality": (
        "tests.test_local_actor.LocalActorTests.test_neural_actor_passes_environment_metadata_and_teammate_locality_audit",
        "tests.test_local_actor.LocalActorTests.test_weights_are_shared_but_agent_memories_are_independent",
        "tests.test_mappo_recurrent.MappoRecurrentTests.test_own_history_locality_and_reset_are_preserved",
    ),
    "frozen_world_model": (
        "tests.test_mappo_updater.RecurrentMAPPOTests.test_update_changes_only_policy_networks_and_uses_all_whole_sequences_each_epoch",
        "tests.test_mappo_updater.RecurrentMAPPOTests.test_update_preserves_world_gradients_modes_global_rng_and_sampler_state",
    ),
    "padding_resets_and_rng": (
        "tests.test_local_actor.LocalActorTests.test_start_resets_memory_and_blocks_previous_episode_gradients",
        "tests.test_local_actor.LocalActorTests.test_padding_skips_encoding_and_invalid_action_slots_without_changing_memory",
        "tests.test_rollout_streams.RolloutStreamTests.test_agent_and_posterior_draws_cannot_advance_other_components",
        "tests.test_imagined_rollout_generator.ImaginedRolloutGeneratorTests.test_mixed_lengths_freeze_finished_world_rows_and_never_draw_padding_actions",
    ),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def source_identity() -> dict[str, str]:
    """Bind production, acceptance, config, scenario and dependency source bytes.

    Test files are recorded separately by ``verify`` so deployed evaluation can
    use this same identity without importing or depending on the tests package.
    Earlier P3.1-P3.5 receipts remain untouched by these additive CLI tools.
    """
    files = set((ROOT / "src/world_models").rglob("*.py"))
    files.update((ROOT / "configs").rglob("*.yaml"))
    files.update((ROOT / "configs").rglob("*.json"))
    files.update((ROOT / "scenarios").rglob("*.acs"))
    files.update((ROOT / "scripts/phase36_support").glob("*.py"))
    files.update(ROOT / f"scripts/phase{number}.py" for number in range(31, 37))
    files.add(ROOT / "scripts/phase24_freeze.py")
    files.update(ROOT.glob("requirements*.lock"))
    files.update(ROOT.glob("requirements*.txt"))
    files.add(ROOT / "pyproject.toml")
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sorted(files)}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _path(path) -> Path:
    candidate = Path(path)
    if ".." in candidate.parts or any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError("evidence path cannot traverse parents or symbolic links")
    return candidate.resolve()


def write_report(path: str | Path, report: dict) -> Path:
    """Publish a new JSON file and final SHA256 receipt; never replace evidence."""
    if type(report) is not dict:
        raise ValueError("report must be a JSON object")
    content = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    destination = _path(path)
    if destination.suffix != ".json":
        raise ValueError("report destination must end in .json")
    receipt = destination.with_suffix(".sha256")
    temporary = receipt.with_suffix(".sha256.tmp")
    if any(item.exists() for item in (destination, receipt, temporary)):
        raise FileExistsError("preserve existing evidence; use a fresh output path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    with temporary.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{hashlib.sha256(content).hexdigest()}  {destination.name}\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, receipt)
    return destination


def load_report(path: str | Path) -> dict:
    """Verify exact bytes before parsing; partial or edited evidence is invalid."""
    source = _path(path)
    receipt = _path(source.with_suffix(".sha256"))
    content = source.read_bytes()
    expected = f"{hashlib.sha256(content).hexdigest()}  {source.name}"
    if receipt.read_text(encoding="ascii").strip() != expected:
        raise ValueError(f"evidence receipt differs: {source.name}")
    result = json.loads(content, object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    if type(result) is not dict:
        raise ValueError("evidence must contain a JSON object")
    return result
