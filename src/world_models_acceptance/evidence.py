"""Named acceptance requirements and exact source-bound CPU verification.

Passing test counts alone never grant acceptance. Each requirement names its
specific checks, and the gate verifies the current code, report and test log.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from world_models.data.ledger import atomic_json, file_sha256
from world_models.infrastructure.reproducibility import canonical_hash


ROOT = Path(__file__).resolve().parents[2]


def _ids(module, cls, *names):
    return tuple(f"tests.{module}.{cls}.{name}" for name in names)


REQUIRED_CHECKS = {
    "teacher_and_student_information_boundaries": (
        *_ids("test_teacher_policy", "GlobalTeacherTests",
              "test_global_tokens_and_identity_change_action_but_never_memory",
              "test_shared_tokens_cannot_enter_history_or_carry_gradients_into_world"),
        *_ids("test_local_actor", "LocalActorTests",
              "test_neural_actor_passes_environment_metadata_and_teammate_locality_audit"),
        *_ids("test_phase45_invariants", "Phase45CrossBoundaryTests",
              "test_privileged_sidecars_do_not_change_local_student_replay")),
    "prefix_alignment_and_executed_history": (
        *_ids("test_imagination_anchors", "ImaginationAnchorTests",
              "test_anchor_image_is_consumed_once_and_actor_output_does_not_encode_again"),
        *_ids("test_imagined_rollout_generator", "ImaginedRolloutGeneratorTests",
              "test_one_joint_prior_step_no_real_suffix_or_posterior_and_no_anchor_reencoding"),
        *_ids("test_teacher_learning", "TeacherLearningTests",
              "test_full_prefix_memory_consumes_anchor_once_and_matches_independent_player_histories",
              "test_teacher_rollout_never_reads_real_suffix_or_runs_posterior_and_rng_restores_on_failure"),
        *_ids("test_distillation_learning", "DistillationTests",
              "test_targets_temperature_and_execution_alignment_have_no_teacher_memory",
              "test_prefix_reconstructed_current_weights_anchor_consumed_once")),
    "teacher_targets_and_masks": _ids("test_distillation_learning", "DistillationTests",
              "test_teacher_replay_rejects_forged_logits_even_with_matching_version",
              "test_padding_and_terminal_actions_are_masked_without_discarding_terminals",
              "test_buffer_rejects_invalid_targets_actions_and_prefixes"),
    "kl_and_gradient_isolation": (
        *_ids("test_distillation_learning", "DistillationTests",
              "test_plain_forward_kl_matches_manual_valid_player_mean",
              "test_updates_change_only_student_and_preserve_modes_global_rng_and_buffer",
              "test_critic_must_be_frozen_and_no_gradient_state_imported"),
        *_ids("test_phase45_invariants", "Phase45CrossBoundaryTests",
              "test_zero_kl_and_detached_teacher_targets_have_no_teacher_gradient")),
    "paired_initialization_and_unguided_ppo": (
        *_ids("test_local_policy_initialization", "LocalPolicyInitializationTests",
              "test_paired_initialization_matches_with_disjoint_parameter_storage"),
        *_ids("test_refinement_runner", "RefinementRunnerTests",
              "test_prepare_imports_warm_started_actor_and_original_critic_with_fresh_adams",
              "test_collection_and_updater_consume_only_new_local_student_rollouts")),
    "durable_recovery_and_tamper_guards": (
        *_ids("test_distillation_learning", "DistillationTests",
              "test_resume_preserves_minibatch_rng_adam_and_result_exactly"),
        *_ids("test_refinement_runner", "RefinementRunnerTests",
              "test_real_local_rollout_and_ppo_resume_are_exact"),
        *_ids("test_campaign_controller", "CampaignControllerTests",
              "test_child_commit_survives_controller_failure_without_repeating_optimizer_work",
              "test_source_runtime_frozen_binding_and_manifest_receipt_changes_fail_closed",
              "test_child_checkpoint_tampering_cannot_fall_back_to_earlier_progress")),
    "frozen_inputs_accounting_and_finalization": (
        *_ids("test_campaign_controller", "CampaignControllerTests",
              "test_prepare_saves_paired_student_before_any_teacher_work",
              "test_full_fixed_schedule_runs_ordered_units_once_and_publishes_local_export",
              "test_final_export_publication_retries_without_repeating_any_learner_units"),
        *_ids("test_campaign_stages", "CampaignPhysicalWorkTests",
              "test_shared_journal_counts_each_stage_once_and_preserves_interrupted_work",
              "test_truncated_last_row_reports_unknown_compute_without_changing_bytes"),
        *_ids("test_phase45_audit", "Phase45AuditTests",
              "test_complete_campaign_readonly_without_optimizers_resume_or_queries",
              "test_complete_gate_audit_and_isolated_deployment_produce_engineering_only_acceptance")),
    "isolated_local_actor_deployment": _ids("test_phase45_invariants", "Phase45DeploymentTests",
              "test_fresh_process_executes_local_actor_with_learning_imports_blocked",
              "test_origin_device_hash_matches_registered_algorithm",
              "test_probe_rejects_full_controller_checkpoint_before_tensor_loading",
              "test_manifest_and_payload_tampering_rejected",
              "test_export_metadata_and_actor_versions_are_checked"),
}

# Run the named numerical/recurrent checks plus integration, gate and CLI suites.
# Related tests in each class remain adjacent to share expensive class fixtures.
_NAMED = tuple(dict.fromkeys(test for group in REQUIRED_CHECKS.values() for test in group))
_FULL_MODULES = ("tests.test_phase45_audit", "tests.test_phase45_invariants",
                 "tests.test_phase45_gate", "tests.test_phase45_cli", "tests.test_campaign_controller")
TEST_MODULES = tuple(sorted(test for test in _NAMED if not any(test.startswith(module + ".") for module in _FULL_MODULES))) + _FULL_MODULES


def verification_contract_identity():
    return canonical_hash({"requirements": REQUIRED_CHECKS, "test_modules": TEST_MODULES})


def requirement_coverage(passed_ids):
    passed = set(passed_ids)
    return {category: {"status": "passed" if set(required) <= passed else "missing",
            "test_ids": list(required), "missing_test_ids": sorted(set(required) - passed)}
            for category, required in REQUIRED_CHECKS.items()}


def source_identity():
    paths = []
    for name in ("world_models", "world_models_campaign", "world_models_acceptance"):
        paths.extend((ROOT / "src" / name).rglob("*.py"))
    paths.extend(ROOT / f"scripts/phase{number}.py" for number in (41, 42, 43, 44, 45))
    paths.extend((ROOT / "tests").rglob("*.py"))
    paths.append(ROOT / "configs/studies/relay_four_day_v1.yaml")
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sorted(set(paths))}


def read_report(path):
    path = Path(path)
    receipt = path.with_suffix(".sha256")
    if (not path.is_file() or path.is_symlink() or not receipt.is_file() or receipt.is_symlink()
            or receipt.read_text(encoding="ascii").strip() != f"{file_sha256(path)}  {path.name}"):
        raise ValueError(f"acceptance evidence has no valid receipt: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if type(result) is not dict:
        raise ValueError("acceptance evidence must be a JSON object")
    return result


def write_report(path, report):
    path = Path(path)
    if path.exists() or path.with_suffix(".sha256").exists():
        raise FileExistsError("acceptance report must have a fresh destination")
    atomic_json(path, report)
    content = f"{file_sha256(path)}  {path.name}\n"
    path.with_suffix(".sha256").write_text(content, encoding="ascii")
    return path


def load_valid_verification(path):
    """Validate explicit requirements, exact current sources and the test log."""
    path = Path(path).resolve()
    report = read_report(path)
    if (report.get("schema_version") != "p45-implementation-v1"
            or report.get("status") != "complete_p45_implementation"
            or report.get("source_unchanged_during_verification") is not True
            or report.get("verification_contract_sha256") != verification_contract_identity()
            or report.get("test_modules") != list(TEST_MODULES)
            or report.get("source_sha256") != source_identity()):
        raise ValueError("verification schema, contract, sources or completion is stale or invalid")
    for key in ("failures", "errors", "skipped"):
        if type(report.get(key)) is not int or report[key] != 0:
            raise ValueError(f"verification {key} must be zero")
    for key in ("research_training_executed", "research_replay_used", "gpu_runtime_measured", "phase4_acceptance"):
        if report.get(key) is not False:
            raise ValueError("verification must remain separate from research execution and acceptance")
    passed = report.get("passed_tests")
    if (type(passed) is not list or any(type(test) is not str for test in passed)
            or type(report.get("tests_run")) is not int
            or len(passed) != len(set(passed)) or len(passed) != report["tests_run"]):
        raise ValueError("verification passed-test identities or counters differ")
    coverage = requirement_coverage(passed)
    if (report.get("requirement_coverage") != coverage or report.get("all_requirements_passed") is not True
            or any(row["status"] != "passed" for row in coverage.values())):
        raise ValueError("verification is missing required Phase 4 engineering evidence")
    log = report.get("tests_log")
    if type(log) is not dict or log.get("path") != "tests.txt":
        raise ValueError("verification must bind its adjacent tests.txt log")
    log_path = path.parent / "tests.txt"
    if log_path.is_symlink() or file_sha256(log_path) != log.get("sha256"):
        raise ValueError("verification test-log checksum differs")
    content = log_path.read_text(encoding="utf-8")
    # Python versions render either method (module.Class.method) or the
    # earlier method (module.Class). Authenticate both without duplicating it.
    logged = [entry if entry.endswith("." + method) else f"{entry}.{method}" for method, entry in re.findall(
        r"^(\S+) \(([^\n]+)\) \.\.\. ok$", content, flags=re.MULTILINE)]
    if (len(logged) != len(set(logged)) or set(logged) != set(passed)
            or not re.search(rf"^Ran {report['tests_run']} tests? in ", content, flags=re.MULTILINE)
            or not re.search(r"^OK\s*$", content, flags=re.MULTILINE)):
        raise ValueError("verification test-log identities, total or result differ")
    return report, coverage, {"path": str(path), "sha256": file_sha256(path),
        "tests_log_sha256": file_sha256(log_path), "verified_at": datetime.now(timezone.utc).isoformat()}
