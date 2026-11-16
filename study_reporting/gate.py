"""P5.8 evidence-completion gate. Never launches an optimizer or a simulator.

The caller authenticates the frozen cohort, actors, traces, reproduced endpoint
analysis and final ledger. This module checks their agreement and reads only
the exact, hash-bound attempt-journal prefixes retained by that ledger. A pass
means the registered experiment is complete; it is independent of D winning.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA = "p58-phase5-acceptance-v1"
AUTHENTICATION = ("cohort", "analysis", "final_ledger", "final_actors",
                  "evaluation_traces", "input_hashes", "frozen_sources")
CASES = tuple((variant, seed) for seed in (0, 1) for variant in ("B", "D"))
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_NUMERICAL = re.compile(r"non[ -]?finite|\bnan\b|\binfinity\b|\binfinite\b|"
                        r"gradient.*(?:overflow|invalid)|loss.*(?:overflow|invalid)", re.I)
_INFRA = re.compile(r"CUDA out of memory|driver shutting down|CUDA.*(?:unavailable|busy)|"
                    r"no space left on device|disk quota exceeded|stale file handle|"
                    r"connection (?:reset|timed out|aborted)|transport endpoint|"
                    r"input/output error|device or resource busy|broken pipe", re.I)


def _integer(value, name, target=None, maximum=None):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be an explicit nonnegative integer")
    if target is not None and value != target:
        raise ValueError(f"{name} differs from the registered value {target}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds the registered maximum {maximum}")
    return value


def _sha(value, name):
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError(f"{name} has no SHA-256 identity")
    return value


def _case(row):
    variant, seed = row.get("variant"), row.get("master_seed")
    if type(seed) is not int or (variant, seed) not in CASES:
        raise ValueError("a run lies outside the four registered variant/seed slots")
    return variant, seed


def _index(rows, name):
    if type(rows) is not list or len(rows) != 4:
        raise ValueError(f"{name} needs all four registered slots; missing evidence is not zero")
    result = {}
    for row in rows:
        if type(row) is not dict:
            raise ValueError(f"{name} row is not an object")
        case = _case(row)
        if case in result:
            raise ValueError(f"{name} contains a duplicate arm")
        result[case] = row
    if set(result) != set(CASES):
        raise ValueError(f"{name} omits a registered arm")
    return result


def _journal_rows(stage, run_dir):
    """Read a frozen journal prefix, rejecting later append or altered bytes."""
    prefix = stage.get("source_prefix")
    if type(prefix) is not dict:
        raise ValueError("completed training has no authenticated attempt journal prefix")
    original = Path(prefix.get("path", "")).absolute()
    path = original.resolve()
    if original != path or original.is_symlink():
        raise ValueError("attempt journal must have a normalized path without symbolic links")
    if not path.is_relative_to(Path(run_dir).resolve()) or path.name != "attempts.jsonl":
        raise ValueError("attempt journal escapes its registered run")
    length = _integer(prefix.get("bytes"), "attempt journal bytes")
    digest = _sha(prefix.get("sha256"), "attempt journal prefix")
    if path.stat().st_size != length:
        raise ValueError("attempt journal advanced beyond the final ledger; refresh the ledger")
    raw = path.read_bytes()
    if len(raw) != length or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("attempt journal prefix changed after the final ledger")
    rows, damaged = [], []
    for line, text in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(text)
            if type(value) is not dict or type(value.get("event")) is not str:
                raise ValueError("invalid event")
            rows.append((line, value))
        except (ValueError, UnicodeError):
            damaged.append(line)
    return rows, damaged, {"path": str(path), "bytes": length, "sha256": digest}


def _failure_kind(row):
    kind, message = row.get("error_type"), row.get("error")
    if type(kind) is not str or type(message) is not str:
        return "unclassified"
    if kind in {"FloatingPointError", "ArithmeticError", "OverflowError", "ZeroDivisionError"} or _NUMERICAL.search(message):
        return "numerical"
    if kind in {"OutOfMemoryError", "TimeoutError", "ConnectionError", "BrokenPipeError"} or _INFRA.search(message):
        return "infrastructure"
    # OSError/RuntimeError alone does not establish infrastructure. A missing
    # input or deterministic program bug must not be relabeled to obtain a pass.
    return "unclassified"


def audit_retries(runs, ledger):
    """Audit recorded failures without counting D parent/child reports twice.

    Every child is scanned for numerical errors. Retry counts use only the
    canonical parent invocation journal (B's learner or D's controller), since
    one failing child normally also raises the same exception in its parent.
    Unknown hard interruptions remain unresolved rather than becoming zero.
    """
    indexed = _index(runs, "retry audit")
    result, reasons, warnings = [], [], []
    for variant, seed in CASES:
        key = f"{variant}/seed_{seed}"
        run = indexed[variant, seed]
        entry = ledger["runs"][key]
        physical = entry.get("physical_work_by_stage")
        expected = {"local_cycle"} if variant == "B" else {"teacher", "buffer", "distillation", "local_cycle", "controller"}
        if type(physical) is not dict or set(physical) != expected:
            raise ValueError(f"{key}: physical stage journals are missing or unexpected")
        unique, stage_rows = {}, {}
        for name, stage in sorted(physical.items()):
            if type(stage) is not dict or type(stage.get("additional_partial_work_unknown")) is not bool:
                raise ValueError(f"{key}/{name}: physical-work uncertainty is not explicit")
            rows, damaged, reference = _journal_rows(stage, run["run_dir"])
            stage_rows[name] = rows
            unique[reference["path"]] = (rows, damaged, reference)
            if stage["additional_partial_work_unknown"]:
                if stage.get("additional_partial_optimizer_work") is not None:
                    raise ValueError(f"{key}/{name}: unknown interrupted optimizer work was imputed")
                warnings.append(f"{key}/{name}: interrupted or unmeasured physical work remains unknown")
            for field in ("recorded_elapsed_seconds",):
                value = stage.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{key}/{name}: invalid recorded elapsed lower bound")
        canonical_stage = "local_cycle" if variant == "B" else "controller"
        canonical = stage_rows[canonical_stage]
        canonical_path = str(Path(physical[canonical_stage]["source_prefix"]["path"]).resolve())
        incidents, child_numerical = [], []
        for _, (rows, damaged, reference) in sorted(unique.items()):
            if damaged:
                reasons.append(f"{key}: damaged attempt rows prevent a complete retry audit")
            for line, row in rows:
                if row["event"].endswith("failed") and _failure_kind(row) == "numerical":
                    child_numerical.append({"line": line, "journal": reference, "error_type": row.get("error_type"),
                                            "error": row.get("error")})
        for line, row in canonical:
            if row["event"].endswith("failed"):
                incidents.append({"line": line, "attempt": row.get("attempt"), "invocation": row.get("invocation"),
                                  "classification": _failure_kind(row), "error_type": row.get("error_type"),
                                  "error": row.get("error"), "same_frozen_seed_config_required": True})
        parent_signatures = Counter((row["error_type"], row["error"]) for row in incidents)
        child_signatures = Counter()
        for path, (rows, _, _) in sorted(unique.items()):
            if path == canonical_path:
                continue
            for _, row in rows:
                if row["event"].endswith("failed"):
                    child_signatures[row.get("error_type"), row.get("error")] += 1
                    if _failure_kind(row) == "unclassified":
                        reasons.append(f"{key}: a child failure has no established infrastructure classification")
        if any(count > parent_signatures[signature] for signature, count in child_signatures.items()):
            reasons.append(f"{key}: child failures are not completely represented in the canonical parent retry journal")
        infrastructure = sum(row["classification"] == "infrastructure" for row in incidents)
        if infrastructure > 2:
            reasons.append(f"{key}: more than two infrastructure retries were recorded")
        if any(row["classification"] == "unclassified" for row in incidents):
            reasons.append(f"{key}: a failed invocation has no established infrastructure classification")
        if child_numerical:
            reasons.append(f"{key}: numerical failure was resumed; it must remain an outcome")
        starts = {row.get("attempt") for _, row in canonical if row["event"].endswith("_started")}
        ended = {row.get("attempt") for _, row in canonical if row["event"].endswith("_committed")
                 or row["event"] == "unit_cancelled_before_training"}
        failed = {row["attempt"] for row in incidents}
        unexplained = sorted(str(value) for value in starts - ended - failed)
        if unexplained:
            reasons.append(f"{key}: unmatched attempts have no recorded failure classification")
        result.append({"case": key, "infrastructure_retries": infrastructure,
                       "failed_invocations": incidents, "numerical_failure_records": child_numerical,
                       "unclassified_unmatched_attempts": unexplained,
                       "canonical_parent_journal": "local_cycle" if variant == "B" else "controller",
                       "parent_child_failures_counted_twice": False,
                       "journal_prefixes": [value[2] for _, value in sorted(unique.items())]})
    return {"status": "complete" if not reasons else "needs_review", "runs": result,
            "reasons": reasons, "warnings": sorted(set(warnings)), "max_infrastructure_retries_per_run": 2,
            "unrecorded_failures_assumed_absent": False}


def _check_verification(context, verification):
    if (type(verification) is not dict or verification.get("schema_version") != "p565758-implementation-v1"
            or verification.get("status") != "complete_p565758_implementation"):
        raise ValueError("P5.6/P5.7/P5.8 implementation verification is missing or incomplete")
    _integer(verification.get("tests_run"), "verification tests")
    if verification["tests_run"] == 0:
        raise ValueError("implementation verification ran no tests")
    for name in ("failures", "errors", "skipped"):
        _integer(verification.get(name), f"verification {name}", target=0)
    for name in ("research_training_executed", "research_evaluation_executed", "gpu_runtime_measured"):
        if verification.get(name) is not False:
            raise ValueError("implementation fixtures must not be presented as research execution")
    if (verification.get("frozen_source_sha256") != context["cohort"]["source_sha256"]
            or verification.get("source_sha256") != context.get("source_identity")
            or not verification.get("source_sha256")):
        raise ValueError("implementation proof does not bind current frozen and extension sources")


def _check_runs(context):
    cohort, ledger = context["cohort"], context["ledger"]
    indexed = _index(context["runs"], "training audit")
    _index(cohort.get("run_matrix"), "frozen cohort")
    if (cohort.get("historical_results_promoted") is True or cohort.get("historical_runs_promoted") is True
            or cohort.get("candidate_only") is not False):
        raise ValueError("historical or candidate evidence cannot become a prospective completed cohort")
    if (ledger.get("status") != "complete" or ledger.get("cohort_identity") != cohort["cohort_identity"]
            or ledger.get("missing_run_slots") != [] or ledger.get("evaluation_enters_training") is not False
            or set(ledger.get("runs", {})) != {f"{v}/seed_{s}" for v, s in CASES}):
        raise ValueError("final ledger has incomplete, extra or mismatched registered runs")
    shared = ledger["shared_initial_costs"]
    for field, target in (("physical_joint_transitions", 100000),
                          ("configured_four_run_attributed_joint_transitions", 200000),
                          ("physical_selected_model_prefix_optimizer_steps", 9000),
                          ("configured_four_run_attributed_selected_model_steps", 18000)):
        _integer(shared.get(field), field, target=target)
    summaries = []
    for variant, seed in CASES:
        run, key = indexed[variant, seed], f"{variant}/seed_{seed}"
        entry = ledger["runs"][key]
        if (run.get("status") != "complete" or run.get("phase") != "complete" or run.get("training_complete") is not True
                or run.get("cohort_identity") != cohort["cohort_identity"]
                or entry.get("run_identity") != run.get("run_identity")
                or entry.get("last_checkpoint") != run.get("last_checkpoint")
                or entry.get("run_dir") != run.get("run_dir") or entry.get("status") != "complete"):
            raise ValueError(f"{key}: incomplete training, checkpoint or ledger disagreement")
        binding = run.get("run_binding", {})
        if (binding.get("run_kind") != "prospective_paired_cohort" or binding.get("historical_adoption") is not False
                or binding.get("cohort_identity") != cohort["cohort_identity"]
                or binding.get("variant") != variant or binding.get("master_seed") != seed):
            raise ValueError(f"{key}: original fresh-run binding differs")
        expected = {"local_cycles_completed": 20, "local_actor_optimizer_steps": 320,
                    "local_critic_optimizer_steps": 320, "real_joint_transitions": 50000,
                    "model_updates": 4500, "new_environment_interactions": 0, "new_model_updates": 0,
                    "teacher_cycles_completed": 25 if variant == "D" else 0,
                    "teacher_actor_optimizer_steps": 400 if variant == "D" else 0,
                    "teacher_critic_optimizer_steps": 400 if variant == "D" else 0,
                    "trajectories_completed": 128 if variant == "D" else 0,
                    "distillation_optimizer_steps": 100 if variant == "D" else 0}
        for field, target in expected.items():
            _integer(run.get(field), f"{key}/{field}", target=target)
        counters = entry["counters"]
        for run_name, ledger_name in (("local_cycles_completed", "local_PPO_cycles"),
                ("local_actor_optimizer_steps", "local_actor_steps"), ("local_critic_optimizer_steps", "local_critic_steps"),
                ("teacher_cycles_completed", "teacher_PPO_cycles"), ("teacher_actor_optimizer_steps", "teacher_actor_steps"),
                ("teacher_critic_optimizer_steps", "teacher_critic_steps"), ("trajectories_completed", "warm_start_buffer_trajectories"),
                ("distillation_optimizer_steps", "distillation_optimizer_steps"), ("model_updates", "world_model_optimizer_steps_initial"),
                ("new_model_updates", "new_model_updates"), ("new_environment_interactions", "new_environment_interactions")):
            _integer(counters.get(ledger_name), f"{key}/ledger/{ledger_name}", target=run[run_name])
        for field, maximum in (("imagined_joint_steps_local", 20 * 256 * 15),
                               ("imagined_joint_steps_teacher", 25 * 256 * 15 if variant == "D" else 0),
                               ("imagined_joint_steps_buffer", 128 * 15 if variant == "D" else 0)):
            _integer(run.get(field), f"{key}/{field}", maximum=maximum)
        if type(run.get("actor_export")) is not str or not run["actor_export"]:
            raise ValueError(f"{key}: final actor export is missing")
        pairing = binding.get("pairing")
        if type(pairing) is not dict or set(pairing) != {"actor_cpu", "critic_cpu", "input_binding_sha256",
                "world_model_checkpoint_sha256", "normalization_identity_sha256", "replay_manifest_sha256"}:
            raise ValueError(f"{key}: exact initial model/normalization/data/policy pairing is missing")
        for field, value in pairing.items():
            _sha(value, f"{key}/{field}")
        other = indexed["D" if variant == "B" else "B", seed]
        if other.get("run_binding", {}).get("pairing") != pairing:
            raise ValueError(f"{key}: B and D no longer share their registered initial artifacts")
        summaries.append({"case": key, "run_identity": run["run_identity"], "final_checkpoint": run["last_checkpoint"],
                          "actor_export": run["actor_export"], "counts": expected,
                          "initial_pairing": deepcopy(pairing)})
    return summaries


def _check_endpoints(context):
    cohort, analysis = context["cohort"], context["analysis"]
    if (analysis.get("status") != "complete" or analysis.get("source_unchanged") is not True
            or analysis.get("missing_evidence_imputed") is not False
            or analysis.get("cohort_identity") != cohort["cohort_identity"]
            or analysis.get("protocol_sha256") != cohort["protocol_sha256"]
            or analysis.get("bank_sha256") != cohort["bank_sha256"]):
        raise ValueError("authenticated paired analysis is incomplete or belongs to another cohort/bank")
    metrics = analysis.get("analysis")
    if (type(metrics) is not dict or metrics.get("evidence_kind") != "native_final_endpoint"
            or metrics.get("training_seeds") != [0, 1] or metrics.get("runs") != 4
            or metrics.get("episodes_per_run") != 20 or metrics.get("total_logical_episodes") != 80
            or metrics.get("paired_seed_count") != 2):
        raise ValueError("paired endpoint analysis does not contain all 80 native final records")
    arms = _index(metrics.get("arms"), "paired analysis arms")
    for (variant, seed), arm in arms.items():
        _integer(arm.get("successes"), f"{variant}/seed_{seed} successes", maximum=20)
    if type(metrics.get("descriptive_summary")) is not dict:
        raise ValueError("paired descriptive differences are missing")
    for field in ("confidence_intervals", "p_values", "bootstrap", "auc", "superiority_claim", "equivalence_claim"):
        if metrics.get(field) is not False:
            raise ValueError("two-pair analysis overstates its statistical scope")
    reports = _index(context["evaluation_reports"], "native evaluation")
    actors = set()
    for variant, seed in CASES:
        report = reports[variant, seed]
        if (report.get("status") != "complete" or report.get("cohort_identity") != cohort["cohort_identity"]
                or report.get("protocol_sha256") != cohort["protocol_sha256"]
                or report.get("bank_sha256") != cohort["bank_sha256"]
                or report.get("fixture_only") is not False or report.get("final_local_cycles") != 20
                or any(report.get(name) is not True for name in ("native_backend", "native_engine",
                    "native_environment_evaluation_executed", "research_evaluation_executed"))
                or any(report.get(name) is not False for name in ("world_model_loaded", "critic_loaded", "evaluation_data_enters_training", "learning_eligible"))):
            raise ValueError("native final actor-only evaluation binding is incomplete or differs")
        for name in ("world_model_updates", "policy_optimizer_updates", "real_ppo_updates", "training_interactions_added"):
            _integer(report.get(name), f"final evaluation/{name}", target=0)
        actors.add(_sha(report.get("final_actor_identity"), "final actor provenance"))
        episodes = report.get("episodes")
        if type(episodes) is not list or len(episodes) != 20 or {r.get("episode_index") for r in episodes} != set(range(20)):
            raise ValueError("every actor must retain exactly its 20 native bank records")
        accounting = report.get("interaction_accounting", {})
        # The already frozen evaluator allows two retries PER EPISODE, not two
        # across its entire bank. Authenticate those readers before entering
        # this gate, then retain their exact grouped publication structure.
        retries = _integer(accounting.get("infrastructure_retries"), "evaluation infrastructure retries")
        attempts = report.get("attempts")
        if type(attempts) is not list:
            raise ValueError("evaluation attempt receipts are missing")
        grouped = {}
        for attempt in attempts:
            match = re.fullmatch(r"episodes/([0-9]{6})/attempt_([0-9]{2})/attempt\.json", attempt.get("path", ""))
            if match is None:
                raise ValueError("evaluation attempt receipt path differs from the fixed episode/attempt structure")
            index, number = map(int, match.groups())
            _sha(attempt.get("sha256"), "evaluation attempt receipt")
            if number in grouped.setdefault(index, []):
                raise ValueError("evaluation contains a duplicate episode attempt receipt")
            grouped[index].append(number)
        if (set(grouped) != set(range(20)) or any(sorted(numbers) != list(range(len(numbers)))
                or len(numbers) > 3 for numbers in grouped.values())
                or sum(len(numbers) - 1 for numbers in grouped.values()) != retries):
            raise ValueError("evaluation episode retries exceed two, omit attempts or disagree with the frozen accounting")
        if accounting.get("training_interactions_added") != 0 or accounting.get("learning_eligible") is not False:
            raise ValueError("evaluation physical costs entered training")
        entry = context["ledger"]["runs"][f"{variant}/seed_{seed}"]["evaluation"]
        if (entry.get("status") != "complete" or entry.get("evaluation_enters_training") is not False
                or entry.get("accounting") != accounting):
            raise ValueError("final ledger predates or differs from completed native evaluation accounting")
    if len(actors) != 4:
        raise ValueError("four distinct final actor identities are required")
    return metrics


def build_acceptance(context, secondary, resources, verification):
    """Return an honest gate report; missing/broken evidence is ``needs_review``.

    Numerical test results and poor task performance never cause automatic
    retraining. Missing resource measurements remain null/warnings, rather than
    preventing completion of an otherwise authentic, fully executed study.
    """
    reasons, warnings, checks = [], [], {}
    run_summaries, metrics, retry = [], None, None
    def check(name, operation):
        try:
            result = operation()
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
            message = f"{name}: {type(error).__name__}: {error}"
            reasons.append(message)
            checks[name] = {"passed": False, "reason": message}
            return None
        checks[name] = {"passed": True}
        return result

    def authentication():
        if type(context) is not dict or type(context.get("authentication")) is not dict:
            raise ValueError("authenticated evidence context is missing")
        if any(context["authentication"].get(name) is not True for name in AUTHENTICATION):
            raise ValueError("cohort, sources, final actors, traces, input hashes, analysis and final ledger must all be authenticated")
        cohort = context["cohort"]
        for field in ("cohort_identity", "protocol_sha256", "bank_sha256"):
            _sha(cohort.get(field), field)
        if cohort.get("status") != "cohort_frozen" or cohort.get("execution_ready") is not True:
            raise ValueError("cohort was not prospectively frozen for execution")
        references = context.get("references")
        if not references or type(references) not in (dict, list):
            raise ValueError("authenticated evidence references must be retained")
    check("authenticated_evidence", authentication)
    check("implementation_verification", lambda: _check_verification(context, verification))
    run_summaries = check("all_registered_training_stages", lambda: _check_runs(context)) or []
    metrics = check("native_fixed_endpoint_records", lambda: _check_endpoints(context))
    retry = check("retry_and_interruption_accounting", lambda: audit_retries(context["runs"], context["ledger"]))
    if retry:
        warnings.extend(retry["warnings"])
        reasons.extend(retry["reasons"])
        if retry["status"] != "complete":
            checks["retry_and_interruption_accounting"]["passed"] = False
    def reporting():
        for name, report in (("secondary", secondary), ("resources", resources)):
            if type(report) is not dict or report.get("status") != "complete":
                raise ValueError(f"{name} evidence report is incomplete")
            if report.get("cohort_identity") != context["cohort"]["cohort_identity"]:
                raise ValueError(f"{name} report belongs to another cohort")
            warnings.extend(report.get("warnings", []))
            missing = report.get("missing_measurements", [])
            if missing:
                warnings.append(f"{name}: missing measurements remain explicit: {missing}")
        deliverables = context.get("deliverables")
        if type(deliverables) is not dict or not all(deliverables.get(name) for name in ("secondary", "resources", "figures")):
            raise ValueError("secondary evidence, resource tables or the two endpoint figures are missing")
        figures = deliverables["figures"]
        if type(figures) not in (list, dict) or len(figures) != 2:
            raise ValueError("exactly two deterministic endpoint figures are required")
        graphics = context.get("figure_evidence")
        if (type(graphics) is not dict or graphics.get("fixtures_only") is not False
                or graphics.get("actual_graphics_rendered") is not True
                or graphics.get("matplotlib_version") != "3.11.2"):
            raise ValueError("endpoint figures require actual rendered graphics provenance, not fixture publications")
    check("secondary_resource_and_endpoint_deliverables", reporting)
    passed = not reasons
    successes = sum(row["successes"] for row in metrics["arms"]) if metrics is not None else None
    effect = deepcopy(metrics["descriptive_summary"]) if metrics is not None else None
    cohort_for_summary = context.get("cohort") if type(context) is dict else None
    return {"schema_version": SCHEMA, "status": "passed" if passed else "needs_review",
        "phase5_acceptance": passed, "execution_acceptance": passed, "evidence_acceptance": passed,
        "cohort_acceptance": passed, "cohort_acceptance_scope": "execution and evidence completeness only",
        "cohort_identity": cohort_for_summary.get("cohort_identity") if type(cohort_for_summary) is dict else None,
        "reasons": reasons, "warnings": sorted(set(str(w) for w in warnings)), "gate_checks": checks,
        "completed_run_budgets": run_summaries, "retry_audit": retry,
        "successful_final_episodes_observed": successes, "paired_descriptive_effects": effect,
        "D_wins_required_for_acceptance": False, "statistical_superiority_established": False,
        "successful_transfer_established": False, "quality_acceptance_implied_by_completion": False,
        "missing_evidence_imputed": False, "zero_negative_results_are_valid_outcomes": True,
        "unmeasured_costs_assumed_zero": False,
        "deviations": deepcopy(context.get("deviations", [])) if type(context) is dict else [],
        "evidence_references": deepcopy(context.get("references", {})) if type(context) is dict else {},
        "deliverables": deepcopy(context.get("deliverables", {})) if type(context) is dict else {},
        "source_sha256": deepcopy(context.get("source_identity", {})) if type(context) is dict else {},
        "implementation_verification": {name: deepcopy(verification.get(name)) for name in
            ("schema_version", "status", "tests_run", "failures", "errors", "skipped")} if type(verification) is dict else None,
        "research_training_executed_by_gate": False, "research_evaluation_executed_by_gate": False,
        "native_environment_opened": False, "new_environment_interactions": 0,
        "new_model_updates": 0, "new_policy_optimizer_updates": 0,
        "next_step": "P6 interpretation and release" if passed else "resolve missing or invalid evidence; do not replace unfavorable outcomes"}
