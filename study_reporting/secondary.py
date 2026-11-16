"""P5.6 summaries from authenticated, already completed server evidence.

This module does not construct networks, optimizers or simulators. Saved
checkpoint dictionaries are deserialized on CPU only to read existing reports;
targets, trajectories and losses are never recomputed.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re

from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_endpoint import checkpoints
from world_models_study.evidence import read_report, safe_path


SCHEMA = "p56-essential-secondary-evidence-v1"
_AUTH = ("cohort", "analysis", "final_ledger", "final_actors", "evaluation_traces",
         "input_hashes", "frozen_sources")
_CASES = {(seed, variant) for seed in (0, 1) for variant in ("B", "D")}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _identity(value, name):
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 identity")
    return value


def _bind(run, path, sources):
    path = safe_path(path)
    if not path.is_relative_to(safe_path(run["run_dir"])):
        raise ValueError("secondary diagnostic source escaped its registered run")
    expected = {row["path"]: row["sha256"] for row in run["files"]}
    if str(path) not in expected or not path.is_file() or file_sha256(path) != expected[str(path)]:
        raise ValueError(f"secondary diagnostic source is not audit-bound: {path}")
    sources[str(path)] = expected[str(path)]
    return path


def _checkpoint(run, directory, loader, sources):
    directory = safe_path(directory)
    for name in ("manifest.json", "manifest.sha256", "state.pt"):
        _bind(run, directory / name, sources)
    manifest = checkpoints.authenticate(directory)
    state, metadata = loader(directory)
    if metadata != manifest["metadata"] or type(state) is not dict:
        raise ValueError("secondary checkpoint metadata differs from its immutable receipt")
    if any(metadata.get(key) != 0 for key in ("new_environment_interactions", "new_model_updates")):
        raise ValueError("diagnostic checkpoint contains new data collection or model updates")
    return state, metadata


def _cpu_checkpoint(directory):
    from world_models.cycle.checkpoints import load_cycle_checkpoint
    return load_cycle_checkpoint(directory, device="cpu")


def _ppo(run, checkpoint, model_version, stage, loader, sources, warnings):
    saved, metadata = _checkpoint(run, checkpoint, loader, sources)
    teacher = stage == "teacher"
    cycles = 25 if teacher else 20
    key = "initial_model_version" if teacher else "model_version"
    schema = ("p41-teacher-checkpoint-v1" if teacher else
              "p52-b-local-checkpoint-v1" if run["variant"] == "B" else "p43-local-refinement-checkpoint-v1")
    if (metadata.get("schema_version") != schema or metadata.get(key) != model_version
            or metadata.get("teacher_cycles_completed" if teacher else "local_cycles_completed") != cycles):
        raise ValueError("PPO diagnostic checkpoint model identity or completed stage changed")
    report = saved.get("last_report")
    if report is None:
        warnings.append(f"{run['case']}/{stage}: final PPO report is unavailable; no values inferred")
        return {"status": "unavailable", "checkpoint": str(checkpoint), "metrics": None,
                "cycles_completed": cycles, "scope": "saved final imagined-training cycle"}
    if (type(report) is not dict or report.get("schema_version") !=
            ("p41-teacher-update-v1" if teacher else "p34-update-v1")
            or report.get("cycles_completed") != cycles
            or report.get("world_model_version") != model_version
            or report.get("actor_optimizer_steps") != 16 * cycles
            or report.get("critic_optimizer_steps") != 16 * cycles
            or report.get("new_environment_interactions") != 0):
        raise ValueError("saved PPO diagnostic report differs from the registered final stage")
    if not teacher and any(report.get(key) != 0 for key in
            ("teacher_kl_weight", "teacher_queries", "teacher_optimizer_steps", "distillation_optimizer_steps")):
        raise ValueError("local PPO diagnostics contain teacher queries or auxiliary updates")
    metrics = report.get("metrics")
    if type(metrics) is not dict:
        raise ValueError("saved PPO metrics must be an object")
    retained = {}
    for name, value in metrics.items():
        if name == "explained_variance" and value is None:
            retained[name] = None  # Undefined target variance is not a zero variance score.
        else:
            retained[name] = _number(value, f"{stage}/{name}")
    for name in ("approx_kl", "entropy", "clip_fraction", "value_loss", "value_error",
                 "actor_grad_norm", "critic_grad_norm", "mean_trajectory_length",
                 "predicted_termination_frequency"):
        if name not in retained:
            retained[name] = None
            warnings.append(f"{run['case']}/{stage}: saved {name} measurement is unavailable")
    if (retained["entropy"] is not None and not -1e-6 <= retained["entropy"] <= math.log(12) + 1e-5
            or retained["clip_fraction"] is not None and not 0 <= retained["clip_fraction"] <= 1
            or retained["predicted_termination_frequency"] is not None
            and not 0 <= retained["predicted_termination_frequency"] <= 1):
        raise ValueError("PPO entropy or probability diagnostic is outside its valid range")
    steps = _integer(report.get("imagined_joint_steps"), "last-cycle imagined joint steps", minimum=1)
    causes = report.get("termination_counts")
    if type(causes) is not dict or set(causes) != {"success", "failure", "deadline", "horizon"}:
        raise ValueError("PPO imagined termination counters are missing or malformed")
    if sum(_integer(value, "imagined termination count") for value in causes.values()) != 256 or steps > 3840:
        raise ValueError("PPO final cycle exceeds its 256-start, 15-step budget")
    if (retained["mean_trajectory_length"] is not None
            and not math.isclose(retained["mean_trajectory_length"], steps / 256, abs_tol=1e-6)):
        raise ValueError("PPO mean trajectory length disagrees with its saved joint-step count")
    return {"status": "available", "checkpoint": str(checkpoint), "cycles_completed": cycles,
            "actor_optimizer_steps": 16 * cycles, "critic_optimizer_steps": 16 * cycles,
            "imagined_joint_steps_in_last_cycle": steps, "termination_counts": deepcopy(causes),
            "metrics": retained, "model_version": model_version,
            "progression": None, "scope": "final committed imagined-training cycle only; not real performance",
            "progression_note": "Earlier metric logs are not used as authenticated checkpoint diagnostics."}


def _distillation(run, model_version, loader, sources, warnings):
    evidence = run["stage_evidence"]["distillation"]
    directory = safe_path(evidence["path"])
    checkpoint = safe_path(evidence["checkpoint"])
    saved, metadata = _checkpoint(run, checkpoint, loader, sources)
    if (metadata.get("schema_version") != "p42-distillation-checkpoint-v1"
            or metadata.get("distillation_optimizer_steps") != 100
            or metadata.get("trajectories_completed") != 128):
        raise ValueError("distillation diagnostic checkpoint differs from the completed fixed buffer budget")
    path = directory / "warm_started_student.json"
    _bind(run, path, sources)
    handoff = read_report(path)
    # The handoff receipt need not appear separately in old audit.files. It is
    # authenticated by read_report and bound here for the final publication barrier.
    sources[str(path.with_suffix(".sha256"))] = file_sha256(path.with_suffix(".sha256"))
    if (handoff.get("schema_version") != "p42-warm-started-student-v1"
            or safe_path(directory / handoff.get("checkpoint", "")) != checkpoint
            or handoff.get("checkpoint_manifest_sha256") != file_sha256(checkpoint / "manifest.json")
            or handoff.get("run_identity") != metadata.get("run_identity")
            or handoff.get("distillation_optimizer_steps") != 100 or handoff.get("trajectories") != 128
            or handoff.get("frozen_versions", {}).get("model") != model_version
            or any(handoff.get(key) != 0 for key in ("new_environment_interactions", "new_model_updates"))):
        raise ValueError("distillation handoff identity differs from the audited completed checkpoint")
    values, missing = {}, False
    steps = _integer(metadata.get("imagined_joint_steps"), "distillation imagined steps", minimum=1)
    for name, updates in (("diagnostics_before", 0), ("diagnostics_after", 100)):
        value = saved.get(name)
        if canonical_hash(value) != canonical_hash(handoff.get(name)):
            raise ValueError("distillation diagnostic JSON differs from its committed checkpoint")
        if value is None:
            warnings.append(f"{run['case']}: {name} is unavailable; no KL/entropy values inferred")
            values[name] = None
            missing = True
            continue
        expected = {"schema_version", "forward_kl", "student_entropy", "valid_player_decisions",
                    "trajectories", "optimizer_updates", "scope"}
        if (type(value) is not dict or set(value) != expected
                or value.get("schema_version") != "p42-training-diagnostics-v1"
                or value.get("scope") != "fixed training buffer; not validation or real performance"
                or type(value.get("optimizer_updates")) is not int or value["optimizer_updates"] != updates
                or type(value.get("trajectories")) is not int or value["trajectories"] != 128
                or type(value.get("valid_player_decisions")) is not int
                or value["valid_player_decisions"] != 2 * steps
                or _number(value.get("forward_kl"), "distillation KL") < -1e-6
                or not -1e-6 <= _number(value.get("student_entropy"), "distillation entropy") <= math.log(12) + 1e-5):
            raise ValueError("distillation KL/entropy diagnostics differ from their training-buffer definition")
        values[name] = deepcopy(value)
    before, after = values["diagnostics_before"], values["diagnostics_after"]
    return {"status": "unavailable" if missing else "available", **values,
            "forward_kl_change_after_minus_before": None if missing else after["forward_kl"] - before["forward_kl"],
            "student_entropy_change_after_minus_before": None if missing else after["student_entropy"] - before["student_entropy"],
            "trajectories": 128, "optimizer_updates": 100, "valid_player_decisions": 2 * steps,
            "training_buffer_only": True, "real_cooperation_established": False,
            "checkpoint": str(checkpoint), "handoff": str(path)}


def _endpoint_rows(metrics):
    if (type(metrics) is not dict or metrics.get("evidence_kind") != "native_final_endpoint"
            or metrics.get("total_logical_episodes") != 80 or metrics.get("paired_seed_count") != 2):
        raise ValueError("secondary evidence requires the completed native four-arm paired endpoint analysis")
    arms, layouts = metrics.get("arms"), metrics.get("per_layout")
    if (type(arms) is not list or len(arms) != 4
            or {(r.get("master_seed"), r.get("variant")) for r in arms} != _CASES
            or type(layouts) is not list or len(layouts) != 8
            or {(r.get("master_seed"), r.get("variant"), r.get("template_id")) for r in layouts}
                != {(s, v, t) for s, v in _CASES for t in ("relay_test_00", "relay_test_01")}):
        raise ValueError("secondary endpoint summaries omit or duplicate an arm or authored layout")
    for rows, episodes in ((arms, 20), (layouts, 10)):
        for row in rows:
            successes = _integer(row.get("successes"), "endpoint successes")
            if row.get("episodes") != episodes or successes > episodes:
                raise ValueError("secondary endpoint denominator differs from the fixed test bank")
            if row.get("successful_completion_length_denominator") != successes or row.get("successful_completion_tics_denominator") != successes:
                raise ValueError("conditional completion denominator differs from success count")
            for name in ("mean_successful_completion_length", "mean_successful_completion_tics"):
                if successes == 0 and row.get(name) is not None:
                    raise ValueError("zero-success conditional completion time must be null, never zero-imputed")
                if successes:
                    _number(row.get(name), name)
    return deepcopy(arms), deepcopy(layouts)


def validate_secondary_sources(report):
    """Recheck every checkpoint/report byte actually consumed by this summary."""
    rows = report.get("diagnostic_sources")
    if type(rows) is not list:
        raise ValueError("secondary diagnostic source bindings are missing")
    for row in rows:
        path = safe_path(row["path"])
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise ValueError("secondary diagnostic evidence changed before publication")


def build_secondary(context, *, checkpoint_loader=None):
    """Build P5.6 from the gate's authenticated final cohort context.

    A custom loader is a synthetic-fixture seam; the server entry point always
    uses the CPU weights-only loader. No saved optimization state is restored
    into a learner and no recorded scientific outcome is recalculated.
    """
    if any(context.get("authentication", {}).get(key) is not True for key in _AUTH):
        raise ValueError("P5.6 requires authenticated cohort, analysis, ledger, actors, traces, inputs and sources")
    cohort, analysis, ledger = context["cohort"], context["analysis"], context["ledger"]
    if (analysis.get("status") != "complete" or analysis.get("cohort_identity") != cohort["cohort_identity"]
            or ledger.get("status") != "complete" or ledger.get("cohort_identity") != cohort["cohort_identity"]):
        raise ValueError("secondary analysis or final ledger belongs to a different or incomplete cohort")
    arms, layouts = _endpoint_rows(analysis.get("analysis"))
    runs = context.get("runs")
    if type(runs) is not list or len(runs) != 4 or {(r.get("master_seed"), r.get("variant")) for r in runs} != _CASES:
        raise ValueError("P5.6 requires all four authenticated final training runs")
    indexed = {(r["master_seed"], r["variant"]): r for r in runs}
    loader, warnings, sources, diagnostics, invariants = checkpoint_loader or _cpu_checkpoint, [], {}, [], []
    for seed, variant in sorted(_CASES):
        run = indexed[seed, variant]
        if (run.get("training_complete") is not True or run.get("status") != "complete"
                or run.get("cohort_identity") != cohort["cohort_identity"]
                or any(run.get(key) != 0 for key in ("new_model_updates", "new_environment_interactions"))):
            raise ValueError("secondary diagnostics require completed training with zero new collection/model updates")
        pairing = run["run_binding"]["pairing"]
        b = indexed[seed, "B"]
        expected_model = _identity(b["run_manifest"]["initial_imported_versions"]["model"], "frozen model version")
        normalization = _identity(run["initial_binding"]["normalization"]["identity_sha256"], "normalization identity")
        if (pairing != b["run_binding"]["pairing"]
                or pairing["normalization_identity_sha256"] != normalization
                or run["initial_binding"] != b["initial_binding"]):
            raise ValueError("paired model/replay/normalization initialization identities changed")
        if variant == "B":
            local = _ppo(run, run["last_checkpoint"], expected_model, "local", loader, sources, warnings)
            teacher, distilled = {"status": "not_applicable", "metrics": None}, {"status": "not_applicable"}
        else:
            stages = run["stage_evidence"]
            teacher = _ppo(run, stages["teacher"]["checkpoint"], expected_model, "teacher", loader, sources, warnings)
            distilled = _distillation(run, expected_model, loader, sources, warnings)
            local = _ppo(run, stages["local_cycle"]["checkpoint"], expected_model, "local", loader, sources, warnings)
        diagnostics.append({"master_seed": seed, "variant": variant, "teacher_PPO": teacher,
                            "distillation": distilled, "local_PPO": local})
        invariants.append({"master_seed": seed, "variant": variant,
            "world_model_version": expected_model,
            "selected_model_checkpoint_sha256": pairing["world_model_checkpoint_sha256"],
            "normalization_identity_sha256": normalization,
            "replay_manifest_sha256": pairing["replay_manifest_sha256"],
            "normalization_unchanged": True, "world_model_unchanged": True,
            "new_model_updates": 0, "new_environment_interactions": 0,
            "basis": "receipt-bound initial inputs and final committed stage metadata; no target regeneration"})
    report = {"schema_version": SCHEMA, "status": "complete",
        "cohort_identity": cohort["cohort_identity"], "protocol_sha256": cohort["protocol_sha256"],
        "warnings": warnings, "diagnostics_complete": not warnings,
        "diagnostic_measurement_status": "complete" if not warnings else "complete_with_explicit_measurement_gaps",
        "per_seed": arms, "per_layout": layouts,
        "paired_secondary_differences": deepcopy(analysis["analysis"].get("paired_differences")),
        "training_diagnostics": diagnostics, "frozen_invariants": invariants,
        "work_volumes": {key: deepcopy(ledger.get(key)) for key in
            ("shared_initial_costs", "sum_run_logical_counters", "evaluation_joint_transitions",
             "evaluation_known_simulator_tics", "evaluation_unknown_dispatch_attempts",
             "evaluation_unknown_simulator_tics", "historical_and_development_costs")},
        "resource_measurements": "reported separately by P5.7; absent historical timings/memory are not inferred",
        "diagnostic_sources": [{"path": p, "sha256": digest} for p, digest in sorted(sources.items())],
        "source_sha256": deepcopy(context.get("source_identity", {})),
        "references": deepcopy(context.get("references", [])),
        "completion_time_definition": "joint decisions and simulator tics conditional on success; null with denominator zero",
        "layout_replication": "two authored unseen Relay layouts; training master seed is the replication unit",
        "compute_matched": False, "same_initial_real_samples": True,
        "low_KL_establishes_real_cooperation": False, "model_exploitation_excluded": False,
        "deferred": ["teacher real-environment reference evaluation", "further world-model screen sweeps",
                     "video collections", "additional ablations"],
        "research_training_executed": False, "research_evaluation_executed": False,
        "native_environment_opened": False, "new_model_updates": 0, "new_policy_optimizer_updates": 0}
    validate_secondary_sources(report)
    return report
