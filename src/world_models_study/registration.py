"""Read-only input binding and immutable protocol registration.

This freezes choices and provenance, not an execution-ready comparative cohort.
Existing completed/adopted D runs are never silently relabelled as prospective.
"""
from __future__ import annotations

from pathlib import Path

from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from .bank import build_test_bank, validate_test_bank
from .evidence import (ROOT, read_report, safe_path, source_identity, timestamp,
                       validate_verification, write_report)
from .inputs import bind_initial_inputs
from .protocol import load_protocol, plan, protocol_identity


PENDING = ["P5.2 matched reduced B runner", "P5.4 endpoint evaluator", "P5.5 paired endpoint reader",
           "prospective cohort eligibility and source freeze"]
_MARKERS = {"protocol_frozen": True, "candidate_only": True, "execution_ready": False,
            "cohort_frozen": False, "cohort_acceptance": False, "fresh_cohort_eligible": False,
            "research_training_executed": False, "research_evaluation_executed": False,
            "new_environment_interactions": 0, "new_model_updates": 0,
            "test_bank_executed": False, "test_bank_prior_use_confirmed_absent": False,
            "automatic_training_launch": False}


def _cohort_policy():
    return {"required_runs": ["relay/B/seed_0", "relay/D/seed_0", "relay/B/seed_1", "relay/D/seed_1"],
        "historical_runs_promoted": False, "adopted_development_eligible_automatically": False,
        "registration_chronology": "candidate choices and inputs only; no prospective cohort declared",
        "test_bank_prior_use": "unverified; operator and future execution registry must audit before final evaluation",
        "existing_development": "retain original manifests, acceptance decisions and all outcomes; never rewrite history"}


def _bind_rows(rows):
    if type(rows) is not list or not rows:
        raise ValueError("protocol registration needs bound immutable input files")
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256"}:
            raise ValueError("protocol input binding is malformed")
        path = safe_path(row["path"])
        if str(path) != row["path"] or str(path) in seen or not path.is_file():
            raise ValueError("protocol input path is duplicated, unavailable or not absolute")
        seen.add(str(path))
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"registered input changed: {path}")


def _fresh_output(output, readonly):
    if output.exists():
        raise FileExistsError("protocol registration requires a fresh output directory")
    guarded = [ROOT / name for name in ("src", "scripts", "tests", "configs", "scenarios")]
    guarded += [safe_path(path) for path in readonly]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in guarded):
        raise ValueError("protocol output must be separate from source, input and verification directories")


def register_protocol(output_dir, *, initial_selection, verification_report,
                      data_root=ROOT / "data/initial_replay/pilot",
                      statistics_root=ROOT / "artifacts/phase23/normalization",
                      config_root=ROOT / "configs"):
    """Publish static choices only after authenticating verification and inputs."""
    output, verification = safe_path(output_dir), safe_path(verification_report)
    selection = safe_path(initial_selection)
    roots = {"initial_selection": str(selection), "data_root": str(safe_path(data_root)),
             "statistics_root": str(safe_path(statistics_root)), "config_root": str(safe_path(config_root))}
    _fresh_output(output, [selection.parent, data_root, statistics_root, config_root, verification.parent])
    before = source_identity()
    validate_verification(verification)
    evidence = {"path": str(verification), "sha256": file_sha256(verification),
                "tests_log_sha256": file_sha256(verification.parent / "tests.txt")}
    inputs = bind_initial_inputs(**roots)
    _bind_rows(inputs.get("bound_files"))
    _fresh_output(output, {safe_path(row["path"]).parent for row in inputs["bound_files"]})
    bank = build_test_bank()
    validate_test_bank(bank)
    study_plan = plan()
    document = {"schema_version": "p51-protocol-registration-v1", "status": "protocol_registered",
        "created_at": timestamp(), "protocol": load_protocol(), "protocol_sha256": protocol_identity(),
        "source_sha256": before, "inputs": inputs, "input_roots": roots,
        "verification": evidence, "run_plan": study_plan, "test_bank_sha256": canonical_hash(bank),
        "test_bank_file": "test_bank.json", "pending": list(PENDING),
        "cohort_policy": _cohort_policy(),
        "report_path": str(output / "protocol_registration.json"), **_MARKERS}
    # Scientific choices and all referenced bytes must stay fixed throughout.
    validate_verification(verification)
    _bind_rows(inputs["bound_files"])
    if (before != source_identity() or file_sha256(verification) != evidence["sha256"]
            or file_sha256(verification.parent / "tests.txt") != evidence["tests_log_sha256"]):
        raise ValueError("protocol sources or verification evidence changed during registration")
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "test_bank.json", bank)
    write_report(output / "protocol_registration.json", document)
    return document


def inspect_registration(path):
    """Recheck an immutable candidate; inspection cannot promote readiness."""
    path = safe_path(path)
    report = read_report(path)
    if (report.get("schema_version") != "p51-protocol-registration-v1"
            or report.get("status") != "protocol_registered"
            or report.get("report_path") != str(path)
            or report.get("protocol") != load_protocol()
            or report.get("protocol_sha256") != protocol_identity()
            or report.get("source_sha256") != source_identity()
            or report.get("run_plan") != plan()
            or report.get("cohort_policy") != _cohort_policy()
            or report.get("pending") != PENDING
            or any(type(report.get(key)) is not type(value) or report.get(key) != value
                   for key, value in _MARKERS.items())):
        raise ValueError("registered protocol, sources or candidate-only execution boundary changed")
    if report.get("test_bank_file") != "test_bank.json":
        raise ValueError("registered test bank path differs")
    bank = read_report(path.parent / "test_bank.json")
    validate_test_bank(bank)
    if canonical_hash(bank) != report.get("test_bank_sha256"):
        raise ValueError("registered test bank identity changed")
    _bind_rows(report.get("inputs", {}).get("bound_files"))
    roots = report.get("input_roots")
    if type(roots) is not dict or set(roots) != {"initial_selection", "data_root", "statistics_root", "config_root"}:
        raise ValueError("registered initial input roots differ")
    if canonical_hash(bind_initial_inputs(**roots)) != canonical_hash(report["inputs"]):
        raise ValueError("registered input metadata differs from its actual immutable files")
    evidence = report.get("verification", {})
    verification = safe_path(evidence["path"])
    validate_verification(verification)
    if (file_sha256(verification) != evidence.get("sha256")
            or file_sha256(verification.parent / "tests.txt") != evidence.get("tests_log_sha256")):
        raise ValueError("registered implementation evidence changed")
    return report
