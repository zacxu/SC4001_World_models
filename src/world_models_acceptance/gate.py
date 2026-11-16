"""Read-only single-run D engineering acceptance; never a performance gate."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from world_models.data.ledger import file_sha256
from world_models.teacher.runner import _root
from .audit import audit_campaign
from .deployment import probe_actor_export
from .evidence import (ROOT, load_valid_verification, read_report, source_identity, write_report)


def _output_directory(run, output, verification):
    if output.exists():
        raise FileExistsError("acceptance requires a fresh output directory")
    readonly = [run, ROOT / "src", ROOT / "scripts", ROOT / "tests", ROOT / "configs", verification.parent]
    manifest_path = run / "run_manifest.json"
    if manifest_path.is_file():
        # Only interpret valid, receipt-bound paths. An invalid campaign is
        # reported later, without writing into any of its known input stores.
        try:
            manifest = read_report(manifest_path)
        except (OSError, ValueError):
            manifest = {}
        readonly.extend(_root(path) for path in manifest.get("paths", {}).values())
        inputs = manifest.get("inputs", {})
        readonly.extend(_root(inputs[key]) for key in ("data_root", "statistics_root", "config_root") if key in inputs)
        if "initial_selection" in inputs:
            readonly.append(_root(inputs["initial_selection"]).parent)
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in readonly):
        raise ValueError("acceptance output must be separate from training, inputs, source and verification evidence")
    output.mkdir(parents=True, exist_ok=False)


def _bound_files_unchanged(rows):
    if type(rows) is not list or not rows:
        raise ValueError("campaign audit did not bind its immutable input artifacts")
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256"} or type(row["path"]) is not str:
            raise ValueError("campaign audit artifact binding is malformed")
        path = _root(row["path"])
        if str(path) in seen or not path.is_file() or file_sha256(path) != row["sha256"]:
            raise ValueError(f"campaign input changed during acceptance: {path}")
        seen.add(str(path))


def accept_campaign(run_dir, output_dir, *, verification_report, device="cpu"):
    """Audit complete immutable work and publish a separate engineering decision.

    Invalid/missing evidence produces needs_review with explicit reasons. No
    existing run, source, journal, checkpoint or acceptance flag is rewritten.
    A pass does not establish cooperation or B-versus-D transfer benefit.
    """
    run, output, verification = _root(run_dir), _root(output_dir), _root(verification_report)
    _output_directory(run, output, verification)
    sources = source_identity()
    report = {"schema_version": "p45-phase4-acceptance-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "needs_review", "phase4_acceptance": False, "phase4_engineering_acceptance": False,
        "acceptance_scope": "one completed D run: engineering correctness, immutable lineage and local deployment",
        "run_dir": str(run), "verification_report": str(verification), "source_sha256": sources,
        "requirement_coverage": None, "audit": None, "deployment": None, "reasons": [],
        "research_training_executed": False, "new_environment_interactions": 0, "new_model_updates": 0,
        "native_environment_evaluation_executed": False, "learned_student_quality_established": False,
        "b_versus_d_efficacy_established": False, "cohort_acceptance": False,
        "fresh_cohort_eligible": False, "automatic_phase5_launch": False,
        "pending": ["matched reduced B execution", "prospective cohort freeze and eligibility review",
                    "P5 final real-environment evaluation and descriptive paired analysis"]}
    try:
        _verified, coverage, provenance = load_valid_verification(verification)
        report["requirement_coverage"], report["verification_evidence"] = coverage, provenance
        audit = audit_campaign(run, device=device)
        report["audit"] = audit
        if audit.get("engineering_passed") is not True:
            raise ValueError("completed campaign did not satisfy the engineering artifact audit")
        exported = audit.get("actor_export", {})
        path = _root(exported["path"])
        deployment = probe_actor_export(path, output_dir=output / "deployment",
                                        source_device=exported.get("source_device"))
        report["deployment"] = deployment
        if (deployment.get("status") != "passed" or deployment.get("passed") is not True
                or deployment.get("local_only") is not True or deployment.get("forbidden_modules_loaded") != []
                or any(deployment.get(key) != exported.get(key) for key in ("manifest_sha256", "payload_sha256"))):
            raise ValueError("isolated deployment did not authenticate the audited local actor export")
        _bound_files_unchanged(audit.get("bound_files"))
        # Recheck the implementation evidence as well as the campaign after the
        # fresh subprocess; a moving source/test/log cannot produce acceptance.
        load_valid_verification(verification)
        if source_identity() != sources:
            raise ValueError("acceptance implementation sources changed during the audit")
        report.update(status="passed", phase4_acceptance=True, phase4_engineering_acceptance=True)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        report["reasons"].append(f"{type(error).__name__}: {error}")
    report["source_unchanged_during_acceptance"] = source_identity() == sources
    if not report["source_unchanged_during_acceptance"]:
        report.update(status="needs_review", phase4_acceptance=False, phase4_engineering_acceptance=False)
        if not any("sources changed" in reason for reason in report["reasons"]):
            report["reasons"].append("acceptance implementation sources changed during the audit")
    # These outputs describe this acceptance operation; they never promote or
    # mutate the historical false flags inside P4.4 and local actor exports.
    if report["audit"] is not None:
        write_report(output / "campaign_audit.json", report["audit"])
        report["campaign_audit"] = {"path": "campaign_audit.json", "sha256": file_sha256(output / "campaign_audit.json")}
    write_report(output / "phase4_acceptance.json", report)
    return report
