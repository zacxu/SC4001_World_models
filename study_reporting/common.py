"""Authenticate the completed cohort on its original Linux server, read-only."""
from __future__ import annotations

from pathlib import Path
import platform
from types import SimpleNamespace

from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from .evidence import ROOT, frozen_source_identity, read_report, safe_path


AUTH_KEYS = ("cohort", "analysis", "final_ledger", "final_actors", "evaluation_traces", "input_hashes", "frozen_sources")


def require_server():
    if platform.system() != "Linux":
        raise ValueError("run scientific post-processing on the original Linux server; local verification uses fixtures only")


def reference(path, *, receipt=True):
    path = safe_path(path)
    if receipt:
        read_report(path)
    if not path.is_file():
        raise ValueError(f"missing reporting evidence: {path}")
    return {"path": str(path), "sha256": file_sha256(path)}


def validate_references(rows):
    if type(rows) is not list:
        raise ValueError("reporting evidence references must be a list")
    seen = {}
    for row in rows:
        path = safe_path(row["path"])
        if (str(path) != row["path"] or not path.is_file()
                or file_sha256(path) != row["sha256"]
                or row["path"] in seen and seen[row["path"]] != row["sha256"]):
            raise ValueError(f"reporting evidence changed: {path}")
        seen[row["path"]] = row["sha256"]


def apis():
    # Evaluation is a saved-record reader here. Never call evaluate_endpoint,
    # native environment constructors, controller run/resume or an optimizer.
    from world_models_endpoint import analysis, cohort, evaluation, ledger
    return SimpleNamespace(inspect_cohort=cohort.inspect_cohort, audit_run=cohort.audit_run,
        inspect_run=cohort.inspect_run, load_evaluation=evaluation.load_evaluation,
        analyze_cohort=analysis.analyze_cohort, aggregate_records=analysis.aggregate_records,
        summarize_cohort=ledger.summarize_cohort)


def guard_output(output, cohort, proof_path):
    output = safe_path(output)
    if output.exists():
        raise FileExistsError("completion needs a fresh directory; retain previous results")
    protected = [ROOT / name for name in ("src", "scripts", "tests", "configs", "scenarios", "study_reporting")]
    protected += [safe_path(row["run_dir"]) for row in cohort["run_matrix"]]
    protected += [safe_path(row["path"]).parent for row in cohort["inputs"]["bound_files"]]
    protected += [safe_path(proof_path).parent]
    for row in cohort["run_matrix"]:
        path = Path(row["run_dir"]) / "endpoint_test_claim.json"
        if path.exists() or path.with_suffix(".sha256").exists():
            claim = read_report(path)
            protected.append(safe_path(claim["output_dir"]))
        else:
            protected.append(safe_path(cohort["report_path"]).parent / "evaluation")
    if (safe_path(cohort["report_path"]).is_relative_to(output)
            or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected)):
        raise ValueError("completion output must be separate from frozen sources, training, test outputs and input evidence")
    return output


def evaluation_paths(cohort):
    paths = []
    for row in cohort["run_matrix"]:
        claim_path = Path(row["run_dir"]) / "endpoint_test_claim.json"
        claim = read_report(claim_path)
        if (claim.get("schema_version") != "p54-single-final-test-claim-v1"
                or claim.get("cohort_identity") != cohort["cohort_identity"]
                or safe_path(claim["run_dir"]) != safe_path(row["run_dir"])
                or claim.get("bank_sha256") != cohort["bank_sha256"]):
            raise ValueError("final-test claim differs from its frozen training slot")
        paths.append(safe_path(claim["output_dir"]) / "evaluation.json")
    if len(paths) != 4 or len(set(paths)) != 4:
        raise ValueError("completion requires four distinct claimed final-test reports")
    return paths


def _analysis_files(report):
    parent = safe_path(report["report_path"]).parent
    if set(report.get("files", {})) != {"arms.csv", "paired_differences.csv", "per_layout.csv", "episodes.csv"}:
        raise ValueError("paired analysis is missing required deterministic tables")
    rows = []
    for name, row in report["files"].items():
        if row.get("path") != name:
            raise ValueError("paired table path differs from its published name")
        rows.append({"path": str(parent / name), "sha256": row["sha256"]})
    validate_references(rows)
    return rows


def load_context(cohort_path, output, *, backend=None):
    """Freshly summarize existing saved episodes; no new scientific interaction."""
    backend = backend or apis()
    cohort_path, output = safe_path(cohort_path), safe_path(output)
    c = backend.inspect_cohort(cohort_path, check_inputs=True)
    if c["source_sha256"] != frozen_source_identity():
        raise ValueError("the registered training source set changed")
    runs = []
    for row in c["run_matrix"]:
        backend.audit_run(row["run_dir"])
        run = backend.inspect_run(row["run_dir"])
        if run["status"] != "complete" or run.get("training_complete") is not True:
            raise ValueError(f"training is unfinished: {row['case']}")
        runs.append(run)
    paths = evaluation_paths(c)
    report = backend.analyze_cohort(cohort_path, paths, output / "endpoint_analysis")
    if report.get("status") != "complete" or report.get("analysis") is None:
        raise ValueError(f"paired analysis needs review: {report.get('reasons', [])}")
    publications = backend.summarize_cohort(cohort_path, output / "ledger")
    ledger_path = safe_path(publications["snapshot_path"])
    ledger = read_report(ledger_path)["snapshot"]
    if ledger.get("status") != "complete" or ledger.get("missing_run_slots"):
        raise ValueError("the final ledger does not contain four completed runs")
    evaluations = [backend.load_evaluation(path, require_native=True) for path in paths]
    metrics = backend.aggregate_records(evaluations, bank=c["bank"], cohort_identity=c["cohort_identity"],
        protocol_sha256=c["protocol_sha256"], require_native=True)
    if (canonical_hash(report["analysis"]) != canonical_hash(metrics)
            or report.get("cohort_identity") != c["cohort_identity"]
            or report.get("source_sha256") != c["source_sha256"]
            or report.get("source_unchanged") is not True):
        raise ValueError("paired endpoint analysis does not reproduce the complete native records")
    expected_refs = {str(path): file_sha256(path) for path in paths}
    if {r["path"]: r["sha256"] for r in report["evaluations"]} != expected_refs:
        raise ValueError("paired analysis substituted an evaluation source")
    for run in runs:
        key = f"{run['variant']}/seed_{run['master_seed']}"
        summary = ledger["runs"][key]
        if summary["run_identity"] != run["run_identity"] or summary["last_checkpoint"] != run["last_checkpoint"]:
            raise ValueError("ledger and checkpoint audits describe different training boundaries")
        evaluation = next(e for e in evaluations if e["variant"] == run["variant"] and e["master_seed"] == run["master_seed"])
        cost = evaluation["interaction_accounting"]
        if (summary["evaluation"]["status"] != "complete"
                or summary["evaluation"]["evaluation_joint_transitions"] != cost["published_real_joint_transitions"]
                or summary["evaluation"]["evaluation_simulator_tics"] != cost["known_simulator_tics_total"]):
            raise ValueError("ledger evaluation accounting predates or differs from final test records")
    refs = [reference(cohort_path), reference(report["report_path"]), reference(ledger_path)]
    refs += [reference(p) for p in paths] + _analysis_files(report)
    refs += [reference(run["audit_path"]) for run in runs]
    context = {"schema_version": "p5-completion-context-v1", "cohort": c, "analysis": report,
        "ledger": ledger, "runs": runs, "evaluation_reports": evaluations,
        "evaluation_paths": [str(p) for p in paths], "ledger_path": str(ledger_path),
        "authentication": {key: True for key in AUTH_KEYS}, "references": refs,
        "scientific_protocol_amended": False, "new_training_or_test_execution": False}
    return context


def validate_context(context, *, backend=None):
    """Recheck moving inputs immediately before final publication/inspection."""
    backend = backend or apis()
    c = context["cohort"]
    current = backend.inspect_cohort(c["report_path"], check_inputs=True)
    if canonical_hash(current) != canonical_hash(c) or c["source_sha256"] != frozen_source_identity():
        raise ValueError("frozen cohort changed during reporting")
    validate_references(context["references"])
    if (canonical_hash(read_report(context["analysis"]["report_path"])) != canonical_hash(context["analysis"])
            or canonical_hash(read_report(context["ledger_path"])["snapshot"]) != canonical_hash(context["ledger"])):
        raise ValueError("embedded analysis or ledger differs from its authenticated publication")
    if (len(context["runs"]) != 4 or len(context["evaluation_reports"]) != 4
            or context["evaluation_paths"] != [str(p) for p in evaluation_paths(c)]
            or {r["run_dir"] for r in context["runs"]} != {r["run_dir"] for r in c["run_matrix"]}):
        raise ValueError("completion context must retain all four registered runs and native reports")
    for run in context["runs"]:
        now = backend.inspect_run(run["run_dir"])
        if canonical_hash(now) != canonical_hash(run) or now["status"] != "complete":
            raise ValueError("a training boundary changed during post-processing")
    for path, prior in zip(context["evaluation_paths"], context["evaluation_reports"]):
        current_report = backend.load_evaluation(path, require_native=True)
        if canonical_hash(current_report) != canonical_hash(prior):
            raise ValueError("native episode evidence changed during reporting")
    metrics = backend.aggregate_records(context["evaluation_reports"], bank=c["bank"],
        cohort_identity=c["cohort_identity"], protocol_sha256=c["protocol_sha256"], require_native=True)
    if canonical_hash(metrics) != canonical_hash(context["analysis"]["analysis"]):
        raise ValueError("saved native records no longer reproduce the paired analysis")
    return True
