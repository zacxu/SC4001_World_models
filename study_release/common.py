"""Original-server evidence handoff; no training, model inference or simulator run."""
from __future__ import annotations

from pathlib import Path
import platform

from .evidence import ROOT, canonical_hash, read_report, reference, safe_path, validate_references


def require_server():
    if platform.system() != "Linux":
        raise ValueError("run scientific diagnosis and packaging on the original Linux server")


def inspect_phase5(path):
    from study_reporting.phase5 import inspect_completion
    return inspect_completion(path)


def load_completed(path):
    require_server()
    path = safe_path(path)
    completion = inspect_phase5(path)
    if completion.get("status") != "passed" or completion.get("phase5_acceptance") is not True:
        raise ValueError("P6.1-P6.2 require a passed, authenticated Phase 5 completion")
    context = read_report(completion["context"]["path"])
    context["completion"] = completion
    context["completion_path"] = str(path)
    context["secondary"] = read_report(context["deliverables"]["secondary"]["path"])
    context["resources"] = read_report(context["deliverables"]["resources"]["path"])
    manifest = read_report(completion["reporting_manifest"]["path"])
    proof = Path(manifest["verification"]["path"])
    context["release_inputs"] = [reference(path), reference(completion["context"]["path"]),
        reference(completion["reporting_manifest"]["path"]), reference(proof),
        reference(proof.parent / "tests.txt", receipt=False)]
    context["commands_provenance"] = {"historical_shell_commands_recorded": False,
        "physical_CUDA_VISIBLE_DEVICES_recorded": False,
        "evidence": "run/endpoint manifests and attempt journals bind configuration, runtime and invocation IDs; shell argv was not recorded",
        "reproduction_recipes_are_historical_command_transcripts": False,
        "limitation": "exact original shell commands and physical GPU assignments cannot be recovered from invocation UUIDs"}
    return context


def validate_completed(context):
    now = inspect_phase5(context["completion_path"])
    if canonical_hash(now) != canonical_hash(context["completion"]):
        raise ValueError("Phase 5 completion evidence changed during Phase 6 processing")
    validate_references(context["release_inputs"])
    return True


def attach_history(context):
    """Retain the already cited development reports separately, when available."""
    names = (
        ("cover_model_validation", "artifacts/phase25_runs/shortened_20261109_100750/campaign/report.json"),
        ("B_development", "artifacts/phase36_runs/relay_week_seed0_100k_20261111_082445/baseline/evaluation.json"),
        ("B_development", "artifacts/phase36_runs/relay_week_seed0_100k_20261111_082445/trained/evaluation.json"),
        ("B_development", "artifacts/phase35_runs/B_week_check_fp32_v1/relay/seed_0/checkpoints/state_00000020_86a71d32/manifest.json"))
    history, unavailable = [], []
    for kind, name in names:
        path = ROOT / name
        if not path.is_file() or not path.with_suffix(".sha256").is_file():
            unavailable.append({"kind": kind, "path": str(path), "status": "missing_report_or_receipt"})
            continue
        row = reference(path)
        context["release_inputs"].append(row)
        history.append({"kind": kind, "report_path": str(path), "sha256": row["sha256"],
            "summary": "separate development provenance; excluded from four-run paired endpoint metrics"})
        if kind == "cover_model_validation":
            campaign = read_report(path)
            for case in campaign.get("cases", []):
                case_path = safe_path(case["report_path"])
                ref = reference(case_path)
                if not case_path.is_relative_to(ROOT) or ref["sha256"] != case["report_sha256"]:
                    raise ValueError("historical P2.5 case differs from its archived campaign")
                context["release_inputs"].append(ref)
    context["historical_evidence"] = history
    context["historical_evidence_unavailable"] = unavailable
    return context


def guard_output(output, context, proof_path):
    output = safe_path(output)
    if output.exists():
        raise FileExistsError("Phase 6 requires a fresh output directory")
    protected = [ROOT / name for name in ("src", "scripts", "tests", "configs", "scenarios", "study_reporting", "study_release")]
    protected += [safe_path(row["run_dir"]) for row in context["cohort"]["run_matrix"]]
    protected += [safe_path(row["path"]).parent for row in context["cohort"]["inputs"]["bound_files"]]
    protected += [safe_path(context["completion_path"]).parent, safe_path(proof_path).parent]
    protected += [safe_path(p).parent for p in context["evaluation_paths"]]
    if (safe_path(context["cohort"]["report_path"]).is_relative_to(output)
            or any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected)):
        raise ValueError("Phase 6 outputs must be separate from sources and all prior evidence")
    return output
