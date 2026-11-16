"""P6.1-P6.2: original-server saved-trace diagnosis and evidence packaging."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys

from . import common, evidence
from .evidence import canonical_hash, read_report, reference, safe_path, timestamp, validate_references, write_report


def _components():
    from .diagnosis import build_diagnosis, validate_diagnosis_sources, render_markdown
    from .packaging import build_release, inspect_release
    return build_diagnosis, validate_diagnosis_sources, render_markdown, build_release, inspect_release


def _selected_csv(report):
    output = io.StringIO(newline="")
    fields = ("master_seed", "variant", "category", "episode_index", "template_id", "termination_cause",
        "episode_return", "latch_activated", "joint_transitions", "trace_path", "terminal_line", "terminal_row_sha256")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for selected in report["selected_traces"]:
        last = selected["summary"]["last_transition"]
        writer.writerow({**{key: selected[key] for key in fields[:5]},
            **{key: selected["record"][key] for key in fields[5:9]},
            "trace_path": last["path"], "terminal_line": last["line"], "terminal_row_sha256": last["row_sha256"]})
    return output.getvalue()


def complete(phase5_acceptance, output_dir, verification_report):
    common.require_server()
    proof = evidence.validate_verification(verification_report)
    context = common.load_completed(phase5_acceptance)
    if (proof["frozen_source_sha256"] != context["cohort"]["source_sha256"]
            or proof["reporting_source_sha256"] != context["source_identity"]):
        raise ValueError("Phase 6 proof differs from the completed experiment/reporting sources")
    output = common.guard_output(output_dir, context, verification_report)
    common.attach_history(context)
    before = evidence.source_identity()
    output.mkdir(parents=True)
    manifest = {"schema_version": "p6162-processing-run-v1", "created_at": timestamp(),
        "phase5_acceptance": reference(phase5_acceptance), "verification": reference(verification_report),
        "cohort_identity": context["cohort"]["cohort_identity"], "source_sha256": before,
        "frozen_source_sha256": proof["frozen_source_sha256"],
        "reporting_source_sha256": proof["reporting_source_sha256"],
        "phase6_argv": list(sys.argv), "phase6_argv_is_historical_training_command": False,
        "scientific_protocol_amended": False, "existing_outcomes_replaced": False,
        "research_training_executed": False, "research_evaluation_executed": False,
        "new_environment_interactions": 0, "new_model_updates": 0, "new_policy_optimizer_updates": 0}
    write_report(output / "processing_manifest.json", manifest)
    try:
        (build_diagnosis, check_diagnosis, markdown, build_release, inspect_release) = _components()
        diagnosis = build_diagnosis(context)
        if (diagnosis.get("status") != "complete" or diagnosis.get("fixtures_only") is not False
                or diagnosis.get("cohort_identity") != context["cohort"]["cohort_identity"]):
            raise ValueError("P6.1 requires complete, authenticated saved native traces, not fixture outputs")
        check_diagnosis(diagnosis)
        directory = output / "diagnosis"
        directory.mkdir()
        diagnosis["publication"] = {"implementation_verification": reference(verification_report),
            "phase5_acceptance": reference(phase5_acceptance), "source_sha256": before,
            "processing_manifest": reference(output / "processing_manifest.json")}
        write_report(directory / "diagnosis.json", diagnosis)
        (directory / "diagnosis.md").write_text(markdown(diagnosis), encoding="utf-8", newline="\n")
        (directory / "selected_traces.csv").write_text(_selected_csv(diagnosis), encoding="utf-8", newline="")
        context["diagnosis_path"] = str(directory / "diagnosis.json")
        context["diagnosis_artifacts"] = [reference(directory / name, receipt=name.endswith(".json"))
            for name in ("diagnosis.json", "diagnosis.md", "selected_traces.csv")]
        context["release_inputs"] += [*context["diagnosis_artifacts"], reference(verification_report),
            reference(Path(verification_report).parent / "tests.txt", receipt=False),
            reference(output / "processing_manifest.json")]
        context["release_source_identity"] = before
        release = build_release(context, diagnosis, output / "release")
        if release.get("status") != "complete" or release.get("cohort_identity") != context["cohort"]["cohort_identity"]:
            raise ValueError("P6.2 did not create a complete release of this cohort")
        check_diagnosis(diagnosis)
        common.validate_completed(context)
        evidence.validate_verification(verification_report)
        validate_references(context["diagnosis_artifacts"])
        inspected = inspect_release(output / "release")
        if canonical_hash(inspected) != canonical_hash(release) or before != evidence.source_identity():
            raise ValueError("release or implementation changed during final publication")
        report = {"schema_version": "p6162-completion-v1", "status": "complete", "created_at": timestamp(),
            "cohort_identity": context["cohort"]["cohort_identity"], "P6.1_complete": True, "P6.2_complete": True,
            "phase6_acceptance": False, "P6.3_paper_review_complete": False,
            "next_step": "P6.3 update manuscript and review claims; run saved-record reproduction separately",
            "phase5_acceptance": reference(phase5_acceptance), "processing_manifest": reference(output / "processing_manifest.json"),
            "diagnosis": context["diagnosis_artifacts"], "selected_trace_count": diagnosis["selected_count"],
            "missing_outcome_categories": diagnosis["missing_categories"],
            "release_manifest": reference(output / "release/release_manifest.json"),
            "release_files": len(release["files"]), "release_byte_integrity_checked": True,
            "saved_record_reproduction_executed": False,
            "independent_retraining_without_original_archive": False,
            "historical_evidence_unavailable": context.get("historical_evidence_unavailable", []),
            "source_sha256": before, "scientific_protocol_amended": False,
            "research_training_executed": False, "research_evaluation_executed": False,
            "native_environment_opened": False, "new_environment_interactions": 0,
            "new_model_updates": 0, "new_policy_optimizer_updates": 0, "successful_transfer_established": False,
            "report_path": str(output / "phase61_62_completion.json")}
        write_report(output / "phase61_62_completion.json", report)
        return report
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        report = {"schema_version": "p6162-completion-v1", "status": "needs_review", "created_at": timestamp(),
            "P6.1_complete": False, "P6.2_complete": False, "phase6_acceptance": False,
            "cohort_identity": context["cohort"]["cohort_identity"], "reasons": [f"{type(error).__name__}: {error}"],
            "research_training_executed": False, "research_evaluation_executed": False,
            "native_environment_opened": False, "new_environment_interactions": 0, "new_model_updates": 0,
            "new_policy_optimizer_updates": 0, "report_path": str(output / "phase61_62_completion.json")}
        write_report(output / "phase61_62_completion.json", report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="synthetic CPU implementation checks only")
    verify.add_argument("--output-dir", required=True)
    finish = commands.add_parser("complete", help="server CPU: diagnose saved traces and package evidence")
    finish.add_argument("--phase5-acceptance", required=True)
    finish.add_argument("--verification-report", required=True)
    finish.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = evidence.verify(args.output_dir) if args.command == "verify" else complete(
            args.phase5_acceptance, args.output_dir, args.verification_report)
        print(json.dumps({key: result[key] for key in ("status", "tests_run", "failures", "errors", "skipped",
            "P6.1_complete", "P6.2_complete", "phase6_acceptance", "selected_trace_count", "release_files",
            "next_step", "reasons", "report_path") if key in result}, indent=2))
        return 0 if result["status"] in ("complete", evidence.STATUS) else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"P6.1-P6.2 error ({type(error).__name__}): {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
