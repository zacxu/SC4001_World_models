"""Finish P5.6-P5.8 on the original Linux server, using saved evidence only."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys

from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from . import common, evidence
from .evidence import read_report, safe_path, timestamp, write_report


def _components():
    from .secondary import build_secondary, validate_secondary_sources
    from .resources import build_resources, validate_resource_sources
    from .figures import write_figures
    from .gate import build_acceptance
    return build_secondary, validate_secondary_sources, build_resources, validate_resource_sources, write_figures, build_acceptance


def _csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: "" if row.get(k) is None else row[k] for k in fields})
    path.write_text(stream.getvalue(), encoding="utf-8", newline="")
    return common.reference(path, receipt=False)


def _leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, member in sorted(value.items()):
            yield from _leaves(member, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, member in enumerate(value):
            yield from _leaves(member, f"{prefix}[{index}]")
    else:
        yield prefix, value


def write_tables(output, secondary, resources):
    rows = []
    for record in resources["per_run"]:
        counts = record["logical_counters"]
        rows.append({**{k: record.get(k) for k in ("case", "variant", "master_seed",
            "recorded_scientific_unit_elapsed_seconds", "complete_scientific_elapsed_cost_known",
            "actual_GPU_occupancy_seconds", "process_CPU_seconds",
            "peak_GPU_memory_bytes", "peak_CPU_memory_bytes", "current_retained_file_bytes")},
            **{k: counts[k] for k in ("local_PPO_cycles", "local_actor_steps", "local_critic_steps",
                "teacher_PPO_cycles", "teacher_actor_steps", "teacher_critic_steps", "distillation_optimizer_steps",
                "imagined_joint_transitions_total", "new_environment_interactions", "new_model_updates")}})
    fields = tuple(rows[0]) if rows else ("case",)
    refs = [_csv(output / "resources.csv", rows, fields)]
    diagnostics = []
    for arm in secondary["training_diagnostics"]:
        for stage in ("teacher_PPO", "distillation", "local_PPO"):
            for metric, value in _leaves(arm[stage]):
                diagnostics.append({"master_seed": arm["master_seed"], "variant": arm["variant"],
                    "stage": stage, "metric": metric, "value": value})
    refs.append(_csv(output / "training_diagnostics.csv", diagnostics,
                     ("master_seed", "variant", "stage", "metric", "value")))
    return refs


def _failure(output, error, manifest):
    report = {"schema_version": "p58-phase5-acceptance-v1", "status": "needs_review",
        "phase5_acceptance": False, "execution_acceptance": False, "evidence_acceptance": False,
        "cohort_acceptance": False, "reasons": [f"{type(error).__name__}: {error}"],
        "created_at": timestamp(), "cohort_identity": manifest.get("cohort_identity"),
        "source_sha256": manifest.get("postprocessing_source_sha256"),
        "successful_transfer_established": False, "statistical_superiority_established": False,
        "missing_evidence_imputed": False, "research_training_executed_by_gate": False,
        "research_evaluation_executed_by_gate": False, "native_environment_opened": False,
        "new_environment_interactions": 0, "new_model_updates": 0, "new_policy_optimizer_updates": 0,
        "report_path": str(output / "phase5_acceptance.json")}
    write_report(output / "phase5_acceptance.json", report)
    return report


def complete(cohort_path, output_dir, verification_report):
    common.require_server()
    proof = evidence.validate_verification(verification_report)
    api = common.apis()
    cohort_path = safe_path(cohort_path)
    cohort = api.inspect_cohort(cohort_path, check_inputs=False)
    output = common.guard_output(output_dir, cohort, verification_report)
    before = evidence.source_identity()
    if proof["frozen_source_sha256"] != cohort["source_sha256"]:
        raise ValueError("reporting verification does not preserve the frozen scientific sources")
    output.mkdir(parents=True)
    manifest = {"schema_version": "p565758-reporting-run-v1", "created_at": timestamp(),
        "cohort": common.reference(cohort_path), "cohort_identity": cohort["cohort_identity"],
        "verification": common.reference(verification_report),
        "frozen_source_sha256": cohort["source_sha256"], "postprocessing_source_sha256": before,
        "scientific_protocol_amended": False, "existing_outcomes_replaced": False,
        "scientific_training_or_test_executed": False, "reporting_runtime": {"python": sys.version,
        "platform": sys.platform}, "output_dir": str(output)}
    write_report(output / "completion_manifest.json", manifest)
    try:
        context = common.load_context(cohort_path, output, backend=api)
        context["source_identity"] = before
        context["references"] += [common.reference(verification_report), common.reference(output / "completion_manifest.json")]
        (build_secondary, check_secondary, build_resources, check_resources, write_figures, build_gate) = _components()
        secondary = build_secondary(context)
        resources = build_resources(context)
        write_report(output / "secondary_evidence.json", secondary)
        write_report(output / "resource_summary.json", resources)
        tables = write_tables(output, secondary, resources)
        figures = write_figures(context, resources, output / "figures")
        if (figures.get("status") != "complete" or figures.get("fixtures_only") is not False
                or figures.get("actual_graphics_rendered") is not True
                or figures.get("matplotlib_version") != "3.11.2"):
            raise ValueError("formal completion requires two actual pinned-renderer figures, not fixture graphics")
        write_report(output / "figures.json", figures)
        figure_files = figures["files"]
        figure_refs = list(figure_files.values()) if isinstance(figure_files, dict) else figure_files
        svg = [row for row in figure_refs if str(row["path"]).endswith(".svg")]
        if len(svg) != 2:
            raise ValueError("reporting requires exactly two endpoint SVG figures")
        common.validate_references(figure_refs)
        limitations = {"schema_version": "p5-deviation-and-limitations-v1", "cohort_identity": cohort["cohort_identity"],
            "scientific_protocol_amended": False, "analysis_changed_after_outcomes": False,
            "reporting_extension_added_after_training": True,
            "resource_measurement_gaps": resources.get("missing_measurements", []),
            "diagnostic_warnings": secondary.get("warnings", []),
            "scope_disclosures": ["two reused development seeds; two authored unseen Relay layouts",
                "validation-informed Relay selection; formal P2.5 quality acceptance remains unresolved",
                "D uses additional teacher/distillation compute; no compute-matched control",
                "one fixed stochastic final endpoint; no online AUC, significance or component-ablation claim"],
            "measured_outcomes_preserved": True, "failed_or_unknown_costs_imputed": False}
        write_report(output / "deviations.json", limitations)
        context["deviations"] = [limitations]
        context["figure_evidence"] = figures
        context["deliverables"] = {"secondary": common.reference(output / "secondary_evidence.json"),
            "resources": common.reference(output / "resource_summary.json"), "figures": svg,
            "figure_manifest": common.reference(output / "figures.json"), "tables": tables,
            "deviations": common.reference(output / "deviations.json")}
        context["references"] += [*tables, *figure_refs, *figures.get("data", [])]
        context["references"] += [row for name, row in context["deliverables"].items() if name not in ("figures", "tables")]
        check_secondary(secondary)
        check_resources(resources)
        common.validate_context(context, backend=api)
        evidence.validate_verification(verification_report)
        if before != evidence.source_identity():
            raise ValueError("reporting source changed during completion")
        context_path = output / "evidence_context.json"
        write_report(context_path, context)
        report = build_gate(context, secondary, resources, proof)
        report.update(created_at=timestamp(), report_path=str(output / "phase5_acceptance.json"),
            context=common.reference(context_path), reporting_manifest=common.reference(output / "completion_manifest.json"))
        write_report(output / "phase5_acceptance.json", report)
        return report
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        return _failure(output, error, manifest)


def inspect_completion(path):
    common.require_server()
    path = safe_path(path)
    report = read_report(path)
    if report.get("schema_version") != "p58-phase5-acceptance-v1" or report.get("report_path") != str(path):
        raise ValueError("not a canonical Phase 5 completion report")
    if report.get("status") != "passed":
        return report
    context_ref = report["context"]
    common.validate_references([context_ref, report["reporting_manifest"]])
    context = read_report(context_ref["path"])
    manifest = read_report(report["reporting_manifest"]["path"])
    proof = evidence.validate_verification(manifest["verification"]["path"])
    if (manifest["postprocessing_source_sha256"] != evidence.source_identity()
            or context["source_identity"] != evidence.source_identity()
            or report["source_sha256"] != context["source_identity"]):
        raise ValueError("completion reporting sources changed")
    common.validate_context(context)
    (_, check_secondary, _, check_resources, _, build_gate) = _components()
    secondary = read_report(context["deliverables"]["secondary"]["path"])
    resources = read_report(context["deliverables"]["resources"]["path"])
    check_secondary(secondary)
    check_resources(resources)
    rebuilt = build_gate(context, secondary, resources, proof)
    original = {k: v for k, v in report.items() if k not in ("created_at", "report_path", "context", "reporting_manifest")}
    if canonical_hash(rebuilt) != canonical_hash(original):
        raise ValueError("Phase 5 acceptance no longer reproduces its authenticated evidence")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="synthetic CPU implementation checks only")
    verify.add_argument("--output-dir", required=True)
    finish = commands.add_parser("complete", help="server CPU: summarize saved records and audit final completeness")
    finish.add_argument("--cohort", required=True)
    finish.add_argument("--output-dir", required=True)
    finish.add_argument("--verification-report", required=True)
    inspect = commands.add_parser("inspect", help="server CPU: authenticate a saved completion report")
    inspect.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            report = evidence.verify(args.output_dir)
            good = report["status"] == evidence.STATUS
        elif args.command == "complete":
            report = complete(args.cohort, args.output_dir, args.verification_report)
            good = report["status"] == "passed"
        else:
            report = inspect_completion(args.report)
            good = report["status"] == "passed"
        keys = ("status", "tests_run", "failures", "errors", "skipped", "phase5_acceptance", "execution_acceptance",
            "successful_transfer_established", "successful_final_episodes_observed", "reasons", "warnings", "report_path")
        print(json.dumps({k: report[k] for k in keys if k in report}, indent=2, allow_nan=False))
        return 0 if good else 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        print(f"P5.6-P5.8 error ({type(error).__name__}): {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
