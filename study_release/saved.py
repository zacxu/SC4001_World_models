"""Relocatable CPU reproduction from a sealed evidence release; no simulator."""
from __future__ import annotations

import sys
# Packaged sources are sealed bytes. Use python -B as documented so even this
# module's initial import cannot create an unlisted __pycache__ directory.
sys.dont_write_bytecode = True

import argparse
import csv
import io
import json
from pathlib import Path

from .evidence import canonical_hash, file_sha256, reference, safe_path, timestamp, write_report
from .packaging import inspect_release


def _components():
    from world_models_endpoint.analysis import aggregate_records
    from study_reporting.figures import build_figure_data, write_figures
    from study_reporting.phase5 import write_tables
    return aggregate_records, build_figure_data, write_figures, write_tables


def _user_path(value):
    # Users may legitimately say '..' from the packaged source/ directory;
    # internal archive references remain confined POSIX paths without '..'.
    raw = Path(value).absolute()
    if any(p.is_symlink() for p in (raw, *raw.parents)):
        raise ValueError("saved reproduction paths cannot use symbolic links")
    return raw.resolve()


def _check_origins(components, source):
    for component in components:
        module = sys.modules.get(component.__module__)
        filename = getattr(module, "__file__", None)
        if filename is None or not Path(filename).resolve().is_relative_to(source):
            raise ValueError("start a fresh Python process from the sealed release's source directory")


def _input(root, manifest, row):
    if type(row) is not dict or type(row.get("path")) is not str:
        raise ValueError("saved reproduction input needs an indexed package reference")
    relative = Path(row["path"])
    if relative.is_absolute() or "\\" in row["path"] or any(p in (".", "..") for p in row["path"].split("/")):
        raise ValueError("saved reproduction input escaped its package")
    path = safe_path(root / relative)
    indexed = next((item for item in manifest["files"] if item["path"] == row["path"]), None)
    if (not path.is_relative_to(root) or indexed is None or indexed["sha256"] != row.get("sha256")
            or file_sha256(path) != row["sha256"] or "bytes" in row and path.stat().st_size != row["bytes"]):
        raise ValueError("saved reproduction input differs from the sealed inventory")
    return path


def _json(root, manifest, row):
    return json.loads(_input(root, manifest, row).read_text(encoding="utf-8"))


def table_bytes(metrics):
    """Use the frozen P5.5 column order and representation exactly."""
    common = ("master_seed", "variant", "episodes", "successes", "success_rate", "return_sum", "mean_return",
        "latch_episodes", "latch_episode_rate", "joint_transitions", "simulator_tics", "mean_length",
        "successful_completion_length_denominator", "mean_successful_completion_length",
        "successful_completion_tics_denominator", "mean_successful_completion_tics")
    tables = {"arms.csv": (metrics["arms"], (*common, "final_actor_identity")),
        "paired_differences.csv": (metrics["paired_differences"], tuple(metrics["paired_differences"][0])),
        "per_layout.csv": (metrics["per_layout"], (*common, "template_id")),
        "episodes.csv": (metrics["episode_records"], tuple(metrics["episode_records"][0]))}
    result = {}
    for name, (rows, fields) in tables.items():
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row[key] for key in fields})
        result[name] = stream.getvalue().encode("utf-8")
    return result


def regenerate(release_dir, output_dir):
    root, output = _user_path(release_dir), _user_path(output_dir)
    if output.exists():
        raise FileExistsError("saved reproduction requires a fresh output directory")
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("regeneration output must stay outside the sealed release")
    manifest = inspect_release(root)
    inputs = json.loads((root / "reproduction_inputs.json").read_text(encoding="utf-8"))
    if (inputs.get("schema_version") != "p62-reproduction-inputs-v1"
            or inputs.get("cohort_identity") != manifest["cohort_identity"]
            or inputs.get("protocol_sha256") != manifest["protocol_sha256"]):
        raise ValueError("saved reproduction mapping belongs to a different cohort or protocol")
    # Pure original readers discover protocol/config files relative to their
    # packaged source tree; neither original absolute server paths nor model
    # payloads are opened by this path.
    source = root / "source"
    sys.path.insert(0, str(source / "src"))
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    components = _components()
    _check_origins(components, source)
    aggregate, figure_data, render_figures, resource_tables = components
    evaluations = [_json(root, manifest, row) for row in inputs["evaluations"]]
    bank = _json(root, manifest, inputs["bank"])
    if inputs.get("bank_key") is not None:
        bank = bank[inputs["bank_key"]]
    archived_analysis = _json(root, manifest, inputs["analysis"])
    metrics = aggregate(evaluations, bank=bank, cohort_identity=manifest["cohort_identity"],
        protocol_sha256=manifest["protocol_sha256"], require_native=True)
    if canonical_hash(metrics) != canonical_hash(archived_analysis["analysis"]):
        raise ValueError("saved native episodes do not reproduce the archived paired analysis")
    resources, secondary = (_json(root, manifest, inputs[key]) for key in ("resources", "secondary"))
    context = {"cohort": {"cohort_identity": manifest["cohort_identity"], "protocol_sha256": manifest["protocol_sha256"]},
        "analysis": {"status": "complete", "cohort_identity": manifest["cohort_identity"], "analysis": metrics},
        "references": [], "source_identity": manifest["reporting_source_sha256"]}
    numbers = figure_data(context, resources)
    archived_data = {_input(root, manifest, row).name: _json(root, manifest, row) for row in inputs["figure_data"]}
    if (set(archived_data) != {"endpoint_results_data.json", "paired_effects_compute_data.json"}
            or canonical_hash(numbers[0]) != canonical_hash(archived_data["endpoint_results_data.json"])
            or canonical_hash(numbers[1]) != canonical_hash(archived_data["paired_effects_compute_data.json"])):
        raise ValueError("regenerated figure numbers differ from the archived plotted data")
    expected = {_input(root, manifest, row).name: _input(root, manifest, row) for row in inputs["tables"]}
    required = {"arms.csv", "paired_differences.csv", "per_layout.csv", "episodes.csv", "resources.csv", "training_diagnostics.csv"}
    if len(inputs["tables"]) != 6 or set(expected) != required:
        raise ValueError("saved reproduction requires all six archived table files")
    tables = table_bytes(metrics)
    if any(data != expected[name].read_bytes() for name, data in tables.items()):
        raise ValueError("regenerated performance table bytes differ from the archive")
    output.mkdir(parents=True)
    for name, data in tables.items():
        (output / name).write_bytes(data)
    resource_tables(output, secondary, resources)
    if any((output / name).read_bytes() != path.read_bytes() for name, path in expected.items()):
        raise ValueError("regenerated resource/diagnostic tables differ from the archived bytes")
    figures = render_figures(context, resources, output / "figures")
    if (figures.get("status") != "complete" or figures.get("fixtures_only") is not False
            or figures.get("actual_graphics_rendered") is not True):
        raise ValueError("saved reproduction requires actual figure rendering")
    # Recheck the source release after computation; never repair/mutate it.
    inspect_release(root)
    write_report(output / "paired_results.json", {"schema_version": "p62-saved-paired-results-v1",
        "cohort_identity": manifest["cohort_identity"], "analysis": metrics,
        "archived_native_outcomes_reused": True, "research_evaluation_executed": False})
    report = {"schema_version": "p62-saved-reproduction-v1", "status": "complete", "created_at": timestamp(),
        "cohort_identity": manifest["cohort_identity"], "release_manifest": reference(root / "release_manifest.json"),
        "numerical_reproduction": True, "six_tables_byte_identical": True,
        "two_figures_rendered": True, "graphics_bytes_platform_independent": False,
        "tables": [reference(output / name, receipt=False) for name in sorted(required)], "figures": figures["files"],
        "research_training_executed": False, "research_evaluation_executed": False,
        "model_or_actor_weights_loaded": False, "native_environment_opened": False,
        "new_environment_interactions": 0, "new_model_updates": 0, "new_policy_optimizer_updates": 0,
        "successful_transfer_established": False, "phase6_acceptance": False,
        "report_path": str(output / "reproduction_report.json")}
    write_report(output / "reproduction_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("regenerate")
    run.add_argument("--release-dir", required=True)
    run.add_argument("--output-dir", required=True)
    check = commands.add_parser("inspect")
    check.add_argument("--release-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = regenerate(args.release_dir, args.output_dir) if args.command == "regenerate" else inspect_release(_user_path(args.release_dir))
        print(json.dumps({key: result[key] for key in ("status", "cohort_identity", "numerical_reproduction",
            "six_tables_byte_identical", "two_figures_rendered", "report_path") if key in result}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"P6 saved reproduction error ({type(error).__name__}): {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
