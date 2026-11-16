"""Descriptive paired final-endpoint results, never a training or test launcher.

The pure aggregation helper does not authenticate evidence. The public reader
authenticates the frozen cohort and every native report before aggregating, then
rechecks the bindings before publishing immutable analysis artifacts.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import io
import math
from pathlib import Path
import re

from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_study.bank import validate_test_bank
from world_models_study.evidence import ROOT, safe_path, source_identity, timestamp, write_report


SCHEMA_VERSION = "p55-paired-endpoint-analysis-v1"
EVALUATION_SCHEMA = "p54-final-test-evaluation-v1"
_CASES = tuple((seed, variant) for seed in (0, 1) for variant in ("B", "D"))
_NATIVE_FLAGS = ("native_backend", "native_engine", "native_environment_evaluation_executed",
                 "research_evaluation_executed")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_CAUSES = {"success", "deadline", "failure", "death"}


def _sha(value, name):
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 identity")
    return value


def _integer(value, name, *, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in the registered range")
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _sum(values):
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise ValueError("endpoint aggregation overflowed finite numeric bounds") from error
    return _number(result, "aggregated endpoint value")


def _difference(d, b):
    return _number(d - b, "paired endpoint difference")


def _episode_rows(rows, bank):
    if type(rows) is not list or len(rows) != 20:
        raise ValueError("every final actor needs exactly 20 complete bank episodes; missing evidence cannot be imputed")
    by_index = {}
    for row in rows:
        if type(row) is not dict:
            raise ValueError("endpoint episode must be an object")
        index = _integer(row.get("episode_index"), "episode index", maximum=19)
        if index in by_index:
            raise ValueError("duplicate endpoint bank episode")
        expected = bank["episodes"][index]
        if row.get("template_id") != expected["template_id"]:
            raise ValueError("endpoint episode layout differs from its fixed bank entry")
        if type(row.get("success")) is not bool or type(row.get("latch_activated")) is not bool:
            raise ValueError("episode success and latch indicators must be booleans")
        _number(row.get("episode_return"), "episode return")
        _integer(row.get("joint_transitions"), "episode joint decisions", minimum=1, maximum=400)
        _integer(row.get("simulator_tics"), "episode simulator tics")
        cause = row.get("termination_cause")
        if type(cause) is not str or cause not in _CAUSES or row["success"] != (cause == "success"):
            raise ValueError("episode success and terminal cause disagree")
        by_index[index] = deepcopy(row)
    if set(by_index) != set(range(20)):
        raise ValueError("endpoint episodes omit part of the registered bank")
    return [by_index[index] for index in range(20)]


def _metrics(rows):
    count = len(rows)
    successes = sum(row["success"] for row in rows)
    latches = sum(row["latch_activated"] for row in rows)
    success_lengths = [row["joint_transitions"] for row in rows if row["success"]]
    success_tics = [row["simulator_tics"] for row in rows if row["success"]]
    returns = _sum(row["episode_return"] for row in rows)
    lengths = sum(row["joint_transitions"] for row in rows)
    tics = sum(row["simulator_tics"] for row in rows)
    return {"episodes": count, "successes": successes, "success_rate": successes / count if count else None,
        "return_sum": returns, "mean_return": returns / count if count else None,
        "latch_episodes": latches, "latch_episode_rate": latches / count if count else None,
        "joint_transitions": lengths, "simulator_tics": tics,
        "mean_length": lengths / count if count else None,
        "termination_counts": {name: Counter(row["termination_cause"] for row in rows)[name]
                               for name in sorted(_CAUSES)},
        "successful_completion_length_denominator": successes,
        "mean_successful_completion_length": _sum(success_lengths) / successes if successes else None,
        "successful_completion_tics_denominator": successes,
        "mean_successful_completion_tics": _sum(success_tics) / successes if successes else None}


def aggregate_records(evaluations, *, bank, cohort_identity, protocol_sha256, require_native=True):
    """Deterministically summarize four already-authenticated final reports.

    This helper performs structural/protocol checks but cannot authenticate
    in-memory dictionaries or their origin. Explicit ``require_native=False``
    permits homogeneous synthetic fixtures and labels their output accordingly;
    the public scientific reader never enables that mode.
    """
    bank = validate_test_bank(bank)
    _sha(cohort_identity, "cohort identity")
    _sha(protocol_sha256, "scientific protocol identity")
    if protocol_sha256 != bank["protocol_sha256"]:
        raise ValueError("analysis scientific amendment differs from the fixed bank")
    if type(require_native) is not bool or type(evaluations) is not list or len(evaluations) != 4:
        raise ValueError("paired endpoint analysis requires exactly four registered final reports")
    reports, native_kinds, rng_backends, torch_versions = {}, set(), set(), set()
    for report in evaluations:
        if type(report) is not dict:
            raise ValueError("endpoint report must be an object")
        seed, variant = report.get("master_seed"), report.get("variant")
        if type(seed) is not int or type(variant) is not str or (seed, variant) not in _CASES:
            raise ValueError("endpoint variant or master seed is outside the frozen four-run matrix")
        key = seed, variant
        if key in reports:
            raise ValueError("duplicate final endpoint actor or seed/variant report")
        if (report.get("schema_version") != EVALUATION_SCHEMA or report.get("status") != "complete"
                or report.get("task_family") != "relay"
                or report.get("cohort_identity") != cohort_identity
                or report.get("protocol_sha256") != protocol_sha256
                or report.get("bank_sha256") != canonical_hash(bank)):
            raise ValueError("endpoint schema, completion, task, cohort, amendment or bank differs")
        if type(report.get("final_local_cycles")) is not int or report["final_local_cycles"] != 20:
            raise ValueError("only the final actor after all 20 local cycles can enter endpoint analysis")
        _sha(report.get("final_actor_identity"), "final actor identity")
        flags = [report.get(key) for key in _NATIVE_FLAGS]
        if any(type(value) is not bool for value in flags) or len(set(flags)) != 1:
            raise ValueError("endpoint report has inconsistent native/fixture execution flags")
        native = flags[0]
        if require_native and native is not True:
            raise ValueError("synthetic fixture output cannot enter scientific endpoint analysis")
        native_kinds.add(native)
        runtime = report.get("runtime")
        if type(runtime) is not dict or runtime.get("action_rng_backend") not in ("cpu", "cuda"):
            raise ValueError("endpoint runtime must record its CPU/CUDA action RNG backend")
        device = runtime.get("device")
        if (type(device) is not str or re.fullmatch(r"cpu|cuda(?::(?:0|[1-9][0-9]*))?", device) is None
                or runtime["action_rng_backend"] != device.split(":")[0]):
            raise ValueError("endpoint action RNG backend differs from its evaluation device")
        if (type(runtime.get("torch")) is not str or not runtime["torch"]
                or runtime.get("action_sampler") != "torch.multinomial_per_player"):
            raise ValueError("endpoint runtime must bind the PyTorch version and registered per-player action sampler")
        rng_backends.add(runtime["action_rng_backend"])
        torch_versions.add(runtime["torch"])
        reports[key] = {"report": report, "episodes": _episode_rows(report.get("episodes"), bank)}
    if set(reports) != set(_CASES) or len(native_kinds) != 1:
        raise ValueError("analysis needs complete paired arms and cannot mix native and fixture evidence")
    if len(rng_backends) != 1:
        raise ValueError("all four final actors must use the same evaluation action RNG backend; CPU/CUDA seeds are not equivalent streams")
    if len(torch_versions) != 1:
        raise ValueError("all four final actors must use the same evaluation PyTorch version to retain paired action RNG semantics")
    actor_ids = [reports[key]["report"]["final_actor_identity"] for key in _CASES]
    if len(set(actor_ids)) != 4:
        raise ValueError("four distinct final actor provenance identities are required")
    arms, layouts, episodes = [], [], []
    for seed, variant in _CASES:
        item, report = reports[seed, variant], reports[seed, variant]["report"]
        rows = item["episodes"]
        arms.append({"master_seed": seed, "variant": variant,
                     "final_actor_identity": report["final_actor_identity"], **_metrics(rows)})
        for template_id in ("relay_test_00", "relay_test_01"):
            selected = [row for row in rows if row["template_id"] == template_id]
            if len(selected) != 10:
                raise ValueError("per-layout endpoint evidence must retain all ten episodes")
            layouts.append({"master_seed": seed, "variant": variant, "template_id": template_id,
                            **_metrics(selected)})
        for row in rows:
            episodes.append({"master_seed": seed, "variant": variant,
                **{key: row[key] for key in ("episode_index", "template_id", "success", "episode_return",
                   "latch_activated", "joint_transitions", "simulator_tics", "termination_cause")}})
    indexed = {(row["master_seed"], row["variant"]): row for row in arms}
    pairs = []
    for seed in (0, 1):
        b, d = indexed[seed, "B"], indexed[seed, "D"]
        pairs.append({"master_seed": seed, "episodes_per_arm": 20,
            "B_successes": b["successes"], "D_successes": d["successes"],
            "B_success_rate": b["success_rate"], "D_success_rate": d["success_rate"],
            "success_count_difference_D_minus_B": d["successes"] - b["successes"],
            "success_rate_difference_D_minus_B": _difference(d["success_rate"], b["success_rate"]),
            "mean_return_difference_D_minus_B": _difference(d["mean_return"], b["mean_return"]),
            "latch_count_difference_D_minus_B": d["latch_episodes"] - b["latch_episodes"],
            "latch_episode_rate_difference_D_minus_B": _difference(d["latch_episode_rate"], b["latch_episode_rate"])})
    summaries = {}
    for key in ("success_rate_difference_D_minus_B", "mean_return_difference_D_minus_B",
                "latch_episode_rate_difference_D_minus_B"):
        values = [row[key] for row in pairs]
        summaries[key] = {"paired_seeds": 2, "values_by_master_seed": {str(s): values[s] for s in (0, 1)},
            "descriptive_mean": _sum(value / 2 for value in values), "observed_min": min(values), "observed_max": max(values)}
    return {"evidence_kind": "native_final_endpoint" if True in native_kinds else "synthetic_fixture",
        "evaluation_device_backend": next(iter(rng_backends)),
        "evaluation_pytorch_version": next(iter(torch_versions)),
        "training_seeds": [0, 1], "paired_seed_count": 2, "runs": 4, "episodes_per_run": 20,
        "total_logical_episodes": 80, "arms": arms, "paired_differences": pairs,
        "descriptive_summary": summaries, "per_layout": layouts, "episode_records": episodes,
        "primary_metric": "final_success_rate_D_minus_B", "replication_unit": "training_master_seed",
        "generalization_scope": "two_authored_unseen_relay_layouts",
        "confidence_intervals": False, "p_values": False, "bootstrap": False, "auc": False,
        "superiority_claim": False, "equivalence_claim": False,
        "interpretation": "descriptive exploratory results from two training seed pairs; latch-only gains are partial progress"}


def _inspect_cohort(path):
    from .cohort import inspect_cohort
    return inspect_cohort(path)


def _load_evaluation(path, *, cohort):
    from .evaluation import validate_evaluation_report
    return validate_evaluation_report(path, cohort=cohort, require_native=True)


def _guard_output(output, cohort, evaluations):
    if output.exists():
        raise FileExistsError("analysis requires a fresh directory; preserve all earlier evidence")
    protected = [ROOT / name for name in ("src", "scripts", "tests", "configs", "scenarios")]
    protected += [path.parent for path in evaluations]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError("analysis output must be separate from sources and evaluation evidence")
    if cohort.is_relative_to(output) or output == cohort.parent:
        raise ValueError("analysis output must not replace or contain the frozen cohort")


def _csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})
    return stream.getvalue()


def analyze_cohort(cohort, evaluations, output_dir):
    """Authenticate four native endpoint reports and publish immutable tables.

    Invalid or incomplete scientific evidence produces ``needs_review`` with no
    aggregate metrics. It never fills absent episodes, actors or pairs with zero.
    This function reads saved results only and does not open a native environment.
    """
    cohort_path, output = safe_path(cohort), safe_path(output_dir)
    if type(evaluations) not in (list, tuple):
        raise ValueError("evaluation evidence must be a list of report paths")
    paths = [safe_path(path) for path in evaluations]
    _guard_output(output, cohort_path, paths)
    before, reasons, metrics, manifest, bindings = source_identity(), [], None, None, []
    try:
        manifest = _inspect_cohort(cohort_path)
        identity = _sha(manifest.get("cohort_identity"), "frozen cohort identity")
        bank = validate_test_bank(manifest.get("bank"))
        if manifest.get("bank_sha256") != canonical_hash(bank):
            raise ValueError("frozen cohort bank identity differs")
        expected = {(row["master_seed"], row["variant"]) for row in manifest["run_matrix"]}
        if len(manifest["run_matrix"]) != 4 or expected != set(_CASES):
            raise ValueError("frozen cohort does not contain the registered four distinct runs")
        if len(paths) != 4 or len(set(paths)) != 4:
            raise ValueError("exactly four distinct final endpoint report paths are required; missing or duplicate inputs")
        cohort_hash = file_sha256(cohort_path)
        reports = []
        for path in paths:
            reports.append(_load_evaluation(path, cohort=manifest))
            bindings.append({"path": str(path), "sha256": file_sha256(path)})
        metrics = aggregate_records(reports, bank=bank, cohort_identity=identity,
            protocol_sha256=manifest["protocol_sha256"], require_native=True)
        # A report's receipt authenticates its linked episode and trace files.
        # Revalidate all links immediately before publication, not only the
        # report JSON, so moving input files cannot enter a mixed-time summary.
        for row, original in zip(bindings, reports):
            reread = _load_evaluation(row["path"], cohort=manifest)
            if row["sha256"] != file_sha256(row["path"]) or canonical_hash(reread) != canonical_hash(original):
                raise ValueError("endpoint evidence changed during paired analysis")
        checked = _inspect_cohort(cohort_path)
        if (file_sha256(cohort_path) != cohort_hash or canonical_hash(checked) != canonical_hash(manifest)
                or source_identity() != before):
            raise ValueError("cohort or implementation sources changed during paired analysis")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        reasons.append(f"{type(error).__name__}: {error}")
        metrics = None
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    if metrics is not None:
        common = ("master_seed", "variant", "episodes", "successes", "success_rate", "return_sum", "mean_return",
            "latch_episodes", "latch_episode_rate", "joint_transitions", "simulator_tics", "mean_length",
            "successful_completion_length_denominator", "mean_successful_completion_length",
            "successful_completion_tics_denominator", "mean_successful_completion_tics")
        tables = {"arms.csv": (metrics["arms"], (*common, "final_actor_identity")),
            "paired_differences.csv": (metrics["paired_differences"], tuple(metrics["paired_differences"][0])),
            "per_layout.csv": (metrics["per_layout"], (*common, "template_id")),
            "episodes.csv": (metrics["episode_records"], tuple(metrics["episode_records"][0]))}
        for name, (rows, fields) in tables.items():
            path = output / name
            path.write_text(_csv_text(rows, fields), encoding="utf-8", newline="")
            files[name] = {"path": name, "sha256": file_sha256(path)}
    source_unchanged = source_identity() == before
    if not source_unchanged and metrics is not None:
        reasons.append("ValueError: implementation sources changed while publishing analysis tables")
        metrics = None
        files = {}
    report = {"schema_version": SCHEMA_VERSION, "created_at": timestamp(),
        "status": "complete" if metrics is not None else "needs_review", "reasons": reasons,
        "cohort": {"path": str(cohort_path), "sha256": file_sha256(cohort_path) if cohort_path.is_file() else None},
        "cohort_identity": manifest.get("cohort_identity") if type(manifest) is dict else None,
        "protocol_sha256": manifest.get("protocol_sha256") if type(manifest) is dict else None,
        "bank_sha256": manifest.get("bank_sha256") if type(manifest) is dict else None,
        "evaluations": bindings, "provided_evaluation_paths": [str(path) for path in paths],
        "source_sha256": before, "source_unchanged": source_unchanged,
        "analysis": metrics, "files": files, "missing_evidence_imputed": False,
        "research_training_executed": False, "research_evaluation_executed": False,
        "native_environment_opened": False, "new_environment_interactions": 0,
        "new_model_updates": 0, "new_policy_optimizer_updates": 0,
        "statistical_superiority_established": False, "cohort_acceptance": False,
        "report_path": str(output / "paired_analysis.json")}
    write_report(output / "paired_analysis.json", report)
    return report
