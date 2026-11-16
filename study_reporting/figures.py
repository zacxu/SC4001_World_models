"""Two reproducible endpoint figures from saved, authenticated study records.

The Matplotlib renderer is imported lazily and uses the noninteractive Agg
backend. Fixture renderers test data and publication contracts without making
claims about a graphics installation or rerunning any scientific experiment.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_study.evidence import safe_path, write_report


MATPLOTLIB_VERSION = "3.11.2"
SCHEMA = "p5-endpoint-figures-v1"
_CASES = tuple((s, v) for s in (0, 1) for v in ("B", "D"))
_FILES = ("endpoint_results.svg", "endpoint_results.png", "paired_effects_compute.svg", "paired_effects_compute.png")
_COUNT_FIELDS = ("local_actor_steps", "local_critic_steps", "teacher_actor_steps", "teacher_critic_steps",
                 "distillation_optimizer_steps")


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"figure {name} must be a finite recorded value")
    return value


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"figure {name} must be a nonnegative recorded integer")
    return value


def _index(rows, name):
    if type(rows) is not list or len(rows) != 4:
        raise ValueError(f"figure {name} requires all four registered runs")
    result = {}
    for row in rows:
        key = row.get("master_seed"), row.get("variant")
        if type(key[0]) is not int or type(key[1]) is not str or key not in _CASES or key in result:
            raise ValueError(f"figure {name} has a missing, extra or duplicate seed/arm")
        result[key] = row
    return result


def build_figure_data(context, resources):
    """Extract exact plotted numbers; null measurements remain explicitly null."""
    analysis, cohort = context["analysis"], context["cohort"]
    if (analysis.get("status") != "complete" or analysis.get("cohort_identity") != cohort["cohort_identity"]
            or resources.get("cohort_identity") != cohort["cohort_identity"]
            or resources.get("status") not in ("complete", "complete_with_explicit_measurement_gaps")):
        raise ValueError("figures require complete analysis and matching completed resource evidence")
    metrics = analysis.get("analysis", {})
    if (metrics.get("evidence_kind") != "native_final_endpoint" or metrics.get("total_logical_episodes") != 80
            or metrics.get("paired_seed_count") != 2):
        raise ValueError("figures require the native final four-run endpoint, not development or synthetic outcomes")
    arms = _index(metrics.get("arms"), "endpoint data")
    costs = _index(resources.get("per_run"), "compute data")
    raw, compute = [], []
    for seed, variant in _CASES:
        row, cost = arms[seed, variant], costs[seed, variant]
        successes = _count(row.get("successes"), "successes")
        latches = _count(row.get("latch_episodes"), "latch episodes")
        if (row.get("episodes") != 20 or successes > 20 or latches > 20
                or row.get("success_rate") != successes / 20 or row.get("latch_episode_rate") != latches / 20):
            raise ValueError("figure success/latch counts and rates do not reconcile with 20 episodes")
        raw.append({"master_seed": seed, "variant": variant, "episodes": 20, "successes": successes,
            "success_rate": successes / 20, "mean_return": _finite(row.get("mean_return"), "mean return"),
            "latch_episodes": latches, "latch_episode_rate": latches / 20})
        counters = cost.get("logical_counters", {})
        counts = {name: _count(counters.get(name), name) for name in _COUNT_FIELDS}
        expected = dict(zip(_COUNT_FIELDS, (320, 320, 400 if variant == "D" else 0,
                                            400 if variant == "D" else 0, 100 if variant == "D" else 0)))
        if counts != expected:
            raise ValueError("figure optimizer counts differ from the completed registered schedule")
        imagined = {name: _count(counters.get(name), name) for name in
            ("imagined_joint_transitions_local", "imagined_joint_transitions_teacher",
             "imagined_joint_transitions_buffer_generation", "imagined_joint_transitions_total")}
        if sum(value for name, value in imagined.items() if name != "imagined_joint_transitions_total") != imagined["imagined_joint_transitions_total"]:
            raise ValueError("figure imagined volumes overlap or fail to reconcile")
        seconds = cost.get("recorded_scientific_unit_elapsed_seconds")
        if seconds is not None:
            _finite(seconds, "recorded elapsed seconds")
            if seconds < 0:
                raise ValueError("figure recorded elapsed time cannot be negative")
        known = cost.get("complete_scientific_elapsed_cost_known")
        if type(known) is not bool:
            raise ValueError("figure elapsed measurement completeness must be explicit")
        if seconds == 0 and not known:
            seconds = None  # No recorded duration is not a measured zero-second experiment.
        compute.append({"master_seed": seed, "variant": variant, "optimizer_steps": counts,
            "imagined_joint_transitions": imagined, "recorded_unit_elapsed_seconds": seconds,
            "complete_elapsed_cost_known": known, "GPU_occupancy_seconds": None})
    indexed = {(row["master_seed"], row["variant"]): row for row in raw}
    paired = []
    for seed in (0, 1):
        b, d = indexed[seed, "B"], indexed[seed, "D"]
        paired.append({"master_seed": seed,
            "success_rate_difference_D_minus_B": d["success_rate"] - b["success_rate"],
            "mean_return_difference_D_minus_B": d["mean_return"] - b["mean_return"],
            "latch_episode_rate_difference_D_minus_B": d["latch_episode_rate"] - b["latch_episode_rate"]})
    original = metrics.get("paired_differences")
    if type(original) is not list or len(original) != 2 or {r.get("master_seed") for r in original} != {0, 1}:
        raise ValueError("figure paired differences are missing or duplicated")
    for row in original:
        expected = paired[row["master_seed"]]
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("figure paired differences disagree with raw per-seed endpoint data")
    common = {"cohort_identity": cohort["cohort_identity"], "protocol_sha256": cohort["protocol_sha256"],
        "training_seeds": [0, 1], "replication_unit": "training master seed", "confidence_intervals": False,
        "p_values": False, "bootstrap": False, "superiority_claim": False,
        "missing_values_imputed": False, "online_learning_curves": False}
    endpoint = {"schema_version": "p5-endpoint-figure-data-v1", **common,
        "figure": "final_success_return_latch", "per_seed": raw,
        "scope": "20 fixed held-out Relay episodes per final actor; two authored test layouts"}
    paired_compute = {"schema_version": "p5-paired-compute-figure-data-v1", **common,
        "figure": "paired_effects_and_actual_compute", "paired_differences": paired, "compute_by_run": compute,
        "optimizer_step_totals": {name: sum(row["optimizer_steps"][name] for row in compute) for name in _COUNT_FIELDS},
        "optimizer_steps_are_FLOPs": False, "GPU_occupancy_measured": False,
        "elapsed_scope": "recorded scientific-unit host wall seconds; lower bounds when interrupted work is unknown; not GPU occupancy or campaign wall time",
        "compute_matched": False}
    return endpoint, paired_compute


def _validate_graphic(path):
    raw = path.read_bytes()
    if path.suffix == ".svg":
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as error:
            raise ValueError("renderer did not publish valid SVG") from error
        if root.tag != "{http://www.w3.org/2000/svg}svg":
            raise ValueError("renderer output is not an SVG figure")
    elif (len(raw) < 45 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR"
          or not all(struct.unpack(">II", raw[16:24])) or raw[-8:-4] != b"IEND"):
        raise ValueError("renderer did not publish a valid standalone PNG")


def _save(fig, directory, stem):
    fig.savefig(directory / f"{stem}.svg", format="svg", metadata={"Date": None, "Creator": "WM1 study_reporting"})
    fig.savefig(directory / f"{stem}.png", format="png", dpi=160, metadata={"Software": "WM1 study_reporting"})


def _matplotlib_renderer(endpoint, paired_compute, output):
    try:
        import matplotlib
    except ImportError as error:
        raise ValueError("install study_reporting/requirements.txt in the server environment before rendering") from error
    if matplotlib.__version__ != MATPLOTLIB_VERSION:
        raise ValueError(f"endpoint rendering requires matplotlib=={MATPLOTLIB_VERSION}; observed {matplotlib.__version__}")
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    settings = {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                "axes.spines.right": False, "svg.hashsalt": canonical_hash([endpoint, paired_compute])}
    with matplotlib.rc_context(settings):
        colors, shapes = {"B": "#3465a4", "D": "#c86b25"}, {0: "o", 1: "s"}
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
        fields = (("success_rate", "Final team success rate", "successes"),
                  ("mean_return", "Mean team return", None),
                  ("latch_episode_rate", "Latch activation rate", "latch_episodes"))
        for ax, (field, title, count_field) in zip(axes, fields):
            for row in endpoint["per_seed"]:
                x = (0 if row["variant"] == "B" else 1) + (-0.07 if row["master_seed"] == 0 else 0.07)
                y = row[field]
                ax.scatter([x], [y], color=colors[row["variant"]], marker=shapes[row["master_seed"]],
                    s=65, zorder=3, label=f"{row['variant']} seed {row['master_seed']}")
                label = f"s{row['master_seed']}: {row[count_field]}/20" if count_field else f"s{row['master_seed']}: {y:.4g}"
                ax.annotate(label, (x, y), xytext=(-30 if row["master_seed"] == 0 else 4, 10 if row["master_seed"] == 0 else -17),
                            textcoords="offset points", fontsize=9)
            ax.axhline(0, color="#999999", linewidth=0.7, zorder=0)
            ax.set(xticks=[0, 1], xticklabels=["B", "D"], xlim=(-0.45, 1.5), title=title)
            if count_field:
                ax.set_ylim(-0.12, 1.08)
                ax.set_ylabel("Episode fraction")
        fig.suptitle("Final held-out Relay endpoint: both training seeds shown; no confidence intervals")
        _save(fig, output, "endpoint_results")
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
        fields = (("success_rate_difference_D_minus_B", "Paired success-rate difference"),
                  ("mean_return_difference_D_minus_B", "Paired mean-return difference"),
                  ("latch_episode_rate_difference_D_minus_B", "Paired latch-rate difference"))
        for ax, (field, title) in zip(axes[0], fields):
            ax.axhline(0, color="#777777", linewidth=0.8)
            for row in paired_compute["paired_differences"]:
                ax.scatter([row["master_seed"]], [row[field]], color="#73559a", marker=shapes[row["master_seed"]], s=70)
                ax.annotate(f"{row[field]:.4g}", (row["master_seed"], row[field]), xytext=(0, 10), textcoords="offset points")
            ax.set(xticks=[0, 1], xticklabels=["Seed 0", "Seed 1"], xlim=(-0.4, 1.4), title=title, ylabel="D minus B")
        rows = paired_compute["compute_by_run"]
        labels = [f"{r['variant']} s{r['master_seed']}" for r in rows]
        ax = axes[1, 0]
        bottoms = [0] * 4
        for field, label in zip(_COUNT_FIELDS, ("Local actor", "Local critic", "Teacher actor", "Teacher critic", "Distillation")):
            values = [row["optimizer_steps"][field] for row in rows]
            ax.bar(range(4), values, bottom=bottoms, label=label)
            bottoms = [a + b for a, b in zip(bottoms, values)]
        ax.set(title="Actual optimizer updates per run", ylabel="Steps (different networks; not FLOPs)", xticks=range(4), xticklabels=labels)
        ax.legend(fontsize=8)
        ax = axes[1, 1]
        bottoms = [0] * 4
        for field, label in (("imagined_joint_transitions_local", "Local"),
                             ("imagined_joint_transitions_teacher", "Teacher"),
                             ("imagined_joint_transitions_buffer_generation", "Buffer generation")):
            values = [row["imagined_joint_transitions"][field] for row in rows]
            ax.bar(range(4), values, bottom=bottoms, label=label)
            bottoms = [a + b for a, b in zip(bottoms, values)]
        ax.set(title="Actual predicted joint steps", ylabel="Joint steps (prefix reconstruction excluded)", xticks=range(4), xticklabels=labels)
        ax.legend(fontsize=8)
        ax = axes[1, 2]
        for index, row in enumerate(rows):
            seconds = row["recorded_unit_elapsed_seconds"]
            if seconds is None:
                ax.text(index, 0.02, "Unmeasured", transform=ax.get_xaxis_transform(), rotation=90, ha="center", va="bottom")
            else:
                ax.bar([index], [seconds / 3600], color=colors[row["variant"]],
                       hatch=None if row["complete_elapsed_cost_known"] else "//")
                suffix = "" if row["complete_elapsed_cost_known"] else " (partial)"
                ax.annotate(f"{seconds / 3600:.3g} h{suffix}", (index, seconds / 3600), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        ax.set(title="Recorded unit elapsed time", ylabel="Host wall hours; GPU occupancy unknown", xticks=range(4), xticklabels=labels)
        fig.suptitle("Paired endpoint effects and D's additional compute: descriptive two-seed study")
        _save(fig, output, "paired_effects_compute")
        plt.close(fig)
    return {name: output / name for name in _FILES}


def write_figures(context, resources, output_dir, *, renderer=None):
    """Publish deterministic data, two standalone figures and their SHA receipts."""
    output = safe_path(output_dir)
    if output.exists():
        raise FileExistsError("figures require a fresh output directory; preserve prior versions")
    endpoint, paired_compute = build_figure_data(context, resources)
    output.mkdir(parents=True)
    data_refs = []
    for name, data in (("endpoint_results_data.json", endpoint), ("paired_effects_compute_data.json", paired_compute)):
        path = output / name
        write_report(path, data)
        data_refs.append({"path": str(path), "sha256": file_sha256(path)})
    rendered = (renderer or _matplotlib_renderer)(deepcopy(endpoint), deepcopy(paired_compute), output)
    if type(rendered) is not dict or set(rendered) != set(_FILES):
        raise ValueError("renderer must publish exactly two SVG and two PNG endpoint figures")
    files = []
    for name in _FILES:
        path = safe_path(rendered[name])
        if path != output / name or not path.is_file():
            raise ValueError("renderer file escaped the figure directory or differs from its required name")
        _validate_graphic(path)
        digest = file_sha256(path)
        path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {name}\n", encoding="ascii")
        files.append({"path": str(path), "sha256": digest})
    report = {"schema_version": SCHEMA, "status": "complete", "cohort_identity": context["cohort"]["cohort_identity"],
        "protocol_sha256": context["cohort"]["protocol_sha256"], "figure_count": 2, "files": files, "data": data_refs,
        "matplotlib_version": MATPLOTLIB_VERSION if renderer is None else None,
        "backend": "Agg" if renderer is None else "fixture_renderer",
        "fixtures_only": renderer is not None, "actual_graphics_rendered": renderer is None,
        "two_training_seed_values_retained": True, "confidence_intervals": False, "p_values": False,
        "unmeasured_costs_assumed_zero": False, "research_training_executed": False,
        "research_evaluation_executed": False, "native_environment_opened": False,
        "source_sha256": deepcopy(context.get("source_identity", {})),
        "references": deepcopy(context.get("references", []))}
    write_report(output / "figures.json", report)
    return report
