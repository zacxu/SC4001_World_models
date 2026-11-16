"""Figure-data and mock rendering contracts; real graphics run on the server."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

from study_reporting.figures import build_figure_data, write_figures, _matplotlib_renderer
from world_models_study.evidence import read_report


def _png():
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b""))


def fixture_renderer(endpoint, paired, output):
    result = {}
    for stem in ("endpoint_results", "paired_effects_compute"):
        for suffix in ("svg", "png"):
            path = output / f"{stem}.{suffix}"
            path.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>' if suffix == "svg" else _png())
            result[path.name] = path
    return result


def fixture_context():
    arms, pairs, costs = [], [], []
    for seed in (0, 1):
        for variant in ("B", "D"):
            arms.append({"master_seed": seed, "variant": variant, "episodes": 20,
                "successes": 0, "success_rate": 0, "mean_return": -0.4,
                "latch_episodes": 0, "latch_episode_rate": 0})
            costs.append({"master_seed": seed, "variant": variant,
                "logical_counters": {"local_actor_steps": 320, "local_critic_steps": 320,
                    "teacher_actor_steps": 400 if variant == "D" else 0,
                    "teacher_critic_steps": 400 if variant == "D" else 0,
                    "distillation_optimizer_steps": 100 if variant == "D" else 0,
                    "imagined_joint_transitions_local": 75000,
                    "imagined_joint_transitions_teacher": 90000 if variant == "D" else 0,
                    "imagined_joint_transitions_buffer_generation": 1800 if variant == "D" else 0,
                    "imagined_joint_transitions_total": 166800 if variant == "D" else 75000},
                "recorded_scientific_unit_elapsed_seconds": None,
                "complete_scientific_elapsed_cost_known": False})
        pairs.append({"master_seed": seed, "success_rate_difference_D_minus_B": 0,
            "mean_return_difference_D_minus_B": 0, "latch_episode_rate_difference_D_minus_B": 0})
    context = {"cohort": {"cohort_identity": "a" * 64, "protocol_sha256": "b" * 64},
        "analysis": {"status": "complete", "cohort_identity": "a" * 64, "analysis": {
            "evidence_kind": "native_final_endpoint", "total_logical_episodes": 80,
            "paired_seed_count": 2, "arms": arms, "paired_differences": pairs}}}
    resources = {"cohort_identity": "a" * 64, "status": "complete_with_explicit_measurement_gaps", "per_run": costs}
    return context, resources


class FakeAxis:
    def __init__(self):
        self.points, self.bars, self.texts = [], [], []

    def scatter(self, xs, ys, **kwargs):
        self.points.append((xs, ys, kwargs))

    def bar(self, xs, ys, **kwargs):
        self.bars.append((list(xs), list(ys), kwargs))

    def text(self, *args, **kwargs):
        self.texts.append(args)

    def get_xaxis_transform(self):
        return None

    def annotate(self, *args, **kwargs): pass
    def axhline(self, *args, **kwargs): pass
    def set(self, *args, **kwargs): pass
    def set_ylim(self, *args, **kwargs): pass
    def set_ylabel(self, *args, **kwargs): pass
    def legend(self, *args, **kwargs): pass


class FakeFigure:
    def suptitle(self, *args, **kwargs): pass

    def savefig(self, path, **kwargs):
        path.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"/>' if path.suffix == ".svg" else _png())


class FakePyplot:
    def __init__(self):
        self.grids = []

    def subplots(self, nrows, ncols, **kwargs):
        grid = [[FakeAxis() for _ in range(ncols)] for _ in range(nrows)]
        self.grids.append(grid)
        class Grid:
            def __getitem__(self, key):
                return grid[key[0]][key[1]] if isinstance(key, tuple) else grid[key]
        return FakeFigure(), grid[0] if nrows == 1 else Grid()

    def close(self, *args): pass


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.context, self.resources = fixture_context()

    def path(self, name="figures"):
        return Path(self.temporary.name) / name

    def test_zero_outcomes_all_seed_values_and_exact_optimizer_units_retained(self):
        endpoint, paired = build_figure_data(self.context, self.resources)
        self.assertEqual(len(endpoint["per_seed"]), 4)
        self.assertEqual(len(paired["paired_differences"]), 2)
        self.assertEqual(paired["optimizer_step_totals"]["local_actor_steps"], 1280)
        self.assertEqual(paired["optimizer_step_totals"]["local_critic_steps"], 1280)
        self.assertEqual(paired["optimizer_step_totals"]["teacher_actor_steps"], 800)
        self.assertEqual(paired["optimizer_step_totals"]["distillation_optimizer_steps"], 200)
        self.assertTrue(all(row["recorded_unit_elapsed_seconds"] is None for row in paired["compute_by_run"]))
        self.assertFalse(paired["GPU_occupancy_measured"])

    def test_descriptive_pair_difference_is_d_minus_b_including_negative_values(self):
        arms = self.context["analysis"]["analysis"]["arms"]
        arms[1].update(successes=2, success_rate=0.1, mean_return=-0.6, latch_episodes=3, latch_episode_rate=0.15)
        self.context["analysis"]["analysis"]["paired_differences"][0].update(
            success_rate_difference_D_minus_B=0.1, mean_return_difference_D_minus_B=-0.19999999999999996,
            latch_episode_rate_difference_D_minus_B=0.15)
        _, data = build_figure_data(self.context, self.resources)
        self.assertAlmostEqual(data["paired_differences"][0]["mean_return_difference_D_minus_B"], -0.2)
        self.assertFalse(data["confidence_intervals"])

    def test_partial_missing_elapsed_is_not_imputed_as_zero(self):
        self.resources["per_run"][0]["recorded_scientific_unit_elapsed_seconds"] = 0
        self.resources["per_run"][1]["recorded_scientific_unit_elapsed_seconds"] = 123.5
        _, data = build_figure_data(self.context, self.resources)
        self.assertIsNone(data["compute_by_run"][0]["recorded_unit_elapsed_seconds"])
        self.assertEqual(data["compute_by_run"][1]["recorded_unit_elapsed_seconds"], 123.5)
        self.assertFalse(data["compute_by_run"][1]["complete_elapsed_cost_known"])

    def test_fixture_publication_has_real_format_contract_and_explicit_fixture_scope(self):
        report = write_figures(self.context, self.resources, self.path(), renderer=fixture_renderer)
        self.assertTrue(report["fixtures_only"])
        self.assertFalse(report["actual_graphics_rendered"])
        self.assertIsNone(report["matplotlib_version"])
        self.assertEqual(len(report["files"]), 4)
        self.assertEqual(len(report["data"]), 2)
        self.assertEqual(read_report(self.path() / "figures.json"), report)
        self.assertEqual(read_report(Path(report["data"][0]["path"]))["per_seed"][0]["successes"], 0)

    def test_data_bytes_reproduce_independently_of_output_path(self):
        a = write_figures(self.context, self.resources, self.path("a"), renderer=fixture_renderer)
        b = write_figures(self.context, self.resources, self.path("b"), renderer=fixture_renderer)
        self.assertEqual([r["sha256"] for r in a["data"]], [r["sha256"] for r in b["data"]])

    def test_headless_plot_contract_keeps_zero_points_and_unknown_time_labels(self):
        endpoint, paired = build_figure_data(self.context, self.resources)
        pyplot, calls = FakePyplot(), []
        mpl = ModuleType("matplotlib")
        mpl.__version__ = "3.11.2"
        mpl.use = lambda *args, **kwargs: calls.append((args, kwargs))
        mpl.rc_context = lambda *args, **kwargs: nullcontext()
        mpl.pyplot = pyplot
        output = self.path(); output.mkdir()
        with patch.dict("sys.modules", {"matplotlib": mpl, "matplotlib.pyplot": pyplot}):
            result = _matplotlib_renderer(endpoint, paired, output)
        self.assertEqual(calls, [(("Agg",), {"force": True})])
        self.assertEqual(len(result), 4)
        self.assertTrue(all(len(axis.points) == 4 for axis in pyplot.grids[0][0]))
        self.assertTrue(all(len(axis.points) == 2 for axis in pyplot.grids[1][0]))
        self.assertEqual(len(pyplot.grids[1][1][2].bars), 0)
        self.assertEqual(len(pyplot.grids[1][1][2].texts), 4)

    def test_unpinned_matplotlib_rejected_before_any_graphics(self):
        mpl = ModuleType("matplotlib"); mpl.__version__ = "0.0.0"
        with patch.dict("sys.modules", {"matplotlib": mpl}):
            with self.assertRaisesRegex(ValueError, "3.11.2"):
                _matplotlib_renderer(*build_figure_data(self.context, self.resources), self.path())

    def test_invalid_graphics_bytes_rejected(self):
        def bad(endpoint, paired, output):
            rows = fixture_renderer(endpoint, paired, output)
            rows["endpoint_results.png"].write_bytes(b"not a PNG")
            return rows
        with self.assertRaisesRegex(ValueError, "PNG"):
            write_figures(self.context, self.resources, self.path(), renderer=bad)

    def test_renderer_cannot_escape_output_directory(self):
        def bad(endpoint, paired, output):
            rows = fixture_renderer(endpoint, paired, output)
            rows["endpoint_results.svg"] = Path(self.temporary.name) / "elsewhere.svg"
            return rows
        with self.assertRaisesRegex(ValueError, "escaped"):
            write_figures(self.context, self.resources, self.path(), renderer=bad)

    def test_duplicate_or_missing_arm_fails_instead_of_zero_filling(self):
        arms = self.context["analysis"]["analysis"]["arms"]
        arms[1] = deepcopy(arms[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_figure_data(self.context, self.resources)

    def test_nonfinite_return_rejected(self):
        self.context["analysis"]["analysis"]["arms"][0]["mean_return"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            build_figure_data(self.context, self.resources)

    def test_success_count_rate_disagreement_rejected(self):
        self.context["analysis"]["analysis"]["arms"][0]["success_rate"] = 0.5
        with self.assertRaisesRegex(ValueError, "reconcile"):
            build_figure_data(self.context, self.resources)

    def test_mismatched_paired_effect_rejected(self):
        self.context["analysis"]["analysis"]["paired_differences"][0]["success_rate_difference_D_minus_B"] = 0.1
        with self.assertRaisesRegex(ValueError, "raw per-seed"):
            build_figure_data(self.context, self.resources)

    def test_compute_does_not_match_registered_schedule_rejected(self):
        self.resources["per_run"][1]["logical_counters"]["teacher_actor_steps"] = 0
        with self.assertRaisesRegex(ValueError, "registered schedule"):
            build_figure_data(self.context, self.resources)

    def test_reusing_output_never_overwrites_prior_evidence(self):
        write_figures(self.context, self.resources, self.path(), renderer=fixture_renderer)
        with self.assertRaises(FileExistsError):
            write_figures(self.context, self.resources, self.path(), renderer=fixture_renderer)


if __name__ == "__main__":
    unittest.main()
