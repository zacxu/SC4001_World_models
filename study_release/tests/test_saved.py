"""Portable saved-record reproduction from artificial episodes and mock graphics."""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from study_release import evidence, packaging, saved
from study_reporting.figures import build_figure_data
from study_reporting.phase5 import write_tables
from study_reporting.tests.test_figures import fixture_context, fixture_renderer
from tests import test_endpoint_analysis as _episode_factory
from world_models_endpoint.analysis import aggregate_records


class SavedFixture:
    def __init__(self, root):
        self.root, self.refs = root, []
        original = root / "original"
        original.mkdir()
        self.original = original
        source = self.file("src/fixture.py", b"# Synthetic package source\n", include=False)
        reporting = self.file("study_reporting/fixture.py", b"# Synthetic reporting source\n", include=False)
        self.file("study_release/saved.py", b"# Synthetic entry point\n", include=False)
        self.file("study_release/requirements.txt", b"PyYAML==6.0.3\nmatplotlib==3.11.2\n", include=False)
        factory = _episode_factory.EndpointAnalysisTests()
        factory.setUp()
        self.reports = deepcopy(factory.reports)
        for report in self.reports:
            report.update(native_backend=True, native_engine=True,
                native_environment_evaluation_executed=True, research_evaluation_executed=True, attempts=[])
        self.metrics = aggregate_records(self.reports, bank=factory.bank, cohort_identity=factory.cohort_identity,
            protocol_sha256=factory.protocol, require_native=True)
        self.context = {"cohort": {"cohort_identity": factory.cohort_identity, "protocol_sha256": factory.protocol,
            "source_sha256": {"src/fixture.py": source["sha256"]}, "inputs": {"bound_files": []}},
            "source_identity": {"study_reporting/fixture.py": reporting["sha256"]},
            "references": self.refs, "runs": [], "evaluation_paths": [], "evaluation_reports": self.reports}
        for report in self.reports:
            v, s = report["variant"], report["master_seed"]
            prefix = f"artifacts/runs/{v}/seed_{s}"
            actor = self.original / prefix / "actor_export"
            actor_refs = [self.report(prefix + "/actor_export/manifest.json", {"synthetic": True}),
                self.file(prefix + "/actor_export/state.pt", b"synthetic actor bytes")]
            actor_refs.append(evidence.reference(actor / "manifest.sha256", receipt=False))
            self.context["runs"].append({"variant": v, "master_seed": s, "status": "complete", "training_complete": True,
                "run_dir": str(self.original / prefix), "actor_export": str(actor), "files": actor_refs})
            prefix = f"artifacts/evaluation/{v}/seed_{s}"
            path = self.report(prefix + "/evaluation.json", report)["path"]
            self.report(prefix + "/evaluation_manifest.json", {"synthetic": True})
            self.report(prefix + "/bank.json", factory.bank)
            self.context["evaluation_paths"].append(path)
        analysis_ref = self.report("artifacts/completion/paired_analysis.json", {"status": "complete",
            "cohort_identity": factory.cohort_identity, "analysis": self.metrics})
        self.context["analysis"] = {"status": "complete", "cohort_identity": factory.cohort_identity,
            "analysis": self.metrics, "report_path": analysis_ref["path"]}
        for name, data in saved.table_bytes(self.metrics).items():
            self.file("artifacts/completion/" + name, data)
        _, resources = fixture_context()
        resources["cohort_identity"] = factory.cohort_identity
        resources["status"] = "complete"
        for row in resources["per_run"]:
            row["case"] = f"{row['variant']}/seed_{row['master_seed']}"
            row["logical_counters"].update(local_PPO_cycles=20, teacher_PPO_cycles=25 if row["variant"] == "D" else 0,
                new_environment_interactions=0, new_model_updates=0)
        secondary = {"training_diagnostics": []}
        secondary_ref = self.report("artifacts/completion/secondary.json", secondary)
        resource_ref = self.report("artifacts/completion/resources.json", resources)
        tables = write_tables(self.original / "artifacts/completion", secondary, resources)
        self.refs += tables
        self.context["deliverables"] = {"resources": resource_ref, "secondary": secondary_ref, "tables": tables}
        plot_data = build_figure_data(self.context, resources)
        data_refs = [self.report("artifacts/completion/" + name, data) for name, data in zip(
            ("endpoint_results_data.json", "paired_effects_compute_data.json"), plot_data)]
        self.context["figure_evidence"] = {"data": data_refs}
        self.diagnosis = {"status": "complete", "cohort_identity": factory.cohort_identity}
        dpath = self.report("artifacts/diagnosis/diagnosis.json", self.diagnosis)
        self.context["diagnosis_path"] = dpath["path"]
        self.context["diagnosis_artifacts"] = [dpath, self.file("artifacts/diagnosis/diagnosis.md", b"Synthetic only\n"),
            self.file("artifacts/diagnosis/selected_traces.csv", b"seed,variant\n")]
        self.release = root / "sealed"
        self.manifest = packaging.build_release(self.context, self.diagnosis, self.release, root=self.original)

    def file(self, name, data, *, include=True):
        path = self.original / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        row = evidence.reference(path, receipt=False)
        if include:
            self.refs.append(row)
        return row

    def report(self, name, data):
        path = self.original / name
        evidence.write_report(path, data)
        row = evidence.reference(path)
        self.refs.append(row)
        return row


class SavedReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.fixture = SavedFixture(self.root)
        self.output = self.root / "regenerated"
        self.enterContext(patch.object(saved, "_check_origins", return_value=None))
        self.enterContext(patch("study_reporting.figures._matplotlib_renderer", side_effect=fixture_renderer))

    def test_six_byte_identical_tables_and_mocked_figure_plumbing(self):
        result = saved.regenerate(self.fixture.release, self.output)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["six_tables_byte_identical"])
        self.assertTrue(result["numerical_reproduction"])
        self.assertFalse(result["research_evaluation_executed"])
        self.assertFalse(result["model_or_actor_weights_loaded"])
        self.assertFalse(result["phase6_acceptance"])
        self.assertEqual(len(result["tables"]), 6)
        self.assertEqual(len(result["figures"]), 4)
        packaging.inspect_release(self.fixture.release)

    def test_moving_release_needs_no_original_server_paths(self):
        import shutil
        moved = self.root / "moved"
        self.fixture.release.rename(moved)
        shutil.rmtree(self.fixture.original)
        self.assertEqual(saved.regenerate(moved, self.output)["status"], "complete")

    def test_inside_release_and_existing_outputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the sealed"):
            saved.regenerate(self.fixture.release, self.fixture.release / "new")
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            saved.regenerate(self.fixture.release, self.output)

    def test_tampered_archive_is_rejected_before_aggregation(self):
        data = self.fixture.release / "reproduction_inputs.json"
        data.write_text("{}")
        with self.assertRaisesRegex(ValueError, "inventory"):
            saved.regenerate(self.fixture.release, self.output)
        self.assertFalse(self.output.exists())

    def test_structural_reader_rejects_native_report_mismatch(self):
        original_components = saved._components()
        def changed(*args, **kwargs):
            rows = deepcopy(args[0])
            rows[0]["episodes"].pop()
            return aggregate_records(rows, **kwargs)
        with patch.object(saved, "_components", return_value=(changed, *original_components[1:])):
            with self.assertRaises(ValueError):
                saved.regenerate(self.fixture.release, self.output)

    def test_changed_plot_numbers_cannot_be_rendered(self):
        original_components = saved._components()
        def changed(*args):
            first, second = original_components[1](*args)
            first["per_seed"][0]["mean_return"] += 1
            return first, second
        with patch.object(saved, "_components", return_value=(original_components[0], changed, *original_components[2:])):
            with self.assertRaisesRegex(ValueError, "figure numbers"):
                saved.regenerate(self.fixture.release, self.output)
        self.assertFalse(self.output.exists())

    def test_fixture_renderer_publication_cannot_certify_real_graphics(self):
        original_components = saved._components()
        with patch.object(saved, "_components", return_value=(*original_components[:2],
                lambda *args: {"status": "complete", "fixtures_only": True, "actual_graphics_rendered": False}, original_components[3])):
            with self.assertRaisesRegex(ValueError, "actual figure rendering"):
                saved.regenerate(self.fixture.release, self.output)

    def test_confined_input_reference_rejects_parent_traversal(self):
        with self.assertRaisesRegex(ValueError, "escaped"):
            saved._input(self.fixture.release, self.fixture.manifest, {"path": "../outside", "sha256": "x"})

    def test_user_directory_arguments_allow_normal_relative_parent_paths(self):
        result = saved._user_path(self.fixture.release / "source" / "..")
        self.assertEqual(result, self.fixture.release)


class SavedImportAndCliTests(unittest.TestCase):
    def test_fresh_pure_components_import_neither_torch_nor_simulator(self):
        code = "from study_release.saved import _components; _components(); import sys; assert 'torch' not in sys.modules; assert 'vizdoom' not in sys.modules; print('pure')"
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=evidence.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "pure")

    def test_component_origins_must_belong_to_the_sealed_source_tree(self):
        with self.assertRaisesRegex(ValueError, "fresh Python process"):
            saved._check_origins(saved._components(), evidence.ROOT / "other_source")
        saved._check_origins(saved._components(), evidence.ROOT)

    def test_cli_dispatch_error_and_explicit_required_arguments(self):
        with patch.object(saved, "regenerate", return_value={"status": "complete"}) as run:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(saved.main(["regenerate", "--release-dir", "r", "--output-dir", "o"]), 0)
            run.assert_called_once_with("r", "o")
        with patch.object(saved, "inspect_release", side_effect=ValueError("tampered")):
            with redirect_stderr(io.StringIO()):
                self.assertEqual(saved.main(["inspect", "--release-dir", "."]), 1)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            saved.main(["regenerate", "--release-dir", "r"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
