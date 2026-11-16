"""Synthetic package fixtures only; no saved research results are processed."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from study_release import packaging as p


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, root):
        self.root = root
        self.refs = []
        for name in ("src/example.py", "study_reporting/example.py", "study_release/saved.py"):
            self.file(name, b"# synthetic fixture source\n", referenced=False)
        self.file("study_release/requirements.txt", b"PyYAML==6.0.3\nmatplotlib==3.11.2\n", referenced=False)
        initial = self.file("artifacts/initial/model.pt", b"SYNTHETIC LARGE MODEL")
        self.analysis = self.report("artifacts/completion/endpoint_analysis/paired_analysis.json", {"status": "complete"})
        for name in ("arms.csv", "paired_differences.csv", "per_layout.csv", "episodes.csv"):
            self.file("artifacts/completion/endpoint_analysis/" + name, b"fixture,value\nzero,0\n")
        resources = self.report("artifacts/completion/resource_summary.json", {"status": "complete"})
        secondary = self.report("artifacts/completion/secondary_evidence.json", {"status": "complete"})
        tables = [self.file("artifacts/completion/" + name, b"case,value\nB,0\n") for name in
                  ("resources.csv", "training_diagnostics.csv")]
        figures = [self.report("artifacts/completion/figures/" + name, {"synthetic": True}) for name in
                   ("endpoint_results_data.json", "paired_effects_compute_data.json")]
        self.diagnosis = {"status": "complete", "cohort_identity": "fixture-cohort"}
        diagnosis = self.report("artifacts/phase6/diagnosis/diagnosis.json", self.diagnosis)
        diagnosis_artifacts = [diagnosis,
            self.file("artifacts/phase6/diagnosis/diagnosis.md", b"Synthetic diagnosis\n"),
            self.file("artifacts/phase6/diagnosis/selected_traces.csv", b"case,episode\nB,0\n")]
        self.runs, paths, reports = [], [], []
        for seed in (0, 1):
            for variant in ("B", "D"):
                base = f"artifacts/cohort/runs/{variant}/seed_{seed}"
                actor = root / base / "actor_export"
                actor_rows = [self.report(base + "/actor_export/manifest.json", {"actor_only": True}),
                    self.file(base + "/actor_export/state.pt", b"SYNTHETIC ACTOR")]
                actor_rows.append({"path": str(actor / "manifest.sha256"), "sha256": sha(actor / "manifest.sha256")})
                self.runs.append({"variant": variant, "master_seed": seed, "run_dir": str(root / base),
                    "status": "complete", "training_complete": True, "actor_export": str(actor), "files": actor_rows})
                evaluation = f"artifacts/cohort/evaluation/{variant}/seed_{seed}"
                self.report(evaluation + "/bank.json", {"bank_seed": 5600, "episodes": []})
                self.report(evaluation + "/evaluation_manifest.json", {"native_backend": True})
                journal = self.file(evaluation + "/episodes/000000/attempt_00/events.jsonl", b'{"fixture":true}\n')
                attempt = self.report(evaluation + "/episodes/000000/attempt_00/attempt.json", {
                    "journal": {"path": "events.jsonl", "sha256": journal["sha256"]}, "status": "complete"})
                report = {"status": "complete", "native_backend": True, "native_engine": True,
                    "attempts": [{"path": "episodes/000000/attempt_00/attempt.json", "sha256": attempt["sha256"]}]}
                reference = self.report(evaluation + "/evaluation.json", report)
                paths.append(reference["path"]); reports.append(report)
        self.context = {"cohort": {"cohort_identity": "fixture-cohort", "protocol_sha256": "fixture-protocol",
            "source_sha256": {"src/example.py": sha(root / "src/example.py")},
            "inputs": {"bound_files": [initial]}}, "source_identity": {
                "study_reporting/example.py": sha(root / "study_reporting/example.py")},
            "runs": self.runs, "evaluation_paths": paths, "evaluation_reports": reports,
            "references": self.refs, "analysis": {"report_path": self.analysis["path"]},
            "deliverables": {"resources": resources, "secondary": secondary, "tables": tables},
            "figure_evidence": {"data": figures}, "diagnosis_path": diagnosis["path"],
            "diagnosis_artifacts": diagnosis_artifacts}

    def file(self, name, payload, *, referenced=True):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        reference = {"path": str(path), "sha256": sha(path)}
        if referenced:
            self.refs.append(reference)
        return reference

    def report(self, name, value):
        reference = self.file(name, (json.dumps(value, sort_keys=True) + "\n").encode())
        path = Path(reference["path"])
        path.with_suffix(".sha256").write_text(f"{reference['sha256']}  {path.name}\n", encoding="ascii")
        return reference


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.f = Fixture(self.root)
        self.output = self.root / "artifacts/release/package"

    def build(self, **kwargs):
        return p.build_release(self.f.context, self.f.diagnosis, self.output, root=self.root, **kwargs)

    def test_package_copies_four_actors_and_complete_portable_inputs(self):
        manifest = self.build()
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(manifest["local_actor_exports"]), 4)
        self.assertFalse(manifest["phase6_complete"])
        self.assertFalse(manifest["published"])
        data = json.loads((self.output / "reproduction_inputs.json").read_text())
        self.assertEqual(len(data["evaluations"]), 4)
        self.assertEqual(len(data["tables"]), 6)
        self.assertEqual(len(data["figure_data"]), 2)
        self.assertIsNone(data["bank_key"])
        for row in data["tables"] + data["evaluations"] + data["figure_data"]:
            self.assertTrue((self.output / row["path"]).is_file())
            self.assertFalse(Path(row["path"]).is_absolute())
        self.assertTrue(any(r["path"].endswith("events.jsonl") for r in manifest["files"]))

    def test_binary_archive_has_exact_original_size_and_hash(self):
        self.build()
        archive = json.loads((self.output / "archive_locations.json").read_text())
        model = next(r for r in archive["files"] if r["original_path"].endswith("model.pt"))
        self.assertEqual(model["sha256"], sha(Path(model["original_path"])))
        self.assertEqual(model["bytes"], len(b"SYNTHETIC LARGE MODEL"))
        self.assertFalse(any("actor_export" in r["original_path"] for r in archive["files"]))

    def test_deterministic_across_fresh_destination_directories(self):
        a = self.build()
        other = self.root / "artifacts/release/package2"
        b = p.build_release(self.f.context, self.f.diagnosis, other, root=self.root)
        self.assertEqual(a, b)
        self.assertEqual((self.output / "SHA256SUMS").read_bytes(), (other / "SHA256SUMS").read_bytes())

    def test_checksum_and_inventory_use_case_sensitive_posix_order(self):
        self.f.file("artifacts/Z_table.csv", b"fixture,value\nupper,0\n")
        self.f.file("artifacts/a_table.csv", b"fixture,value\nlower,0\n")
        manifest = self.build()
        names = [row["path"] for row in manifest["files"]]
        self.assertEqual(names, sorted(names))
        sums = (self.output / "SHA256SUMS").read_text(encoding="ascii")
        listed = [line.split("  ", 1)[1] for line in sums.splitlines()]
        self.assertEqual(listed, sorted(listed))
        z = next(name for name in listed if name.endswith("Z_table.csv"))
        a = next(name for name in listed if name.endswith("a_table.csv"))
        self.assertLess(listed.index(z), listed.index(a))
        self.assertEqual(p.inspect_release(self.output), manifest)

    def test_missing_actor_payload_or_invalid_audit_ref_fails_closed(self):
        state = Path(self.f.runs[0]["actor_export"]) / "state.pt"
        state.write_bytes(b"CHANGED")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_frozen_source_change_is_rejected_before_any_output(self):
        (self.root / "src/example.py").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_wrong_cohort_or_incomplete_diagnosis_is_rejected(self):
        for update in ({"cohort_identity": "different"}, {"status": "needs_review"}):
            with self.subTest(update=update):
                diagnosis = {**self.f.diagnosis, **update}
                with self.assertRaises(ValueError):
                    p.build_release(self.f.context, diagnosis, self.output, root=self.root)

    def test_all_diagnosis_artifacts_required(self):
        self.f.context["diagnosis_artifacts"].pop()
        with self.assertRaisesRegex(ValueError, "diagnosis JSON"):
            self.build()

    def test_diagnosis_bytes_and_native_report_requirements(self):
        altered = {**self.f.diagnosis, "changed": True}
        with self.assertRaisesRegex(ValueError, "publication differs"):
            p.build_release(self.f.context, altered, self.output, root=self.root)
        self.f.context["evaluation_reports"][0]["native_engine"] = False
        with self.assertRaisesRegex(ValueError, "native final-bank"):
            self.build()

    def test_historical_windows_path_is_not_reinterpreted_as_host_relative(self):
        self.f.report("artifacts/history.json", {"prior": {"path": "Z:\\old\\record.json", "sha256": "a" * 64}})
        manifest = self.build()
        self.assertEqual(manifest["nested_provenance_references_retained_without_reanchoring"], 1)
        archive = json.loads((self.output / "archive_locations.json").read_text())
        self.assertEqual(len(archive["unresolved_nested_provenance"]), 1)

    def test_protected_or_existing_output_rejected(self):
        for out in (self.root / "src/nested", Path(self.f.runs[0]["run_dir"]) / "release", self.root):
            with self.subTest(out=out), self.assertRaises((ValueError, FileExistsError)):
                p.build_release(self.f.context, self.f.diagnosis, out, root=self.root)
        self.output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.build()

    def test_secrets_environments_and_outside_inputs_rejected(self):
        for name in (".envs/token.json", ".aws/credentials", "artifacts/private.key"):
            with self.subTest(name=name):
                ref = self.f.file(name, b"not a real secret", referenced=False)
                c = deepcopy(self.f.context); c["release_inputs"] = [ref]
                with self.assertRaisesRegex(ValueError, "secrets|secret"):
                    p.build_release(c, self.f.diagnosis, self.output, root=self.root)

    def test_commercial_wad_and_external_tool_never_copied(self):
        self.f.file("artifacts/scenarios/DOOM2.wad", b"synthetic external asset")
        self.f.file(".tools/compiler/acc", b"synthetic executable")
        manifest = self.build()
        archive = json.loads((self.output / "archive_locations.json").read_text())
        for suffix in ("DOOM2.wad", "compiler/acc"):
            self.assertTrue(any(r["original_path"].replace("\\", "/").endswith(suffix) for r in archive["files"]))
            self.assertFalse(any(r["path"].endswith(suffix) for r in manifest["files"]))

    def test_historical_commands_not_invented(self):
        self.build()
        record = json.loads((self.output / "command_provenance.json").read_text())
        self.assertFalse(record["historical_shell_commands_recorded"])
        self.assertFalse(record["historical_shell_commands_invented"])
        self.assertEqual(record["saved_command_evidence"], [])
        self.assertIn("../../regenerated", record["reproduction_command"])

    def test_native_scenario_ini_is_included_compiled_object_is_archived(self):
        directory = Path(self.f.context["evaluation_paths"][0]).parent
        config = directory / "episodes/000000/attempt_00/scenario/player_0.ini"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("# synthetic native configuration\n")
        binary = config.with_name("task.o"); binary.write_bytes(b"SYNTHETIC OBJECT")
        manifest = self.build()
        self.assertTrue(any(row["path"].endswith("player_0.ini") for row in manifest["files"]))
        archive = json.loads((self.output / "archive_locations.json").read_text())
        self.assertTrue(any(row["original_path"].endswith("task.o") for row in archive["files"]))

    def test_operational_helpers_captured_without_retrospective_freeze_claim(self):
        self.f.file("scripts/gpu_server_env.sh", b"# synthetic shell helper\n", referenced=False)
        self.f.file("linux_gpu_server_setup.md", b"Synthetic server setup guide\n", referenced=False)
        manifest = self.build()
        helper = "source/scripts/gpu_server_env.sh"
        self.assertTrue((self.output / helper).is_file())
        self.assertTrue((self.output / "source/linux_gpu_server_setup.md").is_file())
        self.assertNotIn("scripts/gpu_server_env.sh", manifest["frozen_source_sha256"])
        capture = manifest["operational_helper_capture"]
        self.assertIn("current working-tree", capture["scope"])
        self.assertTrue(any(row["path"] == helper for row in capture["files"]))

    def test_missing_historical_nested_reference_is_disclosed(self):
        self.f.report("artifacts/history.json", {"prior": {"path": "missing.json", "sha256": "a" * 64}})
        manifest = self.build()
        self.assertEqual(manifest["nested_provenance_references_retained_without_reanchoring"], 1)
        archive = json.loads((self.output / "archive_locations.json").read_text())
        self.assertEqual(archive["unresolved_nested_provenance"][0]["status"], "relative_reference_retained_with_original_anchor_semantics")

    def test_added_changed_missing_and_manifest_tampering_detected(self):
        self.build()
        target = self.output / "README.md"
        original = target.read_bytes()
        target.write_bytes(original + b"changed")
        with self.assertRaises(ValueError): p.inspect_release(self.output)
        target.write_bytes(original)
        extra = self.output / "unexpected.txt"; extra.write_text("unlisted")
        with self.assertRaisesRegex(ValueError, "unlisted"): p.inspect_release(self.output)
        extra.unlink()
        target.unlink()
        with self.assertRaises(ValueError): p.inspect_release(self.output)
        target.write_bytes(original)
        manifest = self.output / "release_manifest.json"
        manifest.write_text(manifest.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "receipt"): p.inspect_release(self.output)

    def test_path_traversal_absolute_and_duplicate_manifest_entries_rejected(self):
        self.build()
        path = self.output / "release_manifest.json"
        original = json.loads(path.read_text())
        for bad in ("../escape", "/absolute/path", "C:/absolute/path", "source\\escape"):
            with self.subTest(bad=bad):
                record = deepcopy(original); record["files"][0]["path"] = bad
                p._json(path, record)
                (self.output / "release_manifest.sha256").write_text(f"{sha(path)}  release_manifest.json\n", encoding="ascii")
                with self.assertRaises(ValueError): p.inspect_release(self.output)
        record = deepcopy(original); record["files"].append(record["files"][0])
        p._json(path, record)
        (self.output / "release_manifest.sha256").write_text(f"{sha(path)}  release_manifest.json\n", encoding="ascii")
        with self.assertRaises(ValueError): p.inspect_release(self.output)

    def test_size_limit_and_unsupported_extension_fail_closed(self):
        with self.assertRaises(ValueError): self.build(archive_binary_limit=0)
        self.f.file("artifacts/unreviewed.blob", b"opaque")
        with self.assertRaisesRegex(ValueError, "allowlist"): self.build()

    def test_symlink_or_junction_input_rejected_without_following(self):
        link = self.root / "artifacts/link.json"
        target = self.root / "artifacts/completion/resource_summary.json"
        # The path predicate is mocked because Windows may disallow making links.
        original = Path.is_symlink
        with patch.object(Path, "is_symlink", lambda self: self == link or original(self)):
            with self.assertRaisesRegex(ValueError, "links"):
                p._safe(link)

    def test_package_reader_has_no_torch_simulator_network_imports(self):
        text = Path(p.__file__).read_text()
        self.assertNotIn("import torch", text)
        self.assertNotIn("import vizdoom", text)
        self.assertNotIn("subprocess", text)
        self.assertNotIn("urlopen", text)


if __name__ == "__main__":
    unittest.main()
