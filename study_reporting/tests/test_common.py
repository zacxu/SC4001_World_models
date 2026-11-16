"""CPU-only metadata fixtures for the original-server reporting handoff."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from study_reporting import common
from study_reporting.evidence import read_report, write_report


class ReportingContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "completion"
        self.output.mkdir()
        self.cpath = self.root / "cohort.json"
        self.frozen = {"original.py": "a" * 64}
        self.enterContext(patch.object(common, "frozen_source_identity", return_value=self.frozen))
        self.c = {"report_path": str(self.cpath), "cohort_identity": "c" * 64,
            "protocol_sha256": "d" * 64, "bank_sha256": "b" * 64,
            "source_sha256": self.frozen, "bank": {}, "inputs": {"bound_files": []}, "run_matrix": []}
        self.runs, self.evaluations = {}, {}
        self.metrics = {"fixture_only": True, "all_80": True}
        for seed in (0, 1):
            for variant in ("B", "D"):
                directory = self.root / "runs" / variant / f"seed_{seed}"
                directory.mkdir(parents=True)
                case = f"relay/{variant}/seed_{seed}"
                self.c["run_matrix"].append({"case": case, "variant": variant, "master_seed": seed, "run_dir": str(directory)})
                audit = directory / "audit.json"
                write_report(audit, {"fixture_only": True})
                run = {"status": "complete", "training_complete": True, "run_dir": str(directory),
                    "variant": variant, "master_seed": seed, "run_identity": f"{variant}{seed}",
                    "last_checkpoint": str(directory / "checkpoint"), "audit_path": str(audit),
                    "audit_sha256": common.file_sha256(audit)}
                self.runs[str(directory)] = run
                output = self.root / "evaluation" / variant / f"seed_{seed}"
                output.mkdir(parents=True)
                epath = output / "evaluation.json"
                cost = {"published_real_joint_transitions": 8000, "known_simulator_tics_total": 32000}
                evaluation = {"variant": variant, "master_seed": seed, "interaction_accounting": cost}
                write_report(epath, evaluation)
                self.evaluations[str(epath)] = evaluation
                write_report(directory / "endpoint_test_claim.json", {"schema_version": "p54-single-final-test-claim-v1",
                    "run_dir": str(directory), "output_dir": str(output), "cohort_identity": self.c["cohort_identity"],
                    "bank_sha256": self.c["bank_sha256"]})
        write_report(self.cpath, self.c)
        self.api = SimpleNamespace(inspect_cohort=lambda path, **kw: deepcopy(self.c), audit_run=lambda path: None,
            inspect_run=lambda path: deepcopy(self.runs[str(path)]), analyze_cohort=self.analyze,
            summarize_cohort=self.ledger, load_evaluation=lambda path, **kw: deepcopy(self.evaluations[str(path)]),
            aggregate_records=lambda *args, **kw: deepcopy(self.metrics))

    def analyze(self, cohort, paths, output):
        output.mkdir()
        files = {}
        for name in ("arms.csv", "paired_differences.csv", "per_layout.csv", "episodes.csv"):
            p = output / name
            p.write_text("synthetic fixture\n")
            files[name] = {"path": name, "sha256": common.file_sha256(p)}
        report = {"report_path": str(output / "paired_analysis.json"), "status": "complete",
            "analysis": deepcopy(self.metrics), "cohort_identity": self.c["cohort_identity"],
            "source_sha256": self.frozen, "source_unchanged": True, "files": files,
            "evaluations": [common.reference(path) for path in paths]}
        write_report(output / "paired_analysis.json", report)
        return report

    def ledger(self, cohort, output):
        output.mkdir()
        runs = {}
        for run in self.runs.values():
            runs[f"{run['variant']}/seed_{run['master_seed']}"] = {"run_identity": run["run_identity"],
                "last_checkpoint": run["last_checkpoint"], "evaluation": {"status": "complete",
                    "evaluation_joint_transitions": 8000, "evaluation_simulator_tics": 32000}}
        snapshot = {"status": "complete", "missing_run_slots": [], "runs": runs}
        path = output / "snapshot_00000001.json"
        write_report(path, {"snapshot": snapshot})
        return {"snapshot_path": str(path)}

    def context(self):
        return common.load_context(self.cpath, self.output, backend=self.api)

    def test_complete_saved_record_handoff_and_final_barrier(self):
        c = self.context()
        self.assertTrue(all(c["authentication"].values()))
        self.assertEqual(len(c["runs"]), 4)
        self.assertTrue(common.validate_context(c, backend=self.api))
        self.assertFalse(c["new_training_or_test_execution"])

    def test_unfinished_training_and_changed_sources_fail_closed(self):
        first = next(iter(self.runs.values()))
        first["training_complete"] = False
        with self.assertRaisesRegex(ValueError, "unfinished"):
            self.context()
        first["training_complete"] = True
        self.c["source_sha256"] = {"changed": "f" * 64}
        with self.assertRaisesRegex(ValueError, "source set changed"):
            self.context()

    def test_changed_native_report_or_checkpoint_after_context_is_rejected(self):
        c = self.context()
        first = next(iter(self.runs.values()))
        first["last_checkpoint"] += "_new"
        with self.assertRaisesRegex(ValueError, "training boundary"):
            common.validate_context(c, backend=self.api)
        first["last_checkpoint"] = c["runs"][0]["last_checkpoint"]
        self.evaluations[c["evaluation_paths"][0]]["changed"] = True
        with self.assertRaisesRegex(ValueError, "episode evidence"):
            common.validate_context(c, backend=self.api)

    def test_stale_zero_evaluation_ledger_is_rejected(self):
        original = self.ledger
        def stale(cohort, output):
            result = original(cohort, output)
            path = Path(result["snapshot_path"])
            value = read_report(path)
            next(iter(value["snapshot"]["runs"].values()))["evaluation"]["evaluation_joint_transitions"] = 0
            path.unlink(); path.with_suffix(".sha256").unlink()
            write_report(path, value)
            return result
        self.api.summarize_cohort = stale
        with self.assertRaisesRegex(ValueError, "predates"):
            self.context()

    def test_substituted_analysis_and_native_claim_are_rejected(self):
        self.api.aggregate_records = lambda *args, **kw: {"substituted": True}
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            self.context()

    def test_public_server_guard_blocks_windows_and_allows_linux(self):
        with patch.object(common.platform, "system", return_value="Windows"):
            with self.assertRaisesRegex(ValueError, "original Linux server"):
                common.require_server()
        with patch.object(common.platform, "system", return_value="Linux"):
            common.require_server()

    def test_output_cannot_overwrite_inputs_or_sources(self):
        proof = self.root / "proof/checks.json"
        proof.parent.mkdir()
        output = self.root / "fresh"
        self.assertEqual(common.guard_output(output, self.c, proof), output)
        with self.assertRaises(ValueError):
            common.guard_output(self.root / "runs/B/seed_0/new", self.c, proof)
        with self.assertRaises((ValueError, FileExistsError)):
            common.guard_output(self.root, self.c, proof)

    def test_reference_mutation_and_wrong_claim_bank_are_rejected(self):
        path = self.root / "raw"
        path.write_text("before")
        ref = common.reference(path, receipt=False)
        path.write_text("after")
        with self.assertRaisesRegex(ValueError, "changed"):
            common.validate_references([ref])
        first = Path(self.c["run_matrix"][0]["run_dir"]) / "endpoint_test_claim.json"
        value = read_report(first)
        first.unlink(); first.with_suffix(".sha256").unlink()
        value["bank_sha256"] = "e" * 64
        write_report(first, value)
        with self.assertRaisesRegex(ValueError, "claim differs"):
            common.evaluation_paths(self.c)

    def test_embedded_analysis_and_ledger_cannot_differ_from_publications(self):
        context = self.context()
        for key in ("analysis", "ledger"):
            changed = deepcopy(context)
            changed[key]["substituted"] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "embedded analysis or ledger"):
                common.validate_context(changed, backend=self.api)

    def test_context_cannot_truncate_or_substitute_run_metadata(self):
        context = self.context()
        changed = deepcopy(context)
        changed["evaluation_reports"].pop()
        with self.assertRaisesRegex(ValueError, "all four"):
            common.validate_context(changed, backend=self.api)
        changed = deepcopy(context)
        changed["runs"][0]["fabricated_counter"] = 1
        with self.assertRaisesRegex(ValueError, "training boundary"):
            common.validate_context(changed, backend=self.api)

    def test_reinspection_reproduces_metrics_from_saved_native_records(self):
        context = self.context()
        self.api.aggregate_records = lambda *args, **kwargs: {"different": True}
        with self.assertRaisesRegex(ValueError, "no longer reproduce"):
            common.validate_context(context, backend=self.api)
