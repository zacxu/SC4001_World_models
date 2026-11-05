"""Bounded P2.5 dispatch, structural evidence reuse and source-transfer errors."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import phase25 as cli
from world_models.evaluation import verification as checks


class CommandTests(unittest.TestCase):
    def call(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = cli.main(arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_verify_dispatches_fixtures_without_evaluation(self):
        with patch.object(cli, "verify_implementation", return_value={"status": "passed"}) as verify, \
             patch.object(cli, "evaluate_checkpoint") as evaluate, patch.object(cli, "evaluate_campaign") as campaign:
            code, output, _ = self.call(["verify", "--output-dir", "evidence"])
        self.assertEqual(code, 0)
        verify.assert_called_once_with(Path("evidence"))
        evaluate.assert_not_called()
        campaign.assert_not_called()
        self.assertEqual(json.loads(output)["status"], "passed")

    def test_single_model_defaults_are_eight_windows_seed_zero_and_checkpoint_precision(self):
        with patch.object(cli, "evaluate_checkpoint", return_value={"pilot_readiness": {"status": "needs_review"}}) as evaluate:
            code, output, _ = self.call(["evaluate", "--checkpoint", "update_00003000.pt", "--output-dir", "screen"])
        self.assertEqual(code, 0)
        args, kwargs = evaluate.call_args
        self.assertEqual(args, (Path("update_00003000.pt"), Path("screen")))
        self.assertEqual(kwargs["windows"], 8)
        self.assertEqual(kwargs["seed"], 0)
        self.assertEqual(kwargs["device"], "cuda:0")
        self.assertIsNone(kwargs["precision"])
        self.assertFalse(json.loads(output)["P2.5_quality_acceptance"])

    def test_campaign_requires_explicit_update_and_preserves_paths_limits_and_receipt(self):
        with patch.object(cli, "evaluate_campaign", return_value={"pilot_readiness": {"status": "ready_for_exploratory_phase3"}}) as evaluate:
            code, output, _ = self.call(["campaign", "--run-root", "fits", "--checkpoint-update", "3000",
                "--output-dir", "campaign", "--windows", "64", "--seed", "17", "--device", "cpu",
                "--precision", "float32", "--data-dir", "replay", "--statistics-dir", "stats",
                "--verification-report", "checks/implementation_checks.json"])
        self.assertEqual(code, 0)
        args, kwargs = evaluate.call_args
        self.assertEqual(args, (Path("fits"), 3000, Path("campaign")))
        self.assertEqual(kwargs["windows"], 64)
        self.assertEqual(kwargs["seed"], 17)
        self.assertEqual(kwargs["data_root"], Path("replay"))
        self.assertEqual(kwargs["statistics_root"], Path("stats"))
        self.assertEqual(kwargs["verification_report"], Path("checks/implementation_checks.json"))
        self.assertEqual(json.loads(output)["pilot_readiness"]["status"], "ready_for_exploratory_phase3")

    def test_invalid_windows_seed_and_missing_checkpoint_update_fail_before_dispatch(self):
        base = ["evaluate", "--checkpoint", "model.pt", "--output-dir", "screen"]
        with patch.object(cli, "evaluate_checkpoint") as evaluate:
            for option, value in (("--windows", "0"), ("--windows", "65"), ("--windows", "1.5"), ("--seed", "-1")):
                with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main([*base, option, value])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(["campaign", "--run-root", "fits", "--output-dir", "screen"])
        evaluate.assert_not_called()

    def test_failed_readiness_and_errors_return_one_without_phase3_dispatch(self):
        arguments = ["evaluate", "--checkpoint", "model.pt", "--output-dir", "screen"]
        with patch.object(cli, "evaluate_checkpoint", return_value={"pilot_readiness": {"status": "failed", "reasons": ["fixture"]}}):
            self.assertEqual(self.call(arguments)[0], 1)
        with patch.object(cli, "evaluate_checkpoint", side_effect=ValueError("source mismatch")):
            code, _, error = self.call(arguments)
        self.assertEqual(code, 1)
        self.assertIn("source mismatch", error)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.identity = {"scripts/phase25.py": "source", "src/world_models/models/core.py": "model"}
        self.p24 = {"src/world_models/models/core.py": "model"}

    def report(self):
        return {"schema_version": checks.SCHEMA_VERSION, "status": "passed", "scope": "required_structural_checks",
                "source_sha256": deepcopy(self.identity), "p24_source_sha256": deepcopy(self.p24),
                "source_unchanged": True, "structural_tests": list(checks.STRUCTURAL_TESTS),
                "structural_checks_passed": True, "passed_test_ids": list(checks.STRUCTURAL_TESTS),
                "tests_run": len(checks.STRUCTURAL_TESTS), "failures": 0, "errors": 0,
                "research_training_executed": False, "research_checkpoint_evaluated": False,
                "new_environment_interactions": 0, "P2.5_quality_acceptance": False}

    def save(self, value=None, name="implementation_checks.json"):
        path = self.root / name
        checks.write_json(path, self.report() if value is None else value)
        return path

    def reuse(self, path, output="copy"):
        with patch.object(checks, "source_identity", return_value=self.identity), \
             patch.object(checks, "p24_source_identity", return_value=self.p24):
            return checks.prepare_verification(self.root / output, reuse_path=path)

    def test_fresh_json_receipts_are_hash_bound_and_immutable(self):
        path = self.save()
        self.assertEqual(path.with_suffix(".sha256").read_text().strip(),
                         f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        with self.assertRaises(FileExistsError):
            checks.write_json(path, {})

    def test_valid_reuse_copies_evidence_and_never_runs_tests(self):
        path = self.save()
        with patch.object(checks, "_run") as run:
            result = self.reuse(path)
        run.assert_not_called()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["reused"])
        self.assertEqual(Path(result["report_path"]).read_bytes(), path.read_bytes())
        self.assertEqual(result["report_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            self.reuse(path)

    def test_receipt_tampering_and_missing_receipt_fail_before_output_creation(self):
        path = self.save()
        original = path.read_bytes()
        path.write_bytes(original+b"changed")
        with self.assertRaisesRegex(ValueError, "receipt mismatch"):
            self.reuse(path)
        self.assertFalse((self.root / "copy").exists())
        path.with_suffix(".sha256").unlink()
        with self.assertRaisesRegex(ValueError, "both required"):
            self.reuse(path)

    def test_p24_wrong_schema_failed_status_missing_test_and_changed_sources_are_rejected(self):
        mutations = [lambda value: value.update(schema_version="p24-implementation-v1"),
                     lambda value: value.update(status="failed"),
                     lambda value: value.update(structural_checks_passed=False),
                     lambda value: value.update(passed_test_ids=list(checks.STRUCTURAL_TESTS[:-1])),
                     lambda value: value.update(source_sha256={"changed": "source"}),
                     lambda value: value.update(p24_source_sha256={"changed": "source"}),
                     lambda value: value.update(research_training_executed=True),
                     lambda value: value.update({"P2.5_quality_acceptance": True})]
        for index, mutate in enumerate(mutations):
            value = self.report()
            mutate(value)
            path = self.save(value, name=f"invalid_{index}.json")
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.reuse(path, output=f"copy_{index}")

    def test_lightweight_prepare_runs_only_required_structural_suite(self):
        with patch.object(checks, "_run", return_value={"status": "passed"}) as run:
            result = checks.prepare_verification(self.root / "fresh")
        self.assertEqual(result["status"], "passed")
        run.assert_called_once_with(self.root / "fresh", full=False)
        with patch.object(checks, "_run", return_value={"status": "passed"}) as run:
            checks.verify_implementation(self.root / "full")
        run.assert_called_once_with(self.root / "full", full=True)

    def test_missing_tests_initializer_or_shared_fixture_has_clear_preflight(self):
        package = self.root / "tests"
        package.mkdir()
        with patch.object(checks, "ROOT", self.root), self.assertRaisesRegex(FileNotFoundError, "tests/__init__.py"):
            checks._preflight(checks.STRUCTURAL_TESTS)
        for name in ("__init__", "test_world_model", "test_model_heads", "test_model_sampling", "test_targets"):
            (package / f"{name}.py").write_text("")
        with patch.object(checks, "ROOT", self.root), self.assertRaisesRegex(FileNotFoundError, "test_replay_storage"):
            checks._preflight(checks.STRUCTURAL_TESTS)
        (package / "test_replay_storage.py").write_text("")
        with patch.object(checks, "ROOT", self.root), \
             patch.object(checks.importlib.util, "find_spec", return_value=SimpleNamespace(origin="/foreign/tests/__init__.py")), \
             self.assertRaisesRegex(ImportError, "resolved tests"):
            checks._preflight(checks.STRUCTURAL_TESTS)

    def test_failed_structural_test_writes_failed_evidence_and_cannot_be_reused(self):
        class Failing(unittest.TestCase):
            def runTest(self):
                self.fail("fixture structural failure")
        with patch.object(checks, "_preflight"), \
             patch.object(checks, "source_identity", return_value=self.identity), \
             patch.object(checks, "p24_source_identity", return_value=self.p24), \
             patch.object(checks.unittest.defaultTestLoader, "loadTestsFromName",
                          side_effect=lambda _: unittest.TestSuite([Failing()])):
            with self.assertRaisesRegex(RuntimeError, "implementation checks failed"):
                checks.prepare_verification(self.root / "failed")
        path = self.root / "failed/implementation_checks.json"
        self.assertEqual(json.loads(path.read_text())["status"], "failed")
        self.assertIn("fixture structural failure", (self.root / "failed/tests.txt").read_text())
        with self.assertRaises(ValueError):
            self.reuse(path)


if __name__ == "__main__":
    unittest.main()
