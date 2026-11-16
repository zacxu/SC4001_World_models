"""Fresh, hash-bound P2.5 implementation evidence, separate from model quality.

The lightweight preparation gate runs synthetic structural fixtures before a
research checkpoint is loaded. Reuse requires this exact schema, successful
required test IDs, byte receipts and the current execution-source identity.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import unittest
import uuid

from world_models.training.runner import ROOT, source_identity as p24_source_identity


SCHEMA_VERSION = "compact-p25-implementation-v1"
STRUCTURAL_TESTS = (
    "tests.test_world_model.WorldModelTests.test_autonomous_rollout_has_no_future_image_or_privileged_route",
    "tests.test_world_model.WorldModelTests.test_shared_representation_reaches_decoder_once_for_both_views",
    "tests.test_world_model.WorldModelTests.test_ordered_actions_are_preserved_in_transition_heads",
    "tests.test_world_model.WorldModelTests.test_independent_batch_termination_freezes_finished_rows",
    "tests.test_world_model.WorldModelTests.test_prior_preserved_when_either_terminal_view_missing",
    "tests.test_model_heads.TaskHeadsTests.test_joint_outcome_truth_table_and_deadline_precedence",
    "tests.test_model_heads.TaskHeadsTests.test_reward_labels_separate_non_deadline_outcome_from_clock",
    "tests.test_model_sampling.ReplaySamplingTests.test_windows_align_labels_across_chunks_and_keep_terminal_masks",
    "tests.test_model_sampling.ReplaySamplingTests.test_fit_and_diagnostic_partitions_have_separate_rng_and_purpose_guards",
)
NEW_TEST_MODULES = ("tests.test_phase25_selection", "tests.test_phase25_metrics",
                    "tests.test_phase25_evaluation", "tests.test_phase25_cli", "tests.test_initial_selection")
_FIXTURE_MODULES = ("tests.test_replay_storage", "tests.test_targets")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value: dict) -> dict:
    """Publish fresh JSON and a SHA256 commit receipt, never overwrite evidence."""
    path = Path(path)
    receipt = path.with_suffix(".sha256")
    if path == receipt:
        raise ValueError("JSON output cannot use the reserved .sha256 extension")
    if path.exists() or receipt.exists():
        raise FileExistsError("P2.5 evidence is immutable; use a fresh output path")
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp-" + uuid.uuid4().hex
    temporary = path.with_name(path.name + suffix)
    temporary_receipt = receipt.with_name(receipt.name + suffix)
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        with temporary_receipt.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(f"{digest}  {path.name}\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or receipt.exists():
            raise FileExistsError("P2.5 evidence is immutable; use a fresh output path")
        os.replace(temporary, path)
        os.replace(temporary_receipt, receipt)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_receipt.unlink(missing_ok=True)
    return {"path": str(path), "receipt_path": str(receipt), "sha256": digest}


def source_identity() -> dict[str, str]:
    """Bind the unchanged P2.4 checkpoint sources plus evaluation code/tests."""
    identity = dict(p24_source_identity(ROOT))
    paths = [*sorted((ROOT / "src/world_models/evaluation").glob("*.py")), ROOT / "scripts/phase25.py",
             ROOT / "src/world_models/training/selection.py", ROOT / "scripts/phase24_freeze.py",
             *(ROOT / (name.replace(".", "/") + ".py") for name in NEW_TEST_MODULES),
             ROOT / "tests/__init__.py", ROOT / "tests/test_world_model.py", ROOT / "tests/test_model_heads.py",
             ROOT / "tests/test_model_sampling.py", *(ROOT / (name.replace(".", "/")+".py") for name in _FIXTURE_MODULES)]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("P2.5 source is incomplete; transfer evaluation modules, scripts and complete tests/. "
                                "Missing: " + ", ".join(missing))
    identity.update({path.relative_to(ROOT).as_posix(): _hash(path) for path in paths})
    return identity


def _preflight(names) -> None:
    modules = {".".join(name.split(".")[:2]) for name in names}
    modules.update(_FIXTURE_MODULES)
    required = [ROOT / "tests/__init__.py", *(ROOT / (name.replace(".", "/")+".py") for name in sorted(modules))]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("P2.5 verification needs the complete checkout tests/ package, including tests/__init__.py "
                                "and shared fixtures. Missing: " + ", ".join(missing))
    try:
        spec = importlib.util.find_spec("tests")
    except (ImportError, ValueError) as error:
        raise ImportError("P2.5 cannot resolve this checkout's tests package; run from the project root") from error
    actual = Path(spec.origin).resolve() if spec is not None and spec.origin else None
    expected = (ROOT / "tests/__init__.py").resolve()
    if actual != expected:
        raise ImportError(f"P2.5 resolved tests to {actual}, expected {expected}; run from the checkout root "
                          "with the existing environment and complete tests/ directory")


class _RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed_ids.append(test.id())


def _evidence(path: Path, report: dict, *, reused: bool) -> dict:
    return {"status": "passed", "report_path": str(path), "report_sha256": _hash(path),
            "source_sha256": report["source_sha256"], "structural_tests": report["structural_tests"],
            "tests_run": report["tests_run"], "scope": report["scope"], "reused": reused,
            "P2.5_quality_acceptance": False}


def _run(output_dir, *, full: bool) -> dict:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("P2.5 verification requires a fresh output directory")
    names = (*STRUCTURAL_TESTS, *NEW_TEST_MODULES) if full else STRUCTURAL_TESTS
    _preflight(names)
    before = source_identity()
    p24 = p24_source_identity(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
    with (output / "tests.txt").open("x", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=_RecordedResult).run(suite)
    after = source_identity()
    structural_passed = set(STRUCTURAL_TESTS).issubset(result.passed_ids)
    passed = result.wasSuccessful() and structural_passed and before == after
    report = {"schema_version": SCHEMA_VERSION, "status": "passed" if passed else "failed",
              "scope": "full_targeted_implementation" if full else "required_structural_checks",
              "source_sha256": before, "p24_source_sha256": p24,
              "source_unchanged": before == after, "structural_tests": list(STRUCTURAL_TESTS),
              "structural_checks_passed": structural_passed, "requested_tests": list(names),
              "passed_test_ids": result.passed_ids, "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
              "elapsed_seconds": time.monotonic()-started,
              "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "research_training_executed": False, "research_checkpoint_evaluated": False,
              "new_environment_interactions": 0, "gpu_training_measured": False,
              "fixture_note": "synthetic implementation fixtures only; trained-model quality is not established",
              "P2.5_quality_acceptance": False}
    path = output / "implementation_checks.json"
    write_json(path, report)
    if not passed:
        raise RuntimeError(f"P2.5 implementation checks failed; inspect {output / 'tests.txt'}")
    return _evidence(path, report, reused=False)


def _read_verified(path) -> dict:
    path = Path(path)
    receipt = path.with_suffix(".sha256")
    if not path.is_file() or not receipt.is_file():
        raise ValueError("P2.5 verification JSON and SHA receipt are both required")
    try:
        parts = receipt.read_text(encoding="ascii").strip().split(maxsplit=1)
    except (OSError, UnicodeError) as error:
        raise ValueError("P2.5 verification receipt cannot be read") from error
    if len(parts) != 2 or parts[1] != path.name or parts[0] != _hash(path):
        raise ValueError("P2.5 verification SHA receipt mismatch")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError("P2.5 verification JSON is invalid") from error
    if (not isinstance(report, dict) or report.get("schema_version") != SCHEMA_VERSION
            or report.get("status") != "passed" or report.get("structural_checks_passed") is not True
            or report.get("source_unchanged") is not True
            or report.get("research_training_executed") is not False
            or report.get("research_checkpoint_evaluated") is not False
            or report.get("P2.5_quality_acceptance") is not False
            or type(report.get("new_environment_interactions")) is not int
            or report["new_environment_interactions"] != 0
            or type(report.get("tests_run")) is not int or report["tests_run"] < len(STRUCTURAL_TESTS)
            or type(report.get("failures")) is not int or report["failures"] != 0
            or type(report.get("errors")) is not int or report["errors"] != 0):
        raise ValueError("P2.5 verification report has the wrong schema, scope or success status")
    for name in ("structural_tests", "passed_test_ids"):
        values = report.get(name)
        if (not isinstance(values, list) or any(not isinstance(item, str) for item in values)
                or not set(STRUCTURAL_TESTS).issubset(values)):
            raise ValueError("P2.5 verification does not prove every required structural test passed")
    if report.get("scope") not in ("full_targeted_implementation", "required_structural_checks"):
        raise ValueError("P2.5 verification scope is unsupported")
    if report.get("source_sha256") != source_identity() or report.get("p24_source_sha256") != p24_source_identity(ROOT):
        raise ValueError("P2.5 verification source identity changed; run verify into a fresh directory")
    return report


def prepare_verification(output_dir, reuse_path=None) -> dict:
    """Run only required structural fixtures, or reuse exact verified evidence.

    The caller invokes this before loading any research checkpoint. A campaign
    should call it once and share the returned evidence across its four cases.
    The output directory must be fresh even when copying reusable evidence.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("P2.5 verification requires a fresh output directory")
    if reuse_path is None:
        return _run(output, full=False)
    report = _read_verified(reuse_path)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "implementation_checks.json"
    write_json(path, report)
    return {**_evidence(path, report, reused=True), "reused_from": str(reuse_path)}


def verify_implementation(output_dir) -> dict:
    """Run structural checks and all targeted P2.5 implementation unit tests."""
    return _run(output_dir, full=True)
