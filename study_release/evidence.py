"""Independent Phase 6 source receipts; earlier scientific source sets stay fixed."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "p6162-implementation-v1"
STATUS = "complete_p6162_implementation"
TEST_MODULES = ("study_release.tests.test_evidence", "study_release.tests.test_common",
    "study_release.tests.test_diagnosis", "study_release.tests.test_packaging",
    "study_release.tests.test_saved", "study_release.tests.test_phase6")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe_path(value):
    path = Path(value).absolute()
    if path.is_symlink() or path != path.resolve():
        raise ValueError("release evidence requires normalized paths without symbolic links")
    return path.resolve()


def read_report(path):
    path = safe_path(path)
    receipt = path.with_suffix(".sha256")
    if (not path.is_file() or not receipt.is_file() or receipt.is_symlink()
            or receipt.read_text(encoding="ascii").strip() != f"{file_sha256(path)}  {path.name}"):
        raise ValueError(f"release evidence has no valid receipt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("release evidence must be a JSON object")
    return value


def write_report(path, value):
    path = safe_path(path)
    receipt = path.with_suffix(".sha256")
    if path.exists() or receipt.exists():
        raise FileExistsError("release evidence needs a fresh destination")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    receipt.write_text(f"{file_sha256(path)}  {path.name}\n", encoding="ascii")
    return path


def reference(path, *, receipt=True):
    path = safe_path(path)
    if receipt:
        read_report(path)
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def validate_references(rows):
    if type(rows) is not list:
        raise ValueError("release references must be an explicit list")
    for row in rows:
        path = safe_path(row["path"])
        if (str(path) != row["path"] or not path.is_file() or file_sha256(path) != row["sha256"]
                or "bytes" in row and path.stat().st_size != row["bytes"]):
            raise ValueError(f"release evidence changed: {path}")


def source_identity():
    paths = list((ROOT / "study_release").rglob("*.py")) + [ROOT / "study_release/requirements.txt"]
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(paths)}


def prior_identities():
    from study_reporting.evidence import source_identity as reporting_identity
    from world_models_study.evidence import source_identity as scientific_identity
    return {"frozen_source_sha256": scientific_identity(), "reporting_source_sha256": reporting_identity()}


def _ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from _ids(test)
        else:
            yield test.id()


def expected_ids():
    result = list(_ids(unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)))
    if not result or len(set(result)) != len(result) or any("_FailedTest" in name for name in result):
        raise ValueError("cannot discover the complete Phase 6 fixture suite")
    return set(result)


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


def verify(output_dir):
    output = safe_path(output_dir)
    if output.exists():
        raise FileExistsError("verification needs a fresh directory")
    before, prior, started = source_identity(), prior_identities(), time.perf_counter()
    output.mkdir(parents=True)
    with (output / "tests.txt").open("x", encoding="utf-8", newline="\n") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(
                unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES))
    unchanged = before == source_identity() and prior == prior_identities()
    complete = result.testsRun > 0 and result.wasSuccessful() and not result.skipped and unchanged
    report = {"schema_version": SCHEMA, "status": STATUS if complete else "failed",
        "created_at": timestamp(), "test_modules": list(TEST_MODULES), "tests_run": result.testsRun,
        "passed_tests": sorted(result.passed_ids), "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "source_sha256": before, **prior,
        "source_unchanged_during_verification": unchanged,
        "tests_log": {"path": "tests.txt", "sha256": file_sha256(output / "tests.txt")},
        "elapsed_seconds": time.perf_counter() - started, "device": "cpu",
        "scope": "synthetic saved-trace, packaging, receipt and reproduction fixtures only",
        "research_training_executed": False, "research_evaluation_executed": False,
        "actual_scientific_diagnosis_executed": False, "actual_scientific_release_generated": False,
        "actual_scientific_graphics_rendered": False, "gpu_runtime_measured": False,
        "phase6_acceptance": False, "new_environment_interactions": 0, "new_model_updates": 0,
        "new_policy_optimizer_updates": 0}
    write_report(output / "implementation_checks.json", report)
    return report


def validate_verification(path):
    path = safe_path(path)
    report = read_report(path)
    if (report.get("schema_version") != SCHEMA or report.get("status") != STATUS
            or report.get("source_sha256") != source_identity()
            or any(report.get(k) != v for k, v in prior_identities().items())
            or report.get("test_modules") != list(TEST_MODULES)
            or report.get("source_unchanged_during_verification") is not True):
        raise ValueError("Phase 6 implementation proof is stale or incomplete")
    if any(type(report.get(k)) is not int or report[k] != 0 for k in ("failures", "errors", "skipped")):
        raise ValueError("all required Phase 6 checks must pass without skips")
    if any(report.get(k) is not False for k in ("research_training_executed", "research_evaluation_executed",
            "actual_scientific_diagnosis_executed", "actual_scientific_release_generated",
            "actual_scientific_graphics_rendered", "gpu_runtime_measured", "phase6_acceptance")):
        raise ValueError("synthetic verification cannot claim scientific execution")
    if any(type(report.get(k)) is not int or report[k] != 0 for k in
            ("new_environment_interactions", "new_model_updates", "new_policy_optimizer_updates")):
        raise ValueError("synthetic verification cannot claim scientific work counters")
    passed = report.get("passed_tests")
    if (type(passed) is not list or len(passed) != len(set(passed)) or set(passed) != expected_ids()
            or type(report.get("tests_run")) is not int or report["tests_run"] != len(passed)):
        raise ValueError("Phase 6 proof omits or replaces required test IDs")
    log = path.parent / "tests.txt"
    if report.get("tests_log") != {"path": "tests.txt", "sha256": file_sha256(log)}:
        raise ValueError("Phase 6 test log changed")
    body = log.read_text(encoding="utf-8")
    logged = [entry if entry.endswith("." + method) else entry + "." + method
        for method, entry in re.findall(r"^(\S+) \(([^\n]+)\) \.\.\. ok$", body, re.MULTILINE)]
    if (len(logged) != len(set(logged)) or set(logged) != set(passed)
            or not re.search(rf"^Ran {len(passed)} tests? in ", body, re.MULTILINE)
            or not re.search(r"^OK\s*$", body, re.MULTILINE)):
        raise ValueError("Phase 6 proof log is truncated or inconsistent")
    return report
