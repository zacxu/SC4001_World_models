"""Independent source receipts for reporting without changing a frozen cohort."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import re
import time
import unittest
from pathlib import Path

from world_models.data.ledger import file_sha256
from world_models_study.evidence import (
    read_report, safe_path, source_identity as frozen_source_identity, timestamp, write_report,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "p565758-implementation-v1"
STATUS = "complete_p565758_implementation"
TEST_MODULES = (
    "study_reporting.tests.test_evidence", "study_reporting.tests.test_common",
    "study_reporting.tests.test_secondary", "study_reporting.tests.test_resources",
    "study_reporting.tests.test_gate", "study_reporting.tests.test_figures",
    "study_reporting.tests.test_phase5",
)


def source_identity():
    paths = list((ROOT / "study_reporting").rglob("*.py"))
    paths += [ROOT / "study_reporting/requirements.txt"]
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(paths)}


def expected_ids():
    def visit(suite):
        for item in suite:
            if isinstance(item, unittest.TestSuite):
                yield from visit(item)
            else:
                yield item.id()
    result = list(visit(unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)))
    if not result or len(result) != len(set(result)) or any("_FailedTest" in x for x in result):
        raise ValueError("reporting verification cannot discover its complete fixture suite")
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
        raise FileExistsError("verification requires a fresh directory")
    before, frozen = source_identity(), frozen_source_identity()
    output.mkdir(parents=True)
    started = time.perf_counter()
    import torch
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (output / "tests.txt").open("x", encoding="utf-8", newline="\n") as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(
                    unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES))
    finally:
        torch.set_num_threads(threads)
    unchanged = before == source_identity() and frozen == frozen_source_identity()
    complete = result.testsRun > 0 and result.wasSuccessful() and not result.skipped and unchanged
    report = {"schema_version": SCHEMA, "status": STATUS if complete else "failed",
        "created_at": timestamp(), "tests_run": result.testsRun, "passed_tests": sorted(result.passed_ids),
        "test_modules": list(TEST_MODULES), "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "source_sha256": before, "frozen_source_sha256": frozen,
        "source_unchanged_during_verification": unchanged,
        "tests_log": {"path": "tests.txt", "sha256": file_sha256(output / "tests.txt")},
        "elapsed_seconds": time.perf_counter() - started, "device": "cpu",
        "scope": "synthetic reporting, accounting, receipt and acceptance fixtures only",
        "actual_scientific_graphics_rendered": False,
        "research_training_executed": False, "research_evaluation_executed": False,
        "gpu_runtime_measured": False, "new_environment_interactions": 0,
        "new_model_updates": 0, "new_policy_optimizer_updates": 0, "phase5_acceptance": False}
    write_report(output / "implementation_checks.json", report)
    return report


def validate_verification(path):
    path = safe_path(path)
    report = read_report(path)
    if (report.get("schema_version") != SCHEMA or report.get("status") != STATUS
            or report.get("source_sha256") != source_identity()
            or report.get("frozen_source_sha256") != frozen_source_identity()
            or report.get("source_unchanged_during_verification") is not True
            or report.get("test_modules") != list(TEST_MODULES)):
        raise ValueError("reporting implementation proof is stale or incomplete")
    if any(type(report.get(k)) is not int or report[k] != 0 for k in ("failures", "errors", "skipped")):
        raise ValueError("all required reporting checks must pass without skips")
    if any(report.get(k) is not False for k in
           ("research_training_executed", "research_evaluation_executed", "gpu_runtime_measured")):
        raise ValueError("fixture verification cannot certify scientific execution")
    passed = report.get("passed_tests")
    if (type(passed) is not list or len(passed) != len(set(passed))
            or set(passed) != expected_ids() or type(report.get("tests_run")) is not int
            or report["tests_run"] != len(passed)):
        raise ValueError("reporting proof omits or replaces required test IDs")
    log = path.parent / "tests.txt"
    if report.get("tests_log") != {"path": "tests.txt", "sha256": file_sha256(log)}:
        raise ValueError("reporting proof test log changed")
    body = log.read_text(encoding="utf-8")
    logged = [entry if entry.endswith("." + method) else entry + "." + method
              for method, entry in re.findall(r"^(\S+) \(([^\n]+)\) \.\.\. ok$", body, re.MULTILINE)]
    if (len(logged) != len(set(logged)) or set(logged) != set(passed)
            or not re.search(rf"^Ran {len(passed)} tests? in ", body, re.MULTILINE)
            or not re.search(r"^OK\s*$", body, re.MULTILINE)):
        raise ValueError("reporting proof log is truncated or inconsistent")
    return report
