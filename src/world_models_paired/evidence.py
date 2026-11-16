"""Authenticate P5.2 implementation evidence without executing research work."""
from __future__ import annotations

import re
import unittest

from world_models.data.ledger import file_sha256
from world_models_study.evidence import read_report, safe_path, source_identity, write_report


TEST_MODULES = ("tests.test_paired_b_runner", "tests.test_paired_d_runner",
                "tests.test_paired_cohort", "tests.test_phase52_cli")


def _expected_ids():
    def visit(suite):
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                yield from visit(test)
            else:
                yield test.id()
    ids = list(visit(unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)))
    if not ids or len(set(ids)) != len(ids) or any("_FailedTest" in name for name in ids):
        raise ValueError("P5.2 verification tests cannot be discovered completely")
    return set(ids)


def validate_verification(path):
    path = safe_path(path)
    report = read_report(path)
    if (report.get("schema_version") != "p52-implementation-v1"
            or report.get("status") != "complete_p52_implementation"
            or report.get("test_modules") != list(TEST_MODULES)
            or report.get("source_sha256") != source_identity()
            or report.get("source_unchanged_during_verification") is not True):
        raise ValueError("P5.2 verification schema, sources or completion is stale")
    for key in ("failures", "errors", "skipped"):
        if type(report.get(key)) is not int or report[key] != 0:
            raise ValueError(f"P5.2 verification {key} must be zero")
    for key in ("research_training_executed", "research_evaluation_executed", "gpu_runtime_measured"):
        if report.get(key) is not False:
            raise ValueError("P5.2 CPU checks cannot certify research execution")
    passed = report.get("passed_tests")
    if (type(passed) is not list or any(type(name) is not str for name in passed)
            or len(set(passed)) != len(passed) or set(passed) != _expected_ids()
            or type(report.get("tests_run")) is not int or report["tests_run"] != len(passed)):
        raise ValueError("P5.2 verification omits, repeats or replaces required checks")
    log = path.parent / "tests.txt"
    if (report.get("tests_log") != {"path": "tests.txt", "sha256": file_sha256(log)}
            or log.is_symlink()):
        raise ValueError("P5.2 verification test log differs")
    contents = log.read_text(encoding="utf-8")
    logged = [entry if entry.endswith("." + method) else f"{entry}.{method}"
              for method, entry in re.findall(r"^(\S+) \(([^\n]+)\) \.\.\. ok$", contents, re.MULTILINE)]
    if (len(logged) != len(set(logged)) or set(logged) != set(passed)
            or not re.search(rf"^Ran {len(passed)} tests? in ", contents, re.MULTILINE)
            or not re.search(r"^OK\s*$", contents, re.MULTILINE)):
        raise ValueError("P5.2 verification test-log identities or result differ")
    return report
