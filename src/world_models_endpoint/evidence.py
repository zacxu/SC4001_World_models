"""Source-bound checks and immutable reports for P5.3-P5.5."""
from __future__ import annotations

import re
import unittest

from world_models.data.ledger import file_sha256
from world_models_study.evidence import ROOT, read_report, safe_path, source_identity, timestamp, write_report


TEST_MODULES = (
    "tests.test_study_protocol", "tests.test_study_bank", "tests.test_study_inputs",
    "tests.test_study_registration", "tests.test_phase51_cli",
    "tests.test_paired_b_runner", "tests.test_paired_d_runner", "tests.test_paired_cohort", "tests.test_phase52_cli",
    "tests.test_endpoint_cohort", "tests.test_endpoint_ledger", "tests.test_endpoint_evaluation",
    "tests.test_endpoint_analysis", "tests.test_phase5_cli",
    "tests.test_endpoint_pipeline",
)


def expected_ids():
    def visit(suite):
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                yield from visit(test)
            else:
                yield test.id()
    ids = list(visit(unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)))
    if not ids or len(ids) != len(set(ids)) or any("_FailedTest" in name for name in ids):
        raise ValueError("endpoint verification cannot discover every required test")
    return set(ids)


def validate_verification(path, *, discover=True):
    path = safe_path(path)
    r = read_report(path)
    if (r.get("schema_version") != "p535455-implementation-v1"
            or r.get("status") != "complete_p535455_implementation"
            or r.get("source_sha256") != source_identity()
            or r.get("test_modules") != list(TEST_MODULES)
            or r.get("source_unchanged_during_verification") is not True):
        raise ValueError("endpoint verification is stale or incomplete")
    if any(type(r.get(k)) is not int or r[k] != 0 for k in ("failures", "errors", "skipped")):
        raise ValueError("endpoint verification must have no failures, errors or skips")
    if any(r.get(k) is not False for k in ("research_training_executed", "research_evaluation_executed", "gpu_runtime_measured")):
        raise ValueError("fixture checks cannot certify research execution")
    passed = r.get("passed_tests")
    if (type(passed) is not list or any(type(x) is not str for x in passed)
            or len(passed) != len(set(passed)) or type(r.get("tests_run")) is not int
            or r["tests_run"] != len(passed) or not passed
            or (discover and set(passed) != expected_ids())):
        raise ValueError("endpoint verification omits or replaces required tests")
    log = path.parent / "tests.txt"
    if r.get("tests_log") != {"path": "tests.txt", "sha256": file_sha256(log)}:
        raise ValueError("endpoint verification log changed")
    contents = log.read_text(encoding="utf-8")
    logged = [entry if entry.endswith("." + method) else f"{entry}.{method}"
              for method, entry in re.findall(r"^(\S+) \(([^\n]+)\) \.\.\. ok$", contents, re.MULTILINE)]
    if (len(logged) != len(set(logged)) or set(logged) != set(passed)
            or not re.search(rf"^Ran {len(passed)} tests? in ", contents, re.MULTILINE)
            or not re.search(r"^OK\s*$", contents, re.MULTILINE)):
        raise ValueError("endpoint verification log is truncated or inconsistent")
    return r
