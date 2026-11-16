"""Immutable protocol evidence and source-bound static CPU checks."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unittest

from world_models.data.ledger import atomic_json, file_sha256


ROOT = Path(__file__).resolve().parents[2]
TEST_MODULES = (
    "tests.test_study_protocol", "tests.test_study_bank", "tests.test_study_inputs",
    "tests.test_study_registration", "tests.test_phase51_cli",
)


def safe_path(value):
    candidate = Path(value)
    if ".." in candidate.parts or any(p.is_symlink() for p in (candidate, *candidate.parents)):
        raise ValueError("protocol paths cannot traverse parents or symbolic links")
    return candidate.resolve()


def source_identity():
    """Freeze all current sources, task definitions, configurations and locks.

    Later implementations require a new registration, rather than making an
    earlier candidate manifest retrospectively execution-ready.
    """
    paths = [ROOT / "pyproject.toml"]
    for directory in ("src", "scripts", "tests"):
        paths.extend((ROOT / directory).rglob("*.py"))
    for directory, suffixes in (("configs", {".yaml", ".json"}), ("scenarios", {".acs", ".py"})):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.suffix in suffixes)
    paths.extend(ROOT.glob("requirements*.lock"))
    paths.extend(ROOT.glob("requirements*.txt"))
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(set(paths))}


def read_report(path):
    path = safe_path(path)
    receipt = path.with_suffix(".sha256")
    if (not path.is_file() or not receipt.is_file() or receipt.is_symlink()
            or receipt.read_text(encoding="ascii").strip() != f"{file_sha256(path)}  {path.name}"):
        raise ValueError(f"protocol evidence has no valid receipt: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if type(report) is not dict:
        raise ValueError("protocol evidence must be a JSON object")
    return report


def write_report(path, report):
    path = safe_path(path)
    if path.exists() or path.with_suffix(".sha256").exists():
        raise FileExistsError("protocol evidence needs a fresh destination; preserve earlier versions")
    atomic_json(path, report)
    path.with_suffix(".sha256").write_text(f"{file_sha256(path)}  {path.name}\n", encoding="ascii")
    return path


def _expected_ids():
    def visit(suite):
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                yield from visit(test)
            else:
                yield test.id()
    result = list(visit(unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)))
    if not result or len(set(result)) != len(result) or any("_FailedTest" in test for test in result):
        raise ValueError("protocol verification tests cannot be discovered completely")
    return set(result)


def validate_verification(path):
    """Authenticate the exact current static suite and adjacent unittest log."""
    path = safe_path(path)
    report = read_report(path)
    if (report.get("schema_version") != "p51-implementation-v1"
            or report.get("status") != "complete_p51_implementation"
            or report.get("test_modules") != list(TEST_MODULES)
            or report.get("source_sha256") != source_identity()
            or report.get("source_unchanged_during_verification") is not True):
        raise ValueError("protocol verification schema, sources or completion is stale")
    for key in ("failures", "errors", "skipped"):
        if type(report.get(key)) is not int or report[key] != 0:
            raise ValueError(f"protocol verification {key} must be zero")
    for key in ("research_training_executed", "research_evaluation_executed", "gpu_runtime_measured"):
        if report.get(key) is not False:
            raise ValueError("protocol verification must remain separate from research execution")
    passed = report.get("passed_tests")
    if (type(passed) is not list or any(type(p) is not str for p in passed)
            or type(report.get("tests_run")) is not int
            or len(passed) != len(set(passed)) or len(passed) != report["tests_run"]
            or set(passed) != _expected_ids()):
        raise ValueError("protocol verification omits, repeats or replaces required tests")
    binding = report.get("tests_log")
    log = path.parent / "tests.txt"
    if (type(binding) is not dict or binding.get("path") != "tests.txt"
            or not log.is_file() or log.is_symlink() or binding.get("sha256") != file_sha256(log)):
        raise ValueError("protocol verification test log differs")
    content = log.read_text(encoding="utf-8")
    logged = [entry if entry.endswith("." + method) else f"{entry}.{method}"
              for method, entry in re.findall(r"^(\S+) \(([^\n]+)\) \.\.\. ok$", content, re.MULTILINE)]
    if (len(logged) != len(set(logged)) or set(logged) != set(passed)
            or not re.search(rf"^Ran {len(passed)} tests? in ", content, re.MULTILINE)
            or not re.search(r"^OK\s*$", content, re.MULTILINE)):
        raise ValueError("protocol verification test-log identities or result differ")
    return report


def timestamp():
    return datetime.now(timezone.utc).isoformat()
