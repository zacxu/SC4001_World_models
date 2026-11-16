"""P5.1: register the fixed-data exploratory protocol without running the study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import unittest

from world_models_study.evidence import TEST_MODULES


ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    from world_models_study.evidence import source_identity as sources
    return sources()


def write_report(path, report):
    from world_models_study.evidence import write_report as write
    return write(path, report)


def plan():
    from world_models_study.protocol import plan as protocol_plan
    return protocol_plan()


def register_protocol(output_dir, **kwargs):
    from world_models_study.registration import register_protocol as register
    return register(output_dir, **kwargs)


def inspect_registration(path):
    from world_models_study.registration import inspect_registration as inspect
    return inspect(path)


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


def verify(output_dir):
    """Check static definitions and fail-closed registration using CPU fixtures."""
    spec = importlib.util.find_spec("tests")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != ROOT / "tests/__init__.py":
        raise ImportError("run from the complete project root, including tests/")
    before = source_identity()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with (output / "tests.txt").open("x", encoding="utf-8", newline="\n") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(
            unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES))
    unchanged = before == source_identity()
    passed = result.testsRun > 0 and result.wasSuccessful() and not result.skipped and unchanged
    report = {
        "schema_version": "p51-implementation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_p51_implementation" if passed else "failed",
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "passed_tests": sorted(result.passed_ids), "test_modules": list(TEST_MODULES),
        "source_sha256": before, "source_unchanged_during_verification": unchanged,
        "tests_log": {"path": "tests.txt", "sha256": hashlib.sha256((output / "tests.txt").read_bytes()).hexdigest()},
        "scope": "static protocol, test-bank definitions, frozen input metadata and candidate registration fixtures",
        "python": platform.python_version(), "device": "cpu",
        "elapsed_seconds": time.perf_counter() - started,
        "research_training_executed": False, "research_evaluation_executed": False,
        "native_environment_evaluation_executed": False, "research_replay_used": False,
        "fixture_optimizer_updates_executed": False, "new_environment_interactions": 0,
        "new_model_updates": 0, "gpu_runtime_measured": False,
        "learned_student_quality_established": False, "protocol_frozen": False,
        "execution_ready": False, "cohort_frozen": False,
    }
    write_report(output / "implementation_checks.json", report)
    return report


def _summary(report, *, report_path=None):
    keys = ("status", "protocol_frozen", "candidate_only", "execution_ready", "cohort_frozen",
            "registration_identity", "reasons", "pending")
    result = {key: report[key] for key in keys if key in report}
    result["report_path"] = report.get("report_path", report_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="preview the fixed protocol and pending execution prerequisites")
    check = commands.add_parser("verify", help="run static CPU fixture checks; no optimization or native episodes")
    check.add_argument("--output-dir", required=True)
    register = commands.add_parser("register", help="freeze a candidate protocol and input metadata without training")
    register.add_argument("--output-dir", required=True)
    register.add_argument("--initial-selection", required=True)
    register.add_argument("--data-root", default=str(ROOT / "data/initial_replay/pilot"))
    register.add_argument("--statistics-root", default=str(ROOT / "artifacts/phase23/normalization"))
    register.add_argument("--config-root", default=str(ROOT / "configs"))
    register.add_argument("--verification-report", required=True)
    inspect = commands.add_parser("inspect", help="validate a saved registration and its immutable inputs")
    inspect.add_argument("--registration", required=True)
    args = parser.parse_args(argv)
    code = 0
    if args.command == "plan":
        result = plan()
    elif args.command == "verify":
        report = verify(args.output_dir)
        result = {key: report[key] for key in ("status", "tests_run", "failures", "errors", "skipped",
                                               "research_training_executed", "research_evaluation_executed",
                                               "execution_ready", "cohort_frozen")}
        result["report_path"] = str(Path(args.output_dir) / "implementation_checks.json")
        code = 0 if report["status"] == "complete_p51_implementation" else 1
    elif args.command == "register":
        report = register_protocol(args.output_dir, initial_selection=args.initial_selection,
            data_root=args.data_root, statistics_root=args.statistics_root, config_root=args.config_root,
            verification_report=args.verification_report)
        result = _summary(report, report_path=str(Path(args.output_dir) / "protocol_registration.json"))
    else:
        result = _summary(inspect_registration(args.registration), report_path=args.registration)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"P5.1 error ({type(error).__name__}): {error}", file=sys.stderr)
        raise SystemExit(1)
