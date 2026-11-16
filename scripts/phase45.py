"""P4.5: engineering acceptance for a completed fixed-data D campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
from world_models_acceptance.evidence import TEST_MODULES


def _torch():
    import torch
    return torch


def source_identity():
    from world_models_acceptance.evidence import source_identity as sources
    return sources()


def requirement_coverage(passed_ids):
    from world_models_acceptance.evidence import requirement_coverage as coverage
    return coverage(passed_ids)


def verification_contract_identity():
    from world_models_acceptance.evidence import verification_contract_identity as identity
    return identity()


def _accept(run_dir, output_dir, *, verification_report, device):
    from world_models_acceptance.gate import accept_campaign
    return accept_campaign(run_dir, output_dir, verification_report=verification_report, device=device)


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


def verify(output_dir):
    """Exercise synthetic CPU mechanics; do not load the research campaign."""
    spec = importlib.util.find_spec("tests")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != ROOT / "tests/__init__.py":
        raise ImportError("run from the complete project root, including tests/")
    before = source_identity()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch = _torch()
    previous_threads, started = torch.get_num_threads(), time.perf_counter()
    torch.set_num_threads(1)
    try:
        with (output / "tests.txt").open("x", encoding="utf-8", newline="\n") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(
                unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES))
        unchanged = before == source_identity()
        coverage = requirement_coverage(result.passed_ids)
        covered = bool(coverage) and all(item["status"] == "passed" for item in coverage.values())
        passed = (result.testsRun > 0 and result.wasSuccessful() and not result.skipped
                  and unchanged and covered)
        report = {
            "schema_version": "p45-implementation-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete_p45_implementation" if passed else "failed",
            "tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "passed_tests": sorted(result.passed_ids), "test_modules": list(TEST_MODULES),
            "source_sha256": before, "source_unchanged_during_verification": unchanged,
            "requirement_coverage": coverage, "all_requirements_passed": covered,
            "verification_contract_sha256": verification_contract_identity(),
            "tests_log": {"path": "tests.txt", "sha256": hashlib.sha256((output / "tests.txt").read_bytes()).hexdigest()},
            "scope": "CPU fixtures and tiny synthetic D smoke check; completed-run acceptance is separate",
            "pytorch": str(torch.__version__), "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "research_training_executed": False, "research_replay_used": False,
            "fixture_optimizer_updates_executed": True, "new_environment_interactions": 0,
            "gpu_runtime_measured": False, "learned_student_quality_measured": False,
            "phase4_acceptance": False,
        }
        path = output / "implementation_checks.json"
        content = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        path.write_bytes(content)
        path.with_suffix(".sha256").write_text(
            f"{hashlib.sha256(content).hexdigest()}  {path.name}\n", encoding="ascii")
        return report
    finally:
        torch.set_num_threads(previous_threads)


def _runtime_setup():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = _torch()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="CPU fixture checks; safe while research training continues")
    check.add_argument("--output-dir", required=True)
    accept = commands.add_parser("accept", help="audit a completed P4.4 campaign without training or native evaluation")
    accept.add_argument("--run-dir", required=True)
    accept.add_argument("--verification-report", required=True)
    accept.add_argument("--output-dir", required=True)
    accept.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "verify":
        report = verify(args.output_dir)
        result = {key: report[key] for key in ("status", "tests_run", "failures", "errors", "skipped",
                                               "all_requirements_passed", "research_training_executed",
                                               "phase4_acceptance")}
        code = 0 if report["status"] == "complete_p45_implementation" else 1
    else:
        _runtime_setup()
        report = _accept(args.run_dir, args.output_dir,
                         verification_report=args.verification_report, device=args.device)
        code = 0 if report.get("phase4_engineering_acceptance") is True else 1
        result = {
            "status": report.get("status", "passed" if code == 0 else "needs_review"),
            "phase4_engineering_acceptance": report.get("phase4_engineering_acceptance") is True,
            "phase4_acceptance": report.get("phase4_acceptance") is True,
            "reasons": report.get("reasons", []),
            "report_path": str(Path(args.output_dir) / "phase4_acceptance.json"),
            "learned_student_quality_established": False,
            "cohort_acceptance": False,
        }
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"P4.5 error ({type(error).__name__}): {error}", file=sys.stderr)
        raise SystemExit(1)
