"""P5.2: prepare and resume the paired fixed-data B/D training runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = (
    "tests.test_paired_b_runner", "tests.test_paired_d_runner",
    "tests.test_paired_cohort", "tests.test_phase52_cli",
)


def _torch():
    import torch
    return torch


def _cohort():
    from world_models_paired import cohort
    return cohort


def source_identity():
    from world_models_study.evidence import source_identity as sources
    return sources()


def write_report(path, report):
    from world_models_study.evidence import write_report as write
    return write(path, report)


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


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


def verify(output_dir):
    """Run CPU fixture checks, never a research training or evaluation command."""
    spec = importlib.util.find_spec("tests")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != ROOT / "tests/__init__.py":
        raise ImportError("run from the complete project root, including tests/")
    before = source_identity()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch = _torch()
    threads, started = torch.get_num_threads(), time.perf_counter()
    torch.set_num_threads(1)
    try:
        with (output / "tests.txt").open("x", encoding="utf-8", newline="\n") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(
                unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES))
        unchanged = before == source_identity()
        passed = result.testsRun > 0 and result.wasSuccessful() and not result.skipped and unchanged
        report = {
            "schema_version": "p52-implementation-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete_p52_implementation" if passed else "failed",
            "tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "passed_tests": sorted(result.passed_ids), "test_modules": list(TEST_MODULES),
            "source_sha256": before, "source_unchanged_during_verification": unchanged,
            "tests_log": {"path": "tests.txt", "sha256": hashlib.sha256((output / "tests.txt").read_bytes()).hexdigest()},
            "scope": "CPU fixtures: paired initial inputs, fixed B/D schedules, durable checkpoints, branch isolation and safe stop/resume",
            "python": platform.python_version(), "pytorch": str(torch.__version__), "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "research_training_executed": False, "research_evaluation_executed": False,
            "native_environment_evaluation_executed": False, "research_replay_used": False,
            "fixture_optimizer_updates_executed": True, "new_environment_interactions": 0,
            "new_model_updates": 0, "gpu_runtime_measured": False,
            "learned_student_quality_established": False, "cohort_acceptance": False,
            "execution_ready": False,
        }
        write_report(output / "implementation_checks.json", report)
        return report
    finally:
        torch.set_num_threads(threads)


def _positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _summary(report, *, report_path=None):
    keys = ("status", "protocol_frozen", "candidate_only", "execution_ready", "training_tools_ready",
            "cohort_frozen", "cohort_acceptance", "cohort_identity", "registration_identity",
            "run_dir", "training_executed", "reasons", "pending")
    result = {key: report[key] for key in keys if key in report}
    if "report_path" in report or report_path is not None:
        result["report_path"] = report.get("report_path", report_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="preview the paired fixed-data budgets without loading models")
    check = commands.add_parser("verify", help="CPU fixtures only; no research training or native episodes")
    check.add_argument("--output-dir", required=True)
    register = commands.add_parser("register", help="bind a new paired-training registration; no training")
    register.add_argument("--output-dir", required=True)
    register.add_argument("--registration", required=True, help="current P5.1 protocol registration")
    register.add_argument("--verification-report", required=True)
    inspect = commands.add_parser("inspect", help="validate saved paired-training registration")
    inspect.add_argument("--cohort", required=True)
    prepare = commands.add_parser("prepare", help="prepare one fresh B or D branch; no optimizer updates")
    prepare.add_argument("--cohort", required=True)
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--variant", choices=("B", "D"), required=True)
    prepare.add_argument("--seed", type=int, choices=(0, 1), required=True)
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--development-only", action="store_true",
                         help="explicitly prepare an ineligible development run while the endpoint workflow is pending")
    run = commands.add_parser("run", help="resume only this branch from its committed checkpoint")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--max-units", type=_positive, help="pause after this many additional committed units")
    for name in ("status", "stop"):
        command = commands.add_parser(name, help="read branch progress" if name == "status" else
                                      "request a checkpointed pause at the next stage-unit boundary")
        command.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        report = verify(args.output_dir)
        result = {key: report[key] for key in ("status", "tests_run", "failures", "errors", "skipped",
            "research_training_executed", "research_evaluation_executed", "execution_ready")}
        result["report_path"] = str(Path(args.output_dir) / "implementation_checks.json")
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0 if report["status"] == "complete_p52_implementation" else 1
    if args.command == "plan":
        result = _cohort().plan()
    elif args.command == "register":
        result = _summary(_cohort().register_cohort(args.output_dir, registration=args.registration,
            verification_report=args.verification_report))
    elif args.command == "inspect":
        result = _summary(_cohort().inspect_cohort(args.cohort), report_path=args.cohort)
    elif args.command == "status":
        result = _cohort().read_status(args.run_dir)
    elif args.command == "stop":
        result = _cohort().request_stop(args.run_dir)
    else:
        _runtime_setup()
        if args.command == "prepare":
            status = _cohort().prepare_run(args.run_dir, cohort=args.cohort, variant=args.variant,
                                          seed=args.seed, device=args.device,
                                          development_only=args.development_only)
            result = {**status, "training_executed": False}
        else:
            controller = _cohort().resume_run(args.run_dir, device=args.device)
            originals = {}

            def stop_requested(_signal, _frame):
                controller.request_stop()
                print("Stop requested; finishing and checkpointing the current stage unit.",
                      file=sys.stderr, flush=True)

            try:
                for name in ("SIGINT", "SIGTERM"):
                    if hasattr(signal, name):
                        signum = getattr(signal, name)
                        originals[signum] = signal.signal(signum, stop_requested)
                result = controller.run(max_units=args.max_units)
            finally:
                for signum, handler in originals.items():
                    signal.signal(signum, handler)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"P5.2 error ({type(error).__name__}): {error}", file=sys.stderr)
        raise SystemExit(1)
