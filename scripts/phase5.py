"""Verify, freeze and explicitly train the complete paired endpoint workflow."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import unittest

from world_models_endpoint.evidence import ROOT, TEST_MODULES, source_identity, write_report


def _cohort():
    from world_models_endpoint import cohort
    return cohort


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


def verify(output_dir):
    import torch
    spec = importlib.util.find_spec("tests")
    if spec is None or not spec.origin or Path(spec.origin).resolve() != ROOT / "tests/__init__.py":
        raise ImportError("run from the complete project root including tests")
    before, output = source_identity(), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    threads, started = torch.get_num_threads(), time.perf_counter()
    torch.set_num_threads(1)
    try:
        with (output / "tests.txt").open("x", encoding="utf-8", newline="\n") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(
                unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES))
        unchanged = source_identity() == before
        good = result.testsRun > 0 and result.wasSuccessful() and not result.skipped and unchanged
        from world_models.data.ledger import file_sha256
        report = {"schema_version": "p535455-implementation-v1", "status": "complete_p535455_implementation" if good else "failed",
            "created_at": datetime.now(timezone.utc).isoformat(), "tests_run": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
            "passed_tests": sorted(result.passed_ids), "test_modules": list(TEST_MODULES),
            "source_sha256": before, "source_unchanged_during_verification": unchanged,
            "tests_log": {"path": "tests.txt", "sha256": file_sha256(output / "tests.txt")},
            "device": "cpu", "pytorch": str(torch.__version__), "elapsed_seconds": time.perf_counter()-started,
            "scope": "synthetic fixtures, tiny PPO optimizer updates, mock native engine and receipt-bound integration",
            "fixture_optimizer_updates_executed": True, "research_training_executed": False,
            "research_evaluation_executed": False, "gpu_runtime_measured": False,
            "new_environment_interactions": 0, "new_model_updates": 0,
            "cohort_acceptance": False, "learned_student_quality_established": False}
        write_report(output / "implementation_checks.json", report)
        return report
    finally:
        torch.set_num_threads(threads)


def _positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def _summary(row):
    keys = ("schema_version", "status", "report_path", "cohort_path", "cohort_identity", "run_dir", "case",
        "variant", "master_seed", "execution_ready", "cohort_frozen", "local_cycles_completed", "target_local_cycles",
        "teacher_cycles_completed", "trajectories_completed", "distillation_optimizer_steps", "training_complete",
        "actor_export", "new_environment_interactions", "new_model_updates")
    return {k: row[k] for k in keys if k in row}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    check = subs.add_parser("verify", help="CPU fixture checks, no research training/test episodes")
    check.add_argument("--output-dir", required=True)
    freeze = subs.add_parser("freeze", help="prospective source freeze of the complete four-run workflow")
    freeze.add_argument("--matrix", required=True)
    freeze.add_argument("--verification-report", required=True)
    freeze.add_argument("--output-dir", required=True)
    freeze.add_argument("--confirm-test-bank-unused", action="store_true",
                        help="attest prior nonuse after reviewing development/test exposure")
    inspect = subs.add_parser("inspect")
    inspect.add_argument("--cohort", required=True)
    prepare = subs.add_parser("prepare", help="initialize one fresh registered branch, no optimizer updates")
    prepare.add_argument("--cohort", required=True)
    prepare.add_argument("--variant", choices=("B", "D"), required=True)
    prepare.add_argument("--seed", type=int, choices=(0, 1), required=True)
    prepare.add_argument("--device", default="cuda:0")
    run = subs.add_parser("run", help="resume committed optimizer/RNG state and train this branch")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--max-units", type=_positive)
    status = subs.add_parser("status", help="refresh CPU checkpoint audits; no training")
    status_args = status.add_mutually_exclusive_group(required=True)
    status_args.add_argument("--run-dir")
    status_args.add_argument("--cohort")
    stop = subs.add_parser("stop", help="request a checkpointed pause after the current unit")
    stop.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        r = verify(args.output_dir)
        print(json.dumps({k: r[k] for k in ("status", "tests_run", "failures", "errors", "skipped",
            "research_training_executed", "research_evaluation_executed")}, indent=2))
        return 0 if r["status"] == "complete_p535455_implementation" else 1
    if args.command == "freeze":
        r = _cohort().freeze_cohort(args.output_dir, matrix=args.matrix, verification_report=args.verification_report,
                                   confirm_test_bank_unused=args.confirm_test_bank_unused)
    elif args.command == "inspect":
        r = _cohort().inspect_cohort(args.cohort)
    elif args.command == "status":
        if args.run_dir:
            r = _cohort().audit_run(args.run_dir)
        else:
            c = _cohort().inspect_cohort(args.cohort)
            rows = []
            for row in c["run_matrix"]:
                if not (Path(row["run_dir"]) / "endpoint_run.sha256").is_file():
                    rows.append({"case": row["case"], "status": "not_prepared"})
                else:
                    rows.append(_summary(_cohort().audit_run(row["run_dir"])))
            print(json.dumps({"cohort_identity": c["cohort_identity"], "runs": rows}, indent=2))
            return 0
    elif args.command == "stop":
        r = _cohort().request_stop(args.run_dir)
    else:
        from scripts.phase52 import _runtime_setup
        _runtime_setup()
        if args.command == "prepare":
            r = _cohort().prepare_run(args.cohort, variant=args.variant, seed=args.seed, device=args.device)
        else:
            worker = _cohort().resume_run(args.run_dir, device=args.device)
            saved = {}
            def pause(_signal, _frame):
                worker.request_stop()
                print("Pause requested; finishing and checkpointing the current unit.", file=sys.stderr, flush=True)
            try:
                for name in ("SIGINT", "SIGTERM"):
                    if hasattr(signal, name):
                        number = getattr(signal, name)
                        saved[number] = signal.signal(number, pause)
                r = worker.run(max_units=args.max_units)
            finally:
                for number, handler in saved.items():
                    signal.signal(number, handler)
    print(json.dumps(_summary(r), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Phase 5 error ({type(error).__name__}): {error}", file=sys.stderr)
        raise SystemExit(1)
