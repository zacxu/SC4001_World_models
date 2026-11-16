"""P4.2: prepare, explicitly distill, resume and safely stop a local student."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import unittest

import torch

from scripts.phase31 import RecordedResult
from world_models.data.ledger import atomic_json
from world_models.distillation.protocol import plan


ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = (
    "tests.test_distillation_device_identity",
    "tests.test_distillation_learning",
    "tests.test_distillation_handoff", "tests.test_distillation_runner", "tests.test_phase42_cli",
    "tests.test_local_actor", "tests.test_local_critic", "tests.test_local_policy_initialization",
    "tests.test_teacher_policy", "tests.test_teacher_learning", "tests.test_teacher_runner",
    "tests.test_imagination_anchors", "tests.test_imagination_sampling", "tests.test_imagination_versions",
    "tests.test_imagined_rollout_generator", "tests.test_imagined_rollout_schema",
    "tests.test_rollout_storage", "tests.test_rollout_streams",
    "tests.test_mappo_objectives", "tests.test_mappo_recurrent", "tests.test_mappo_updater",
    "tests.test_cycle_checkpoints",
)


def source_identity():
    from world_models.distillation.runner import source_identity as production_sources
    result = production_sources()
    # Bind test helpers as well as the suite actually executed in this report.
    for path in [ROOT / "scripts/phase31.py", ROOT / "scripts/phase42.py",
                 *sorted((ROOT / "tests").rglob("*.py"))]:
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def verify(output_dir):
    """Run CPU fixtures and tiny optimizer tests, without research checkpoints."""
    spec = importlib.util.find_spec("tests")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != ROOT / "tests/__init__.py":
        raise ImportError("run from the complete project root, including tests/")
    before = source_identity()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    threads, started = torch.get_num_threads(), time.perf_counter()
    torch.set_num_threads(1)
    try:
        with (output / "tests.txt").open("x", encoding="utf-8", newline="\n") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(
                unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES))
        unchanged = before == source_identity()
        passed = result.wasSuccessful() and not result.skipped and unchanged
        report = {
            "schema_version": "p42-implementation-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete_p42_implementation" if passed else "failed",
            "tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "passed_tests": sorted(result.passed_ids), "source_sha256": before,
            "source_unchanged_during_verification": unchanged,
            "scope": "CPU fixtures: teacher handoff, local-history distillation, target isolation, checkpoint/RNG resume and policy regressions",
            "pytorch": str(torch.__version__), "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "research_training_executed": False, "research_replay_used": False,
            "fixture_optimizer_updates_executed": True, "new_environment_interactions": 0,
            "gpu_runtime_measured": False, "learned_student_quality_measured": False,
            "full_d_pipeline_implemented": False, "phase4_acceptance": False,
        }
        path = output / "implementation_checks.json"
        content = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        path.write_bytes(content)
        path.with_suffix(".sha256").write_text(
            f"{hashlib.sha256(content).hexdigest()}  {path.name}\n", encoding="ascii")
        return report
    finally:
        torch.set_num_threads(threads)


def _runtime_setup():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _runner():
    from world_models.distillation.runner import DistillationRunner
    return DistillationRunner


def _positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _request_stop(directory):
    directory = Path(directory)
    path = directory / "run_manifest.json"
    content = path.read_bytes()
    manifest = json.loads(content)
    if manifest.get("schema_version") != "p42-distillation-run-v1":
        raise ValueError("stop requires an existing P4.2 distillation run")
    receipt = path.with_suffix(".sha256")
    expected = f"{hashlib.sha256(content).hexdigest()}  {path.name}"
    if not receipt.is_file() or receipt.read_text(encoding="ascii").strip() != expected:
        raise ValueError("P4.2 run manifest has no valid receipt")
    atomic_json(directory / "stop.request", {"requested_at": datetime.now(timezone.utc).isoformat()})
    return {"status": "stop_requested", "run_dir": str(directory)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="CPU fixture checks; no research training")
    check.add_argument("--output-dir", required=True)
    preview = commands.add_parser("plan", help="fixed distillation budget; no weights or GPU")
    preview.add_argument("--seed", type=int, choices=(0, 1), default=0)
    prepare = commands.add_parser("prepare", help="validate completed teacher and initialize paired student; no training")
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--teacher-run", required=True)
    prepare.add_argument("--device", default="cuda:0")
    run = commands.add_parser("run", help="generate the fixed buffer and perform 100 total distillation updates")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--max-units", type=_positive,
                     help="pause after this many additional committed stage units")
    for name in ("status", "stop"):
        command = commands.add_parser(name, help="read progress" if name == "status" else "request a checkpointed pause")
        command.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan(args.seed)
    elif args.command == "verify":
        report = verify(args.output_dir)
        result = {key: report[key] for key in ("status", "tests_run", "failures", "errors", "skipped",
                                               "research_training_executed", "phase4_acceptance")}
        print(json.dumps(result, indent=2))
        return 0 if report["status"] == "complete_p42_implementation" else 1
    elif args.command == "status":
        result = json.loads((Path(args.run_dir) / "status.json").read_text(encoding="utf-8"))
    elif args.command == "stop":
        result = _request_stop(args.run_dir)
    else:
        _runtime_setup()
        if args.command == "prepare":
            controller = _runner().prepare(args.run_dir, teacher_run=args.teacher_run, device=args.device)
            result = {**controller.status(), "training_executed": False}
        else:
            controller = _runner().resume(args.run_dir, device=args.device)
            originals = {}

            def stop_requested(_signal, _frame):
                controller.request_stop()
                print("Stop requested; finishing and checkpointing the current distillation stage unit.",
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
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"P4.2 error ({type(error).__name__}): {error}", file=sys.stderr)
        raise SystemExit(1)
