"""Verify Phase 3, exercise PPO mechanics, and evaluate exported local actors."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import time
import unittest

from scripts.phase36_support.evidence import (
    INVARIANT_TESTS, ROOT, file_sha256, source_identity, write_report,
)


TEST_MODULES = (
    "tests.test_contracts", "tests.test_model_heads", "tests.test_local_actor",
    "tests.test_mappo_objectives", "tests.test_mappo_recurrent", "tests.test_mappo_updater",
    "tests.test_imagined_rollout_generator", "tests.test_rollout_streams",
    "tests.test_phase36_evidence", "tests.test_phase36_evaluation",
    "tests.test_phase36_mechanics", "tests.test_phase36_gate", "tests.test_phase36_cli", "tests.test_cycle_study",
)


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


def _test_sources():
    # Fixtures imported by selected suites are evidence dependencies too.
    return {path.relative_to(ROOT).as_posix(): file_sha256(path)
            for path in sorted((ROOT / "tests").glob("*.py"))}


def verify(output_dir):
    """Run CPU fixtures; never launch native engines or a research training run."""
    import torch

    spec = importlib.util.find_spec("tests")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != ROOT / "tests/__init__.py":
        raise ImportError("run from the complete project root, including tests/")
    before, test_before = source_identity(), _test_sources()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    threads, started = torch.get_num_threads(), time.perf_counter()
    torch.set_num_threads(1)
    try:
        with (output / "tests.txt").open("x", encoding="utf-8", newline="\n") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(
                unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES))
        unchanged = before == source_identity() and test_before == _test_sources()
        passed_ids = set(result.passed_ids)
        invariants = {name: {"status": "passed" if set(ids) <= passed_ids else "failed",
                             "test_ids": list(ids)} for name, ids in INVARIANT_TESTS.items()}
        passed = (result.wasSuccessful() and not result.skipped and unchanged
                  and all(item["status"] == "passed" for item in invariants.values()))
        report = {"schema_version": "p36-implementation-v1",
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "status": "complete_p36_implementation" if passed else "failed",
                  "scope": "CPU invariant, synthetic PPO mechanics, evaluation and evidence-integrity fixtures",
                  "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                  "skipped": len(result.skipped), "passed_tests": sorted(result.passed_ids),
                  "invariants": invariants, "source_sha256": before, "test_source_sha256": test_before,
                  "source_unchanged_during_verification": unchanged,
                  "tests_log": {"path": "tests.txt", "sha256": file_sha256(output / "tests.txt")},
                  "pytorch": str(torch.__version__), "device": "cpu",
                  "elapsed_seconds": time.perf_counter() - started,
                  "new_environment_interactions": 0, "research_training_executed": False,
                  "research_replay_used": False, "fixture_optimizer_updates_executed": True,
                  "native_engine_runtime_measured": False, "gpu_runtime_measured": False,
                  "learned_native_policy_quality_measured": False, "phase3_acceptance": False}
        write_report(output / "implementation_checks.json", report)
        return report
    finally:
        torch.set_num_threads(threads)


def _runtime_setup():
    # Set this before any CUDA context is initialized. Do not import the P3.5
    # training CLI: evaluation must load only the deployed actor's parameters.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
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
    check = commands.add_parser("verify", help="CPU fixtures; no native interactions or research training")
    check.add_argument("--output-dir", required=True)
    mechanics = commands.add_parser("mechanics", help="tiny synthetic visual-cue PPO trainability check")
    mechanics.add_argument("--output-dir", required=True)
    mechanics.add_argument("--seed", type=int, default=3606)
    mechanics.add_argument("--cycles", type=int, default=40)
    mechanics.add_argument("--batch-size", type=int, default=32)
    evaluate = commands.add_parser("evaluate", help="frozen local-actor evaluation on native validation layouts")
    evaluate.add_argument("--actor-export", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--episodes", type=int, default=8)
    evaluate.add_argument("--bank-seed", type=int, default=3600)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--source-device", help="original export device if automatic identity checks cannot locate it")
    evaluate.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    evaluate.add_argument("--mode", choices=("stochastic", "greedy"), default="stochastic")
    gate = commands.add_parser("accept", help="combine verified evidence; report missing native evidence explicitly")
    gate.add_argument("--checks", required=True, help="implementation_checks.json from verify")
    gate.add_argument("--mechanics", help="mechanics report from the synthetic PPO check")
    gate.add_argument("--baseline", help="native evaluation.json for the untrained zero-interaction export")
    gate.add_argument("--trained", help="native evaluation.json for the learned actor from that same run")
    gate.add_argument("--output", required=True, help="fresh phase3_acceptance.json path")
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify(args.output_dir)
        print(json.dumps({key: result[key] for key in ("status", "tests_run", "failures", "errors", "skipped",
                                                       "phase3_acceptance")}, indent=2))
        return 0 if result["status"] == "complete_p36_implementation" else 1
    if args.command == "accept":
        from scripts.phase36_support.gate import aggregate_acceptance
        result = aggregate_acceptance(invariant_report=args.checks, mechanics_report=args.mechanics,
                                      baseline_report=args.baseline, trained_report=args.trained, output_path=args.output)
        print(json.dumps(result, indent=2))
        return 0 if result["phase3_acceptance"] else 2
    _runtime_setup()
    if args.command == "mechanics":
        from scripts.phase36_support.mechanics import run_mechanics
        result = run_mechanics(args.output_dir, seed=args.seed, cycles=args.cycles, batch_size=args.batch_size)
    else:
        from scripts.phase36_support.evaluation import evaluate_actor_export
        result = evaluate_actor_export(args.actor_export, args.output_dir, episodes=args.episodes,
                                      bank_seed=args.bank_seed, device=args.device, mode=args.mode,
                                      precision=args.precision, source_device=args.source_device)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in ("passed", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
