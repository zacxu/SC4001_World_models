"""Verify recurrent MAPPO with CPU fixtures; no research training campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import unittest

import torch

from scripts.phase31 import RecordedResult
from scripts.phase33 import source_identity as rollout_source_identity
from world_models.mappo import MAPPOConfig
from dataclasses import asdict


ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = ("tests.test_mappo_config", "tests.test_mappo_objectives", "tests.test_mappo_recurrent",
                "tests.test_mappo_updater", "tests.test_local_actor", "tests.test_local_critic",
                "tests.test_imagined_rollout_generator", "tests.test_imagined_rollout_schema",
                "tests.test_rollout_storage", "tests.test_rollout_streams")


def source_identity() -> dict[str, str]:
    sources = rollout_source_identity()
    files = [*sorted((ROOT / "src/world_models/mappo").glob("*.py")), ROOT / "scripts/phase34.py",
             *(ROOT / (name.replace(".", "/") + ".py") for name in TEST_MODULES)]
    sources.update({path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in files})
    return sources


def verify(output_dir: str | Path) -> dict:
    spec = importlib.util.find_spec("tests")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != ROOT / "tests/__init__.py":
        raise ImportError("run from the project root with the complete checkout tests/ package")
    before = source_identity()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (output_dir / "tests.txt").open("x", encoding="utf-8", newline="\n") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(
                unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES))
        unchanged = before == source_identity()
        passed = result.wasSuccessful() and not result.skipped and unchanged
        report = {"schema_version": "p34-implementation-v1",
                  "status": "complete_p34_implementation" if passed else "failed",
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "scope": "CPU synthetic full-prefix replay, hand-calculated GAE/PPO fixtures and tiny optimizer cycles",
                  "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                  "skipped": len(result.skipped), "passed_tests": sorted(result.passed_ids),
                  "source_sha256": before, "source_unchanged_during_verification": unchanged,
                  "defaults": asdict(MAPPOConfig()), "pytorch": torch.__version__, "device": "cpu",
                  "elapsed_seconds": time.perf_counter() - started,
                  "new_environment_interactions": 0, "research_training_executed": False,
                  "research_replay_used": False, "fixture_optimizer_updates_executed": True,
                  "learned_policy_quality_measured": False, "gpu_runtime_measured": False, "phase3_acceptance": False}
        content = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        path = output_dir / "implementation_checks.json"
        with path.open("xb") as stream:
            stream.write(content)
        path.with_suffix(".sha256").write_text(f"{hashlib.sha256(content).hexdigest()}  {path.name}\n", encoding="ascii")
        return report
    finally:
        torch.set_num_threads(threads)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="run CPU fixtures and save fresh implementation evidence")
    check.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = verify(args.output_dir)
    print(json.dumps({key: report[key] for key in ("status", "tests_run", "failures", "errors", "skipped",
                                                  "research_training_executed", "phase3_acceptance")}, indent=2))
    return 0 if report["status"] == "complete_p34_implementation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
