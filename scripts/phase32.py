"""Verify P3.2 imagination anchors on synthetic CPU fixtures, without training."""
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

from scripts.phase31 import RecordedResult, source_identity as policy_source_identity


ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = ("tests.test_imagination_sampling", "tests.test_imagination_versions", "tests.test_imagination_anchors",
                "tests.test_local_actor", "tests.test_local_critic", "tests.test_local_policy_initialization",
                "tests.test_model_sampling")


def source_identity() -> dict[str, str]:
    sources = policy_source_identity()
    files = [*sorted((ROOT / "src/world_models/imagination").glob("*.py")), ROOT / "scripts/phase32.py",
             *(ROOT / (name.replace(".", "/") + ".py") for name in TEST_MODULES),
             ROOT / "tests/test_replay_storage.py", ROOT / "tests/test_targets.py"]
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
        report = {"schema_version": "p32-implementation-v1",
                  "status": "complete_p32_implementation" if passed else "failed",
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "scope": "CPU synthetic replay and untrained world-model/local-actor anchor invariants",
                  "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                  "skipped": len(result.skipped), "passed_tests": sorted(result.passed_ids),
                  "source_sha256": before, "source_unchanged_during_verification": unchanged,
                  "defaults": {"anchors": 256, "recent_fraction": 0.5, "recent_transitions": 100000,
                               "cache_capacity": 512, "precision": "float32", "prefix_policy": "full_from_reset",
                               "sampling": "fixed-quota uniform/recent eligible source states with replacement"},
                  "pytorch": torch.__version__, "device": "cpu", "elapsed_seconds": time.perf_counter() - started,
                  "new_environment_interactions": 0, "research_training_executed": False,
                  "research_replay_used": False, "learned_policy_quality_measured": False,
                  "gpu_runtime_measured": False, "phase3_acceptance": False}
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
    check = commands.add_parser("verify", help="run CPU fixture checks and save fresh implementation evidence")
    check.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = verify(args.output_dir)
    print(json.dumps({key: report[key] for key in ("status", "tests_run", "failures", "errors", "skipped",
                                                  "research_training_executed", "phase3_acceptance")}, indent=2))
    return 0 if report["status"] == "complete_p32_implementation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
