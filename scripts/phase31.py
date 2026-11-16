"""Verify P3.1 network implementation on CPU without research training or replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import unittest

import torch

from world_models.infrastructure.config import resolve_config
from world_models.policies import build_actor_critic
from world_models.training.runner import source_identity as model_source_identity


ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = ("tests.test_local_actor", "tests.test_local_critic", "tests.test_local_policy_initialization",
                "tests.test_contracts", "tests.test_config", "tests.test_reproducibility")


def source_identity() -> dict[str, str]:
    sources = model_source_identity(ROOT)
    files = [*sorted((ROOT / "src/world_models/policies").glob("*.py")),
             ROOT / "src/world_models/contracts.py", ROOT / "src/world_models/env/locality.py",
             ROOT / "src/world_models/env/coop.py", ROOT / "scripts/phase31.py", ROOT / "tests/__init__.py",
             *(ROOT / (module.replace(".", "/") + ".py") for module in TEST_MODULES),
             *sorted((ROOT / "configs").rglob("*.yaml"))]
    sources.update({path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in files})
    return sources


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed_ids = []

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


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
        cases = []
        for family in ("relay", "cover"):
            settings = resolve_config(family, "B", "pilot", ROOT / "configs").settings
            policy = build_actor_critic(settings, master_seed=0, variant="B")
            cases.append({"task_family": family, "policy_config": asdict(policy.actor.config),
                          "actor_parameters": sum(p.numel() for p in policy.actor.parameters()),
                          "critic_parameters": sum(p.numel() for p in policy.critic.parameters()),
                          "shared_actor_weights": True, "separate_external_agent_memories": True})
        report = {"schema_version": "p31-implementation-v1",
                  "status": "complete_p31_implementation" if passed else "failed",
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "scope": "CPU synthetic actor/critic and integration invariants only",
                  "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                  "skipped": len(result.skipped), "passed_tests": sorted(result.passed_ids),
                  "source_sha256": before, "source_unchanged_during_verification": unchanged,
                  "cases": cases, "pytorch": torch.__version__, "device": "cpu",
                  "elapsed_seconds": time.perf_counter() - started,
                  "new_environment_interactions": 0, "research_training_executed": False,
                  "learned_policy_quality_measured": False, "phase3_acceptance": False}
        content = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        path = output_dir / "implementation_checks.json"
        with path.open("xb") as stream:
            stream.write(content)
        digest = hashlib.sha256(content).hexdigest()
        path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
        return report
    finally:
        torch.set_num_threads(threads)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("verify", help="run synthetic CPU checks and write fresh implementation evidence")
    check.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = verify(args.output_dir)
    print(json.dumps({key: report[key] for key in ("status", "tests_run", "failures", "errors", "skipped",
                                                  "research_training_executed", "phase3_acceptance")}, indent=2))
    return 0 if report["status"] == "complete_p31_implementation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
