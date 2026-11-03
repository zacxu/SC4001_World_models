"""Validate and record Phase 0 without importing simulator/training libraries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from world_models.infrastructure.config import ConfigError, resolve_config, validate_pair
from world_models.infrastructure.inventory import collect_inventory
from world_models.infrastructure.reproducibility import PAIRED_INITIAL_COMPONENTS, seed_manifest


ROOT = Path(__file__).resolve().parents[1]


def configuration_report(experiment: str) -> dict:
    pairs = []
    for family in ("relay", "cover"):
        b, d = [resolve_config(family, variant, experiment, ROOT / "configs") for variant in ("B", "D")]
        validate_pair(b, d)
        pairs.append({"task_family": family, "B": b.as_dict(), "D": d.as_dict()})
    return {"experiment": experiment, "comparison": "B_vs_D", "pairs": pairs}


def run_matrix(experiment: str) -> dict:
    runs = []
    seed_namespaces: dict[tuple, int] = {}
    for family in ("relay", "cover"):
        b, d = [resolve_config(family, variant, experiment, ROOT / "configs") for variant in ("B", "D")]
        validate_pair(b, d)
        for config in (b, d):
            variant = config.settings["variant"]["name"]
            for master_seed in config.settings["experiment"]["master_seeds"]:
                seeds = seed_manifest(family, master_seed, variant)
                for component, value in seeds["components"].items():
                    branch = "paired" if component in PAIRED_INITIAL_COMPONENTS else variant
                    seed_namespaces[(family, master_seed, component, branch, None)] = value
                for agent, value in seeds["local_action_sampling"].items():
                    seed_namespaces[(family, master_seed, "local_action_sampling", variant, agent)] = value
                runs.append({"run_id": f"{experiment}-{family}-{variant}-{master_seed}-{config.config_hash[:10]}",
                             "task_family": family, "variant": variant, "master_seed": master_seed,
                             "config_hash": config.config_hash, "derived": config.derived,
                             "real_budget": config.settings["experiment"]["real_budget"],
                             "seeds": seeds, "execution_status": "not_started"})
    reverse: dict[int, tuple] = {}
    collisions = []
    for namespace, value in seed_namespaces.items():
        if value in reverse and reverse[value] != namespace:
            collisions.append({"seed": value, "namespaces": [reverse[value], namespace]})
        reverse[value] = namespace
    if collisions:
        raise ConfigError(f"derived uint32 RNG collisions: {collisions}")
    return {"experiment": experiment, "run_count": len(runs),
            "distinct_seed_namespaces": len(seed_namespaces), "seed_collisions": collisions, "runs": runs}


def _source_hashes() -> dict[str, str]:
    files = set()
    for folder, pattern in (("src", "*.py"), ("scripts", "*.py"), ("tests", "*.py"),
                            ("configs", "*.yaml"), ("docs", "*.md"), ("scenarios", "*.md")):
        files.update((ROOT / folder).rglob(pattern))
    for name in ("README.md", "pyproject.toml", "requirements-phase0.lock", "research_implementation_proposal.md", "detailed_research_implementation_plan.md"):
        files.add(ROOT / name)
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files) if path.is_file()}


def acceptance_report() -> tuple[dict, bool]:
    reports = {name: configuration_report(name) for name in ("pilot", "final")}
    matrices = {name: run_matrix(name) for name in ("pilot", "final")}
    test_command = [sys.executable, "-m", "unittest", "-v", "tests.test_config", "tests.test_contracts",
                    "tests.test_inventory", "tests.test_phase0_cli", "tests.test_reproducibility"]
    tests = subprocess.run(test_command,
                           cwd=ROOT, capture_output=True, text=True, timeout=180, check=False)
    locks = {}
    for line in (ROOT / "requirements-phase0.lock").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        locks[name] = {"expected": version, "observed": actual, "matches": actual == version}
    passed = tests.returncode == 0 and all(item["matches"] for item in locks.values())
    expected_documents = ["docs/research_contract.md", "docs/interface_contract.md", "docs/phase0_status.md", "docs/runbook.md"]
    documents = {name: (ROOT / name).is_file() for name in expected_documents}
    passed = passed and all(documents.values())
    summary = {name: {"run_count": matrices[name]["run_count"],
                      "local_ppo_cycles": reports[name]["pairs"][0]["B"]["derived"]["local_ppo_cycles"],
                      "checkpoint_count": reports[name]["pairs"][0]["B"]["derived"]["checkpoint_count"],
                      "paired_common_settings_valid": True,
                      "seed_collisions": matrices[name]["seed_collisions"]} for name in reports}
    inventory = collect_inventory(ROOT)
    return {"schema_version": 1, "phase": 0, "status": "complete" if passed else "failed",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "research_contract_and_project_foundation",
            "configuration_checks": summary, "bootstrap_lock": locks, "documents": documents,
            "tests": {"command": test_command,
                      "exit_code": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr},
            "inventory": inventory, "source_sha256": _source_hashes(),
            "phase1_handoff": {"simulator_smoke_test": "not_run",
                               "training_host_runtime_lock": "pending_phase1_smoke_test",
                               "declared_four_h200_allocation_verified": False,
                               "note": "Measured current host inventory is not verification of a separate training allocation."}}, passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "matrix", "inventory", "check"))
    parser.add_argument("--experiment", choices=("pilot", "final"), default="final")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        passed = True
        if args.command == "validate":
            report = configuration_report(args.experiment)
        elif args.command == "matrix":
            report = run_matrix(args.experiment)
        elif args.command == "inventory":
            report = collect_inventory(ROOT)
        else:
            report, passed = acceptance_report()
        encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
            print(json.dumps({"output": str(args.output.resolve()), "status": "passed" if passed else "failed"}))
        else:
            print(encoded, end="")
        return 0 if passed else 1
    except (ConfigError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"Phase 0 error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
