"""Explicitly segregate mechanics/acceptance interactions from learning replay."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading


class ValidationLedger:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "validation_interactions.jsonl"
        self._lock = threading.Lock()

    def record(self, stage: str, env, published_transitions: int, *, expected_error: bool = False) -> dict:
        consumed = env.coordinator.consumed_joint_decisions
        initialization = env.initialization_joint_decisions
        failed = consumed - initialization - published_transitions
        if failed < 0 or failed and not expected_error:
            raise ValueError("validation interaction accounting does not reconcile")
        record = {"stage": stage, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                  "purpose": "environment_validation", "learning_eligible": False,
                  "enters_training_replay": False, "diagnostic_import_requires_training_ledger": True,
                  "task_family": env.task_family, "scenario_sha256": env.scenario["hashes"]["wad_sha256"],
                  "coordinator_run_dir": str(env.coordinator.run_dir),
                  "consumed_joint_decisions": consumed, "published_transitions": published_transitions,
                  "initialization_joint_decisions": initialization, "failed_dispatch_attempts": failed}
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record

    def manifest(self) -> dict:
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()] if self.path.exists() else []
        identities = [row["coordinator_run_dir"] for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError("an environment was entered twice in the validation ledger")
        return {"schema_version": "phase1-validation-data-v1", "purpose": "environment_validation",
                "enters_training_replay": False, "learning_eligible": False,
                "diagnostic_import_requires_training_ledger": True,
                "import_rule": "No automatic learning import. Any future import must explicitly enter the original real interactions in the training budget ledger and respect split restrictions.",
                "ledger_path": str(self.path.resolve()), "environments": len(rows),
                "consumed_joint_decisions": sum(row["consumed_joint_decisions"] for row in rows),
                "published_transitions": sum(row["published_transitions"] for row in rows),
                "initialization_joint_decisions": sum(row["initialization_joint_decisions"] for row in rows),
                "failed_dispatch_attempts": sum(row["failed_dispatch_attempts"] for row in rows),
                "additional_diagnostic_sources": "Benchmark/replay modules retain separate per-case consumed counts and coordinator ledgers in their reports; these remain validation-only too."}
