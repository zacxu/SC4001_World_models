"""Durable accounting for real replay collection, separate from model trajectories.

Published random/script decisions are the research joint-transition unit.
Reset synchronization advances are separately reported physical simulator work.
No coordinator interaction can disappear if replay writing fails.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any


LEDGER_VERSION = "real-replay-ledger-v1"


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TrainingInteractionLedger:
    """One append-only ledger per initial family/seed dataset.

    A valid transition is debited before storage append. Thus a storage failure
    cannot make already consumed real experience look free. Incomplete datasets
    are never sealed or silently retried by this collector.
    """

    def __init__(self, directory: str | Path, *, task_family: str, run_seed: int, learning_eligible: bool = True):
        if type(learning_eligible) is not bool:
            raise ValueError("learning_eligible must be boolean")
        self.learning_eligible = learning_eligible
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "training_interactions.jsonl"
        if self.path.exists():
            raise FileExistsError("refuse to overwrite or forget an existing training ledger")
        self.identity = {"task_family": task_family, "run_seed": run_seed}
        self._physical = 0
        self._initialization = 0
        self._published = 0
        self._published_tics = 0
        self._failed = 0
        self._committed = 0
        self._kinds = {"random": 0, "scripted": 0}
        self._seen: dict[str, tuple[int, int]] = {}
        self._sequence = 0
        self._append("created", learning_eligible=learning_eligible, data_source="real_environment",
                     purpose="initial_learning_replay" if learning_eligible else "collection_preflight")

    def _append(self, event: str, **fields: Any) -> dict:
        record = {
            "schema_version": LEDGER_VERSION, "sequence": self._sequence,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **self.identity, "event": event, **fields,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._sequence += 1
        return record

    @staticmethod
    def _snapshot(env) -> tuple[str, int, int]:
        key = str(Path(env.coordinator.run_dir).resolve())
        physical = env.coordinator.consumed_joint_decisions
        initialization = env.initialization_joint_decisions
        if any(type(value) is not int or value < 0 for value in (physical, initialization)):
            raise ValueError("invalid real-interaction accounting counters")
        return key, physical, initialization

    def _deltas(self, env) -> tuple[str, int, int, int, int]:
        key, physical, initialization = self._snapshot(env)
        previous, previous_init = self._seen.get(key, (0, 0))
        if physical < previous or initialization < previous_init:
            raise ValueError("real-interaction counters regressed")
        return key, physical, initialization, physical - previous, initialization - previous_init

    def record_reset(self, env, episode_id: str) -> None:
        key, physical, initialization, delta, init_delta = self._deltas(env)
        if delta != init_delta:
            raise ValueError("unaccounted coordinator interaction during reset")
        self._append("reset", episode_id=episode_id, coordinator_run_dir=key,
                     physical_delta=delta, initialization_delta=init_delta,
                     published_delta=0, failed_delta=0,
                     physical_consumed_at_source=physical,
                     initialization_consumed_at_source=initialization)
        self._seen[key] = physical, initialization
        self._physical += delta
        self._initialization += init_delta

    def record_transition(self, env, episode_id: str, collector_kind: str, step) -> None:
        if collector_kind not in self._kinds:
            raise ValueError("only real random/scripted collection is allowed")
        key, physical, initialization, delta, init_delta = self._deltas(env)
        if delta != 1 or init_delta:
            raise ValueError("one published transition must consume exactly one joint decision")
        self._append("transition", episode_id=episode_id, coordinator_run_dir=key,
                     collector_kind=collector_kind,
                     decision_index=step.metadata.decision_index,
                     start_tic=step.metadata.start_tic, end_tic=step.metadata.end_tic,
                     actual_tics=step.metadata.actual_simulator_tics,
                     physical_delta=1, initialization_delta=0,
                     published_delta=1, failed_delta=0,
                     physical_consumed_at_source=physical,
                     initialization_consumed_at_source=initialization,
                     replay_status="pending_episode_commit")
        self._seen[key] = physical, initialization
        self._physical += 1
        self._published += 1
        self._published_tics += step.metadata.actual_simulator_tics
        self._kinds[collector_kind] += 1

    def commit_episode(self, episode_id: str, valid_length: int) -> None:
        if type(valid_length) is not int or valid_length <= 0:
            raise ValueError("committed episode must contain real transitions")
        if self._committed + valid_length > self._published:
            raise ValueError("cannot commit transitions that have not been charged")
        self._append("episode_committed", episode_id=episode_id, valid_length=valid_length)
        self._committed += valid_length

    def record_failure(self, env, episode_id: str, operation: str, error: BaseException) -> None:
        """Retain all extra dispatches; never manufacture a task reward."""
        fields: dict[str, Any] = {
            "episode_id": episode_id, "operation": operation,
            "error_type": type(error).__name__, "error": str(error),
            "physical_delta": 0, "initialization_delta": 0,
            "published_delta": 0, "failed_delta": 0,
        }
        if env is not None:
            key, physical, initialization, delta, init_delta = self._deltas(env)
            # A reset dispatch can fail after incrementing the initialization
            # attempt counter. Count actually consumed dispatches once only.
            actual_initialization = min(delta, init_delta)
            failed = delta - actual_initialization
            fields.update(coordinator_run_dir=key, physical_delta=delta,
                          initialization_delta=actual_initialization,
                          failed_delta=failed,
                          failed_initialization_dispatches=actual_initialization if operation == "reset" else 0,
                          physical_consumed_at_source=physical,
                          initialization_consumed_at_source=initialization,
                          initialization_attempts_at_source=init_delta)
            self._seen[key] = physical, initialization
            self._physical += delta
            self._initialization += actual_initialization
            self._failed += failed
        self._append("infrastructure_or_storage_error", **fields)

    def archive_to(self, destination: str | Path, *, spool_root: str | Path) -> dict:
        """Durably copy a closed ledger snapshot; retain the authoritative hot file.

        If final-report publication fails after an earlier snapshot, the error
        row is appended to the same hot ledger and a newer complete snapshot
        atomically replaces the cold copy. No earlier interaction is omitted.
        """
        from world_models.data.spool import archive_closed_file
        destination = Path(destination).resolve()
        source = self.path.resolve()
        if source == destination:
            return {"status": "already_at_archive", "path": destination.name,
                    "sha256": file_sha256(source), "bytes": source.stat().st_size,
                    "source_path_at_collection": str(source), "source_removed": False}
        staging = destination.with_name(destination.name + ".archive-" + uuid.uuid4().hex)
        receipt = archive_closed_file(source, staging, spool_root=spool_root, remove_source=False)
        os.replace(staging, destination)
        if file_sha256(destination) != receipt["sha256"] or destination.stat().st_size != receipt["bytes"]:
            raise RuntimeError("final ledger archive differs from its verified durable snapshot")
        return {**receipt, "path": destination.name, "archive_path": str(destination)}

    def manifest(self) -> dict:
        if self._physical != self._published + self._initialization + self._failed:
            raise ValueError("physical simulator accounting does not reconcile")
        return {
            "schema_version": LEDGER_VERSION, **self.identity,
            "learning_eligible": self.learning_eligible,
            "purpose": "initial_learning_replay" if self.learning_eligible else "collection_preflight",
            "ledger_path": "training_interactions.jsonl",
            "ledger_sha256": file_sha256(self.path),
            "real_joint_transitions_total": self._published + self._failed,
            "published_real_joint_transitions": self._published,
            "published_real_simulator_tics_total": self._published_tics,
            "replay_committed_joint_transitions": self._committed,
            "initialization_joint_decisions": self._initialization,
            "failed_dispatch_attempts": self._failed,
            "physical_joint_decisions_including_initialization": self._physical,
            "collector_counts": dict(self._kinds),
            "accounting_rule": (
                "Each published scripted/random joint decision is charged once; "
                "failed non-initialization dispatches remain charged. Reset "
                "synchronization dispatches are retained as separately reported "
                "physical simulator overhead, not fabricated replay transitions."
            ),
            "validation_data_imported": False,
            "synthetic_data_imported": False,
        }
