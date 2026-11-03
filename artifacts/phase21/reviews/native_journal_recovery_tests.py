"""Artifact-only synthetic recovery tests; no simulator/project imports or D edits."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import psutil

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r"D:\world models\artifacts\phase21\preflight_data\relay\seed_0")
spec = importlib.util.spec_from_file_location("native_journal_recovery_audit", ROOT / "native_journal_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def encoded(rows):
    return b"".join((json.dumps({**row, "sequence": index}, sort_keys=True) + "\n").encode()
                    for index, row in enumerate(rows))


def failed_environment(path, source, episode_id):
    path.mkdir(parents=True)
    journal = path / "env-synthetic" / "dispatch_ledger.jsonl"
    journal.parent.mkdir()
    rows = [{"kind": "dispatch", "timestamp_utc": "2026-11-02T00:00:00+00:00", "episode_id": 0,
             "decision_index": 0, "dispatch_id": 1, "consumed_joint_decisions": 0,
             "command": "start", "actions": None},
            {"kind": "infrastructure_error", "timestamp_utc": "2026-11-02T00:00:01+00:00", "episode_id": 0,
             "decision_index": 0, "dispatch_id": 1, "consumed_joint_decisions": 0,
             "error": {"message": "synthetic zero-physics start fixture", "consumed_joint_decisions": 0}}]
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    inventory = {"directories": ["env-synthetic"], "files": [{"path": "env-synthetic/dispatch_ledger.jsonl",
                 "bytes": journal.stat().st_size, "sha256": audit.sha256(journal)}]}
    receipt = {"status": "verified_closed_archive", "source_path_at_collection": str(source),
               **inventory, "content_sha256": audit.canonical_hash(inventory),
               "file_count": 1, "total_bytes": journal.stat().st_size}
    write_json(path / "archive_receipt.json", receipt)
    return {**receipt, "episode_id": episode_id, "category": "environments",
            "receipt_sha256": audit.sha256(path / "archive_receipt.json"), "source_removed": True}


def make_controlled_fixture(destination, *, retry_first=False):
    """Reuse real validation payloads solely inside a disposable audit fixture.

    Inject bookkeeping for an explicitly synthetic failed start. This tests the
    independent audit, not a claim that a new native failure/interaction occurred.
    """
    shutil.copytree(SOURCE, destination)
    report = audit.json_read(destination / "collection_report.json")
    manifest = audit.json_read(destination / "replay/manifest.json")
    original_ledger_rows = [json.loads(line) for line in (destination / "training_interactions.jsonl").read_text().splitlines()]
    prefix_episode, next_episode = manifest["episodes"][:2]
    prefix = prefix_episode["valid_length"]
    identifier = next_episode["episode_id"]
    recovery_id = "synthetic-controlled-resume"
    recovery_dir = destination / "recovery" / recovery_id
    before = recovery_dir / "before"
    before.mkdir(parents=True)
    successful = report["journal_storage"]["runtime_archives"]
    old_hot = destination / "fixture_original_hot_ledger.jsonl"
    new_hot = destination / "fixture_current_hot_ledger.jsonl"
    failed_root = destination / "fixture_historical_removed_runtime" / identifier
    failed_env = destination / "runtime/fixture_failed/environments" / identifier
    environment = failed_environment(failed_env, failed_root, identifier)
    environment["archive_path"] = failed_env.relative_to(destination).as_posix()
    source_scene = next(entry for entry in successful if entry["episode_id"] == identifier and entry["category"] == "scenarios")
    failed_scene = destination / "runtime/fixture_failed/scenarios" / identifier
    shutil.copytree(destination / source_scene["archive_path"], failed_scene)
    scene_receipt = audit.json_read(failed_scene / "archive_receipt.json")
    scene_receipt["source_path_at_collection"] = str(destination / "fixture_historical_removed_scenario" / identifier)
    write_json(failed_scene / "archive_receipt.json", scene_receipt)
    scene = {**source_scene, **scene_receipt, "archive_path": failed_scene.relative_to(destination).as_posix(),
             "receipt_sha256": audit.sha256(failed_scene / "archive_receipt.json"), "source_removed": True}
    failed = [scene, environment]
    cut = next(index + 1 for index, row in enumerate(original_ledger_rows)
               if row["event"] == "episode_committed" and row["episode_id"] == prefix_episode["episode_id"])
    prefix_rows = deepcopy(original_ledger_rows[:cut])
    prefix_rows.append({"event": "infrastructure_or_storage_error", "episode_id": identifier, "operation": "reset",
        "physical_delta": 0, "initialization_delta": 0, "published_delta": 0, "failed_delta": 0,
        "physical_consumed_at_source": 0, "initialization_consumed_at_source": 0,
        "coordinator_run_dir": str(failed_root / "env-synthetic"),
        "task_family": report["task_family"], "run_seed": report["run_seed"]})
    before_bytes = encoded(prefix_rows)
    old_hot.write_bytes(before_bytes)
    (before / "training_interactions.jsonl").write_bytes(before_bytes)
    partial = deepcopy(manifest)
    partial.update(status="collecting", active_episode=None, transition_count=prefix,
                   episodes=[deepcopy(prefix_episode)])
    partial.pop("sealed_at_utc", None)
    partial.pop("dataset_identity_sha256", None)
    write_json(before / "manifest.partial.json", partial)
    failure = {"status": "failed", "operation": "reset", "unreconciled": False,
               "episode_id": identifier, "interaction_accounting": {"ledger_sha256": audit.sha256(old_hot)},
               "journal_storage": {"ledger_hot_path_at_collection": str(old_hot),
                    "runtime_archives": [entry for entry in successful if entry["episode_id"] == prefix_episode["episode_id"]] + failed}}
    write_json(destination / "collection_error.json", failure)
    shutil.copyfile(destination / "collection_error.json", before / "collection_error.json")
    shutil.copyfile(destination / "collection_spec.json", before / "collection_spec.json")
    write_json(before / "collection_progress.json", {"published_joint_transitions": prefix, "completed_episodes": 1})
    snapshots = {path.name: {"path": path.relative_to(destination).as_posix(),
                  "bytes": path.stat().st_size, "sha256": audit.sha256(path)} for path in before.iterdir()}
    prefix_files = [{"path": "replay/" + prefix_episode["manifest_path"], "sha256": prefix_episode["manifest_sha256"]}]
    prefix_files.extend({"path": "replay/" + chunk["path"], "sha256": chunk["sha256"]} for chunk in prefix_episode["chunks"])
    lineage = {"recovery_id": recovery_id, "prefix_joint_transitions": prefix, "prefix_episodes": 1,
               "maximum_fresh_startup_attempts": 2, "step_failures_retried": False,
               "new_random_or_scripted_interactions_discarded": 0, "snapshots": snapshots,
               "original_metadata_sha256": audit.canonical_hash(manifest["metadata"]),
               "original_prefix_files": prefix_files, "original_failure_report_path": "collection_error.json",
               "original_failure_sha256": audit.sha256(destination / "collection_error.json")}
    write_json(recovery_dir / "lineage.json", lineage)
    summary = {"path": (recovery_dir / "lineage.json").relative_to(destination).as_posix(),
               "sha256": audit.sha256(recovery_dir / "lineage.json"), "recovery_id": recovery_id,
               "prefix_joint_transitions": prefix, "prefix_episodes": 1}
    rows = prefix_rows + [{"event": "intentional_resume", "lineage": {key: summary[key] for key in ("path", "sha256", "recovery_id")},
                         "prefix_joint_transitions": prefix, "original_ledger_sha256": audit.sha256(old_hot),
                         "physical_delta": 0, "initialization_delta": 0, "published_delta": 0, "failed_delta": 0}]
    for episode in manifest["episodes"][1:]:
        attempt = 2 if retry_first and episode["episode_id"] == identifier else 1
        if attempt == 2:
            new_failed_root = destination / "fixture_historical_removed_retry" / identifier
            new_failed_env = destination / f"runtime/recovery/{recovery_id}/attempt-001/environments" / identifier
            new_environment = failed_environment(new_failed_env, new_failed_root, identifier)
            new_environment.update(archive_path=new_failed_env.relative_to(destination).as_posix(),
                                   recovery_id=recovery_id, startup_attempt=1)
            new_failed_scene = destination / f"runtime/recovery/{recovery_id}/attempt-001/scenarios" / identifier
            shutil.copytree(failed_scene, new_failed_scene)
            new_scene_receipt = audit.json_read(new_failed_scene / "archive_receipt.json")
            new_scene_receipt["source_path_at_collection"] = str(destination / "fixture_historical_removed_retry_scene" / identifier)
            write_json(new_failed_scene / "archive_receipt.json", new_scene_receipt)
            new_scene = {**scene, **new_scene_receipt, "archive_path": new_failed_scene.relative_to(destination).as_posix(),
                         "receipt_sha256": audit.sha256(new_failed_scene / "archive_receipt.json"),
                         "recovery_id": recovery_id, "startup_attempt": 1}
            failed.extend([new_scene, new_environment])
            rows.append({"event": "startup_attempt", "episode_id": identifier, "recovery_id": recovery_id,
                         "startup_attempt": 1, **{key: episode[key] for key in ("environment_seed", "layout_seed", "collector_seed")},
                         "template_index": int(episode["layout_id"].rsplit("_", 1)[-1]),
                         "physical_delta": 0, "initialization_delta": 0, "published_delta": 0, "failed_delta": 0})
            rows.append({"event": "infrastructure_or_storage_error", "episode_id": identifier, "operation": "reset",
                         "coordinator_run_dir": str(new_failed_root / "env-synthetic"),
                         "physical_delta": 0, "initialization_delta": 0, "published_delta": 0, "failed_delta": 0,
                         "physical_consumed_at_source": 0, "initialization_consumed_at_source": 0})
        episode.update(recovery_id=recovery_id, startup_attempt=attempt)
        episode_path = destination / "replay" / episode["manifest_path"]
        stored_episode = audit.json_read(episode_path)
        stored_episode.update(recovery_id=recovery_id, startup_attempt=attempt)
        write_json(episode_path, stored_episode)
        episode["manifest_sha256"] = audit.sha256(episode_path)
        rows.append({"event": "startup_attempt", "episode_id": episode["episode_id"], "recovery_id": recovery_id,
                     "startup_attempt": attempt, **{key: episode[key] for key in ("environment_seed", "layout_seed", "collector_seed")},
                     "template_index": int(episode["layout_id"].rsplit("_", 1)[-1]),
                     "physical_delta": 0, "initialization_delta": 0, "published_delta": 0, "failed_delta": 0})
        rows.extend(deepcopy(row) for row in original_ledger_rows[cut:] if row.get("episode_id") == episode["episode_id"])
        for entry in successful:
            if entry["episode_id"] == episode["episode_id"]:
                old_path = entry["archive_path"]
                new_path = f"runtime/recovery/{recovery_id}/attempt-{attempt:03d}/{entry['category']}/{episode['episode_id']}"
                shutil.copytree(destination / old_path, destination / new_path)
                entry.update(archive_path=new_path, recovery_id=recovery_id, startup_attempt=attempt)
    (destination / "training_interactions.jsonl").write_bytes(encoded(rows))
    new_hot.write_bytes((destination / "training_interactions.jsonl").read_bytes())
    manifest["dataset_identity_sha256"] = audit.canonical_hash({key: value for key, value in manifest.items()
                                                   if key not in ("dataset_identity_sha256", "sealed_at_utc")})
    write_json(destination / "replay/manifest.json", manifest)
    (destination / "replay/manifest.sha256").write_text(audit.sha256(destination / "replay/manifest.json") + "  manifest.json\n")
    report.update(dataset_identity_sha256=manifest["dataset_identity_sha256"],
                  dataset_manifest_sha256=audit.sha256(destination / "replay/manifest.json"),
                  recovery_history=[summary], recovery_lineage=summary, recorded_zero_physics_startup_failures=len(failed)//2)
    report["interaction_accounting"]["ledger_sha256"] = audit.sha256(new_hot)
    storage = report["journal_storage"]
    storage.update(ledger_hot_path_at_collection=str(new_hot), failed_runtime_archives=failed,
        retained_prior_hot_ledger_copies=[{"source_path_at_collection": str(old_hot),
            "matching_archive_path": snapshots["training_interactions.jsonl"]["path"],
            "bytes": old_hot.stat().st_size, "sha256": audit.sha256(old_hot)}])
    storage["ledger_archive"].update(sha256=audit.sha256(new_hot), bytes=new_hot.stat().st_size)
    for variant in ("B", "D"):
        path = destination / report["bindings"][variant]["path"]
        binding = audit.json_read(path)
        binding.update(dataset_identity_sha256=manifest["dataset_identity_sha256"], manifest_sha256=report["dataset_manifest_sha256"])
        write_json(path, binding)
        report["bindings"][variant]["sha256"] = audit.sha256(path)
    write_json(destination / "collection_report.json", report)
    return report


def rewrite_ledger(root, edit):
    path = root / "training_interactions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    edit(rows)
    path.write_bytes(encoded(rows))
    report = audit.json_read(root / "collection_report.json")
    hot = Path(report["journal_storage"]["ledger_hot_path_at_collection"])
    hot.write_bytes(path.read_bytes())
    report["interaction_accounting"]["ledger_sha256"] = audit.sha256(path)
    report["journal_storage"]["ledger_archive"]["sha256"] = audit.sha256(path)
    write_json(root / "collection_report.json", report)


class RecoveryAuditTests(unittest.TestCase):
    def fixture(self, *, retry_first=False):
        temporary = tempfile.TemporaryDirectory(prefix="native-recovery-audit-fixture-", dir=ROOT)
        directory = Path(temporary.name).resolve()
        self.assertTrue(directory.is_relative_to(ROOT.resolve()))
        self.addCleanup(temporary.cleanup)
        target = directory / "case"
        make_controlled_fixture(target, retry_first=retry_first)
        return target

    def test_full_native_payload_reconciles_with_explicit_synthetic_zero_startup(self):
        root = self.fixture()
        result = audit.audit_dataset(root, audit.Checks())
        self.assertEqual(result["published"], 1000)
        self.assertEqual(result["zero_physics_startup_failures"], 1)
        self.assertEqual(result["failed_startup_control_dispatches"], 1)
        self.assertEqual(result["recovery_audit"]["explicit_continuations"], 1)
        self.assertEqual(result["recovery_audit"]["startup_intents"], result["episodes"] - 1)
        self.assertEqual(result["archives_verified"], 2 * result["episodes"] + 2)

    def test_recovery_marker_cannot_reference_another_lineage_hash(self):
        root = self.fixture()
        def change(rows):
            next(row for row in rows if row["event"] == "intentional_resume")["lineage"]["sha256"] = "0" * 64
        rewrite_ledger(root, change)
        with self.assertRaisesRegex(ValueError, "lineage/ledger"):
            audit.audit_dataset(root, audit.Checks())

    def test_bounded_second_start_requires_both_verified_failure_archives_and_intents(self):
        root = self.fixture(retry_first=True)
        result = audit.audit_dataset(root, audit.Checks())
        self.assertEqual(result["published"], 1000)
        self.assertEqual(result["zero_physics_startup_failures"], 2)
        self.assertEqual(result["failed_startup_control_dispatches"], 2)
        self.assertEqual(result["recovery_audit"]["startup_intents"], result["episodes"])
        self.assertEqual(result["archives_verified"], 2 * result["episodes"] + 4)

    def test_startup_intent_cannot_change_a_seed(self):
        root = self.fixture()
        def change(rows):
            next(row for row in rows if row["event"] == "startup_attempt")["environment_seed"] += 1
        rewrite_ledger(root, change)
        with self.assertRaisesRegex(ValueError, "seed or template"):
            audit.audit_dataset(root, audit.Checks())

    def test_an_undeclared_zero_debit_error_is_still_rejected(self):
        root = self.fixture()
        def change(rows):
            index = next(index for index, row in enumerate(rows) if row["event"] == "startup_attempt")
            rows.insert(index, {"event": "infrastructure_or_storage_error", "operation": "reset",
                "episode_id": rows[index]["episode_id"], "coordinator_run_dir": "undeclared-native-process"})
        rewrite_ledger(root, change)
        with self.assertRaisesRegex(ValueError, "undeclared or duplicated"):
            audit.audit_dataset(root, audit.Checks())

    def test_original_hot_prefix_must_match_its_own_snapshot_not_final_ledger(self):
        root = self.fixture()
        report = audit.json_read(root / "collection_report.json")
        old_hot = Path(report["journal_storage"]["retained_prior_hot_ledger_copies"][0]["source_path_at_collection"])
        old_hot.write_bytes((root / "training_interactions.jsonl").read_bytes())
        with self.assertRaisesRegex(ValueError, "original hot ledger"):
            audit.audit_dataset(root, audit.Checks())

    def test_original_prefix_payload_is_hash_protected(self):
        root = self.fixture()
        manifest = audit.json_read(root / "replay/manifest.json")
        chunk = root / "replay" / manifest["episodes"][0]["chunks"][0]["path"]
        with chunk.open("ab") as stream: stream.write(b"changed-prefix")
        with self.assertRaisesRegex(ValueError, "original replay prefix payload"):
            audit.audit_dataset(root, audit.Checks())

    def test_zero_startup_validator_rejects_ack_or_action_even_with_zero_debits(self):
        for invalid in ({"kind": "acknowledged"}, {"kind": "dispatch", "command": "step", "actions": [0, 0]}):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory(prefix="native-startup-proof-", dir=ROOT) as directory:
                path = Path(directory) / "journal.jsonl"
                path.write_text(json.dumps({"timestamp_utc": "2026-11-02T00:00:00+00:00", "episode_id": 0,
                    "decision_index": 0, "dispatch_id": 1, "consumed_joint_decisions": 0, **invalid}) + "\n")
                with self.assertRaises(ValueError):
                    audit.zero_physics_startup(path, audit.Checks())


if __name__ == "__main__":
    process = psutil.Process()
    prior = process.cpu_affinity()
    if not {0, 1, 2, 3} <= set(prior):
        raise RuntimeError("CPU0..3 allocation unavailable")
    process.cpu_affinity([0, 1, 2, 3])
    stream = io.StringIO()
    try:
        results = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryAuditTests))
    finally:
        process.cpu_affinity(prior)
    report = {"status": "passed" if results.wasSuccessful() else "failed", "tests_run": results.testsRun,
              "failures": len(results.failures), "errors": len(results.errors), "output": stream.getvalue(),
              "helper_sha256": audit.sha256(ROOT / "native_journal_audit.py"),
              "test_helper_sha256": audit.sha256(__file__), "source_preflight_report_sha256": audit.sha256(SOURCE / "collection_report.json"),
              "scope": "Copied real preflight payloads plus explicitly synthetic recovery bookkeeping; no simulator interactions or D: dataset/source writes.",
              "cpu_ids": [0, 1, 2, 3], "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(ROOT / "native_journal_recovery_tests.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "output"}))
    if not results.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)
