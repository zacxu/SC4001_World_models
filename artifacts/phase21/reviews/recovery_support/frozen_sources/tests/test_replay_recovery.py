"""Explicit verified continuation preserves prefix, seeds, and physical debits."""

from __future__ import annotations

import json
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from world_models.data.collection import collect_initial_dataset, verify_initial_collection
from world_models.data.ledger import TrainingInteractionLedger, file_sha256
from world_models.data.recovery import (
    _remaining_segments, inspect_recoverable_prefix, resume_initial_dataset, main,
)
from world_models.data.replay import ReplayDataset
from world_models.env.coop import TwoPlayerCoopEnv
from world_models.env.coordinator import CoordinatorError
from tests.test_replay_collection import (
    CountingCoordinator, spooled_environment_fixture, spooled_scenario_fixture,
)


def make_prefix(output, total=100):
    calls = []

    def factory(family, **kwargs):
        env = spooled_environment_fixture(family, **kwargs)
        calls.append(env)
        if len(calls) == 2:
            def failed_reset(**kwargs):
                raise CoordinatorError("injected startup EOF before any physics", {"physical": 0})
            env.reset = failed_reset
        return env

    try:
        collect_initial_dataset("relay", 0, output, total_transitions=total, chunk_transitions=3,
            journal_root=output.parent / "original-hot", env_factory=factory,
            scenario_builder=spooled_scenario_fixture)
    except CoordinatorError:
        pass
    else:
        raise AssertionError("fixture did not produce its zero-physics startup failure")
    return file_sha256(output / "collection_error.json")


class ReplayRecoveryTests(unittest.TestCase):
    def recover(self, output, expected, **kwargs):
        arguments = {"journal_root": output.parent / "continued-hot", "recovery_id": "resume-001",
            "expected_failure_sha256": expected, "env_factory": spooled_environment_fixture,
            "scenario_builder": spooled_scenario_fixture}
        arguments.update(kwargs)
        return resume_initial_dataset(output, **arguments)

    def test_remaining_32700_schedule_and_quotas_match_original(self):
        remaining, counts = _remaining_segments(50000, .3, 1000, 32700)
        self.assertEqual(counts, {"random": 23100, "scripted": 9600})
        self.assertEqual(remaining[0], (65, "scripted", 300))
        self.assertEqual(sum(n for _, kind, n in remaining if kind == "random"), 11900)
        self.assertEqual(sum(n for _, kind, n in remaining if kind == "scripted"), 5400)

    def test_explicit_continuation_preserves_all_prefix_bytes_and_original_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "initial"
            expected = make_prefix(output)
            before_error = (output / "collection_error.json").read_bytes()
            before_ledger = (output / "training_interactions.jsonl").read_bytes()
            before_manifest = (output / "replay/manifest.partial.json").read_bytes()
            original = json.loads(before_manifest)
            prefix_files = {p: p.read_bytes() for p in (output / "replay/episodes").rglob("*") if p.is_file()}
            reports = []
            result = self.recover(output, expected, progress_callback=reports.append)
            self.assertEqual(result["published_joint_transitions"], 100)
            self.assertEqual(result["collector_counts"], {"random": 70, "scripted": 30})
            self.assertEqual(result["interaction_accounting"]["failed_dispatch_attempts"], 0)
            self.assertEqual(result["interaction_accounting"]["published_real_simulator_tics_total"], 400)
            self.assertEqual(result["recovery_id"], "resume-001")
            self.assertAlmostEqual(result["total_collection_attempt_elapsed_seconds"],
                result["original_failed_collection_attempt_elapsed_seconds"] + result["continuation_elapsed_seconds"])
            self.assertAlmostEqual(result["elapsed_seconds"], result["total_collection_attempt_elapsed_seconds"])
            self.assertGreaterEqual(result["original_failed_collection_attempt_elapsed_seconds"],
                                    result["original_prefix_elapsed_seconds"])
            self.assertTrue(any(r["status"] == "recovering" and r["published_joint_transitions"] == 4 for r in reports))
            self.assertEqual((output / "collection_error.json").read_bytes(), before_error)
            self.assertTrue((output / "training_interactions.jsonl").read_bytes().startswith(before_ledger))
            self.assertEqual((output / "recovery/resume-001/before/manifest.partial.json").read_bytes(), before_manifest)
            self.assertEqual((output / "recovery/resume-001/before/collection_error.json").read_bytes(), before_error)
            self.assertEqual((output / "recovery/resume-001/before/training_interactions.jsonl").read_bytes(), before_ledger)
            for path, data in prefix_files.items():
                self.assertEqual(path.read_bytes(), data)
            reader = ReplayDataset(output / "replay")
            self.assertEqual(reader.manifest["metadata"], original["metadata"])
            self.assertEqual(reader.episodes[1]["episode_id"], "relay-s0-e000001")
            self.assertEqual(len(result["journal_storage"]["failed_runtime_archives"]), 2)
            self.assertEqual(len(result["journal_storage"]["runtime_archives"]), 2 * result["completed_episodes"])
            self.assertEqual(verify_initial_collection(output)["joint_transitions"], 100)

    def test_resumed_arrays_and_seed_sequence_equal_uninterrupted_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "interrupted"
            expected = make_prefix(output)
            self.recover(output, expected)
            collect_initial_dataset("relay", 0, root / "uninterrupted", total_transitions=100,
                chunk_transitions=3, journal_root=root / "baseline-hot",
                env_factory=spooled_environment_fixture, scenario_builder=spooled_scenario_fixture)
            resumed = ReplayDataset(output / "replay")
            baseline = ReplayDataset(root / "uninterrupted/replay")
            self.assertEqual(len(resumed.episodes), len(baseline.episodes))
            for a, b in zip(resumed.episodes, baseline.episodes):
                for key in ("episode_id", "environment_seed", "layout_seed", "collector_seed", "partition",
                            "collector_kind", "policy_version", "layout_id", "mixture_segment", "valid_length"):
                    self.assertEqual(a[key], b[key])
                left = resumed.load_episode(a["episode_id"]).arrays()
                right = baseline.load_episode(b["episode_id"]).arrays()
                for name in left:
                    np.testing.assert_array_equal(left[name], right[name])

    def test_one_zero_physics_startup_retry_keeps_seed_and_records_both_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "initial"
            expected = make_prefix(output)
            observed = []

            def factory(family, **kwargs):
                env = spooled_environment_fixture(family, **kwargs)
                observed.append(kwargs)
                if len(observed) == 1:
                    def failed_reset(**kwargs):
                        raise CoordinatorError("second independent zero-physics startup failure", {})
                    env.reset = failed_reset
                return env

            result = self.recover(output, expected, env_factory=factory)
            self.assertEqual(observed[0]["seed"], observed[1]["seed"])
            reader = ReplayDataset(output / "replay")
            self.assertEqual(reader.episodes[1]["startup_attempt"], 2)
            self.assertEqual(len(result["journal_storage"]["failed_runtime_archives"]), 4)
            rows = [json.loads(line) for line in (output / "training_interactions.jsonl").read_text().splitlines()]
            attempts = [r for r in rows if r["event"] == "startup_attempt" and r["episode_id"] == "relay-s0-e000001"]
            self.assertEqual([r["startup_attempt"] for r in attempts], [1, 2])
            self.assertEqual(attempts[0]["environment_seed"], attempts[1]["environment_seed"])
            self.assertEqual(result["interaction_accounting"]["failed_dispatch_attempts"], 0)

    def test_two_zero_physics_startup_failures_stop_without_third_attempt_or_data_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "initial"
            expected = make_prefix(output)
            old_error = (output / "collection_error.json").read_bytes()
            calls = []

            def factory(family, **kwargs):
                env = spooled_environment_fixture(family, **kwargs)
                calls.append(env)
                def failed_reset(**kwargs):
                    raise CoordinatorError("persistent zero-physics startup crash", {})
                env.reset = failed_reset
                return env

            with self.assertRaises(CoordinatorError):
                self.recover(output, expected, env_factory=factory)
            self.assertEqual(len(calls), 2)
            self.assertEqual((output / "collection_error.json").read_bytes(), old_error)
            failure = json.loads((output / "recovery/resume-001/continuation_error.json").read_text())
            self.assertEqual(failure["published_joint_transitions"], 4)
            self.assertEqual(failure["interaction_accounting"]["failed_dispatch_attempts"], 0)
            self.assertEqual(len(failure["journal_storage"]["failed_runtime_archives"]), 6)
            self.assertFalse((output / "replay/manifest.json").exists())

    def test_physics_failure_has_no_retry_and_keeps_failed_dispatch_charged(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "initial"
            expected = make_prefix(output)
            calls = []

            def factory(family, **kwargs):
                env = spooled_environment_fixture(family, **kwargs)
                env.coordinator.failure_at = 2
                calls.append(env)
                return env

            with self.assertRaises(CoordinatorError):
                self.recover(output, expected, env_factory=factory)
            self.assertEqual(len(calls), 1)
            failure = json.loads((output / "recovery/resume-001/continuation_error.json").read_text())
            self.assertEqual(failure["operation"], "step")
            self.assertEqual(failure["interaction_accounting"]["published_real_joint_transitions"], 5)
            self.assertEqual(failure["interaction_accounting"]["replay_committed_joint_transitions"], 4)
            self.assertEqual(failure["interaction_accounting"]["failed_dispatch_attempts"], 1)
            self.assertIsNotNone(json.loads((output / "replay/manifest.partial.json").read_text())["active_episode"])

    def test_wrong_failure_hash_or_corrupt_prefix_is_rejected_before_new_append(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "initial"
            expected = make_prefix(output)
            original_ledger = (output / "training_interactions.jsonl").read_bytes()
            with self.assertRaises(ValueError):
                self.recover(output, "0" * 64)
            self.assertFalse((output / "recovery").exists())
            chunk = next((output / "replay/episodes").rglob("*.npz"))
            with chunk.open("ab") as stream:
                stream.write(b"corrupted")
            with self.assertRaises(ValueError):
                self.recover(output, expected)
            self.assertFalse((output / "recovery").exists())
            self.assertEqual((output / "training_interactions.jsonl").read_bytes(), original_ledger)

    def test_ledger_resume_rejects_unaccounted_source_counter_jump(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "initial"
            make_prefix(output)
            source = output / "training_interactions.jsonl"
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            for row in rows:
                if row["event"] == "transition":
                    row["physical_consumed_at_source"] += 100
                    break
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaises(ValueError):
                TrainingInteractionLedger.resume_from_verified(Path(directory) / "new-hot",
                    source_path=source, expected_sha256=file_sha256(source), task_family="relay", run_seed=0,
                    expected_committed=4, learning_eligible=False, lineage={})
            self.assertFalse((Path(directory) / "new-hot").exists())


class RecoveryCliTests(unittest.TestCase):
    def arguments(self, root, extra=()):
        return ["--dataset", str(root / "dataset"), "--recovery-id", "explicit-cli",
                "--failure-sha256", "0" * 64, "--output", str(root / "cli-report.json"), *extra]

    def test_invalid_startup_attempt_count_never_calls_collection(self):
        with tempfile.TemporaryDirectory() as directory, patch("world_models.data.recovery.resume_initial_dataset") as resume:
            root = Path(directory)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(self.arguments(root, ("--max-startup-attempts", "3")))
            self.assertEqual(error.exception.code, 2)
            resume.assert_not_called()
            self.assertFalse((root / "cli-report.json").exists())

    def test_native_resume_requires_explicit_fresh_journal_path(self):
        with tempfile.TemporaryDirectory() as directory, patch("world_models.data.recovery.resume_initial_dataset") as resume:
            root = Path(directory)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(self.arguments(root))
            self.assertEqual(error.exception.code, 2)
            resume.assert_not_called()

    def test_cli_report_cannot_overwrite_original_dataset_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            original = root / "dataset/collection_error.json"
            original.write_bytes(b"immutable original startup failure")
            arguments = self.arguments(root, ("--inspect-only", "--output", str(original)))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(arguments)
            self.assertEqual(original.read_bytes(), b"immutable original startup failure")

    def test_inspect_only_has_no_native_resume_and_writes_readonly_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = {"count": 4, "episodes": [{}], "expected_next_episode_id": "relay-s0-e000001",
                "remaining_segments": [(0, "random", 66), (1, "scripted", 30)],
                "native_zero_physics_failure_checked": True, "original_generation_proof": {}}
            resources = MagicMock()
            resources.__enter__.return_value = resources
            resources.report.return_value = {"scope": "continuation only"}
            with patch("world_models.data.recovery.inspect_recoverable_prefix", return_value=prefix) as inspect_prefix, \
                    patch("world_models.data.recovery.resume_initial_dataset") as resume, \
                    patch("world_models.data.resources.CollectionResources", return_value=resources), \
                    patch("world_models.env.acceptance_metrics.cpu_budget", return_value=nullcontext({"cpu_ids": [0,1,2,3]})), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(main(self.arguments(root, ("--inspect-only",))), 0)
            resume.assert_not_called()
            inspect_prefix.assert_called_once()
            report = json.loads((root / "cli-report.json").read_text())
            self.assertEqual(report["status"], "passed_inspection")
            self.assertEqual(report["prefix"]["remaining_transitions"], 96)
            self.assertFalse(report["source_overlays_or_dependency_upgrades"])

    def test_failed_inspection_never_starts_native_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resources = MagicMock()
            resources.__enter__.return_value = resources
            resources.report.return_value = {"scope": "continuation only"}
            with patch("world_models.data.recovery.inspect_recoverable_prefix", side_effect=ValueError("invalid immutable prefix")), \
                    patch("world_models.data.recovery.resume_initial_dataset") as resume, \
                    patch("world_models.data.resources.CollectionResources", return_value=resources), \
                    patch("world_models.env.acceptance_metrics.cpu_budget", return_value=nullcontext({"cpu_ids": [0,1,2,3]})), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(main(self.arguments(root, ("--inspect-only",))), 1)
            resume.assert_not_called()
            report = json.loads((root / "cli-report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("invalid immutable prefix", report["error"])


if __name__ == "__main__":
    unittest.main()
