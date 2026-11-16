"""Shortened-pilot selections use tiny saved fixtures, never research training."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import phase24_freeze as cli
from tests import test_phase24_relocate as baseline
from world_models.data.replay import sha256_file
from world_models.training import runner, selection


class InitialSelectionTests(unittest.TestCase):
    setUp = baseline.RelocationTests.setUp
    load_case = staticmethod(baseline.RelocationTests.load_case)
    snapshot = baseline.RelocationTests.snapshot

    def freeze(self):
        with patch.object(runner, "load_case", side_effect=self.load_case):
            return selection.freeze_selection(self.source, self.output, reason="Submission deadline")

    def selection_path(self):
        return self.output / "selection.json"

    def rewrite_selection(self, mutate):
        path = self.selection_path()
        manifest = json.loads(path.read_text())
        mutate(manifest)
        baseline.json_write(path, manifest)
        path.with_suffix(".sha256").write_text(f"{sha256_file(path)}  {path.name}\n", encoding="ascii")

    def test_unequal_saved_counts_preserve_source_bytes_rng_and_paired_identity(self):
        snapshot = self.snapshot()
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        manifest = self.freeze()
        self.assertEqual(self.snapshot(), snapshot)
        self.assertEqual(random.getstate(), before[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertEqual(runner.source_identity(), self.source_hashes)
        self.assertEqual(selection.load_selection(self.selection_path()), manifest)
        self.assertEqual([row["model_updates"] for row in manifest["cases"]], self.counts)
        self.assertFalse(manifest["full_schedule_completion_claim"])
        self.assertFalse(manifest["P2.5_quality_acceptance"])
        for row, source in zip(manifest["cases"], self.source_checkpoints):
            self.assertEqual(row["original_target_stage_updates"], 5000)
            self.assertEqual(row["shared_initial_variants"], ["B", "D"])
            self.assertEqual(row["checkpoint_sha256"], sha256_file(source))
            self.assertFalse(Path(row["checkpoint_path"]).is_absolute())
            resolved = selection.resolve_selected_case(self.selection_path(), row["task"], row["seed"])
            self.assertEqual(Path(resolved["checkpoint_path"]), source)
            self.assertEqual(resolved["selection_sha256"], sha256_file(self.selection_path()))

    def test_latest_committed_corruption_rejects_instead_of_using_older_checkpoint(self):
        checkpoint = self.source_checkpoints[0]
        checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_uncommitted_newer_payload_and_temp_are_ignored_without_modification(self):
        directory = self.source_checkpoints[0].parent
        (directory / "update_00004999.pt").write_bytes(b"incomplete payload")
        (directory / "update_00005000.pt.tmp-writing").write_bytes(b"incomplete temp")
        before = self.snapshot()
        manifest = self.freeze()
        self.assertEqual(manifest["cases"][0]["model_updates"], self.counts[0])
        self.assertEqual(self.snapshot(), before)

    def test_active_launcher_and_each_case_lock_prevent_selection(self):
        for directory in (self.source, *(path.parent.parent for path in self.source_checkpoints)):
            with self.subTest(directory=directory), runner.run_lock(directory):
                with self.assertRaisesRegex(RuntimeError, "another worker"):
                    self.freeze()
                self.assertFalse(self.output.exists())

    def test_output_is_fresh_immutable_and_outside_source(self):
        self.freeze()
        before = self.selection_path().read_bytes()
        with self.assertRaises(FileExistsError):
            self.freeze()
        self.assertEqual(self.selection_path().read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "separate"):
            selection.freeze_selection(self.source, self.source / "selection", reason="deadline")

    def test_complete_initial_plan_and_all_four_cases_required(self):
        path = self.source / "launch_plan.json"
        plan = json.loads(path.read_text())
        plan["jobs"] = plan["jobs"][:3]
        baseline.json_write(path, plan)
        with self.assertRaisesRegex(ValueError, "complete four-case"):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_stale_source_identity_rejects_before_output(self):
        with patch.object(runner, "source_identity", return_value={"changed.py": "changed"}):
            with self.assertRaisesRegex(ValueError, "source or update mismatch"):
                self.freeze()
        self.assertFalse(self.output.exists())

    def test_current_data_provenance_must_match_original_training(self):
        def changed(task, seed, **kwargs):
            cfg, data, stats, provenance = self.load_case(task, seed)
            provenance["replays"][0]["manifest_sha256"] = "changed replay"
            return cfg, data, stats, provenance
        with patch.object(runner, "load_case", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "replay/normalization"):
                selection.freeze_selection(self.source, self.output, reason="deadline")
        self.assertFalse(self.output.exists())

    def test_checkpoint_metadata_must_match_original_manifest(self):
        baseline.rewrite_checkpoint(self.source_checkpoints[0],
            lambda payload: payload["metadata"].update(variant="B"))
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_selection_receipt_and_selected_checkpoint_tampering_reject(self):
        self.freeze()
        original = self.selection_path().read_bytes()
        self.selection_path().write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "receipt mismatch"):
            selection.load_selection(self.selection_path())
        self.selection_path().write_bytes(original)
        checkpoint = self.source_checkpoints[2]
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            selection.load_selection(self.selection_path())

    def test_changed_count_pairing_case_or_target_reject_even_with_rehashed_selection(self):
        self.freeze()
        original = deepcopy(json.loads(self.selection_path().read_text()))
        mutations = [lambda item: item["cases"][0].update(model_updates=5000),
                     lambda item: item["cases"][0].update(shared_initial_variants=["B"]),
                     lambda item: item["cases"][0].update(original_target_stage_updates=1000),
                     lambda item: item["cases"][0].update(checkpoint_path=item["cases"][1]["checkpoint_path"]),
                     lambda item: item.update(full_schedule_completion_claim=True)]
        for mutate in mutations:
            baseline.json_write(self.selection_path(), original)
            self.rewrite_selection(mutate)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                selection.load_selection(self.selection_path())

    def test_frozen_selection_does_not_follow_later_source_checkpoints(self):
        self.freeze()
        before = self.selection_path().read_bytes()
        old = self.source_checkpoints[0]
        newer = old.with_name("update_00002000.pt")
        payload = torch.load(old, map_location="cpu", weights_only=True)
        payload["update"] = 2000
        torch.save(payload, newer)
        newer.with_suffix(".sha256").write_text(f"{sha256_file(newer)}  {newer.name}\n", encoding="ascii")
        manifest = selection.load_selection(self.selection_path())
        self.assertEqual(manifest["cases"][0]["model_updates"], self.counts[0])
        self.assertEqual(self.selection_path().read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "not the frozen selection"):
            selection.resolve_selected_case(self.selection_path(), "relay", 0, checkpoint=newer)

    def test_selection_survives_copying_whole_project_tree(self):
        self.freeze()
        portable = self.root / "transferred"
        portable.mkdir()
        shutil.copytree(self.source, portable / self.source.name)
        shutil.copytree(self.output, portable / self.output.name)
        manifest = selection.load_selection(portable / self.output.name / "selection.json")
        self.assertEqual([row["model_updates"] for row in manifest["cases"]], self.counts)

    def test_original_config_hash_paths_are_portable_between_linux_and_windows(self):
        for checkpoint in self.source_checkpoints:
            path = checkpoint.parent.parent / "run_manifest.json"
            manifest = json.loads(path.read_text())
            original = manifest["resolved_config"]["source_hashes"]
            # Use the separator opposite to the current host; digest values
            # and every resolved configuration value remain unmodified.
            separator = "/" if any("\\" in key for key in original) else "\\"
            manifest["resolved_config"]["source_hashes"] = {
                key.replace("\\", "/").replace("/", separator): digest
                for key, digest in original.items()}
            baseline.json_write(path, manifest)
        before = self.snapshot()
        self.freeze()
        manifest = selection.load_selection(self.selection_path())
        self.assertEqual([row["model_updates"] for row in manifest["cases"]], self.counts)
        self.assertEqual(self.snapshot(), before)

    def test_config_path_normalization_preserves_hash_checks_and_rejects_collisions(self):
        path = self.source_checkpoints[0].parent.parent / "run_manifest.json"
        original = json.loads(path.read_text())
        for collision in (False, True):
            manifest = deepcopy(original)
            hashes = manifest["resolved_config"]["source_hashes"]
            key = next(key for key in hashes if key != "base.yaml")
            if collision:
                other = key.replace("\\", "/") if "\\" in key else key.replace("/", "\\")
                hashes[other] = hashes[key]
            else:
                hashes[key] = "0" * 64
            baseline.json_write(path, manifest)
            with self.subTest(collision=collision), self.assertRaises(ValueError):
                self.freeze()
            self.assertFalse(self.output.exists())

    def test_cli_creates_and_inspects_selection_without_running_fit(self):
        with patch.object(runner, "load_case", side_effect=self.load_case), \
             patch.object(runner, "fit_case") as fitting, redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(cli.main(["--run-root", str(self.source), "--output-dir", str(self.output),
                                       "--reason", "Time-limited pilot"]), 0)
            self.assertEqual(cli.main(["--inspect", str(self.selection_path())]), 0)
        fitting.assert_not_called()
        self.assertIn("1000 saved optimizer updates (original target 5000)", stdout.getvalue())
        self.assertIn('"full_schedule_completion_claim": false', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
