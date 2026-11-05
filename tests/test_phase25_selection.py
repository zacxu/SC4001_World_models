"""Lazy held-out P2.5 cohorts, deterministic prefixes, and rare-event support."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_model_sampling import identifier, make_store
from world_models.data.replay import EpisodeView, ReplayDataset, canonical_hash, read_npz, sha256_file, write_npz
from world_models.data.schema import field_manifest
from world_models.evaluation.selection import select_diagnostic_cohort
from world_models.training.sampling import ReplaySampler


class DiagnosticSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.fit_id = identifier("selection-fit", "fit")
        cls.diagnostic_ids = [identifier(f"selection-diagnostic-{index}", "diagnostic") for index in range(16)]
        cls.dataset = make_store(Path(cls.temporary.name)/"replay",
            [(cls.fit_id, 5, False), *((name, 20, True) for name in cls.diagnostic_ids)], chunk_transitions=4)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_bounded_strata_are_lazy_and_four_eight_windows_are_exact_prefixes(self):
        with patch.object(ReplaySampler, "window", side_effect=AssertionError("eager full-prefix RGB load")), \
             patch.object(EpisodeView, "arrays", side_effect=AssertionError("whole episode materialization")):
            full = select_diagnostic_cohort([self.dataset], seed=27)
            four = select_diagnostic_cohort([self.dataset], seed=27, max_windows=4)
            eight = select_diagnostic_cohort([self.dataset], seed=27, max_windows=8)
        self.assertEqual(len(full), 64)
        self.assertEqual(full.report["selected_strata"], {"uniform": 32, "event": 32})
        self.assertEqual(four.records, full.records[:4])
        self.assertEqual(eight.records, full.records[:8])
        self.assertEqual(full.prefix(4).cohort_identity, four.cohort_identity)
        self.assertEqual(full.prefix(8).cohort_identity, eight.cohort_identity)
        self.assertEqual([record["stratum"] for record in full.records[:8]], ["uniform", "event"]*4)
        uniform = [record for record in full.records if record["stratum"] == "uniform"]
        self.assertEqual(len({(record["dataset_index"], record["episode_id"], record["anchor"]) for record in uniform}), 32)
        self.assertTrue(all(record["episode_id"] in self.diagnostic_ids for record in full.records))
        self.assertFalse(any(record["episode_id"] == self.fit_id for record in full.records))

    def test_loaded_windows_preserve_full_real_prefix_clocks_and_terminal_masks(self):
        cohort = select_diagnostic_cohort([self.dataset], seed=31)
        index = next(index for index, record in enumerate(cohort.records)
                     if record["stratum"] == "event" and any("terminal/success" in event["kinds"]
                                                             for event in record["event_instances"]))
        record = cohort.records[index]
        with patch.object(EpisodeView, "arrays", side_effect=AssertionError("whole episode materialization")):
            window = cohort.window(record)
        self.assertEqual((window.start, window.anchor, window.stop), (0, 15, 20))
        self.assertEqual(window.burn_in, 15)
        self.assertEqual(window.loss_steps, 5)
        self.assertEqual(window.transitions["decision_index"].tolist(), list(range(20)))
        self.assertTrue(window.states["image_valid"][window.anchor].all())
        self.assertFalse(window.states["image_valid"][-1, 0])
        self.assertTrue(window.states["image_valid"][-1, 1])
        self.assertTrue(window.states["privileged_validity_mask"][-1].all())
        self.assertTrue(window.transitions["terminated"][-1])
        self.assertAlmostEqual(float(window.transitions["team_reward"][-1]), .999)
        self.assertEqual(window.event_fields, tuple(self.dataset.manifest["schema"]["event_fields"]))
        self.assertFalse(window.states["rgb"].flags.writeable)
        self.assertEqual(len(window.states["rgb"]), 21)
        self.assertEqual(next(cohort.prefix(1).iter_windows()).episode_id, cohort.records[0]["episode_id"])

    def test_event_instances_have_no_repeated_event_targets_and_full_source_support_is_explicit(self):
        cohort = select_diagnostic_cohort([self.dataset], seed=29)
        full = cohort.report["full_heldout_support"]
        self.assertEqual((full["episodes"], full["transitions"], full["eligible_anchors"]), (16, 320, 320))
        self.assertEqual(full["rare_event_instances"], 32)
        self.assertEqual(full["rare_event_kinds"]["milestone/latch_activations"], 16)
        self.assertEqual(full["rare_event_kinds"]["terminal/success"], 16)
        self.assertEqual(full["episode_end_reasons"]["success"], 16)
        self.assertEqual(full["episode_end_reasons"]["death"], 0)
        self.assertEqual(full["transition_reasons"]["death"], 0)
        self.assertEqual(full["events"]["latch_activations"]["positive"], 16)
        self.assertEqual(full["events"]["latch_activations"]["valid"], 320)
        enriched = [record for record in cohort.records if record["stratum"] == "event"]
        event_ids = [event_id for record in enriched for event_id in record["event_ids"]]
        self.assertEqual(len(event_ids), len(set(event_ids)))
        self.assertEqual(len(event_ids), 32)
        self.assertEqual(cohort.report["overlap_counts"]["rare_event_instances_repeated_within_event"], 0)
        uniform_events = Counter(event_id for record in cohort.records if record["stratum"] == "uniform"
                                 for event_id in record["event_ids"])
        self.assertEqual(cohort.report["overlap_counts"]["rare_event_instances_between_strata"],
                         len(uniform_events.keys() & set(event_ids)))
        self.assertEqual(cohort.report["selected_event_support"]["event"]["unique_event_instances"], 32)

    def test_coincident_and_nearby_events_are_not_counted_as_duplicate_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            same = identifier("coincident-selection", "diagnostic")
            dataset = make_store(Path(directory)/"one", [(same, 1, True)])
            cohort = select_diagnostic_cohort([dataset], seed=2)
            event_records = [record for record in cohort.records if record["stratum"] == "event"]
            self.assertEqual(cohort.report["full_heldout_support"]["rare_event_instances"], 1)
            self.assertEqual(len(event_records), 1)
            event = event_records[0]["event_instances"][0]
            self.assertEqual(event["transition_index"], 0)
            self.assertIn("milestone/latch_activations", event["kinds"])
            self.assertIn("terminal/success", event["kinds"])
            self.assertEqual(cohort.report["overlap_counts"]["identical_anchors_between_strata"], 1)
            nearby = identifier("nearby-selection", "diagnostic")
            dataset = make_store(Path(directory)/"nearby", [(nearby, 5, False)])
            cohort = select_diagnostic_cohort([dataset], seed=2)
            event_records = [record for record in cohort.records if record["stratum"] == "event"]
            self.assertEqual(cohort.report["full_heldout_support"]["rare_event_instances"], 2)
            self.assertEqual(len(event_records), 1)
            self.assertEqual(len(event_records[0]["event_ids"]), 2)
            self.assertEqual(cohort.report["overlap_counts"]["rare_event_instances_repeated_within_event"], 0)

    def test_reports_are_defensive_and_selection_never_changes_sources_or_fit_rng(self):
        before_files = {path: sha256_file(path) for path in self.dataset.path.rglob("manifest*") if path.is_file()}
        fitting = ReplaySampler([self.dataset], seed=63)
        fitting_state = deepcopy(fitting.state_dict())
        global_rng = np.random.get_state()
        cohort = select_diagnostic_cohort([self.dataset], seed=63, max_windows=4)
        original = cohort.records
        changed = cohort.records
        changed[0]["anchor"] = 999
        changed[0]["event_ids"].append("invented")
        report = cohort.report
        report["full_heldout_support"]["episodes"] = 999
        self.assertEqual(cohort.records, original)
        self.assertEqual(cohort.report["full_heldout_support"]["episodes"], 16)
        self.assertEqual(fitting.state_dict(), fitting_state)
        current_rng = np.random.get_state()
        self.assertEqual(global_rng[0], current_rng[0])
        np.testing.assert_array_equal(global_rng[1], current_rng[1])
        self.assertEqual(global_rng[2:], current_rng[2:])
        self.assertEqual({path: sha256_file(path) for path in before_files}, before_files)
        with self.assertRaisesRegex(ValueError, "outside the frozen diagnostic cohort"):
            cohort.window(changed[0])

    def test_guarded_sources_empty_diagnostic_partition_and_invalid_limits_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            preflight = make_store(Path(directory)/"preflight", [(identifier("preflight-selection", "diagnostic"), 5, False)],
                                   preflight=True)
            with self.assertRaisesRegex(ValueError, "excluded from learning"):
                select_diagnostic_cohort([preflight], seed=1)
            only_fit = make_store(Path(directory)/"fit", [(identifier("fit-only-selection", "fit"), 5, False)])
            with self.assertRaisesRegex(ValueError, "no valid source-image anchors"):
                select_diagnostic_cohort([only_fit], seed=1)
            with self.assertRaisesRegex(ValueError, "duplicate dataset content"):
                select_diagnostic_cohort([self.dataset, ReplayDataset(self.dataset.path)], seed=1)
        for kwargs in ({"max_windows": 65}, {"max_windows": 0}, {"horizon": 16}, {"horizon": 0},
                       {"uniform_limit": 33}, {"event_limit": 33}, {"uniform_limit": 0, "event_limit": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_diagnostic_cohort([self.dataset], seed=1, **kwargs)
        cohort = select_diagnostic_cohort([self.dataset], seed=1, max_windows=4)
        with self.assertRaises(IndexError):
            cohort.window(4)
        with self.assertRaises(ValueError):
            cohort.window(True)
        with self.assertRaises(ValueError):
            cohort.prefix(0)


class SelectionSchemaOrderTests(unittest.TestCase):
    def test_noncanonical_stored_event_order_selects_the_correct_milestone_column(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostic = identifier("reordered-selection", "diagnostic")
            original = make_store(Path(directory)/"replay", [(diagnostic, 1, True)])
            manifest = deepcopy(original.manifest)
            events = tuple(reversed(manifest["schema"]["event_fields"]))
            # Re-encode a fresh synthetic fixture with a permitted event-column
            # order and matching immutable receipts; no project replay is used.
            for episode in manifest["episodes"]:
                for chunk in episode["chunks"]:
                    arrays = dict(read_npz(original.path/chunk["path"]))
                    for name in ("cumulative_event_counters", "event_validity_mask",
                                 "event_deltas", "event_delta_validity_mask"):
                        arrays[name] = arrays[name][..., ::-1].copy()
                    chunk.update(write_npz(original.path/chunk["path"], arrays))
                episode_path = original.path/episode["manifest_path"]
                episode_path.write_text(json.dumps({key: value for key, value in episode.items()
                    if key not in ("manifest_path", "manifest_sha256")}), encoding="utf-8")
                episode["manifest_sha256"] = sha256_file(episode_path)
            manifest["schema"] = field_manifest(events)
            manifest["dataset_identity_sha256"] = canonical_hash({key: value for key, value in manifest.items()
                if key not in ("sealed_at_utc", "dataset_identity_sha256")})
            manifest_path = original.path/"manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (original.path/"manifest.sha256").write_text(sha256_file(manifest_path)+"  manifest.json\n", encoding="ascii")
            dataset = ReplayDataset(original.path)
            cohort = select_diagnostic_cohort([dataset], seed=11)
            self.assertEqual(cohort.report["full_heldout_support"]["events"]["latch_activations"]["positive"], 1)
            enriched = next(record for record in cohort.records if record["stratum"] == "event")
            self.assertIn("milestone/latch_activations", enriched["event_instances"][0]["kinds"])
            self.assertEqual(cohort.window(enriched).event_fields, events)


if __name__ == "__main__":
    unittest.main()
