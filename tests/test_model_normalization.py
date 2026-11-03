"""Fitting-only statistics, independent target masks, and immutable metadata."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_replay_storage import episode_fixture, write_episode
from world_models.data.replay import ReplayDataset, ReplayDatasetWriter, canonical_hash
from world_models.data.schema import PLAYER_FIELDS, episode_partition
from world_models.models.heads import auxiliary_fields
from world_models.training.normalization import (
    FieldStatistics, NormalizationStats, continuous_field_names, fit_field_statistics, fit_normalization,
)


FIELD = "player_0/position_x"


def make_store(directory, *, family="relay", preflight=False, missing_final=False, missing_state=False,
               include_diagnostic=True):
    identifiers = [f"normalization-episode-{index}" for index in range(100)]
    fit_id = next(name for name in identifiers if episode_partition(name) == "fit")
    diagnostic_id = next(name for name in identifiers if episode_partition(name) == "diagnostic")
    writer = ReplayDatasetWriter(Path(directory) / "replay", {
        "task_family": family, "run_seed": 0, "source": "real_environment",
        "budget_scope": "collection_preflight" if preflight else "initial_collection",
        "target_transitions": 10 if include_diagnostic else 5}, chunk_transitions=2)
    write_episode(writer, episode_fixture(family=family, missing_final=missing_final, missing_state=missing_state), episode_id=fit_id)
    if include_diagnostic:
        results = episode_fixture(family=family)
        for result in results:
            for player in result.training_only_info["privileged_targets"]["players"]:
                # A large diagnostic outlier must not affect fitting moments.
                player["normalized"]["position_x"] = 10000.
        write_episode(writer, results, episode_id=diagnostic_id)
    writer.seal()
    return ReplayDataset(writer.path, cache_chunks=1), fit_id, diagnostic_id


class NormalizationMomentTests(unittest.TestCase):
    def test_population_moments_match_independent_numpy_reduction(self):
        values = np.array([2., -3., 7., 9., 1.], dtype=np.float64)
        stats = fit_field_statistics([{FIELD: (values, np.ones(5, dtype=bool))}], "relay")
        field = stats.fields[FIELD]
        self.assertEqual(field.count, 5)
        self.assertAlmostEqual(field.mean, values.mean(), places=14)
        self.assertAlmostEqual(field.variance, values.var(ddof=0), places=14)
        self.assertAlmostEqual(field.std, values.std(ddof=0), places=14)
        self.assertEqual(field.minimum, -3.)
        self.assertEqual(field.maximum, 9.)

    def test_chunked_stable_welford_handles_large_offsets(self):
        values = 1e12 + np.array([-4., -2., 0., 2., 4.])
        stats = fit_field_statistics(({FIELD: (chunk, np.ones(chunk.shape, dtype=bool))}
                                     for chunk in (values[:2], values[2:4], values[4:])), "relay")
        self.assertAlmostEqual(stats.fields[FIELD].mean, values.mean(), places=8)
        self.assertAlmostEqual(stats.fields[FIELD].variance, values.var(), places=10)

    def test_validity_masks_ignore_nonfinite_padding_and_reject_valid_nan(self):
        values = np.array([2., np.nan, 6.])
        stats = fit_field_statistics([{FIELD: (values, np.array([True, False, True]))}], "relay")
        self.assertEqual(stats.fields[FIELD].count, 2)
        self.assertEqual(stats.fields[FIELD].mean, 4.)
        with self.assertRaisesRegex(ValueError, "nonfinite valid"):
            fit_field_statistics([{FIELD: (values, np.ones(3, dtype=bool))}], "relay")
        with self.assertRaisesRegex(ValueError, "boolean validity"):
            fit_field_statistics([{FIELD: (values, np.array([1, 0, 1]))}], "relay")

    def test_constant_scale_and_unavailable_fields_never_invent_statistics(self):
        stats = fit_field_statistics([{FIELD: (np.array([3., 3.]), np.ones(2, dtype=bool))}], "relay")
        self.assertTrue(stats.fields[FIELD].constant)
        self.assertEqual(stats.fields[FIELD].scale, 1.)
        torch.testing.assert_close(stats.normalize(FIELD, torch.tensor([3., 4.])), torch.tensor([0., 1.]))
        unknown = "player_0/velocity_x"
        self.assertFalse(stats.fields[unknown].available)
        self.assertIsNone(stats.fields[unknown].mean)
        with self.assertRaisesRegex(ValueError, "no fitting-data statistics"):
            stats.normalize(unknown, torch.tensor([0.]))
        self.assertEqual(stats.normalize(unknown, torch.tensor([])).numel(), 0)

    def test_normalization_float32_gradients_and_roundtrip_units(self):
        stats = fit_field_statistics([{FIELD: (np.array([1., 3.]), np.ones(2, dtype=bool))}], "relay")
        values = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
        normalized = stats.normalize(FIELD, values)
        self.assertEqual(normalized.dtype, torch.float32)
        torch.testing.assert_close(normalized, torch.tensor([0., 1.]))
        torch.testing.assert_close(stats.denormalize(FIELD, normalized), values.float())
        normalized.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones(2, dtype=torch.float64))
        with self.assertRaises(ValueError):
            stats.normalize(FIELD, torch.tensor([1]))
        with self.assertRaises(ValueError):
            stats.normalize(FIELD, torch.tensor([float("nan")]))

    def test_registry_contains_only_declared_continuous_state_fields(self):
        for family, count in (("relay", 22), ("cover", 30)):
            stats = fit_field_statistics([], family)
            expected = {field.name for field in auxiliary_fields(family) if field.kind == "continuous"}
            self.assertEqual(set(stats.fields), expected)
            self.assertEqual(len(stats.fields), count)
            self.assertFalse(any(name.startswith("event/") for name in stats.fields))
        with self.assertRaises(ValueError):
            fit_field_statistics([{"task/stage": (np.zeros(1), np.ones(1, dtype=bool))}], "relay")

    def test_metadata_roundtrip_deterministic_identity_and_no_overwrite(self):
        stats = fit_field_statistics([{FIELD: (np.arange(4.), np.ones(4, dtype=bool))}], "relay")
        self.assertEqual(NormalizationStats.from_dict(stats.as_dict()).as_dict(), stats.as_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "statistics.json"
            receipt = stats.save(path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), receipt["sha256"])
            self.assertEqual(NormalizationStats.load(path).as_dict(), stats.as_dict())
            with self.assertRaises(FileExistsError):
                stats.save(path)
            path.write_text(path.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                NormalizationStats.load(path)

    def test_metadata_rejects_tampering_missing_fields_and_bad_variance(self):
        stats = fit_field_statistics([{FIELD: (np.arange(4.), np.ones(4, dtype=bool))}], "relay")
        modified = stats.as_dict()
        modified["fields"][FIELD]["mean"] = 999.
        with self.assertRaisesRegex(ValueError, "identity hash"):
            NormalizationStats.from_dict(modified)
        modified = stats.as_dict()
        modified["fields"].pop(FIELD)
        modified["identity_sha256"] = canonical_hash({name: value for name, value in modified.items() if name != "identity_sha256"})
        with self.assertRaisesRegex(ValueError, "every continuous"):
            NormalizationStats.from_dict(modified)
        with self.assertRaises(ValueError):
            FieldStatistics(2, 0., -1., 1., 1., -1., 1.)
        with self.assertRaises(ValueError):
            FieldStatistics(0, None, None, None, True, None, None)


class ReplayNormalizationTests(unittest.TestCase):
    def test_fit_only_unique_observations_including_reset_and_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, fit_id, diagnostic_id = make_store(directory)
            manifest_before = dataset.manifest_sha256
            with patch.object(type(dataset.load_episode(fit_id)), "arrays", side_effect=AssertionError("full episode materialized")):
                stats = fit_normalization(dataset)
            self.assertEqual(stats.fields[FIELD].count, 6)
            expected = dataset.load_episode(fit_id, purpose="world_model_fit").arrays()["normalized_player_state"][:, 0, PLAYER_FIELDS.index("position_x")]
            self.assertAlmostEqual(stats.fields[FIELD].mean, float(expected.astype(np.float64).mean()), places=14)
            self.assertLess(stats.fields[FIELD].maximum, 10000.)
            self.assertEqual(stats.provenance["fitting_episode_ids"], [fit_id])
            self.assertNotIn(diagnostic_id, stats.provenance["fitting_episode_ids"])
            self.assertEqual(stats.provenance["fitting_transition_count"], 5)
            self.assertEqual(stats.provenance["unique_state_count"], 6)
            self.assertEqual(stats.provenance["excluded_model_diagnostic_episode_count"], 1)
            self.assertEqual(stats.provenance["shared_by_variants"], ["B", "D"])
            self.assertEqual(dataset.manifest_sha256, manifest_before)
            self.assertLessEqual(len(dataset._cache), 1)

    def test_terminal_image_mask_does_not_erase_independent_state_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, _, _ = make_store(directory, missing_final=True, include_diagnostic=False)
            stats = fit_normalization(dataset)
            self.assertEqual(stats.fields[FIELD].count, 6)
        with tempfile.TemporaryDirectory() as directory:
            dataset, _, _ = make_store(directory, missing_state=True, include_diagnostic=False)
            stats = fit_normalization(dataset)
            self.assertEqual(stats.fields[FIELD].count, 5)

    def test_label_profile_uses_independent_state_and_transition_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, _, _ = make_store(directory)
            progress = []
            stats = fit_normalization(dataset, progress=progress.append)
            profile = stats.provenance["label_profile"]
            self.assertEqual(profile["learned_outcome"]["counts"], [4, 1, 0])
            self.assertEqual(profile["learned_outcome"]["valid"], 5)
            self.assertEqual(profile["task_stage"]["valid"], 6)
            self.assertEqual(sum(profile["task_stage"]["counts"]), 6)
            self.assertEqual(profile["binary_fields"]["event/latch_activations"]["valid"], 5)
            self.assertEqual(profile["binary_fields"]["event/latch_activations"]["positive"], 1)
            self.assertEqual(profile["binary_fields"]["task/latch_active"]["valid"], 6)
            self.assertEqual(profile["binary_fields"]["task/latch_active"]["positive"], 5)
            self.assertAlmostEqual(profile["binary_fields"]["task/latch_active"]["positive_fraction"], 5/6)
            self.assertEqual(profile["deadline_transitions"], 0)
            self.assertEqual(progress, [{"fitting_episodes": 1, "unique_states": 6, "fitting_transitions": 5, "chunks": 3}])

    def test_real_preflight_and_wrong_store_provenance_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, _, _ = make_store(directory, preflight=True)
            with self.assertRaisesRegex(ValueError, "learning-eligible"):
                fit_normalization(dataset)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            dataset, _, _ = make_store(first)
            other, _, _ = make_store(second, missing_final=True)
            stats = fit_normalization(dataset)
            stats.validate_for_dataset(dataset)
            with self.assertRaisesRegex(ValueError, "do not belong"):
                stats.validate_for_dataset(other)
        with self.assertRaises(ValueError):
            fit_normalization(object())

    def test_cover_enemy_fields_and_empty_fit_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, _, _ = make_store(directory, family="cover", include_diagnostic=False)
            stats = fit_normalization(dataset)
            self.assertIn("enemy_1/health", stats.fields)
            self.assertEqual(len(stats.fields), 30)
        with tempfile.TemporaryDirectory() as directory:
            dataset, _, _ = make_store(directory, include_diagnostic=False)
            with patch.object(dataset, "iter_episodes", return_value=iter(())):
                with self.assertRaisesRegex(ValueError, "at least one fitting"):
                    fit_normalization(dataset)


if __name__ == "__main__":
    unittest.main()
