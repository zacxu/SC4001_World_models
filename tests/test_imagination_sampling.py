"""Fitting-only, outcome-independent anchors and complete causal prefixes."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_model_sampling import identifier, make_store
from tests.test_replay_storage import episode_fixture, rewrite_chunk_with_receipts, write_episode
from world_models.contracts import BoundaryReason, EpisodeBoundary
from world_models.data.replay import EpisodeView, ReplayDataset, ReplayDatasetWriter
from world_models.data.schema import REASONS
from world_models.imagination.replay import AnchorPrefix, AnchorRef, AnchorSampler
from world_models.infrastructure.config import resolve_config


class AnchorSamplingTests(unittest.TestCase):
    def test_complete_prefix_alignment_and_owned_readonly_arrays_across_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            episode_id = identifier("anchor-alignment", "fit")
            dataset = make_store(Path(directory)/"replay", [(episode_id, 7, True)])
            sampler = AnchorSampler([dataset], seed=8, deadline=300)
            expected = dataset.load_episode(episode_id).arrays()
            state = sampler.state_dict()
            ref = sampler.reference(0, episode_id, 4)
            with patch.object(EpisodeView, "arrays", side_effect=AssertionError("whole episode read")):
                prefix = sampler.read_prefix(ref)
            self.assertEqual(ref.prefix_length, 5)
            np.testing.assert_array_equal(prefix.rgb, expected["rgb"][:5])
            np.testing.assert_array_equal(prefix.actions, expected["actions"][:4])
            np.testing.assert_array_equal(prefix.decision_indices, np.arange(5))
            self.assertEqual(prefix.actions.dtype, np.int64)
            self.assertEqual(state, sampler.state_dict())
            for value in (prefix.rgb, prefix.actions, prefix.decision_indices):
                self.assertTrue(value.flags.owndata)
                self.assertFalse(value.flags.writeable)
            self.assertEqual(set(vars(prefix)), {"ref", "rgb", "actions", "decision_indices"})
            # Outgoing terminal transitions do not disqualify their source.
            self.assertEqual(sampler.reference(0, episode_id, 6).decision_index, 6)
            with self.assertRaisesRegex(ValueError, "nonterminal source"):
                sampler.reference(0, episode_id, 7)

    def test_prefix_reader_uses_only_intersecting_chunks_once_and_no_future_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            episode_id = identifier("anchor-causal", "fit")
            dataset = make_store(Path(directory)/"replay", [(episode_id, 9, False)], chunk_transitions=2)
            sampler = AnchorSampler([dataset], seed=4, deadline=300)
            ref = sampler.reference(0, episode_id, 3)
            expected = sampler.read_prefix(ref)
            original = dataset._chunk
            paths = []

            class PrefixFields(dict):
                def __getitem__(self, name):
                    if name not in {"rgb", "image_valid", "actions", "decision_index"}:
                        raise AssertionError(f"non-prefix field accessed: {name}")
                    return super().__getitem__(name)

            def altered_chunk(chunk):
                paths.append(chunk["path"])
                arrays = {name: value.copy() for name, value in original(chunk).items()}
                for local in range(chunk["state_count"]):
                    if chunk["state_start"] + local > ref.decision_index:
                        arrays["rgb"][local] = 255
                        arrays["image_valid"][local] = False
                for local in range(chunk["transition_count"]):
                    if chunk["transition_start"] + local >= ref.decision_index:
                        arrays["actions"][local] = 11
                        arrays["decision_index"][local] = 999
                return PrefixFields(arrays)

            with patch.object(dataset, "_chunk", side_effect=altered_chunk), patch.object(
                    EpisodeView, "arrays", side_effect=AssertionError("whole episode read")):
                actual = sampler.read_prefix(ref)
            np.testing.assert_array_equal(actual.rgb, expected.rgb)
            np.testing.assert_array_equal(actual.actions, expected.actions)
            np.testing.assert_array_equal(actual.decision_indices, expected.decision_indices)
            self.assertEqual(paths, [row["path"] for row in dataset.episodes[0]["chunks"][:2]])
            self.assertEqual(len(paths), len(set(paths)))
            self.assertLessEqual(len(dataset._cache), 1)

    def test_reset_deadline_and_final_cutoff_destination_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            episode_id = identifier("anchor-cutoff", "fit")
            writer = ReplayDatasetWriter(Path(directory)/"replay", {
                "task_family": "relay", "run_seed": 0, "source": "real_environment",
                "target_transitions": 5}, chunk_transitions=2)
            results = episode_fixture(5)
            cutoff = EpisodeBoundary(BoundaryReason.COLLECTION_CUTOFF)
            results[-1] = replace(results[-1], boundary=cutoff,
                                  metadata=replace(results[-1].metadata, boundary=cutoff))
            write_episode(writer, results, episode_id=episode_id)
            writer.seal()
            dataset = ReplayDataset(writer.path)
            sampler = AnchorSampler([dataset], seed=0, deadline=4)
            self.assertEqual(sampler.anchor_count, 4)
            self.assertEqual(sampler.reference(0, episode_id, 3).decision_index, 3)
            prefix = sampler.read_prefix(sampler.reference(0, episode_id, 0))
            self.assertEqual(prefix.rgb.shape, (1, 2, 3, 96, 128))
            self.assertEqual(prefix.actions.shape, (0, 2))
            self.assertEqual(prefix.decision_indices.tolist(), [0])
            for index in (4, 5):
                with self.assertRaisesRegex(ValueError, "before its deadline"):
                    sampler.reference(0, episode_id, index)
            normal = AnchorSampler([dataset], seed=0, deadline=300)
            self.assertEqual(normal.anchor_count, 5)
            self.assertEqual(normal.reference(0, episode_id, 4).decision_index, 4)
            with self.assertRaises(ValueError):
                normal.reference(0, episode_id, 5)

    def test_uniform_population_tracks_current_transition_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            short, long = identifier("anchor-short", "fit"), identifier("anchor-long", "fit")
            dataset = make_store(Path(directory)/"replay", [(short, 2, False), (long, 8, False)])
            sampler = AnchorSampler([dataset], seed=311, deadline=300, recent_fraction=0.)
            self.assertEqual(sampler.anchor_count, 10)
            draws = sampler.sample(5000)
            fraction = sum(ref.episode_id == long for ref in draws)/len(draws)
            self.assertTrue(.77 < fraction < .83)

    def test_recent_pool_uses_physical_chronology_including_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, diagnostic = (identifier("anchor-old", "fit"), identifier("anchor-new", "fit"),
                                    identifier("anchor-recent-diagnostic", "diagnostic"))
            first = make_store(Path(directory)/"first", [(old, 4, False)])
            second = make_store(Path(directory)/"second", [(new, 5, False), (diagnostic, 2, False)])
            sampler = AnchorSampler([first, second], seed=6, deadline=300,
                                    recent_fraction=1., recent_transitions=4)
            self.assertEqual((sampler.anchor_count, sampler.recent_anchor_count), (9, 2))
            self.assertEqual({(ref.dataset_index, ref.episode_id, ref.decision_index) for ref in sampler.sample(40)},
                             {(1, new, 3), (1, new, 4)})
            with self.assertRaisesRegex(ValueError, "suffix contains no eligible"):
                AnchorSampler([first, second], seed=0, deadline=300, recent_transitions=2)
            reversed_sampler = AnchorSampler([second, first], seed=6, deadline=300,
                                             recent_fraction=1., recent_transitions=4)
            self.assertEqual({ref.episode_id for ref in reversed_sampler.sample(20)}, {old})

    def test_exact_round_half_up_mixture_quota_and_shuffle(self):
        with tempfile.TemporaryDirectory() as directory:
            episode_id = identifier("anchor-quota", "fit")
            dataset = make_store(Path(directory)/"replay", [(episode_id, 8, False)])
            sampler = AnchorSampler([dataset], seed=1, deadline=300, recent_transitions=2)

            class RecordingGenerator:
                def __init__(self):
                    self.calls = []

                def integers(self, high, *, size):
                    self.calls.append((high, size))
                    return np.zeros(size, dtype=np.int64)

                def permutation(self, size):
                    return np.arange(size-1, -1, -1)

            generator = RecordingGenerator()
            sampler._rng = generator
            refs = sampler.sample(5)
            self.assertEqual(generator.calls, [(8, 2), (2, 3)])
            self.assertEqual([ref.decision_index for ref in refs], [6, 6, 6, 0, 0])

    def test_rng_resume_is_exact_and_does_not_touch_global_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            episode_id = identifier("anchor-rng", "fit")
            dataset = make_store(Path(directory)/"replay", [(episode_id, 8, False)])
            arguments = dict(deadline=300, recent_fraction=.35, recent_transitions=3, batch_size=9)
            sampler = AnchorSampler([dataset], seed=4, **arguments)
            other = AnchorSampler([dataset], seed=999, **arguments)
            global_before = np.random.get_state()
            sampler.sample(3)
            saved = sampler.state_dict()
            expected = sampler.sample()
            other.load_state_dict(saved)
            self.assertEqual(other.sample(), expected)
            global_after = np.random.get_state()
            self.assertEqual(global_before[0], global_after[0])
            np.testing.assert_array_equal(global_before[1], global_after[1])
            self.assertEqual(global_before[2:], global_after[2:])
            saved["rng_state"]["state"]["state"] = 0
            self.assertNotEqual(saved["rng_state"], other.state_dict()["rng_state"])
            before = other.state_dict()
            malformed = deepcopy(before)
            malformed["rng_state"] = {"bit_generator": "bad"}
            with self.assertRaisesRegex(ValueError, "invalid anchor sampler RNG"):
                other.load_state_dict(malformed)
            self.assertEqual(before, other.state_dict())

    def test_relocation_preserves_identity_but_reordering_or_parameters_reject_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            first = make_store(Path(directory)/"first", [(identifier("anchor-cohort-first", "fit"), 3, False)])
            second = make_store(Path(directory)/"second", [(identifier("anchor-cohort-second", "fit"), 4, False)])
            sampler = AnchorSampler([first, second], seed=3, deadline=300)
            shutil.copytree(first.path, Path(directory)/"moved")
            moved = AnchorSampler([ReplayDataset(Path(directory)/"moved"), second], seed=0, deadline=300)
            moved.load_state_dict(sampler.state_dict())
            self.assertEqual(moved.sample(12), sampler.sample(12))
            for stores, extra in (([second, first], {}), ([first, second], {"deadline": 250}),
                                  ([first, second], {"recent_fraction": 0.}),
                                  ([first, second], {"batch_size": 128})):
                arguments = {"seed": 3, "deadline": 300, **extra}
                incompatible = AnchorSampler(stores, **arguments)
                with self.assertRaisesRegex(ValueError, "cohort or sampling configuration"):
                    incompatible.load_state_dict(sampler.state_dict())
            ref = sampler.sample(1)[0]
            with self.assertRaisesRegex(ValueError, "different replay cohort"):
                sampler.validate_reference(replace(ref, dataset_identity="a"*64))

    def test_diagnostic_episodes_and_incompatible_stores_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            fit, diagnostic = identifier("anchor-guard", "fit"), identifier("anchor-guard", "diagnostic")
            first = make_store(Path(directory)/"first", [(fit, 3, False), (diagnostic, 4, False)])
            sampler = AnchorSampler([first], seed=0, deadline=300)
            self.assertEqual({ref.episode_id for ref in sampler.sample()}, {fit})
            with self.assertRaisesRegex(ValueError, "guarded fitting cohort"):
                sampler.reference(0, diagnostic, 0)
            preflight = make_store(Path(directory)/"preflight", [(identifier("anchor-preflight", "fit"), 3, False)],
                                   preflight=True)
            duplicate = make_store(Path(directory)/"duplicate", [(fit, 4, False)])
            cover = make_store(Path(directory)/"cover", [(identifier("anchor-cover", "fit"), 3, False)], family="cover")
            for sources, message in (([preflight], "excluded from learning"),
                                     ([first, ReplayDataset(first.path)], "duplicate dataset content"),
                                     ([first, duplicate], "duplicate episode IDs"),
                                     ([first, cover], "share task family")):
                with self.assertRaisesRegex(ValueError, message):
                    AnchorSampler(sources, seed=0, deadline=300)
            for key, value, message in (("run_seed", 1, "share task family"),
                                        ("data_source", "imagination", "excluded from learning")):
                original = deepcopy(duplicate.manifest["metadata"])
                duplicate.manifest["metadata"][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    AnchorSampler([first, duplicate], seed=0, deadline=300)
                duplicate.manifest["metadata"] = original

    def test_semantic_integrity_is_required_even_when_dataset_open_skipped_verification(self):
        for corruption in ("clock", "early_terminal", "missing_reset"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                episode_id = identifier("anchor-corrupt", "fit")
                dataset = make_store(Path(directory)/"replay", [(episode_id, 5, False)])

                def transform(arrays):
                    if corruption == "clock":
                        arrays["decision_index"][0] = 4
                    elif corruption == "early_terminal":
                        arrays["terminated"][0] = True
                        arrays["termination_reason"][0] = REASONS.index("success")
                    else:
                        arrays["image_valid"][0, 0] = False
                        arrays["rgb"][0, 0] = 0

                rewrite_chunk_with_receipts(dataset.path, transform)
                unchecked = ReplayDataset(dataset.path, verify=False)
                with self.assertRaises(ValueError):
                    AnchorSampler([unchecked], seed=0, deadline=300)

    def test_prefix_owns_inputs_and_rejects_invalid_types_shapes_actions_and_clocks(self):
        ref = AnchorRef("a"*64, 0, "b"*64, "episode", 1)
        rgb = np.zeros((2, 2, 3, 96, 128), dtype=np.uint8)
        actions = np.zeros((1, 2), dtype=np.int64)
        clocks = np.arange(2, dtype=np.int64)
        prefix = AnchorPrefix(ref, rgb, actions, clocks)
        rgb[:] = 25
        actions[:] = 11
        clocks[:] = 99
        self.assertFalse(prefix.rgb.any())
        self.assertFalse(prefix.actions.any())
        self.assertEqual(prefix.decision_indices.tolist(), [0, 1])
        for changes in ({"rgb": prefix.rgb.astype(np.float32)}, {"rgb": prefix.rgb[:-1]},
                        {"actions": prefix.actions.astype(np.uint8)},
                        {"actions": np.array([[0, 12]], dtype=np.int64)},
                        {"actions": np.array([[-1, 0]], dtype=np.int64)},
                        {"decision_indices": np.array([1, 2], dtype=np.int64)},
                        {"decision_indices": np.array([0, 1], dtype=np.int32)}, {"ref": None}):
            with self.assertRaises(ValueError):
                replace(prefix, **changes)

    def test_input_validation_and_empty_fit_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            fit = identifier("anchor-inputs", "fit")
            dataset = make_store(Path(directory)/"replay", [(fit, 3, False)])
            for changes in ({"seed": -1}, {"seed": True}, {"deadline": 0}, {"deadline": 1.5},
                            {"batch_size": 0}, {"recent_fraction": float("nan")},
                            {"recent_fraction": True}, {"recent_fraction": -1}, {"recent_transitions": 0}):
                with self.assertRaises(ValueError):
                    AnchorSampler([dataset], **{"seed": 0, "deadline": 300, **changes})
            sampler = AnchorSampler([dataset], seed=0, deadline=300)
            for size in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    sampler.sample(size)
            for coordinates in ((True, fit, 0), (0, fit, True), (0, [], 0), (1, fit, 0)):
                with self.assertRaises(ValueError):
                    sampler.reference(*coordinates)
            for changes in ({"cohort_identity": "bad"}, {"dataset_identity": "A"*64},
                            {"dataset_index": -1}, {"episode_id": ""}, {"decision_index": True}):
                with self.assertRaises(ValueError):
                    replace(sampler.reference(0, fit, 0), **changes)
            diagnostic = make_store(Path(directory)/"diagnostic", [(identifier("anchor-only-diagnostic", "diagnostic"), 3, False)])
            with self.assertRaisesRegex(ValueError, "no valid nonterminal"):
                AnchorSampler([diagnostic], seed=0, deadline=300)

    def test_experiment_defaults_and_paired_variant_sampling_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_store(Path(directory)/"replay", [(identifier("anchor-experiment", "fit"), 5, False)])
            baseline = resolve_config("relay", "B", "pilot").settings
            guided = resolve_config("relay", "D", "pilot").settings
            left = AnchorSampler.from_experiment([dataset], baseline, seed=4)
            right = AnchorSampler.from_experiment([dataset], guided, seed=4)
            self.assertEqual(left.batch_size, 256)
            self.assertEqual(left.recent_fraction, .5)
            self.assertEqual(left.deadline, baseline["task"]["deadline"])
            self.assertEqual(left.sample(), right.sample())
            with self.assertRaisesRegex(ValueError, "experiment task differs"):
                AnchorSampler.from_experiment([dataset], resolve_config("cover", "B", "pilot").settings, seed=4)


if __name__ == "__main__":
    unittest.main()
