"""Episode boundaries, physical recency, source guards, and exact sampling resume."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_replay_storage import episode_fixture, write_episode
from world_models.data.replay import EpisodeView, ReplayDataset, ReplayDatasetWriter
from world_models.data.schema import STATE_ARRAYS, TRANSITION_ARRAYS, episode_partition
from world_models.training.sampling import ReplaySampler


def identifier(prefix: str, partition: str) -> str:
    return next(f"{prefix}-{index}" for index in range(1000)
                if episode_partition(f"{prefix}-{index}") == partition)


def make_store(path, episodes, *, preflight=False, chunk_transitions=2, family="relay"):
    writer = ReplayDatasetWriter(path, {
        "task_family": family, "run_seed": 0, "source": "real_environment",
        "budget_scope": "collection_preflight" if preflight else "initial_collection",
        "target_transitions": sum(length for _, length, _ in episodes)}, chunk_transitions=chunk_transitions)
    for name, length, missing in episodes:
        write_episode(writer, episode_fixture(length, missing_final=missing, family=family), episode_id=name)
    writer.seal()
    return ReplayDataset(path, cache_chunks=1)


def coordinates(windows):
    return [(window.dataset_index, window.episode_id, window.start, window.anchor, window.stop) for window in windows]


class ReplaySamplingTests(unittest.TestCase):
    def test_windows_align_labels_across_chunks_and_keep_terminal_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            fit = identifier("alignment", "fit")
            dataset = make_store(Path(directory)/"replay", [(fit, 7, True)])
            sampler = ReplaySampler([dataset], seed=7, burn_in=2, loss_steps=3)
            expected = dataset.load_episode(fit).arrays()
            with patch.object(EpisodeView, "arrays", side_effect=AssertionError("whole episode materialization")):
                window = sampler.window(0, fit, 4)
            self.assertEqual((window.start, window.anchor, window.stop, window.burn_in, window.loss_steps), (2, 4, 7, 2, 3))
            self.assertEqual(set(window.states), set(STATE_ARRAYS))
            self.assertEqual(set(window.transitions), set(TRANSITION_ARRAYS))
            self.assertEqual(window.event_fields, tuple(dataset.manifest["schema"]["event_fields"]))
            for name in STATE_ARRAYS:
                np.testing.assert_array_equal(window.states[name], expected[name][2:8])
                self.assertFalse(window.states[name].flags.writeable)
            for name in TRANSITION_ARRAYS:
                np.testing.assert_array_equal(window.transitions[name], expected[name][2:7])
                self.assertFalse(window.transitions[name].flags.writeable)
            self.assertFalse(window.states["image_valid"][-1, 0])
            self.assertTrue(window.states["image_valid"][-1, 1])
            self.assertTrue(window.states["privileged_validity_mask"][-1].all())
            self.assertAlmostEqual(float(window.transitions["team_reward"][-1]), .999)
            self.assertTrue(window.transitions["terminated"][-1])
            with self.assertRaisesRegex(ValueError, "nonterminal source"):
                sampler.window(0, fit, 7)

    def test_clipping_never_pads_or_crosses_episode_and_prefix_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = identifier("first", "fit"), identifier("second", "fit")
            dataset = make_store(Path(directory)/"replay", [(first, 5, False), (second, 7, False)])
            sampler = ReplaySampler([dataset], seed=8)
            early = sampler.window(0, first, 0)
            late = sampler.window(0, first, 4)
            self.assertEqual((early.start, early.anchor, early.stop), (0, 0, 5))
            self.assertEqual((late.start, late.anchor, late.stop), (0, 4, 5))
            self.assertEqual(late.transitions["episode_id"].tolist(), [first]*5)
            short = sampler.window(0, second, 4, burn_in=1, loss_steps=2)
            full = sampler.window(0, second, 4, burn_in=1, loss_steps=2, full_prefix=True)
            self.assertEqual((short.start, short.stop, short.burn_in), (3, 6, 1))
            self.assertEqual((full.start, full.stop, full.burn_in), (0, 6, 4))
            self.assertEqual(len(short.states["rgb"]), 4)
            self.assertEqual(len(short.transitions["actions"]), 3)
            self.assertEqual(short.transitions["decision_index"].tolist(), [3, 4, 5])

    def test_explicit_window_reads_only_intersecting_chunks_once(self):
        with tempfile.TemporaryDirectory() as directory:
            fit = identifier("bounded", "fit")
            dataset = make_store(Path(directory)/"replay", [(fit, 11, False)], chunk_transitions=2)
            sampler = ReplaySampler([dataset], seed=0, burn_in=1, loss_steps=2)
            dataset._cache.clear()
            with patch.object(dataset, "_chunk", wraps=dataset._chunk) as reader:
                window = sampler.window(0, fit, 5)
            self.assertEqual((window.start, window.stop), (4, 7))
            called = [call.args[0]["path"] for call in reader.call_args_list]
            self.assertEqual(called, [chunk["path"] for chunk in dataset.episodes[0]["chunks"][1:4]])
            self.assertEqual(len(called), len(set(called)))
            self.assertLessEqual(len(dataset._cache), 1)

    def test_uniform_population_is_transitions_not_episode_count(self):
        with tempfile.TemporaryDirectory() as directory:
            short, long = identifier("short", "fit"), identifier("long", "fit")
            dataset = make_store(Path(directory)/"replay", [(short, 2, False), (long, 8, False)])
            sampler = ReplaySampler([dataset], seed=311, burn_in=0, loss_steps=1)
            self.assertEqual(sampler.anchor_count, 10)
            population = sampler._anchors
            self.assertEqual(sum(row[1] == short for row in population), 2)
            self.assertEqual(sum(row[1] == long for row in population), 8)
            # Intercept the final read to count draws without allocating image batches.
            with patch.object(sampler, "window", side_effect=lambda *args: args):
                draws = sampler.sample(5000)
            long_fraction = sum(row[1] == long for row in draws)/len(draws)
            self.assertTrue(.77 < long_fraction < .83)

    def test_fit_and_diagnostic_partitions_have_separate_rng_and_purpose_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            fit, diagnostic = identifier("partition", "fit"), identifier("partition", "diagnostic")
            dataset = make_store(Path(directory)/"replay", [(fit, 5, False), (diagnostic, 5, False)])
            fitting = ReplaySampler([dataset], seed=2, burn_in=0, loss_steps=1)
            reference = ReplaySampler([dataset], seed=2, burn_in=0, loss_steps=1)
            diagnostics = ReplaySampler([dataset], seed=9, partition="diagnostic", burn_in=0, loss_steps=1)
            diagnostics.sample(5)
            self.assertEqual(coordinates(fitting.sample(5)), coordinates(reference.sample(5)))
            self.assertTrue(all(window.episode_id == fit for window in fitting.sample(5)))
            self.assertTrue(all(window.episode_id == diagnostic for window in diagnostics.sample(5)))
            with self.assertRaisesRegex(ValueError, "guarded partition"):
                fitting.window(0, diagnostic, 0)
            with self.assertRaisesRegex(ValueError, "guarded partition"):
                diagnostics.window(0, fit, 0)

    def test_recent_mixture_uses_physical_positions_across_ordered_stores(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new, diagnostic = identifier("old", "fit"), identifier("new", "fit"), identifier("new", "diagnostic")
            first = make_store(Path(directory)/"first", [(old, 4, False)])
            second = make_store(Path(directory)/"second", [(new, 5, False), (diagnostic, 2, False)])
            sampler = ReplaySampler([first, second], seed=6, burn_in=0, loss_steps=1,
                                    recent_fraction=1., recent_transitions=4)
            self.assertEqual(sampler.anchor_count, 9)
            self.assertEqual(sampler.recent_anchor_count, 2)
            self.assertEqual(sampler._recent, ((1, new, 3), (1, new, 4)))
            self.assertEqual({(w.dataset_index, w.episode_id, w.anchor) for w in sampler.sample(30)},
                             {(1, new, 3), (1, new, 4)})
            # Both final physical transitions are diagnostic: never silently fall back.
            with self.assertRaisesRegex(ValueError, "suffix contains no eligible"):
                ReplaySampler([first, second], seed=6, recent_fraction=.5, recent_transitions=2)
            reversed_order = ReplaySampler([second, first], seed=6, recent_fraction=1., recent_transitions=4)
            self.assertEqual(reversed_order._recent, tuple((1, old, index) for index in range(4)))

    def test_sampler_rng_restore_is_exact_and_explicit_windows_do_not_consume_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            fit = identifier("rng", "fit")
            dataset = make_store(Path(directory)/"replay", [(fit, 7, False)])
            sampler = ReplaySampler([dataset], seed=4, recent_fraction=.35, recent_transitions=3,
                                    burn_in=1, loss_steps=1)
            sampler.sample(3)
            state = sampler.state_dict()
            sampler.window(0, fit, 3, full_prefix=True)
            self.assertEqual(state, sampler.state_dict())
            expected = coordinates(sampler.sample(12))
            resumed = ReplaySampler([dataset], seed=999, recent_fraction=.35, recent_transitions=3,
                                    burn_in=1, loss_steps=1)
            resumed.load_state_dict(state)
            self.assertEqual(coordinates(resumed.sample(12)), expected)
            state["rng_state"]["state"]["state"] = 0
            self.assertNotEqual(state["rng_state"], resumed.state_dict()["rng_state"])
            incompatible = ReplaySampler([dataset], seed=4, burn_in=2)
            with self.assertRaisesRegex(ValueError, "cohort or sampling configuration"):
                incompatible.load_state_dict(resumed.state_dict())
            before = resumed.state_dict()
            broken = deepcopy(before)
            broken["rng_state"] = {"bit_generator": "invalid"}
            with self.assertRaisesRegex(ValueError, "invalid sampler RNG"):
                resumed.load_state_dict(broken)
            self.assertEqual(resumed.state_dict(), before)

    def test_resume_identity_survives_relocation_but_rejects_store_order_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            first = make_store(Path(directory)/"first", [(identifier("identity1", "fit"), 3, False)])
            second = make_store(Path(directory)/"second", [(identifier("identity2", "fit"), 4, False)])
            sampler = ReplaySampler([first, second], seed=3)
            shutil.copytree(first.path, Path(directory)/"relocated")
            relocated = ReplaySampler([ReplayDataset(Path(directory)/"relocated"), second], seed=0)
            relocated.load_state_dict(sampler.state_dict())
            self.assertEqual(coordinates(relocated.sample(3)), coordinates(sampler.sample(3)))
            reordered = ReplaySampler([second, first], seed=3)
            with self.assertRaisesRegex(ValueError, "cohort or sampling configuration"):
                reordered.load_state_dict(sampler.state_dict())

    def test_preflight_incompatible_sources_and_duplicate_content_or_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fit = identifier("guard", "fit")
            dataset = make_store(Path(directory)/"first", [(fit, 4, False)])
            preflight = make_store(Path(directory)/"preflight", [(identifier("preflight", "fit"), 4, False)], preflight=True)
            for partition in ("fit", "diagnostic"):
                with self.assertRaisesRegex(ValueError, "excluded from learning"):
                    ReplaySampler([preflight], seed=0, partition=partition)
            with self.assertRaisesRegex(ValueError, "duplicate dataset content"):
                ReplaySampler([dataset, ReplayDataset(dataset.path)], seed=0)
            duplicate_id = make_store(Path(directory)/"duplicate", [(fit, 3, False)])
            with self.assertRaisesRegex(ValueError, "duplicate episode IDs"):
                ReplaySampler([dataset, duplicate_id], seed=0)
            cover = make_store(Path(directory)/"cover", [(identifier("cover", "fit"), 3, False)], family="cover")
            with self.assertRaisesRegex(ValueError, "share task family, run seed, and schema"):
                ReplaySampler([dataset, cover], seed=0)
            for key, value, pattern in (("run_seed", 1, "share task family"),
                                        ("data_source", "imagination", "excluded from learning")):
                original = deepcopy(duplicate_id.manifest["metadata"])
                duplicate_id.manifest["metadata"][key] = value
                with self.assertRaisesRegex(ValueError, pattern):
                    ReplaySampler([dataset, duplicate_id], seed=0)
                duplicate_id.manifest["metadata"] = original

    def test_empty_partition_and_invalid_sampling_arguments_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            fit = identifier("validation", "fit")
            dataset = make_store(Path(directory)/"replay", [(fit, 3, False)])
            with self.assertRaisesRegex(ValueError, "no valid source-image anchors"):
                ReplaySampler([dataset], seed=0, partition="diagnostic")
            invalid = ({"burn_in": -1}, {"loss_steps": 0}, {"recent_fraction": float("nan")},
                       {"recent_fraction": True}, {"recent_transitions": 0}, {"partition": "validation"})
            for arguments in invalid:
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    ReplaySampler([dataset], seed=0, **arguments)
            sampler = ReplaySampler([dataset], seed=0)
            for size in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    sampler.sample(size)
            for arguments in ({"anchor": 3}, {"anchor": 0, "burn_in": -1}, {"anchor": 0, "loss_steps": 0}):
                with self.assertRaises(ValueError):
                    sampler.window(0, fit, **arguments)
            window = sampler.window(0, fit, 1)
            with self.assertRaisesRegex(ValueError, "start <= anchor < stop"):
                replace(window, stop=1)
            with self.assertRaisesRegex(ValueError, "complete typed replay schema"):
                replace(window, states={"rgb": window.states["rgb"]})
            with self.assertRaisesRegex(ValueError, "inconsistent leading lengths"):
                replace(window, states={name: value[:-1] for name, value in window.states.items()})

    def test_real_forward_support_matches_label_denominators_after_burn_in(self):
        import torch
        from world_models.models.world_model import SamplingStreams, SharedWorldModel
        from world_models.training.fitting import LossSupport, forward_windows, loss_support
        from world_models.training.normalization import fit_normalization
        from world_models.training.objective import WorldModelObjective

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                fit = identifier("support", "fit")
                dataset = make_store(Path(directory)/"replay", [(fit, 5, True)])
                sampler = ReplaySampler([dataset], seed=4, burn_in=1, loss_steps=2)
                windows = [sampler.window(0, fit, anchor) for anchor in (1, 3)]
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(43)
                    model = SharedWorldModel()
                objective = WorldModelObjective(model.task, fit_normalization(dataset))
                with torch.no_grad(), patch.object(model.core, "observe", wraps=model.core.observe) as observer:
                    result = forward_windows(model, windows, SamplingStreams.from_seed(17), objective)
                # A truncated context starts at its absolute source clock before
                # the very first posterior draw, then burn-in advances once.
                self.assertEqual(observer.call_args_list[0].args[0].decision_index.tolist(), [0, 2])
                self.assertEqual(observer.call_args_list[1].args[0].decision_index.tolist(), [1, 3])
                expected = LossSupport.from_results([result])
                actual = loss_support(windows, model.task, windows[0].event_fields)
                self.assertEqual(actual.components, expected.components)
                self.assertEqual(actual.fields, expected.fields)
                self.assertEqual(actual.components["rgb"], 5*2*3*96*128)
                self.assertEqual(actual.components["dynamics_kl"], 3)
                self.assertEqual(actual.components["reward"], 4)
                self.assertEqual(actual.components["outcome"], 4)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
