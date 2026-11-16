"""P3.5 persistent refresh checks on tiny synthetic real-format fixtures."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_model_sampling import identifier, make_store
from tests.test_replay_storage import episode_fixture, write_episode
from world_models.cycle.refresh import ModelRefresher
from world_models.data.replay import ReplayDataset, ReplayDatasetWriter, canonical_hash
from world_models.infrastructure.config import resolve_config
from world_models.models.world_model import SharedWorldModel
from world_models.training.checkpoints import read_checkpoint, save_checkpoint
from world_models.training.fitting import FitSettings
from world_models.training.normalization import fit_normalization
from world_models.training.runner import source_identity


ROOT = Path(__file__).resolve().parents[1]


class CycleRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fit = identifier("cycle-refresh-fit", "fit")
        self.diagnostic = identifier("cycle-refresh-diagnostic", "diagnostic")
        self.dataset = make_store(self.root / "initial", [(self.fit, 3, False),
                                                         (self.diagnostic, 2, False)])
        self.normalization = fit_normalization(self.dataset)
        self.settings = resolve_config("relay", "B", "pilot", ROOT / "configs").settings
        self.fit_settings = FitSettings(effective_batch=1, microbatch=1, burn_in=0,
                                        loss_steps=1, precision="float32")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(305)
            self.model = SharedWorldModel()

    def refresher(self, **kwargs):
        return ModelRefresher(self.model, self.normalization, self.settings, master_seed=0,
                              fit_settings=self.fit_settings, **kwargs)

    def online(self, variant="B", suffix="online"):
        writer = ReplayDatasetWriter(self.root / suffix, {
            "task_family": "relay", "run_seed": 0, "source": "real_environment",
            "budget_scope": "online_collection", "target_transitions": 3,
            "training_variant": variant}, chunk_transitions=2)
        write_episode(writer, episode_fixture(3), episode_id=identifier(suffix, "fit"))
        writer.seal()
        return ReplayDataset(self.root / suffix)

    def parent_metadata(self, refresher):
        common = deepcopy(self.settings)
        common.pop("variant")
        return {"task_family": "relay", "run_seed": 0,
                "common_config_sha256": canonical_hash(common), "source_sha256": source_identity(),
                "stage": "initial", "variant": "shared_initial:B,D",
                "initial_normalization": {"identity_sha256": self.normalization.as_dict()["identity_sha256"]},
                "replays": refresher._replay_identity([self.dataset])}

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_one_real_window_update_has_finite_metrics_fp32_moments_and_no_policy_or_global_rng(self):
        refresh = self.refresher()
        refresh.set_replay([self.dataset])
        self.model.eval()
        self.model.decoder.train()
        modes = [module.training for module in self.model.modules()]
        rng = torch.random.get_rng_state().clone()
        before = self.model.decoder.agent_queries.detach().clone()
        # Count actual fitting windows and prove no diagnostic episode is selected.
        original = refresh.sampler.sample
        sampled = []
        def sample(size):
            values = original(size)
            sampled.extend(values)
            return values
        with patch.object(refresh.sampler, "sample", side_effect=sample):
            metrics = refresh.update()
        self.assertEqual([window.episode_id for window in sampled], [self.fit])
        self.assertEqual((metrics["model_updates"], metrics["online_model_updates"], metrics["sequences"]), (1, 1, 1))
        self.assertEqual(metrics["real_joint_transitions_collected"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["total"])))
        self.assertFalse(torch.equal(before, self.model.decoder.agent_queries))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertTrue(refresh.optimizer.state)
        for state in refresh.optimizer.state.values():
            self.assertEqual(float(state["step"]), 1.)
            self.assertEqual(state["exp_avg"].dtype, torch.float32)

    def test_p24_checkpoint_preserves_adamw_moments_and_absolute_step_count(self):
        initial = self.refresher()
        initial.set_replay([self.dataset])
        initial.update()
        path = self.root / "p24.pt"
        save_checkpoint(path, model=self.model, optimizer=initial.optimizer, streams=initial.streams,
                        sampler=initial.sampler, update=1, metadata=self.parent_metadata(initial))
        expected_optimizer = initial.optimizer.state_dict()
        payload = read_checkpoint(path)
        refresh = self.refresher(initial_checkpoint=payload)
        self.assertEqual((refresh.initial_model_updates, refresh.total_model_updates), (1, 1))
        self.assert_tree_equal(expected_optimizer, refresh.optimizer.state_dict())
        refresh.set_replay([self.dataset, self.online()])
        self.assertEqual(refresh.update()["model_updates"], 2)
        self.assertTrue(all(float(state["step"]) == 2 for state in refresh.optimizer.state.values()))
        # The run-level checkpoint owns current model weights. Reattaching the
        # initial optimizer would require old weights, so resume instead creates
        # a pristine adapter and adopts saved counters together with its moments.
        saved = refresh.state_dict()
        resumed = self.refresher()
        resumed.set_replay(refresh.datasets)
        resumed.load_state_dict(saved)
        self.assertEqual((resumed.initial_model_updates, resumed.total_model_updates), (1, 2))
        self.assert_tree_equal(refresh.optimizer.state_dict(), resumed.optimizer.state_dict())

    def test_exact_next_update_resume_matches_uninterrupted_model_optimizer_and_private_rng(self):
        first = self.refresher()
        first.set_replay([self.dataset])
        first.update()
        saved_weights = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        state = first.state_dict()
        expected_metrics = first.update()
        expected_weights = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        expected_optimizer = first.optimizer.state_dict()
        self.model.load_state_dict(saved_weights)
        resumed = self.refresher()
        resumed.set_replay([self.dataset])
        resumed.load_state_dict(state)
        self.assertEqual(expected_metrics, resumed.update())
        self.assert_tree_equal(expected_weights, self.model.state_dict())
        self.assert_tree_equal(expected_optimizer, resumed.optimizer.state_dict())
        self.assert_tree_equal(first.streams.state_dict(), resumed.streams.state_dict())
        self.assertEqual(first.sampler.state_dict(), resumed.sampler.state_dict())

    def test_cohort_expansion_preserves_rng_and_uses_half_recent_fit_sampling(self):
        refresh = self.refresher()
        refresh.set_replay([self.dataset])
        refresh.sampler.sample(5)
        before = refresh.sampler.state_dict()["rng_state"]
        online = self.online()
        refresh.set_replay([self.dataset, online])
        self.assertEqual(before, refresh.sampler.state_dict()["rng_state"])
        self.assertEqual(refresh.sampler.recent_fraction, .5)
        self.assertEqual(refresh.sampler.partition, "fit")
        self.assertEqual(refresh.sampler.recent_transitions, 100000)
        self.assertEqual(refresh.total_model_updates, 0)
        with self.assertRaisesRegex(ValueError, "preserve and extend"):
            refresh.set_replay([self.dataset])
        self.assertEqual(refresh.datasets, (self.dataset, online))

    def test_normalization_is_frozen_to_initial_fit_episodes_and_owned_copy(self):
        refresh = self.refresher()
        original = refresh.normalization.as_dict()
        self.normalization.provenance["external_mutation"] = True
        refresh.set_replay([self.dataset, self.online()])
        self.assertEqual(refresh.normalization.as_dict(), original)
        refresh.normalization.provenance["mutation"] = True
        with self.assertRaisesRegex(ValueError, "normalization was mutated"):
            refresh.update()

    def test_rejects_wrong_branch_preflight_synthetic_diagnostic_geometry_and_seed(self):
        refresh = self.refresher()
        with self.assertRaisesRegex(ValueError, "training_variant"):
            refresh.set_replay([self.dataset, self.online("D")])
        preflight = make_store(self.root / "preflight", [(identifier("preflight", "fit"), 2, False)], preflight=True)
        with self.assertRaises(ValueError):
            refresh.set_replay([preflight])
        with self.assertRaisesRegex(ValueError, "another task or training seed"):
            source = self.dataset.manifest["metadata"]
            source["run_seed"] = 1
            try:
                refresh.set_replay([self.dataset])
            finally:
                source["run_seed"] = 0
        refresh.set_replay([self.dataset])
        self.assertEqual({item[1] for item in refresh.sampler._anchors}, {self.fit})

    def test_invalid_resume_is_transactional_and_checks_moments_model_normalizer_and_source(self):
        refresh = self.refresher()
        refresh.set_replay([self.dataset])
        good = refresh.state_dict()
        for mutation in (
            lambda state: state.update(total_model_updates=-1),
            lambda state: state.update(model_version="0" * 64),
            lambda state: state["identity"].update(source_sha256={}),
            lambda state: state["identity"]["normalization"].update(identity_sha256="0" * 64),
            lambda state: state["streams"].update(posterior=torch.tensor([], dtype=torch.uint8)),
            lambda state: state["sampler"].update(cohort_identity="0" * 64),
            lambda state: state["optimizer"]["param_groups"][0].update(lr=0.2),
        ):
            state = deepcopy(good)
            mutation(state)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                refresh.load_state_dict(state)
            self.assert_tree_equal(good, refresh.state_dict())

    def test_snapshot_is_owned_cpu_tensors_and_round_trips_restricted_loader(self):
        refresh = self.refresher()
        refresh.set_replay([self.dataset])
        state = refresh.state_dict()
        path = self.root / "state.pt"
        torch.save(state, path)
        loaded = torch.load(path, weights_only=True)
        self.assert_tree_equal(state, loaded)
        state["streams"]["prior"].zero_()
        state["identity"]["fit_settings"]["microbatch"] = 100
        self.assertFalse(torch.equal(state["streams"]["prior"], refresh.streams.state_dict()["prior"]))
        self.assertEqual(refresh.fit_settings.microbatch, 1)

    def test_branch_fitting_streams_are_independent_and_initial_model_updates_cannot_reset_moments(self):
        left = self.refresher()
        settings = resolve_config("relay", "D", "pilot", ROOT / "configs").settings
        right = ModelRefresher(self.model, self.normalization, settings, master_seed=0,
                               fit_settings=self.fit_settings)
        self.assertNotEqual(left.seed, right.seed)
        self.assertFalse(torch.equal(left.streams.prior.get_state(), right.streams.prior.get_state()))
        with self.assertRaisesRegex(ValueError, "optimizer moments"):
            self.refresher(initial_model_updates=3750)

    def test_parent_checkpoint_provenance_and_already_loaded_weights_are_required(self):
        refresh = self.refresher()
        parent = {"model": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                  "optimizer": refresh.optimizer.state_dict(), "metadata": self.parent_metadata(refresh), "update": 0}
        for name, changed in (("task_family", "cover"), ("run_seed", 1), ("source_sha256", {}),
                              ("stage", "overfit"), ("variant", "B"),
                              ("initial_normalization", {"identity_sha256": "0" * 64})):
            metadata = deepcopy(parent["metadata"])
            metadata[name] = changed
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.refresher(initial_checkpoint={**parent, "metadata": metadata})
        parent["model"] = dict(parent["model"])
        key = next(iter(parent["model"]))
        parent["model"][key] = parent["model"][key].clone() + 1
        with self.assertRaisesRegex(ValueError, "exact P2.4 checkpoint model weights"):
            self.refresher(initial_checkpoint=parent)

    def test_update_without_replay_and_invalid_settings_or_frozen_parameters_rejected(self):
        with self.assertRaisesRegex(ValueError, "set_replay"):
            self.refresher().update()
        for kwargs in ({"precision": "float16"}, {"initial_model_updates": -1},
                       {"fit_settings": object()}):
            arguments = {"fit_settings": self.fit_settings, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelRefresher(self.model, self.normalization, self.settings, master_seed=0, **arguments)
        parameter = next(self.model.parameters())
        parameter.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "refresh gradients"):
            self.refresher()

    def test_mutating_resolved_settings_cannot_change_fitting_after_initialization(self):
        refresh = self.refresher()
        refresh.settings["world_model"]["recent_fraction"] = 0.
        with self.assertRaisesRegex(ValueError, "settings were mutated"):
            refresh.set_replay([self.dataset])

    def test_failed_update_clears_gradients_preserves_modes_and_counter(self):
        refresh = self.refresher()
        refresh.set_replay([self.dataset])
        self.model.eval()
        before = [module.training for module in self.model.modules()]
        with patch("world_models.cycle.refresh.optimizer_update", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                refresh.update()
        self.assertEqual(refresh.total_model_updates, 0)
        self.assertEqual(before, [module.training for module in self.model.modules()])
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))


if __name__ == "__main__":
    unittest.main()
