"""Durable P3.5 integration on tiny CPU fixtures, without a native simulator.

The model architecture is unchanged. Research-scale data collection, fitting,
policy training and evaluation are never launched by these numerical fixtures.
"""

from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_model_sampling import identifier, make_store
from tests.test_replay_collection import CountingCoordinator, environment_fixture, scenario_fixture
from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.cycle.collection import collect_policy_block
from world_models.cycle.controller import CycleController, CycleOptions
from world_models.cycle.refresh import ModelRefresher
from world_models.env.coop import TwoPlayerCoopEnv
from world_models.imagination.versions import module_version
from world_models.infrastructure.config import resolve_config, validate_config
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.models.world_model import build_model
from world_models.training.checkpoints import read_checkpoint, save_checkpoint
from world_models.training.fitting import FitSettings
from world_models.training.normalization import fit_normalization
from world_models.training.runner import source_identity


ROOT = Path(__file__).resolve().parents[1]


def controller_settings():
    settings = deepcopy(resolve_config("relay", "B", "pilot", ROOT / "configs").settings)
    settings["experiment"].update(initial_transitions=6, real_budget=12, collection_block=3,
                                  checkpoints=[0, 6, 9, 12])
    settings["world_model"].update(initial_updates=1, online_updates=1, effective_batch=1,
                                   burn_in=0, sequence_length=1)
    settings["mappo"].update(starts=2, horizon=1, epochs=1, recurrent_minibatches=1,
                             initial_local_cycles=1, online_local_cycles=1)
    settings["actor"].update(embedding_size=8, hidden_size=8)
    settings["critic"]["hidden_sizes"] = [16, 16]
    validate_config(settings)
    return settings


class CycleControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.inputs = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.inputs.cleanup)
        cls.root = Path(cls.inputs.name)
        cls.settings = controller_settings()
        cls.dataset = make_store(cls.root / "initial", [(identifier("p35-controller", "fit"), 6, False)])
        cls.normalization = fit_normalization(cls.dataset)
        model = build_model(cls.settings, master_seed=0, variant="B")
        refresh = ModelRefresher(model, cls.normalization, cls.settings, master_seed=0,
                                 fit_settings=FitSettings.from_experiment(cls.settings, microbatch=1,
                                                                         precision="float32"))
        refresh.set_replay([cls.dataset])
        refresh.update()
        common = deepcopy(cls.settings)
        common.pop("variant")
        metadata = {"task_family": "relay", "run_seed": 0, "stage": "initial",
                    "variant": "shared_initial:B,D", "target_stage_updates": 1,
                    "common_config_sha256": canonical_hash(common), "source_sha256": source_identity(),
                    "initial_normalization": {"identity_sha256": cls.normalization.as_dict()["identity_sha256"]},
                    "replays": refresh._replay_identity([cls.dataset])}
        cls.initial_checkpoint = cls.root / "p24.pt"
        save_checkpoint(cls.initial_checkpoint, model=model, optimizer=refresh.optimizer,
                        streams=refresh.streams, sampler=refresh.sampler, update=1, metadata=metadata)
        del model, refresh
        # Same one-update numerical fixture, with a two-update registered target
        # to exercise explicit early stopping. No research metadata is edited.
        cls.short_settings = deepcopy(cls.settings)
        cls.short_settings["world_model"]["initial_updates"] = 2
        common = deepcopy(cls.short_settings)
        common.pop("variant")
        payload = read_checkpoint(cls.initial_checkpoint)
        payload["metadata"].update(target_stage_updates=2, common_config_sha256=canonical_hash(common))
        cls.short_checkpoint = cls.root / "short_p24.pt"
        torch.save(payload, cls.short_checkpoint)
        digest = hashlib.sha256(cls.short_checkpoint.read_bytes()).hexdigest()
        cls.short_checkpoint.with_suffix(".sha256").write_text(f"{digest}  {cls.short_checkpoint.name}\n", encoding="ascii")
        cls.selected_row = {"task": "relay", "seed": 0, "model_updates": 1, "original_target_stage_updates": 2,
            "common_config_sha256": canonical_hash(common), "checkpoint_path": str(cls.short_checkpoint),
            "checkpoint_sha256": digest, "selection_path": str(cls.root / "selection.json"), "selection_sha256": "b"*64}
        del payload
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "run"
        self.calls = []

    def collector(self, *args, **kwargs):
        self.calls.append(kwargs["block_index"])
        return collect_policy_block(*args, env_factory=environment_fixture,
                                    scenario_builder=scenario_fixture, chunk_transitions=2, **kwargs)

    def prepare(self, **kwargs):
        arguments = dict(settings=deepcopy(self.settings), master_seed=0, initial_dataset=self.dataset,
                         normalization=self.normalization, initial_checkpoint=self.initial_checkpoint,
                         device="cpu", options=CycleOptions(precision="float32", model_microbatch=1,
                                                            checkpoint_cycles=1, checkpoint_model_updates=1),
                         collector=self.collector)
        arguments.update(kwargs)
        return CycleController.prepare(self.directory, **arguments)

    def resume(self, collector=None):
        return CycleController.resume(self.directory, device="cpu", collector=collector or self.collector)

    def reach_collection(self, controller):
        report = controller.run(max_units=2)
        self.assertEqual((report["phase"], report["block_index"], report["real_joint_transitions"]),
                         ("collect", 1, 6))
        return report

    def test_complete_actual_pipeline_stop_resume_counts_weights_and_evaluation_exports(self):
        controller = self.prepare()
        initial_actor = module_version(controller.actor)
        initial_model = module_version(controller.model)
        frozen_normalization = controller.normalization.as_dict()
        self.assertEqual((controller.refresher.model_updates, controller.learner.cycles_completed), (1, 0))
        self.assertEqual(self.calls, [])
        report = controller.run(max_units=1)
        self.assertEqual((report["phase"], report["local_cycles_completed"], report["real_joint_transitions"]),
                         ("local", 1, 6))
        self.assertNotEqual(initial_actor, module_version(controller.actor))
        self.assertEqual(initial_model, module_version(controller.model))
        saved_actor = module_version(controller.actor)
        del controller
        gc.collect()
        controller = self.resume()
        self.assertEqual(module_version(controller.actor), saved_actor)
        self.assertEqual(controller.learner.cycles_completed, 1)
        report = controller.run()
        self.assertEqual(report["status"], "complete")
        self.assertEqual((report["real_joint_transitions"], report["model_updates"],
                          report["local_cycles_completed"], report["actor_optimizer_steps"],
                          report["critic_optimizer_steps"]), (12, 3, 3, 3, 3))
        self.assertEqual(self.calls, [0, 1])
        self.assertNotEqual(initial_model, module_version(controller.model))
        self.assertEqual(frozen_normalization, controller.normalization.as_dict())
        self.assertEqual(sum(data.manifest["transition_count"] for data in controller.datasets), 12)
        self.assertEqual(report["local_player_decisions"], 2 * report["imagined_joint_steps"])
        exports = json.loads((self.directory / "evaluation_checkpoints.json").read_text())
        self.assertEqual([row["real_joint_transitions"] for row in exports], [0, 6, 9, 12])
        for row in exports:
            state, metadata = load_cycle_checkpoint(self.directory / row["path"])
            self.assertEqual(set(state), {"actor"})
            self.assertFalse(metadata["evaluation_executed"])
            self.assertFalse(metadata["phase3_acceptance"])
            self.assertEqual(metadata["real_joint_transitions"], row["real_joint_transitions"])
        before = deepcopy(controller.state)
        completed_again = controller.run()
        self.assertEqual(completed_again["units_executed"], 0)
        self.assertEqual(self.calls, [0, 1])
        self.assertEqual(before["real_joint_transitions"], controller.state["real_joint_transitions"])

    def test_sealed_collection_before_return_failure_is_attached_without_double_collection(self):
        controller = self.prepare()
        self.reach_collection(controller)
        def fail_after_seal(*args, **kwargs):
            self.collector(*args, **kwargs)
            raise RuntimeError("fixture lost return after sealing")
        controller.collector = fail_after_seal
        with self.assertRaisesRegex(RuntimeError, "lost return"):
            controller.run(max_units=1)
        with self.assertRaisesRegex(RuntimeError, "resume"):
            controller.run(max_units=1)
        self.assertEqual(self.calls, [0])
        del controller
        gc.collect()
        controller = self.resume()
        report = controller.run(max_units=1)
        self.assertEqual((report["phase"], report["real_joint_transitions"], report["model_updates"]),
                         ("refresh", 9, 1))
        self.assertEqual(self.calls, [0])
        self.assertEqual(len(controller.datasets), 2)

    def test_sealed_collection_after_checkpoint_failure_is_attached_without_repeat(self):
        controller = self.prepare()
        self.reach_collection(controller)
        with patch("world_models.cycle.controller.save_cycle_checkpoint", side_effect=OSError("fixture checkpoint disk")):
            with self.assertRaisesRegex(OSError, "checkpoint disk"):
                controller.run(max_units=1)
        with self.assertRaisesRegex(RuntimeError, "resume"):
            controller.run(max_units=1)
        self.assertEqual(self.calls, [0])
        del controller
        gc.collect()
        controller = self.resume()
        self.assertEqual(controller.state["real_joint_transitions"], 6)
        report = controller.run(max_units=1)
        self.assertEqual((report["phase"], report["real_joint_transitions"]), ("refresh", 9))
        self.assertEqual(self.calls, [0])
        coherent = deepcopy(controller.state)
        controller.state["phase"] = "local"
        with self.assertRaises(ValueError):
            controller._validate_state()
        controller.state = coherent
        controller._validate_state()

    def test_interrupted_real_collection_preserves_debits_and_refuses_automatic_retry(self):
        controller = self.prepare()
        self.reach_collection(controller)
        def fail_env(family, **kwargs):
            return TwoPlayerCoopEnv(family, seed=kwargs["seed"], scenario_manifest=kwargs["scenario_manifest"],
                                   coordinator=CountingCoordinator(kwargs["work_root"], failure_at=2))
        def fail_collection(*args, **kwargs):
            self.calls.append(kwargs["block_index"])
            return collect_policy_block(*args, env_factory=fail_env, scenario_builder=scenario_fixture, **kwargs)
        controller.collector = fail_collection
        with self.assertRaises(RuntimeError):
            controller.run(max_units=1)
        with self.assertRaisesRegex(RuntimeError, "resume"):
            controller.run(max_units=1)
        failure_path = self.directory / "real_replay/block_0001/collection_error.json"
        failure = json.loads(failure_path.read_text())
        self.assertEqual(failure["interaction_accounting"]["published_real_joint_transitions"], 1)
        self.assertEqual(failure["interaction_accounting"]["failed_dispatch_attempts"], 1)
        self.assertEqual(self.calls, [0])
        del controller
        gc.collect()
        controller = self.resume()
        with self.assertRaisesRegex(RuntimeError, "reconciled"):
            controller.run(max_units=1)
        self.assertEqual(self.calls, [0])
        self.assertEqual(controller.state["real_joint_transitions"], 6)
        self.assertEqual(json.loads(failure_path.read_text()), failure)

    def test_preparation_rejects_d_wrong_seed_or_shared_checkpoint_budget_without_creating_run(self):
        settings_d = deepcopy(self.settings)
        settings_d["variant"] = deepcopy(resolve_config("relay", "D", "pilot", ROOT / "configs").settings["variant"])
        wrong_budget = deepcopy(self.settings)
        wrong_budget["world_model"]["initial_updates"] = 2
        for arguments in ({"settings": settings_d}, {"master_seed": 100}, {"settings": wrong_budget}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.prepare(**arguments)
            self.assertFalse(self.directory.exists())

    def test_shortened_selection_preserves_adam_counts_through_refresh_resume_and_completion(self):
        with patch("world_models.training.selection.resolve_selected_case", return_value=self.selected_row):
            controller = self.prepare(settings=self.short_settings, initial_checkpoint=None,
                                      initial_selection=self.selected_row["selection_path"])
            self.assertEqual(controller.schedule.initial_model_updates, 1)
            self.assertEqual(controller.manifest["registered_schedule"]["initial_model_updates"], 2)
            self.assertEqual(controller.manifest["initial_selection"]["model_updates"], 1)
            self.assertFalse(controller.manifest["research_quality_acceptance"])
            original = read_checkpoint(self.short_checkpoint)["optimizer"]["state"]
            retained = controller.refresher.optimizer.state_dict()["state"]
            self.assertEqual(original.keys(), retained.keys())
            for key in original:
                for moment in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(original[key][moment], retained[key][moment]))
            self.reach_collection(controller)
            report = controller.run(max_units=2)
            self.assertEqual((report["phase"], report["model_updates"]), ("local", 2))
            del controller
            gc.collect()
            controller = self.resume()
            self.assertEqual((controller.refresher.initial_model_updates, controller.refresher.model_updates), (1, 2))
            report = controller.run()
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["model_updates"], 3)
            self.assertEqual(report["initial_model_updates"], 1)
            self.assertEqual(report["registered_initial_model_updates"], 2)
            self.assertEqual(report["target_model_updates"], 3)
            self.assertEqual(self.calls, [0, 1])
            self.assertEqual(controller.settings, self.short_settings)

    def test_short_checkpoint_requires_selection_and_rejects_changed_selection_on_resume(self):
        with self.assertRaises(ValueError):
            self.prepare(settings=self.short_settings, initial_checkpoint=self.short_checkpoint)
        self.assertFalse(self.directory.exists())
        with patch("world_models.training.selection.resolve_selected_case", return_value=self.selected_row):
            controller = self.prepare(settings=self.short_settings, initial_checkpoint=None,
                                      initial_selection=self.selected_row["selection_path"])
        del controller
        gc.collect()
        changed = {**self.selected_row, "selection_sha256": "c"*64}
        with patch("world_models.training.selection.resolve_selected_case", return_value=changed):
            with self.assertRaisesRegex(ValueError, "selection"):
                self.resume()

    def test_selection_rejects_unsigned_or_wrong_checkpoint_before_creating_run(self):
        unsigned = Path(self.temp.name) / "selection.json"
        unsigned.write_text("{}", encoding="utf-8")
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.prepare(settings=self.short_settings, initial_checkpoint=None, initial_selection=unsigned)
        wrong = {**self.selected_row, "checkpoint_path": str(self.initial_checkpoint)}
        with patch("world_models.training.selection.resolve_selected_case", return_value=wrong):
            with self.assertRaises(ValueError):
                self.prepare(settings=self.short_settings, initial_checkpoint=None, initial_selection=unsigned)
        self.assertFalse(self.directory.exists())

    def test_invalid_run_unit_limits_and_stop_do_not_collect(self):
        controller = self.prepare()
        coherent = deepcopy(controller.state)
        for changes in ({"phase": "refresh"}, {"phase": "complete"},
                        {"real_joint_transitions": 7}, {"local_player_decisions": 1},
                        {"local_cycles_in_block": 1}, {"model_updates_in_block": 1}):
            controller.state = {**deepcopy(coherent), **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                controller._validate_state()
        controller.state = coherent
        for maximum in (0, -1, True, "1", 1.0):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                controller.run(max_units=maximum)
        controller.request_stop()
        result = controller.run()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["units_executed"], 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(controller.learner.cycles_completed, 0)

    def test_durable_stop_request_file_is_consumed_at_a_safe_checkpoint(self):
        controller = self.prepare()
        stop_file = self.directory / "stop.request"
        stop_file.write_text("stop at the next complete unit\n", encoding="utf-8")
        result = controller.run()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["units_executed"], 0)
        self.assertEqual(result["local_cycles_completed"], 0)
        self.assertEqual(result["real_joint_transitions"], 6)
        self.assertEqual(self.calls, [])
        self.assertFalse(stop_file.exists())
        self.assertTrue((controller.last_checkpoint / "manifest.sha256").is_file())
        del controller
        gc.collect()
        controller = self.resume()
        result = controller.run(max_units=1)
        self.assertEqual(result["local_cycles_completed"], 1)
        self.assertEqual(result["real_joint_transitions"], 6)

    def test_amended_budget_preserves_parent_optimizer_then_trains_and_resumes_actual_fixture(self):
        # Profile schema/allowlist checks are exercised in test_cycle_study.
        # Tiny synthetic settings here exercise the real controller/optimizer
        # bridge without consuming the research replay or launching an engine.
        settings = deepcopy(self.short_settings)
        settings["experiment"].update(real_budget=9, checkpoints=[0, 6, 9])
        settings["world_model"]["online_updates"] = 2
        settings["mappo"]["initial_local_cycles"] = 2
        amendment = {"name": "synthetic_fixture_only"}
        with patch("world_models.cycle.study.parent_settings_for_study", return_value=self.short_settings), \
             patch("world_models.training.selection.resolve_selected_case", return_value=self.selected_row):
            controller = self.prepare(settings=settings, initial_checkpoint=None,
                                      initial_selection="fixture-selection.json", study_amendment=amendment)
            initial = read_checkpoint(self.short_checkpoint)["optimizer"]
            resumed_optimizer = controller.refresher.optimizer.state_dict()
            self.assertEqual(canonical_hash(initial["param_groups"]), canonical_hash(resumed_optimizer["param_groups"]))
            for key, values in initial["state"].items():
                for name, value in values.items():
                    self.assertTrue(torch.equal(value, resumed_optimizer["state"][key][name]))
            self.assertEqual(controller.schedule.initial_model_updates, 1)
            self.assertEqual(controller.manifest["parent_schedule"]["real_budget"], 12)
            controller.run(max_units=1)
            trained_actor = module_version(controller.actor)
            del controller
            gc.collect()
            controller = self.resume()
            self.assertEqual(module_version(controller.actor), trained_actor)
            self.assertEqual(controller.learner.cycles_completed, 1)
            self.assertEqual(controller.refresher.model_updates, 1)
            report = controller.run()
            self.assertEqual(report["status"], "complete")
            self.assertEqual((report["real_joint_transitions"], report["model_updates"], report["local_cycles_completed"]),
                             (9, 3, 3))
            self.assertEqual(self.calls, [0])


if __name__ == "__main__":
    unittest.main()
