"""P3.5 real collector invariants using counted CPU coordinator fixtures only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tests.test_replay_collection import CountingCoordinator, environment_fixture, scenario_fixture
from world_models.cycle.collection import (
    COLLECTOR_KIND, OnlineReplayDatasetWriter, collect_policy_block,
)
from world_models.data.ledger import file_sha256
from world_models.data.replay import ReplayDataset
from world_models.env.coop import TwoPlayerCoopEnv
from world_models.imagination.versions import module_version
from world_models.infrastructure.config import resolve_config
from world_models.policies import LocalActor, PolicyConfig


def fixture_settings(*, variant="B", total=10):
    settings = deepcopy(resolve_config("relay", variant, "pilot").settings)
    settings["experiment"].update(initial_transitions=10, collection_block=total,
                                  real_budget=10+2*total, checkpoints=[0, 10, 10+total, 10+2*total])
    settings["actor"].update(embedding_size=8, hidden_size=8)
    settings["critic"]["hidden_sizes"] = [16, 16]
    return settings


class CycleCollectionTests(unittest.TestCase):
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
        self.settings = fixture_settings()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(355)
            self.actor = LocalActor(PolicyConfig.from_experiment(self.settings))

    def collect(self, name="block", **kwargs):
        options = dict(actor=self.actor, settings=self.settings, block_index=0,
                       env_factory=environment_fixture, scenario_builder=scenario_fixture,
                       chunk_transitions=3)
        options.update(kwargs)
        return collect_policy_block("relay", 0, self.settings["variant"]["name"],
                                    Path(self.directory.name)/name, **options)

    def test_exact_block_debits_and_readable_online_real_replay(self):
        report = self.collect()
        root = Path(self.directory.name)/"block"
        self.assertEqual(report["status"], "complete_online_replay")
        self.assertEqual(report["published_joint_transitions"], 10)
        self.assertEqual(report["real_ppo_updates"], 0)
        self.assertEqual(report["actor_version_before"], report["actor_version_after"])
        self.assertEqual(report["dataset_manifest_sha256"], file_sha256(root/"replay/manifest.json"))
        accounting = report["interaction_accounting"]
        self.assertEqual(accounting["published_real_joint_transitions"], 10)
        self.assertEqual(accounting["replay_committed_joint_transitions"], 10)
        self.assertEqual(accounting["physical_joint_decisions_including_initialization"],
                         10+accounting["initialization_joint_decisions"])
        self.assertEqual(accounting["collector_counts"], {COLLECTOR_KIND: 10})
        self.assertEqual(accounting["failed_dispatch_attempts"], 0)
        self.assertEqual(accounting["purpose"], "online_local_learning_replay")
        reader = ReplayDataset(root/"replay", verify=True)
        self.assertEqual(reader.manifest["metadata"]["training_variant"], "B")
        self.assertFalse(reader.manifest["metadata"]["branch_data_shared"])
        self.assertTrue(reader.manifest["metadata"]["learning_eligible"])
        self.assertEqual(reader.manifest["metadata"]["data_source"], "real_environment")
        self.assertTrue(all(ep["variant"] == "B" and ep["collector_kind"] == COLLECTOR_KIND
                            for ep in reader.episodes))
        self.assertIn("collection_cutoff", {ep["end_reason"] for ep in reader.episodes})
        self.assertFalse((root/"bindings").exists())

    def test_only_own_images_previous_executed_actions_and_own_memory_reach_actor(self):
        calls = []
        def hook(_module, arguments):
            self.assertEqual(len(arguments), 3)
            calls.append(tuple(arg.detach().clone() for arg in arguments))
        handle = self.actor.register_forward_pre_hook(hook)
        try:
            self.collect()
        finally:
            handle.remove()
        reader = ReplayDataset(Path(self.directory.name)/"block/replay")
        index = 0
        for ep in reader.episodes:
            data = reader.load_episode(ep["episode_id"]).arrays()
            for step in range(ep["valid_length"]):
                rgb, previous, hidden = calls[index]
                self.assertEqual(rgb.dtype, torch.uint8)
                self.assertEqual(tuple(rgb.shape), (2, 3, 96, 128))
                self.assertTrue(torch.equal(rgb, torch.from_numpy(data["rgb"][step].copy())))
                if step == 0:
                    self.assertTrue(torch.equal(previous, torch.full((2,), 12)))
                    self.assertEqual(hidden.count_nonzero().item(), 0)
                else:
                    self.assertTrue(torch.equal(previous, torch.tensor(data["actions"][step-1].copy(), dtype=torch.long)))
                index += 1
        self.assertEqual(index, 10)

    def test_independent_reproducible_agent_streams_preserve_global_rng_and_weights_and_modes(self):
        self.actor.train()
        self.actor.encoder[0].eval()
        modes = [module.training for module in self.actor.modules()]
        weights = module_version(self.actor)
        global_rng = torch.random.get_rng_state().clone()
        first = self.collect("first")
        second = self.collect("second")
        self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
        self.assertEqual(weights, module_version(self.actor))
        self.assertEqual(modes, [module.training for module in self.actor.modules()])
        roots = [Path(self.directory.name)/name for name in ("first", "second")]
        specs = [json.loads((root/"collection_spec.json").read_text()) for root in roots]
        self.assertNotEqual(*specs[0]["actor_action_seeds"])
        for a, b in zip(ReplayDataset(roots[0]/"replay").episodes, ReplayDataset(roots[1]/"replay").episodes):
            self.assertEqual(a["episode_id"], b["episode_id"])
            self.assertEqual(a["partition"], b["partition"])
            aa = ReplayDataset(roots[0]/"replay").load_episode(a["episode_id"]).arrays()
            bb = ReplayDataset(roots[1]/"replay").load_episode(b["episode_id"]).arrays()
            self.assertEqual(aa["actions"].tolist(), bb["actions"].tolist())
        self.assertEqual(first["collector_counts"], second["collector_counts"])

    def test_completed_or_failed_output_cannot_be_retried_or_overwritten(self):
        self.collect()
        ledger = Path(self.directory.name)/"block/training_interactions.jsonl"
        before = ledger.read_bytes()
        with self.assertRaises(FileExistsError):
            self.collect()
        self.assertEqual(before, ledger.read_bytes())

    def test_consumed_worker_failure_is_charged_without_fabricated_replay_or_retry(self):
        def fail_env(family, **kwargs):
            return TwoPlayerCoopEnv(family, seed=kwargs["seed"], scenario_manifest=kwargs["scenario_manifest"],
                                    coordinator=CountingCoordinator(kwargs["work_root"], failure_at=2))
        with self.assertRaises(RuntimeError):
            self.collect(env_factory=fail_env)
        root = Path(self.directory.name)/"block"
        failure = json.loads((root/"collection_error.json").read_text())
        accounting = failure["interaction_accounting"]
        self.assertEqual(accounting["published_real_joint_transitions"], 1)
        self.assertEqual(accounting["failed_dispatch_attempts"], 1)
        self.assertEqual(accounting["physical_joint_decisions_including_initialization"], 3)
        self.assertEqual(accounting["replay_committed_joint_transitions"], 0)
        self.assertFalse((root/"replay/manifest.json").exists())
        with self.assertRaises(FileExistsError):
            self.collect()

    def test_storage_failure_retains_published_debit(self):
        class FailingWriter(OnlineReplayDatasetWriter):
            def append(self, step):
                raise OSError("fixture disk full")
        with self.assertRaises(OSError):
            self.collect(writer_factory=FailingWriter)
        failure = json.loads((Path(self.directory.name)/"block/collection_error.json").read_text())
        accounting = failure["interaction_accounting"]
        self.assertEqual(accounting["published_real_joint_transitions"], 1)
        self.assertEqual(accounting["failed_dispatch_attempts"], 0)
        self.assertEqual(accounting["replay_committed_joint_transitions"], 0)
        self.assertFalse(failure["unreconciled"])

    def test_train_provenance_and_selected_template_are_enforced_before_any_physics(self):
        for mutation in ({"purpose": "validation"}, {"split": "test"},
                         {"template_id": "relay_validation_00"}, {"learning_eligible": False}):
            def bad_scenario(*args, **kwargs):
                scenario = scenario_fixture(*args, **kwargs)
                scenario["split_provenance"].update(mutation)
                return scenario
            name = "bad_" + str(len(list(Path(self.directory.name).iterdir())))
            with self.assertRaises(ValueError):
                self.collect(name, scenario_builder=bad_scenario)
            failure = json.loads((Path(self.directory.name)/name/"collection_error.json").read_text())
            self.assertEqual(failure["interaction_accounting"]["physical_joint_decisions_including_initialization"], 0)

    def test_exact_budget_task_branch_and_index_rejected_before_output_creation(self):
        for changes in ({"total_transitions": 9}, {"total_transitions": True},
                        {"block_index": -1}, {"block_index": 2}, {"block_index": True}):
            with self.assertRaises(ValueError):
                self.collect(**changes)
            self.assertFalse((Path(self.directory.name)/"block").exists())
        wrong = deepcopy(self.settings)
        wrong["actor"]["hidden_size"] += 1
        with self.assertRaises(ValueError):
            self.collect(settings=wrong)

    def test_next_block_uses_fresh_episode_namespace_memory_and_rng(self):
        self.collect("first", block_index=0)
        self.collect("second", block_index=1)
        readers = [ReplayDataset(Path(self.directory.name)/name/"replay") for name in ("first", "second")]
        self.assertTrue(set(ep["episode_id"] for ep in readers[0].episodes).isdisjoint(
            ep["episode_id"] for ep in readers[1].episodes))
        specs = [json.loads((Path(self.directory.name)/name/"collection_spec.json").read_text())
                 for name in ("first", "second")]
        self.assertNotEqual(specs[0]["collection_seed"], specs[1]["collection_seed"])
        self.assertNotEqual(specs[0]["actor_action_seeds"], specs[1]["actor_action_seeds"])

    def test_resource_guard_prevents_dispatch_and_keeps_zero_debit_failure(self):
        def deny():
            raise RuntimeError("fixture resource unavailable")
        with self.assertRaisesRegex(RuntimeError, "resource unavailable"):
            self.collect(resource_guard=deny)
        failure = json.loads((Path(self.directory.name)/"block/collection_error.json").read_text())
        self.assertEqual(failure["interaction_accounting"]["physical_joint_decisions_including_initialization"], 0)

    def test_real_success_on_final_budget_decision_takes_precedence_over_external_cutoff(self):
        self.settings = fixture_settings(total=4)
        self.collect()
        reader = ReplayDataset(Path(self.directory.name)/"block/replay")
        self.assertEqual(len(reader.episodes), 1)
        self.assertEqual(reader.episodes[0]["end_reason"], "success")
        data = reader.load_episode(reader.episodes[0]["episode_id"]).arrays()
        self.assertTrue(data["terminated"][-1])
        self.assertFalse(data["truncated"][-1])

    def test_changed_task_reward_is_rejected_before_output_or_physics(self):
        changed = deepcopy(self.settings)
        changed["task"]["success_reward"] += 1
        with self.assertRaisesRegex(ValueError, "task rules"):
            self.collect(settings=changed)
        self.assertFalse((Path(self.directory.name)/"block").exists())

    def test_actor_mutation_is_detected_before_replay_is_sealed(self):
        def mutate(progress):
            if progress["completed_episodes"] == 1:
                with torch.no_grad():
                    next(self.actor.parameters()).add_(0.01)
        with self.assertRaisesRegex(RuntimeError, "actor changed"):
            self.collect(progress_callback=mutate)
        root = Path(self.directory.name)/"block"
        self.assertFalse((root/"replay/manifest.json").exists())
        failure = json.loads((root/"collection_error.json").read_text())
        self.assertEqual(failure["interaction_accounting"]["published_real_joint_transitions"], 10)
        self.assertEqual(failure["operation"], "actor_version_check")


if __name__ == "__main__":
    unittest.main()
