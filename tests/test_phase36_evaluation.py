"""Real-evaluation protocol checks with counted coordinator fixtures, never ViZDoom."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.phase36_support.evaluation import (
    actor_version, evaluate_actor_export, load_actor_export, validation_bank,
)
from scripts.phase36_support.evidence import load_report
from tests.test_replay_collection import CountingCoordinator, environment_fixture, scenario_fixture
from world_models.cycle.checkpoints import save_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.env.coop import TwoPlayerCoopEnv
from world_models.infrastructure.config import resolve_config
from world_models.policies.actor import LocalActor
from world_models.policies.config import PolicyConfig


def validation_scenario(family, split, template_index=0, **kwargs):
    result = scenario_fixture(family, split, template_index, **kwargs)
    result["split_provenance"]["template_id"] = f"{family}_{split}_{template_index:02d}"
    return result


def export_fixture(path, *, seed=0, origin_device="cpu", actor=None, extra_state=None,
                   mutate_metadata=None, run_manifest=False):
    if actor is None:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(361)
            actor = LocalActor(PolicyConfig(encoder_channels=(8, 8, 8, 8), embedding_size=8,
                                            hidden_size=8, critic_hidden_sizes=(16, 16)))
    metadata = {"schema_version": "p35-local-actor-v1", "task_family": "relay", "variant": "B",
                "run_seed": seed, "real_joint_transitions": 0, "actual_initial_replay_interactions": 50000,
                "untrained_zero_interaction_baseline": True, "actor_version": actor_version(actor, origin_device=origin_device),
                "actor_config": json.loads(json.dumps(asdict(actor.config))), "world_model_version": "f" * 64,
                "model_updates": 5000, "local_ppo_cycles": 0, "actor_optimizer_steps": 0, "critic_optimizer_steps": 0,
                "evaluation_split": "validation", "on_registered_evaluation_grid": True,
                "evaluation_executed": False, "phase3_acceptance": False}
    if mutate_metadata:
        mutate_metadata(metadata)
    save_cycle_checkpoint(path, state={"actor": actor.state_dict(), **(extra_state or {})}, metadata=metadata)
    if run_manifest:
        settings = deepcopy(resolve_config("relay", "B", "pilot").settings)
        settings["world_model"]["encoder_channels"] = list(actor.config.encoder_channels)
        settings["actor"].update(embedding_size=actor.config.embedding_size, hidden_size=actor.config.hidden_size)
        settings["critic"]["hidden_sizes"] = list(actor.config.critic_hidden_sizes)
        manifest = {"schema_version": "p35-run-v1", "master_seed": seed, "device": origin_device, "settings": settings}
        (Path(path).parent.parent / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return actor, metadata


class Phase36EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.export = self.root / "actor"
        self.actor, self.metadata = export_fixture(self.export)
        # Source edits elsewhere in concurrently developed modules are irrelevant
        # to these protocol fixtures; the real evaluator guards complete sources.
        self.sources = patch("scripts.phase36_support.evaluation.source_identity", return_value={"fixture": "0" * 64})
        self.sources.start(); self.addCleanup(self.sources.stop)

    def evaluate(self, name="evaluation", **kwargs):
        options = {"episodes": 2, "env_factory": environment_fixture, "scenario_builder": validation_scenario}
        options.update(kwargs)
        return evaluate_actor_export(self.export, self.root / name, **options)

    def test_eight_whole_episodes_balanced_validation_metrics_and_receipts(self):
        result = self.evaluate(episodes=8)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["native_backend"])
        self.assertFalse(result["native_engine"])
        self.assertFalse(result["phase3_acceptance"])
        self.assertFalse(result["learning_eligible"])
        self.assertEqual(result["training_interactions_added"], 0)
        self.assertFalse(result["world_model_loaded"])
        self.assertFalse(result["critic_loaded"])
        self.assertEqual(result["metrics"]["episodes"], 8)
        self.assertEqual(result["metrics"]["success_rate"], 1.0)
        self.assertEqual(result["metrics"]["mean_length"], 4)
        self.assertEqual(result["metrics"]["mean_return"], .996)
        self.assertEqual(result["metrics"]["event_totals"]["successes"], 8)
        self.assertEqual(result["metrics"]["mean_event_progress"], 0.0)
        self.assertGreater(result["metrics"]["mean_action_entropy"], 0)
        self.assertEqual([row["template_index"] for row in result["episodes"]], [0, 1] * 4)
        accounting = result["interaction_accounting"]
        self.assertEqual(accounting["published_real_joint_transitions"], 32)
        self.assertEqual(accounting["completed_episode_joint_transitions"], 32)
        self.assertEqual(accounting["initialization_joint_decisions"], 8)
        self.assertEqual(accounting["physical_joint_decisions_including_initialization"], 40)
        self.assertEqual(accounting["failed_dispatch_attempts"], 0)
        self.assertEqual(accounting["purpose"], "local_actor_validation")
        self.assertFalse((self.root / "evaluation/training_interactions.jsonl").exists())
        self.assertFalse((self.root / "evaluation/replay").exists())
        self.assertEqual(load_report(self.root / "evaluation/evaluation.json"), result)
        for entry in result["files"].values():
            self.assertEqual(file_sha256(self.root / "evaluation" / entry["path"]), entry["sha256"])
        self.assertEqual(Path(result["actor_export"]["path"]), Path("../actor"))

    def test_actor_identity_matches_existing_version_and_cross_device_export_loads_on_cpu(self):
        from world_models.imagination.versions import module_version
        self.assertEqual(actor_version(self.actor), module_version(self.actor))
        gpu_export = self.root / "gpu_actor"
        export_fixture(gpu_export, actor=self.actor, origin_device="cuda:3")
        loaded, provenance = load_actor_export(gpu_export)
        self.assertEqual(provenance["verified_origin_device"], "cuda:3")
        self.assertEqual(provenance["portable_weights_sha256"], actor_version(self.actor))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(loaded.parameters(), self.actor.parameters())))

    def test_nonstandard_origin_device_requires_explicit_label_and_never_cuda_allocation(self):
        export_fixture(self.root / "gpu19", actor=self.actor, origin_device="cuda:19")
        with self.assertRaisesRegex(ValueError, "source-device"):
            load_actor_export(self.root / "gpu19")
        loaded, provenance = load_actor_export(self.root / "gpu19", source_device="cuda:19")
        self.assertEqual(next(loaded.parameters()).device.type, "cpu")
        self.assertEqual(provenance["verified_origin_device"], "cuda:19")
        with self.assertRaises(ValueError):
            load_actor_export(self.export, source_device="cuda:19")

    def test_validation_bank_is_paired_deterministic_balanced_and_does_not_use_test_or_train(self):
        bank = validation_bank("relay", episodes=8, bank_seed=42)
        self.assertEqual(bank, validation_bank("relay", episodes=8, bank_seed=42))
        self.assertNotEqual(bank, validation_bank("relay", episodes=8, bank_seed=43))
        self.assertTrue(all("validation" in row["template_id"] for row in bank["episodes"]))
        self.assertTrue(all(row["action_seeds"][0] != row["action_seeds"][1] for row in bank["episodes"]))
        for episodes in (0, 1, 3, True):
            with self.assertRaises(ValueError):
                validation_bank("relay", episodes=episodes)
        for bank_seed in (-1, True):
            with self.assertRaises(ValueError):
                validation_bank("relay", bank_seed=bank_seed)

    def test_sampling_repeats_without_changing_global_rng_actor_weights_or_modes(self):
        loaded, ref = load_actor_export(self.export)
        loaded.train(); loaded.encoder[0].eval()
        modes = [module.training for module in loaded.modules()]
        global_rng = torch.random.get_rng_state().clone()
        version = actor_version(loaded)
        with patch("scripts.phase36_support.evaluation.load_actor_export", side_effect=lambda *a, **k: (loaded, deepcopy(ref))):
            self.evaluate("first")
            self.evaluate("second")
        self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
        self.assertEqual(version, actor_version(loaded))
        self.assertEqual(modes, [module.training for module in loaded.modules()])
        self.assertEqual((self.root / "first/transitions.jsonl").read_bytes(),
                         (self.root / "second/transitions.jsonl").read_bytes())

    def test_each_actor_call_receives_only_local_rgb_executed_previous_actions_and_reset_memory(self):
        loaded, ref = load_actor_export(self.export)
        calls = []
        handle = loaded.register_forward_pre_hook(lambda _module, args: calls.append(tuple(value.clone() for value in args)))
        self.addCleanup(handle.remove)
        with patch("scripts.phase36_support.evaluation.load_actor_export", return_value=(loaded, ref)):
            self.evaluate()
        transitions = [json.loads(line) for line in (self.root / "evaluation/transitions.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), len(transitions))
        for index, (images, previous, hidden) in enumerate(calls):
            self.assertEqual(tuple(images.shape), (2, 3, 96, 128))
            self.assertEqual(images.dtype, torch.uint8)
            if transitions[index]["decision_index"] == 0:
                self.assertEqual(previous.tolist(), [12, 12])
                self.assertEqual(hidden.count_nonzero().item(), 0)
            else:
                self.assertEqual(previous.tolist(), transitions[index - 1]["actions"])

    def test_greedy_is_explicit_separate_protocol_and_bfloat16_is_recorded(self):
        result = self.evaluate(mode="greedy", precision="bfloat16")
        self.assertEqual(result["protocol"]["mode"], "greedy")
        self.assertEqual(result["protocol"]["precision"], "bfloat16")
        self.assertIsNone(result["protocol"]["temperature"])
        for kwargs in ({"mode": "teacher"}, {"precision": "float16"}):
            with self.assertRaises(ValueError):
                self.evaluate("invalid", **kwargs)
        self.assertFalse((self.root / "invalid").exists())

    def test_consumed_failure_retains_debit_and_failed_receipted_report_without_fabricated_episode(self):
        def fail_env(family, **kwargs):
            return TwoPlayerCoopEnv(family, seed=kwargs["seed"], scenario_manifest=kwargs["scenario_manifest"],
                                    coordinator=CountingCoordinator(kwargs["work_root"], failure_at=2))
        with self.assertRaisesRegex(RuntimeError, "consumed decision"):
            self.evaluate(env_factory=fail_env)
        report = load_report(self.root / "evaluation/evaluation.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["episodes"], [])
        self.assertEqual(report["interaction_accounting"]["published_real_joint_transitions"], 1)
        self.assertEqual(report["interaction_accounting"]["failed_dispatch_attempts"], 1)
        self.assertEqual(report["interaction_accounting"]["initialization_joint_decisions"], 1)
        self.assertEqual(report["interaction_accounting"]["physical_joint_decisions_including_initialization"], 3)
        before = (self.root / "evaluation/evaluation_interactions.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            self.evaluate()
        self.assertEqual(before, (self.root / "evaluation/evaluation_interactions.jsonl").read_bytes())

    def test_wrong_layout_provenance_rejected_before_any_physics(self):
        for field, wrong in (("split", "test"), ("purpose", "training"),
                             ("learning_eligible", True), ("template_id", "relay_train_00")):
            def bad_scenario(*args, **kwargs):
                scenario = validation_scenario(*args, **kwargs)
                scenario["split_provenance"][field] = wrong
                return scenario
            with self.assertRaises(ValueError):
                self.evaluate(field, scenario_builder=bad_scenario)
            report = load_report(self.root / field / "evaluation.json")
            self.assertEqual(report["interaction_accounting"]["physical_joint_decisions_including_initialization"], 0)

    def test_full_training_checkpoint_and_corrupted_export_are_rejected_before_native_calls(self):
        export_fixture(self.root / "full", extra_state={"world_model": {"weight": torch.zeros(1)}})
        with patch("scripts.phase36_support.evaluation.load_cycle_checkpoint") as loader:
            with self.assertRaisesRegex(ValueError, "actor-only"):
                load_actor_export(self.root / "full")
            loader.assert_not_called()
        with (self.export / "state.pt").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaises(ValueError):
            self.evaluate()
        self.assertFalse((self.root / "evaluation").exists())

    def test_missing_final_export_identifies_pending_publication_before_loading_or_native_calls(self):
        missing = self.root / "run/evaluation/real_000100000"
        with patch("scripts.phase36_support.evaluation.load_cycle_checkpoint") as loader, patch(
                "scripts.phase36_support.evaluation._NATIVE_ENV") as native:
            with self.assertRaisesRegex(ValueError, "directory does not exist.*publication may still be pending"):
                evaluate_actor_export(missing, self.root / "result")
            loader.assert_not_called()
            native.assert_not_called()
        self.assertFalse(missing.exists())
        self.assertFalse((self.root / "result").exists())
        with self.assertRaisesRegex(ValueError, "directory does not exist") as error:
            load_actor_export(self.root / "missing")
        self.assertNotIn("publication may still be pending", str(error.exception))

    def test_missing_export_manifest_is_distinguished_before_loading_or_native_calls(self):
        missing = self.root / "empty_export"
        missing.mkdir()
        with patch("scripts.phase36_support.evaluation.load_cycle_checkpoint") as loader, patch(
                "scripts.phase36_support.evaluation._NATIVE_ENV") as native:
            with self.assertRaisesRegex(ValueError, "incomplete: missing manifest.json"):
                evaluate_actor_export(missing, self.root / "result")
            loader.assert_not_called()
            native.assert_not_called()
        self.assertEqual(list(missing.iterdir()), [])
        self.assertFalse((self.root / "result").exists())

    def test_missing_export_receipt_is_distinguished_and_never_repaired_or_evaluated(self):
        receipt = self.export / "manifest.sha256"
        receipt.unlink()
        with patch("scripts.phase36_support.evaluation.load_cycle_checkpoint") as loader, patch(
                "scripts.phase36_support.evaluation._NATIVE_ENV") as native:
            with self.assertRaisesRegex(ValueError, "incomplete: missing manifest.sha256"):
                evaluate_actor_export(self.export, self.root / "result")
            loader.assert_not_called()
            native.assert_not_called()
        self.assertFalse(receipt.exists())
        self.assertFalse((self.root / "result").exists())

    def test_export_manifest_tampering_is_distinguished_before_loading_or_native_calls(self):
        manifest = self.export / "manifest.json"
        receipt_before = (self.export / "manifest.sha256").read_bytes()
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        with patch("scripts.phase36_support.evaluation.load_cycle_checkpoint") as loader, patch(
                "scripts.phase36_support.evaluation._NATIVE_ENV") as native:
            with self.assertRaisesRegex(ValueError, "manifest receipt mismatch"):
                evaluate_actor_export(self.export, self.root / "result")
            loader.assert_not_called()
            native.assert_not_called()
        self.assertEqual((self.export / "manifest.sha256").read_bytes(), receipt_before)
        self.assertFalse((self.root / "result").exists())

    def test_metadata_weight_mismatch_and_invalid_baseline_counters_are_rejected(self):
        for name, mutation in (("version", lambda metadata: metadata.update(actor_version="a" * 64)),
                               ("baseline", lambda metadata: metadata.update(local_ppo_cycles=1)),
                               ("dtype", lambda metadata: metadata["actor_config"].update(hidden_size=16))):
            export_fixture(self.root / name, mutate_metadata=mutation)
            with self.assertRaises((ValueError, RuntimeError)):
                load_actor_export(self.root / name)

    def test_parent_run_identity_binding_supports_arbitrary_cuda_origin(self):
        path = self.root / "run/evaluation/real_000000000"
        export_fixture(path, origin_device="cuda:9", run_manifest=True)
        _, provenance = load_actor_export(path)
        self.assertEqual(provenance["verified_origin_device"], "cuda:9")
        self.assertEqual(len(provenance["run_identity"]), 64)
        manifest_path = self.root / "run/run_manifest.json"
        self.assertEqual(provenance["run_manifest"]["sha256"], file_sha256(manifest_path))
        manifest = json.loads(manifest_path.read_text())
        manifest["master_seed"] += 1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            load_actor_export(path)

    def test_trace_storage_failure_keeps_published_interaction_and_restores_actor_modes(self):
        loaded, reference = load_actor_export(self.export)
        loaded.train(); loaded.encoder[0].eval()
        modes = [module.training for module in loaded.modules()]
        with patch("scripts.phase36_support.evaluation.load_actor_export", return_value=(loaded, reference)), patch(
                "scripts.phase36_support.evaluation._append_jsonl", side_effect=OSError("fixture disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.evaluate()
        report = load_report(self.root / "evaluation/evaluation.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["operation"], "trace_write")
        self.assertEqual(report["interaction_accounting"]["published_real_joint_transitions"], 1)
        self.assertEqual(report["interaction_accounting"]["failed_dispatch_attempts"], 0)
        self.assertEqual(report["interaction_accounting"]["completed_episode_joint_transitions"], 0)
        self.assertEqual(modes, [module.training for module in loaded.modules()])

    def test_source_change_invalidates_result_after_retaining_all_consumed_interactions(self):
        with patch("scripts.phase36_support.evaluation.source_identity", side_effect=[{"old": "a"}, {"new": "b"}, {"new": "b"}]):
            with self.assertRaisesRegex(RuntimeError, "implementation sources"):
                self.evaluate()
        report = load_report(self.root / "evaluation/evaluation.json")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["source_unchanged"])
        self.assertEqual(report["interaction_accounting"]["published_real_joint_transitions"], 8)
        self.assertEqual(report["metrics"]["episodes"], 2)

    def test_fresh_evaluator_import_does_not_import_model_or_training_runtime(self):
        program = "import sys; import scripts.phase36_support.evaluation; assert not any(x.startswith(('world_models.models', 'world_models.training', 'world_models.cycle.controller')) for x in sys.modules)"
        result = subprocess.run([sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
