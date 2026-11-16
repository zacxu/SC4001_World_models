"""Final actor projection checks; synthetic tensors, no research training."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from scripts.phase35_export import ROOT, _ACTOR_SOURCES, export_final_actor
from scripts.phase36_support.evaluation import actor_version, load_actor_export
from world_models.cycle.checkpoints import save_cycle_checkpoint
from world_models.cycle.schedule import Schedule
from world_models.data.ledger import file_sha256
from world_models.infrastructure.config import resolve_config
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.policies.actor import LocalActor
from world_models.policies.config import PolicyConfig


class Phase35ExportTests(unittest.TestCase):
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
        self.run = self.root / "run"
        self.run.mkdir()
        (self.run / ".worker.lock").write_bytes(b"")
        self.config = PolicyConfig(encoder_channels=(8, 8, 8, 8), embedding_size=8,
                                   hidden_size=8, critic_hidden_sizes=(16, 16))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(365)
            self.actor = LocalActor(self.config)
        settings = deepcopy(resolve_config("relay", "B", "pilot").settings)
        settings["world_model"]["encoder_channels"] = [8, 8, 8, 8]
        settings["world_model"]["online_updates"] = 250
        settings["actor"].update(embedding_size=8, hidden_size=8)
        settings["critic"]["hidden_sizes"] = [16, 16]
        settings["experiment"].update(real_budget=100000, collection_block=25000,
                                      checkpoints=[0, 50000, 75000, 100000])
        settings["mappo"].update(initial_local_cycles=40, online_local_cycles=20)
        schedule = replace(Schedule.from_experiment(settings), initial_model_updates=4500)
        self.manifest = {"schema_version": "p35-run-v1", "settings": settings, "master_seed": 0,
            "schedule": schedule.describe(), "device": "cuda:19",
            "source_sha256": {name: file_sha256(ROOT / name) for name in _ACTOR_SOURCES},
            "initial_selection": {"model_updates": 4500, "original_target_stage_updates": 5000,
                                  "selection_sha256": "a" * 64, "checkpoint_sha256": "b" * 64},
            "initial_checkpoint_sha256": "b" * 64,
            "registered_schedule": Schedule.from_experiment(settings).describe(),
            "initial_training_policy": "frozen_last_saved_checkpoints"}
        # An unrelated source difference is preserved as historical provenance.
        self.manifest["source_sha256"]["src/world_models/teacher/runner.py"] = "0" * 64
        self.controller = {"sequence": 20, "block_index": 2, "phase": "local",
            "real_joint_transitions": 100000, "model_updates_in_block": 250,
            "local_cycles_in_block": 20, "imagined_joint_steps": 20,
            "local_player_decisions": 40, "evaluation_exports": []}
        self.saved = {"controller": self.controller, "model": {"fixture": torch.zeros(1)},
            "actor": dict(self.actor.state_dict()), "critic": {"fixture": torch.zeros(1)},
            "mappo": {"schema_version": "p34-optimizer-v1", "cycles_completed": 80,
                "actor_optimizer_steps": 1280, "critic_optimizer_steps": 1280,
                "config": {"epochs": 4, "minibatches": 4},
                "network_versions": ("f" * 64, actor_version(self.actor, origin_device="cuda:19"), "e" * 64)},
            "refresher": {"schema_version": "p35-model-refresher-v1", "initial_model_updates": 4500,
                "total_model_updates": 5000, "model_version": "f" * 64},
            "rollout_streams": {}, "anchor_sampler": {}, "replays": []}
        self.meta = {"run_identity": canonical_hash(self.manifest), "phase": "local", "block_index": 2,
            "completed_block": False, "real_joint_transitions": 100000, "evaluation_executed": False}
        self.output = self.run / "evaluation/final_100k_export_v1"
        self.checkpoint = self.run / "checkpoints/state_00000020_01234567"

    def publish(self):
        (self.run / "run_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.meta["run_identity"] = canonical_hash(self.manifest)
        save_cycle_checkpoint(self.checkpoint, state=self.saved, metadata=self.meta)
        (self.run / "status.json").write_text('{"status":"stopped"}', encoding="utf-8")

    def files(self):
        return {p.relative_to(self.run).as_posix(): file_sha256(p) for p in self.run.rglob("*") if p.is_file()}

    def test_export_terminal_projection_preserves_run_rng_and_exact_actor(self):
        self.publish()
        before, rng = self.files(), torch.random.get_rng_state().clone()
        report = export_final_actor(self.run, self.output)
        self.assertTrue(report["training_budget_complete"])
        self.assertFalse(report["training_executed"])
        self.assertFalse(report["controller_finalization_performed"])
        self.assertEqual(report["source_checkpoint"], self.checkpoint.name)
        self.assertEqual(report["source_checkpoint_payload_sha256"], file_sha256(self.checkpoint / "state.pt"))
        self.assertIn("scripts/phase35_export.py", report["export_source_sha256"])
        self.assertIn("src/world_models/teacher/runner.py", report["source_differences"]["changed"])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        loaded, provenance = load_actor_export(self.output, source_device="cuda:19")
        self.assertEqual(provenance["metadata"]["local_ppo_cycles"], 80)
        self.assertEqual(provenance["metadata"]["model_updates"], 5000)
        self.assertTrue(all(torch.equal(v, loaded.state_dict()[k]) for k, v in self.actor.state_dict().items()))
        self.assertEqual(before, {k:v for k,v in self.files().items() if k in before})
        self.assertEqual((self.output / "export_provenance.sha256").read_text().strip(),
                         f"{file_sha256(self.output / 'export_provenance.json')}  export_provenance.json")

    def test_completed_controller_can_also_be_projected_without_retraining(self):
        self.controller["phase"] = "complete"
        self.meta.update(phase="complete", completed_block=True)
        self.publish()
        report = export_final_actor(self.run, self.root / "external_export")
        self.assertEqual(report["source_checkpoint_phase"], "complete")
        self.assertFalse(report["world_model_instantiated"])

    def test_reject_incomplete_ppo_counter_or_model_counter_budget(self):
        for section, name, value in (("controller", "local_cycles_in_block", 19),
                ("controller", "model_updates_in_block", 249), ("mappo", "cycles_completed", 79),
                ("mappo", "critic_optimizer_steps", 1279), ("refresher", "total_model_updates", 4999)):
            with self.subTest(section=section, name=name):
                old = self.saved[section][name]
                self.saved[section][name] = value
                self.checkpoint = self.run / f"checkpoints/state_00000020_{len(list((self.run/'checkpoints').glob('*'))) if (self.run/'checkpoints').exists() else 0:08x}"
                self.publish()
                with self.assertRaises(ValueError):
                    export_final_actor(self.run, self.output)
                self.assertFalse(self.output.exists())
                for p in self.checkpoint.iterdir(): p.unlink()
                self.checkpoint.rmdir()
                self.saved[section][name] = old

    def test_reject_actor_tensor_mutation(self):
        self.saved["actor"]["action_head.2.bias"] = self.saved["actor"]["action_head.2.bias"] + 1
        self.publish()
        with self.assertRaisesRegex(ValueError, "original version"):
            export_final_actor(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_reject_non_fp32_actor_before_publication(self):
        name = next(iter(self.saved["actor"]))
        self.saved["actor"][name] = self.saved["actor"][name].to(torch.float64)
        self.publish()
        with self.assertRaisesRegex(ValueError, "FP32"):
            export_final_actor(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_reject_disagreeing_recorded_model_versions(self):
        self.saved["refresher"]["model_version"] = "a" * 64
        self.publish()
        with self.assertRaisesRegex(ValueError, "model version"):
            export_final_actor(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_reject_boolean_counter_and_checkpoint_sequence_mismatch(self):
        self.controller["imagined_joint_steps"] = True
        self.publish()
        with self.assertRaisesRegex(ValueError, "integer"):
            export_final_actor(self.run, self.output)
        for p in self.checkpoint.iterdir(): p.unlink()
        self.checkpoint.rmdir()
        self.controller.update(imagined_joint_steps=20, sequence=19)
        self.publish()
        with self.assertRaisesRegex(ValueError, "final training budget"):
            export_final_actor(self.run, self.output)

    def test_concurrent_run_lock_rejects_export(self):
        from scripts.phase35_export import _exclusive_existing_run
        self.publish()
        with _exclusive_existing_run(self.run):
            with self.assertRaisesRegex(RuntimeError, "another worker"):
                export_final_actor(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_corrupt_latest_never_falls_back_to_earlier_checkpoint(self):
        self.publish()
        later = self.run / "checkpoints/state_00000021_11111111"
        save_cycle_checkpoint(later, state=self.saved, metadata=self.meta)
        (later / "manifest.sha256").write_text("0" * 64, encoding="ascii")
        with self.assertRaisesRegex(ValueError, "receipt"):
            export_final_actor(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_uncommitted_later_checkpoint_is_ignored_but_duplicates_rejected(self):
        self.publish()
        (self.run / "checkpoints/state_00000099_aaaaaaaa").mkdir()
        report = export_final_actor(self.run, self.output)
        self.assertEqual(report["source_checkpoint"], self.checkpoint.name)
        save_cycle_checkpoint(self.run / "checkpoints/state_00000020_22222222", state=self.saved, metadata=self.meta)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            export_final_actor(self.run, self.root / "new_output")

    def test_reject_unbound_manifest_and_changed_deployable_actor_source(self):
        self.publish()
        self.manifest["master_seed"] = 1
        (self.run / "run_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bound"):
            export_final_actor(self.run, self.output)
        self.manifest["source_sha256"][_ACTOR_SOURCES[0]] = "0" * 64
        (self.run / "run_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "actor implementation changed"):
            export_final_actor(self.run, self.output)

    def test_fresh_output_only_and_never_write_controller_paths(self):
        self.publish()
        for output in (self.run, self.checkpoint, self.run / "new_child"):
            with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                export_final_actor(self.run, output)
        export_final_actor(self.run, self.output)
        with self.assertRaises(FileExistsError):
            export_final_actor(self.run, self.output)

    def test_cli_does_not_import_worldmodel_controller_optimizer_or_cuda(self):
        self.publish()
        code = """
import sys, torch
from scripts.phase35_export import export_final_actor
export_final_actor(sys.argv[1], sys.argv[2])
assert not torch.cuda.is_initialized()
assert 'world_models.cycle.controller' not in sys.modules
assert 'world_models.cycle.refresh' not in sys.modules
assert 'world_models.models.world_model' not in sys.modules
assert 'world_models.mappo.updater' not in sys.modules
assert 'world_models.training.runner' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", "import torch; torch.set_num_threads(1);" + code,
                                 str(self.run), str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
