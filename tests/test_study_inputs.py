"""Temporary receipt-bound files, with the original CPU validator injected.

The injected validator stands for existing P2.4 checks tested separately. These
tests exercise registration pairing, byte bindings, metadata and moving-input
guards without loading research tensors, building models or opening ViZDoom.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from world_models.data.ledger import file_sha256
from world_models.infrastructure.config import resolve_config
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_study import inputs


ROOT = Path(__file__).resolve().parents[1]


def _json(path, value, *, receipt=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    if receipt:
        path.with_suffix(".sha256").write_text(f"{file_sha256(path)}  {path.name}\n", encoding="ascii")


class StudyInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.selection = self.root / "selection/selection.json"
        self.data = self.root / "data"
        self.norm = self.root / "normalization"
        self.config = ROOT / "configs"
        self.contexts, cases = {}, []
        sources = {"src/world_models/contracts.py": file_sha256(ROOT / "src/world_models/contracts.py")}
        plan = self.root / "phase24/launch_plan.json"
        _json(plan, {"fixture": "unchanged original four-case launch plan"})
        for family, seed in (("relay", 0), ("relay", 1), ("cover", 0), ("cover", 1)):
            checkpoint = self.root / "phase24" / family / f"seed_{seed}/checkpoints/update_00004500.pt"
            _json(checkpoint, {"fixture_weights": [family, seed], "update": 4500}, receipt=True)
            manifest = checkpoint.parents[1] / "run_manifest.json"
            _json(manifest, {"fixture_manifest": [family, seed]})
            row = {"task": family, "seed": seed,
                "checkpoint_path": os.path.relpath(checkpoint, self.selection.parent).replace("\\", "/"),
                "checkpoint_sha256": file_sha256(checkpoint), "model_updates": 4500,
                "original_target_stage_updates": 5000,
                "run_manifest_path": os.path.relpath(manifest, self.selection.parent).replace("\\", "/"),
                "run_manifest_sha256": file_sha256(manifest), "source_sha256": sources,
                "shared_initial_variants": ["B", "D"]}
            if family == "relay":
                replay = self.data / family / f"seed_{seed}/replay"
                replay_manifest = {"dataset_identity_sha256": str(seed + 1) * 64, "transition_count": 50000,
                    "metadata": {"task_family": "relay", "run_seed": seed, "learning_eligible": True,
                                 "data_source": "real_environment"}}
                _json(replay / "manifest.json", replay_manifest, receipt=True)
                (replay / "chunk.npz").write_bytes(f"synthetic immutable replay {seed}".encode())
                dataset = SimpleNamespace(path=replay, manifest=replay_manifest,
                                          manifest_sha256=file_sha256(replay / "manifest.json"))
                normalization = {"task_family": "relay", "identity_sha256": str(seed + 3) * 64,
                                 "fixture": "fitting-only frozen statistics"}
                norm_path = self.norm / family / f"seed_{seed}.json"
                _json(norm_path, normalization, receipt=True)
                provenance = {"initial_normalization": {"identity_sha256": normalization["identity_sha256"],
                    "file_sha256": file_sha256(norm_path)}, "replays": [{
                    "manifest_sha256": dataset.manifest_sha256,
                    "dataset_identity_sha256": replay_manifest["dataset_identity_sha256"], "transitions": 50000}]}
                row.update(deepcopy(provenance))
                for variant in ("B", "D"):
                    _json(self.data / family / f"seed_{seed}/bindings/{variant}.json", {
                        "variant": variant, "dataset_path": "../replay",
                        "manifest_sha256": dataset.manifest_sha256,
                        "dataset_identity_sha256": replay_manifest["dataset_identity_sha256"],
                        "attributed_initial_joint_transitions": 50000,
                        "physical_collection_shared": True, "new_physical_transitions": 0})
                settings = resolve_config("relay", "D", "pilot", self.config).settings
                stats = SimpleNamespace(as_dict=lambda value=normalization: deepcopy(value))
                selected = {**deepcopy(row), "checkpoint_path": str(checkpoint), "run_manifest_path": str(manifest),
                            "selection_path": str(self.selection)}
                binding = {"selection": selected, "provenance": provenance,
                           "replay_path": str(replay), "normalization": normalization, "settings": settings}
                self.contexts[seed] = [settings, dataset, stats, {"update": 4500}, binding]
            cases.append(row)
        _json(self.selection, {"source_run_root": "../phase24", "source_launch_plan_sha256": file_sha256(plan),
                               "cases": cases}, receipt=True)
        for context in self.contexts.values():
            context[-1]["selection"]["selection_sha256"] = file_sha256(self.selection)
        self.loader = self.enterContext(patch.object(inputs, "_load_initial_inputs",
            side_effect=lambda **arguments: self.contexts[arguments["master_seed"]]))

    def bind(self):
        return inputs.bind_initial_inputs(initial_selection=self.selection, data_root=self.data,
                                         statistics_root=self.norm, config_root=self.config)

    def test_two_seed_inputs_are_paired_receipt_bound_json_and_readonly(self):
        before = {str(path): file_sha256(path) for path in self.root.rglob("*") if path.is_file()}
        result = self.bind()
        self.assertEqual(set(result["per_seed"]), {"0", "1"})
        self.assertEqual(result["physical_initial_joint_transitions"], 100000)
        self.assertEqual(result["sum_four_run_attributed_initial_joint_transitions"], 200000)
        self.assertEqual(result["physical_selected_model_optimizer_updates"], 9000)
        self.assertEqual(result["sum_four_run_attributed_model_optimizer_updates"], 18000)
        self.assertEqual(self.loader.call_count, 2)
        self.assertEqual([call.kwargs["master_seed"] for call in self.loader.call_args_list], [0, 1])
        for seed, row in result["per_seed"].items():
            self.assertEqual(row["selected_model"]["model_updates"], 4500)
            self.assertEqual(row["original_parent"]["resolved_by_variant"]["D"]["settings"]["world_model"]["initial_updates"], 5000)
            self.assertEqual(row["effective_fixed_data_contract"]["local_cycles"], 20)
            self.assertEqual(row["effective_fixed_data_contract"]["new_model_updates"], 0)
            effective = row["effective_fixed_data_contract"]
            self.assertEqual(effective["distillation_trajectory_source"], "teacher")
            self.assertEqual(effective["distillation_temperature"], 1.0)
            self.assertEqual(effective["distillation_loss"], "forward_kl_teacher_to_student")
            self.assertEqual(effective["local_ppo_optimizer_initialization"],
                             {"B": "fresh_from_paired_initialization", "D": "fresh_after_distillation"})
            self.assertEqual(effective["local_actor_compute_precision"], "float32")
            self.assertEqual(effective["teacher_policy_compute_precision"], "float32")
            self.assertEqual(effective["world_model_compute_precision"], "bfloat16")
            self.assertEqual(effective["teacher_kl_weight"], 0.0)
            self.assertFalse(effective["guidance_decay_enabled"])
            self.assertEqual(effective["evaluation_split"], "test")
            self.assertEqual(effective["evaluation_episodes_per_actor"], 20)
            self.assertEqual(effective["evaluation_bank_seed"], 5600)
            self.assertEqual(effective["primary_metric"], "final_success_rate")
            self.assertIn("historical", row["scientific_recipe_role"])
            for variant in ("B", "D"):
                self.assertEqual(row["replay"]["branch_bindings"][variant]["binding"]["manifest_sha256"], row["replay"]["manifest_sha256"])
        for name in ("research_training_executed", "model_constructed", "model_inference_executed",
                     "optimizer_constructed_or_restored", "native_environment_evaluation_executed"):
            self.assertFalse(result[name])
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual(before, {str(path): file_sha256(path) for path in self.root.rglob("*") if path.is_file()})
        bound = {row["path"] for row in result["bound_files"]}
        self.assertIn(str(self.root / "phase24/cover/seed_1/checkpoints/update_00004500.pt"), bound)
        self.assertIn(str(self.data / "relay/seed_0/replay/chunk.npz"), bound)

    def test_seed_budget_update_and_normalization_mismatches_fail_closed(self):
        checks = (("wrong selected update", lambda c: c[-1]["selection"].update(model_updates=4000)),
                  ("wrong model update", lambda c: c[3].update(update=4000)),
                  ("missing paired variant", lambda c: c[-1]["selection"].update(shared_initial_variants=["D"])),
                  ("wrong seed", lambda c: c[-1]["selection"].update(seed=0)),
                  ("wrong replay budget", lambda c: c[1].manifest.update(transition_count=49999)),
                  ("wrong statistics", lambda c: c[-1]["provenance"]["initial_normalization"].update(identity_sha256="f" * 64)))
        for name, mutation in checks:
            original = deepcopy(self.contexts[1])
            try:
                mutation(self.contexts[1])
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.bind()
            finally:
                self.contexts[1] = original

    def test_missing_seed_and_missing_legacy_cover_checkpoint_are_rejected(self):
        self.loader.side_effect = lambda **arguments: self.contexts[arguments["master_seed"]] if arguments["master_seed"] == 0 else (_ for _ in ()).throw(ValueError("missing seed 1"))
        with self.assertRaisesRegex(ValueError, "missing seed 1"):
            self.bind()
        self.loader.side_effect = lambda **arguments: self.contexts[arguments["master_seed"]]
        path = self.root / "phase24/cover/seed_0/checkpoints/update_00004500.pt"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.bind()

    def test_replay_binding_flags_and_wrong_store_are_rejected(self):
        path = self.data / "relay/seed_1/bindings/D.json"
        original = json.loads(path.read_text())
        for key, value in (("physical_collection_shared", False), ("new_physical_transitions", 1),
                           ("attributed_initial_joint_transitions", 49999), ("dataset_path", "../../seed_0/replay")):
            _json(path, {**original, key: value})
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, "replay bindings"):
                self.bind()

    def test_receipt_and_selected_source_tampering_are_rejected(self):
        receipt = self.norm / "relay/seed_0.sha256"
        receipt.write_text("0" * 64 + "  seed_0.json\n")
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.bind()
        _json(self.norm / "relay/seed_0.json", self.contexts[0][2].as_dict(), receipt=True)
        self.contexts[0][-1]["selection"]["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.bind()

    def test_changed_bytes_during_second_seed_validation_are_rejected(self):
        chunk = self.data / "relay/seed_0/replay/chunk.npz"
        def moving(**arguments):
            if arguments["master_seed"] == 1:
                chunk.write_bytes(b"mutated after seed zero was bound")
            return self.contexts[arguments["master_seed"]]
        self.loader.side_effect = moving
        with self.assertRaisesRegex(ValueError, "changed while being bound"):
            self.bind()

    def test_hash_cache_adds_each_file_once_and_final_recheck_once(self):
        observed = {}
        original = inputs.file_sha256
        def counted(path):
            key = str(Path(path))
            observed[key] = observed.get(key, 0) + 1
            return original(path)
        with patch.object(inputs, "file_sha256", side_effect=counted):
            result = self.bind()
        self.assertEqual(set(observed), {row["path"] for row in result["bound_files"]})
        self.assertTrue(all(count == 2 for count in observed.values()), observed)

    def test_import_does_not_load_torch_models_or_native_runtime(self):
        code = "import sys; import world_models_study.inputs; assert 'torch' not in sys.modules; assert 'vizdoom' not in sys.modules; assert 'world_models.teacher.runner' not in sys.modules"
        process = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
