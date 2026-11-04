"""Validation geometry separation, native linkage, receipts and burn-in cohorts."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_replay_storage import episode_fixture
from world_models.data.replay import ReplayDataset, canonical_hash, read_npz, write_npz
from world_models.env.splits import get_template, list_templates
from world_models.infrastructure.config import resolve_config
from world_models.training.validation import ValidationBank, ValidationBankWriter, collect_validation_bank


def results(length=20, *, missing_final=False, family="relay", split="validation"):
    values = episode_fixture(length, missing_final=missing_final, family=family)
    _, _, provenance = get_template(family, list_templates(family, split)[0]["id"],
                                     split=split, purpose="validation" if split == "validation" else "training")
    scenario = deepcopy(values[0].training_only_info["scenario_manifest"])
    scenario["split_provenance"] = provenance
    scenario["template_id"] = provenance["template_id"]
    for index, value in enumerate(values):
        value.training_only_info.update(scenario_manifest=deepcopy(scenario), dispatch_id=4 + 4*index,
                                        consumed_joint_decisions=1 + index)
        if index:
            deltas = value.training_only_info["events"]["deltas"]
            milestone = (deltas["latch_activations"] * .1 if family == "relay" else
                         (deltas["console_1_completions"] + deltas["console_2_completions"]) * .05)
            values[index] = replace(value, team_reward=-.001 + milestone + (1. if index == length else 0.))
    return values


def evidence(directory, values, *, wrong_action=False):
    journal = Path(directory) / "runtime" / "dispatch_ledger.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for step in values[1:]:
        for repeat in range(step.metadata.actual_simulator_tics):
            identifier = 4 + 4*step.metadata.decision_index + repeat + 1
            tic = step.metadata.start_tic + repeat + 1
            records.extend((dict(kind="dispatch", dispatch_id=identifier, command="step",
                                 actions=[0, 0] if wrong_action else list(step.metadata.actions)),
                            dict(kind="acknowledged", dispatch_id=identifier, command="step", tics=[tic, tic])))
    journal.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return [{"path": "runtime/dispatch_ledger.jsonl", "bytes": journal.stat().st_size,
             "sha256": hashlib.sha256(journal.read_bytes()).hexdigest()}]


def make_bank(directory, *, length=20, missing_final=False, family="relay"):
    configuration = resolve_config(family, "B", "pilot", "configs").settings
    writer = ValidationBankWriter(directory, task_family=family, run_seed=0, configuration=configuration)
    values = results(length, missing_final=missing_final, family=family)
    writer.append_episode("validation-episode", values[0], values[1:], collector_kind="scripted",
                          policy_version="fixture-script", engine_version="fixture-native-contract",
                          native_evidence=evidence(directory, values))
    writer.seal()
    return ValidationBank(directory)


def rewrite_manifest(directory, transform):
    path = Path(directory) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    transform(manifest)
    manifest["identity_sha256"] = canonical_hash({key: value for key, value in manifest.items() if key != "identity_sha256"})
    path.write_text(json.dumps(manifest), encoding="utf-8")
    path.with_suffix(".sha256").write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n", encoding="ascii")


class ValidationBankTests(unittest.TestCase):
    def test_roundtrip_immutable_windows_preserve_source_clock_and_terminal_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bank"
            bank = make_bank(root, missing_final=True)
            window = bank.window(0, "validation-episode", 18, burn_in=16, loss_steps=48)
            self.assertEqual((window.start, window.anchor, window.stop), (2, 18, 20))
            np.testing.assert_array_equal(window.transitions["decision_index"], np.arange(2, 20))
            self.assertFalse(window.states["image_valid"][-1, 0])
            self.assertTrue(window.transitions["terminated"][-1])
            self.assertTrue(window.states["privileged_validity_mask"][-1].all())
            self.assertEqual(float(window.transitions["team_reward"][-1]), .999)
            self.assertEqual(len(window.event_fields), window.transitions["event_deltas"].shape[-1])
            self.assertTrue(all(not array.flags.writeable for array in (*window.states.values(), *window.transitions.values())))
            with self.assertRaises(ValueError):
                window.states["rgb"][0, 0, 0, 0, 0] = 1
            with self.assertRaises(ValueError):
                ReplayDataset(root)
            self.assertFalse(bank.manifest["learning_eligible"])
            self.assertEqual(bank.manifest["purpose"], "model_validation")

    def test_burn_in_comparison_has_identical_anchors_targets_and_no_global_rng_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = make_bank(Path(directory) / "bank")
            before = np.random.get_state()
            cohorts = [bank.windows(burn_in=burn, max_windows=7, seed=19) for burn in (16, 32, 64)]
            after = np.random.get_state()
            self.assertTrue(np.array_equal(before[1], after[1]))
            for windows in zip(*cohorts):
                self.assertEqual(len({(window.episode_id, window.anchor, window.stop) for window in windows}), 1)
                labels = [window.transitions["team_reward"][window.burn_in:] for window in windows]
                np.testing.assert_array_equal(labels[0], labels[1])
                np.testing.assert_array_equal(labels[1], labels[2])
            prefix = bank.window(0, "validation-episode", 19, full_prefix=True)
            self.assertEqual((prefix.start, prefix.burn_in), (0, 19))
            for args in ((1, "validation-episode", 0), (0, "validation-episode", 20), (0, "missing", 0)):
                with self.assertRaises(ValueError):
                    bank.window(*args)

    def test_training_geometry_synthetic_source_and_unlinked_native_actions_are_rejected(self):
        configuration = resolve_config("relay", "B", "pilot", "configs").settings
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "synthetic"):
                ValidationBankWriter(Path(directory) / "synthetic", task_family="relay", run_seed=0,
                                     configuration=configuration, data_source="synthetic_ppo")
            writer = ValidationBankWriter(Path(directory) / "bank", task_family="relay", run_seed=0,
                                          configuration=configuration)
            for values, bad in ((results(split="train"), False), (results(), True)):
                with self.assertRaises(ValueError):
                    writer.append_episode("bad", values[0], values[1:], collector_kind="scripted",
                        policy_version="fixture", engine_version="fixture", native_evidence=evidence(writer.path, values, wrong_action=bad))

    def test_payload_and_native_journal_tampering_fail_closed(self):
        for member in ("episodes/episode_000000.npz", "runtime/dispatch_ledger.jsonl"):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "bank"
                make_bank(root)
                path = root / member
                with path.open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaises(ValueError):
                    ValidationBank(root)

    def test_semantic_corruption_with_refreshed_receipts_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bank"
            make_bank(root)
            path = root / "episodes/episode_000000.npz"
            arrays = {name: value.copy() for name, value in read_npz(path).items()}
            arrays["decision_index"][1] = 0
            receipt = write_npz(path, arrays)
            path.with_suffix(".sha256").write_text(f"{receipt['sha256']}  {path.name}\n", encoding="ascii")
            rewrite_manifest(root, lambda manifest: manifest["episodes"][0]["payload"].update(receipt))
            with self.assertRaisesRegex(ValueError, "indexing"):
                ValidationBank(root)

    def test_manifest_guards_prevent_train_test_or_synthetic_relabeling(self):
        for key, value in (("split", "train"), ("split", "test"), ("data_source", "synthetic_ppo"),
                           ("learning_eligible", True), ("enters_training_replay", True)):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "bank"
                make_bank(root)
                rewrite_manifest(root, lambda manifest: manifest.update({key: value}))
                with self.assertRaises(ValueError):
                    ValidationBank(root)

    def test_existing_bank_cannot_be_overwritten_and_missing_receipt_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bank"
            bank = make_bank(root)
            with self.assertRaises(FileExistsError):
                ValidationBankWriter(root, task_family="relay", run_seed=0,
                                     configuration=bank.manifest["configuration"])
            (root / "manifest.sha256").unlink()
            with self.assertRaisesRegex(ValueError, "SHA receipt"):
                ValidationBank(root)

    def test_native_collection_api_segregates_both_validation_geometries_and_accounts_resets(self):
        class FixtureEnvironment:
            def __init__(self, family, *, scenario_manifest, work_root, **kwargs):
                self.task_family, self.scenario = family, scenario_manifest
                self.values = results(3, family=family)
                for value in self.values:
                    value.training_only_info["scenario_manifest"] = deepcopy(scenario_manifest)
                self.coordinator = SimpleNamespace(run_dir=Path(work_root) / "fixture",
                                                   consumed_joint_decisions=0)
                self.coordinator.run_dir.mkdir(parents=True)
                evidence(self.coordinator.run_dir, self.values)
                self.initialization_joint_decisions = 0
                self.index = 0

            def reset(self, *, seed):
                self.initialization_joint_decisions = 1
                self.coordinator.consumed_joint_decisions = 1
                return self.values[0]

            def step(self, *actions, collection_cutoff=False):
                self.index += 1
                self.coordinator.consumed_joint_decisions += 1
                # A mocked native result publishes the executed action, as the
                # real coordinator does; make its retained trace agree too.
                self.values[self.index] = replace(self.values[self.index],
                    metadata=replace(self.values[self.index].metadata, actions=tuple(actions)))
                evidence(self.coordinator.run_dir, self.values)
                return self.values[self.index]

            def close(self):
                pass

        selected = []
        class FixturePolicy:
            version = "fixture-script"

            def __init__(self, *args):
                pass

            def actions(self, info):
                return (0, 0)

        def build(family, split, template_index, *, purpose, seed, **kwargs):
            selected.append((split, purpose, template_index))
            scenario = deepcopy(results(3, family=family)[0].training_only_info["scenario_manifest"])
            _, _, provenance = get_template(family, list_templates(family, split)[template_index]["id"],
                                             split=split, purpose=purpose)
            scenario.update(split_provenance=provenance, template_id=provenance["template_id"], seed=seed)
            return scenario
        with tempfile.TemporaryDirectory() as directory, \
                patch("world_models.env.coop.TwoPlayerCoopEnv", FixtureEnvironment), \
                patch("world_models.env.splits.build_split_scenario", side_effect=build), \
                patch("world_models.data.collection.ScriptedExplorationPolicy", FixturePolicy), \
                patch("world_models.training.validation.importlib.metadata.version", return_value="fixture"):
            root = Path(directory) / "bank"
            manifest = collect_validation_bank("relay", 0, root, episodes_per_template=2)
            bank = ValidationBank(root)
            self.assertEqual(len(bank.manifest["episodes"]), 4)
            self.assertEqual(selected, [("validation", "validation", index) for index in (0, 0, 1, 1)])
            self.assertEqual(manifest["collection_accounting"]["published_transitions"], 12)
            self.assertEqual(manifest["collection_accounting"]["initialization_joint_decisions"], 4)
            self.assertEqual(manifest["collection_accounting"]["consumed_joint_decisions"], 16)
            self.assertEqual(manifest["collection_accounting"]["failed_dispatch_attempts"], 0)
            self.assertEqual({episode["collector_kind"] for episode in bank.manifest["episodes"]}, {"random", "scripted"})

    def test_failed_native_attempt_is_charged_and_cleanup_does_not_mask_primary_error(self):
        values = results(3)
        class FailingEnvironment:
            def __init__(self, family, *, scenario_manifest, work_root, **kwargs):
                self.task_family, self.scenario = family, scenario_manifest
                self.coordinator = SimpleNamespace(run_dir=Path(work_root) / "fixture",
                                                   consumed_joint_decisions=0)
                self.coordinator.run_dir.mkdir(parents=True)
                self.initialization_joint_decisions = 0
                self.index = 0

            def reset(self, **kwargs):
                self.initialization_joint_decisions = 1
                self.coordinator.consumed_joint_decisions = 1
                return values[0]

            def step(self, *args, **kwargs):
                self.index += 1
                self.coordinator.consumed_joint_decisions += 1
                if self.index == 2:
                    raise RuntimeError("primary native fixture failure")
                return values[self.index]

            def close(self):
                raise OSError("secondary cleanup fixture failure")

        with tempfile.TemporaryDirectory() as directory, \
                patch("world_models.env.coop.TwoPlayerCoopEnv", FailingEnvironment), \
                patch("world_models.env.splits.build_split_scenario", return_value=values[0].training_only_info["scenario_manifest"]), \
                patch("world_models.training.validation.importlib.metadata.version", return_value="fixture"):
            root = Path(directory) / "bank"
            with self.assertRaisesRegex(RuntimeError, "primary native") as raised:
                collect_validation_bank("relay", 0, root, episodes_per_template=2)
            self.assertIn("cleanup", " ".join(raised.exception.__notes__))
            rows = [json.loads(line) for line in (root / "validation_interactions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["consumed_joint_decisions"], 3)
            self.assertEqual(rows[0]["published_transitions"], 1)
            self.assertEqual(rows[0]["initialization_joint_decisions"], 1)
            self.assertEqual(rows[0]["failed_dispatch_attempts"], 1)
            self.assertFalse((root / "manifest.json").exists())
            self.assertEqual(json.loads((root / "collection_error.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
