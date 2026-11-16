"""Static final-bank provenance, pairing and mutation rejection without native I/O."""
from collections import Counter
from copy import deepcopy
import hashlib
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from world_models.env.splits import SPEC_PATH, list_templates, load_split_spec, randomization_draw
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_study.bank import build_test_bank, evaluation_action_seed, validate_test_bank


class StudyBankTests(unittest.TestCase):
    def test_bank_has_twenty_deterministic_balanced_heldout_records(self):
        bank = build_test_bank()
        self.assertEqual(bank, build_test_bank())
        self.assertEqual(bank["bank_seed"], 5600)
        self.assertEqual(bank["split"], "test")
        self.assertEqual(bank["episodes_per_actor"], 20)
        self.assertEqual([row["episode_index"] for row in bank["episodes"]], list(range(20)))
        self.assertEqual(Counter(row["template_id"] for row in bank["episodes"]),
                         Counter({"relay_test_00": 10, "relay_test_01": 10}))
        self.assertEqual(bank["split_manifest_sha256"], canonical_hash(load_split_spec()))
        self.assertEqual(bank["split_registry_file_sha256"], hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest())
        self.assertEqual(bank["episodes"][0]["layout_seed"], 3352166895)
        self.assertEqual(bank["episodes"][0]["engine_seed"], 2884349978)
        self.assertEqual(bank["episodes"][0]["action_seeds_by_master_seed"],
                         {"0": [1880476010, 2888434087], "1": [3228139473, 2315445552]})
        for key in ("scenario_built", "native_environment_opened", "native_evaluation_executed", "learning_eligible"):
            self.assertIs(bank[key], False)

    def test_action_namespaces_pair_branches_but_not_players_or_training_seeds(self):
        bank, used = build_test_bank(), []
        for row in bank["episodes"]:
            used += [row["layout_seed"], row["engine_seed"]]
            for seed in (0, 1):
                b = [evaluation_action_seed(seed, row["episode_index"], player) for player in (0, 1)]
                d = [evaluation_action_seed(seed, row["episode_index"], player) for player in (0, 1)]
                self.assertEqual(b, d)
                self.assertEqual(b, row["action_seeds_by_master_seed"][str(seed)])
                self.assertNotEqual(b[0], b[1])
                used += b
            self.assertNotEqual(row["action_seeds_by_master_seed"]["0"], row["action_seeds_by_master_seed"]["1"])
        self.assertEqual(len(used), 120)
        self.assertEqual(len(set(used)), 120)
        for args in ((True, 0, 0), (2, 0, 0), (0, True, 0), (0, 20, 0), (0, -1, 0), (0, 0, True), (0, 0, 2)):
            with self.assertRaises(ValueError):
                evaluation_action_seed(*args)

    def test_nuisance_draws_and_geometry_are_from_registered_test_templates(self):
        spec, bank = load_split_spec(), build_test_bank()
        templates = {row["id"]: row for row in list_templates("relay", "test")}
        for row in bank["episodes"]:
            template = templates[row["template_id"]]
            self.assertEqual(row["geometry"], template["geometry"])
            self.assertEqual(row["geometry_sha256"], canonical_hash(template["geometry"]))
            self.assertEqual(row["nuisance_draw"], randomization_draw("relay", row["layout_seed"], spec["banks"][template["bank"]]))
            provenance = row["split_provenance"]
            self.assertEqual((provenance["split"], provenance["purpose"]), ("test", "test"))
            self.assertFalse(provenance["learning_eligible"])
            self.assertFalse(provenance["diagnostic_data_enters_replay"])
        bank["episodes"][0]["geometry"]["obstacles"][0][0] = -99
        self.assertNotEqual(bank, build_test_bank())

    def test_validate_rejects_missing_duplicate_reordered_or_tampered_bank_evidence(self):
        original = build_test_bank()
        mutations = (lambda b: b["episodes"].pop(),
                     lambda b: b["episodes"].append(deepcopy(b["episodes"][0])),
                     lambda b: b["episodes"].reverse(),
                     lambda b: b.update(split="validation"),
                     lambda b: b.update(bank_seed=5601),
                     lambda b: b.update(split_manifest_sha256="0" * 64),
                     lambda b: b["episodes"][0].update(template_id="relay_validation_00"),
                     lambda b: b["episodes"][0].update(layout_seed=1),
                     lambda b: b["episodes"][0]["action_seeds_by_master_seed"]["0"].reverse(),
                     lambda b: b.update(learning_eligible=True))
        for index, mutate in enumerate(mutations):
            bank = deepcopy(original)
            mutate(bank)
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_test_bank(bank)
        checked = validate_test_bank(original)
        self.assertEqual(checked, original)
        self.assertIsNot(checked, original)

    def test_reordered_or_changed_test_layout_registry_fails_closed(self):
        templates = list_templates("relay", "test")
        with patch("world_models_study.bank.list_templates", return_value=list(reversed(templates))):
            with self.assertRaisesRegex(ValueError, "test layouts"):
                build_test_bank()

    def test_construction_preserves_rng_and_imports_no_torch_or_native_backend(self):
        before = random.getstate()
        build_test_bank()
        self.assertEqual(random.getstate(), before)
        code = ("import sys; from world_models_study.bank import build_test_bank; "
                "b=build_test_bank(); assert len(b['episodes'])==20; "
                "assert 'torch' not in sys.modules; assert 'vizdoom' not in sys.modules; "
                "assert 'world_models.env.coop' not in sys.modules; "
                "assert 'world_models.env.scenario_builder' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
