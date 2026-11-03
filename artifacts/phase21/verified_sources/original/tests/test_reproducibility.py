import math
import unittest

from world_models.infrastructure.reproducibility import (
    PAIRED_INITIAL_COMPONENTS,
    canonical_hash,
    canonical_json,
    derive_seed,
    seed_manifest,
)


class SeedDerivationTests(unittest.TestCase):
    def test_initialization_components_are_paired(self):
        for component in PAIRED_INITIAL_COMPONENTS:
            with self.subTest(component=component):
                self.assertEqual(derive_seed("relay", 1001, component, "B"), derive_seed("relay", 1001, component, "D"))

    def test_online_branches_families_agents_and_components_are_separate(self):
        values = {
            derive_seed("relay", 1001, "real_collection", "B"),
            derive_seed("relay", 1001, "real_collection", "D"),
            derive_seed("cover", 1001, "real_collection", "B"),
            derive_seed("relay", 1002, "real_collection", "B"),
            derive_seed("relay", 1001, "imagination", "B"),
            derive_seed("relay", 1001, "local_action_sampling", "B", 0),
            derive_seed("relay", 1001, "local_action_sampling", "B", 1),
        }
        self.assertEqual(len(values), 7)
        self.assertTrue(all(0 <= value <= 2**32 - 1 for value in values))

    def test_seed_requests_are_stable_and_order_independent(self):
        expected = derive_seed("relay", 1001, "imagination", "D")
        derive_seed("cover", 12, "unrelated_component", "B")
        self.assertEqual(expected, derive_seed("relay", 1001, "imagination", "D"))
        self.assertEqual(expected, 1141243886)

    def test_invalid_namespaces_are_rejected(self):
        for arguments in [
            ("unknown", 1, "imagination", "B"),
            ("relay", True, "imagination", "B"),
            ("relay", -1, "imagination", "B"),
            ("relay", 2**63, "imagination", "B"),
            ("relay", 1, "", "B"),
            ("relay", 1, " imagination", "B"),
            ("relay", 1, "imagination", "A"),
            ("relay", 1, "imagination", "B", 2),
            ("relay", 1, "imagination", "B", True),
        ]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                derive_seed(*arguments)

    def test_manifests_are_json_serializable_and_canonical(self):
        manifest = seed_manifest("cover", 1001, "D")
        self.assertIn("local_actor_initialization", manifest["components"])
        self.assertNotEqual(manifest["local_action_sampling"]["0"], manifest["local_action_sampling"]["1"])
        self.assertEqual(canonical_hash({"b": 2, "a": 1}), canonical_hash({"a": 1, "b": 2}))
        self.assertIn('"variant":"D"', canonical_json(manifest))
        with self.assertRaises(ValueError):
            canonical_json({"invalid": math.nan})


if __name__ == "__main__":
    unittest.main()
