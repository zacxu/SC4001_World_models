"""Geometry holdout allocation and diagnostic/training provenance checks."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from world_models.env.scenario_builder import (
    _parameters, _template_parameters, build_scenario, compiler_path,
    geometry_fingerprint, validate_geometry,
)
from world_models.env.splits import (
    build_split_scenario, get_template, list_templates, load_split_spec,
    randomization_draw, split_manifest, write_split_manifest,
)


class SplitManifestTests(unittest.TestCase):
    def test_six_two_two_allocation_holds_out_realized_geometry(self):
        manifest = split_manifest()
        self.assertTrue(manifest["geometry_disjoint_across_splits"])
        for family in ("relay", "cover"):
            records = [row for row in manifest["templates"] if row["task_family"] == family]
            self.assertEqual(Counter(row["split"] for row in records), {"train":6,"validation":2,"test":2})
            self.assertEqual(len({row["geometry_sha256"] for row in records}), 10)
            self.assertTrue(all(all(row["checks"].values()) for row in records))
            self.assertTrue(all(row["all_spawn_role_enemy_bank_combinations_validated"] == (6 if family == "relay" else 18) for row in records))
        self.assertIn("not arbitrary-map", manifest["scope"])

    def test_geometry_is_constant_across_nuisance_randomization_seeds(self):
        for family in ("relay", "cover"):
            for template in list_templates(family):
                signatures = set()
                for seed in (0, 1, 2, 17, 101):
                    params, cells, _ = _template_parameters(family, seed, template["id"])
                    validate_geometry(params, cells)
                    signatures.add(geometry_fingerprint(params, cells))
                self.assertEqual(len(signatures), 1)

    def test_training_cannot_select_validation_test_or_legacy_geometry(self):
        for family in ("relay", "cover"):
            for split in ("validation", "test"):
                for template in list_templates(family, split):
                    with self.assertRaisesRegex(ValueError, "training may not select"):
                        get_template(family, template["id"], purpose="training")
                    with self.assertRaises(ValueError):
                        build_split_scenario(family, split, purpose="training")
        with self.assertRaisesRegex(ValueError, "explicitly allocated"):
            build_scenario("relay", purpose="training")

    def test_purpose_split_and_bank_selection_are_explicit(self):
        _, _, diagnostic = get_template("relay", "relay_train_00")
        self.assertFalse(diagnostic["learning_eligible"])
        self.assertFalse(diagnostic["diagnostic_data_enters_replay"])
        self.assertTrue(diagnostic["diagnostic_import_requires_training_ledger"])
        _, _, training = get_template("relay", "relay_train_00", purpose="training")
        self.assertTrue(training["learning_eligible"])
        for kwargs in ({"purpose":"unknown"}, {"split":"test"}, {"randomization_bank":"cover-shared-v1"}, {"purpose":"test"}):
            with self.assertRaises(ValueError):
                get_template("relay", "relay_train_00", **kwargs)
        with self.assertRaises(ValueError):
            list_templates("relay", "held_out_unknown")
        for index in (-1, True, 6):
            with self.assertRaises(ValueError):
                build_split_scenario("relay", "train", index)

    def test_split_and_bank_manifests_are_defensive_stable_snapshots(self):
        first = split_manifest()
        original_hash = first["sha256"]
        first["specification"]["banks"]["relay-shared-v1"]["wall_textures"][0] = "CORRUPTED"
        first["templates"][0]["checks"].clear()
        second = split_manifest()
        self.assertEqual(second["sha256"], original_hash)
        self.assertNotEqual(second["specification"]["banks"]["relay-shared-v1"]["wall_textures"][0], "CORRUPTED")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            result = write_split_manifest(path)
            self.assertEqual(json.loads(path.read_text()), result)

    def test_versioned_banks_vary_roles_spawns_textures_and_cover_enemies(self):
        spec = load_split_spec()
        for family in ("relay", "cover"):
            bank = spec["banks"][f"{family}-shared-v1"]
            samples = [randomization_draw(family, seed, bank) for seed in range(64)]
            self.assertEqual({draw["operator"] for draw in samples}, {0, 1})
            for field in ("spawn", "texture", "enemy"):
                self.assertEqual({draw["indices"][field] for draw in samples}, set(range(1 if family == "relay" and field == "enemy" else 3)))
            self.assertEqual(randomization_draw(family, 12, bank), randomization_draw(family, 12, bank))

    def test_every_spawn_bank_choice_preserves_zero_task_state_on_reset(self):
        spec = load_split_spec()
        for family in ("relay", "cover"):
            for template in list_templates(family):
                params, cells, _ = _template_parameters(family, 0, template["id"])
                for pair in spec["banks"][template["bank"]]["spawn_pairs"]:
                    for operator in (0, 1):
                        trial = deepcopy(params)
                        trial["role_assignment"] = {"operator":operator,"runner":1-operator}
                        trial["spawns"] = [pair["operator"],pair["runner"]] if operator == 0 else [pair["runner"],pair["operator"]]
                        self.assertTrue(validate_geometry(trial, cells)["fresh_spawns_outside_task_zones"])
        # This spawn is reachable and otherwise geometrically valid, but is
        # inside the real pressure-plate rectangle and would trigger events in
        # the reset warm-up before the first actor decision.
        params, cells, _ = _template_parameters("relay", 0, "relay_train_00")
        params["spawns"][params["role_assignment"]["operator"]] = [128,96]
        with self.assertRaisesRegex(ValueError, "fresh_spawns_outside_task_zones.*False"):
            validate_geometry(params, cells)

    def test_geometry_validation_rejects_broken_cooperation_and_inaccessible_console(self):
        relay, cells, _ = _parameters("relay", 0)
        broken = deepcopy(cells)
        for props in broken.values():
            if props.get("tag") == 11:
                props["ceiling"] = 128
        with self.assertRaisesRegex(ValueError, "generated layout failed"):
            validate_geometry(relay, broken)
        cover, cells, _ = _parameters("cover", 0)
        for x in range(256, 384, 64):
            for y in range(128, 256, 64):
                cells.pop((x,y), None)
        with self.assertRaisesRegex(ValueError, "generated layout failed"):
            validate_geometry(cover, cells)

    def test_geometry_fingerprint_excludes_appearance_but_detects_geometry_changes(self):
        params, cells, _ = _parameters("cover", 0)
        before = geometry_fingerprint(params, cells)
        for props in cells.values():
            props.update(floor="ANOTHER", light=64)
        self.assertEqual(geometry_fingerprint(params, cells), before)
        cells.pop((64, 64))
        self.assertNotEqual(geometry_fingerprint(params, cells), before)


class CompiledSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            compiler_path()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc))

    def test_all_twenty_templates_compile_with_registry_and_split_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            for family in ("relay", "cover"):
                for split in ("train", "validation", "test"):
                    for index, template in enumerate(list_templates(family, split)):
                        with self.subTest(template=template["id"]):
                            result = build_split_scenario(family, split, index, seed=7, output_dir=Path(directory)/template["id"])
                            self.assertEqual(result["seed"], 7)
                            self.assertEqual(result["template_id"], template["id"])
                            self.assertEqual(result["split_provenance"]["split"], split)
                            self.assertFalse(result["data_provenance"]["learning_eligible"])
                            self.assertTrue(all(result["geometry_checks"].values()))
                            self.assertEqual(result["privileged_registry_version"], "privileged-targets-v1")
                            self.assertEqual(result["user_variables"]["extraction_unlocked"],40)
                            self.assertTrue(Path(result["wad_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
