"""Scenario geometry, compiler provenance, and binary packaging checks."""

from collections import deque
import hashlib
import io
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from world_models.env.scenario_builder import _parameters, build_scenario, compiler_path
from world_models.env.scripted import _arrived, _move


def reachable(cells,start,target,opened_tags=()):
    def cell(point):
        return int(point[0]//64)*64,int(point[1]//64)*64
    begin,end=cell(start),cell(target)
    allowed={point for point,props in cells.items() if props.get("ceiling",128)>0 or props.get("tag") in opened_tags}
    seen={begin}
    queue=deque([begin])
    while queue:
        x,y=queue.popleft()
        if (x,y)==end:
            return True
        for neighbor in ((x+64,y),(x-64,y),(x,y+64),(x,y-64)):
            if neighbor in allowed and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return False


class ScenarioGeometryTests(unittest.TestCase):
    def test_diagnostic_navigation_brakes_before_advancing_a_fast_waypoint(self):
        moving={"POSITION_X":144,"POSITION_Y":96,"VELOCITY_X":8,"VELOCITY_Y":0}
        self.assertFalse(_arrived(moving,(144,96)))
        self.assertEqual(_move(moving,(144,96)),2)

    def test_small_residual_velocity_does_not_starve_the_other_navigation_axis(self):
        player={"POSITION_X":692,"POSITION_Y":648,"VELOCITY_X":0,"VELOCITY_Y":-1.4}
        self.assertEqual(_move(player,(960,648),y_first=True),1)

    def test_relay_gate_and_latch_have_real_separated_route_dependencies(self):
        for seed in range(8):
            params,cells,_=_parameters("relay",seed)
            operator=params["role_assignment"]["operator"]
            runner=1-operator
            latch=params["scripted_waypoints"]["runner"][0]
            extraction=params["scripted_waypoints"]["runner"][1]
            self.assertFalse(reachable(cells,params["spawns"][runner],latch))
            self.assertFalse(reachable(cells,params["spawns"][operator],latch,opened_tags=(11,)))
            self.assertTrue(reachable(cells,params["spawns"][runner],latch,opened_tags=(11,)))
            self.assertTrue(reachable(cells,params["spawns"][operator],extraction,opened_tags=(11,12)))

    def test_seed_variation_is_reproducible_and_changes_roles_geometry_or_textures(self):
        self.assertEqual(_parameters("relay",4),_parameters("relay",4))
        variations=[_parameters("relay",seed)[0] for seed in range(8)]
        self.assertEqual({p["role_assignment"]["operator"] for p in variations},{0,1})
        self.assertGreater(len({p["common_x"] for p in variations}),1)
        self.assertGreater(len({p["wall_texture"] for p in variations}),1)

    def test_cover_consoles_and_extraction_are_reachable_without_crossing_pillars(self):
        params,cells,_=_parameters("cover",0)
        for start in params["spawns"]:
            for destination in ((320,192),(704,576),(params["zones"]["extraction"][0]+32,672)):
                self.assertTrue(reachable(cells,start,destination))
        first,second=params["zones"]["console_1"],params["zones"]["console_2"]
        self.assertTrue(first[2]<second[0] and first[3]<second[1])

    def test_extraction_marked_sector_union_matches_success_rectangle_across_layouts(self):
        for task in ("relay", "cover"):
            for seed in range(8):
                with self.subTest(task=task, seed=seed):
                    params, cells, _ = _parameters(task, seed)
                    x1, y1, x2, y2 = params["zones"]["extraction"]
                    marker = params["extraction_marker"]
                    self.assertEqual(marker["bounds"], [x1, y1, x2, y2])
                    expected = {(x,y) for x in range(x1,x2,64) for y in range(y1,y2,64)}
                    actual = {cell for cell, props in cells.items()
                              if props.get("tag") == marker["sector_tag"]}
                    self.assertEqual(actual, expected)
                    self.assertTrue(all(cells[cell]["floor"] == marker["locked_floor"] for cell in expected))
                    east_wall = {cell for cell, props in cells.items() if props.get("exit_east_wall")}
                    self.assertEqual(east_wall, {(x2-64,y) for y in range(y1,y2,64)})
                    self.assertTrue(all((x2,y) not in cells for y in range(y1,y2,64)))
                    # Continuous 128px sign spans adjacent 64-unit wall panels.
                    self.assertEqual({cells[cell]["exit_wall_offset"] for cell in east_wall}, {0,64})

    def test_invalid_build_inputs_are_rejected_before_tool_execution(self):
        for args in (("other",0,4),("relay",True,4),("cover",0,False),("relay",-1,4)):
            with self.subTest(args=args),self.assertRaises(ValueError):
                build_scenario(*args)


class ScenarioBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.compiler=compiler_path()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc))

    def test_actual_compiler_produces_deterministic_pwad_with_editable_sources(self):
        for task in ("relay","cover"):
            with self.subTest(task=task),tempfile.TemporaryDirectory() as directory:
                first=build_scenario(task,7,output_dir=Path(directory)/"first")
                second=build_scenario(task,7,output_dir=Path(directory)/"second")
                self.assertEqual(first["hashes"]["wad_sha256"],second["hashes"]["wad_sha256"])
                self.assertEqual(first["compiler"]["version"],"1.60")
                binary=Path(first["wad_path"]).read_bytes()
                magic,count,directory_start=struct.unpack_from("<4sii",binary)
                self.assertEqual(magic,b"PWAD")
                lumps={}
                for index in range(count):
                    offset,size,name=struct.unpack_from("<ii8s",binary,directory_start+index*16)
                    self.assertLessEqual(offset+size,directory_start)
                    lumps[name.rstrip(b"\0").decode()]=binary[offset:offset+size]
                self.assertTrue({"MAP01","TEXTMAP","BEHAVIOR","SCRIPTS","ENDMAP"}<=set(lumps))
                self.assertIn(lumps["BEHAVIOR"][:4],(b"ACS\0",b"ACSE",b"ACSe"))
                self.assertIn(b'namespace = "ZDoom"',lumps["TEXTMAP"])
                self.assertNotIn(b"@",lumps["SCRIPTS"])
                self.assertEqual(first["user_variables"]["registered_players"],10)
                self.assertEqual(first["user_variables"]["enemy_2_alive"],20)
                self.assertTrue({"FF_START", "FF_END", "TX_START", "TX_END"} <= set(lumps))
                from PIL import Image
                for name, asset in first["marker_assets"].items():
                    self.assertEqual(hashlib.sha256(lumps[name]).hexdigest(), asset["sha256"])
                    self.assertEqual(Path(asset["path"]).read_bytes(), lumps[name])
                    with Image.open(io.BytesIO(lumps[name])) as image:
                        self.assertEqual(image.size, (64,64) if name.startswith("EXF") else (128,128))
                self.assertNotEqual(lumps["EXFLOCK"], lumps["EXFOPEN"])
                self.assertNotEqual(lumps["EXSLOCK"], lumps["EXSOPEN"])

    def test_compiler_version_must_be_verified_not_only_assumed(self):
        fake=subprocess.CompletedProcess([],0,"This is version 9.99","")
        with tempfile.TemporaryDirectory() as directory,patch("world_models.env.scenario_builder.subprocess.run",return_value=fake):
            with self.assertRaisesRegex(RuntimeError,"version 1.60"):
                build_scenario("relay",output_dir=directory)


if __name__=="__main__":
    unittest.main()
