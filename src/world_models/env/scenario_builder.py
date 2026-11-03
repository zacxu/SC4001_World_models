"""Deterministic editable UDMF/ACS cooperative scenario generation.

Only original geometry/scripts/marker art are packaged. Engine/base textures come from the
separately installed, redistributable Freedoom base game bundled with ViZDoom.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
from collections import deque

from world_models.infrastructure.config import resolve_config


ROOT = Path(__file__).resolve().parents[3]
SCENARIO_VERSION = "coop-mechanics-v3"
USER_VARIABLES = {
    "stage": 1, "success": 2, "failure": 3, "milestone_count": 4,
    "progress_1": 5, "progress_2": 6, "plate_active": 7,
    "latch_active": 8, "elapsed_decisions": 9, "registered_players": 10,
    "start_tic": 11, "enemy_health": 12,
    # Minimal privileged fields for the action-only moving-enemy diagnostic.
    # These are never added to either actor's local observation.
    "enemy_1_x": 13, "enemy_1_y": 14, "enemy_1_health": 15, "enemy_1_alive": 16,
    "enemy_2_x": 17, "enemy_2_y": 18, "enemy_2_health": 19, "enemy_2_alive": 20,
    "plate_entry_count": 21, "plate_exit_count": 22,
    "gate_open_count": 23, "gate_close_count": 24,
    "latch_activation_count": 25, "console_1_completion_count": 26,
    "console_2_completion_count": 27, "success_count": 28, "failure_count": 29,
    "player_0_death_count": 30, "player_1_death_count": 31,
    "enemy_1_death_count": 32, "enemy_2_death_count": 33,
    "gate_1_open": 34, "gate_2_open": 35,
    "console_1_complete": 36, "console_2_complete": 37,
    "player_0_alive": 38, "player_1_alive": 39, "extraction_unlocked": 40,
}

EXTRACTION_SECTOR_TAG = 31
EXTRACTION_WALL_ID = 21


def _marker_assets() -> dict[str, bytes]:
    """Original deterministic art, independent of installed fonts/base textures."""
    from PIL import Image, ImageDraw

    glyphs = {
        "E": (31, 16, 16, 30, 16, 16, 31), "X": (17, 17, 10, 4, 10, 17, 17),
        "I": (31, 4, 4, 4, 4, 4, 31), "T": (31, 4, 4, 4, 4, 4, 4),
        "L": (16, 16, 16, 16, 16, 16, 31), "O": (14, 17, 17, 17, 17, 17, 14),
        "C": (15, 16, 16, 16, 16, 16, 15), "K": (17, 18, 20, 24, 20, 18, 17),
        "D": (30, 17, 17, 17, 17, 17, 30), "P": (30, 17, 17, 30, 16, 16, 16),
        "N": (17, 25, 25, 21, 19, 19, 17),
    }

    def word(draw, text, y, scale):
        x = (128 - (len(text) * 6 - 1) * scale) // 2
        for letter in text:
            for row, bits in enumerate(glyphs[letter]):
                for column in range(5):
                    if bits & (16 >> column):
                        left, top = x + column * scale, y + row * scale
                        draw.rectangle((left, top, left + scale - 1, top + scale - 1), fill="white")
            x += 6 * scale

    assets = {}
    for opened in (False, True):
        suffix = "OPEN" if opened else "LOCK"
        color = (24, 216, 80) if opened else (248, 168, 24)
        floor = Image.new("RGB", (64, 64), (12, 20, 24))
        draw = ImageDraw.Draw(floor)
        # High contrast checker/rim distinguishes the entire sector footprint.
        for y in range(0, 64, 16):
            for x in range(0, 64, 16):
                if (x // 16 + y // 16) % 2 == 0:
                    draw.rectangle((x, y, x + 15, y + 15), fill=color)
        draw.rectangle((1, 1, 62, 62), outline=color, width=4)
        stream = io.BytesIO()
        floor.save(stream, format="PNG", optimize=False)
        assets["EXF" + suffix] = stream.getvalue()
        sign = Image.new("RGB", (128, 128), (8, 16, 20))
        draw = ImageDraw.Draw(sign)
        draw.rectangle((3, 3, 124, 124), outline=color, width=6)
        draw.rectangle((9, 9, 118, 112), fill=(12, 112, 44) if opened else (128, 48, 8))
        word(draw, "EXIT", 18, 4)
        word(draw, "OPEN" if opened else "LOCKED", 55, 3)
        # A thick lock/check shape also communicates status when text is small.
        if opened:
            draw.line(((45, 93), (58, 106), (83, 81)), fill="white", width=8)
        else:
            draw.arc((51, 79, 77, 105), 180, 360, fill="white", width=5)
            draw.rectangle((49, 92, 79, 111), fill="white")
            draw.rectangle((62, 97, 66, 106), fill=(128, 48, 8))
        stream = io.BytesIO()
        sign.save(stream, format="PNG", optimize=False)
        assets["EXS" + suffix] = stream.getvalue()
    return assets


def _mark_extraction(params: dict, cells: dict) -> None:
    """One half-open rectangle drives both full-cell appearance and ACS success."""
    x1, y1, x2, y2 = params["zones"]["extraction"]
    if any(coordinate % 64 for coordinate in (x1, y1, x2, y2)):
        raise ValueError("Extraction must align to visible 64-unit sector boundaries")
    if any((x2, y) in cells for y in range(y1, y2, 64)):
        raise ValueError("Extraction sign requires an exposed eastern boundary wall")
    for x in range(x1, x2, 64):
        for y in range(y1, y2, 64):
            if (x, y) not in cells:
                raise ValueError("Extraction rectangle must contain only accessible sectors")
            cells[(x, y)].update(floor="EXFLOCK", tag=EXTRACTION_SECTOR_TAG, light=224,
                                  exit_east_wall=(x + 64 == x2),
                                  exit_wall_offset=(y2 - y - 64) % 128)
    params["extraction_marker"] = {
        "bounds": list(params["zones"]["extraction"]),
        "containment": "player center; x1 <= x < x2 and y1 <= y < y2",
        "sector_tag": EXTRACTION_SECTOR_TAG, "wall_line_id": EXTRACTION_WALL_ID,
        "locked_floor": "EXFLOCK", "unlocked_floor": "EXFOPEN",
        "locked_sign": "EXSLOCK", "unlocked_sign": "EXSOPEN",
        "sign_wall": "east", "unlock_condition": "latch" if params["task_family"] == "relay" else "both_consoles",
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiler_path() -> Path:
    explicit = os.environ.get("WORLD_MODELS_ACC")
    names = [Path(explicit)] if explicit else []
    names += [ROOT / ".tools" / "acc-1.60" / ("acc.exe" if os.name == "nt" else "acc")]
    if shutil.which("acc"):
        names.append(Path(shutil.which("acc")))
    for path in names:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("ACC 1.60 is required. See scenarios/README.md; set WORLD_MODELS_ACC if necessary.")


def _wad(path: Path, lumps: list[tuple[str, bytes]]) -> None:
    payload = bytearray()
    entries = []
    for name, contents in lumps:
        entries.append((12 + len(payload), len(contents), name.encode("ascii").ljust(8, b"\0")))
        payload.extend(contents)
    directory = b"".join(struct.pack("<ii8s", *entry) for entry in entries)
    path.write_bytes(struct.pack("<4sii", b"PWAD", len(lumps), 12 + len(payload)) + payload + directory)


def _textmap(cells: dict[tuple[int, int], dict], things: list[dict], wall: str) -> str:
    """Write connected 64-unit cells; ZDoom builds nodes for UDMF at load time."""
    vertices: dict[tuple[int, int], int] = {}
    edges: dict[tuple[int, int], dict] = {}
    sectors, sides = [], []
    for (x, y), properties in sorted(cells.items()):
        sector_index = len(sectors)
        sectors.append({"heightfloor": 0, "heightceiling": properties.get("ceiling", 128),
                        "texturefloor": properties.get("floor", "FLOOR4_8"),
                        "textureceiling": "CEIL3_5", "lightlevel": properties.get("light", 192),
                        "id": properties.get("tag", 0)})
        # Clockwise front sides face the cell interior.
        corners = [(x, y), (x, y + 64), (x + 64, y + 64), (x + 64, y)]
        indices = []
        for point in corners:
            if point not in vertices:
                vertices[point] = len(vertices)
            indices.append(vertices[point])
        for a, b in zip(indices, indices[1:] + indices[:1]):
            side = len(sides)
            exit_sign = properties.get("exit_east_wall", False) and (a, b) == (indices[2], indices[3])
            sides.append({"sector": sector_index, "texturemiddle": "EXSLOCK" if exit_sign else wall,
                          "texturetop": wall, "texturebottom": wall})
            if exit_sign:
                # Adjacent 64-unit linedefs display one continuous 128px sign.
                sides[-1]["offsetx"] = properties["exit_wall_offset"]
            opposite = edges.get((b, a))
            if opposite is not None:
                opposite["sideback"] = side
                opposite["twosided"] = True
                opposite["blocking"] = False
                sides[opposite["sidefront"]]["texturemiddle"] = "-"
                sides[side]["texturemiddle"] = "-"
            else:
                edges[(a, b)] = {"v1": a, "v2": b, "sidefront": side, "blocking": True}
                if exit_sign:
                    edges[(a, b)]["id"] = EXTRACTION_WALL_ID

    def block(kind: str, values: dict) -> str:
        def value(item):
            if type(item) is bool:
                return "true" if item else "false"
            if isinstance(item, str):
                return json.dumps(item)
            return str(item)
        return kind + "\n{\n" + "".join(f"  {key} = {value(item)};\n" for key, item in values.items()) + "}\n"

    text = 'namespace = "ZDoom";\n'
    for (x, y), _ in sorted(vertices.items(), key=lambda pair: pair[1]):
        text += block("vertex", {"x": x, "y": y})
    for kind, rows in (("sector", sectors), ("sidedef", sides), ("linedef", list(edges.values())), ("thing", things)):
        for row in rows:
            text += block(kind, row)
    return text


def _player(number: int, x: int, y: int, angle: int = 0) -> dict:
    return {"type": number + 1, "x": x, "y": y, "angle": angle,
            "skill1": True, "skill2": True, "skill3": True, "skill4": True,
            "skill5": True, "single": True, "coop": True, "dm": False}


def _zone(x: int, y: int, radius: int = 40) -> list[int]:
    return [x-radius, y-radius, x+radius, y+radius]


def _parameters(task: str, seed: int) -> tuple[dict, dict, list[dict]]:
    rng = random.Random(seed)
    swapped = bool(rng.randrange(2))
    role_assignment = {"operator": int(swapped), "runner": int(not swapped)}
    if task == "relay":
        common = rng.choice((768, 832))
        gate_x = rng.choice((256, 320))
        latch_x = common - 96
        zones = {"plate": _zone(144, 96, 40), "latch": _zone(latch_x, 352, 40),
                 "extraction": [common, 256, common+128, 448]}
        cells = {}
        for x in range(0, common+128, 64):
            for y in range(0, 448, 64):
                if y == 192 and x < common:
                    continue
                props = {}
                if x == gate_x and y >= 256:
                    props.update(tag=11, ceiling=0, floor="FLOOR0_1")
                if x == common-192 and y < 192:
                    props.update(tag=12, ceiling=0, floor="FLOOR0_1")
                if 96 <= x <= 160 and y in (64,128):
                    props.update(floor="FLOOR0_1", light=240)
                if x >= common:
                    props.update(floor="FLAT14", light=240)
                if x == common-128 and y == 320:
                    props.update(floor="FLOOR0_1", light=255)
                cells[(x,y)] = props
        # Original obstacle geometry changes with the seed, leaving center routes open.
        for position in [(rng.choice((384,448)), 128), (rng.choice((448,512)), 256)]:
            cells.pop(position, None)
        lower = (64,96)
        upper = (64,352)
        spawns = [upper,lower] if swapped else [lower,upper]
        points = {"operator": [[144,96], [common-64,96], [common+72,96], [common+72,352]],
                  "runner": [[latch_x,352], [common+72,352]]}
        params = {"task_family": task, "deadline": 400, "role_assignment": role_assignment,
                  "zones": zones, "spawns": spawns, "gate_x": gate_x,
                  "common_x": common, "scripted_waypoints": points}
    else:
        width = rng.choice((960,1024))
        cells = {(x,y): {} for x in range(0,width,64) for y in range(0,768,64)}
        # Two cover pillars beside, rather than on, the console-to-enemy sight lines.
        for x,y in [(448,256),(512,256),(448,448),(512,448)]:
            cells.pop((x,y),None)
        console1, console2 = (320,192),(704,576)
        zones = {"console_1": _zone(*console1,40), "console_2": _zone(*console2,40),
                 "extraction": [width-128,640,width,768]}
        for (x,y),props in cells.items():
            if abs(x-console1[0]) <= 64 and abs(y-console1[1]) <=64:
                props.update(floor="FLOOR0_1",light=224)
            if abs(x-console2[0]) <=64 and abs(y-console2[1]) <=64:
                props.update(floor="FLAT14",light=224)
        positions = [(192,192),(192,352)]
        spawns = list(reversed(positions)) if swapped else positions
        points = {"operator": [[320,192],[704,576],[width-64,672]],
                  "runner": [[448,192],[704,576],[width-64,672]]}
        params = {"task_family": task, "deadline": 600, "role_assignment": role_assignment,
                  "zones": zones, "spawns": spawns, "scripted_waypoints": points,
                  "enemy_positions": [[704,192],[256,576]], "enemy_ids": [200,201],
                  "dwell_decisions": 20}
    _mark_extraction(params, cells)
    params["wall_texture"] = rng.choice(("STARTAN3","BROWN1","STONE2"))
    things = [_player(index,*position) for index,position in enumerate(params["spawns"])]
    return params,cells,things


def _template_parameters(task: str, seed: int, template_id: str, *, split: str | None = None,
                         purpose: str = "diagnostic", randomization_bank: str | None = None) -> tuple[dict, dict, list[dict]]:
    """Realize fixed geometry and independently versioned nuisance draws."""
    from world_models.env.splits import get_template, randomization_draw
    template, bank, provenance = get_template(task, template_id, split=split, purpose=purpose,
                                             randomization_bank=randomization_bank)
    draw = randomization_draw(task, seed, bank)
    geometry = template["geometry"]
    operator = draw["operator"]
    pair = draw["spawn_pair"]
    spawns = [pair["operator"], pair["runner"]] if operator == 0 else [pair["runner"], pair["operator"]]
    roles = {"operator": operator, "runner": 1 - operator}
    if task == "relay":
        common, gate_x = geometry["common_x"], geometry["gate_x"]
        latch_x = common - 96
        zones = {"plate": _zone(144, 96, 40), "latch": _zone(latch_x, 352, 40),
                 "extraction": [common, 256, common + 128, 448]}
        cells = {}
        for x in range(0, common + 128, 64):
            for y in range(0, 448, 64):
                if y == 192 and x < common:
                    continue
                props = {}
                if x == gate_x and y >= 256:
                    props.update(tag=11, ceiling=0, floor="FLOOR0_1")
                if x == common - 192 and y < 192:
                    props.update(tag=12, ceiling=0, floor="FLOOR0_1")
                if 96 <= x <= 160 and y in (64, 128):
                    props.update(floor="FLOOR0_1", light=240)
                if x >= common:
                    props.update(floor="FLAT14", light=240)
                if x == common - 128 and y == 320:
                    props.update(floor="FLOOR0_1", light=255)
                cells[(x, y)] = props
        for position in geometry["obstacles"]:
            if tuple(position) not in cells:
                raise ValueError("template obstacle is outside the authored floor")
            cells.pop(tuple(position))
        points = {"operator": [[144, 96], [common - 64, 96], [common + 72, 96], [common + 72, 352]],
                  "runner": [[latch_x, 352], [common + 72, 352]]}
        params = {"task_family": task, "deadline": 400, "role_assignment": roles,
                  "zones": zones, "spawns": spawns, "gate_x": gate_x,
                  "common_x": common, "scripted_waypoints": points}
    else:
        width = geometry["width"]
        cells = {(x, y): {} for x in range(0, width, 64) for y in range(0, 768, 64)}
        for position in geometry["pillars"]:
            if tuple(position) not in cells:
                raise ValueError("template pillar is outside the authored floor")
            cells.pop(tuple(position))
        console1, console2 = (320, 192), (704, 576)
        zones = {"console_1": _zone(*console1, 40), "console_2": _zone(*console2, 40),
                 "extraction": [width - 128, 640, width, 768]}
        for (x, y), props in cells.items():
            if abs(x - console1[0]) <= 64 and abs(y - console1[1]) <= 64:
                props.update(floor="FLOOR0_1", light=224)
            if abs(x - console2[0]) <= 64 and abs(y - console2[1]) <= 64:
                props.update(floor="FLAT14", light=224)
        points = {"operator": [[320, 192], [704, 576], [width - 64, 672]],
                  "runner": [[448, 192], [704, 576], [width - 64, 672]]}
        params = {"task_family": task, "deadline": 600, "role_assignment": roles,
                  "zones": zones, "spawns": spawns, "scripted_waypoints": points,
                  "enemy_positions": draw["enemy_positions"], "enemy_ids": [200, 201],
                  "dwell_decisions": 20}
    _mark_extraction(params, cells)
    params.update(wall_texture=draw["wall_texture"], template_id=template_id,
                  split_provenance=provenance, randomization_draw=draw["indices"],
                  geometry_template=geometry)
    return params, cells, [_player(index, *position) for index, position in enumerate(spawns)]


def geometry_fingerprint(params: dict, cells: dict) -> str:
    """Hash realized collision/door/interaction geometry, excluding appearance."""
    payload = {"task_family": params["task_family"], "zones": params["zones"],
               "cells": [[x, y, props.get("ceiling", 128), props.get("tag", 0)]
                         for (x, y), props in sorted(cells.items())]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_geometry(params: dict, cells: dict) -> dict:
    """Check cell-center routes and the relay dependency before compilation.

    A 64-unit traversable corridor leaves clearance for the ordinary player
    radius. Native scripted checks additionally validate collisions and actions.
    """
    def reachable(start, target, opened=()):
        begin = (int(start[0] // 64) * 64, int(start[1] // 64) * 64)
        end = (int(target[0] // 64) * 64, int(target[1] // 64) * 64)
        allowed = {point for point, props in cells.items()
                   if props.get("ceiling", 128) >= 56 or props.get("tag") in opened}
        if begin not in allowed or end not in allowed:
            return False
        seen, pending = {begin}, deque([begin])
        while pending:
            x, y = pending.popleft()
            if (x, y) == end:
                return True
            for point in ((x + 64, y), (x - 64, y), (x, y + 64), (x, y - 64)):
                if point in allowed and point not in seen:
                    seen.add(point)
                    pending.append(point)
        return False

    bounds = params["zones"]["extraction"]
    extraction = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    def inside_task_zone(point, name, zone):
        x, y = point
        x1, y1, x2, y2 = zone
        if name == "extraction":
            return x1 <= x < x2 and y1 <= y < y2
        # ACS interaction zones use inclusive integer bounds. A fresh spawn on
        # the plate would immediately create occupancy/entry events during
        # reset warm-up, violating the validated zero-state reset contract.
        return x1 <= x <= x2 and y1 <= y <= y2

    checks = {"both_spawn_points_on_floor": all(tuple((int(p[0] // 64) * 64, int(p[1] // 64) * 64)) in cells for p in params["spawns"]),
              "fresh_spawns_outside_task_zones": all(not inside_task_zone(point, name, zone)
                  for point in params["spawns"] for name, zone in params["zones"].items())}
    if params["task_family"] == "relay":
        operator = params["role_assignment"]["operator"]
        runner = 1 - operator
        latch = params["scripted_waypoints"]["runner"][0]
        checks.update(operator_can_reach_plate=reachable(params["spawns"][operator], (144, 96)),
                      runner_blocked_before_plate=not reachable(params["spawns"][runner], latch),
                      runner_can_pass_when_plate_opens_gate=reachable(params["spawns"][runner], latch, (11,)),
                      plate_holder_cannot_do_runner_role_before_latch=not reachable(params["spawns"][operator], latch, (11,)),
                      both_can_extract_after_latch=all(reachable(start, extraction, (11, 12)) for start in params["spawns"]))
    else:
        checks.update(all_consoles_reachable=all(reachable(start, console) for start in params["spawns"] for console in ((320, 192), (704, 576))),
                      both_can_reach_extraction=all(reachable(start, extraction) for start in params["spawns"]),
                      authored_enemies_on_reachable_floor=all(reachable(params["spawns"][0], point) for point in params["enemy_positions"]))
    marked = {point for point, props in cells.items() if props.get("tag") == EXTRACTION_SECTOR_TAG}
    expected = {(x, y) for x in range(bounds[0], bounds[2], 64) for y in range(bounds[1], bounds[3], 64)}
    checks["marked_extraction_matches_success_rectangle"] = marked == expected
    if not all(checks.values()):
        raise ValueError(f"generated layout failed geometry/mechanic validation: {checks}")
    return checks


def build_scenario(task_family: str, seed: int = 0, frame_skip: int = 4,
                   output_dir: str | Path | None = None, *, template_id: str | None = None,
                   split: str | None = None, purpose: str = "diagnostic",
                   randomization_bank: str | None = None) -> dict:
    if task_family not in ("relay","cover"):
        raise ValueError("task_family must be relay or cover")
    if type(seed) is not int or seed < 0 or type(frame_skip) is not int or frame_skip < 1:
        raise ValueError("seed must be nonnegative and frame_skip positive integers")
    if template_id is None and (split is not None or randomization_bank is not None or purpose != "diagnostic"):
        raise ValueError("formal training/evaluation requires an explicitly allocated template")
    params,cells,things = (_parameters(task_family,seed) if template_id is None else
                          _template_parameters(task_family,seed,template_id,split=split,purpose=purpose,
                                               randomization_bank=randomization_bank))
    geometry_checks = validate_geometry(params, cells)
    default = (ROOT / "artifacts" / "phase1" / "scenarios" / f"{task_family}_{seed}_f{frame_skip}" if template_id is None
               else ROOT / "artifacts" / "phase1" / "split_scenarios" / template_id / f"{seed}_f{frame_skip}")
    target = Path(output_dir) if output_dir is not None else default
    target.mkdir(parents=True,exist_ok=True)
    task_config = resolve_config(task_family,"B","pilot",ROOT/"configs").settings["task"]
    params["deadline"] = task_config["deadline"]
    if task_family == "cover":
        params["dwell_decisions"] = task_config["dwell_decisions"]
    params.update(seed=seed,frame_skip=frame_skip,scenario_version=SCENARIO_VERSION)
    template = ROOT / "scenarios" / task_family / "task.acs"
    acs = template.read_text(encoding="utf-8")
    replacements = {"FRAME_SKIP":frame_skip,"DEADLINE":params["deadline"],
                    "DWELL":task_config["dwell_decisions"],
                    "OPERATOR":params["role_assignment"]["operator"],
                    "EXTRACTION_SECTOR_TAG":EXTRACTION_SECTOR_TAG,
                    "EXTRACTION_WALL_ID":EXTRACTION_WALL_ID}
    for name,zone in params["zones"].items():
        for suffix,value in zip(("X1","Y1","X2","Y2"),zone):
            replacements[f"{name.upper()}_{suffix}"] = value
    for index,(x,y) in enumerate(params["spawns"]):
        replacements[f"SPAWN{index}_X"] = x
        replacements[f"SPAWN{index}_Y"] = y
    for index,(x,y) in enumerate(params.get("enemy_positions", []), start=1):
        replacements[f"ENEMY{index}_X"] = x
        replacements[f"ENEMY{index}_Y"] = y
    for key,value in replacements.items():
        acs = acs.replace("@"+key+"@",str(value))
    if "@" in acs:
        raise ValueError("unresolved scenario template placeholder")
    source = target / "task.acs"
    source.write_text(acs,encoding="utf-8")
    behavior = target / "BEHAVIOR.o"
    compiler = compiler_path()
    completed = subprocess.run([str(compiler),"-i",str(compiler.parent),str(source.resolve()),str(behavior.resolve())],capture_output=True,text=True,timeout=30)
    if completed.returncode:
        raise RuntimeError(f"ACS compilation failed: {completed.stdout}\n{completed.stderr}")
    banner=completed.stdout+"\n"+completed.stderr
    version=re.search(r"This is version\s+(\d+\.\d+)",banner)
    if not version or version.group(1)!="1.60":
        raise RuntimeError("The scenario compiler must report ACC version 1.60")
    textmap = _textmap(cells,things,params["wall_texture"])
    (target/"TEXTMAP.txt").write_text(textmap,encoding="utf-8")
    wad = target / f"{task_family}.wad"
    marker_assets = _marker_assets()
    for name, contents in marker_assets.items():
        (target / (name + ".png")).write_bytes(contents)
    _wad(wad,[("MAP01",b""),("TEXTMAP",textmap.encode()),("BEHAVIOR",behavior.read_bytes()),
              ("SCRIPTS",acs.encode()),("ENDMAP",b""),
              ("MAPINFO",f'map MAP01 "Cooperative {task_family}" {{ next = "MAP01" }}\n'.encode()),
              ("FF_START", b""), ("EXFLOCK", marker_assets["EXFLOCK"]),
              ("EXFOPEN", marker_assets["EXFOPEN"]), ("FF_END", b""),
              ("TX_START", b""), ("EXSLOCK", marker_assets["EXSLOCK"]),
              ("EXSOPEN", marker_assets["EXSOPEN"]), ("TX_END", b"")])
    from world_models.env.targets import authored_objects, registry_manifest
    registry = registry_manifest()
    result = {**params,"wad_path":str(wad.resolve()),"acs_path":str(source.resolve()),
              "textmap_path":str((target/"TEXTMAP.txt").resolve()),"user_variables":USER_VARIABLES,
              "geometry_checks":geometry_checks,
              "data_provenance":{"purpose":purpose,"learning_eligible":purpose == "training",
                                 "diagnostic_data_enters_replay":False,"diagnostic_import_requires_training_ledger":True},
              "privileged_registry_version":registry["schema_version"],
              "privileged_registry_sha256":registry["sha256"],"authored_objects":authored_objects(params),
              "user_variable_encodings":{name:"acs_fixed_16_16" for name in ("enemy_1_x","enemy_1_y","enemy_2_x","enemy_2_y")},
              "hashes":{"wad_sha256":_sha(wad),"acs_sha256":_sha(source),
                        "geometry_sha256":geometry_fingerprint(params,cells),
                        "compiler_sha256":_sha(compiler),"template_sha256":_sha(template)},
              "compiler":{"version":version.group(1),"path":str(compiler),
                          "source":"https://github.com/ZDoom/acc/releases/tag/1.60"},
              "marker_assets": {name: {"path": str((target / (name + ".png")).resolve()),
                                         "sha256": hashlib.sha256(contents).hexdigest(),
                                         "provenance": "original deterministic pixel art in scenario_builder.py"}
                                for name, contents in marker_assets.items()}}
    (target/"manifest.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result
