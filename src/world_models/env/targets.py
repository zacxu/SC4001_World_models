"""Versioned privileged supervision and observation-aligned event deltas.

This module is deliberately absent from LocalActorInput. Missing supervision is
represented by None plus a false validity mask, never an invented zero target.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any


REGISTRY_VERSION = "privileged-targets-v1"
USER_FIELDS = {
    "stage":1,"success":2,"failure":3,"milestone_count":4,"progress_1":5,"progress_2":6,
    "plate_active":7,"latch_active":8,"elapsed_decisions":9,"registered_players":10,"start_tic":11,
    "enemy_health":12,"enemy_1_x":13,"enemy_1_y":14,"enemy_1_health":15,"enemy_1_alive":16,
    "enemy_2_x":17,"enemy_2_y":18,"enemy_2_health":19,"enemy_2_alive":20,
    "plate_entry_count":21,"plate_exit_count":22,"gate_open_count":23,"gate_close_count":24,
    "latch_activation_count":25,"console_1_completion_count":26,"console_2_completion_count":27,
    "success_count":28,"failure_count":29,"player_0_death_count":30,"player_1_death_count":31,
    "enemy_1_death_count":32,"enemy_2_death_count":33,"gate_1_open":34,"gate_2_open":35,
    "console_1_complete":36,"console_2_complete":37,"player_0_alive":38,"player_1_alive":39,"extraction_unlocked":40,
}
FIXED_POSITION_FIELDS = frozenset(("enemy_1_x", "enemy_1_y", "enemy_2_x", "enemy_2_y"))
EVENT_FIELDS = {
    "plate_entries": "plate_entry_count", "plate_exits": "plate_exit_count",
    "gate_opens": "gate_open_count", "gate_closes": "gate_close_count",
    "latch_activations": "latch_activation_count",
    "console_1_completions": "console_1_completion_count", "console_2_completions": "console_2_completion_count",
    "successes": "success_count", "failures": "failure_count",
    "player_0_deaths": "player_0_death_count", "player_1_deaths": "player_1_death_count",
    "enemy_1_deaths": "enemy_1_death_count", "enemy_2_deaths": "enemy_2_death_count",
}
NATIVE_COUNTERS = ("DEATHCOUNT", "HITCOUNT", "HITS_TAKEN", "DAMAGECOUNT", "DAMAGE_TAKEN", "KILLCOUNT", "ITEMCOUNT")
PLAYER_SOURCES = {
    "position_x": "POSITION_X", "position_y": "POSITION_Y", "position_z": "POSITION_Z",
    "velocity_x": "VELOCITY_X", "velocity_y": "VELOCITY_Y", "health": "HEALTH",
    "armor": "ARMOR", "ammo": "SELECTED_WEAPON_AMMO",
}
TASK_FIELDS = ("stage", "plate_active", "latch_active", "gate_1_open", "gate_2_open",
               "progress_1", "progress_2", "console_1_complete", "console_2_complete",
               "extraction_unlocked", "success", "failure", "milestone_count", "elapsed_decisions")


def registry_manifest() -> dict:
    fields = []
    for name, source in PLAYER_SOURCES.items():
        units = "map_units" if name.startswith("position") else "map_units_per_tic" if name.startswith("velocity") else "hit_points" if name in ("health", "armor") else "pistol_rounds"
        normalization = ("2*x/map_extent_x-1" if name == "position_x" else "2*y/map_extent_y-1" if name == "position_y"
                         else "z/128" if name == "position_z" else "value/32" if name.startswith("velocity")
                         else "max(value,0)/100" if name in ("health", "armor") else "value/200")
        fields.append({"scope":"player", "name":name, "source":source, "units":units,
                       "encoding":"native_ViZDoom_float", "normalization":normalization,
                       "validity":"finite source present; ammo >= 0; native state remains valid independently of terminal RGB"})
    fields.extend([
        {"scope":"player", "name":"heading_sin", "source":"ANGLE", "units":"unitless", "encoding":"sin(degrees*pi/180)", "normalization":"identity", "validity":"finite ANGLE"},
        {"scope":"player", "name":"heading_cos", "source":"ANGLE", "units":"unitless", "encoding":"cos(degrees*pi/180)", "normalization":"identity", "validity":"finite ANGLE"},
        {"scope":"player", "name":"alive", "source":"DoomGame.is_player_dead", "units":"boolean", "encoding":"0/1", "normalization":"identity", "validity":"native boolean always required"},
    ])
    for name in TASK_FIELDS:
        fields.append({"scope":"task", "name":name, "source":"ACS USER registry", "units":"decisions" if name.startswith("progress") or name == "elapsed_decisions" else "integer_or_boolean",
                       "encoding":"ACS integer", "normalization":"progress/dwell; elapsed/deadline; stage/(milestone_count_max+1); milestones/milestone_count_max; booleans identity",
                       "validity":"declared field and applicable family; plate/gates/latch only relay; consoles only cover"})
    fields.extend({"scope":"enemy", "name":name, "source":"ACS GetActorX/Y of stable TID", "units":"map_units",
                   "encoding":"signed_ACS_16_16", "normalization":"2*coordinate/map_extent-1",
                   "validity":"cover; enemy spawned and alive"} for name in sorted(FIXED_POSITION_FIELDS))
    fields.extend([
        {"scope":"enemy","name":"health","source":"ACS GetActorProperty(TID, APROP_Health)","units":"hit_points","encoding":"ACS integer clamped to nonnegative","normalization":"health/20","validity":"cover; enemy spawned, including dead health=0"},
        {"scope":"enemy","name":"alive","source":"ACS health > 0","units":"boolean","encoding":"0/1","normalization":"identity","validity":"cover; stable enemy slot, unspawned means alive=0"},
        {"scope":"enemy","name":"spawned","source":"authored stage schedule","units":"boolean","encoding":"0/1","normalization":"identity","validity":"cover; enemy1 at reset, enemy2 after console1"},
    ])
    core = {"schema_version":REGISTRY_VERSION, "actor_access":"forbidden; training_only_info only",
            "normalization_clipping":"none except negative health/armor clamp to zero; raw values retained",
            "fields":fields,"user_variables":dict(USER_FIELDS),
            "events":[{"name":name,"source":source,"units":"episode_cumulative_count", "validity":"nonnegative integral, nondecreasing within episode", "normalization":"raw integer; divide by deadline*frame_skip*2 for generic bounded auxiliary scaling"} for name,source in EVENT_FIELDS.items()],
            "native_counters":list(NATIVE_COUNTERS),
            "native_counter_baseline":"Raw native counters are baselined at every validated episode reset; ACS task counters must reset to zero. Deltas are differences of episode-relative cumulative values.",
            "reward_reservation":{"acs_global":0,"encoding":"signed_16_16","value":0.0,"purpose":"reserved engine reward, intentionally zero; one shared reward computed centrally"},
            "sources":["https://vizdoom.farama.org/environments/creating_custom/", "https://vizdoom.farama.org/api/python/enums/", "https://vizdoom.farama.org/api/python/utils/"]}
    digest = hashlib.sha256(json.dumps(core,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return {**core,"sha256":digest}


def authored_objects(scenario: dict) -> list[dict]:
    """IDs refer to map-authored TIDs/tags/zones, never renderer label indices."""
    objects = [{"object_id":f"player_{i}","kind":"player","engine_tid":100+i,"player_index":i} for i in (0,1)]
    for name,bounds in scenario.get("zones",{}).items():
        objects.append({"object_id":name,"kind":"authored_interaction_region","bounds":list(bounds),
                        "sector_tag":31 if name == "extraction" else None})
    if scenario["task_family"] == "relay":
        objects.extend({"object_id":f"gate_{i}","kind":"gate","sector_tag":10+i} for i in (1,2))
    else:
        objects.extend({"object_id":f"enemy_{i}","kind":"enemy","engine_tid":199+i,"initial_position":list(position)}
                       for i,position in enumerate(scenario["enemy_positions"],1))
    return objects


def decode_user_field(name: str, value: float, scenario: dict) -> int | float:
    if not math.isfinite(value) or value != int(value):
        raise ValueError(f"non-integral ACS wire value for {name}")
    if scenario.get("user_variable_encodings",{}).get(name) == "acs_fixed_16_16":
        return int(value) / 65536.0
    return int(value)


def _applicable(name: str, family: str) -> bool:
    return not ((family == "relay" and (name.startswith("progress_") or name.startswith("console_")))
                or (family == "cover" and (name in ("plate_active","latch_active") or name.startswith("gate_"))))


def _extent(scenario: dict) -> tuple[float,float]:
    if scenario["task_family"] == "relay":
        return float(scenario.get("common_x",896)+128),448.0
    return float(scenario.get("zones",{}).get("extraction",[0,0,1024,768])[2]),768.0


def _modern(scenario: dict) -> bool:
    version = scenario.get("privileged_registry_version")
    if version is None:
        return False  # Minimal legacy contract fixtures have explicit missing masks.
    registry = registry_manifest()
    if version != REGISTRY_VERSION or scenario.get("privileged_registry_sha256") != registry["sha256"]:
        raise ValueError("scenario privileged registry version/hash mismatch")
    if scenario.get("user_variables") != USER_FIELDS:
        raise ValueError("scenario USER mapping differs from the versioned privileged registry")
    if scenario.get("user_variable_encodings") != {name:"acs_fixed_16_16" for name in FIXED_POSITION_FIELDS}:
        raise ValueError("scenario USER wire encodings differ from the versioned privileged registry")
    if scenario.get("authored_objects") != authored_objects(scenario):
        raise ValueError("scenario authored object IDs or geometry differ from their registry")
    return True


def build_privileged_targets(raw: Any, flags: dict, scenario: dict) -> dict:
    modern = _modern(scenario)
    width,height = _extent(scenario)
    players = []
    for participant in raw.participants:
        values, normalized, validity = {},{},{}
        for name,source in PLAYER_SOURCES.items():
            value = participant.variables.get(source)
            valid = value is not None and math.isfinite(value) and (name != "ammo" or value >= 0)
            if modern and value is None:
                raise ValueError(f"missing declared privileged player target {source}")
            values[name] = float(value) if value is not None else None
            validity[name] = valid
            if not valid:
                normalized[name] = None
            elif name == "position_x": normalized[name] = 2*value/width-1
            elif name == "position_y": normalized[name] = 2*value/height-1
            elif name == "position_z": normalized[name] = value/128
            elif name.startswith("velocity"): normalized[name] = value/32
            elif name in ("health","armor"): normalized[name] = max(0,value)/100
            else: normalized[name] = value/200
        angle = participant.variables.get("ANGLE")
        if modern and angle is None:
            raise ValueError("missing declared privileged player target ANGLE")
        for name,operation in (("heading_sin",math.sin),("heading_cos",math.cos)):
            valid = angle is not None and math.isfinite(angle)
            values[name] = operation(math.radians(angle)) if valid else None
            normalized[name],validity[name] = values[name],valid
        values["alive"] = int(not participant.dead)
        normalized["alive"],validity["alive"] = values["alive"],True
        players.append({"object_id":f"player_{participant.player_index}","raw":values,"normalized":normalized,"valid":validity})
    task_raw, task_norm, task_valid = {},{},{}
    for name in TASK_FIELDS:
        valid = name in flags and _applicable(name,scenario["task_family"])
        value = flags.get(name) if valid else None
        task_raw[name],task_valid[name] = value,valid
        denominator = (scenario.get("dwell_decisions",20) if name.startswith("progress_") else scenario["deadline"] if name == "elapsed_decisions"
                       else (2 if scenario["task_family"] == "relay" else 3) if name == "stage"
                       else (1 if scenario["task_family"] == "relay" else 2) if name == "milestone_count" else 1)
        task_norm[name] = value/denominator if valid else None
    objects = deepcopy(scenario.get("authored_objects",[]))
    for obj in objects:
        name = obj["object_id"]
        if name.startswith("enemy_"):
            spawned = name == "enemy_1" or flags.get("stage",0)>=1
            alive = bool(flags.get(name+"_alive",0))
            state = {"position_x":flags.get(name+"_x"),"position_y":flags.get(name+"_y"),"health":flags.get(name+"_health"),"alive":int(alive),"spawned":int(spawned)}
            obj["state"] = {"raw":state,"valid":{"position_x":spawned and alive,"position_y":spawned and alive,"health":spawned,"alive":True,"spawned":True},
                            "normalized":{"position_x":2*state["position_x"]/width-1 if alive else None,
                                          "position_y":2*state["position_y"]/height-1 if alive else None,
                                          "health":max(0,state["health"])/20 if spawned else None,"alive":int(alive),"spawned":int(spawned)}}
    return {"schema_version":REGISTRY_VERSION,"registry_sha256":registry_manifest()["sha256"],
            "players":players,"task":{"raw":task_raw,"normalized":task_norm,"valid":task_valid},"objects":objects}


def native_counter_baseline(raw: Any) -> dict:
    return {f"player_{p.player_index}_{name.lower()}":int(p.variables[name])
            for p in raw.participants for name in NATIVE_COUNTERS if name in p.variables}


def event_snapshot(raw: Any, flags: dict, scenario: dict, previous: dict | None = None, *, boundary: Any = None,
                   native_baseline: dict | None = None) -> dict:
    modern = _modern(scenario)
    cumulative = {}
    native_baseline = native_counter_baseline(raw) if native_baseline is None else native_baseline
    native_raw = {}
    for name,source in EVENT_FIELDS.items():
        value = flags.get(source)
        if modern and value is None:
            raise ValueError(f"missing declared cumulative event counter {source}")
        if value is None:
            continue
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid cumulative event counter {name}")
        cumulative[name] = value
    for participant in raw.participants:
        for source in NATIVE_COUNTERS:
            value = participant.variables.get(source)
            if modern and value is None:
                raise ValueError(f"missing native event counter {source}")
            if value is None:
                continue
            if not math.isfinite(value) or value != int(value) or value < 0:
                raise ValueError(f"invalid native event counter {source}")
            key = f"player_{participant.player_index}_{source.lower()}"
            native_raw[key] = int(value)
            cumulative[key] = int(value)-native_baseline.get(key,0)
        # With automatic respawn disabled death is irreversible. The engine's
        # terminal dead flag also captures a death on the final sub-tic before
        # the ACS monitor gets its next turn to increment its own counter.
        name = f"player_{participant.player_index}_deaths"
        if modern:
            cumulative[name] = max(cumulative.get(name,0),cumulative[f"player_{participant.player_index}_deathcount"],int(participant.dead))
    previous = previous or {}
    if previous and set(previous) != set(cumulative):
        raise ValueError("event counter schema changed inside an episode")
    deltas = {}
    for name,value in cumulative.items():
        before = previous.get(name,0)
        if value < before:
            raise ValueError(f"authoritative cumulative event counter regressed: {name}")
        deltas[name] = value-before if previous else 0
    if modern:
        expected = (cumulative["latch_activations"] if scenario["task_family"] == "relay"
                    else cumulative["console_1_completions"]+cumulative["console_2_completions"])
        if expected != flags["milestone_count"]:
            raise ValueError("milestone counter disagrees with authoritative event counters")
    reason = boundary.reason.value if boundary is not None else None
    counter_scale = scenario["deadline"]*scenario.get("frame_skip",4)*2
    valid = {name:not ((scenario["task_family"] == "relay" and name.startswith(("console_","enemy_")))
                       or (scenario["task_family"] == "cover" and name in ("plate_entries","plate_exits","gate_opens","gate_closes","latch_activations")))
             for name in cumulative}
    return {"schema_version":REGISTRY_VERSION,"cumulative":cumulative,"deltas":deltas,
            "normalized_cumulative":{name:value/counter_scale if valid[name] else None for name,value in cumulative.items()},"valid":valid,
            "native_raw_counters":native_raw,"native_counter_baseline":dict(native_baseline),
            "boundary_events":{name:int(reason==name) for name in ("success","death","failure","deadline","collection_cutoff")}}


def reward_components(rule: Any, boundary: Any, milestone_bonus: float, total: float) -> dict:
    components = {"step":rule.step_cost,"terminal":rule.terminal_component(boundary),"milestone":milestone_bonus,"total":total,"shared_rewards":[total,total]}
    if not math.isclose(sum(components[name] for name in ("step","terminal","milestone")),total,abs_tol=1e-12):
        raise ValueError("shared reward components do not sum to the published reward")
    return components
