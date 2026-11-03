"""Independent post-collection P1.5/1.6 row audit; starts no engines.

Run after the final benchmark to audit all 10,000 stress rows without affecting
the benchmark. Before then only already completed mechanics JSON files are read.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
registry_path = ROOT / "privileged_registry.json"
registry = json.loads(registry_path.read_text())
registry_core = {k:v for k,v in registry.items() if k != "sha256"}
registry_sha = hashlib.sha256(json.dumps(registry_core,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
assert registry_sha == registry["sha256"]
wire_map = registry["user_variables"]
event_map = {item["name"]:item["source"] for item in registry["events"]}
native_sources = registry["native_counters"]
fixed = {"enemy_1_x","enemy_1_y","enemy_2_x","enemy_2_y"}
player_sources = {"position_x":"POSITION_X","position_y":"POSITION_Y","position_z":"POSITION_Z",
                  "velocity_x":"VELOCITY_X","velocity_y":"VELOCITY_Y","health":"HEALTH",
                  "armor":"ARMOR","ammo":"SELECTED_WEAPON_AMMO"}
TERMINALS = {"success","death","failure","deadline"}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def applicable(name,family):
    return not ((family=="relay" and (name.startswith("progress_") or name.startswith("console_")))
                or (family=="cover" and (name in ("plate_active","latch_active") or name.startswith("gate_"))))


def audit_rows(rows, scenario=None, initial=None):
    summary={"rows_audited":0,"checks_evaluated":0,"failed_checks":0,"first_failures":[],
             "boundary_counts":Counter(),"missing_image_slots":0,"fractional_npc_coordinate_rows":0,
             "episode_ids":set(),"counter_positive_deltas":Counter(),"max_once_counter":{},"episode_resets_in_trace":0}
    previous = dict(initial) if initial is not None else None
    previous_episode=None
    previous_native_baseline=None

    def check(condition,label,row_index):
        summary["checks_evaluated"]+=1
        if not condition:
            summary["failed_checks"]+=1
            if len(summary["first_failures"])<20:
                summary["first_failures"].append({"row":row_index,"check":label})

    def close(a,b):
        if a is None or b is None: return a is b
        return isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=0.,abs_tol=1e-12)

    for index,row in enumerate(rows,1):
        summary["rows_audited"]+=1
        info=row["training_only_info"]
        scene=info["scenario_manifest"] if scenario is None else scenario
        family=scene["task_family"]
        width,height=(scene["common_x"]+128,448) if family=="relay" else (scene["zones"]["extraction"][2],768)
        flags=info["task_flags"]
        events=info["events"]
        targets=info["privileged_targets"]
        reason=row["boundary"]
        summary["boundary_counts"][reason]+=1
        episode=info["episode_id"]
        summary["episode_ids"].add(episode)
        if previous_episode is not None and episode != previous_episode:
            summary["episode_resets_in_trace"]+=1
            check(episode>previous_episode,"new episode ID increases",index)
            previous=None
            previous_native_baseline=None
        cumulative=events["cumulative"]
        if previous is None: previous={name:0 for name in cumulative}
        check(set(previous)==set(cumulative)==set(events["deltas"]),"event schema stable",index)
        for name,value in cumulative.items():
            check(type(value) is int and value>=0,"nonnegative integral cumulative "+name,index)
            check(value>=previous[name] and events["deltas"][name]==value-previous[name],"delta "+name,index)
            if events["deltas"][name]: summary["counter_positive_deltas"][name]+=events["deltas"][name]
            if name in event_map:
                is_once=name not in ("plate_entries","plate_exits","gate_opens","gate_closes")
                if is_once:
                    check(value<=1,"one-time counter bounded "+name,index)
                    summary["max_once_counter"][name]=max(value,summary["max_once_counter"].get(name,0))
                if not name.startswith("player_"):
                    check(value==flags[event_map[name]],"ACS event snapshot "+name,index)
            valid=not ((family=="relay" and name.startswith(("console_","enemy_")))
                       or (family=="cover" and name in ("plate_entries","plate_exits","gate_opens","gate_closes","latch_activations")))
            check(events["valid"][name] is valid,"event applicability "+name,index)
            expected=value/(scene["deadline"]*scene["frame_skip"]*2) if valid else None
            check(close(events["normalized_cumulative"][name],expected),"event normalization "+name,index)
        if previous_native_baseline is not None:
            check(events["native_counter_baseline"]==previous_native_baseline,"native baseline immutable within episode",index)
        previous_native_baseline=dict(events["native_counter_baseline"])
        check(targets["schema_version"]==scene["privileged_registry_version"]==registry["schema_version"],"registry version",index)
        check(targets["registry_sha256"]==scene["privileged_registry_sha256"]==registry_sha,"registry hash",index)
        check(len(info["players"])==len(targets["players"])==2,"two privileged player slots",index)
        fraction=False
        for player,(native,target) in enumerate(zip(info["players"],targets["players"])):
            check(target["object_id"]==f"player_{player}","stable player ID",index)
            for name,source in player_sources.items():
                value=native[source]
                valid=name!="ammo" or value>=0
                check(target["valid"][name] is valid,"player validity "+name,index)
                check(close(target["raw"][name],value),"player raw source "+name,index)
                expected=None
                if valid:
                    if name=="position_x": expected=2*value/width-1
                    elif name=="position_y": expected=2*value/height-1
                    elif name=="position_z": expected=value/128
                    elif name.startswith("velocity"): expected=value/32
                    elif name in ("health","armor"): expected=max(value,0)/100
                    else: expected=value/200
                check(close(target["normalized"][name],expected),"player normalization "+name,index)
            for name,operation in (("heading_sin",math.sin),("heading_cos",math.cos)):
                expected=operation(math.radians(native["ANGLE"]))
                check(close(target["raw"][name],expected) and close(target["normalized"][name],expected),"heading conversion "+name,index)
                check(target["valid"][name] is True,"heading valid",index)
            check(target["raw"]["alive"]==int(native["HEALTH"]>0),"alive agrees with health",index)
            check(target["normalized"]["alive"]==target["raw"]["alive"] and target["valid"]["alive"] is True,"alive validity/normalization",index)
            for name,slot in wire_map.items():
                value=native[f"USER{slot}"]
                check(value==int(value),"integral ACS wire "+name,index)
                decoded=value/65536 if name in fixed else int(value)
                check(flags[name]==decoded,"ACS decoded source both peers "+name,index)
                fraction |= name in fixed and bool(value%65536)
            for source in native_sources:
                name=f"player_{player}_{source.lower()}"
                check(events["native_raw_counters"][name]==native[source],"native raw counter "+name,index)
                check(cumulative[name]==native[source]-events["native_counter_baseline"][name],"native baseline/cumulative "+name,index)
            death=f"player_{player}_deaths"
            expected=max(flags[f"player_{player}_death_count"],cumulative[f"player_{player}_deathcount"],int(native["HEALTH"]<=0))
            check(cumulative[death]==expected,"canonical death capture "+death,index)
        summary["fractional_npc_coordinate_rows"]+=int(fraction)
        for name,value in targets["task"]["raw"].items():
            valid=applicable(name,family)
            check(targets["task"]["valid"][name] is valid,"task applicability "+name,index)
            check(value==(flags[name] if valid else None),"task raw "+name,index)
            divisor=(scene.get("dwell_decisions",20) if name.startswith("progress_") else scene["deadline"] if name=="elapsed_decisions"
                     else (2 if family=="relay" else 3) if name=="stage" else (1 if family=="relay" else 2) if name=="milestone_count" else 1)
            check(close(targets["task"]["normalized"][name],flags[name]/divisor if valid else None),"task normalization "+name,index)
        objects={obj["object_id"]:obj for obj in targets["objects"]}
        check(len(objects)==len(targets["objects"])==len(scene["authored_objects"]),"unique authored object slots",index)
        for obj in scene["authored_objects"]:
            state=objects[obj["object_id"]]
            check({key:value for key,value in state.items() if key!="state"}==obj,"stable authored object metadata",index)
            if obj["kind"]=="enemy":
                name=obj["object_id"]; values=state["state"]
                spawned=name=="enemy_1" or flags["stage"]>=1
                alive=bool(flags[name+"_alive"])
                for axis,extent in (("x",width),("y",height)):
                    key="position_"+axis
                    check(values["valid"][key] is (spawned and alive),"enemy position mask",index)
                    check(values["raw"][key]==flags[name+"_"+axis],"enemy position source",index)
                    check(close(values["normalized"][key],2*flags[name+"_"+axis]/extent-1 if alive else None),"enemy position normalization",index)
                check(values["valid"]["health"] is spawned,"enemy health mask",index)
                check(values["raw"]["health"]==flags[name+"_health"],"enemy health source",index)
                check(close(values["normalized"]["health"],max(flags[name+"_health"],0)/20 if spawned else None),"enemy health normalization",index)
                check(values["raw"]["alive"]==int(alive) and values["raw"]["spawned"]==int(spawned),"enemy life state",index)
        components=info["reward_components"]
        bonus=(events["deltas"]["latch_activations"]*.1 if family=="relay" else
               (events["deltas"]["console_1_completions"]+events["deltas"]["console_2_completions"])*.05)
        terminal=1. if reason=="success" else -1. if reason in TERMINALS else 0.
        check(close(components["step"],-.001) and close(components["milestone"],bonus) and close(components["terminal"],terminal),"approved event-derived reward components",index)
        check(close(sum(components[k] for k in ("step","terminal","milestone")),row["reward"]) and components["total"]==row["reward"],"reward sums",index)
        check(components["shared_rewards"]==[row["reward"],row["reward"]],"same reward both players",index)
        for name,value in events["boundary_events"].items(): check(value==int(name==reason),"published boundary event "+name,index)
        valid=[value is not None for value in row["rgb_sha256"]]
        summary["missing_image_slots"]+=valid.count(False)
        check(valid==row["next_observation_valid"]==info["image_reconstruction_valid"],"actual RGB hash/validity mask agreement",index)
        check(all(type(value) is bool for value in row["next_observation_valid"]+info["image_reconstruction_valid"]),"image validity boolean types",index)
        check(row["bootstrap_mask"]==(0. if reason in TERMINALS else 1.),"terminal/truncation bootstrap",index)
        if reason not in TERMINALS: check(all(valid),"nonterminal next images available",index)
        check(info["decision_index"]==row["metadata"]["decision_index"]+1,"observation/transition alignment",index)
        actual=row["metadata"]["end_tic"]-row["metadata"]["start_tic"]
        check(0<actual<=scene["frame_skip"] and (actual==scene["frame_skip"] or reason in TERMINALS),"actual action-repeat tics",index)
        previous=dict(cumulative); previous_episode=episode
    summary["episode_ids"]=sorted(summary["episode_ids"])
    summary["boundary_counts"]=dict(summary["boundary_counts"])
    summary["counter_positive_deltas"]=dict(summary["counter_positive_deltas"])
    summary["status"]="passed" if summary["failed_checks"]==0 else "failed"
    return summary


campaign=json.loads((ROOT/"phase1_acceptance.json").read_text())
output_path=ROOT/"target_event_audit.json"
prior=json.loads(output_path.read_text()) if output_path.exists() else {}
cached=({case["path"]:case for case in prior.get("mechanics",[])}
        if prior.get("audit_program_sha256")==digest(Path(__file__)) else {})
cases=[]
for path in sorted((ROOT/"mechanics").rglob("mechanics_trace.json")):
    try:
        payload=path.read_bytes(); data=json.loads(payload)
    except (OSError,ValueError): continue
    if data.get("status")!="passed": continue
    signature=hashlib.sha256(payload).hexdigest()
    old=cached.get(str(path))
    if old and old["sha256"]==signature:
        cases.append(old); continue
    summary=audit_rows(data["records"],data["scenario_manifest"],data["initial_events"])
    summary.update(path=str(path),sha256=signature,task_family=data["task_family"],template_id=data["template_id"],
                   expected_rows=data["decisions"],sampling="all persisted mechanics rows",locality=data.get("locality"))
    if path.read_bytes()!=payload: raise RuntimeError("completed mechanics evidence changed during audit")
    cases.append(summary)
stress=[]
safe_for_heavy=campaign.get("status")=="complete_phase1" and campaign.get("stages",{}).get("benchmark",{}).get("status")=="passed"
if safe_for_heavy:
    for family in campaign["stages"]["stress"]["families"]:
        path=Path(family["trace_path"])
        signature=digest(path)
        assert signature==family["trace_sha256"]
        with path.open() as stream:
            summary=audit_rows((json.loads(line) for line in stream))
        summary.update(path=str(path),sha256=signature,task_family=family["task_family"],expected_rows=family["published_joint_decisions"],sampling="all persisted stress rows; read only after benchmark completed")
        assert summary["rows_audited"]==family["published_joint_decisions"]
        assert digest(path)==signature
        stress.append(summary)
event_probe=campaign.get("stages",{}).get("events")
boundaries=[]
for case in campaign.get("stages",{}).get("boundaries",{}).get("cases",[]):
    if "final_transition" not in case:
        continue
    row=case["final_transition"]
    events=row["training_only_info"]["events"]
    inferred_previous={name:value-events["deltas"][name] for name,value in events["cumulative"].items()}
    summary=audit_rows([row],initial=inferred_previous)
    summary.update(path=case["path"],sampling="one persisted final boundary transition; preceding event counts reconstructed from the retained delta, not independently replayed history")
    boundaries.append(summary)
result={"schema_version":1,"kind":"independent_readonly_privileged_target_and_event_trace_audit",
        "audited_at_utc":datetime.now(timezone.utc).isoformat(),"campaign_status_observed":campaign.get("status"),
        "registry_file":{"path":str(registry_path),"sha256":digest(registry_path),"canonical_registry_sha256":registry_sha},
        "audit_program_sha256":digest(Path(__file__)),"mechanics":cases,"stress":stress,"boundaries":boundaries,
        "event_probe_reported_result":event_probe,
        "sampling":"All rows of completed mechanics traces. All stress rows only once the final benchmark and parent campaign have completed.",
        "limits":["Alive labels are cross-checked with retained health; raw is_player_dead booleans are not separately serialized in these traces.",
                  "Native engine reward scalar is guarded against nonzero by the wrapper before publication, but that scalar is not retained in row JSON.",
                  "Reconstruction float-mask property is exercised by the native checker and unit tests; persisted JSON independently retains RGB hashes and boolean validity.",
                  "Locality evidence uses LocalContractProbe on actual image/action histories. It is not a trained B/D actor or a model-training result.",
                  "The dedicated native event probe retains reported selected endpoints/counter differences, not a complete independent per-step trace."]}
failed=any(case["status"]!="passed" for case in cases+stress+boundaries)
complete=safe_for_heavy and len(cases)==24 and len(stress)==2
result["status"]="failed" if failed else "passed" if complete else "in_progress"
result["totals"]={"mechanics_rows":sum(case["rows_audited"] for case in cases),"stress_rows":sum(case["rows_audited"] for case in stress),"final_boundary_rows":len(boundaries),
                  "checks_evaluated":sum(case["checks_evaluated"] for case in cases+stress+boundaries),"failed_checks":sum(case["failed_checks"] for case in cases+stress+boundaries),
                  "missing_image_slots_observed":sum(case["missing_image_slots"] for case in cases+stress+boundaries)}
output_path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":result["status"],"mechanics_cases":len(cases),"mechanics_rows":sum(c["rows_audited"] for c in cases),"stress_rows":sum(c["rows_audited"] for c in stress),"failed_checks":sum(c["failed_checks"] for c in cases+stress)}))
