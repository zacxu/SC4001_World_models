"""Independent, read-only audit of P2.1 native journals and real replay debits.

This standalone evidence helper imports no project modules. It streams one
episode and one NPZ chunk at a time. It never launches a simulator or writes into
an audited dataset. Main source/config/test fingerprints therefore stay frozen.

Usage:
  python native_journal_audit.py --datasets CASE_DIR ... --output AUDIT.json

Run before hot-ledger cleanup by default. For a later relocated cold-only audit,
pass --allow-cleaned-hot-ledgers; archived member hashes remain mandatory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

VERSION = "phase21-native-journal-audit-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Checks:
    def __init__(self):
        self.count = 0

    def require(self, condition, message):
        self.count += 1
        if not condition:
            raise ValueError(message)


def inside(root, relative, checks):
    member = Path(relative)
    checks.require(not member.is_absolute() and not member.drive and ".." not in member.parts,
                   "archive/data member is not a safe relative path")
    path = (root/member).resolve()
    checks.require(path.is_relative_to(root.resolve()), "archive/data path escapes dataset")
    return path


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class LedgerStream:
    """Validate monotonic sequence while retaining only a current episode."""
    def __init__(self, path, checks):
        self.file = Path(path).open(encoding="utf-8")
        self.checks = checks
        self.sequence = 0
        self.totals = Counter()
        self.kinds = Counter()

    def next(self):
        line = self.file.readline()
        if not line:
            return None
        row = json.loads(line)
        self.checks.require(row["sequence"] == self.sequence, "training ledger sequence has a gap")
        self.sequence += 1
        for field in ("physical_delta", "initialization_delta", "published_delta", "failed_delta"):
            value = row.get(field, 0)
            self.checks.require(type(value) is int and value >= 0, "negative/nonintegral training debit")
            self.totals[field] += value
        if row["event"] == "transition":
            self.kinds[row["collector_kind"]] += 1
            self.totals["published_tics"] += row["actual_tics"]
        elif row["event"] == "episode_committed":
            self.totals["committed"] += row["valid_length"]
        return row

    def episode(self, episode_id):
        reset = self.next()
        self.checks.require(reset is not None and reset["event"] == "reset"
                            and reset["episode_id"] == episode_id,
                            "episode does not begin with its accounted reset")
        transitions = []
        while True:
            row = self.next()
            self.checks.require(row is not None and row.get("episode_id") == episode_id,
                                "training ledger crosses episode before commit")
            if row["event"] == "episode_committed":
                return reset, transitions, row
            self.checks.require(row["event"] == "transition", "unexpected error/retry in a completed dataset")
            transitions.append(row)


def archive_inventory(archive, expected, checks):
    receipt_path = archive/"archive_receipt.json"
    checks.require(sha256(receipt_path) == expected["receipt_sha256"], "archive receipt bytes changed")
    receipt = json_read(receipt_path)
    checks.require(receipt["status"] == "verified_closed_archive", "runtime archive was not verified")
    source = Path(receipt["source_path_at_collection"])
    checks.require(not source.exists() and expected.get("source_removed") is True,
                   "closed hot runtime source still exists")
    expected_files = {record["path"]: record for record in receipt["files"]}
    actual_files, actual_dirs = set(), set()
    for path in archive.rglob("*"):
        checks.require(not path.is_symlink(), "symbolic link in a runtime archive")
        relative = path.relative_to(archive).as_posix()
        if path.is_file() and relative != "archive_receipt.json":
            actual_files.add(relative)
        elif path.is_dir():
            actual_dirs.add(relative)
    checks.require(actual_files == set(expected_files), "runtime archive file inventory changed")
    checks.require(actual_dirs == set(receipt["directories"]), "runtime archive directories changed")
    for relative, record in expected_files.items():
        path = inside(archive, relative, checks)
        checks.require(path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"],
                       "archived runtime member bytes changed")
    content = {"directories": receipt["directories"], "files": receipt["files"]}
    checks.require(canonical_hash(content) == receipt["content_sha256"], "runtime inventory identity mismatch")
    checks.require(receipt["file_count"] == len(expected_files)
                   and receipt["total_bytes"] == sum(r["bytes"] for r in expected_files.values()),
                   "runtime archive totals disagree")
    return receipt


def native_episode(journal, episode, reset, transitions, checks):
    """Reconcile every native sub-tic dispatch/ACK, preserving terminal shortening."""
    dispatches, acknowledgements, captures, cleanups = [], [], [], []
    with journal.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            kind = row["kind"]
            checks.require(kind in ("dispatch", "acknowledged", "native_processes", "native_process_cleanup"),
                           "native error/retry requires explicit independent review")
            if kind == "dispatch":
                dispatches.append(row)
            elif kind == "acknowledged":
                acknowledgements.append(row)
            elif kind == "native_processes":
                captures.append(row)
            else:
                cleanups.append(row)
    checks.require(len(dispatches) == len(acknowledgements) > 1, "unacknowledged native dispatch")
    checks.require([r["dispatch_id"] for r in dispatches] == list(range(1,len(dispatches)+1)),
                   "native dispatch IDs are not contiguous")
    checks.require([r["dispatch_id"] for r in acknowledgements] == list(range(1,len(acknowledgements)+1)),
                   "native ACK IDs are not contiguous")
    groups = defaultdict(list)
    previous_tic = None
    start_tic = None
    for sent, received in zip(dispatches, acknowledgements):
        for key in ("episode_id", "decision_index", "dispatch_id", "consumed_joint_decisions", "command"):
            checks.require(sent[key] == received[key], "dispatch and ACK identity differ")
        checks.require(sent["episode_id"] == episode["engine_episode_id"], "native engine episode changed")
        tics = received["tics"]
        checks.require(len(tics) == 2 and tics[0] == tics[1], "participant native tics disagree")
        if previous_tic is None:
            checks.require(sent["command"] == "start" and sent["consumed_joint_decisions"] == 0,
                           "native journal lacks a fresh startup")
            start_tic = tics[0]
            iwads = received["selected_iwads"]
            checks.require(len(iwads) == 2 and iwads[0]["sha256"] == iwads[1]["sha256"]
                           and iwads[0]["path"] == iwads[1]["path"],
                           "participants loaded different IWADs")
        else:
            checks.require(sent["command"] == "step" and tics[0] == previous_tic+1,
                           "native advancement is not one synchronized tic")
            groups[sent["decision_index"]].append((sent,received))
        previous_tic = tics[0]
    initialization = reset["initialization_delta"]
    checks.require(reset["physical_delta"] == initialization >= 1, "reset debit differs from native initialization")
    checks.require(sorted(groups) == list(range(1,initialization+len(transitions)+1)),
                   "native logical decisions have a gap")
    repeat = episode["action_repeat"]
    for decision, group in groups.items():
        checks.require(1 <= len(group) <= repeat, "invalid repeated-action native duration")
        checks.require([r[0]["substep_index"] for r in group] == list(range(len(group))),
                       "repeated-action sub-tic IDs have a gap")
        checks.require(all(r[0]["substep_count"] == repeat for r in group), "action-repeat declaration changed")
        checks.require(all(r[0]["consumed_joint_decisions"] == decision for r in group),
                       "physical consumed counter not stable within a joint decision")
        checks.require(all(r[0]["actions"] == group[0][0]["actions"] for r in group),
                       "player actions changed inside a repeated decision")
        if decision <= initialization:
            checks.require(len(group) == repeat and group[0][0]["actions"] == [0,0],
                           "reset initialization was not complete synchronized no-op stepping")
    checks.require(reset["physical_consumed_at_source"] == initialization,
                   "reset source counter does not match actual native dispatches")
    checks.require(len(captures) == len(cleanups) == 1, "engine ownership capture/cleanup missing")
    checks.require(len(captures[0]["processes"]) == len(cleanups[0]["processes"]) == 2,
                   "episode did not own exactly two native engines")
    checks.require([row["player_index"] for row in captures[0]["processes"]]==[0,1]
                   and len({(row["pid"],row["create_time"]) for row in captures[0]["processes"]})==2
                   and len({row["instance_id"] for row in captures[0]["processes"]})==2,
                   "two players did not own distinct captured native engines")
    cleanup_counts = Counter()
    for source, target in zip(captures[0]["processes"],cleanups[0]["processes"]):
        for key in ("player_index","pid","create_time","executable","instance_id"):
            checks.require(source[key] == target[key], "cleanup does not reference captured native identity")
        checks.require(target["cleanup"] in (
            "already_exited", "terminated_owned_engine", "identity_changed_no_signal_sent",
            "identity_changed_before_force_no_signal_sent"), "owned native cleanup failed")
        cleanup_counts[target["cleanup"]] += 1
    return {
        "groups":groups, "initialization":initialization,
        "dispatches":len(dispatches), "acknowledgements":len(acknowledgements),
        "physical_decisions":len(groups), "physical_tics":previous_tic-start_tic,
        "initialization_tics":sum(len(groups[i]) for i in range(1,initialization+1)),
        "cleanup_counts":cleanup_counts,
    }


def audit_dataset(path, checks, *, allow_cleaned_hot=False, process=None):
    import numpy as np
    output=Path(path).resolve()
    report=json_read(output/"collection_report.json")
    checks.require(report["status"] == "complete_initial_replay", "collection is incomplete")
    manifest_path=output/"replay/manifest.json"
    manifest=json_read(manifest_path)
    checks.require(sha256(manifest_path) == report["dataset_manifest_sha256"], "replay manifest changed")
    checks.require(sha256(output/"collection_spec.json") == report["collection_spec_sha256"], "collection specification changed")
    checks.require(manifest["status"] == "sealed", "replay is not sealed")
    checks.require(canonical_hash({k:v for k,v in manifest.items() if k not in ("sealed_at_utc","dataset_identity_sha256")})
                   == manifest["dataset_identity_sha256"] == report["dataset_identity_sha256"],
                   "dataset content identity mismatch")
    metadata=manifest["metadata"]
    checks.require(report["learning_eligible"] == metadata["learning_eligible"]
                   == (report["budget_scope"]=="approved_initial"),
                   "preflight/learning eligibility differs between report and store")
    episodes=manifest["episodes"]
    checks.require(len(episodes)==report["completed_episodes"],"episode count differs")
    event_fields=manifest["schema"]["event_fields"]
    reasons=manifest["schema"]["termination_reasons"]
    archives=report["journal_storage"]["runtime_archives"]
    storage=report["journal_storage"]
    checks.require(storage["spooled"] is True, "checker requires the archived bounded-journal pipeline")
    checks.require(0 < storage["peak_hot_bytes"] <= storage["maximum_hot_bytes"], "hot spool bound was exceeded")
    archive_env={}
    archive_keys=set()
    archive_types=Counter()
    archive_files=archive_bytes=0
    for entry in archives:
        archive=inside(output,entry["archive_path"],checks)
        receipt=archive_inventory(archive,entry,checks)
        key=(entry["episode_id"],entry["category"])
        checks.require(key not in archive_keys,"duplicate archived episode/category")
        archive_keys.add(key)
        archive_types[entry["category"]]+=1
        archive_files+=receipt["file_count"]
        archive_bytes+=receipt["total_bytes"]
        if entry["category"]=="environments":
            checks.require(entry["episode_id"] not in archive_env,"duplicate environment archive")
            archive_env[entry["episode_id"]]=(archive,Path(receipt["source_path_at_collection"]))
    checks.require(archive_types=={"scenarios":len(episodes),"environments":len(episodes)}
                   and archive_keys=={(episode["episode_id"],category) for episode in episodes
                                      for category in ("scenarios","environments")},
                   "not every completed episode has both runtime archive categories")
    ledger_path=output/"training_interactions.jsonl"
    accounting=report["interaction_accounting"]
    checks.require(sha256(ledger_path)==accounting["ledger_sha256"]
                   == storage["ledger_archive"]["sha256"],"cold ledger changed")
    hot_ledger=Path(storage["ledger_hot_path_at_collection"])
    hot_exists=hot_ledger.is_file()
    checks.require(hot_exists or allow_cleaned_hot, "hot ledger missing before final verified cleanup")
    if hot_exists:
        checks.require(sha256(hot_ledger)==sha256(ledger_path), "hot/cold ledgers differ")
    ledger=LedgerStream(ledger_path,checks)
    created=ledger.next()
    checks.require(created["event"]=="created" and created["learning_eligible"]==report["learning_eligible"],
                   "ledger creation eligibility is inconsistent")
    checks.require(created["task_family"]==metadata["task_family"]==report["task_family"]
                   and created["run_seed"]==metadata["run_seed"]==report["run_seed"],
                   "report, dataset, and training ledger identities disagree")
    totals=Counter(); mixture=Counter(); event_counts=Counter(); ends=Counter(); cleanup_counts=Counter()
    checks_start=checks.count
    next_transition=0
    peak_rss=process.memory_info().rss if process else None
    try:
        for episode in episodes:
            episode_id=episode["episode_id"]
            reset,transitions,commit=ledger.episode(episode_id)
            checks.require(len(transitions)==commit["valid_length"]==episode["valid_length"],
                           "committed episode length differs from charged real decisions")
            checks.require(episode["transition_id_start"]==next_transition,"global real transition sequence has a gap")
            checks.require(episode["learning_eligible"]==report["learning_eligible"],"episode eligibility disagrees")
            archive,original_root=archive_env[episode_id]
            original_env=Path(reset["coordinator_run_dir"])
            checks.require(original_env.is_relative_to(original_root),"native source path does not match relocation map")
            cold_env=(archive/original_env.relative_to(original_root)).resolve()
            checks.require(cold_env.is_relative_to(archive.resolve()),"native journal mapping escaped archive")
            journals=list(cold_env.rglob("dispatch_ledger.jsonl"))
            checks.require(len(journals)==1,"expected one native coordinator journal per episode")
            native=native_episode(journals[0],episode,reset,transitions,checks)
            previous_state=None; state_count=transition_count=0; ended=False; final_reason=None
            for chunk in episode["chunks"]:
                chunk_path=inside(output/"replay",chunk["path"],checks)
                checks.require(sha256(chunk_path)==chunk["sha256"],"NPZ chunk changed")
                with np.load(chunk_path,allow_pickle=False) as source:
                    arrays={name:source[name] for name in source.files}
                checks.require(chunk["state_start"]==state_count and chunk["transition_start"]==transition_count,
                               "observation/transition chunk sequence has a gap")
                checks.require(arrays["rgb"].dtype==np.uint8
                               and arrays["rgb"].shape==(chunk["state_count"],2,3,96,128),
                               "real RGB storage shape/dtype changed")
                checks.require(arrays["image_valid"].dtype==np.bool_
                               and arrays["image_valid"].shape==(chunk["state_count"],2),
                               "image masks do not identify both player slots")
                checks.require(not np.any(arrays["rgb"][~arrays["image_valid"]]),
                               "missing-image storage slots contain fabricated usable pixels")
                states={}
                for local in range(chunk["state_count"]):
                    states[state_count+local]={
                        "tic":int(arrays["state_engine_tic"][local]),
                        "valid":arrays["image_valid"][local],
                        "events":arrays["cumulative_event_counters"][local],
                    }
                for local in range(chunk["transition_count"]):
                    index=transition_count+local
                    row=transitions[index]
                    group=native["groups"][native["initialization"]+index+1]
                    current=states.get(index,previous_state)
                    following=states.get(index+1)
                    checks.require(current is not None and following is not None,"source/next state absent across chunk")
                    checks.require(not ended,"real transition follows terminal before reset")
                    checks.require(int(arrays["decision_index"][local])==index==row["decision_index"],
                                   "stored and charged decision indices disagree")
                    checks.require(int(arrays["transition_id"][local])==next_transition+index,
                                   "stored real transition IDs disagree")
                    checks.require(str(arrays["episode_id"][local])==episode_id==row["episode_id"],
                                   "stored and charged episode identities disagree")
                    checks.require(int(arrays["observation_index"][local])==index
                                   and int(arrays["next_observation_index"][local])==index+1,
                                   "observation indices do not encode t->t+1")
                    checks.require(arrays["actions"][local].tolist()==group[0][0]["actions"],
                                   "replay joint action differs from actually dispatched actions")
                    checks.require(row["physical_consumed_at_source"]==native["initialization"]+index+1,
                                   "training consumed counter differs from real native decisions")
                    checks.require(row["coordinator_run_dir"]==reset["coordinator_run_dir"],
                                   "ledger transition switched coordinator")
                    checks.require(row["start_tic"]==int(arrays["start_tic"][local])==current["tic"]
                                   ==group[0][1]["tics"][0]-1,"source-state tic/ledger/native action effect misaligned")
                    checks.require(row["end_tic"]==int(arrays["end_tic"][local])==following["tic"]
                                   ==group[-1][1]["tics"][0],"next-state tic/ledger/native action effect misaligned")
                    checks.require(row["actual_tics"]==int(arrays["actual_tics"][local])==len(group),
                                   "actual physical tic duration differs from replay")
                    checks.require(row["collector_kind"]==episode["collector_kind"],"collector attribution changed")
                    checks.require(np.array_equal(arrays["next_obs_valid"][local],following["valid"]),
                                   "next-image reconstruction mask is not state-aligned")
                    checks.require(np.array_equal(following["events"]-current["events"],arrays["event_deltas"][local]),
                                   "event deltas not aligned with source/next states")
                    final_reason=reasons[int(arrays["termination_reason"][local])]
                    terminal=final_reason in ("success","death","failure","deadline")
                    cutoff=final_reason=="collection_cutoff"
                    checks.require(bool(arrays["terminated"][local])==terminal
                                   and bool(arrays["truncated"][local])==cutoff,
                                   "task-terminal and collection-cutoff semantics differ")
                    checks.require(terminal or following["valid"].all(),"nonterminal next image is missing")
                    checks.require(current["valid"].all(),"real action source image is missing")
                    checks.require(len(group)==episode["action_repeat"] or terminal,
                                   "only true terminals may shorten a repeated native action")
                    ended=terminal or cutoff
                    totals["published_tics"]+=len(group)
                    previous_state=following
                event_counts.update(dict(zip(event_fields,arrays["event_deltas"].sum(axis=0).tolist())))
                state_count+=chunk["state_count"]
                transition_count+=chunk["transition_count"]
                del arrays
                if process:
                    peak_rss=max(peak_rss,process.memory_info().rss)
            checks.require(ended and final_reason==episode["end_reason"],"episode end reason differs from stored boundary")
            checks.require(state_count==transition_count+1==episode["observation_count"],"observations duplicated or missing")
            checks.require(transition_count==episode["valid_length"],"episode transition count differs")
            totals.update({key:native[key] for key in (
                "dispatches","acknowledgements","physical_decisions","physical_tics","initialization_tics")})
            cleanup_counts.update(native["cleanup_counts"])
            totals["initialization"]+=native["initialization"]
            totals["published"]+=transition_count
            mixture[episode["collector_kind"]]+=transition_count
            ends[episode["end_reason"]]+=1
            next_transition+=transition_count
        checks.require(ledger.next() is None,"ledger contains unreported extra interactions after final episode")
    finally:
        ledger.file.close()
    checks.require(totals["published"]==report["target_transitions"]==report["published_joint_transitions"]
                   ==accounting["published_real_joint_transitions"]==accounting["replay_committed_joint_transitions"]
                   ==manifest["transition_count"],"published/committed/report replay totals differ")
    checks.require(totals["physical_decisions"]==accounting["physical_joint_decisions_including_initialization"],
                   "physical joint-decision total differs from actual native journals")
    checks.require(totals["initialization"]==accounting["initialization_joint_decisions"],"initialization count differs")
    checks.require(totals["published_tics"]==accounting["published_real_simulator_tics_total"],"published native tics differ")
    checks.require(totals["physical_tics"]==totals["published_tics"]+totals["initialization_tics"],
                   "physical native tics do not reconcile")
    checks.require(accounting["failed_dispatch_attempts"]==ledger.totals["failed_delta"]==0
                   and accounting["real_joint_transitions_total"]==totals["published"],
                   "a completed initial dataset contains unexplained failed decisions or debit aliases")
    checks.require(ledger.totals["physical_delta"]==totals["physical_decisions"]
                   and ledger.totals["initialization_delta"]==totals["initialization"]
                   and ledger.totals["published_delta"]==ledger.totals["committed"]==totals["published"]
                   and ledger.totals["published_tics"]==totals["published_tics"],
                   "native journal and durable training ledger debits differ")
    checks.require(mixture==report["collector_counts"]==accounting["collector_counts"]==ledger.kinds,
                   "scripted/random joint-transition attribution differs")
    checks.require(mixture["scripted"]==round(totals["published"]*.3)
                   and mixture["random"]==totals["published"]-mixture["scripted"],
                   "initial data does not have exact 70/30 transition mixture")
    checks.require(event_counts==report["event_counts"] and ends==report["terminal_counts"],
                   "stored event/outcome coverage differs from report")
    for variant in ("B","D"):
        binding_path=inside(output,report["bindings"][variant]["path"],checks)
        checks.require(sha256(binding_path)==report["bindings"][variant]["sha256"],"branch binding changed")
        binding=json_read(binding_path)
        linked=Path(binding["dataset_path"])
        if not linked.is_absolute():
            linked=binding_path.parent/linked
        checks.require(linked.resolve()==(output/"replay").resolve()
                       and binding["variant"]==variant
                       and binding["manifest_sha256"]==sha256(manifest_path)
                       and binding["dataset_identity_sha256"]==manifest["dataset_identity_sha256"]
                       and binding["attributed_initial_joint_transitions"]==totals["published"],
                       "B/D branch binding does not share exactly one immutable initial store")
    return {
        "status":"passed","dataset_path":str(output),"task_family":report["task_family"],
        "run_seed":report["run_seed"],"budget_scope":report["budget_scope"],
        "learning_eligible":report["learning_eligible"],"episodes":len(episodes),
        **dict(totals),"collector_counts":dict(mixture),"end_reasons":dict(ends),
        "nonzero_events":{key:value for key,value in event_counts.items() if value},
        "cleanup_status_counts":dict(cleanup_counts),"archives_verified":len(archives),
        "archive_member_files_verified":archive_files,"archive_member_bytes":archive_bytes,
        "hot_closed_runtime_remaining":False,"hot_ledger_present_and_hash_verified":hot_exists,
        "peak_hot_bytes":storage["peak_hot_bytes"],"maximum_hot_bytes":storage["maximum_hot_bytes"],
        "bounded_peak_audit_rss_bytes":peak_rss,"native_and_replay_checks":checks.count-checks_start,
        "dataset_manifest_sha256":sha256(manifest_path),"collection_report_sha256":sha256(output/"collection_report.json"),
        "training_ledger_sha256":sha256(ledger_path),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",nargs="+",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-cleaned-hot-ledgers",action="store_true")
    args=parser.parse_args()
    output=args.output.resolve()
    datasets=[path.resolve() for path in args.datasets]
    if any(output.is_relative_to(path) for path in datasets):
        raise ValueError("audit output must remain outside every audited immutable dataset")
    import psutil
    process=psutil.Process()
    original=process.cpu_affinity()
    requested=[0,1,2,3]
    if not set(requested).issubset(original):
        raise RuntimeError("CPU affinity 0..3 is unavailable; do not silently exceed or replace the audit CPU budget")
    process.cpu_affinity(requested)
    checks=Checks(); start=time.perf_counter(); cpu_start=process.cpu_times()
    report={
        "schema_version":VERSION,"status":"running",
        "recorded_at_utc":datetime.now(timezone.utc).isoformat(),
        "helper_path":str(Path(__file__).resolve()),"helper_sha256":sha256(__file__),
        "cpu_ids":requested,"thread_environment":{name:os.environ[name] for name in (
            "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS")},
        "scope":"Only explicitly supplied completed datasets; separate abandoned/development costs must be reported elsewhere.",
        "independence":"No project modules imported; native journal, NPZ, archive, and ledger values independently compared.",
        "cases":[],
    }
    try:
        for dataset in datasets:
            case=audit_dataset(dataset,checks,allow_cleaned_hot=args.allow_cleaned_hot_ledgers,process=process)
            report["cases"].append(case)
        report["status"]="passed"
        report["totals"]={key:sum(case.get(key,0) for case in report["cases"]) for key in (
            "published","episodes","physical_decisions","initialization","published_tics","initialization_tics",
            "physical_tics","dispatches","acknowledgements","archives_verified","archive_member_files_verified")}
    except BaseException as error:
        report.update(status="failed",error=repr(error),traceback=traceback.format_exc())
    finally:
        cpu_end=process.cpu_times(); elapsed=time.perf_counter()-start
        report.update(assertions=checks.count,wall_seconds=elapsed,
                      process_cpu_seconds=(cpu_end.user+cpu_end.system-cpu_start.user-cpu_start.system),
                      cpu_percent_of_four_core_budget=100*(cpu_end.user+cpu_end.system-cpu_start.user-cpu_start.system)/(elapsed*4))
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("w",encoding="utf-8",newline="\n") as stream:
            json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        process.cpu_affinity(original)
    print(json.dumps({"status":report["status"],"cases":len(report["cases"]),
                      "assertions":checks.count,"output":str(output),"totals":report.get("totals"),
                      "error":report.get("error")}),flush=True)
    return 0 if report["status"]=="passed" else 1


if __name__=="__main__":
    raise SystemExit(main())
