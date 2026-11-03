"""Tiny synthetic evidence-contract fixtures; no simulator, image payloads, or D: writes."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import finalize_phase21 as finalizer


def write(path, value, receipt=True):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    if receipt:path.with_suffix(".sha256").write_text(finalizer.sha(path)+"  "+path.name+"\n",encoding="ascii")


def source_index(root, location, entries, *, original=False):
    rows=[]
    for name,content in entries.items():
        snapshot=location.parent/"snapshots"/name;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
        rows.append({"project_relative_path":name,"snapshot_relative_path":snapshot.relative_to(location.parent).as_posix(),"sha256":finalizer.sha(snapshot),"bytes":len(content)})
    hashes={row["project_relative_path"]:row["sha256"] for row in rows}
    data={"status":"verified_original_execution_source_snapshot" if original else "frozen","files":rows}
    if original:data["original_execution_source_sha256"]=hashes
    write(location,data)
    return hashes


def synthetic_case(root, directory, family, seed, metadata, hot, *, full=True, recovered=False, prefix_source=None):
    directory.mkdir(parents=True,exist_ok=True)
    ledger=directory/"training_interactions.jsonl"
    ledger.write_bytes((b"old verified ledger prefix\n" if recovered else b"initial ledger\n")+b"completed ledger tail\n")
    hot_ledger=hot/"ledger/training_interactions.jsonl";hot_ledger.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ledger,hot_ledger)
    count=50000 if full else 1000
    manifest={"status":"sealed","metadata":metadata,"transition_count":count,"episodes":[],
              "storage":{"compression":"deflate","compression_level":1,"chunk_transitions":128}}
    manifest["dataset_identity_sha256"]=finalizer.canonical(manifest)
    write(directory/"replay/manifest.json",manifest)
    write(directory/"collection_spec.json",{"origin_provenance":metadata.get("origin_provenance",{})})
    journal={"spooled":True,"spool_root_at_collection":str(hot),"ledger_hot_path_at_collection":str(hot_ledger),
             "ledger_archive":{"source_path_at_collection":str(hot_ledger),"sha256":finalizer.sha(ledger),"bytes":ledger.stat().st_size},"runtime_archives":[]}
    if recovered:
        oldhot=prefix_source
        journal["retained_prior_hot_ledger_copies"]=[{"source_path_at_collection":str(oldhot),"matching_archive_path":"recovery/continue/before/training_interactions.jsonl","sha256":finalizer.sha(oldhot),"bytes":oldhot.stat().st_size}]
    report={"status":"complete_initial_replay","task_family":family,"run_seed":seed,"target_transitions":count,"published_joint_transitions":count,
            "collector_counts":{"random":35000 if full else 700,"scripted":15000 if full else 300},
            "dataset_manifest_sha256":finalizer.sha(directory/"replay/manifest.json"),"dataset_identity_sha256":manifest["dataset_identity_sha256"],
            "collection_spec_sha256":finalizer.sha(directory/"collection_spec.json"),"bindings":{},"journal_storage":journal,
            "interaction_accounting":{"ledger_sha256":finalizer.sha(ledger),"published_real_joint_transitions":count,"replay_committed_joint_transitions":count,
                "real_joint_transitions_total":count,"failed_dispatch_attempts":0,"physical_joint_decisions_including_initialization":count+(2 if full else 6),
                "initialization_joint_decisions":2 if full else 6,"collector_counts":{"random":35000 if full else 700,"scripted":15000 if full else 300}}}
    for variant in ("B","D"):
        path=directory/"bindings"/(variant+".json")
        write(path,{"variant":variant,"dataset_path":"../replay","manifest_sha256":report["dataset_manifest_sha256"],
                    "dataset_identity_sha256":manifest["dataset_identity_sha256"],"attributed_initial_joint_transitions":count,"new_physical_transitions":0})
        report["bindings"][variant]={"path":"bindings/"+variant+".json","sha256":finalizer.sha(path)}
    return report,manifest


def fixture(directory):
    root=Path(directory)/"project";root.mkdir()
    mainhot=Path(directory)/"world-models-p21-fixture-main";recoveryhot=Path(directory)/"world-models-p21-fixture-recovery"
    sources=source_index(root,root/"reviews/original/index.json",{
        "src/"+name.replace(".","/")+".py":("frozen "+name).encode() for name in finalizer.GENERATION_MODULES},original=True)
    overlay=source_index(root,root/"reviews/overlay/index.json",{"src/world_models/data/"+name+".py":name.encode() for name in ("ledger","replay","recovery")})
    driver_source=root/"reviews/recovery_driver.py";driver_source.write_bytes(b"frozen driver")
    raw={"status":"failed","command":"complete","profile":"pilot","cases":[list(case) for case in sorted(finalizer.CASES)],
         "target_transitions_per_dataset":50000,"model_training_started":False,"final_seed_collection_started":False,
         "source_sha256_at_start":sources,"journal_storage":{"temporary_campaign_root":str(mainhot)}}
    inputs={"project_root":root,"data_dir":root/"data/initial_replay/pilot","raw_report":root/"reports/raw_failed.json",
            "production_report":root/"reports/verify.json","arrays_report":root/"reports/arrays.json","native_report":root/"reports/native.json",
            "unit_report":root/"reports/unit.json","storage_report":root/"reports/storage.json","preflight_report":root/"reports/pref/native_preflight_report.json",
            "prior_diagnostic_report":root/"reports/aborted.json","original_source_index":root/"reviews/original/index.json",
            "recovery_report":root/"reports/recovery.json","recovery_source_index":root/"reviews/overlay/index.json","recovery_driver_source":driver_source}
    write(inputs["raw_report"],raw)
    production=[];arrays=[];native=[];preflight=[];preflight_datasets=[]
    for family,seed in sorted(finalizer.CASES):
        recovered=(family,seed)==("relay",1)
        base=inputs["data_dir"]/family/f"seed_{seed}"
        metadata={"task_family":family,"run_seed":seed,"learning_eligible":True,"budget_scope":"approved_initial","data_source":"real_environment",
                  "origin_provenance":{"execution_source_sha256":sources}}
        prefix_hot=mainhot/"initial"/family/f"seed_{seed}"/"ledger/training_interactions.jsonl"
        if recovered:
            prefix_hot.parent.mkdir(parents=True,exist_ok=True);prefix_hot.write_bytes(b"old verified ledger prefix\n")
        report,manifest=synthetic_case(root,base,family,seed,metadata,recoveryhot if recovered else prefix_hot.parent.parent,
                                       recovered=recovered,prefix_source=prefix_hot)
        if recovered:
            before=base/"recovery/continue/before"
            failure={"status":"failed","operation":"reset","unreconciled":False,"interaction_accounting":{"failed_dispatch_attempts":0}}
            write(base/"collection_error.json",failure)
            originals={"collection_error.json":base/"collection_error.json","collection_spec.json":base/"collection_spec.json"}
            before.mkdir(parents=True)
            write(before/"manifest.partial.json",{"active_episode":None,"transition_count":32700})
            write(before/"collection_progress.json",{"published_joint_transitions":32700})
            shutil.copyfile(prefix_hot,before/"training_interactions.jsonl")
            for name,path in originals.items():shutil.copyfile(path,before/name)
            snapshots={p.name:{"path":p.relative_to(base).as_posix(),"sha256":finalizer.sha(p),"bytes":p.stat().st_size} for p in before.iterdir() if p.is_file() and p.suffix!=".sha256"}
            payload=base/"replay/preserved-prefix.bin";payload.write_bytes(b"unchanged committed payload")
            lineage={"recovery_id":"continue","original_failure_sha256":finalizer.sha(base/"collection_error.json"),"original_failure_report_path":"collection_error.json",
                "prefix_joint_transitions":32700,"prefix_episodes":226,"new_random_or_scripted_interactions_discarded":0,"step_failures_retried":False,
                "original_metadata_sha256":finalizer.canonical(metadata),"snapshots":snapshots,
                "original_generation_proof":{"original_execution_sources_verified":len(sources),"actual_loaded_generation_modules_verified":14,"actual_authored_generation_inputs_verified":11},
                "original_prefix_files":[{"path":payload.relative_to(base).as_posix(),"sha256":finalizer.sha(payload)}]}
            lineage_path=base/"recovery/continue/lineage.json";write(lineage_path,lineage)
            report["recovery_lineage"]={"path":lineage_path.relative_to(base).as_posix(),"sha256":finalizer.sha(lineage_path)}
            write(base/"collection_report.json",report)
            write(lineage_path.parent/"continuation_result.json",{"status":"passed","prefix_joint_transitions":32700,"added_joint_transitions":17300,"final_joint_transitions":50000,
                "original_metadata_unchanged":True,"original_prefix_files_unchanged":1,"collection_report_sha256":finalizer.sha(base/"collection_report.json")})
            write(inputs["recovery_report"],{"status":"complete_initial_replay","collection_status":"complete_initial_replay","inspect_only":False,"published_joint_transitions":50000,
                "collection_report_sha256":finalizer.sha(base/"collection_report.json"),"original_execution_sources_unchanged":True,"overlay_sources_unchanged":True,"driver_source_unchanged":True,
                "recovery_id":"continue","requested_failure_sha256":lineage["original_failure_sha256"],"resource_measurements":{"logical_cpu_limit":4},
                "implementation_provenance":{"original_execution_source_sha256":sources,"driver_sha256":finalizer.sha(driver_source),
                    "original_generation_modules":{name:{"sha256":sources["src/"+name.replace(".","/")+".py"]} for name in finalizer.GENERATION_MODULES},
                    "loaded_module_overlays":{"world_models.data."+name:{"sha256":overlay["src/world_models/data/"+name+".py"]} for name in ("ledger","replay","recovery")}}})
        else:write(base/"collection_report.json",report)
        common={"status":"passed","task_family":family,"run_seed":seed}
        production.append({**common,"transition_count":50000,"manifest_sha256":report["dataset_manifest_sha256"]})
        hashed={"dataset_manifest_sha256":report["dataset_manifest_sha256"],"collection_report_sha256":finalizer.sha(base/"collection_report.json"),"training_ledger_sha256":finalizer.sha(base/"training_interactions.jsonl")}
        arrays.append({**common,**hashed,"transition_count":50000})
        native.append({**common,**hashed,"published":50000,"physical_decisions":50002,"collector_counts":{"random":35000,"scripted":15000}})
        prefbase=inputs["preflight_report"].parent/"preflight_data"/family/f"seed_{seed}"
        pref,_=synthetic_case(root,prefbase,family,seed,{"task_family":family,"run_seed":seed},mainhot/"preflight"/family/f"seed_{seed}",full=False)
        write(prefbase/"collection_report.json",pref)
        preflight.append({**common,"transition_count":1000});preflight_datasets.append({"path":str(prefbase),"report":pref})
    write(inputs["production_report"],{"status":"passed","command":"verify","stages":{"dataset_audits":production}})
    write(inputs["arrays_report"],{"status":"passed","cpu_ids":[0,1,2,3],"total_transitions_checked":200000,"datasets":arrays})
    write(inputs["native_report"],{"status":"passed","cpu_ids":[0,1,2,3],"totals":{"published":200000,"physical_decisions":200008},"cases":native})
    write(inputs["unit_report"],{"status":"passed","returncode":0})
    write(inputs["storage_report"],{"status":"passed","all_payloads_exact":True,"source":{"source_kind":"sealed_real_replay"},"cpu_budget":{"logical_core_budget":4},"trials":[{"payload_roundtrip_exact":True}],"selected":{"compression":"deflate","compression_level":1,"chunk_transitions":128}})
    write(inputs["preflight_report"],{"audits":preflight,"datasets":preflight_datasets,"published_diagnostic_joint_transitions":4000,"enters_initial_training_replay":False,"resources":{"logical_cpu_limit":4}})
    write(inputs["prior_diagnostic_report"],{"status":"aborted_diagnostic_preflight_before_full_collection","all_costs_retained":True,
        "cases":[{"enters_learning_replay":False,"physical_joint_decisions_from_native_journals":611,"published_validation_transitions":608,"unpublished_or_pending_initialization_decisions":1} for _ in range(2)]})
    return inputs


class FinalizerArtifactTests(unittest.TestCase):
    def test_four_cases_finalize_without_relabeling_raw_failure_and_no_prefix_double_count(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs=fixture(directory);raw=inputs["raw_report"].read_bytes()
            report,plan=finalizer.finalize(inputs)
            self.assertEqual(report["status"],"complete_p21_pilot")
            self.assertEqual(report["completion_method"],"verified_explicit_continuation")
            self.assertEqual(inputs["raw_report"].read_bytes(),raw)
            self.assertEqual(report["original_pipeline_status"],"failed")
            accounting=report["interaction_accounting"]
            self.assertEqual(accounting["physical_initial_replay_joint_transitions"],200000)
            self.assertEqual(accounting["attributed_initial_joint_transitions_per_variant"],{"B":200000,"D":200000})
            self.assertEqual(accounting["all_collection_and_diagnostic_physical_decisions"],200008+4024+1222)
            self.assertEqual(len(plan["roots"]),2)
            self.assertTrue(all(Path(row["absolute_collection_host_root"]).exists() for row in plan["roots"]))
            self.assertFalse(plan["deletion_performed_by_this_helper"])

    def test_missing_case_stale_audit_wrong_mixture_and_diagnostic_costs_fail_closed(self):
        for damage in ("missing_case","stale_audit","wrong_mixture","diagnostic_cost"):
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as directory:
                inputs=fixture(directory)
                if damage=="missing_case":
                    path=inputs["arrays_report"];value=finalizer.read(path);value["datasets"].pop();write(path,value)
                elif damage=="stale_audit":
                    path=inputs["native_report"];value=finalizer.read(path);value["cases"][0]["dataset_manifest_sha256"]="stale";write(path,value)
                elif damage=="wrong_mixture":
                    path=inputs["data_dir"]/"cover/seed_0/collection_report.json";value=finalizer.read(path);value["collector_counts"]={"random":35001,"scripted":14999};write(path,value)
                else:
                    path=inputs["prior_diagnostic_report"];value=finalizer.read(path);value["cases"][0]["physical_joint_decisions_from_native_journals"]+=1;write(path,value)
                with self.assertRaises(ValueError):finalizer.finalize(inputs)

    def test_recovery_source_or_original_payload_tamper_is_rejected(self):
        for damage in ("source","payload","driver","generation_record"):
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as directory:
                inputs=fixture(directory)
                if damage=="generation_record":
                    value=finalizer.read(inputs["recovery_report"])
                    value["implementation_provenance"]["original_generation_modules"]["world_models.env.workers"]["sha256"]="changed"
                    write(inputs["recovery_report"],value)
                    with self.assertRaises(ValueError):finalizer.finalize(inputs)
                    continue
                if damage=="source":path=inputs["original_source_index"].parent/"snapshots/src/world_models/data/collection.py"
                elif damage=="payload":path=inputs["data_dir"]/"relay/seed_1/replay/preserved-prefix.bin"
                else:path=inputs["recovery_driver_source"]
                path.write_bytes(b"tamper")
                with self.assertRaises(ValueError):finalizer.finalize(inputs)

    def test_hot_cleanup_is_blocked_by_unknown_or_different_bytes(self):
        for damage in ("unknown","changed"):
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as directory:
                inputs=fixture(directory);main=Path(directory)/"world-models-p21-fixture-main"
                if damage=="unknown":(main/"unarchived.txt").write_bytes(b"preserve")
                else:next(main.rglob("training_interactions.jsonl")).write_bytes(b"changed")
                with self.assertRaises(ValueError):finalizer.finalize(inputs)
                self.assertTrue(main.exists())

    def test_cleanup_verifier_requires_absence_and_rechecks_archived_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs=fixture(directory);report,plan=finalizer.finalize(inputs)
            path=inputs["project_root"]/"reports/cleanup_plan.json";finalizer.write_new(path,plan)
            with self.assertRaisesRegex(ValueError,"still exists"):
                finalizer.verify_cleanup(inputs["project_root"],path)
            # Only fixture-owned temporary roots are removed here, emulating an
            # operator after the plan verified every retained ledger copy.
            for entry in plan["roots"]:
                target=Path(entry["absolute_collection_host_root"]).resolve()
                self.assertTrue(target.is_relative_to(Path(directory).resolve()))
                shutil.rmtree(target)
            verified=finalizer.verify_cleanup(inputs["project_root"],path)
            self.assertEqual(verified["status"],"passed")
            prefix=inputs["data_dir"]/"relay/seed_1/recovery/continue/before/training_interactions.jsonl"
            prefix.write_bytes(b"changed cold snapshot")
            with self.assertRaisesRegex(ValueError,"cold copy changed"):
                finalizer.verify_cleanup(inputs["project_root"],path)


if __name__=="__main__":unittest.main()
