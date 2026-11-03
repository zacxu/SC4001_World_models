"""Finalize P2.1 from immutable measured evidence, preserving the failed raw run.

No simulator starts and no data/source files change. This helper only writes a
new acceptance report and a separately verifiable hot-ledger cleanup plan.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import tempfile


CASES = {(family, seed) for family in ("relay", "cover") for seed in (0, 1)}
GENERATION_MODULES = (
    "world_models.data.collection", "world_models.data.schema", "world_models.env.coop",
    "world_models.env.coordinator", "world_models.env.workers", "world_models.env.scenario_builder",
    "world_models.env.scripted", "world_models.env.splits", "world_models.env.targets", "world_models.env.actions",
    "world_models.env.observations", "world_models.contracts", "world_models.infrastructure.config",
    "world_models.infrastructure.reproducibility",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def member(root, encoded):
    require(isinstance(encoded, str) and encoded.strip() and "\0" not in encoded, "empty/invalid evidence path")
    relative = PurePosixPath(encoded.replace("\\", "/"))
    require(not relative.is_absolute() and not PureWindowsPath(encoded).drive and not PureWindowsPath(encoded).root and ".." not in relative.parts,
            "evidence member must be a safe portable relative path")
    target = (Path(root) / Path(*relative.parts)).resolve()
    require(target.is_relative_to(Path(root).resolve()), "evidence member escaped its intended directory")
    return target


def record(root, path):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and path.is_relative_to(root), "promote evidence under the project before finalization")
    return {"path": path.relative_to(root).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}


def document(root, path, *, receipt=False):
    artifact = record(root, path)
    if receipt:
        receipt_path = Path(path).with_suffix(".sha256")
        require(receipt_path.is_file() and receipt_path.read_text(encoding="ascii").split()[0] == artifact["sha256"],
                "report receipt is absent or disagrees: " + str(path))
    return read(path), artifact


def case_index(rows, *, label):
    result = {}
    for row in rows:
        case = row["task_family"], row["run_seed"]
        require(case in CASES and case not in result, label + " has an unexpected or repeated case")
        require(row.get("status") == "passed", label + " did not pass")
        result[case] = row
    require(set(result) == CASES, label + " must cover all four pilot cases")
    return result


def verify_source_index(root, path, expected, *, allow_extra=False):
    index, artifact = document(root, path)
    require(index.get("status") in ("frozen", "verified_original_execution_source_snapshot", "passed"), "source snapshot index is not verified")
    if "original_execution_source_sha256" in index:
        require(index["original_execution_source_sha256"] == expected, "source index pins different original code")
    seen = {}
    for entry in index["files"]:
        relative = entry.get("project_relative_path", entry.get("path"))
        require((relative in expected or allow_extra) and relative not in seen, "source snapshot has duplicate or unexpected files")
        snapshot = entry.get("snapshot_relative_path", entry.get("snapshot"))
        target = member(Path(path).parent, snapshot)
        require(target.is_relative_to(root) and sha(target) == entry["sha256"], "source snapshot bytes differ from measured execution")
        if relative in expected: require(entry["sha256"] == expected[relative], "pinned execution source snapshot changed")
        if "bytes" in entry: require(target.stat().st_size == entry["bytes"], "source snapshot byte count changed")
        seen[relative] = entry["sha256"]
    require({relative:seen[relative] for relative in expected if relative in seen} == expected, "source snapshot must preserve every pinned execution file")
    return {**artifact, "verified_files": len(seen), "matched_execution_files":len(expected)}


def check_dataset(root, directory, case):
    report, report_file = document(root, directory / "collection_report.json")
    manifest, manifest_file = document(root, directory / "replay/manifest.json", receipt=True)
    metadata = manifest["metadata"]
    require(report["status"] == "complete_initial_replay" and manifest["status"] == "sealed", "dataset is not complete/sealed")
    require((report["task_family"], report["run_seed"]) == case == (metadata["task_family"], metadata["run_seed"]), "dataset case identity mismatch")
    require(manifest["transition_count"] == report["target_transitions"] == report["published_joint_transitions"] == 50000,
            "each pilot dataset must contain exactly 50,000 transitions")
    require(report["collector_counts"] == {"random": 35000, "scripted": 15000}, "dataset mixture is not exactly 70/30")
    require(metadata["learning_eligible"] is True and metadata["budget_scope"] == "approved_initial" and metadata["data_source"] == "real_environment",
            "only approved real initial replay may satisfy acceptance")
    require(report["dataset_manifest_sha256"] == manifest_file["sha256"], "dataset manifest no longer matches collection")
    require(report["dataset_identity_sha256"] == manifest["dataset_identity_sha256"], "reported shared initialization identity changed")
    require(manifest["dataset_identity_sha256"] == canonical({key: value for key, value in manifest.items() if key not in ("sealed_at_utc", "dataset_identity_sha256")}), "dataset logical identity changed")
    specification, specification_file = document(root, directory / "collection_spec.json")
    require(specification_file["sha256"] == report["collection_spec_sha256"] and specification["origin_provenance"] == metadata["origin_provenance"], "initial collection procedure changed")
    accounting = report["interaction_accounting"]
    ledger = record(root, directory / "training_interactions.jsonl")
    require(ledger["sha256"] == accounting["ledger_sha256"], "final real-interaction ledger changed")
    require(accounting["published_real_joint_transitions"] == accounting["replay_committed_joint_transitions"] == accounting["real_joint_transitions_total"] == 50000,
            "replay and charged real interactions do not reconcile")
    require(accounting["failed_dispatch_attempts"] == 0 and accounting["physical_joint_decisions_including_initialization"] == 50000 + accounting["initialization_joint_decisions"],
            "unaccounted failed physics or initialization decisions")
    require(accounting["collector_counts"] == report["collector_counts"], "ledger/collector mixture changed")
    for variant in ("B", "D"):
        pointer_path = member(directory, report["bindings"][variant]["path"])
        pointer, pointer_file = document(root, pointer_path)
        require(pointer_file["sha256"] == report["bindings"][variant]["sha256"], "B/D pointer bytes changed")
        encoded = pointer["dataset_path"]
        require(isinstance(encoded, str) and encoded.strip(), "empty dataset pointer")
        relative = Path(encoded.replace("\\", "/"))
        require(not relative.is_absolute() and not PureWindowsPath(encoded).drive and (pointer_path.parent / relative).resolve() == (directory / "replay").resolve(), "B/D must use portable pointers to one physical store")
        require(pointer["variant"] == variant and pointer["manifest_sha256"] == manifest_file["sha256"]
                and pointer["dataset_identity_sha256"] == manifest["dataset_identity_sha256"]
                and pointer["attributed_initial_joint_transitions"] == 50000 and pointer["new_physical_transitions"] == 0,
                "B/D immutable data attribution changed")
    return {"case": case, "directory": directory, "report": report, "manifest": manifest,
            "report_file": report_file, "manifest_file": manifest_file, "ledger_file": ledger}


def verify_recovery(root, item, recovery_report, original_sources, overlay_index, driver_source):
    directory, collection = item["directory"], item["report"]
    driver, driver_file = document(root, recovery_report)
    require(driver["status"] == driver.get("collection_status") == "complete_initial_replay" and driver.get("inspect_only") is False,
            "recovery driver must have completed actual continuation")
    require(driver["published_joint_transitions"] == 50000 and driver["collection_report_sha256"] == item["report_file"]["sha256"], "recovery report is stale or incomplete")
    require(all(driver.get(name) is True for name in ("original_execution_sources_unchanged", "overlay_sources_unchanged", "driver_source_unchanged")), "recovery sources changed during execution")
    provenance = driver["implementation_provenance"]
    require(provenance["original_execution_source_sha256"] == original_sources, "recovery changed original generation-source identity")
    generation = provenance["original_generation_modules"]
    require(set(generation) == set(GENERATION_MODULES), "recovery loaded unexpected generation modules")
    for name, entry in generation.items():
        relative = "src/" + name.replace(".", "/") + ".py"
        require(entry["sha256"] == original_sources[relative], "a loaded recovery generation module differs from original execution")
    overlays = provenance["loaded_module_overlays"]
    require(set(overlays) == {"world_models.data." + name for name in ("ledger", "replay", "recovery")}, "unexpected recovery module overlay")
    expected_overlay = {"src/world_models/data/" + name.rsplit(".", 1)[-1] + ".py": entry["sha256"] for name, entry in overlays.items()}
    overlay_evidence = verify_source_index(root, overlay_index, expected_overlay, allow_extra=True)
    driver_artifact = record(root, driver_source)
    require(driver_artifact["sha256"] == provenance["driver_sha256"], "recovery driver source snapshot changed")
    require(driver["resource_measurements"]["logical_cpu_limit"] == 4, "recovery did not enforce four logical CPUs")
    ref = collection["recovery_lineage"]
    lineage_path = member(directory, ref["path"])
    lineage, lineage_file = document(root, lineage_path)
    require(lineage_file["sha256"] == ref["sha256"] and lineage["recovery_id"] == driver["recovery_id"], "recovery lineage identity changed")
    require(lineage["original_failure_sha256"] == driver["requested_failure_sha256"], "recovery did not name the original failure")
    require(sha(member(directory,lineage["original_failure_report_path"])) == driver["requested_failure_sha256"], "original failed collection report was not preserved")
    require(lineage["prefix_joint_transitions"] == 32700 and lineage["prefix_episodes"] == 226
            and lineage["new_random_or_scripted_interactions_discarded"] == 0 and lineage["step_failures_retried"] is False,
            "recovery discarded interactions or changed the approved prefix")
    require(lineage["original_metadata_sha256"] == canonical(item["manifest"]["metadata"]), "recovery changed original dataset metadata")
    proof = lineage["original_generation_proof"]
    require(proof["original_execution_sources_verified"] == len(original_sources)
            and proof.get("actual_loaded_generation_modules_verified", 0) >= 14
            and proof.get("actual_authored_generation_inputs_verified", 0) >= 11, "original loaded generation/input proof is incomplete")
    snapshots = {}
    for name, snapshot in lineage["snapshots"].items():
        path = member(directory, snapshot["path"])
        current = record(root, path)
        require(current["sha256"] == snapshot["sha256"] and current["bytes"] == snapshot["bytes"], "original failure snapshot changed")
        snapshots[name] = path
    require(set(snapshots) >= {"collection_error.json", "manifest.partial.json", "training_interactions.jsonl", "collection_spec.json", "collection_progress.json"}, "recovery omitted an original evidence snapshot")
    failure = read(snapshots["collection_error.json"])
    require(sha(snapshots["collection_error.json"]) == driver["requested_failure_sha256"] and failure["operation"] == "reset"
            and failure["unreconciled"] is False and failure["interaction_accounting"]["failed_dispatch_attempts"] == 0,
            "recovery was not a verified startup-only failure")
    original_partial = read(snapshots["manifest.partial.json"])
    require(original_partial["active_episode"] is None and original_partial["transition_count"] == 32700, "recovery prefix was active/uncommitted")
    with snapshots["training_interactions.jsonl"].open("rb") as prefix, (directory / "training_interactions.jsonl").open("rb") as final:
        for block in iter(lambda: prefix.read(1024*1024), b""):
            require(final.read(len(block)) == block, "final ledger does not preserve the exact original byte prefix")
    for entry in lineage["original_prefix_files"]:
        require(sha(member(directory, entry["path"])) == entry["sha256"], "a committed original payload changed during recovery")
    result, result_file = document(root, lineage_path.parent / "continuation_result.json")
    require(result["status"] == "passed" and result["prefix_joint_transitions"] == 32700 and result["added_joint_transitions"] == 17300
            and result["final_joint_transitions"] == 50000 and result["original_metadata_unchanged"] is True
            and result["original_prefix_files_unchanged"] == len(lineage["original_prefix_files"])
            and result["collection_report_sha256"] == item["report_file"]["sha256"], "continuation result is incomplete")
    return {"driver_report": driver_file, "driver_source": driver_artifact, "overlay_source_index": overlay_evidence,
            "lineage": lineage_file, "continuation_result": result_file,
            "original_failure_preserved": True, "prefix_joint_transitions_preserved": 32700, "added_joint_transitions": 17300,
            "original_payload_files_preserved": len(lineage["original_prefix_files"]),
            "no_physics_failures_retried": True, "original_dataset_metadata_unchanged": True,
            "resource_measurements": driver["resource_measurements"]}


def cleanup_plan(root, raw, datasets, preflight, preflight_path):
    """Map every retained hot ledger to its exact verified D: copy; never delete."""
    sources = {}
    roots = [Path(raw["journal_storage"]["temporary_campaign_root"]).resolve()]
    reports = [(item["directory"], item["report"]) for item in datasets]
    for item in preflight["datasets"]:
        directory = Path(item["path"])
        # Original reports may have an old absolute project location. Derive
        # preflight case location from this report's portable relationship.
        family, seed = item["report"]["task_family"], item["report"]["run_seed"]
        directory = Path(preflight_path).parent / "preflight_data" / family / f"seed_{seed}"
        reports.append((directory, read(directory / "collection_report.json")))
    for directory, report in reports:
        journal = report["journal_storage"]
        if not journal["spooled"]: continue
        roots.append(Path(journal["spool_root_at_collection"]).resolve())
        entries = [{"source_path_at_collection": journal["ledger_hot_path_at_collection"],
                    "matching_archive_path": "training_interactions.jsonl", **journal["ledger_archive"]}]
        entries += journal.get("retained_prior_hot_ledger_copies", [])
        for entry in entries:
            source = Path(entry["source_path_at_collection"]).resolve()
            target = member(directory, entry["matching_archive_path"])
            cold = record(root, target)
            require(cold["sha256"] == entry["sha256"] and cold["bytes"] == entry["bytes"], "hot cleanup's cold evidence copy does not match")
            if source in sources:
                require(sources[source]["sha256"] == cold["sha256"], "one hot file has conflicting cold copies")
            sources[source] = {"hot_source_path_on_collection_host": str(source), "cold_copy": cold,
                               "sha256": cold["sha256"], "bytes": cold["bytes"]}
    temp = Path(tempfile.gettempdir()).resolve()
    unique = sorted(set(roots), key=lambda path: len(path.parts))
    selected = []
    for candidate in unique:
        if any(candidate.is_relative_to(parent) for parent in selected): continue
        require(candidate != temp and candidate.is_relative_to(temp) and candidate.name.startswith("world-models-p21-"), "cleanup root is not an explicitly named P2.1 child of the temporary directory")
        require(not candidate.is_relative_to(root) and not root.is_relative_to(candidate), "cleanup root overlaps original project/data")
        selected.append(candidate)
    records = []
    for candidate in selected:
        known = {source: entry for source, entry in sources.items() if source.is_relative_to(candidate)}
        require(known, "hot cleanup root has no verified ledger copies")
        actual = set()
        if candidate.exists():
            for path in candidate.rglob("*"):
                require(not path.is_symlink() and not getattr(path, "is_junction", lambda: False)(), "cleanup source contains a link/junction")
                if path.is_file(): actual.add(path.resolve())
            require(actual <= set(known), "hot root contains unarchived/unknown files; retain it for inspection")
        for source, entry in known.items():
            if source.exists():
                require(source.stat().st_size == entry["bytes"] and sha(source) == entry["sha256"], "hot ledger differs from its exact durable cold copy")
            entry["hot_source_present_when_planned"] = source.exists()
        records.append({"absolute_collection_host_root": str(candidate), "present_when_planned": candidate.exists(),
                        "verified_hot_files": list(known.values()), "unknown_files": [],
                        "safe_after_owned_processes_stopped": True})
    require(all(any(source.is_relative_to(candidate) for candidate in selected) for source in sources), "cleanup left a hot source outside the named roots")
    return {"schema_version": "p21-verified-hot-cleanup-plan-v1", "status": "verified_cold_copies_ready_for_operator_cleanup",
            "collection_host_os_name": os.name, "temporary_parent_on_collection_host": str(temp), "roots": records,
            "deletion_performed_by_this_helper": False,
            "operator_rule": "After confirming all owned collection/recovery processes stopped, verify each exact absolute root is still the same strict P2.1 temporary child and contains only these matching files; remove that named root. Never delete data directories or mutate historical reports.",
            "portable_evidence_rule": "Cold copy paths are relative to project root. Absolute hot paths are cleanup instructions for the original collection host only."}


def finalize(inputs):
    root = Path(inputs["project_root"]).resolve(strict=True)
    data = Path(inputs["data_dir"]).resolve(strict=True)
    require(data.is_relative_to(root), "initial replay must be under the named project for portable evidence links")
    raw, raw_file = document(root, inputs["raw_report"], receipt=True)
    require(raw["status"] == "failed" and raw["command"] == "complete" and raw["profile"] == "pilot", "preserve and identify the failed original complete pipeline")
    require(set(map(tuple, raw["cases"])) == CASES and raw["target_transitions_per_dataset"] == 50000, "original campaign had a different scope")
    require(raw["model_training_started"] is False and raw["final_seed_collection_started"] is False, "scope expanded beyond P2.1 pilot collection")
    original_sources = raw["source_sha256_at_start"]
    original_index = verify_source_index(root, inputs["original_source_index"], original_sources)
    production, production_file = document(root, inputs["production_report"], receipt=True)
    require(production["status"] == "passed" and production["command"] == "verify", "production read-only replay verification did not pass")
    production_cases = case_index(production["stages"]["dataset_audits"], label="production verification")
    arrays, arrays_file = document(root, inputs["arrays_report"])
    native, native_file = document(root, inputs["native_report"])
    require(arrays["status"] == native["status"] == "passed" and arrays["total_transitions_checked"] == native["totals"]["published"] == 200000, "independent audits did not pass all 200,000 transitions")
    array_cases = case_index(arrays["datasets"], label="array audit")
    native_cases = case_index(native["cases"], label="native audit")
    require(arrays["cpu_ids"] == native["cpu_ids"] == [0,1,2,3], "independent audits exceeded the four-core allocation")
    units, units_file = document(root, inputs["unit_report"], receipt=True)
    require(units["status"] == "passed" and units["returncode"] == 0, "unit suite did not pass")
    storage, storage_file = document(root, inputs["storage_report"], receipt=True)
    require(storage["status"] == "passed" and storage["all_payloads_exact"] is True
            and storage["source"]["source_kind"] == "sealed_real_replay"
            and storage["cpu_budget"]["logical_core_budget"] == 4
            and all(row["payload_roundtrip_exact"] for row in storage["trials"]), "production-data storage benchmark did not pass")
    preflight, preflight_file = document(root, inputs["preflight_report"], receipt=True)
    preflight_cases = case_index(preflight["audits"], label="native preflight")
    require(all(row["transition_count"] == 1000 for row in preflight_cases.values())
            and preflight["published_diagnostic_joint_transitions"] == 4000
            and preflight["enters_initial_training_replay"] is False
            and preflight["resources"]["logical_cpu_limit"] == 4, "preflight coverage/scope differs")
    preflight_physical = sum(item["report"]["interaction_accounting"]["physical_joint_decisions_including_initialization"] for item in preflight["datasets"])
    require(preflight_physical == 4024, "passed preflight physical accounting differs from preserved evidence")
    prior, prior_file = document(root, inputs["prior_diagnostic_report"])
    require(prior["status"] == "aborted_diagnostic_preflight_before_full_collection" and prior["all_costs_retained"] is True
            and all(row["enters_learning_replay"] is False for row in prior["cases"]), "aborted diagnostic costs were not retained separately")
    prior_physical = sum(row["physical_joint_decisions_from_native_journals"] for row in prior["cases"])
    require(prior_physical == 1222, "aborted diagnostic accounting changed")
    datasets = []
    for case in sorted(CASES):
        family, seed = case
        item = check_dataset(root, data / family / f"seed_{seed}", case)
        require(all(item["manifest"]["storage"][key] == storage["selected"][key]
                    for key in ("compression", "compression_level", "chunk_transitions")), "dataset did not use the measured storage setting")
        manifest_sha, report_sha = item["manifest_file"]["sha256"], item["report_file"]["sha256"]
        require(item["manifest"]["metadata"]["origin_provenance"]["execution_source_sha256"] == original_sources, "initial source identity differs across paired datasets")
        require(production_cases[case]["transition_count"] == 50000 and production_cases[case]["manifest_sha256"] == manifest_sha, "production verification is stale")
        require(array_cases[case]["transition_count"] == 50000 and array_cases[case]["dataset_manifest_sha256"] == manifest_sha
                and array_cases[case]["collection_report_sha256"] == report_sha and array_cases[case]["training_ledger_sha256"] == item["ledger_file"]["sha256"], "independent array audit is stale")
        require(native_cases[case]["published"] == 50000 and native_cases[case]["dataset_manifest_sha256"] == manifest_sha
                and native_cases[case]["collection_report_sha256"] == report_sha
                and native_cases[case]["training_ledger_sha256"] == item["ledger_file"]["sha256"], "independent native audit is stale")
        require(native_cases[case]["physical_decisions"] == item["report"]["interaction_accounting"]["physical_joint_decisions_including_initialization"], "native/ledger physical work differs")
        require(native_cases[case]["collector_counts"] == {"random":35000,"scripted":15000}, "native executed-action mixture differs")
        datasets.append(item)
    recovered = [item for item in datasets if item["report"].get("recovery_lineage")]
    require(len(recovered) == 1 and recovered[0]["case"] == ("relay",1), "unexpected controlled-recovery scope")
    recovery = verify_recovery(root, recovered[0], inputs["recovery_report"], original_sources,
                               inputs["recovery_source_index"], inputs["recovery_driver_source"])
    physical = sum(item["report"]["interaction_accounting"]["physical_joint_decisions_including_initialization"] for item in datasets)
    initialization = sum(item["report"]["interaction_accounting"]["initialization_joint_decisions"] for item in datasets)
    require(physical == 200000 + initialization and native["totals"]["physical_decisions"] == physical, "whole final replay accounting does not reconcile")
    plan = cleanup_plan(root, raw, datasets, preflight, inputs["preflight_report"])
    result = {"schema_version":"phase21-finalized-acceptance-v1", "status":"complete_p21_pilot",
              "completion_method":"verified_explicit_continuation", "recorded_at_utc":datetime.now(timezone.utc).isoformat(),
              "scope":"P2.1 pilot initialization only; four datasets, no world-model/policy training or final-seed collection",
              "model_training_started":False, "final_seed_collection_started":False,
              "original_pipeline_status":"failed", "original_failed_pipeline_report":raw_file,
              "original_execution_source_snapshot":original_index,
              "required_evidence":{"production_verify":production_file,"independent_arrays":arrays_file,"independent_native_journals":native_file,
                  "unit_suite":units_file,"storage_benchmark":storage_file,"native_preflight":preflight_file,"aborted_slow_drive_preflight":prior_file},
              "datasets":[{"task_family":item["case"][0],"run_seed":item["case"][1],"path":item["directory"].relative_to(root).as_posix(),
                  "transition_count":50000,"collector_counts":{"random":35000,"scripted":15000},
                  "collection_report":item["report_file"],"dataset_manifest":item["manifest_file"],"training_ledger":item["ledger_file"]} for item in datasets],
              "recovery":recovery,
              "interaction_accounting":{"physical_initial_replay_joint_transitions":200000,"attributed_initial_joint_transitions_per_variant":{"B":200000,"D":200000},
                  "physical_dataset_copies":4,"initialization_joint_decisions":initialization,"failed_noninitialization_dispatches":0,
                  "final_replay_physical_decisions_including_initialization":physical,
                  "passed_preflight_physical_decisions":preflight_physical,"aborted_diagnostic_physical_decisions":prior_physical,
                  "passed_preflight_published_transitions":4000,
                  "aborted_diagnostic_published_transitions":sum(row["published_validation_transitions"] for row in prior["cases"]),
                  "aborted_diagnostic_unpublished_or_pending_initialization_attempts":sum(row.get("unpublished_or_pending_initialization_decisions",0) for row in prior["cases"]),
                  "all_collection_and_diagnostic_physical_decisions":physical+preflight_physical+prior_physical,
                  "diagnostic_work_enters_learning_replay":False,"recovery_prefix_counted_once":True},
              "storage_setting":storage["selected"],
              "resources":{"preflight":preflight["resources"],"controlled_continuation":recovery["resource_measurements"],
                  "original_main_final_resource_summary_available":"collection_resources" in raw,
                  "limitation":"The failed original orchestration did not assign its final initial-collection resource summary. Source-enforced four-core limits, measured preflight/continuation resources, and a separate point-in-time observation are evidence; no complete original initial-collection CPU/RAM time series is claimed."},
              "finalizer_sha256":sha(Path(__file__)),"source_and_raw_data_mutated_by_finalizer":False}
    if inputs.get("resource_observation"):
        observation, observation_file = document(root, inputs["resource_observation"])
        require(observation["status"] == "passed" and observation["logical_cpu_limit"] == 4
                and all(set(row["cpu_ids"]) <= {0,1,2,3} for row in observation["processes"]), "point-in-time affinity observation failed")
        result["resources"]["point_in_time_affinity_observation"] = observation_file
    return result, plan


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), "never overwrite finalized evidence: " + str(path))
    require(not path.with_suffix(".sha256").exists(), "a report receipt already exists; choose a fresh output")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    receipt = path.with_suffix(".sha256")
    with receipt.open("x",encoding="ascii") as stream:
        stream.write(sha(path)+"  "+path.name+"\n");stream.flush();os.fsync(stream.fileno())


def verify_cleanup(root, plan_path, acceptance_path=None):
    root = Path(root).resolve()
    plan, plan_file = document(root, plan_path, receipt=True)
    require(plan["status"] == "verified_cold_copies_ready_for_operator_cleanup" and plan["collection_host_os_name"] == os.name,
            "cleanup verification must run on the original collection-host OS")
    require(Path(plan["temporary_parent_on_collection_host"]).resolve() == Path(tempfile.gettempdir()).resolve(), "cleanup temporary parent differs from the collection host")
    for entry in plan["roots"]:
        hot = Path(entry["absolute_collection_host_root"])
        require(not hot.exists(), "hot journal root still exists: " + str(hot))
        for file in entry["verified_hot_files"]:
            cold = member(root,file["cold_copy"]["path"])
            require(sha(cold) == file["sha256"] and cold.stat().st_size == file["bytes"], "cold copy changed after hot cleanup")
    result = {"schema_version":"p21-hot-cleanup-verification-v1","status":"passed","recorded_at_utc":datetime.now(timezone.utc).isoformat(),
            "cleanup_plan":plan_file,"all_named_hot_roots_absent":True,"cold_evidence_copies_reverified":sum(len(row["verified_hot_files"]) for row in plan["roots"]),
            "deletion_performed_by_this_helper":False,"raw_dataset_and_historical_report_bytes_changed":False}
    if acceptance_path is not None:
        acceptance, acceptance_file = document(root,acceptance_path,receipt=True)
        require(acceptance["status"] == "complete_p21_pilot" and acceptance["cleanup_plan"]["sha256"] == plan_file["sha256"], "cleanup plan does not belong to the completed acceptance")
        result["acceptance_report"] = acceptance_file
        result["current_cleanup_state"] = "verified_complete; acceptance's earlier pending state is preserved as historical evidence"
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("finalize","verify-cleanup"))
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cleanup-plan",type=Path,required=True)
    parser.add_argument("--acceptance-report",type=Path,help="Optional completed acceptance to link from a new cleanup verification; it is never rewritten")
    for name in ("data-dir","raw-report","production-report","arrays-report","native-report","unit-report","storage-report","preflight-report",
                 "prior-diagnostic-report","original-source-index","recovery-report","recovery-source-index","recovery-driver-source","resource-observation"):
        parser.add_argument("--"+name,type=Path)
    args=parser.parse_args(argv);values=vars(args)
    root=args.project_root.resolve(strict=True)
    require(args.output.resolve().is_relative_to(root) and args.cleanup_plan.resolve().is_relative_to(root), "finalized evidence and cleanup plan must live under the named project")
    require(not args.output.exists(),"choose a new output report")
    if args.command=="verify-cleanup":
        result=verify_cleanup(args.project_root,args.cleanup_plan,args.acceptance_report)
    else:
        required=[name.replace("-","_") for name in ("data-dir","raw-report","production-report","arrays-report","native-report","unit-report","storage-report","preflight-report","prior-diagnostic-report","original-source-index","recovery-report","recovery-source-index","recovery-driver-source")]
        for name in required:require(values[name] is not None,"missing required --"+name.replace("_","-"))
        require(not args.cleanup_plan.exists(),"choose a new cleanup plan path")
        result,plan=finalize(values)
        write_new(args.cleanup_plan,plan)
        result["cleanup_plan"]=record(args.project_root.resolve(),args.cleanup_plan)
        result["cleanup_state"]="separate_verification_pending; canonical acceptance does not claim deletion"
    write_new(args.output,result)
    print(json.dumps({"status":result["status"],"output":str(args.output)}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
