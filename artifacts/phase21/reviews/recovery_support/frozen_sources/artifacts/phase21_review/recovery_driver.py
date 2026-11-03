"""Isolated P2.1 continuation driver; never edits original project sources.

Only staged ledger/replay/recovery modules are overlaid in this process. The
native environment, collection action policies, schema, compiler, and spawn
worker classes continue to resolve from the untouched original project.
"""


def main():
    import argparse
    import hashlib
    import importlib
    import importlib.util
    import json
    import os
    from pathlib import Path
    import sys
    import time
    import traceback

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--journal-root", type=Path)
    parser.add_argument("--recovery-id", default="startup-continuation-001")
    parser.add_argument("--failure-sha256", required=True)
    parser.add_argument("--original-project", type=Path, required=True)
    parser.add_argument("--staged-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if not args.inspect_only and args.journal_root is None:
        parser.error("a fresh --journal-root is required for native continuation")
    if args.output.resolve().is_relative_to(args.dataset.resolve()):
        parser.error("driver report must stay outside the immutable/original dataset evidence")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import psutil
    process = psutil.Process()
    previous_affinity = process.cpu_affinity()
    if not set([0, 1, 2, 3]).issubset(previous_affinity):
        raise RuntimeError("the required four-CPU allocation 0..3 is unavailable")
    process.cpu_affinity([0, 1, 2, 3])
    project = args.original_project.resolve()
    staged = args.staged_src.resolve()
    sys.path.insert(0, str(project / "src"))
    import world_models.data

    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    overlays = {}
    for short_name in ("ledger", "replay", "recovery"):
        name = "world_models.data." + short_name
        source = staged / "world_models" / "data" / (short_name + ".py")
        specification = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        setattr(world_models.data, short_name, module)
        specification.loader.exec_module(module)
        overlays[name] = {"path": str(source), "sha256": sha(source)}
    original_modules = {}
    for name in ("world_models.data.collection", "world_models.data.schema",
                 "world_models.env.coop", "world_models.env.coordinator", "world_models.env.workers",
                 "world_models.env.scenario_builder", "world_models.env.scripted", "world_models.env.splits",
                 "world_models.env.targets", "world_models.env.actions", "world_models.env.observations",
                 "world_models.contracts", "world_models.infrastructure.config",
                 "world_models.infrastructure.reproducibility"):
        module = importlib.import_module(name)
        source = Path(module.__file__).resolve()
        if not source.is_relative_to(project):
            raise RuntimeError("native generation unexpectedly resolved outside original project: " + name)
        original_modules[name] = {"path": str(source), "sha256": sha(source)}
    from world_models.data.ledger import atomic_json
    from world_models.data.recovery import inspect_recoverable_prefix, resume_initial_dataset
    from world_models.data.resources import CollectionResources
    original_manifest = json.loads((args.dataset / "replay/manifest.partial.json").read_text(encoding="utf-8"))
    expected_sources = original_manifest["metadata"]["origin_provenance"]["execution_source_sha256"]
    observed_sources = {relative: sha(project / relative) for relative in expected_sources}
    if observed_sources != expected_sources:
        raise RuntimeError("original execution sources changed before continuation")
    provenance = {"driver_path": str(Path(__file__).resolve()), "driver_sha256": sha(__file__),
                  "loaded_module_overlays": overlays, "original_generation_modules": original_modules,
                  "original_execution_source_sha256": observed_sources}
    report = {"status": "running", "inspect_only": args.inspect_only,
              "recovery_id": args.recovery_id, "dataset": str(args.dataset.resolve()),
              "requested_failure_sha256": args.failure_sha256, "implementation_provenance": provenance,
              "cpu_ids": [0, 1, 2, 3]}
    start = time.perf_counter()
    resources = None

    def progress(record):
        count = record["published_joint_transitions"]
        if count % 1000 == 0 or record["status"] == "complete_initial_replay":
            print(json.dumps({"status": record["status"], "recovery_id": args.recovery_id,
                "task_family": record["task_family"], "run_seed": record["run_seed"],
                "published": count, "target": record["target_transitions"],
                "completed_episodes": record["completed_episodes"]}), flush=True)

    try:
        with CollectionResources([0, 1, 2, 3]) as resources:
            if args.inspect_only:
                prefix = inspect_recoverable_prefix(args.dataset,
                    expected_failure_sha256=args.failure_sha256, origin_project_root=project)
                report["prefix"] = {"published": prefix["count"], "episodes": len(prefix["episodes"]),
                    "next_episode": prefix["expected_next_episode_id"],
                    "next_segment": prefix["remaining_segments"][0],
                    "remaining_transitions": sum(length for _, _, length in prefix["remaining_segments"]),
                    "native_zero_physics_failure_checked": prefix["native_zero_physics_failure_checked"],
                    "generation_proof": prefix["original_generation_proof"]}
            else:
                result = resume_initial_dataset(args.dataset, journal_root=args.journal_root,
                    recovery_id=args.recovery_id, expected_failure_sha256=args.failure_sha256,
                    origin_project_root=project, config_dir=project / "configs",
                    max_startup_attempts=2, progress_callback=progress,
                    recovery_origin_provenance=provenance)
                report["collection_status"] = result["status"]
                report["published_joint_transitions"] = result["published_joint_transitions"]
                report["interaction_accounting"] = result["interaction_accounting"]
                report["collection_report_sha256"] = sha(args.dataset / "collection_report.json")
        report["resource_measurements"] = resources.report()
        report["original_execution_sources_unchanged"] = observed_sources == {
            relative: sha(project / relative) for relative in observed_sources}
        report["overlay_sources_unchanged"] = all(sha(record["path"]) == record["sha256"]
                                                   for record in overlays.values())
        report["driver_source_unchanged"] = sha(__file__) == provenance["driver_sha256"]
        if (not report["original_execution_sources_unchanged"] or not report["overlay_sources_unchanged"]
                or not report["driver_source_unchanged"]):
            raise RuntimeError("original/continuation execution sources changed during recovery")
        report["status"] = "passed_inspection" if args.inspect_only else "complete_initial_replay"
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        if resources is not None and hasattr(resources, "elapsed"):
            try:
                report["resource_measurements"] = resources.report()
            except BaseException as resource_error:
                report["resource_report_error"] = repr(resource_error)
        report["wall_seconds"] = time.perf_counter() - start
        atomic_json(args.output, report)
        process.cpu_affinity(previous_affinity)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "wall_seconds": report["wall_seconds"], "error": report.get("error")}), flush=True)
    return 0 if report["status"] in ("passed_inspection", "complete_initial_replay") else 1


if __name__ == "__main__":
    raise SystemExit(main())
