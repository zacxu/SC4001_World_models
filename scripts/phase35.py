"""Plan, prepare and explicitly run variant B's P3.5 training cycle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import time
import unittest

import torch

from scripts.phase31 import RecordedResult
from world_models.data.ledger import atomic_json
from world_models.cycle.controller import CycleController, CycleOptions
from world_models.cycle.provenance import ROOT, source_identity as production_source_identity
from world_models.cycle.schedule import Schedule
from world_models.cycle.study import STUDY_PROFILES, apply_study_profile
from world_models.infrastructure.config import resolve_config
from world_models.training.runner import load_case


TEST_MODULES = ("tests.test_cycle_schedule", "tests.test_cycle_checkpoints", "tests.test_cycle_collection",
                "tests.test_cycle_refresh", "tests.test_cycle_controller", "tests.test_phase35_cli",
                "tests.test_cycle_study", "tests.test_cycle_recovery", "tests.test_local_actor",
                "tests.test_mappo_recurrent", "tests.test_mappo_objectives", "tests.test_mappo_updater")


def source_identity():
    sources = production_source_identity()
    sources.update({name.replace(".", "/") + ".py": hashlib.sha256(
        (ROOT / (name.replace(".", "/") + ".py")).read_bytes()).hexdigest() for name in TEST_MODULES})
    return sources


def verify(output_dir):
    spec = importlib.util.find_spec("tests")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != ROOT / "tests/__init__.py":
        raise ImportError("run from the complete project root, including tests/")
    before = source_identity()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    threads, started = torch.get_num_threads(), time.perf_counter()
    torch.set_num_threads(1)
    try:
        with (output / "tests.txt").open("x", encoding="utf-8", newline="\n") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(
                unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES))
        unchanged = before == source_identity()
        passed = result.wasSuccessful() and not result.skipped and unchanged
        report = {"schema_version": "p35-implementation-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                  "status": "complete_p35_implementation" if passed else "failed",
                  "scope": "CPU synthetic replay, mock engine fixtures, actual tiny model/PPO updates and controller resume",
                  "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                  "skipped": len(result.skipped), "passed_tests": sorted(result.passed_ids), "source_sha256": before,
                  "source_unchanged_during_verification": unchanged, "pytorch": str(torch.__version__), "device": "cpu",
                  "elapsed_seconds": time.perf_counter()-started, "new_environment_interactions": 0,
                  "research_training_executed": False, "research_replay_used": False,
                  "fixture_optimizer_updates_executed": True, "native_engine_runtime_measured": False,
                  "gpu_runtime_measured": False, "learned_policy_quality_measured": False, "phase3_acceptance": False}
        content = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        path = output / "implementation_checks.json"
        path.write_bytes(content)
        path.with_suffix(".sha256").write_text(f"{hashlib.sha256(content).hexdigest()}  {path.name}\n", encoding="ascii")
        return report
    finally:
        torch.set_num_threads(threads)


def _runtime_setup():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="CPU implementation checks only")
    check.add_argument("--output-dir", required=True)
    plan = commands.add_parser("plan", help="print budgets without allocating a GPU or starting training")
    prepare = commands.add_parser("prepare", help="bind a completed P2.4 checkpoint or frozen pilot selection and initialize B")
    for command in (plan, prepare):
        command.add_argument("--task", choices=("relay", "cover"), required=True)
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--profile", choices=("pilot", "final"), default="pilot")
        command.add_argument("--study-profile", choices=STUDY_PROFILES,
                             help="explicit downstream budget amendment; requires the frozen initial selection")
        command.add_argument("--config-root", default=str(ROOT / "configs"))
    prepare.add_argument("--run-dir", required=True)
    plan.add_argument("--initial-selection", help="frozen P2.4 pilot selection; use actual per-case model updates")
    initial = prepare.add_mutually_exclusive_group(required=True)
    initial.add_argument("--initial-checkpoint", help="completed registered P2.4 checkpoint")
    initial.add_argument("--initial-selection", help="receipt-verified frozen P2.4 pilot selection")
    prepare.add_argument("--data-root", default="data/initial_replay/pilot")
    prepare.add_argument("--statistics-root", default="artifacts/phase23/normalization")
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    prepare.add_argument("--checkpoint-cycles", type=int, default=10)
    prepare.add_argument("--checkpoint-model-updates", type=int, default=250)
    prepare.add_argument("--model-microbatch", type=int, default=4)
    run = commands.add_parser("run", help="resume the latest committed state and execute the training cycle")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--max-units", type=int, help="stop and checkpoint after this many complete work units")
    recover = commands.add_parser("recover", help="preserve a failed initial run in a fresh actor-FP32 continuation")
    recover.add_argument("--source-run", required=True)
    recover.add_argument("--run-dir", required=True, help="fresh destination; keep the original run")
    recover.add_argument("--device", default="cuda:0", help="same logical GPU/runtime as the original run")
    status = commands.add_parser("status", help="read status without loading weights or using a GPU")
    status.add_argument("--run-dir", required=True)
    stop = commands.add_parser("stop", help="request a checkpointed stop after the current complete work unit")
    stop.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command in ("plan", "prepare") and args.study_profile:
        if args.profile != "pilot" or not args.initial_selection:
            parser.error("--study-profile requires --profile pilot and --initial-selection")
        if args.task != "relay" or args.seed not in (0, 1):
            parser.error("relay_week_v1 permits Relay master seeds 0 and 1 only")
    if args.command == "verify":
        result = verify(args.output_dir)
        print(json.dumps({key: result[key] for key in ("status", "tests_run", "failures", "errors", "skipped",
                                                       "research_training_executed", "phase3_acceptance")}, indent=2))
        return 0 if result["status"] == "complete_p35_implementation" else 1
    if args.command == "status":
        print((Path(args.run_dir) / "status.json").read_text(encoding="utf-8"))
        return 0
    if args.command == "stop":
        directory = Path(args.run_dir)
        if not (directory / "run_manifest.json").is_file():
            parser.error("stop requires an existing prepared run")
        atomic_json(directory / "stop.request", {"requested_at": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"status": "stop_requested", "run_dir": str(directory)}))
        return 0
    if args.command == "plan":
        settings = resolve_config(args.task, "B", args.profile, args.config_root).settings
        if args.seed not in settings["experiment"]["master_seeds"]:
            parser.error("seed is outside the selected profile")
        parent_settings, amendment = settings, None
        if args.study_profile:
            settings, amendment = apply_study_profile(settings, args.study_profile, master_seed=args.seed)
        schedule = Schedule.from_experiment(settings)
        selection = None
        if args.initial_selection:
            amended = {"study_amendment": amendment} if amendment is not None else {}
            schedule, selection = Schedule.from_initial_selection(settings, args.seed, args.initial_selection, **amended)
        report = {"task": args.task, "variant": "B", "seed": args.seed, "profile": args.profile,
                  **schedule.describe(), "ppo_optimizer_steps_per_cycle":
                  settings["mappo"]["epochs"] * settings["mappo"]["recurrent_minibatches"],
                  "native_environment_required_for_run": True, "training_executed": False}
        if selection is not None:
            report.update(initial_selection=selection, study_scope="exploratory_pilot",
                          registered_schedule=Schedule.from_experiment(settings).describe())
        if amendment is not None:
            report.update(study_amendment=amendment,
                          parent_schedule=Schedule.from_experiment(parent_settings).describe())
        print(json.dumps(report, indent=2))
        return 0
    _runtime_setup()
    if args.command == "recover":
        from world_models.cycle.recovery import recover_run
        report = recover_run(args.source_run, args.run_dir, device=args.device)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    if args.command == "prepare":
        cfg, datasets, stats, _ = load_case(args.task, args.seed, data_root=args.data_root,
            statistics_root=args.statistics_root, config_root=args.config_root, profile=args.profile)
        settings, amended = cfg.settings, {}
        if args.study_profile:
            settings, amendment = apply_study_profile(settings, args.study_profile, master_seed=args.seed)
            amended = {"study_amendment": amendment}
        initial_input = ({"initial_selection": args.initial_selection} if args.initial_selection else
                         {"initial_checkpoint": args.initial_checkpoint})
        controller = CycleController.prepare(args.run_dir, settings=settings, master_seed=args.seed,
            initial_dataset=datasets[0], normalization=stats, **initial_input, device=args.device,
            options=CycleOptions(precision=args.precision, checkpoint_cycles=args.checkpoint_cycles,
                                 checkpoint_model_updates=args.checkpoint_model_updates, model_microbatch=args.model_microbatch),
            **amended)
        print(json.dumps({"status": "prepared", "run_dir": str(controller.directory),
                          "schedule": controller.schedule.describe(), "training_executed": False}, indent=2))
        return 0
    controller = CycleController.resume(args.run_dir, device=args.device)
    originals = {}
    def stop_requested(_signal, _frame):
        controller.request_stop()
        print("Stop requested; finishing the current real block, model update or PPO cycle before checkpointing.", flush=True)
    try:
        for name in ("SIGINT", "SIGTERM"):
            if hasattr(signal, name):
                signum = getattr(signal, name)
                originals[signum] = signal.signal(signum, stop_requested)
        print(json.dumps(controller.run(max_units=args.max_units), sort_keys=True, indent=2))
    finally:
        for signum, handler in originals.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
