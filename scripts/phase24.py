"""P2.4 fitting CLI. `plan`/`verify` never start research training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

import torch

from world_models.infrastructure.config import resolve_config
from world_models.training.fitting import FitSettings
from world_models.training.runner import (CASES, ROOT, evaluate_checkpoint, fit_case, load_case,
                                          run_lock, source_identity, write_report)


def _common(parser):
    parser.add_argument("--data-dir", type=Path, default=Path("data/initial_replay/pilot"))
    parser.add_argument("--statistics-dir", type=Path, default=Path("artifacts/phase23/normalization"))
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--profile", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--effective-batch", type=int)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--loss-steps", type=int)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-windows", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)


def _fitting(parser):
    _common(parser)
    parser.add_argument("--task", choices=("relay", "cover"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--resume", action="store_true")


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    subs = command.add_subparsers(dest="command", required=True)
    for name in ("initial", "overfit", "refresh"):
        item = subs.add_parser(name)
        _fitting(item)
        if name == "refresh":
            item.add_argument("--variant", choices=("B", "D"), required=True)
            item.add_argument("--parent-checkpoint", type=Path, required=True)
            item.add_argument("--replay", type=Path, action="append", required=True,
                              help="ordered online replay shards, oldest first; include all parent online shards")
    for name in ("plan", "launch"):
        item = subs.add_parser(name)
        _common(item)
        item.add_argument("--output-dir", type=Path, required=True)
        item.add_argument("--gpus", default="0", help="visible logical GPU indices, comma separated")
        item.add_argument("--workers-per-gpu", type=int, default=4)
        item.add_argument("--stage", choices=("initial", "overfit"), default="initial")
        item.add_argument("--updates", type=int)
        item.add_argument("--resume", action="store_true")
        if name == "plan":
            item.add_argument("--report", type=Path)
    for name in ("validate", "burn-in"):
        item = subs.add_parser(name)
        _common(item)
        item.add_argument("--checkpoint", type=Path, required=True)
        item.add_argument("--validation-bank", type=Path, required=name == "burn-in")
        item.add_argument("--device", default="cuda:0")
        item.add_argument("--output", type=Path, required=True)
    item = subs.add_parser("collect-validation")
    item.add_argument("--task", choices=("relay", "cover"), required=True)
    item.add_argument("--seed", type=int, required=True)
    item.add_argument("--profile", choices=("pilot", "final"), default="pilot")
    item.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    item.add_argument("--episodes-per-template", type=int, default=4)
    item.add_argument("--output-dir", type=Path, required=True)
    item = subs.add_parser("status")
    item.add_argument("--output-dir", type=Path, required=True)
    item = subs.add_parser("verify")
    item.add_argument("--output-dir", type=Path, required=True)
    return command


def settings_from_args(args, task="relay"):
    cfg = resolve_config(task, "B", args.profile, args.config_dir).settings
    changes = {name: getattr(args, name) for name in
               ("microbatch", "effective_batch", "burn_in", "loss_steps", "precision", "checkpoint_every",
                "validate_every", "validation_windows", "log_every") if getattr(args, name) is not None}
    return FitSettings.from_experiment(cfg, **changes)


def gpu_indices(value):
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("--gpus must be comma-separated visible logical GPU indices") from exc
    if not values or any(index < 0 for index in values) or len(set(values)) != len(values):
        raise ValueError("GPU indices must be unique and nonnegative")
    return values


def launch_plan(args):
    devices = gpu_indices(args.gpus)
    if type(args.workers_per_gpu) is not int or args.workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive")
    if args.updates is not None and (type(args.updates) is not int or args.updates < 1):
        raise ValueError("updates must be positive")
    slots = [gpu for _ in range(args.workers_per_gpu) for gpu in devices]
    if len(slots) < len(CASES):
        raise ValueError("four pilot fits need at least four parallel worker slots")
    if args.profile != "pilot":
        raise ValueError("four-case launcher is pilot-only; run final initial fits explicitly by task/seed")
    settings = settings_from_args(args)
    jobs = []
    for (task, seed), gpu in zip(CASES, slots):
        cfg, datasets, stats, provenance = load_case(task, seed, data_root=args.data_dir,
            statistics_root=args.statistics_dir, config_root=args.config_dir, profile=args.profile)
        output = args.output_dir.resolve() / task / f"seed_{seed}"
        command = [sys.executable, "-m", "scripts.phase24", args.stage, "--task", task, "--seed", str(seed),
                   "--output-dir", str(output), "--data-dir", str(args.data_dir.resolve()),
                   "--statistics-dir", str(args.statistics_dir.resolve()), "--config-dir", str(args.config_dir.resolve()),
                   "--profile", args.profile, "--device", "cuda:0"]
        for name in ("microbatch", "effective_batch", "burn_in", "loss_steps", "precision", "checkpoint_every",
                     "validate_every", "validation_windows", "log_every"):
            command += ["--" + name.replace("_", "-"), str(getattr(settings, name))]
        target_updates = args.updates if args.updates is not None else (50 if args.stage == "overfit" else
                                                                    cfg.settings["world_model"]["initial_updates"])
        command += ["--updates", str(target_updates)]
        if args.resume and (output / "run_manifest.json").is_file():
            command.append("--resume")
        jobs.append({"task": task, "seed": seed, "logical_gpu": gpu, "output_dir": str(output),
                     "command": command, "target_stage_updates": target_updates,
                     "data": provenance})
    return {"schema_version": "p24-launch-plan-v1", "mode": "parallel independent workers",
            "workers_per_gpu": args.workers_per_gpu, "gpus": devices, "worker_count": len(jobs),
            "stage": args.stage, "settings": asdict(settings), "new_environment_interactions": 0, "optimizer_updates_executed": 0,
            "training_executed": False, "jobs": jobs,
            "memory_note": "microbatch controls each worker's memory; aggregate GPU memory is measured on the server"}


def gpu_environment(logical_gpu, inherited=None):
    if type(logical_gpu) is not int or logical_gpu < 0:
        raise ValueError("logical GPU index must be a nonnegative integer")
    env = dict(os.environ if inherited is None else inherited)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if logical_gpu >= len(tokens):
            raise ValueError("logical GPU index is outside inherited CUDA_VISIBLE_DEVICES")
        selected = tokens[logical_gpu]
    else:
        selected = str(logical_gpu)
    env.update(CUDA_VISIBLE_DEVICES=selected, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    return env


def launch(args):
    plan = launch_plan(args)
    if not torch.cuda.is_available() or max(plan["gpus"]) >= torch.cuda.device_count():
        raise RuntimeError("selected GPU is unavailable; run launch inside your GPU allocation")
    output = args.output_dir.resolve()
    # Parent directory exists for the launcher; per-case directories remain
    # absent until each worker starts. A second parent cannot launch duplicates.
    output.mkdir(parents=True, exist_ok=True)
    processes, streams = [], []
    with run_lock(output):
        plan_path = output / "launch_plan.json"
        if plan_path.exists() and not args.resume:
            raise FileExistsError("launch output already used; choose fresh output or explicit --resume")
        if args.resume:
            if not plan_path.is_file():
                raise ValueError("resume needs the original launch plan")
            previous = json.loads(plan_path.read_text(encoding="utf-8"))
            def stable(value):
                value = json.loads(json.dumps(value))
                for job in value["jobs"]:
                    job["command"] = [part for part in job["command"] if part != "--resume"]
                return value
            if stable(previous) != stable(plan):
                raise ValueError("launch plan changed; refusing to replace the original plan or resume")
            write_report(output / f"launch_plan_resume_{time.time_ns()}.json", plan)
        else:
            write_report(plan_path, plan)
        (output / "worker_logs").mkdir(exist_ok=True)
        try:
            # Start every worker before waiting on any one of them.
            for job in plan["jobs"]:
                log = (output / "worker_logs" / f"{job['task']}_seed_{job['seed']}.log").open("a", encoding="utf-8")
                streams.append(log)
                process = subprocess.Popen(job["command"], cwd=ROOT, env=gpu_environment(job["logical_gpu"]),
                                           stdout=log, stderr=subprocess.STDOUT)
                processes.append((job, process))
            while any(process.poll() is None for _, process in processes):
                state = [{"task": job["task"], "seed": job["seed"], "pid": process.pid,
                          "returncode": process.poll(), "logical_gpu": job["logical_gpu"]}
                         for job, process in processes]
                write_report(output / "launch_status.json", {"status": "running", "workers": state})
                time.sleep(2)
        except BaseException:
            for _, process in processes:
                if process.poll() is None:
                    process.terminate()
            for _, process in processes:
                process.wait()
            raise
        finally:
            for stream in streams:
                stream.close()
        codes = [{"task": job["task"], "seed": job["seed"], "returncode": process.returncode}
                 for job, process in processes]
        complete = all(row["returncode"] == 0 for row in codes)
        write_report(output / "launch_status.json", {"status": "complete_fit_stages" if complete else "failed",
                                                     "workers": codes, "P2.5_quality_acceptance": False})
        if not complete:
            raise RuntimeError("one or more workers failed; inspect worker_logs and per-case status before --resume")
        return {"status": "complete_fit_stages", "workers": codes, "P2.5_quality_acceptance": False}


def verify(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("verification requires a fresh evidence directory")
    output.mkdir(parents=True)
    names = ("tests.test_model_sampling", "tests.test_model_checkpoints", "tests.test_model_fitting",
             "tests.test_model_validation_bank", "tests.test_phase24_cli")
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
    with (output / "tests.txt").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"schema_version": "p24-implementation-v1", "status": "complete_p24_implementation" if result.wasSuccessful() else "failed",
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "source_sha256": source_identity(),
              "research_training_executed": False, "new_environment_interactions": 0,
              "local_checks": "CPU fixtures and tiny optimizer unit tests only",
              "gpu_training_measured": False, "P2.5_quality_acceptance": False}
    write_report(output / "phase24_acceptance.json", report)
    if not result.wasSuccessful():
        raise RuntimeError("P2.4 implementation checks failed; inspect tests.txt")
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "verify":
        result = verify(args.output_dir)
    elif args.command == "status":
        results = []
        for task, seed in CASES:
            path = args.output_dir / task / f"seed_{seed}" / "status.json"
            results.append(json.loads(path.read_text()) if path.is_file() else {"task_family": task, "run_seed": seed, "status": "not_started"})
        result = {"cases": results, "P2.5_quality_acceptance": False}
    elif args.command == "plan":
        result = launch_plan(args)
        if args.report:
            write_report(args.report, result)
    elif args.command == "launch":
        result = launch(args)
    elif args.command == "collect-validation":
        from world_models.training.validation import collect_validation_bank
        result = collect_validation_bank(args.task, args.seed, args.output_dir,
            episodes_per_template=args.episodes_per_template, profile=args.profile, config_dir=args.config_dir)
    elif args.command in ("validate", "burn-in"):
        if args.output.exists():
            raise FileExistsError("diagnostic output already exists; use a fresh path")
        result = evaluate_checkpoint(args.checkpoint, data_root=args.data_dir, statistics_root=args.statistics_dir,
            settings=settings_from_args(args), device=args.device, validation_bank=args.validation_bank,
            burn_in_study=args.command == "burn-in", config_root=args.config_dir)
        write_report(args.output, result)
    else:
        settings = settings_from_args(args, args.task)
        if args.command == "overfit" and args.updates is None:
            args.updates = 50
        result = fit_case(args.task, args.seed, args.output_dir, data_root=args.data_dir,
            statistics_root=args.statistics_dir, config_root=args.config_dir, profile=args.profile,
            settings=settings, device=args.device, updates=args.updates, resume=args.resume,
            stage=args.command, variant=getattr(args, "variant", None),
            parent_checkpoint=getattr(args, "parent_checkpoint", None), additional_replays=getattr(args, "replay", ()))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
