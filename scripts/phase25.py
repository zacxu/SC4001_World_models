"""Compact P2.5 engineering screen; never starts training or simulator collection."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from world_models.evaluation.verification import verify_implementation
from world_models.training.runner import ROOT


def _positive_windows(value):
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("windows must be an integer from 1 to 64") from error
    if not 1 <= count <= 64:
        raise argparse.ArgumentTypeError("windows must be an integer from 1 to 64")
    return count


def _nonnegative(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return number


def _gpu_indices(value):
    try:
        indices = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("gpus must be distinct nonnegative logical indices, e.g. 0,1") from error
    if not indices or len(indices) > 4 or min(indices) < 0 or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("gpus must contain one to four distinct nonnegative logical indices")
    return indices


def _workers(value):
    number = _nonnegative(value)
    if not 1 <= number <= 4:
        raise argparse.ArgumentTypeError("workers-per-gpu must be between 1 and 4")
    return number


def _evaluation_args(command, *, parallel=False):
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--data-dir", type=Path, default=Path("data/initial_replay/pilot"))
    command.add_argument("--statistics-dir", type=Path, default=Path("artifacts/phase23/normalization"))
    command.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    devices = command.add_mutually_exclusive_group() if parallel else command
    devices.add_argument("--device", default="cuda:0")
    if parallel:
        devices.add_argument("--gpus", type=_gpu_indices,
                             help="allocated logical GPU indices; each worker sees only its assigned GPU")
        command.add_argument("--workers-per-gpu", type=_workers, default=1,
                             help="parallel workers per GPU; provide at least four total slots")
    command.add_argument("--windows", type=_positive_windows, default=8)
    command.add_argument("--seed", type=_nonnegative, default=0)
    command.add_argument("--precision", choices=("float32", "bfloat16"), default=None)
    command.add_argument("--verification-report", type=Path)


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    subs = command.add_subparsers(dest="command", required=True)
    verify = subs.add_parser("verify", help="implementation fixtures only; no research model evaluated")
    verify.add_argument("--output-dir", type=Path, required=True)
    evaluate = subs.add_parser("evaluate", help="screen one explicit checkpoint")
    _evaluation_args(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    campaign = subs.add_parser("campaign", help="screen relay/cover seeds 0/1, optionally in parallel")
    _evaluation_args(campaign, parallel=True)
    selection = campaign.add_mutually_exclusive_group(required=True)
    selection.add_argument("--selection", type=Path,
                           help="immutable P2.4 checkpoint selection JSON; permits unequal update counts")
    selection.add_argument("--run-root", type=Path)
    campaign.add_argument("--checkpoint-update", type=_nonnegative,
                          help="explicit common update; required with --run-root, forbidden with --selection")
    return command


def evaluate_checkpoint(*args, **kwargs):
    # Keep implementation verification independent of checkpoint/data loaders.
    from world_models.evaluation.compact import evaluate_checkpoint as evaluate
    return evaluate(*args, **kwargs)


def evaluate_campaign(*args, **kwargs):
    from world_models.evaluation.compact import evaluate_campaign as evaluate
    return evaluate(*args, **kwargs)


def _gpu_environments(indices):
    import torch
    from scripts.phase24 import gpu_environment
    if not torch.cuda.is_available() or max(indices) >= torch.cuda.device_count():
        raise ValueError("selected GPUs are unavailable; run campaign inside the GPU allocation")
    environments = [gpu_environment(index) for index in indices]
    if len({env["CUDA_VISIBLE_DEVICES"] for env in environments}) != len(indices):
        raise ValueError("selected GPUs must have distinct allocated CUDA visibility tokens")
    return environments


@contextmanager
def _interrupt_on_termination():
    # Translate scheduler termination into normal finally-block cleanup.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"P2.5 campaign received signal {signum}")
    installed = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, interrupted) if installed else None
    try:
        yield
    finally:
        if installed:
            signal.signal(signal.SIGTERM, previous)


def _run_workers(jobs, environments):
    processes, streams = [], []
    try:
        with _interrupt_on_termination():
            for job, environment in zip(jobs, environments):
                stream = Path(job["log_path"]).open("x", encoding="utf-8")
                streams.append(stream)
                # stderr inherits the terminal/PBS log so window progress remains live.
                process = subprocess.Popen(job["command"], cwd=ROOT, env=environment, stdout=stream)
                processes.append(process)
                print(f"P2.5 {job['task_family']}/seed_{job['run_seed']}: started on logical GPU "
                      f"{job['logical_gpu']}; log {job['log_path']}", file=sys.stderr, flush=True)
            pending = set(range(len(processes)))
            while pending:
                for index in tuple(pending):
                    code = processes[index].poll()
                    if code is None:
                        continue
                    pending.remove(index)
                    if code != 0:
                        job = jobs[index]
                        raise RuntimeError(f"P2.5 worker {job['task_family']}/seed_{job['run_seed']} "
                                           f"exited {code}; inspect the job stderr and {job['log_path']}")
                if pending:
                    time.sleep(.2)
    finally:
        # A failed launch, worker failure, Ctrl-C or SIGTERM must leave no workers running.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in streams:
            stream.close()


def _read_case_report(job, *, checkpoint_update, windows, seed, source, evidence):
    from world_models.data.replay import sha256_file
    from world_models.evaluation.compact import VERSION
    path = Path(job["output_dir"])/"report.json"
    receipt = path.with_suffix(".sha256")
    if not path.is_file() or not receipt.is_file():
        raise ValueError(f"worker did not commit its evaluation report: {path}")
    digest = sha256_file(path)
    if receipt.read_text(encoding="ascii").strip().split() != [digest, path.name]:
        raise ValueError(f"worker evaluation report SHA receipt mismatch: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    cp = report.get("checkpoint", {})
    if ((cp.get("task_family"), cp.get("run_seed"), cp.get("update")) !=
            (job["task_family"], job["run_seed"], checkpoint_update)
            or cp.get("sha256") != job["checkpoint_sha256"]
            or Path(cp.get("path", "")).resolve() != Path(job["checkpoint_path"])
            or cp.get("shared_by_variants") != ["B", "D"]):
        raise ValueError("campaign checkpoint metadata does not match its explicit task/seed/update path")
    if "original_target_stage_updates" in job and (
            cp.get("original_target_stage_updates") != job["original_target_stage_updates"]
            or cp.get("original_training_target_completed") is not
            (checkpoint_update >= job["original_target_stage_updates"])):
        raise ValueError("worker checkpoint report changed the original training target or completion state")
    readiness = report.get("pilot_readiness", {})
    verification = report.get("verification", {})
    if (report.get("schema_version") != VERSION or report.get("P2.5_quality_acceptance") is not False
            or readiness.get("status") not in ("needs_review", "ready_for_exploratory_phase3")
            or report.get("source_sha256") != source or report.get("seed") != seed
            or verification.get("status") != "passed"
            or verification.get("report_sha256") != evidence["report_sha256"]
            or not isinstance(report.get("windows"), list) or not 1 <= len(report["windows"]) <= windows
            or report.get("new_optimizer_updates") != 0 or report.get("new_real_interactions") != 0):
        raise ValueError(f"worker evaluation is incomplete or differs from the requested campaign: {path}")
    return {"report_path": str(path), "report_sha256": digest, "checkpoint": cp,
            "pilot_readiness": readiness, "timing": report["timing"]}


def evaluate_parallel_campaign(run_root, checkpoint_update, output_dir, *, gpus, workers_per_gpu,
                               data_root, statistics_root, config_root=None, windows=8, seed=0,
                               precision=None, verification_report=None, selection=None, device="cuda:0"):
    """Screen explicit checkpoints, with optional parallel GPU workers.

    The frozen-selection path also supports sequential CPU evaluation. Selection
    receipt verification fixes the four files before any evaluation begins; it
    never performs another latest-checkpoint search.
    """
    import torch
    from world_models.data.replay import canonical_hash, sha256_file
    from world_models.evaluation import compact
    from world_models.training.checkpoints import read_checkpoint
    compact._bounds(windows, seed)
    if selection is None and (type(checkpoint_update) is not int or checkpoint_update < 1):
        raise ValueError("campaign checkpoint update must be a positive explicit common update")
    if selection is not None and (run_root is not None or checkpoint_update is not None):
        raise ValueError("--selection cannot be combined with --run-root or --checkpoint-update")
    parallel = gpus is not None
    if parallel and (not isinstance(gpus, list) or not gpus or len(gpus) > 4
            or any(type(index) is not int or index < 0 for index in gpus) or len(set(gpus)) != len(gpus)
            or type(workers_per_gpu) is not int or not 1 <= workers_per_gpu <= 4
            or len(gpus)*workers_per_gpu < 4):
        raise ValueError("parallel campaign needs at least four slots; use --gpus 0,1 --workers-per-gpu 2")
    if not parallel and (selection is None or workers_per_gpu != 1):
        raise ValueError("--workers-per-gpu requires --gpus; sequential campaign requires a frozen selection")
    directory = compact._fresh(Path(output_dir).resolve())
    started, threads = time.perf_counter(), torch.get_num_threads()
    reports, jobs = [], []
    frozen, selection_evidence, selection_path = None, None, None
    try:
        torch.set_num_threads(1)
        source = compact.evaluator_source_identity()
        evidence = compact.prepare_verification(directory/"verification", reuse_path=verification_report)
        if selection is not None:
            from world_models.training.selection import load_selection
            selection_path = Path(selection).resolve()
            frozen = load_selection(selection_path)
            selected = {(row["task"], row["seed"]): row for row in frozen["cases"]}
            case_rows = [selected[case] for case in compact.CASES]
            paths = [(selection_path.parent/row["checkpoint_path"]).resolve() for row in case_rows]
            updates = [row["model_updates"] for row in case_rows]
            selection_evidence = {"path": str(selection_path), "sha256": sha256_file(selection_path),
                "schema_version": frozen["schema_version"], "selection_rule": frozen["selection_rule"],
                "reason": frozen["reason"],
                "user_authorized_shortened_pilot": frozen["user_authorized_shortened_pilot"],
                "full_schedule_completion_claim": frozen["full_schedule_completion_claim"],
                "manifest": frozen}
        else:
            case_rows = None
            paths = [(Path(run_root)/task/f"seed_{run_seed}"/"checkpoints"/
                      f"update_{checkpoint_update:08d}.pt").resolve() for task, run_seed in compact.CASES]
            updates = [checkpoint_update]*len(compact.CASES)
        for path in paths:
            if not path.is_file() or not path.with_suffix(".sha256").is_file():
                raise FileNotFoundError(f"missing common-update checkpoint/receipt: {path}")
        # Read and validate every committed checkpoint on CPU before launching any GPU work.
        if frozen is None:  # load_selection has already validated all four full payloads.
            for (task, run_seed), path in zip(compact.CASES, paths):
                payload = read_checkpoint(path)
                meta = payload["metadata"]
                if ((meta.get("task_family"), meta.get("run_seed"), payload["update"]) !=
                        (task, run_seed, checkpoint_update)
                        or (meta.get("profile"), meta.get("stage"), meta.get("variant")) !=
                        ("pilot", "initial", "shared_initial:B,D")):
                    raise ValueError("campaign checkpoint metadata does not match its explicit task/seed/update path")
                del payload
        environments = _gpu_environments(gpus) if parallel else []
        if parallel:
            (directory/"worker_logs").mkdir()
        worker_environments = []
        for index, ((task, run_seed), path) in enumerate(zip(compact.CASES, paths)):
            output = directory/task/f"seed_{run_seed}"
            job = {"task_family": task, "run_seed": run_seed, "checkpoint_update": updates[index],
                   "checkpoint_path": str(path), "checkpoint_sha256": sha256_file(path),
                   "output_dir": str(output)}
            if case_rows is not None:
                row = case_rows[index]
                if job["checkpoint_sha256"] != row["checkpoint_sha256"]:
                    raise ValueError("selected checkpoint changed after frozen selection verification")
                job["original_target_stage_updates"] = row["original_target_stage_updates"]
            jobs.append(job)
            if not parallel:
                continue
            slot = index % len(gpus)
            command = [sys.executable, "-m", "scripts.phase25", "evaluate", "--checkpoint", str(path),
                       "--output-dir", str(output), "--data-dir", str(Path(data_root).resolve()),
                       "--statistics-dir", str(Path(statistics_root).resolve()), "--config-dir",
                       str(Path(config_root or ROOT/"configs").resolve()), "--device", "cuda:0",
                       "--windows", str(windows), "--seed", str(seed), "--verification-report",
                       str(Path(evidence["report_path"]).resolve())]
            if precision is not None:
                command.extend(("--precision", precision))
            job.update({"logical_gpu": gpus[slot],
                         "cuda_visible_devices": environments[slot]["CUDA_VISIBLE_DEVICES"],
                         "command": command,
                         "log_path": str(directory/"worker_logs"/f"{task}_seed_{run_seed}.log")})
            worker_environments.append(environments[slot])
        launch = {"gpus": gpus, "workers_per_gpu": workers_per_gpu,
                  "worker_count": 4 if parallel else 1, "mode": "parallel" if parallel else "sequential",
                  "jobs": jobs}
        compact.write_json(directory/"launch_plan.json", launch)
        if parallel:
            _run_workers(jobs, worker_environments)
        for job in jobs:
            if not parallel:
                case_directory = compact._fresh(Path(job["output_dir"]))
                compact._evaluate_one(Path(job["checkpoint_path"]), case_directory,
                    verification=evidence, data_root=data_root, statistics_root=statistics_root,
                    config_root=config_root, device=device, windows=windows, seed=seed, precision=precision)
            reports.append(_read_case_report(job, checkpoint_update=job["checkpoint_update"], windows=windows,
                                            seed=seed, source=source, evidence=evidence))
        if selection_path is not None:
            if (load_selection(selection_path) != frozen
                    or sha256_file(selection_path) != selection_evidence["sha256"]):
                raise ValueError("frozen checkpoint selection changed during evaluation")
        if compact.evaluator_source_identity() != source:
            raise RuntimeError("source changed during evaluation; evidence invalid")
        ready = all(row["pilot_readiness"]["status"] == "ready_for_exploratory_phase3" for row in reports)
        result = {"schema_version": compact.VERSION, "P2.5_quality_acceptance": False,
            "pilot_readiness": {"status": "ready_for_exploratory_phase3" if ready else "needs_review",
                "reasons": [f"{row['checkpoint']['task_family']}/seed_{row['checkpoint']['run_seed']}: {reason}"
                            for row in reports for reason in row["pilot_readiness"]["reasons"]],
                "unchecked_classes": sorted({item for row in reports for item in row["pilot_readiness"]["unchecked_classes"]}),
                "scope": "four shared initial models; exploratory integration pilot only", "automatic_phase3_launch": False},
            "cases": reports, "checkpoint_update": checkpoint_update, "windows_per_model_cap": windows,
            "seed": seed, "verification": evidence, "source_sha256": source, "launch": launch,
            "new_optimizer_updates": 0, "new_real_interactions": 0, "total_seconds": time.perf_counter()-started}
        frozen_context = {}
        if selection_evidence is not None:
            frozen_context = {"initial_checkpoint_selection": selection_evidence,
                "protocol_deviation": {"kind": "shortened_initial_model_training", "reason": frozen["reason"],
                    "unequal_updates_allowed": True, "full_schedule_completion_claim": False,
                    "comparison_unit": "B and D share the same selected checkpoint within each task/seed"},
                "checkpoint_updates": {f"{job['task_family']}/seed_{job['run_seed']}": job["checkpoint_update"]
                                       for job in jobs}}
            result.update(frozen_context)
        model_selection = compact.write_json(directory/"pilot_model_selection.json", {"schema_version": compact.VERSION,
            "P2.5_quality_acceptance": False, "pilot_readiness": result["pilot_readiness"],
            "campaign_identity_sha256": canonical_hash(result),
            **frozen_context,
            "models": [{"checkpoint": row["checkpoint"], "evaluation_report_path": row["report_path"],
                        "evaluation_report_sha256": row["report_sha256"]} for row in reports]})
        result["pilot_model_selection"] = model_selection
        compact.write_json(directory/"report.json", result)
        return result
    except BaseException as error:
        compact._failure(directory, error, cases=reports, checkpoint_update=checkpoint_update,
                         worker_jobs=jobs, initial_checkpoint_selection=selection_evidence)
        raise
    finally:
        torch.set_num_threads(threads)


def evaluate_selected_campaign(selection, output_dir, *, gpus=None, workers_per_gpu=1, **kwargs):
    return evaluate_parallel_campaign(None, None, output_dir, selection=selection,
                                      gpus=gpus, workers_per_gpu=workers_per_gpu, **kwargs)


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.command == "campaign":
        if args.selection is not None and args.checkpoint_update is not None:
            command.error("--checkpoint-update cannot be combined with --selection")
        if args.run_root is not None and args.checkpoint_update is None:
            command.error("--run-root requires --checkpoint-update")
    try:
        if args.command == "verify":
            result = verify_implementation(args.output_dir)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0
        kwargs = {"data_root": args.data_dir, "statistics_root": args.statistics_dir,
                  "config_root": args.config_dir, "device": args.device, "windows": args.windows,
                  "seed": args.seed, "precision": args.precision, "verification_report": args.verification_report}
        if args.command == "evaluate":
            result = evaluate_checkpoint(args.checkpoint, args.output_dir, **kwargs)
        elif args.selection is not None:
            result = evaluate_selected_campaign(args.selection, args.output_dir, gpus=args.gpus,
                                               workers_per_gpu=args.workers_per_gpu, **kwargs)
        elif args.gpus is not None:
            kwargs.pop("device")
            result = evaluate_parallel_campaign(args.run_root, args.checkpoint_update, args.output_dir,
                                               gpus=args.gpus, workers_per_gpu=args.workers_per_gpu, **kwargs)
        else:
            if args.workers_per_gpu != 1:
                raise ValueError("--workers-per-gpu requires --gpus")
            result = evaluate_campaign(args.run_root, args.checkpoint_update, args.output_dir, **kwargs)
        readiness = result.get("pilot_readiness", {})
        print(json.dumps({"report_path": str(args.output_dir / "report.json"),
                          "pilot_readiness": readiness, "P2.5_quality_acceptance": False},
                         indent=2, sort_keys=True, allow_nan=False))
        return 1 if readiness.get("status") == "failed" else 0
    except Exception as error:
        print(f"P2.5 error ({type(error).__name__}): {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("P2.5 campaign interrupted; launched workers stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
