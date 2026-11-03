"""P2.3 fitting-label statistics and objective verification; no optimization."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import unittest


CASES = tuple((family, seed) for family in ("relay", "cover") for seed in (0, 1))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    path.with_suffix(".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="ascii")


def execution_sources(root: Path) -> dict:
    paths = [*sorted((root / "src/world_models/training").glob("*.py")),
             *sorted((root / "src/world_models/models").glob("*.py")),
             *sorted((root / "tests").glob("test_model_*.py")), root / "scripts/phase23.py",
             root / "src/world_models/contracts.py", root / "src/world_models/data/schema.py",
             root / "src/world_models/data/replay.py", root / "src/world_models/env/targets.py",
             root / "src/world_models/infrastructure/config.py", root / "src/world_models/infrastructure/reproducibility.py",
             root / "configs/base.yaml", root / "configs/tasks/relay.yaml", root / "configs/tasks/cover.yaml",
             root / "requirements-phase2-cpu.lock"]
    return {path.relative_to(root).as_posix(): sha256(path) for path in paths}


def fit_statistics(root: Path, data: Path, output: Path, *, workers: int = 1) -> dict:
    """One immutable statistics object per task/seed, shared by both variants."""
    from world_models.data.replay import ReplayDataset
    from world_models.infrastructure.config import resolve_config
    from world_models.models.heads import TaskSpec
    from world_models.training.normalization import fit_normalization
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("statistics workers must be between one and four")
    output.mkdir(parents=True, exist_ok=False)
    sources = execution_sources(root)
    started = time.perf_counter()
    def fit(case):
        family, seed = case
        dataset = ReplayDataset(data / family / f"seed_{seed}" / "replay", verify=False)
        config = resolve_config(family, "B", "pilot", root / "configs").settings
        stats = fit_normalization(dataset, task=TaskSpec.from_experiment(config))
        path = output / family / f"seed_{seed}.json"
        stats.save(path)
        print(json.dumps({"statistics_complete": family, "seed": seed, "states": stats.provenance["unique_state_count"]}), flush=True)
        return {"family": family, "seed": seed, "path": path.relative_to(output).as_posix(), "sha256": sha256(path),
                "states": stats.provenance["unique_state_count"], "transitions": stats.provenance["fitting_transition_count"]}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cases = list(pool.map(fit, CASES))
    if sources != execution_sources(root):
        raise RuntimeError("execution source changed while fitting statistics")
    report = {"status": "passed", "source_sha256": sources, "cases": cases, "workers": workers,
              "wall_seconds": time.perf_counter()-started, "updates": 0, "interactions": 0}
    write_json(output / "generation.json", report)
    return report


def verify(root: Path, data: Path, statistics_dir: Path, output: Path) -> dict:
    import torch
    from world_models.data.replay import ReplayDataset
    from world_models.data.schema import STATE_ARRAYS, TRANSITION_ARRAYS
    from world_models.infrastructure.config import resolve_config
    from world_models.models.world_model import SamplingStreams, build_model
    from world_models.training.diagnostics import gradient_diagnostics
    from world_models.training.normalization import NormalizationStats
    from world_models.training.objective import ObjectiveConfig, WorldModelObjective
    from world_models.training.objective_batch import objective_inputs

    output.mkdir(parents=True, exist_ok=False)
    sources = execution_sources(root)
    started = time.perf_counter()
    report = {"schema_version": "p23-objective-v1", "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "P2.3 objectives, frozen initial fitting statistics, and loss diagnostics; no model optimization or P2.5 quality gate",
              "source_sha256": sources, "model_updates": 0, "new_environment_interactions": 0,
              "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                          "cpu_threads": torch.get_num_threads(), "cuda_tested": False}, "cases": []}
    try:
        generation_path = statistics_dir / "generation.json"
        receipt = generation_path.with_suffix(".sha256")
        parts = receipt.read_text(encoding="ascii").strip().split(maxsplit=1)
        if len(parts) != 2 or parts[0] != sha256(generation_path) or parts[1] != generation_path.name:
            raise ValueError("statistics generation receipt is missing or does not match")
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if generation["status"] != "passed" or len(generation["cases"]) != 4:
            raise ValueError("statistics generation must contain four passed pilot cases")
        for path, digest in generation["source_sha256"].items():
            if sha256(root / path) != digest:
                raise ValueError(f"statistics implementation changed: {path}")
        write_json(output / "statistics_generation.json", generation)
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
            ("tests.test_model_loss_primitives", "tests.test_model_normalization", "tests.test_model_objective_diagnostics",
             "tests.test_model_objective", "tests.test_model_objective_accumulation"))
        with (output / "objective_tests.txt").open("w", encoding="utf-8") as stream:
            tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report["unit_tests"] = {"run": tests.testsRun, "passed": tests.wasSuccessful(), "errors": len(tests.errors),
                                "failures": len(tests.failures), "skipped": len(tests.skipped)}
        if not tests.wasSuccessful():
            raise AssertionError("P2.3 objective tests failed")
        bindings = []
        for family, seed in CASES:
            dataset = ReplayDataset(data / family / f"seed_{seed}" / "replay", verify=False)
            before = dataset.manifest_sha256
            source = statistics_dir / family / f"seed_{seed}.json"
            entry = next(item for item in generation["cases"] if (item["family"], item["seed"]) == (family, seed))
            if sha256(source) != entry["sha256"]:
                raise ValueError("normalization source differs from its generation receipt")
            stats = NormalizationStats.load(source)
            stats.validate_for_dataset(dataset)
            target = output / "normalization" / family / f"seed_{seed}.json"
            stats.save(target)
            if sha256(target) != sha256(source):
                raise ValueError("statistics serialization changed during verified copy")
            cfg = resolve_config(family, "B", "pilot", root / "configs").settings
            model = build_model(cfg, master_seed=seed, variant="B")
            objective = WorldModelObjective(model.task, stats, ObjectiveConfig.from_experiment(cfg))
            fit = [episode for episode in dataset.episodes if episode["partition"] == "fit"]
            selected = {fit[0]["episode_id"]: fit[0]}
            for reason in sorted({episode["end_reason"] for episode in fit}):
                episode = next(episode for episode in fit if episode["end_reason"] == reason)
                selected[episode["episode_id"]] = episode
            checked = []
            model_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            windows = [(episode, max(0, episode["valid_length"]-3), episode["valid_length"], "episode_end")
                       for episode in selected.values()]
            # Exercise actual positive milestone labels as well as terminal
            # labels. This is an objective/alignment diagnostic, not reweighting.
            success = next((episode for episode in selected.values() if episode["end_reason"] == "success"), None)
            if success is not None:
                import numpy as np
                success_arrays = dataset.load_episode(success["episode_id"], purpose="world_model_fit").arrays()
                events = dataset.manifest["schema"]["event_fields"]
                milestone_names = ("latch_activations",) if family == "relay" else ("console_1_completions", "console_2_completions")
                for name in milestone_names:
                    indices = np.flatnonzero(success_arrays["event_deltas"][:, events.index(name)] > 0)
                    if not indices.size:
                        raise AssertionError("successful fitting fixture lacks its milestone event")
                    end = int(indices[0])+1
                    windows.append((success, max(0, end-3), end, name))
            for episode, start, length, window_kind in windows:
                arrays = dataset.load_episode(episode["episode_id"], purpose="world_model_fit").arrays()
                streams = SamplingStreams.from_seed(200+seed)
                # Reconstruct the entire real prefix, without retaining a graph.
                # This is an interface/gradient check, not a P2.4 burn-in choice.
                with torch.no_grad():
                    state = model.initialize(torch.tensor(arrays["rgb"][0:1]), streams=streams)
                    for index in range(start):
                        prior = model.core.prior_step(state, torch.tensor(arrays["actions"][index:index+1]).long(), generator=streams.prior)
                        if arrays["image_valid"][index+1].all():
                            state = model.core.observe(prior, torch.tensor(arrays["rgb"][index+1:index+2]).float()/255, generator=streams.posterior)
                        else:
                            state = prior
                initial = model.reconstruct(state.detach())
                predictions = []
                for index in range(start, length):
                    step = model.observe_step(state, torch.tensor(arrays["actions"][index:index+1]).long(),
                        torch.tensor(arrays["rgb"][index+1:index+2]), streams=streams,
                        next_image_valid=torch.tensor(arrays["image_valid"][index+1:index+2]))
                    predictions.append(step)
                    state = step.state
                states = {name: torch.tensor(arrays[name][start:length+1]).unsqueeze(1) for name in STATE_ARRAYS}
                # Only numeric arrays needed by the adapters; replay IDs stay metadata.
                names = set(TRANSITION_ARRAYS)-{"episode_id", "collector_policy_version"}
                transitions = {name: torch.tensor(arrays[name][start:length]).unsqueeze(1) for name in names}
                state_input, transition_input = objective_inputs(initial, predictions, states, transitions, model.task,
                    event_fields=dataset.manifest["schema"]["event_fields"])
                model.zero_grad(set_to_none=True)
                result = objective(state_input, transition_input)
                result.total.backward()
                gradients = gradient_diagnostics(model)
                if not gradients["global_l2"] or not torch.isfinite(result.total):
                    raise AssertionError("real-data objective/gradient check failed")
                positive_bonus = int(torch.count_nonzero(transition_input.targets.bonus[transition_input.targets.bonus_valid] > 1e-6))
                if window_kind != "episode_end" and positive_bonus == 0:
                    raise AssertionError("milestone fixture does not carry its real bonus label")
                checked.append({"episode_id": episode["episode_id"], "partition": "fit", "episode_end_reason": episode["end_reason"],
                                "window_kind": window_kind, "positive_bonus_labels": positive_bonus,
                                "complete_prefix_decisions": start, "loss_window_transitions": length-start,
                                "loss_window_states": length-start+1, "objective": result.metrics(), "gradients": gradients})
            if any(not torch.equal(parameter, model_before[name]) for name, parameter in model.named_parameters()):
                raise AssertionError("verification unexpectedly changed model weights")
            if dataset.manifest_sha256 != before:
                raise AssertionError("replay manifest changed")
            normalizer_path = target.relative_to(output).as_posix()
            bindings.append({"task_family": family, "seed": seed, "B": normalizer_path, "D": normalizer_path,
                             "normalization_sha256": sha256(target), "dataset_identity_sha256": dataset.manifest["dataset_identity_sha256"]})
            report["cases"].append({"task_family": family, "seed": seed, "normalization_path": normalizer_path,
                "normalization_sha256": sha256(target), "normalization_identity": stats.as_dict()["identity_sha256"],
                "fitting_transitions": stats.provenance["fitting_transition_count"], "unique_fitting_states": stats.provenance["unique_state_count"],
                "fitting_episodes": stats.provenance["fitting_episode_count"], "excluded_diagnostic_episodes": stats.provenance["excluded_model_diagnostic_episode_count"],
                "constant_fields": [name for name, item in stats.fields.items() if item.constant],
                "unavailable_fields": [name for name, item in stats.fields.items() if not item.available],
                "label_profile": stats.provenance["label_profile"], "dataset_manifest_sha256": before,
                "dataset_manifest_unchanged": True, "model_weights_unchanged": True, "checks": checked})
            print(json.dumps({"objective_verified": family, "seed": seed, "fit_episodes_checked": len(selected), "windows_checked": len(checked)}), flush=True)
        write_json(output / "normalization_bindings.json", {"schema_version": "paired-initial-normalization-v1", "bindings": bindings})
        write_json(output / "normalization" / "generation.json", generation)
        if sources != execution_sources(root):
            raise RuntimeError("source changed during P2.3 verification")
        report["status"] = "complete_p23"
        report["source_unchanged_during_verification"] = True
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["wall_seconds"] = time.perf_counter()-started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "phase23_acceptance.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stats", "verify", "complete"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/initial_replay/pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase23"))
    parser.add_argument("--statistics-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    import psutil
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:4])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    data, output = args.data_dir.resolve(), args.output_dir.resolve()
    try:
        if args.command == "stats":
            report = fit_statistics(root, data, output, workers=args.workers)
        elif args.command == "verify":
            if args.statistics_dir is None:
                parser.error("verify requires --statistics-dir; complete fits new statistics")
            report = verify(root, data, args.statistics_dir.resolve(), output)
        else:
            if output.exists():
                raise FileExistsError("use a fresh output directory to preserve reports")
            statistics = output.with_name(output.name+"_statistics")
            fit_statistics(root, data, statistics, workers=args.workers)
            report = verify(root, data, statistics, output)
    except Exception as error:
        print(f"P2.3 failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
