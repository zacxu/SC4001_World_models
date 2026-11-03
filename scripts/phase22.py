"""Reproducible P2.2 architecture verification, without fitting or engine calls."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import unittest


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    path.with_suffix(".sha256").write_text(f"{file_hash(path)}  {path.name}\n", encoding="ascii")


def verify(root: Path, output: Path, data_dir: Path | None, *, tests: bool = True) -> dict:
    import torch
    from world_models.infrastructure.config import resolve_config
    from world_models.models.core import categorical_probs
    from world_models.models.heads import encode_auxiliary_targets, encode_transition_targets
    from world_models.models.world_model import SamplingStreams, build_model

    if output.exists():
        raise FileExistsError("use a fresh output directory; previous verification evidence is immutable")
    output.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source_paths = [*sorted((root / "src/world_models/models").glob("*.py")),
                    *sorted((root / "tests").glob("test_model_*.py")), root / "tests/test_world_model.py",
                    root / "scripts/phase22.py", root / "configs/base.yaml", root / "configs/tasks/relay.yaml",
                    root / "configs/tasks/cover.yaml", root / "pyproject.toml", root / "requirements-phase2-cpu.lock",
                    root / "src/world_models/contracts.py", root / "src/world_models/data/schema.py",
                    root / "src/world_models/env/targets.py", root / "src/world_models/infrastructure/config.py",
                    root / "src/world_models/infrastructure/reproducibility.py"]
    source_hashes = {path.relative_to(root).as_posix(): file_hash(path) for path in source_paths}
    report = {"schema_version": "p22-architecture-v1", "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hashes,
              "scope": "P2.2 architecture and interfaces only; no fitting, policy learning, or Phase 2 quality gate",
              "runtime": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
                          "device": "cpu", "intraop_threads": 1, "interop_threads": 1,
                          "cuda_available": torch.cuda.is_available(), "cuda_tested": False,
                          "packages": {name: importlib.metadata.version(name) for name in
                                       ("numpy", "PyYAML", "torch", "filelock", "fsspec", "Jinja2", "MarkupSafe", "mpmath", "networkx", "sympy")}},
              "training_updates": 0, "new_environment_interactions": 0, "tasks": []}
    try:
        if tests:
            suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name)
                                       for name in ("tests.test_model_core", "tests.test_model_heads", "tests.test_world_model"))
            with (output / "architecture_tests.txt").open("w", encoding="utf-8") as stream:
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            report["unit_tests"] = {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                                    "skipped": len(result.skipped), "passed": result.wasSuccessful()}
            if not result.wasSuccessful():
                raise AssertionError("architecture unit tests failed; see architecture_tests.txt")
        for family in ("relay", "cover"):
            config_b = resolve_config(family, "B", "pilot", root / "configs")
            config_d = resolve_config(family, "D", "pilot", root / "configs")
            model = build_model(config_b.settings, master_seed=0, variant="B").eval()
            paired = build_model(config_d.settings, master_seed=0, variant="D").eval()
            assert all(torch.equal(value, paired.state_dict()[key]) for key, value in model.state_dict().items())
            del paired
            streams = SamplingStreams.from_seed(91)
            observations = torch.randint(0, 256, (2, 2, 3, 96, 128), dtype=torch.uint8,
                                         generator=torch.Generator().manual_seed(13))
            with torch.no_grad():
                initial = model.initialize(observations, streams=streams)
                reconstructed = model.reconstruct(initial)
                actions = torch.tensor([[1, 8], [10, 3]])
                observed = model.observe_step(initial, actions, observations.flip(-1), streams=streams)
                assert categorical_probs(initial.logits).dtype == torch.float32
                assert observed.posterior_valid.all()
                # Structural stress fixture only. No weights/checkpoint or result
                # from this deliberately forced outcome head are used for research.
                model.transition_heads.outcome_head.weight.zero_()
                model.transition_heads.outcome_head.bias.copy_(torch.tensor([100., -100., -100.]))
                rng = streams.state_dict()
                rollout_actions = actions.unsqueeze(0).expand(15, -1, -1)
                rollout = model.imagine(observed.state, rollout_actions, streams=streams)
                streams.load_state_dict(rng)
                repeated = model.imagine(observed.state, rollout_actions, streams=streams)
                assert len(rollout) == 15
                assert all(torch.equal(a.transition.rgb, b.transition.rgb) for a, b in zip(rollout, repeated))
                near_deadline = replace(initial, decision_index=torch.tensor([model.task.deadline-1, model.task.deadline-3]))
                deadline_rollout = model.imagine(near_deadline, rollout_actions, streams=streams)
                assert [step.valid.tolist() for step in deadline_rollout] == [[True, True], [False, True], [False, True]]
            task_report = {"family": family, "config": asdict(model.config), "task": asdict(model.task),
                           "paired_B_D_weights_identical": True,
                           "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                           "image_tokens": [2, 48, 256], "memory": list(initial.memory.shape),
                           "categorical_latent": list(initial.stochastic.shape),
                           "shared_representation": list(reconstructed.representation.shape),
                           "decoded_rgb": list(reconstructed.rgb.shape),
                           "auxiliary_fields": [asdict(field) for field in model.auxiliary_heads.fields],
                           "fixed_action_rollout_steps": len(rollout), "rng_resume_identical": True,
                           "deadline_mask_trace": [step.valid.tolist() for step in deadline_rollout],
                           "rollout_fixture": "untrained model with continue-only outcome head, structural verification only",
                           "replay_checks": []}
            # Fresh unmodified initialization for the real replay interface check.
            model = build_model(config_b.settings, master_seed=0, variant="B").eval()
            if data_dir is not None:
                from world_models.data.replay import ReplayDataset
                for seed in (0, 1):
                    path = data_dir / family / f"seed_{seed}" / "replay"
                    dataset = ReplayDataset(path, verify=False)
                    manifest_before = dataset.manifest_sha256
                    candidates = [episode for episode in dataset.episodes if episode["partition"] == "fit"]
                    if not candidates:
                        raise ValueError("replay has no fitting episodes")
                    selected = {candidates[0]["episode_id"]: candidates[0]}
                    for reason in sorted({episode["end_reason"] for episode in candidates}):
                        representative = next(episode for episode in candidates if episode["end_reason"] == reason)
                        selected[representative["episode_id"]] = representative
                    episodes = []
                    for episode in selected.values():
                        arrays = dataset.load_episode(episode["episode_id"], purpose="world_model_fit").arrays()
                        labels = encode_transition_targets(arrays, model.task)
                        auxiliary = encode_auxiliary_targets(arrays, arrays, model.task, dataset.manifest["schema"]["event_fields"])
                        assert len(auxiliary) == len(model.auxiliary_heads.fields)
                        assert labels.bonus_valid.all() and labels.outcome_valid.all()
                        # Only ordinary fitting prefixes enter the image model.
                        streams = SamplingStreams.from_seed(100+seed)
                        with torch.no_grad():
                            state = model.initialize(torch.tensor(arrays["rgb"][0:1]), streams=streams)
                            for index in range(min(3, episode["valid_length"])):
                                transition = model.observe_step(state, torch.tensor(arrays["actions"][index:index+1]).long(),
                                    torch.tensor(arrays["rgb"][index+1:index+2]), streams=streams,
                                    next_image_valid=torch.tensor(arrays["image_valid"][index+1:index+2]))
                                state = transition.state
                            assert state.decision_index.item() == min(3, episode["valid_length"])
                        episodes.append({"episode_id": episode["episode_id"], "partition": episode["partition"],
                                         "end_reason": episode["end_reason"], "transitions_checked": episode["valid_length"],
                                         "source_state_targets": int(arrays["rgb"].shape[0]),
                                         "valid_bonus_labels": int(labels.bonus_valid.sum()),
                                         "valid_outcome_labels": int(labels.outcome_valid.sum()),
                                         "auxiliary_fields_checked": len(auxiliary)})
                    assert manifest_before == dataset.manifest_sha256
                    task_report["replay_checks"].append({"seed": seed, "dataset_manifest_sha256": manifest_before,
                        "dataset_identity_sha256": dataset.manifest["dataset_identity_sha256"],
                        "immutable_manifest_unchanged": True, "chunk_hashes_verified_on_read": True, "episodes": episodes})
            report["tasks"].append(task_report)
        assert all(file_hash(root / path) == digest for path, digest in source_hashes.items())
        report["status"] = "complete_p22" if data_dir is not None and tests else "passed_architecture_smoke_only"
        report["source_unchanged_during_verification"] = True
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["wall_seconds"] = time.perf_counter()-started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "phase22_acceptance.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase22"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/initial_replay/pilot"))
    parser.add_argument("--no-replay", action="store_true", help="synthetic architecture smoke only; cannot mark P2.2 complete")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    data = None if args.no_replay else args.data_dir.resolve()
    try:
        report = verify(root, args.output_dir.resolve(), data)
    except Exception as error:
        print(f"P2.2 verification failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "report": str(args.output_dir / "phase22_acceptance.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
