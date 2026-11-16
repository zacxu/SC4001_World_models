"""Freeze existing initial checkpoints for an explicitly shortened pilot.

This selection is separate evidence: it never changes checkpoint bytes, the
registered fitting target, or a training completion/quality acceptance record.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PureWindowsPath
import re

from world_models.data.replay import canonical_hash, sha256_file
from world_models.infrastructure.config import resolve_config, validate_pair
from world_models.infrastructure.reproducibility import derive_seed
from . import runner
from .checkpoints import read_checkpoint
from .fitting import FitSettings


SCHEMA_VERSION = "p24-initial-selection-v1"
SELECTION_RULE = "latest_committed_per_case"
_NUMBERED = re.compile(r"update_([0-9]{8})\.pt\Z")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(path, directory):
    return Path(os.path.relpath(Path(path).resolve(), directory)).as_posix()


def _resolve(directory, value):
    if (not isinstance(value, str) or not value or "\\" in value
            or Path(value).is_absolute() or PureWindowsPath(value).drive):
        raise ValueError("selection paths must be portable relative paths")
    return (directory / value).resolve()


def _portable_config(value):
    """Compare historical config hash keys across Linux/Windows separators.

    ``ResolvedConfig.as_dict`` records platform-native relative filenames.
    Normalize those names only; all hashes, settings and derived values remain
    exact, and ambiguous duplicate normalized names are rejected.
    """
    result = deepcopy(value)
    hashes = result.get("source_hashes")
    if not isinstance(hashes, dict) or any(not isinstance(key, str) for key in hashes):
        raise ValueError("resolved configuration source hashes are malformed")
    normalized = {key.replace("\\", "/"): digest for key, digest in hashes.items()}
    if len(normalized) != len(hashes):
        raise ValueError("resolved configuration contains ambiguous source hash paths")
    result["source_hashes"] = normalized
    return result


def _plan(root):
    plan = _json(root / "launch_plan.json")
    if (plan.get("schema_version") != "p24-launch-plan-v1" or plan.get("stage") != "initial"
            or plan.get("worker_count") != 4 or not isinstance(plan.get("jobs"), list)
            or [(row.get("task"), row.get("seed")) for row in plan["jobs"]] != list(runner.CASES)):
        raise ValueError("selection requires the complete four-case initial launch plan")
    FitSettings(**plan["settings"])
    return plan


def _latest(case):
    candidates = [path for path in (case / "checkpoints").glob("update_*.pt")
                  if _NUMBERED.fullmatch(path.name) and path.with_suffix(".sha256").is_file()]
    if not candidates:
        raise ValueError(f"no committed numbered checkpoint: {case}")
    # Validate exactly the latest committed pair. A corrupt pair never falls
    # back to a less recent checkpoint; orphan payload/temp files are ignored.
    return max(candidates, key=lambda path: int(_NUMBERED.fullmatch(path.name)[1]))


def _inspect(root, checkpoint, job, plan):
    task, seed = job["task"], job["seed"]
    case = root / task / f"seed_{seed}"
    if checkpoint.parent != (case / "checkpoints").resolve():
        raise ValueError("selected checkpoint is outside its recorded task/seed run")
    payload = read_checkpoint(checkpoint)
    metadata = payload["metadata"]
    manifest_path = case / "run_manifest.json"
    manifest = _json(manifest_path)
    left = resolve_config(task, "B", "pilot", runner.ROOT / "configs")
    right = resolve_config(task, "D", "pilot", runner.ROOT / "configs")
    validate_pair(left, right)
    common = deepcopy(left.settings)
    common.pop("variant")
    target = job["target_stage_updates"]
    expected = {"task_family": task, "run_seed": seed, "profile": "pilot", "stage": "initial",
                "variant": "shared_initial:B,D", "source_sha256": runner.source_identity(),
                "common_config_sha256": canonical_hash(common), "target_stage_updates": target,
                "settings": plan["settings"], "precision": plan["settings"]["precision"],
                "fitting_seed": derive_seed(task, seed, "world_model_fitting_initial", "B"),
                **job["data"]}
    if (type(target) is not int or target != left.settings["world_model"]["initial_updates"]
            or type(payload["update"]) is not int or not 0 < payload["update"] <= target
            or checkpoint.name != f"update_{payload['update']:08d}.pt"
            or manifest.get("schema_version") != "p24-run-v1"
            or manifest.get("metadata") != metadata
            or _portable_config(manifest.get("resolved_config", {})) != _portable_config(left.as_dict())
            or any(metadata.get(key) != value for key, value in expected.items())):
        raise ValueError(f"selected checkpoint configuration, provenance, source or update mismatch: {case}")
    if (len(metadata["replays"]) != 1
            or metadata["replays"][0]["transitions"] != left.settings["experiment"]["initial_transitions"]):
        raise ValueError("initial selection must retain the shared initial replay budget")
    return {"task": task, "seed": seed, "checkpoint_path": checkpoint,
            "checkpoint_sha256": sha256_file(checkpoint), "model_updates": payload["update"],
            "original_target_stage_updates": target, "run_manifest_path": manifest_path,
            "run_manifest_sha256": sha256_file(manifest_path),
            "common_config_sha256": metadata["common_config_sha256"],
            "source_sha256": metadata["source_sha256"],
            "initial_normalization": metadata["initial_normalization"], "replays": metadata["replays"],
            "shared_initial_variants": ["B", "D"]}


def _portable(row, directory):
    return {**row, **{key: _relative(row[key], directory)
                     for key in ("checkpoint_path", "run_manifest_path")}}


def freeze_selection(run_root, output_dir, *, reason, data_root=None, statistics_root=None,
                     config_root=None):
    """Select latest saved states under exclusive source locks, without fitting.

    Stop the remote training job and finish synchronizing its files first. Local
    OS locks detect local writers; they cannot detect a writer on another host.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("an explicit shortened-pilot reason is required")
    root, output = Path(run_root).resolve(), Path(output_dir).resolve()
    if not root.is_dir():
        raise ValueError("source run root does not exist")
    if output.exists():
        raise FileExistsError("selection is immutable; choose a fresh output directory")
    if output == root or root in output.parents:
        raise ValueError("selection output must be separate from the source run")
    for task, seed in runner.CASES:
        if not (root / task / f"seed_{seed}").is_dir():
            raise ValueError("selection requires all four existing case directories")
    with ExitStack() as stack:
        stack.enter_context(runner.run_lock(root))
        for task, seed in runner.CASES:
            stack.enter_context(runner.run_lock(root / task / f"seed_{seed}"))
        plan = _plan(root)
        rows = []
        for job in plan["jobs"]:
            task, seed = job["task"], job["seed"]
            cfg, _, _, provenance = runner.load_case(task, seed,
                data_root=data_root or runner.ROOT / "data/initial_replay/pilot",
                statistics_root=statistics_root or runner.ROOT / "artifacts/phase23/normalization",
                config_root=config_root, profile="pilot")
            if job["data"] != provenance:
                raise ValueError("current replay/normalization differs from the saved initial training data")
            if _portable_config(cfg.as_dict()) != _portable_config(
                    resolve_config(task, "B", "pilot", runner.ROOT / "configs").as_dict()):
                raise ValueError("selection requires the unchanged current study configuration")
            rows.append(_portable(_inspect(root, _latest(root / task / f"seed_{seed}"),
                                           job, plan), output))
        manifest = {"schema_version": SCHEMA_VERSION,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(), "reason": reason.strip(),
                    "user_authorized_shortened_pilot": True, "selection_rule": SELECTION_RULE,
                    "source_run_root": _relative(root, output),
                    "source_launch_plan_sha256": sha256_file(root / "launch_plan.json"),
                    "cases": rows, "full_schedule_completion_claim": False,
                    "P2.5_quality_acceptance": False, "automatic_phase3_launch": False,
                    "optimizer_updates_executed": 0, "new_environment_interactions": 0,
                    "scope": "shared B/D initial checkpoints; shortened exploratory pilot"}
        output.mkdir(parents=True, exist_ok=False)
        path = output / "selection.json"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        with path.with_suffix(".sha256").open("x", encoding="ascii", newline="\n") as stream:
            stream.write(f"{sha256_file(path)}  {path.name}\n")
            stream.flush()
            os.fsync(stream.fileno())
        return manifest


def load_selection(path):
    """Verify the frozen selection and its original four committed checkpoints.

    Newer source checkpoints do not change an already frozen selection. Paths
    remain relative to this JSON file's directory for copying the whole project.
    """
    path = Path(path).resolve()
    receipt = path.with_suffix(".sha256")
    if not path.is_file() or not receipt.is_file():
        raise ValueError("selection JSON and SHA256 receipt are both required")
    parts = receipt.read_text(encoding="ascii").strip().split(maxsplit=1)
    if len(parts) != 2 or parts != [sha256_file(path), path.name]:
        raise ValueError("selection SHA256 receipt mismatch")
    manifest = _json(path)
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("selection_rule") != SELECTION_RULE
            or manifest.get("user_authorized_shortened_pilot") is not True
            or manifest.get("full_schedule_completion_claim") is not False
            or manifest.get("P2.5_quality_acceptance") is not False
            or manifest.get("automatic_phase3_launch") is not False
            or manifest.get("optimizer_updates_executed") != 0
            or manifest.get("new_environment_interactions") != 0
            or not isinstance(manifest.get("reason"), str) or not manifest["reason"].strip()
            or not isinstance(manifest.get("cases"), list)
            or [(row.get("task"), row.get("seed")) for row in manifest["cases"]] != list(runner.CASES)):
        raise ValueError("invalid shortened initial checkpoint selection")
    root = _resolve(path.parent, manifest["source_run_root"])
    if sha256_file(root / "launch_plan.json") != manifest["source_launch_plan_sha256"]:
        raise ValueError("source launch plan changed after selection")
    plan = _plan(root)
    for row, job in zip(manifest["cases"], plan["jobs"]):
        checkpoint = _resolve(path.parent, row["checkpoint_path"])
        expected = _portable(_inspect(root, checkpoint, job, plan), path.parent)
        if row != expected:
            raise ValueError(f"frozen selection/checkpoint identity mismatch: {row['task']}/seed_{row['seed']}")
    return manifest


def resolve_selected_case(selection_path, task, seed, checkpoint=None):
    """Return validated selected case plus absolute paths and selection identity."""
    path = Path(selection_path).resolve()
    manifest = load_selection(path)
    for row in manifest["cases"]:
        if (row["task"], row["seed"]) == (task, seed):
            selected = _resolve(path.parent, row["checkpoint_path"])
            if checkpoint is not None and Path(checkpoint).resolve() != selected:
                raise ValueError("checkpoint is not the frozen selection for this task and seed")
            return {**row, "checkpoint_path": str(selected),
                    "run_manifest_path": str(_resolve(path.parent, row["run_manifest_path"])),
                    "selection_path": str(path), "selection_sha256": sha256_file(path)}
    raise ValueError("task/seed is outside the four-case shortened selection")
