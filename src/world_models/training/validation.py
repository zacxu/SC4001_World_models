"""Immutable real validation-layout episodes, kept outside all learning replay.

The native collector is an explicit user-run command. Loading or sampling a
bank never starts ViZDoom, updates normalization, or makes fitting anchors.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from world_models.data.replay import canonical_hash, read_npz, sha256_file, write_npz
from world_models.data.schema import (REASONS, STATE_ARRAYS, TRANSITION_ARRAYS, array_specs,
                                     event_field_names, extract_state, extract_transition, field_manifest)
from world_models.env.coop import ResetResult, StepResult
from world_models.env.splits import get_template, list_templates
from .sampling import ReplayWindow


VERSION = "model-validation-bank-v1"
ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = ("src/world_models/training/validation.py", "src/world_models/data/schema.py",
                "src/world_models/env/coop.py", "src/world_models/env/splits.py",
                "src/world_models/data/collection.py", "src/world_models/env/scripted.py")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _member(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("validation member must be a relative POSIX path")
    value = (root / name).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError("validation member escapes the bank")
    return value


def _write_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _write_receipt(path)


def _write_receipt(path: Path) -> None:
    with path.with_suffix(".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{sha256_file(path)}  {path.name}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _check_receipt(path: Path) -> None:
    receipt = path.with_suffix(".sha256")
    if not path.is_file() or not receipt.is_file():
        raise ValueError(f"validation payload and SHA receipt are required: {path.name}")
    parts = receipt.read_text(encoding="ascii").strip().split(maxsplit=1)
    if len(parts) != 2 or parts != [sha256_file(path), path.name]:
        raise ValueError(f"validation SHA receipt does not match: {path.name}")


def _scenario_provenance(scenario: dict, family: str) -> dict:
    provenance = scenario.get("split_provenance", {})
    if (scenario.get("task_family") != family or provenance.get("split") != "validation"
            or provenance.get("purpose") != "validation" or provenance.get("learning_eligible") is not False):
        raise ValueError("model validation requires validation geometry, never train/test/synthetic replay")
    _, _, expected = get_template(family, provenance.get("template_id"), split="validation", purpose="validation")
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("validation geometry provenance differs from the versioned split")
    if not isinstance(scenario.get("hashes", {}).get("wad_sha256"), str):
        raise ValueError("validation scenario needs its compiled WAD hash")
    return deepcopy(provenance)


def _validate_arrays(arrays: dict, episode: dict, events: tuple[str, ...], task: dict) -> None:
    length = _integer(episode["valid_length"], "episode length", 1)
    specifications = array_specs(events)
    if set(arrays) != set(specifications):
        raise ValueError("validation episode arrays differ from the typed real-data schema")
    for name, specification in specifications.items():
        array = arrays[name]
        count = length + 1 if name in STATE_ARRAYS else length
        if array.dtype.str != specification["dtype"] or array.shape != (count, *specification["row_shape"]):
            raise ValueError(f"validation array shape/dtype differs: {name}")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"nonfinite validation values: {name}")
    indices = np.arange(length)
    for name in ("decision_index", "observation_index"):
        if not np.array_equal(arrays[name], indices):
            raise ValueError("validation action/state indexing is not episode-relative")
    if (not np.array_equal(arrays["next_observation_index"], indices + 1)
            or not np.array_equal(arrays["transition_id"], indices + episode["transition_id_start"])
            or not np.all(arrays["episode_id"] == episode["episode_id"])
            or np.any(arrays["actions"] >= 12)):
        raise ValueError("validation episode identities/actions are inconsistent")
    if not np.all(arrays["collector_policy_version"] == episode["policy_version"]):
        raise ValueError("validation collector policy changed inside an episode")
    if not arrays["image_valid"][0].all() or not arrays["image_valid"][:-1].all():
        raise ValueError("validation reset/nonterminal source observations must be present")
    if not np.array_equal(arrays["next_obs_valid"], arrays["image_valid"][1:]):
        raise ValueError("validation destination image masks disagree")
    if (not np.array_equal(arrays["start_tic"], arrays["state_engine_tic"][:-1])
            or not np.array_equal(arrays["end_tic"], arrays["state_engine_tic"][1:])
            or not np.array_equal(arrays["actual_tics"], arrays["end_tic"] - arrays["start_tic"])
            or np.any(arrays["actual_tics"] == 0) or np.any(arrays["actual_tics"] > episode["action_repeat"])):
        raise ValueError("validation simulator tics do not reconcile")
    if (not np.array_equal(np.diff(arrays["cumulative_event_counters"], axis=0), arrays["event_deltas"])
            or np.any(arrays["event_deltas"] < 0)):
        raise ValueError("validation native cumulative events/deltas disagree")
    if np.any(arrays["termination_reason"] >= len(REASONS)):
        raise ValueError("validation boundary reason is invalid")
    reasons = [REASONS[int(value)] for value in arrays["termination_reason"]]
    terminal = np.array([reason in ("success", "death", "failure", "deadline") for reason in reasons])
    cutoff = np.array([reason == "collection_cutoff" for reason in reasons])
    if (any(reason != "continue" for reason in reasons[:-1]) or not (terminal[-1] or cutoff[-1])
            or not np.array_equal(arrays["terminated"], terminal)
            or not np.array_equal(arrays["truncated"], cutoff) or reasons[-1] != episode["end_reason"]
            or np.any((arrays["actual_tics"] != episode["action_repeat"]) & ~terminal)
            or length > task["deadline"]):
        raise ValueError("validation sequence crosses or mislabels an episode boundary")
    if reasons[-1] == "deadline" and length != task["deadline"]:
        raise ValueError("validation deadline does not agree with its absolute clock")
    milestones = ("latch_activations",) if episode["task_family"] == "relay" else (
        "console_1_completions", "console_2_completions")
    bonus = sum(arrays["event_deltas"][:, events.index(name)] for name in milestones) * task["milestone_bonus"]
    final = np.array([task["success_reward"] if reason == "success" else
                      task["failure_reward"] if reason in ("death", "failure", "deadline") else 0.
                      for reason in reasons])
    if not np.allclose(arrays["team_reward"], task["step_cost"] + bonus + final, atol=1e-9, rtol=0):
        raise ValueError("validation team rewards disagree with authoritative events/boundaries")


def _native_evidence(root: Path, records: list[dict], arrays: dict, initial_dispatch: int) -> None:
    """Bind each stored action/tic to retained native dispatch and ACK records."""
    journals = []
    for record in records:
        member = _member(root, record["path"])
        if not member.is_file() or member.stat().st_size != record["bytes"] or sha256_file(member) != record["sha256"]:
            raise ValueError("validation native evidence differs from its receipt")
        if member.name == "dispatch_ledger.jsonl":
            journals.append(member)
    if len(journals) != 1:
        raise ValueError("validation episode needs exactly one native dispatch journal")
    dispatches, acknowledgements = {}, {}
    for line in journals[0].read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        target = dispatches if record.get("kind") == "dispatch" else acknowledgements if record.get("kind") == "acknowledged" else None
        if target is not None:
            identifier = record.get("dispatch_id")
            if type(identifier) is not int or identifier in target:
                raise ValueError("validation native dispatch/ACK identity is invalid")
            target[identifier] = record
    current = _integer(initial_dispatch, "initial native dispatch")
    for index, actions in enumerate(arrays["actions"]):
        for repeat in range(int(arrays["actual_tics"][index])):
            current += 1
            dispatch, acknowledgement = dispatches.get(current, {}), acknowledgements.get(current, {})
            tic = int(arrays["start_tic"][index]) + repeat + 1
            if (dispatch.get("command") != "step" or dispatch.get("actions") != actions.tolist()
                    or acknowledgement.get("command") != "step" or acknowledgement.get("tics") != [tic, tic]):
                raise ValueError("validation stored actions/tics lack matching native dispatch/ACK evidence")


class ValidationBankWriter:
    """Write one new bank from published native results; no raw-array import API."""

    def __init__(self, directory: str | Path, *, task_family: str, run_seed: int,
                 configuration: dict, data_source: str = "real_environment"):
        if data_source != "real_environment":
            raise ValueError("synthetic data cannot enter a real model-validation bank")
        if task_family not in ("relay", "cover"):
            raise ValueError("unknown validation task family")
        _integer(run_seed, "run seed")
        if configuration.get("task", {}).get("name") != task_family:
            raise ValueError("validation task configuration differs")
        self.path = Path(directory).resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "episodes").mkdir()
        self.task_family, self.run_seed = task_family, run_seed
        shared = deepcopy(configuration)
        shared.pop("variant", None)
        self.configuration = shared
        self._episodes, self._events, self._transitions = [], None, 0
        self._sealed = False
        self._source_hashes = {name: sha256_file(ROOT / name) for name in SOURCE_PATHS}

    def append_episode(self, episode_id: str, reset: ResetResult, steps: Iterable[StepResult], *,
                       collector_kind: str, policy_version: str, engine_version: str,
                       native_evidence: list[dict], data_source: str = "real_environment") -> dict:
        if self._sealed:
            raise ValueError("validation bank is sealed")
        if (data_source != "real_environment" or not isinstance(reset, ResetResult)
                or collector_kind not in ("random", "scripted") or not isinstance(policy_version, str)
                or not policy_version or not isinstance(engine_version, str) or not engine_version):
            raise ValueError("validation accepts only published real native random/scripted episodes")
        if (not isinstance(episode_id, str) or not episode_id or len(episode_id) > 128
                or any(episode["episode_id"] == episode_id for episode in self._episodes)):
            raise ValueError("validation episode ID must be unique and at most 128 characters")
        scenario = reset.training_only_info["scenario_manifest"]
        provenance = _scenario_provenance(scenario, self.task_family)
        if (scenario.get("deadline") != self.configuration["task"]["deadline"]
                or scenario.get("frame_skip") != self.configuration["environment"]["frame_skip"]):
            raise ValueError("validation scenario reward clock/action repeat differs from configuration")
        events = event_field_names(reset)
        if self._events is not None and self._events != events:
            raise ValueError("validation event schema changed")
        self._events = events
        if reset.training_only_info.get("decision_index") != 0:
            raise ValueError("validation reset clock must be zero")
        if not native_evidence or not any(Path(record["path"]).name == "dispatch_ledger.jsonl" for record in native_evidence):
            raise ValueError("validation native dispatch journal evidence is required")
        for record in native_evidence:
            member = _member(self.path, record["path"])
            if not member.is_file() or member.stat().st_size != record["bytes"] or sha256_file(member) != record["sha256"]:
                raise ValueError("validation native evidence does not match its receipt")
        parts = {name: [] for name in (*STATE_ARRAYS, *TRANSITION_ARRAYS)}
        state = extract_state(reset, events)
        for name in STATE_ARRAYS:
            parts[name].append(state[name])
        metadata = {"episode_id": episode_id, "task_family": self.task_family, "run_seed": self.run_seed,
                    "data_source": "real_environment", "purpose": "model_validation", "split": "validation",
                    "learning_eligible": False, "split_provenance": provenance,
                    "scenario_sha256": scenario["hashes"]["wad_sha256"], "scenario_manifest": deepcopy(scenario),
                    "collector_kind": collector_kind, "policy_version": policy_version,
                    "engine_version": engine_version, "action_repeat": scenario["frame_skip"],
                    "environment_seed": reset.training_only_info.get("engine_seed"),
                    "layout_seed": scenario.get("seed"),
                    "initial_dispatch_id": reset.training_only_info.get("dispatch_id"),
                    "transition_id_start": self._transitions, "native_evidence": deepcopy(native_evidence)}
        previous = reset.training_only_info
        ended = False
        length = 0
        for index, step in enumerate(steps):
            if ended or not isinstance(step, StepResult):
                raise ValueError("validation episode crosses a boundary or contains synthetic results")
            info = step.training_only_info
            if (info.get("scenario_manifest") != scenario or info.get("decision_index") != index + 1
                    or info.get("episode_id") != previous.get("episode_id")):
                raise ValueError("validation environment/scenario changed inside an episode")
            for key in ("consumed_joint_decisions", "dispatch_id"):
                if type(info.get(key)) is not int or type(previous.get(key)) is not int:
                    raise ValueError("validation results require native consumed-decision/dispatch counters")
            if (info["consumed_joint_decisions"] != previous["consumed_joint_decisions"] + 1
                    or info["dispatch_id"] - previous["dispatch_id"] != step.metadata.actual_simulator_tics):
                raise ValueError("validation native publication counters are inconsistent")
            row = extract_transition(step, events, transition_id=self._transitions + index,
                                     observation_index=index, episode_metadata=metadata)
            state = extract_state(step, events)
            for name in STATE_ARRAYS:
                parts[name].append(state[name])
            for name in TRANSITION_ARRAYS:
                parts[name].append(row[name])
            previous, ended, length = info, step.terminated or step.truncated, index + 1
        if not ended:
            raise ValueError("validation episode must finish at a true terminal or explicit collection cutoff")
        arrays = {name: np.stack(values) for name, values in parts.items()}
        metadata.update(valid_length=length, end_reason=REASONS[int(arrays["termination_reason"][-1])])
        _validate_arrays(arrays, metadata, events, self.configuration["task"])
        _native_evidence(self.path, native_evidence, arrays, metadata["initial_dispatch_id"])
        path = self.path / "episodes" / f"episode_{len(self._episodes):06d}.npz"
        if path.exists() or path.with_suffix(".sha256").exists():
            raise FileExistsError("validation episode storage is immutable")
        receipt = write_npz(path, arrays)
        _write_receipt(path)
        metadata["payload"] = {"path": path.relative_to(self.path).as_posix(), **receipt}
        self._episodes.append(metadata)
        self._transitions += length
        return deepcopy(metadata)

    def seal(self, *, collection_accounting: dict | None = None) -> dict:
        if self._sealed or not self._episodes:
            raise ValueError("seal a nonempty, not-yet-sealed validation bank")
        if self._source_hashes != {name: sha256_file(ROOT / name) for name in SOURCE_PATHS}:
            raise ValueError("validation collection source changed before sealing")
        manifest = {"schema_version": VERSION, "status": "sealed", "purpose": "model_validation",
                    "data_source": "real_environment", "split": "validation", "learning_eligible": False,
                    "enters_training_replay": False, "task_family": self.task_family, "run_seed": self.run_seed,
                    "configuration": self.configuration, "configuration_sha256": canonical_hash(self.configuration),
                    "task_sha256": canonical_hash(self.configuration["task"]),
                    "source_sha256": self._source_hashes,
                    "schema": field_manifest(self._events), "episodes": self._episodes,
                    "transition_count": self._transitions, "collection_accounting": collection_accounting,
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
        manifest["identity_sha256"] = canonical_hash(manifest)
        _write_json(self.path / "manifest.json", manifest)
        self._sealed = True
        return deepcopy(manifest)


class ValidationBank:
    """Receipt-verified validation windows with no fitting/imagination capability."""

    def __init__(self, directory: str | Path, *, verify_native_evidence: bool = True):
        if type(verify_native_evidence) is not bool:
            raise ValueError("native evidence verification flag must be boolean")
        self.path = Path(directory).resolve()
        _check_receipt(self.path / "manifest.json")
        manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != VERSION or manifest.get("status") != "sealed"
                or manifest.get("purpose") != "model_validation" or manifest.get("split") != "validation"
                or manifest.get("data_source") != "real_environment" or manifest.get("learning_eligible") is not False
                or manifest.get("enters_training_replay") is not False):
            raise ValueError("only sealed real validation-only banks are readable")
        if manifest.get("identity_sha256") != canonical_hash({key: value for key, value in manifest.items() if key != "identity_sha256"}):
            raise ValueError("validation bank identity hash differs")
        if (manifest["configuration_sha256"] != canonical_hash(manifest["configuration"])
                or manifest["task_sha256"] != canonical_hash(manifest["configuration"]["task"])):
            raise ValueError("validation configuration/task identity differs")
        self.manifest = deepcopy(manifest)
        self.task_family, self.run_seed = manifest["task_family"], manifest["run_seed"]
        self.cohort_identity = manifest["identity_sha256"]
        self._events = tuple(manifest["schema"]["event_fields"])
        if manifest["schema"] != field_manifest(self._events):
            raise ValueError("validation typed schema differs from the implementation")
        self._episodes, self._anchors = {}, []
        offset = 0
        for episode in manifest["episodes"]:
            identifier = episode["episode_id"]
            if (identifier in self._episodes or episode["task_family"] != self.task_family
                    or episode["run_seed"] != self.run_seed or episode["transition_id_start"] != offset
                    or episode["data_source"] != "real_environment" or episode["purpose"] != "model_validation"
                    or episode["learning_eligible"] is not False or episode["split"] != "validation"):
                raise ValueError("validation episode identity/purpose differs")
            if _scenario_provenance(episode["scenario_manifest"], self.task_family) != episode["split_provenance"]:
                raise ValueError("validation episode split provenance differs")
            scenario = episode["scenario_manifest"]
            if (scenario["hashes"]["wad_sha256"] != episode["scenario_sha256"]
                    or scenario["deadline"] != manifest["configuration"]["task"]["deadline"]
                    or scenario["frame_skip"] != episode["action_repeat"]
                    or episode["action_repeat"] != manifest["configuration"]["environment"]["frame_skip"]):
                raise ValueError("validation scenario/task/action-repeat identity differs")
            if not episode.get("native_evidence"):
                raise ValueError("validation native evidence is required")
            self._episodes[identifier] = deepcopy(episode)
            arrays = self._arrays(identifier)
            if verify_native_evidence:
                _native_evidence(self.path, episode["native_evidence"], arrays, episode["initial_dispatch_id"])
            self._anchors.extend((identifier, int(index)) for index in
                                 np.flatnonzero(arrays["image_valid"][:-1].all(axis=-1)))
            offset += episode["valid_length"]
        if not self._anchors or offset != manifest["transition_count"]:
            raise ValueError("validation bank is empty or its transition count differs")

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.path / "manifest.json")

    def _arrays(self, episode_id: str) -> dict:
        episode = self._episodes[episode_id]
        path = _member(self.path, episode["payload"]["path"])
        _check_receipt(path)
        if sha256_file(path) != episode["payload"]["sha256"] or path.stat().st_size != episode["payload"]["bytes"]:
            raise ValueError("validation episode differs from its bank manifest")
        arrays = read_npz(path)
        _validate_arrays(arrays, episode, self._events, self.manifest["configuration"]["task"])
        return arrays

    def window(self, dataset_index: int, episode_id: str, anchor: int, *, burn_in: int = 16,
               loss_steps: int = 48, full_prefix: bool = False) -> ReplayWindow:
        if type(dataset_index) is not int or dataset_index != 0:
            raise ValueError("a validation bank has dataset_index zero only")
        _integer(anchor, "anchor")
        _integer(burn_in, "burn-in")
        _integer(loss_steps, "loss steps", 1)
        if type(full_prefix) is not bool or episode_id not in self._episodes:
            raise ValueError("invalid validation window request")
        arrays = self._arrays(episode_id)
        length = self._episodes[episode_id]["valid_length"]
        if anchor >= length or not arrays["image_valid"][anchor].all():
            raise ValueError("validation anchor must be a nonterminal source with both views")
        start, stop = 0 if full_prefix else max(0, anchor-burn_in), min(length, anchor+loss_steps)
        # Copies release the rest of the decoded episode: a short window must
        # not retain hundreds of unused RGB states through a NumPy base view.
        states = {name: arrays[name][start:stop+1].copy() for name in STATE_ARRAYS}
        transitions = {name: arrays[name][start:stop].copy() for name in TRANSITION_ARRAYS}
        for array in (*states.values(), *transitions.values()):
            array.flags.writeable = False
        return ReplayWindow(0, episode_id, start, anchor, stop, states, transitions, self._events)

    def windows(self, *, burn_in: int = 16, loss_steps: int = 48, max_windows: int = 32,
                seed: int = 0, full_prefix: bool = False) -> list[ReplayWindow]:
        """Same anchor/target cohort for every burn-in; never consume fitting RNG."""
        _integer(max_windows, "maximum windows", 1)
        _integer(seed, "validation seed")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self._anchors), min(max_windows, len(self._anchors)), replace=False)
        return [self.window(0, *self._anchors[int(index)], burn_in=burn_in, loss_steps=loss_steps,
                            full_prefix=full_prefix) for index in indices]


def collect_validation_bank(task_family: str, run_seed: int, output_dir: str | Path, *,
                            episodes_per_template: int = 4, profile: str = "pilot",
                            config_dir: str | Path | None = None,
                            max_episode_steps: int | None = None) -> dict:
    """Collect both validation geometries; all costs stay outside learning budgets.

    Each episode has a fresh real native environment. Random/scripted episodes
    alternate; the mixture is descriptive, not the fitting replay's 70/30 quota.
    A failure leaves its runtime journals and separately charged ledger intact.
    """
    from world_models.data.collection import (CorrelatedRandomPolicy, RANDOM_POLICY_VERSION,
                                              ScriptedExplorationPolicy)
    from world_models.env.coop import TwoPlayerCoopEnv
    from world_models.env.splits import build_split_scenario
    from world_models.env.validation_data import ValidationLedger
    from world_models.infrastructure.config import resolve_config
    from world_models.infrastructure.reproducibility import derive_seed

    _integer(episodes_per_template, "episodes per validation template", 2)
    config_dir = Path(config_dir or ROOT / "configs")
    settings = resolve_config(task_family, "B", profile, config_dir).settings
    if type(run_seed) is not int or run_seed not in settings["experiment"]["master_seeds"]:
        raise ValueError("validation run seed must belong to the selected profile")
    limit = settings["task"]["deadline"] if max_episode_steps is None else _integer(max_episode_steps, "episode limit", 1)
    if limit > settings["task"]["deadline"]:
        raise ValueError("validation episode limit exceeds the task deadline")
    writer = ValidationBankWriter(output_dir, task_family=task_family, run_seed=run_seed, configuration=settings)
    ledger = ValidationLedger(writer.path)
    engine_version = importlib.metadata.version("vizdoom")
    for template_index, _ in enumerate(list_templates(task_family, "validation")):
        for repetition in range(episodes_per_template):
            episode_id = f"validation-{task_family}-s{run_seed}-t{template_index}-e{repetition:05d}"
            seed = derive_seed(task_family, run_seed, f"model_validation_{template_index}_{repetition}", "B")
            scenario = build_split_scenario(task_family, "validation", template_index, seed=seed,
                purpose="validation", frame_skip=settings["environment"]["frame_skip"],
                output_dir=writer.path / "runtime" / "scenarios" / episode_id)
            _scenario_provenance(scenario, task_family)
            env = TwoPlayerCoopEnv(task_family, seed=seed, scenario_manifest=scenario,
                frame_skip=settings["environment"]["frame_skip"], config_dir=config_dir,
                work_root=writer.path / "runtime" / "environments" / episode_id)
            published, steps, primary_error = 0, [], None
            try:
                reset = env.reset(seed=seed)
                scripted = repetition % 2 == 1
                policy = ScriptedExplorationPolicy(scenario, repetition // 2) if scripted else CorrelatedRandomPolicy(seed)
                info = reset.training_only_info
                for index in range(limit):
                    actions = policy.actions(info) if scripted else policy.actions()
                    step = env.step(*actions, collection_cutoff=index + 1 == limit)
                    published += 1
                    steps.append(step)
                    info = step.training_only_info
                    if step.terminated or step.truncated:
                        break
                ledger.record("model_validation", env, published)
            except BaseException as error:
                primary_error = error
                try:
                    ledger.record("model_validation_failed", env, published, expected_error=True)
                except BaseException as ledger_error:
                    error.add_note(f"validation accounting also failed: {ledger_error}")
                _write_json(writer.path / "collection_error.json", {
                    "status": "failed", "episode_id": episode_id, "error": str(error),
                    "published_transitions": published, "purpose": "model_validation", "learning_eligible": False})
                raise
            finally:
                try:
                    env.close()
                except BaseException as close_error:
                    if primary_error is None:
                        raise
                    primary_error.add_note(f"validation environment cleanup also failed: {close_error}")
            evidence = []
            for directory in (env.coordinator.run_dir, writer.path / "runtime" / "scenarios" / episode_id):
                for path in sorted(directory.rglob("*")):
                    if path.is_file():
                        evidence.append({"path": path.relative_to(writer.path).as_posix(),
                                         "bytes": path.stat().st_size, "sha256": sha256_file(path)})
            writer.append_episode(episode_id, reset, steps, collector_kind="scripted" if scripted else "random",
                policy_version=policy.version if scripted else RANDOM_POLICY_VERSION,
                engine_version=engine_version, native_evidence=evidence)
    return writer.seal(collection_accounting=ledger.manifest())
