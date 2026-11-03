"""Chunked immutable real replay; source-state and transition labels stay separate."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile

import numpy as np

from world_models.contracts import RGBFrame
from world_models.data.schema import (
    DATASET_VERSION, REASONS, STATE_ARRAYS, TRANSITION_ARRAYS, episode_partition,
    array_specs, event_field_names, extract_state, extract_transition, field_manifest,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_npz(path: str | Path, arrays: dict[str, np.ndarray], *, compression: str = "deflate",
              compression_level: int = 1) -> dict:
    """Production codec, also used by the storage benchmark. Never serializes pickle."""
    if compression not in ("stored", "deflate") or type(compression_level) is not int or not 0 <= compression_level <= 9:
        raise ValueError("compression must be stored/deflate with a level from 0 to 9")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    method = zipfile.ZIP_STORED if compression == "stored" else zipfile.ZIP_DEFLATED
    try:
        with temporary.open("wb") as stream:
            with zipfile.ZipFile(stream, "w", compression=method, compresslevel=compression_level) as archive:
                for name in sorted(arrays):
                    array = np.asarray(arrays[name])
                    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or array.dtype.hasobject:
                        raise ValueError("NPZ members must be named typed arrays without Python objects")
                    payload = io.BytesIO()
                    np.save(payload, array, allow_pickle=False)
                    # Deterministic ZIP metadata makes identical chunks byte-identical.
                    member = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                    member.compress_type = method
                    archive.writestr(member, payload.getvalue(), compress_type=method, compresslevel=compression_level)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Decode typed arrays only; object/pickle payloads are rejected by NumPy."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("object arrays are forbidden in replay")
    for value in arrays.values():
        value.flags.writeable = False
    return arrays


def _safe_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("replay manifest path escapes the dataset")
    return path


def _real_metadata(metadata: dict) -> dict:
    metadata = deepcopy(metadata)
    source = metadata.pop("source", metadata.get("data_source", "real_environment"))
    if source != "real_environment" or metadata.get("data_source", source) != "real_environment":
        raise ValueError("synthetic PPO/imagination data must use a separate store")
    if metadata.get("task_family") not in ("relay", "cover") or type(metadata.get("run_seed")) is not int:
        raise ValueError("dataset needs its task family and integer run seed")
    metadata["data_source"] = "real_environment"
    metadata["variant"] = "shared_initial:B,D"
    metadata["paired_variants"] = ["B", "D"]
    eligible = metadata.get("learning_eligible", True)
    if type(eligible) is not bool:
        raise ValueError("dataset learning eligibility must be a boolean")
    metadata["learning_eligible"] = eligible and metadata.get("budget_scope") != "collection_preflight"
    metadata["dataset_version"] = DATASET_VERSION
    json.dumps(metadata, allow_nan=False)
    return metadata


class ReplayDatasetWriter:
    """Single-writer append pipeline; a sealed dataset can never be reopened for writes."""

    def __init__(self, path: str | Path, metadata: dict, *, chunk_transitions: int = 128,
                 compression: str = "deflate", compression_level: int = 1, resume: bool = False):
        self.path = Path(path).resolve()
        self.metadata = _real_metadata(metadata)
        if type(chunk_transitions) is not int or chunk_transitions < 1:
            raise ValueError("chunk_transitions must be a positive integer")
        if compression not in ("stored", "deflate") or type(compression_level) is not int or not 0 <= compression_level <= 9:
            raise ValueError("invalid chunk compression")
        self.chunk_transitions, self.compression, self.compression_level = chunk_transitions, compression, compression_level
        self._active = None
        self._sealed = False
        self._sealed_manifest = None
        self._event_fields = None
        self._episodes = []
        self._count = 0
        if (self.path / "manifest.json").exists():
            raise FileExistsError("dataset is sealed; create a new dataset instead of mutating shared initialization")
        if self.path.exists() and any(self.path.iterdir()):
            if not resume:
                raise FileExistsError("replay output is not empty; use a fresh directory")
            self._resume()
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            self._checkpoint()

    @property
    def transition_count(self) -> int:
        return self._count

    @property
    def manifest(self) -> dict:
        if self._sealed_manifest is not None:
            return deepcopy(self._sealed_manifest)
        return self._manifest("sealed" if self._sealed else "collecting")

    def _manifest(self, status: str) -> dict:
        return {"dataset_version": DATASET_VERSION, "status": status, "metadata": deepcopy(self.metadata),
                "storage": {"format": "npz-typed-chunks-v1", "compression": self.compression,
                            "compression_level": self.compression_level, "chunk_transitions": self.chunk_transitions},
                "schema": field_manifest(self._event_fields or ()), "episodes": deepcopy(self._episodes),
                "transition_count": self._count,
                "partition_rule": "sha256(real-replay-v1|episode_id) first64bits mod10;0 diagnostic,otherwise fit",
                "partition_transition_counts": {part: sum(ep["valid_length"] for ep in self._episodes if ep["partition"] == part)
                                                for part in ("fit", "diagnostic")},
                "learning_guards": {"fit_and_imagination_allow_only": "fit", "allowed_geometry_split": "train",
                                    "synthetic_data_allowed": False, "diagnostic_import_requires_explicit_training_ledger": True},
                "active_episode": self._active["metadata"]["episode_id"] if self._active else None}

    def _checkpoint(self) -> None:
        _atomic_json(self.path / "manifest.partial.json", self._manifest("collecting"))

    def _resume(self) -> None:
        path = self.path / "manifest.partial.json"
        if not path.is_file():
            raise ValueError("no resumable replay manifest")
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["metadata"] != self.metadata or saved["storage"] != {
                "format": "npz-typed-chunks-v1", "compression": self.compression,
                "compression_level": self.compression_level, "chunk_transitions": self.chunk_transitions}:
            raise ValueError("resume metadata or storage setting differs")
        if saved.get("active_episode"):
            raise ValueError("interrupted active episode requires explicit accounting/recovery; refusing silent data loss")
        self._episodes, self._count = saved["episodes"], saved["transition_count"]
        self._event_fields = tuple(saved["schema"]["event_fields"]) or None
        for episode in self._episodes:
            if sha256_file(_safe_member(self.path, episode["manifest_path"])) != episode["manifest_sha256"]:
                raise ValueError("completed episode changed before resume")
            for chunk in episode["chunks"]:
                if sha256_file(_safe_member(self.path, chunk["path"])) != chunk["sha256"]:
                    raise ValueError("completed replay chunk changed before resume")

    def start_episode(self, metadata: dict, reset) -> None:
        if self._sealed or self._active:
            raise RuntimeError("cannot start an episode while another is active or after sealing")
        metadata = deepcopy(metadata)
        info = reset.training_only_info
        scenario = info.get("scenario_manifest", {})
        provenance = scenario.get("split_provenance", {})
        if (provenance.get("split") != "train" or provenance.get("purpose") != "training"
                or provenance.get("learning_eligible") is not True):
            raise ValueError("learning replay requires training geometry/provenance; validation trajectories need explicit counted import")
        if metadata.get("split", "train") != "train" or metadata.get("layout_id", provenance.get("template_id")) != provenance.get("template_id"):
            raise ValueError("episode layout/split metadata differs from the authoritative scenario")
        if metadata.get("data_source", "real_environment") != "real_environment" or metadata.get("purpose") == "environment_validation":
            raise ValueError("diagnostic/synthetic trajectories may not automatically enter training replay")
        episode_id = metadata.get("episode_id")
        if not isinstance(episode_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", episode_id):
            raise ValueError("episode_id must be a safe stable ID of at most 128 characters")
        if episode_id in {episode["episode_id"] for episode in self._episodes}:
            raise ValueError("duplicate episode ID")
        if metadata.get("collector_kind") not in ("random", "scripted"):
            raise ValueError("initial replay collector must be random or scripted")
        policy = metadata.get("policy_version")
        if not isinstance(policy, str) or not 1 <= len(policy) <= 128:
            raise ValueError("a collector policy version is required")
        if metadata.get("run_seed", self.metadata["run_seed"]) != self.metadata["run_seed"]:
            raise ValueError("episode run seed differs from the dataset")
        family = metadata.get("task_family", info.get("task_family"))
        if family != self.metadata["task_family"] or scenario.get("task_family", family) != family:
            raise ValueError("episode belongs to another task family")
        environment_seed = metadata.get("environment_seed", info.get("engine_seed"))
        if type(environment_seed) is not int or not 0 <= environment_seed < 2**32:
            raise ValueError("an episode needs a uint32 engine/environment seed")
        if info.get("engine_seed") is not None and environment_seed != info["engine_seed"]:
            raise ValueError("episode environment seed differs from the actual reset")
        if type(metadata.get("collection_block", 0)) is not int or metadata.get("collection_block", 0) < 0:
            raise ValueError("collection block must be a nonnegative integer")
        version = metadata.get("engine_version", self.metadata.get("engine_version"))
        if not isinstance(version, str) or not version:
            raise ValueError("pin the episode engine version")
        scenario_hash = metadata.get("scenario_hash", scenario.get("hashes", {}).get("wad_sha256"))
        if not isinstance(scenario_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", scenario_hash):
            raise ValueError("episode requires its scenario SHA256")
        if scenario_hash != scenario.get("hashes", {}).get("wad_sha256"):
            raise ValueError("episode scenario hash differs from the authoritative scenario")
        fields = event_field_names(reset)
        if self._event_fields is not None and fields != self._event_fields:
            raise ValueError("episode event schema differs from dataset")
        first = extract_state(reset, fields)
        if not first["image_valid"].all() or info.get("decision_index") != 0:
            raise ValueError("an episode starts from two validated reset images at decision zero")
        if metadata.get("initial_tic", int(first["state_engine_tic"])) != int(first["state_engine_tic"]):
            raise ValueError("episode initial tic differs from the actual reset")
        engine_episode_id = info.get("episode_id")
        if type(engine_episode_id) is not int or engine_episode_id < 0:
            raise ValueError("reset lacks an authoritative engine episode ID")
        self._event_fields = fields
        metadata.update(dataset_version=DATASET_VERSION, task_family=family, run_seed=self.metadata["run_seed"],
                        variant="shared_initial:B,D", environment_seed=environment_seed,
                        layout_id=metadata.get("layout_id", provenance.get("template_id")), split="train",
                        scenario_hash=scenario_hash, engine_version=version, initial_tic=int(first["state_engine_tic"]),
                        partition=episode_partition(episode_id), split_provenance=deepcopy(provenance),
                        data_source="real_environment", learning_eligible=self.metadata["learning_eligible"],
                        action_repeat=scenario.get("frame_skip", 4), engine_episode_id=engine_episode_id,
                        layout_seed=info.get("layout_seed", scenario.get("seed")),
                        native_counter_baseline=deepcopy((info.get("events") or {}).get("native_counter_baseline", {})))
        if not metadata["layout_id"]:
            raise ValueError("episode requires a layout/template ID")
        if type(metadata["action_repeat"]) is not int or not 1 <= metadata["action_repeat"] <= 65535:
            raise ValueError("episode action repeat must fit the declared real-transition dtype")
        json.dumps(metadata, allow_nan=False)
        directory = self.path / "episodes" / episode_id
        if directory.exists():
            raise FileExistsError("episode output already exists")
        directory.mkdir(parents=True)
        self._active = {"metadata": metadata, "directory": directory, "states": [first], "transitions": [],
                        "chunks": [], "length": 0, "state_count": 0, "transition_count": 0,
                        "previous_state": first, "boundary": None, "start_transition_id": self._count}
        self._checkpoint()

    def append(self, step) -> None:
        active = self._active
        if self._sealed or active is None or active["boundary"] is not None:
            raise RuntimeError("append requires a live non-ended episode")
        info, metadata = step.training_only_info, step.metadata
        if info.get("task_family") != self.metadata["task_family"]:
            raise ValueError("transition task family changed")
        if info.get("episode_id") != active["metadata"]["engine_episode_id"]:
            raise ValueError("transition crossed an engine episode/reset boundary")
        if info.get("scenario_manifest", {}).get("hashes", {}).get("wad_sha256") != active["metadata"]["scenario_hash"]:
            raise ValueError("transition changed the episode scenario")
        if metadata.boundary != step.boundary:
            raise ValueError("transition metadata and result boundaries disagree")
        if info.get("decision_index") != active["length"] + 1:
            raise ValueError("transition result decision index is not consecutive")
        previous = active["previous_state"]
        if metadata.start_tic != int(previous["state_engine_tic"]):
            raise ValueError("transition is not aligned with its source observation")
        if metadata.action_repeat != active["metadata"]["action_repeat"]:
            raise ValueError("transition changed the episode action-repeat contract")
        state = extract_state(step, self._event_fields)
        if metadata.end_tic != int(state["state_engine_tic"]):
            raise ValueError("transition is not aligned with its next observation")
        transition = extract_transition(step, self._event_fields, transition_id=self._count,
                                        observation_index=active["length"], episode_metadata=active["metadata"])
        if not np.array_equal(state["cumulative_event_counters"] - previous["cumulative_event_counters"], transition["event_deltas"]):
            raise ValueError("event delta does not match the source/next cumulative counters")
        if np.any(transition["event_deltas"] < 0):
            raise ValueError("event counters regressed")
        active["states"].append(state)
        active["transitions"].append(transition)
        active["length"] += 1
        active["previous_state"] = state
        self._count += 1
        if step.terminated or step.truncated:
            active["boundary"] = step.boundary.reason.value
        if len(active["transitions"]) == self.chunk_transitions:
            self._flush()

    def _flush(self) -> None:
        active = self._active
        if not active["transitions"]:
            return
        arrays = {name: np.stack([state[name] for state in active["states"]]) for name in STATE_ARRAYS}
        arrays.update({name: np.stack([step[name] for step in active["transitions"]]) for name in TRANSITION_ARRAYS})
        path = active["directory"] / f"chunk_{len(active['chunks']):06d}.npz"
        receipt = write_npz(path, arrays, compression=self.compression, compression_level=self.compression_level)
        active["chunks"].append({"path": path.relative_to(self.path).as_posix(), **receipt,
                                  "state_start": active["state_count"], "state_count": len(active["states"]),
                                  "transition_start": active["transition_count"], "transition_count": len(active["transitions"])})
        active["state_count"] += len(active["states"])
        active["transition_count"] += len(active["transitions"])
        # The last source state remains only in the previous chunk. Subsequent
        # transition observation_index points back to it, avoiding frame copies.
        active["states"], active["transitions"] = [], []
        _atomic_json(active["directory"] / "manifest.partial.json", {
            "metadata": active["metadata"], "chunks": active["chunks"], "valid_length": active["length"],
            "status": "collecting", "boundary": active["boundary"]})

    def finish_episode(self) -> dict:
        active = self._active
        if self._sealed or active is None or active["boundary"] is None:
            raise RuntimeError("finish_episode requires a real terminal or external collection cutoff")
        self._flush()
        if active["state_count"] != active["length"] + 1 or active["transition_count"] != active["length"]:
            raise RuntimeError("episode state/transition storage counts do not reconcile")
        episode = {**active["metadata"], "end_reason": active["boundary"], "valid_length": active["length"],
                   "observation_count": active["state_count"], "transition_id_start": active["start_transition_id"],
                   "chunks": deepcopy(active["chunks"])}
        manifest_path = active["directory"] / "manifest.json"
        _atomic_json(manifest_path, episode)
        episode["manifest_path"] = manifest_path.relative_to(self.path).as_posix()
        episode["manifest_sha256"] = sha256_file(manifest_path)
        self._episodes.append(episode)
        partial = active["directory"] / "manifest.partial.json"
        if partial.exists():
            partial.unlink()
        self._active = None
        self._checkpoint()
        return deepcopy(episode)

    def seal(self) -> dict:
        if self._sealed or self._active:
            raise RuntimeError("seal requires all episodes complete and an unsealed dataset")
        target = self.metadata.get("target_transitions")
        if target is not None and self._count != target:
            raise ValueError("dataset does not contain its declared initial transition budget")
        if not self._episodes or self._count < 1:
            raise ValueError("cannot seal empty initial replay")
        manifest = self._manifest("sealed")
        manifest["sealed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["dataset_identity_sha256"] = canonical_hash({key: value for key, value in manifest.items() if key != "sealed_at_utc"})
        path = self.path / "manifest.json"
        _atomic_json(path, manifest)
        (self.path / "manifest.sha256").write_text(sha256_file(path) + "  manifest.json\n", encoding="ascii")
        (self.path / "manifest.partial.json").unlink()
        self._sealed = True
        self._sealed_manifest = deepcopy(manifest)
        return deepcopy(manifest)


class EpisodeView:
    def __init__(self, dataset: "ReplayDataset", metadata: dict):
        self.dataset, self.metadata = dataset, metadata

    def chunks(self):
        for chunk in self.metadata["chunks"]:
            yield chunk, self.dataset._chunk(chunk)

    def state(self, index: int) -> dict[str, np.ndarray]:
        if type(index) is not int or not 0 <= index < self.metadata["observation_count"]:
            raise IndexError("state index outside episode")
        for chunk in self.metadata["chunks"]:
            if chunk["state_start"] <= index < chunk["state_start"] + chunk["state_count"]:
                arrays = self.dataset._chunk(chunk)
                local = index - chunk["state_start"]
                return {name: arrays[name][local] for name in STATE_ARRAYS}
        raise ValueError("manifest has a hole in its observation sequence")

    def observation(self, index: int, player: int) -> RGBFrame | None:
        if type(player) is not int or player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        state = self.state(index)
        return RGBFrame(state["rgb"][player].tobytes()) if bool(state["image_valid"][player]) else None

    def valid_anchor_indices(self, *, purpose: str = "imagination") -> np.ndarray:
        """Only fitting episodes and valid source images can seed learning rollouts."""
        if purpose not in ("world_model_fit", "imagination"):
            raise ValueError("anchor selection requires a fitting or imagination purpose")
        self.dataset.load_episode(self.metadata["episode_id"], purpose=purpose)
        indices = []
        for chunk, arrays in self.chunks():
            for local in range(chunk["state_count"]):
                index = chunk["state_start"] + local
                if index < self.metadata["valid_length"] and arrays["image_valid"][local].all():
                    indices.append(index)
        output = np.array(indices, dtype=np.int64)
        output.flags.writeable = False
        return output

    def arrays(self) -> dict[str, np.ndarray]:
        """Materialize one episode only; normal training should use chunk/window access."""
        chunks = [arrays for _, arrays in self.chunks()]
        values = {name: np.concatenate([chunk[name] for chunk in chunks], axis=0) for name in (*STATE_ARRAYS, *TRANSITION_ARRAYS)}
        for array in values.values():
            array.flags.writeable = False
        return values

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays()[name]


class ReplayDataset:
    """Read-only real replay with a bounded LRU chunk cache and purpose guards."""

    def __init__(self, path: str | Path, *, verify: bool = True, cache_chunks: int = 4):
        if type(cache_chunks) is not int or cache_chunks < 1:
            raise ValueError("cache_chunks must be positive")
        self.path = Path(path).resolve()
        manifest_path = self.path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("only sealed datasets are readable for learning")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = self.path / "manifest.sha256"
        if not receipt.is_file() or receipt.read_text(encoding="ascii").split()[0] != sha256_file(manifest_path):
            raise ValueError("dataset manifest changed or has no receipt")
        manifest = self.manifest
        identity = canonical_hash({key: value for key, value in manifest.items() if key not in ("sealed_at_utc", "dataset_identity_sha256")})
        if manifest.get("dataset_identity_sha256") != identity:
            raise ValueError("dataset content identity mismatch")
        if manifest.get("dataset_version") != DATASET_VERSION or manifest.get("status") != "sealed" or manifest.get("active_episode"):
            raise ValueError("unsupported or unsealed replay")
        if _real_metadata(manifest["metadata"]) != manifest["metadata"]:
            raise ValueError("dataset is not declared real learning data")
        self._learning_eligible = manifest["metadata"]["learning_eligible"]
        self._episodes = deepcopy(manifest["episodes"])
        event_fields = tuple(manifest["schema"]["event_fields"])
        if manifest["schema"] != field_manifest(event_fields):
            raise ValueError("replay array/target schema differs from its executable version")
        self._array_specs = array_specs(event_fields)
        self._cache, self.cache_chunks = OrderedDict(), cache_chunks
        self._verify = verify
        self._ids = {}
        expected_transition = 0
        for episode in self._episodes:
            identifier = episode["episode_id"]
            if identifier in self._ids or episode["partition"] != episode_partition(identifier):
                raise ValueError("episode identity/partition is inconsistent")
            self._ids[identifier] = episode
            provenance = episode["split_provenance"]
            if episode["split"] != "train" or provenance.get("purpose") != "training" or not provenance.get("learning_eligible"):
                raise ValueError("evaluation/diagnostic geometry entered learning replay")
            if episode["transition_id_start"] != expected_transition:
                raise ValueError("global transition IDs do not form a unique consecutive sequence")
            expected_transition += episode["valid_length"]
            self._verify_episode_manifest(episode)
            if verify:
                self._verify_episode(episode)
        if expected_transition != manifest["transition_count"]:
            raise ValueError("dataset transition totals do not reconcile")
        if manifest["partition_transition_counts"] != {
                part: sum(episode["valid_length"] for episode in self._episodes if episode["partition"] == part)
                for part in ("fit", "diagnostic")}:
            raise ValueError("model-fitting/diagnostic transition totals do not reconcile")

    @property
    def episodes(self) -> list[dict]:
        return deepcopy(self._episodes)

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.path / "manifest.json")

    def _verify_episode_manifest(self, episode: dict) -> None:
        path = _safe_member(self.path, episode["manifest_path"])
        if sha256_file(path) != episode["manifest_sha256"]:
            raise ValueError("episode manifest changed")
        content = json.loads(path.read_text(encoding="utf-8"))
        if content != {key: value for key, value in episode.items() if key not in ("manifest_path", "manifest_sha256")}:
            raise ValueError("episode manifest and dataset index disagree")

    def _chunk(self, chunk: dict) -> dict[str, np.ndarray]:
        key = chunk["path"]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = _safe_member(self.path, key)
        if sha256_file(path) != chunk["sha256"]:
            raise ValueError("replay chunk bytes changed")
        arrays = read_npz(path)
        if set(arrays) != set((*STATE_ARRAYS, *TRANSITION_ARRAYS)):
            raise ValueError("chunk arrays differ from the declared schema")
        for name, specification in self._array_specs.items():
            value = arrays[name]
            if value.dtype.str != specification["dtype"] or value.shape[1:] != tuple(specification["row_shape"]):
                raise ValueError(f"chunk {name} shape/dtype differs from the declared typed interface")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"nonfinite replay values in {name}")
        if any(len(arrays[name]) != chunk["state_count"] for name in STATE_ARRAYS) or any(len(arrays[name]) != chunk["transition_count"] for name in TRANSITION_ARRAYS):
            raise ValueError("chunk row counts differ from its manifest")
        for name in ("cumulative_event_counters", "event_deltas", "state_engine_tic", "transition_id", "decision_index", "start_tic", "end_tic", "observation_index", "next_observation_index", "collection_block"):
            if np.any(arrays[name] < 0):
                raise ValueError(f"negative replay counter/index in {name}")
        if np.any(arrays["actions"] >= 12) or np.any(arrays["actual_tics"] < 1):
            raise ValueError("invalid action or actual tic count")
        if np.any(arrays["rgb"][~arrays["image_valid"]] != 0):
            raise ValueError("invalid image slots must contain masked storage padding only")
        self._cache[key] = arrays
        while len(self._cache) > self.cache_chunks:
            self._cache.popitem(last=False)
        return arrays

    def _verify_episode(self, episode: dict) -> None:
        state_index = transition_index = 0
        previous_counters = None
        previous_tic = None
        final_reason = None
        ended = False
        for chunk in episode["chunks"]:
            if chunk["state_start"] != state_index or chunk["transition_start"] != transition_index:
                raise ValueError("noncontiguous/duplicated observation or transition chunks")
            arrays = self._chunk(chunk)
            states = {state_index + index: {name: arrays[name][index] for name in STATE_ARRAYS} for index in range(chunk["state_count"])}
            for index in range(chunk["transition_count"]):
                source = transition_index + index
                if ended or int(arrays["observation_index"][index]) != source or int(arrays["next_observation_index"][index]) != source + 1:
                    raise ValueError("transition crosses an episode boundary or has invalid observation indices")
                if int(arrays["transition_id"][index]) != episode["transition_id_start"] + source or int(arrays["decision_index"][index]) != source:
                    raise ValueError("transition ID/decision index mismatch")
                if str(arrays["episode_id"][index]) != episode["episode_id"]:
                    raise ValueError("transition episode identity mismatch")
                current = states.get(source)
                next_state = states.get(source + 1)
                if next_state is None:
                    raise ValueError("chunk is missing a next observation")
                counters = current["cumulative_event_counters"] if current else previous_counters
                tic = int(current["state_engine_tic"]) if current else previous_tic
                if counters is None or not np.array_equal(next_state["cumulative_event_counters"] - counters, arrays["event_deltas"][index]):
                    raise ValueError("state/event target indexing is inconsistent")
                if (int(arrays["start_tic"][index]) != tic or int(arrays["end_tic"][index]) != int(next_state["state_engine_tic"])
                        or int(arrays["end_tic"][index] - arrays["start_tic"][index]) != int(arrays["actual_tics"][index])):
                    raise ValueError("state/action tic indexing is inconsistent")
                if not np.array_equal(arrays["next_obs_valid"][index], next_state["image_valid"]):
                    raise ValueError("next-image masks are not observation aligned")
                reason = int(arrays["termination_reason"][index])
                if not 0 <= reason < len(REASONS):
                    raise ValueError("invalid terminal reason code")
                final_reason = REASONS[reason]
                terminal = final_reason in ("success", "death", "failure", "deadline")
                cutoff = final_reason == "collection_cutoff"
                if bool(arrays["terminated"][index]) != terminal or bool(arrays["truncated"][index]) != cutoff:
                    raise ValueError("boundary flags disagree with their reason")
                if not terminal and not next_state["image_valid"].all():
                    raise ValueError("missing nonterminal next observation")
                actual = int(arrays["actual_tics"][index])
                if actual > episode["action_repeat"] or (actual != episode["action_repeat"] and not terminal):
                    raise ValueError("transition actual tics violate the episode action repeat")
                if str(arrays["collector_policy_version"][index]) != episode["policy_version"]:
                    raise ValueError("transition collector-policy version changed within episode")
                ended = terminal or cutoff
                previous_counters, previous_tic = next_state["cumulative_event_counters"], int(next_state["state_engine_tic"])
            state_index += chunk["state_count"]
            transition_index += chunk["transition_count"]
        if (transition_index != episode["valid_length"] or state_index != transition_index + 1
                or final_reason != episode["end_reason"] or not ended):
            raise ValueError("episode length/end boundary does not reconcile")

    def load_episode(self, episode_id: str, *, purpose: str = "diagnostic") -> EpisodeView:
        if purpose not in ("diagnostic", "world_model_fit", "imagination"):
            raise ValueError("declare diagnostic, world_model_fit, or imagination purpose")
        if purpose != "diagnostic" and not self._learning_eligible:
            raise ValueError("collection-preflight/diagnostic stores are excluded from learning; explicit counted import is required")
        episode = self._ids[episode_id]
        if purpose != "diagnostic" and episode_partition(episode_id) != "fit":
            raise ValueError("model-diagnostic episodes are excluded from fitting and imagination anchors")
        return EpisodeView(self, deepcopy(episode))

    def iter_episodes(self, *, purpose: str = "world_model_fit"):
        if purpose not in ("diagnostic", "world_model_fit", "imagination"):
            raise ValueError("declare diagnostic, world_model_fit, or imagination purpose")
        if purpose != "diagnostic" and not self._learning_eligible:
            raise ValueError("collection-preflight/diagnostic stores are excluded from learning; explicit counted import is required")
        for episode in self._episodes:
            if purpose in ("world_model_fit", "imagination") and episode["partition"] != "fit":
                continue
            yield self.load_episode(episode["episode_id"], purpose=purpose)

    def bind_variant(self, path: str | Path, variant: str) -> dict:
        if variant not in ("B", "D"):
            raise ValueError("only the paired B/D variants are supported")
        path = Path(path)
        if path.exists():
            raise FileExistsError("branch binding already exists")
        value = {"variant": variant, "dataset_path": os.path.relpath(self.path, path.parent.resolve()), "manifest_sha256": self.manifest_sha256,
                 "dataset_identity_sha256": self.manifest["dataset_identity_sha256"],
                 "attributed_initial_joint_transitions": self.manifest["transition_count"],
                 "physical_collection_shared": True, "new_physical_transitions": 0}
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, value)
        return value

    @classmethod
    def from_binding(cls, path: str | Path, *, variant: str | None = None, **kwargs) -> "ReplayDataset":
        path = Path(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("variant") not in ("B", "D") or variant is not None and value["variant"] != variant:
            raise ValueError("shared initialization binding belongs to another variant")
        dataset_path = Path(value["dataset_path"])
        if not dataset_path.is_absolute():
            dataset_path = path.parent / dataset_path
        dataset = cls(dataset_path, **kwargs)
        if (dataset.manifest_sha256 != value["manifest_sha256"]
                or dataset.manifest["dataset_identity_sha256"] != value["dataset_identity_sha256"]
                or dataset.manifest["transition_count"] != value["attributed_initial_joint_transitions"]):
            raise ValueError("shared initialization binding no longer matches immutable data")
        return dataset
