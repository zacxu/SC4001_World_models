"""Measure the production replay codec on genuine actor pixels.

This benchmark does not collect simulator interactions or train a model. Its
rates describe repeated writes and OS-cache-warm reads in the measured process,
not a promise about the eventual learner's throughput.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import tempfile
import time

import numpy as np
from PIL import Image

from world_models.env.acceptance_metrics import ProcessTreeMonitor, cpu_budget, percentile


_FRAME_NAME = re.compile(r"frame(\d+)_player([01])\.png$")


def _append_fsync(path: Path, payload: bytes) -> float:
    """Time the append/open/flush/fsync/close path used by durable journals."""
    began = time.perf_counter()
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return time.perf_counter() - began


def benchmark_fsync_directories(output: str | Path, *, locations: dict[str, str | Path],
                                writes: int = 24, payload_bytes: int = 4096,
                                warmup_writes: int = 2, cpu_limit: int = 4) -> dict:
    """Measure durable small-append latency at explicitly named directories.

    This diagnosis can explain why the same collector behaves differently on
    two volumes without reducing its fsync frequency. Every operation appends
    fresh deterministic pseudorandom bytes; no simulator interactions occur.
    Only probe files created by this call are removed, even on failure.
    """
    from world_models.data.ledger import atomic_json

    if (type(writes) is not int or writes < 1 or type(payload_bytes) is not int or payload_bytes < 1
            or type(warmup_writes) is not int or warmup_writes < 0):
        raise ValueError("writes/payload_bytes must be positive integers and warmup_writes nonnegative")
    if not locations or any(not isinstance(label, str) or not label for label in locations):
        raise ValueError("provide named probe directories")
    roots = {label: Path(directory).resolve(strict=True) for label, directory in locations.items()}
    if any(not path.is_dir() for path in roots.values()):
        raise ValueError("each probe location must be an existing directory")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("preserve previous fsync measurements; choose a fresh output directory")
    destination.mkdir(parents=True)
    owned, cases = {}, {}
    rng = random.Random(21721)
    try:
        with cpu_budget(cpu_limit) as allocation:
            for label, directory in roots.items():
                probe_directory = Path(tempfile.mkdtemp(prefix="world-models-fsync-probe-", dir=directory))
                owned[label] = (probe_directory, probe_directory / "append.bin")
                cases[label] = {"directory": str(directory), "path_anchor": directory.anchor,
                                "samples": [], "expected_sha256": hashlib.sha256(),
                                "fsynced_operations": 0}
            labels = list(roots)
            # Alternate the traversal order to reduce simple time-order bias.
            for index in range(warmup_writes + writes):
                for label in labels if index % 2 == 0 else reversed(labels):
                    payload = rng.randbytes(payload_bytes)
                    elapsed = _append_fsync(owned[label][1], payload)
                    case = cases[label]
                    case["expected_sha256"].update(payload)
                    case["fsynced_operations"] += 1
                    if index >= warmup_writes:
                        case["samples"].append({"write_index": index - warmup_writes,
                                                "elapsed_seconds": elapsed, "bytes": payload_bytes,
                                                "payload_sha256": hashlib.sha256(payload).hexdigest()})
            results = {}
            for label, case in cases.items():
                path = owned[label][1]
                payload = path.read_bytes()
                actual_hash = hashlib.sha256(payload).hexdigest()
                if len(payload) != (warmup_writes + writes) * payload_bytes or actual_hash != case["expected_sha256"].hexdigest():
                    raise RuntimeError("fsynced probe bytes differ from the appended payloads")
                times = [sample["elapsed_seconds"] for sample in case["samples"]]
                results[label] = {"directory": case["directory"], "path_anchor": case["path_anchor"],
                                  "sample_count": len(times), "warmup_writes": warmup_writes,
                                  "payload_bytes_per_write": payload_bytes,
                                  "total_fsynced_write_operations": case["fsynced_operations"],
                                  "total_verified_bytes": len(payload), "appended_bytes_sha256": actual_hash,
                                  "median_latency_seconds": statistics.median(times),
                                  "p95_latency_seconds": percentile(times, .95),
                                  "minimum_latency_seconds": min(times), "maximum_latency_seconds": max(times),
                                  "samples": case["samples"], "payload_roundtrip_exact": True}
        report = {"schema_version": "durable-journal-io-probe-v1",
                  "status": "passed" if writes >= 24 and payload_bytes >= 4096 and warmup_writes >= 2 else "passed_preflight",
                  "cpu_budget": allocation, "locations": results,
                  "ordering": "location order alternates each repetition; 21721 deterministic PRNG, fresh payload per append",
                  "timing_scope": "Open append stream, write bytes, flush, request os.fsync, and close; warmup, directory creation, verification, cleanup and report writes excluded.",
                  "cache_scope": "Ordinary filesystem caching remains enabled. Each append requests fsync; no direct-I/O or power-loss test is claimed.",
                  "interpretation": "Measures this small durable-journal workload at the recorded directories. It does not change storage settings, move learning data, reduce fsync frequency, or benchmark the full collector.",
                  "simulator_interactions": 0, "owned_probe_files_removed": True}
        # Remove only our exact file and empty directory. Never recursively
        # delete a caller-provided location or an unverified computed target.
        for label, (directory, path) in list(owned.items()):
            path.unlink()
            directory.rmdir()
            del owned[label]
        path = destination / "durable_journal_io_probe.json"
        atomic_json(path, report)
        path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n", encoding="ascii")
        return report
    except BaseException as exc:
        atomic_json(destination / "error.json", {"status": "failed", "error": repr(exc),
                                               "locations": {key: str(value) for key, value in roots.items()}})
        raise
    finally:
        for directory, path in owned.values():
            if path.exists():
                path.unlink()
            directory.rmdir()


def load_actor_samples(sample_root: str | Path, *, maximum_joint_observations: int = 256) -> tuple[np.ndarray, dict]:
    """Load paired actor PNGs, preserving each actual observation exactly.

    ``sample_root`` is a mechanics directory with ``relay_train_00/video`` and
    ``cover_train_00/video`` children, or one of those video directories. Only
    these explicit train sources are accepted; this does not import validation
    trajectories into fitting replay.
    """
    if type(maximum_joint_observations) is not int or maximum_joint_observations < 2:
        raise ValueError("maximum_joint_observations must be an integer >= 2")
    root = Path(sample_root).resolve()
    sources = [(family, root / f"{family}_train_00" / "video") for family in ("relay", "cover")]
    if root.name == "video" and root.parent.name in ("relay_train_00", "cover_train_00"):
        sources = [(root.parent.name.removesuffix("_train_00"), root)]
    frames, records = [], []
    # Interleave families so a sample cap cannot silently discard one family.
    by_family = {}
    for family, directory in sources:
        paired = {}
        for path in sorted(directory.glob("*.png")):
            match = _FRAME_NAME.fullmatch(path.name)
            if match:
                paired.setdefault(int(match[1]), {})[int(match[2])] = path
        if not paired:
            raise ValueError(f"no paired actor frames in {directory}")
        rows = []
        for index, paths in sorted(paired.items()):
            if set(paths) != {0, 1}:
                raise ValueError(f"unpaired actor frame {directory} index {index}")
            images, image_records = [], []
            for player in (0, 1):
                path = paths[player]
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.size != (128, 96):
                        raise ValueError(f"expected genuine RGB128x96 actor frame: {path}")
                    values = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
                images.append(values)
                image_records.append({"player": player, "path": str(path),
                                      "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "rgb_sha256": hashlib.sha256(values.tobytes()).hexdigest()})
            rows.append((np.stack(images), {"task_family": family, "frame": index, "images": image_records}))
        by_family[family] = rows
    for index in range(max(len(rows) for rows in by_family.values())):
        for rows in by_family.values():
            if index < len(rows) and len(frames) < maximum_joint_observations:
                frame, record = rows[index]
                frames.append(frame)
                records.append(record)
    if len(frames) < 2:
        raise ValueError("storage benchmark needs at least two paired observations")
    values = np.stack(frames)
    return values, {"source_kind": "phase1_scripted_actor_pngs", "source_root": str(root),
                    "joint_observations": len(values), "families": dict(Counter(row["task_family"] for row in records)),
                    "records": records, "pixels_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    "scope": "Actual native actor images from scripted train00 episodes only; not the Phase2 70/30 collection mixture. No samples enter learning replay."}


def load_replay_samples(dataset_paths: list[str | Path], *, maximum_joint_observations: int = 256) -> tuple[np.ndarray, dict]:
    """Read a bounded real-replay sample; never modify its sealed dataset.

    This API can rerun the codec measurement on the actual collection mixture
    after initial replay exists. It is independent of model-fitting inclusion.
    """
    from world_models.data.replay import ReplayDataset

    if type(maximum_joint_observations) is not int or maximum_joint_observations < 2 or not dataset_paths:
        raise ValueError("provide replay datasets and at least two observation slots")
    frames, records, families = [], [], Counter()
    datasets = [ReplayDataset(path, verify=True) for path in dataset_paths]
    per_dataset_limit = max(1, maximum_joint_observations // len(datasets))
    for dataset in datasets:
        family = dataset.manifest["metadata"]["task_family"]
        total = sum(episode["observation_count"] for episode in dataset.episodes)
        count = min(total, per_dataset_limit)
        # Evenly spread positions cover the whole collected ordering, instead
        # of sampling only the first random episode and omitting scripts.
        positions = [min(total - 1, int((index + .5) * total / count)) for index in range(count)] if count else []
        offset = cursor = 0
        for metadata in dataset.episodes:
            end = offset + metadata["observation_count"]
            selected = []
            while cursor < len(positions) and positions[cursor] < end:
                selected.append(positions[cursor] - offset)
                cursor += 1
            if selected:
                episode = dataset.load_episode(metadata["episode_id"], purpose="diagnostic")
                for index in selected:
                    state = episode.state(index)
                    if not bool(np.all(state["image_valid"])):
                        continue
                    frames.append(np.asarray(state["rgb"], dtype=np.uint8).copy())
                    records.append({"dataset_path": str(Path(dataset.path).resolve()),
                                    "episode_id": metadata["episode_id"], "observation_index": index,
                                    "task_family": family, "collector_kind": metadata["collector_kind"]})
                    families[family] += 1
            offset = end
            if cursor >= len(positions):
                break
    if len(frames) < 2:
        raise ValueError("replay datasets contain too few jointly valid observations")
    values = np.stack(frames)
    if values.shape[1:] != (2, 3, 96, 128):
        raise ValueError("unexpected replay observation shape")
    return values, {"source_kind": "sealed_real_replay", "joint_observations": len(values),
                    "families": dict(families), "records": records,
                    "pixels_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    "scope": "Bounded real-replay sample used only for codec timing; excludes missing images and does not alter dataset partitions or interaction counts."}


def choose_storage_setting(trials: list[dict], *, target_collection_rate: float = 40.0,
                           preferred_chunk_transitions: int = 128) -> dict:
    """Prefer light level1 compression when it saves space and keeps headroom.

    The measured write and sequence-read rates must both exceed twice the
    supplied collection rate. This is a collection-headroom rule, not a GPU
    learner capacity requirement. Read rates are reported for later profiling.
    """
    if not math.isfinite(target_collection_rate) or target_collection_rate <= 0:
        raise ValueError("target_collection_rate must be finite and positive")
    cases = [row for row in trials if row["chunk_transitions"] == preferred_chunk_transitions]
    if not cases:
        raise ValueError("preferred chunk size was not benchmarked")
    baseline = next((row for row in cases if row["compression"] == "stored"), None)
    if baseline is None or baseline["stored_bytes"] <= 0:
        raise ValueError("a stored baseline is required")
    eligible = [row for row in cases if row["compression"] == "deflate"
                and row["stored_bytes"] <= .9 * baseline["stored_bytes"]
                and row["write_joint_observations_per_second"] >= 2 * target_collection_rate
                and row["read_joint_observations_per_second"] >= 2 * target_collection_rate]
    selected = min(eligible, key=lambda row: (row["compression_level"], row["stored_bytes"])) if eligible else baseline
    return {"compression": selected["compression"], "compression_level": selected["compression_level"],
            "chunk_transitions": preferred_chunk_transitions,
            "collection_headroom_threshold_joint_observations_per_second": 2 * target_collection_rate,
            "compressed_setting_qualified": bool(eligible),
            "selection_rule": f"At chunk{preferred_chunk_transitions} prefer the lowest deflate level saving >=10% vs stored with measured durable write and sequence-read rates each >=2x collection rate; otherwise stored. Other chunk sizes are reported comparators.",
            "scope": "Collection storage headroom on this host; eventual model-loader/GPU consumption has not been benchmarked."}


def _chunks(rgb: np.ndarray, size: int) -> list[dict[str, np.ndarray]]:
    # Numeric columns accompany the actual pixels so the benchmark measures the
    # production multi-array codec. They are explicitly dummy timing columns,
    # not state labels or a fabricated learning dataset.
    return [{"rgb": rgb[start:start + size],
             "image_valid": np.ones((len(rgb[start:start + size]), 2), dtype=np.bool_),
             "observation_index": np.arange(start, min(start + size, len(rgb)), dtype=np.int64)}
            for start in range(0, len(rgb), size)]


def _read_window(paths: list[Path], ranges: list[tuple[int, int]], start: int, length: int) -> np.ndarray:
    from world_models.data.replay import read_npz

    result = []
    for path, (low, high) in zip(paths, ranges):
        left, right = max(start, low), min(start + length, high)
        if left < right:
            result.append(read_npz(path)["rgb"][left - low:right - low])
    return np.concatenate(result)


def run_storage_benchmark(output: str | Path, *, sample_root: str | Path | None = None,
                          dataset_paths: list[str | Path] | None = None, cpu_limit: int = 4,
                          minimum_seconds: float = .5, maximum_joint_observations: int = 256,
                          chunk_sizes: tuple[int, ...] = (64, 128), sequence_length: int = 64,
                          target_collection_rate: float = 40.0) -> dict:
    """Benchmark identical samples through the actual replay codec.

    Each timed phase runs at least twice and for the requested duration. Small
    durations/samples produce ``passed_preflight``, never full evidence. Existing
    output directories are refused to preserve previous measured windows.
    """
    from world_models.data.replay import read_npz, write_npz

    if (sample_root is None) == (dataset_paths is None):
        raise ValueError("provide exactly one sample_root or dataset_paths")
    if not math.isfinite(minimum_seconds) or minimum_seconds < 0:
        raise ValueError("minimum_seconds must be finite and nonnegative")
    if not chunk_sizes or any(type(size) is not int or size < 1 for size in chunk_sizes) or len(set(chunk_sizes)) != len(chunk_sizes):
        raise ValueError("chunk sizes must be unique positive integers")
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("use a fresh output directory for each storage benchmark")
    loader = load_actor_samples if sample_root is not None else load_replay_samples
    samples, source = loader(sample_root if sample_root is not None else dataset_paths,
                             maximum_joint_observations=maximum_joint_observations)
    if len(samples) < sequence_length:
        raise ValueError("sample does not contain a full sequence")
    preferred = 128 if 128 in chunk_sizes else max(chunk_sizes)
    destination.mkdir(parents=True)
    trials = []
    try:
        with cpu_budget(cpu_limit) as budget:
            starts = [random.Random(217 + index).randrange(len(samples) - sequence_length + 1) for index in range(64)]
            for size in chunk_sizes:
                chunks = _chunks(samples, size)
                ranges = [(start, min(start + size, len(samples))) for start in range(0, len(samples), size)]
                for compression, level in (("stored", 0), ("deflate", 1), ("deflate", 3)):
                    case = destination / f"chunk{size}_{compression}{level}"
                    case.mkdir()
                    paths = [case / f"chunk{index:04d}.npz" for index in range(len(chunks))]
                    written = 0
                    with ProcessTreeMonitor(budget["cpu_ids"], interval=.05) as write_monitor:
                        began = time.perf_counter()
                        while written < 2 or time.perf_counter() - began < minimum_seconds:
                            for path, arrays in zip(paths, chunks):
                                write_npz(path, arrays, compression=compression, compression_level=level)
                            written += 1
                        write_seconds = time.perf_counter() - began
                    # Equality is an acceptance gate, rather than a reported
                    # flag which can be ignored when choosing compression.
                    for path, expected in zip(paths, chunks):
                        actual = read_npz(path)
                        if set(actual) != set(expected) or any(not np.array_equal(actual[key], values) for key, values in expected.items()):
                            raise RuntimeError("production chunk codec changed benchmark payload")
                    latencies, read_windows = [], 0
                    with ProcessTreeMonitor(budget["cpu_ids"], interval=.05) as read_monitor:
                        began = time.perf_counter()
                        while read_windows < 2 or time.perf_counter() - began < minimum_seconds:
                            start = starts[read_windows % len(starts)]
                            before = time.perf_counter()
                            recovered = _read_window(paths, ranges, start, sequence_length)
                            elapsed = time.perf_counter() - before
                            if not np.array_equal(recovered, samples[start:start + sequence_length]):
                                raise RuntimeError("sequence read differs from its source observations")
                            latencies.append(elapsed)
                            read_windows += 1
                        read_seconds = time.perf_counter() - began
                    stored_bytes = sum(path.stat().st_size for path in paths)
                    trials.append({"chunk_transitions": size, "compression": compression, "compression_level": level,
                                   "payload_pixels_sha256": source["pixels_sha256"], "raw_rgb_bytes": samples.nbytes,
                                   "stored_bytes": stored_bytes, "rgb_to_container_ratio": samples.nbytes / stored_bytes,
                                   "write_passes": written, "write_seconds": write_seconds,
                                   "write_joint_observations_per_second": written * len(samples) / write_seconds,
                                   "write_raw_rgb_MiB_per_second": written * samples.nbytes / write_seconds / 2**20,
                                   "write_resources": write_monitor.report(),
                                   "read_windows": read_windows, "sequence_length": sequence_length, "read_seconds": read_seconds,
                                   "read_joint_observations_per_second": read_windows * sequence_length / read_seconds,
                                   "read_sequence_latency_p50_seconds": statistics.median(latencies),
                                   "read_sequence_latency_p95_seconds": percentile(latencies, .95),
                                   "read_resources": read_monitor.report(), "payload_roundtrip_exact": True,
                                   "chunks": [{"path": str(path.relative_to(destination)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                              for path in paths]})
                    (destination / "trials_progress.json").write_text(json.dumps(trials, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            choice = choose_storage_setting(trials, target_collection_rate=target_collection_rate,
                                            preferred_chunk_transitions=preferred)
        full = (minimum_seconds >= .5 and len(samples) >= 128 and preferred == 128
                and set(source["families"]) == {"relay", "cover"})
        report = {"schema_version": "replay-storage-benchmark-v1", "status": "passed" if full else "passed_preflight",
                  "source": source, "cpu_budget": budget, "minimum_seconds_per_phase": minimum_seconds,
                  "trials": trials, "selected": choice, "all_payloads_exact": True,
                  "timing_scope": "Production NPZ codec, durable/fsynced writes and uncached decompression per sequence; OS page cache remains warm. PNG decoding/source loading, report writes and equality checks are excluded from write timing; read timing includes per-window equality verification.",
                  "metadata_scope": "Real RGB plus dummy validity/index numeric timing columns. Full privileged/event schema, simulator interaction, concurrent GPU training and durable interaction journaling are outside this codec-only benchmark.",
                  "cache_scope": "Reader chunk cache disabled to measure decompression; no OS-cache eviction or direct-I/O claim.",
                  "resource_scope": "This benchmark process and descendants under the recorded logical-CPU budget; original affinity/thread environment restored after timing."}
        path = destination / "storage_benchmark.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n", encoding="ascii")
        return report
    except BaseException as exc:
        (destination / "error.json").write_text(json.dumps({"status": "failed", "error": repr(exc), "completed_trials": len(trials)}, indent=2) + "\n", encoding="utf-8")
        raise
