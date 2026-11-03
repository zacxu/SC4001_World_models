"""Read-only production replay benchmark on already sealed 50k stores.

No simulator is called and no audited file is written. Report output must remain
outside the datasets. NPZ verification/prefetch and metadata-open costs are
reported separately from the timed production reader accesses.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import statistics
import sys
import threading
import time
import traceback

THREAD_NAMES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS")
for name in THREAD_NAMES:
    os.environ[name] = "1"

import psutil


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def snapshot(directory):
    return {"collection_report_sha256": sha256(directory / "collection_report.json"),
            "dataset_manifest_sha256": sha256(directory / "replay/manifest.json"),
            "training_ledger_sha256": sha256(directory / "training_interactions.jsonl")}


def native_process_snapshot(project):
    expected = (project / ".venv/Lib/site-packages/vizdoom/vizdoom.exe").resolve()
    rows = []
    for process in psutil.process_iter(["pid", "name", "exe", "create_time"]):
        try:
            info = process.info
            if info["name"] and info["name"].lower() == "vizdoom.exe" and info["exe"] and Path(info["exe"]).resolve() == expected:
                directory = process.cwd()
                match = re.search(r"(?:relay|cover)-s\d+-e\d+", directory)
                rows.append({"pid": info["pid"], "create_time": info["create_time"], "cpu_ids": process.cpu_affinity(),
                             "episode_id_from_cwd": match[0] if match else None})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return rows


class RSSMonitor:
    def __init__(self, process, interval=.02):
        self.process, self.interval = process, interval
        self.stop = threading.Event()
        self.peak, self.samples, self.error = 0, 0, None

    def sample(self):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.samples += 1

    def loop(self):
        try:
            while not self.stop.wait(self.interval):
                self.sample()
        except BaseException as error:
            self.error = error

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("reader memory sampler did not stop")
        self.sample()
        if self.error:
            raise RuntimeError("reader memory sampling failed") from self.error


def select_requests(episodes, *, seed, scattered=128, windows=8, window_length=50):
    fitting = [episode for episode in episodes if episode["partition"] == "fit"]
    rng = random.Random(seed)
    ends, count = [], 0
    for episode in fitting:
        count += episode["valid_length"]
        ends.append(count)
    if count < scattered:
        raise ValueError("not enough fitting source states for the requested shuffled sample")
    independent = []
    for position in rng.sample(range(count), scattered):
        index = bisect.bisect_right(ends, position)
        low = ends[index - 1] if index else 0
        independent.append((fitting[index], position - low, 1))
    eligible = [episode for episode in fitting if episode["valid_length"] >= window_length]
    window_ends, count = [], 0
    for episode in eligible:
        count += episode["valid_length"] - window_length + 1
        window_ends.append(count)
    if count < windows:
        raise ValueError("not enough fitting within-episode windows")
    recurrent = []
    for position in rng.sample(range(count), windows):
        index = bisect.bisect_right(window_ends, position)
        low = window_ends[index - 1] if index else 0
        recurrent.append((eligible[index], position - low, window_length))
    return {"shuffled_single_states": independent, "contiguous_50_state_windows": recurrent}


def touched_chunks(episode, start, length):
    chosen = [chunk for chunk in episode["chunks"]
              if max(start, chunk["state_start"]) < min(start + length, chunk["state_start"] + chunk["state_count"])]
    covered = sum(min(start + length, chunk["state_start"] + chunk["state_count"])
                  - max(start, chunk["state_start"]) for chunk in chosen)
    if covered != length:
        raise ValueError("request does not form one contiguous observation sequence")
    return chosen


def run_case(directory, *, project, process, reader_class, numpy):
    anchor_start = time.perf_counter()
    before = snapshot(directory)
    anchor_seconds = time.perf_counter() - anchor_start
    begin = time.perf_counter()
    manifest = json.loads((directory / "replay/manifest.json").read_text(encoding="utf-8"))
    manifest_seconds = time.perf_counter() - begin
    collection = json.loads((directory / "collection_report.json").read_text(encoding="utf-8"))
    if (collection["status"] != "complete_initial_replay" or manifest["transition_count"] != 50000
            or not manifest["metadata"]["learning_eligible"] or collection["published_joint_transitions"] != 50000):
        raise ValueError("this full-scale reader measurement requires a sealed, completed, learning-eligible 50k store")
    if before["dataset_manifest_sha256"] != collection["dataset_manifest_sha256"] or before["training_ledger_sha256"] != collection["interaction_accounting"]["ledger_sha256"]:
        raise ValueError("dataset metadata anchors disagree with its collection report")
    rss_before_open = process.memory_info().rss
    with RSSMonitor(process) as opening_memory:
        begin = time.perf_counter()
        reader = reader_class(directory / "replay", verify=False, cache_chunks=4)
        open_seconds = time.perf_counter() - begin
    if reader.calls or reader._cache:
        raise ValueError("metadata-only reader construction unexpectedly loaded NPZ chunks")
    plans = select_requests(reader.episodes, seed=21721 + (0 if collection["task_family"] == "relay" else 1))
    all_touched = {}
    plan_records = {}
    for name, requests in plans.items():
        plan_records[name] = []
        for episode, start, length in requests:
            chunks = touched_chunks(episode, start, length)
            all_touched.update({chunk["path"]: chunk for chunk in chunks})
            plan_records[name].append({"episode_id": episode["episode_id"], "start": start, "length": length,
                "collector_kind": episode["collector_kind"], "chunk_paths": [chunk["path"] for chunk in chunks]})
    begin = time.perf_counter()
    warmed_bytes = 0
    for relative, chunk in sorted(all_touched.items()):
        path = directory / "replay" / relative
        if sha256(path) != chunk["sha256"]:
            raise ValueError("a selected production chunk differs from its previously audited manifest")
        warmed_bytes += path.stat().st_size
    prefetch_seconds = time.perf_counter() - begin
    results = {}
    for name, requests in plans.items():
        reader.clear_measurement()
        gc.collect()
        digest = hashlib.sha256()
        latencies = []
        joint_observations = 0
        rss_before = process.memory_info().rss
        cpu_before = time.process_time()
        with RSSMonitor(process) as memory:
            begin = time.perf_counter()
            for episode, start, length in requests:
                request_start = time.perf_counter()
                view = reader.load_episode(episode["episode_id"], purpose="world_model_fit")
                frames = []
                for index in range(start, start + length):
                    state = view.state(index)
                    if not bool(state["image_valid"].all()):
                        raise ValueError("a fitting action-source state lacks either actual view")
                    frames.append(state["rgb"])
                pixels = numpy.stack(frames)
                latencies.append(time.perf_counter() - request_start)
                if pixels.dtype != numpy.uint8 or pixels.shape != (length, 2, 3, 96, 128):
                    raise ValueError("reader changed its joint two-view uint8 contract")
                digest.update(pixels.tobytes())
                joint_observations += length
                del pixels, frames, state, view
            wall_seconds = time.perf_counter() - begin
        cpu_seconds = time.process_time() - cpu_before
        if len(reader._cache) > 4 or reader.peak_cached_chunks > 4:
            raise ValueError("production reader exceeded its configured four-chunk LRU")
        result = {"requests": len(requests), "joint_observations_returned": joint_observations,
            "two_rgb_views_counted_per_joint_observation": 2, "wall_seconds": wall_seconds,
            "joint_observations_per_second": joint_observations / wall_seconds,
            "requests_per_second": len(requests) / wall_seconds,
            "returned_rgb_MiB_per_second": joint_observations * 73728 / wall_seconds / 2**20,
            "request_latency_p50_seconds": statistics.median(latencies),
            "request_latency_p95_seconds": percentile(latencies, .95),
            "request_latency_min_seconds": min(latencies), "request_latency_max_seconds": max(latencies),
            "sum_request_service_seconds": sum(latencies), "returned_pixels_sha256": digest.hexdigest(),
            "cache_capacity_chunks": 4, "cache_initial_state": "empty", "chunk_cache_calls": reader.calls,
            "chunk_cache_hits": reader.hits, "chunk_cache_misses": reader.misses,
            "chunk_cache_hit_fraction": reader.hits / reader.calls,
            "unique_chunk_paths_requested": len({path for request in plan_records[name] for path in request["chunk_paths"]}),
            "unique_chunk_paths_loaded": len(reader.loaded), "compressed_bytes_on_cache_misses": reader.compressed_bytes,
            "decoded_array_bytes_on_cache_misses": reader.decoded_bytes,
            "decoded_array_bytes_per_returned_rgb_byte": reader.decoded_bytes / (joint_observations * 73728),
            "peak_cached_chunks": reader.peak_cached_chunks, "peak_cached_array_bytes": reader.peak_cached_bytes,
            "process_rss_before_bytes": rss_before, "sampled_peak_process_rss_bytes": memory.peak,
            "rss_sample_interval_seconds": memory.interval, "rss_sample_count": memory.samples,
            "process_cpu_seconds": cpu_seconds, "process_cpu_percent_of_four_core_budget": 100 * cpu_seconds / wall_seconds / 4,
            "unique_episodes_requested": len({request[0]["episode_id"] for request in requests}),
            "windows_crossing_chunk_boundary": sum(len(record["chunk_paths"]) > 1 for record in plan_records[name]),
            "joint_observation_counts_by_collector_kind": dict(Counter({kind: sum(length for episode, _, length in requests
                       if episode["collector_kind"] == kind) for kind in ("random", "scripted")})),
            "request_plan": plan_records[name], "request_latencies_seconds": latencies}
        results[name] = result
        reader.clear_measurement()
    after = snapshot(directory)
    if after != before:
        raise ValueError("immutable dataset report/manifest/ledger changed during reader measurements")
    return {"status": "passed", "dataset_path": str(directory), "task_family": collection["task_family"],
        "run_seed": collection["run_seed"], "dataset_snapshot_sha256_at_start": before,
        "dataset_snapshot_sha256_at_end": after, "dataset_snapshot_unchanged": True,
        "dataset_identity_sha256": manifest["dataset_identity_sha256"], "real_joint_transitions": manifest["transition_count"],
        "stored_joint_observation_slots": sum(episode["observation_count"] for episode in reader.episodes),
        "episode_count": len(reader.episodes), "total_chunks": sum(len(episode["chunks"]) for episode in reader.episodes),
        "fitting_joint_transitions_available": sum(episode["valid_length"] for episode in reader.episodes if episode["partition"] == "fit"),
        "storage": manifest["storage"], "hash_anchor_read_seconds_excluded_from_throughput": anchor_seconds,
        "manifest_json_read_parse_seconds": manifest_seconds,
        "first_measured_reader_open_metadata_validation_seconds": open_seconds,
        "reader_open_process_rss_before_bytes": rss_before_open,
        "reader_open_sampled_peak_process_rss_bytes": opening_memory.peak,
        "metadata_open_scope": "verify=False still validates dataset/episode manifests, schema, identities and counts; full NPZ hash/content verification is excluded. The manifest-only phase and earlier hash anchors have already touched metadata.",
        "selected_chunks_hashed_and_prefetched": len(all_touched),
        "compressed_bytes_hashed_and_prefetched": warmed_bytes,
        "selected_chunk_hash_and_prefetch_seconds_excluded_from_throughput": prefetch_seconds,
        "workloads": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(r"D:\world models"))
    parser.add_argument("--datasets", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("production_reader_benchmark.json"))
    args = parser.parse_args()
    project = args.project_root.resolve()
    datasets = [path.resolve() for path in args.datasets] if args.datasets else [project / "data/initial_replay/pilot" / family / "seed_0" for family in ("relay", "cover")]
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(path) for path in datasets):
        raise ValueError("use a fresh output path outside every immutable dataset")
    process = psutil.Process()
    prior = process.cpu_affinity()
    if not {0, 1, 2, 3} <= set(prior):
        raise RuntimeError("CPU0..3 allocation unavailable")
    process.cpu_affinity([0, 1, 2, 3])
    report = {"schema_version": "production-reader-benchmark-v1", "status": "running",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(__file__), "cpu_ids": [0, 1, 2, 3],
        "thread_environment": {name: os.environ[name] for name in THREAD_NAMES},
        "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                     "host_logical_cpus": psutil.cpu_count(), "host_memory_bytes": psutil.virtual_memory().total,
                     "available_memory_bytes_at_start": psutil.virtual_memory().available},
        "native_jobs_at_start": native_process_snapshot(project), "cases": [],
        "read_integrity_scope": "Known previously fully audited sealed datasets; anchors are rebound before/after. Selected compressed chunks are SHA-verified while prefetching outside timed reads. Production verify=False reads still perform typed-array and validity checks.",
        "cache_scope": "Selected compressed chunks are deliberately prefetched, intending warm OS cache. OS residency/eviction is not enforced or measured. The production decompressed-chunk LRU is cleared at each workload start, then retains at most four chunks.",
        "timing_scope": "Throughput includes the complete request loop, uint8 view stacking, image-validity assertions and returned-pixel SHA consumption. Per-request latency excludes the returned-pixel SHA update. Metadata open, source hashes, chunk hashes and prefetch are separately reported and excluded from throughput.",
        "contention_scope": "Measured concurrently with the recorded native collection jobs sharing logical CPUs0..3. These are observed contended-host rates, not dedicated-host or OS-cold rates.",
        "sampling_scope": "128 distinct uniformly shuffled fitting action-source states, plus eight uniformly sampled valid within-episode 50-state windows. Terminal next-state slots and model-diagnostic episodes are excluded. Both RGB views form one joint observation.",
        "limitations": ["Only two completed family/seed stores and a modest sample are measured; there is no learning-performance claim.",
            "Scalar state() calls exercise today's production API; a later batched loader can have different performance.",
            "No float32 preprocessing, PyTorch, GPU transfer, model training, actor inference or network data path is simulated.",
            "RSS is sampled process memory including Python/NumPy/metadata and temporary buffers, not an isolated cache-only measurement.",
            "Warm-OS-cache timing does not establish cold disk throughput. Native collection competes for the same CPU allocation."]}
    began = time.perf_counter()
    try:
        # Affinity/thread settings above precede NumPy and the production reader.
        sys.path.insert(0, str(project / "src"))
        import numpy as np
        import world_models.data.replay as production
        if Path(production.__file__).resolve() != project / "src/world_models/data/replay.py":
            raise ValueError("benchmark imported a different replay reader checkout")
        report["production_reader_path"] = str(Path(production.__file__).resolve())
        report["production_reader_sha256_at_start"] = sha256(production.__file__)
        report["packages"] = {name: importlib.metadata.version(name) for name in ("numpy", "psutil", "PyYAML")}

        class MeasuredReader(production.ReplayDataset):
            def __init__(self, *args, **kwargs):
                self.calls = self.hits = self.misses = self.compressed_bytes = self.decoded_bytes = 0
                self.loaded = set()
                self.peak_cached_chunks = self.peak_cached_bytes = 0
                super().__init__(*args, **kwargs)

            def clear_measurement(self):
                self._cache.clear()
                self.calls = self.hits = self.misses = self.compressed_bytes = self.decoded_bytes = 0
                self.loaded = set()
                self.peak_cached_chunks = self.peak_cached_bytes = 0

            def _chunk(self, chunk):
                hit = chunk["path"] in self._cache
                self.calls += 1
                self.hits += int(hit)
                self.misses += int(not hit)
                arrays = super()._chunk(chunk)
                if not hit:
                    self.loaded.add(chunk["path"])
                    self.compressed_bytes += chunk["bytes"]
                    self.decoded_bytes += sum(array.nbytes for array in arrays.values())
                    self.peak_cached_bytes = max(self.peak_cached_bytes,
                        sum(array.nbytes for cached in self._cache.values() for array in cached.values()))
                self.peak_cached_chunks = max(self.peak_cached_chunks, len(self._cache))
                return arrays

        for dataset in datasets:
            report["cases"].append(run_case(dataset, project=project, process=process,
                                           reader_class=MeasuredReader, numpy=np))
        report["production_reader_sha256_at_end"] = sha256(production.__file__)
        if report["production_reader_sha256_at_end"] != report["production_reader_sha256_at_start"]:
            raise ValueError("production reader execution source changed during measurement")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        report["wall_seconds"] = time.perf_counter() - began
        report["native_jobs_at_end"] = native_process_snapshot(project)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        output.with_suffix(".sha256").write_text(sha256(output) + "  " + output.name + "\n", encoding="ascii")
        process.cpu_affinity(prior)
    summary = {"status": report["status"], "output": str(output), "report_sha256": sha256(output),
        "cases": [{"family": case["task_family"], "open_seconds": case["first_measured_reader_open_metadata_validation_seconds"],
                    "workloads": {name: {key: row[key] for key in ("joint_observations_per_second", "request_latency_p50_seconds",
                        "request_latency_p95_seconds", "chunk_cache_misses", "unique_chunk_paths_loaded", "sampled_peak_process_rss_bytes")}
                        for name, row in case["workloads"].items()}} for case in report["cases"]], "error": report.get("error")}
    print(json.dumps(summary), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
