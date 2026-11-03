"""Measured collector throughput and process-tree resource use for Phase 1."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import os
import json
from pathlib import Path
import statistics
import threading
import time

import psutil


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("percentile requires values and a fraction in [0,1]")
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


@contextmanager
def cpu_budget(cores: int = 4):
    """Constrain this acceptance job and its future children; restore on exit."""
    process = psutil.Process()
    prior = process.cpu_affinity()
    if type(cores) is not int or cores < 1 or len(prior) < cores:
        raise ValueError("the requested CPU budget is unavailable")
    selected = sorted(prior)[:cores]
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous_environment = {name: os.environ.get(name) for name in names}
    try:
        process.cpu_affinity(selected)
        for name in names:
            os.environ[name] = "1"
        yield {"cpu_ids": selected, "logical_core_budget": cores,
               "host_logical_cores": psutil.cpu_count(), "library_threads": 1}
    finally:
        process.cpu_affinity(prior)
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class ProcessTreeMonitor:
    """Sample this job, participant workers and their native game children."""

    def __init__(self, cpu_ids: list[int], interval: float = .1):
        self.root = psutil.Process()
        self.cpu_ids, self.interval = list(cpu_ids), interval
        self.samples: list[dict] = []
        self.start_cpu: dict[tuple[int, float], float] = {}
        self.latest_cpu: dict[tuple[int, float], float] = {}
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread: threading.Thread | None = None

    def sample(self, *, initial=False):
        rss, private, identities = 0, 0, []
        for process in [self.root, *self.root.children(recursive=True)]:
            try:
                with process.oneshot():
                    identity = process.pid, process.create_time()
                    affinity = process.cpu_affinity()
                    if not set(affinity) <= set(self.cpu_ids):
                        process.cpu_affinity(self.cpu_ids)
                        affinity = process.cpu_affinity()
                    memory = process.memory_info()
                    cpu = process.cpu_times()
                    used = cpu.user + cpu.system
                    rss += memory.rss
                    private += getattr(memory, "private", 0)
                    identities.append({"pid": process.pid, "name": process.name(), "cpu_ids": affinity})
                if not initial and identity not in self.start_cpu:
                    raise RuntimeError("a new process appeared during the timed benchmark")
                self.start_cpu.setdefault(identity, used)
                self.latest_cpu[identity] = used
            except psutil.NoSuchProcess:
                continue
        self.samples.append({"monotonic": time.perf_counter(), "rss_bytes": rss,
                             "private_bytes_windows": private, "processes": identities})

    def _run(self):
        try:
            while not self.stop_event.wait(self.interval):
                self.sample()
        except BaseException as exc:
            self.error = exc
            self.stop_event.set()

    def __enter__(self):
        self.sample(initial=True)
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        assert self.thread is not None
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("resource monitor failed to stop")
        self.sample()
        self.elapsed = time.perf_counter() - self.started
        if self.error is not None:
            raise RuntimeError("resource sampling failed") from self.error

    def report(self) -> dict:
        cpu_seconds = sum(value - self.start_cpu[key] for key, value in self.latest_cpu.items())
        return {"cpu_seconds_process_tree": cpu_seconds,
                "cpu_percent_one_core_100": 100 * cpu_seconds / self.elapsed,
                "cpu_percent_of_budget": 100 * cpu_seconds / self.elapsed / len(self.cpu_ids),
                "peak_sum_rss_bytes": max(item["rss_bytes"] for item in self.samples),
                "peak_sum_private_bytes_windows": max(item["private_bytes_windows"] for item in self.samples),
                "sample_interval_seconds": self.interval, "sample_count": len(self.samples),
                "monitor_wall_seconds": self.elapsed,
                "observed_processes": {str(item["pid"]): item for row in self.samples for item in row["processes"]},
                "ram_definition": "Sampled sum of per-process RSS; shared pages can be counted more than once. Windows private memory is also reported.",
                "scope": "Acceptance parent + its descendants only; startup/teardown excluded."}


def choose_collectors(aggregates: dict) -> dict:
    checks = {}
    for task, rows in aggregates.items():
        single, double = rows["1"], rows["2"]
        checks[task] = {
            "throughput_gain_at_least_10_percent": double["median_transitions_per_second"] >= 1.1 * single["median_transitions_per_second"],
            "throughput_cv_at_most_25_percent": double["throughput_coefficient_of_variation"] <= .25,
            "p95_no_more_than_3x_single": double["median_p95_decision_latency_seconds"] <= 3 * single["median_p95_decision_latency_seconds"]}
    baseline_stable = bool(aggregates) and all(rows["1"]["throughput_coefficient_of_variation"] <= .25 for rows in aggregates.values())
    selected = (2 if all(all(row.values()) for row in checks.values()) else 1) if baseline_stable else None
    return {"selected_collectors":selected,"selection_checks":checks,"single_collector_baseline_stable":baseline_stable,
            "status":"passed" if baseline_stable else "failed_unstable_benchmark_baseline"}


def benchmark_collectors(output_dir: str | Path, *, cpu_ids: list[int], steps_per_env: int = 128,
                         repetitions: int = 3, warmup_steps: int = 16) -> dict:
    from world_models.env.coop import TwoPlayerCoopEnv
    from world_models.env.splits import build_split_scenario

    if steps_per_env < 16 or repetitions < 2 or warmup_steps < 0:
        raise ValueError("benchmark requires >=16 steps and >=2 repetitions")
    output = Path(output_dir)
    trials = []
    for task in ("relay", "cover"):
        scenario = build_split_scenario(task, "train", 0, seed=0, purpose="diagnostic",
                                        output_dir=output/"scenarios"/task)
        for repetition in range(repetitions):
            for count in ((1, 2) if repetition % 2 == 0 else (2, 1)):
                environments = []
                startup_seconds = []
                try:
                    for index in range(count):
                        env = TwoPlayerCoopEnv(task, seed=0, scenario_manifest=scenario,
                                               work_root=output/"environments")
                        environments.append(env)
                        startup_started = time.perf_counter()
                        env.reset(seed=7000)
                        startup_seconds.append(time.perf_counter() - startup_started)
                    with ThreadPoolExecutor(max_workers=count) as pool:
                        for _ in range(warmup_steps):
                            results = [future.result() for future in [pool.submit(env.step, 0, 0) for env in environments]]
                            if any(result.terminated for result in results):
                                raise RuntimeError("benchmark warmup terminated unexpectedly")
                        latencies, tics = [], 0
                        physical_hashers = [hashlib.sha256() for _ in environments]
                        def advance(env):
                            before = time.perf_counter()
                            result = env.step(0, 0)
                            return result, time.perf_counter() - before
                        with ProcessTreeMonitor(cpu_ids) as monitor:
                            started = time.perf_counter()
                            for _ in range(steps_per_env):
                                results = [future.result() for future in [pool.submit(advance, env) for env in environments]]
                                for environment_index,(result, latency) in enumerate(results):
                                    if result.terminated or result.truncated:
                                        raise RuntimeError("benchmark episode terminated; workload is no longer matched")
                                    latencies.append(latency)
                                    tics += result.metadata.actual_simulator_tics
                                    info = result.training_only_info
                                    physical_hashers[environment_index].update(json.dumps(
                                        {"players":info["players"],"task_flags":info["task_flags"],"events":info["events"],
                                         "reward":result.team_reward,"boundary":result.boundary.reason.value},
                                        sort_keys=True,separators=(",",":"),allow_nan=False).encode())
                            elapsed = time.perf_counter() - started
                        transitions = len(latencies)
                        trials.append({"task_family": task, "joint_environments": count, "repetition": repetition,
                                       "joint_transitions": transitions, "wall_seconds": elapsed,
                                       "warmup_joint_decisions":warmup_steps*count,
                                       "transitions_per_second": transitions / elapsed,
                                       "joint_tics_per_second": tics / elapsed,
                                       "participant_tics_per_second": 2 * tics / elapsed,
                                       "decision_latency_p50_seconds": percentile(latencies, .5),
                                       "decision_latency_p95_seconds": percentile(latencies, .95),
                                       "latencies_seconds": latencies, "resources": monitor.report(),
                                       "scenario_hash": scenario["hashes"]["wad_sha256"],
                                       "engine_seed":7000,
                                       "physical_workload_trace_sha256":[value.hexdigest() for value in physical_hashers],
                                       "scheduler":"Submit every environment's action before awaiting any result; next batch waits for all environments.",
                                       "startup_reset_seconds_per_env": startup_seconds,
                                       "consumed_joint_decisions": sum(env.coordinator.consumed_joint_decisions for env in environments),
                                       "initialization_joint_decisions": sum(env.initialization_joint_decisions for env in environments),
                                       "coordinator_ledgers": [str(env.coordinator.run_dir/"dispatch_ledger.jsonl") for env in environments],
                                       "startup_warmup_teardown_excluded": True})
                        output.mkdir(parents=True,exist_ok=True)
                        (output/"benchmark_trials.json").write_text(json.dumps(trials,indent=2)+"\n",encoding="utf-8")
                        print(json.dumps({"progress":"benchmark_trial","task":task,"collectors":count,
                                          "repetition":repetition,"transitions_per_second":transitions/elapsed}),flush=True)
                finally:
                    for env in environments:
                        env.close()
    aggregates = {}
    for task in ("relay", "cover"):
        aggregates[task] = {}
        for count in (1, 2):
            cases = [row for row in trials if row["task_family"] == task and row["joint_environments"] == count]
            rates = [row["transitions_per_second"] for row in cases]
            mean = statistics.mean(rates)
            aggregates[task][str(count)] = {
                "median_transitions_per_second": statistics.median(rates),
                "throughput_coefficient_of_variation": statistics.pstdev(rates) / mean,
                "median_p95_decision_latency_seconds": statistics.median(row["decision_latency_p95_seconds"] for row in cases),
                "peak_sum_rss_bytes": max(row["resources"]["peak_sum_rss_bytes"] for row in cases)}
    workload_matches = {task:len({digest for row in trials if row["task_family"] == task
                                  for digest in row["physical_workload_trace_sha256"]}) == 1 for task in ("relay","cover")}
    return {**choose_collectors(aggregates), "trials": trials, "aggregate": aggregates,
            "physical_workload_trace_matches_across_counts_and_repetitions":workload_matches,
            "selection_rule": "Use two only if both families gain >=10% median throughput, CV<=25%, p95<=3x single and all trials have zero infrastructure errors; otherwise retain one.",
            "cpu_ids": cpu_ids, "benchmark_action": [0, 0],
            "workload_control":"All repetitions and collector counts use the identical engine seed 7000 and layout seed0; repetitions vary execution, not the sampled scene.",
            "scope": "Collector environment only, including durable dispatch logging/RGB/targets; no learner overlap or GPU inference."}
