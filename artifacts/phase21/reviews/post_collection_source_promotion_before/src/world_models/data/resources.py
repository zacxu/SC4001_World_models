"""Bound collection to the declared CPU budget, including newly spawned engines."""

from __future__ import annotations

import math
import threading
import time

import psutil


class CollectionResources:
    """Sample this collection job's changing process tree without retaining frames."""

    def __init__(self, cpu_ids: list[int], interval: float = .25):
        self.cpu_ids = list(cpu_ids)
        if (not self.cpu_ids or len(set(self.cpu_ids)) != len(self.cpu_ids)
                or any(type(value) is not int or value < 0 for value in self.cpu_ids)
                or not math.isfinite(interval) or interval <= 0):
            raise ValueError("provide a CPU allocation and positive sample interval")
        self.root = psutil.Process()
        self.interval = interval
        self.stop = threading.Event()
        self.error = None
        self.identities = {}
        self.latest_cpu = {}
        self.first_cpu = {}
        self.peak_rss = 0
        self.samples = 0

    def sample(self):
        rss = 0
        for process in (self.root, *self.root.children(recursive=True)):
            try:
                identity = (process.pid, process.create_time())
                affinity = process.cpu_affinity()
                if not set(affinity) <= set(self.cpu_ids):
                    process.cpu_affinity(self.cpu_ids)
                if not set(process.cpu_affinity()) <= set(self.cpu_ids):
                    raise RuntimeError("collection child escaped its CPU allocation")
                cpu = process.cpu_times()
                used = cpu.user + cpu.system
                self.first_cpu.setdefault(identity, used)
                self.latest_cpu[identity] = used
                self.identities[identity] = process.name()
                rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        self.peak_rss = max(self.peak_rss, rss)
        self.samples += 1

    def _monitor(self):
        try:
            while not self.stop.wait(self.interval):
                self.sample()
        except BaseException as exc:
            self.error = exc
            self.stop.set()

    def __enter__(self):
        self.started = time.perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("collection resource monitor did not stop")
        self.sample()
        self.elapsed = time.perf_counter() - self.started
        if self.error is not None:
            raise RuntimeError("collection resource enforcement failed") from self.error

    def report(self) -> dict:
        cpu_seconds = sum(self.latest_cpu[key] - value for key, value in self.first_cpu.items())
        return {"cpu_ids": self.cpu_ids, "logical_cpu_limit": len(self.cpu_ids),
                "elapsed_seconds": self.elapsed, "sample_count": self.samples,
                "sample_interval_seconds": self.interval, "process_identities_seen": len(self.identities),
                "peak_sum_rss_bytes": self.peak_rss, "cpu_seconds_after_first_sample": cpu_seconds,
                "cpu_percent_of_budget_after_first_sample": 100 * cpu_seconds / self.elapsed / len(self.cpu_ids),
                "scope": "Collection parent and observed descendants, including initialization and teardown.",
                "limitation": "Sampling can miss very short-lived processes and CPU before a child's first sample; summed RSS can double-count shared pages."}
