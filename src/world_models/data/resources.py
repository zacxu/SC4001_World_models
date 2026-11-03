"""Bound collection to the declared CPU budget, including newly spawned engines."""

from __future__ import annotations

import math
from collections import Counter, deque
from datetime import datetime, timezone
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
        self.sample_attempts = 0
        self.started = self.elapsed = self.ended_at = None
        self.thread = None
        self._lock = threading.RLock()
        self._entered = False
        self._last_sample_at = None
        self._first_error_at = None
        self._errors = deque(maxlen=16)
        self._departures = deque(maxlen=32)
        self._departure_counts = Counter()
        self._health_checks = 0
        self._context_error = None

    def _record_failure(self, error, operation, process=None, identity=None):
        with self._lock:
            now = time.perf_counter()
            if self.error is None:
                self.error, self._first_error_at = error, now
            self._errors.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic": now, "operation": operation,
                "pid": getattr(process, "pid", None), "creation_time": identity[1] if identity else None,
                "error_type": type(error).__name__, "error": repr(error)})

    def _departure(self, process, identity, operation, reason):
        with self._lock:
            self._departure_counts[reason] += 1
            self._departures.append({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "operation": operation, "pid": process.pid,
                "creation_time": identity[1] if identity else None, "reason": reason})

    def _proven_departure(self, process):
        # A denial is not evidence of death. Only skip an observed non-root
        # child when a subsequent identity/liveness query proves it gone/reused.
        if process.pid == self.root.pid:
            return False
        try:
            if not process.is_running():
                return True
            return process.status() in (psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            return True
        except (psutil.AccessDenied, AttributeError):
            return False

    def _report_for_error(self):
        try:
            return self.report()
        except BaseException as export_error:
            return {"status": "unavailable", "error_type": type(export_error).__name__,
                    "error": repr(export_error), "observer_error": repr(self.error),
                    "scope": "resource report export failed; no measurements were inferred"}

    def sample(self):
        rss = 0
        with self._lock:
            self.sample_attempts += 1
        try:
            processes = (self.root, *self.root.children(recursive=True))
        except BaseException as error:
            self._record_failure(error, "enumerate_children", self.root)
            raise
        for process in processes:
            identity = None
            operation = "create_time"
            try:
                identity = (process.pid, process.create_time())
                operation = "cpu_affinity_get"
                affinity = process.cpu_affinity()
                if not set(affinity) <= set(self.cpu_ids):
                    operation = "cpu_affinity_set"
                    process.cpu_affinity(self.cpu_ids)
                operation = "cpu_affinity_verify"
                if not set(process.cpu_affinity()) <= set(self.cpu_ids):
                    raise RuntimeError("collection child escaped its CPU allocation")
                operation = "cpu_times"
                cpu = process.cpu_times()
                used = cpu.user + cpu.system
                operation = "name"
                name = process.name()
                operation = "memory_info"
                memory = process.memory_info()
                with self._lock:
                    self.first_cpu.setdefault(identity, used)
                    self.latest_cpu[identity] = used
                    self.identities[identity] = name
                rss += memory.rss
            except psutil.NoSuchProcess:
                if process.pid == self.root.pid:
                    self._record_failure(RuntimeError("resource root identity disappeared"), operation, process, identity)
                    raise
                self._departure(process, identity, operation, "no_such_process")
            except psutil.AccessDenied as error:
                if self._proven_departure(process):
                    self._departure(process, identity, operation, "access_denied_then_confirmed_exited_or_reused")
                else:
                    self._record_failure(error, operation, process, identity)
                    raise
            except BaseException as error:
                self._record_failure(error, operation, process, identity)
                raise
        with self._lock:
            self.peak_rss = max(self.peak_rss, rss)
            self.samples += 1
            self._last_sample_at = time.perf_counter()

    def _monitor(self):
        try:
            while not self.stop.wait(self.interval):
                self.sample()
        except BaseException as exc:
            if self.error is None:
                self._record_failure(exc, "background_sample")
            self.stop.set()

    def __enter__(self):
        if self._entered:
            raise RuntimeError("a resource observer cannot be entered twice")
        self._entered = True
        self.started = time.perf_counter()
        try:
            self.sample()
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()
            self.check_health()
        except BaseException as error:
            if self.error is None:
                self._record_failure(error, "context_enter")
            self.stop.set()
            if self.thread is not None and self.thread.ident is not None:
                try:
                    self.thread.join(timeout=5)
                except BaseException as cleanup_error:
                    self._record_failure(cleanup_error, "context_enter_cleanup")
            self.ended_at = time.perf_counter()
            self.elapsed = self.ended_at - self.started
            raise
        return self

    def check_health(self):
        """Fail before the next dispatch when background observation has failed."""
        with self._lock:
            self._health_checks += 1
            error = self.error
        if error is None and (not self._entered or self.elapsed is not None):
            error = RuntimeError("collection resource observer is not active")
        if error is None and (self.thread is None or not self.thread.is_alive()):
            error = RuntimeError("collection resource sampling thread stopped unexpectedly")
            self._record_failure(error, "check_health")
        if error is not None:
            failure = RuntimeError("collection resource enforcement failed")
            failure.collection_resource_report = self._report_for_error()
            raise failure from error

    def __exit__(self, exception_type, exception, traceback):
        self.stop.set()
        try:
            if self.thread is not None:
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    raise RuntimeError("collection resource monitor did not stop")
            self.sample()
        except BaseException as error:
            if self.error is None:
                self._record_failure(error, "context_exit_sample")
        finally:
            self.ended_at = time.perf_counter()
            self.elapsed = self.ended_at - self.started if self.started is not None else 0.
        if exception is not None:
            self._context_error = {"error_type": type(exception).__name__, "error": repr(exception)}
            if self.error is not None:
                try:
                    exception.collection_resource_report = self._report_for_error()
                    exception.add_note("Collection resource observation also failed; its report is attached as collection_resource_report. The original collection exception remains primary.")
                except BaseException:
                    pass
            return False
        if self.error is not None:
            failure = RuntimeError("collection resource enforcement failed")
            failure.collection_resource_report = self._report_for_error()
            raise failure from self.error
        return False

    def report(self) -> dict:
        with self._lock:
            elapsed = self.elapsed if self.elapsed is not None else time.perf_counter() - self.started if self.started is not None else 0.
            cpu_seconds = sum(self.latest_cpu[key] - value for key, value in self.first_cpu.items())
            end = self.ended_at if self.ended_at is not None else time.perf_counter()
            return {"status": "failed" if self.error is not None else "complete" if self.elapsed is not None else "observing" if self.started is not None else "not_started",
                "cpu_ids": self.cpu_ids, "logical_cpu_limit": len(self.cpu_ids),
                "elapsed_seconds": elapsed, "sample_count": self.samples, "sample_attempts": self.sample_attempts,
                "sample_interval_seconds": self.interval, "process_identities_seen": len(self.identities),
                "peak_sum_rss_bytes": self.peak_rss if self.samples else None,
                "cpu_seconds_after_first_sample": cpu_seconds if self.samples else None,
                "cpu_percent_of_budget_after_first_sample": 100 * cpu_seconds / elapsed / len(self.cpu_ids) if elapsed > 0 and self.samples else None,
                "health_check_calls": self._health_checks,
                "last_successful_sample_monotonic": self._last_sample_at,
                "first_observer_error_monotonic": self._first_error_at,
                "coverage_gap_seconds_after_observer_error": max(0., end-self._first_error_at) if self._first_error_at is not None else 0.,
                "observer_error_type": type(self.error).__name__ if self.error is not None else None,
                "observer_error": repr(self.error) if self.error is not None else None,
                "observer_error_records": list(self._errors),
                "departed_process_observation_counts": dict(self._departure_counts),
                "recent_departed_process_observations": list(self._departures),
                "collection_context_exception": self._context_error,
                "measurement_complete": self.elapsed is not None and self.error is None and self.samples > 0,
                "scope": "Collection parent and observed descendants, including initialization and teardown.",
                "limitation": "Sampling can miss short-lived processes and CPU before a child's first sample; RSS can double-count shared pages. After observer failure the unsampled interval is a coverage gap, not proof of a CPU breach or continuous successful enforcement. Call check_health before each real reset/step."}
