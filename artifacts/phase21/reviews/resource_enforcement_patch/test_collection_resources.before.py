"""CPU enforcement and changing-process accounting without timing assertions."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import psutil

from world_models.data.resources import CollectionResources


class Process:
    def __init__(self, pid, created=1., *, cpus=(2, 3), cpu=0., rss=100, ignores_affinity=False):
        self.pid, self.created = pid, created
        self.affinity, self.cpu, self.rss = list(cpus), cpu, rss
        self.ignores_affinity, self.children_list = ignores_affinity, []

    def create_time(self): return self.created
    def children(self, recursive=True): return list(self.children_list)
    def name(self): return f"fixture-{self.pid}"
    def cpu_times(self): return SimpleNamespace(user=self.cpu, system=0.)
    def memory_info(self): return SimpleNamespace(rss=self.rss)

    def cpu_affinity(self, value=None):
        if value is not None and not self.ignores_affinity:
            self.affinity = list(value)
        return list(self.affinity)


class CollectionResourceTests(unittest.TestCase):
    def test_invalid_allocations_and_nonfinite_intervals_are_rejected(self):
        for ids in ([], [1, 1], [-1], [True], ["0"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                CollectionResources(ids)
        for interval in (0., -1., float("nan"), float("inf")):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                CollectionResources([0], interval)

    def test_new_children_are_constrained_and_pid_reuse_does_not_lose_cpu(self):
        root = Process(1, cpu=5)
        old_child = Process(2, created=1., cpus=(0, 1, 2, 3), cpu=10, rss=200)
        root.children_list = [old_child]
        with patch("world_models.data.resources.psutil.Process", return_value=root):
            resources = CollectionResources([2, 3])
        resources.sample()
        self.assertEqual(old_child.affinity, [2, 3])
        root.cpu, old_child.cpu = 9, 15
        resources.sample()
        replacement = Process(2, created=2., cpus=(0, 1), cpu=100, rss=250)
        root.children_list = [replacement]
        resources.sample()
        root.cpu, replacement.cpu = 11, 101
        resources.sample()
        resources.elapsed = 2.
        report = resources.report()
        self.assertEqual(replacement.affinity, [2, 3])
        self.assertEqual(report["process_identities_seen"], 3)
        self.assertEqual(report["cpu_seconds_after_first_sample"], 12.)
        self.assertEqual(report["peak_sum_rss_bytes"], 350)
        self.assertEqual(report["sample_count"], 4)

    def test_unenforceable_child_affinity_fails_closed(self):
        root = Process(1)
        root.children_list = [Process(2, cpus=(0, 1), ignores_affinity=True)]
        with patch("world_models.data.resources.psutil.Process", return_value=root):
            resources = CollectionResources([2, 3])
        with self.assertRaisesRegex(RuntimeError, "escaped"):
            resources.sample()

    def test_vanished_process_is_skipped_but_access_denial_is_not_hidden(self):
        root, child = Process(1), Process(2)
        root.children_list = [child]
        child.memory_info = Mock(side_effect=psutil.NoSuchProcess(2))
        with patch("world_models.data.resources.psutil.Process", return_value=root):
            resources = CollectionResources([2, 3])
        resources.sample()
        self.assertEqual(resources.peak_rss, 100)
        child.memory_info = Mock(side_effect=psutil.AccessDenied(2))
        with self.assertRaises(psutil.AccessDenied):
            resources.sample()

    def test_background_enforcement_error_is_reported_at_context_exit(self):
        root = Process(1)
        with patch("world_models.data.resources.psutil.Process", return_value=root):
            resources = CollectionResources([2, 3])
        failure = psutil.AccessDenied(2)
        with patch.object(resources.stop, "wait", return_value=False), patch.object(resources, "sample", side_effect=failure):
            resources._monitor()
        self.assertIs(resources.error, failure)
        self.assertTrue(resources.stop.is_set())
        resources.started = 0.
        resources.thread = Mock()
        resources.thread.is_alive.return_value = False
        with patch.object(resources, "sample"), patch("world_models.data.resources.time.perf_counter", return_value=2.):
            with self.assertRaisesRegex(RuntimeError, "enforcement failed"):
                resources.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
