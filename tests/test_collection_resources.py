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
    def is_running(self): return True
    def status(self): return psutil.STATUS_RUNNING

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

    def test_access_denial_is_skipped_only_after_confirmed_child_exit_or_pid_reuse(self):
        for confirmation in ("not_running", "no_such_process", "dead"):
            with self.subTest(confirmation=confirmation):
                root, child = Process(1), Process(2)
                root.children_list = [child]
                child.cpu_affinity = Mock(side_effect=psutil.AccessDenied(2))
                if confirmation == "not_running": child.is_running = Mock(return_value=False)
                if confirmation == "no_such_process": child.is_running = Mock(side_effect=psutil.NoSuchProcess(2))
                if confirmation == "dead": child.status = Mock(return_value=psutil.STATUS_DEAD)
                with patch("world_models.data.resources.psutil.Process", return_value=root):
                    resources = CollectionResources([2,3])
                resources.sample()
                report = resources.report()
                self.assertIsNone(resources.error)
                self.assertEqual(report["departed_process_observation_counts"],
                                 {"access_denied_then_confirmed_exited_or_reused": 1})
                self.assertEqual(report["sample_count"], 1)
                self.assertEqual(report["peak_sum_rss_bytes"], 100)

    def test_live_or_uninspectable_access_denied_child_is_fatal_and_recorded(self):
        for inspectable in (True, False):
            with self.subTest(inspectable=inspectable):
                root, child = Process(1), Process(2)
                root.children_list = [child]
                failure = psutil.AccessDenied(2)
                child.cpu_affinity = Mock(side_effect=failure)
                if not inspectable: child.is_running = Mock(side_effect=psutil.AccessDenied(2))
                with patch("world_models.data.resources.psutil.Process", return_value=root):
                    resources = CollectionResources([2,3])
                with self.assertRaises(psutil.AccessDenied): resources.sample()
                report = resources.report()
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["measurement_complete"])
                self.assertEqual(report["observer_error_records"][0]["pid"], 2)
                self.assertEqual(report["observer_error_records"][0]["operation"], "cpu_affinity_get")
                self.assertEqual(report["departed_process_observation_counts"], {})
                self.assertIs(resources.error, failure)

    def test_root_access_denial_is_not_misclassified_as_departed_child(self):
        root = Process(1)
        root.cpu_affinity = Mock(side_effect=psutil.AccessDenied(1))
        root.is_running = Mock(return_value=False)
        with patch("world_models.data.resources.psutil.Process", return_value=root):
            resources = CollectionResources([2,3])
        with self.assertRaises(psutil.AccessDenied): resources.sample()
        self.assertEqual(resources.report()["departed_process_observation_counts"], {})

    def test_report_is_available_before_start_and_after_failed_enter(self):
        root = Process(1)
        with patch("world_models.data.resources.psutil.Process", return_value=root):
            resources = CollectionResources([2,3])
        before = resources.report()
        self.assertEqual(before["status"], "not_started")
        self.assertIsNone(before["peak_sum_rss_bytes"])
        self.assertIsNone(before["cpu_percent_of_budget_after_first_sample"])
        failure = RuntimeError("could not start sampler thread")
        with patch("world_models.data.resources.threading.Thread") as constructor:
            constructor.return_value.start.side_effect = failure
            constructor.return_value.ident = None
            with self.assertRaisesRegex(RuntimeError, "sampler thread"): resources.__enter__()
        report = resources.report()
        self.assertEqual(report["status"], "failed")
        self.assertIsNotNone(resources.elapsed)
        self.assertEqual(report["observer_error_records"][0]["operation"], "context_enter")

    def test_original_collection_exception_survives_observer_and_export_failure(self):
        for export_failure in (False, True):
            with self.subTest(export_failure=export_failure):
                root = Process(1)
                with patch("world_models.data.resources.psutil.Process", return_value=root):
                    resources = CollectionResources([2,3])
                resources.started, resources._entered = 1., True
                resources.thread = Mock()
                resources.thread.is_alive.return_value = False
                original = ValueError("original native collection failure")
                patches = [patch.object(CollectionResources, "__enter__", return_value=resources),
                           patch.object(resources, "sample", side_effect=psutil.AccessDenied(2))]
                from contextlib import ExitStack
                with ExitStack() as stack:
                    for item in patches: stack.enter_context(item)
                    if export_failure: stack.enter_context(patch.object(resources, "report", side_effect=RuntimeError("secondary export failure")))
                    with self.assertRaises(ValueError) as caught:
                        with resources:
                            raise original
                self.assertIs(caught.exception, original)
                self.assertIsNotNone(resources.elapsed)
                self.assertEqual(original.collection_resource_report["status"], "unavailable" if export_failure else "failed")
                self.assertTrue(any("original collection exception remains primary" in note for note in original.__notes__))

    def test_health_guard_prevents_next_action_after_sampler_failure(self):
        root = Process(1)
        with patch("world_models.data.resources.psutil.Process", return_value=root):
            resources = CollectionResources([2,3])
        resources.started, resources._entered = 1., True
        resources.thread = Mock()
        resources.thread.is_alive.return_value = True
        failure = psutil.AccessDenied(2)
        resources._record_failure(failure, "cpu_affinity_get", Process(2), (2,1.))
        dispatch = Mock()
        with self.assertRaisesRegex(RuntimeError, "enforcement failed") as caught:
            resources.check_health()
            dispatch()
        dispatch.assert_not_called()
        self.assertIs(caught.exception.__cause__, failure)
        self.assertFalse(caught.exception.collection_resource_report["measurement_complete"])

    def test_unexpected_stopped_monitor_is_not_silently_healthy(self):
        root = Process(1)
        with patch("world_models.data.resources.psutil.Process", return_value=root):
            resources = CollectionResources([2,3])
        resources.started, resources._entered = 1., True
        resources.thread = Mock()
        resources.thread.is_alive.return_value = False
        with self.assertRaisesRegex(RuntimeError, "enforcement failed"):
            resources.check_health()
        self.assertIn("stopped unexpectedly", resources.report()["observer_error"])

    def test_error_coverage_gap_is_reported_without_claiming_continuous_enforcement(self):
        root = Process(1)
        with patch("world_models.data.resources.psutil.Process", return_value=root):
            resources = CollectionResources([2,3])
        resources.started = 100.
        with patch("world_models.data.resources.time.perf_counter", return_value=110.):
            resources._record_failure(psutil.AccessDenied(2), "cpu_affinity_get", Process(2), (2,1.))
        resources.ended_at, resources.elapsed = 113., 13.
        report = resources.report()
        self.assertEqual(report["coverage_gap_seconds_after_observer_error"], 3.)
        self.assertEqual(report["first_observer_error_monotonic"], 110.)
        self.assertFalse(report["measurement_complete"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["observer_error_records"][0]["creation_time"], 1.)


if __name__ == "__main__":
    unittest.main()
