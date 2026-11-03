"""Fail-closed P2.1 orchestration without native collection or large disk writes."""

from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import phase21


STORAGE = {"compression": "deflate", "compression_level": 1, "chunk_transitions": 128}
RUNTIME = {"packages": {"numpy": {"matches": True}}, "engine": "pinned-test-engine", "recorded_at_utc": "test"}
SOURCES = {"src/example.py": "a"*64}


@contextmanager
def fake_cpu_budget(count):
    if count != 4:
        raise AssertionError("orchestration must enforce the approved four-core budget")
    yield {"cpu_ids": [0, 1, 2, 3]}


class FakeResources:
    def __init__(self, cpu_ids):
        self.cpu_ids = cpu_ids

    def __enter__(self): return self
    def __exit__(self, *args): return False
    def report(self): return {"logical_cpu_limit": len(self.cpu_ids), "cpu_ids": self.cpu_ids}
    def check_health(self): return None


class Phase21CLITests(unittest.TestCase):
    def fake_collection(self, cases, directory, *, profile, transitions, **kwargs):
        return [{"path": str(directory/family/f"seed_{seed}"), "reused": False,
                 "report": {"task_family": family, "run_seed": seed, "target_transitions": transitions, "profile": profile}}
                for family, seed in cases]

    def fake_audit(self, directory, *, expected_transitions, **kwargs):
        directory = Path(directory)
        return {"status": "passed", "task_family": directory.parent.name,
                "run_seed": int(directory.name.removeprefix("seed_")), "transition_count": expected_transitions}

    def mocks(self, stack):
        def disk_check(path, *args):
            Path(path).mkdir(parents=True, exist_ok=True)
            return {"free_bytes_at_start": 10**12}
        values = {
            "execution_sources": stack.enter_context(patch.object(phase21, "execution_sources", return_value=deepcopy(SOURCES))),
            "runtime_manifest": stack.enter_context(patch.object(phase21, "runtime_manifest", return_value=deepcopy(RUNTIME))),
            "check_disk": stack.enter_context(patch.object(phase21, "check_disk", side_effect=disk_check)),
            "unit_suite": stack.enter_context(patch.object(phase21, "unit_suite", return_value={"status": "passed"})),
            "collect_cases": stack.enter_context(patch.object(phase21, "collect_cases", side_effect=self.fake_collection)),
            "audit_initial_dataset": stack.enter_context(patch.object(phase21, "audit_initial_dataset", side_effect=self.fake_audit)),
            "run_storage_benchmark": stack.enter_context(patch.object(phase21, "run_storage_benchmark", return_value={
                "status": "passed", "selected": deepcopy(STORAGE), "all_payloads_exact": True})),
            "benchmark_fsync_directories": stack.enter_context(patch.object(phase21, "benchmark_fsync_directories", return_value={"status":"passed"})),
        }
        stack.enter_context(patch.object(phase21, "cpu_budget", side_effect=fake_cpu_budget))
        stack.enter_context(patch.object(phase21, "CollectionResources", FakeResources))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        return values

    def arguments(self, directory, command="complete", *extra):
        return [command, "--output-dir", str(Path(directory)/"reports"), "--data-dir", str(Path(directory)/"data"),
                "--journal-root", str(Path(directory)/"hot_journals"), *extra]

    def test_default_case_matrix_and_profile_seeds_are_exact(self):
        self.assertEqual(phase21.dataset_cases("pilot", "both", None), [("relay", 0), ("relay", 1), ("cover", 0), ("cover", 1)])
        self.assertEqual(phase21.dataset_cases("pilot", "cover", [1]), [("cover", 1)])
        final = phase21.dataset_cases("final", "both", None)
        self.assertEqual(len(final), 20)
        self.assertEqual({seed for _, seed in final}, set(range(1001, 1011)))
        for seeds in ([], [0, 0], [1001], [True]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                phase21.dataset_cases("pilot", "both", seeds)

    def test_complete_refuses_partial_matrix_or_reduced_budget_before_native_work(self):
        cases = (("--task", "relay"), ("--seeds", "0"), ("--transitions", "1000"), ("--profile", "final"))
        for extra in cases:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                mocks = self.mocks(stack)
                with self.assertRaises(SystemExit) as error:
                    phase21.main(self.arguments(directory, "complete", *extra))
                self.assertEqual(error.exception.code, 2)
                mocks["collect_cases"].assert_not_called()
                mocks["runtime_manifest"].assert_not_called()

    def test_invalid_quotas_and_profile_mismatched_seeds_are_usage_errors(self):
        for extra in (("--transitions", "0"), ("--transitions", "11"), ("--transitions", "100000"),
                      ("--seeds", "1001"), ("--seeds", "0", "0")):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                mocks = self.mocks(stack)
                with self.assertRaises(SystemExit) as error:
                    phase21.main(self.arguments(directory, "collect", *extra))
                self.assertEqual(error.exception.code, 2)
                mocks["collect_cases"].assert_not_called()

    def test_existing_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            mocks = self.mocks(stack)
            target = Path(directory)/"reports/phase21_acceptance.json"
            target.parent.mkdir()
            target.write_bytes(b"original acceptance evidence\n")
            with self.assertRaises(SystemExit):
                phase21.main(self.arguments(directory))
            self.assertEqual(target.read_bytes(), b"original acceptance evidence\n")
            mocks["collect_cases"].assert_not_called()

    def test_final_seeds_require_source_pinned_frozen_protocol(self):
        for protocol_kind in ("absent", "unfrozen", "stale"):
            with self.subTest(protocol=protocol_kind), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                mocks = self.mocks(stack)
                extra = ["--profile", "final", "--task", "relay", "--seeds", "1001", "--transitions", "10"]
                if protocol_kind != "absent":
                    protocol = Path(directory)/"protocol.json"
                    protocol.write_text(json.dumps({"status": "draft" if protocol_kind == "unfrozen" else "frozen",
                                                    "execution_source_sha256": {} if protocol_kind == "stale" else SOURCES}))
                    extra.extend(["--frozen-protocol", str(protocol)])
                with self.assertRaises(SystemExit):
                    phase21.main(self.arguments(directory, "collect", *extra))
                mocks["collect_cases"].assert_not_called()
                mocks["runtime_manifest"].assert_not_called()

    def test_complete_requires_all_four_full_audits_and_records_shared_attribution(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            mocks = self.mocks(stack)
            self.assertEqual(phase21.main(self.arguments(directory)), 0)
            report = json.loads((Path(directory)/"reports/phase21_acceptance.json").read_text())
            self.assertEqual(report["status"], "complete_p21_pilot")
            self.assertEqual(report["cases"], [["relay",0],["relay",1],["cover",0],["cover",1]])
            self.assertEqual(report["physical_initial_replay_joint_transitions"], 200000)
            self.assertEqual(report["attributed_initial_joint_transitions_per_variant"], {"B":200000,"D":200000})
            self.assertFalse(report["model_training_started"])
            self.assertFalse(report["final_seed_collection_started"])
            self.assertEqual([call.kwargs["transitions"] for call in mocks["collect_cases"].call_args_list], [1000,50000])
            self.assertEqual(len(report["stages"]["dataset_audits"]), 4)
            self.assertEqual(report["stages"]["preflight"]["published_diagnostic_joint_transitions"], 4000)
            self.assertFalse(report["stages"]["preflight"]["enters_initial_training_replay"])
            self.assertEqual(len(mocks["run_storage_benchmark"].call_args.kwargs["dataset_paths"]), 4)
            self.assertTrue((Path(directory)/"reports/phase21_acceptance.sha256").is_file())

    def test_storage_preflight_or_failed_benchmark_cannot_produce_complete_status(self):
        for status in ("failed", "passed_preflight"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                mocks = self.mocks(stack)
                mocks["run_storage_benchmark"].return_value = {"status":status,"selected":deepcopy(STORAGE),"all_payloads_exact":True}
                self.assertEqual(phase21.main(self.arguments(directory)), 1)
                report = json.loads((Path(directory)/"reports/phase21_acceptance.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertEqual(mocks["collect_cases"].call_count, 1)  # diagnostic preflights only

    def test_dataset_audit_failure_prevents_complete_status_and_preserves_failure_report(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            mocks = self.mocks(stack)
            original = self.fake_audit
            def fail_full_dataset(path, **kwargs):
                if kwargs["expected_transitions"] == 50000:
                    raise ValueError("injected source/next-state alignment failure")
                return original(path, **kwargs)
            mocks["audit_initial_dataset"].side_effect = fail_full_dataset
            self.assertEqual(phase21.main(self.arguments(directory)), 1)
            report = json.loads((Path(directory)/"reports/phase21_acceptance.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("alignment failure", report["error"])
            self.assertEqual(mocks["collect_cases"].call_count, 2)

    def test_source_drift_and_runtime_drift_fail_closed(self):
        for drift in ("source", "runtime"):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                mocks = self.mocks(stack)
                if drift == "source":
                    mocks["execution_sources"].side_effect = [SOURCES, {"src/example.py": "changed"}]
                else:
                    mocks["runtime_manifest"].side_effect = [RUNTIME, {**RUNTIME, "engine": "changed"}]
                self.assertEqual(phase21.main(self.arguments(directory, "benchmark")), 1)
                report = json.loads((Path(directory)/"reports/benchmark_report.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertIn("changed during", report["error"])

    def test_unpinned_runtime_stops_before_collection(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            mocks = self.mocks(stack)
            mocks["runtime_manifest"].return_value = {"packages":{"numpy":{"matches":False}}}
            self.assertEqual(phase21.main(self.arguments(directory, "collect")), 1)
            mocks["collect_cases"].assert_not_called()
            report = json.loads((Path(directory)/"reports/collect_report.json").read_text())
            self.assertEqual(report["status"], "failed")

    def test_frozen_final_mock_run_records_its_actual_scope(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.mocks(stack)
            protocol = Path(directory)/"protocol.json"
            protocol.write_text(json.dumps({"status":"frozen","execution_source_sha256":SOURCES}))
            self.assertEqual(phase21.main(self.arguments(directory, "collect", "--profile","final", "--task","relay",
                                                        "--seeds","1001", "--transitions","10", "--frozen-protocol",str(protocol))), 0)
            report = json.loads((Path(directory)/"reports/collect_report.json").read_text())
            self.assertEqual(report["status"], "passed_preflight")
            self.assertTrue(report["final_seed_collection_started"])
            self.assertIn("final", report["scope"])

    def test_reuse_requires_matching_verified_complete_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root/"relay/seed_0"
            target.mkdir(parents=True)
            report = {"target_transitions":1000,"task_family":"relay","run_seed":0,"profile":"pilot"}
            (target/"collection_report.json").write_text(json.dumps(report))
            origin = {"execution_source_sha256":SOURCES,"runtime":RUNTIME,"journal_mode":"dataset_local"}
            (target/"collection_spec.json").write_text(json.dumps({"origin_provenance":{**origin,"storage_setting":STORAGE}}))
            with patch.object(phase21,"collection_origin",return_value=deepcopy(origin)), patch.object(phase21,"verify_initial_collection",return_value={"status":"passed"}) as verify, patch.object(phase21,"collect_initial_dataset") as collect, redirect_stdout(io.StringIO()):
                cases = phase21.collect_cases([("relay",0)],root,profile="pilot",transitions=1000,jobs=1,storage=STORAGE,reuse=True)
                self.assertTrue(cases[0]["reused"])
                verify.assert_called_once_with(target)
                collect.assert_not_called()
                with self.assertRaises(ValueError):
                    phase21.collect_cases([("relay",0)],root,profile="pilot",transitions=50000,jobs=1,storage=STORAGE,reuse=True)
                with self.assertRaises(FileExistsError):
                    phase21.collect_cases([("relay",0)],root,profile="pilot",transitions=1000,jobs=1,storage=STORAGE,reuse=False)

    def test_stale_cached_origin_stops_before_any_other_case_can_consume_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            target=root/"relay/seed_0"
            target.mkdir(parents=True)
            (target/"collection_report.json").write_text(json.dumps({
                "target_transitions":1000,"task_family":"relay","run_seed":0,"profile":"pilot"}))
            (target/"collection_spec.json").write_text(json.dumps({"origin_provenance":{"execution_source_sha256":{"old":"old"}}}))
            with patch.object(phase21,"collection_origin",return_value={"execution_source_sha256":SOURCES,"runtime":RUNTIME,"journal_mode":"dataset_local"}), patch.object(phase21,"verify_initial_collection",return_value={"status":"passed"}), patch.object(phase21,"collect_initial_dataset") as collect:
                # A new case appears first. Cache compatibility must be checked
                # for the entire matrix before a thread starts that new case.
                with self.assertRaisesRegex(ValueError,"origin"):
                    phase21.collect_cases([("cover",0),("relay",0)],root,profile="pilot",transitions=1000,jobs=2,storage=STORAGE,reuse=True)
                collect.assert_not_called()
            self.assertFalse((root/"cover").exists())

    def test_collection_origin_pins_sources_runtime_and_journal_mode(self):
        with patch.object(phase21,"execution_sources",return_value=SOURCES), patch.object(phase21,"runtime_manifest",return_value=RUNTIME):
            local=phase21.collection_origin(None)
            hot=phase21.collection_origin(Path("temporary"))
        self.assertEqual(local["execution_source_sha256"],SOURCES)
        self.assertNotIn("recorded_at_utc",local["runtime"])
        self.assertEqual(local["journal_mode"],"dataset_local")
        self.assertEqual(hot["journal_mode"],"durable_temporary_spool")

    def test_complete_propagates_active_resource_guard_to_each_collection_stage(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            mocks = self.mocks(stack)
            self.assertEqual(phase21.main(self.arguments(directory)), 0)
            self.assertEqual(mocks["collect_cases"].call_count, 2)
            for call in mocks["collect_cases"].call_args_list:
                guard = call.kwargs["resource_guard"]
                self.assertTrue(callable(guard))
                self.assertIsInstance(guard.__self__, FakeResources)
                self.assertEqual(guard.__self__.cpu_ids, [0,1,2,3])

    def test_case_executor_passes_health_guard_to_new_collector(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = lambda: None
            with patch.object(phase21,"collection_origin",return_value={}), patch.object(phase21,"collect_initial_dataset",return_value={"task_family":"relay","run_seed":0}) as collect:
                phase21.collect_cases([("relay",0)],Path(directory),profile="pilot",transitions=1000,jobs=1,
                                      storage=STORAGE,reuse=False,resource_guard=guard)
            self.assertIs(collect.call_args.kwargs["resource_guard"], guard)

    def test_disk_provisioning_does_not_assume_compression_savings(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(phase21.shutil,"disk_usage",return_value=type("Usage",(),{"free":1000})()):
                with self.assertRaisesRegex(RuntimeError,"insufficient free disk"):
                    phase21.check_disk(Path(directory),50000,4)
            with patch.object(phase21.shutil,"disk_usage",return_value=type("Usage",(),{"free":10**12})()):
                report=phase21.check_disk(Path(directory),50000,4)
                self.assertGreater(report["conservative_required_bytes"],4*50000*73728)


if __name__ == "__main__":
    unittest.main()
