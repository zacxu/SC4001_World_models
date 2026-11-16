"""Synthetic metadata fixtures; no GPU, training or native evaluation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import tempfile
import unittest

from study_reporting.resources import build_resources, validate_resource_sources


class ResourceFixture:
    def __init__(self, root):
        self.root = root
        self.context = {"cohort": {"cohort_identity": "c" * 64, "inputs": {"bound_files": []}},
            "ledger": {"cohort_identity": "c" * 64, "status": "complete", "runs": {},
                "shared_initial_costs": {"physical_joint_transitions": 100000,
                    "configured_four_run_attributed_joint_transitions": 200000,
                    "physical_selected_model_prefix_optimizer_steps": 9000,
                    "configured_four_run_attributed_selected_model_steps": 18000},
                "sum_run_logical_counters": {"local_PPO_cycles": 80},
                "evaluation_joint_transitions": 32000, "evaluation_known_simulator_tics": 256000,
                "evaluation_unknown_dispatch_attempts": 0}, "runs": [], "analysis": {}}
        shared = root / "shared.dat"; shared.write_bytes(b"shared model and replay")
        self.context["cohort"]["inputs"]["bound_files"] = [{"path": str(shared), "sha256": self.sha(shared)}]
        for seed in (0, 1):
            for variant in ("B", "D"):
                self.run(variant, seed)

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def manifest(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        value = {"created_at": "2026-11-12T09:00:00+00:00", "runtime": {
            "logical_device": "cuda:0", "gpu_name": "fixture GPU", "cpu_threads": 1}}
        (directory / "run_manifest.json").write_text(json.dumps(value), encoding="utf-8")
        return value

    def stage(self, directory, stage, units, seconds, parent=False):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "attempts.jsonl"
        rows = []
        for unit in range(units):
            phase = {"phase": stage} if stage in ("buffer", "distillation") else {}
            rows += [{"event": "unit_started", "attempt": f"{stage}-{unit}", **phase},
                {"event": "unit_committed", "attempt": f"{stage}-{unit}",
                 "elapsed_seconds": seconds, **phase}]
        with path.open("a", encoding="utf-8") as stream:
            stream.writelines(json.dumps(row) + "\n" for row in rows)
        return {"journal": str(path), "committed_units": units, "logical_completed_units": units,
            "recorded_elapsed_seconds": units * seconds, "additional_partial_work_unknown": False,
            "unmeasured_elapsed_commits": 0, "malformed_rows": 0}

    def run(self, variant, seed):
        directory = self.root / variant / f"seed_{seed}"
        manifest = self.manifest(directory)
        stage_paths, stages = {}, {}
        if variant == "B":
            stages["local_cycle"] = self.stage(directory, "local_cycle", 20, 2)
        else:
            stage_paths = {name: str(directory / "stages" / name)
                for name in ("teacher", "distillation", "local_cycle")}
            stage_paths["buffer"] = stage_paths["distillation"]
            for p in set(stage_paths.values()):
                self.manifest(Path(p))
            for name, count, duration in (("teacher", 25, 4), ("buffer", 128, 0.5),
                    ("distillation", 100, 0.25), ("local_cycle", 20, 2)):
                stages[name] = self.stage(Path(stage_paths[name]), name, count, duration)
            stages["controller"] = self.stage(directory, "controller", 274, 10, parent=True)
        for value in stages.values():
            value["journal_sha256"] = self.sha(Path(value["journal"]))
        metrics = [{"stage": "teacher", "elapsed_controller_unit_seconds": 10}] if variant == "D" else [{"metrics": {}}]
        (directory / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in metrics))
        record = {"cohort_identity": "c" * 64, "variant": variant, "master_seed": seed,
            "training_complete": True, "run_dir": str(directory), "run_manifest": manifest,
            "stage_directories": stage_paths}
        self.context["runs"].append(record)
        self.context["ledger"]["runs"][f"{variant}/seed_{seed}"] = {
            "run_dir": str(directory), "physical_work_by_stage": stages,
            "counters": {"imagined_joint_transitions_total": 1000 if variant == "B" else 2000,
                "imagined_joint_transitions_local": 1000},
            "evaluation": {"evaluation_joint_transitions": 8000, "evaluation_simulator_tics": 64000,
                "sources": []}}


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = ResourceFixture(Path(self.temp.name))
        self.context = self.fixture.context

    def test_child_elapsed_is_not_double_charged_with_D_controller(self):
        report = build_resources(self.context)
        d = next(row for row in report["per_run"] if row["variant"] == "D")
        self.assertEqual(d["recorded_scientific_unit_elapsed_seconds"], 229)
        self.assertEqual(d["D_controller_elapsed_seconds_informational"], 2740)
        self.assertEqual(report["cohort_totals"]["recorded_scientific_unit_elapsed_seconds"], 538)
        self.assertFalse(d["controller_elapsed_added_to_child_time"])
        self.assertFalse(next(s for s in d["stages"] if s["stage"] == "controller")["scientific_compute_debit"])

    def test_missing_telemetry_stays_null_and_elapsed_is_not_occupancy(self):
        report = build_resources(self.context)
        for row in report["per_run"]:
            for key in ("actual_GPU_occupancy_seconds", "process_CPU_seconds", "queue_wait_seconds",
                    "peak_GPU_memory_bytes", "peak_CPU_memory_bytes", "historical_peak_storage_bytes"):
                self.assertIsNone(row[key])
            self.assertIsNone(row["evaluation"]["episodes_per_second"])
            self.assertEqual(row["runtime_manifests"][0]["runtime"]["gpu_name"], "fixture GPU")
        self.assertTrue(report["cohort_totals"]["sum_is_parallel_learner_elapsed_not_campaign_wall_time"])
        self.assertIsNone(report["cohort_totals"]["campaign_wall_seconds"])
        self.assertFalse(report["measured_gpu_occupancy_inferred_from_elapsed"])

    def test_shared_physical_and_four_arm_attribution_remain_distinct(self):
        report = build_resources(self.context)
        costs = report["shared_initial_costs"]
        self.assertEqual(costs["physical_joint_transitions"], 100000)
        self.assertEqual(costs["configured_four_run_attributed_joint_transitions"], 200000)
        self.assertEqual(costs["physical_selected_model_prefix_optimizer_steps"], 9000)
        self.assertEqual(costs["configured_four_run_attributed_selected_model_steps"], 18000)
        self.assertEqual(report["workload_caps"]["combined_imagined_joint_steps"], 503040)
        self.assertFalse(report["caps_are_measured_volumes"])
        self.assertFalse(report["compute_matched"])

    def test_exact_stage_rates_from_recorded_durations(self):
        report = build_resources(self.context)
        b = next(row for row in report["per_run"] if row["variant"] == "B")
        self.assertEqual(b["recorded_imagined_joint_steps_per_elapsed_second"], 25)
        stage = b["stages"][0]
        self.assertEqual(stage["recorded_unit_times"]["samples"], 20)
        self.assertEqual(stage["recorded_unit_times"]["median_seconds"], 2)
        self.assertEqual(len(stage["unit_time_history"]), 20)

    def test_changed_attempt_journal_rejected_against_ledger(self):
        run = self.context["runs"][0]
        path = Path(run["run_dir"]) / "attempts.jsonl"
        path.write_text(path.read_text() + json.dumps({"event": "extra"}) + "\n")
        with self.assertRaisesRegex(ValueError, "differs from the final ledger"):
            build_resources(self.context)

    def test_missing_journal_time_is_unknown_without_zero_throughput(self):
        entry = self.context["ledger"]["runs"]["B/seed_0"]
        debit = entry["physical_work_by_stage"]["local_cycle"]
        Path(debit["journal"]).unlink()
        debit["committed_units"] = 0
        debit["recorded_elapsed_seconds"] = 0
        debit["additional_partial_work_unknown"] = True
        report = build_resources(self.context)
        b = next(row for row in report["per_run"] if row["case"] == "B/seed_0")
        self.assertFalse(b["complete_scientific_elapsed_cost_known"])
        self.assertIsNone(b["recorded_imagined_joint_steps_per_elapsed_second"])

    def test_mid_publication_source_mutation_is_rejected(self):
        report = build_resources(self.context)
        ref = report["evidence_references"][0]
        with Path(ref["path"]).open("a") as stream:
            stream.write("changed")
        with self.assertRaises(RuntimeError):
            validate_resource_sources(report)

    def test_retained_storage_growth_detected_without_inventing_peak(self):
        report = build_resources(self.context)
        root = Path(self.context["runs"][0]["run_dir"])
        (root / "late.bin").write_bytes(b"new storage")
        with self.assertRaisesRegex(RuntimeError, "storage membership changed"):
            validate_resource_sources(report)
        self.assertIsNone(report["cohort_totals"]["historical_peak_storage_bytes"])

    def test_deduplicated_current_storage_is_not_historical_peak(self):
        report = build_resources(self.context)
        union = report["storage_inventories"][-2]
        paths = [row["path"] for row in union["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(union["file_count"], len(paths))
        self.assertGreater(union["retained_file_bytes"], 0)
        self.assertIsNone(union["historical_peak_storage_bytes"])
        self.assertEqual(report["cohort_totals"]["current_shared_input_retained_file_bytes"], len(b"shared model and replay"))

    def test_incomplete_or_duplicate_arm_is_rejected(self):
        self.context["runs"][0]["training_complete"] = False
        with self.assertRaisesRegex(ValueError, "completed cohort"):
            build_resources(self.context)
        self.context["runs"][0] = deepcopy(self.context["runs"][1])
        with self.assertRaisesRegex(ValueError, "four distinct"):
            build_resources(self.context)

    def test_recorded_evaluation_invocation_time_is_not_full_bank_throughput(self):
        entry = self.context["ledger"]["runs"]["B/seed_0"]
        path = self.fixture.root / "evaluation.json"
        path.write_text(json.dumps({"episodes": [{}] * 20,
            "elapsed_seconds_this_invocation": 125.5,
            "completed_at_utc": "2026-11-13T08:10:00+00:00"}))
        entry["evaluation"]["sources"] = [{"path": str(path), "sha256": self.fixture.sha(path)}]
        report = build_resources(self.context)
        evaluation = next(r for r in report["per_run"] if r["case"] == "B/seed_0")["evaluation"]
        self.assertEqual(evaluation["last_invocation_elapsed_seconds"], 125.5)
        self.assertEqual(evaluation["completed_at_utc"], "2026-11-13T08:10:00+00:00")
        self.assertEqual(evaluation["completed_episodes"], 20)
        self.assertIsNone(evaluation["elapsed_seconds"])
        self.assertIsNone(evaluation["episodes_per_second"])

    def test_measured_stage_span_includes_pauses_and_is_separate_from_unit_time(self):
        debit = self.context["ledger"]["runs"]["B/seed_0"]["physical_work_by_stage"]["local_cycle"]
        path = Path(debit["journal"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["created_at"] = "2026-11-13T09:00:00+00:00"
        rows[-1]["created_at"] = "2026-11-13T10:00:00+00:00"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        debit["journal_sha256"] = self.fixture.sha(path)
        report = build_resources(self.context)
        stage = next(r for r in report["per_run"] if r["case"] == "B/seed_0")["stages"][0]
        self.assertEqual(stage["recorded_elapsed_seconds"], 40)
        self.assertEqual(stage["recorded_stage_wall_span_seconds"], 3600)
        self.assertIsNone(stage["actual_GPU_occupancy_seconds"])

    def test_hard_links_charged_once_as_retained_file_bytes(self):
        directory = Path(self.context["runs"][0]["run_dir"])
        original = directory / "payload.bin"
        original.write_bytes(b"0123456789")
        os.link(original, directory / "payload_copy.bin")
        report = build_resources(self.context)
        inventory = report["storage_inventories"][0]
        self.assertEqual(inventory["logical_path_bytes"] - inventory["retained_file_bytes"], 10)
        self.assertEqual(inventory["file_count"] - inventory["distinct_file_count"], 1)

    def test_ledger_time_mismatch_rejected(self):
        self.context["ledger"]["runs"]["B/seed_0"]["physical_work_by_stage"]["local_cycle"]["recorded_elapsed_seconds"] += 1
        with self.assertRaisesRegex(ValueError, "do not reconcile"):
            build_resources(self.context)


if __name__ == "__main__":
    unittest.main()
