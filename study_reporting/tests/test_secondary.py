"""Synthetic CPU checkpoint/report fixtures; no research data or optimization."""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from world_models.cycle.checkpoints import save_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models_endpoint import checkpoints
from world_models_study.evidence import write_report

from study_reporting.secondary import build_secondary, validate_secondary_sources


def _ppo_report(teacher, model):
    cycles = 25 if teacher else 20
    return {"schema_version": "p41-teacher-update-v1" if teacher else "p34-update-v1",
        "cycles_completed": cycles, "world_model_version": model,
        "actor_optimizer_steps": cycles * 16, "critic_optimizer_steps": cycles * 16,
        "new_environment_interactions": 0, "teacher_kl_weight": 0, "teacher_queries": 0,
        "teacher_optimizer_steps": 0, "distillation_optimizer_steps": 0,
        "imagined_joint_steps": 3840,
        "termination_counts": {"horizon": 256, "success": 0, "failure": 0, "deadline": 0},
        "metrics": {"approx_kl": 0.01, "entropy": 2.0, "clip_fraction": 0.1,
            "value_loss": 0.2, "value_error": 0.4, "actor_grad_norm": 0.2,
            "critic_grad_norm": 0.1, "mean_trajectory_length": 15,
            "predicted_termination_frequency": 0, "explained_variance": None}}


def _diagnostic(updates, *, kl=None):
    return {"schema_version": "p42-training-diagnostics-v1", "optimizer_updates": updates,
        "forward_kl": (0.4 if updates == 0 else 0.1) if kl is None else kl,
        "student_entropy": 2.4 if updates == 0 else 2.0, "valid_player_decisions": 3840,
        "trajectories": 128, "scope": "fixed training buffer; not validation or real performance"}


def _metric(seed, variant, episodes, template=None):
    row = {"master_seed": seed, "variant": variant, "episodes": episodes,
        "successes": 0, "success_rate": 0, "mean_return": -0.4,
        "latch_episodes": 0, "latch_episode_rate": 0,
        "successful_completion_length_denominator": 0, "mean_successful_completion_length": None,
        "successful_completion_tics_denominator": 0, "mean_successful_completion_tics": None,
        "termination_counts": {"deadline": episodes, "success": 0, "death": 0, "failure": 0}}
    if template:
        row["template_id"] = template
    return row


class SecondaryFixture:
    def __init__(self, root, *, missing_ppo=False, missing_distillation=False, kl_after=0.1):
        self.root = Path(root)
        self.states = {}
        self.context = {"authentication": {key: True for key in
            ("cohort", "analysis", "final_ledger", "final_actors", "evaluation_traces", "input_hashes", "frozen_sources")},
            "cohort": {"cohort_identity": "a" * 64, "protocol_sha256": "b" * 64},
            "analysis": {"cohort_identity": "a" * 64, "status": "complete", "analysis": {
                "evidence_kind": "native_final_endpoint", "total_logical_episodes": 80, "paired_seed_count": 2,
                "arms": [_metric(s, v, 20) for s in (0, 1) for v in ("B", "D")],
                "per_layout": [_metric(s, v, 10, t) for s in (0, 1) for v in ("B", "D")
                    for t in ("relay_test_00", "relay_test_01")], "paired_differences": []}},
            "ledger": {"cohort_identity": "a" * 64, "status": "complete",
                "sum_run_logical_counters": {"new_model_updates": 0},
                "shared_initial_costs": {"physical_joint_transitions": 100000},
                "evaluation_joint_transitions": 32000, "evaluation_known_simulator_tics": 128000,
                "evaluation_unknown_dispatch_attempts": 0, "evaluation_unknown_simulator_tics": False},
            "source_identity": {"study_reporting/secondary.py": "c" * 64}, "references": [], "runs": []}
        for seed in (0, 1):
            model, norm = str(seed + 1) * 64, str(seed + 3) * 64
            pairing = {"world_model_checkpoint_sha256": str(seed + 5) * 64,
                "normalization_identity_sha256": norm, "replay_manifest_sha256": str(seed + 7) * 64}
            for variant in ("B", "D"):
                directory = self.root / variant / f"seed_{seed}"
                run = {"case": f"relay/{variant}/seed_{seed}", "master_seed": seed, "variant": variant,
                    "run_dir": str(directory.resolve()), "training_complete": True, "status": "complete",
                    "cohort_identity": "a" * 64, "new_model_updates": 0, "new_environment_interactions": 0,
                    "run_manifest": {"initial_imported_versions": {"model": model}},
                    "run_binding": {"pairing": deepcopy(pairing)},
                    "initial_binding": {"normalization": {"identity_sha256": norm}}, "files": [],
                    "stage_evidence": {}}
                local_dir = directory if variant == "B" else directory / "stages" / "refinement"
                local = self._checkpoint(run, local_dir / "checkpoints" / "state_00000021_00000000",
                    {"last_report": None if missing_ppo else _ppo_report(False, model)},
                    {"schema_version": "p52-b-local-checkpoint-v1" if variant == "B" else "p43-local-refinement-checkpoint-v1",
                     "model_version": model, "local_cycles_completed": 20})
                run["last_checkpoint"] = str(local)
                if variant == "D":
                    teacher = self._checkpoint(run, directory / "stages/teacher/checkpoints/state_00000026_00000000",
                        {"last_report": None if missing_ppo else _ppo_report(True, model)},
                        {"schema_version": "p41-teacher-checkpoint-v1", "initial_model_version": model,
                         "teacher_cycles_completed": 25})
                    distill_dir = directory / "stages/distillation"
                    before = None if missing_distillation else _diagnostic(0)
                    after = None if missing_distillation else _diagnostic(100, kl=kl_after)
                    distill = self._checkpoint(run, distill_dir / "checkpoints/state_00000229_00000000",
                        {"diagnostics_before": before, "diagnostics_after": after},
                        {"schema_version": "p42-distillation-checkpoint-v1", "distillation_optimizer_steps": 100,
                         "trajectories_completed": 128, "imagined_joint_steps": 1920, "run_identity": "d" * 64})
                    handoff = {"schema_version": "p42-warm-started-student-v1",
                        "checkpoint": distill.relative_to(distill_dir).as_posix(),
                        "checkpoint_manifest_sha256": file_sha256(distill / "manifest.json"),
                        "run_identity": "d" * 64, "distillation_optimizer_steps": 100, "trajectories": 128,
                        "frozen_versions": {"model": model}, "diagnostics_before": before, "diagnostics_after": after,
                        "new_environment_interactions": 0, "new_model_updates": 0}
                    path = distill_dir / "warm_started_student.json"
                    write_report(path, handoff)
                    run["files"].append({"path": str(path.resolve()), "sha256": file_sha256(path)})
                    run["stage_evidence"] = {"teacher": {"checkpoint": str(teacher)},
                        "distillation": {"path": str(distill_dir.resolve()), "checkpoint": str(distill)},
                        "local_cycle": {"checkpoint": str(local)}}
                self.context["runs"].append(run)

    def _checkpoint(self, run, path, state, metadata):
        metadata = {**metadata, "new_environment_interactions": 0, "new_model_updates": 0}
        save_cycle_checkpoint(path, state=state, metadata=metadata)
        run["files"] += checkpoints.files(path)
        path = path.resolve()
        self.states[str(path)] = state, metadata
        return path

    def loader(self, path):
        return deepcopy(self.states[str(Path(path).resolve())])


class SecondaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = SecondaryFixture(self.temporary.name)

    def build(self):
        return build_secondary(self.fixture.context, checkpoint_loader=self.fixture.loader)

    def test_complete_zero_success_summary_retains_null_conditional_lengths(self):
        report = self.build()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["per_seed"]), 4)
        self.assertEqual(len(report["per_layout"]), 8)
        self.assertTrue(all(r["mean_successful_completion_length"] is None for r in report["per_seed"]))
        self.assertTrue(all(r["successful_completion_length_denominator"] == 0 for r in report["per_seed"]))
        self.assertEqual(report["per_seed"][0]["mean_return"], -0.4)
        self.assertFalse(report["compute_matched"])
        self.assertFalse(report["research_training_executed"])

    def test_distillation_before_after_changes_are_signed_and_not_quality_gates(self):
        report = self.build()
        distilled = [r["distillation"] for r in report["training_diagnostics"] if r["variant"] == "D"]
        self.assertTrue(all(math.isclose(r["forward_kl_change_after_minus_before"], -0.3) for r in distilled))
        self.assertTrue(all(r["valid_player_decisions"] == 3840 for r in distilled))
        self.assertFalse(report["low_KL_establishes_real_cooperation"])
        other = SecondaryFixture(Path(self.temporary.name) / "worse", kl_after=0.7)
        result = build_secondary(other.context, checkpoint_loader=other.loader)
        self.assertEqual(result["status"], "complete")
        self.assertGreater(result["training_diagnostics"][1]["distillation"]["forward_kl_change_after_minus_before"], 0)

    def test_missing_ppo_measurements_stay_null_without_invalidating_completed_summary(self):
        other = SecondaryFixture(Path(self.temporary.name) / "missing", missing_ppo=True)
        report = build_secondary(other.context, checkpoint_loader=other.loader)
        self.assertEqual(report["status"], "complete")
        self.assertIsNone(report["training_diagnostics"][0]["local_PPO"]["metrics"])
        self.assertFalse(report["diagnostics_complete"])
        self.assertEqual(report["diagnostic_measurement_status"], "complete_with_explicit_measurement_gaps")
        self.assertTrue(report["warnings"])

    def test_missing_distillation_measurements_not_inferred_from_optimizer_steps(self):
        other = SecondaryFixture(Path(self.temporary.name) / "missing", missing_distillation=True)
        report = build_secondary(other.context, checkpoint_loader=other.loader)
        self.assertEqual(report["status"], "complete")
        self.assertIsNone(report["training_diagnostics"][1]["distillation"]["diagnostics_after"])
        self.assertIsNone(report["training_diagnostics"][1]["distillation"]["forward_kl_change_after_minus_before"])
        self.assertFalse(report["diagnostics_complete"])
        self.assertTrue(report["warnings"])

    def test_unrecorded_single_ppo_metric_is_warning_and_explicit_null(self):
        state = next(iter(self.fixture.states.values()))[0]
        del state["last_report"]["metrics"]["value_error"]
        report = self.build()
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["diagnostics_complete"])
        self.assertIsNone(report["training_diagnostics"][0]["local_PPO"]["metrics"]["value_error"])
        self.assertTrue(any("value_error" in warning for warning in report["warnings"]))

    def test_default_loader_reads_only_cpu_and_never_constructs_optimizer(self):
        import torch
        with patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer forbidden")), \
                patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU query forbidden")), \
                patch.object(torch.cuda, "init", side_effect=AssertionError("CUDA forbidden")):
            report = build_secondary(self.fixture.context)
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["native_environment_opened"])

    def test_new_model_updates_rejected(self):
        self.fixture.context["runs"][0]["new_model_updates"] = 1
        with self.assertRaisesRegex(ValueError, "zero new"):
            self.build()

    def test_different_pair_normalization_rejected(self):
        self.fixture.context["runs"][1]["run_binding"]["pairing"]["normalization_identity_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "normalization"):
            self.build()

    def test_wrong_model_version_in_committed_report_rejected(self):
        state = next(iter(self.fixture.states.values()))[0]
        state["last_report"]["world_model_version"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "PPO diagnostic"):
            self.build()

    def test_local_teacher_queries_rejected(self):
        state = next(iter(self.fixture.states.values()))[0]
        state["last_report"]["teacher_queries"] = 1
        with self.assertRaisesRegex(ValueError, "teacher queries"):
            self.build()

    def test_nonfinite_ppo_measurement_rejected(self):
        state = next(iter(self.fixture.states.values()))[0]
        state["last_report"]["metrics"]["approx_kl"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.build()

    def test_zero_success_time_cannot_be_imputed(self):
        self.fixture.context["analysis"]["analysis"]["arms"][0]["mean_successful_completion_length"] = 0
        with self.assertRaisesRegex(ValueError, "zero-success"):
            self.build()

    def test_wrong_conditional_denominator_rejected(self):
        self.fixture.context["analysis"]["analysis"]["per_layout"][0]["successful_completion_tics_denominator"] = 1
        with self.assertRaisesRegex(ValueError, "denominator"):
            self.build()

    def test_duplicate_layout_rejected(self):
        rows = self.fixture.context["analysis"]["analysis"]["per_layout"]
        rows[1] = deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError, "omit or duplicate"):
            self.build()

    def test_mutated_checkpoint_payload_rejected_before_read(self):
        path = Path(self.fixture.context["runs"][0]["last_checkpoint"]) / "state.pt"
        path.write_bytes(path.read_bytes() + b"mutation")
        with self.assertRaisesRegex(ValueError, "not audit-bound"):
            self.build()

    def test_bound_sources_checked_again_before_publication(self):
        report = self.build()
        path = Path(report["diagnostic_sources"][0]["path"])
        path.write_bytes(path.read_bytes() + b"mutation")
        with self.assertRaisesRegex(ValueError, "changed before publication"):
            validate_secondary_sources(report)

    def test_unverified_context_rejected(self):
        self.fixture.context["authentication"]["evaluation_traces"] = False
        with self.assertRaisesRegex(ValueError, "authenticated"):
            self.build()

    def test_incomplete_run_rejected(self):
        self.fixture.context["runs"][0]["training_complete"] = False
        with self.assertRaisesRegex(ValueError, "completed training"):
            self.build()

    def test_foreign_analysis_rejected(self):
        self.fixture.context["analysis"]["cohort_identity"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "different or incomplete"):
            self.build()


if __name__ == "__main__":
    unittest.main()
