"""P5.4 uses synthetic coordinators only; these tests never open native engines."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.phase36_support.evaluation import actor_version, load_actor_export
from tests.test_phase36_evaluation import export_fixture
from tests.test_replay_collection import CountingCoordinator, environment_fixture, scenario_fixture
from world_models.env.coop import TwoPlayerCoopEnv
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_study.bank import build_test_bank
from world_models_endpoint.evaluation import (
    _Lease, _metrics, _native_claim, evaluate_endpoint, load_evaluation, load_partial_accounting,
)
from world_models_endpoint.evidence import read_report, write_report


_FIXTURE_ENTRIES = {row["layout_seed"]: row for row in build_test_bank()["episodes"]}


def final_scenario(family, split, template_index=0, **kwargs):
    entry = _FIXTURE_ENTRIES[kwargs["seed"]]
    result = scenario_fixture(family, split, template_index, **kwargs)
    result.update(split_provenance=deepcopy(entry["split_provenance"]),
                  geometry_template=deepcopy(entry["geometry"]),
                  randomization_draw=deepcopy(entry["nuisance_draw"]["indices"]))
    return result


class EndpointEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.export = self.root / "actor"
        self.sources = {"fixture_only": "a" * 64}
        self.source_patch = patch("world_models_endpoint.evaluation.source_identity", return_value=self.sources)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.actor, self.meta = export_fixture(self.export, mutate_metadata=self.final_metadata)
        bank = build_test_bank()
        self.bound = {"run_dir": str(self.run), "actor_export": str(self.export), "variant": "B", "master_seed": 0,
            "cohort_identity": "b" * 64, "protocol_sha256": bank["protocol_sha256"], "bank": bank,
            "bank_sha256": canonical_hash(bank), "binding": {"fixture_final_actor": "c" * 64},
            "source_sha256": self.sources}

    @staticmethod
    def final_metadata(meta):
        meta.update(local_ppo_cycles=20, actor_optimizer_steps=320, critic_optimizer_steps=320,
                    model_updates=4500, real_joint_transitions=50000, untrained_zero_interaction_baseline=False)

    def evaluate(self, name="evaluation", **options):
        kwargs = dict(final_binding=self.bound, env_factory=environment_fixture, scenario_builder=final_scenario)
        kwargs.update(options)
        return evaluate_endpoint(self.run, self.root / name, **kwargs)

    def load(self, name="evaluation", **options):
        kwargs = dict(require_native=False, final_binding=self.bound)
        kwargs.update(options)
        return load_evaluation(self.root / name / "evaluation.json", **kwargs)

    def test_exact_twenty_test_episodes_metrics_counts_and_receipts(self):
        result = self.evaluate()
        self.assertEqual(result, self.load())
        self.assertEqual(result["schema_version"], "p54-final-test-evaluation-v1")
        self.assertEqual(result["metrics"]["episodes"], 20)
        self.assertEqual(result["metrics"]["successes"], 20)
        self.assertEqual(result["metrics"]["mean_length"], 4)
        self.assertAlmostEqual(result["metrics"]["mean_return"], .996)
        self.assertEqual([r["template_id"] for r in result["episodes"]], ["relay_test_00", "relay_test_01"] * 10)
        for key in ("native_backend", "native_engine", "native_environment_evaluation_executed", "research_evaluation_executed"):
            self.assertIs(result[key], False)
        self.assertTrue(result["fixture_only"])
        self.assertEqual(result["runtime"]["action_rng_backend"], "cpu")
        self.assertEqual(result["runtime"]["action_sampler"], "torch.multinomial_per_player")
        self.assertFalse(result["world_model_loaded"])
        self.assertFalse(result["critic_loaded"])
        self.assertFalse(result["learning_eligible"])
        accounting = result["interaction_accounting"]
        self.assertEqual(accounting["published_real_joint_transitions"], 80)
        self.assertEqual(accounting["completed_episode_joint_transitions"], 80)
        self.assertEqual(accounting["initialization_joint_decisions"], 20)
        self.assertEqual(accounting["physical_joint_decisions_including_initialization"], 100)
        self.assertEqual(accounting["published_real_simulator_tics_total"], 320)
        # The synthetic engine begins at tic 1 and reset synchronization adds
        # four tics. Keep all five, rather than inferring tics from decisions.
        self.assertEqual(accounting["initialization_simulator_tics"], 100)
        self.assertEqual(accounting["simulator_tics_total"], 420)
        self.assertTrue(accounting["accounting_exact"])
        self.assertFalse((self.run / "endpoint_test_claim.json").exists())

    def test_fixture_evidence_is_rejected_by_default_scientific_reader(self):
        self.evaluate()
        with self.assertRaisesRegex(ValueError, "synthetic fixture"):
            self.load(require_native=True)

    def test_sampling_is_reproducible_and_preserves_global_rng_weights_and_modes(self):
        loaded, export = load_actor_export(self.export)
        loaded.train()
        loaded.encoder[0].eval()
        modes = [module.training for module in loaded.modules()]
        rng, version = torch.random.get_rng_state().clone(), actor_version(loaded)
        loader = lambda *a, **k: (loaded, deepcopy(export))
        self.evaluate("first", actor_loader=loader)
        self.evaluate("second", actor_loader=loader)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(version, actor_version(loaded))
        self.assertEqual(modes, [module.training for module in loaded.modules()])
        for index in range(20):
            left = read_report(self.root / "first" / "episodes" / f"{index:06d}" / "attempt_00/attempt.json")
            right = read_report(self.root / "second" / "episodes" / f"{index:06d}" / "attempt_00/attempt.json")
            self.assertEqual(left["record"]["action_seeds"], right["record"]["action_seeds"])
            lp = self.root / "first" / "episodes" / f"{index:06d}" / "attempt_00/events.jsonl"
            rp = self.root / "second" / "episodes" / f"{index:06d}" / "attempt_00/events.jsonl"
            self.assertEqual(lp.read_bytes(), rp.read_bytes())

    def test_each_call_uses_only_local_rgb_executed_previous_action_and_reset_memory(self):
        loaded, export = load_actor_export(self.export)
        calls = []
        hook = loaded.register_forward_pre_hook(lambda module, values: calls.append(tuple(v.clone() for v in values)))
        self.addCleanup(hook.remove)
        result = self.evaluate(actor_loader=lambda *a, **k: (loaded, export))
        self.assertEqual(len(calls), 80)
        for index, (images, previous, hidden) in enumerate(calls):
            self.assertEqual(tuple(images.shape), (2, 3, 96, 128))
            self.assertEqual(images.dtype, torch.uint8)
            if index % 4 == 0:
                self.assertEqual(previous.tolist(), [12, 12])
                self.assertEqual(hidden.count_nonzero().item(), 0)
        self.assertTrue(all(row["action_seeds"][0] != row["action_seeds"][1] for row in result["episodes"]))

    def test_b_and_d_use_identical_paired_action_seeds_and_accept_preserved_validation_export_tag(self):
        left = self.evaluate("B")
        d_export = self.root / "D_actor"
        def d_metadata(meta):
            self.final_metadata(meta)
            meta["variant"] = "D"
        export_fixture(d_export, actor=self.actor, mutate_metadata=d_metadata)
        d_bound = {**self.bound, "variant": "D", "actor_export": str(d_export),
                   "binding": {"fixture_final_actor": "d" * 64}}
        right = self.evaluate("D", final_binding=d_bound)
        self.assertEqual(left["actor_export"]["metadata"]["evaluation_split"], "validation")
        self.assertEqual(right["protocol"]["split"], "test")
        self.assertEqual([r["action_seeds"] for r in left["episodes"]], [r["action_seeds"] for r in right["episodes"]])

    def test_wrong_bank_or_master_seed_or_source_fails_before_actor_or_native_work(self):
        for name, mutation in (
            ("bank", lambda b: b["bank"]["episodes"].pop()),
            ("seed", lambda b: b.update(master_seed=True)),
            ("source", lambda b: b.update(source_sha256={"changed": "f" * 64})),
        ):
            bound = deepcopy(self.bound)
            mutation(bound)
            with patch("world_models_endpoint.evaluation.load_actor_export") as loader:
                with self.assertRaises(ValueError):
                    self.evaluate(name, final_binding=bound)
                loader.assert_not_called()
            self.assertFalse((self.root / name).exists())

    def test_initial_or_intermediate_export_cannot_be_evaluated_as_final(self):
        bad = self.root / "intermediate"
        def metadata(meta):
            self.final_metadata(meta)
            meta.update(local_ppo_cycles=19, actor_optimizer_steps=304, critic_optimizer_steps=304)
        export_fixture(bad, mutate_metadata=metadata)
        with self.assertRaisesRegex(ValueError, "completed 20-cycle"):
            self.evaluate(final_binding={**self.bound, "actor_export": str(bad)})
        self.assertFalse((self.root / "evaluation").exists())

    def test_wrong_scenario_provenance_geometry_or_nuisance_is_rejected_before_physics(self):
        for name in ("split", "geometry", "nuisance"):
            def bad_scenario(*args, **kwargs):
                scenario = final_scenario(*args, **kwargs)
                if name == "split":
                    scenario["split_provenance"]["split"] = "validation"
                elif name == "geometry":
                    scenario["geometry_template"]["gate_x"] += 64
                else:
                    scenario["randomization_draw"]["spawn"] += 1
                return scenario
            with patch("world_models_endpoint.evaluation._NATIVE_ENV") as native:
                with self.assertRaisesRegex(RuntimeError, "held-out test geometry"):
                    self.evaluate(name, scenario_builder=bad_scenario)
                native.assert_not_called()
            costs = load_partial_accounting(self.root / name, final_binding=self.bound, require_native=False)
            self.assertEqual(costs["physical_joint_decisions_including_initialization"], 0)
            self.assertEqual(costs["attempts_total"], 1)

    def test_infrastructure_retry_preserves_failed_consumption_and_uses_same_seeds(self):
        calls = []
        class Fault(CountingCoordinator):
            def step(self, actions):
                if self.index >= 2:
                    self.count += 1
                    raise TimeoutError("fixture watchdog timeout after consumed decision")
                return super().step(actions)
        def create(family, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return TwoPlayerCoopEnv(family, seed=kwargs["seed"], scenario_manifest=kwargs["scenario_manifest"],
                                       coordinator=Fault(kwargs["work_root"]))
            return environment_fixture(family, **kwargs)
        result = self.evaluate(env_factory=create)
        costs = result["interaction_accounting"]
        self.assertEqual(costs["attempts_total"], 21)
        self.assertEqual(costs["infrastructure_retries"], 1)
        self.assertEqual(costs["failed_dispatch_attempts"], 1)
        self.assertEqual(costs["published_real_joint_transitions"], 81)
        self.assertEqual(costs["completed_episode_joint_transitions"], 80)
        self.assertTrue(costs["unknown_simulator_tics"])
        self.assertIsNone(costs["simulator_tics_total"])
        self.assertFalse(costs["accounting_exact"])
        self.assertEqual(calls[0]["seed"], calls[1]["seed"])
        self.assertEqual(calls[0]["scenario_manifest"]["randomization_draw"], calls[1]["scenario_manifest"]["randomization_draw"])

    def test_numerical_failure_is_never_retried(self):
        def bad_env(family, **kwargs):
            env = environment_fixture(family, **kwargs)
            env.step = lambda *args: (_ for _ in ()).throw(ValueError("nonfinite policy/task output"))
            return env
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            self.evaluate(env_factory=bad_env)
        report = read_report(self.root / "evaluation/episodes/000000/attempt_00/attempt.json")
        self.assertFalse(report["retryable_infrastructure"])
        with self.assertRaisesRegex(ValueError, "outside the infrastructure retry policy"):
            self.evaluate(resume=True, env_factory=bad_env)
        self.assertFalse((self.root / "evaluation/episodes/000000/attempt_01").exists())

    def test_at_most_two_infrastructure_retries_without_seed_replacement(self):
        def broken(*args, **kwargs):
            raise OSError("fixture unavailable worker")
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            self.evaluate(env_factory=broken)
        costs = load_partial_accounting(self.root / "evaluation", final_binding=self.bound, require_native=False)
        self.assertEqual(costs["attempts_total"], 3)
        self.assertEqual(costs["infrastructure_retries"], 2)
        self.assertGreater(costs["unknown_dispatch_attempts"], 0)
        self.assertFalse((self.root / "evaluation/episodes/000001").exists())

    def test_completed_output_cannot_be_resampled_and_resume_only_reads_its_first_outcomes(self):
        original = self.evaluate()
        with self.assertRaises(FileExistsError):
            self.evaluate()
        with patch("world_models_endpoint.evaluation._attempt") as attempt:
            resumed = self.evaluate(resume=True)
            attempt.assert_not_called()
        self.assertEqual(original, resumed)

    def test_unsealed_completed_journal_is_sealed_on_resume_without_rerunning_episode(self):
        original = self.evaluate()
        for name in ("evaluation.json", "evaluation.sha256", "episodes/000019/attempt_00/attempt.json", "episodes/000019/attempt_00/attempt.sha256"):
            (self.root / "evaluation" / name).unlink()
        with patch("world_models_endpoint.evaluation._attempt") as attempt:
            resumed = self.evaluate(resume=True)
            attempt.assert_not_called()
        self.assertEqual(original["episodes"], resumed["episodes"])
        self.assertEqual(original["interaction_accounting"], resumed["interaction_accounting"])

    def test_interrupted_pending_dispatch_is_retained_as_unknown_on_resume(self):
        self.evaluate()
        base = self.root / "evaluation"
        for name in ("evaluation.json", "evaluation.sha256", "episodes/000019/attempt_00/attempt.json", "episodes/000019/attempt_00/attempt.sha256"):
            (base / name).unlink()
        journal = base / "episodes/000019/attempt_00/events.jsonl"
        rows = journal.read_text().splitlines()
        intent = next(i for i, line in enumerate(rows) if json.loads(line).get("event") == "intent"
                      and json.loads(line).get("operation") == "step")
        journal.write_text("\n".join(rows[:intent + 1]) + "\n", encoding="utf-8")
        costs = load_partial_accounting(base, final_binding=self.bound, require_native=False)
        self.assertEqual(costs["unknown_dispatch_attempts"], 1)
        self.assertFalse(costs["accounting_exact"])
        resumed = self.evaluate(resume=True)
        self.assertEqual(resumed["metrics"]["episodes"], 20)
        self.assertEqual(resumed["interaction_accounting"]["unknown_dispatch_attempts"], 1)
        self.assertEqual(resumed["interaction_accounting"]["attempts_total"], 21)

    def test_durable_terminal_trace_is_never_resampled_if_summary_publication_was_interrupted(self):
        original = self.evaluate()
        base = self.root / "evaluation"
        for name in ("evaluation.json", "evaluation.sha256", "episodes/000019/attempt_00/attempt.json", "episodes/000019/attempt_00/attempt.sha256"):
            (base / name).unlink()
        journal = base / "episodes/000019/attempt_00/events.jsonl"
        rows = journal.read_text().splitlines()
        self.assertEqual(json.loads(rows[-1])["event"], "episode_complete")
        journal.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
        with patch("world_models_endpoint.evaluation._attempt") as attempt:
            resumed = self.evaluate(resume=True)
            attempt.assert_not_called()
        self.assertEqual(original["episodes"], resumed["episodes"])
        self.assertEqual(original["interaction_accounting"], resumed["interaction_accounting"])

    def test_missing_attempt_and_final_publication_receipts_recover_without_rerunning_valid_episodes(self):
        original = self.evaluate()
        base = self.root / "evaluation"
        (base / "evaluation.sha256").unlink()
        (base / "episodes/000019/attempt_00/attempt.sha256").unlink()
        with patch("world_models_endpoint.evaluation._attempt") as attempt:
            resumed = self.evaluate(resume=True)
            attempt.assert_not_called()
        self.assertEqual(original["episodes"], resumed["episodes"])
        self.assertEqual(original["interaction_accounting"], resumed["interaction_accounting"])
        self.assertEqual(len(list(base.glob("evaluation.uncommitted_*.json"))), 1)
        self.assertEqual(len(list((base / "episodes/000019/attempt_00").glob("attempt.uncommitted_*.json"))), 1)

    def test_tampered_trace_receipt_or_forged_metrics_fail_closed(self):
        self.evaluate()
        report_path = self.root / "evaluation/evaluation.json"
        original = read_report(report_path)
        mutated = deepcopy(original)
        mutated["metrics"]["successes"] = 0
        report_path.unlink()
        report_path.with_suffix(".sha256").unlink()
        write_report(report_path, mutated)
        with self.assertRaisesRegex(ValueError, "metrics or costs"):
            self.load()
        report_path.unlink()
        report_path.with_suffix(".sha256").unlink()
        write_report(report_path, original)
        trace = self.root / "evaluation/episodes/000000/attempt_00/events.jsonl"
        with trace.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "file changed"):
            self.load()

    def test_missing_or_extra_or_retried_valid_episode_fails_closed(self):
        self.evaluate()
        attempt = self.root / "evaluation/episodes/000000/attempt_01"
        attempt.mkdir()
        with self.assertRaisesRegex(ValueError, "retried a valid"):
            self.load()

    def test_action_rng_backend_must_match_the_recorded_evaluation_device(self):
        result = self.evaluate()
        path = self.root / "evaluation/evaluation.json"
        path.unlink()
        path.with_suffix(".sha256").unlink()
        result["runtime"]["action_rng_backend"] = "cuda"
        write_report(path, result)
        with self.assertRaisesRegex(ValueError, "RNG backend disagrees"):
            self.load()

    def test_resume_cannot_change_the_original_action_rng_backend_or_torch_version(self):
        self.evaluate()
        from world_models_endpoint.evaluation import _sampling_runtime
        original = _sampling_runtime(self.actor)
        for field, changed in (("action_rng_backend", "cuda"), ("torch", "different-version")):
            runtime = {**original, field: changed}
            with patch("world_models_endpoint.evaluation._sampling_runtime", return_value=runtime), patch(
                    "world_models_endpoint.evaluation._attempt") as attempt:
                with self.assertRaisesRegex(ValueError, "resume changed actor"):
                    self.evaluate(resume=True)
                attempt.assert_not_called()

    def test_no_success_completion_length_is_null_with_zero_denominator(self):
        records = [dict(success=False, episode_return=-1.4, latch_activated=False,
                        joint_transitions=400, simulator_tics=1600, termination_cause="deadline",
                        template_id="relay_test_00")]
        metrics = _metrics(records)
        self.assertIsNone(metrics["mean_successful_completion_length"])
        self.assertIsNone(metrics["mean_successful_completion_simulator_tics"])
        self.assertEqual(metrics["successful_completion_length_denominator"], 0)

    def test_claim_forbids_new_output_for_same_actor_and_reader_does_not_repair_missing_claim(self):
        output = self.root / "first"
        first = _native_claim(self.bound, output)
        self.assertEqual(first, _native_claim(self.bound, output))
        with self.assertRaisesRegex(ValueError, "already claimed another"):
            _native_claim(self.bound, self.root / "second")

    def test_writer_lock_rejects_overlapping_processes(self):
        path = self.root / "lock"
        with _Lease(path):
            program = "from pathlib import Path; from world_models_endpoint.evaluation import _Lease; import sys; p=Path(sys.argv[1]);\nwith _Lease(p): pass"
            result = subprocess.run([sys.executable, "-c", program, str(path)], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("another process owns", result.stderr)

    def test_readers_and_evaluator_import_no_world_model_or_training_runtime(self):
        program = ("import sys; import world_models_endpoint.evaluation; "
            "assert not any(x.startswith(('world_models.models','world_models.training','world_models.teacher',"
            "'world_models.distillation','world_models.cycle.controller')) for x in sys.modules)")
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cli_has_only_explicit_final_evaluate_and_read_only_inspect(self):
        from scripts.phase54 import main
        with patch("world_models_endpoint.evaluation.evaluate_endpoint", return_value={
            "status": "complete", "variant": "B", "master_seed": 0, "native_engine": True,
            "metrics": {}, "interaction_accounting": {}}) as evaluate:
            self.assertEqual(main(["evaluate", "--run-dir", "fixture", "--output-dir", "fixture-out", "--resume"]), 0)
            evaluate.assert_called_once_with("fixture", "fixture-out", device="cpu", resume=True)
        with self.assertRaises(SystemExit):
            main(["evaluate", "--run-dir", "fixture", "--output-dir", "out", "--episodes", "8"])


if __name__ == "__main__":
    unittest.main()
