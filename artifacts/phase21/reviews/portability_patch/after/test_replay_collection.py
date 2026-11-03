"""Collector integration tests: budgets, durable errors, and paired real replay."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

from world_models.data.collection import (
    CorrelatedRandomPolicy, ScriptedExplorationPolicy, collect_initial_dataset,
    mixture_segments, verify_initial_collection, _portable_report_relative_path,
)
from world_models.data.ledger import TrainingInteractionLedger, file_sha256
from world_models.data.replay import ReplayDataset, ReplayDatasetWriter
from world_models.env.coop import TwoPlayerCoopEnv
from tests.test_targets import ModernCoordinator, fixture


def scenario_fixture(family, split, template_index=0, *, seed=0, purpose="training", **kwargs):
    scenario, _, _ = fixture(family)
    scenario.update(seed=seed, hashes={"wad_sha256": "a"*64}, split_provenance={
        "template_id": f"{family}_train_{template_index:02d}",
        "split": split, "purpose": purpose, "learning_eligible": purpose == "training",
    })
    return scenario


class CountingCoordinator(ModernCoordinator):
    def __init__(self, work_root, *, failure_at=None):
        super().__init__()
        self.run_dir = Path(work_root)
        self.failure_at = failure_at

    @property
    def consumed_joint_decisions(self):
        return self.count

    def step(self, actions):
        if self.failure_at is not None and self.index >= self.failure_at:
            self.count += 1
            raise RuntimeError("injected worker fault after actual consumed decision")
        if self.index == 4:
            self.flags.update(success=1, success_count=1)
        return super().step(actions)


def environment_fixture(family, *, seed, scenario_manifest, work_root, frame_skip=4):
    return TwoPlayerCoopEnv(family, seed=seed, scenario_manifest=scenario_manifest,
                           coordinator=CountingCoordinator(work_root), frame_skip=frame_skip)


def spooled_scenario_fixture(*args, output_dir, **kwargs):
    path=Path(output_dir)
    path.mkdir(parents=True)
    (path/"scenario_source.txt").write_text("closed scenario source")
    return scenario_fixture(*args,**kwargs)


def spooled_environment_fixture(family,**kwargs):
    path=Path(kwargs["work_root"])
    path.mkdir(parents=True)
    (path/"worker_journal.txt").write_text("worker journal")
    env=environment_fixture(family,**kwargs)
    close=env.close
    def closed():
        close()
        (path/"participants_closed.txt").write_text("both participants closed")
    env.close=closed
    return env


class ReplayCollectionTests(unittest.TestCase):
    def collect(self, destination, **kwargs):
        return collect_initial_dataset("relay", 0, destination, total_transitions=100,
                                       env_factory=environment_fixture, scenario_builder=scenario_fixture,
                                       chunk_transitions=3, **kwargs)

    def test_exact_transition_mixture_is_not_an_episode_ratio(self):
        segments = mixture_segments(50000)
        self.assertEqual(sum(n for kind, n in segments if kind == "random"), 35000)
        self.assertEqual(sum(n for kind, n in segments if kind == "scripted"), 15000)
        self.assertEqual(segments[:4], [("random",700),("scripted",300)]*2)
        with self.assertRaises(ValueError):
            mixture_segments(11)

    def test_random_action_histories_are_reproducible_and_temporally_correlated(self):
        first, second = CorrelatedRandomPolicy(90), CorrelatedRandomPolicy(90)
        a, b = [first.actions() for _ in range(100)], [second.actions() for _ in range(100)]
        self.assertEqual(a, b)
        self.assertTrue(all(0 <= action < 12 for pair in a for action in pair))
        self.assertGreater(sum(a[i] == a[i-1] for i in range(1,len(a))), 10)
        self.assertGreater(len(set(a)), 10)
        self.assertNotEqual(a, [CorrelatedRandomPolicy(91).actions()]*100)

    def test_collection_charges_all_decisions_and_binds_one_shared_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"initial"
            report = self.collect(output)
            self.assertEqual(report["status"], "complete_initial_replay")
            self.assertEqual(report["budget_scope"], "collection_preflight")
            self.assertFalse(report["learning_eligible"])
            self.assertEqual(report["collector_counts"], {"random":70,"scripted":30})
            self.assertEqual(report["published_joint_transitions"],100)
            self.assertEqual(report["physical_initial_dataset_copies"],1)
            accounting = report["interaction_accounting"]
            self.assertEqual(accounting["published_real_joint_transitions"],100)
            self.assertFalse(accounting["learning_eligible"])
            self.assertEqual(accounting["published_real_simulator_tics_total"],400)
            self.assertEqual(accounting["replay_committed_joint_transitions"],100)
            self.assertEqual(accounting["physical_joint_decisions_including_initialization"],
                             100+accounting["initialization_joint_decisions"])
            self.assertEqual(accounting["failed_dispatch_attempts"],0)
            verified = verify_initial_collection(output)
            self.assertEqual(verified["joint_transitions"],100)
            self.assertEqual(verified["published_real_simulator_tics_total"],400)
            left = ReplayDataset.from_binding(output/"bindings/B.json",variant="B")
            right = ReplayDataset.from_binding(output/"bindings/D.json",variant="D")
            self.assertEqual(left.path,right.path)
            self.assertEqual(left.manifest["dataset_identity_sha256"],right.manifest["dataset_identity_sha256"])
            self.assertTrue(set(report["template_episodes"]).issubset({f"relay_train_{i:02d}" for i in range(6)}))
            self.assertEqual(len(report["template_episodes"]),6)
            ends = {episode["end_reason"] for episode in left.episodes}
            self.assertIn("success",ends)
            self.assertIn("collection_cutoff",ends)
            self.assertEqual(sum(ep["valid_length"] for ep in left.episodes),100)
            self.assertTrue(report["no_world_model_or_policy_training_started"])

    def test_cover_profiles_visit_every_training_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"cover"
            collect_initial_dataset("cover",0,output,total_transitions=320,
                                    env_factory=environment_fixture,scenario_builder=scenario_fixture,
                                    chunk_transitions=8)
            reader=ReplayDataset(output/"replay")
            coverage={}
            for episode in reader.episodes:
                if episode["collector_kind"]=="scripted":
                    coverage.setdefault(episode["layout_id"],set()).add(episode["scripted_profile"])
            self.assertEqual(len(coverage),6)
            for profiles in coverage.values():
                self.assertEqual(profiles,set(ScriptedExplorationPolicy.PROFILES))

    def test_report_bindings_are_posix_and_legacy_windows_bytes_stay_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"initial"
            report=self.collect(output)
            self.assertEqual(report["bindings"]["B"]["path"],"bindings/B.json")
            self.assertEqual(report["bindings"]["D"]["path"],"bindings/D.json")
            self.assertEqual(_portable_report_relative_path("bindings\\B.json"),
                             PurePosixPath("bindings/B.json"))
            for entry in report["bindings"].values():
                entry["path"]=entry["path"].replace("/","\\")
            path=output/"collection_report.json"
            path.write_text(json.dumps(report,indent=2),encoding="utf-8")
            original=path.read_bytes()
            binding_bytes={variant:(output/"bindings"/f"{variant}.json").read_bytes()
                           for variant in ("B","D")}
            self.assertEqual(verify_initial_collection(output)["joint_transitions"],100)
            self.assertEqual(path.read_bytes(),original)
            for variant,data in binding_bytes.items():
                self.assertEqual((output/"bindings"/f"{variant}.json").read_bytes(),data)

    def test_binding_report_paths_reject_cross_platform_escapes(self):
        for encoded in ("../outside.json","bindings/../../outside.json",
                        "..\\outside.json","/outside.json",
                        "C:\\outside.json","C:outside.json",
                        "\\\\server\\share\\outside.json","",None,"bad\0path"):
            with self.subTest(path=encoded):
                with self.assertRaises(ValueError):
                    _portable_report_relative_path(encoded)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"initial"
            report=self.collect(output)
            report["bindings"]["B"]["path"]="../outside.json"
            (output/"collection_report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,"escapes"):
                verify_initial_collection(output)

    def test_deterministic_collection_ids_streams_and_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs=[Path(directory)/"a",Path(directory)/"b"]
            for output in outputs:
                self.collect(output)
            readers=[ReplayDataset(output/"replay") for output in outputs]
            for a,b in zip(readers[0].episodes,readers[1].episodes):
                for field in ("episode_id","environment_seed","layout_seed","collector_seed","partition","collector_kind","policy_version"):
                    self.assertEqual(a[field],b[field])
                aa=readers[0].load_episode(a["episode_id"]).arrays()
                bb=readers[1].load_episode(b["episode_id"]).arrays()
                self.assertEqual(aa["actions"].tolist(),bb["actions"].tolist())

    def test_no_validation_import_or_completed_store_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"initial"
            self.collect(output)
            before=(output/"training_interactions.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                self.collect(output)
            self.assertEqual((output/"training_interactions.jsonl").read_bytes(),before)

            def bad_scenario(*args,**kwargs):
                scenario=scenario_fixture(*args,**kwargs)
                scenario["split_provenance"]["purpose"]="diagnostic"
                scenario["split_provenance"]["learning_eligible"]=False
                return scenario
            with self.assertRaises(ValueError):
                collect_initial_dataset("relay",0,Path(directory)/"bad",total_transitions=10,
                                        scenario_builder=bad_scenario,env_factory=environment_fixture)
            failed=json.loads((Path(directory)/"bad/collection_error.json").read_text())
            self.assertEqual(failed["interaction_accounting"]["physical_joint_decisions_including_initialization"],0)

    def test_consumed_worker_failure_produces_no_reward_transition_or_silent_retry(self):
        environments=[]
        def fail_env(family,**kwargs):
            env=TwoPlayerCoopEnv(family,seed=kwargs["seed"],scenario_manifest=kwargs["scenario_manifest"],
                                coordinator=CountingCoordinator(kwargs["work_root"],failure_at=2))
            environments.append(env)
            return env
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"initial"
            with self.assertRaises(RuntimeError):
                collect_initial_dataset("relay",0,output,total_transitions=10,
                                        env_factory=fail_env,scenario_builder=scenario_fixture)
            failure=json.loads((output/"collection_error.json").read_text())
            accounting=failure["interaction_accounting"]
            self.assertEqual(accounting["published_real_joint_transitions"],1)
            self.assertEqual(accounting["published_real_simulator_tics_total"],4)
            self.assertEqual(accounting["failed_dispatch_attempts"],1)
            self.assertEqual(accounting["physical_joint_decisions_including_initialization"],3)
            self.assertFalse((output/"replay/manifest.json").exists())
            rows=[json.loads(line) for line in (output/"training_interactions.jsonl").read_text().splitlines()]
            self.assertEqual(sum(row.get("published_delta",0) for row in rows),1)
            self.assertEqual(len(environments),1)

    def test_storage_failure_still_charges_the_valid_real_transition(self):
        class FailingWriter(ReplayDatasetWriter):
            def append(self,step):
                raise OSError("injected disk error")
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"initial"
            with self.assertRaises(OSError):
                self.collect(output,writer_factory=FailingWriter)
            failure=json.loads((output/"collection_error.json").read_text())
            accounting=failure["interaction_accounting"]
            self.assertEqual(accounting["published_real_joint_transitions"],1)
            self.assertEqual(accounting["replay_committed_joint_transitions"],0)
            self.assertEqual(accounting["failed_dispatch_attempts"],0)
            self.assertEqual(accounting["physical_joint_decisions_including_initialization"],2)
            self.assertFalse((output/"collection_report.json").exists())

    def test_bounded_hot_spool_archives_runtime_and_ledger_but_not_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            output,hot=Path(directory)/"cold",Path(directory)/"hot"
            report=collect_initial_dataset(
                "relay",0,output,total_transitions=100,chunk_transitions=3,
                env_factory=spooled_environment_fixture,scenario_builder=spooled_scenario_fixture,
                journal_root=hot,origin_provenance={"execution_fingerprint":"original-fingerprint"})
            storage=report["journal_storage"]
            self.assertTrue(storage["spooled"])
            self.assertEqual(len(storage["runtime_archives"]),report["completed_episodes"]*2)
            self.assertLessEqual(storage["peak_hot_bytes"],storage["maximum_hot_bytes"])
            self.assertEqual(storage["ledger_archive"]["sha256"],file_sha256(output/"training_interactions.jsonl"))
            self.assertEqual(file_sha256(hot/"ledger/training_interactions.jsonl"),
                             file_sha256(output/"training_interactions.jsonl"))
            self.assertEqual(list(hot.rglob("*.npz")),[])
            self.assertTrue(list((output/"replay").rglob("*.npz")))
            self.assertEqual([p.relative_to(hot).as_posix() for p in hot.rglob("*") if p.is_file()],
                             ["ledger/training_interactions.jsonl"])
            for archive in storage["runtime_archives"]:
                self.assertTrue(archive["source_removed"])
                if archive["category"]=="environments":
                    self.assertTrue((output/archive["archive_path"]/"participants_closed.txt").is_file())
            self.assertEqual(verify_initial_collection(output)["joint_transitions"],100)
            manifest=ReplayDataset(output/"replay").manifest
            self.assertEqual(manifest["metadata"]["origin_provenance"],{"execution_fingerprint":"original-fingerprint"})
            first=storage["runtime_archives"][0]
            member=first["files"][0]["path"]
            (output/first["archive_path"]/member).write_text("changed archived source")
            with self.assertRaises(ValueError):
                verify_initial_collection(output)

    def test_archive_failure_retains_hot_runtime_and_cold_ledger_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            output,hot=Path(directory)/"cold",Path(directory)/"hot"
            with patch("world_models.data.spool.archive_closed_tree",side_effect=OSError("injected archive failure")):
                with self.assertRaises(OSError):
                    collect_initial_dataset(
                        "relay",0,output,total_transitions=100,
                        env_factory=spooled_environment_fixture,scenario_builder=spooled_scenario_fixture,
                        journal_root=hot)
            failure=json.loads((output/"collection_error.json").read_text())
            self.assertEqual(failure["operation"],"runtime_archive")
            self.assertEqual(failure["interaction_accounting"]["published_real_joint_transitions"],4)
            self.assertTrue(list((hot/"runtime/scenarios").rglob("scenario_source.txt")))
            self.assertTrue(list((hot/"runtime/environments").rglob("participants_closed.txt")))
            self.assertEqual(file_sha256(hot/"ledger/training_interactions.jsonl"),
                             file_sha256(output/"training_interactions.jsonl"))
            self.assertFalse((output/"collection_report.json").exists())

    def test_spool_limit_fails_before_an_engine_decision_and_keeps_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            output,hot=Path(directory)/"cold",Path(directory)/"hot"
            with self.assertRaises(OSError):
                collect_initial_dataset(
                    "relay",0,output,total_transitions=100,journal_root=hot,
                    journal_spool_max_bytes=1,
                    env_factory=spooled_environment_fixture,scenario_builder=spooled_scenario_fixture)
            failure=json.loads((output/"collection_error.json").read_text())
            self.assertEqual(failure["interaction_accounting"]["physical_joint_decisions_including_initialization"],0)
            self.assertEqual(file_sha256(hot/"ledger/training_interactions.jsonl"),
                             file_sha256(output/"training_interactions.jsonl"))
            self.assertGreater(failure["journal_storage"]["peak_hot_bytes"],1)

    def test_invalid_profile_seed_is_rejected_before_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                collect_initial_dataset("relay",1001,Path(directory)/"initial",profile="pilot")
            self.assertFalse((Path(directory)/"initial").exists())

    def test_ledger_refuses_reuse_and_regressing_or_unaccounted_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=TrainingInteractionLedger(directory,task_family="relay",run_seed=0)
            with self.assertRaises(FileExistsError):
                TrainingInteractionLedger(directory,task_family="relay",run_seed=0)
            scenario=scenario_fixture("relay","train")
            env=environment_fixture("relay",seed=0,scenario_manifest=scenario,work_root=Path(directory)/"env")
            env.reset()
            ledger.record_reset(env,"episode")
            step=env.step(0,0)
            ledger.record_transition(env,"episode","random",step)
            env.coordinator.count=0
            with self.assertRaises(ValueError):
                ledger.record_failure(env,"episode","step",RuntimeError("failure"))


if __name__=="__main__":
    unittest.main()
