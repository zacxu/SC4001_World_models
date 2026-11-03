"""Independent P2.1 checks must reject validly stored but wrong labels/accounting."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tests.test_replay_collection import environment_fixture, scenario_fixture
from tests.test_replay_storage import episode_fixture, metadata
from world_models.contracts import BoundaryReason, EpisodeBoundary
from world_models.data.audit import audit_initial_dataset
from world_models.data.collection import collect_initial_dataset
from world_models.data.replay import ReplayDataset, ReplayDatasetWriter
from world_models.data.schema import episode_partition


def labelled_episode(length, *, layout=0, failure=False, missing_image=False, omit_event=None, bad_reward=False):
    """Use the storage action-effect fixture with exact native reward semantics."""
    results = episode_fixture(length, missing_final=missing_image)
    for index, result in enumerate(results):
        info = result.training_only_info
        info["scenario_manifest"]["split_provenance"]["template_id"] = f"relay_train_{layout:02d}"
        if omit_event:
            info["events"]["cumulative"].pop(omit_event)
            info["events"]["deltas"].pop(omit_event)
        if index:
            terminal = index == length
            outcome = -1. if failure else 1.
            reward = -.001 + (.1 if index == 1 else 0.) + (outcome if terminal else 0.)
            boundary = result.boundary
            if terminal:
                reason = BoundaryReason.FAILURE if failure else BoundaryReason.SUCCESS
                boundary = EpisodeBoundary(reason, result.boundary.next_observation_valid)
                event = "failures" if failure else "successes"
                if event != omit_event:
                    info["events"]["cumulative"][event] = 1
                    info["events"]["deltas"][event] = 1
                target = "failure" if failure else "success"
                info["privileged_targets"]["task"]["raw"][target] = 1
                info["privileged_targets"]["task"]["normalized"][target] = 1.
            results[index] = replace(result, team_reward=reward + (.05 if bad_reward and index == 1 else 0.),
                                     boundary=boundary, metadata=replace(result.metadata, boundary=boundary))
    return results


def make_store(root, *, full=False, omit_event=None, wrong_mix=False, bad_reward=False, missing_image=False):
    root = Path(root)
    layouts = range(6) if full else range(1)
    writer = ReplayDatasetWriter(root / "replay", {"task_family": "relay", "run_seed": 0,
        "source": "real_environment", "profile": "pilot", "target_transitions": 60 if full else 10}, chunk_transitions=2)
    for layout in layouts:
        for kind, length in (("random", 7), ("scripted", 3)):
            # Deliberately include independent fitting and diagnostic episodes.
            target_partition = "diagnostic" if kind == "scripted" and layout == 0 else "fit"
            identifier = next(f"audit-{layout}-{kind}-{index}" for index in range(1000)
                              if episode_partition(f"audit-{layout}-{kind}-{index}") == target_partition)
            results = labelled_episode(length, layout=layout, failure=full and kind == "scripted" and layout == 0,
                                       omit_event=omit_event, bad_reward=bad_reward,
                                       missing_image=missing_image)
            writer.start_episode(metadata(identifier, collector_kind="scripted" if wrong_mix else kind), results[0])
            for result in results[1:]:
                writer.append(result)
            writer.finish_episode()
    writer.seal()
    dataset = ReplayDataset(writer.path)
    for variant in ("B", "D"):
        dataset.bind_variant(root / "bindings" / f"{variant}.json", variant)
    return dataset


class ReplayAuditTests(unittest.TestCase):
    def test_actual_collector_ledger_and_shared_bindings_are_independently_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "collection"
            collect_initial_dataset("relay", 0, output, total_transitions=100,
                                    env_factory=environment_fixture, scenario_builder=scenario_fixture,
                                    chunk_transitions=3)
            report = audit_initial_dataset(output, expected_transitions=100)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["aligned_transitions_checked"], 100)
            self.assertEqual(report["collector_transition_counts"], {"random": 70, "scripted": 30})
            self.assertEqual(report["interaction_ledger_verification"]["joint_transitions"], 100)
            with (output / "training_interactions.jsonl").open("ab") as stream:
                stream.write(b"{}\n")
            with self.assertRaisesRegex(ValueError, "ledger changed"):
                audit_initial_dataset(output)

    def test_full_coverage_fixture_has_both_partitions_collectors_geometries_and_events(self):
        with tempfile.TemporaryDirectory() as directory:
            make_store(directory, full=True, missing_image=True)
            with patch("world_models.data.collection.verify_initial_collection", return_value={"status": "fixture"}):
                report = audit_initial_dataset(directory, expected_transitions=60, require_full_coverage=True)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(len(report["layout_collector_transition_counts"]), 6)
            self.assertEqual(report["collector_transition_counts"], {"random": 42, "scripted": 18})
            self.assertEqual(set(report["partition_episode_counts"]), {"fit", "diagnostic"})
            self.assertGreater(report["event_counts"]["successes"], 0)
            self.assertGreater(report["event_counts"]["failures"], 0)

    def test_valid_npz_with_wrong_reward_or_mixture_is_rejected(self):
        for options, message in (({"bad_reward": True}, "reward"), ({"wrong_mix": True}, "mixture")):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                make_store(directory, **options)
                with patch("world_models.data.collection.verify_initial_collection", return_value={"status": "fixture"}):
                    with self.assertRaisesRegex(ValueError, message):
                        audit_initial_dataset(directory)

    def test_full_coverage_cannot_silently_omit_a_required_event_field(self):
        with tempfile.TemporaryDirectory() as directory:
            make_store(directory, full=True, omit_event="successes")
            with patch("world_models.data.collection.verify_initial_collection", return_value={"status": "fixture"}):
                with self.assertRaisesRegex(ValueError, "successes"):
                    audit_initial_dataset(directory, require_full_coverage=True)

    def test_equal_bytes_in_two_physical_stores_do_not_satisfy_shared_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "initial"
            make_store(output)
            copy = Path(directory) / "second_physical_copy"
            shutil.copytree(output / "replay", copy)
            binding_path = output / "bindings/D.json"
            binding = json.loads(binding_path.read_text())
            binding["dataset_path"] = str(copy)
            binding_path.write_text(json.dumps(binding), encoding="utf-8")
            with patch("world_models.data.collection.verify_initial_collection", return_value={"status": "fixture"}):
                with self.assertRaisesRegex(ValueError, "physical replay"):
                    audit_initial_dataset(output)


if __name__ == "__main__":
    unittest.main()
