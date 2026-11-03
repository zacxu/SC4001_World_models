"""Real-replay alignment, immutable pairing, and independent missing-target masks."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_targets import fixture
from world_models.contracts import BoundaryReason, EpisodeBoundary, JointTransitionMetadata, RGBFrame
from world_models.data.replay import ReplayDataset, ReplayDatasetWriter, canonical_hash, read_npz, write_npz
from world_models.data.schema import PLAYER_FIELDS, TASK_FIELDS, episode_partition
from world_models.env.coop import ResetResult, StepResult
from world_models.env.targets import build_privileged_targets, event_snapshot, native_counter_baseline


def episode_fixture(length=5, *, missing_final=False, missing_state=False, family="relay"):
    scenario, flags, raw = fixture(family)
    scenario.update(hashes={"wad_sha256": "a" * 64}, split_provenance={
        "split": "train", "purpose": "training", "learning_eligible": True,
        "template_id": f"{family}_train_00", "split_manifest_sha256": "b" * 64})
    baseline = native_counter_baseline(raw)
    previous = None
    results = []
    for index in range(length + 1):
        local_flags = deepcopy(flags)
        local_flags["elapsed_decisions"] = index
        if index:
            if family == "relay":
                local_flags.update(latch_activation_count=1, latch_active=1, milestone_count=1)
            else:
                local_flags.update(console_1_completion_count=1, console_1_complete=1, milestone_count=1)
        current = replace(raw, participants=tuple(replace(player, tic=4 + 4*index,
            variables={**player.variables, "POSITION_X": 10.125 + index}) for player in raw.participants))
        targets = build_privileged_targets(current, local_flags, scenario)
        boundary = EpisodeBoundary(BoundaryReason.SUCCESS if index == length else BoundaryReason.CONTINUE,
                                   (False, True) if missing_final and index == length else (True, True))
        events = event_snapshot(current, local_flags, scenario, previous, boundary=boundary, native_baseline=baseline)
        previous = events["cumulative"]
        info = {"task_family": family, "engine_seed": 123, "decision_index": index, "episode_id": 1,
                "engine_tic": 4 + 4*index, "scenario_manifest": deepcopy(scenario),
                "privileged_targets": None if missing_state and index == length else targets,
                "events": events}
        frames = tuple(RGBFrame(np.full((3, 96, 128), 10*index + player, dtype=np.uint8).tobytes())
                       if not (player == 0 and missing_final and index == length) else None for player in (0, 1))
        if index == 0:
            results.append(ResetResult(frames, info))
        else:
            metadata = JointTransitionMetadata(index-1, 4*index, 4+4*index, (index % 12, (index+1) % 12), boundary)
            results.append(StepResult(frames, 0.999 if index == length else -0.001, boundary, metadata, info))
    return results


def metadata(episode_id="episode-0", **extra):
    return {"episode_id": episode_id, "collector_kind": "scripted", "policy_version": "scripted-v1",
            "run_seed": 0, "environment_seed": 123, "engine_version": "1.3.2", **extra}


def write_episode(writer, results, *, episode_id="episode-0"):
    writer.start_episode(metadata(episode_id), results[0])
    for result in results[1:]:
        writer.append(result)
    return writer.finish_episode()


def rewrite_chunk_with_receipts(root, transform):
    """Corrupt semantics while keeping receipts consistent: exercise validation, not only hashes."""
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    episode = manifest["episodes"][0]
    chunk = episode["chunks"][0]
    arrays = {name: value.copy() for name, value in read_npz(root/chunk["path"]).items()}
    transform(arrays)
    chunk.update(write_npz(root/chunk["path"], arrays))
    episode_path = root/episode["manifest_path"]
    episode_path.write_text(json.dumps({key: value for key, value in episode.items() if key not in ("manifest_path", "manifest_sha256")}), encoding="utf-8")
    episode["manifest_sha256"] = hashlib.sha256(episode_path.read_bytes()).hexdigest()
    manifest["dataset_identity_sha256"] = canonical_hash({key: value for key, value in manifest.items() if key not in ("sealed_at_utc", "dataset_identity_sha256")})
    (root/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root/"manifest.sha256").write_text(hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest()+"  manifest.json\n")


class ReplayStorageTests(unittest.TestCase):
    def make_writer(self, directory, *, target=5, **extra):
        return ReplayDatasetWriter(Path(directory)/"real", {
            "task_family": "relay", "run_seed": 0, "source": "real_environment", "target_transitions": target}, **extra)

    def test_roundtrip_preserves_source_state_and_transition_alignment_across_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            results = episode_fixture()
            writer = self.make_writer(directory, chunk_transitions=2)
            episode = write_episode(writer, results)
            manifest = writer.seal()
            self.assertEqual([(row["state_start"], row["state_count"]) for row in episode["chunks"]], [(0, 3), (3, 2), (5, 1)])
            self.assertEqual(sum(row["state_count"] for row in episode["chunks"]), 6)
            dataset = ReplayDataset(writer.path, cache_chunks=1)
            view = dataset.load_episode("episode-0")
            arrays = view.arrays()
            self.assertEqual(arrays["rgb"].shape, (6, 2, 3, 96, 128))
            self.assertEqual(arrays["rgb"].dtype, np.uint8)
            np.testing.assert_array_equal(arrays["privileged_player_state"][:, 0, PLAYER_FIELDS.index("position_x")], 10.125 + np.arange(6))
            np.testing.assert_array_equal(arrays["observation_index"], np.arange(5))
            np.testing.assert_array_equal(arrays["next_observation_index"], np.arange(1, 6))
            np.testing.assert_array_equal(arrays["actions"], [result.metadata.actions for result in results[1:]])
            np.testing.assert_allclose(arrays["team_reward"], [result.team_reward for result in results[1:]], atol=0, rtol=0)
            self.assertEqual(arrays["task_state"][0, TASK_FIELDS.index("milestone_count")], 0)
            self.assertEqual(arrays["task_state"][1, TASK_FIELDS.index("milestone_count")], 1)
            self.assertEqual(view.observation(2, 1), results[2].observations[1])
            self.assertLessEqual(len(dataset._cache), 1)
            self.assertEqual(manifest["transition_count"], 5)
            self.assertFalse(arrays["rgb"].flags.writeable)

    def test_terminal_image_missing_keeps_valid_state_reward_and_events(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory)
            write_episode(writer, episode_fixture(missing_final=True))
            writer.seal()
            view = ReplayDataset(writer.path).load_episode("episode-0")
            arrays = view.arrays()
            self.assertIsNone(view.observation(5, 0))
            self.assertIsNotNone(view.observation(5, 1))
            self.assertFalse(arrays["image_valid"][-1, 0])
            self.assertTrue(arrays["privileged_validity_mask"][-1, 0].all())
            self.assertTrue(arrays["terminated"][-1])
            self.assertEqual(arrays["team_reward"][-1], 0.999)
            self.assertEqual(arrays["event_deltas"].sum(), 1)

    def test_state_missing_does_not_mask_valid_final_image(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory)
            write_episode(writer, episode_fixture(missing_state=True))
            writer.seal()
            arrays = ReplayDataset(writer.path).load_episode("episode-0").arrays()
            self.assertTrue(arrays["image_valid"][-1].all())
            self.assertFalse(arrays["privileged_validity_mask"][-1].any())
            self.assertFalse(arrays["task_validity_mask"][-1].any())

    def test_model_diagnostic_episode_cannot_be_fit_or_imagination_anchor(self):
        identifiers = [f"episode-{index}" for index in range(100)]
        identifier = next(value for value in identifiers if episode_partition(value) == "diagnostic")
        fitting = next(value for value in identifiers if episode_partition(value) == "fit")
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory, target=10)
            write_episode(writer, episode_fixture(), episode_id=identifier)
            write_episode(writer, episode_fixture(), episode_id=fitting)
            writer.seal()
            dataset = ReplayDataset(writer.path)
            self.assertEqual(len(list(dataset.iter_episodes(purpose="diagnostic"))), 2)
            self.assertEqual(len(list(dataset.iter_episodes(purpose="world_model_fit"))), 1)
            self.assertEqual(len(list(dataset.iter_episodes(purpose="imagination"))), 1)
            for purpose in ("world_model_fit", "imagination"):
                with self.assertRaisesRegex(ValueError, "excluded"):
                    dataset.load_episode(identifier, purpose=purpose)
            self.assertEqual(episode_partition(identifier), episode_partition(identifier))

    def test_hash_partition_tracks_episode_id_not_collection_order(self):
        before = {name: episode_partition(name) for name in (f"episode-{i}" for i in range(1000))}
        after = {name: episode_partition(name) for name in reversed(list(before))}
        self.assertEqual(before, after)
        self.assertTrue(60 < sum(value == "diagnostic" for value in before.values()) < 140)

    def test_sealed_writer_and_new_writer_reject_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory)
            results = episode_fixture()
            write_episode(writer, results)
            writer.seal()
            with self.assertRaises(RuntimeError): writer.start_episode(metadata("episode-2"), results[0])
            with self.assertRaises(RuntimeError): writer.append(results[-1])
            with self.assertRaises(RuntimeError): writer.seal()
            with self.assertRaises(FileExistsError): self.make_writer(directory)

    def test_b_and_d_bind_to_same_immutable_dataset_and_full_interaction_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory)
            write_episode(writer, episode_fixture())
            writer.seal()
            dataset = ReplayDataset(writer.path)
            bindings = [dataset.bind_variant(Path(directory)/f"{variant}.json", variant) for variant in ("B", "D")]
            self.assertEqual(bindings[0]["manifest_sha256"], bindings[1]["manifest_sha256"])
            self.assertEqual([binding["attributed_initial_joint_transitions"] for binding in bindings], [5, 5])
            self.assertTrue(all(binding["new_physical_transitions"] == 0 for binding in bindings))
            self.assertEqual(ReplayDataset.from_binding(Path(directory)/"D.json").manifest_sha256, dataset.manifest_sha256)
            with self.assertRaisesRegex(ValueError, "another variant"):
                ReplayDataset.from_binding(Path(directory)/"D.json", variant="B")

    def test_shared_binding_survives_relocation_of_whole_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory)/"original"
            writer = self.make_writer(original)
            write_episode(writer, episode_fixture())
            writer.seal()
            dataset = ReplayDataset(writer.path)
            binding = dataset.bind_variant(original/"branches/B.json", "B")
            self.assertFalse(Path(binding["dataset_path"]).is_absolute())
            self.assertEqual(binding["dataset_path"], "../real")
            self.assertNotIn("\\", binding["dataset_path"])
            destination = Path(directory)/"relocated"
            shutil.copytree(original, destination)
            relocated = ReplayDataset.from_binding(destination/"branches/B.json", variant="B")
            self.assertEqual(relocated.manifest_sha256, dataset.manifest_sha256)
            self.assertEqual(relocated.load_episode("episode-0").observation(1, 0), episode_fixture()[1].observations[0])

    def test_legacy_windows_binding_is_normalized_before_posix_path_parsing(self):
        value = {"variant": "B", "dataset_path": "..\\replay", "manifest_sha256": "a"*64,
                 "dataset_identity_sha256": "b"*64, "attributed_initial_joint_transitions": 5}
        class BindingPath:
            parent = PurePosixPath("/relocated/branches")
            def read_text(self, **kwargs): return json.dumps(value)
        class CapturingDataset(ReplayDataset):
            def __init__(self, path, **kwargs):
                self.received_path = path
                self.manifest = {"dataset_identity_sha256": "b"*64, "transition_count": 5}
            @property
            def manifest_sha256(self): return "a"*64
        sentinel = object()
        def posix_paths(argument):
            return BindingPath() if argument is sentinel else PurePosixPath(argument)
        with patch("world_models.data.replay.Path", side_effect=posix_paths):
            dataset = CapturingDataset.from_binding(sentinel, variant="B")
        self.assertEqual(dataset.received_path, PurePosixPath("/relocated/branches/../replay"))
        self.assertNotIn("\\", str(dataset.received_path))

    def test_legacy_windows_binding_reads_without_modifying_pointer_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory)
            write_episode(writer, episode_fixture())
            writer.seal()
            dataset = ReplayDataset(writer.path)
            path = Path(directory)/"branches/B.json"
            value = dataset.bind_variant(path, "B")
            value["dataset_path"] = "..\\real"
            path.write_text(json.dumps(value), encoding="utf-8")
            before = path.read_bytes()
            reader = ReplayDataset.from_binding(path, variant="B")
            self.assertEqual(reader.manifest_sha256, dataset.manifest_sha256)
            self.assertEqual(path.read_bytes(), before)

    def test_binding_path_requires_a_nonempty_string(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"binding.json"
            for value in (None, "", "   ", 0, [], {}, "bad\0path"):
                with self.subTest(value=value):
                    path.write_text(json.dumps({"variant": "B", "dataset_path": value}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "nonempty dataset_path string"):
                        ReplayDataset.from_binding(path)

    def test_anchor_selection_excludes_diagnostic_episodes_and_final_state(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory, target=10, chunk_transitions=2)
            fit = next(f"episode-{index}" for index in range(100) if episode_partition(f"episode-{index}") == "fit")
            diagnostic = next(f"episode-{index}" for index in range(100) if episode_partition(f"episode-{index}") == "diagnostic")
            write_episode(writer, episode_fixture(missing_final=True), episode_id=fit)
            write_episode(writer, episode_fixture(), episode_id=diagnostic)
            writer.seal()
            dataset = ReplayDataset(writer.path)
            np.testing.assert_array_equal(dataset.load_episode(fit).valid_anchor_indices(), np.arange(5))
            with self.assertRaisesRegex(ValueError, "excluded"):
                dataset.load_episode(diagnostic).valid_anchor_indices()
            # Returned episode metadata cannot change the dataset's purpose guards.
            exposed = dataset.episodes
            next(value for value in exposed if value["episode_id"] == diagnostic)["partition"] = "fit"
            with self.assertRaisesRegex(ValueError, "excluded"):
                dataset.load_episode(diagnostic, purpose="imagination")

    def test_preflight_store_cannot_fit_or_seed_imagination_even_for_fitting_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = ReplayDatasetWriter(Path(directory)/"preflight", {
                "task_family": "relay", "run_seed": 0, "data_source": "real_environment",
                "budget_scope": "collection_preflight", "target_transitions": 5})
            identifier = next(f"episode-{i}" for i in range(100) if episode_partition(f"episode-{i}") == "fit")
            write_episode(writer, episode_fixture(), episode_id=identifier)
            manifest = writer.seal()
            self.assertFalse(manifest["metadata"]["learning_eligible"])
            self.assertFalse(manifest["episodes"][0]["learning_eligible"])
            reader = ReplayDataset(writer.path)
            self.assertEqual(len(list(reader.iter_episodes(purpose="diagnostic"))), 1)
            view = reader.load_episode(identifier)
            reader.manifest["metadata"]["learning_eligible"] = True  # Returned metadata cannot bypass the guard.
            for purpose in ("world_model_fit", "imagination"):
                with self.assertRaisesRegex(ValueError, "explicit counted import"):
                    reader.load_episode(identifier, purpose=purpose)
                with self.assertRaisesRegex(ValueError, "explicit counted import"):
                    list(reader.iter_episodes(purpose=purpose))
                with self.assertRaisesRegex(ValueError, "explicit counted import"):
                    view.valid_anchor_indices(purpose=purpose)

    def test_changed_chunk_manifest_and_episode_are_detected(self):
        for victim in ("chunk", "episode", "manifest"):
            with self.subTest(victim=victim), tempfile.TemporaryDirectory() as directory:
                writer = self.make_writer(directory)
                write_episode(writer, episode_fixture())
                manifest = writer.seal()
                first = manifest["episodes"][0]
                path = writer.path / ({"chunk": first["chunks"][0]["path"], "episode": first["manifest_path"], "manifest": "manifest.json"}[victim])
                with path.open("ab") as stream: stream.write(b" ")
                with self.assertRaises(ValueError): ReplayDataset(writer.path)

    def test_nontraining_geometry_and_diagnostic_trajectories_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory)
            for split, purpose, eligible in (("validation", "validation", False), ("test", "test", False), ("train", "diagnostic", False)):
                reset = deepcopy(episode_fixture()[0])
                reset.training_only_info["scenario_manifest"]["split_provenance"].update(split=split, purpose=purpose, learning_eligible=eligible)
                with self.assertRaisesRegex(ValueError, "training geometry"):
                    writer.start_episode(metadata(), reset)
            with self.assertRaisesRegex(ValueError, "diagnostic/synthetic"):
                writer.start_episode(metadata(purpose="environment_validation"), episode_fixture()[0])
            with self.assertRaisesRegex(ValueError, "synthetic"):
                ReplayDatasetWriter(Path(directory)/"synthetic", {"task_family": "relay", "run_seed": 0, "data_source": "synthetic_ppo"})
            forged = deepcopy(episode_fixture()[0])
            forged.training_only_info["scenario_manifest"].pop("split_provenance")
            with self.assertRaisesRegex(ValueError, "training geometry"):
                writer.start_episode(metadata(split_provenance={"split": "train", "purpose": "training", "learning_eligible": True}), forged)

    def test_episode_metadata_cannot_mislabel_authoritative_layout_seed_hash_or_tic(self):
        cases = ({"layout_id": "relay_validation_00"}, {"split": "validation"},
                 {"environment_seed": 456}, {"scenario_hash": "c"*64}, {"initial_tic": 999})
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory)
            reset = episode_fixture()[0]
            for changed in cases:
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    writer.start_episode(metadata(**changed), reset)
            self.assertEqual(writer.transition_count, 0)
            self.assertFalse((writer.path/"episodes").exists())

    def test_reader_rejects_wrong_typed_shapes_and_invalid_semantics_even_with_valid_hashes(self):
        changes = {
            "mask_dtype": lambda arrays: arrays.update(image_valid=arrays["image_valid"].astype(np.uint8)),
            "state_shape": lambda arrays: arrays.update(task_state=arrays["task_state"][:, :-1]),
            "nan_reward": lambda arrays: arrays["team_reward"].__setitem__(0, np.nan),
            "negative_counter": lambda arrays: arrays["cumulative_event_counters"].__setitem__((0, 0), -1),
            "invalid_action": lambda arrays: arrays["actions"].__setitem__((0, 0), 12),
            "zero_tics": lambda arrays: arrays["actual_tics"].__setitem__(0, 0),
            "wrong_episode": lambda arrays: arrays["episode_id"].__setitem__(0, "different"),
            "wrong_mask": lambda arrays: arrays["next_obs_valid"].__setitem__((0, 0), False),
        }
        for label, change in changes.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                writer = self.make_writer(directory)
                write_episode(writer, episode_fixture())
                writer.seal()
                rewrite_chunk_with_receipts(writer.path, change)
                with self.assertRaises(ValueError): ReplayDataset(writer.path)

    def test_invalid_counter_delta_is_rejected_without_advancing_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory)
            results = episode_fixture()
            writer.start_episode(metadata(), results[0])
            bad = deepcopy(results[1])
            bad.training_only_info["events"]["deltas"]["latch_activations"] = 0
            with self.assertRaisesRegex(ValueError, "event delta"):
                writer.append(bad)
            self.assertEqual(writer.transition_count, 0)
            writer.append(results[1])
            self.assertEqual(writer.transition_count, 1)

    def test_cross_reset_transition_is_rejected_even_when_tics_and_decisions_match(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory)
            results = episode_fixture()
            writer.start_episode(metadata(), results[0])
            changed = deepcopy(results[1])
            changed.training_only_info["episode_id"] += 1
            with self.assertRaisesRegex(ValueError, "reset boundary"):
                writer.append(changed)
            self.assertEqual(writer.transition_count, 0)

    def test_incomplete_episode_cannot_be_sealed_or_silently_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory)
            results = episode_fixture()
            writer.start_episode(metadata(), results[0])
            writer.append(results[1])
            with self.assertRaises(RuntimeError): writer.finish_episode()
            with self.assertRaises(RuntimeError): writer.seal()
            with self.assertRaisesRegex(ValueError, "interrupted active episode"):
                self.make_writer(directory, resume=True)

    def test_completed_episodes_resume_and_global_transition_ids_are_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory, target=10)
            prefix = write_episode(writer, episode_fixture())
            prefix_paths = [prefix["manifest_path"], *(chunk["path"] for chunk in prefix["chunks"])]
            prefix_bytes = {relative: (writer.path/relative).read_bytes() for relative in prefix_paths}
            resumed = self.make_writer(directory, target=10, resume=True)
            self.assertEqual(resumed.transition_count, 5)
            write_episode(resumed, episode_fixture(), episode_id="episode-1")
            resumed.seal()
            for relative, expected in prefix_bytes.items():
                self.assertEqual((writer.path/relative).read_bytes(), expected)
            dataset = ReplayDataset(resumed.path)
            values = np.concatenate([view.arrays()["transition_id"] for view in dataset.iter_episodes(purpose="diagnostic")])
            np.testing.assert_array_equal(values, np.arange(10))

    def test_resume_rejects_nonconserved_prefix_counts_duplicates_and_orphans(self):
        for damage in ("count", "partition_count", "duplicate_episode", "orphan"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                writer = self.make_writer(directory, target=10)
                write_episode(writer, episode_fixture())
                path = writer.path/"manifest.partial.json"
                saved = json.loads(path.read_text())
                if damage == "count": saved["transition_count"] += 1
                elif damage == "partition_count": saved["partition_transition_counts"]["fit"] += 1
                elif damage == "duplicate_episode": saved["episodes"].append(deepcopy(saved["episodes"][0]))
                else: (writer.path/"episodes/unaccounted").mkdir()
                path.write_text(json.dumps(saved), encoding="utf-8")
                before = {p.relative_to(writer.path).as_posix():p.read_bytes() for p in writer.path.rglob("*") if p.is_file()}
                with self.assertRaises(ValueError): self.make_writer(directory, target=10, resume=True)
                self.assertEqual(before, {p.relative_to(writer.path).as_posix():p.read_bytes() for p in writer.path.rglob("*") if p.is_file()})

    def test_resume_checks_completed_prefix_payload_semantics_even_when_hashes_match(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory, target=10)
            write_episode(writer, episode_fixture())
            path = writer.path/"manifest.partial.json"
            saved = json.loads(path.read_text())
            episode = saved["episodes"][0]
            chunk = episode["chunks"][0]
            arrays = {name: value.copy() for name,value in read_npz(writer.path/chunk["path"]).items()}
            arrays["actions"][0,0] = 12
            chunk.update(write_npz(writer.path/chunk["path"], arrays))
            manifest_path = writer.path/episode["manifest_path"]
            manifest_path.write_text(json.dumps({key:value for key,value in episode.items() if key not in ("manifest_path","manifest_sha256")}), encoding="utf-8")
            episode["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid action"):
                self.make_writer(directory, target=10, resume=True)

    def test_dataset_target_count_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = self.make_writer(directory, target=10)
            write_episode(writer, episode_fixture())
            with self.assertRaisesRegex(ValueError, "budget"):
                writer.seal()

    def test_codec_is_deterministic_lossless_and_disallows_object_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"chunk.npz"
            original = {"pixels": np.arange(512, dtype=np.uint8), "values": np.array([1.25, -2.5], dtype=np.float64)}
            first = write_npz(path, original)
            second = write_npz(path, original)
            self.assertEqual(first, second)
            decoded = read_npz(path)
            for name, value in original.items(): np.testing.assert_array_equal(value, decoded[name])
            with self.assertRaisesRegex(ValueError, "objects"):
                write_npz(path, {"bad": np.array([object()], dtype=object)})
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), first["sha256"])


if __name__ == "__main__":
    unittest.main()
