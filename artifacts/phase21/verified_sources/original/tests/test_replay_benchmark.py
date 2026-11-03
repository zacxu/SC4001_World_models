"""Storage measurements must preserve pixels, scope and selection safeguards."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import psutil

from world_models.data.benchmark import (
    _append_fsync, benchmark_fsync_directories, choose_storage_setting,
    load_actor_samples, load_replay_samples, run_storage_benchmark,
)


def samples(directory: Path, count: int = 3):
    for family_index, family in enumerate(("relay", "cover")):
        target = directory / f"{family}_train_00" / "video"
        target.mkdir(parents=True)
        for index in range(count):
            for player in (0, 1):
                image = np.full((96, 128, 3), 40 * family_index + 3 * index + player, dtype=np.uint8)
                image[:, index, player] = 255
                Image.fromarray(image).save(target / f"frame{index:04d}_player{player}.png")


class ReplayBenchmarkTests(unittest.TestCase):
    def test_journal_append_requests_flush_and_fsync_before_close(self):
        events = []
        class Stream:
            def __enter__(self): return self
            def __exit__(self, *args): events.append("close")
            def write(self, payload): events.append(("write", payload))
            def flush(self): events.append("flush")
            def fileno(self): return 217
        with patch.object(Path, "open", return_value=Stream()) as opened:
            with patch("world_models.data.benchmark.os.fsync", side_effect=lambda descriptor: events.append(("fsync", descriptor))):
                _append_fsync(Path("owned-probe.bin"), b"measured-bytes")
        opened.assert_called_once_with("ab")
        self.assertEqual(events, [("write", b"measured-bytes"), "flush", ("fsync", 217), "close"])

    def test_fsync_probe_records_actual_named_locations_verified_bytes_and_cleans_owned_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locations = {name: root / name for name in ("journal_spool", "dataset_volume")}
            for path in locations.values(): path.mkdir()
            before = psutil.Process().cpu_affinity()
            report = benchmark_fsync_directories(root / "report", locations=locations,
                                                 writes=3, payload_bytes=4096, warmup_writes=1,
                                                 cpu_limit=min(4, len(before)))
            self.assertEqual(psutil.Process().cpu_affinity(), before)
            self.assertEqual(report["status"], "passed_preflight")
            self.assertEqual(set(report["locations"]), set(locations))
            self.assertEqual(report["simulator_interactions"], 0)
            for name, location in locations.items():
                case = report["locations"][name]
                self.assertEqual(case["directory"], str(location.resolve()))
                self.assertEqual(case["sample_count"], 3)
                self.assertEqual(case["total_fsynced_write_operations"], 4)
                self.assertEqual(case["total_verified_bytes"], 4 * 4096)
                self.assertEqual(len({sample["payload_sha256"] for sample in case["samples"]}), 3)
                self.assertTrue(case["payload_roundtrip_exact"])
                self.assertEqual(list(location.iterdir()), [])
            path = root / "report/durable_journal_io_probe.json"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), path.with_suffix(".sha256").read_text().split()[0])
            with self.assertRaises(FileExistsError):
                benchmark_fsync_directories(root / "report", locations=locations)

    def test_fsync_probe_error_retains_failure_record_and_removes_only_owned_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            location = root / "volume"
            location.mkdir()
            preserved = location / "unrelated.txt"
            preserved.write_text("keep this", encoding="utf-8")
            with patch("world_models.data.benchmark._append_fsync", side_effect=OSError("injected fsync failure")):
                with self.assertRaisesRegex(OSError, "fsync failure"):
                    benchmark_fsync_directories(root / "report", locations={"volume": location},
                                                 writes=1, warmup_writes=0,
                                                 cpu_limit=min(4, len(psutil.Process().cpu_affinity())))
            self.assertEqual(list(location.iterdir()), [preserved])
            self.assertEqual(preserved.read_text(), "keep this")
            self.assertEqual(json.loads((root / "report/error.json").read_text())["status"], "failed")
            self.assertFalse((root / "report/durable_journal_io_probe.json").exists())

    def test_fsync_probe_rejects_invalid_workloads_without_mutating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for options in ({"writes": 0}, {"writes": True}, {"payload_bytes": 0}, {"warmup_writes": -1}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    benchmark_fsync_directories(root / "report", locations={"volume": root}, **options)
            self.assertFalse((root / "report").exists())

    def test_actor_source_is_paired_uint8_exact_and_capped_across_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples(root)
            values, info = load_actor_samples(root, maximum_joint_observations=4)
            self.assertEqual(values.dtype, np.uint8)
            self.assertEqual(values.shape, (4, 2, 3, 96, 128))
            self.assertEqual(info["families"], {"relay": 2, "cover": 2})
            self.assertEqual(info["pixels_sha256"], hashlib.sha256(values.tobytes()).hexdigest())
            first = info["records"][0]
            for player in (0, 1):
                with Image.open(first["images"][player]["path"]) as image:
                    np.testing.assert_array_equal(values[0, player], np.asarray(image).transpose(2, 0, 1))

    def test_missing_partner_and_wrong_resolution_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples(root)
            partner = root / "relay_train_00/video/frame0000_player1.png"
            partner.unlink()
            with self.assertRaisesRegex(ValueError, "unpaired"):
                load_actor_samples(root)
            Image.new("RGB", (127, 96)).save(partner)
            with self.assertRaisesRegex(ValueError, "RGB128x96"):
                load_actor_samples(root)

    def test_selection_requires_space_savings_and_collection_headroom(self):
        def trial(compression, level, size, write, read):
            return {"chunk_transitions": 128, "compression": compression, "compression_level": level,
                    "stored_bytes": size, "write_joint_observations_per_second": write,
                    "read_joint_observations_per_second": read}
        baseline = trial("stored", 0, 1000, 1000, 1000)
        level1 = trial("deflate", 1, 800, 200, 200)
        level3 = trial("deflate", 3, 700, 120, 120)
        choice = choose_storage_setting([baseline, level1, level3])
        self.assertEqual((choice["compression"], choice["compression_level"]), ("deflate", 1))
        level1["write_joint_observations_per_second"] = 20
        self.assertEqual(choose_storage_setting([baseline, level1, level3])["compression_level"], 3)
        level3["stored_bytes"] = 950
        self.assertEqual(choose_storage_setting([baseline, level1, level3])["compression"], "stored")
        with self.assertRaisesRegex(ValueError, "stored baseline"):
            choose_storage_setting([level1])

    def test_real_replay_sampling_spans_collectors_and_skips_missing_images(self):
        class Episode:
            def __init__(self, identifier):
                self.identifier = identifier

            def state(self, index):
                missing = self.identifier == "scripted" and index == 2
                return {"rgb": np.full((2, 3, 96, 128), 200 if self.identifier == "scripted" else 20, dtype=np.uint8),
                        "image_valid": np.asarray([not missing, True], dtype=np.bool_)}

        class Dataset:
            def __init__(self, path, verify=True):
                self.path = Path(path)
                self.manifest = {"metadata": {"task_family": self.path.name}}
                self.episodes = [{"episode_id": "random", "observation_count": 7, "collector_kind": "random"},
                                 {"episode_id": "scripted", "observation_count": 3, "collector_kind": "scripted"}]

            def load_episode(self, identifier, purpose):
                if purpose != "diagnostic":
                    raise AssertionError("benchmark requested a fitting view")
                return Episode(identifier)

        with patch("world_models.data.replay.ReplayDataset", Dataset):
            rgb, report = load_replay_samples([Path("relay"), Path("cover")], maximum_joint_observations=20)
        self.assertEqual(rgb.shape, (18, 2, 3, 96, 128))
        self.assertEqual(report["families"], {"relay": 9, "cover": 9})
        self.assertEqual(sum(record["collector_kind"] == "scripted" for record in report["records"]), 4)
        self.assertEqual(int(np.sum(rgb[:, 0, 0, 0, 0] == 200)), 4)

    def test_codec_roundtrip_preflight_hash_receipt_and_affinity_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "benchmark"
            samples(source)
            process = psutil.Process()
            before = process.cpu_affinity()
            report = run_storage_benchmark(output, sample_root=source, cpu_limit=min(4, len(before)),
                                           minimum_seconds=0, chunk_sizes=(2, 4), sequence_length=2)
            self.assertEqual(process.cpu_affinity(), before)
            self.assertEqual(report["status"], "passed_preflight")
            self.assertEqual(len(report["trials"]), 6)
            self.assertTrue(report["all_payloads_exact"])
            self.assertTrue(all(row["write_passes"] >= 2 and row["read_windows"] >= 2 for row in report["trials"]))
            for row in report["trials"]:
                self.assertEqual(row["payload_pixels_sha256"], report["source"]["pixels_sha256"])
                self.assertTrue(row["payload_roundtrip_exact"])
                for chunk in row["chunks"]:
                    self.assertEqual(hashlib.sha256((output / chunk["path"]).read_bytes()).hexdigest(), chunk["sha256"])
            path = output / "storage_benchmark.json"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), path.with_suffix(".sha256").read_text().split()[0])
            self.assertEqual(json.loads(path.read_text())["status"], "passed_preflight")
            with self.assertRaises(FileExistsError):
                run_storage_benchmark(output, sample_root=source, minimum_seconds=0, chunk_sizes=(2, 4), sequence_length=2)

    def test_payload_mismatch_fails_instead_of_reporting_pass(self):
        from world_models.data.replay import read_npz

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples(root / "source")
            def wrong(path):
                arrays = read_npz(path)
                arrays["rgb"] = arrays["rgb"].copy()
                arrays["rgb"][0, 0, 0, 0, 0] ^= 255
                return arrays
            with patch("world_models.data.replay.read_npz", side_effect=wrong):
                with self.assertRaisesRegex(RuntimeError, "changed benchmark payload"):
                    run_storage_benchmark(root / "output", sample_root=root / "source", minimum_seconds=0,
                                          chunk_sizes=(2,), sequence_length=2,
                                          cpu_limit=min(4, len(psutil.Process().cpu_affinity())))
            failure = json.loads((root / "output/error.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertFalse((root / "output/storage_benchmark.json").exists())


if __name__ == "__main__":
    unittest.main()
