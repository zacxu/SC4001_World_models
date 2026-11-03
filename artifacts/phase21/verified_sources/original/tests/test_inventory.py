import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from world_models.infrastructure.inventory import _gpu_inventory, collect_inventory


class InventoryTests(unittest.TestCase):
    @patch("world_models.infrastructure.inventory.shutil.which", return_value=None)
    @patch("world_models.infrastructure.inventory.subprocess.run")
    def test_missing_nvidia_smi_is_not_claimed_as_no_hardware(self, run, which):
        result = _gpu_inventory()
        self.assertEqual(result["status"], "not_found")
        self.assertIn("does not establish", result["note"])
        run.assert_not_called()

    @patch("world_models.infrastructure.inventory.shutil.which", return_value="nvidia-smi")
    @patch("world_models.infrastructure.inventory.subprocess.run")
    def test_gpu_query_is_parsed_without_importing_training_stack(self, run, which):
        run.return_value = subprocess.CompletedProcess([], 0, "0, NVIDIA H200, 143771, 570.0\n1, NVIDIA H200, 143771, 570.0\n", "")
        result = _gpu_inventory()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["gpus"]), 2)
        self.assertEqual(result["gpus"][0]["memory_total_mib"], 143771)
        self.assertEqual(run.call_args.kwargs["timeout"], 10)

    @patch("world_models.infrastructure.inventory.shutil.which", return_value="nvidia-smi")
    @patch("world_models.infrastructure.inventory.subprocess.run")
    def test_probe_failures_are_inventory_data(self, run, which):
        for output in [
            subprocess.CompletedProcess([], 1, "", "driver unavailable"),
            subprocess.CompletedProcess([], 0, "invalid row", ""),
        ]:
            with self.subTest(output=output):
                run.side_effect = None
                run.return_value = output
                self.assertEqual(_gpu_inventory()["status"], "error")
        run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 10)
        self.assertEqual(_gpu_inventory()["status"], "timeout")
        run.side_effect = OSError("not executable")
        self.assertEqual(_gpu_inventory()["status"], "error")

    @patch("world_models.infrastructure.inventory.shutil.which", return_value=None)
    def test_inventory_separates_requested_resources_from_observations(self, which):
        with tempfile.TemporaryDirectory() as folder:
            result = collect_inventory(Path(folder))
        self.assertEqual(result["requested_resources"]["gpu_count"], 4)
        self.assertIsNone(result["requested_resource_comparison"]["observed_h200_count"])
        self.assertEqual(result["disk"]["status"], "ok")
        self.assertIn("not been verified", result["scope"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
