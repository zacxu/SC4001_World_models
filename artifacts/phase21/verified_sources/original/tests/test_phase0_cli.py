"""End-to-end foundation entry points without a GPU/simulator dependency."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.phase0 import configuration_report, run_matrix


ROOT = Path(__file__).resolve().parents[1]


class Phase0CommandTests(unittest.TestCase):
    def test_every_comparison_resolves_to_matching_local_learning_budgets(self):
        for experiment, expected in (("pilot", 220), ("final", 860)):
            report = configuration_report(experiment)
            self.assertEqual(len(report["pairs"]), 2)
            for pair in report["pairs"]:
                for variant in ("B", "D"):
                    self.assertEqual(pair[variant]["derived"]["local_ppo_cycles"], expected)

    def test_run_matrices_are_unexecuted_unique_and_have_no_seed_collisions(self):
        for experiment, count in (("pilot", 8), ("final", 40)):
            report = run_matrix(experiment)
            self.assertEqual(report["run_count"], count)
            self.assertEqual(report["seed_collisions"], [])
            self.assertEqual(len({run["run_id"] for run in report["runs"]}), count)
            self.assertTrue(all(run["execution_status"] == "not_started" for run in report["runs"]))

    def test_module_command_writes_a_parseable_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "matrix.json"
            completed = subprocess.run(
                [sys.executable, "-m", "scripts.phase0", "matrix", "--experiment", "pilot", "--output", str(destination)],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "passed")
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8"))["run_count"], 8)

    def test_command_rejects_an_unapproved_experiment(self):
        completed = subprocess.run(
            [sys.executable, "-m", "scripts.phase0", "matrix", "--experiment", "unapproved"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("invalid choice", completed.stderr)


if __name__ == "__main__":
    unittest.main()
