"""P5.3: inspect cumulative interaction/compute costs without scientific work."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cohort = commands.add_parser("ledger", help="snapshot all registered run costs")
    cohort.add_argument("--cohort", required=True)
    cohort.add_argument("--output-dir", required=True, help="append-only snapshot directory; reuse to record progress")
    run = commands.add_parser("run", help="inspect one registered run")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--output-dir", help="optional append-only snapshot directory")
    args = parser.parse_args(argv)
    try:
        from world_models_endpoint.ledger import snapshot_run, summarize_cohort
        result = (summarize_cohort(args.cohort, args.output_dir) if args.command == "ledger" else
                  snapshot_run(args.run_dir, args.output_dir))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"P5.3 error ({type(error).__name__}): {error}", file=sys.stderr)
        return 1
    keys = ("status", "scope", "snapshot_path", "snapshot_sequence", "new_snapshot_written",
            "missing_run_slots", "shared_initial_costs", "sum_run_logical_counters", "counters",
            "evaluation_joint_transitions", "evaluation_known_simulator_tics", "evaluation_unknown_dispatch_attempts")
    print(json.dumps({key: result[key] for key in keys if key in result}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
