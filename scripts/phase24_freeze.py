"""Freeze latest saved shared P2.4 checkpoints for a shortened pilot; never train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_models.training.selection import freeze_selection, load_selection


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-root", type=Path, help="stopped, fully synchronized initial training run")
    mode.add_argument("--inspect", type=Path, help="verify and display an existing selection JSON")
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--reason", help="record the authorized shortened-study decision")
    command.add_argument("--data-root", type=Path)
    command.add_argument("--statistics-root", type=Path)
    command.add_argument("--config-root", type=Path)
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.run_root and (args.output_dir is None or args.reason is None):
        command.error("--run-root requires --output-dir and --reason")
    if args.inspect and any(getattr(args, key) is not None for key in
                            ("output_dir", "reason", "data_root", "statistics_root", "config_root")):
        command.error("--inspect cannot be combined with selection creation options")
    try:
        if args.run_root:
            manifest = freeze_selection(args.run_root, args.output_dir, reason=args.reason,
                data_root=args.data_root, statistics_root=args.statistics_root, config_root=args.config_root)
            path = args.output_dir / "selection.json"
        else:
            path = args.inspect
            manifest = load_selection(path)
        for row in manifest["cases"]:
            print(f"{row['task']}/seed_{row['seed']}: {row['model_updates']} saved optimizer updates "
                  f"(original target {row['original_target_stage_updates']})")
        print(json.dumps({"selection": str(path), "P2.5_quality_acceptance": False,
                          "full_schedule_completion_claim": False}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"P2.4 selection error ({type(error).__name__}): {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
