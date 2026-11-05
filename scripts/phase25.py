"""Compact P2.5 engineering screen; never starts training or simulator collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from world_models.evaluation.verification import verify_implementation
from world_models.training.runner import ROOT


def _positive_windows(value):
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("windows must be an integer from 1 to 64") from error
    if not 1 <= count <= 64:
        raise argparse.ArgumentTypeError("windows must be an integer from 1 to 64")
    return count


def _nonnegative(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return number


def _evaluation_args(command):
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--data-dir", type=Path, default=Path("data/initial_replay/pilot"))
    command.add_argument("--statistics-dir", type=Path, default=Path("artifacts/phase23/normalization"))
    command.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--windows", type=_positive_windows, default=8)
    command.add_argument("--seed", type=_nonnegative, default=0)
    command.add_argument("--precision", choices=("float32", "bfloat16"), default=None)
    command.add_argument("--verification-report", type=Path)


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    subs = command.add_subparsers(dest="command", required=True)
    verify = subs.add_parser("verify", help="implementation fixtures only; no research model evaluated")
    verify.add_argument("--output-dir", type=Path, required=True)
    evaluate = subs.add_parser("evaluate", help="screen one explicit checkpoint")
    _evaluation_args(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    campaign = subs.add_parser("campaign", help="screen relay/cover seeds 0/1 sequentially")
    _evaluation_args(campaign)
    campaign.add_argument("--run-root", type=Path, required=True)
    campaign.add_argument("--checkpoint-update", type=_nonnegative, required=True)
    return command


def evaluate_checkpoint(*args, **kwargs):
    # Keep implementation verification independent of checkpoint/data loaders.
    from world_models.evaluation.compact import evaluate_checkpoint as evaluate
    return evaluate(*args, **kwargs)


def evaluate_campaign(*args, **kwargs):
    from world_models.evaluation.compact import evaluate_campaign as evaluate
    return evaluate(*args, **kwargs)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_implementation(args.output_dir)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0
        kwargs = {"data_root": args.data_dir, "statistics_root": args.statistics_dir,
                  "config_root": args.config_dir, "device": args.device, "windows": args.windows,
                  "seed": args.seed, "precision": args.precision, "verification_report": args.verification_report}
        if args.command == "evaluate":
            result = evaluate_checkpoint(args.checkpoint, args.output_dir, **kwargs)
        else:
            result = evaluate_campaign(args.run_root, args.checkpoint_update, args.output_dir, **kwargs)
        readiness = result.get("pilot_readiness", {})
        print(json.dumps({"report_path": str(args.output_dir / "report.json"),
                          "pilot_readiness": readiness, "P2.5_quality_acceptance": False},
                         indent=2, sort_keys=True, allow_nan=False))
        return 1 if readiness.get("status") == "failed" else 0
    except Exception as error:
        print(f"P2.5 error ({type(error).__name__}): {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
