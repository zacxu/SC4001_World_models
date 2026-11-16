"""P5.5: analyze saved final native endpoint evaluations without new episodes."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="authenticate all four saved final actors and paired episode banks")
    analyze.add_argument("--cohort", required=True, help="prospectively frozen endpoint cohort manifest")
    analyze.add_argument("--evaluation", required=True, action="append", help="saved evaluation.json; repeat four times")
    analyze.add_argument("--output-dir", required=True, help="fresh directory for immutable JSON and CSV tables")
    args = parser.parse_args(argv)
    try:
        from world_models_endpoint.analysis import analyze_cohort
        report = analyze_cohort(args.cohort, args.evaluation, args.output_dir)
        print(json.dumps({key: report[key] for key in ("status", "reasons", "cohort_identity", "report_path",
            "research_training_executed", "research_evaluation_executed", "missing_evidence_imputed")},
            indent=2, sort_keys=True, allow_nan=False))
        return 0 if report["status"] == "complete" else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"P5.5 error ({type(error).__name__}): {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
