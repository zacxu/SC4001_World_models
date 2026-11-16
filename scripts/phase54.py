"""P5.4: evaluate a registered final local actor on the immutable test bank."""
from __future__ import annotations

import argparse
import json
import os


def _runtime_setup():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="explicit native final-test execution after all four training runs complete")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--resume", action="store_true", help="continue the same output, preserving first valid episodes and all failed costs")
    inspect = commands.add_parser("inspect", help="authenticate completed final evaluation without executing the engine")
    inspect.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            _runtime_setup()
            from world_models_endpoint.evaluation import evaluate_endpoint
            result = evaluate_endpoint(args.run_dir, args.output_dir, device=args.device, resume=args.resume)
        else:
            from world_models_endpoint.evaluation import load_evaluation
            result = load_evaluation(args.report)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"P5.4 error ({type(error).__name__}): {error}")
        return 2
    print(json.dumps({key: result[key] for key in (
        "status", "variant", "master_seed", "native_engine", "metrics", "interaction_accounting")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
