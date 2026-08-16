"""Argument parser and process entrypoint for the installed CLI."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Sequence

from fedference.research_registry import DATASET_SPECS, EXPERIMENT_SPECS, registry_manifest

from ._commands import _benchmark_command, _replay_command, _run_command, _verify_command


def _build_parser() -> argparse.ArgumentParser:
    """Build the stable ``fedference`` subcommand parser."""
    parser = argparse.ArgumentParser(
        prog="fedference",
        description="Run and verify source-bound Active Fedference evidence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list research registry entries")
    list_parser.add_argument("--json", action="store_true", dest="as_json")

    run_parser = subparsers.add_parser("run", help="run one executable experiment")
    run_parser.add_argument("experiment_id", choices=[spec.experiment_id for spec in EXPERIMENT_SPECS])
    run_parser.add_argument("--profile", default="smoke")
    run_parser.add_argument("--seed", type=int, action="append", default=[])
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--cache-dir")
    run_parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu")
    run_parser.add_argument("--project-root", default=".")

    benchmark_parser = subparsers.add_parser("benchmark", help="run the registered external dataset pack")
    benchmark_parser.add_argument(
        "--dataset-id",
        choices=["all", *(spec.dataset_id for spec in DATASET_SPECS)],
        default="all",
    )
    benchmark_parser.add_argument("--profile", choices=["smoke", "pilot", "confirmatory"], default="smoke")
    benchmark_parser.add_argument("--seed", type=int, action="append", default=[])
    benchmark_parser.add_argument("--n-clients", type=int, default=5)
    benchmark_parser.add_argument("--n-contaminated", type=int, default=2)
    benchmark_parser.add_argument("--contamination-rate", type=float, default=1.0)
    benchmark_parser.add_argument("--robustness", type=float, default=1.5)
    benchmark_parser.add_argument("--entropy-weight", type=float, default=1.0)
    benchmark_parser.add_argument("--cache-dir", required=True)
    benchmark_parser.add_argument("--output-dir", required=True)
    benchmark_parser.add_argument("--project-root", default=".")

    verify_parser = subparsers.add_parser("verify", help="verify a run receipt")
    verify_parser.add_argument("receipt")
    verify_parser.add_argument("--root")
    verify_parser.add_argument(
        "--project-root",
        help=(
            "checkout whose live commit, tree state, and uv.lock must match the "
            "receipt; defaults to the current checkout in strict mode"
        ),
    )
    verify_parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="fail unless the run receipt records a clean Git tree",
    )

    replay_parser = subparsers.add_parser("replay", help="verify a socket replay against supplied beliefs")
    replay_parser.add_argument("--replay", required=True)
    replay_parser.add_argument("--beliefs", required=True)
    replay_parser.add_argument("--consensus", required=True)
    replay_parser.add_argument("--method", choices=["naive", "robust", "variational"], default="robust")
    replay_parser.add_argument("--robustness", type=float, default=0.0)
    replay_parser.add_argument("--entropy-weight", type=float, default=1.0)
    replay_parser.add_argument("--max-iter", type=int, default=64)
    replay_parser.add_argument("--tol", type=float, default=1e-9)
    replay_parser.add_argument("--single-start", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, dispatch a command, and return a process status code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            if args.as_json:
                print(json.dumps(registry_manifest(), indent=2, sort_keys=True))
            else:
                for spec in EXPERIMENT_SPECS:
                    runner = "executable" if spec.runner else "declared"
                    print(f"{spec.experiment_id:24} {spec.state:8} {runner:10} {spec.title}")
            return 0
        if args.command == "run":
            if not args.seed:
                args.seed = [0]
            args.started_at = datetime.now(timezone.utc).isoformat()
            return _run_command(args)
        if args.command == "benchmark":
            if not args.seed:
                args.seed = [0]
            args.started_at = datetime.now(timezone.utc).isoformat()
            return _benchmark_command(args)
        if args.command == "verify":
            return _verify_command(args)
        if args.command == "replay":
            return _replay_command(args)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"unhandled command: {args.command}")


__all__ = ["main"]
