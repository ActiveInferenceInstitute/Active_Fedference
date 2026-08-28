"""Argument parser and process entrypoint for the installed CLI."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Sequence

from fedference.research_registry import DATASET_SPECS, EXPERIMENT_SPECS, registry_manifest

from ._commands import (
    _aggregate_command,
    _benchmark_command,
    _replay_command,
    _run_command,
    _verify_command,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the stable ``fedference`` subcommand parser."""
    parser = argparse.ArgumentParser(
        prog="fedference",
        description="Run and verify source-bound Active Fedference evidence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list research registry entries")
    list_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the complete registry manifest as sorted JSON",
    )

    run_parser = subparsers.add_parser("run", help="run one executable experiment")
    run_parser.add_argument(
        "experiment_id",
        choices=[spec.experiment_id for spec in EXPERIMENT_SPECS if spec.runner is not None],
        help="registry experiment with an executable runner (inspect declarations with 'list')",
    )
    run_parser.add_argument(
        "--profile",
        default="smoke",
        help="registry-declared execution profile for this experiment (default: smoke)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=[],
        help="non-negative seed; repeat for runners that accept multiple seeds (default: 0)",
    )
    run_parser.add_argument(
        "--output-dir",
        required=True,
        help="new or empty caller-owned directory outside the committed output/ tree",
    )
    run_parser.add_argument(
        "--cache-dir",
        help="caller-owned dataset cache; required only for external-tabular",
    )
    run_parser.add_argument(
        "--device",
        choices=["cpu", "mps", "auto"],
        default="cpu",
        help="requested device for fedgvi-bnn (default: cpu; ignored by other runners)",
    )
    run_parser.add_argument(
        "--project-root",
        default=".",
        help="source checkout bound into the receipt (default: current directory)",
    )

    benchmark_parser = subparsers.add_parser("benchmark", help="run the registered external dataset pack")
    benchmark_parser.add_argument(
        "--dataset-id",
        choices=["all", *(spec.dataset_id for spec in DATASET_SPECS)],
        default="all",
        help="registered dataset to execute, or all registered datasets (default: all)",
    )
    benchmark_parser.add_argument(
        "--profile",
        choices=["smoke", "pilot", "confirmatory"],
        default="smoke",
        help="execution profile (default: smoke; confirmatory is blocked until frozen)",
    )
    benchmark_parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=[],
        help="non-negative split/training seed; repeat for multiple seeds (default: 0)",
    )
    benchmark_parser.add_argument("--n-clients", type=int, default=5, help="client count (default: 5)")
    benchmark_parser.add_argument(
        "--n-contaminated",
        type=int,
        default=2,
        help="contaminated client count (default: 2)",
    )
    benchmark_parser.add_argument(
        "--contamination-rate",
        type=float,
        default=1.0,
        help="contamination rate in [0, 1] (default: 1.0)",
    )
    benchmark_parser.add_argument(
        "--robustness",
        type=float,
        default=1.5,
        help="non-negative robust aggregation strength (default: 1.5)",
    )
    benchmark_parser.add_argument(
        "--entropy-weight",
        type=float,
        default=1.0,
        help="non-negative variational entropy weight (default: 1.0)",
    )
    benchmark_parser.add_argument(
        "--cache-dir",
        required=True,
        help="caller-owned cache for hash-checked registered dataset archives",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        required=True,
        help="new or empty caller-owned directory outside the committed output/ tree",
    )
    benchmark_parser.add_argument(
        "--project-root",
        default=".",
        help="source checkout bound into the receipt (default: current directory)",
    )

    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="aggregate a strict labeled categorical request",
    )
    aggregate_parser.add_argument(
        "--input",
        required=True,
        help="labeled aggregation request JSON",
    )
    aggregate_parser.add_argument(
        "--output-dir",
        required=True,
        help="new or empty caller-owned directory outside the committed output/ tree",
    )
    aggregate_parser.add_argument(
        "--project-root",
        help="explicit Git checkout to bind into source provenance",
    )
    aggregate_parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="fail before writing unless the selected/current Git checkout is clean",
    )

    verify_parser = subparsers.add_parser("verify", help="verify a run receipt")
    verify_parser.add_argument("receipt")
    verify_parser.add_argument(
        "--root",
        help="artifact root for receipt-relative config/report paths (default: receipt directory)",
    )
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
    verify_parser.add_argument(
        "--require-nominal-solver",
        action="store_true",
        help="fail unless an application receipt records nominal solver health",
    )

    replay_parser = subparsers.add_parser("replay", help="verify a socket replay against supplied beliefs")
    replay_parser.add_argument("--replay", required=True, help="persisted digest-only replay JSON")
    replay_parser.add_argument("--beliefs", required=True, help="JSON matrix of original worker beliefs")
    replay_parser.add_argument("--consensus", required=True, help="JSON vector of the reported consensus")
    replay_parser.add_argument(
        "--method",
        choices=["naive", "robust", "variational"],
        default=None,
        help="assert the recorded aggregation method (default: read replay)",
    )
    replay_parser.add_argument(
        "--robustness",
        type=float,
        default=None,
        help="assert the recorded non-negative robustness strength",
    )
    replay_parser.add_argument(
        "--entropy-weight",
        type=float,
        default=None,
        help="assert the recorded variational entropy weight",
    )
    replay_parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="assert the recorded solver iteration budget",
    )
    replay_parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help="assert the recorded solver tolerance",
    )
    multistart_group = replay_parser.add_mutually_exclusive_group()
    multistart_group.add_argument(
        "--multistart",
        action="store_true",
        dest="multistart",
        help="assert that variational multistart was enabled",
    )
    multistart_group.add_argument(
        "--single-start",
        action="store_false",
        dest="multistart",
        help="assert that variational multistart was disabled",
    )
    replay_parser.set_defaults(multistart=None)
    replay_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the structured replay verdict as sorted JSON",
    )
    replay_parser.add_argument(
        "--require-nominal-solver",
        action="store_true",
        help="fail when replay integrity passes but solver health is non-nominal",
    )
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
        if args.command == "aggregate":
            args.started_at = datetime.now(timezone.utc).isoformat()
            return _aggregate_command(args)
        if args.command == "verify":
            return _verify_command(args)
        if args.command == "replay":
            return _replay_command(args)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"unhandled command: {args.command}")


__all__ = ["main"]
