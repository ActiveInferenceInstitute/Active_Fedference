"""Aggregate labeled caller data through Python and the receipt-writing CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from fedference import (
    LabeledAggregationRequest,
    LabeledAggregationResult,
    aggregate_labeled,
)


def _prepare_empty_directory(path: str | Path) -> Path:
    """Create a caller-owned directory without deleting or overwriting content."""
    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _run_cli(arguments: Sequence[str], *, project_root: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the installed public CLI as a real child process."""
    return subprocess.run(
        [sys.executable, "-m", "fedference_cli", *arguments],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _require_success(result: subprocess.CompletedProcess[str], operation: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(
            f"{operation} failed with exit code {result.returncode}: {result.stderr.strip()}"
        )


def _build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(project_root / "examples" / "data" / "labeled_aggregation_request.json"),
        help="strict labeled request JSON (default: packaged example data)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new or empty caller-owned directory for request, result, and receipt",
    )
    parser.add_argument(
        "--project-root",
        default=str(project_root),
        help="Active Fedference checkout bound into source provenance",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the labeled API and prove the CLI artifacts encode the same result."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    try:
        output_dir = _prepare_empty_directory(args.output_dir)
        request = LabeledAggregationRequest.from_json(
            input_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    python_result = aggregate_labeled(request)
    aggregation = _run_cli(
        [
            "aggregate",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--project-root",
            str(project_root),
        ],
        project_root=project_root,
    )
    _require_success(aggregation, "fedference aggregate")
    advertised_paths = json.loads(aggregation.stdout)
    result_path = Path(advertised_paths["result"])
    receipt_path = Path(advertised_paths["receipt"])
    cli_result = LabeledAggregationResult.from_json(
        result_path.read_text(encoding="utf-8")
    )

    verification = _run_cli(
        ["verify", str(receipt_path), "--require-nominal-solver"],
        project_root=project_root,
    )
    _require_success(verification, "fedference verify")

    canonical_request = json.loads((output_dir / "request.json").read_text(encoding="utf-8"))
    payload = {
        "artifacts": sorted(path.name for path in output_dir.iterdir()),
        "canonical_request": {
            "agent_ids": [agent["agent_id"] for agent in canonical_request["agents"]],
            "default_weight_expanded": canonical_request["agents"][1]["base_weight"] == 1.0,
            "request_sha256": request.request_sha256,
            "state_labels": canonical_request["state_labels"],
        },
        "example": "labeled_application",
        "receipt": {
            "artifact_integrity_verified": verification.stdout.startswith(
                "PASS: application artifact integrity verified"
            ),
            "nominal_solver_required": "PASS: solver_status=nominal" in verification.stdout,
            "proves_scientific_validity": False,
        },
        "result": {
            "consensus": [round(float(value), 6) for value in cli_result.aggregation.consensus],
            "normalized_first_row": [
                round(float(value), 6)
                for value in cli_result.normalized_local_posteriors[0]
            ],
            "python_cli_bit_identical": bool(
                python_result.as_dict() == cli_result.as_dict()
            ),
            "solver_status": cli_result.aggregation.solver_status,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
