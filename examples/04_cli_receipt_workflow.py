"""Run and verify an isolated CLI smoke receipt without exposing volatile fields."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _prepare_empty_directory(path: str | Path) -> Path:
    """Create a caller-owned directory, refusing to mix with existing files."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new or empty caller-owned directory for this example workflow",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Active Fedference checkout used to bind Git and uv.lock provenance",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Exercise CLI run, receipt verification, and both isolation failures."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    try:
        workflow_dir = _prepare_empty_directory(args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    run_dir = workflow_dir / "server-theory-smoke"

    run_arguments = [
        "run",
        "server-theory",
        "--profile",
        "smoke",
        "--seed",
        "0",
        "--output-dir",
        str(run_dir),
        "--project-root",
        str(project_root),
    ]
    first_run = _run_cli(run_arguments, project_root=project_root)
    _require_success(first_run, "fedference run")
    advertised_paths = json.loads(first_run.stdout)
    receipt_path = Path(advertised_paths["receipt"])

    verification = _run_cli(
        ["verify", str(receipt_path)],
        project_root=project_root,
    )
    _require_success(verification, "fedference verify")

    repeated_run = _run_cli(run_arguments, project_root=project_root)
    repeated_run_rejected = (
        repeated_run.returncode == 2
        and "output directory must be empty" in repeated_run.stderr
    )
    if not repeated_run_rejected:
        raise RuntimeError("fedference run did not reject a non-empty evidence directory")

    protected_probe = project_root / "output" / ".example-cli-isolation-probe"
    if protected_probe.exists():
        raise RuntimeError(f"protected-output probe already exists: {protected_probe}")
    protected_arguments = list(run_arguments)
    protected_arguments[protected_arguments.index("--output-dir") + 1] = str(protected_probe)
    protected_run = _run_cli(protected_arguments, project_root=project_root)
    protected_output_rejected = (
        protected_run.returncode == 2
        and "may not write into the committed reviewer output tree" in protected_run.stderr
        and not protected_probe.exists()
    )
    if not protected_output_rejected:
        raise RuntimeError("fedference run did not protect the committed reviewer output tree")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    git_metadata_bound = (
        (
            isinstance(receipt["git_commit"], str)
            and len(receipt["git_commit"]) == 40
            and receipt["git_tree_state"] in {"clean", "dirty"}
        )
        or (
            receipt["git_commit"] == "unavailable"
            and receipt["git_tree_state"] == "unavailable"
        )
    )
    payload = {
        "committed_output_rejected": protected_output_rejected,
        "example": "cli_receipt_workflow",
        "receipt": {
            "experiment_id": receipt["experiment_id"],
            "git_metadata_bound": git_metadata_bound,
            "output_names": sorted(record["name"] for record in receipt["outputs"]),
            "profile": receipt["profile"],
            "schema_version": receipt["schema_version"],
            "seeds": receipt["seeds"],
            "status": receipt["status"],
        },
        "report_declares_no_claim": bool(report["experiment_spec"]["no_claim"]),
        "rerun_rejected": repeated_run_rejected,
        "run_files": sorted(path.name for path in run_dir.iterdir()),
        "verify_passed": verification.stdout.startswith("PASS:"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
