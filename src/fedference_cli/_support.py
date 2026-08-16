"""Shared safety and evidence helpers for the installed CLI.

This module owns the reusable boundary mechanics used by command handlers:
project-root resolution, explicit output-directory checks, deterministic JSON
writes, registry declarations, and content-bound run receipts.  It deliberately
does not choose an experiment or parse command-line arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fedference.aggregation import AggregationConfig
from fedference.evidence import (
    GitTreeState,
    RunReceipt,
    canonical_sha256,
    make_artifact_record,
    sha256_file,
    validate_evidence_report,
    write_run_receipt,
)
from fedference.research_registry import get_experiment_spec


def _write_json(path: Path, payload: object) -> Path:
    """Atomically write finite, sorted JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _resolve_project_root(path: str | Path) -> Path:
    """Resolve a nested checkout path to its Git root when available."""
    candidate = Path(path).resolve()
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedcache=false",
            "-C",
            str(candidate),
            "rev-parse",
            "--show-toplevel",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip()).resolve()
    return candidate


def _validate_output_dir(path: str, project_root: Path) -> Path:
    """Reject committed-output writes and non-empty run destinations."""
    output_dir = Path(path).resolve()
    reviewer_output = (project_root / "output").resolve()
    try:
        output_dir.relative_to(reviewer_output)
    except ValueError:
        pass
    else:
        raise ValueError("CLI research runs may not write into the committed reviewer output tree")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    return output_dir


def _prepare_output_dir(path: str, project_root: Path) -> Path:
    """Validate and create a caller-owned empty evidence directory."""
    output_dir = _validate_output_dir(path, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _require_locked_profile(profile: str, *, confirmatory_ready: bool) -> None:
    """Reject confirmatory execution until its registry contract is frozen."""
    if "confirmatory" in profile and not confirmatory_ready:
        raise ValueError(
            f"profile {profile!r} is blocked until its pilot freezes the "
            "effect, MCSE target, budget, comparison family, and configuration"
        )


def _git_revision(project_root: Path) -> tuple[str, GitTreeState]:
    """Return the live commit and hermetic Git-tree state for a run receipt."""
    commit_result = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedcache=false",
            "-C",
            str(project_root),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_result.returncode != 0 or not commit_result.stdout.strip():
        return "unavailable", "unavailable"
    status_result = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedcache=false",
            "-C",
            str(project_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    tree_state: GitTreeState = (
        "unavailable" if status_result.returncode != 0 else ("dirty" if status_result.stdout else "clean")
    )
    return commit_result.stdout.strip(), tree_state


def _validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Validate non-empty, unique, non-negative integer seed selections."""
    normalized = tuple(seeds)
    if not normalized:
        raise ValueError("seeds must be non-empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in normalized):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("seeds must be unique")
    return normalized


def _validate_benchmark_controls(args: argparse.Namespace) -> None:
    """Validate benchmark controls before creating output directories."""
    if args.n_clients < 1:
        raise ValueError("n_clients must be positive")
    if not 0 <= args.n_contaminated <= args.n_clients:
        raise ValueError("n_contaminated must lie in [0, n_clients]")
    if not np.isfinite(args.contamination_rate) or not 0.0 <= args.contamination_rate <= 1.0:
        raise ValueError("contamination_rate must lie in [0, 1]")
    AggregationConfig(method="robust", robustness=args.robustness)
    AggregationConfig(
        method="variational",
        robustness=args.robustness,
        entropy_weight=args.entropy_weight,
    )


def _environment_lock_digest(project_root: Path) -> tuple[str, tuple[str, ...]]:
    """Return the lockfile digest or an explicit unavailable fallback."""
    lock = project_root / "uv.lock"
    if lock.is_file():
        return sha256_file(lock), ()
    return (
        canonical_sha256({"environment_lock": "unavailable"}),
        ("uv.lock was unavailable to the installed CLI",),
    )


def _dataset_digests(report: dict[str, Any]) -> dict[str, str]:
    """Extract registered dataset archive digests from a report payload."""
    rows = report.get("rows")
    if isinstance(rows, list) and all(
        isinstance(row, dict) and "dataset_id" in row and "dataset_archive_sha256" in row for row in rows
    ):
        return {str(row["dataset_id"]): str(row["dataset_archive_sha256"]) for row in rows}
    if "dataset_id" in report:
        return {str(report["dataset_id"]): str(report["dataset_archive_sha256"])}
    return {}


def _report_fallbacks(report: dict[str, Any]) -> tuple[str, ...]:
    """Summarize algorithm fallbacks and non-convergence for a run receipt."""
    raw_rows = report.get("rows")
    rows = (
        tuple(row for row in raw_rows if isinstance(row, dict)) if isinstance(raw_rows, list) else (report,)
    )
    events: list[str] = []
    device = report.get("device")
    if isinstance(device, dict) and device.get("fallback"):
        events.append(f"device fallback: {device['fallback']}")
    for row in rows:
        dataset = str(row.get("dataset_id", row.get("dataset", "unregistered")))
        seed = row.get("seed", "unknown")
        n_test = row.get("n_test", "unknown")
        for method in ("naive", "robust", "variational"):
            fallback_count = row.get(f"{method}_fallback_predictions", 0)
            if (
                isinstance(fallback_count, int)
                and not isinstance(fallback_count, bool)
                and fallback_count > 0
            ):
                events.append(
                    "aggregation fallback: "
                    f"dataset={dataset} seed={seed} method={method} "
                    f"predictions={fallback_count}/{n_test}"
                )
            nonconverged_count = row.get(f"{method}_nonconverged_predictions", 0)
            if (
                isinstance(nonconverged_count, int)
                and not isinstance(nonconverged_count, bool)
                and nonconverged_count > 0
            ):
                events.append(
                    "aggregation non-convergence: "
                    f"dataset={dataset} seed={seed} method={method} "
                    f"predictions={nonconverged_count}/{n_test}"
                )
    return tuple(dict.fromkeys(events))


def _experiment_summary(experiment_id: str) -> dict[str, Any]:
    """Return the complete registry declaration embedded in a run report."""
    spec = get_experiment_spec(experiment_id)
    return {
        "experiment_id": spec.experiment_id,
        "version": spec.version,
        "state": spec.state,
        "source_bundle": list(spec.source_ids),
        "primary_estimand": spec.primary_estimand,
        "independent_unit": spec.independent_unit,
        "smallest_effect_of_interest": spec.smallest_effect_of_interest,
        "mcse_stopping_target": spec.mcse_stopping_target,
        "maximum_budget": spec.maximum_budget,
        "comparison_family": spec.comparison_family,
        "falsifier": spec.falsifier,
        "no_claim": spec.no_claim,
        "profiles": list(spec.profiles),
        "confirmatory_ready": spec.confirmatory_ready,
        "runner": spec.runner,
    }


def _write_evidence_run(
    *,
    output_dir: Path,
    project_root: Path,
    experiment_id: str,
    profile: str,
    seeds: tuple[int, ...],
    config: dict[str, Any],
    report: dict[str, Any],
    started_at: str,
    git_revision: tuple[str, GitTreeState],
    backend: str = "numpy",
    device: str | None = None,
    checkpoints: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    """Validate, write, and receipt-bind one CLI evidence run."""
    validate_evidence_report(report)
    config_path = _write_json(output_dir / "config.json", config)
    report_path = _write_json(output_dir / "report.json", report)
    spec = get_experiment_spec(experiment_id)
    lock_digest, environment_fallbacks = _environment_lock_digest(project_root)
    fallbacks = tuple(dict.fromkeys((*environment_fallbacks, *_report_fallbacks(report))))
    completed_at = datetime.now(timezone.utc).isoformat()
    receipt = RunReceipt(
        run_id=(
            f"{experiment_id}-{profile}-{canonical_sha256({'started_at': started_at, 'config': config})[:12]}"
        ),
        experiment_id=experiment_id,
        experiment_version=spec.version,
        profile=profile,
        git_commit=git_revision[0],
        git_tree_state=git_revision[1],
        environment_lock_sha256=lock_digest,
        config_sha256=canonical_sha256(config),
        dataset_sha256=_dataset_digests(report),
        seeds=seeds,
        device=device or f"{platform.system()}-{platform.machine()}",
        backend=backend,
        fallbacks=fallbacks,
        checkpoints=checkpoints,
        outputs=(
            make_artifact_record("config", config_path, root=output_dir),
            make_artifact_record("report", report_path, root=output_dir),
        ),
        status="completed",
        started_at_utc=started_at,
        completed_at_utc=completed_at,
    )
    receipt_path = write_run_receipt(output_dir / "receipt.json", receipt)
    return report_path, receipt_path


__all__ = [
    "_dataset_digests",
    "_environment_lock_digest",
    "_experiment_summary",
    "_git_revision",
    "_prepare_output_dir",
    "_report_fallbacks",
    "_require_locked_profile",
    "_resolve_project_root",
    "_validate_benchmark_controls",
    "_validate_output_dir",
    "_validate_seeds",
    "_write_evidence_run",
]
