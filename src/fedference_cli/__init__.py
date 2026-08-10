"""Installed command-line interface for evidence-bound Active Fedference runs."""

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
from fedference.benchmark import run_external_benchmark_pack
from fedference.evidence import (
    GitTreeState,
    RunReceipt,
    canonical_sha256,
    load_run_receipt,
    make_artifact_record,
    sha256_file,
    validate_evidence_report,
    verify_run_receipt,
    write_run_receipt,
)
from fedference.experiments import run_heuristic_characterization
from fedference.federation import load_socket_replay, validate_socket_replay
from fedference.hierarchy_tasks import run_hierarchy_task_pilot
from fedference.hybrid_tracking import run_hybrid_tracking_comparison
from fedference.protocol_parity import run_friston_protocol_audit
from fedference.research_registry import (
    DATASET_SPECS,
    EXPERIMENT_SPECS,
    get_experiment_spec,
    registry_fingerprint,
    registry_manifest,
)
from fedference.single_machine import run_calibration_pilot, run_fedgvi_bnn_pilot


def _write_json(path: Path, payload: object) -> Path:
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
    output_dir = _validate_output_dir(path, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _require_locked_profile(profile: str, *, confirmatory_ready: bool) -> None:
    if "confirmatory" in profile and not confirmatory_ready:
        raise ValueError(
            f"profile {profile!r} is blocked until its pilot freezes the "
            "effect, MCSE target, budget, comparison family, and configuration"
        )


def _git_revision(project_root: Path) -> tuple[str, GitTreeState]:
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
    normalized = tuple(seeds)
    if not normalized:
        raise ValueError("seeds must be non-empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in normalized):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("seeds must be unique")
    return normalized


def _validate_benchmark_controls(args: argparse.Namespace) -> None:
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
    lock = project_root / "uv.lock"
    if lock.is_file():
        return sha256_file(lock), ()
    return (
        canonical_sha256({"environment_lock": "unavailable"}),
        ("uv.lock was unavailable to the installed CLI",),
    )


def _dataset_digests(report: dict[str, Any]) -> dict[str, str]:
    rows = report.get("rows")
    if isinstance(rows, list) and all(
        isinstance(row, dict) and "dataset_id" in row and "dataset_archive_sha256" in row
        for row in rows
    ):
        return {str(row["dataset_id"]): str(row["dataset_archive_sha256"]) for row in rows}
    if "dataset_id" in report:
        return {str(report["dataset_id"]): str(report["dataset_archive_sha256"])}
    return {}


def _report_fallbacks(report: dict[str, Any]) -> tuple[str, ...]:
    """Summarize algorithm fallbacks and non-convergence for the run receipt."""
    raw_rows = report.get("rows")
    rows = (
        tuple(row for row in raw_rows if isinstance(row, dict))
        if isinstance(raw_rows, list)
        else (report,)
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
    """Return the complete governance declaration embedded in run reports."""
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
    validate_evidence_report(report)
    config_path = _write_json(output_dir / "config.json", config)
    report_path = _write_json(output_dir / "report.json", report)
    spec = get_experiment_spec(experiment_id)
    lock_digest, environment_fallbacks = _environment_lock_digest(project_root)
    fallbacks = tuple(
        dict.fromkeys(
            (
                *environment_fallbacks,
                *_report_fallbacks(report),
            )
        )
    )
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


def _run_command(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project_root)
    spec = get_experiment_spec(args.experiment_id)
    if args.profile not in spec.profiles:
        raise ValueError(f"profile {args.profile!r} is not declared for {args.experiment_id!r}")
    _require_locked_profile(
        args.profile,
        confirmatory_ready=spec.confirmatory_ready,
    )
    if spec.runner is None:
        raise ValueError(f"{args.experiment_id!r} is registry-declared but has no executable CLI runner")
    _validate_output_dir(args.output_dir, project_root)
    git_revision = _git_revision(project_root)
    started_at = datetime.now(timezone.utc).isoformat()
    seeds = _validate_seeds(args.seed)
    config: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "profile": args.profile,
        "seeds": list(seeds),
        "registry_fingerprint": registry_fingerprint(),
        "runner": spec.runner,
    }
    if args.experiment_id == "fedgvi-bnn":
        config["requested_device"] = args.device
    if args.experiment_id == "server-theory":
        if len(seeds) != 1:
            raise ValueError("server-theory runner accepts exactly one seed")
        report = run_heuristic_characterization(seeds[0])
    elif args.experiment_id == "external-tabular":
        if args.cache_dir is None:
            raise ValueError("external-tabular requires --cache-dir")
        dataset_ids = (
            ("uci-banknote",)
            if args.profile == "smoke"
            else tuple(dataset.dataset_id for dataset in DATASET_SPECS)
        )
        config["dataset_ids"] = list(dataset_ids)
        report = run_external_benchmark_pack(
            cache_dir=args.cache_dir,
            seeds=seeds,
            dataset_ids=dataset_ids,
        )
    elif args.experiment_id == "robustness-calibration":
        if len(seeds) != 1:
            raise ValueError("robustness-calibration runner accepts exactly one seed")
        report = run_calibration_pilot(seed=seeds[0], profile=args.profile)
    elif args.experiment_id == "fedgvi-bnn":
        if args.profile == "source_5090":
            raise ValueError("source_5090 is declarative and is not executed on this workstation")
        if len(seeds) != 1:
            raise ValueError("fedgvi-bnn runner accepts exactly one seed per evidence directory")
        bnn_profile = "pilot" if args.profile == "pilot" else "smoke"
        report = run_fedgvi_bnn_pilot(
            seed=seeds[0],
            profile=bnn_profile,
            requested_device=args.device,
        )
    elif args.experiment_id == "hybrid-tracking":
        runs = [
            run_hybrid_tracking_comparison(seed=seed)
            for seed in seeds
        ]
        report = runs[0] if len(runs) == 1 else {
            "status": "pilot",
            "runs": runs,
            "seeds": list(seeds),
            "primary_estimand": "held-out posterior-predictive log score per seeded tracking world",
            "independent_unit": "seeded tracking world",
            "no_claim": "multiple pilot seeds do not establish general continuous control",
        }
    elif args.experiment_id == "hierarchy-tasks":
        report = run_hierarchy_task_pilot(seeds=seeds)
    elif args.experiment_id == "friston-protocol":
        if len(seeds) != 1:
            raise ValueError("friston-protocol runner accepts exactly one seed")
        report = run_friston_protocol_audit()
    else:
        raise ValueError(f"{args.experiment_id!r} is registry-declared but has no executable CLI runner")
    output_dir = _prepare_output_dir(args.output_dir, project_root)
    report["experiment_spec"] = _experiment_summary(spec.experiment_id)
    report_path, receipt_path = _write_evidence_run(
        output_dir=output_dir,
        project_root=project_root,
        experiment_id=args.experiment_id,
        profile=args.profile,
        seeds=seeds,
        config=config,
        report=report,
        started_at=started_at,
        git_revision=git_revision,
        backend=str(report.get("backend", "numpy")),
        device=(
            str(report["device"].get("resolved"))
            if isinstance(report.get("device"), dict) and report["device"].get("resolved")
            else None
        ),
        checkpoints=tuple(str(value) for value in report.get("checkpoint_fingerprints", [])),
    )
    print(json.dumps({"report": str(report_path), "receipt": str(receipt_path)}))
    return 0


def _benchmark_command(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project_root)
    spec = get_experiment_spec("external-tabular")
    _require_locked_profile(
        args.profile,
        confirmatory_ready=spec.confirmatory_ready,
    )
    _validate_output_dir(args.output_dir, project_root)
    _validate_benchmark_controls(args)
    git_revision = _git_revision(project_root)
    started_at = datetime.now(timezone.utc).isoformat()
    dataset_ids = (
        tuple(spec.dataset_id for spec in DATASET_SPECS) if args.dataset_id == "all" else (args.dataset_id,)
    )
    seeds = _validate_seeds(args.seed)
    config = {
        "experiment_id": "external-tabular",
        "profile": args.profile,
        "dataset_ids": list(dataset_ids),
        "seeds": list(seeds),
        "n_clients": args.n_clients,
        "n_contaminated": args.n_contaminated,
        "contamination_rate": args.contamination_rate,
        "robustness": args.robustness,
        "entropy_weight": args.entropy_weight,
        "registry_fingerprint": registry_fingerprint(),
    }
    report = run_external_benchmark_pack(
        cache_dir=args.cache_dir,
        seeds=seeds,
        dataset_ids=dataset_ids,
        n_clients=args.n_clients,
        n_contaminated=args.n_contaminated,
        contamination_rate=args.contamination_rate,
        robustness=args.robustness,
        entropy_weight=args.entropy_weight,
    )
    output_dir = _prepare_output_dir(args.output_dir, project_root)
    report["experiment_spec"] = _experiment_summary(spec.experiment_id)
    report_path, receipt_path = _write_evidence_run(
        output_dir=output_dir,
        project_root=project_root,
        experiment_id="external-tabular",
        profile=args.profile,
        seeds=seeds,
        config=config,
        report=report,
        started_at=started_at,
        git_revision=git_revision,
    )
    print(json.dumps({"report": str(report_path), "receipt": str(receipt_path)}))
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    receipt_path = Path(args.receipt).resolve()
    receipt = load_run_receipt(receipt_path)
    root = Path(args.root).resolve() if args.root else receipt_path.parent
    project_root = (
        _resolve_project_root(args.project_root)
        if args.project_root is not None
        else (_resolve_project_root(".") if args.require_clean_git else None)
    )
    findings = verify_run_receipt(
        receipt,
        root=root,
        require_clean_git=args.require_clean_git,
        project_root=project_root,
    )
    if findings:
        for finding in findings:
            print(f"FAIL: {finding}")
        return 1
    print(f"PASS: {receipt.run_id} (git_tree_state={receipt.git_tree_state})")
    return 0


def _replay_command(args: argparse.Namespace) -> int:
    replay = load_socket_replay(args.replay)
    beliefs_raw = json.loads(Path(args.beliefs).read_text(encoding="utf-8"))
    consensus_raw = json.loads(Path(args.consensus).read_text(encoding="utf-8"))
    config = AggregationConfig(
        method=args.method,
        robustness=args.robustness,
        entropy_weight=args.entropy_weight,
        max_iter=args.max_iter,
        tol=args.tol,
        multistart=not args.single_start,
    )
    valid = validate_socket_replay(
        replay,
        np.asarray(beliefs_raw, dtype=np.float64),
        np.asarray(consensus_raw, dtype=np.float64),
        config=config,
    )
    print("PASS" if valid else "FAIL")
    return 0 if valid else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fedference",
        description="Run and verify source-bound Active Fedference evidence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list research registry entries")
    list_parser.add_argument("--json", action="store_true", dest="as_json")

    run_parser = subparsers.add_parser("run", help="run one executable experiment")
    run_parser.add_argument(
        "experiment_id",
        choices=[spec.experiment_id for spec in EXPERIMENT_SPECS],
    )
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
    """CLI entry point; return a process-compatible status code."""
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
            return _run_command(args)
        if args.command == "benchmark":
            if not args.seed:
                args.seed = [0]
            return _benchmark_command(args)
        if args.command == "verify":
            return _verify_command(args)
        if args.command == "replay":
            return _replay_command(args)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"unhandled command: {args.command}")


__all__ = ["main"]
