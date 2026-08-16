"""Command handlers for the evidence-bound Active Fedference CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fedference.aggregation import AggregationConfig
from fedference.benchmark import run_external_benchmark_pack
from fedference.evidence import load_run_receipt, verify_run_receipt
from fedference.experiments import run_heuristic_characterization
from fedference.federation import load_socket_replay, validate_socket_replay
from fedference.hierarchy_tasks import run_hierarchy_task_pilot
from fedference.hybrid_tracking import run_hybrid_tracking_comparison
from fedference.protocol_parity import run_friston_protocol_audit
from fedference.research_registry import DATASET_SPECS, get_experiment_spec, registry_fingerprint
from fedference.single_machine import run_calibration_pilot, run_fedgvi_bnn_pilot

from ._support import (
    _experiment_summary,
    _git_revision,
    _prepare_output_dir,
    _require_locked_profile,
    _resolve_project_root,
    _validate_benchmark_controls,
    _validate_output_dir,
    _validate_seeds,
    _write_evidence_run,
)


def _run_command(args: argparse.Namespace) -> int:
    """Run one registry-declared executable experiment and write its receipt."""
    project_root = _resolve_project_root(args.project_root)
    spec = get_experiment_spec(args.experiment_id)
    if args.profile not in spec.profiles:
        raise ValueError(f"profile {args.profile!r} is not declared for {args.experiment_id!r}")
    _require_locked_profile(args.profile, confirmatory_ready=spec.confirmatory_ready)
    if spec.runner is None:
        raise ValueError(f"{args.experiment_id!r} is registry-declared but has no executable CLI runner")
    _validate_output_dir(args.output_dir, project_root)
    git_revision = _git_revision(project_root)
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
            raise ValueError("fedgvi-bnn runner accepts one seed per evidence directory")
        bnn_profile = "pilot" if args.profile == "pilot" else "smoke"
        report = run_fedgvi_bnn_pilot(
            seed=seeds[0],
            profile=bnn_profile,
            requested_device=args.device,
        )
    elif args.experiment_id == "hybrid-tracking":
        runs = [run_hybrid_tracking_comparison(seed=seed) for seed in seeds]
        report = (
            runs[0]
            if len(runs) == 1
            else {
                "status": "pilot",
                "runs": runs,
                "seeds": list(seeds),
                "primary_estimand": "held-out posterior-predictive log score per seeded tracking world",
                "independent_unit": "seeded tracking world",
                "no_claim": "multiple pilot seeds do not establish general continuous control",
            }
        )
    elif args.experiment_id == "hierarchy-tasks":
        report = run_hierarchy_task_pilot(seeds=seeds)
    elif args.experiment_id == "friston-protocol":
        if len(seeds) != 1:
            raise ValueError("friston-protocol runner accepts one seed")
        report = run_friston_protocol_audit()
    else:
        raise ValueError(f"{args.experiment_id!r} is registry-declared but has no executable CLI runner")
    output_dir = _prepare_output_dir(args.output_dir, project_root)
    report["experiment_spec"] = _experiment_summary(spec.experiment_id)
    report_path, receipt_path = _write_evidence_run(
        output_dir=output_dir,
        project_root=project_root,
        experiment_id=spec.experiment_id,
        profile=args.profile,
        seeds=seeds,
        config=config,
        report=report,
        started_at=args.started_at,
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
    """Run the registered external benchmark pack and write its receipt."""
    project_root = _resolve_project_root(args.project_root)
    spec = get_experiment_spec("external-tabular")
    _require_locked_profile(args.profile, confirmatory_ready=spec.confirmatory_ready)
    _validate_output_dir(args.output_dir, project_root)
    _validate_benchmark_controls(args)
    git_revision = _git_revision(project_root)
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
        experiment_id=spec.experiment_id,
        profile=args.profile,
        seeds=seeds,
        config=config,
        report=report,
        started_at=args.started_at,
        git_revision=git_revision,
    )
    print(json.dumps({"report": str(report_path), "receipt": str(receipt_path)}))
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    """Verify a previously written evidence receipt."""
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
    """Validate a persisted socket replay against supplied belief arrays."""
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


__all__ = ["_benchmark_command", "_replay_command", "_run_command", "_verify_command"]
