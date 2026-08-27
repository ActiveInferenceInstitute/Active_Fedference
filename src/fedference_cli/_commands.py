"""Command handlers for the evidence-bound Active Fedference CLI."""

from __future__ import annotations

import argparse
import json
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from fedference._validation import as_pmf_matrix
from fedference.application import aggregate_labeled, load_labeled_aggregation_request
from fedference.benchmark import run_external_benchmark_pack
from fedference.evidence import (
    ApplicationReceipt,
    load_receipt,
    verify_application_receipt,
    verify_run_receipt,
)
from fedference.experiments import run_heuristic_characterization
from fedference.federation import inspect_socket_replay, load_socket_replay
from fedference.hierarchy_tasks import run_hierarchy_task_pilot
from fedference.hybrid_tracking import run_hybrid_tracking_comparison
from fedference.protocol_parity import run_friston_protocol_audit
from fedference.provenance import (
    active_fedference_checkout_version,
    collect_source_provenance,
    runtime_provenance,
)
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
    _write_application_run,
    _write_evidence_run,
)


def _load_numeric_json_array(path: str, *, name: str, ndim: int) -> np.ndarray:
    """Load one finite numeric JSON array with a process-facing error message."""

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant is not allowed: {value}")

    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except ValueError as exc:
        raise ValueError(f"{name} must contain valid finite JSON: {path}") from exc

    def contains_invalid_scalar(value: object) -> bool:
        if isinstance(value, list):
            return any(contains_invalid_scalar(item) for item in value)
        return isinstance(value, bool) or not isinstance(value, Real)

    if isinstance(raw, list) and contains_invalid_scalar(raw):
        raise ValueError(
            f"{name} must contain only numeric non-boolean JSON values"
        )
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain a numeric JSON array") from exc
    dimensionality = {1: "one-dimensional", 2: "two-dimensional"}.get(
        ndim,
        f"{ndim}-dimensional",
    )
    if array.ndim != ndim or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain a non-empty finite {dimensionality} numeric JSON array"
        )
    return array


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


def _aggregate_command(args: argparse.Namespace) -> int:
    """Aggregate one strict labeled request and retain a self-verifying receipt."""
    request = load_labeled_aggregation_request(args.input)
    project_root = (
        _resolve_project_root(args.project_root)
        if args.project_root is not None
        else (_resolve_project_root(".") if args.require_clean_git else None)
    )
    checkout_version: str | None = None
    if project_root is not None:
        checkout_version = active_fedference_checkout_version(project_root)
        if checkout_version is None:
            raise ValueError(
                "--project-root must select an Active Fedference checkout "
                "with matching project metadata and source layout"
            )
        runtime_version = runtime_provenance().distribution_version
        if checkout_version != runtime_version:
            raise ValueError(
                "selected checkout version does not match the running distribution: "
                f"{checkout_version!r} != {runtime_version!r}"
            )
    source_provenance = collect_source_provenance(project_root)
    if args.project_root is not None and source_provenance.source_kind != "git_checkout":
        raise ValueError("--project-root must select a readable Git checkout")
    if args.require_clean_git:
        if source_provenance.source_kind != "git_checkout":
            raise ValueError("--require-clean-git requires an explicit readable Git checkout")
        if source_provenance.git_tree_state != "clean":
            raise ValueError(
                "--require-clean-git requires a clean Git tree; recorded state is "
                f"{source_provenance.git_tree_state!r}"
            )
    safety_root = project_root or _resolve_project_root(".")
    _validate_output_dir(args.output_dir, safety_root)
    result = aggregate_labeled(request)
    if checkout_version is not None and result.software_version != checkout_version:
        raise ValueError(
            "selected checkout version does not match the result software version: "
            f"{checkout_version!r} != {result.software_version!r}"
        )
    output_dir = _prepare_output_dir(args.output_dir, safety_root)
    request_path, result_path, receipt_path = _write_application_run(
        output_dir=output_dir,
        request=request,
        result=result,
        source_provenance=source_provenance,
        started_at=args.started_at,
    )
    print(
        json.dumps(
            {
                "receipt": str(receipt_path.resolve()),
                "request": str(request_path.resolve()),
                "result": str(result_path.resolve()),
                "solver_status": result.aggregation.solver_status,
            },
            sort_keys=True,
        )
    )
    return 0 if result.aggregation.solver_status == "nominal" else 1


def _verify_command(args: argparse.Namespace) -> int:
    """Auto-detect and verify a research or application receipt."""
    receipt_path = Path(args.receipt).resolve()
    receipt = load_receipt(receipt_path)
    root = Path(args.root).resolve() if args.root else receipt_path.parent
    project_root = (
        _resolve_project_root(args.project_root)
        if args.project_root is not None
        else (_resolve_project_root(".") if args.require_clean_git else None)
    )
    if (
        project_root is not None
        and active_fedference_checkout_version(project_root) is None
    ):
        raise ValueError(
            "--project-root must select an Active Fedference checkout "
            "with matching project metadata and source layout"
        )
    if isinstance(receipt, ApplicationReceipt):
        artifact_findings = verify_application_receipt(
            receipt,
            root=root,
            scope="artifacts",
        )
        source_findings = verify_application_receipt(
            receipt,
            root=root,
            require_clean_git=args.require_clean_git,
            project_root=project_root,
            scope="source",
        )
        solver_findings = verify_application_receipt(
            receipt,
            root=root,
            require_nominal_solver=args.require_nominal_solver,
            scope="solver",
        )
        findings = artifact_findings + source_findings + solver_findings
    else:
        if args.require_nominal_solver:
            raise ValueError(
                "--require-nominal-solver applies only to application receipts"
            )
        findings = verify_run_receipt(
            receipt,
            root=root,
            require_clean_git=args.require_clean_git,
            project_root=project_root,
        )
    if isinstance(receipt, ApplicationReceipt):
        if artifact_findings:
            for finding in artifact_findings:
                print(f"FAIL: application artifact integrity: {finding}")
        else:
            print("PASS: application artifact integrity verified")
        if project_root is not None:
            if source_findings:
                for finding in source_findings:
                    print(f"FAIL: source equivalence: {finding}")
            else:
                print("PASS: source equivalence verified")
        elif receipt.source_provenance.source_kind == "unavailable":
            print("INFO: source equivalence unavailable")
        else:
            print("INFO: source equivalence not requested")
        if solver_findings:
            for finding in solver_findings:
                print(f"FAIL: solver policy: {finding}")
        else:
            print(f"PASS: solver_status={receipt.solver_status}")
        return 1 if findings else 0
    if findings:
        for finding in findings:
            print(f"FAIL: {finding}")
        return 1
    else:
        print(f"PASS: {receipt.run_id} (git_tree_state={receipt.git_tree_state})")
    return 0


def _replay_command(args: argparse.Namespace) -> int:
    """Inspect replay integrity and optionally require nominal solver health."""
    replay = load_socket_replay(args.replay)
    if any(
        event.get("event") in {"belief_received", "consensus_broadcast"}
        and type(event.get("worker_id")) is not int
        for event in replay
    ):
        raise ValueError(
            "replay worker_id values must be exact non-boolean JSON integers"
        )
    beliefs = _load_numeric_json_array(args.beliefs, name="beliefs", ndim=2)
    consensus = _load_numeric_json_array(args.consensus, name="consensus", ndim=1)
    # Belief frames retain caller-supplied positive masses; the replayed server
    # normalizes once. The reported consensus is an on-wire result and must
    # already satisfy the unit-simplex contract.
    as_pmf_matrix(beliefs, name="beliefs")
    if np.any(consensus < 0.0) or not np.isclose(
        float(consensus.sum()),
        1.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("consensus must be a finite non-negative unit probability vector")
    config_assertions = {
        name: value
        for name, value in (
            ("method", args.method),
            ("robustness", args.robustness),
            ("entropy_weight", args.entropy_weight),
            ("max_iter", args.max_iter),
            ("tol", args.tol),
            ("multistart", args.multistart),
        )
        if value is not None
    }
    verdict = inspect_socket_replay(
        replay,
        beliefs,
        consensus,
        config_assertions=config_assertions,
    )
    solver_failure = args.require_nominal_solver and not verdict.nominal_solver
    if args.as_json:
        print(json.dumps(verdict.as_dict(), sort_keys=True))
    elif verdict.integrity_valid:
        print(
            "PASS: replay integrity verified "
            f"(solver_status={verdict.solver_status})"
        )
        for finding in verdict.findings:
            prefix = "FAIL" if solver_failure and finding.category == "solver" else "INFO"
            print(f"{prefix}: {finding.code}: {finding.message}")
        if solver_failure and not any(
            finding.category == "solver" for finding in verdict.findings
        ):
            print("FAIL: solver.not_nominal: replay solver health is not nominal")
    else:
        for finding in verdict.findings:
            print(f"FAIL: {finding.code}: {finding.message}")
    return 1 if not verdict.integrity_valid or solver_failure else 0


__all__ = [
    "_aggregate_command",
    "_benchmark_command",
    "_replay_command",
    "_run_command",
    "_verify_command",
]
