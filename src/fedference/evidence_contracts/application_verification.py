"""Application artifact and semantic binding verification."""

from __future__ import annotations

from pathlib import Path

from .application_schema import ApplicationReceipt
from .hashing import sha256_file
from .schema import ApplicationVerificationScope
from .source_verification import _application_source_findings


def verify_application_receipt(
    receipt: ApplicationReceipt,
    *,
    root: str | Path,
    require_clean_git: bool = False,
    project_root: str | Path | None = None,
    require_nominal_solver: bool = False,
    scope: ApplicationVerificationScope = "all",
) -> tuple[str, ...]:
    """Verify artifact/envelope bindings, source, and the requested solver policy.

    Verification intentionally does not rerun aggregation, so a receipt remains
    an artifact-integrity check across compatible software versions. The result
    bytes and solver diagnostics are hash-bound, while normalized caller rows
    are recomputed because their semantics are part of the stable input schema.
    """
    from .._validation import as_pmf_matrix
    from ..application import LabeledAggregationRequest, LabeledAggregationResult

    if not isinstance(receipt, ApplicationReceipt):
        raise ValueError("receipt must be an ApplicationReceipt")
    if scope not in ("all", "artifacts", "source", "solver"):
        raise ValueError("scope must be 'all', 'artifacts', 'source', or 'solver'")
    if scope == "source":
        return _application_source_findings(
            receipt,
            require_clean_git=require_clean_git,
            project_root=project_root,
        )
    if scope == "solver":
        if require_nominal_solver and receipt.solver_status != "nominal":
            return (f"solver status is {receipt.solver_status!r}, not 'nominal'",)
        return ()
    findings: list[str] = []
    if receipt.status != "completed":  # defensive against non-dataclass decoders
        findings.append(f"application status is {receipt.status!r}, not 'completed'")
    if (
        scope == "all"
        and require_nominal_solver
        and receipt.solver_status != "nominal"
    ):
        findings.append(
            f"solver status is {receipt.solver_status!r}, not 'nominal'"
        )
    root_path = Path(root).resolve()
    artifact_paths: dict[str, Path] = {}
    for artifact in receipt.outputs:
        path = (root_path / artifact.path).resolve()
        try:
            path.relative_to(root_path)
        except ValueError:
            findings.append(f"artifact escapes receipt root: {artifact.path}")
            continue
        artifact_paths[artifact.name] = path
        if not path.is_file():
            findings.append(f"missing artifact: {artifact.path}")
            continue
        if path.stat().st_size != artifact.bytes:
            findings.append(f"artifact byte-size mismatch: {artifact.path}")
        if sha256_file(path) != artifact.sha256:
            findings.append(f"artifact digest mismatch: {artifact.path}")

    request: LabeledAggregationRequest | None = None
    result: LabeledAggregationResult | None = None
    request_path = artifact_paths.get("request")
    if request_path is not None and request_path.is_file():
        try:
            request = LabeledAggregationRequest.from_json(
                request_path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError) as exc:
            findings.append(f"request artifact is invalid: {exc}")
    result_path = artifact_paths.get("result")
    if result_path is not None and result_path.is_file():
        try:
            result = LabeledAggregationResult.from_json(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError) as exc:
            findings.append(f"result artifact is invalid: {exc}")
    if request is not None:
        if request.request_sha256 != receipt.request_sha256:
            findings.append("request semantic hash does not match receipt")
        if request.config.fingerprint != receipt.config_fingerprint:
            findings.append("request configuration fingerprint does not match receipt")
    if result is not None:
        if (
            result.software_version
            != receipt.runtime_provenance.distribution_version
        ):
            findings.append(
                "result software version does not match runtime provenance"
            )
        if result.request_sha256 != receipt.request_sha256:
            findings.append("result request hash does not match receipt")
        if result.config.fingerprint != receipt.config_fingerprint:
            findings.append("result configuration fingerprint does not match receipt")
        if result.aggregation.solver_status != receipt.solver_status:
            findings.append("result solver status does not match receipt")
        if result.aggregation.fallback_events != receipt.fallback_events:
            findings.append("result fallback events do not match receipt")
    if request is not None and result is not None:
        if request.state_labels != result.state_labels:
            findings.append("request and result state-label order do not match")
        if tuple(agent.agent_id for agent in request.agents) != result.agent_ids:
            findings.append("request and result agent order do not match")
        expected_rows = as_pmf_matrix(
            (agent.posterior for agent in request.agents),
            name="agents.posterior",
        )
        if expected_rows.tolist() != result.normalized_local_posteriors.tolist():
            findings.append(
                "result normalized local posteriors do not match the request"
            )

    if scope == "all":
        findings.extend(
            _application_source_findings(
                receipt,
                require_clean_git=require_clean_git,
                project_root=project_root,
            )
        )
    return tuple(dict.fromkeys(findings))

