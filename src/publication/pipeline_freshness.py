"""Content-bound freshness receipts for the publication pipeline.

The release manifest proves that a bundle is internally byte-consistent and
that its declared source tree has not drifted since the bundle was built. This
module adds the missing upstream/downstream check: each pipeline stage records
hashes of the inputs it consumed and outputs it produced. A later validator can
therefore reject an output snapshot whose source, upstream reports, or
rendered surfaces changed without the dependent stage being rerun.

The receipts are release-integrity metadata only. They are not scientific
evidence and do not establish correctness, generalisation, or performance.
Schema 2 omits wall-clock completion time by default so recording an unchanged
stage is byte-idempotent. A canonical UTC timestamp remains an explicit input
for an approved, externally anchored build.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from publication.release_manifest import validate_utc_timestamp

PIPELINE_RECEIPT_SCHEMA_VERSION = 2
PIPELINE_RECEIPT_PATH = Path("output/data/pipeline_provenance.json")
ANALYSIS_EXECUTION_PATH = Path("output/data/analysis_execution.json")
ANALYSIS_EXECUTION_SCHEMA_VERSION = 2
PIPELINE_RECEIPT_GENERATOR = "src.publication.pipeline_freshness"

_ANALYSIS_REPORT_NAMES: tuple[str, ...] = (
    "belief_quality.json",
    "belief_sharing.json",
    "bnn_robustness.json",
    "bnn_torch.json",
    "contamination_gallery.json",
    "complexity_scaling.json",
    "conditional_world.json",
    "cross_study_summary.json",
    "disjoint_fov_world.json",
    "efe_decomposition.json",
    "emergence.json",
    "heuristic_characterization.json",
    "hierarchical_bmr.json",
    "hierarchical_world.json",
    "language_acquisition.json",
    "moving_world.json",
    "nlevel3_world.json",
    "parameter_recovery.json",
    "robust_influence_weights.json",
    "robustness_onset.json",
    "robustness_review_grid.json",
    "robustness_sweep.json",
    "variational_aggregation.json",
)


@dataclass(frozen=True)
class PipelineStageSpec:
    """Declared content boundary for one pipeline stage."""

    name: str
    input_patterns: tuple[str, ...]
    output_patterns: tuple[str, ...]
    dependencies: tuple[str, ...] = ()


_ANALYSIS_OUTPUTS = tuple(f"output/reports/{name}" for name in _ANALYSIS_REPORT_NAMES) + (
    "output/figures/**/*",
    "output/data/stage_timings.json",
    ANALYSIS_EXECUTION_PATH.as_posix(),
)
_ANALYSIS_INPUTS = (
    "src/**/*.py",
    "scripts/02_run_analysis.py",
    "manuscript/config.yaml",
    # The analysis-owned figure registry parses source captions and labels.
    # Bind every manuscript source it consumes, but never a prior hydrated tree:
    # generated output would make analysis freshness depend on a downstream
    # stage and could reintroduce stale-caption promotion.
    "manuscript/**/*.md",
    "experiment_plan.yaml",
    "pyproject.toml",
    "uv.lock",
)
_HYDRATION_INPUTS = (
    "ISA.md",
    "manuscript/**/*.md",
    "manuscript/**/*.bib",
    "manuscript/**/*.yaml",
    "manuscript/**/*.tex",
    "src/manuscript_vars/**/*.py",
    "src/experiment_config.py",
    "scripts/z_generate_manuscript_variables.py",
    *(f"output/reports/{name}" for name in _ANALYSIS_REPORT_NAMES),
    "output/data/stage_timings.json",
    # This receipt is intentionally an input rather than a pipeline-stage
    # dependency.  It is produced only after a provisional hydrate/render pass
    # has enabled the complete test suite, so making it a graph dependency
    # would create a cycle.  The final hydration CLI validates it explicitly
    # before writing and records that exact receipt hash here.
    "output/data/test_coverage_receipt.json",
)
_HYDRATION_OUTPUTS = (
    "output/data/manuscript_variables.json",
    "output/manuscript/**/*.md",
    "output/manuscript/**/*.bib",
    "output/manuscript/**/*.yaml",
    "output/manuscript/**/*.tex",
)
_RENDER_INPUTS = (
    "output/manuscript/**/*",
    "output/figures/**/*",
)
_RENDER_OUTPUTS = (
    "output/pdf/**/*",
    "output/slides/**/*",
    "output/web/**/*",
)

PIPELINE_STAGES: tuple[PipelineStageSpec, ...] = (
    PipelineStageSpec("analysis", _ANALYSIS_INPUTS, _ANALYSIS_OUTPUTS),
    PipelineStageSpec(
        "hydration",
        _HYDRATION_INPUTS,
        _HYDRATION_OUTPUTS,
        dependencies=("analysis",),
    ),
    PipelineStageSpec(
        "render",
        _RENDER_INPUTS,
        _RENDER_OUTPUTS,
        dependencies=("analysis", "hydration"),
    ),
)
_STAGES_BY_NAME = {stage.name: stage for stage in PIPELINE_STAGES}


def _receipt_path(root: Path) -> Path:
    return root / PIPELINE_RECEIPT_PATH


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_files(root: Path, patterns: Iterable[str], *, role: str) -> dict[str, Path]:
    """Resolve glob patterns to a sorted, de-duplicated relative path map."""
    resolved: dict[str, Path] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            resolved[relative] = path
    if not resolved:
        raise ValueError(f"pipeline {role} boundary resolved no files")
    return dict(sorted(resolved.items()))


def _hash_files(root: Path, patterns: Iterable[str], *, role: str) -> dict[str, str]:
    return {relative: _sha256(path) for relative, path in _resolve_files(root, patterns, role=role).items()}


def _map_digest(file_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in file_hashes.items():
        digest.update(f"{file_hash}  {relative}\n".encode("utf-8"))
    return digest.hexdigest()


def _empty_receipt() -> dict[str, Any]:
    return {"schema_version": PIPELINE_RECEIPT_SCHEMA_VERSION, "stages": {}}


def _load_receipt(
    root: Path,
    *,
    reset_unsupported: bool = False,
) -> dict[str, Any]:
    path = _receipt_path(root)
    if not path.is_file():
        return _empty_receipt()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid pipeline provenance receipt: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), dict):
        raise ValueError(f"invalid pipeline provenance shape: {path}")
    if payload.get("schema_version") != PIPELINE_RECEIPT_SCHEMA_VERSION:
        if reset_unsupported:
            return _empty_receipt()
        raise ValueError(f"unsupported pipeline provenance schema in {path}")
    return payload


def _stage(stage_name: str) -> PipelineStageSpec:
    try:
        return _STAGES_BY_NAME[stage_name]
    except KeyError as exc:
        known = ", ".join(sorted(_STAGES_BY_NAME))
        raise ValueError(f"unknown pipeline stage {stage_name!r}; expected one of {known}") from exc


def capture_analysis_input_snapshot(project_root: Path) -> dict[str, Any]:
    """Capture the exact analysis-input boundary immediately before a run.

    A publication receipt is useful only when it attests the source and
    configuration that the long-running producer actually consumed.  The
    workflow captures this snapshot before it starts writing reports, and the
    publication recorder compares it with the inputs present immediately before
    it writes the analysis receipt.
    """
    root = Path(project_root).resolve()
    stage = _stage("analysis")
    input_hashes = _hash_files(root, stage.input_patterns, role="analysis inputs")
    return {
        "input_hashes": input_hashes,
        "input_digest": _map_digest(input_hashes),
    }


def pipeline_stage_record(project_root: Path, stage_name: str) -> dict[str, Any]:
    """Return a recorded stage mapping or fail closed when it is absent.

    This narrow public reader lets independent receipts bind a validated
    upstream stage without treating them as graph dependencies.  Callers that
    need freshness as well as presence should first call
    :func:`require_fresh_pipeline_stages`.
    """
    root = Path(project_root).resolve()
    stage = _stage(stage_name)
    receipt = _load_receipt(root)
    stages = receipt.get("stages", {})
    record = stages.get(stage.name) if isinstance(stages, dict) else None
    if not isinstance(record, dict):
        raise ValueError(f"{stage.name}: missing pipeline provenance receipt")
    return dict(record)


def _normalized_analysis_input_snapshot(payload: object) -> dict[str, Any]:
    """Validate the producer-owned pre-analysis input snapshot fail closed."""
    if not isinstance(payload, Mapping):
        raise ValueError("analysis: execution sidecar has no valid input snapshot")
    raw_hashes = payload.get("input_hashes")
    if not isinstance(raw_hashes, Mapping) or not raw_hashes:
        raise ValueError("analysis: execution sidecar has no valid input snapshot")
    input_hashes: dict[str, str] = {}
    for relative, digest in raw_hashes.items():
        if not isinstance(relative, str) or not relative or not isinstance(digest, str) or not digest:
            raise ValueError("analysis: execution sidecar has no valid input snapshot")
        input_hashes[relative] = digest
    input_hashes = dict(sorted(input_hashes.items()))
    input_digest = payload.get("input_digest")
    if input_digest != _map_digest(input_hashes):
        raise ValueError("analysis: execution sidecar has an invalid input snapshot digest")
    return {
        "input_hashes": input_hashes,
        "input_digest": input_digest,
    }


def _publication_analysis_execution(root: Path) -> dict[str, Any]:
    """Read and validate the producer-owned publication-profile sidecar."""
    path = root / ANALYSIS_EXECUTION_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"analysis: missing or invalid execution sidecar: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != ANALYSIS_EXECUTION_SCHEMA_VERSION:
        raise ValueError("analysis: unsupported execution sidecar schema")
    expected = {
        "schema_version": str(ANALYSIS_EXECUTION_SCHEMA_VERSION),
        "configured_profile": "publication",
        "effective_profile": "publication",
        "producer": "analysis.workflow.run_analysis_pipeline",
    }
    observed = {key: str(payload.get(key, "")) for key in expected}
    if observed != expected:
        raise ValueError(
            "analysis: execution sidecar does not attest a configured and effective publication profile"
        )
    return {
        **observed,
        "analysis_input_snapshot": _normalized_analysis_input_snapshot(
            payload.get("analysis_input_snapshot")
        ),
    }


def _compare_hash_maps(
    recorded: Any,
    current: dict[str, str],
    *,
    label: str,
) -> list[str]:
    if not isinstance(recorded, dict):
        return [f"{label}: receipt is missing its hash map"]
    recorded_map = {str(path): str(value) for path, value in recorded.items()}
    changed = sorted(
        path for path in set(recorded_map) | set(current) if recorded_map.get(path) != current.get(path)
    )
    if not changed:
        return []
    shown = ", ".join(changed[:12])
    suffix = "" if len(changed) <= 12 else f" (+{len(changed) - 12} more)"
    return [f"{label}: changed paths: {shown}{suffix}"]


def _validate_stage_record(
    root: Path,
    stage: PipelineStageSpec,
    receipt: dict[str, Any],
    *,
    visited: set[str],
) -> list[str]:
    """Validate one stage and all of its upstream dependencies."""
    if stage.name in visited:
        return []
    visited.add(stage.name)
    findings: list[str] = []
    stages = receipt.get("stages", {})
    record = stages.get(stage.name) if isinstance(stages, dict) else None
    for dependency_name in stage.dependencies:
        findings.extend(
            _validate_stage_record(
                root,
                _stage(dependency_name),
                receipt,
                visited=visited,
            )
        )
    if not isinstance(record, dict):
        return findings + [f"{stage.name}: missing pipeline provenance receipt"]
    if record.get("stage") != stage.name:
        findings.append(f"{stage.name}: persisted stage identity drift")
    if record.get("dependencies") != list(stage.dependencies):
        findings.append(f"{stage.name}: dependency contract drift")
    if record.get("input_patterns") != list(stage.input_patterns):
        findings.append(f"{stage.name}: input pattern contract drift")
    if record.get("output_patterns") != list(stage.output_patterns):
        findings.append(f"{stage.name}: output pattern contract drift")
    recorded_at_present = "recorded_at" in record
    recorded_at = record.get("recorded_at")
    timestamp_policy = record.get("timestamp_policy")
    if not recorded_at_present:
        findings.append(f"{stage.name}: recorded_at field missing")
    elif recorded_at is None:
        if timestamp_policy != "omitted":
            findings.append(f"{stage.name}: omitted recorded_at requires timestamp_policy=omitted")
    elif isinstance(recorded_at, str):
        try:
            validate_utc_timestamp(recorded_at)
        except ValueError:
            findings.append(f"{stage.name}: recorded_at is not canonical UTC")
        if timestamp_policy != "recorded":
            findings.append(f"{stage.name}: populated recorded_at requires timestamp_policy=recorded")
    else:
        findings.append(f"{stage.name}: recorded_at must be a canonical UTC string or null")
    try:
        current_inputs = _hash_files(root, stage.input_patterns, role=f"{stage.name} inputs")
        current_outputs = _hash_files(root, stage.output_patterns, role=f"{stage.name} outputs")
    except ValueError as exc:
        return findings + [f"{stage.name}: {exc}"]
    findings.extend(
        _compare_hash_maps(record.get("input_hashes"), current_inputs, label=f"{stage.name} inputs")
    )
    findings.extend(
        _compare_hash_maps(record.get("output_hashes"), current_outputs, label=f"{stage.name} outputs")
    )
    if record.get("input_digest") != _map_digest(current_inputs):
        findings.append(f"{stage.name}: input digest mismatch")
    if record.get("output_digest") != _map_digest(current_outputs):
        findings.append(f"{stage.name}: output digest mismatch")
    return findings


def _record_pipeline_stage(
    project_root: Path,
    stage_name: str,
    *,
    renderer: str | None = None,
    timestamp: str | None = None,
    analysis_execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record one successful stage after its inputs and outputs exist.

    Dependencies are checked before a new receipt is written. The external
    template renderer is represented by the ``renderer`` label for audit
    context; its implementation is outside this standalone repository and is
    therefore not falsely presented as content-fingerprinted here. Recording
    analysis under a newer schema replaces an incompatible older receipt, so
    the downstream chain must then be regenerated in order.
    """
    root = Path(project_root).resolve()
    stage = _stage(stage_name)
    if analysis_execution is not None and stage.name != "analysis":
        raise ValueError("analysis execution metadata is valid only for the analysis stage")
    receipt = _load_receipt(
        root,
        reset_unsupported=stage.name == "analysis",
    )
    dependency_findings: list[str] = []
    for dependency_name in stage.dependencies:
        dependency_findings.extend(
            _validate_stage_record(root, _stage(dependency_name), receipt, visited=set())
        )
    if dependency_findings:
        raise ValueError(
            f"cannot record {stage.name} until upstream stages are fresh: " + "; ".join(dependency_findings)
        )
    input_hashes = _hash_files(root, stage.input_patterns, role=f"{stage.name} inputs")
    if analysis_execution is not None:
        expected_snapshot = _normalized_analysis_input_snapshot(
            analysis_execution.get("analysis_input_snapshot")
        )
        if input_hashes != expected_snapshot["input_hashes"]:
            raise ValueError(
                "analysis: inputs changed while the producer ran; "
                "refusing to attest post-run inputs as publication analysis"
            )
    output_hashes = _hash_files(root, stage.output_patterns, role=f"{stage.name} outputs")
    stamp = validate_utc_timestamp(timestamp)
    record: dict[str, Any] = {
        "stage": stage.name,
        "dependencies": list(stage.dependencies),
        "input_patterns": list(stage.input_patterns),
        "output_patterns": list(stage.output_patterns),
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "input_digest": _map_digest(input_hashes),
        "output_digest": _map_digest(output_hashes),
        "recorded_at": stamp,
        "timestamp_policy": "recorded" if stamp is not None else "omitted",
    }
    if stage.name == "render":
        record["renderer"] = renderer or "external-template-renderer (not content-fingerprinted)"
    if analysis_execution is not None:
        record["analysis_execution"] = dict(analysis_execution)
    receipt["schema_version"] = PIPELINE_RECEIPT_SCHEMA_VERSION
    receipt["generated_by"] = PIPELINE_RECEIPT_GENERATOR
    receipt["stages"][stage.name] = record
    output_path = _receipt_path(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return record


def record_pipeline_stage(
    project_root: Path,
    stage_name: str,
    *,
    renderer: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Record a generic stage receipt for internal fixtures or external rendering.

    The analysis and hydration producers own their receipt metadata. In
    particular, this generic helper deliberately cannot attest a publication
    analysis profile; release-facing callers must use
    :func:`record_publication_analysis_stage` after the real analysis producer
    has written its execution sidecar.
    """
    return _record_pipeline_stage(
        project_root,
        stage_name,
        renderer=renderer,
        timestamp=timestamp,
    )


def record_publication_analysis_stage(
    project_root: Path,
    *,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Record the analysis stage only after a publication-profile producer run."""
    root = Path(project_root).resolve()
    execution = _publication_analysis_execution(root)
    return _record_pipeline_stage(
        root,
        "analysis",
        timestamp=timestamp,
        analysis_execution=execution,
    )


def validate_pipeline_freshness(
    project_root: Path,
    stages: Iterable[str] | None = None,
) -> list[str]:
    """Return fail-closed freshness findings for the requested stage closure."""
    root = Path(project_root).resolve()
    requested = tuple(stages or tuple(stage.name for stage in PIPELINE_STAGES))
    receipt = _load_receipt(root)
    findings: list[str] = []
    if receipt.get("stages") and receipt.get("generated_by") != PIPELINE_RECEIPT_GENERATOR:
        findings.append("pipeline receipt: generated_by identity drift")
    visited: set[str] = set()
    for stage_name in requested:
        findings.extend(_validate_stage_record(root, _stage(stage_name), receipt, visited=visited))
    return findings


def require_fresh_pipeline_stages(
    project_root: Path,
    stages: Iterable[str],
) -> None:
    """Raise when any requested content-bound pipeline stage is stale.

    Orchestrators call this *before* mutating a downstream output tree.  The
    helper keeps that preflight policy separate from stage recording, whose
    checks otherwise occur only after a producer has completed its writes.
    """
    requested = tuple(stages)
    if not requested:
        raise ValueError("at least one pipeline stage is required for preflight")
    findings = validate_pipeline_freshness(project_root, stages=requested)
    if findings:
        raise ValueError("pipeline freshness preflight failed: " + "; ".join(findings))


def require_fresh_publication_analysis(project_root: Path) -> dict[str, Any]:
    """Require a fresh analysis receipt bound to a publication-profile run.

    Content hashes alone cannot distinguish a bounded smoke execution from a
    publication execution. The analysis producer therefore writes a sidecar,
    and the receipt must embed the same producer-owned metadata before a
    non-draft hydration or release-facing operation may proceed.
    """
    root = Path(project_root).resolve()
    require_fresh_pipeline_stages(root, ("analysis",))
    execution = _publication_analysis_execution(root)
    record = pipeline_stage_record(root, "analysis")
    if record.get("analysis_execution") != execution:
        raise ValueError(
            "analysis: receipt is not bound to the producer-owned publication execution metadata"
        )
    return record


def validate_publication_pipeline_freshness(
    project_root: Path,
    stages: Iterable[str] | None = None,
) -> list[str]:
    """Validate stage hashes plus the publication-profile analysis boundary."""
    findings = validate_pipeline_freshness(project_root, stages=stages)
    if findings:
        return findings
    try:
        require_fresh_publication_analysis(project_root)
    except ValueError as exc:
        return [str(exc)]
    return []


__all__ = [
    "ANALYSIS_EXECUTION_SCHEMA_VERSION",
    "ANALYSIS_EXECUTION_PATH",
    "PIPELINE_RECEIPT_PATH",
    "PIPELINE_RECEIPT_SCHEMA_VERSION",
    "PIPELINE_RECEIPT_GENERATOR",
    "PIPELINE_STAGES",
    "PipelineStageSpec",
    "capture_analysis_input_snapshot",
    "pipeline_stage_record",
    "record_publication_analysis_stage",
    "record_pipeline_stage",
    "require_fresh_publication_analysis",
    "require_fresh_pipeline_stages",
    "validate_publication_pipeline_freshness",
    "validate_pipeline_freshness",
]
