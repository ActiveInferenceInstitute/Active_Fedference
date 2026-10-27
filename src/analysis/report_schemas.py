"""Canonical report validation and figure dependency boundary."""

from __future__ import annotations

from collections.abc import Mapping

from analysis.schemas.aggregation_diagnostics import (
    HeuristicCharacterizationReport as HeuristicCharacterizationReport,
)
from analysis.schemas.aggregation_diagnostics import (
    RobustInfluenceWeightsReport as RobustInfluenceWeightsReport,
)
from analysis.schemas.aggregation_diagnostics import (
    VariationalAggregationReport as VariationalAggregationReport,
)
from analysis.schemas.application_integrity_flow import (
    ApplicationIntegrityFlowReport as ApplicationIntegrityFlowReport,
)
from analysis.schemas.application_integrity_flow import _check_application_integrity_flow
from analysis.schemas.belief_sharing import BeliefSharingReport as BeliefSharingReport
from analysis.schemas.belief_sharing import _check_belief_sharing
from analysis.schemas.bnn_robustness import BnnRobustnessReport as BnnRobustnessReport
from analysis.schemas.bnn_torch import BnnTorchOkReport as BnnTorchOkReport
from analysis.schemas.bnn_torch import BnnTorchSkippedReport as BnnTorchSkippedReport
from analysis.schemas.bnn_torch import _check_bnn_torch
from analysis.schemas.bnn_validation import _check_bnn_robustness
from analysis.schemas.contamination import ContaminationGalleryCell as ContaminationGalleryCell
from analysis.schemas.contamination import ContaminationGalleryReport as ContaminationGalleryReport
from analysis.schemas.contamination import RobustnessOnsetCell as RobustnessOnsetCell
from analysis.schemas.contamination import RobustnessOnsetReport as RobustnessOnsetReport
from analysis.schemas.evidence_replication_map import (
    EvidenceReplicationMapReport as EvidenceReplicationMapReport,
)
from analysis.schemas.evidence_replication_map import _check_evidence_replication_map
from analysis.schemas.figure_dependencies import FIGURE_DEPENDENCY_CONTRACTS as FIGURE_DEPENDENCY_CONTRACTS
from analysis.schemas.figure_exact_values import _check_figure_exact_values
from analysis.schemas.figure_registry import FigureMetadataEntry as FigureMetadataEntry
from analysis.schemas.figure_registry import FigureRegistryPayload as FigureRegistryPayload
from analysis.schemas.figure_registry import _check_figure_registry
from analysis.schemas.hierarchical_bmr import HierarchicalBmrReport as HierarchicalBmrReport
from analysis.schemas.hierarchical_bmr import _check_hierarchical_bmr
from analysis.schemas.learning import EfeDecompositionReport as EfeDecompositionReport
from analysis.schemas.learning import EmergenceReport as EmergenceReport
from analysis.schemas.learning import LanguageAcquisitionReport as LanguageAcquisitionReport
from analysis.schemas.observer_worlds import ComplexityScalingReport as ComplexityScalingReport
from analysis.schemas.observer_worlds import ConditionalWorldReport as ConditionalWorldReport
from analysis.schemas.observer_worlds import DisjointFovWorldReport as DisjointFovWorldReport
from analysis.schemas.parameter_recovery import ParameterRecoveryReport as ParameterRecoveryReport
from analysis.schemas.parameter_recovery import _check_parameter_recovery
from analysis.schemas.presentation import _check_presentation_manifest
from analysis.schemas.primitives import FigureDependencyContract as FigureDependencyContract
from analysis.schemas.primitives import ReportSchemaError as ReportSchemaError
from analysis.schemas.primitives import SchemaDefinition as SchemaDefinition
from analysis.schemas.primitives import _check, _check_finite_json
from analysis.schemas.registry import _REPORT_SCHEMAS as _REPORT_SCHEMAS
from analysis.schemas.registry import _SUPPORTED_REPORT_SCHEMA_VERSIONS as _SUPPORTED_REPORT_SCHEMA_VERSIONS
from analysis.schemas.review_grid import ReviewGridReport as ReviewGridReport
from analysis.schemas.review_validation import _check_review_grid
from analysis.schemas.robustness_sweep import RobustnessSweepReport as RobustnessSweepReport
from analysis.schemas.robustness_sweep import _check_robustness_sweep
from analysis.schemas.sensitivity import SensitivityReport as SensitivityReport
from analysis.schemas.sensitivity import _check_sensitivity
from analysis.schemas.source_render_provenance import (
    SourceRenderProvenanceReport as SourceRenderProvenanceReport,
)
from analysis.schemas.source_render_provenance import _check_source_render_provenance
from analysis.schemas.study_summaries import BeliefQualityReport as BeliefQualityReport
from analysis.schemas.study_summaries import CrossStudySummaryReport as CrossStudySummaryReport
from analysis.schemas.worlds import HierarchicalWorldReport as HierarchicalWorldReport
from analysis.schemas.worlds import MovingWorldReport as MovingWorldReport
from analysis.schemas.worlds import NLevel3WorldReport as NLevel3WorldReport
from analysis.visual_contracts import (
    SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY as SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY,
)
from analysis.visual_contracts import (
    SOURCE_RENDER_PRODUCER_EDGE_INVENTORY as SOURCE_RENDER_PRODUCER_EDGE_INVENTORY,
)


def validate_report(schema: str, payload: Mapping[str, object]) -> None:
    """Validate one report or figure-registry payload before it is written."""

    if not isinstance(payload, Mapping):
        raise ReportSchemaError(f"{schema} payload must be a mapping")
    _check_finite_json(payload)
    if schema == "presentation_manifest":
        _check_presentation_manifest(payload)
        return
    if schema == "figure_registry":
        _check_figure_registry(payload)
        return
    if schema == "figure_exact_values":
        _check_figure_exact_values(payload)
        return
    if schema == "bnn_torch":
        _check_bnn_torch(payload)
        return
    if schema not in _REPORT_SCHEMAS:
        raise ReportSchemaError(f"Unknown report schema: {schema}")
    _check(schema, payload, _REPORT_SCHEMAS[schema])
    supported_versions = _SUPPORTED_REPORT_SCHEMA_VERSIONS.get(schema)
    if supported_versions is not None:
        version = payload["schema_version"]
        assert isinstance(version, str)
        if version not in supported_versions:
            raise ReportSchemaError(
                f"{schema} payload has unsupported schema_version {version!r}; "
                f"supported versions are {sorted(supported_versions)}"
            )
    if schema == "application_integrity_flow":
        _check_application_integrity_flow(payload)
    elif schema == "belief_sharing":
        _check_belief_sharing(payload)
    elif schema == "bnn_robustness":
        _check_bnn_robustness(payload)
    elif schema == "hierarchical_bmr":
        _check_hierarchical_bmr(payload)
    elif schema == "parameter_recovery":
        _check_parameter_recovery(payload)
    elif schema == "robustness_sweep":
        _check_robustness_sweep(payload)
    elif schema == "evidence_replication_map":
        _check_evidence_replication_map(payload)
    elif schema == "source_render_provenance":
        _check_source_render_provenance(payload)
    elif schema == "sensitivity":
        _check_sensitivity(payload)
    if schema == "robustness_review_grid":
        _check_review_grid(payload)


def check_figure_contract(
    generator: str,
    report_name: str,
    report: Mapping[str, object],
) -> None:
    """Validate the declared report fields consumed by one figure generator."""

    if not isinstance(report, Mapping):
        raise ReportSchemaError(f"figure {generator!r} report {report_name!r} must be a mapping")
    _check_finite_json(report)
    if generator not in FIGURE_DEPENDENCY_CONTRACTS:
        raise ReportSchemaError(f"Unknown figure contract for generator {generator!r}")
    for contract in FIGURE_DEPENDENCY_CONTRACTS[generator]:
        if contract.report_name == report_name:
            _check(
                f"figure {generator!r} using report {report_name!r}",
                report,
                SchemaDefinition(
                    required=contract.required_fields,
                    optional=contract.optional_fields,
                ),
            )
            return
    raise ReportSchemaError(f"Figure {generator!r} does not declare a dependency on report {report_name!r}")


__all__ = [
    "ApplicationIntegrityFlowReport",
    "BnnRobustnessReport",
    "BnnTorchOkReport",
    "BnnTorchSkippedReport",
    "ContaminationGalleryCell",
    "ContaminationGalleryReport",
    "CrossStudySummaryReport",
    "DisjointFovWorldReport",
    "EfeDecompositionReport",
    "EmergenceReport",
    "EvidenceReplicationMapReport",
    "FIGURE_DEPENDENCY_CONTRACTS",
    "FigureDependencyContract",
    "FigureMetadataEntry",
    "FigureRegistryPayload",
    "HeuristicCharacterizationReport",
    "HierarchicalBmrReport",
    "HierarchicalWorldReport",
    "LanguageAcquisitionReport",
    "MovingWorldReport",
    "NLevel3WorldReport",
    "ParameterRecoveryReport",
    "ReportSchemaError",
    "RobustInfluenceWeightsReport",
    "RobustnessOnsetCell",
    "RobustnessOnsetReport",
    "RobustnessSweepReport",
    "SchemaDefinition",
    "SensitivityReport",
    "SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY",
    "SOURCE_RENDER_PRODUCER_EDGE_INVENTORY",
    "SourceRenderProvenanceReport",
    "VariationalAggregationReport",
    "BeliefSharingReport",
    "check_figure_contract",
    "validate_report",
]
