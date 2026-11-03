"""Registry report contracts."""

from __future__ import annotations

from analysis.schemas.aggregation_diagnostics import (
    _HEURISTIC_CHARACTERIZATION_SCHEMA,
    _ROBUST_INFLUENCE_WEIGHTS_SCHEMA,
    _VARIATIONAL_AGGREGATION_SCHEMA,
)
from analysis.schemas.application_integrity_flow import _APPLICATION_INTEGRITY_FLOW_SCHEMA
from analysis.schemas.belief_sharing import _BELIEF_SHARING_SCHEMA
from analysis.schemas.bnn_robustness import _BNN_ROBUSTNESS_SCHEMA
from analysis.schemas.contamination import _CONTAMINATION_GALLERY_SCHEMA, _ROBUSTNESS_ONSET_SCHEMA
from analysis.schemas.evidence_replication_map import _EVIDENCE_REPLICATION_MAP_SCHEMA
from analysis.schemas.hierarchical_bmr import _HIERARCHICAL_BMR_SCHEMA
from analysis.schemas.learning import (
    _EFE_DECOMPOSITION_SCHEMA,
    _EMERGENCE_SCHEMA,
    _LANGUAGE_ACQUISITION_SCHEMA,
)
from analysis.schemas.observer_worlds import (
    _COMPLEXITY_SCALING_SCHEMA,
    _CONDITIONAL_WORLD_SCHEMA,
    _DISJOINT_FOV_WORLD_SCHEMA,
)
from analysis.schemas.parameter_recovery import _PARAMETER_RECOVERY_SCHEMA
from analysis.schemas.primitives import SchemaDefinition
from analysis.schemas.review_grid import _ROBUSTNESS_REVIEW_GRID_SCHEMA
from analysis.schemas.robustness_sweep import _ROBUSTNESS_SWEEP_SCHEMA
from analysis.schemas.sensitivity import _SENSITIVITY_SCHEMA
from analysis.schemas.source_render_provenance import _SOURCE_RENDER_PROVENANCE_SCHEMA
from analysis.schemas.study_summaries import _BELIEF_QUALITY_SCHEMA, _CROSS_STUDY_SUMMARY_SCHEMA
from analysis.schemas.worlds import _HIERARCHICAL_WORLD_SCHEMA, _MOVING_WORLD_SCHEMA, _NLEVEL3_WORLD_SCHEMA

_REPORT_SCHEMAS: dict[str, SchemaDefinition] = {
    "belief_sharing": _BELIEF_SHARING_SCHEMA,
    "language_acquisition": _LANGUAGE_ACQUISITION_SCHEMA,
    "emergence": _EMERGENCE_SCHEMA,
    "robustness_sweep": _ROBUSTNESS_SWEEP_SCHEMA,
    "hierarchical_bmr": _HIERARCHICAL_BMR_SCHEMA,
    "heuristic_characterization": _HEURISTIC_CHARACTERIZATION_SCHEMA,
    "efe_decomposition": _EFE_DECOMPOSITION_SCHEMA,
    "robust_influence_weights": _ROBUST_INFLUENCE_WEIGHTS_SCHEMA,
    "bnn_robustness": _BNN_ROBUSTNESS_SCHEMA,
    "variational_aggregation": _VARIATIONAL_AGGREGATION_SCHEMA,
    "contamination_gallery": _CONTAMINATION_GALLERY_SCHEMA,
    "robustness_onset": _ROBUSTNESS_ONSET_SCHEMA,
    "moving_world": _MOVING_WORLD_SCHEMA,
    "parameter_recovery": _PARAMETER_RECOVERY_SCHEMA,
    "hierarchical_world": _HIERARCHICAL_WORLD_SCHEMA,
    "nlevel3_world": _NLEVEL3_WORLD_SCHEMA,
    "cross_study_summary": _CROSS_STUDY_SUMMARY_SCHEMA,
    "disjoint_fov_world": _DISJOINT_FOV_WORLD_SCHEMA,
    "complexity_scaling": _COMPLEXITY_SCALING_SCHEMA,
    "conditional_world": _CONDITIONAL_WORLD_SCHEMA,
    "robustness_review_grid": _ROBUSTNESS_REVIEW_GRID_SCHEMA,
    "belief_quality": _BELIEF_QUALITY_SCHEMA,
    "application_integrity_flow": _APPLICATION_INTEGRITY_FLOW_SCHEMA,
    "evidence_replication_map": _EVIDENCE_REPLICATION_MAP_SCHEMA,
    "source_render_provenance": _SOURCE_RENDER_PROVENANCE_SCHEMA,
    "sensitivity": _SENSITIVITY_SCHEMA,
}


_SUPPORTED_REPORT_SCHEMA_VERSIONS: dict[str, frozenset[str]] = {
    "application_integrity_flow": frozenset({"1.0"}),
    "belief_sharing": frozenset({"1.0"}),
    "bnn_robustness": frozenset({"1.0"}),
    "evidence_replication_map": frozenset({"1.0"}),
    "heuristic_characterization": frozenset({"2.0"}),
    "hierarchical_bmr": frozenset({"1.0"}),
    "parameter_recovery": frozenset({"1.0"}),
    "robustness_sweep": frozenset({"2.0"}),
    "robust_influence_weights": frozenset({"2.0"}),
    "robustness_review_grid": frozenset({"1.1"}),
    "source_render_provenance": frozenset({"1.0"}),
    "sensitivity": frozenset({"1.0"}),
}
