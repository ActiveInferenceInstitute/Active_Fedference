"""Typed report and figure-registry schemas for the analysis write boundary.

The analysis pipeline writes JSON payloads that are later consumed by figures,
publication tooling, and release checks. This module documents those top-level
shapes with :class:`typing.TypedDict` and enforces them with a small shallow
runtime validator at the single write boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, NoReturn, TypedDict

from typing_extensions import NotRequired, Required

from analysis.figure_exact_values import EXACT_VALUE_GENERATORS, EXACT_VALUE_REPORT_BINDINGS
from analysis.visual_contracts import (
    APPLICATION_ARTIFACT_WRITE_ORDER,
    APPLICATION_FLOW_EDGE_INVENTORY,
    APPLICATION_PANEL_NODE_INVENTORY,
    COMPLEX_FIGURE_GENERATORS,
    SOURCE_RENDER_INPUT_INVENTORY,
    SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY,
    SOURCE_RENDER_PRODUCER_EDGE_INVENTORY,
    SOURCE_RENDER_STAGE_INVENTORY,
)

TypeTag = Literal["bool", "dict", "int", "list", "number", "str"]


class BeliefSharingReport(TypedDict):
    """Top-level payload written to ``belief_sharing.json``."""

    schema_version: str
    analysis_unit: str
    communicating_free_energy: list[float]
    communicating_mean: float
    communication_helps: bool
    difference_definition: str
    free_energy_gap: float
    incommunicado_free_energy: list[float]
    incommunicado_mean: float
    interval_method: str
    interval_percent: int
    n_agents: int
    n_seeds: int
    paired_difference_ci: list[float]
    paired_difference_mean: float
    paired_free_energy_difference: list[float]
    replication_unit: str


class LanguageAcquisitionReport(TypedDict):
    """Top-level payload written to ``language_acquisition.json``."""

    final_kl: float
    initial_kl: float
    kl_trajectory: list[float]
    kl_trajectory_by_seed: list[object]
    monotone_decreasing: bool
    n_points: int
    n_seeds: int
    num_steps: int
    seed: int
    trajectory_ci_hi: list[float]
    trajectory_ci_lo: list[float]


class EmergenceReport(TypedDict):
    """Top-level payload written to ``emergence.json``."""

    convergence: bool
    delta_F_redundant: float
    delta_F_supported: float
    n: int
    n_states: int
    seed: int


class RobustnessSweepReport(TypedDict):
    """Top-level payload written to ``robustness_sweep.json``."""

    schema_version: str
    accuracy_at_verdict_rate: dict[str, object]
    accuracy_by_method_and_rate: dict[str, object]
    accuracy_threshold: float
    any_robust_wins: bool
    attack_target_state: int
    divergences: list[str]
    fdr_alpha: float
    headline_method: str
    headline_selection_rule: str
    headline_tie_break: str
    headline_tie_set: list[str]
    headline_is_display_selection: bool
    largest_mean_difference_method: str
    headline_n_for_target_power: int
    headline_power: float
    kind: str
    n: int
    n_agents: int
    n_contaminated: int
    n_trials: int
    naive_degrades_with_rate: bool
    naive_verdict_rate_mean: float
    paired_tests_by_rate: dict[str, object]
    per_rate_summary: dict[str, object]
    power_alpha: float
    power_alternative: str
    prospective_n_for_target_power: int
    rates: list[float]
    robust_above_threshold_at_worst_rate: bool
    seed: int
    server_robustness_by_label: dict[str, object]
    target_power: float
    true_state: int
    verdict: dict[str, object]
    verdict_rate: float
    trial_structure: str
    analysis_unit: str
    worst_rate_best_method: str
    worst_rate: float
    paired_test_alternative: str
    fdr_family_ownership: str
    d_equivalent_status: str


class HierarchicalBmrReport(TypedDict):
    """Top-level payload written to ``hierarchical_bmr.json``."""

    acuity: float
    claim_boundary: str
    configured_control_passed: bool
    decision_rule: str
    degenerate: dict[str, object]
    degenerate_recommends_prune_top: bool
    degenerate_top_surprise: float
    informative: dict[str, object]
    informative_keeps_top: bool
    informative_top_surprise: float
    n_iters: int
    n_levels: int
    obs: int
    primary_control: str
    schema_version: str
    secondary_diagnostic: str
    seed: int
    seed_role: str
    study_status: str
    surprise_tol: float


class HeuristicCharacterizationReport(TypedDict):
    """Top-level payload written to ``heuristic_characterization.json``."""

    breakdown: dict[str, object]
    claim_level: str
    formal_no_go: dict[str, object]
    grid: dict[str, object]
    independent_unit: str
    influence_naive: dict[str, object]
    influence_robust: dict[str, object]
    no_claim: str
    schema_version: str
    seed: int
    theory_status: str


class EfeDecompositionReport(TypedDict):
    """Top-level payload written to ``efe_decomposition.json``."""

    ambiguity: float
    epistemic_value: float
    identity_residual: float
    policy: list[int]
    pragmatic_value: float
    prior_type: str
    risk: float
    total: float


class RobustInfluenceWeightsReport(TypedDict):
    """Top-level payload written to ``robust_influence_weights.json``."""

    normalized_effective_weights: list[float]
    agent_weights: NotRequired[list[float]]
    contaminated_indices: list[int]
    n_agents: int
    n_contaminated: int
    robustness: float
    schema_version: str
    true_state: int


class BnnRobustnessReport(TypedDict):
    """Top-level payload written to ``bnn_robustness.json``."""

    accuracy_by_config: dict[str, object]
    accuracy_ci_by_config: dict[str, object]
    accuracy_seed_values_by_config: dict[str, object]
    analysis_unit: str
    ci_percent: int
    claim_boundary: str
    configuration_boundary: str
    contamination_levels: list[float]
    interval_method: str
    model_family: str
    n_bootstrap: int
    n_per: int
    n_seeds: int
    peak_margin: NotRequired[float]
    peak_margin_contamination: NotRequired[float]
    replication_unit: str
    robust_loss_param: float
    robust_minus_standard: list[float]
    schema_version: str
    seed: int
    selection_disclosure: str
    study_status: str


class BnnTorchOkReport(TypedDict):
    """Executed PyTorch complement payload written to ``bnn_torch.json``."""

    accuracy_by_config: dict[str, object]
    beta: float
    consensus_max_simplex_deviation: float
    contamination_levels: list[float]
    deterministic: bool
    hidden_dim: int
    n_clients: int
    n_steps: int
    reported_contamination: float
    robust_accuracy: float
    robustness: float
    seed: int
    standard_accuracy: float
    status: Literal["ok"]
    torch_version: str


class BnnTorchSkippedReport(TypedDict, total=False):
    """Degradation payload when the PyTorch optional extra is unavailable."""

    status: Required[str]


class VariationalAggregationReport(TypedDict):
    """Top-level payload written to ``variational_aggregation.json``."""

    capture_gap: NotRequired[float]
    converged: bool
    drifts: list[float]
    free_energy_history: list[float]
    iterations: int
    multi_start_final_f: NotRequired[float]
    multi_start_history: list[float]
    n_agents: int
    n_contaminated: int
    naive_influence: float
    robustness: float
    single_start_final_f: NotRequired[float]
    single_start_history: list[float]
    true_state: int
    variational_influence: list[float]


class ContaminationGalleryCell(TypedDict):
    """One seed-aggregated contamination-gallery mechanism cell."""

    best_robust_method: str
    diff_ci: list[float]
    directional: bool
    mean_diff: float
    naive_ci: list[float]
    naive_mean: float
    reliably_beats: bool
    robust_ci: list[float]
    robust_mean: float
    win_fraction: float


class ContaminationGalleryReport(TypedDict):
    """Top-level payload written to ``contamination_gallery.json``."""

    by_kind: dict[str, ContaminationGalleryCell]
    directional_kinds: list[str]
    entropy_kinds: list[str]
    entropy_naive_robust: bool
    kinds: list[str]
    n_agents: int
    n_contaminated: int
    n_seeds: int
    n_trials: int
    rate: float
    reliable_kinds: list[str]
    reliable_win_fraction: float
    seed: int


class RobustnessOnsetCell(TypedDict):
    """One pooled-method, seed-aggregated robustness-onset rate cell."""

    best_robust_method_by_rate: list[str]
    naive_ci: list[list[float]]
    naive_curve: list[float]
    onset_rate: float | None
    rates: list[float]
    robust_ci: list[list[float]]
    robust_curve: list[float]
    win_curve: list[float]
    by_rate: NotRequired[dict[str, object]]


class RobustnessOnsetReport(TypedDict):
    """Top-level payload written to ``robustness_onset.json``."""

    by_kind: dict[str, RobustnessOnsetCell]
    kinds: list[str]
    n_agents: int
    n_contaminated: int
    n_seeds: int
    n_trials: int
    onset_win_fraction: float
    rates: list[float]
    seed: int


class MovingWorldReport(TypedDict):
    """Top-level payload written to ``moving_world.json``."""

    accuracy: dict[str, object]
    free_energy_gap: dict[str, object]
    multiseed: dict[str, object]
    n_agents: int
    n_positions: int
    n_steps: int
    n_steps_to_consensus: dict[str, object]
    n_trials: int
    seed: int


class ParameterRecoveryReport(TypedDict):
    """Top-level payload written to ``parameter_recovery.json``."""

    abs_error: list[float]
    acuity_grid: list[float]
    analysis_unit: str
    claim_boundary: str
    estimator: str
    fit_bounds: list[float]
    fit_grid: list[float]
    fit_resolution: int
    interval_method: str
    interval_percent: int
    mean_abs_error: float
    n_grid_points: int
    n_independent_trials: int
    n_observations: int
    n_trials: int
    r_squared: float
    recovered_acuity: list[float]
    recovered_acuity_by_trial: list[object]
    recovered_acuity_ci_hi: list[float]
    recovered_acuity_ci_lo: list[float]
    replication_unit: str
    schema_version: str
    seed: int
    seed_role: str
    scope: str
    true_acuity: list[float]


class HierarchicalWorldReport(TypedDict):
    """Top-level payload written to ``hierarchical_world.json``."""

    acuity: float
    context_accuracy: float
    free_energy_gap: dict[str, object]
    location_accuracy: dict[str, object]
    location_accuracy_gap: float
    multiseed: dict[str, object]
    n_agents: int
    n_contexts: int
    n_iters: int
    n_trials: int
    seed: int


class NLevel3WorldReport(TypedDict):
    """Top-level payload written to ``nlevel3_world.json``."""

    acuity: float
    context_accuracy: float
    free_energy_gap: dict[str, object]
    location_accuracy: dict[str, object]
    location_accuracy_gap: float
    meta_context_accuracy: float
    multiseed: dict[str, object]
    n_agents: int
    n_contexts: int
    n_iters: int
    n_levels: int
    n_meta_contexts: int
    n_trials: int
    seed: int


class CrossStudySummaryReport(TypedDict):
    """Top-level payload written to ``cross_study_summary.json``."""

    n_seeds: int
    n_trials: int
    seed: int
    studies: list[object]


class DisjointFovWorldReport(TypedDict):
    """Top-level payload written to ``disjoint_fov_world.json``."""

    communicating_accuracy: float
    efe_navigation: dict[str, object]
    fov_width: int
    gap: float
    isolated_accuracy: float
    multiseed: dict[str, object]
    n_agents: int
    n_positions: int
    n_steps: int
    n_trials: int
    seed: int


class ComplexityScalingReport(TypedDict):
    """Top-level payload written to ``complexity_scaling.json``."""

    analytic_specs: list[object]
    benchmark: dict[str, object]
    claim_boundary: str
    machine: dict[str, object]
    measurements: list[object]
    schema_version: str
    seed: int
    status: str


class ConditionalWorldReport(TypedDict):
    """Top-level payload written to ``conditional_world.json``."""

    by_scenario: dict[str, object]
    claim_status: str
    controls: dict[str, object]
    grid: list[object]
    independent_unit: str
    n_agents: int
    n_seeds: int
    n_states: int
    n_trials: int
    primary_estimand: str
    robustness: float
    schema_version: str
    seed: int


class ReviewGridReport(TypedDict):
    """Top-level payload written to ``robustness_review_grid.json``."""

    analysis_profile: str
    attack_mechanisms: list[str]
    conditional_world: dict[str, object]
    controls: dict[str, object]
    divergences: list[str]
    directional_mechanisms: list[str]
    entropy_controls: list[str]
    independent_unit: str
    n_agents: int
    n_seeds: int
    n_trials: int
    primary_estimand: str
    precision_plan: dict[str, object]
    rates: list[float]
    rate_profiles: dict[str, object]
    robustness: float
    schema_version: str
    seed: int
    seed_schedule: dict[str, object]
    selection_status: str
    statistics: dict[str, object]
    trial_structure: str


class BeliefQualityReport(TypedDict):
    """Top-level payload written to ``belief_quality.json``."""

    by_scenario: dict[str, object]
    control_scores: dict[str, object]
    controls: dict[str, object]
    independent_unit: str
    n_agents: int
    n_seeds: int
    n_trials: int
    primary_estimand: str
    robustness: float
    schema_version: str
    seed: int


class ApplicationIntegrityFlowReport(TypedDict):
    """Source-owned payload for the application integrity flow figure."""

    schema_version: str
    study_status: str
    source_relation: str
    panels: list[dict[str, object]]
    solver_statuses: list[dict[str, object]]
    verification_levels: list[dict[str, object]]
    no_claims: list[str]


class EvidenceReplicationMapReport(TypedDict):
    """Source-owned payload for the evidence and replication map."""

    schema_version: str
    study_status: str
    source_relation: str
    evidence_classes: list[dict[str, object]]
    lanes: list[dict[str, object]]
    nesting: list[dict[str, object]]
    nesting_edges: list[dict[str, object]]
    no_claims: list[str]


class SourceRenderProvenanceReport(TypedDict):
    """Source-owned payload for producer order and stale invalidation."""

    schema_version: str
    study_status: str
    source_relation: str
    inputs: list[dict[str, object]]
    stages: list[dict[str, object]]
    terminals: list[dict[str, object]]
    producer_edges: list[dict[str, object]]
    invalidation_edges: list[dict[str, object]]
    authorization_boundary: str
    manifest_cycle_boundary: str
    no_claims: list[str]


class SensitivityReport(TypedDict):
    """Source-bound payload for the two configured sensitivity grids."""

    schema_version: str
    seed: int
    n_trials: int
    acuity_values: list[float]
    n_agents_values: list[int]
    noise_floor: float
    belief_sharing: dict[str, object]
    hierarchical: dict[str, object]
    estimands: dict[str, str]
    unit: str
    analysis_unit: str
    replication_unit: str
    uncertainty: str
    claim_boundary: str


class FigureMetadataEntry(TypedDict):
    """Per-figure metadata payload written inside ``figure_registry.json``."""

    label: str
    filename: str
    path: str
    source_manuscript: str
    caption: str
    generated_by: str
    status: str
    source_relation: str
    source_figure: str
    source_equation: str
    source_citation: str
    estimand: str
    unit: str
    uncertainty: str
    replication_unit: str
    alt_text: str
    long_description: NotRequired[str]
    exact_value_fallback: NotRequired[str]


class FigureRegistryPayload(TypedDict):
    """Top-level figure registry payload."""

    schema_version: str
    generated_by: str
    figures: list[FigureMetadataEntry]
    exact_value_artifact: NotRequired[dict[str, object]]


class ReportSchemaError(ValueError):
    """Raised when a report or figure-registry payload violates its schema."""


@dataclass(frozen=True)
class SchemaDefinition:
    """Shallow top-level schema definition."""

    required: Mapping[str, TypeTag]
    optional: Mapping[str, TypeTag] = field(default_factory=dict)


@dataclass(frozen=True)
class FigureDependencyContract:
    """Declared top-level report fields consumed by one figure generator."""

    report_name: str
    required_fields: Mapping[str, TypeTag]
    optional_fields: Mapping[str, TypeTag] = field(default_factory=dict)


_REPORT_SCHEMAS: dict[str, SchemaDefinition] = {
    "belief_sharing": SchemaDefinition(
        required={
            "schema_version": "str",
            "analysis_unit": "str",
            "communicating_free_energy": "list",
            "communicating_mean": "number",
            "communication_helps": "bool",
            "difference_definition": "str",
            "free_energy_gap": "number",
            "incommunicado_free_energy": "list",
            "incommunicado_mean": "number",
            "interval_method": "str",
            "interval_percent": "int",
            "n_agents": "int",
            "n_seeds": "int",
            "paired_difference_ci": "list",
            "paired_difference_mean": "number",
            "paired_free_energy_difference": "list",
            "replication_unit": "str",
        }
    ),
    "language_acquisition": SchemaDefinition(
        required={
            "final_kl": "number",
            "initial_kl": "number",
            "kl_trajectory": "list",
            "kl_trajectory_by_seed": "list",
            "monotone_decreasing": "bool",
            "n_points": "int",
            "n_seeds": "int",
            "num_steps": "int",
            "seed": "int",
            "trajectory_ci_hi": "list",
            "trajectory_ci_lo": "list",
        }
    ),
    "emergence": SchemaDefinition(
        required={
            "convergence": "bool",
            "delta_F_redundant": "number",
            "delta_F_supported": "number",
            "n": "int",
            "n_states": "int",
            "seed": "int",
        }
    ),
    "robustness_sweep": SchemaDefinition(
        required={
            "schema_version": "str",
            "accuracy_at_verdict_rate": "dict",
            "accuracy_by_method_and_rate": "dict",
            "accuracy_threshold": "number",
            "any_robust_wins": "bool",
            "attack_target_state": "int",
            "divergences": "list",
            "fdr_alpha": "number",
            "headline_method": "str",
            "headline_selection_rule": "str",
            "headline_tie_break": "str",
            "headline_tie_set": "list",
            "headline_is_display_selection": "bool",
            "largest_mean_difference_method": "str",
            "headline_n_for_target_power": "int",
            "headline_power": "number",
            "kind": "str",
            "n": "int",
            "n_agents": "int",
            "n_contaminated": "int",
            "n_trials": "int",
            "naive_degrades_with_rate": "bool",
            "naive_verdict_rate_mean": "number",
            "paired_tests_by_rate": "dict",
            "per_rate_summary": "dict",
            "power_alpha": "number",
            "power_alternative": "str",
            "prospective_n_for_target_power": "int",
            "rates": "list",
            "robust_above_threshold_at_worst_rate": "bool",
            "seed": "int",
            "server_robustness_by_label": "dict",
            "target_power": "number",
            "true_state": "int",
            "verdict": "dict",
            "verdict_rate": "number",
            "trial_structure": "str",
            "analysis_unit": "str",
            "worst_rate_best_method": "str",
            "worst_rate": "number",
            "paired_test_alternative": "str",
            "fdr_family_ownership": "str",
            "d_equivalent_status": "str",
        }
    ),
    "hierarchical_bmr": SchemaDefinition(
        required={
            "acuity": "number",
            "claim_boundary": "str",
            "configured_control_passed": "bool",
            "decision_rule": "str",
            "degenerate": "dict",
            "degenerate_recommends_prune_top": "bool",
            "degenerate_top_surprise": "number",
            "informative": "dict",
            "informative_keeps_top": "bool",
            "informative_top_surprise": "number",
            "n_iters": "int",
            "n_levels": "int",
            "obs": "int",
            "primary_control": "str",
            "schema_version": "str",
            "secondary_diagnostic": "str",
            "seed": "int",
            "seed_role": "str",
            "study_status": "str",
            "surprise_tol": "number",
        }
    ),
    "heuristic_characterization": SchemaDefinition(
        required={
            "breakdown": "dict",
            "claim_level": "str",
            "formal_no_go": "dict",
            "grid": "dict",
            "independent_unit": "str",
            "influence_naive": "dict",
            "influence_robust": "dict",
            "no_claim": "str",
            "schema_version": "str",
            "seed": "int",
            "theory_status": "str",
        }
    ),
    "efe_decomposition": SchemaDefinition(
        required={
            "ambiguity": "number",
            "epistemic_value": "number",
            "identity_residual": "number",
            "policy": "list",
            "pragmatic_value": "number",
            "prior_type": "str",
            "risk": "number",
            "total": "number",
        }
    ),
    "robust_influence_weights": SchemaDefinition(
        required={
            "normalized_effective_weights": "list",
            "contaminated_indices": "list",
            "n_agents": "int",
            "n_contaminated": "int",
            "robustness": "number",
            "schema_version": "str",
            "true_state": "int",
        }
    ),
    "bnn_robustness": SchemaDefinition(
        required={
            "accuracy_by_config": "dict",
            "accuracy_ci_by_config": "dict",
            "accuracy_seed_values_by_config": "dict",
            "analysis_unit": "str",
            "ci_percent": "int",
            "claim_boundary": "str",
            "configuration_boundary": "str",
            "contamination_levels": "list",
            "interval_method": "str",
            "model_family": "str",
            "n_bootstrap": "int",
            "n_per": "int",
            "n_seeds": "int",
            "replication_unit": "str",
            "robust_loss_param": "number",
            "robust_minus_standard": "list",
            "schema_version": "str",
            "seed": "int",
            "selection_disclosure": "str",
            "study_status": "str",
        },
        optional={
            "peak_margin": "number",
            "peak_margin_contamination": "number",
        },
    ),
    "variational_aggregation": SchemaDefinition(
        required={
            "converged": "bool",
            "drifts": "list",
            "free_energy_history": "list",
            "iterations": "int",
            "multi_start_history": "list",
            "n_agents": "int",
            "n_contaminated": "int",
            "naive_influence": "number",
            "robustness": "number",
            "single_start_history": "list",
            "true_state": "int",
            "variational_influence": "list",
        },
        optional={
            "capture_gap": "number",
            "multi_start_final_f": "number",
            "single_start_final_f": "number",
        },
    ),
    "contamination_gallery": SchemaDefinition(
        required={
            "by_kind": "dict",
            "directional_kinds": "list",
            "entropy_kinds": "list",
            "entropy_naive_robust": "bool",
            "kinds": "list",
            "n_agents": "int",
            "n_contaminated": "int",
            "n_seeds": "int",
            "n_trials": "int",
            "rate": "number",
            "reliable_kinds": "list",
            "reliable_win_fraction": "number",
            "seed": "int",
        }
    ),
    "robustness_onset": SchemaDefinition(
        required={
            "by_kind": "dict",
            "kinds": "list",
            "n_agents": "int",
            "n_contaminated": "int",
            "n_seeds": "int",
            "n_trials": "int",
            "onset_win_fraction": "number",
            "rates": "list",
            "seed": "int",
        }
    ),
    "moving_world": SchemaDefinition(
        required={
            "accuracy": "dict",
            "free_energy_gap": "dict",
            "multiseed": "dict",
            "n_agents": "int",
            "n_positions": "int",
            "n_steps": "int",
            "n_steps_to_consensus": "dict",
            "n_trials": "int",
            "seed": "int",
        }
    ),
    "parameter_recovery": SchemaDefinition(
        required={
            "abs_error": "list",
            "acuity_grid": "list",
            "analysis_unit": "str",
            "claim_boundary": "str",
            "estimator": "str",
            "fit_bounds": "list",
            "fit_grid": "list",
            "fit_resolution": "int",
            "interval_method": "str",
            "interval_percent": "int",
            "mean_abs_error": "number",
            "n_grid_points": "int",
            "n_independent_trials": "int",
            "n_observations": "int",
            "n_trials": "int",
            "r_squared": "number",
            "recovered_acuity": "list",
            "recovered_acuity_by_trial": "list",
            "recovered_acuity_ci_hi": "list",
            "recovered_acuity_ci_lo": "list",
            "replication_unit": "str",
            "schema_version": "str",
            "seed": "int",
            "seed_role": "str",
            "scope": "str",
            "true_acuity": "list",
        },
    ),
    "hierarchical_world": SchemaDefinition(
        required={
            "acuity": "number",
            "context_accuracy": "number",
            "free_energy_gap": "dict",
            "location_accuracy": "dict",
            "location_accuracy_gap": "number",
            "multiseed": "dict",
            "n_agents": "int",
            "n_contexts": "int",
            "n_iters": "int",
            "n_trials": "int",
            "seed": "int",
        }
    ),
    "nlevel3_world": SchemaDefinition(
        required={
            "acuity": "number",
            "context_accuracy": "number",
            "free_energy_gap": "dict",
            "location_accuracy": "dict",
            "location_accuracy_gap": "number",
            "meta_context_accuracy": "number",
            "multiseed": "dict",
            "n_agents": "int",
            "n_contexts": "int",
            "n_iters": "int",
            "n_levels": "int",
            "n_meta_contexts": "int",
            "n_trials": "int",
            "seed": "int",
        }
    ),
    "cross_study_summary": SchemaDefinition(
        required={
            "n_seeds": "int",
            "n_trials": "int",
            "seed": "int",
            "studies": "list",
        }
    ),
    "disjoint_fov_world": SchemaDefinition(
        required={
            "communicating_accuracy": "number",
            "efe_navigation": "dict",
            "fov_width": "int",
            "gap": "number",
            "isolated_accuracy": "number",
            "multiseed": "dict",
            "n_agents": "int",
            "n_positions": "int",
            "n_steps": "int",
            "n_trials": "int",
            "seed": "int",
        }
    ),
    "complexity_scaling": SchemaDefinition(
        required={
            "analytic_specs": "list",
            "benchmark": "dict",
            "claim_boundary": "str",
            "machine": "dict",
            "measurements": "list",
            "schema_version": "str",
            "seed": "int",
            "status": "str",
        }
    ),
    "conditional_world": SchemaDefinition(
        required={
            "by_scenario": "dict",
            "claim_status": "str",
            "controls": "dict",
            "grid": "list",
            "independent_unit": "str",
            "n_agents": "int",
            "n_seeds": "int",
            "n_states": "int",
            "n_trials": "int",
            "primary_estimand": "str",
            "robustness": "number",
            "schema_version": "str",
            "seed": "int",
        }
    ),
    "robustness_review_grid": SchemaDefinition(
        required={
            "analysis_profile": "str",
            "attack_mechanisms": "list",
            "conditional_world": "dict",
            "controls": "dict",
            "divergences": "list",
            "directional_mechanisms": "list",
            "entropy_controls": "list",
            "independent_unit": "str",
            "n_agents": "int",
            "n_seeds": "int",
            "n_trials": "int",
            "primary_estimand": "str",
            "precision_plan": "dict",
            "rates": "list",
            "rate_profiles": "dict",
            "robustness": "number",
            "schema_version": "str",
            "seed": "int",
            "seed_schedule": "dict",
            "selection_status": "str",
            "statistics": "dict",
            "trial_structure": "str",
        }
    ),
    "belief_quality": SchemaDefinition(
        required={
            "by_scenario": "dict",
            "control_scores": "dict",
            "controls": "dict",
            "independent_unit": "str",
            "n_agents": "int",
            "n_seeds": "int",
            "n_trials": "int",
            "primary_estimand": "str",
            "robustness": "number",
            "schema_version": "str",
            "seed": "int",
        }
    ),
    "application_integrity_flow": SchemaDefinition(
        required={
            "schema_version": "str",
            "study_status": "str",
            "source_relation": "str",
            "panels": "list",
            "solver_statuses": "list",
            "verification_levels": "list",
            "no_claims": "list",
        }
    ),
    "evidence_replication_map": SchemaDefinition(
        required={
            "schema_version": "str",
            "study_status": "str",
            "source_relation": "str",
            "evidence_classes": "list",
            "lanes": "list",
            "nesting": "list",
            "nesting_edges": "list",
            "no_claims": "list",
        }
    ),
    "source_render_provenance": SchemaDefinition(
        required={
            "schema_version": "str",
            "study_status": "str",
            "source_relation": "str",
            "inputs": "list",
            "stages": "list",
            "terminals": "list",
            "producer_edges": "list",
            "invalidation_edges": "list",
            "authorization_boundary": "str",
            "manifest_cycle_boundary": "str",
            "no_claims": "list",
        }
    ),
    "sensitivity": SchemaDefinition(
        required={
            "schema_version": "str",
            "seed": "int",
            "n_trials": "int",
            "acuity_values": "list",
            "n_agents_values": "list",
            "noise_floor": "number",
            "belief_sharing": "dict",
            "hierarchical": "dict",
            "estimands": "dict",
            "unit": "str",
            "analysis_unit": "str",
            "replication_unit": "str",
            "uncertainty": "str",
            "claim_boundary": "str",
        }
    ),
}

# Versioned payloads are intentionally fail-closed. Compatibility keys may be
# retained inside a supported payload, but a reader must never guess how an
# unsupported report version maps onto the canonical fields.
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

_FIGURE_METADATA_SCHEMA = SchemaDefinition(
    required={
        "label": "str",
        "filename": "str",
        "path": "str",
        "source_manuscript": "str",
        "caption": "str",
        "generated_by": "str",
        "status": "str",
        "source_relation": "str",
        "source_figure": "str",
        "source_equation": "str",
        "source_citation": "str",
        "estimand": "str",
        "unit": "str",
        "uncertainty": "str",
        "replication_unit": "str",
        "alt_text": "str",
    },
    optional={
        "long_description": "str",
        "exact_value_fallback": "str",
    },
)

_FIGURE_REGISTRY_SCHEMA = SchemaDefinition(
    required={
        "schema_version": "str",
        "generated_by": "str",
        "figures": "list",
    },
    optional={"exact_value_artifact": "dict"},
)

_FIGURE_EXACT_VALUES_SCHEMA = SchemaDefinition(
    required={
        "schema_version": "str",
        "generated_by": "str",
        "tables": "list",
    }
)

FIGURE_DEPENDENCY_CONTRACTS: dict[str, tuple[FigureDependencyContract, ...]] = {
    "application_integrity_flow": (
        FigureDependencyContract(
            report_name="application_integrity_flow",
            required_fields={
                "panels": "list",
                "solver_statuses": "list",
                "verification_levels": "list",
                "no_claims": "list",
                "source_relation": "str",
            },
        ),
    ),
    "evidence_replication_map": (
        FigureDependencyContract(
            report_name="evidence_replication_map",
            required_fields={
                "evidence_classes": "list",
                "lanes": "list",
                "nesting": "list",
                "nesting_edges": "list",
                "no_claims": "list",
                "source_relation": "str",
            },
        ),
    ),
    "source_render_provenance": (
        FigureDependencyContract(
            report_name="source_render_provenance",
            required_fields={
                "inputs": "list",
                "stages": "list",
                "terminals": "list",
                "producer_edges": "list",
                "invalidation_edges": "list",
                "authorization_boundary": "str",
                "manifest_cycle_boundary": "str",
                "no_claims": "list",
            },
        ),
    ),
    "sensitivity_heatmap": (
        FigureDependencyContract(
            report_name="sensitivity",
            required_fields={
                "acuity_values": "list",
                "n_agents_values": "list",
                "noise_floor": "number",
                "belief_sharing": "dict",
                "hierarchical": "dict",
                "analysis_unit": "str",
                "replication_unit": "str",
                "uncertainty": "str",
                "claim_boundary": "str",
            },
        ),
    ),
    "free_energy_comparison": (
        FigureDependencyContract(
            report_name="belief_sharing",
            required_fields={
                "incommunicado_free_energy": "list",
                "communicating_free_energy": "list",
                "paired_free_energy_difference": "list",
                "paired_difference_mean": "number",
                "paired_difference_ci": "list",
                "difference_definition": "str",
                "analysis_unit": "str",
                "replication_unit": "str",
                "interval_method": "str",
                "interval_percent": "int",
            },
        ),
    ),
    "robustness_sweep": (
        FigureDependencyContract(
            report_name="robustness_sweep",
            required_fields={
                "accuracy_by_method_and_rate": "dict",
                "accuracy_threshold": "number",
                "analysis_unit": "str",
                "divergences": "list",
                "headline_is_display_selection": "bool",
                "n_trials": "int",
                "per_rate_summary": "dict",
                "rates": "list",
                "server_robustness_by_label": "dict",
                "trial_structure": "str",
            },
        ),
    ),
    "language_kl_decay": (
        FigureDependencyContract(
            report_name="language_acquisition",
            required_fields={
                "kl_trajectory": "list",
                "trajectory_ci_lo": "list",
                "trajectory_ci_hi": "list",
            },
            optional_fields={
                "monotone_decreasing": "bool",
                "n_seeds": "int",
            },
        ),
    ),
    "emergence_bmr": (
        FigureDependencyContract(
            report_name="emergence",
            required_fields={
                "delta_F_redundant": "number",
                "delta_F_supported": "number",
            },
            optional_fields={"convergence": "bool"},
        ),
    ),
    "hierarchical_bmr": (
        FigureDependencyContract(
            report_name="hierarchical_bmr",
            required_fields={
                "degenerate": "dict",
                "informative": "dict",
                "surprise_tol": "number",
                "decision_rule": "str",
                "secondary_diagnostic": "str",
                "claim_boundary": "str",
            },
        ),
    ),
    "heuristic_breakdown": (
        FigureDependencyContract(
            report_name="heuristic_characterization",
            required_fields={
                "influence_naive": "dict",
                "influence_robust": "dict",
                "breakdown": "dict",
            },
            optional_fields={"grid": "dict"},
        ),
    ),
    "efe_decomposition": (
        FigureDependencyContract(
            report_name="efe_decomposition",
            required_fields={
                "risk": "number",
                "ambiguity": "number",
                "pragmatic_value": "number",
                "epistemic_value": "number",
            },
        ),
    ),
    "robust_influence_weights": (
        FigureDependencyContract(
            report_name="robust_influence_weights",
            required_fields={
                "normalized_effective_weights": "list",
                "contaminated_indices": "list",
            },
        ),
    ),
    "bnn_robustness": (
        FigureDependencyContract(
            report_name="bnn_robustness",
            required_fields={
                "accuracy_by_config": "dict",
                "accuracy_ci_by_config": "dict",
                "accuracy_seed_values_by_config": "dict",
                "ci_percent": "int",
                "contamination_levels": "list",
                "interval_method": "str",
                "model_family": "str",
                "n_bootstrap": "int",
                "n_per": "int",
                "n_seeds": "int",
                "robust_loss_param": "number",
                "study_status": "str",
                "selection_disclosure": "str",
                "analysis_unit": "str",
                "replication_unit": "str",
                "claim_boundary": "str",
                "configuration_boundary": "str",
            },
        ),
    ),
    "aggregation_descent": (
        FigureDependencyContract(
            report_name="variational_aggregation",
            required_fields={
                "free_energy_history": "list",
                "converged": "bool",
            },
        ),
    ),
    "bounded_influence": (
        FigureDependencyContract(
            report_name="variational_aggregation",
            required_fields={
                "drifts": "list",
                "variational_influence": "list",
                "naive_influence": "number",
            },
        ),
    ),
    "contamination_gallery": (
        FigureDependencyContract(
            report_name="contamination_gallery",
            required_fields={"by_kind": "dict"},
        ),
    ),
    "descent_comparison": (
        FigureDependencyContract(
            report_name="variational_aggregation",
            required_fields={
                "single_start_history": "list",
                "multi_start_history": "list",
            },
        ),
    ),
    "robustness_onset": (
        FigureDependencyContract(
            report_name="robustness_onset",
            required_fields={"by_kind": "dict"},
        ),
    ),
    "conditional_world": (
        FigureDependencyContract(
            report_name="conditional_world",
            required_fields={"by_scenario": "dict", "controls": "dict"},
        ),
    ),
    "robustness_review_grid": (
        FigureDependencyContract(
            report_name="robustness_review_grid",
            required_fields={
                "conditional_world": "dict",
                "rate_profiles": "dict",
                "controls": "dict",
            },
        ),
    ),
    "belief_quality": (
        FigureDependencyContract(
            report_name="belief_quality",
            required_fields={"control_scores": "dict", "controls": "dict"},
        ),
    ),
    "moving_world": (
        FigureDependencyContract(
            report_name="moving_world",
            required_fields={
                "accuracy": "dict",
                "free_energy_gap": "dict",
                "n_steps_to_consensus": "dict",
            },
        ),
    ),
    "parameter_recovery": (
        FigureDependencyContract(
            report_name="parameter_recovery",
            required_fields={
                "true_acuity": "list",
                "recovered_acuity": "list",
                "recovered_acuity_ci_lo": "list",
                "recovered_acuity_ci_hi": "list",
                "abs_error": "list",
                "recovered_acuity_by_trial": "list",
                "fit_grid": "list",
                "fit_resolution": "int",
                "analysis_unit": "str",
                "replication_unit": "str",
                "scope": "str",
                "claim_boundary": "str",
                "r_squared": "number",
                "mean_abs_error": "number",
                "n_trials": "int",
                "n_observations": "int",
            },
        ),
    ),
    "cross_study_summary": (
        FigureDependencyContract(
            report_name="cross_study_summary",
            required_fields={"studies": "list"},
            optional_fields={"n_seeds": "int"},
        ),
    ),
    "hierarchical_pomdp": (
        FigureDependencyContract(
            report_name="hierarchical_world",
            required_fields={"location_accuracy_gap": "number"},
            optional_fields={"n_trials": "int"},
        ),
        FigureDependencyContract(
            report_name="nlevel3_world",
            required_fields={"location_accuracy_gap": "number"},
        ),
    ),
    "disjoint_fov_world": (
        FigureDependencyContract(
            report_name="disjoint_fov_world",
            required_fields={"multiseed": "dict"},
        ),
    ),
    "complexity_scaling": (
        FigureDependencyContract(
            report_name="complexity_scaling",
            required_fields={
                "analytic_specs": "list",
                "benchmark": "dict",
                "measurements": "list",
            },
        ),
    ),
}


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _check_finite_json(value: object, *, path: str = "payload") -> None:
    """Reject non-standard JSON numbers at every nested report position."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ReportSchemaError(f"{path} contains a non-finite JSON number")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _check_finite_json(nested, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _check_finite_json(nested, path=f"{path}[{index}]")


def _matches_type(tag: TypeTag, value: object) -> bool:
    if tag == "bool":
        return isinstance(value, bool)
    if tag == "dict":
        return isinstance(value, Mapping)
    if tag == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if tag == "list":
        return isinstance(value, list)
    if tag == "number":
        return _is_number(value)
    if tag == "str":
        return isinstance(value, str)
    raise AssertionError(f"Unhandled type tag: {tag}")


def _describe_value(value: object) -> str:
    return type(value).__name__


def _raise_missing(schema_name: str, field_name: str, tag: TypeTag) -> NoReturn:
    raise ReportSchemaError(f"{schema_name} payload missing required field {field_name!r} (expected {tag})")


def _raise_type_error(schema_name: str, field_name: str, tag: TypeTag, value: object) -> NoReturn:
    raise ReportSchemaError(
        f"{schema_name} payload field {field_name!r} expected {tag}, got {_describe_value(value)}"
    )


def _check(
    schema_name: str,
    payload: Mapping[str, object],
    spec: SchemaDefinition,
) -> None:
    for field_name, tag in spec.required.items():
        if field_name not in payload:
            _raise_missing(schema_name, field_name, tag)
        value = payload[field_name]
        if not _matches_type(tag, value):
            _raise_type_error(schema_name, field_name, tag, value)
    for field_name, tag in spec.optional.items():
        if field_name in payload and not _matches_type(tag, payload[field_name]):
            _raise_type_error(schema_name, field_name, tag, payload[field_name])


def _number_list(value: object, *, schema: str, field: str) -> list[float]:
    """Return a finite numeric list with a field-specific diagnostic."""
    if not isinstance(value, list) or any(not _is_number(item) for item in value):
        raise ReportSchemaError(f"{schema} payload field {field!r} must be a list of finite numbers")
    return [float(item) for item in value]


def _number(value: object, *, schema: str, field: str) -> float:
    """Return one finite non-Boolean number with a field-specific diagnostic."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportSchemaError(f"{schema} payload field {field!r} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ReportSchemaError(f"{schema} payload field {field!r} must be a finite number")
    return number


def _mapping(value: object, *, schema: str, field: str) -> Mapping[str, object]:
    """Return a nested mapping with a field-specific diagnostic."""
    if not isinstance(value, Mapping):
        raise ReportSchemaError(f"{schema} payload field {field!r} must be a mapping")
    return value


def _check_belief_sharing(payload: Mapping[str, object]) -> None:
    """Bind the paired free-energy estimand to its source arrays."""
    communicating = _number_list(
        payload["communicating_free_energy"], schema="belief_sharing", field="communicating_free_energy"
    )
    incommunicado = _number_list(
        payload["incommunicado_free_energy"], schema="belief_sharing", field="incommunicado_free_energy"
    )
    paired = _number_list(
        payload["paired_free_energy_difference"],
        schema="belief_sharing",
        field="paired_free_energy_difference",
    )
    n_seeds = payload["n_seeds"]
    assert isinstance(n_seeds, int) and not isinstance(n_seeds, bool)
    if n_seeds < 2 or len(communicating) != n_seeds or len(incommunicado) != n_seeds:
        raise ReportSchemaError("belief_sharing paired arrays must each contain n_seeds >= 2 values")
    expected = [inco - comm for inco, comm in zip(incommunicado, communicating)]
    if len(paired) != n_seeds or any(
        not math.isclose(observed, target, rel_tol=0.0, abs_tol=1e-12)
        for observed, target in zip(paired, expected)
    ):
        raise ReportSchemaError(
            "belief_sharing paired_free_energy_difference must equal "
            "incommunicado_free_energy - communicating_free_energy elementwise"
        )
    comm_mean = _number(payload["communicating_mean"], schema="belief_sharing", field="communicating_mean")
    incom_mean = _number(payload["incommunicado_mean"], schema="belief_sharing", field="incommunicado_mean")
    paired_mean = _number(
        payload["paired_difference_mean"],
        schema="belief_sharing",
        field="paired_difference_mean",
    )
    if not math.isclose(comm_mean, sum(communicating) / n_seeds, rel_tol=0.0, abs_tol=1e-12):
        raise ReportSchemaError("belief_sharing communicating_mean disagrees with source values")
    if not math.isclose(incom_mean, sum(incommunicado) / n_seeds, rel_tol=0.0, abs_tol=1e-12):
        raise ReportSchemaError("belief_sharing incommunicado_mean disagrees with source values")
    if not math.isclose(paired_mean, sum(paired) / n_seeds, rel_tol=0.0, abs_tol=1e-12):
        raise ReportSchemaError("belief_sharing paired_difference_mean disagrees with paired values")
    free_energy_gap = _number(payload["free_energy_gap"], schema="belief_sharing", field="free_energy_gap")
    if not math.isclose(free_energy_gap, paired_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ReportSchemaError("belief_sharing free_energy_gap must equal paired_difference_mean")
    if payload["communication_helps"] is not (comm_mean < incom_mean):
        raise ReportSchemaError("belief_sharing communication_helps disagrees with paired means")
    interval = _number_list(
        payload["paired_difference_ci"], schema="belief_sharing", field="paired_difference_ci"
    )
    if len(interval) != 2 or interval[0] > paired_mean or paired_mean > interval[1]:
        raise ReportSchemaError("belief_sharing paired_difference_ci must be ordered around the mean")
    if payload["difference_definition"] != "incommunicado_minus_communicating":
        raise ReportSchemaError(
            "belief_sharing difference_definition must be 'incommunicado_minus_communicating'"
        )
    if payload["replication_unit"] != "configured seed":
        raise ReportSchemaError("belief_sharing replication_unit must be 'configured seed'")
    analysis_unit = payload["analysis_unit"]
    if analysis_unit != "configured seed with within-seed paired conditions":
        raise ReportSchemaError("belief_sharing analysis_unit must identify within-seed pairing")
    if payload["interval_percent"] != 95:
        raise ReportSchemaError("belief_sharing interval_percent must be 95")


def _check_parameter_recovery(payload: Mapping[str, object]) -> None:
    """Validate the complete finite-grid recovery design and trial summaries."""
    true_values = _number_list(payload["true_acuity"], schema="parameter_recovery", field="true_acuity")
    acuity_grid = _number_list(payload["acuity_grid"], schema="parameter_recovery", field="acuity_grid")
    recovered = _number_list(
        payload["recovered_acuity"], schema="parameter_recovery", field="recovered_acuity"
    )
    ci_lo = _number_list(
        payload["recovered_acuity_ci_lo"], schema="parameter_recovery", field="recovered_acuity_ci_lo"
    )
    ci_hi = _number_list(
        payload["recovered_acuity_ci_hi"], schema="parameter_recovery", field="recovered_acuity_ci_hi"
    )
    abs_error = _number_list(payload["abs_error"], schema="parameter_recovery", field="abs_error")
    fit_grid = _number_list(payload["fit_grid"], schema="parameter_recovery", field="fit_grid")
    fit_bounds = _number_list(payload["fit_bounds"], schema="parameter_recovery", field="fit_bounds")
    n_grid_points = payload["n_grid_points"]
    n_trials = payload["n_trials"]
    fit_resolution = payload["fit_resolution"]
    assert isinstance(n_grid_points, int) and not isinstance(n_grid_points, bool)
    assert isinstance(n_trials, int) and not isinstance(n_trials, bool)
    assert isinstance(fit_resolution, int) and not isinstance(fit_resolution, bool)
    if n_grid_points < 1 or n_trials < 1 or fit_resolution < 2:
        raise ReportSchemaError("parameter_recovery grid, trial, and fit budgets must be positive")
    if true_values != acuity_grid:
        raise ReportSchemaError("parameter_recovery true_acuity must preserve acuity_grid exactly")
    grid_fields = (true_values, recovered, ci_lo, ci_hi, abs_error)
    if any(len(values) != n_grid_points for values in grid_fields):
        raise ReportSchemaError("parameter_recovery per-grid fields must match n_grid_points")
    if len(fit_grid) != fit_resolution or fit_grid != sorted(set(fit_grid)):
        raise ReportSchemaError("parameter_recovery fit_grid must be strictly increasing at fit_resolution")
    if len(fit_bounds) != 2 or fit_bounds != [fit_grid[0], fit_grid[-1]]:
        raise ReportSchemaError("parameter_recovery fit_bounds must equal the fit_grid endpoints")
    trial_rows = payload["recovered_acuity_by_trial"]
    assert isinstance(trial_rows, list)
    if len(trial_rows) != n_grid_points:
        raise ReportSchemaError("parameter_recovery recovered_acuity_by_trial has wrong grid count")
    for index, row_value in enumerate(trial_rows):
        row = _number_list(
            row_value,
            schema="parameter_recovery",
            field=f"recovered_acuity_by_trial[{index}]",
        )
        if len(row) != n_trials:
            raise ReportSchemaError("parameter_recovery trial row has wrong n_trials")
        mean = sum(row) / n_trials
        if not math.isclose(recovered[index], mean, rel_tol=0.0, abs_tol=1e-12):
            raise ReportSchemaError("parameter_recovery recovered_acuity disagrees with trial values")
        if ci_lo[index] > mean or mean > ci_hi[index]:
            raise ReportSchemaError("parameter_recovery empirical interval is not ordered around its mean")
        expected_abs = sum(abs(value - true_values[index]) for value in row) / n_trials
        if not math.isclose(abs_error[index], expected_abs, rel_tol=0.0, abs_tol=1e-12):
            raise ReportSchemaError("parameter_recovery abs_error disagrees with trial values")
    if payload["n_independent_trials"] != n_grid_points * n_trials:
        raise ReportSchemaError(
            "parameter_recovery n_independent_trials disagrees with grid and trial counts"
        )
    if not math.isclose(
        _number(
            payload["mean_abs_error"],
            schema="parameter_recovery",
            field="mean_abs_error",
        ),
        sum(abs_error) / n_grid_points,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ReportSchemaError("parameter_recovery mean_abs_error disagrees with grid errors")
    if payload["estimator"] != "finite_grid_maximum_marginal_likelihood":
        raise ReportSchemaError("parameter_recovery estimator must identify finite-grid MML")
    if payload["replication_unit"] != "synthetic trial":
        raise ReportSchemaError("parameter_recovery replication_unit must be 'synthetic trial'")
    if "seed is not" not in str(payload["seed_role"]):
        raise ReportSchemaError("parameter_recovery seed_role must distinguish seed from analysis unit")
    if "finite" not in str(payload["scope"]) or "no continuous-parameter" not in str(
        payload["claim_boundary"]
    ):
        raise ReportSchemaError("parameter_recovery scope and claim boundary must remain finite-grid bounded")


def _check_hierarchical_bmr(payload: Mapping[str, object]) -> None:
    """Bind the configured control verdict to its declared surprise threshold."""
    tolerance = _number(payload["surprise_tol"], schema="hierarchical_bmr", field="surprise_tol")
    if tolerance <= 0.0:
        raise ReportSchemaError("hierarchical_bmr surprise_tol must be positive")
    worlds: dict[str, Mapping[str, object]] = {}
    for world_name in ("degenerate", "informative"):
        world = _mapping(payload[world_name], schema="hierarchical_bmr", field=world_name)
        levels = world.get("levels")
        if not isinstance(levels, list) or not levels:
            raise ReportSchemaError(f"hierarchical_bmr {world_name} must contain non-leaf levels")
        for index, level_value in enumerate(levels):
            level = _mapping(level_value, schema="hierarchical_bmr", field=f"{world_name}.levels[{index}]")
            surprise = level.get("bayesian_surprise")
            prunable = level.get("prunable")
            if not _is_number(surprise) or not isinstance(prunable, bool):
                raise ReportSchemaError(
                    f"hierarchical_bmr {world_name} level must report surprise and prunable"
                )
            surprise_value = _number(
                surprise,
                schema="hierarchical_bmr",
                field=f"{world_name}.levels[{index}].bayesian_surprise",
            )
            if prunable is not (surprise_value < tolerance):
                raise ReportSchemaError(
                    f"hierarchical_bmr {world_name} prunable flag disagrees with surprise_tol"
                )
        worlds[world_name] = world
    deg_levels = worlds["degenerate"]["levels"]
    inf_levels = worlds["informative"]["levels"]
    assert isinstance(deg_levels, list) and isinstance(inf_levels, list)
    deg_top = next(
        (
            _mapping(level, schema="hierarchical_bmr", field="degenerate top")
            for level in deg_levels
            if isinstance(level, Mapping) and level.get("level") == 0
        ),
        None,
    )
    inf_top = next(
        (
            _mapping(level, schema="hierarchical_bmr", field="informative top")
            for level in inf_levels
            if isinstance(level, Mapping) and level.get("level") == 0
        ),
        None,
    )
    if deg_top is None or inf_top is None:
        raise ReportSchemaError("hierarchical_bmr controls must report top level 0")
    if not math.isclose(
        _number(
            payload["degenerate_top_surprise"],
            schema="hierarchical_bmr",
            field="degenerate_top_surprise",
        ),
        _number(
            deg_top["bayesian_surprise"],
            schema="hierarchical_bmr",
            field="degenerate.bayesian_surprise",
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        _number(
            payload["informative_top_surprise"],
            schema="hierarchical_bmr",
            field="informative_top_surprise",
        ),
        _number(
            inf_top["bayesian_surprise"],
            schema="hierarchical_bmr",
            field="informative.bayesian_surprise",
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ReportSchemaError("hierarchical_bmr headline surprise disagrees with nested level")
    deg_prunes = worlds["degenerate"].get("recommended_prune") == 0
    inf_keeps = worlds["informative"].get("recommended_prune") != 0
    if payload["degenerate_recommends_prune_top"] is not deg_prunes:
        raise ReportSchemaError("hierarchical_bmr degenerate verdict disagrees with nested result")
    if payload["informative_keeps_top"] is not inf_keeps:
        raise ReportSchemaError("hierarchical_bmr informative verdict disagrees with nested result")
    if payload["configured_control_passed"] is not (deg_prunes and inf_keeps):
        raise ReportSchemaError("hierarchical_bmr configured_control_passed disagrees with verdicts")
    if payload["study_status"] != "configured_deterministic_control":
        raise ReportSchemaError("hierarchical_bmr study_status must identify a configured control")
    if "does not determine" not in str(payload["secondary_diagnostic"]):
        raise ReportSchemaError("hierarchical_bmr secondary diagnostic must not own the decision")
    if "no universal" not in str(payload["claim_boundary"]):
        raise ReportSchemaError("hierarchical_bmr claim_boundary must deny universal emergence")


def _check_bnn_robustness(payload: Mapping[str, object]) -> None:
    """Keep the legacy BNN stem bound to the point-estimate exploratory model."""
    if payload["model_family"] != "point_estimate_logistic_regression":
        raise ReportSchemaError("bnn_robustness model_family must be 'point_estimate_logistic_regression'")
    if payload["study_status"] != "exploratory_conditional_synthetic_sweep":
        raise ReportSchemaError("bnn_robustness study_status must remain exploratory and conditional")
    selection_disclosure = str(payload["selection_disclosure"])
    required_selection_fragments = (
        "peak_margin_contamination is selected within the displayed contamination sweep",
        "composite-configuration held-out accuracy margin",
        "robust_loss_param and n_per are configured inputs, not selected by this report",
    )
    if any(fragment not in selection_disclosure for fragment in required_selection_fragments):
        raise ReportSchemaError(
            "bnn_robustness selection_disclosure must limit selection to peak contamination"
        )
    expected_design_strings = {
        "analysis_unit": "synthetic-data seed within contamination operating point",
        "replication_unit": "synthetic-data seed",
        "interval_method": "percentile bootstrap across synthetic-data seeds",
    }
    for field_name, expected in expected_design_strings.items():
        if payload[field_name] != expected:
            raise ReportSchemaError(
                f"bnn_robustness {field_name} must remain {expected!r}"
            )
    ci_percent = payload["ci_percent"]
    n_bootstrap = payload["n_bootstrap"]
    n_per = payload["n_per"]
    assert isinstance(ci_percent, int) and not isinstance(ci_percent, bool)
    assert isinstance(n_bootstrap, int) and not isinstance(n_bootstrap, bool)
    assert isinstance(n_per, int) and not isinstance(n_per, bool)
    if ci_percent != 95 or n_bootstrap != 5000:
        raise ReportSchemaError(
            "bnn_robustness intervals must retain the 95-percent, 5000-resample design"
        )
    if n_per < 1:
        raise ReportSchemaError("bnn_robustness n_per must be a positive points-per-class count")
    robust_loss_param = _number(
        payload["robust_loss_param"],
        schema="bnn_robustness",
        field="robust_loss_param",
    )
    if not 0.0 <= robust_loss_param <= 1.0:
        raise ReportSchemaError("bnn_robustness robust_loss_param must lie in [0, 1]")
    boundary = str(payload["claim_boundary"])
    required_boundaries = (
        "no posterior-uncertainty BNN",
        "leakage-free calibration",
        "universal robustness",
        "source-protocol BNN replication",
        "legacy AR argument selects stronger L2",
        "does not evaluate an Alpha-Renyi objective",
    )
    if any(fragment not in boundary for fragment in required_boundaries):
        raise ReportSchemaError("bnn_robustness claim_boundary must retain every declared no-claim")
    levels = _number_list(
        payload["contamination_levels"], schema="bnn_robustness", field="contamination_levels"
    )
    if not levels or len(set(levels)) != len(levels) or any(level < 0.0 or level > 1.0 for level in levels):
        raise ReportSchemaError("bnn_robustness contamination levels must be unique and lie in [0, 1]")
    curves = _mapping(payload["accuracy_by_config"], schema="bnn_robustness", field="accuracy_by_config")
    intervals = _mapping(
        payload["accuracy_ci_by_config"], schema="bnn_robustness", field="accuracy_ci_by_config"
    )
    seed_values = _mapping(
        payload["accuracy_seed_values_by_config"],
        schema="bnn_robustness",
        field="accuracy_seed_values_by_config",
    )
    configuration_boundary = str(payload["configuration_boundary"])
    required_configuration_fragments = (
        "composite two-configuration contrast",
        "changes both client loss and L2 shrinkage",
        "L2 coefficient 0.05",
        "L2 coefficient 0.10",
        "cannot identify an RCCE-only effect",
        "neither branch computes a weight-space divergence",
    )
    if any(fragment not in configuration_boundary for fragment in required_configuration_fragments):
        raise ReportSchemaError(
            "bnn_robustness configuration_boundary must expose the proxy's L2 semantics"
        )
    expected_configs = {
        "nll / L2=0.05 (standard proxy)",
        "rcce / L2=0.10 (exploratory proxy)",
    }
    if (
        set(curves) != expected_configs
        or set(intervals) != expected_configs
        or set(seed_values) != expected_configs
    ):
        raise ReportSchemaError("bnn_robustness configuration mappings must retain both declared baselines")
    n_seeds = payload["n_seeds"]
    assert isinstance(n_seeds, int) and not isinstance(n_seeds, bool)
    if n_seeds < 2:
        raise ReportSchemaError("bnn_robustness n_seeds must be >= 2")
    for config in expected_configs:
        curve = _number_list(curves[config], schema="bnn_robustness", field=f"accuracy_by_config.{config}")
        config_intervals = intervals[config]
        config_seed_values = seed_values[config]
        if (
            len(curve) != len(levels)
            or not isinstance(config_intervals, list)
            or not isinstance(config_seed_values, list)
            or len(config_intervals) != len(levels)
            or len(config_seed_values) != len(levels)
        ):
            raise ReportSchemaError(
                "bnn_robustness curves, intervals, and seed rows must match operating points"
            )
        for index, value in enumerate(curve):
            if not 0.0 <= value <= 1.0:
                raise ReportSchemaError("bnn_robustness accuracy must lie in [0, 1]")
            interval = _number_list(
                config_intervals[index],
                schema="bnn_robustness",
                field=f"accuracy_ci_by_config.{config}[{index}]",
            )
            row = _number_list(
                config_seed_values[index],
                schema="bnn_robustness",
                field=f"accuracy_seed_values_by_config.{config}[{index}]",
            )
            seed_row_out_of_bounds = any(
                seed_accuracy < 0.0 or seed_accuracy > 1.0 for seed_accuracy in row
            )
            if len(row) != n_seeds or seed_row_out_of_bounds:
                raise ReportSchemaError(
                    "bnn_robustness seed accuracy row must match n_seeds and lie in [0, 1]"
                )
            row_mean = math.fsum(row) / len(row)
            if not math.isclose(value, row_mean, rel_tol=0.0, abs_tol=1e-12):
                raise ReportSchemaError(
                    "bnn_robustness reported mean disagrees with its seed accuracy row"
                )
            if (
                len(interval) != 2
                or interval[0] < 0.0
                or interval[1] > 1.0
                or interval[0] > value
                or value > interval[1]
            ):
                raise ReportSchemaError(
                    "bnn_robustness interval must lie in [0, 1] and contain its seed-row mean"
                )
    standard = _number_list(
        curves["nll / L2=0.05 (standard proxy)"],
        schema="bnn_robustness",
        field="standard",
    )
    robust = _number_list(
        curves["rcce / L2=0.10 (exploratory proxy)"],
        schema="bnn_robustness",
        field="robust",
    )
    gaps = _number_list(
        payload["robust_minus_standard"], schema="bnn_robustness", field="robust_minus_standard"
    )
    expected_gaps = [rob - std for rob, std in zip(robust, standard)]
    if len(gaps) != len(levels) or any(
        not math.isclose(gap, expected, rel_tol=0.0, abs_tol=1e-12)
        for gap, expected in zip(gaps, expected_gaps)
    ):
        raise ReportSchemaError("bnn_robustness robust_minus_standard disagrees with accuracy curves")
    if "peak_margin" in payload or "peak_margin_contamination" in payload:
        if "peak_margin" not in payload or "peak_margin_contamination" not in payload:
            raise ReportSchemaError("bnn_robustness peak selection fields must appear together")
        peak_index = max(range(len(gaps)), key=lambda index: gaps[index])
        if not math.isclose(
            _number(payload["peak_margin"], schema="bnn_robustness", field="peak_margin"),
            gaps[peak_index],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            _number(
                payload["peak_margin_contamination"],
                schema="bnn_robustness",
                field="peak_margin_contamination",
            ),
            levels[peak_index],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ReportSchemaError("bnn_robustness peak fields disagree with within-sweep selection")


def _check_robustness_sweep(payload: Mapping[str, object]) -> None:
    """Bind the display sweep to matched trials and actual server presets."""
    divergences = payload["divergences"]
    rates = payload["rates"]
    if (
        not isinstance(divergences, list)
        or not divergences
        or any(not isinstance(item, str) for item in divergences)
    ):
        raise ReportSchemaError("robustness_sweep divergences must be a non-empty string list")
    if len(set(divergences)) != len(divergences) or "KLD" not in divergences:
        raise ReportSchemaError("robustness_sweep divergences must be unique and include KLD")
    rate_values = _number_list(rates, schema="robustness_sweep", field="rates")
    if not rate_values or len(set(rate_values)) != len(rate_values):
        raise ReportSchemaError("robustness_sweep rates must be non-empty and unique")
    presets = _mapping(
        payload["server_robustness_by_label"],
        schema="robustness_sweep",
        field="server_robustness_by_label",
    )
    if set(presets) != set(divergences):
        raise ReportSchemaError("robustness_sweep server presets must match divergence labels")
    for label, value in presets.items():
        preset = _number(
            value,
            schema="robustness_sweep",
            field=f"server_robustness_by_label.{label}",
        )
        if (label == "KLD" and preset != 0.0) or (label != "KLD" and preset <= 0.0):
            raise ReportSchemaError(
                "robustness_sweep server presets must expose naive zero and positive robust values"
            )
    if payload["headline_is_display_selection"] is not True:
        raise ReportSchemaError("robustness_sweep headline must remain disclosed as display selection")
    if "matched trial" not in str(payload["analysis_unit"]):
        raise ReportSchemaError("robustness_sweep analysis_unit must identify matched trials")
    if "true state" not in str(payload["trial_structure"]) or "attack target" not in str(
        payload["trial_structure"]
    ):
        raise ReportSchemaError("robustness_sweep trial_structure must identify fixed world variables")
    n_trials = payload["n_trials"]
    assert isinstance(n_trials, int) and not isinstance(n_trials, bool)
    profiles = _mapping(payload["per_rate_summary"], schema="robustness_sweep", field="per_rate_summary")
    expected_rate_keys = {f"{rate:g}" for rate in rate_values}
    if set(profiles) != expected_rate_keys:
        raise ReportSchemaError("robustness_sweep per_rate_summary must cover every declared rate")
    for rate_key, profile_value in profiles.items():
        profile = _mapping(profile_value, schema="robustness_sweep", field=f"per_rate_summary.{rate_key}")
        methods = _mapping(
            profile.get("methods"), schema="robustness_sweep", field=f"per_rate_summary.{rate_key}.methods"
        )
        if profile.get("n") != n_trials or set(methods) != set(divergences):
            raise ReportSchemaError(
                "robustness_sweep per-rate summaries must retain matched trial budgets and presets"
            )


def _check_bnn_torch(payload: Mapping[str, object]) -> None:
    if "status" not in payload:
        _raise_missing("bnn_torch", "status", "str")
    status = payload["status"]
    if not isinstance(status, str):
        _raise_type_error("bnn_torch", "status", "str", status)
    if status == "ok":
        _check(
            "bnn_torch",
            payload,
            SchemaDefinition(
                required={
                    "accuracy_by_config": "dict",
                    "beta": "number",
                    "consensus_max_simplex_deviation": "number",
                    "contamination_levels": "list",
                    "deterministic": "bool",
                    "hidden_dim": "int",
                    "n_clients": "int",
                    "n_steps": "int",
                    "reported_contamination": "number",
                    "robust_accuracy": "number",
                    "robustness": "number",
                    "seed": "int",
                    "standard_accuracy": "number",
                    "status": "str",
                    "torch_version": "str",
                }
            ),
        )
        return
    if status.startswith("skipped"):
        return
    raise ReportSchemaError("bnn_torch payload field 'status' must be 'ok' or start with 'skipped'")


def _check_figure_registry(payload: Mapping[str, object]) -> None:
    _check("figure_registry", payload, _FIGURE_REGISTRY_SCHEMA)
    if payload["schema_version"] != "1.2":
        raise ReportSchemaError("figure_registry schema_version must be '1.2'")
    if payload["generated_by"] != "analysis.workflow.run_analysis_pipeline":
        raise ReportSchemaError("figure_registry generated_by is not the canonical producer")
    figures = payload["figures"]
    assert isinstance(figures, list)
    if not figures:
        raise ReportSchemaError("figure_registry requires at least one figure")
    labels: list[str] = []
    filenames: list[str] = []
    generators: list[str] = []
    for index, entry in enumerate(figures):
        if not isinstance(entry, Mapping):
            raise ReportSchemaError(
                "figure_registry payload field 'figures' expected list of dict, "
                f"got {_describe_value(entry)} at index {index}"
            )
        label = entry.get("label")
        entry_name = (
            f"figure_registry figure {label!r}"
            if isinstance(label, str)
            else f"figure_registry figures[{index}]"
        )
        _check(entry_name, entry, _FIGURE_METADATA_SCHEMA)
        label = str(entry["label"])
        filename = str(entry["filename"])
        generator = str(entry["generated_by"])
        if not label.startswith("fig:"):
            raise ReportSchemaError(f"{entry_name} label must start with 'fig:'")
        if filename != f"{generator}.png":
            raise ReportSchemaError(
                f"{entry_name} filename must be the canonical PNG for generator {generator!r}"
            )
        if entry["path"] != f"output/figures/{filename}":
            raise ReportSchemaError(f"{entry_name} path does not match its filename")
        source_manuscript = str(entry["source_manuscript"])
        if not source_manuscript.startswith("manuscript/") or not source_manuscript.endswith(".md"):
            raise ReportSchemaError(f"{entry_name} has an unsafe source_manuscript path")
        if len(str(entry["alt_text"])) > 500:
            raise ReportSchemaError(f"{entry_name} alt_text exceeds 500 characters")
        if "long_description" in entry and not str(entry["long_description"]).strip():
            raise ReportSchemaError(f"{entry_name} long_description must be non-empty")
        if generator in COMPLEX_FIGURE_GENERATORS and not str(
            entry.get("long_description", "")
        ).strip():
            raise ReportSchemaError(
                f"{entry_name} complex figure requires a structured long_description"
            )
        fallback = entry.get("exact_value_fallback")
        if fallback is not None and not str(fallback).startswith("fig-values:"):
            raise ReportSchemaError(f"{entry_name} exact_value_fallback is malformed")
        labels.append(label)
        filenames.append(filename)
        generators.append(generator)
    if labels != sorted(labels):
        raise ReportSchemaError("figure_registry figures must be sorted by label")
    for field_name, values in (
        ("label", labels),
        ("filename", filenames),
        ("generated_by", generators),
    ):
        if len(values) != len(set(values)):
            raise ReportSchemaError(f"figure_registry contains duplicate {field_name} entries")
    exact_artifact = payload.get("exact_value_artifact")
    declared = sorted(
        str(entry["exact_value_fallback"])
        for entry in figures
        if isinstance(entry, Mapping) and entry.get("exact_value_fallback")
    )
    if declared and exact_artifact is None:
        raise ReportSchemaError(
            "figure_registry exact-value fallbacks require exact_value_artifact"
        )
    if exact_artifact is not None:
        assert isinstance(exact_artifact, Mapping)
        _check(
            "figure_registry exact_value_artifact",
            exact_artifact,
            SchemaDefinition(
                required={
                    "json_path": "str",
                    "markdown_path": "str",
                    "identifiers": "list",
                }
            ),
        )
        identifiers = exact_artifact["identifiers"]
        assert isinstance(identifiers, list)
        if exact_artifact["json_path"] != "output/figures/figure_exact_values.json":
            raise ReportSchemaError("figure_registry exact-value JSON path is not canonical")
        if exact_artifact["markdown_path"] != "output/figures/figure_exact_values.md":
            raise ReportSchemaError("figure_registry exact-value Markdown path is not canonical")
        if (
            any(not isinstance(item, str) or not item.startswith("fig-values:") for item in identifiers)
            or identifiers != sorted(set(identifiers))
        ):
            raise ReportSchemaError(
                "figure_registry exact_value_artifact identifiers must be sorted, unique fig-values:* strings"
            )
        if declared != identifiers:
            raise ReportSchemaError(
                "figure_registry exact-value identifiers do not match per-figure fallback declarations"
            )


def _check_figure_exact_values(payload: Mapping[str, object]) -> None:
    """Validate identifier-addressable tables used as non-visual fallbacks."""
    _check("figure_exact_values", payload, _FIGURE_EXACT_VALUES_SCHEMA)
    if payload["schema_version"] != "1.0":
        raise ReportSchemaError("figure_exact_values schema_version must be '1.0'")
    if payload["generated_by"] != "analysis.figure_exact_values.build_figure_exact_values":
        raise ReportSchemaError("figure_exact_values generated_by is not the canonical producer")
    tables = payload["tables"]
    assert isinstance(tables, list)
    if not tables:
        raise ReportSchemaError("figure_exact_values requires at least one table")
    identifiers: list[str] = []
    generators: list[str] = []
    for index, table in enumerate(tables):
        if not isinstance(table, Mapping):
            raise ReportSchemaError(f"figure_exact_values tables[{index}] must be a mapping")
        _check(
            f"figure_exact_values tables[{index}]",
            table,
            SchemaDefinition(
                required={
                    "identifier": "str",
                    "generator": "str",
                    "source_report": "str",
                    "columns": "list",
                    "rows": "list",
                    "note": "str",
                }
            ),
        )
        expected_table_fields = {
            "identifier",
            "generator",
            "source_report",
            "columns",
            "rows",
            "note",
        }
        if set(table) != expected_table_fields:
            raise ReportSchemaError(
                f"figure_exact_values tables[{index}] has unknown or missing fields"
            )
        identifier = str(table["identifier"])
        generator = str(table["generator"])
        expected_report = dict(EXACT_VALUE_REPORT_BINDINGS).get(generator)
        if identifier != f"fig-values:{generator.replace('_', '-')}":
            raise ReportSchemaError(
                f"figure_exact_values identifier/generator mismatch at tables[{index}]"
            )
        if expected_report is not None and table["source_report"] != f"{expected_report}.json":
            raise ReportSchemaError(
                f"figure_exact_values source_report mismatch at tables[{index}]"
            )
        if not str(table["note"]).strip():
            raise ReportSchemaError(f"figure_exact_values tables[{index}] note must be non-empty")
        columns = table["columns"]
        rows = table["rows"]
        assert isinstance(columns, list) and isinstance(rows, list)
        if (
            not columns
            or any(not isinstance(column, str) or not column for column in columns)
            or len(columns) != len(set(columns))
        ):
            raise ReportSchemaError(f"figure_exact_values tables[{index}] has invalid columns")
        if not rows:
            raise ReportSchemaError(f"figure_exact_values tables[{index}] has no rows")
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != set(columns):
                raise ReportSchemaError(
                    f"figure_exact_values tables[{index}].rows[{row_index}] does not match columns"
                )
            if any(
                value is not None and not isinstance(value, (str, int, float, bool))
                for value in row.values()
            ):
                raise ReportSchemaError(
                    f"figure_exact_values tables[{index}].rows[{row_index}] "
                    "contains a non-scalar value"
                )
        identifiers.append(identifier)
        generators.append(generator)
    if identifiers != sorted(set(identifiers)) or len(generators) != len(set(generators)):
        raise ReportSchemaError("figure_exact_values tables must be sorted and unique")
    expected_generators = set(EXACT_VALUE_GENERATORS)
    if set(generators) != expected_generators:
        raise ReportSchemaError(
            "figure_exact_values generator inventory mismatch: "
            f"missing={sorted(expected_generators - set(generators))}, "
            f"extra={sorted(set(generators) - expected_generators)}"
        )


def _review_mapping(value: object, *, field: str) -> Mapping[str, object]:
    """Return a nested review-grid mapping or raise a named schema error."""
    if not isinstance(value, Mapping):
        raise ReportSchemaError(f"robustness_review_grid field {field!r} must be a mapping")
    return value


def _review_list(value: object, *, field: str) -> list[object]:
    """Return a nested review-grid list or raise a named schema error."""
    if not isinstance(value, list):
        raise ReportSchemaError(f"robustness_review_grid field {field!r} must be a list")
    return value


def _review_number(value: object, *, field: str) -> float:
    """Validate a finite non-bool number in a review-grid payload."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportSchemaError(f"robustness_review_grid field {field!r} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ReportSchemaError(f"robustness_review_grid field {field!r} must be a finite number")
    return number


def _review_string_set(value: object, *, field: str) -> set[str]:
    """Validate a unique list of strings and return its set representation."""
    values = _review_list(value, field=field)
    if not values or any(not isinstance(item, str) for item in values):
        raise ReportSchemaError(f"robustness_review_grid field {field!r} must be a non-empty list of strings")
    result = {str(item) for item in values}
    if len(result) != len(values):
        raise ReportSchemaError(f"robustness_review_grid field {field!r} must not contain duplicate strings")
    return result


def _check_review_grid(payload: Mapping[str, object]) -> None:
    """Fail closed on nested data consumed by the review-grid public figure.

    A shallow top-level schema is insufficient here: a missing interval or a
    malformed method row once caused the figure to silently omit uncertainty.
    This validator binds the complete signed method × rate surface, its seed
    replication unit, and the observed (not merely declared) attack controls.
    """
    n_seeds = payload["n_seeds"]
    n_trials = payload["n_trials"]
    assert isinstance(n_seeds, int) and not isinstance(n_seeds, bool)
    assert isinstance(n_trials, int) and not isinstance(n_trials, bool)
    if n_seeds < 2 or n_trials < 2:
        raise ReportSchemaError("robustness_review_grid requires n_seeds and n_trials >= 2")

    divergences = _review_list(payload["divergences"], field="divergences")
    if any(not isinstance(method, str) for method in divergences) or "KLD" not in divergences:
        raise ReportSchemaError("robustness_review_grid divergences must be strings including 'KLD'")
    robust_methods = [str(method) for method in divergences if method != "KLD"]
    if not robust_methods or len(set(robust_methods)) != len(robust_methods):
        raise ReportSchemaError("robustness_review_grid must declare one or more unique non-KLD methods")

    rates_raw = _review_list(payload["rates"], field="rates")
    rates = [_review_number(rate, field="rates[]") for rate in rates_raw]
    if not rates or any(rate < 0.0 or rate > 1.0 for rate in rates):
        raise ReportSchemaError("robustness_review_grid rates must be non-empty and lie in [0, 1]")
    if rates != sorted(rates) or len(set(rates)) != len(rates):
        raise ReportSchemaError("robustness_review_grid rates must be strictly increasing")
    rate_keys = {f"{rate:g}" for rate in rates}
    directional = _review_string_set(payload["directional_mechanisms"], field="directional_mechanisms")
    declared_attacks = _review_string_set(payload["attack_mechanisms"], field="attack_mechanisms")

    conditional = _review_mapping(payload["conditional_world"], field="conditional_world")
    conditional_rows = _review_mapping(conditional.get("by_scenario"), field="conditional_world.by_scenario")
    if not conditional_rows:
        raise ReportSchemaError("robustness_review_grid conditional_world has no cells")
    conditional_attacks: set[str] = set()
    for scenario_id, row_value in conditional_rows.items():
        row = _review_mapping(row_value, field=f"conditional_world.by_scenario.{scenario_id}")
        attack = row.get("attack")
        if not isinstance(attack, str):
            raise ReportSchemaError(f"robustness_review_grid conditional cell {scenario_id!r} lacks attack")
        conditional_attacks.add(attack)
        if row.get("n_seeds") != n_seeds or row.get("n_trials") != n_trials:
            raise ReportSchemaError(
                f"robustness_review_grid conditional cell {scenario_id!r} has a mismatched budget"
            )
        contrast = _review_list(
            row.get("contrast_by_seed"),
            field=f"conditional_world.by_scenario.{scenario_id}.contrast_by_seed",
        )
        if len(contrast) != n_seeds:
            raise ReportSchemaError(
                f"robustness_review_grid conditional cell {scenario_id!r} has wrong seed count"
            )
        for index, value in enumerate(contrast):
            _review_number(
                value,
                field=f"conditional_world.by_scenario.{scenario_id}.contrast_by_seed[{index}]",
            )
        interval = _review_list(
            row.get("contrast_ci"),
            field=f"conditional_world.by_scenario.{scenario_id}.contrast_ci",
        )
        if len(interval) != 2:
            raise ReportSchemaError(f"robustness_review_grid conditional cell {scenario_id!r} has invalid CI")
        lo = _review_number(interval[0], field="conditional contrast CI lower")
        hi = _review_number(interval[1], field="conditional contrast CI upper")
        mean = _review_number(row.get("contrast_mean"), field="conditional contrast mean")
        if lo > mean or mean > hi:
            raise ReportSchemaError(
                f"robustness_review_grid conditional cell {scenario_id!r} has unordered CI"
            )

    profiles = _review_mapping(payload["rate_profiles"], field="rate_profiles")
    by_kind = _review_mapping(profiles.get("by_kind"), field="rate_profiles.by_kind")
    if set(by_kind) != directional:
        raise ReportSchemaError(
            "robustness_review_grid rate profile mechanisms differ from directional_mechanisms"
        )
    for mechanism, profile_value in by_kind.items():
        profile = _review_mapping(profile_value, field=f"rate_profiles.by_kind.{mechanism}")
        profile_rates = [
            _review_number(rate, field=f"rate_profiles.by_kind.{mechanism}.rates[]")
            for rate in _review_list(profile.get("rates"), field=f"rate_profiles.by_kind.{mechanism}.rates")
        ]
        if profile_rates != rates:
            raise ReportSchemaError(
                f"robustness_review_grid profile {mechanism!r} has rates inconsistent with top level"
            )

    statistics = _review_mapping(payload["statistics"], field="statistics")
    if statistics.get("selection_free") is not True:
        raise ReportSchemaError("robustness_review_grid statistics must declare selection_free=true")
    for alpha_field in ("fdr_alpha", "power_alpha"):
        alpha = _review_number(statistics.get(alpha_field), field=f"statistics.{alpha_field}")
        if not 0.0 < alpha < 1.0:
            raise ReportSchemaError(f"robustness_review_grid statistics {alpha_field} must lie in (0, 1)")
    planning_alternative = statistics.get("planning_alternative")
    if planning_alternative not in ("greater", "less", "two-sided"):
        raise ReportSchemaError("robustness_review_grid statistics planning_alternative is unsupported")
    by_mechanism = _review_mapping(statistics.get("by_mechanism"), field="statistics.by_mechanism")
    if set(by_mechanism) != directional:
        raise ReportSchemaError(
            "robustness_review_grid statistics mechanisms differ from directional_mechanisms"
        )
    observed_cell_mcses: list[float] = []
    for mechanism, mechanism_value in by_mechanism.items():
        mechanism_rows = _review_mapping(mechanism_value, field=f"statistics.by_mechanism.{mechanism}")
        methods = _review_list(
            mechanism_rows.get("methods"), field=f"statistics.by_mechanism.{mechanism}.methods"
        )
        if methods != robust_methods:
            raise ReportSchemaError(f"robustness_review_grid statistics methods differ for {mechanism!r}")
        rate_rows = _review_mapping(
            mechanism_rows.get("by_rate"),
            field=f"statistics.by_mechanism.{mechanism}.by_rate",
        )
        if set(rate_rows) != rate_keys:
            raise ReportSchemaError(f"robustness_review_grid statistics rate rows differ for {mechanism!r}")
        for rate_key, rate_row_value in rate_rows.items():
            rate_row = _review_mapping(
                rate_row_value,
                field=f"statistics.by_mechanism.{mechanism}.by_rate.{rate_key}",
            )
            if rate_row.get("n_seeds") != n_seeds:
                raise ReportSchemaError(
                    "robustness_review_grid statistics row has a mismatched seed count: "
                    f"{mechanism!r}/{rate_key!r}"
                )
            method_rows = _review_mapping(
                rate_row.get("methods"),
                field=f"statistics.by_mechanism.{mechanism}.by_rate.{rate_key}.methods",
            )
            if set(method_rows) != set(robust_methods):
                raise ReportSchemaError(
                    "robustness_review_grid statistics row has missing or unexpected methods: "
                    f"{mechanism!r}/{rate_key!r}"
                )
            for method, method_value in method_rows.items():
                method_row = _review_mapping(
                    method_value,
                    field=(f"statistics.by_mechanism.{mechanism}.by_rate.{rate_key}.methods.{method}"),
                )
                contrast = _review_list(
                    method_row.get("contrast_by_seed"),
                    field=f"statistics contrast_by_seed {mechanism}/{rate_key}/{method}",
                )
                if len(contrast) != n_seeds:
                    raise ReportSchemaError(
                        "robustness_review_grid method contrast has a wrong seed count: "
                        f"{mechanism!r}/{rate_key!r}/{method!r}"
                    )
                for index, value in enumerate(contrast):
                    _review_number(
                        value,
                        field=(f"statistics contrast_by_seed {mechanism}/{rate_key}/{method}[{index}]"),
                    )
                summary = _review_mapping(
                    method_row.get("summary"),
                    field=f"statistics summary {mechanism}/{rate_key}/{method}",
                )
                if summary.get("n") != n_seeds:
                    raise ReportSchemaError(
                        "robustness_review_grid summary has a mismatched seed count: "
                        f"{mechanism!r}/{rate_key!r}/{method!r}"
                    )
                summary_mean = _review_number(summary.get("mean"), field="statistics summary mean")
                summary_lo = _review_number(summary.get("ci_lo"), field="statistics summary ci_lo")
                summary_hi = _review_number(summary.get("ci_hi"), field="statistics summary ci_hi")
                mcse = _review_number(summary.get("mcse"), field="statistics summary mcse")
                mde = _review_number(summary.get("mde"), field="statistics summary mde")
                if summary_lo > summary_mean or summary_mean > summary_hi or mcse < 0 or mde < 0:
                    raise ReportSchemaError(
                        "robustness_review_grid summary has invalid uncertainty ordering: "
                        f"{mechanism!r}/{rate_key!r}/{method!r}"
                    )
                observed_cell_mcses.append(mcse)
                interval = _review_list(
                    method_row.get("contrast_ci"),
                    field=f"statistics contrast_ci {mechanism}/{rate_key}/{method}",
                )
                if len(interval) != 2:
                    raise ReportSchemaError(
                        "robustness_review_grid method interval must have exactly two values: "
                        f"{mechanism!r}/{rate_key!r}/{method!r}"
                    )
                interval_lo = _review_number(interval[0], field="statistics contrast_ci lower")
                interval_hi = _review_number(interval[1], field="statistics contrast_ci upper")
                if not (
                    math.isclose(interval_lo, summary_lo, abs_tol=1e-12)
                    and math.isclose(interval_hi, summary_hi, abs_tol=1e-12)
                ):
                    raise ReportSchemaError(
                        "robustness_review_grid method interval disagrees with its summary: "
                        f"{mechanism!r}/{rate_key!r}/{method!r}"
                    )

    controls = _review_mapping(payload["controls"], field="controls")
    observed = _review_string_set(
        controls.get("observed_attack_mechanisms"), field="controls.observed_attack_mechanisms"
    )
    if observed != declared_attacks or observed != conditional_attacks | directional:
        raise ReportSchemaError(
            "robustness_review_grid observed attack controls do not match the emitted components"
        )
    if controls.get("all_declared_attack_controls_present") is not True:
        raise ReportSchemaError("robustness_review_grid controls must prove declared attack coverage")
    if controls.get("conditional_zero_robustness_control_passed") is not True:
        raise ReportSchemaError("robustness_review_grid controls must retain the zero-robustness check")

    precision = _review_mapping(payload["precision_plan"], field="precision_plan")
    observed_mcse = _review_number(
        precision.get("observed_max_mcse"), field="precision_plan.observed_max_mcse"
    )
    if observed_mcse < 0.0:
        raise ReportSchemaError("robustness_review_grid precision plan must report a non-negative MCSE")
    expected_signed_cells = len(directional) * len(rates) * len(robust_methods)
    signed_cells = precision.get("n_signed_method_rate_cells")
    if (
        isinstance(signed_cells, bool)
        or not isinstance(signed_cells, int)
        or signed_cells != expected_signed_cells
    ):
        raise ReportSchemaError(
            "robustness_review_grid precision plan has a mismatched signed method-rate cell count"
        )
    if not observed_cell_mcses or not math.isclose(
        observed_mcse,
        max(observed_cell_mcses),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ReportSchemaError(
            "robustness_review_grid precision plan maximum MCSE disagrees with signed method-rate summaries"
        )
    target = precision.get("target_max_mcse")
    target_status = precision.get("target_status")
    target_met = precision.get("target_met")
    if target is not None:
        target_value = _review_number(target, field="precision_plan.target_max_mcse")
        if (
            target_value <= 0.0
            or target_status != "met"
            or target_met is not True
            or observed_mcse > target_value
        ):
            raise ReportSchemaError("robustness_review_grid precision plan has an unmet MCSE target")
    elif target_status != "not_evaluated" or target_met is not None:
        raise ReportSchemaError(
            "robustness_review_grid precision plan must mark an absent target as not_evaluated"
        )


def _record_list(value: object, *, schema: str, field: str) -> list[Mapping[str, object]]:
    """Return a list of mappings with an exact field-specific diagnostic."""
    if not isinstance(value, list):
        raise ReportSchemaError(f"{schema} payload field {field!r} must be a list")
    records: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ReportSchemaError(f"{schema} payload field {field}[{index}] must be a mapping")
        records.append(item)
    return records


def _record_ids(
    records: list[Mapping[str, object]],
    *,
    schema: str,
    field: str,
) -> list[str]:
    """Return unique, non-empty string ids from a record list."""
    identifiers: list[str] = []
    for index, record in enumerate(records):
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ReportSchemaError(f"{schema} payload field {field}[{index}].id must be a non-empty string")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ReportSchemaError(f"{schema} payload field {field!r} contains duplicate ids")
    return identifiers


def _check_edges(
    records: list[Mapping[str, object]],
    *,
    schema: str,
    field: str,
    valid_nodes: set[str],
    allowed_dispositions: set[str],
) -> None:
    """Validate directed edge records against a declared node inventory."""
    seen: set[tuple[str, str]] = set()
    for index, edge in enumerate(records):
        source = edge.get("source")
        target = edge.get("target")
        label = edge.get("label")
        disposition = edge.get("disposition")
        if source not in valid_nodes or target not in valid_nodes:
            raise ReportSchemaError(f"{schema} payload field {field}[{index}] references an unknown node")
        if not isinstance(label, str) or not label:
            raise ReportSchemaError(f"{schema} payload field {field}[{index}].label must be non-empty")
        if disposition not in allowed_dispositions:
            raise ReportSchemaError(f"{schema} payload field {field}[{index}] has invalid disposition")
        # A directed dependency is identified by its endpoints.  Treating the
        # human-readable label as part of the identity would allow the same
        # producer or invalidation relation to be duplicated under two labels.
        identity = (str(source), str(target))
        if identity in seen:
            raise ReportSchemaError(f"{schema} payload field {field!r} contains duplicate edges")
        seen.add(identity)


def _check_nonempty_string_fields(
    record: Mapping[str, object],
    *,
    schema: str,
    field: str,
    names: tuple[str, ...],
) -> None:
    """Require the listed record fields to contain non-empty strings."""
    for name in names:
        value = record.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ReportSchemaError(f"{schema} payload field {field}.{name} must be a non-empty string")


def _check_application_integrity_flow(payload: Mapping[str, object]) -> None:
    """Bind the application flow to implemented status and verification states."""
    schema = "application_integrity_flow"
    if payload["study_status"] != "source_owned_explanatory_contract":
        raise ReportSchemaError(f"{schema} study_status must identify an explanatory contract")
    panels = _record_list(payload["panels"], schema=schema, field="panels")
    panel_ids = _record_ids(panels, schema=schema, field="panels")
    if panel_ids != ["validation", "solver_health", "receipt"]:
        raise ReportSchemaError(f"{schema} panels must preserve validation, solver, receipt order")
    global_nodes: set[str] = set()
    edges: list[Mapping[str, object]] = []
    edge_inventory: list[tuple[str, str, str, str]] = []
    for index, panel in enumerate(panels):
        panel_id = str(panel["id"])
        _check_nonempty_string_fields(
            panel,
            schema=schema,
            field=f"panels[{index}]",
            names=("label",),
        )
        nodes = _record_list(panel.get("nodes"), schema=schema, field=f"panels[{index}].nodes")
        node_ids = _record_ids(nodes, schema=schema, field=f"panels[{index}].nodes")
        expected_node_ids = dict(APPLICATION_PANEL_NODE_INVENTORY)[panel_id]
        if node_ids != list(expected_node_ids):
            raise ReportSchemaError(
                f"{schema} panel {panel_id!r} node inventory or order is invalid"
            )
        overlap = global_nodes & set(node_ids)
        if overlap:
            raise ReportSchemaError(f"{schema} nodes are duplicated across panels: {sorted(overlap)}")
        global_nodes.update(node_ids)
        for node_index, node in enumerate(nodes):
            _check_nonempty_string_fields(
                node,
                schema=schema,
                field=f"panels[{index}].nodes[{node_index}]",
                names=("label", "role"),
            )
        panel_edges = _record_list(
            panel.get("edges"), schema=schema, field=f"panels[{index}].edges"
        )
        edges.extend(panel_edges)
        edge_inventory.extend(
            (
                panel_id,
                str(edge.get("source")),
                str(edge.get("target")),
                str(edge.get("disposition")),
            )
            for edge in panel_edges
        )
        if panel_id == "receipt":
            write_order = panel.get("write_order")
            if not isinstance(write_order, list) or tuple(write_order) != APPLICATION_ARTIFACT_WRITE_ORDER:
                raise ReportSchemaError(
                    f"{schema} receipt artifact write order is invalid"
                )
        elif "write_order" in panel:
            raise ReportSchemaError(
                f"{schema} write_order belongs only to the receipt panel"
            )
    _check_edges(
        edges,
        schema=schema,
        field="panels[].edges",
        valid_nodes=global_nodes,
        allowed_dispositions={"normal", "rejected", "retained_warning", "data_dependency"},
    )
    if tuple(edge_inventory) != APPLICATION_FLOW_EDGE_INVENTORY:
        raise ReportSchemaError(f"{schema} edge inventory, panel ownership, or order is invalid")
    statuses = _record_list(payload["solver_statuses"], schema=schema, field="solver_statuses")
    status_ids = _record_ids(statuses, schema=schema, field="solver_statuses")
    expected_status = {
        "nominal": (True, False),
        "converged_with_fallback": (True, True),
        "not_converged": (False, False),
        "not_converged_with_fallback": (False, True),
    }
    if status_ids != list(expected_status):
        raise ReportSchemaError(f"{schema} solver status order or inventory is invalid")
    for status in statuses:
        identifier = str(status["id"])
        if (
            status.get("label") != identifier
            or (status.get("converged"), status.get("fallbacks")) != expected_status[identifier]
        ):
            raise ReportSchemaError(f"{schema} solver state {identifier!r} has invalid semantics")
    levels = _record_list(payload["verification_levels"], schema=schema, field="verification_levels")
    if _record_ids(levels, schema=schema, field="verification_levels") != [
        "artifact_integrity",
        "source_equivalence",
        "nominal_solver",
    ]:
        raise ReportSchemaError(f"{schema} verification levels are incomplete or reordered")
    for index, level in enumerate(levels):
        _check_nonempty_string_fields(
            level,
            schema=schema,
            field=f"verification_levels[{index}]",
            names=("label", "establishes", "does_not_establish"),
        )
    no_claims = payload["no_claims"]
    if not isinstance(no_claims, list) or set(no_claims) != {
        "calibration",
        "domain suitability",
        "scientific validity",
        "downstream decision correctness",
        "acceptance",
    }:
        raise ReportSchemaError(f"{schema} must retain the complete receipt no-claim boundary")


def _check_evidence_replication_map(payload: Mapping[str, object]) -> None:
    """Keep evidence classes and independent-unit ownership in separate lanes."""
    schema = "evidence_replication_map"
    if payload["study_status"] != "source_owned_evidence_classification":
        raise ReportSchemaError(f"{schema} study_status must identify a source-owned classification")
    classes = _record_list(payload["evidence_classes"], schema=schema, field="evidence_classes")
    class_ids = _record_ids(classes, schema=schema, field="evidence_classes")
    expected_classes = [
        "formal_executable",
        "source_conditional",
        "conditional_empirical",
        "scoped_implementation",
        "open",
    ]
    if class_ids != expected_classes:
        raise ReportSchemaError(f"{schema} evidence-class inventory or order is invalid")
    for index, evidence_class in enumerate(classes):
        _check_nonempty_string_fields(
            evidence_class,
            schema=schema,
            field=f"evidence_classes[{index}]",
            names=("label", "meaning"),
        )
    lanes = _record_list(payload["lanes"], schema=schema, field="lanes")
    lane_ids = _record_ids(lanes, schema=schema, field="lanes")
    expected_lane_classes = {
        "project_identity": "formal_executable",
        "client_loss": "source_conditional",
        "belief_sharing_free_energy": "conditional_empirical",
        "likelihood_learning": "conditional_empirical",
        "bmr_sign_control": "formal_executable",
        "client_loss_baseline": "conditional_empirical",
        "server_heuristic": "conditional_empirical",
        "variational_server": "formal_executable",
        "moving_disjoint_fov": "conditional_empirical",
        "hierarchy_sensitivity": "conditional_empirical",
        "parameter_recovery": "conditional_empirical",
        "protocol_integrity": "scoped_implementation",
        "application_provenance": "scoped_implementation",
        "external_confirmation": "open",
    }
    if lane_ids != list(expected_lane_classes):
        raise ReportSchemaError(f"{schema} claim-owner lanes are incomplete or reordered")
    for index, lane in enumerate(lanes):
        _check_nonempty_string_fields(
            lane,
            schema=schema,
            field=f"lanes[{index}]",
            names=(
                "headline",
                "evidence_class",
                "estimand",
                "unit",
                "replication_unit",
                "nesting",
                "permitted_interpretation",
                "prohibited_generalization",
            ),
        )
        if lane["evidence_class"] not in set(class_ids):
            raise ReportSchemaError(f"{schema} lane references an unknown evidence class")
        expected_class = expected_lane_classes[str(lane["id"])]
        if lane["evidence_class"] != expected_class:
            raise ReportSchemaError(
                f"{schema} lane {lane['id']!r} must retain evidence class {expected_class!r}"
            )
        display = _mapping(lane.get("display"), schema=schema, field=f"lanes[{index}].display")
        if set(display) != {
            "estimand_unit",
            "replication",
            "nesting",
            "permitted",
            "prohibited",
        }:
            raise ReportSchemaError(f"{schema} lane display summary has an invalid field set")
        _check_nonempty_string_fields(
            display,
            schema=schema,
            field=f"lanes[{index}].display",
            names=("estimand_unit", "replication", "nesting", "permitted", "prohibited"),
        )
    nesting = _record_list(payload["nesting"], schema=schema, field="nesting")
    nesting_ids = _record_ids(nesting, schema=schema, field="nesting")
    if nesting_ids != ["study", "seed", "trial", "agent", "step"]:
        raise ReportSchemaError(f"{schema} nesting inventory or order is invalid")
    _check_edges(
        _record_list(payload["nesting_edges"], schema=schema, field="nesting_edges"),
        schema=schema,
        field="nesting_edges",
        valid_nodes=set(nesting_ids),
        allowed_dispositions={"nesting"},
    )
    no_claims = payload["no_claims"]
    if (
        not isinstance(no_claims, list)
        or len(no_claims) < 3
        or any(not isinstance(item, str) or not item for item in no_claims)
    ):
        raise ReportSchemaError(f"{schema} must retain explicit non-transfer boundaries")


def _edge_inventory(records: list[Mapping[str, object]]) -> tuple[tuple[str, str, str], ...]:
    """Return ordered endpoint/disposition triples for an exact flow inventory."""

    return tuple(
        (str(edge["source"]), str(edge["target"]), str(edge["disposition"]))
        for edge in records
    )


def _check_source_render_provenance(payload: Mapping[str, object]) -> None:
    """Validate producer order, reverse invalidation, and authorization gates."""
    schema = "source_render_provenance"
    if payload["study_status"] != "source_owned_pipeline_contract":
        raise ReportSchemaError(f"{schema} study_status must identify a pipeline contract")
    inputs = _record_list(payload["inputs"], schema=schema, field="inputs")
    stages = _record_list(payload["stages"], schema=schema, field="stages")
    terminals = _record_list(payload["terminals"], schema=schema, field="terminals")
    input_ids = _record_ids(inputs, schema=schema, field="inputs")
    stage_ids = _record_ids(stages, schema=schema, field="stages")
    terminal_ids = _record_ids(terminals, schema=schema, field="terminals")
    if input_ids != list(SOURCE_RENDER_INPUT_INVENTORY):
        raise ReportSchemaError(f"{schema} input inventory or order is invalid")
    if stage_ids != list(SOURCE_RENDER_STAGE_INVENTORY):
        raise ReportSchemaError(f"{schema} producer-stage inventory or order is invalid")
    if terminal_ids != ["github", "zenodo"]:
        raise ReportSchemaError(f"{schema} terminal inventory or order is invalid")
    for index, terminal in enumerate(terminals):
        _check_nonempty_string_fields(
            terminal,
            schema=schema,
            field=f"terminals[{index}]",
            names=("label", "role"),
        )
        if terminal.get("authorization_required") is not True:
            raise ReportSchemaError(f"{schema} publication terminals must require authorization")
    all_nodes = set(input_ids) | set(stage_ids) | set(terminal_ids)
    producer_edges = _record_list(payload["producer_edges"], schema=schema, field="producer_edges")
    _check_edges(
        producer_edges,
        schema=schema,
        field="producer_edges",
        valid_nodes=all_nodes,
        allowed_dispositions={"producer", "gate", "receipt", "authorization"},
    )
    if _edge_inventory(producer_edges) != SOURCE_RENDER_PRODUCER_EDGE_INVENTORY:
        raise ReportSchemaError(
            f"{schema} producer edge inventory, order, or disposition is invalid"
        )
    producer_pairs = {(edge["source"], edge["target"]) for edge in producer_edges}
    if ("release_manifest", "github") not in producer_pairs or ("github", "zenodo") not in producer_pairs:
        raise ReportSchemaError(f"{schema} must retain both publication authorization gates")
    invalidation_edges = _record_list(
        payload["invalidation_edges"], schema=schema, field="invalidation_edges"
    )
    _check_edges(
        invalidation_edges,
        schema=schema,
        field="invalidation_edges",
        valid_nodes=all_nodes,
        allowed_dispositions={"invalidation"},
    )
    if _edge_inventory(invalidation_edges) != SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY:
        raise ReportSchemaError(
            f"{schema} invalidation edge inventory, order, or disposition is invalid"
        )
    producer_pairs = {
        (str(edge["source"]), str(edge["target"])) for edge in producer_edges
    }
    for index, edge in enumerate(invalidation_edges):
        pair = (str(edge["source"]), str(edge["target"]))
        if pair not in producer_pairs:
            raise ReportSchemaError(
                f"{schema} payload field invalidation_edges[{index}] must name an "
                "immediate producer dependency"
            )
    authorization = payload["authorization_boundary"]
    cycle_boundary = payload["manifest_cycle_boundary"]
    assert isinstance(authorization, str) and isinstance(cycle_boundary, str)
    if "do not automatically authorize" not in authorization:
        raise ReportSchemaError(f"{schema} must deny automatic publication authorization")
    if "never reads" not in cycle_boundary or "release manifest" not in cycle_boundary:
        raise ReportSchemaError(f"{schema} must deny a generated-manifest dependency cycle")
    no_claims = payload["no_claims"]
    if (
        not isinstance(no_claims, list)
        or len(no_claims) < 4
        or any(not isinstance(item, str) or not item for item in no_claims)
    ):
        raise ReportSchemaError(f"{schema} must retain engineering and accessibility no-claims")


def _check_sensitivity(payload: Mapping[str, object]) -> None:
    """Keep both displayed sensitivity grids aligned with their declared axes."""
    schema = "sensitivity"
    acuity = _number_list(payload["acuity_values"], schema=schema, field="acuity_values")
    raw_agents = payload["n_agents_values"]
    if (
        not isinstance(raw_agents, list)
        or not raw_agents
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in raw_agents)
    ):
        raise ReportSchemaError(f"{schema} n_agents_values must contain positive integers")
    if not acuity or any(value <= 0.0 or value > 1.0 for value in acuity):
        raise ReportSchemaError(f"{schema} acuity_values must lie in (0, 1]")
    if acuity != sorted(set(acuity)):
        raise ReportSchemaError(f"{schema} acuity_values must be strictly increasing and unique")
    if raw_agents != sorted(set(raw_agents)):
        raise ReportSchemaError(
            f"{schema} n_agents_values must be strictly increasing and unique"
        )
    noise_floor = _number(payload["noise_floor"], schema=schema, field="noise_floor")
    if noise_floor < 0.0 or noise_floor > 1.0:
        raise ReportSchemaError(f"{schema} noise_floor must lie in [0, 1]")
    n_trials = payload["n_trials"]
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials <= 0:
        raise ReportSchemaError(f"{schema} n_trials must be positive")

    for panel_name in ("belief_sharing", "hierarchical"):
        nested = payload[panel_name]
        assert isinstance(nested, Mapping)
        if nested.get("acuity_values") != payload["acuity_values"]:
            raise ReportSchemaError(
                f"{schema} {panel_name} acuity axis does not match the top-level axis"
            )
        if nested.get("n_agents_values") != raw_agents:
            raise ReportSchemaError(
                f"{schema} {panel_name} colony-size axis does not match the top-level axis"
            )
        if nested.get("seed") != payload["seed"] or nested.get("n_trials") != n_trials:
            raise ReportSchemaError(
                f"{schema} {panel_name} execution identity does not match the report"
            )
        raw_grid = nested.get("accuracy_gap_grid")
        if not isinstance(raw_grid, list) or len(raw_grid) != len(acuity):
            raise ReportSchemaError(
                f"{schema} {panel_name} accuracy_gap_grid has the wrong row count"
            )
        for row_index, row in enumerate(raw_grid):
            if (
                not isinstance(row, list)
                or len(row) != len(raw_agents)
                or any(not _is_number(value) for value in row)
            ):
                raise ReportSchemaError(
                    f"{schema} {panel_name} accuracy_gap_grid row {row_index} "
                    "does not match the colony axis"
                )
            if any(abs(float(value)) > 1.0 for value in row):
                raise ReportSchemaError(
                    f"{schema} {panel_name} accuracy_gap_grid values must lie in [-1, 1]"
                )
    estimands = payload["estimands"]
    if not isinstance(estimands, Mapping) or set(estimands) != {"belief_sharing", "hierarchical"}:
        raise ReportSchemaError(f"{schema} estimands must name both displayed panels")
    if payload["unit"] != "fraction":
        raise ReportSchemaError(f"{schema} unit must be 'fraction'")
    if "not a confidence interval" not in str(payload["claim_boundary"]):
        raise ReportSchemaError(f"{schema} claim_boundary must explain the hatch semantics")


def validate_report(schema: str, payload: Mapping[str, object]) -> None:
    """Validate one report or figure-registry payload before it is written."""

    if not isinstance(payload, Mapping):
        raise ReportSchemaError(f"{schema} payload must be a mapping")
    _check_finite_json(payload)
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
