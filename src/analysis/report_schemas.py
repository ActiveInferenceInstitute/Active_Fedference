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

TypeTag = Literal["bool", "dict", "int", "list", "number", "str"]


class BeliefSharingReport(TypedDict):
    """Top-level payload written to ``belief_sharing.json``."""

    communicating_free_energy: list[float]
    communicating_mean: float
    communication_helps: bool
    free_energy_gap: float
    incommunicado_free_energy: list[float]
    incommunicado_mean: float
    n_agents: int
    n_seeds: int


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
    degenerate: dict[str, object]
    degenerate_recommends_prune_top: bool
    degenerate_top_surprise: float
    informative: dict[str, object]
    informative_keeps_top: bool
    informative_top_surprise: float
    n_levels: int
    obs: int


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
    ci_percent: int
    contamination_levels: list[float]
    n_bootstrap: int
    n_per: int
    n_seeds: int
    peak_margin: NotRequired[float]
    peak_margin_contamination: NotRequired[float]
    robust_loss_param: float
    robust_minus_standard: list[float]
    seed: int


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
    interval_method: str
    interval_percent: int
    mean_abs_error: NotRequired[float]
    n_observations: NotRequired[int]
    n_trials: NotRequired[int]
    r_squared: NotRequired[float]
    recovered_acuity: list[float]
    recovered_acuity_ci_hi: list[float]
    recovered_acuity_ci_lo: list[float]
    seed: int
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


class FigureRegistryPayload(TypedDict):
    """Top-level figure registry payload."""

    schema_version: str
    generated_by: str
    figures: list[FigureMetadataEntry]


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
            "communicating_free_energy": "list",
            "communicating_mean": "number",
            "communication_helps": "bool",
            "free_energy_gap": "number",
            "incommunicado_free_energy": "list",
            "incommunicado_mean": "number",
            "n_agents": "int",
            "n_seeds": "int",
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
            "degenerate": "dict",
            "degenerate_recommends_prune_top": "bool",
            "degenerate_top_surprise": "number",
            "informative": "dict",
            "informative_keeps_top": "bool",
            "informative_top_surprise": "number",
            "n_levels": "int",
            "obs": "int",
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
            "ci_percent": "int",
            "contamination_levels": "list",
            "n_bootstrap": "int",
            "n_per": "int",
            "n_seeds": "int",
            "robust_loss_param": "number",
            "robust_minus_standard": "list",
            "seed": "int",
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
            "interval_method": "str",
            "interval_percent": "int",
            "recovered_acuity": "list",
            "recovered_acuity_ci_hi": "list",
            "recovered_acuity_ci_lo": "list",
            "seed": "int",
            "true_acuity": "list",
        },
        optional={
            "mean_abs_error": "number",
            "n_observations": "int",
            "n_trials": "int",
            "r_squared": "number",
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
}

# Versioned payloads are intentionally fail-closed. Compatibility keys may be
# retained inside a supported payload, but a reader must never guess how an
# unsupported report version maps onto the canonical fields.
_SUPPORTED_REPORT_SCHEMA_VERSIONS: dict[str, frozenset[str]] = {
    "heuristic_characterization": frozenset({"2.0"}),
    "robustness_sweep": frozenset({"2.0"}),
    "robust_influence_weights": frozenset({"2.0"}),
    "robustness_review_grid": frozenset({"1.1"}),
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
    }
)

_FIGURE_REGISTRY_SCHEMA = SchemaDefinition(
    required={
        "schema_version": "str",
        "generated_by": "str",
        "figures": "list",
    }
)

FIGURE_DEPENDENCY_CONTRACTS: dict[str, tuple[FigureDependencyContract, ...]] = {
    "free_energy_comparison": (
        FigureDependencyContract(
            report_name="belief_sharing",
            required_fields={
                "incommunicado_free_energy": "list",
                "communicating_free_energy": "list",
            },
        ),
    ),
    "robustness_sweep": (
        FigureDependencyContract(
            report_name="robustness_sweep",
            required_fields={"accuracy_by_method_and_rate": "dict"},
            optional_fields={
                "accuracy_threshold": "number",
                "per_rate_summary": "dict",
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
                "contamination_levels": "list",
            },
            optional_fields={"accuracy_ci_by_config": "dict"},
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
            },
            optional_fields={
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
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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
    figures = payload["figures"]
    assert isinstance(figures, list)
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
            raise ReportSchemaError(
                f"robustness_review_grid statistics {alpha_field} must lie in (0, 1)"
            )
    planning_alternative = statistics.get("planning_alternative")
    if planning_alternative not in ("greater", "less", "two-sided"):
        raise ReportSchemaError(
            "robustness_review_grid statistics planning_alternative is unsupported"
        )
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


def validate_report(schema: str, payload: Mapping[str, object]) -> None:
    """Validate one report or figure-registry payload before it is written."""

    if not isinstance(payload, Mapping):
        raise ReportSchemaError(f"{schema} payload must be a mapping")
    _check_finite_json(payload)
    if schema == "figure_registry":
        _check_figure_registry(payload)
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
    "BnnRobustnessReport",
    "BnnTorchOkReport",
    "BnnTorchSkippedReport",
    "ContaminationGalleryCell",
    "ContaminationGalleryReport",
    "CrossStudySummaryReport",
    "DisjointFovWorldReport",
    "EfeDecompositionReport",
    "EmergenceReport",
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
    "VariationalAggregationReport",
    "BeliefSharingReport",
    "check_figure_contract",
    "validate_report",
]
