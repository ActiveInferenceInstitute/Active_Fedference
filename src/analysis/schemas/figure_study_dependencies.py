"""Figure study dependencies report contracts."""

from __future__ import annotations

from analysis.schemas.primitives import FigureDependencyContract

_FIGURE_STUDY_DEPENDENCIES: dict[str, tuple[FigureDependencyContract, ...]] = {
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
}
