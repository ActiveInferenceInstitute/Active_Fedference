"""Figure world dependencies report contracts."""

from __future__ import annotations

from analysis.schemas.primitives import FigureDependencyContract

_FIGURE_WORLD_DEPENDENCIES: dict[str, tuple[FigureDependencyContract, ...]] = {
    "conditional_world": (
        FigureDependencyContract(
            report_name="conditional_world",
            required_fields={"by_scenario": "dict", "controls": "dict"},
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
