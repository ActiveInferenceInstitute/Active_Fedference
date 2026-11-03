"""Existing diagnostics report production; no persistence or workflow imports.

Friston et al. (2024) and Mildner et al. (2025) study meanings and configured
seed/budget controls are retained from the canonical analysis workflow.
"""

from __future__ import annotations

from experiment_config import (
    DEFAULT_SENSITIVITY_ACUITY,
    DEFAULT_SENSITIVITY_COLONY_SIZES,
    SENSITIVITY_NOISE_FLOOR,
    ExperimentConfig,
)
from fedference.experiments import (
    run_belief_quality_sensitivity,
    run_belief_sharing_sensitivity,
    run_hierarchical_sensitivity,
)


def _sensitivity_report(*, seed: int, n_trials: int) -> dict[str, object]:
    """Return the two configured sensitivity grids consumed by Figure 8."""
    acuity_values = tuple(float(value) for value in DEFAULT_SENSITIVITY_ACUITY)
    n_agents_values = tuple(int(value) for value in DEFAULT_SENSITIVITY_COLONY_SIZES)
    belief_sharing = run_belief_sharing_sensitivity(
        seed,
        acuity_values=acuity_values,
        n_agents_values=n_agents_values,
        n_trials=n_trials,
    )
    hierarchical = run_hierarchical_sensitivity(
        seed,
        acuity_values=acuity_values,
        n_agents_values=n_agents_values,
        n_trials=n_trials,
    )
    return {
        "schema_version": "1.0",
        "seed": seed,
        "n_trials": n_trials,
        "acuity_values": list(acuity_values),
        "n_agents_values": list(n_agents_values),
        "noise_floor": float(SENSITIVITY_NOISE_FLOOR),
        "belief_sharing": belief_sharing,
        "hierarchical": hierarchical,
        "estimands": {
            "belief_sharing": "communicating minus isolated mean accuracy",
            "hierarchical": "hierarchical minus flat location accuracy",
        },
        "unit": "fraction",
        "analysis_unit": "configured acuity-by-colony-size cell",
        "replication_unit": "synthetic trial nested within configured cell",
        "uncertainty": "none displayed; each cell is a mean over nested trials",
        "claim_boundary": (
            "hatching denotes only the configured display band; it is not a confidence interval, "
            "significance test, unreliability flag, or proof of zero effect"
        ),
    }


def _belief_quality_report(config: ExperimentConfig) -> dict:
    """Proper-score and reliability diagnostics on the conditional subset."""
    return run_belief_quality_sensitivity(
        config.seeds[0],
        n_seeds=config.conditional_world_n_seeds,
        n_trials=config.conditional_world_n_trials,
        n_agents=config.n_agents,
        robustness=1.5,
    )
