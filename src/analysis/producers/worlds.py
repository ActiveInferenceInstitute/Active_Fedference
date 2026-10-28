"""Existing worlds report production; no persistence or workflow imports.

Friston et al. (2024) and Mildner et al. (2025) study meanings and configured
seed/budget controls are retained from the canonical analysis workflow.
"""

from __future__ import annotations

from experiment_config import ExperimentConfig
from fedference.experiments import (
    moving_world_report,
    run_conditional_world_generalization,
    run_parameter_recovery,
)


def _moving_world_report(config: ExperimentConfig) -> dict:
    """Moving-world experiment: isolated vs communicating vs EFE-guided (V4)."""
    seed = config.seeds[0]
    return moving_world_report(
        seed,
        n_trials=config.n_trials,
        n_multiseed=config.replicate_seeds,
    )


def _parameter_recovery_report(config: ExperimentConfig, *, smoke: bool = False) -> dict:
    """Measure finite-grid acuity recovery for the configured estimator (Study 9)."""
    seed = config.seeds[0]
    return dict(
        run_parameter_recovery(
            seed,
            n_trials=config.n_trials,
            n_observations=24 if smoke else 200,
            fit_resolution=20 if smoke else 80,
        )
    )


def _conditional_world_report(config: ExperimentConfig) -> dict:
    """Conditional world/attack geometry grid with seed-level contrasts."""
    return run_conditional_world_generalization(
        config.seeds[0],
        n_seeds=config.conditional_world_n_seeds,
        n_trials=config.conditional_world_n_trials,
        n_agents=config.n_agents,
        robustness=1.5,
    )
