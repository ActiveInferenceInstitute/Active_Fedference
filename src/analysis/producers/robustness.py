"""Existing robustness report production; no persistence or workflow imports.

Friston et al. (2024) and Mildner et al. (2025) study meanings and configured
seed/budget controls are retained from the canonical analysis workflow.
"""

from __future__ import annotations

from experiment_config import ExperimentConfig
from fedference.experiments import (
    run_contamination_gallery,
    run_influence_weights_report,
    run_review_grid,
    run_robustness_onset,
    run_robustness_sweep,
    run_variational_aggregation_report,
)


def _robustness_report(config: ExperimentConfig) -> dict:
    """Robustness sweep at the first seed, using the configured rates/divergences.

    ``run_robustness_sweep`` does not echo back ``n_agents`` / ``n_contaminated``,
    so we record the colony shape we asked for alongside its output (the
    manuscript prose reports both).
    """
    seed = config.seeds[0]
    n_contaminated = max(1, config.n_agents // 3)
    report = run_robustness_sweep(
        seed,
        rates=config.contamination_rates,
        divergences=config.divergences,
        n_agents=config.n_agents,
        n_contaminated=n_contaminated,
        n_trials=config.n_trials,
        fdr_alpha=config.fdr_alpha,
        power_alpha=config.power_alpha,
        power_alternative=config.power_alternative,
        target_power=config.target_power,
    )
    report["n_agents"] = config.n_agents
    report["n_contaminated"] = n_contaminated
    return report


def _influence_weights(config: ExperimentConfig) -> dict:
    """Server-side robust pooling influence weights on a contaminated colony."""
    return run_influence_weights_report(config.seeds[0], n_agents=config.n_agents)


def _contamination_gallery_report(config: ExperimentConfig, *, smoke: bool = False) -> dict:
    """Descriptive pooled-method gallery across the declared mechanisms.

    Runs :func:`fedference.experiments.run_contamination_gallery` over
    confident-wrong, byzantine, drift, uniform, and label-noise mechanisms. The
    gallery chooses a pooled display method from these same seed-level results,
    so it is not the selection-free inferential surface and cannot establish a
    mechanism-wide or universal robustness conclusion. The review grid retains
    every configured method and owns the all-method signed comparisons.
    """
    n_contaminated = max(1, config.n_agents // 3)
    # The frozen config owns every stochastic budget. Smoke uses the same real
    # code path with explicitly smaller units for temporary-project tests.
    return run_contamination_gallery(
        config.seeds[0],
        n_agents=config.n_agents,
        n_contaminated=n_contaminated,
        divergences=config.divergences,
        n_trials=min(config.gallery_n_trials, 4) if smoke else config.gallery_n_trials,
        n_seeds=min(config.gallery_n_seeds, 2) if smoke else config.gallery_n_seeds,
    )


def _robustness_onset_report(config: ExperimentConfig, *, smoke: bool = False) -> dict:
    """Per-mechanism robustness-onset curves for the pooled display contrast.

    Maps the rate dependence the single-strength gallery cannot: for each
    directional mechanism, the naive and pooled-display accuracy curves over a rate
    grid plus the onset rate. Its seed and nested-trial budgets are frozen in
    :class:`ExperimentConfig`, so the publication profile cannot silently use
    the former hard-coded small grid.
    """
    n_contaminated = max(1, config.n_agents // 3)
    return run_robustness_onset(
        config.seeds[0],
        n_agents=config.n_agents,
        n_contaminated=n_contaminated,
        divergences=config.divergences,
        n_trials=min(config.onset_n_trials, 4) if smoke else config.onset_n_trials,
        n_seeds=min(config.onset_n_seeds, 2) if smoke else config.onset_n_seeds,
    )


def _variational_aggregation_report(config: ExperimentConfig) -> dict:
    """Diagnostics for the objective-backed variational aggregator."""
    return run_variational_aggregation_report(config.seeds[0], n_agents=config.n_agents)


def _review_grid_report(config: ExperimentConfig, *, smoke: bool = False) -> dict:
    """Run the bounded source-bound review grid with explicit seed nesting."""
    return run_review_grid(
        config.seeds[0],
        n_seeds=min(config.review_grid_n_seeds, 4) if smoke else config.review_grid_n_seeds,
        n_trials=min(config.review_grid_n_trials, 4) if smoke else config.review_grid_n_trials,
        n_agents=config.n_agents,
        robustness=1.5,
        rates=config.review_grid_rates,
        divergences=config.divergences,
        fdr_alpha=config.fdr_alpha,
        power_alpha=config.power_alpha,
        planning_alternative=config.power_alternative,
        # Smoke verifies the same producer but intentionally does not claim to
        # meet the publication precision stopping rule.
        target_max_mcse=None if smoke else config.review_grid_target_max_mcse,
    )
