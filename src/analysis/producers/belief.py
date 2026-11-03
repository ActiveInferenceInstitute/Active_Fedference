"""Existing belief report production; no persistence or workflow imports.

Friston et al. (2024) and Mildner et al. (2025) study meanings and configured
seed/budget controls are retained from the canonical analysis workflow.
"""

from __future__ import annotations

import numpy as np

from experiment_config import ExperimentConfig
from fedference.aggregation import log_linear_pool
from fedference.belief_updating import infer_states
from fedference.experiments import (
    run_belief_sharing,
    run_efe_decomposition_report,
    run_emergence,
    summarize_language_acquisition,
)
from fedference.pomdp import N_LOCATIONS, build_sentinel_world
from fedference.statistics import bootstrap_ci


def _belief_sharing_report(config: ExperimentConfig) -> dict:
    """Aggregate per-seed belief-sharing free energies (communicating vs not)."""
    communicating: list[float] = []
    incommunicado: list[float] = []
    for seed in config.seeds:
        comm = run_belief_sharing(seed, communicate=True, n_agents=config.n_agents)
        incom = run_belief_sharing(seed, communicate=False, n_agents=config.n_agents)
        communicating.append(float(comm["mean_free_energy"]))
        incommunicado.append(float(incom["mean_free_energy"]))
    comm_mean = float(np.mean(communicating))
    incom_mean = float(np.mean(incommunicado))
    paired = [float(incom - comm) for incom, comm in zip(incommunicado, communicating)]
    paired_mean = float(np.mean(paired))
    paired_ci = bootstrap_ci(
        np.asarray(paired, dtype=np.float64),
        alpha=0.05,
        n_boot=10_000,
        rng=np.random.default_rng(config.seeds[0] + 5_000_003),
    )
    return {
        "schema_version": "1.0",
        "communicating_free_energy": communicating,
        "incommunicado_free_energy": incommunicado,
        "communicating_mean": comm_mean,
        "incommunicado_mean": incom_mean,
        "paired_free_energy_difference": paired,
        "paired_difference_mean": paired_mean,
        "paired_difference_ci": [float(paired_ci[0]), float(paired_ci[1])],
        "difference_definition": "incommunicado_minus_communicating",
        "analysis_unit": "configured seed with within-seed paired conditions",
        "replication_unit": "configured seed",
        "interval_method": "percentile_bootstrap_of_paired_seed_differences",
        "interval_percent": 95,
        "free_energy_gap": paired_mean,
        "communication_helps": bool(comm_mean < incom_mean),
        "n_agents": config.n_agents,
        "n_seeds": config.n_seeds,
    }


def _language_report(config: ExperimentConfig) -> dict:
    """Seed-level language-acquisition trajectory summary for publication."""
    return summarize_language_acquisition(config.seeds)


def _emergence_report(config: ExperimentConfig) -> dict:
    """Bayesian-model-reduction emergence contrast at the first configured seed."""
    seed = config.seeds[0]
    result = run_emergence(seed)
    return dict(result)


def _efe_terms(config: ExperimentConfig) -> dict:
    """Closed-form EFE decomposition of one sentinel policy."""
    return run_efe_decomposition_report(config.seeds[0])


def _heatmap_inputs(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build a seeded colony of posteriors plus their consensus for the heatmap."""
    rng = np.random.default_rng(config.seeds[0])
    world = build_sentinel_world(rng, acuity=0.6)
    A = np.asarray(world["A"][0], dtype=np.float64)  # type: ignore[index]
    n_s = int(N_LOCATIONS)
    true_state = int(rng.integers(0, n_s))
    log_prior = np.log(np.full(n_s, 1.0 / n_s))
    local_posteriors = np.empty((config.n_agents, n_s), dtype=np.float64)
    for n in range(config.n_agents):
        column = A[:, true_state]
        o = int(rng.choice(n_s, p=column / column.sum()))
        local_posteriors[n] = infer_states(A, o, log_prior)
    consensus = log_linear_pool(local_posteriors=local_posteriors)
    return local_posteriors, consensus
