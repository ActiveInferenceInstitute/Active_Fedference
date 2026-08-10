"""Active Fedference — public API surface.

Re-exports the FedGVI core (divergences, robust losses, generalized Bayes,
federated aggregation) and the belief-sharing bridge from the
:mod:`fedference` domain package. Active-inference, ensemble, experiment,
statistics and baseline modules are imported from their submodules on demand.
"""

from __future__ import annotations

from fedference import (  # noqa: F401
    AggregationResult,
    SharingDiagnostics,
    aggregate,
    alpha_renyi_divergence,
    beta_loss,
    cavity,
    divergence,
    friston_belief_share,
    generalized_posterior,
    kl_divergence,
    log_linear_pool,
    loss_vector,
    nll,
    rcce,
    renyi_divergence,
    reverse_kl,
    robust_aggregate,
    share_round,
    softmax,
    total_variation,
    update_factor,
)

__all__ = [
    "kl_divergence", "reverse_kl", "renyi_divergence", "alpha_renyi_divergence",
    "total_variation", "divergence",
    "nll", "rcce", "beta_loss", "loss_vector",
    "generalized_posterior", "cavity", "update_factor", "softmax",
    "log_linear_pool", "friston_belief_share", "robust_aggregate", "aggregate",
    "AggregationResult", "share_round", "SharingDiagnostics",
]
