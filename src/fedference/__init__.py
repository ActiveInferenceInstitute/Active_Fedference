"""Active Fedference — robust federated active inference.

A discrete-categorical reimplementation of FedGVI (Federated Generalized
Variational Inference; Mildner et al., 2025, arXiv:2502.00846) applied to the
federated belief-sharing scenario of Friston et al. (2024), *Federated
inference and belief sharing* (Neurosci. Biobehav. Rev. 156:105500).

The public surface is the FedGVI core (divergences, losses, generalized Bayes,
aggregation) and the belief-sharing bridge. Active-inference machinery (POMDP,
Dirichlet learning, expected free energy, Bayesian model reduction), the agent
ensemble, the experiment harness, statistics and the classification baselines are imported
lazily by callers from their submodules to keep the core import light.
"""

from __future__ import annotations

from .aggregation import (
    AggregationConfig,
    AggregationMethod,
    AggregationResult,
    AggregatorProtocol,
    aggregate,
    aggregate_result,
    aggregation_free_energy,
    friston_belief_share,
    log_linear_pool,
    robust_aggregate,
    variational_aggregate,
)
from .bayesian_model_reduction import greedy_reduce, hierarchical_reduce
from .belief_sharing import SharingDiagnostics, share_round
from .continuous_recovery import (
    conjugate_gaussian_posterior,
    recovery_residuals,
    robust_gaussian_posterior,
)
from .dirichlet_learning import DirichletLearningResult, learn_likelihood
from .divergences import (
    alpha_renyi_divergence,
    divergence,
    gaussian_alpha_renyi,
    kl_divergence,
    renyi_divergence,
    reverse_kl,
    total_variation,
)
from .evidence import DatasetSpec, ExperimentSpec, RunReceipt
from .generalized_bayes import (
    cavity,
    generalized_posterior,
    softmax,
    update_factor,
)
from .hybrid import HybridAggregationResult, HybridBelief, hybrid_aggregate, hybrid_log_linear_pool
from .losses import beta_loss, loss_vector, nll, rcce
from .statistics import (
    bh_fdr,
    bootstrap_ci,
    d_equivalent_from_rank_biserial,
    minimum_detectable_effect,
    multiseed_summary,
    paired_test,
    power_analysis,
    summary_statistics,
)

__all__ = [
    "hierarchical_reduce",
    "conjugate_gaussian_posterior",
    "robust_gaussian_posterior",
    "recovery_residuals",
    # divergences
    "kl_divergence",
    "reverse_kl",
    "renyi_divergence",
    "alpha_renyi_divergence",
    "gaussian_alpha_renyi",
    "total_variation",
    "divergence",
    # losses
    "nll",
    "rcce",
    "beta_loss",
    "loss_vector",
    # generalized Bayes
    "generalized_posterior",
    "cavity",
    "update_factor",
    "softmax",
    # aggregation
    "log_linear_pool",
    "friston_belief_share",
    "robust_aggregate",
    "aggregate",
    "aggregate_result",
    "variational_aggregate",
    "aggregation_free_energy",
    "AggregationConfig",
    "AggregationMethod",
    "AggregationResult",
    "AggregatorProtocol",
    # belief sharing
    "share_round",
    "SharingDiagnostics",
    # evidence contracts
    "DatasetSpec",
    "ExperimentSpec",
    "RunReceipt",
    "HybridAggregationResult",
    "HybridBelief",
    "hybrid_aggregate",
    "hybrid_log_linear_pool",
    # Dirichlet learning
    "DirichletLearningResult",
    "learn_likelihood",
    # Bayesian model reduction
    "greedy_reduce",
    # statistics
    "paired_test",
    "bh_fdr",
    "bootstrap_ci",
    "power_analysis",
    "multiseed_summary",
    "minimum_detectable_effect",
    "summary_statistics",
    "d_equivalent_from_rank_biserial",
]
