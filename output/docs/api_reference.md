# Active Fedference API Reference

This document provides API reference for the Active Fedference project's
federated active inference modules.

## Core packages

### `fedference.divergences`

Statistical divergences over categorical distributions for FedGVI:
- `kl_divergence(q, p)` — KL divergence, the naive baseline (Friston Eq. 7 limit).
- `reverse_kl(q, p)` — Reverse KL, FedGVI RKL client divergence.
- `renyi_divergence(q, p, alpha)` — standard Rényi; `alpha -> 1` recovers KL.
- `alpha_renyi_divergence(q, p, alpha)` — FedGVI Alpha-Rényi normalization.
- `total_variation(q, p)` — Total variation distance.

### `fedference.losses`

Loss functions for generalized Bayesian inference:
- `nll(q, outcome)` — Negative log likelihood (standard Bayes).
- `beta_loss(q, outcome, beta)` — Beta-loss; `beta -> 0` recovers NLL.
- `rcce(q, outcome, q_reg)` — Robust categorical cross-entropy.

### `fedference.aggregation`

Belief aggregation primitives for federated consensus:
- `log_linear_pool(local_posteriors, base_weights=None)` — Product-of-experts = Friston Eq. 7.
- `robust_aggregate(local_posteriors, robustness, base_weights=None)` — Divergence-reweighted pooling.
- `variational_aggregate(local_posteriors, robustness, base_weights=None)` —
  Objective-backed conservative rule.
- `aggregation_free_energy(consensus_posterior, raw_effective_weights,`
  `local_posteriors, robustness)` — Variational free energy.

### `fedference.experiments`

Reproducible study harness:
- `run_belief_sharing(seed)` — categorical source-mechanism analogue related to Friston Fig. 5.
- `run_language_acquisition(seed)` — categorical language-learning trajectory related to Friston Fig. 7.
- `run_emergence(seed)` — categorical BMR diagnostic related to Friston Fig. 9.
- `run_robustness_sweep(seed)` — Contamination robustness sweep.
- `run_contamination_gallery(seed)` — Multi-mechanism contamination comparison.

See the full docstrings in each module for parameter details.
