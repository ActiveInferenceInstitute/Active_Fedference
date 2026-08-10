"""Continuous-state (1-D Gaussian) generalized-Bayes recovery limits (MAJ-3 slice).

The discrete-categorical core proves that robust generalized Bayes recovers
standard Bayes in a named limit (``beta -> 0``, ``alpha -> 1``). This module
carries the SAME limit-as-proof contract into the continuous setting for the
canonical conjugate problem — inferring a Gaussian mean with known observation
variance — using the density-power (``beta``) robust weighting.

* :func:`conjugate_gaussian_posterior` — the standard closed-form Normal-Normal
  update (the reference the robust rule must recover).
* :func:`robust_gaussian_posterior` — the density-power robust update: each
  observation is weighted by its density under the current estimate raised to
  ``beta``, iterated to a fixed point. At ``beta = 0`` every weight is
  ``density^0 = 1`` and the update is bit-identical to the conjugate posterior;
  at small ``beta > 0`` the fixed point departs by ``O(beta)`` and the residual
  shrinks monotonically toward zero — an exact-at-the-corner, convergent-off-it
  recovery mirroring the discrete off-switch witness.

Scope (kept honest): this is the recovery-limit slice only. It does NOT deliver
continuous-state active inference, a Gaussian belief-sharing colony, or any
robustness-superiority claim — only the conjugate recovery limit and the
density-power down-weighting of a genuine outlier.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .divergences import gaussian_kl

ArrayF = np.ndarray


def conjugate_gaussian_posterior(
    prior_mean: float, prior_var: float, obs: ArrayF, obs_var: float
) -> dict[str, float]:
    """Closed-form Normal-Normal posterior for a mean with known ``obs_var``.

    Precision adds: ``1/post_var = 1/prior_var + n/obs_var`` and the posterior
    mean is the precision-weighted average of the prior mean and the sample mean.
    """
    if prior_var <= 0.0 or obs_var <= 0.0:
        raise ValueError("variances must be strictly positive")
    x = np.asarray(obs, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        raise ValueError("obs must be non-empty")
    post_prec = 1.0 / prior_var + n / obs_var
    post_mean = (prior_mean / prior_var + x.sum() / obs_var) / post_prec
    return {"mean": float(post_mean), "var": float(1.0 / post_prec)}


def _gaussian_density(x: ArrayF, mu: float, var: float) -> ArrayF:
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def robust_gaussian_posterior(
    prior_mean: float,
    prior_var: float,
    obs: ArrayF,
    obs_var: float,
    *,
    beta: float = 0.0,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> dict[str, Any]:
    """Density-power (beta) robust Normal-Normal posterior via weighted fixed point.

    Each observation ``x_i`` receives weight ``w_i = density(x_i)^beta`` under the
    current estimate; the mean update is the ``w``-weighted conjugate formula.
    ``beta = 0`` gives ``w_i = 1`` and returns the conjugate posterior exactly (no
    iteration). For ``beta > 0`` the fixed point down-weights low-density
    (outlying) observations.

    Returns ``mean``, ``var``, ``weights`` (final per-obs, normalized to mean 1),
    ``iterations`` and ``converged``.
    """
    if prior_var <= 0.0 or obs_var <= 0.0:
        raise ValueError("variances must be strictly positive")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    x = np.asarray(obs, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        raise ValueError("obs must be non-empty")

    if beta == 0.0:
        base = conjugate_gaussian_posterior(prior_mean, prior_var, x, obs_var)
        return {"mean": base["mean"], "var": base["var"],
                "weights": np.ones(n), "iterations": 0, "converged": True}

    mu = conjugate_gaussian_posterior(prior_mean, prior_var, x, obs_var)["mean"]
    converged = False
    it = 0
    weights = np.ones(n)
    for it in range(1, max_iter + 1):
        raw = _gaussian_density(x, mu, obs_var) ** beta
        weights = raw / raw.mean()  # normalize to mean 1 so beta->0 is the n-obs limit
        eff_n = float(weights.sum())
        post_prec = 1.0 / prior_var + eff_n / obs_var
        mu_new = (prior_mean / prior_var + (weights * x).sum() / obs_var) / post_prec
        if abs(mu_new - mu) < tol:
            mu = mu_new
            converged = True
            break
        mu = mu_new
    eff_n = float(weights.sum())
    post_var = 1.0 / (1.0 / prior_var + eff_n / obs_var)
    return {"mean": float(mu), "var": float(post_var), "weights": weights,
            "iterations": int(it), "converged": converged}


def recovery_residuals(
    prior_mean: float,
    prior_var: float,
    obs: ArrayF,
    obs_var: float,
    *,
    betas: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4),
) -> dict[str, Any]:
    """Off-corner recovery witness: the robust-vs-conjugate posterior gap as a
    function of ``beta``.

    For each ``beta`` we measure ``|mean_robust - mean_conjugate|`` and the
    symmetric ``KL`` between the two posterior Gaussians. Both shrink toward zero
    as ``beta -> 0`` (monotone in this grid), demonstrating genuine numerical
    convergence to the conjugate limit — not merely a ``beta == 0`` code branch.
    """
    ref = conjugate_gaussian_posterior(prior_mean, prior_var, obs, obs_var)
    mean_gaps: list[float] = []
    kl_gaps: list[float] = []
    for beta in betas:
        rob = robust_gaussian_posterior(prior_mean, prior_var, obs, obs_var, beta=beta)
        mean_gaps.append(abs(rob["mean"] - ref["mean"]))
        kl = 0.5 * (
            gaussian_kl(rob["mean"], rob["var"], ref["mean"], ref["var"])
            + gaussian_kl(ref["mean"], ref["var"], rob["mean"], rob["var"])
        )
        kl_gaps.append(float(kl))
    return {
        "betas": list(betas),
        "mean_gap": mean_gaps,
        "kl_gap": kl_gaps,
        "conjugate": ref,
    }


__all__ = [
    "conjugate_gaussian_posterior",
    "recovery_residuals",
    "robust_gaussian_posterior",
]
