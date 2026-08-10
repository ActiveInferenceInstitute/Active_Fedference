"""Continuous 1-D Gaussian generalized-Bayes recovery limits (MAJ-3) — no mocks.

Mirrors the discrete recovery contract in the continuous setting:

1. beta = 0 recovers the conjugate Normal-Normal posterior EXACTLY (bit-level).
2. Off the corner (beta = 1e-1 .. 1e-4, genuine values, not a code branch) the
   robust-vs-conjugate gap shrinks monotonically toward zero — real numerical
   convergence, not an equality flag.
3. Negative control / robustness witness: with a genuine outlier the robust
   posterior down-weights it (posterior mean stays near the clean data while the
   conjugate posterior is dragged toward the outlier).
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.continuous_recovery import (
    conjugate_gaussian_posterior,
    recovery_residuals,
    robust_gaussian_posterior,
)

_PRIOR_MEAN = 0.0
_PRIOR_VAR = 10.0
_OBS_VAR = 0.25
_CLEAN = np.array([0.9, 1.1, 1.0, 0.95, 1.05])


def test_beta_zero_recovers_conjugate_posterior_exactly() -> None:
    ref = conjugate_gaussian_posterior(_PRIOR_MEAN, _PRIOR_VAR, _CLEAN, _OBS_VAR)
    rob = robust_gaussian_posterior(_PRIOR_MEAN, _PRIOR_VAR, _CLEAN, _OBS_VAR, beta=0.0)
    # Bit-identical: the beta=0 branch runs the exact conjugate formula.
    assert rob["mean"] == ref["mean"]
    assert rob["var"] == ref["var"]
    assert rob["iterations"] == 0


def test_off_corner_gap_shrinks_monotonically_to_zero() -> None:
    res = recovery_residuals(_PRIOR_MEAN, _PRIOR_VAR, _CLEAN, _OBS_VAR,
                             betas=(1e-1, 1e-2, 1e-3, 1e-4))
    mean_gap = res["mean_gap"]
    kl_gap = res["kl_gap"]
    # Genuine convergence: each smaller beta gives a strictly smaller gap.
    assert all(mean_gap[i] > mean_gap[i + 1] for i in range(len(mean_gap) - 1))
    assert all(kl_gap[i] > kl_gap[i + 1] for i in range(len(kl_gap) - 1))
    # And the gaps are nonzero off the corner (so beta=0 is not vacuously tested).
    assert mean_gap[0] > 0.0
    # O(beta) convergence: a 10x smaller beta gives ~10x smaller gap.
    assert mean_gap[0] / mean_gap[-1] > 100.0


def test_conjugate_posterior_precision_adds() -> None:
    ref = conjugate_gaussian_posterior(_PRIOR_MEAN, _PRIOR_VAR, _CLEAN, _OBS_VAR)
    n = _CLEAN.size
    expected_prec = 1.0 / _PRIOR_VAR + n / _OBS_VAR
    assert abs(1.0 / ref["var"] - expected_prec) < 1e-12
    # Posterior mean lies between prior mean and sample mean.
    assert _PRIOR_MEAN < ref["mean"] < float(_CLEAN.mean()) + 1e-9


def test_robust_posterior_downweights_a_genuine_outlier() -> None:
    obs = np.array([1.0, 1.0, 1.0, 1.0, 8.0])  # last observation is a gross outlier
    conj = conjugate_gaussian_posterior(_PRIOR_MEAN, _PRIOR_VAR, obs, _OBS_VAR)
    rob = robust_gaussian_posterior(_PRIOR_MEAN, _PRIOR_VAR, obs, _OBS_VAR, beta=0.5)
    # The conjugate (equal-weight) posterior is dragged well above the clean value.
    assert conj["mean"] > 2.0
    # The robust posterior stays near the clean cluster and crushes the outlier.
    assert abs(rob["mean"] - 1.0) < 0.1
    assert rob["weights"][-1] < 1e-3
    assert rob["converged"] is True


def test_rejects_nonpositive_variance() -> None:
    for bad in (0.0, -1.0):
        try:
            conjugate_gaussian_posterior(0.0, bad, _CLEAN, _OBS_VAR)
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "variance" in str(exc)


def test_robust_posterior_input_guards() -> None:
    with pytest.raises(ValueError, match="variance"):
        robust_gaussian_posterior(0.0, -1.0, _CLEAN, _OBS_VAR, beta=0.5)
    with pytest.raises(ValueError, match="beta must be non-negative"):
        robust_gaussian_posterior(0.0, _PRIOR_VAR, _CLEAN, _OBS_VAR, beta=-0.1)
    with pytest.raises(ValueError, match="non-empty"):
        robust_gaussian_posterior(0.0, _PRIOR_VAR, np.array([]), _OBS_VAR, beta=0.5)
    with pytest.raises(ValueError, match="non-empty"):
        conjugate_gaussian_posterior(0.0, _PRIOR_VAR, np.array([]), _OBS_VAR)
