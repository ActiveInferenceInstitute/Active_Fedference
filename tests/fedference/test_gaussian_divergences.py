"""Continuous-state (1-D Gaussian) divergence bridge (no mocks).

OUT OF SCOPE for the categorical federated experiments — these closed forms
exist only to show the KL/Renyi family carries over to the Gaussian beliefs a
continuous-state active-inference extension would use. The load-bearing checks
mirror the categorical ones: KL is zero iff the Gaussians coincide, the analytic
two-Gaussian KL matches a hand value, and Renyi -> KL as alpha -> 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.divergences import gaussian_alpha_renyi, gaussian_kl, gaussian_renyi


def test_gaussian_kl_zero_for_identical():
    assert gaussian_kl(1.5, 2.0, 1.5, 2.0) == pytest.approx(0.0, abs=1e-12)


def test_gaussian_kl_matches_analytic_value():
    # KL(N(0,1) || N(1,4)) = 0.5*(1/4 + 1/4 - 1 + log 4)
    expected = 0.5 * (0.25 + 0.25 - 1.0 + np.log(4.0))
    assert gaussian_kl(0.0, 1.0, 1.0, 4.0) == pytest.approx(expected, abs=1e-12)


def test_gaussian_kl_is_nonnegative():
    assert gaussian_kl(0.0, 1.0, 3.0, 0.5) > 0.0
    assert gaussian_kl(-2.0, 3.0, 2.0, 1.0) > 0.0


def test_gaussian_kl_asymmetric():
    fwd = gaussian_kl(0.0, 1.0, 1.0, 4.0)
    rev = gaussian_kl(1.0, 4.0, 0.0, 1.0)
    assert fwd != pytest.approx(rev)


def test_gaussian_renyi_recovers_kl_as_alpha_to_one():
    kl = gaussian_kl(0.0, 1.0, 1.0, 2.0)
    near = gaussian_renyi(0.0, 1.0, 1.0, 2.0, alpha=1.0 + 1e-5)
    assert near == pytest.approx(kl, rel=1e-3)
    # inside the stability band the closed form is exactly KL
    assert gaussian_renyi(0.0, 1.0, 1.0, 2.0, alpha=1.0) == pytest.approx(kl, abs=1e-12)


def test_gaussian_renyi_zero_for_identical():
    assert gaussian_renyi(2.0, 1.5, 2.0, 1.5, alpha=0.5) == pytest.approx(0.0, abs=1e-12)


def test_gaussian_renyi_positive_for_distinct():
    assert gaussian_renyi(0.0, 1.0, 2.0, 1.0, alpha=0.5) > 0.0


def test_gaussian_alpha_renyi_has_fedgvi_normalization_and_kl_limit():
    standard = gaussian_renyi(0.0, 1.0, 2.0, 1.0, alpha=0.5)
    assert gaussian_alpha_renyi(0.0, 1.0, 2.0, 1.0, alpha=0.5) == pytest.approx(
        standard / 0.5
    )
    assert gaussian_alpha_renyi(0.0, 1.0, 2.0, 1.0, alpha=1.0) == pytest.approx(
        gaussian_kl(0.0, 1.0, 2.0, 1.0)
    )


def test_gaussian_invalid_variance_raises():
    with pytest.raises(ValueError, match="variances must be strictly positive"):
        gaussian_kl(0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="variances must be strictly positive"):
        gaussian_renyi(0.0, -1.0, 1.0, 1.0, alpha=0.5)


def test_gaussian_renyi_nonpositive_alpha_raises():
    with pytest.raises(ValueError, match="alpha must be positive"):
        gaussian_renyi(0.0, 1.0, 1.0, 1.0, alpha=-0.5)


def test_gaussian_renyi_divergent_interpolated_variance_raises():
    # alpha > 1 with var_p < var_q can drive var_alpha <= 0 (the Renyi integral
    # diverges); the closed form raises rather than returning a bogus number.
    with pytest.raises(ValueError, match="interpolated variance"):
        gaussian_renyi(0.0, 10.0, 5.0, 0.1, alpha=3.0)
