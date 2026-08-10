"""Conjugate Dirichlet likelihood-learning tests (Friston et al., 2024, Eqs. 9-12).

No mocks: every assertion is a real seeded computation on small categorical
likelihood matrices with explicit numeric expectations.

ISC-17: KL(target || learned) is monotone non-increasing and drops below 1e-2.
ISC-18: with the eta forgetting hyperprior, the total Dirichlet count mass
        saturates at eta.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.dirichlet_learning import (
    CONVERGENCE_KL_ATOL,
    DEFAULT_COUNT_SCALE,
    DEFAULT_PRIOR_CONCENTRATION,
    DirichletLearningResult,
    _as_likelihood,
    _kl_columns,
    expected_likelihood,
    learn_likelihood,
)

# A small, non-degenerate true likelihood (n_o=3 outcomes, n_s=2 states).
TRUE_A = np.array(
    [
        [0.80, 0.10],
        [0.15, 0.20],
        [0.05, 0.70],
    ]
)


# ---- expected_likelihood -------------------------------------------------

def test_expected_likelihood_column_normalizes():
    a = np.array([[2.0, 1.0], [2.0, 3.0]])
    e = expected_likelihood(a)
    assert e == pytest.approx(np.array([[0.5, 0.25], [0.5, 0.75]]))
    assert np.allclose(e.sum(axis=0), 1.0)


def test_expected_likelihood_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        expected_likelihood(np.array([1.0, 2.0, 3.0]))


def test_expected_likelihood_rejects_empty_column():
    with pytest.raises(ValueError, match="positive mass"):
        expected_likelihood(np.array([[0.0], [0.0]]))


# ---- _as_likelihood guards ----------------------------------------------

def test_as_likelihood_renormalises_columns():
    raw = np.array([[8.0, 1.0], [2.0, 9.0]])
    norm = _as_likelihood(raw)
    assert norm == pytest.approx(np.array([[0.8, 0.1], [0.2, 0.9]]))


def test_as_likelihood_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        _as_likelihood(np.array([0.5, 0.5]))


def test_as_likelihood_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        _as_likelihood(np.empty((0, 0)))


def test_as_likelihood_rejects_negative():
    with pytest.raises(ValueError, match="negative"):
        _as_likelihood(np.array([[-0.5], [1.5]]))


def test_as_likelihood_rejects_zero_column():
    with pytest.raises(ValueError, match="positive mass"):
        _as_likelihood(np.array([[0.0, 0.5], [0.0, 0.5]]))


# ---- _kl_columns ---------------------------------------------------------

def test_kl_columns_zero_for_identical():
    assert _kl_columns(TRUE_A, _as_likelihood(TRUE_A)) == pytest.approx(0.0, abs=1e-12)


def test_kl_columns_positive_for_mismatch():
    flat = np.full_like(TRUE_A, 1.0 / TRUE_A.shape[0])
    assert _kl_columns(_as_likelihood(TRUE_A), flat) > 0.0


# ---- ISC-17: monotone convergence to < 1e-2 ------------------------------

def test_isc17_kl_monotone_decreasing_to_below_tol_deterministic():
    res = learn_likelihood(TRUE_A, num_steps=40)
    traj = res.kl_trajectory
    # Monotone non-increasing.
    assert res.is_monotone_decreasing
    for earlier, later in zip(traj, traj[1:]):
        assert later <= earlier + 1e-12
    # Starts above tolerance, ends below it.
    assert traj[0] > CONVERGENCE_KL_ATOL
    assert res.final_kl < CONVERGENCE_KL_ATOL
    assert res.steps_to_converge < len(traj)
    # Learned likelihood recovers the true matrix.
    assert res.expected_a == pytest.approx(_as_likelihood(TRUE_A), abs=2e-2)


def test_isc17_holds_with_seeded_rng_jitter():
    rng = np.random.default_rng(20240624)
    res = learn_likelihood(TRUE_A, num_steps=60, rng=rng)
    # Jittered counts still drive KL below tolerance.
    assert res.final_kl < CONVERGENCE_KL_ATOL
    # Final trajectory tail is much smaller than the prior KL.
    assert res.final_kl < res.initial_kl


def test_seeded_rng_is_reproducible():
    a = learn_likelihood(TRUE_A, num_steps=12, rng=np.random.default_rng(7))
    b = learn_likelihood(TRUE_A, num_steps=12, rng=np.random.default_rng(7))
    assert a.kl_trajectory == b.kl_trajectory
    assert np.array_equal(a.expected_a, b.expected_a)


def test_default_run_uses_documented_defaults():
    res = learn_likelihood(TRUE_A, num_steps=5)
    assert res.count_scale == DEFAULT_COUNT_SCALE
    assert res.prior_concentration == DEFAULT_PRIOR_CONCENTRATION
    assert res.eta is None
    assert res.num_obs == 3
    assert res.num_states == 2
    assert len(res.kl_trajectory) == 6  # prior + one point per batch


# ---- ISC-18: eta forgetting hyperprior saturates total counts ------------

def test_isc18_total_counts_saturate_at_eta():
    eta = 50.0
    res = learn_likelihood(TRUE_A, num_steps=30, count_scale=10.0, eta=eta)
    # After the first eta-decayed update the total is pinned to eta and stays.
    assert res.total_count == pytest.approx(eta, abs=1e-6)
    assert res.eta == eta
    # The mass does not run away: every recorded total stays bounded by eta.
    assert max(res.concentration_totals) <= eta + 1e-6


def test_isc18_eta_total_is_fixed_point_across_horizons():
    eta = 80.0
    short = learn_likelihood(TRUE_A, num_steps=3, eta=eta)
    long = learn_likelihood(TRUE_A, num_steps=25, eta=eta)
    assert short.total_count == pytest.approx(eta, abs=1e-6)
    assert long.total_count == pytest.approx(eta, abs=1e-6)


def test_eta_run_still_converges_in_kl():
    eta = 200.0
    res = learn_likelihood(TRUE_A, num_steps=50, count_scale=20.0, eta=eta)
    assert res.is_monotone_decreasing
    assert res.final_kl < CONVERGENCE_KL_ATOL
    assert res.total_count == pytest.approx(eta, abs=1e-6)


def test_without_eta_total_grows_unbounded():
    res = learn_likelihood(TRUE_A, num_steps=10, count_scale=10.0)
    # No saturation: total = prior_mass + num_steps * batch_total.
    prior_mass = DEFAULT_PRIOR_CONCENTRATION * TRUE_A.size
    batch_total = 10.0 * TRUE_A.shape[1]  # sum over a column-normalized matrix
    assert res.total_count == pytest.approx(prior_mass + 10 * batch_total)


# ---- error paths ---------------------------------------------------------

def test_rejects_non_positive_num_steps():
    with pytest.raises(ValueError, match="num_steps"):
        learn_likelihood(TRUE_A, num_steps=0)


def test_rejects_non_positive_count_scale():
    with pytest.raises(ValueError, match="count_scale"):
        learn_likelihood(TRUE_A, num_steps=3, count_scale=0.0)


def test_rejects_non_positive_prior_concentration():
    with pytest.raises(ValueError, match="prior_concentration"):
        learn_likelihood(TRUE_A, num_steps=3, prior_concentration=-1.0)


def test_rejects_non_positive_eta():
    with pytest.raises(ValueError, match="eta must be positive"):
        learn_likelihood(TRUE_A, num_steps=3, eta=0.0)


def test_rejects_eta_below_batch_mass():
    # batch_total = count_scale * n_s = 10 * 2 = 20; eta must exceed it.
    with pytest.raises(ValueError, match="must exceed the per-step batch mass"):
        learn_likelihood(TRUE_A, num_steps=3, count_scale=10.0, eta=15.0)


def test_result_dataclass_properties():
    res = learn_likelihood(TRUE_A, num_steps=4)
    assert isinstance(res, DirichletLearningResult)
    assert res.initial_kl == res.kl_trajectory[0]
    assert res.final_kl == res.kl_trajectory[-1]
    assert 0 <= res.steps_to_converge <= len(res.kl_trajectory)


def test_steps_to_converge_returns_horizon_when_never_converged():
    # One tiny step from a flat prior: KL stays above tolerance the whole horizon.
    res = learn_likelihood(TRUE_A, num_steps=1, count_scale=0.001)
    assert res.steps_to_converge == len(res.kl_trajectory)


def test_learn_likelihood_multi_column_converges_column_by_column():
    """learn_likelihood on a wider matrix converges independently per column.

    The existing tests use a 3×2 TRUE_A. A 3×4 matrix exposes whether
    normalisation and count accumulation are truly column-independent:
    each column's argmax after learning must match the true matrix, and
    the sum-to-one column-stochastic property must hold for every column.
    """
    TRUE_A_WIDE = np.array([
        [0.80, 0.10, 0.05, 0.05],
        [0.10, 0.80, 0.05, 0.10],
        [0.10, 0.10, 0.90, 0.85],
    ])  # 3 outcomes x 4 states

    res = learn_likelihood(TRUE_A_WIDE, num_steps=60)

    # KL must drop below tolerance (ISC-17 generalised to the wide matrix).
    assert res.final_kl < CONVERGENCE_KL_ATOL, (
        f"final_kl={res.final_kl:.4f}: should converge for a 3×4 wide matrix"
    )
    assert res.is_monotone_decreasing

    # Shape must be preserved.
    assert res.expected_a.shape == TRUE_A_WIDE.shape

    # Each column must still be column-stochastic after learning.
    np.testing.assert_allclose(
        res.expected_a.sum(axis=0), 1.0, atol=1e-12,
        err_msg="Learned likelihood columns must sum to 1.0"
    )

    # Column-independence check: argmax per column must match the true matrix.
    for col in range(TRUE_A_WIDE.shape[1]):
        learned_argmax = int(np.argmax(res.expected_a[:, col]))
        true_argmax = int(np.argmax(TRUE_A_WIDE[:, col]))
        assert learned_argmax == true_argmax, (
            f"Column {col}: learned argmax={learned_argmax}, "
            f"true argmax={true_argmax} — columns must converge independently"
        )
