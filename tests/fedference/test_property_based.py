"""Property-based tests for the fedference core using the Hypothesis framework.

These tests complement the 400+ exact-numeric tests with generative,
randomised counterexample search.  Rather than checking specific numbers, every
test encodes an *invariant* — a mathematical property that must hold for ALL
valid inputs — and lets Hypothesis find any input that breaks it.

Coverage map
------------
Divergences  (D1-D5) : kl_divergence, renyi_divergence, total_variation,
                        gaussian_kl / gaussian_renyi
Losses       (L1-L3) : nll, beta_loss, rcce
Aggregation  (A1-A5) : log_linear_pool, robust_aggregate, variational_aggregate,
                        aggregate dispatch, agent_weights normalization
Gen-Bayes    (G1-G2) : generalized_posterior closed form, cavity roundtrip
Belief share (S1-S2) : share_round pmf validity, exclude_self differentiation
Statistics   (T1-T4) : power_analysis in [0,1], bootstrap_ci lo<=hi,
                        bh_fdr q-values in [0,1], cohens_d sign consistency

18 property-based tests (>= 12 required).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Shared strategies — all use lazy imports inside each test body so that any
# ImportError surfaces as a test failure rather than a collection error.
# ---------------------------------------------------------------------------

_EPS = 1e-12

# Small strictly-positive floats suitable for PMF component weights.
_pos_weight = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)


def _pmf_of_size(k: int):
    """Strategy: k positive floats normalized to a pmf."""
    return st.lists(_pos_weight, min_size=k, max_size=k).map(
        lambda xs: np.array(xs, dtype=np.float64) / sum(xs)
    )


def _pmf_strategy(min_k: int = 2, max_k: int = 8):
    """Strategy: a single random pmf with between min_k and max_k states."""
    return st.integers(min_value=min_k, max_value=max_k).flatmap(_pmf_of_size)


def _paired_pmf_strategy(min_k: int = 2, max_k: int = 8):
    """Strategy: two pmfs over the SAME k states — avoids assume() filtering."""
    return st.integers(min_value=min_k, max_value=max_k).flatmap(
        lambda k: st.tuples(_pmf_of_size(k), _pmf_of_size(k))
    )


def _belief_list_strategy(
    min_agents: int = 2,
    max_agents: int = 5,
    min_states: int = 2,
    max_states: int = 6,
):
    """Strategy: list of pmfs all sharing the same n_states."""
    return st.integers(min_value=min_states, max_value=max_states).flatmap(
        lambda n_states: st.integers(min_value=min_agents, max_value=max_agents).flatmap(
            lambda n_agents: st.lists(_pmf_of_size(n_states), min_size=n_agents, max_size=n_agents)
        )
    )


_pos_var = st.floats(min_value=1e-4, max_value=1e4, allow_nan=False, allow_infinity=False)
_mean_float = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)

_pvalues_strategy = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=20,
)

_data_strategy = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=30,
)


def _is_valid_pmf(arr: np.ndarray, atol: float = 1e-6) -> bool:
    """Return True iff arr is a valid pmf: 1-D, non-negative, sums to 1."""
    return arr.ndim == 1 and arr.size > 0 and bool(np.all(arr >= 0.0)) and abs(arr.sum() - 1.0) < atol


# ===========================================================================
# D1 — KL divergence is non-negative (Gibbs' inequality)
# ===========================================================================


@given(_paired_pmf_strategy())
@settings(max_examples=300, deadline=None)
def test_kl_non_negative(qp):
    """KL(q||p) >= 0 for all categorical pmfs (Gibbs' inequality)."""
    from fedference.divergences import kl_divergence

    q, p = qp
    assert kl_divergence(q, p) >= -1e-12


# ===========================================================================
# D2 — KL divergence is zero iff distributions are equal
# ===========================================================================


@given(_pmf_strategy())
@settings(max_examples=200, deadline=None)
def test_kl_zero_iff_equal(q):
    """KL(q||q) == 0  (self-divergence is always zero)."""
    from fedference.divergences import kl_divergence

    assert kl_divergence(q, q) == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# D3 — Rényi divergence recovers KL exactly at alpha=1
# ===========================================================================


@given(_paired_pmf_strategy())
@settings(max_examples=200, deadline=None)
def test_renyi_recovers_kl_at_alpha_one(qp):
    """renyi(q, p, alpha=1.0) == kl(q, p) exactly (library's built-in KL branch)."""
    from fedference.divergences import kl_divergence, renyi_divergence

    q, p = qp
    kl = kl_divergence(q, p)
    r_exact = renyi_divergence(q, p, alpha=1.0)
    assert r_exact == pytest.approx(kl, abs=1e-10)


# ===========================================================================
# D4 — Total variation is in [0, 1] and symmetric
# ===========================================================================


@given(_paired_pmf_strategy())
@settings(max_examples=300, deadline=None)
def test_total_variation_bounded_and_symmetric(qp):
    """TV(q, p) is always in [0, 1] and TV(q,p) == TV(p,q)."""
    from fedference.divergences import total_variation

    q, p = qp
    tv_qp = total_variation(q, p)
    tv_pq = total_variation(p, q)
    assert 0.0 - 1e-12 <= tv_qp <= 1.0 + 1e-12
    assert tv_qp == pytest.approx(tv_pq, abs=1e-12)


# ===========================================================================
# D5 — Gaussian KL is non-negative
# ===========================================================================


@given(_mean_float, _pos_var, _mean_float, _pos_var)
@settings(max_examples=300, deadline=None)
def test_gaussian_kl_non_negative(mu_q, var_q, mu_p, var_p):
    """KL(N(mu_q,var_q) || N(mu_p,var_p)) >= 0 for any valid Gaussian parameters."""
    from fedference.divergences import gaussian_kl

    assert gaussian_kl(mu_q, var_q, mu_p, var_p) >= -1e-10


# ===========================================================================
# L1 — beta_loss approaches NLL as beta → 0
# ===========================================================================


@given(_pmf_strategy(min_k=2, max_k=6), st.integers(min_value=0, max_value=5))
@settings(max_examples=200, deadline=None)
def test_beta_loss_recovers_nll_as_beta_to_zero(p, o_raw):
    """beta_loss(p, o, beta≈0) ≈ nll(p, o) for any pmf and outcome."""
    from fedference.losses import beta_loss, nll

    o = o_raw % p.size
    nll_val = nll(p, o)
    beta_val = beta_loss(p, o, beta=1e-8)
    assert beta_val == pytest.approx(nll_val, rel=1e-4, abs=1e-6)


# ===========================================================================
# L2 — rcce approaches NLL as q → 0
# ===========================================================================


@given(_pmf_strategy(min_k=2, max_k=6), st.integers(min_value=0, max_value=5))
@settings(max_examples=200, deadline=None)
def test_rcce_recovers_nll_as_q_loss_to_zero(p, o_raw):
    """rcce(p, o, q_loss≈0) ≈ nll(p, o) for any pmf and outcome."""
    from fedference.losses import nll, rcce

    o = o_raw % p.size
    nll_val = nll(p, o)
    rcce_val = rcce(p, o, q_loss=1e-9)
    assert rcce_val == pytest.approx(nll_val, rel=1e-4, abs=1e-6)


# ===========================================================================
# L3 — NLL is always non-negative
# ===========================================================================


@given(_pmf_strategy(min_k=2, max_k=8), st.integers(min_value=0, max_value=7))
@settings(max_examples=300, deadline=None)
def test_nll_non_negative(p, o_raw):
    """NLL = -log p(o|s) is non-negative for any valid pmf and outcome."""
    from fedference.losses import nll

    o = o_raw % p.size
    assert nll(p, o) >= 0.0


# ===========================================================================
# A1 — All aggregators produce valid pmfs
# ===========================================================================


@given(_belief_list_strategy())
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_all_aggregators_return_valid_pmf(beliefs):
    """log_linear_pool, robust_aggregate, and variational_aggregate all return valid pmfs."""
    from fedference.aggregation import (
        log_linear_pool,
        robust_aggregate,
        variational_aggregate,
    )

    naive = log_linear_pool(beliefs)
    assert _is_valid_pmf(naive), f"log_linear_pool not a valid pmf: {naive}"

    robust_res = robust_aggregate(beliefs, robustness=1.0)
    assert _is_valid_pmf(robust_res.consensus), f"robust_aggregate not a valid pmf: {robust_res.consensus}"

    var_res = variational_aggregate(beliefs, robustness=1.0)
    assert _is_valid_pmf(var_res.consensus), f"variational_aggregate not a valid pmf: {var_res.consensus}"


# ===========================================================================
# A2 — robust_aggregate(robustness=0) == log_linear_pool  (exact identity)
# ===========================================================================


@given(_belief_list_strategy())
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_robust_aggregate_zero_robustness_equals_log_linear_pool(beliefs):
    """robust_aggregate(robustness=0) is bit-identical to log_linear_pool for ANY beliefs."""
    from fedference.aggregation import log_linear_pool, robust_aggregate

    naive = log_linear_pool(beliefs)
    robust_res = robust_aggregate(beliefs, robustness=0.0)
    np.testing.assert_allclose(
        robust_res.consensus,
        naive,
        atol=1e-12,
        err_msg="robust_aggregate(robustness=0) diverged from log_linear_pool",
    )


# ===========================================================================
# A3 — variational_aggregate(robustness=0) == log_linear_pool  (exact identity)
# ===========================================================================


@given(_belief_list_strategy())
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_variational_aggregate_zero_robustness_equals_log_linear_pool(beliefs):
    """variational_aggregate(robustness=0, entropy_weight=1) == log_linear_pool."""
    from fedference.aggregation import log_linear_pool, variational_aggregate

    naive = log_linear_pool(beliefs)
    var_res = variational_aggregate(beliefs, robustness=0.0, entropy_weight=1.0)
    np.testing.assert_allclose(
        var_res.consensus,
        naive,
        atol=1e-10,
        err_msg="variational_aggregate(robustness=0) diverged from log_linear_pool",
    )


# ===========================================================================
# A4 — aggregate dispatch(method='naive') equals log_linear_pool
# ===========================================================================


@given(_belief_list_strategy())
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_aggregate_dispatch_naive_consistent(beliefs):
    """aggregate(method='naive') returns the same result as log_linear_pool directly."""
    from fedference.aggregation import aggregate, log_linear_pool

    direct = log_linear_pool(beliefs)
    dispatched = aggregate(beliefs, method="naive")
    np.testing.assert_allclose(direct, dispatched, atol=1e-12)


# ===========================================================================
# A5 — agent_weights from robust_aggregate sum to 1
# ===========================================================================


@given(
    _belief_list_strategy(min_agents=2, max_agents=5),
    st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_robust_aggregate_agent_weights_sum_to_one(beliefs, robustness):
    """robust_aggregate always returns normalized agent_weights (sum to 1)."""
    from fedference.aggregation import robust_aggregate

    res = robust_aggregate(beliefs, robustness=robustness)
    assert abs(res.agent_weights.sum() - 1.0) < 1e-9, (
        f"agent_weights sum to {res.agent_weights.sum()} instead of 1"
    )


# ===========================================================================
# G1 — generalized_posterior(KLD, NLL) == standard Bayes closed form
# ===========================================================================


@given(
    _pmf_strategy(min_k=2, max_k=6),  # prior
    _pmf_strategy(min_k=2, max_k=6),  # likelihood per state, already a valid pmf
)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
def test_generalized_posterior_kld_nll_equals_bayes(prior, likelihood_row):
    """KL-divergence + NLL loss == standard Bayes: posterior ∝ prior × likelihood."""
    from fedference.generalized_bayes import generalized_posterior
    from fedference.losses import loss_vector

    assume(prior.size == likelihood_row.size)

    # Build a (2, n_s) likelihood matrix with column-stochastic columns (each
    # column is a valid p(o|s) distribution over observations).
    # Row 0 = likelihood values; row 1 = complement, so each column sums to 1.
    lik = np.clip(likelihood_row, _EPS, 1.0 - _EPS)
    row1 = 1.0 - lik  # complement: columns now sum to 1 (valid distributions)
    A = np.stack([lik, row1], axis=0)  # shape (2, n_s), each col sums to 1

    o = 0
    lv = loss_vector(A, o, loss="nll")  # length n_s
    log_prior = np.log(prior)

    q = generalized_posterior(log_prior, lv, learning_rate=1.0, divergence="KLD")

    # Standard Bayes: posterior ∝ prior × p(o=0|s) = prior × lik
    exact = prior * lik
    exact = exact / exact.sum()
    np.testing.assert_allclose(
        q, exact, atol=1e-7, err_msg="generalized_posterior(KLD, NLL) != standard Bayes"
    )


# ===========================================================================
# G2 — cavity(posterior, factor) re-multiplied by factor restores posterior
# ===========================================================================


@given(_paired_pmf_strategy())
@settings(max_examples=200, deadline=None)
def test_cavity_factor_roundtrip(posterior_factor):
    """cavity(q, t) re-combined with t restores q up to numerical precision."""
    from fedference.generalized_bayes import cavity, softmax

    posterior, factor = posterior_factor
    cav = cavity(posterior, factor)
    # Restore: softmax(log cav + log factor)
    log_cav = np.log(np.clip(cav, _EPS, None))
    log_factor = np.log(np.clip(factor, _EPS, None))
    restored = softmax(log_cav + log_factor)
    np.testing.assert_allclose(restored, posterior, atol=1e-7, err_msg="cavity roundtrip failed")


# ===========================================================================
# S1 — share_round always returns valid pmfs for every agent
# ===========================================================================


@given(
    _belief_list_strategy(min_agents=2, max_agents=5),
    st.sampled_from(["naive", "robust"]),
)
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_share_round_returns_valid_pmfs(beliefs, method):
    """share_round always produces valid (non-negative, normalized) per-agent beliefs."""
    from fedference.belief_sharing import share_round

    diag = share_round(beliefs, method=method, robustness=1.0, exclude_self=True)
    for i, row in enumerate(diag.shared_beliefs):
        assert _is_valid_pmf(row, atol=1e-6), f"Agent {i}'s shared belief is not a valid pmf: {row}"
    assert _is_valid_pmf(diag.consensus, atol=1e-6), f"Global consensus is not a valid pmf: {diag.consensus}"


# ===========================================================================
# S2 — share_round with exclude_self=True: every recipient omits itself
# ===========================================================================


@given(_belief_list_strategy(min_agents=3, max_agents=6))
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_share_round_exclude_self_uses_each_recipients_leave_one_out_pool(beliefs):
    """Each recipient receives the pool of every broadcast except its own.

    This directly checks the semantic contract rather than assuming the
    leave-one-out results must be numerically distinct.  With extreme but
    valid beliefs, product pooling can legitimately saturate every recipient
    at the same state even though each pool has a different membership.
    """
    from fedference.aggregation import log_linear_pool
    from fedference.belief_sharing import share_round

    beliefs_arr = np.asarray(beliefs, dtype=np.float64)
    diag = share_round(beliefs_arr, method="naive", exclude_self=True)
    all_indices = np.arange(len(beliefs_arr))
    for recipient in all_indices:
        expected = log_linear_pool(beliefs_arr[all_indices != recipient])
        np.testing.assert_allclose(
            diag.shared_posteriors[recipient],
            expected,
            atol=1e-12,
            rtol=0.0,
            err_msg=f"recipient {recipient} did not receive its leave-one-out pool",
        )


# ===========================================================================
# T1 — power_analysis returns power in [0, 1]
# ===========================================================================


@given(
    st.floats(min_value=1e-3, max_value=5.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=2, max_value=500),
    st.floats(min_value=0.001, max_value=0.499, allow_nan=False, allow_infinity=False),
    st.sampled_from(["greater", "less", "two-sided"]),
)
@settings(max_examples=300, deadline=None)
def test_power_analysis_returns_valid_power(effect_size, n, alpha, alternative):
    """power_analysis always returns power in [0, 1] for any valid inputs.

    We restrict effect_size away from near-zero to avoid the overflow in
    sample_size_for_power (((z_a + z_p) / eff)**2 overflows for eff ≈ 0).
    Near-zero effect sizes are tested via the unit tests; this property
    checks the non-degenerate range.
    """
    from fedference.statistics import power_analysis

    result = power_analysis(effect_size, n, alpha=alpha, alternative=alternative)
    assert 0.0 <= result["power"] <= 1.0, (
        f"power={result['power']} outside [0,1] for effect={effect_size}, n={n}"
    )
    assert result["n"] == n
    assert result["effect_size"] == pytest.approx(effect_size)


# ===========================================================================
# T2 — bootstrap_ci returns (lo, hi) with lo <= hi
# ===========================================================================


@given(_data_strategy, st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_bootstrap_ci_lo_leq_hi(data, seed):
    """bootstrap_ci always returns lo <= hi regardless of data distribution."""
    from fedference.statistics import bootstrap_ci

    assume(len(data) >= 2)
    rng = np.random.default_rng(seed)
    lo, hi = bootstrap_ci(data, alpha=0.05, n_boot=500, rng=rng)
    assert lo <= hi, f"bootstrap_ci returned lo={lo} > hi={hi}"


# ===========================================================================
# T3 — bh_fdr q-values are bounded in [0, 1]
# ===========================================================================


@given(
    _pvalues_strategy,
    st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300, deadline=None)
def test_bh_fdr_qvalues_bounded(pvalues, alpha):
    """BH FDR q-values always lie in [0, 1] for any p-value family."""
    from fedference.statistics import bh_fdr

    assume(len(pvalues) >= 1)
    result = bh_fdr(pvalues, alpha=alpha)
    qv = result["qvalues"]
    assert np.all(qv >= 0.0), f"Some q-values < 0: {qv}"
    assert np.all(qv <= 1.0), f"Some q-values > 1: {qv}"
    assert len(qv) == len(pvalues)
    assert len(result["rejected"]) == len(pvalues)


# ===========================================================================
# T4 — cohens_d_from_rank_biserial: sign consistency and finiteness
# ===========================================================================


@given(
    st.floats(min_value=-0.99, max_value=0.99, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300, deadline=None)
def test_cohens_d_from_rank_biserial_sign_and_zero(r):
    """cohens_d(r) shares the sign of r; cohens_d(0) == 0; result is finite."""
    from fedference.statistics import cohens_d_from_rank_biserial

    d = cohens_d_from_rank_biserial(r)
    assert math.isfinite(d), f"cohens_d({r}) returned non-finite {d}"
    if r > 1e-10:
        assert d > 0.0, f"cohens_d({r}) should be positive, got {d}"
    elif r < -1e-10:
        assert d < 0.0, f"cohens_d({r}) should be negative, got {d}"
    else:
        assert abs(d) < 1e-9, f"cohens_d(~0) should be ~0, got {d}"
