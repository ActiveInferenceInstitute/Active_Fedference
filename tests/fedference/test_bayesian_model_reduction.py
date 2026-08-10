"""Bayesian model reduction tests (Friston & Penny 2011; Friston 2024 Eq. 13).

No mocks: every value is a real ``scipy.special.gammaln`` computation on small
Dirichlet concentration vectors with explicit, hand-derived expectations. The
load-bearing checks (ISC-20) are:

* ``delta_F`` matches a closed-form hand value on a small known reduced prior
  (``prior=[1,1]``, ``post=[3,1]``, ``reduced_prior=[0.5,0.5]`` collapses, via
  ``lnB(prior)=0`` and the Beta-function identity, to exactly ``ln(9/8)``);
* pruning a redundant column (a barely-supported state shrunk toward zero)
  yields ``delta_F > 0`` — the reduced model has more evidence.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gammaln

from fedference.bayesian_model_reduction import (
    greedy_reduce,
    log_beta,
    reduce,
    reduced_posterior,
)


def _lnB(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.sum(gammaln(a)) - gammaln(a.sum()))


# ---- log_beta normalizer -------------------------------------------------

def test_log_beta_matches_reference_formula():
    a = np.array([3.0, 1.0, 2.0])
    assert log_beta(a) == pytest.approx(_lnB(a), abs=1e-12)


def test_log_beta_symmetric_uniform_pair():
    # lnB([1,1]) = gammaln(1)+gammaln(1)-gammaln(2) = 0.
    assert log_beta([1.0, 1.0]) == pytest.approx(0.0, abs=1e-12)
    # lnB([0.5,0.5]) = 2*gammaln(0.5) - gammaln(1) = ln(pi).
    assert log_beta([0.5, 0.5]) == pytest.approx(np.log(np.pi), abs=1e-12)


# ---- closed-form reduced posterior --------------------------------------

def test_reduced_posterior_is_post_plus_rprior_minus_prior():
    post = np.array([3.0, 1.0])
    prior = np.array([1.0, 1.0])
    rprior = np.array([0.5, 0.5])
    rpost = reduced_posterior(post, prior, rprior)
    np.testing.assert_allclose(rpost, np.array([2.5, 0.5]), atol=1e-12)


# ---- ISC-20: hand-computed delta_F --------------------------------------

def test_delta_F_matches_hand_computed_value():
    """prior=[1,1], post=[3,1], reduced_prior=[0.5,0.5] -> dF = ln(9/8) exactly.

    Derivation: lnB(prior)=0; the remaining three log-Beta terms collapse to
    ln(0.375 * 3) = ln(1.125) = ln(9/8).
    """
    result = reduce(
        posterior_counts=[3.0, 1.0],
        prior_counts=[1.0, 1.0],
        reduced_prior_counts=[0.5, 0.5],
    )
    assert result["delta_F"] == pytest.approx(np.log(9.0 / 8.0), abs=1e-12)
    # And the explicit four-term assembly agrees with the independent reference.
    assert result["delta_F"] == pytest.approx(
        _lnB([1.0, 1.0]) + _lnB([2.5, 0.5]) - _lnB([3.0, 1.0]) - _lnB([0.5, 0.5]),
        abs=1e-12,
    )
    np.testing.assert_allclose(result["reduced_posterior"], [2.5, 0.5], atol=1e-12)


def test_components_are_reported():
    result = reduce([3.0, 1.0], [1.0, 1.0], [0.5, 0.5])
    assert result["log_beta_prior"] == pytest.approx(0.0, abs=1e-12)
    assert result["log_beta_reduced_prior"] == pytest.approx(np.log(np.pi), abs=1e-12)
    assert result["log_beta_posterior"] == pytest.approx(np.log(1.0 / 3.0), abs=1e-12)


# ---- ISC-20: pruning a redundant column gives dF > 0 --------------------

def test_pruning_redundant_column_increases_evidence():
    """A barely-supported third state, pruned toward zero, raises free energy."""
    prior = [1.0, 1.0, 1.0]
    post = [10.0, 10.0, 1.0]  # third state carries essentially no data
    reduced_prior = [1.0, 1.0, 0.125]  # shrink the redundant column
    result = reduce(post, prior, reduced_prior)
    assert result["delta_F"] > 0.0


def test_no_reduction_gives_zero_delta_F():
    """reduced_prior == prior is not a reduction: dF is identically zero."""
    result = reduce([4.0, 2.0, 3.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert result["delta_F"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(result["reduced_posterior"], [4.0, 2.0, 3.0], atol=1e-12)


def test_pruning_supported_column_destroys_evidence():
    """Pruning a *well-supported* state is penalized: dF < 0, and by a real margin.

    The sign alone is too weak a check — a near-zero negative delta (an
    almost-indifferent reduction) would pass it. The actual gammaln computation
    on these counts gives dF ~ -1.635, so a -0.5 floor is cleared comfortably
    while still failing any regression that collapses the penalty toward zero.
    """
    prior = [1.0, 1.0, 1.0]
    post = [10.0, 10.0, 10.0]  # all three states strongly supported
    reduced_prior = [1.0, 1.0, 0.125]  # wrongly prune a real column
    result = reduce(post, prior, reduced_prior)
    assert result["delta_F"] < -0.5


# ---- error paths ---------------------------------------------------------

def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        log_beta([])


def test_negative_concentration_raises():
    with pytest.raises(ValueError, match="negative"):
        reduce([3.0, 1.0], [-1.0, 1.0], [0.5, 0.5])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="share a shape"):
        reduce([3.0, 1.0, 2.0], [1.0, 1.0], [0.5, 0.5])


def test_reduced_posterior_shape_mismatch_raises():
    with pytest.raises(ValueError, match="share a shape"):
        reduced_posterior([3.0, 1.0], [1.0, 1.0, 1.0], [0.5, 0.5])


# ---- greedy multi-hypothesis structure learning --------------------------

def test_greedy_recovers_sparse_model_from_redundant_one():
    # Five states; the data support only the first two (heavy counts), the last
    # three are redundant (no evidence accrues). Greedy reduction should prune
    # exactly the three redundant states and keep the two supported ones.
    prior = np.ones(5)
    counts = np.array([60.0, 50.0, 0.0, 0.0, 0.0])
    posterior = prior + counts
    out = greedy_reduce(posterior, prior)
    assert out["pruned_states"] == [2, 3, 4]
    assert out["n_pruned"] == 3
    assert out["total_delta_F"] > 0.0


def test_greedy_steps_are_monotone_improving():
    prior = np.ones(5)
    posterior = prior + np.array([60.0, 50.0, 0.0, 0.0, 0.0])
    out = greedy_reduce(posterior, prior)
    steps = out["steps"]
    assert len(steps) == 3
    # every accepted prune has a strictly positive incremental gain...
    assert all(s["delta_F_step"] > 0.0 for s in steps)
    # ...so the cumulative evidence is monotone-increasing.
    cum = [s["cumulative_delta_F"] for s in steps]
    assert all(cum[i] < cum[i + 1] for i in range(len(cum) - 1))


def test_greedy_keeps_all_supported_states():
    # When every state is well supported, no prune improves evidence -> none taken.
    prior = np.ones(4)
    posterior = prior + np.array([30.0, 28.0, 32.0, 29.0])
    out = greedy_reduce(posterior, prior)
    assert out["pruned_states"] == []
    assert out["n_pruned"] == 0
    assert out["total_delta_F"] == 0.0


def test_greedy_respects_max_prunes():
    prior = np.ones(5)
    posterior = prior + np.array([60.0, 50.0, 0.0, 0.0, 0.0])
    out = greedy_reduce(posterior, prior, max_prunes=1)
    assert out["n_pruned"] == 1
    # the single accepted prune is one of the redundant states.
    assert out["pruned_states"][0] in (2, 3, 4)


def test_greedy_too_few_states_raises():
    with pytest.raises(ValueError, match="at least two states"):
        greedy_reduce([5.0], [1.0])


def test_greedy_shape_mismatch_raises():
    with pytest.raises(ValueError, match="share a shape"):
        greedy_reduce([3.0, 1.0, 2.0], [1.0, 1.0])


def test_greedy_reduce_selection_is_always_the_lowest_supported_state():
    """Greedy pruning selects zero-evidence states before positive-evidence states.

    The existing tests verify *which* states are ultimately pruned, but not that
    evidence-free states are exhausted before touching any positive-count state.
    This test uses three genuinely zero-count states alongside two well-supported
    states; all three zeros must be pruned (their delta_F > 0) and the two
    well-supported states must be kept.  The step key is 'state' (not
    'pruned_state') — pinned here to prevent silent key renames.
    """
    prior = np.ones(5)
    # States 2, 3, 4 have zero observed counts — they are purely prior mass.
    counts = np.array([60.0, 50.0, 0.0, 0.0, 0.0])
    posterior = prior + counts
    out = greedy_reduce(posterior, prior)
    steps = out["steps"]

    # All three zero-evidence states must be pruned.
    assert out["n_pruned"] == 3, (
        f"Expected 3 pruned states, got {out['n_pruned']}"
    )
    pruned = set(out["pruned_states"])
    assert pruned == {2, 3, 4}, (
        f"Expected {{2,3,4}} to be pruned, got {pruned}"
    )

    # Each step must use the 'state' key and report strictly positive delta_F.
    for i, s in enumerate(steps):
        assert "state" in s, f"Step {i} missing 'state' key; got keys {list(s.keys())}"
        assert s["delta_F_step"] > 0.0, (
            f"Step {i}: delta_F_step={s['delta_F_step']:.6f} must be > 0"
        )
        assert s["state"] in {2, 3, 4}, (
            f"Step {i}: pruned state {s['state']} is well-supported — should not be pruned"
        )
