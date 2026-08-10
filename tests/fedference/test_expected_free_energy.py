"""EFE decomposition tests (no mocks, real seeded categorical POMDPs).

These pin Friston et al. (2024) Eq. 2: the two EFE decompositions are the same
scalar, so for every policy

    risk + ambiguity == -(pragmatic_value + epistemic_value)

to ``EFE_IDENTITY_ATOL`` (project gate ISC-19; the same shared constant
the ``ISC_EFE_TOLERANCE`` manuscript token renders). All distributions are explicit small arrays
or seeded draws via ``np.random.default_rng`` — never global RNG, never mocks.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from fedference.expected_free_energy import (
    EFE_IDENTITY_ATOL,
    EFETerms,
    decompose,
    preferred_outcomes,
)


def _identity_transition(n_s: int) -> np.ndarray:
    """One action that holds the state belief fixed (identity transition)."""
    b: np.ndarray = np.zeros((n_s, n_s, 1), dtype=np.float64)
    b[:, :, 0] = np.eye(n_s)
    return b


def _two_action_transition(n_s: int) -> np.ndarray:
    """Action 0 = identity; action 1 = cyclic shift of the hidden state."""
    b: np.ndarray = np.zeros((n_s, n_s, 2), dtype=np.float64)
    b[:, :, 0] = np.eye(n_s)
    b[:, :, 1] = np.roll(np.eye(n_s), shift=1, axis=0)
    return b


# ---- preferred outcomes --------------------------------------------------

def test_preferred_outcomes_is_softmax_of_C():
    c = np.array([0.0, 2.0, 1.0])
    pref = preferred_outcomes(c)
    manual = np.exp(c - c.max())
    manual = manual / manual.sum()
    assert np.allclose(pref, manual, atol=1e-12)
    assert pref.sum() == pytest.approx(1.0, abs=1e-12)


# ---- the central identity (ISC-19) ---------------------------------------

def test_decomposition_identity_single_step():
    a = np.array([[0.9, 0.2], [0.1, 0.8]])  # (n_o=2, n_s=2)
    b = _identity_transition(2)
    c = np.array([2.0, 0.0])  # prefer outcome 0
    prior = np.array([0.6, 0.4])
    terms = decompose(a, b, c, prior, policy=[0])
    assert isinstance(terms, EFETerms)
    assert terms.total == pytest.approx(terms.risk + terms.ambiguity, abs=1e-12)
    # risk + ambiguity == -(pragmatic + epistemic)
    assert terms.identity_residual == pytest.approx(0.0, abs=EFE_IDENTITY_ATOL)


def test_decomposition_identity_holds_for_all_policies():
    rng = np.random.default_rng(20240624)
    n_o, n_s = 3, 3
    # Random column-stochastic likelihood A.
    a = rng.dirichlet(np.ones(n_o), size=n_s).T  # (n_o, n_s)
    b = _two_action_transition(n_s)
    c = rng.normal(size=n_o)
    prior = rng.dirichlet(np.ones(n_s))

    max_residual = 0.0
    n_a = b.shape[2]
    for policy in product(range(n_a), repeat=3):  # all length-3 policies
        terms = decompose(a, b, c, prior, policy=list(policy))
        max_residual = max(max_residual, abs(terms.identity_residual))
    assert max_residual < EFE_IDENTITY_ATOL


def test_empty_policy_yields_zero_terms():
    a = np.array([[0.7, 0.3], [0.3, 0.7]])
    b = _identity_transition(2)
    c = np.array([1.0, 0.0])
    prior = np.array([0.5, 0.5])
    terms = decompose(a, b, c, prior, policy=[])
    assert terms.risk == 0.0
    assert terms.ambiguity == 0.0
    assert terms.pragmatic_value == 0.0
    assert terms.epistemic_value == 0.0
    assert terms.total == 0.0
    assert terms.identity_residual == pytest.approx(0.0, abs=1e-12)


# ---- term semantics ------------------------------------------------------

def test_deterministic_likelihood_has_zero_ambiguity():
    # Each state maps to a distinct outcome with certainty -> H[p(o|s)] = 0.
    a = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    b = _identity_transition(3)
    c = np.array([0.0, 0.0, 0.0])  # flat preference
    prior = np.array([0.2, 0.3, 0.5])
    terms = decompose(a, b, c, prior, policy=[0])
    assert terms.ambiguity == pytest.approx(0.0, abs=1e-12)
    # Flat preference + deterministic A: epistemic value equals H[q(o)] > 0.
    assert terms.epistemic_value > 0.0
    assert terms.identity_residual == pytest.approx(0.0, abs=EFE_IDENTITY_ATOL)


def test_uninformative_likelihood_has_zero_epistemic_value():
    # Every state emits the same outcome distribution -> outcomes carry no info.
    row = np.array([0.6, 0.4])
    a = np.column_stack([row, row, row])  # (2, 3), identical columns
    b = _identity_transition(3)
    c = np.array([0.0, 0.0])
    prior = np.array([0.2, 0.3, 0.5])
    terms = decompose(a, b, c, prior, policy=[0])
    # H[q(o)] == E_q(s)[H[p(o|s)]] when A's columns are identical -> I(s;o)=0.
    assert terms.epistemic_value == pytest.approx(0.0, abs=1e-12)
    assert terms.identity_residual == pytest.approx(0.0, abs=EFE_IDENTITY_ATOL)


def test_preference_match_lowers_risk():
    # Predicted outcomes that match the preference give lower risk than a mismatch.
    a = np.array([[0.95, 0.05], [0.05, 0.95]])
    b = _identity_transition(2)
    prior_match = np.array([0.95, 0.05])  # drives q(o) toward outcome 0
    c_for_o0 = np.array([3.0, 0.0])  # strongly prefer outcome 0
    aligned = decompose(a, b, c_for_o0, prior_match, policy=[0])
    misaligned = decompose(a, b, c_for_o0, np.array([0.05, 0.95]), policy=[0])
    assert aligned.risk < misaligned.risk


def test_multistep_terms_accumulate_over_horizon():
    a = np.array([[0.8, 0.3], [0.2, 0.7]])
    b = _two_action_transition(2)
    c = np.array([1.5, 0.0])
    prior = np.array([0.7, 0.3])
    one = decompose(a, b, c, prior, policy=[0])
    two = decompose(a, b, c, prior, policy=[0, 0])
    # Identity transition (action 0) repeats the same step, so two-step is 2x.
    assert two.ambiguity == pytest.approx(2.0 * one.ambiguity, abs=1e-12)
    assert two.risk == pytest.approx(2.0 * one.risk, abs=1e-12)
    assert two.identity_residual == pytest.approx(0.0, abs=1e-9)


# ---- error paths ---------------------------------------------------------

def test_decompose_rejects_non_2d_likelihood():
    with pytest.raises(ValueError, match="2-D"):
        decompose(np.array([0.5, 0.5]), _identity_transition(2),
                  np.array([1.0, 0.0]), np.array([0.5, 0.5]), policy=[0])


def test_decompose_rejects_non_3d_transition():
    a = np.array([[0.7, 0.3], [0.3, 0.7]])
    with pytest.raises(ValueError, match="3-D"):
        decompose(a, np.eye(2), np.array([1.0, 0.0]), np.array([0.5, 0.5]), policy=[0])


def test_decompose_rejects_transition_state_mismatch():
    a = np.array([[0.7, 0.3], [0.3, 0.7]])  # n_s = 2
    b = _identity_transition(3)  # n_s = 3
    with pytest.raises(ValueError, match="leading dimensions"):
        decompose(a, b, np.array([1.0, 0.0]), np.array([0.5, 0.5]), policy=[0])


def test_decompose_rejects_preference_length_mismatch():
    a = np.array([[0.7, 0.3], [0.3, 0.7]])  # n_o = 2
    b = _identity_transition(2)
    with pytest.raises(ValueError, match="C length"):
        decompose(a, b, np.array([1.0, 0.0, 0.0]), np.array([0.5, 0.5]), policy=[0])


def test_decompose_rejects_prior_length_mismatch():
    a = np.array([[0.7, 0.3], [0.3, 0.7]])  # n_s = 2
    b = _identity_transition(2)
    with pytest.raises(ValueError, match="prior length"):
        decompose(a, b, np.array([1.0, 0.0]), np.array([0.3, 0.3, 0.4]), policy=[0])


def test_decompose_rejects_out_of_range_action():
    a = np.array([[0.7, 0.3], [0.3, 0.7]])
    b = _identity_transition(2)  # only action 0 exists
    with pytest.raises(ValueError, match="out of range"):
        decompose(a, b, np.array([1.0, 0.0]), np.array([0.5, 0.5]), policy=[1])
