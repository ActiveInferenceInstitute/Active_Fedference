"""Tests for single-agent variational state inference (Friston 2024, Eq. 4).

ISC-16: the one-step variational posterior produced by :func:`infer_states`
must equal a hand-computed prior x likelihood normalization
``q(s) = D(s) A[o, s] / sum_s D(s) A[o, s]``. All numbers are real seeded
computations with explicit numeric expectations — no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.belief_updating import infer_states, vfe
from fedference.generalized_bayes import generalized_posterior, softmax
from fedference.losses import loss_vector

# ---- ISC-16: posterior == hand-computed prior*likelihood normalization ----

def test_posterior_equals_hand_computed_normalization():
    # A is (n_o=2, n_s=3); columns are p(o | s).
    A = np.array([[0.8, 0.5, 0.1],
                  [0.2, 0.5, 0.9]])
    prior = np.array([0.2, 0.3, 0.5])
    o = 0

    qs = infer_states(A, o, np.log(prior))

    # Hand computation: unnormalized = D(s) * A[o, s].
    unnorm = prior * A[o, :]            # [0.16, 0.15, 0.05]
    expected = unnorm / unnorm.sum()    # / 0.36
    assert qs == pytest.approx(expected, abs=1e-12)
    # Explicit pinned numbers.
    assert qs == pytest.approx([0.16 / 0.36, 0.15 / 0.36, 0.05 / 0.36], abs=1e-12)
    assert qs.sum() == pytest.approx(1.0, abs=1e-12)


def test_posterior_other_observation():
    A = np.array([[0.8, 0.5, 0.1],
                  [0.2, 0.5, 0.9]])
    prior = np.array([0.2, 0.3, 0.5])
    qs = infer_states(A, 1, np.log(prior))
    unnorm = prior * A[1, :]            # [0.04, 0.15, 0.45]
    assert qs == pytest.approx(unnorm / unnorm.sum(), abs=1e-12)


def test_uniform_prior_posterior_is_normalised_likelihood_column_row():
    # With a flat prior the posterior is just the normalized A[o, :] row.
    A = np.array([[0.7, 0.2, 0.1],
                  [0.3, 0.8, 0.9]])
    prior = np.ones(3) / 3.0
    qs = infer_states(A, 0, np.log(prior))
    assert qs == pytest.approx(A[0, :] / A[0, :].sum(), abs=1e-12)


# ---- multi-modality: additive log-likelihood messages --------------------

def test_two_modalities_multiply_likelihoods():
    A1 = np.array([[0.8, 0.3],
                   [0.2, 0.7]])
    A2 = np.array([[0.6, 0.1],
                   [0.4, 0.9]])
    prior = np.array([0.5, 0.5])
    qs = infer_states([A1, A2], [0, 1], np.log(prior))
    # Hand: D(s) * A1[0, s] * A2[1, s].
    unnorm = prior * A1[0, :] * A2[1, :]   # [0.5*0.8*0.4, 0.5*0.3*0.9]
    assert qs == pytest.approx(unnorm / unnorm.sum(), abs=1e-12)


def test_single_in_list_matches_bare_array():
    A = np.array([[0.8, 0.5, 0.1],
                  [0.2, 0.5, 0.9]])
    lp = np.log(np.array([0.2, 0.3, 0.5]))
    assert infer_states([A], [0], lp) == pytest.approx(infer_states(A, 0, lp), abs=1e-12)


def test_log_prior_need_not_be_normalized():
    # Adding a constant to log_prior is absorbed by the softmax.
    A = np.array([[0.8, 0.5, 0.1],
                  [0.2, 0.5, 0.9]])
    lp = np.log(np.array([0.2, 0.3, 0.5]))
    a = infer_states(A, 0, lp)
    b = infer_states(A, 0, lp + 7.0)
    assert a == pytest.approx(b, abs=1e-12)


# ---- equivalence with the locked generalized_posterior (NLL, w=1) --------

def test_matches_generalized_posterior_with_nll():
    A = np.array([[0.8, 0.5, 0.1],
                  [0.2, 0.5, 0.9]])
    lp = np.log(np.array([0.2, 0.3, 0.5]))
    o = 0
    gp = generalized_posterior(lp, loss_vector(A, o, loss="nll"), learning_rate=1.0)
    assert infer_states(A, o, lp) == pytest.approx(gp, abs=1e-12)


# ---- variational free energy ---------------------------------------------

def test_vfe_at_minimiser_equals_negative_log_evidence():
    A = np.array([[0.8, 0.5, 0.1],
                  [0.2, 0.5, 0.9]])
    prior = np.array([0.2, 0.3, 0.5])
    lp = np.log(prior)
    o = 0
    qs = infer_states(A, o, lp)
    # Model evidence Z = sum_s D(s) A[o, s] = 0.36; F* = -ln Z.
    z = float((prior * A[o, :]).sum())
    assert vfe(qs, A, o, lp) == pytest.approx(-np.log(z), abs=1e-12)
    assert vfe(qs, A, o, lp) == pytest.approx(-np.log(0.36), abs=1e-12)


def test_vfe_is_minimised_by_the_one_step_posterior():
    A = np.array([[0.8, 0.5, 0.1],
                  [0.2, 0.5, 0.9]])
    lp = np.log(np.array([0.2, 0.3, 0.5]))
    o = 0
    qs_star = infer_states(A, o, lp)
    f_star = vfe(qs_star, A, o, lp)
    rng = np.random.default_rng(2024)
    for _ in range(50):
        perturbed = softmax(np.log(np.clip(qs_star, 1e-9, None)) + rng.normal(0, 0.5, size=3))
        assert vfe(perturbed, A, o, lp) >= f_star - 1e-12


def test_vfe_multi_modality_at_minimizer():
    A1 = np.array([[0.8, 0.3],
                   [0.2, 0.7]])
    A2 = np.array([[0.6, 0.1],
                   [0.4, 0.9]])
    prior = np.array([0.5, 0.5])
    lp = np.log(prior)
    obs = [0, 1]
    qs = infer_states([A1, A2], obs, lp)
    z = float((prior * A1[0, :] * A2[1, :]).sum())
    assert vfe(qs, [A1, A2], obs, lp) == pytest.approx(-np.log(z), abs=1e-12)


# ---- error paths ----------------------------------------------------------

def test_empty_modality_list_raises():
    with pytest.raises(ValueError, match="at least one likelihood"):
        infer_states([], 0, np.log(np.array([0.5, 0.5])))


def test_non_2d_likelihood_raises():
    with pytest.raises(ValueError, match="2-D"):
        infer_states(np.array([0.5, 0.5]), 0, np.log(np.array([0.5, 0.5])))


def test_mismatched_n_s_across_modalities_raises():
    A1 = np.array([[0.8, 0.2]])
    A2 = np.array([[0.8, 0.1, 0.1]])
    with pytest.raises(ValueError, match="same n_s"):
        infer_states([A1, A2], [0, 0], np.log(np.array([0.5, 0.5])))


def test_wrong_number_of_observations_raises():
    A1 = np.array([[0.8, 0.2], [0.2, 0.8]])
    A2 = np.array([[0.6, 0.4], [0.4, 0.6]])
    with pytest.raises(ValueError, match="expected 2 observation"):
        infer_states([A1, A2], [0], np.log(np.array([0.5, 0.5])))


def test_log_prior_wrong_length_raises():
    A = np.array([[0.8, 0.2], [0.2, 0.8]])
    with pytest.raises(ValueError, match="log_prior length"):
        infer_states(A, 0, np.log(np.array([0.3, 0.3, 0.4])))


def test_observation_index_out_of_range_raises():
    A = np.array([[0.8, 0.2], [0.2, 0.8]])
    with pytest.raises(ValueError, match="out of range"):
        infer_states(A, 5, np.log(np.array([0.5, 0.5])))


def test_vfe_wrong_length_raises():
    A = np.array([[0.8, 0.2], [0.2, 0.8]])
    qs = np.array([0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match="must equal n_s"):
        vfe(qs, A, 0, np.log(np.array([0.5, 0.5])))


def test_obs_as_numpy_integer_accepted():
    A = np.array([[0.8, 0.5, 0.1],
                  [0.2, 0.5, 0.9]])
    lp = np.log(np.array([0.2, 0.3, 0.5]))
    qs = infer_states(A, np.int64(0), lp)
    unnorm = np.array([0.2, 0.3, 0.5]) * A[0, :]
    assert qs == pytest.approx(unnorm / unnorm.sum(), abs=1e-12)


def test_vfe_is_strictly_positive_when_posterior_does_not_match_prior():
    """VFE = KL(q||p) - log Z is non-negative (Jensen inequality).

    The existing tests verify the minimiser equals -log Z and that random
    perturbations stay above the minimum. This test pins the non-negativity
    contract independently of the minimiser: any pmf that differs from the
    optimal posterior must have strictly higher VFE.
    """
    A = np.array([[0.8, 0.5, 0.1],
                  [0.2, 0.5, 0.9]])
    prior = np.array([0.2, 0.3, 0.5])
    lp = np.log(prior)
    o = 0

    qs_star = infer_states(A, o, lp)
    f_star = vfe(qs_star, A, o, lp)

    # A deliberately wrong posterior must have strictly higher VFE.
    not_minimiser = softmax(np.array([0.0, 0.0, 1.0]))
    assert vfe(not_minimiser, A, o, lp) > f_star + 1e-12, (
        "A non-optimal posterior must have strictly higher VFE than the minimiser"
    )

    # Over 30 random pmfs: VFE must always be >= f_star (non-negativity of KL).
    rng = np.random.default_rng(7)
    for trial in range(30):
        q = rng.dirichlet([1.0, 1.0, 1.0])
        f_q = vfe(q, A, o, lp)
        assert f_q >= f_star - 1e-12, (
            f"Trial {trial}: vfe(q)={f_q:.6f} < f_star={f_star:.6f} "
            f"(q={q}): VFE must be non-negative relative to the minimum"
        )
