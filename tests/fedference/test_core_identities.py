"""Core identity & robustness tests for the FedGVI bridge (no mocks).

These tests pin the synthesis: the robust FedGVI machinery must (a) recover the
Kullback-Leibler / standard-Bayes special case in the appropriate limit, which
is exactly Friston et al. (2024) belief-sharing, and (b) genuinely down-weight a
contaminated agent when robustness is engaged. All numbers are real
computations on small categorical distributions.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference import divergences as dv
from fedference import losses as ls
from fedference.aggregation import log_linear_pool, robust_aggregate
from fedference.belief_sharing import share_round
from fedference.generalized_bayes import cavity, generalized_posterior, softmax

# ---- divergence limits ---------------------------------------------------


def test_kl_is_zero_for_identical_distributions():
    p = np.array([0.2, 0.5, 0.3])
    assert dv.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_kl_is_nonnegative():
    q = np.array([0.7, 0.2, 0.1])
    p = np.array([0.2, 0.2, 0.6])
    assert dv.kl_divergence(q, p) > 0.0


def test_renyi_recovers_kl_as_alpha_to_one():
    q = np.array([0.6, 0.3, 0.1])
    p = np.array([0.2, 0.3, 0.5])
    kl = dv.kl_divergence(q, p)
    near = dv.renyi_divergence(q, p, alpha=1.0 + 1e-4)
    assert near == pytest.approx(kl, rel=1e-2)
    # inside the stability band the closed form is exactly KL
    assert dv.renyi_divergence(q, p, alpha=1.0) == pytest.approx(kl, abs=1e-12)


def test_alpha_renyi_is_the_fedgvi_normalization():
    q = np.array([0.6, 0.3, 0.1])
    p = np.array([0.2, 0.3, 0.5])
    assert dv.alpha_renyi_divergence(q, p, 0.5) == pytest.approx(dv.renyi_divergence(q, p, 0.5) / 0.5)
    assert dv.alpha_renyi_divergence(q, p, 1.0) == pytest.approx(dv.kl_divergence(q, p))


def test_divergence_dispatch_matches_named():
    q = np.array([0.5, 0.5])
    p = np.array([0.9, 0.1])
    assert dv.divergence("KLD", q, p) == pytest.approx(dv.kl_divergence(q, p))
    assert dv.divergence("RKL", q, p) == pytest.approx(dv.kl_divergence(p, q))


# ---- loss limits ---------------------------------------------------------


def test_rcce_recovers_nll_as_q_loss_to_zero():
    p = np.array([0.1, 0.7, 0.2])
    assert ls.rcce(p, 1, q_loss=1e-4) == pytest.approx(ls.nll(p, 1), rel=1e-3)


def test_beta_loss_recovers_nll_as_beta_to_zero():
    p = np.array([0.1, 0.7, 0.2])
    assert ls.beta_loss(p, 1, beta=1e-5) == pytest.approx(ls.nll(p, 1), rel=1e-2)


def test_rcce_is_bounded_unlike_nll():
    # a confidently-wrong observation: NLL explodes, rcce(q_loss=1) stays <= 1
    p = np.array([0.999, 0.0005, 0.0005])
    assert ls.nll(p, 2) > 5.0
    assert ls.rcce(p, 2, q_loss=1.0) <= 1.0


# ---- M2: off-switch-point convergence witness -----------------------------
# rcce/beta_loss/renyi_divergence each switch to an exact closed form inside a
# small band around their limit point (q < 1e-9, beta < 1e-9,
# |alpha - 1| < 1e-6) for numerical stability. A residual measured *inside*
# that band is a code-branch equality, not evidence of convergence. These
# tests evaluate strictly *outside* the band and require the gap to shrink
# monotonically toward zero as the parameter approaches the limit — genuine
# numerical convergence of the general formula, not a branch coincidence.


def test_rcce_offswitch_residual_shrinks_toward_the_limit():
    p = np.array([0.1, 0.7, 0.2])
    nll_val = ls.nll(p, 1)
    gaps = [abs(ls.rcce(p, 1, q_loss=q_loss) - nll_val) for q_loss in (1e-4, 1e-5, 1e-6)]
    assert gaps[0] > gaps[1] > gaps[2] > 0.0


def test_beta_loss_offswitch_residual_shrinks_toward_the_limit():
    p = np.array([0.1, 0.7, 0.2])
    nll_val = ls.nll(p, 1)
    gaps = [abs(ls.beta_loss(p, 1, beta=b) - nll_val) for b in (1e-4, 1e-5, 1e-6)]
    assert gaps[0] > gaps[1] > gaps[2] > 0.0


def test_renyi_offswitch_residual_shrinks_toward_the_limit():
    q = np.array([0.6, 0.3, 0.1])
    p = np.array([0.2, 0.3, 0.5])
    kl = dv.kl_divergence(q, p)
    # 1e-6 sits at the float64 precision floor for this pair (gap rounds to
    # exactly 0.0), so this uses a coarser but still off-band offset ladder.
    gaps = [abs(dv.renyi_divergence(q, p, alpha=1.0 + d) - kl) for d in (1e-3, 1e-4, 1e-5)]
    assert gaps[0] > gaps[1] > gaps[2] > 0.0


# ---- generalized Bayes ---------------------------------------------------


def test_generalized_posterior_kl_nll_is_exact_bayes():
    # KL divergence + NLL loss must equal closed-form Bayes: prior * likelihood
    log_prior = np.log(np.array([0.5, 0.3, 0.2]))
    likelihood = np.array(
        [
            [0.8, 0.2, 0.1],  # p(o=0 | s)
            [0.2, 0.8, 0.9],
        ]
    )  # p(o=1 | s)
    o = 0
    loss_vec = ls.loss_vector(likelihood, o, loss="nll")
    q = generalized_posterior(log_prior, loss_vec, divergence="KLD")
    exact = np.array([0.5, 0.3, 0.2]) * likelihood[o]
    exact = exact / exact.sum()
    assert np.allclose(q, exact, atol=1e-9)


def test_cavity_removes_a_factor():
    posterior = np.array([0.6, 0.3, 0.1])
    factor = np.array([0.5, 0.4, 0.1])
    cav = cavity(posterior, factor)
    # re-multiplying the factor into the cavity restores the posterior
    restored = softmax(np.log(cav) + np.log(factor))
    assert np.allclose(restored, posterior, atol=1e-9)


# ---- aggregation: the central identity -----------------------------------


def test_log_linear_pool_is_product_of_experts():
    a = np.array([0.7, 0.2, 0.1])
    b = np.array([0.2, 0.7, 0.1])
    pooled = log_linear_pool([a, b])
    manual = a * b
    manual = manual / manual.sum()
    assert np.allclose(pooled, manual, atol=1e-9)


def test_robust_aggregate_recovers_naive_at_zero_robustness():
    beliefs = [np.array([0.7, 0.2, 0.1]), np.array([0.6, 0.3, 0.1]), np.array([0.1, 0.1, 0.8])]
    naive = log_linear_pool(beliefs)
    res = robust_aggregate(beliefs, robustness=0.0)
    assert np.allclose(res.consensus, naive, atol=1e-12)
    assert res.iterations == 0


def test_robust_aggregate_downweights_the_outlier():
    # two agents agree on state 0; one confidently-wrong agent insists on state 2
    good_a = np.array([0.85, 0.10, 0.05])
    good_b = np.array([0.80, 0.15, 0.05])
    liar = np.array([0.02, 0.03, 0.95])
    naive = log_linear_pool([good_a, good_b, liar])
    res = robust_aggregate([good_a, good_b, liar], robustness=5.0)
    # robust consensus should place MORE mass on the true state 0 than naive
    assert res.consensus[0] > naive[0]
    # the liar should carry the smallest influence weight
    assert np.argmin(res.agent_weights) == 2


# ---- belief sharing bridge ----------------------------------------------


def test_share_round_self_exclusion_changes_per_agent_consensus():
    beliefs = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    diag = share_round(beliefs, method="naive", exclude_self=True, true_state=0)
    # with self-exclusion the three agents do NOT all land on the same belief
    assert not np.allclose(diag.shared_beliefs[0], diag.shared_beliefs[2], atol=1e-6)
    assert 0.0 <= diag.mean_accuracy <= 1.0


def test_robust_sharing_beats_naive_under_a_contaminated_sentinel():
    # ground truth is state 0; two honest agents + one adversary broadcasting state 2
    beliefs = np.array([[0.75, 0.15, 0.10], [0.70, 0.20, 0.10], [0.02, 0.03, 0.95]])
    naive = share_round(beliefs, method="naive", exclude_self=False, true_state=0)
    robust = share_round(beliefs, method="robust", robustness=5.0, exclude_self=False, true_state=0)
    assert robust.mean_accuracy > naive.mean_accuracy
    assert robust.mean_surprise < naive.mean_surprise
