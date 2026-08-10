"""Edge-path / contract tests for the hand-authored FedGVI core.

The identity suite (test_fedference_core.py) pins the *mathematics*; this file
pins the *contract* — validation raises, named-dispatch branches, the alpha!=1
Renyi/AR paths, robust-aggregate weight handling, and the belief-sharing
degenerate branches — so the core modules are covered to the same standard as
the workflow-authored breadth. Still no mocks: every assertion is a real call.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference import divergences as dv
from fedference import losses as ls
from fedference.aggregation import (
    aggregate,
    log_linear_pool,
    robust_aggregate,
    variational_aggregate,
)
from fedference.belief_sharing import share_round
from fedference.generalized_bayes import generalized_posterior, update_factor

# ---- divergences: validation + alternate branches ------------------------


def test_as_pmf_rejects_empty():
    with pytest.raises(ValueError):
        dv.kl_divergence(np.array([]), np.array([]))


def test_as_pmf_rejects_negative_entries():
    with pytest.raises(ValueError):
        dv.kl_divergence(np.array([-0.5, 1.5]), np.array([0.5, 0.5]))


def test_renyi_alpha_not_one_branch_and_total_variation():
    q = np.array([0.7, 0.3])
    p = np.array([0.4, 0.6])
    # alpha far from 1 exercises the non-KL closed form
    assert dv.renyi_divergence(q, p, alpha=0.5) > 0.0
    assert 0.0 <= dv.total_variation(q, p) <= 1.0
    assert dv.divergence("TV", q, p) == pytest.approx(dv.total_variation(q, p))


def test_divergence_ar_requires_param_and_rejects_unknown():
    q = np.array([0.5, 0.5])
    p = np.array([0.2, 0.8])
    assert dv.divergence("AR", q, p, param=0.5) == pytest.approx(dv.alpha_renyi_divergence(q, p, 0.5))
    with pytest.raises(ValueError):
        dv.divergence("AR", q, p)  # missing alpha
    with pytest.raises(ValueError):
        dv.divergence("NOPE", q, p)


# ---- losses: validation + branches ---------------------------------------


def test_rcce_rejects_out_of_range_and_small_q_loss_is_nll():
    p = np.array([0.2, 0.8])
    with pytest.raises(ValueError):
        ls.rcce(p, 0, q_loss=1.5)
    assert ls.rcce(p, 0, q_loss=0.0) == pytest.approx(ls.nll(p, 0))


def test_rcce_deprecated_q_alias_is_explicit_and_unambiguous():
    p = np.array([0.2, 0.8])
    with pytest.deprecated_call(match="use rcce\\(q_loss=...\\)"):
        legacy = ls.rcce(p, 0, q=0.5)
    assert legacy == pytest.approx(ls.rcce(p, 0, q_loss=0.5))
    with pytest.raises(TypeError, match="q_loss or deprecated q"):
        ls.rcce(p, 0, q_loss=0.5, q=0.5)
    with pytest.raises(TypeError, match="requires q_loss"):
        ls.rcce(p, 0)


def test_beta_loss_rejects_negative():
    with pytest.raises(ValueError):
        ls.beta_loss(np.array([0.5, 0.5]), 0, beta=-0.1)


def test_loss_vector_validates_shape_and_loss_name():
    a = np.array([[0.8, 0.2], [0.2, 0.8]])
    assert ls.loss_vector(a, 0, loss="rcce", param=0.5).shape == (2,)
    assert ls.loss_vector(a, 0, loss="beta", param=0.3).shape == (2,)
    with pytest.raises(ValueError):
        ls.loss_vector(np.array([1.0, 0.0]), 0)  # not 2-D
    with pytest.raises(ValueError):
        ls.loss_vector(a, 0, loss="bogus")


# ---- generalized Bayes: AR path, shape guard, factor update --------------


def test_generalized_posterior_ar_path_and_shape_guard():
    log_prior = np.log(np.array([0.5, 0.5]))
    loss_vec = np.array([0.1, 2.0])
    out = generalized_posterior(log_prior, loss_vec, divergence="AR", alpha=0.7)
    assert out.shape == (2,) and np.isclose(out.sum(), 1.0)
    with pytest.raises(ValueError):
        generalized_posterior(log_prior, np.array([0.1, 0.2, 0.3]))


def test_generalized_posterior_ar_is_not_the_standard_renyi_shortcut():
    """The AR update is tied to the named objective, including alpha > 1 faces."""
    log_prior = np.log(np.array([0.5, 0.3, 0.2]))
    loss_vec = np.array([0.1, 2.0, 0.7])
    q = generalized_posterior(log_prior, loss_vec, divergence="AR", alpha=2.0)
    prior = np.exp(log_prior)
    prior /= prior.sum()
    objective = dv.alpha_renyi_divergence(q, prior, 2.0) + float(q @ loss_vec)
    shortcut = np.exp(2.0 * log_prior - loss_vec)
    shortcut /= shortcut.sum()
    shortcut_objective = dv.alpha_renyi_divergence(shortcut, prior, 2.0) + float(shortcut @ loss_vec)
    assert np.isclose(q.sum(), 1.0)
    assert objective < shortcut_objective


def test_generalized_posterior_rejects_nonfinite_controls():
    args = (np.log(np.array([0.5, 0.5])), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="learning_rate"):
        generalized_posterior(*args, learning_rate=np.nan)
    with pytest.raises(ValueError, match="alpha"):
        generalized_posterior(*args, divergence="AR", alpha=0.0)


def test_update_factor_roundtrip():
    """PVI identity: re-multiplying the refreshed factor onto the cavity of the
    OLD posterior recovers the NEW posterior exactly. This binds update_factor to
    its named property (the previous version only asserted the pmf sums to 1, a
    softmax triviality that established nothing)."""
    from fedference.generalized_bayes import cavity, softmax

    old_factor = np.array([0.4, 0.6])
    old_post = np.array([0.5, 0.5])
    new_post = np.array([0.7, 0.3])
    t_new = update_factor(old_factor, old_post, new_post)
    assert np.isclose(t_new.sum(), 1.0)
    # Cavity of the old posterior (old factor removed), re-multiplied by the
    # refreshed factor, must reconstruct the new posterior — the PVI identity.
    q_cavity = cavity(old_post, old_factor)
    reconstructed = softmax(np.log(q_cavity) + np.log(t_new))
    assert np.allclose(reconstructed, new_post, atol=1e-12)
    # Negative control: a WRONG factor must NOT reconstruct new_post, so the
    # identity above is a real constraint, not vacuously true.
    wrong = softmax(np.log(q_cavity) + np.log(np.array([0.5, 0.5])))
    assert not np.allclose(wrong, new_post, atol=1e-3)


# ---- aggregation: weights, validation, dispatch, non-convergence ---------


def test_log_linear_pool_honours_weights():
    a = np.array([0.9, 0.1])
    b = np.array([0.1, 0.9])
    # zero weight on b -> consensus follows a
    pooled = log_linear_pool([a, b], weights=[1.0, 0.0])
    assert pooled[0] > pooled[1]


def test_aggregation_rejects_all_zero_weights():
    beliefs = [np.array([0.8, 0.2]), np.array([0.3, 0.7])]
    with pytest.raises(ValueError, match="at least one positive"):
        log_linear_pool(beliefs, weights=[0.0, 0.0])
    with pytest.raises(ValueError, match="at least one positive"):
        robust_aggregate(beliefs, weights=[0.0, 0.0], robustness=1.0)


def test_aggregation_rejects_nonfinite_controls_and_terminal_weights_match_consensus():
    beliefs = [
        np.array([0.8, 0.1, 0.1]),
        np.array([0.75, 0.15, 0.1]),
        np.array([0.05, 0.05, 0.9]),
    ]
    with pytest.raises(ValueError, match="robustness"):
        robust_aggregate(beliefs, robustness=np.nan)
    with pytest.raises(ValueError, match="max_iter"):
        robust_aggregate(beliefs, robustness=1.0, max_iter=-1)
    with pytest.raises(ValueError, match="tol"):
        robust_aggregate(beliefs, robustness=1.0, tol=np.nan)
    result = robust_aggregate(beliefs, robustness=3.0, max_iter=1)
    expected = np.array([np.exp(-3.0 * dv.kl_divergence(belief, result.consensus)) for belief in beliefs])
    expected /= expected.sum()
    assert np.allclose(result.agent_weights, expected)


def test_aggregate_validation_and_dispatch():
    beliefs = [np.array([0.8, 0.2]), np.array([0.7, 0.3])]
    assert np.allclose(aggregate(beliefs, method="naive"), log_linear_pool(beliefs))
    assert aggregate(beliefs, method="robust").shape == (2,)
    with pytest.raises(ValueError):
        aggregate(beliefs, method="bogus")
    with pytest.raises(ValueError):
        robust_aggregate(beliefs, robustness=-1.0)
    with pytest.raises(ValueError):
        log_linear_pool(beliefs, weights=[1.0])  # wrong length
    with pytest.raises(ValueError):
        log_linear_pool(beliefs, weights=[1.0, -2.0])  # negative


def test_robust_aggregate_reports_iterations_when_engaged():
    beliefs = [np.array([0.8, 0.1, 0.1]), np.array([0.75, 0.15, 0.1]), np.array([0.05, 0.05, 0.9])]
    res = robust_aggregate(beliefs, robustness=3.0, max_iter=50)
    assert res.iterations >= 1
    assert len(res.history) >= 2
    assert np.isclose(res.agent_weights.sum(), 1.0)


# ---- belief sharing: degenerate branches ---------------------------------


def test_share_round_single_agent_and_no_truth():
    diag = share_round(np.array([[0.6, 0.4]]), method="naive")
    assert np.isnan(diag.mean_accuracy)  # true_state=None branch
    assert diag.shared_beliefs.shape == (1, 2)


def test_share_round_rejects_unknown_method():
    with pytest.raises(ValueError):
        share_round(np.array([[0.6, 0.4], [0.5, 0.5]]), method="bogus", true_state=0)


def test_share_round_rejects_invalid_states_and_belief_mass():
    beliefs = np.array([[0.6, 0.4], [0.5, 0.5]])
    with pytest.raises(ValueError, match="true_state"):
        share_round(beliefs, true_state=2)
    with pytest.raises(ValueError, match="true_state"):
        share_round(beliefs, true_state=-1)
    with pytest.raises(ValueError, match="finite"):
        share_round([[0.6, np.nan], [0.5, 0.5]])
    with pytest.raises(ValueError, match="non-negative"):
        share_round([[0.6, -0.1], [0.5, 0.5]])


# ---- additional numerical edge cases (Tier-1 hardening) ------------------


def test_renyi_alpha_to_zero_weights_the_reverse_direction():
    # As alpha -> 0+ the alpha-Renyi divergence stays finite and non-negative
    # (it weights the support of p), distinct from the alpha -> 1 KL value.
    q = np.array([0.6, 0.3, 0.1])
    p = np.array([0.2, 0.3, 0.5])
    d_small = dv.renyi_divergence(q, p, alpha=1e-3)
    assert np.isfinite(d_small)
    assert d_small >= 0.0
    assert d_small != pytest.approx(dv.kl_divergence(q, p), rel=1e-2)


def test_renyi_alpha_large_is_finite_and_dominated_by_max_ratio():
    # As alpha -> infinity the alpha-Renyi divergence approaches log max_k q_k/p_k;
    # it must stay finite and non-negative for well-supported pmfs.
    q = np.array([0.7, 0.2, 0.1])
    p = np.array([0.2, 0.3, 0.5])
    d_big = dv.renyi_divergence(q, p, alpha=50.0)
    assert np.isfinite(d_big)
    assert d_big >= 0.0
    # monotone-ish: a larger alpha emphasises the worst-case ratio, so >= a mid alpha
    assert d_big >= dv.renyi_divergence(q, p, alpha=2.0) - 1e-9


def test_degenerate_one_state_pmf_has_zero_divergence_and_pool():
    one = np.array([1.0])
    assert dv.kl_divergence(one, one) == pytest.approx(0.0, abs=1e-12)
    # a one-state log-linear pool is the (only) certainty
    assert np.allclose(log_linear_pool([one, one]), one, atol=1e-12)


def test_all_mass_on_one_state_into_robust_and_variational_aggregate():
    # An agent placing (almost) all mass on one state is a valid, extreme input;
    # both aggregators must return a normalized pmf without overflow/NaN.
    sharp = np.array([1.0 - 2e-12, 1e-12, 1e-12])
    soft = np.array([0.4, 0.35, 0.25])
    r = robust_aggregate([soft, sharp], robustness=2.0)
    v = variational_aggregate([soft, sharp], robustness=2.0)
    for consensus in (r.consensus, v.consensus):
        assert consensus.shape == (3,)
        assert np.all(np.isfinite(consensus))
        assert consensus.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(consensus >= 0.0)


def test_single_agent_robust_aggregate_is_that_agent():
    # With one agent the robust pool is exactly that agent (nothing to down-weight).
    belief = np.array([0.6, 0.3, 0.1])
    res = robust_aggregate([belief], robustness=3.0)
    assert np.allclose(res.consensus, belief, atol=1e-9)
    assert res.agent_weights.shape == (1,)
    assert res.agent_weights[0] == pytest.approx(1.0, abs=1e-12)


def test_aggregate_dispatch_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        aggregate([np.array([0.5, 0.5])], method="bogus")


# ---- edge cases added to close audit gaps --------------------------------


def test_log_linear_pool_same_beliefs_sharpens():
    """Product-of-experts with identical beliefs amplifies, not recovers, the input.

    With n identical beliefs log-linear pooling raises each p_i to the n-th power
    (then renormalizes), so the dominant state gains mass. The result is strictly
    sharper than the single belief (higher peak), not identical to it.
    """
    p = np.array([0.6, 0.4])
    result = log_linear_pool([p.copy(), p.copy()])
    # The modal state (index 0) should be more probable after pooling.
    assert result[0] > p[0], (
        f"Pooling identical beliefs should sharpen: p0={p[0]:.4f}, pooled={result[0]:.4f}"
    )
    assert np.isclose(result.sum(), 1.0, atol=1e-12)


def test_robust_aggregate_all_agents_vetoed_fallback():
    """robust_aggregate falls back to uniform base weights when all agents are vetoed.

    With robustness=1e12 and highly polarized beliefs the exp(-c*KL) factor
    underflows for every agent, eff.sum() falls below _EPS, and the code
    falls back to base weights (line 128). The result must still be a valid pmf.
    """
    # Two agents with maximally opposed beliefs force every cross-entropy so high
    # that exp(-1e9 * KL) underflows to zero for both.
    a = np.array([1.0 - 1e-9, 1e-9])
    b = np.array([1e-9, 1.0 - 1e-9])
    res = robust_aggregate([a, b], robustness=1e9, max_iter=5)
    assert np.isfinite(res.consensus).all(), "consensus must be finite"
    assert res.consensus.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(res.consensus >= 0.0)
    assert res.fallback_events
    assert res.converged is False


def test_aggregation_free_energy_rejects_negative_agent_weights():
    """aggregation_free_energy raises ValueError for a negative agent weight."""
    from fedference.aggregation import aggregation_free_energy

    beliefs = [np.array([0.6, 0.4]), np.array([0.5, 0.5])]
    with pytest.raises(ValueError, match="non-negative"):
        aggregation_free_energy(
            np.array([0.55, 0.45]),
            np.array([-0.1, 1.1]),  # negative entry
            beliefs,
            robustness=1.0,
        )


def test_variational_aggregate_max_iter_zero_returns_valid_consensus():
    """variational_aggregate with max_iter=0 must not raise IndexError.

    When the inner loop body never executes, fe_history is empty. The
    multi-start comparison must guard against indexing an empty list.
    The returned consensus should be a valid pmf (the seed consensus,
    i.e. the log-linear pool).
    """
    beliefs = [np.array([0.7, 0.3]), np.array([0.6, 0.4])]
    res = variational_aggregate(beliefs, robustness=1.0, max_iter=0)
    assert res.consensus.shape == (2,)
    assert np.isfinite(res.consensus).all()
    assert res.consensus.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(res.consensus >= 0.0)
    # fe_history is empty because no iteration ran
    assert res.free_energy_history == []


# ---- security-critical path: robust + exclude_self=True ------------------


def test_share_round_robust_with_exclude_self_isolates_contaminated_agent():
    """robust + exclude_self=True: the combined path exercises a unique code branch.

    Existing tests use either robust+exclude_self=False or naive+exclude_self=True;
    the combined path (robust+exclude_self=True) is never exercised. This test
    pins the structural contracts — valid pmf output, correct shape, different
    consensus per agent — without asserting accuracy direction, because with only
    3 agents and 1 contaminated the self-excluding pool is 1 honest + 1
    contaminated (1:1 ratio), where no method can guarantee accuracy > 0.5.
    """
    # Agent 2 is contaminated: almost certain about the wrong state (state 2).
    beliefs = np.array(
        [
            [0.80, 0.10, 0.10],  # honest
            [0.75, 0.15, 0.10],  # honest
            [0.02, 0.03, 0.95],  # contaminated
        ]
    )
    robust_excl = share_round(beliefs, method="robust", robustness=5.0, exclude_self=True, true_state=0)
    robust_incl = share_round(beliefs, method="robust", robustness=5.0, exclude_self=False, true_state=0)

    # --- structural contract: output is a valid belief matrix ---
    np.testing.assert_allclose(
        robust_excl.shared_beliefs.sum(axis=1),
        1.0,
        atol=1e-12,
        err_msg="rows of shared_beliefs must sum to 1.0",
    )
    assert robust_excl.shared_beliefs.shape == (3, 3)
    assert np.all(robust_excl.shared_beliefs >= 0.0)

    # --- the combined path changes the result from exclude_self=False ---
    # With self-exclusion, agent 2 (contaminated) excludes itself from its own
    # consensus, so its view must differ from the include-self version.
    assert not np.allclose(robust_excl.shared_beliefs[2], robust_incl.shared_beliefs[2], atol=1e-6), (
        "Contaminated agent 2 self-excluded result must differ from non-excluded"
    )

    # --- contaminated agent 2 gets near-zero weight in both cases ---
    # (it is down-weighted by robust regardless of self-exclusion)
    assert robust_excl.agent_weights[2] < 0.01, (
        f"Contaminated agent 2 weight should be near-zero, got {robust_excl.agent_weights[2]:.4f}"
    )

    # --- mean_accuracy is a valid probability ---
    assert 0.0 <= robust_excl.mean_accuracy <= 1.0


def test_share_round_variational_dispatches_objective_backed_rule():
    beliefs = np.array([[0.82, 0.12, 0.06], [0.78, 0.16, 0.06], [0.05, 0.05, 0.90]])
    from fedference.aggregation import variational_aggregate

    shared = share_round(
        beliefs,
        method="variational",
        robustness=1.5,
        exclude_self=False,
        true_state=0,
    )
    reference = variational_aggregate(beliefs, robustness=1.5)
    assert np.allclose(shared.consensus, reference.consensus)
    assert np.allclose(shared.agent_weights, reference.agent_weights)
    assert np.isclose(shared.mean_accuracy, shared.consensus[0])
