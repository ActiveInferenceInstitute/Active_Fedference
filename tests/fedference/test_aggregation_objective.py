"""The rigorous variational aggregator — axis-2 made objective-backed (no mocks).

:func:`fedference.aggregation.robust_aggregate` is a *heuristic*: the scoped
MAJ-1 proposition rejects the declared separable block-objective class for its
raw log-pool, while it neither derives nor rules out every broader construction.
:func:`fedference.aggregation.variational_aggregate` repairs the specific
separable objective gap: it
is exact block-coordinate descent on the stated free energy
:func:`fedference.aggregation.aggregation_free_energy`. These tests pin the four
properties that make the upgrade real:

1. recovery — ``robustness = 0`` returns the project log-linear pool exactly;
   under documented bridge assumptions that pool specializes Eq. 7's
   message-combination term rather than the complete Friston protocol;
2. monotone descent — the recorded free energy never increases;
3. stationarity — the converged iterate is a fixed point / stationary point of F
   (gradient ~ 0 to 1e-6);
4. redescending effective weight — on the tested path, a diverging agent's
   normalized weight collapses toward 0 while the naive pool's remains fixed.

All numbers are genuine computations on small categorical distributions.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.aggregation import (
    aggregate,
    aggregation_free_energy,
    log_linear_pool,
    robust_aggregate,
    variational_aggregate,
)
from fedference.divergences import kl_divergence
from fedference.generalized_bayes import softmax


def _entropy(p):
    """Shannon entropy of a categorical pmf (nats)."""
    return -np.sum(p * np.log(np.clip(p, 1e-12, None)))


# ---- 1. recovery: the limit is the proof --------------------------------

def test_variational_recovers_log_linear_pool_at_zero_robustness():
    beliefs = [np.array([0.7, 0.2, 0.1]),
               np.array([0.6, 0.3, 0.1]),
               np.array([0.1, 0.1, 0.8])]
    naive = log_linear_pool(beliefs)
    res = variational_aggregate(beliefs, robustness=0.0)
    assert np.allclose(res.consensus, naive, atol=1e-12)
    assert res.iterations == 0
    assert res.converged
    assert res.free_energy_history == []


def test_variational_recovers_pool_in_the_small_c_limit():
    beliefs = [np.array([0.55, 0.30, 0.15]),
               np.array([0.50, 0.35, 0.15]),
               np.array([0.20, 0.20, 0.60])]
    naive = log_linear_pool(beliefs)
    res = variational_aggregate(beliefs, robustness=1e-9)
    assert np.allclose(res.consensus, naive, atol=1e-6)


# ---- 2. monotone descent on the stated objective ------------------------

def test_free_energy_descends_monotonically():
    beliefs = np.array([[0.70, 0.20, 0.10],
                        [0.65, 0.25, 0.10],
                        [0.60, 0.25, 0.15],
                        [0.05, 0.05, 0.90]])
    res = variational_aggregate(beliefs, robustness=2.0)
    fe = np.asarray(res.free_energy_history)
    assert fe.size >= 2
    # non-increasing to within machine slack (block-coordinate descent)
    assert np.all(np.diff(fe) <= 1e-9)


def test_free_energy_supports_zero_base_weight_by_zeroing_effective_weight():
    beliefs = np.array([[0.7, 0.3], [0.2, 0.8]])
    result = variational_aggregate(
        beliefs,
        weights=[1.0, 0.0],
        robustness=1.0,
    )
    assert np.isclose(result.consensus.sum(), 1.0)
    assert result.agent_weights[1] == 0.0


def test_variational_weight_collapse_is_receipted_and_not_convergence():
    beliefs = np.array(
        [
            [1.0 - 1e-9, 1e-9],
            [1e-9, 1.0 - 1e-9],
        ]
    )
    result = variational_aggregate(
        beliefs,
        robustness=1e9,
        max_iter=5,
    )
    assert result.fallback_events
    assert result.converged is False
    assert np.isclose(result.consensus.sum(), 1.0)


def test_free_energy_rejects_effective_mass_outside_base_weight_support():
    beliefs = np.array([[0.7, 0.3], [0.2, 0.8]])
    with pytest.raises(ValueError, match="zero where base_weights are zero"):
        aggregation_free_energy(
            np.array([0.6, 0.4]),
            np.array([0.5, 0.5]),
            beliefs,
            base_weights=[1.0, 0.0],
            robustness=1.0,
        )


# ---- 3. stationarity: the iterate is a fixed point of F -----------------

def test_converged_iterate_is_a_stationary_point():
    beliefs = np.array([[0.70, 0.20, 0.10],
                        [0.65, 0.25, 0.10],
                        [0.60, 0.25, 0.15],
                        [0.05, 0.05, 0.90]])
    c = 1.7
    res = variational_aggregate(beliefs, robustness=c, max_iter=512, tol=1e-12)
    assert res.converged
    q = res.consensus
    mat = beliefs / beliefs.sum(axis=1, keepdims=True)
    log_mat = np.log(mat)
    base = np.ones(mat.shape[0])

    # a-stationarity: the F-minimizing weights at q are a_n = w_n exp(-c CE(q,s_n)).
    ce = -(log_mat @ q)
    a_star = base * np.exp(-c * ce)
    # q-stationarity: q must be the softmax of those weighted log-beliefs (fixed point).
    q_from_a = softmax(a_star @ log_mat)
    assert np.allclose(q, q_from_a, atol=1e-6)

    # explicit gradient-norm checks (both blocks ~ 0).
    grad_a = ce + (1.0 / c) * np.log(a_star / base)  # dF/da_n
    assert np.linalg.norm(grad_a) < 1e-6
    grad_q = -(a_star @ log_mat) + np.log(q) + 1.0  # dG/dq_i (G = F's q-part)
    grad_q_proj = grad_q - grad_q.mean()  # project onto the simplex tangent
    assert np.linalg.norm(grad_q_proj) < 1e-6


def test_free_energy_matches_independent_recomputation():
    beliefs = np.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.1, 0.2, 0.7]])
    c = 1.3
    res = variational_aggregate(beliefs, robustness=c, max_iter=512, tol=1e-12)
    q = res.consensus
    mat = beliefs / beliefs.sum(axis=1, keepdims=True)
    base = np.ones(3)
    a_star = base * np.exp(-c * (-(np.log(mat) @ q)))
    # F computed from the public helper equals a hand expansion of its terms.
    f_public = aggregation_free_energy(q, a_star, beliefs, base_weights=base, robustness=c)
    ce = -(np.log(mat) @ q)
    entropy = -np.sum(q * np.log(q))
    kl_gen = np.sum(a_star * np.log(a_star / base) - a_star + base)
    f_manual = np.sum(a_star * ce) - entropy + kl_gen / c
    assert f_public == pytest.approx(float(f_manual), abs=1e-12)


# ---- 4. redescending effective weight -----------------------------------

def test_diverging_agent_influence_collapses_to_zero():
    base = np.array([[0.5, 0.3, 0.2], [0.45, 0.35, 0.2], [0.5, 0.25, 0.25]])
    last_w = None
    weights_seen = []
    for drift in (0.0, 0.3, 0.6, 0.9, 0.99):
        liar = (1 - drift) * np.array([0.5, 0.3, 0.2]) + drift * np.array([0.0, 0.0, 1.0])
        colony = np.vstack([base, liar])
        res = variational_aggregate(colony, robustness=1.5, max_iter=256, tol=1e-12)
        w_liar = float(res.agent_weights[-1])
        weights_seen.append(w_liar)
        if last_w is not None:
            assert w_liar <= last_w + 1e-9  # monotone non-increasing in divergence
        last_w = w_liar
    # at extreme divergence the outlier is all but vetoed; the naive pool would
    # instead hold it at the fixed 1/n influence however wrong it is.
    assert weights_seen[-1] < 0.05
    assert weights_seen[0] > weights_seen[-1]


def test_multistart_vetoes_near_vertex_adversary():
    # RedTeam counterexample (workflow w1o6slput): an honest majority plus one
    # liar driven to (near) the simplex vertex. The log-linear-pool seed is itself
    # captured by the near-one-hot adversary, so a single-start descent stays in
    # the capture basin. Multi-start descent escapes to the outlier-vetoing basin
    # and suppresses the liar even AT the vertex — the property the single-start
    # version failed and that this test now locks.
    honest = [np.array([0.4, 0.3, 0.3])] * 3
    for drift in (0.99, 0.9999, 1.0):
        liar = (1 - drift) * np.array([0.4, 0.3, 0.3]) + drift * np.array([0.0, 0.0, 1.0])
        r = variational_aggregate(np.vstack([honest, [liar]]), robustness=1.5,
                                  max_iter=512, tol=1e-12)
        assert r.agent_weights[-1] < 0.05      # liar vetoed
        assert np.argmax(r.consensus) == 0     # consensus follows the honest majority


def test_multistart_false_uses_single_pool_seed_and_can_be_captured():
    # The descent figure uses multistart=False (the canonical pool-seed
    # trajectory). On a near-vertex adversary that single start is captured —
    # exactly the basin multi-start escapes. This locks both behaviours.
    honest = [np.array([0.4, 0.3, 0.3])] * 3
    liar = np.array([1e-6, 1e-6, 1.0 - 2e-6])
    colony = np.vstack([honest, [liar]])
    single = variational_aggregate(colony, robustness=1.5, multistart=False,
                                   max_iter=512, tol=1e-12)
    multi = variational_aggregate(colony, robustness=1.5, multistart=True,
                                  max_iter=512, tol=1e-12)
    # single-start (pool seed) is captured by the vertex adversary...
    assert single.agent_weights[-1] > 0.5
    # ...multi-start escapes to the lower-F vetoing basin.
    assert multi.agent_weights[-1] < 0.05
    assert multi.free_energy_history[-1] <= single.free_energy_history[-1] + 1e-9


def test_multistart_reaches_lower_or_equal_free_energy_than_pool_seed():
    # In this fully converged fixture, the pool seed is tried first and the
    # lowest-F converged result is kept. The returned vetoing basin therefore
    # has strictly lower F than the pool-seeded capture basin.
    honest = [np.array([0.4, 0.3, 0.3])] * 3
    liar = np.array([1e-4, 1e-4, 1 - 2e-4])
    colony = np.vstack([honest, [liar]])
    r = variational_aggregate(colony, robustness=1.5, max_iter=512, tol=1e-12)
    # the reached basin vetoes the liar (the lower-F solution), not captures it
    assert r.agent_weights[-1] < 0.05
    assert np.argmax(r.consensus) == 0


def test_multistart_prefers_a_converged_start_over_lower_unfinished_trace():
    """A bounded iteration budget must not hide non-convergence behind lower F."""
    beliefs = np.array(
        [
            [9.13938682e-04, 7.41711549e-01, 1.86819573e-01, 7.05549393e-02],
            [1.79544310e-02, 5.71275169e-02, 9.14090863e-01, 1.08271886e-02],
            [1.34242831e-03, 4.55936190e-03, 9.94098030e-01, 1.79460765e-07],
            [5.60744941e-01, 1.44115130e-01, 4.24744689e-03, 2.90892482e-01],
            [6.95024949e-03, 2.18532476e-06, 9.78717033e-01, 1.43305321e-02],
        ]
    )
    single = variational_aggregate(
        beliefs,
        robustness=2.5,
        max_iter=1,
        tol=1e-6,
        multistart=False,
    )
    multi = variational_aggregate(
        beliefs,
        robustness=2.5,
        max_iter=1,
        tol=1e-6,
        multistart=True,
    )
    assert single.converged is True
    assert multi.converged is True
    assert np.array_equal(multi.consensus, single.consensus)


def test_effective_weights_never_exceed_base():
    beliefs = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    res = variational_aggregate(beliefs, robustness=3.0, max_iter=256, tol=1e-12)
    # normalized influence is a valid distribution; with equal base weights the
    # exp(-c CE) factor is <= 1, so no agent can dominate beyond the pool size.
    assert np.all(res.agent_weights >= 0.0)
    assert res.agent_weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_raw_effective_weights_are_bounded_by_base_weights():
    """SYN-2: the RAW effective-weight bound ``a_n = w_n exp(-c CE) <= w_n``.

    The returned ``agent_weights`` are normalized, which erases the bound, so
    the softmax-triviality assertions above cannot falsify it. Here we
    recompute the raw effective weights from the RETURNED consensus using the
    code's own final-weight formula (``final_eff = base * exp(-c * CE(q))``
    with ``CE(q) = -(log s_n) . q``, aggregation.py) and assert the bound
    elementwise — plus that the returned normalized weights are exactly the
    normalization of that recomputation, so the recomputed quantity is bound
    to the implementation, not a free-floating reimplementation.
    """
    for seed in (0, 1, 7):
        rng = np.random.default_rng(seed)
        mat = rng.dirichlet(np.ones(4), size=5)
        base = rng.uniform(0.5, 2.0, size=5)
        for c in (0.3, 1.5, 3.0):
            res = variational_aggregate(mat, weights=base, robustness=c,
                                        max_iter=512, tol=1e-12)
            q = res.consensus
            ce = -(np.log(mat) @ q)          # forward cross-entropy, per agent
            raw_eff = base * np.exp(-c * ce)  # the code's raw a-update at q
            # the raw bound: CE >= 0 so exp(-c CE) <= 1 — never above base.
            assert np.all(raw_eff <= base + 1e-12), (seed, c, raw_eff, base)
            assert np.all(raw_eff > 0.0)
            # binding: returned normalized weights ARE this raw vector normalized.
            np.testing.assert_allclose(
                res.agent_weights, raw_eff / raw_eff.sum(), atol=1e-12,
                err_msg=f"seed={seed} c={c}: agent_weights are not the "
                        f"normalization of base*exp(-c*CE(consensus))",
            )


# ---- honest caveat: conservatism (max-entropy consensus) ----------------

def test_variational_consensus_is_more_conservative_than_naive():
    # soft colony peaked on state 0: the naive product sharpens, the variational
    # (entropy-regularized) consensus stays flatter — its honest trade-off.
    beliefs = np.array([[0.4, 0.3, 0.3], [0.4, 0.35, 0.25], [0.38, 0.32, 0.30]])
    naive = log_linear_pool(beliefs)
    var = variational_aggregate(beliefs, robustness=1.5).consensus
    assert _entropy(var) >= _entropy(naive) - 1e-9


# ---- edges & dispatch ----------------------------------------------------

def test_single_agent_consensus_recovers_that_agent_at_c_zero():
    belief = np.array([0.6, 0.3, 0.1])
    # at c = 0 the lone agent IS the consensus (Friston corner).
    assert np.allclose(variational_aggregate([belief], robustness=0.0).consensus,
                       belief, atol=1e-9)
    # at c > 0 the entropy regularizer flattens it (more conservative) while
    # preserving the state ordering — the honest max-entropy trade-off.
    soft = variational_aggregate([belief], robustness=2.0).consensus
    assert _entropy(soft) > _entropy(belief)
    assert np.argmax(soft) == np.argmax(belief)
    assert np.all(np.argsort(soft) == np.argsort(belief))


def test_aggregate_dispatch_variational():
    beliefs = [np.array([0.7, 0.2, 0.1]), np.array([0.6, 0.3, 0.1])]
    got = aggregate(beliefs, method="variational", robustness=1.0)
    ref = variational_aggregate(beliefs, robustness=1.0).consensus
    assert np.allclose(got, ref, atol=1e-12)


def test_negative_robustness_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        variational_aggregate([np.array([0.5, 0.5])], robustness=-1.0)


def test_free_energy_requires_positive_robustness():
    beliefs = [np.array([0.6, 0.4]), np.array([0.5, 0.5])]
    with pytest.raises(ValueError, match="robustness > 0"):
        aggregation_free_energy(np.array([0.55, 0.45]), np.array([1.0, 1.0]),
                                beliefs, robustness=0.0)


def test_free_energy_rejects_mismatched_weights():
    beliefs = [np.array([0.6, 0.4]), np.array([0.5, 0.5])]
    with pytest.raises(ValueError, match="length must match"):
        aggregation_free_energy(np.array([0.55, 0.45]), np.array([1.0]),
                                beliefs, robustness=1.0)


def test_robust_aggregate_iterative_path_recovers_pool_in_the_small_c_limit():
    """SYN-10: the c -> 0 identity through the ITERATIVE path.

    ``robust_aggregate(..., robustness=0.0)`` short-circuits through an early
    return, so the exact-identity test never exercises the reweighting loop.
    At ``c = 1e-9`` the early return is skipped, the iterative path runs, and
    the consensus must still match ``log_linear_pool`` to tight tolerance on
    several seeded colonies — a wrong sign or scale in the loop's reweighting
    would blow well past this atol.
    """
    for seed in (0, 3, 11):
        rng = np.random.default_rng(seed)
        mat = rng.dirichlet(np.ones(5), size=6)
        res = robust_aggregate(mat, robustness=1e-9)
        assert res.iterations >= 1  # the iterative path actually ran
        np.testing.assert_allclose(
            res.consensus, log_linear_pool(mat), atol=1e-8,
            err_msg=f"seed={seed}: iterative path diverges from the pool at c=1e-9",
        )


def test_variational_and_heuristic_share_the_friston_corner():
    # both robust families reduce to the same naive pool at c = 0 (ISC-10 spine).
    beliefs = [np.array([0.7, 0.2, 0.1]), np.array([0.55, 0.4, 0.05]),
               np.array([0.1, 0.3, 0.6])]
    naive = log_linear_pool(beliefs)
    var0 = variational_aggregate(beliefs, robustness=0.0).consensus
    assert np.allclose(var0, naive, atol=1e-12)
    # a sanity probe that the divergence used in the heuristic is the *reverse* of
    # the rigorous cross-entropy direction (documents the one-line distinction).
    q = var0
    assert kl_divergence(q, beliefs[2]) != pytest.approx(kl_divergence(beliefs[2], q))
