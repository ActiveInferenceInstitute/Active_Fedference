"""Bayesian model reduction for categorical (Dirichlet) generative models.

This module implements the closed-form *Bayesian model reduction* (BMR) score
of Friston & Penny (2011, NeuroImage 56:2089) — the same Beta-function identity
that appears as Eq. 13 in Friston et al. (2024) on federated active inference.

BMR answers: given a *full* model whose Dirichlet posterior over a categorical
factor is ``post`` (under prior ``prior``), what is the change in (negative)
variational free energy if we swap the prior for a *reduced* prior
``reduced_prior`` — for example one that prunes a redundant column by shrinking
its concentration toward zero — **without re-running inference**? Because the
likelihood is shared, the reduced posterior is available in closed form:

    reduced_post = post + reduced_prior - prior

and the free-energy difference (evidence for the reduced model minus the full
model) is the difference of log multivariate Beta functions:

    dF = lnB(prior) + lnB(reduced_post) - lnB(post) - lnB(reduced_prior)

where ``lnB(a) = sum_k gammaln(a_k) - gammaln(sum_k a_k)`` is the log of the
Dirichlet normalizer (the multivariate Beta function). ``dF > 0`` means the
reduced model has *more* evidence — the pruned structure was redundant and
should be adopted; ``dF < 0`` means the reduction destroyed something the data
support. When ``reduced_prior == prior`` the score is identically zero (no
reduction, no change), which the test-suite pins exactly.

Pure ``numpy`` / ``scipy.special.gammaln``; no sampling, no model re-fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln

ArrayF = np.ndarray
_EPS = 1e-12


def _as_counts(a: ArrayF, name: str) -> ArrayF:
    """Coerce ``a`` to a strictly-positive 1-D Dirichlet concentration vector.

    Concentrations are pseudo-counts: each entry must be ``> 0`` (a proper
    Dirichlet has positive parameters and the log-Beta normalizer is finite only
    there). We clip a tiny floor so a "pruned to zero" reduced prior is handled
    gracefully rather than producing ``gammaln(0) = +inf``.
    """
    arr = np.asarray(a, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if np.any(arr < -1e-9):
        raise ValueError(f"{name} has negative concentrations")
    return np.clip(arr, _EPS, None)


def log_beta(counts: ArrayF) -> float:
    """Return ``lnB(a) = sum_k gammaln(a_k) - gammaln(sum_k a_k)``.

    The log of the multivariate Beta function — i.e. the log normalizing
    constant of a Dirichlet with concentration ``counts``. Used four times to
    assemble the model-reduction free-energy difference.
    """
    a = _as_counts(counts, "counts")
    return float(np.sum(gammaln(a)) - gammaln(a.sum()))


def reduced_posterior(
    posterior_counts: ArrayF,
    prior_counts: ArrayF,
    reduced_prior_counts: ArrayF,
) -> ArrayF:
    """Closed-form reduced posterior ``post + reduced_prior - prior``.

    Because the (shared) likelihood contributes the same expected sufficient
    statistics to both models, the reduced posterior is the full posterior with
    the prior swapped out — no re-inference required.
    """
    post = _as_counts(posterior_counts, "posterior_counts")
    prior = _as_counts(prior_counts, "prior_counts")
    rprior = _as_counts(reduced_prior_counts, "reduced_prior_counts")
    if not (post.shape == prior.shape == rprior.shape):
        raise ValueError("posterior, prior and reduced_prior must share a shape")
    return np.clip(post + rprior - prior, _EPS, None)


def reduce(
    posterior_counts: ArrayF,
    prior_counts: ArrayF,
    reduced_prior_counts: ArrayF,
) -> dict:
    """Score a Dirichlet model reduction via the Beta-function free energy.

    Computes ``reduced_post = post + reduced_prior - prior`` and

        dF = lnB(prior) + lnB(reduced_post) - lnB(post) - lnB(reduced_prior)

    (Friston & Penny 2011; Friston et al. 2024 Eq. 13). ``dF > 0`` favors the
    reduced model. Returns a dict with ``delta_F`` (float, the score),
    ``reduced_posterior`` (the closed-form reduced posterior counts) and the
    four ``log_beta`` components for inspection / plotting.

    Raises ``ValueError`` if the three inputs do not share a common shape or
    carry negative concentrations.
    """
    post = _as_counts(posterior_counts, "posterior_counts")
    prior = _as_counts(prior_counts, "prior_counts")
    rprior = _as_counts(reduced_prior_counts, "reduced_prior_counts")
    rpost = reduced_posterior(post, prior, rprior)

    lb_prior = log_beta(prior)
    lb_rpost = log_beta(rpost)
    lb_post = log_beta(post)
    lb_rprior = log_beta(rprior)
    delta_f = lb_prior + lb_rpost - lb_post - lb_rprior

    return {
        "delta_F": float(delta_f),
        "reduced_posterior": rpost,
        "log_beta_prior": lb_prior,
        "log_beta_reduced_posterior": lb_rpost,
        "log_beta_posterior": lb_post,
        "log_beta_reduced_prior": lb_rprior,
    }


def greedy_reduce(
    posterior_counts: ArrayF,
    prior_counts: ArrayF,
    *,
    tol: float = 1e-9,
    max_prunes: int | None = None,
) -> dict:
    """Greedy multi-hypothesis structure learning by iterated model reduction.

    The single-step :func:`reduce` scores *one* candidate reduced prior. Real
    structure learning (Friston 2024's "emergence" story) prunes a *family* of
    redundant states: starting from the full ``prior``, at each round we score
    pruning each not-yet-pruned state (shrinking its Dirichlet concentration to a
    floor) against the *current* reduced prior, accept the single prune with the
    largest positive free-energy gain, and repeat until no remaining prune
    improves model evidence (or only one state is left, or ``max_prunes`` is
    reached). Because every accepted step has a strictly positive incremental
    ``delta_F``, the cumulative evidence is monotone-increasing, and the greedy
    search recovers the sparse generative model the data actually support: a
    state with genuine evidence yields ``delta_F < 0`` when pruned and is kept.

    Returns a dict with:

    * ``pruned_states`` — sorted list of state indices pruned (the discovered
      redundant structure);
    * ``n_pruned`` — how many;
    * ``reduced_prior`` — the final reduced Dirichlet prior (floored on pruned
      states);
    * ``reduced_posterior`` — its closed-form posterior;
    * ``steps`` — per-accepted-prune ``{state, delta_F_step, cumulative_delta_F}``
      records (each ``delta_F_step > 0``, so ``cumulative_delta_F`` is monotone);
    * ``total_delta_F`` — the honest end-to-end score of the final reduced prior
      against the original ``prior`` (one :func:`reduce` call, not a sum of steps).

    Raises ``ValueError`` if the inputs differ in shape or carry negative
    concentrations, or if there are fewer than two states to choose between.
    """
    post = _as_counts(posterior_counts, "posterior_counts")
    prior = _as_counts(prior_counts, "prior_counts")
    if post.shape != prior.shape:
        raise ValueError("posterior and prior must share a shape")
    n_states = post.shape[0]
    if n_states < 2:
        raise ValueError("greedy_reduce needs at least two states to choose between")
    cap = n_states - 1 if max_prunes is None else min(int(max_prunes), n_states - 1)

    current_prior = prior.copy()
    pruned: list[int] = []
    steps: list[dict] = []
    cumulative = 0.0

    while len(pruned) < cap:
        best_state = -1
        best_df = tol
        best_candidate: ArrayF | None = None
        for k in range(n_states):
            if k in pruned:
                continue
            candidate = current_prior.copy()
            candidate[k] = _EPS
            df = reduce(post, current_prior, candidate)["delta_F"]
            if df > best_df:
                best_df = df
                best_state = k
                best_candidate = candidate
        if best_state < 0 or best_candidate is None:
            break  # no remaining prune improves the evidence — stop
        current_prior = best_candidate
        pruned.append(best_state)
        cumulative += best_df
        steps.append(
            {
                "state": int(best_state),
                "delta_F_step": float(best_df),
                "cumulative_delta_F": float(cumulative),
            }
        )

    total = reduce(post, prior, current_prior)["delta_F"] if pruned else 0.0
    return {
        "pruned_states": sorted(pruned),
        "n_pruned": len(pruned),
        "reduced_prior": current_prior,
        "reduced_posterior": reduced_posterior(post, prior, current_prior),
        "steps": steps,
        "total_delta_F": float(total),
    }


def hierarchical_reduce(
    nlevel_world: dict[str, Any],
    A: ArrayF,
    obs: int,
    *,
    n_iters: int = 8,
    surprise_tol: float = 1e-3,
    alpha0: float = 2.0,
    n_eff: float = 12.0,
) -> dict[str, Any]:
    """Score which levels of a hierarchical POMDP earn their structural keep.

    Structure learning at the *level* granularity: given a built N-level world
    (:func:`fedference.pomdp.build_nlevel_world`) and one leaf observation, run
    the alternating-minimization inference once and, for every **non-leaf**
    level, measure two complementary quantities:

    * ``bayesian_surprise`` — ``KL(posterior_i || empirical_prior_i)``, the
      information the observation added at that level. A level whose belief the
      data never moves from its top-down prior (surprise below ``surprise_tol``)
      carries no structure: its states are behaviourally interchangeable and the
      level can be pruned, recovering the shallower model. This is the
      load-bearing signal — it is an inference-derived divergence, not a
      re-fit, so a non-gating level (identical conditioned priors) scores an
      honest ~0 while an informative level scores strictly positive.
    * ``redundancy_delta_F`` — the Beta-function :func:`greedy_reduce` score on
      the level's own Dirichlet counts (``alpha0 * empirical_prior + n_eff *
      posterior``), a secondary within-level check on whether the level's states
      collapse to one under the model-evidence objective.

    A level is flagged ``prunable`` iff its Bayesian surprise is below
    ``surprise_tol``. The **deepest prunable non-leaf level** (largest index
    among non-leaf levels, i.e. the topmost meta-context that adds nothing) is
    reported as ``recommended_prune`` — pruning it recovers the model with one
    fewer level.

    Returns a dict with:

    * ``levels`` — per non-leaf level: ``{level, label, bayesian_surprise,
      redundancy_delta_F, n_pruned, prunable}``;
    * ``recommended_prune`` — the level index to prune, or ``None`` if every
      non-leaf level is informative;
    * ``n_levels`` — the world's level count.

    Raises ``ValueError`` on a world with fewer than two levels (nothing to
    reduce) or a malformed level structure.
    """
    from fedference.pomdp import nlevel_infer  # local import: avoids a cycle

    n_levels = int(nlevel_world["n_levels"])
    if n_levels < 2:
        raise ValueError("hierarchical_reduce needs an N-level world with n_levels >= 2")

    a = np.asarray(A, dtype=np.float64)
    result = nlevel_infer(a, obs=obs, nlevel_world=nlevel_world, n_iters=n_iters)
    q_levels = result["q_levels"]  # top -> bottom, length n_levels

    # Empirical top-down prior per level (what each level believed BEFORE the
    # bottom-up evidence updated it), reconstructed from the converged parent
    # posteriors — the reference distribution the surprise is measured against.
    level_priors = [np.asarray(p, dtype=np.float64) for p in nlevel_world["level_priors"]]
    conditioned = [
        [np.asarray(p, dtype=np.float64) for p in cp]
        for cp in nlevel_world["conditioned_priors"]
    ]
    # The top level has no parent, so its reference is its GENERATIVE prior
    # (surprise = how far the bottom-up evidence moved it). Each lower level's
    # reference is the top-down empirical prior its parent posterior induces.
    empirical: list[np.ndarray] = [level_priors[0].copy()]
    for depth in range(n_levels - 1):
        parent_q = q_levels[depth]
        child_prior = sum(parent_q[j] * conditioned[depth][j] for j in range(len(parent_q)))
        child_prior = np.clip(np.asarray(child_prior, dtype=np.float64), _EPS, None)
        empirical.append(child_prior / child_prior.sum())

    try:
        labels = [ls.labels for ls in nlevel_world["layers"]]
    except (KeyError, AttributeError):
        labels = [() for _ in range(n_levels)]

    levels: list[dict[str, Any]] = []
    for i in range(n_levels - 1):  # non-leaf levels only
        post = np.clip(np.asarray(q_levels[i], dtype=np.float64), _EPS, None)
        prior = np.clip(empirical[i], _EPS, None)
        surprise = float(np.sum(post * np.log(post / prior)))
        prior_counts = alpha0 * empirical[i] + _EPS
        post_counts = prior_counts + n_eff * np.asarray(q_levels[i], dtype=np.float64)
        greedy = greedy_reduce(post_counts, prior_counts)
        levels.append(
            {
                "level": i,
                "label": labels[i][0] if labels[i] else f"L{i}",
                "bayesian_surprise": surprise,
                "redundancy_delta_F": float(greedy["total_delta_F"]),
                "n_pruned": int(greedy["n_pruned"]),
                "prunable": bool(surprise < surprise_tol),
            }
        )

    prunable = [lv["level"] for lv in levels if lv["prunable"]]
    recommended = max(prunable) if prunable else None
    return {
        "levels": levels,
        "recommended_prune": recommended,
        "n_levels": n_levels,
    }
