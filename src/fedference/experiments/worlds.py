"""Experiment harness submodule — see :mod:`fedference.experiments`."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..aggregation import log_linear_pool
from ..pomdp import (
    N_CONTEXTS,
    N_LOCATIONS,
    LayerSpec,
    build_3level_world,
    build_hierarchical_world,
    build_nlevel_world,
    build_sentinel_world,
    nlevel_infer,
)
from ._common import (
    _EPS,
    _sample_observation,
)


def run_nlevel_world(
    seed: int = 0,
    *,
    layers: list["LayerSpec"] | None = None,
    n_agents: int = 4,
    n_trials: int = 20,
    acuity: float = 0.85,
    n_iters: int = 4,
    depth: int = 2,
) -> dict[str, Any]:
    """Study 7 variant — generic N-level hierarchical federation.

    Wraps :func:`fedference.pomdp.build_nlevel_world` and
    :func:`fedference.pomdp.nlevel_infer` for an arbitrary ``depth`` (default 2,
    the same as :func:`run_hierarchical_world`). When ``layers`` is ``None`` the
    function builds a canonical stack of ``depth`` levels, where the leaf is the
    standard 9-location POMDP and every parent level is a 2-state context variable.

    Returns the same schema as :func:`run_hierarchical_world` extended with
    ``"n_levels"`` and ``"depth"``.
    """
    rng = np.random.default_rng(seed)

    if layers is None:
        from ..pomdp import (
            GRID_SIDE,
            HIER_CONTEXT_LABELS,
            L3_META_LABELS,
            N_CONTEXTS,
            N_LOCATIONS,
        )
        # Build a canonical depth-level stack.
        # Leaf layer: 9-location sentinel world.
        center_idx = (GRID_SIDE // 2) * GRID_SIDE + (GRID_SIDE // 2)
        alert_off = (1.0 - 0.6) / (N_LOCATIONS - 1)
        p_alert = np.full(N_LOCATIONS, alert_off, dtype=np.float64)
        p_alert[center_idx] = 0.6
        p_quiet = np.full(N_LOCATIONS, 1.0 / N_LOCATIONS, dtype=np.float64)
        leaf = LayerSpec(
            n_states=N_LOCATIONS,
            labels=tuple(str(i) for i in range(N_LOCATIONS)),
            default_prior=np.full(N_LOCATIONS, 1.0 / N_LOCATIONS, dtype=np.float64),
            conditioned_priors=None,
        )
        # Context layer (second from bottom): quiet/alert.
        ctx = LayerSpec(
            n_states=N_CONTEXTS,
            labels=HIER_CONTEXT_LABELS,
            default_prior=np.full(N_CONTEXTS, 1.0 / N_CONTEXTS, dtype=np.float64),
            conditioned_priors=[p_quiet, p_alert],
        )
        if depth < 2:
            raise ValueError(f"canonical depth must be >= 2; got {depth}")
        # Bottom two levels are always context -> leaf. Every additional level
        # above them is a 2-state meta-context that gates the level below with
        # one uniform (non-committal) and one peaked conditioned prior — the
        # same low/high-threat template as the canonical 3-level world, applied
        # generically so any depth >= 2 builds a valid stack (MAJ-5).
        l2_low = np.full(N_CONTEXTS, 1.0 / N_CONTEXTS, dtype=np.float64)
        l2_high = np.array([0.2, 0.8], dtype=np.float64)
        layers = [ctx, leaf]
        for level_above in range(depth - 2):
            labels = L3_META_LABELS if level_above == 0 else (
                f"meta{level_above}_low", f"meta{level_above}_high"
            )
            meta = LayerSpec(
                n_states=2,
                labels=labels,
                default_prior=np.full(2, 0.5, dtype=np.float64),
                conditioned_priors=[l2_low.copy(), l2_high.copy()],
            )
            layers = [meta, *layers]

    world = build_nlevel_world(layers, acuity=acuity)
    A_base = np.asarray(world["L1"]["A"][0], dtype=np.float64)
    n_levels: int = int(world["n_levels"])

    # Top-level prior used to sample the true top-level state.
    top_prior = np.asarray(world["level_priors"][0], dtype=np.float64)
    n_top_states = len(top_prior)

    loc_acc: dict[str, float] = {"flat": 0.0, "nlevel": 0.0}
    fe_sum: dict[str, float] = {"flat": 0.0, "nlevel": 0.0}
    top_acc = 0.0

    for _ in range(n_trials):
        true_top = int(rng.choice(n_top_states, p=top_prior))
        # Sample true location using the full top-down prior chain.
        child_prior = np.asarray(world["conditioned_priors"][0][true_top], dtype=np.float64)
        for depth_i in range(1, n_levels - 1):
            next_state = int(rng.choice(len(child_prior), p=child_prior))
            child_prior = np.asarray(world["conditioned_priors"][depth_i][next_state], dtype=np.float64)
        true_state = int(rng.choice(N_LOCATIONS, p=child_prior))

        per_agent_obs = [_sample_observation(A_base, true_state, rng) for _ in range(n_agents)]

        # ---- flat condition ----
        flat_log_prior = np.log(np.full(N_LOCATIONS, 1.0 / N_LOCATIONS))
        from ..belief_updating import infer_states  # noqa: F811
        flat_local_posteriors = [
            infer_states(A_base, o, flat_log_prior) for o in per_agent_obs
        ]
        flat_consensus = log_linear_pool(flat_local_posteriors)
        loc_acc["flat"] += float(np.argmax(flat_consensus) == true_state) / n_trials
        fe_sum["flat"] += float(-np.log(np.clip(flat_consensus[true_state], _EPS, None))) / n_trials

        # ---- N-level condition ----
        nlevel_results = [
            nlevel_infer(A_base, o, world, n_iters=n_iters) for o in per_agent_obs
        ]
        l1_local_posteriors = [r["q_levels"][-1] for r in nlevel_results]
        nlevel_l1_consensus = log_linear_pool(l1_local_posteriors)
        top_local_posteriors = [r["q_levels"][0] for r in nlevel_results]
        top_consensus = log_linear_pool(top_local_posteriors)

        loc_acc["nlevel"] += float(np.argmax(nlevel_l1_consensus) == true_state) / n_trials
        top_acc += float(np.argmax(top_consensus) == true_top) / n_trials
        fe_sum["nlevel"] += float(-np.log(np.clip(nlevel_l1_consensus[true_state], _EPS, None))) / n_trials

    flat_fe = fe_sum["flat"]
    return {
        "location_accuracy": loc_acc,
        "location_accuracy_gap": float(loc_acc["nlevel"] - loc_acc["flat"]),
        "top_level_accuracy": float(top_acc),
        "free_energy_gap": {c: flat_fe - fe_sum[c] for c in ("flat", "nlevel")},
        "n_trials": int(n_trials),
        "n_agents": int(n_agents),
        "acuity": float(acuity),
        "n_iters": int(n_iters),
        "n_levels": n_levels,
        "depth": n_levels,
        "seed": int(seed),
    }


def run_3level_world(
    seed: int = 0,
    *,
    n_agents: int = 4,
    n_trials: int = 20,
    acuity: float = 0.85,
    n_iters: int = 4,
) -> dict[str, Any]:
    """Study 7 — 3-level hierarchical federation (L3=meta-context → L2=context → L1=location).

    A 3-level POMDP (:func:`fedference.pomdp.build_3level_world`) couples location
    inference (L1) to a context (L2) and a meta-context (L3). ``n_agents`` sentinels
    each run :func:`nlevel_infer`; their L1 posteriors are federated via a
    log-linear pool; and L2/L3 beliefs are also federated.

    Returns the same schema as :func:`run_nlevel_world` plus convenience keys
    ``"context_accuracy"`` (L2) and ``"meta_context_accuracy"`` (L3).
    """
    world = build_3level_world(acuity=acuity)
    A_base = np.asarray(world["L1"]["A"][0], dtype=np.float64)
    rng = np.random.default_rng(seed)

    l3_prior = np.asarray(world["L3_prior"], dtype=np.float64)
    l2_given_l3: list[np.ndarray] = list(world["L2_priors_given_l3"])  # type: ignore[arg-type]
    l1_given_l2: list[np.ndarray] = list(world["L1_priors_given_l2"])  # type: ignore[arg-type]

    n_l3 = len(l3_prior)
    n_l2 = N_CONTEXTS
    n_levels: int = int(world["n_levels"])

    loc_acc: dict[str, float] = {"flat": 0.0, "nlevel3": 0.0}
    fe_sum: dict[str, float] = {"flat": 0.0, "nlevel3": 0.0}
    ctx_acc = 0.0
    meta_acc = 0.0

    for _ in range(n_trials):
        true_l3 = int(rng.choice(n_l3, p=l3_prior))
        true_l2 = int(rng.choice(n_l2, p=l2_given_l3[true_l3]))
        true_state = int(rng.choice(N_LOCATIONS, p=l1_given_l2[true_l2]))

        per_agent_obs = [_sample_observation(A_base, true_state, rng) for _ in range(n_agents)]

        from ..pomdp import nlevel_infer
        from ..trials import compare_flat_vs_nlevel

        metrics = compare_flat_vs_nlevel(
            A_base=A_base,
            per_agent_obs=per_agent_obs,
            true_state=true_state,
            n_iters=n_iters,
            infer_fn=nlevel_infer,
            infer_kwargs={"nlevel_world": world},
        )
        loc_acc["flat"] += metrics.flat_loc_correct / n_trials
        loc_acc["nlevel3"] += metrics.nlevel_loc_correct / n_trials
        fe_sum["flat"] += metrics.flat_fe / n_trials
        fe_sum["nlevel3"] += metrics.nlevel_fe / n_trials

        results_3l = [nlevel_infer(A_base, o, world, n_iters=n_iters) for o in per_agent_obs]
        l2_local_posteriors = [r["q_levels"][1] for r in results_3l]
        l3_local_posteriors = [r["q_levels"][0] for r in results_3l]
        l2_consensus = log_linear_pool(l2_local_posteriors)
        l3_consensus = log_linear_pool(l3_local_posteriors)
        ctx_acc += float(np.argmax(l2_consensus) == true_l2) / n_trials
        meta_acc += float(np.argmax(l3_consensus) == true_l3) / n_trials

    flat_fe = fe_sum["flat"]
    return {
        "location_accuracy": loc_acc,
        "location_accuracy_gap": float(loc_acc["nlevel3"] - loc_acc["flat"]),
        "context_accuracy": float(ctx_acc),
        "meta_context_accuracy": float(meta_acc),
        "free_energy_gap": {c: flat_fe - fe_sum[c] for c in ("flat", "nlevel3")},
        "n_trials": int(n_trials),
        "n_agents": int(n_agents),
        "acuity": float(acuity),
        "n_iters": int(n_iters),
        "n_levels": n_levels,
        "n_contexts": int(n_l2),
        "n_meta_contexts": int(n_l3),
        "seed": int(seed),
    }


def run_hierarchical_bmr(
    seed: int = 0,
    *,
    acuity: float = 0.85,
    n_iters: int = 8,
    obs: int = 4,
) -> dict[str, Any]:
    """Hierarchical structure learning by Bayesian model reduction (companion to the N-level study).

    Builds two 3-level worlds that differ ONLY in the top (meta-context) level's
    conditioned priors — a *degenerate* world whose meta-context is non-gating
    (both states predict the same context distribution) and an *informative*
    world whose meta-context sharply distinguishes the two contexts — and runs
    :func:`fedference.bayesian_model_reduction.hierarchical_reduce` on each.

    The verdict is directional: on the degenerate world the top level earns
    ~zero Bayesian surprise and is flagged prunable (BMR recovers the 2-level
    structure); on the informative world it earns strictly positive surprise and
    is kept. The two worlds share every other parameter, so the difference in
    prune verdict is attributable to the meta-context's information alone.

    Returns a JSON-serialisable dict with ``degenerate`` and ``informative``
    (each the ``hierarchical_reduce`` output, arrays elided) plus the headline
    ``degenerate_top_surprise`` / ``informative_top_surprise`` /
    ``degenerate_recommends_prune_top`` / ``informative_keeps_top`` scalars.
    """
    from ..bayesian_model_reduction import hierarchical_reduce

    del seed  # deterministic: the schematic worlds carry no RNG draw

    leaf_A = np.asarray(
        build_sentinel_world(np.random.default_rng(0), acuity=acuity)["A"][0],
        dtype=np.float64,
    )
    loc_a = np.full(N_LOCATIONS, 0.02)
    loc_a[4] = 1.0 - 0.02 * (N_LOCATIONS - 1)
    loc_b = np.full(N_LOCATIONS, 0.02)
    loc_b[0] = 1.0 - 0.02 * (N_LOCATIONS - 1)

    def _world(l3_conditioned: list[np.ndarray]) -> dict[str, Any]:
        l3 = LayerSpec(
            n_states=2,
            labels=("low_threat", "high_threat"),
            default_prior=np.array([0.5, 0.5]),
            conditioned_priors=l3_conditioned,
        )
        l2 = LayerSpec(
            n_states=2,
            labels=("quiet", "alert"),
            default_prior=np.array([0.5, 0.5]),
            conditioned_priors=[loc_a, loc_b],
        )
        leaf = LayerSpec(
            n_states=N_LOCATIONS, labels=tuple(str(i) for i in range(N_LOCATIONS))
        )
        return build_nlevel_world([l3, l2, leaf], acuity=acuity)

    degenerate = hierarchical_reduce(
        _world([np.array([0.5, 0.5]), np.array([0.5, 0.5])]),
        leaf_A,
        obs=obs,
        n_iters=n_iters,
    )
    informative = hierarchical_reduce(
        _world([np.array([0.9, 0.1]), np.array([0.1, 0.9])]),
        leaf_A,
        obs=obs,
        n_iters=n_iters,
    )
    deg_top = next(lv for lv in degenerate["levels"] if lv["level"] == 0)
    inf_top = next(lv for lv in informative["levels"] if lv["level"] == 0)
    return {
        "degenerate": degenerate,
        "informative": informative,
        "degenerate_top_surprise": float(deg_top["bayesian_surprise"]),
        "informative_top_surprise": float(inf_top["bayesian_surprise"]),
        "degenerate_recommends_prune_top": bool(degenerate["recommended_prune"] == 0),
        "informative_keeps_top": bool(informative["recommended_prune"] != 0),
        "n_levels": int(degenerate["n_levels"]),
        "acuity": float(acuity),
        "obs": int(obs),
    }


def run_moving_world(
    seed: int = 0,
    *,
    n_positions: int = 4,
    n_agents: int = 2,
    n_steps: int = 6,
    n_trials: int = 20,
) -> dict[str, Any]:
    """Moving-world federation: isolated vs communicating vs EFE-guided (V4).

    A binary threat occupies one half of a linear ``n_positions`` grid; agents
    have **disjoint** fields-of-view (:func:`fedference.pomdp.build_moving_world`)
    so neither alone can be certain of the hidden state. Three conditions are run
    in lock-step over ``n_trials`` trials of ``n_steps`` steps each:

    * ``isolated`` — random actions, no belief sharing;
    * ``communicating`` — random actions, but a :func:`log_linear_pool` consensus
      is broadcast each step (the agents gossip about the shared latent);
    * ``efe_guided`` — :func:`fedference.pomdp.efe_policy_select` chooses the most
      information-seeking move, plus the same per-step belief sharing.

    Returns per-condition ``accuracy`` (fraction of trials whose consensus
    argmax matches the truth), ``free_energy_gap`` (isolated surprise minus the
    condition's surprise on the true state — positive means lower free energy),
    and ``n_steps_to_consensus`` (a coarse consensus-entropy proxy), plus the run
    parameters.
    """
    from ..pomdp import _moving_likelihood, build_moving_world, efe_policy_select

    rng = np.random.default_rng(seed)
    n_states = 2
    conditions = ["isolated", "communicating", "efe_guided"]
    accuracy = {c: 0.0 for c in conditions}
    fe_sum = {c: 0.0 for c in conditions}
    steps_sum = {c: 0.0 for c in conditions}

    for _ in range(n_trials):
        true_state = int(rng.integers(0, n_states))
        world = build_moving_world(
            n_positions=n_positions,
            n_agents=n_agents,
            n_states=n_states,
            seed=int(rng.integers(0, 100000)),
        )
        B = np.asarray(world["B"], dtype=np.float64)

        local_posteriors_by_condition = {
            c: [np.full(n_states, 1.0 / n_states) for _ in range(n_agents)]
            for c in conditions
        }
        positions_by_cond = {c: list(world["agent_positions"]) for c in conditions}

        for _step in range(n_steps):
            for c in conditions:
                local_posteriors = local_posteriors_by_condition[c]
                positions = positions_by_cond[c]

                # --- action selection ---
                if c == "efe_guided":
                    w_dict = dict(world)
                    w_dict["agent_positions"] = positions
                    actions = efe_policy_select(local_posteriors, w_dict)
                else:
                    actions = [int(rng.integers(0, 3)) for _ in range(n_agents)]

                # --- update positions ---
                for i in range(n_agents):
                    positions[i] = int(np.argmax(B[:, positions[i], actions[i]]))

                # --- observe and update beliefs (likelihood at new position) ---
                for i in range(n_agents):
                    A_i = _moving_likelihood(
                        positions[i],
                        fov_width=int(world["fov_width"]),
                        n_positions=n_positions,
                        n_states=n_states,
                    )
                    p_obs = np.clip(A_i[:, true_state], 0.0, None)
                    p_obs = p_obs / p_obs.sum()
                    o = int(rng.choice(A_i.shape[0], p=p_obs))
                    prior = np.clip(local_posteriors[i], 1e-12, None)
                    posterior = A_i[o, :] * prior
                    posterior = np.clip(posterior, 1e-12, None)
                    posterior = posterior / posterior.sum()
                    local_posteriors[i] = posterior

                # --- communicate ---
                if c in ("communicating", "efe_guided"):
                    consensus = log_linear_pool(local_posteriors)
                    local_posteriors_by_condition[c] = [
                        consensus.copy() for _ in range(n_agents)
                    ]

        # --- score ---
        for c in conditions:
            local_posteriors = local_posteriors_by_condition[c]
            consensus = log_linear_pool(local_posteriors)
            accuracy[c] += float(np.argmax(consensus) == true_state) / n_trials
            fe_sum[c] += float(
                -np.log(np.clip(consensus[true_state], 1e-12, None))
            ) / n_trials
            ent = float(
                -np.sum(
                    np.clip(consensus, 1e-12, None)
                    * np.log(np.clip(consensus, 1e-12, None))
                )
            )
            steps_sum[c] += float(n_steps if ent > 0.5 else max(1, n_steps - 1)) / n_trials

    isolated_fe = fe_sum["isolated"]
    return {
        "accuracy": accuracy,
        "free_energy_gap": {c: isolated_fe - fe_sum[c] for c in conditions},
        "n_steps_to_consensus": steps_sum,
        "n_trials": int(n_trials),
        "n_steps": int(n_steps),
        "n_positions": int(n_positions),
        "n_agents": int(n_agents),
        "seed": int(seed),
    }


def run_hierarchical_world(
    seed: int = 0,
    *,
    n_agents: int = 4,
    n_trials: int = 20,
    acuity: float = 0.85,
    n_iters: int = 4,
) -> dict[str, Any]:
    """Study 6 — hierarchical federation at L1 (location) and L2 (context).

    A 2-level hierarchical POMDP (:func:`fedference.pomdp.build_hierarchical_world`)
    couples location inference (L1, 9 states) to context inference (L2, 2 states:
    ``quiet`` / ``alert``). ``n_agents`` sentinels each run
    :func:`fedference.pomdp.hierarchical_infer` on their own observation; their
    L1 location posteriors are federated via a log-linear pool; and the fused
    location belief is fed back to refine each agent's L2 context belief.

    Two conditions are compared over ``n_trials`` seeded trials:

    * ``flat`` — agents ignore the context hierarchy and infer location with a
      flat (uniform) prior;
    * ``hierarchical`` — agents use :func:`hierarchical_infer` with alternating
      L1/L2 minimization and fuse L1 beliefs across the colony.

    The headline metric is ``location_accuracy`` (fraction of trials in which
    ``argmax q_loc == true_state``). The secondary metric is ``context_accuracy``
    (fraction of trials in which ``argmax q_ctx == true_context``) — only
    meaningful for the hierarchical condition.

    Returns a JSON-serialisable dict with per-condition ``location_accuracy``,
    ``location_accuracy_gap`` (hierarchical minus flat), ``context_accuracy``
    (hierarchical only), ``free_energy_gap`` (flat surprise minus hierarchical
    surprise on the true location state), and run parameters.
    """
    rng = np.random.default_rng(seed)
    n_s = N_LOCATIONS
    n_ctx = N_CONTEXTS

    loc_acc = {"flat": 0.0, "hierarchical": 0.0}
    ctx_acc_hier = 0.0
    fe_sum = {"flat": 0.0, "hierarchical": 0.0}

    hier_world = build_hierarchical_world(acuity=acuity)
    l1 = hier_world["L1"]
    A_base = np.asarray(l1["A"][0], dtype=np.float64)

    from ..pomdp import hierarchical_infer  # noqa: F811
    from ..trials import compare_flat_vs_nlevel

    for _ in range(n_trials):
        true_ctx = int(rng.integers(0, n_ctx))
        # Sample true location from the context-conditioned L1 prior.
        l1_priors_ctx: list[np.ndarray] = list(  # type: ignore[assignment]
            hier_world["L1_priors_given_context"]  # type: ignore[arg-type]
        )
        true_state = int(rng.choice(n_s, p=l1_priors_ctx[true_ctx]))

        # Each agent gets an independent noisy observation from its own sensor.
        per_agent_obs = [
            _sample_observation(A_base, true_state, rng) for _ in range(n_agents)
        ]

        metrics = compare_flat_vs_nlevel(
            A_base=A_base,
            per_agent_obs=per_agent_obs,
            true_state=true_state,
            n_iters=n_iters,
            infer_fn=hierarchical_infer,
            infer_kwargs={"hier_world": hier_world},
        )
        loc_acc["flat"] += metrics.flat_loc_correct / n_trials
        loc_acc["hierarchical"] += metrics.nlevel_loc_correct / n_trials
        fe_sum["flat"] += metrics.flat_fe / n_trials
        fe_sum["hierarchical"] += metrics.nlevel_fe / n_trials

        hier_results = [
            hierarchical_infer(A_base, o, hier_world, n_iters=n_iters)
            for o in per_agent_obs
        ]
        l2_local_posteriors = [r["q_ctx"] for r in hier_results]
        hier_l2_consensus = log_linear_pool(l2_local_posteriors)
        ctx_acc_hier += float(np.argmax(hier_l2_consensus) == true_ctx) / n_trials

    flat_fe = fe_sum["flat"]
    return {
        "location_accuracy": loc_acc,
        "location_accuracy_gap": float(loc_acc["hierarchical"] - loc_acc["flat"]),
        "context_accuracy": float(ctx_acc_hier),
        "free_energy_gap": {c: flat_fe - fe_sum[c] for c in ("flat", "hierarchical")},
        "n_trials": int(n_trials),
        "n_agents": int(n_agents),
        "acuity": float(acuity),
        "n_iters": int(n_iters),
        "n_contexts": int(n_ctx),
        "seed": int(seed),
    }
