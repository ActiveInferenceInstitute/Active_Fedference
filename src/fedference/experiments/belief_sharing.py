"""Experiment harness submodule — see :mod:`fedference.experiments`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..bayesian_model_reduction import reduce as bmr_reduce
from ..belief_sharing import share_round
from ..belief_updating import infer_states, vfe
from ..dirichlet_learning import learn_likelihood
from ..pomdp import N_LOCATIONS, build_sentinel_world
from ..statistics import bootstrap_ci
from ._common import (
    _EPS,
    _N_BOOT,
    _sample_observation,
)

# Single source of truth for the default sensory acuity; the manuscript token
# BELIEF_SHARING_ACUITY is derived from this constant, never re-typed.
DEFAULT_ACUITY: float = 0.55


def run_belief_sharing(
    seed: int,
    *,
    communicate: bool = True,
    n_agents: int = 6,
    acuity: float = DEFAULT_ACUITY,
) -> dict[str, Any]:
    """Run the categorical belief-sharing source-mechanism analogue.

    A colony of ``n_agents`` sentinels each observe the same hidden creature
    location through an independent noisy sensor and form a one-step variational
    posterior (:func:`fedference.belief_updating.infer_states`) from its *own*
    observation. When ``communicate=True`` the agents run one federated
    belief-sharing round (:func:`fedference.belief_sharing.share_round`, the
    project naive pool / categorical Eq. 7 specialization) so each agent's
    posterior moves toward the cross-agent
    consensus. This reduced categorical study does not reconstruct the full
    source message-passing protocol; when ``communicate=False`` the agents keep
    their private posteriors.

    Each agent's *variational free energy is scored against the shared evidence*
    — the full set of all sentinels' observations treated as conditionally
    independent modalities (:func:`fedference.belief_updating.vfe` with a list of
    likelihoods). This is the colony's joint free energy: it is minimized by the
    product-of-experts posterior over every observation. Belief-sharing pushes
    each private posterior toward that joint minimizer, so the mean free energy
    when communicating is strictly below the incommunicado value (ISC-23) — the
    "two heads are better than one" gap of Fig. 5.

    Returns a JSON-serialisable dict with ``mean_free_energy``,
    ``mean_surprise``, ``mean_accuracy``, ``communicate``, ``n_agents``,
    ``true_state`` and ``seed``.
    """
    if n_agents < 2:
        raise ValueError("belief sharing needs at least two agents")
    rng = np.random.default_rng(seed)
    world = build_sentinel_world(rng, acuity=acuity)
    A = np.asarray(world["A"][0], dtype=np.float64)  # type: ignore[index]
    n_s = int(N_LOCATIONS)

    # Hidden truth: an off-center cell so the prior (centered) is genuinely wrong
    # and observations carry information.
    true_state = int(rng.integers(0, n_s))
    log_prior = np.log(np.full(n_s, 1.0 / n_s))  # flat shared prior

    posteriors = np.empty((n_agents, n_s), dtype=np.float64)
    observations = np.empty(n_agents, dtype=np.int64)
    for n in range(n_agents):
        o = _sample_observation(A, true_state, rng)
        observations[n] = o
        posteriors[n] = infer_states(A, o, log_prior)

    if communicate:
        diag = share_round(
            posteriors, method="naive", exclude_self=True, true_state=true_state
        )
        scored = diag.shared_posteriors
        mean_surprise = diag.mean_surprise
        mean_accuracy = diag.mean_accuracy
    else:
        scored = posteriors
        accs = scored[:, true_state]
        mean_accuracy = float(np.mean(accs))
        mean_surprise = float(np.mean(-np.log(np.clip(accs, _EPS, None))))

    # Joint free energy: score each scored belief against the *shared evidence* —
    # the full set of all sentinels' observations as conditionally-independent
    # modalities. Sharing moves each posterior toward this joint minimizer.
    A_modalities = [A] * n_agents
    obs_all = [int(o) for o in observations]
    free_energies = [
        float(vfe(scored[n], A_modalities, obs_all, log_prior))
        for n in range(n_agents)
    ]
    mean_free_energy = float(np.mean(free_energies))

    # Enrichment: the per-agent free energies are the sample behind the headline
    # mean, so a bootstrap CI of that mean quantifies its uncertainty over the
    # colony. Determinism comes from the same seeded generator threaded above.
    fe_ci_lo, fe_ci_hi = bootstrap_ci(
        free_energies, alpha=0.05, n_boot=_N_BOOT, rng=rng
    )

    return {
        # --- existing keys (back-compat) ---------------------------------
        "mean_free_energy": mean_free_energy,
        "mean_surprise": float(mean_surprise),
        "mean_accuracy": float(mean_accuracy),
        "communicate": bool(communicate),
        "n_agents": int(n_agents),
        "true_state": true_state,
        "seed": int(seed),
        # --- enrichment ---------------------------------------------------
        # Sample size behind the colony mean (one free energy per agent).
        "n": int(n_agents),
        "free_energies": free_energies,
        # 95% bootstrap CI of the mean free energy over the colony.
        "mean_free_energy_ci": [fe_ci_lo, fe_ci_hi],
    }


def run_language_acquisition(
    seed: int,
    *,
    num_steps: int = 24,
    n_states: int = 4,
    count_scale: float = 8.0,
) -> dict[str, Any]:
    """Run one categorical language-learning trajectory.

    An agent learns the likelihood of its shared world by conjugate Dirichlet
    updates (:func:`fedference.dirichlet_learning.learn_likelihood`) driven by a
    seeded RNG (so the empirical sufficient statistics fluctuate). The recorded
    ``kl_trajectory`` is ``KL(true A || learned A)`` at the flat-prior start
    and after each count batch (``num_steps + 1`` points) — it
    starts at the flat-prior maximum and declines monotonically toward zero as
    the agent "acquires the language" (ISC-24).

    Returns a dict with ``kl_trajectory`` (list of floats), ``initial_kl``,
    ``final_kl``, ``monotone_decreasing`` and ``seed``. Uncertainty is not
    estimated here: the ordered trajectory points are not independent
    replicates. Use :func:`summarize_language_acquisition` for a seed-level
    ensemble and pointwise intervals.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if n_states < 1:
        raise ValueError("n_states must be >= 1")
    if not np.isfinite(count_scale) or count_scale <= 0.0:
        raise ValueError("count_scale must be finite and positive")
    rng = np.random.default_rng(seed)
    # A confident, near-deterministic data-generating likelihood the agent must
    # discover: each hidden state maps mostly to one matching outcome.
    target = np.full((n_states, n_states), 0.05, dtype=np.float64)
    np.fill_diagonal(target, 1.0)
    target = target / target.sum(axis=0, keepdims=True)

    result = learn_likelihood(
        target, num_steps, count_scale=count_scale, rng=rng
    )

    trajectory = [float(v) for v in result.kl_trajectory]
    return {
        "kl_trajectory": trajectory,
        "initial_kl": float(result.initial_kl),
        "final_kl": float(result.final_kl),
        "monotone_decreasing": bool(result.is_monotone_decreasing),
        "num_steps": int(num_steps),
        "seed": int(seed),
        "n": len(trajectory),
    }


def summarize_language_acquisition(
    seeds: Sequence[int],
    *,
    num_steps: int = 24,
    n_states: int = 4,
    count_scale: float = 8.0,
) -> dict[str, Any]:
    """Summarize language learning across independent seeded trajectories.

    The learning step is the ordered axis, not the replication unit. This
    helper therefore aligns one trajectory per supplied seed, computes the
    seed mean at each step, and bootstraps each step across seeds. It returns
    the mean trajectory plus pointwise percentile intervals for publication
    figures; no time point is treated as an independent observation.
    """
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("summarize_language_acquisition requires at least one seed")
    if len(seed_list) < 2:
        raise ValueError("language seed summaries require at least two independent seeds")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("language seed summaries require distinct independent seeds")
    runs = [
        run_language_acquisition(
            seed,
            num_steps=num_steps,
            n_states=n_states,
            count_scale=count_scale,
        )
        for seed in seed_list
    ]
    trajectories = np.asarray([run["kl_trajectory"] for run in runs], dtype=np.float64)
    if trajectories.ndim != 2 or trajectories.shape[1] != num_steps + 1:
        raise ValueError("language trajectories must align to num_steps + 1 points")
    if not np.all(np.isfinite(trajectories)):
        raise ValueError("language trajectories must contain only finite values")

    mean_trajectory = trajectories.mean(axis=0)
    boot_rng = np.random.default_rng(seed_list[0] + 1_000_003)
    ci_lo: list[float] = []
    ci_hi: list[float] = []
    for step_values in trajectories.T:
        lo, hi = bootstrap_ci(step_values, alpha=0.05, n_boot=_N_BOOT, rng=boot_rng)
        ci_lo.append(lo)
        ci_hi.append(hi)

    return {
        "kl_trajectory": [float(value) for value in mean_trajectory],
        "kl_trajectory_by_seed": trajectories.tolist(),
        "trajectory_ci_lo": ci_lo,
        "trajectory_ci_hi": ci_hi,
        "initial_kl": float(mean_trajectory[0]),
        "final_kl": float(mean_trajectory[-1]),
        "monotone_decreasing": bool(
            np.all(np.diff(mean_trajectory) <= 1e-12)
        ),
        "num_steps": int(num_steps),
        "n_points": int(mean_trajectory.size),
        "n_seeds": int(len(seed_list)),
        "seed": int(seed_list[0]),
    }


def run_emergence(
    seed: int,
    *,
    n_states: int = 4,
    evidence_scale: float = 40.0,
) -> dict[str, Any]:
    """Run the categorical BMR source-mechanism analogue related to Fig. 9.

    A full Dirichlet model carries a redundant state (one column the data never
    support). Bayesian model reduction
    (:func:`fedference.bayesian_model_reduction.reduce`, Friston 2024 Eq. 13)
    scores swapping the prior for a *reduced* prior that prunes that column's
    concentration toward zero. The free-energy difference ``dF`` is positive —
    the simpler, reduced model has more evidence and the run "converges" on it
    (ISC-25). As a control, pruning a column the data *do* support yields
    ``dF < 0`` (the reduction is rejected).

    Returns a dict with ``convergence`` (bool: did the redundant-prune win),
    ``delta_F_redundant``, ``delta_F_supported`` and ``seed``.
    """
    rng = np.random.default_rng(seed)
    prior = np.ones(n_states, dtype=np.float64)

    # Data support every state except the last (the redundant one), with a small
    # seeded jitter so the run is genuinely stochastic but stable in sign.
    support = np.ones(n_states, dtype=np.float64)
    support[-1] = 0.0  # redundant state — no evidence accrues to it
    jitter = rng.uniform(0.9, 1.1, size=n_states)
    counts = evidence_scale * support * jitter
    posterior = prior + counts

    # Reduced prior that prunes the redundant (last) column toward zero.
    reduced_prior_redundant = prior.copy()
    reduced_prior_redundant[-1] = _EPS

    # Control: prune a well-supported column (state 0) — this should be rejected.
    reduced_prior_supported = prior.copy()
    reduced_prior_supported[0] = _EPS

    df_redundant = bmr_reduce(posterior, prior, reduced_prior_redundant)["delta_F"]
    df_supported = bmr_reduce(posterior, prior, reduced_prior_supported)["delta_F"]

    convergence = bool(df_redundant > 0.0 > df_supported)

    return {
        # --- existing keys (back-compat) ---------------------------------
        "convergence": convergence,
        "delta_F_redundant": float(df_redundant),
        "delta_F_supported": float(df_supported),
        "n_states": int(n_states),
        "seed": int(seed),
        # --- enrichment ---------------------------------------------------
        # Sample size: the number of candidate states the reduction ranges over.
        # (A single deterministic BMR evidence comparison — no resampled sample
        # exists here, so no bootstrap CI / paired test is reported, by design.)
        "n": int(n_states),
    }
