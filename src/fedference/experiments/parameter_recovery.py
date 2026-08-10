"""Experiment harness submodule — see :mod:`fedference.experiments`."""

from __future__ import annotations

from typing import Any

import numpy as np


def run_parameter_recovery(
    seed: int,
    *,
    n_agents: int = 4,
    acuity_grid: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90),
    n_observations: int = 50,
    n_trials: int = 20,
    fit_resolution: int = 40,
) -> dict[str, Any]:
    """Validate generative-model identifiability by fitting acuity from synthetic data.

    For each ``true_acuity`` in ``acuity_grid`` the function:

    1. Builds a deterministic sentinel world at that acuity and draws
       ``n_trials`` independent sets of ``n_observations`` synthetic observations
       from the data-generating likelihood ``A_true``. The deterministic model
       construction keeps the declared acuity equal to the data-generating
       acuity rather than adding the colony-heterogeneity jitter used by other
       experiments.
    2. Runs a grid search over ``fit_resolution`` candidate acuity values in
       ``[0.51, 0.99]``; for each candidate it builds a world, computes the
       marginal log-likelihood ``sum_obs log(mean_col(A_c[obs, :]) + eps)``, and
       picks the argmax.
    3. Collects the per-trial recovered acuity, its mean, and the 2.5th / 97.5th
       empirical percentile interval across independent trials. These are
       descriptive trial-distribution quantiles, not bootstrap confidence or
       Bayesian credible intervals.

    The headline statistics are the mean absolute error across the grid and the
    R² of mean-recovered vs true acuity (a measure of calibration).

    Returns a JSON-serialisable dict with keys:

    * ``true_acuity`` — list of true acuity values (one per grid point).
    * ``recovered_acuity`` — list of mean recovered acuity per grid point.
    * ``recovered_acuity_ci_lo`` — list of 2.5th-percentile recovered acuity.
    * ``recovered_acuity_ci_hi`` — list of 97.5th-percentile recovered acuity.
    * ``abs_error`` — list of mean absolute error per grid point.
    * ``mean_abs_error`` — mean of ``abs_error`` across the grid.
    * ``r_squared`` — R² of mean recovered vs true acuity.
    * ``n_trials``, ``n_observations``, ``acuity_grid``, ``seed``.
    * ``interval_method`` and ``interval_percent`` — provenance for the plotted
      empirical percentile interval.
    """
    from fedference.pomdp import build_sentinel_world

    rng = np.random.default_rng(seed)
    fit_grid = np.linspace(0.51, 0.99, fit_resolution)

    true_acuity_list: list[float] = []
    recovered_acuity_list: list[float] = []
    ci_lo_list: list[float] = []
    ci_hi_list: list[float] = []
    abs_error_list: list[float] = []

    for true_acuity in acuity_grid:
        # Keep the likelihood grid fixed across trials and candidate fits. A
        # seeded ``rng`` is reserved for observations, not for perturbing the
        # model being estimated.
        world = build_sentinel_world(None, acuity=float(true_acuity))
        A_true = np.asarray(world["A"][0], dtype=np.float64)
        n_s = A_true.shape[1]

        trial_recovered: list[float] = []
        for _ in range(n_trials):
            true_state = int(rng.integers(0, n_s))
            obs_probs = A_true[:, true_state].copy()
            obs_probs = np.clip(obs_probs, 0.0, None)
            obs_probs = obs_probs / obs_probs.sum()
            observations = rng.choice(len(obs_probs), size=n_observations, p=obs_probs)

            log_likelihoods: list[float] = []
            for cand_acuity in fit_grid:
                world_c = build_sentinel_world(None, acuity=float(cand_acuity))
                A_c = np.asarray(world_c["A"][0], dtype=np.float64)
                n_s_fit = A_c.shape[1]
                # Marginal log-likelihood: log p(obs|acuity) = log sum_s [p(s) prod_obs p(obs|s,acuity)]
                # With uniform state prior: log-sum-exp over states of [sum_obs log A_c[obs,s]] - log(n_s)
                state_log_lls = np.zeros(n_s_fit)
                for obs in observations:
                    state_log_lls += np.log(A_c[int(obs), :] + 1e-12)
                max_ll = float(np.max(state_log_lls))
                ll = (
                    max_ll
                    + float(np.log(float(np.sum(np.exp(state_log_lls - max_ll)))))
                    - float(np.log(n_s_fit))
                )
                log_likelihoods.append(ll)

            best_acuity = float(fit_grid[int(np.argmax(log_likelihoods))])
            trial_recovered.append(best_acuity)

        rec_arr = np.array(trial_recovered, dtype=np.float64)
        mean_rec = float(np.mean(rec_arr))
        ci_lo = float(np.percentile(rec_arr, 2.5))
        ci_hi = float(np.percentile(rec_arr, 97.5))
        abs_err = float(np.mean(np.abs(rec_arr - true_acuity)))

        true_acuity_list.append(float(true_acuity))
        recovered_acuity_list.append(mean_rec)
        ci_lo_list.append(ci_lo)
        ci_hi_list.append(ci_hi)
        abs_error_list.append(abs_err)

    true_arr = np.array(true_acuity_list, dtype=np.float64)
    rec_arr_means = np.array(recovered_acuity_list, dtype=np.float64)
    ss_res = float(np.sum((rec_arr_means - true_arr) ** 2))
    ss_tot = float(np.sum((true_arr - float(np.mean(true_arr))) ** 2))
    r_squared = float(max(0.0, 1.0 - ss_res / ss_tot)) if ss_tot > 0.0 else 1.0

    return {
        "true_acuity": true_acuity_list,
        "recovered_acuity": recovered_acuity_list,
        "recovered_acuity_ci_lo": ci_lo_list,
        "recovered_acuity_ci_hi": ci_hi_list,
        "abs_error": abs_error_list,
        "mean_abs_error": float(np.mean(abs_error_list)),
        "r_squared": r_squared,
        "n_trials": int(n_trials),
        "n_observations": int(n_observations),
        "acuity_grid": list(acuity_grid),
        "interval_method": "empirical_percentile_across_independent_trials",
        "interval_percent": 95,
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# V4 extension: disjoint-FOV multi-agent world (Task 1 additions)
# ---------------------------------------------------------------------------
