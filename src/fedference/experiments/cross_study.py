"""Cross-study summary metrics for all nine federation studies."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..statistics import bootstrap_ci
from .belief_sharing import run_belief_sharing, run_emergence, run_language_acquisition
from .parameter_recovery import run_parameter_recovery
from .robustness import run_robustness_sweep
from .sensitivity import run_belief_sharing_sensitivity
from .worlds import run_3level_world, run_hierarchical_world, run_moving_world

#: Trials-per-cell for the Study 8 cross-study sensitivity scalar. Deliberately
#: smaller than the full-resolution DEFAULT_SENSITIVITY_N_TRIALS grid (a runtime
#: budget: the cross-study loop re-runs this per seed) and surfaced as the
#: CROSS_STUDY_SENS_N_TRIALS manuscript token (src/manuscript_vars/) so prose
#: cannot drift from the executed value.
CROSS_STUDY_SENS_N_TRIALS: int = 3


def summarize_cross_study(
    seed: int, n_seeds: int, *, n_trials: int = 40
) -> dict[str, Any]:
    """Collect per-study federation benefit across multiple seeds.

    ``n_seeds`` is the independent Monte Carlo unit for every row. The
    robustness row additionally uses ``n_trials`` matched trials per seed and
    contamination rate; those trials are reduced within seed before the row is
    summarized, so they are not treated as independent seeds.

    Returns a JSON-serialisable report with a ``studies`` list. Each study entry
    has keys ``study``, ``label``, ``metric``, ``values``, ``unit``, ``mean``,
    ``std``, ``ci_lo``, and ``ci_hi``.
    """
    seeds = list(range(seed, seed + n_seeds))

    bs_gap_vals: list[float] = []
    for s in seeds:
        r_comm = run_belief_sharing(s, communicate=True)
        r_iso = run_belief_sharing(s, communicate=False)
        bs_gap_vals.append(r_comm["mean_accuracy"] - r_iso["mean_accuracy"])

    kl_red_vals: list[float] = []
    for s in seeds:
        r = run_language_acquisition(s)
        kl_red_vals.append(r["initial_kl"] - r["final_kl"])

    emerge_vals: list[float] = []
    for s in seeds:
        r = run_emergence(s)
        emerge_vals.append(r["delta_F_redundant"])

    robust_vals: list[float] = []
    for s in seeds:
        r = run_robustness_sweep(s, n_trials=n_trials)
        worst_rate_key = str(r["worst_rate"])
        # The mechanistic ``accuracy_by_method_and_rate`` curve is a single
        # fixed-colony trajectory. Cross-study inference must use the matched
        # trial profile, reduced within seed, so nested trials are not promoted
        # to independent seed-level observations.
        profile = r.get("per_rate_summary", {}).get(worst_rate_key)
        if profile is None:  # backward-compatible guard for old report schemas
            acc_map = r["accuracy_by_method_and_rate"]
            naive_acc = float(acc_map["KLD"][worst_rate_key])
            best_robust = max(
                (
                    float(acc_map[d][worst_rate_key])
                    for d in r["divergences"]
                    if d != "KLD"
                ),
                default=naive_acc,
            )
            robust_vals.append(best_robust - naive_acc)
            continue
        robust_differences = [
            float(profile["differences"][d]["mean"])
            for d in r["divergences"]
            if d != "KLD"
        ]
        robust_vals.append(max(robust_differences, default=0.0))

    mw_vals: list[float] = []
    for s in seeds:
        r = run_moving_world(s)
        mw_vals.append(r["accuracy"]["efe_guided"] - r["accuracy"]["isolated"])

    hi_vals: list[float] = []
    for s in seeds:
        r = run_hierarchical_world(s)
        hi_vals.append(r["location_accuracy_gap"])

    nl_vals: list[float] = []
    for s in seeds:
        r = run_3level_world(s)
        nl_vals.append(r["location_accuracy_gap"])

    sens_vals: list[float] = []
    for s in seeds:
        r = run_belief_sharing_sensitivity(s, n_trials=CROSS_STUDY_SENS_N_TRIALS)
        sens_vals.append(float(np.mean(r["accuracy_gap_grid"])))

    pr_vals: list[float] = []
    for s in seeds:
        r = run_parameter_recovery(s, n_trials=5, n_observations=20, fit_resolution=20)
        pr_vals.append(float(r["r_squared"]))

    studies: list[dict[str, Any]] = [
        {
            "study": 1,
            "label": "Study 1\nBelief sharing",
            "metric": "Accuracy gain (comm − iso)",
            "values": bs_gap_vals,
            "unit": "fraction",
        },
        {
            "study": 2,
            "label": "Study 2\nLanguage acquisition",
            "metric": "KL reduction (initial − final)",
            "values": kl_red_vals,
            "unit": "nats",
        },
        {
            "study": 3,
            "label": "Study 3\nEmergence (BMR)",
            "metric": "ΔF (redundant pruning)",
            "values": emerge_vals,
            "unit": "nats",
        },
        {
            "study": 4,
            "label": "Study 4\nRobustness sweep",
            "metric": "Trial-mean accuracy gain (best robust − naive, worst rate)",
            "values": robust_vals,
            "unit": "fraction",
            "within_seed_n_trials": int(n_trials),
            "estimand": (
                "best robust-minus-naive mean over matched trials at the worst "
                "contamination rate, reduced within each seed"
            ),
        },
        {
            "study": 5,
            "label": "Study 5\nMoving world (EFE)",
            "metric": "Accuracy gain (EFE − isolated)",
            "values": mw_vals,
            "unit": "fraction",
        },
        {
            "study": 6,
            "label": "Study 6\n2-level hierarchical POMDP",
            "metric": "Location accuracy gap (hier − flat)",
            "values": hi_vals,
            "unit": "fraction",
        },
        {
            "study": 7,
            "label": "Study 7\n3-level hierarchical POMDP",
            "metric": "Location accuracy gap (3-level − flat)",
            "values": nl_vals,
            "unit": "fraction",
        },
        {
            "study": 8,
            "label": "Study 8\nSensitivity sweep",
            "metric": "Mean accuracy gap (comm - iso, 5x5 grid)",
            "values": sens_vals,
            "unit": "fraction",
        },
        {
            "study": 9,
            "label": "Study 9\nParameter recovery",
            "metric": "R-squared (acuity identifiability)",
            "values": pr_vals,
            "unit": "R-sq",
        },
    ]

    rng = np.random.default_rng(seed)
    for entry in studies:
        x = np.array(entry["values"], dtype=np.float64)
        entry["mean"] = float(x.mean())
        entry["std"] = float(x.std(ddof=1)) if x.size >= 2 else 0.0
        ci_lo, ci_hi = bootstrap_ci(x, rng=rng)
        entry["ci_lo"] = float(ci_lo)
        entry["ci_hi"] = float(ci_hi)

    return {
        "studies": studies,
        "seed": int(seed),
        "n_seeds": int(n_seeds),
        "n_trials": int(n_trials),
    }


__all__ = ["CROSS_STUDY_SENS_N_TRIALS", "summarize_cross_study"]
