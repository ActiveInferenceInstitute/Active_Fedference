"""Experiment harness submodule — see :mod:`fedference.experiments`."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..colonies import soft_colony
from ..contamination import _KINDS as _CONTAMINATION_KINDS
from ..contamination import contaminate
from ..pomdp import N_LOCATIONS
from ..statistics import bootstrap_ci
from ._common import (
    _N_BOOT,
    _consensus_accuracy,
)

#: Contamination mechanisms that pull the consensus toward a chosen WRONG state.
_DIRECTIONAL_KINDS = ("confident_wrong", "byzantine", "drift")
#: Mechanisms that raise entropy without a fixed wrong target.
_ENTROPY_KINDS = ("uniform", "label_noise")
#: Across-seed win fraction required for a "reliable" robust advantage.
_GALLERY_RELIABLE_WIN_FRACTION = 0.95


def run_contamination_gallery(
    seed: int,
    *,
    kinds=_CONTAMINATION_KINDS,
    n_agents: int = 7,
    n_contaminated: int = 2,
    healthy_confidence: float = 0.35,
    rate: float = 0.6,
    divergences=("KLD", "RKL", "AR", "beta", "rcce"),
    n_trials: int = 20,
    n_seeds: int = 16,
) -> dict[str, Any]:
    """Descriptive pooled-method summary under each contamination mechanism.

    The headline :func:`run_robustness_sweep` stresses one attack model
    (``confident_wrong``). This gallery descriptively compares the declared
    mechanisms rather than treating a single mechanism or a single seed as
    representative: for every kind it runs ``n_trials`` paired trials across ``n_seeds``
    independent seeds, corrupting ``n_contaminated`` of ``n_agents`` sentinels by
    that mechanism at strength ``rate`` and fusing under the naive log-linear pool
    and each robust divergence (:func:`fedference.contamination.contaminate`
    feeding :func:`_consensus_accuracy`). The ``drift`` mechanism is evaluated at
    its terminal round (full phase) so it acts at full strength in this
    single-round gallery.

    The gallery selects its pooled display method from the same seed-level data
    it summarizes. Its ``reliably_beats`` field is therefore a descriptive
    display label, not selection-free inference or a generalization claim. The
    all-method review grid owns the selection-free signed comparisons. For the
    display summary, a mechanism's pooled-method-minus-naive accuracy difference
    is aggregated over the ``n_seeds`` seeds — its mean, a bootstrap CI, and the
    *win fraction* (seeds in which the displayed method exceeds naive). A
    mechanism is flagged ``reliably_beats`` only if the win fraction reaches
    :data:`_GALLERY_RELIABLE_WIN_FRACTION` *and* the difference-CI excludes zero.
    The directional/entropy split is by attack intent; ``reliably_beats`` is the
    earned, seed-robust verdict (e.g. the multiplicative byzantine attack is
    directional but its advantage is NOT reliable — it escalates to a veto cliff).

    Returns a JSON-serialisable dict with ``by_kind`` (per-mechanism
    ``naive_mean``, ``naive_ci``, ``robust_mean``, ``robust_ci``,
    ``best_robust_method``, ``mean_diff``, ``diff_ci``, ``win_fraction``,
    ``reliably_beats``, ``directional``),
    ``reliable_kinds`` (mechanisms with a reliable robust advantage),
    ``entropy_naive_robust``, and the sweep parameters.
    """
    kinds = [str(k) for k in kinds]
    divergences = [str(d) for d in divergences]
    if "KLD" not in divergences:
        raise ValueError("the naive baseline 'KLD' must be in divergences")
    if n_agents < 3:
        raise ValueError("contamination gallery needs at least three agents")
    if not 1 <= n_contaminated < n_agents:
        raise ValueError("n_contaminated must be in [1, n_agents)")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must lie in [0, 1]")
    if n_trials < 2:
        raise ValueError("n_trials must be >= 2")
    if n_seeds < 2:
        raise ValueError("n_seeds must be >= 2 for a seed-robust verdict")

    robust_methods = [d for d in divergences if d != "KLD"]
    if not robust_methods:
        raise ValueError("at least one robust divergence is required")

    def mechanism_one_seed(s: int, kind: str) -> tuple[float, dict[str, float]]:
        """Mean naive accuracy and per-robust-method mean accuracy for one seed."""
        rng = np.random.default_rng(s)
        n_s = N_LOCATIONS
        true_state = int(rng.integers(0, n_s))
        wrong_state = int((true_state + n_s // 2) % n_s)
        naive: list[float] = []
        robust: dict[str, list[float]] = {d: [] for d in robust_methods}
        for _ in range(n_trials):
            clean = soft_colony(true_state, n_agents, n_s, healthy_confidence, rng, jitter=0.08)
            local_posteriors = clean.copy()
            for k in range(n_contaminated):
                local_posteriors[k] = contaminate(
                    clean[k],
                    kind=kind,
                    rate=rate,
                    rng=rng,
                    wrong_state=wrong_state,
                    target_state=wrong_state,
                    round_index=1 if kind == "drift" else 0,
                    n_rounds=2,
                )
            naive.append(_consensus_accuracy(local_posteriors, "KLD", true_state))
            for d in robust_methods:
                robust[d].append(_consensus_accuracy(local_posteriors, d, true_state))
        return float(np.mean(naive)), {d: float(np.mean(robust[d])) for d in robust_methods}

    seeds = list(range(seed, seed + n_seeds))
    boot_rng = np.random.default_rng(seed)
    by_kind: dict[str, dict] = {}
    for kind in kinds:
        naive_by_seed: list[float] = []
        robust_by_seed: dict[str, list[float]] = {d: [] for d in robust_methods}
        for s in seeds:
            nm, rm = mechanism_one_seed(s, kind)
            naive_by_seed.append(nm)
            for d in robust_methods:
                robust_by_seed[d].append(rm[d])
        # Fix the pooled display method by its mean across seeds (no per-seed cherry-pick),
        # then form the paired per-seed advantage of THAT method over naive.
        method_means = {d: float(np.mean(robust_by_seed[d])) for d in robust_methods}
        best_method = max(method_means, key=lambda d: method_means[d])
        diffs = [robust_by_seed[best_method][i] - naive_by_seed[i] for i in range(n_seeds)]
        diffs_arr = np.array(diffs)
        diff_lo, diff_hi = bootstrap_ci(diffs_arr, alpha=0.05, n_boot=_N_BOOT, rng=boot_rng)
        naive_lo, naive_hi = bootstrap_ci(np.asarray(naive_by_seed), alpha=0.05, n_boot=_N_BOOT, rng=boot_rng)
        robust_lo, robust_hi = bootstrap_ci(
            np.asarray(robust_by_seed[best_method]),
            alpha=0.05,
            n_boot=_N_BOOT,
            rng=boot_rng,
        )
        win_fraction = float(np.mean(diffs_arr > 0.0))
        mean_diff = float(np.mean(diffs_arr))
        reliably_beats = bool(win_fraction >= _GALLERY_RELIABLE_WIN_FRACTION and diff_lo > 0.0)
        by_kind[kind] = {
            "naive_mean": float(np.mean(naive_by_seed)),
            "naive_ci": [naive_lo, naive_hi],
            "robust_mean": method_means[best_method],
            "robust_ci": [robust_lo, robust_hi],
            "best_robust_method": best_method,
            "mean_diff": mean_diff,
            "diff_ci": [diff_lo, diff_hi],
            "win_fraction": win_fraction,
            "reliably_beats": reliably_beats,
            "directional": kind in _DIRECTIONAL_KINDS,
        }

    directional = [k for k in kinds if k in _DIRECTIONAL_KINDS]
    entropy = [k for k in kinds if k in _ENTROPY_KINDS]
    # The earned, seed-robust list: mechanisms where robust reliably beats naive.
    reliable_kinds = [k for k in kinds if by_kind[k]["reliably_beats"]]
    # Entropy attacks should leave the naive pool intact (robust merely conservative):
    # naive within a small margin of robust and no reliable robust win.
    entropy_naive_robust = bool(
        all(
            by_kind[k]["naive_mean"] >= by_kind[k]["robust_mean"] - 0.05 and not by_kind[k]["reliably_beats"]
            for k in entropy
        )
    )

    return {
        "by_kind": by_kind,
        "reliable_kinds": reliable_kinds,
        "entropy_naive_robust": entropy_naive_robust,
        "directional_kinds": directional,
        "entropy_kinds": entropy,
        "reliable_win_fraction": _GALLERY_RELIABLE_WIN_FRACTION,
        "kinds": kinds,
        "rate": float(rate),
        "n_trials": int(n_trials),
        "n_seeds": int(n_seeds),
        "n_agents": int(n_agents),
        "n_contaminated": int(n_contaminated),
        "seed": int(seed),
    }


def run_robustness_onset(
    seed: int,
    *,
    kinds=_DIRECTIONAL_KINDS,
    rates=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    n_agents: int = 7,
    n_contaminated: int = 2,
    healthy_confidence: float = 0.35,
    divergences=("KLD", "RKL", "AR", "beta", "rcce"),
    n_trials: int = 12,
    n_seeds: int = 6,
    onset_win_fraction: float = 0.95,
    include_seed_data: bool = False,
) -> dict[str, Any]:
    """Per-mechanism descriptive onset for a pooled display robust member.

    The gallery fixes one contamination strength; this maps the rate dependence.
    For each directional mechanism it sweeps ``rates`` and, at each rate,
    seed-aggregates (over ``n_seeds`` seeds × ``n_trials`` trials) the naive
    log-linear-pool accuracy and the pooled display member's accuracy, plus the
    across-seed win fraction. The **onset rate** is the smallest rate at which the
    pooled display win fraction reaches ``onset_win_fraction``. Because that
    method is selected from the same seed-level results, the onset is a
    descriptive display summary rather than a selection-free inferential claim.
    Mechanisms differ sharply: the multiplicative byzantine
    attack onsets early but then escalates to a veto cliff, while the additive
    confident-wrong/drift attacks onset later and the robust member stays above.

    Returns ``{by_kind: {kind: {rates, naive_curve, robust_curve, win_curve,
    naive_ci, robust_ci, best_robust_method_by_rate, onset_rate}}}``
    (``onset_rate`` is ``None`` if robust never reliably overtakes within the
    grid), plus the sweep parameters. No verdict is hardcoded.
    """
    kinds = [str(k) for k in kinds]
    rates = [float(r) for r in rates]
    divergences = [str(d) for d in divergences]
    if "KLD" not in divergences:
        raise ValueError("the naive baseline 'KLD' must be in divergences")
    if n_agents < 3 or not 1 <= n_contaminated < n_agents:
        raise ValueError("need n_agents >= 3 and n_contaminated in [1, n_agents)")
    if n_trials < 2 or n_seeds < 2:
        raise ValueError("n_trials and n_seeds must both be >= 2")

    robust_methods = [d for d in divergences if d != "KLD"]
    if not robust_methods:
        raise ValueError("at least one robust divergence is required")
    if not rates:
        raise ValueError("rates must be non-empty")
    if any(not 0.0 <= rate <= 1.0 for rate in rates):
        raise ValueError("rates must lie in [0, 1]")
    if rates != sorted(rates):
        raise ValueError("rates must be sorted in non-decreasing order")
    if not 0.0 <= onset_win_fraction <= 1.0:
        raise ValueError("onset_win_fraction must lie in [0, 1]")

    bootstrap_rng = np.random.default_rng(seed + 2_000_003)

    def cell(kind: str, rate: float) -> dict[str, Any]:
        """Return one rate cell with pooled method selection and seed intervals.

        The robust method is selected once, by its mean across all configured
        seeds, and then evaluated on those same seed-level means. Selecting a
        different method inside each seed would turn ``robust_curve`` into a
        per-seed maximum and overstate the reproducible advantage.
        """
        per_seed_naive: list[float] = []
        per_seed_robust: dict[str, list[float]] = {d: [] for d in robust_methods}
        for s in range(seed, seed + n_seeds):
            rng = np.random.default_rng(s)
            n_s = N_LOCATIONS
            true_state = int(rng.integers(0, n_s))
            wrong_state = int((true_state + n_s // 2) % n_s)
            naive: list[float] = []
            robust: dict[str, list[float]] = {d: [] for d in robust_methods}
            for _ in range(n_trials):
                clean = soft_colony(true_state, n_agents, n_s, healthy_confidence, rng, jitter=0.08)
                local_posteriors = clean.copy()
                for k in range(n_contaminated):
                    local_posteriors[k] = contaminate(
                        clean[k],
                        kind=kind,
                        rate=rate,
                        rng=rng,
                        wrong_state=wrong_state,
                        target_state=wrong_state,
                        round_index=1 if kind == "drift" else 0,
                        n_rounds=2,
                    )
                naive.append(_consensus_accuracy(local_posteriors, "KLD", true_state))
                for d in robust_methods:
                    robust[d].append(_consensus_accuracy(local_posteriors, d, true_state))
            per_seed_naive.append(float(np.mean(naive)))
            for method in robust_methods:
                per_seed_robust[method].append(float(np.mean(robust[method])))
        method_means = {method: float(np.mean(values)) for method, values in per_seed_robust.items()}
        best_method = max(robust_methods, key=lambda method: method_means[method])
        per_seed_naive_arr = np.asarray(per_seed_naive, dtype=np.float64)
        per_seed_best_arr = np.asarray(per_seed_robust[best_method], dtype=np.float64)
        naive_lo, naive_hi = bootstrap_ci(per_seed_naive_arr, alpha=0.05, n_boot=_N_BOOT, rng=bootstrap_rng)
        robust_lo, robust_hi = bootstrap_ci(per_seed_best_arr, alpha=0.05, n_boot=_N_BOOT, rng=bootstrap_rng)
        return {
            "naive_mean": float(np.mean(per_seed_naive_arr)),
            "robust_mean": float(np.mean(per_seed_best_arr)),
            "win_fraction": float(np.mean(per_seed_best_arr > per_seed_naive_arr)),
            "naive_ci": [naive_lo, naive_hi],
            "robust_ci": [robust_lo, robust_hi],
            "best_robust_method": best_method,
            # Preserve the full seed-level method matrix for selection-free
            # review-grid consumers. The legacy onset surface may still display
            # the pooled best method, but the source payload can recompute every
            # contrast without per-seed cherry-picking.
            "method_means": method_means,
            "per_seed_naive": per_seed_naive,
            "per_seed_robust": {
                method: [float(value) for value in values] for method, values in per_seed_robust.items()
            },
        }

    by_kind: dict[str, dict] = {}
    for kind in kinds:
        naive_curve, robust_curve, win_curve = [], [], []
        naive_ci, robust_ci, best_methods = [], [], []
        seed_cells: dict[str, dict[str, Any]] = {}
        onset_rate = None
        for rate in rates:
            result = cell(kind, rate)
            if include_seed_data:
                seed_cells[f"{rate:g}"] = result
            naive_curve.append(result["naive_mean"])
            robust_curve.append(result["robust_mean"])
            win_curve.append(result["win_fraction"])
            naive_ci.append(result["naive_ci"])
            robust_ci.append(result["robust_ci"])
            best_methods.append(result["best_robust_method"])
            if onset_rate is None and result["win_fraction"] >= onset_win_fraction:
                onset_rate = rate
        by_kind[kind] = {
            "rates": rates,
            "naive_curve": naive_curve,
            "robust_curve": robust_curve,
            "win_curve": win_curve,
            "naive_ci": naive_ci,
            "robust_ci": robust_ci,
            "best_robust_method_by_rate": best_methods,
            "onset_rate": onset_rate,
        }
        if include_seed_data:
            by_kind[kind]["by_rate"] = seed_cells

    return {
        "by_kind": by_kind,
        "kinds": kinds,
        "rates": rates,
        "onset_win_fraction": float(onset_win_fraction),
        "n_trials": int(n_trials),
        "n_seeds": int(n_seeds),
        "n_agents": int(n_agents),
        "n_contaminated": int(n_contaminated),
        "seed": int(seed),
    }
