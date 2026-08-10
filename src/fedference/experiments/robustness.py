"""Experiment harness submodule — see :mod:`fedference.experiments`."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..colonies import soft_colony
from ..contamination import contaminate
from ..pomdp import N_LOCATIONS
from ..statistics import (
    bh_fdr,
    bootstrap_ci,
    interpret_effect_size,
    paired_test,
    per_group_test,
    power_analysis,
    sample_size_for_power,
    summary_statistics,
)
from ._common import (
    _N_BOOT,
    ArrayF,
    _consensus_accuracy,
    _divergence_to_robustness,
    _finite_d_equivalent,
)


def run_robustness_sweep(
    seed: int,
    *,
    rates=(0.0, 0.225, 0.45, 0.675, 0.9),
    divergences=("KLD", "RKL", "AR", "beta", "rcce"),
    n_agents: int = 7,
    n_contaminated: int = 2,
    healthy_confidence: float = 0.35,
    accuracy_threshold: float = 0.5,
    n_trials: int = 40,
    verdict_rate: float = 0.8,
    fdr_alpha: float = 0.05,
    power_alpha: float = 0.05,
    power_alternative: str = "greater",
    target_power: float = 0.80,
    kind: str = "confident_wrong",
) -> dict[str, Any]:
    """Sweep contamination rate x divergence; earn the robust-vs-naive verdict.

    ``kind`` selects the contamination mechanism (default ``'confident_wrong'``,
    which leaves the locked headline behaviour bit-identical). Passing
    ``'byzantine'`` or ``'drift'`` produces the same report schema under those
    attacks so the full rate-vs-accuracy curve and per-rate paired tests can be
    inspected for any mechanism; ``'drift'`` is evaluated at its terminal round
    (full phase). Note the honest caveat (see the contamination gallery): a
    multiplicative ``'byzantine'`` attack escalates to a veto cliff at high rate
    where every pool collapses, so its worst-rate "robust holds" assertion need
    not hold — only the additive ``'confident_wrong'``/``'drift'`` attacks keep a
    robust member above threshold at the worst rate.

    A colony of ``n_agents`` sentinels broadcasts soft beliefs about a shared
    hidden state. ``n_contaminated`` of them are *saboteurs* whose belief is
    convex-mixed toward a confident-wrong delta
    (:func:`fedference.contamination.contaminate`, ``confident_wrong``) at each
    sweep ``rate``. We must avoid the degenerate ``rate = 1`` pure-veto limit:
    in a product-of-experts pool a fully-confident wrong delta forces *every*
    pooling rule's accuracy to zero (the outlier's near-zero mass on the true
    state multiplies the consensus down), so the default sweep tops out below 1.

    For every configured operating-point label we fuse the colony — ``'KLD'`` is
    the naive project pool (:func:`fedference.aggregation.log_linear_pool`,
    robustness 0), a categorical Eq. 7 specialization under the documented
    bridge assumptions rather than a full source-protocol reconstruction. The
    rest are robust pools (:func:`fedference.belief_sharing.share_round` with
    a positive robustness). The labels retain the FedGVI vocabulary for
    cross-reference only; the returned ``server_robustness_by_label`` mapping
    records the actual constants used by this server heuristic.

    Two claims are validated (ISC-27, ISC-30):

    * the **naive** (``KLD``) consensus accuracy degrades monotonically as the
      contamination rate rises;
    * at the worst (highest) rate, at least one **robust** method keeps the
      consensus accuracy at or above ``accuracy_threshold``.

    The headline robust-beats-naive verdict is computed, never asserted. We run
    ``n_trials`` conditionally independent contamination trials at
    ``verdict_rate`` (heavy contamination kept below the pure-veto cliff) — each
    trial redraws the heterogeneous healthy colony while the run's true state and
    attack target remain fixed — and for every robust divergence pair its
    per-trial consensus accuracy against the naive pool's via
    :func:`fedference.statistics.paired_test` (Wilcoxon signed-rank, with
    rank-biserial effect size). The family of per-method p-values is then
    deflated with :func:`fedference.statistics.bh_fdr`. A positive effect with
    BH rejection is recorded as a declared server-side statistical signal; it
    is not a unique scientific winner, a universal robustness result, or a
    transfer of the client-side FedGVI guarantee. Pairing many trials at a
    fixed high-contamination level — rather than the few points of the rate
    curve — is what gives the test the power to detect the robustness effect.

    Returns a JSON-serialisable dict with ``accuracy_by_method_and_rate`` (a
    nested ``{method: {rate: accuracy}}`` dict), ``per_rate_summary`` (trial
    means, medians, MCSEs, MDEs, and bootstrap intervals for each rate),
    ``naive_degrades_with_rate``, ``robust_above_threshold_at_worst_rate``,
    ``verdict`` (a per-robust-method
    ``{method: {pvalue, qvalue, effect_size, d_equivalent, rejected, wins, power,
    n_for_target_power}}`` dict), ``any_robust_wins``, the observed-effect
    design ``headline_power`` of the best robust method's paired Wilcoxon at
    ``n_trials``, the fixed ``true_state`` and ``attack_target_state`` used by
    the conditional trial profile, the prospective ``prospective_n_for_target_power`` a
    confirmatory replication should budget to reach ``target_power`` at the
    headline observed effect, and the sweep parameters.
    """
    rates = [float(r) for r in rates]
    divergences = [str(d) for d in divergences]
    if not rates:
        raise ValueError("rates must be non-empty")
    if "KLD" not in divergences:
        raise ValueError("the naive baseline 'KLD' must be in divergences")
    if n_agents < 3:
        raise ValueError("robustness sweep needs at least three agents")
    if not 1 <= n_contaminated < n_agents:
        raise ValueError("n_contaminated must be in [1, n_agents)")
    if not 0.0 < healthy_confidence < 1.0:
        raise ValueError("healthy_confidence must lie in (0, 1)")
    if n_trials < 2:
        raise ValueError("n_trials must be >= 2 for a paired test")
    if not 0.0 <= verdict_rate <= 1.0:
        raise ValueError("verdict_rate must lie in [0, 1]")

    rng = np.random.default_rng(seed)
    n_s = N_LOCATIONS
    true_state = int(rng.integers(0, n_s))
    wrong_state = int((true_state + n_s // 2) % n_s)
    robust_methods = [d for d in divergences if d != "KLD"]

    def contaminate_colony(clean: ArrayF, rate: float) -> ArrayF:
        """Corrupt the first ``n_contaminated`` agents under the selected ``kind``."""
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
        return local_posteriors

    # --- The rate sweep (ISC-27/30 trend claims) ----------------------------
    # One fixed seeded colony; the only varying knob is the contamination rate.
    sweep_colony = soft_colony(
        true_state, n_agents, n_s, healthy_confidence, rng, jitter=0.03
    )
    accuracy: dict[str, dict[str, float]] = {d: {} for d in divergences}
    for rate in rates:
        local_posteriors = contaminate_colony(sweep_colony, rate)
        for d in divergences:
            accuracy[d][f"{rate:g}"] = _consensus_accuracy(
                local_posteriors, d, true_state
            )

    naive_curve = np.array([accuracy["KLD"][f"{r:g}"] for r in rates])
    # Monotone non-increasing (allow a tiny numerical slack).
    naive_degrades = bool(
        all(naive_curve[i] >= naive_curve[i + 1] - 1e-9 for i in range(len(naive_curve) - 1))
        and naive_curve[0] > naive_curve[-1]
    )

    worst_rate = max(rates)
    worst_rate_key = f"{worst_rate:g}"
    robust_above_threshold = bool(
        any(accuracy[d][worst_rate_key] >= accuracy_threshold for d in robust_methods)
    )

    # --- Earn the verdict: many paired trials at ``verdict_rate``, BH-FDR ----
    # Each trial redraws the heterogeneous healthy colony so the pairs are
    # independent samples, giving the Wilcoxon test the power the few-point rate
    # curve lacks. We pair at ``verdict_rate`` — heavy contamination that
    # degrades the naive pool but stays below the pure-veto cliff (where every
    # rule collapses to zero and the contrast vanishes). A null result is
    # recorded if the rate is so low that the paired differences all vanish.
    # The naive log-linear pool sharpens (raw weights sum to ``n_agents``) while
    # the robust pool renormalizes its weights to sum to one, so the two
    # consensuses never coincide exactly even on a homogeneous colony — the
    # paired differences are always non-zero by construction and Wilcoxon is
    # always defined here. No degenerate-tie guard is needed.
    naive_trials: list[float] = []
    robust_trials: dict[str, list[float]] = {d: [] for d in robust_methods}
    for _ in range(n_trials):
        trial_clean = soft_colony(
            true_state, n_agents, n_s, healthy_confidence, rng, jitter=0.08
        )
        trial_local_posteriors = contaminate_colony(trial_clean, verdict_rate)
        naive_trials.append(
            _consensus_accuracy(trial_local_posteriors, "KLD", true_state)
        )
        for d in robust_methods:
            robust_trials[d].append(
                _consensus_accuracy(trial_local_posteriors, d, true_state)
            )

    verdict: dict[str, dict] = {}
    pvalues = []
    effect_sizes = []
    for d in robust_methods:
        test = paired_test(naive_trials, robust_trials[d])
        pvalues.append(test["pvalue"])
        effect_sizes.append(test["effect_size"])
        # Secondary display transform for the headline verdict: convert the
        # rank-biserial r to a d-equivalent and label its magnitude. This is the
        # *aggregation heuristic*'s effect — kept distinct from any beta/rcce
        # per-agent FedGVI guarantee (those live in the agent update, not here).
        d_equivalent = _finite_d_equivalent(test["effect_size"])
        # Observed-effect design power of THIS paired Wilcoxon contrast at the
        # run's n_trials, plus the prospective n needed for ``target_power`` at
        # the observed effect. Power is computed on the finite d-equivalent (the
        # boundary is represented by an explicit saturation sentinel). This quantifies
        # the planning strength of the AGGREGATION HEURISTIC's contrast — it
        # does NOT certify the beta/rcce per-agent FedGVI guarantee.
        pa = power_analysis(
            d_equivalent, n_trials, alpha=power_alpha, alternative=power_alternative
        )
        verdict[d] = {
            "pvalue": float(test["pvalue"]),
            "effect_size": float(test["effect_size"]),
            "statistic": float(test["statistic"]),
            "d_equivalent": float(d_equivalent),
            "effect_label": interpret_effect_size(d_equivalent),
            "power": float(pa["power"]),
            "n_for_target_power": int(pa["n_for_80_power"]),
        }

    fdr = bh_fdr(np.array(pvalues), alpha=fdr_alpha)
    any_robust_wins = False
    for i, d in enumerate(robust_methods):
        rejected = bool(fdr["rejected"][i])
        # effect_size is paired_test(naive, robust): positive means robust > naive.
        wins = bool(rejected and effect_sizes[i] > 0.0)
        verdict[d]["qvalue"] = float(fdr["qvalues"][i])
        # Expose the raw (uncorrected) p-value alongside the BH q-value.
        verdict[d]["raw_pvalue"] = float(fdr["pvalues"][i])
        verdict[d]["rejected"] = rejected
        verdict[d]["wins"] = wins
        any_robust_wins = any_robust_wins or wins

    # --- Enrichment: per-condition means with bootstrap 95% CIs -------------
    # Each method's per-trial accuracy at ``verdict_rate`` is a real sample, so
    # we report its mean and a bootstrap CI of that mean, and (for every robust
    # method) the bootstrap CI of the per-trial robust-minus-naive accuracy
    # difference — the standardized-effect's interval companion for the verdict.
    naive_lo, naive_hi = bootstrap_ci(
        naive_trials, alpha=0.05, n_boot=_N_BOOT, rng=rng
    )
    accuracy_at_verdict_rate: dict[str, dict] = {
        "KLD": {
            "n": int(n_trials),
            "mean": float(np.mean(naive_trials)),
            "ci": [naive_lo, naive_hi],
        }
    }
    for d in robust_methods:
        d_lo, d_hi = bootstrap_ci(
            robust_trials[d], alpha=0.05, n_boot=_N_BOOT, rng=rng
        )
        accuracy_at_verdict_rate[d] = {
            "n": int(n_trials),
            "mean": float(np.mean(robust_trials[d])),
            "ci": [d_lo, d_hi],
        }
        # CI of the paired (robust - naive) accuracy difference: the interval
        # companion to the headline standardized effect size.
        diffs = [
            robust_trials[d][t] - naive_trials[t] for t in range(n_trials)
        ]
        diff_lo, diff_hi = bootstrap_ci(diffs, alpha=0.05, n_boot=_N_BOOT, rng=rng)
        verdict[d]["mean_accuracy_diff"] = float(np.mean(diffs))
        verdict[d]["mean_accuracy_diff_ci"] = [diff_lo, diff_hi]

    # --- Enrichment: per-contamination-rate paired tests --------------------
    # For every sweep rate we draw ``n_trials`` conditionally independent
    # heterogeneous colonies, contaminate at that rate, and pair the naive pool's accuracy
    # against each robust member's via ``per_group_test`` (Wilcoxon +
    # rank-biserial). The collected raw p-values across rates are deflated per
    # method with BH-FDR, so every rate-resolved contrast carries both a raw and
    # an adjusted p-value plus a standardized effect size. This is run AFTER the
    # verdict draws so none of the back-compat values shift.
    per_rate_naive: dict[str, list[float]] = {f"{r:g}": [] for r in rates}
    per_rate_robust: dict[str, dict[str, list[float]]] = {
        f"{r:g}": {d: [] for d in robust_methods} for r in rates
    }
    for rate in rates:
        rkey = f"{rate:g}"
        for _ in range(n_trials):
            trial_clean = soft_colony(
                true_state, n_agents, n_s, healthy_confidence, rng, jitter=0.08
            )
            trial_local_posteriors = contaminate_colony(trial_clean, rate)
            per_rate_naive[rkey].append(
                _consensus_accuracy(trial_local_posteriors, "KLD", true_state)
            )
            for d in robust_methods:
                per_rate_robust[rkey][d].append(
                    _consensus_accuracy(trial_local_posteriors, d, true_state)
                )

    # Build the {method: {rate: contrast}} structure, then BH-deflate each
    # method's family of per-rate raw p-values.
    paired_tests_by_rate: dict[str, dict[str, dict]] = {d: {} for d in robust_methods}
    for d in robust_methods:
        rate_keys = [f"{r:g}" for r in rates]
        groups = [
            (per_rate_naive[rk], per_rate_robust[rk][d]) for rk in rate_keys
        ]
        results = per_group_test(groups)
        raw_p = np.array([res["pvalue"] for res in results])
        fdr_d = bh_fdr(raw_p, alpha=fdr_alpha)
        for j, rk in enumerate(rate_keys):
            res = results[j]
            d_equivalent = _finite_d_equivalent(res["effect_size"])
            paired_tests_by_rate[d][rk] = {
                "statistic": float(res["statistic"]),
                "pvalue": float(res["pvalue"]),
                "raw_pvalue": float(fdr_d["pvalues"][j]),
                "qvalue": float(fdr_d["qvalues"][j]),
                "rejected": bool(fdr_d["rejected"][j]),
                "effect_size": float(res["effect_size"]),
                "d_equivalent": float(d_equivalent),
                "effect_label": interpret_effect_size(d_equivalent),
            }

    # --- Descriptive rate profile ------------------------------------------
    # The original ``accuracy_by_method_and_rate`` curve is intentionally a
    # single fixed seeded colony: it is the mechanistic trajectory. This
    # companion block averages the independent matched trials generated above,
    # so readers can see uncertainty without mistaking a single trajectory for
    # a population estimate. Every summary is over trials, not clients or
    # observations nested inside a trial.
    per_rate_summary: dict[str, dict[str, Any]] = {}
    summary_rng = np.random.default_rng(seed + 10_000)
    for rate in rates:
        rate_key = f"{rate:g}"
        methods = {
            "KLD": summary_statistics(
                per_rate_naive[rate_key], n_boot=_N_BOOT, rng=summary_rng
            )
        }
        for d in robust_methods:
            methods[d] = summary_statistics(
                per_rate_robust[rate_key][d], n_boot=_N_BOOT, rng=summary_rng
            )
        differences = {
            d: summary_statistics(
                [
                    per_rate_robust[rate_key][d][i] - per_rate_naive[rate_key][i]
                    for i in range(n_trials)
                ],
                n_boot=_N_BOOT,
                rng=summary_rng,
            )
            for d in robust_methods
        }
        per_rate_summary[rate_key] = {
            "n": int(n_trials),
            "methods": methods,
            "differences": differences,
        }

    robust_mean_diff = {
        d: float(verdict[d]["mean_accuracy_diff"]) for d in robust_methods
    }
    largest_mean_difference_method = (
        max(robust_methods, key=lambda method: robust_mean_diff[method])
        if robust_methods
        else ""
    )
    worst_profile_methods = per_rate_summary[worst_rate_key]["methods"]
    worst_rate_best_method = (
        max(
            robust_methods,
            key=lambda method: float(worst_profile_methods[method]["mean"]),
        )
        if robust_methods
        else ""
    )

    # --- Headline planning diagnostics + selection disclosure ----------------
    # The headline method is selected by the predeclared largest-positive
    # rank-biserial rule. Ties are first-class: the complete tied set and the
    # stable method-order tie-break are returned, so this label is not a claim of
    # a unique scientific winner. We report the
    # observed-effect design power of its paired Wilcoxon at the run's
    # ``n_trials`` and the prospective sample size a confirmatory replication
    # should budget to reach
    # ``target_power`` at that observed effect. Honesty contract: this is the
    # power of the SERVER-SIDE robust-aggregation heuristic's contrast, NOT a
    # statement about the beta/rcce per-agent generalized-Bayes guarantee.
    selection_rule = "largest positive rank-biserial effect_size; stable method order tie-break"
    best_method = ""
    best_effect = None
    if robust_methods:
        best_effect = max(float(verdict[d]["effect_size"]) for d in robust_methods)
        tie_tolerance = 1e-12
        headline_tie_set = [
            d
            for d in robust_methods
            if abs(float(verdict[d]["effect_size"]) - best_effect) <= tie_tolerance
        ]
        best_method = headline_tie_set[0]
    else:  # pragma: no cover - robust_methods is non-empty whenever KLD has peers
        headline_tie_set = []
    if best_method:
        headline_power = float(verdict[best_method]["power"])
        headline_n_for_target = int(verdict[best_method]["n_for_target_power"])
        headline_effect = float(verdict[best_method]["effect_size"])
    else:  # pragma: no cover - robust_methods is non-empty whenever KLD has peers
        headline_power = 0.0
        headline_n_for_target = 0
        headline_effect = 0.0
    # Prospective sample-size justification computed from the headline observed
    # effect directly (independent of the verdict dict, for the prose token).
    prospective_n = int(
        sample_size_for_power(
            target_power=target_power,
            effect_size=headline_effect,
            alpha=power_alpha,
            alternative=power_alternative,
        )
    )

    return {
        "schema_version": "2.0",
        "accuracy_by_method_and_rate": accuracy,
        "naive_degrades_with_rate": naive_degrades,
        "robust_above_threshold_at_worst_rate": robust_above_threshold,
        "accuracy_threshold": float(accuracy_threshold),
        "verdict": verdict,
        "accuracy_at_verdict_rate": accuracy_at_verdict_rate,
        "per_rate_summary": per_rate_summary,
        "paired_tests_by_rate": paired_tests_by_rate,
        "any_robust_wins": any_robust_wins,
        "worst_rate": float(worst_rate),
        "verdict_rate": float(verdict_rate),
        "kind": str(kind),
        "n_trials": int(n_trials),
        "naive_verdict_rate_mean": float(np.mean(naive_trials)),
        # Sample size behind every paired contrast (trials per condition).
        "n": int(n_trials),
        # Executed BH-FDR level of the verdict panel and the per-rate families
        # (the manuscript's STATISTICS_FDR_ALPHA token reads THIS, so the
        # reported level is always the level the test actually ran at).
        "fdr_alpha": float(fdr_alpha),
        # --- Power analysis of the headline robust-vs-naive verdict ----------
        "power_alpha": float(power_alpha),
        "power_alternative": str(power_alternative),
        "target_power": float(target_power),
        # Achieved power of the best robust method's paired Wilcoxon at n_trials.
        "headline_power": headline_power,
        # Prospective n for ``target_power`` at the headline observed effect.
        "headline_n_for_target_power": headline_n_for_target,
        "prospective_n_for_target_power": prospective_n,
        "headline_method": best_method,
        "headline_selection_rule": selection_rule,
        "headline_tie_set": headline_tie_set,
        "headline_tie_break": "first robust method in divergences order",
        "headline_is_display_selection": True,
        "largest_mean_difference_method": largest_mean_difference_method,
        "worst_rate_best_method": worst_rate_best_method,
        "paired_test_alternative": "two-sided",
        "fdr_family_ownership": (
            "one BH family across robust methods at verdict_rate; one BH family "
            "per robust method across the declared per-rate contrasts"
        ),
        "d_equivalent_status": (
            "secondary rank-biserial-derived display transform; finite saturation "
            "sentinels are not literal million-scale effects"
        ),
        "analysis_unit": (
            "matched trial for the fixed verdict world; per-rate summaries are "
            "nested trial profiles"
        ),
        "trial_structure": (
            "trials redraw a heterogeneous colony while true state and attack "
            "target remain fixed"
        ),
        "true_state": true_state,
        "attack_target_state": wrong_state,
        "rates": rates,
        "divergences": divergences,
        "server_robustness_by_label": {
            label: float(_divergence_to_robustness(label)) for label in divergences
        },
        "seed": int(seed),
    }
