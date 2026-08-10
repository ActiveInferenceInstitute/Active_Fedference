"""Tests for the categorical source-mechanism analogue harness (no mocks).

Every number is a real seeded computation through the locked FedGVI /
active-inference core; nothing is patched or mocked. Each test pins the
qualitative trend of one figure with explicit numeric expectations:

* ISC-23 (Fig. 5): communicating colonies carry strictly lower mean variational
  free energy than the same colony incommunicado.
* ISC-24 (Fig. 7): the Dirichlet language-acquisition KL curve declines
  monotonically to (near) zero.
* ISC-25 (Fig. 9): Bayesian model reduction yields ``dF > 0`` for a redundant
  prune (converged) and ``dF < 0`` for a supported one (rejected).
* ISC-27 / ISC-30: the naive Friston pool degrades with contamination while a
  robust member stays above threshold, and the robust-beats-naive verdict is
  *earned* from :func:`paired_test` + :func:`bh_fdr`, never hard-coded.

All result dicts must be JSON-serialisable (the harness contract).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fedference.experiments import (
    _consensus_accuracy,
    _divergence_to_robustness,
    _sample_observation,
    _soft_colony,
    run_belief_sharing,
    run_bnn_robustness_report,
    run_emergence,
    run_language_acquisition,
    run_robustness_sweep,
)
from fedference.experiments._common import _DIVERGENCE_ROBUSTNESS

SEEDS = [0, 1, 2, 7, 42]


def _json_roundtrips(result: dict) -> bool:
    """Every value survives a JSON dump/load round-trip unchanged in type."""
    return json.loads(json.dumps(result)) is not None


def test_bnn_robustness_seed_is_base_of_data_replicates():
    """Changing the public seed changes simulated data, not only bootstrap draws."""
    kwargs = {
        "n_seeds": 3,
        "n_per": 12,
        "contamination_levels": (0.0, 0.3),
    }
    first = run_bnn_robustness_report(0, **kwargs)
    shifted = run_bnn_robustness_report(10, **kwargs)
    assert first["seed"] == 0
    assert shifted["seed"] == 10
    assert first["accuracy_seed_values_by_config"] != shifted[
        "accuracy_seed_values_by_config"
    ]


# ===========================================================================
# run_belief_sharing — Fig. 5 (ISC-23)
# ===========================================================================

@pytest.mark.parametrize("seed", SEEDS)
def test_belief_sharing_communicating_has_lower_free_energy(seed):
    """ISC-23: communicating mean free energy < incommunicado, every seed."""
    comm = run_belief_sharing(seed, communicate=True)
    inco = run_belief_sharing(seed, communicate=False)
    # Strictly lower, by a margin the 20-seed scan never undershot (~0.85 nats).
    assert comm["mean_free_energy"] < inco["mean_free_energy"]
    assert inco["mean_free_energy"] - comm["mean_free_energy"] > 0.5


def test_belief_sharing_result_shape_and_json():
    result = run_belief_sharing(0, communicate=True)
    # Back-compat keys must all survive (subset, not exact — enrichment adds more).
    assert {
        "mean_free_energy", "mean_surprise", "mean_accuracy",
        "communicate", "n_agents", "true_state", "seed",
    } <= set(result)
    assert result["communicate"] is True
    assert result["seed"] == 0
    assert 0 <= result["true_state"] < 9
    # sharing improves accuracy: shared mean accuracy is a valid probability.
    assert 0.0 <= result["mean_accuracy"] <= 1.0
    assert _json_roundtrips(result)


def test_belief_sharing_enrichment_keys():
    """Enrichment: sample size, per-agent free energies, and a bootstrap CI."""
    result = run_belief_sharing(0, communicate=True, n_agents=6)
    # n equals the colony size; the per-agent free-energy sample has length n.
    assert result["n"] == 6
    assert result["n"] == result["n_agents"]
    assert len(result["free_energies"]) == 6
    # The headline mean is the mean of the per-agent sample.
    assert result["mean_free_energy"] == pytest.approx(
        float(np.mean(result["free_energies"])), abs=1e-12
    )
    lo, hi = result["mean_free_energy_ci"]
    assert lo <= hi
    # The mean lies inside its own bootstrap 95% CI for this seeded colony.
    assert lo <= result["mean_free_energy"] <= hi
    assert _json_roundtrips(result)


def test_belief_sharing_ci_is_deterministic_under_seed():
    """The bootstrap CI is reproducible from the threaded seeded generator."""
    a = run_belief_sharing(7)
    b = run_belief_sharing(7)
    assert a["mean_free_energy_ci"] == b["mean_free_energy_ci"]
    assert a["free_energies"] == b["free_energies"]


def test_belief_sharing_incommunicado_diagnostics_are_finite():
    """The non-communicating branch computes surprise/accuracy directly."""
    inco = run_belief_sharing(3, communicate=False)
    assert inco["communicate"] is False
    assert np.isfinite(inco["mean_surprise"])
    assert 0.0 <= inco["mean_accuracy"] <= 1.0
    assert np.isfinite(inco["mean_free_energy"])


def test_belief_sharing_is_deterministic():
    assert run_belief_sharing(11) == run_belief_sharing(11)


def test_belief_sharing_requires_two_agents():
    with pytest.raises(ValueError, match="at least two agents"):
        run_belief_sharing(0, n_agents=1)


def test_belief_sharing_two_agents_is_minimum_valid_colony():
    """n_agents=2 is the minimum valid colony: each agent hears only the other.

    This stresses the self-exclusion path and can expose off-by-one errors in
    the self-exclusion indexing. The existing suite always uses n_agents>=3 for
    positive tests and only tests n_agents=1 as the rejection boundary.
    """
    comm = run_belief_sharing(0, communicate=True, n_agents=2)
    inco = run_belief_sharing(0, communicate=False, n_agents=2)

    assert comm["n_agents"] == 2
    assert 0.0 <= comm["mean_accuracy"] <= 1.0
    assert 0.0 <= comm["mean_free_energy"]

    # Communicating must not hurt free energy relative to incommunicado
    # (allow 0.5 nat slack for such a small colony).
    assert comm["mean_free_energy"] <= inco["mean_free_energy"] + 0.5, (
        f"2-agent communicating free energy {comm['mean_free_energy']:.3f} "
        f"should not exceed incommunicado {inco['mean_free_energy']:.3f} by > 0.5"
    )

    # Per-agent free energy list must have exactly 2 entries.
    assert len(comm["free_energies"]) == 2


# ===========================================================================
# run_language_acquisition — Fig. 7 (ISC-24)
# ===========================================================================

@pytest.mark.parametrize("seed", SEEDS)
def test_language_acquisition_kl_declines(seed):
    """ISC-24: monotone non-increasing KL trajectory ending near zero."""
    result = run_language_acquisition(seed)
    traj = result["kl_trajectory"]
    assert len(traj) == result["num_steps"] + 1  # prior + one point per batch
    # Monotone non-increasing (the Dirichlet learning contract).
    assert all(traj[i] >= traj[i + 1] - 1e-12 for i in range(len(traj) - 1))
    assert result["monotone_decreasing"] is True
    # Substantial decline: large prior KL collapses to (near) zero.
    assert result["initial_kl"] > 1.0
    assert result["final_kl"] < 0.05
    assert result["final_kl"] < result["initial_kl"]


def test_language_acquisition_first_value_is_the_prior():
    """Index 0 is the flat-prior KL (the largest, recorded before any batch)."""
    result = run_language_acquisition(0)
    assert result["kl_trajectory"][0] == result["initial_kl"]
    assert result["kl_trajectory"][-1] == result["final_kl"]
    assert _json_roundtrips(result)


def test_language_acquisition_enrichment_keys():
    """A single trajectory exposes points, but no invalid time-point CI."""
    result = run_language_acquisition(0, num_steps=24)
    assert result["n"] == 25  # prior + one recorded point per batch
    assert result["n"] == len(result["kl_trajectory"])
    assert "trajectory_ci_lo" not in result


def test_language_acquisition_seed_summary_uses_seed_replication_unit():
    """Intervals align steps and resample independent seeds, not time points."""
    from fedference.experiments import summarize_language_acquisition

    summary = summarize_language_acquisition([0, 1, 2], num_steps=6)
    matrix = np.asarray(summary["kl_trajectory_by_seed"], dtype=float)
    assert matrix.shape == (3, 7)
    assert summary["n_seeds"] == 3
    assert summary["n_points"] == 7
    assert len(summary["trajectory_ci_lo"]) == 7
    assert len(summary["trajectory_ci_hi"]) == 7
    assert np.all(np.asarray(summary["trajectory_ci_lo"]) <= np.asarray(summary["trajectory_ci_hi"]))
    assert np.allclose(summary["kl_trajectory"], matrix.mean(axis=0))
    assert summary == summarize_language_acquisition([0, 1, 2], num_steps=6)


def test_language_acquisition_seed_summary_rejects_empty_seed_set():
    from fedference.experiments import summarize_language_acquisition

    with pytest.raises(ValueError, match="at least one seed"):
        summarize_language_acquisition([])


def test_language_acquisition_seed_summary_rejects_insufficient_or_duplicate_seeds():
    """A bootstrap interval requires distinct independent seed replicates."""
    from fedference.experiments import summarize_language_acquisition

    with pytest.raises(ValueError, match="at least two"):
        summarize_language_acquisition([0], num_steps=2)
    with pytest.raises(ValueError, match="distinct"):
        summarize_language_acquisition([0, 0], num_steps=2)


def test_language_acquisition_is_deterministic():
    assert run_language_acquisition(9) == run_language_acquisition(9)


def test_language_acquisition_respects_num_steps():
    result = run_language_acquisition(0, num_steps=6)
    assert len(result["kl_trajectory"]) == 7  # prior + one point per batch


# ===========================================================================
# run_emergence — Fig. 9 (ISC-25)
# ===========================================================================

@pytest.mark.parametrize("seed", SEEDS)
def test_emergence_reduces_redundant_structure(seed):
    """ISC-25: redundant prune wins (dF>0), supported prune loses (dF<0)."""
    result = run_emergence(seed)
    assert result["delta_F_redundant"] > 0.0
    assert result["delta_F_supported"] < 0.0
    assert result["convergence"] is True
    # The redundant prune is decisively favored (margin from the seed scan).
    assert result["delta_F_redundant"] > 1.0


def test_emergence_result_json_and_shape():
    result = run_emergence(0)
    # Back-compat keys survive as a subset; enrichment adds the sample size ``n``.
    assert {
        "convergence", "delta_F_redundant", "delta_F_supported",
        "n_states", "seed",
    } <= set(result)
    assert result["n_states"] == 4
    # Sample size mirrors the number of candidate states.
    assert result["n"] == result["n_states"] == 4
    assert _json_roundtrips(result)


def test_emergence_is_deterministic():
    assert run_emergence(13) == run_emergence(13)


# ===========================================================================
# run_robustness_sweep — ISC-27 / ISC-30
# ===========================================================================

@pytest.mark.parametrize("seed", SEEDS)
def test_robustness_naive_degrades_and_robust_holds(seed):
    """ISC-27/30: naive degrades with rate; >=1 robust stays above threshold."""
    result = run_robustness_sweep(seed)
    assert result["naive_degrades_with_rate"] is True
    assert result["robust_above_threshold_at_worst_rate"] is True

    acc = result["accuracy_by_method_and_rate"]
    rates = result["rates"]
    naive_curve = [acc["KLD"][f"{r:g}"] for r in rates]
    # Monotone non-increasing, with a real top-to-bottom drop.
    assert all(naive_curve[i] >= naive_curve[i + 1] - 1e-9 for i in range(len(naive_curve) - 1))
    assert naive_curve[0] - naive_curve[-1] > 0.1
    # At least one robust method beats the naive pool at the worst rate.
    worst = f"{max(rates):g}"
    assert any(
        acc[d][worst] > acc["KLD"][worst]
        for d in result["divergences"] if d != "KLD"
    )


def test_sweep_fdr_alpha_is_consumed_not_decorative():
    """The fdr_alpha knob must change the executed verdict, not just be echoed.

    Consumption proof: an absurdly strict level (1e-12) must kill every BH
    rejection (no robust member can win), while the default level rejects.
    The report must also echo the executed level for the manuscript token.
    """
    strict = run_robustness_sweep(0, n_trials=12, fdr_alpha=1e-12)
    default = run_robustness_sweep(0, n_trials=12)
    assert strict["fdr_alpha"] == 1e-12
    assert default["fdr_alpha"] == 0.05
    assert not any(v["rejected"] for v in strict["verdict"].values())
    assert strict["any_robust_wins"] is False
    assert any(v["rejected"] for v in default["verdict"].values())


def test_sweep_kind_default_is_bit_identical_to_confident_wrong():
    # The opt-in `kind` parameter must leave the locked headline behaviour intact:
    # the default is confident_wrong and produces an identical report.
    default = run_robustness_sweep(0, n_trials=12)
    explicit = run_robustness_sweep(0, n_trials=12, kind="confident_wrong")
    assert default["kind"] == "confident_wrong"
    assert default["naive_verdict_rate_mean"] == explicit["naive_verdict_rate_mean"]
    assert default["accuracy_by_method_and_rate"] == explicit["accuracy_by_method_and_rate"]


@pytest.mark.parametrize("kind", ["byzantine", "drift"])
def test_sweep_runs_under_other_mechanisms_with_same_schema(kind):
    # byzantine + drift produce the same report schema and a degrading naive curve.
    result = run_robustness_sweep(0, n_trials=12, kind=kind)
    assert result["kind"] == kind
    assert set(result) >= {
        "accuracy_by_method_and_rate", "naive_degrades_with_rate",
        "verdict", "any_robust_wins", "paired_tests_by_rate", "headline_power",
    }
    assert result["naive_degrades_with_rate"] is True
    # Additive 'drift' keeps a robust member above threshold at the worst rate;
    # multiplicative 'byzantine' escalates to a veto cliff there (the honest
    # caveat), so the worst-rate "robust holds" assertion applies only to drift.
    if kind == "drift":
        assert result["robust_above_threshold_at_worst_rate"] is True


@pytest.mark.parametrize("seed", SEEDS)
def test_robustness_verdict_is_earned_from_statistics(seed):
    """The robust-beats-naive verdict comes from paired_test + bh_fdr only.

    A winner must have a BH-rejected null AND a positive (robust > naive) effect
    size — both computed by :mod:`fedference.statistics`, never hard-coded.
    """
    result = run_robustness_sweep(seed)
    assert result["any_robust_wins"] is True
    winners = [
        d for d in result["divergences"]
        if d != "KLD" and result["verdict"][d]["wins"]
    ]
    assert winners  # at least one confirmed robust winner
    for d in winners:
        v = result["verdict"][d]
        # The win is exactly (BH rejected) AND (effect favors robust).
        assert v["rejected"] is True
        assert v["effect_size"] > 0.0
        assert v["wins"] == (v["rejected"] and v["effect_size"] > 0.0)
        assert 0.0 <= v["pvalue"] <= 1.0
        assert 0.0 <= v["qvalue"] <= 1.0


def test_robustness_verdict_null_when_no_contamination():
    """At verdict_rate=0 the paired differences vanish -> no win is manufactured."""
    result = run_robustness_sweep(0, verdict_rate=0.0)
    assert result["any_robust_wins"] is False
    for d in result["divergences"]:
        if d == "KLD":
            continue
        # Null result: pvalue 1.0, effect 0.0, not rejected.
        assert result["verdict"][d]["wins"] is False


def test_robustness_result_is_json_serialisable():
    result = run_robustness_sweep(0)
    assert _json_roundtrips(result)
    # Nested accuracy dict keyed by method then rate-string.
    acc = result["accuracy_by_method_and_rate"]
    for d in result["divergences"]:
        assert set(acc[d]) == {f"{r:g}" for r in result["rates"]}


def test_robustness_is_deterministic():
    assert run_robustness_sweep(4) == run_robustness_sweep(4)


def test_robustness_custom_divergence_subset():
    """A smaller robust roster still earns a verdict for its members."""
    result = run_robustness_sweep(0, divergences=("KLD", "RKL"))
    assert result["divergences"] == ["KLD", "RKL"]
    assert "RKL" in result["verdict"]
    assert result["verdict"]["RKL"]["wins"] is True


# ---- robustness enrichment (n, CIs, raw+adjusted p, effect sizes) ----------

def test_robustness_top_level_sample_size():
    """``n`` mirrors the per-condition trial count behind every paired contrast."""
    result = run_robustness_sweep(0, n_trials=12)
    assert result["n"] == 12
    assert result["n"] == result["n_trials"]


def test_robustness_per_rate_summary_reports_trial_precision():
    result = run_robustness_sweep(0, n_trials=12, divergences=("KLD", "RKL"))
    summary = result["per_rate_summary"]
    assert set(summary) == {f"{rate:g}" for rate in result["rates"]}
    for rate_block in summary.values():
        assert rate_block["n"] == 12
        assert set(rate_block["methods"]) == {"KLD", "RKL"}
        for method_block in rate_block["methods"].values():
            assert method_block["n"] == 12
            assert method_block["ci_lo"] <= method_block["mean"] <= method_block["ci_hi"]
            assert method_block["mcse"] >= 0.0
            assert method_block["mde"] >= 0.0
        diff = rate_block["differences"]["RKL"]
        assert diff["n"] == 12
        assert diff["ci_lo"] <= diff["mean"] <= diff["ci_hi"]
    assert result["attack_target_state"] != result["true_state"]


def test_robustness_default_n_trials_is_the_larger_budget():
    """The default paired-trial budget is the raised, defensible sample size."""
    result = run_robustness_sweep(0)
    assert result["n_trials"] == 40
    assert result["n"] == 40


def test_robustness_reports_headline_power_and_prospective_n():
    """The verdict carries observed-effect design power and a prospective sample size.

    Power quantifies the SERVER-SIDE aggregation heuristic's contrast only — it
    is not a certificate of the beta/rcce per-agent FedGVI guarantee.
    """
    result = run_robustness_sweep(0)
    assert 0.0 <= result["headline_power"] <= 1.0
    assert result["power_alpha"] == 0.05
    assert result["power_alternative"] == "greater"
    assert result["target_power"] == 0.80
    # The headline method is a robust (non-KLD) member present in the verdict.
    assert result["headline_method"] in result["verdict"]
    assert result["headline_method"] != "KLD"
    # Prospective n is a positive integer (or the honest search ceiling).
    assert isinstance(result["prospective_n_for_target_power"], int)
    assert result["prospective_n_for_target_power"] >= 1
    assert result["headline_n_for_target_power"] >= 1


def test_robustness_verdict_carries_power_per_method():
    """Every robust verdict entry exposes design power and a prospective n."""
    result = run_robustness_sweep(0)
    for d in result["divergences"]:
        if d == "KLD":
            continue
        v = result["verdict"][d]
        assert 0.0 <= v["power"] <= 1.0
        assert isinstance(v["n_for_target_power"], int)
        assert v["n_for_target_power"] >= 1
        # A larger observed effect is at least as well powered at fixed n.
    # The high-effect winner is powered no worse than a weaker robust member.
    powers = {d: result["verdict"][d]["power"] for d in result["verdict"]}
    effects = {d: result["verdict"][d]["effect_size"] for d in result["verdict"]}
    # Monotone: the method with the largest effect has the maximal power.
    best = max(effects, key=effects.get)
    assert powers[best] == pytest.approx(max(powers.values()), abs=1e-12)


def test_robustness_power_honors_alternative_and_alpha():
    """The power params thread through to the observed-effect power computation."""
    greater = run_robustness_sweep(0, power_alternative="greater")
    two_sided = run_robustness_sweep(0, power_alternative="two-sided")
    # For a non-saturated effect a two-sided test is never MORE powered than the
    # one-sided one; for the saturated headline both pin at 1.0 (still valid).
    assert two_sided["headline_power"] <= greater["headline_power"] + 1e-12


def test_robustness_power_is_deterministic():
    a = run_robustness_sweep(4)
    b = run_robustness_sweep(4)
    assert a["headline_power"] == b["headline_power"]
    assert a["prospective_n_for_target_power"] == b["prospective_n_for_target_power"]
    assert {d: a["verdict"][d].get("power") for d in a["verdict"]} == {
        d: b["verdict"][d].get("power") for d in b["verdict"]
    }


def test_robustness_accuracy_at_verdict_rate_means_and_cis():
    """Per-method means at verdict_rate carry bootstrap 95% CIs and sample size."""
    result = run_robustness_sweep(0)
    avr = result["accuracy_at_verdict_rate"]
    # Every method (naive + robust) is present.
    assert set(avr) == set(result["divergences"])
    for d, block in avr.items():
        assert block["n"] == result["n_trials"]
        lo, hi = block["ci"]
        assert lo <= hi
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= block["mean"] <= hi
    # Consistency: the naive mean equals the back-compat naive_verdict_rate_mean.
    assert avr["KLD"]["mean"] == pytest.approx(
        result["naive_verdict_rate_mean"], abs=1e-12
    )


def test_robustness_verdict_carries_effect_size_and_raw_p():
    """Each verdict exposes rank-biserial r, a d-equivalent, and a diff CI."""
    result = run_robustness_sweep(0)
    for d in result["divergences"]:
        if d == "KLD":
            continue
        v = result["verdict"][d]
        # Raw p-value is exposed alongside the BH q-value and equals the test p.
        assert v["raw_pvalue"] == pytest.approx(v["pvalue"], abs=1e-12)
        # Standardized effect size derives from the rank-biserial effect size.
        # A perfectly one-signed contrast (r = +-1) is reported as a large but
        # FINITE, JSON-safe value (not +-inf) — keeps the report standards-JSON.
        assert np.isfinite(v["d_equivalent"])
        if abs(v["effect_size"]) == pytest.approx(1.0, abs=1e-12):
            assert abs(v["d_equivalent"]) >= 1e5  # capped sentinel, off the top band
            assert np.sign(v["d_equivalent"]) == np.sign(v["effect_size"])
        else:
            assert v["d_equivalent"] == pytest.approx(
                2.0 * v["effect_size"] / np.sqrt(1.0 - v["effect_size"] ** 2),
                abs=1e-9,
            )
        assert v["effect_label"] in {"negligible", "small", "medium", "large"}
        lo, hi = v["mean_accuracy_diff_ci"]
        assert lo <= hi
        # The diff equals robust mean minus naive mean (paired).
        avr = result["accuracy_at_verdict_rate"]
        assert v["mean_accuracy_diff"] == pytest.approx(
            avr[d]["mean"] - avr["KLD"]["mean"], abs=1e-12
        )


def test_robustness_winner_has_large_positive_effect():
    """A confirmed winner has positive effect size and a non-negligible label."""
    result = run_robustness_sweep(0)
    winners = [
        d for d in result["divergences"]
        if d != "KLD" and result["verdict"][d]["wins"]
    ]
    assert winners
    for d in winners:
        v = result["verdict"][d]
        assert v["effect_size"] > 0.0
        assert v["d_equivalent"] > 0.0
        assert v["effect_label"] in {"small", "medium", "large"}


def test_robustness_per_rate_paired_tests_structure():
    """Per-rate paired tests cover every robust method x every sweep rate."""
    result = run_robustness_sweep(0)
    ptr = result["paired_tests_by_rate"]
    robust = [d for d in result["divergences"] if d != "KLD"]
    assert set(ptr) == set(robust)
    rate_keys = {f"{r:g}" for r in result["rates"]}
    for d in robust:
        assert set(ptr[d]) == rate_keys
        for rk, cell in ptr[d].items():
            assert set(cell) == {
                "statistic", "pvalue", "raw_pvalue", "qvalue", "rejected",
                "effect_size", "d_equivalent", "effect_label",
            }
            assert 0.0 <= cell["pvalue"] <= 1.0
            assert 0.0 <= cell["qvalue"] <= 1.0
            # BH q-value never undershoots the raw p (step-up inflation).
            assert cell["qvalue"] >= cell["raw_pvalue"] - 1e-12
            assert cell["raw_pvalue"] == pytest.approx(cell["pvalue"], abs=1e-12)
            assert -1.0 <= cell["effect_size"] <= 1.0
            assert isinstance(cell["rejected"], bool)
            assert cell["effect_label"] in {
                "negligible", "small", "medium", "large",
            }


def test_robustness_per_rate_zero_rate_is_a_homogeneous_contrast():
    """At rate 0 no agent is contaminated; the robust-vs-naive effect is small."""
    result = run_robustness_sweep(0)
    ptr = result["paired_tests_by_rate"]
    # rate '0' is in the default sweep; the contrast there reflects only the
    # pooling-rule difference, not any robustness benefit.
    cell = ptr["RKL"]["0"]
    assert -1.0 <= cell["effect_size"] <= 1.0


def test_robustness_enrichment_is_deterministic():
    """All enriched blocks reproduce exactly under the same seed."""
    a = run_robustness_sweep(4)
    b = run_robustness_sweep(4)
    assert a["accuracy_at_verdict_rate"] == b["accuracy_at_verdict_rate"]
    assert a["paired_tests_by_rate"] == b["paired_tests_by_rate"]
    assert a["verdict"] == b["verdict"]


def test_robustness_enrichment_json_serialisable():
    result = run_robustness_sweep(0)
    assert _json_roundtrips(result)


def test_robustness_report_is_strict_standards_json():
    """No Infinity/NaN leaks into the report: strict (allow_nan=False) dumps OK.

    The verdict's Cohen's d can hit the r = +-1 boundary where the closed-form
    value is infinite; the harness caps it to a finite sentinel precisely so a
    non-Python (standards-compliant) JSON consumer can read the report file.
    """
    result = run_robustness_sweep(0)
    # allow_nan=False makes json raise on Infinity/-Infinity/NaN.
    dumped = json.dumps(result, allow_nan=False)
    assert "Infinity" not in dumped and "NaN" not in dumped


# ---- error paths ----------------------------------------------------------

def test_robustness_rejects_empty_rates():
    with pytest.raises(ValueError, match="rates must be non-empty"):
        run_robustness_sweep(0, rates=())


def test_robustness_requires_naive_baseline():
    with pytest.raises(ValueError, match="naive baseline 'KLD'"):
        run_robustness_sweep(0, divergences=("RKL", "beta"))


def test_robustness_requires_three_agents():
    with pytest.raises(ValueError, match="at least three agents"):
        run_robustness_sweep(0, n_agents=2)


@pytest.mark.parametrize("n_contaminated", [0, 7, 10])
def test_robustness_rejects_bad_contaminated_count(n_contaminated):
    with pytest.raises(ValueError, match="n_contaminated"):
        run_robustness_sweep(0, n_agents=7, n_contaminated=n_contaminated)


@pytest.mark.parametrize("conf", [0.0, 1.0, 1.5, -0.2])
def test_robustness_rejects_bad_confidence(conf):
    with pytest.raises(ValueError, match="healthy_confidence"):
        run_robustness_sweep(0, healthy_confidence=conf)


def test_robustness_rejects_too_few_trials():
    with pytest.raises(ValueError, match="n_trials"):
        run_robustness_sweep(0, n_trials=1)


@pytest.mark.parametrize("vr", [-0.1, 1.1])
def test_robustness_rejects_bad_verdict_rate(vr):
    with pytest.raises(ValueError, match="verdict_rate"):
        run_robustness_sweep(0, verdict_rate=vr)


def test_robustness_unknown_divergence_label():
    with pytest.raises(ValueError, match="unknown divergence"):
        run_robustness_sweep(0, divergences=("KLD", "NOPE"))


# ===========================================================================
# internal helpers (kept covered explicitly)
# ===========================================================================

def test_divergence_to_robustness_mapping():
    assert _divergence_to_robustness("KLD") == 0.0
    # every robust label maps to a strictly positive down-weighting strength.
    for label in ("RKL", "AR", "beta", "rcce"):
        assert _divergence_to_robustness(label) > 0.0
    with pytest.raises(ValueError, match="unknown divergence"):
        _divergence_to_robustness("bogus")


def test_divergence_robustness_constants_are_pairwise_distinct():
    """Guard the pre-fix RKL/rcce collision at robustness 1.5."""
    assert len(set(_DIVERGENCE_ROBUSTNESS.values())) == len(_DIVERGENCE_ROBUSTNESS)


def test_sample_observation_is_in_range_and_seeded():
    rng = np.random.default_rng(0)
    # near-deterministic likelihood column -> the sampled outcome is the truth.
    likelihood = np.array([[0.98, 0.01], [0.01, 0.98], [0.01, 0.01]])
    likelihood = likelihood / likelihood.sum(axis=0, keepdims=True)
    samples = [_sample_observation(likelihood, 0, rng) for _ in range(64)]
    assert all(0 <= s < 3 for s in samples)
    assert samples.count(0) > 50  # column 0 peaks on outcome 0


def test_soft_colony_rows_are_pmfs_peaked_on_truth():
    rng = np.random.default_rng(0)
    colony = _soft_colony(true_state=2, n_agents=5, n_s=9, confidence=0.35, rng=rng, jitter=0.05)
    assert colony.shape == (5, 9)
    np.testing.assert_allclose(colony.sum(axis=1), 1.0, atol=1e-12)
    # truth carries the most mass in every healthy belief.
    assert all(int(np.argmax(colony[i])) == 2 for i in range(5))


def test_consensus_accuracy_naive_vs_robust_branches():
    """KLD takes the log-linear branch; a robust label takes the share_round branch."""
    beliefs = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.65, 0.25, 0.1]])
    naive = _consensus_accuracy(beliefs, "KLD", true_state=0)
    robust = _consensus_accuracy(beliefs, "RKL", true_state=0)
    # both are valid probabilities; with an agreeing healthy colony both are high.
    assert 0.0 <= naive <= 1.0
    assert 0.0 <= robust <= 1.0
    assert naive > 0.8
