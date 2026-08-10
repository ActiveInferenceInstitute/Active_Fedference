"""Tests for paired comparison and BH-FDR deflation — Algorithm Gate I (no mocks).

Every number is a real seeded computation with an explicit expectation:

* ISC-28: a clearly-shifted paired sample yields Wilcoxon ``p < 0.05`` and a
  rank-biserial effect size of exactly ``+1`` (every difference one-signed).
* ISC-29: the canonical Benjamini-Hochberg (1995) 15-hypothesis family yields
  the textbook rejection of exactly the four smallest p-values.

All branches of every public function — including the error paths — are
exercised so the project's 90% line+branch coverage gate holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.statistics import (
    bh_fdr,
    bootstrap_ci,
    cohens_d_from_rank_biserial,
    interpret_effect_size,
    minimum_detectable_effect,
    paired_test,
    per_group_test,
    power_analysis,
    rank_stability,
    sample_size_for_power,
    summary_statistics,
)

# ---- paired_test : ISC-28 -------------------------------------------------

def test_paired_test_detects_a_clearly_shifted_pair():
    """ISC-28: a uniformly positive shift is significant at p < 0.05."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 40)
    b = a + 0.8  # deterministic positive shift on every pair
    out = paired_test(a, b)
    assert out["pvalue"] < 0.05
    # every difference b - a = +0.8 > 0 -> rank-biserial pinned at exactly +1
    assert out["effect_size"] == pytest.approx(1.0, abs=1e-12)
    assert out["statistic"] == pytest.approx(0.0, abs=1e-12)


def test_paired_test_effect_size_sign_flips_with_direction():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 30)
    b = a - 0.5  # uniformly negative shift
    out = paired_test(a, b)
    assert out["effect_size"] == pytest.approx(-1.0, abs=1e-12)
    assert out["pvalue"] < 0.05


def test_paired_test_no_effect_for_symmetric_offsets():
    # signed ranks cancel exactly: half the pairs +1, half -1, equal magnitudes
    a = np.array([0.0, 0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, -1.0, -2.0])
    out = paired_test(a, b)
    assert out["effect_size"] == pytest.approx(0.0, abs=1e-12)
    assert out["pvalue"] > 0.05


def test_paired_test_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        paired_test([1.0, 2.0, 3.0], [1.0, 2.0])


def test_paired_test_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        paired_test([], [])


def test_paired_test_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="finite"):
        paired_test([1.0, np.nan], [2.0, 3.0])


def test_paired_test_rejects_all_zero_differences():
    with pytest.raises(ValueError, match="undefined"):
        paired_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])


# ---- bh_fdr : ISC-29 ------------------------------------------------------

def test_bh_fdr_controls_the_family_textbook_rejections():
    """ISC-29: canonical Benjamini-Hochberg (1995) example -> 4 rejections."""
    pvalues = np.array([
        0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
        0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
    ])
    out = bh_fdr(pvalues, alpha=0.05)
    expected = np.zeros(15, dtype=bool)
    expected[:4] = True  # exactly the four smallest survive the step-up rule
    assert np.array_equal(out["rejected"], expected)
    assert int(out["rejected"].sum()) == 4
    # q-values are monotone in the p-value ordering and pinned at the head
    assert out["qvalues"][0] == pytest.approx(0.0015, abs=1e-6)
    assert out["qvalues"][3] == pytest.approx(0.035625, abs=1e-6)
    assert np.all(np.diff(out["qvalues"]) >= -1e-12)  # already sorted by p here
    assert np.all((out["qvalues"] >= 0.0) & (out["qvalues"] <= 1.0))
    # Raw p-values are exposed alongside the q-values, in input order, unchanged.
    assert np.array_equal(out["pvalues"], pvalues)


def test_bh_fdr_preserves_input_order():
    # shuffle the canonical family; rejections must follow the values, not order
    pvalues = np.array([0.5719, 0.0001, 0.0344, 0.0004, 0.0019, 0.0095])
    out = bh_fdr(pvalues, alpha=0.05)
    # the four smallest here are 0.0001, 0.0004, 0.0019, 0.0095 at idx 1,3,4,5
    assert out["rejected"][1]
    assert out["rejected"][3]
    assert not out["rejected"][0]  # 0.5719 never rejected


def test_bh_fdr_rejects_nothing_when_all_large():
    out = bh_fdr([0.6, 0.7, 0.8, 0.9], alpha=0.05)
    assert not out["rejected"].any()
    assert np.all(out["qvalues"] <= 1.0)


def test_bh_fdr_rejects_empty_family():
    with pytest.raises(ValueError, match="non-empty"):
        bh_fdr([])


def test_bh_fdr_rejects_out_of_range_pvalues():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bh_fdr([0.1, 1.2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bh_fdr([-0.01, 0.2])


def test_bh_fdr_rejects_nonfinite_pvalues():
    with pytest.raises(ValueError, match="finite"):
        bh_fdr([0.1, np.nan])


def test_bh_fdr_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        bh_fdr([0.01, 0.02], alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        bh_fdr([0.01, 0.02], alpha=1.5)


# ---- rank_stability -------------------------------------------------------

def test_rank_stability_is_one_for_identical_rankings():
    mat = np.array([[1.0, 2.0, 3.0, 4.0],
                    [10.0, 20.0, 30.0, 40.0],  # same ranking, different scale
                    [0.1, 0.2, 0.3, 0.4]])
    assert rank_stability(mat) == pytest.approx(1.0, abs=1e-12)


def test_rank_stability_matches_hand_computed_mean_spearman():
    mat = np.array([[3.0, 1.0, 2.0, 4.0],
                    [3.0, 2.0, 1.0, 4.0],
                    [4.0, 1.0, 2.0, 3.0]])
    # three pairwise Spearman rhos average to 11/15
    assert rank_stability(mat) == pytest.approx(0.7333333333333333, abs=1e-9)


def test_rank_stability_negative_for_reversed_ranking():
    mat = np.array([[1.0, 2.0, 3.0, 4.0],
                    [4.0, 3.0, 2.0, 1.0]])
    assert rank_stability(mat) == pytest.approx(-1.0, abs=1e-12)


def test_rank_stability_treats_constant_fold_as_zero():
    # second fold is constant -> Spearman is nan -> contributes 0.0
    mat = np.array([[1.0, 2.0, 3.0],
                    [5.0, 5.0, 5.0]])
    assert rank_stability(mat) == pytest.approx(0.0, abs=1e-12)


def test_rank_stability_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        rank_stability(np.array([1.0, 2.0, 3.0]))


def test_rank_stability_rejects_single_fold():
    with pytest.raises(ValueError, match="two folds"):
        rank_stability(np.array([[1.0, 2.0, 3.0]]))


def test_rank_stability_rejects_single_agent():
    with pytest.raises(ValueError, match="two agents"):
        rank_stability(np.array([[1.0], [2.0]]))


def test_rank_stability_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="finite"):
        rank_stability(np.array([[1.0, 2.0], [3.0, np.inf]]))


# ---- bootstrap_ci ---------------------------------------------------------

def test_bootstrap_ci_brackets_the_mean_and_is_seed_reproducible():
    rng = np.random.default_rng(0)
    samples = rng.normal(5.0, 2.0, 50)
    lo, hi = bootstrap_ci(samples, rng=np.random.default_rng(0))
    # Pinned percentile-bootstrap interval for this exact seeded resample.
    assert lo == pytest.approx(4.736037650746902, abs=1e-9)
    assert hi == pytest.approx(5.746173826535622, abs=1e-9)
    assert lo < float(np.mean(samples)) < hi
    # Same generator seed -> identical interval (determinism contract).
    lo2, hi2 = bootstrap_ci(samples, rng=np.random.default_rng(0))
    assert (lo, hi) == (lo2, hi2)


def test_bootstrap_ci_narrows_with_larger_alpha():
    rng = np.random.default_rng(0)
    samples = rng.normal(5.0, 2.0, 50)
    lo95, hi95 = bootstrap_ci(samples, alpha=0.05, rng=np.random.default_rng(0))
    lo90, hi90 = bootstrap_ci(samples, alpha=0.10, rng=np.random.default_rng(0))
    # A 90% interval is strictly inside the 95% interval.
    assert (hi90 - lo90) < (hi95 - lo95)
    assert lo95 <= lo90 and hi90 <= hi95


def test_bootstrap_ci_degenerate_for_constant_sample():
    lo, hi = bootstrap_ci([3.0, 3.0, 3.0], rng=np.random.default_rng(1))
    assert lo == hi == pytest.approx(3.0, abs=1e-12)


def test_bootstrap_ci_default_rng_runs():
    # No rng supplied -> a fresh default generator; interval is still ordered.
    lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert lo <= hi


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci([], rng=np.random.default_rng(0))


def test_bootstrap_ci_rejects_nonfinite_samples():
    with pytest.raises(ValueError, match="finite"):
        bootstrap_ci([1.0, np.inf], rng=np.random.default_rng(0))


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_bootstrap_ci_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci([1.0, 2.0, 3.0], alpha=alpha, rng=np.random.default_rng(0))


def test_bootstrap_ci_rejects_bad_n_boot():
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci([1.0, 2.0, 3.0], n_boot=0, rng=np.random.default_rng(0))


@pytest.mark.parametrize("n_boot", [2.5, True, "200"])
def test_bootstrap_ci_rejects_non_integral_n_boot(n_boot):
    """Counts must not be silently truncated or accept boolean values."""
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap_ci([1.0, 2.0, 3.0], n_boot=n_boot, rng=np.random.default_rng(0))


# ---- per_group_test -------------------------------------------------------

def test_per_group_test_runs_each_pair_in_order():
    groups = [
        ([0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]),  # all positive -> r = +1
        ([0.0, 0.0, 0.0], [-1.0, -2.0, -3.0]),          # all negative -> r = -1
    ]
    out = per_group_test(groups)
    assert len(out) == 2
    assert out[0]["effect_size"] == pytest.approx(1.0, abs=1e-12)
    assert out[1]["effect_size"] == pytest.approx(-1.0, abs=1e-12)
    # Each entry is a full paired_test result dict.
    assert set(out[0]) == {"statistic", "pvalue", "effect_size"}


def test_per_group_test_pvalues_feed_bh_fdr():
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, 30)
    groups = [(base, base + 0.9), (base, base + 0.01 * rng.normal(0, 1, 30))]
    results = per_group_test(groups)
    pvals = [r["pvalue"] for r in results]
    fdr = bh_fdr(np.array(pvals))
    # The strong positive shift is rejected; the near-null group is not.
    assert fdr["rejected"][0]
    assert not fdr["rejected"][1]


def test_per_group_test_rejects_empty():
    with pytest.raises(ValueError, match="at least one group"):
        per_group_test([])


def test_per_group_test_rejects_malformed_pair():
    with pytest.raises(ValueError, match=r"\(a, b\) pair"):
        per_group_test([([1.0, 2.0], [1.0, 3.0], [1.0])])


def test_per_group_test_propagates_pair_validation():
    # An all-zero-difference pair is undefined for Wilcoxon -> error propagates.
    with pytest.raises(ValueError, match="undefined"):
        per_group_test([([1.0, 2.0], [1.0, 2.0])])


# ---- cohens_d_from_rank_biserial ------------------------------------------

def test_cohens_d_zero_at_zero_r():
    assert cohens_d_from_rank_biserial(0.0) == pytest.approx(0.0, abs=1e-12)


def test_cohens_d_matches_closed_form():
    # d = 2r / sqrt(1 - r^2)
    assert cohens_d_from_rank_biserial(0.5) == pytest.approx(1.1547005383792515, abs=1e-12)
    assert cohens_d_from_rank_biserial(-0.5) == pytest.approx(-1.1547005383792515, abs=1e-12)


def test_cohens_d_diverges_at_unit_boundary():
    assert cohens_d_from_rank_biserial(1.0) == float("inf")
    assert cohens_d_from_rank_biserial(-1.0) == float("-inf")


def test_cohens_d_is_monotone_in_r():
    rs = [-0.9, -0.4, 0.0, 0.3, 0.7]
    ds = [cohens_d_from_rank_biserial(r) for r in rs]
    assert all(ds[i] < ds[i + 1] for i in range(len(ds) - 1))


@pytest.mark.parametrize("bad", [1.01, -1.5, 2.0])
def test_cohens_d_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        cohens_d_from_rank_biserial(bad)


# ---- interpret_effect_size ------------------------------------------------

@pytest.mark.parametrize(
    "d,label",
    [
        (0.0, "negligible"),
        (0.19, "negligible"),
        (0.2, "small"),
        (0.49, "small"),
        (0.5, "medium"),
        (0.79, "medium"),
        (0.8, "large"),
        (2.0, "large"),
        (float("inf"), "large"),
    ],
)
def test_interpret_effect_size_thresholds(d, label):
    assert interpret_effect_size(d) == label


def test_interpret_effect_size_ignores_sign():
    # Magnitude only: negative d labels match their positive counterparts.
    assert interpret_effect_size(-0.6) == interpret_effect_size(0.6) == "medium"
    assert interpret_effect_size(-1.5) == "large"
    assert interpret_effect_size(-0.1) == "negligible"


# ---- simulation precision summaries --------------------------------------

def test_minimum_detectable_effect_scales_with_sample_size():
    small = minimum_detectable_effect(1.0, 25)
    large = minimum_detectable_effect(1.0, 100)
    assert small > large > 0.0


def test_minimum_detectable_effect_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="standard_deviation"):
        minimum_detectable_effect(-1.0, 10)
    with pytest.raises(ValueError, match="positive integer"):
        minimum_detectable_effect(1.0, 0)
    with pytest.raises(ValueError, match="alpha"):
        minimum_detectable_effect(1.0, 10, alpha=0.0)
    with pytest.raises(ValueError, match="target_power"):
        minimum_detectable_effect(1.0, 10, target_power=1.0)


@pytest.mark.parametrize("n", [2.5, True, "10"])
def test_minimum_detectable_effect_rejects_non_integral_n(n):
    with pytest.raises(ValueError, match="positive integer"):
        minimum_detectable_effect(1.0, n)


def test_summary_statistics_reports_seed_precision_and_is_reproducible():
    values = np.array([0.2, 0.4, 0.6, 0.8])
    first = summary_statistics(
        values, n_boot=300, rng=np.random.default_rng(5)
    )
    second = summary_statistics(
        values, n_boot=300, rng=np.random.default_rng(5)
    )
    assert first == second
    assert first["median"] == pytest.approx(0.5)
    assert first["mcse"] == pytest.approx(np.std(values, ddof=1) / np.sqrt(4))
    assert first["mde"] > 0.0
    assert first["ci_lo"] <= first["mean"] <= first["ci_hi"]
    assert first["uncertainty_available"] is True
    assert first["ci_method"] == "percentile_bootstrap"


def test_summary_statistics_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        summary_statistics([0.1, float("nan")])


@pytest.mark.parametrize("n_boot", [2.5, True, "200"])
def test_summary_statistics_rejects_non_integral_n_boot(n_boot):
    with pytest.raises(ValueError, match="positive integer"):
        summary_statistics([0.1, 0.2], n_boot=n_boot)


def test_summary_statistics_marks_single_replicate_uncertainty_unavailable():
    result = summary_statistics([0.42], n_boot=100, rng=np.random.default_rng(0))
    assert result["uncertainty_available"] is False
    assert result["mcse"] == 0.0
    assert result["mde"] == 0.0


# ---- power_analysis -------------------------------------------------------

def test_power_analysis_high_power_for_large_effect_and_n():
    """A large effect with a large sample is very well powered (>0.99)."""
    out = power_analysis(0.8, 40)
    assert out["power"] > 0.99
    assert 0.0 <= out["power"] <= 1.0
    assert out["effect_size"] == 0.8
    assert out["n"] == 40
    assert out["alpha"] == 0.05
    assert out["alternative"] == "greater"
    # The prospective n for 80% power at d=0.8 is small and positive.
    assert 1 <= out["n_for_80_power"] <= 20


def test_power_analysis_low_power_for_small_effect_and_n():
    """A small effect at a small sample is underpowered (<0.5)."""
    out = power_analysis(0.2, 20)
    assert out["power"] < 0.5
    # Reaching 80% power for a small effect needs a much larger sample.
    assert out["n_for_80_power"] > 100


def test_power_analysis_rises_monotonically_with_n():
    """Power increases as the sample grows at a fixed effect size."""
    powers = [power_analysis(0.5, n)["power"] for n in (10, 20, 40, 80)]
    assert all(powers[i] < powers[i + 1] for i in range(len(powers) - 1))


def test_power_analysis_two_sided_is_below_one_sided():
    """A two-sided test spends alpha in both tails -> less power than directional."""
    one = power_analysis(0.5, 30, alternative="greater")["power"]
    two = power_analysis(0.5, 30, alternative="two-sided")["power"]
    assert two < one


def test_power_analysis_less_alternative_powered_for_negative_effect():
    """A 'less' test is well powered when the effect is genuinely negative."""
    out = power_analysis(-0.8, 40, alternative="less")
    assert out["power"] > 0.99


def test_power_analysis_is_deterministic():
    assert power_analysis(0.5, 25) == power_analysis(0.5, 25)


@pytest.mark.parametrize("n", [0, -1])
def test_power_analysis_rejects_bad_n(n):
    with pytest.raises(ValueError, match="positive integer"):
        power_analysis(0.5, n)


@pytest.mark.parametrize("n", [2.5, True, "40"])
def test_power_analysis_rejects_non_integral_n(n):
    with pytest.raises(ValueError, match="positive integer"):
        power_analysis(0.5, n)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_power_analysis_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        power_analysis(0.5, 20, alpha=alpha)


def test_power_analysis_rejects_bad_alternative():
    with pytest.raises(ValueError, match="alternative"):
        power_analysis(0.5, 20, alternative="bogus")


def test_power_analysis_rejects_nonfinite_effect():
    with pytest.raises(ValueError, match="finite"):
        power_analysis(float("nan"), 20)


# ---- sample_size_for_power ------------------------------------------------

def test_sample_size_for_power_matches_power_analysis_floor():
    """The returned n is the FIRST n whose power clears the target."""
    n = sample_size_for_power(0.80, 0.5)
    # n itself clears 0.80; n-1 (if >=1) does not.
    assert power_analysis(0.5, n)["power"] >= 0.80
    if n > 1:
        assert power_analysis(0.5, n - 1)["power"] < 0.80


def test_sample_size_for_power_smaller_for_larger_effect():
    """A larger effect needs fewer pairs to reach the same power."""
    n_small = sample_size_for_power(0.80, 0.3)
    n_large = sample_size_for_power(0.80, 0.8)
    assert n_large < n_small


def test_sample_size_for_power_respects_directional_effect_sign():
    """A wrong-signed one-sided effect cannot yield a finite prospective n."""
    from fedference.statistics import _MAX_SAMPLE_SIZE

    assert sample_size_for_power(0.80, 0.5, alternative="greater") < _MAX_SAMPLE_SIZE
    assert sample_size_for_power(0.80, -0.5, alternative="greater") == _MAX_SAMPLE_SIZE
    assert sample_size_for_power(0.80, -0.5, alternative="less") < _MAX_SAMPLE_SIZE
    assert sample_size_for_power(0.80, 0.5, alternative="less") == _MAX_SAMPLE_SIZE
    assert sample_size_for_power(0.80, 0.5, alternative="two-sided") == (
        sample_size_for_power(0.80, -0.5, alternative="two-sided")
    )


def test_sample_size_for_power_zero_effect_hits_ceiling():
    """A zero effect can never beat alpha power -> the search ceiling is returned."""
    from fedference.statistics import _MAX_SAMPLE_SIZE

    assert sample_size_for_power(0.80, 0.0) == _MAX_SAMPLE_SIZE


def test_sample_size_for_power_saturated_effect_returns_feasible_floor():
    """A saturated effect must yield the smallest FEASIBLE Wilcoxon n, not 1.

    With n pairs the most extreme one-sided p is 2**-n, so n=5 is the first
    n that can reject at alpha=0.05 (2**-5 = 0.03125 <= 0.05 < 0.0625 = 2**-4).
    The pre-fix behavior returned an infeasible n=1 for the sweep's saturated
    rank-biserial effect (proof-of-detection for the feasibility floor).
    """
    from fedference.statistics import _wilcoxon_min_feasible_n

    assert _wilcoxon_min_feasible_n(0.05, "greater") == 5
    assert _wilcoxon_min_feasible_n(0.05, "two-sided") == 6
    n = sample_size_for_power(0.80, 1e6)
    assert n == 5
    assert sample_size_for_power(0.80, 1e6, alternative="two-sided") == 6


def test_sample_size_for_power_two_sided_needs_more_than_one_sided():
    """Spending alpha in both tails raises the required sample size."""
    one = sample_size_for_power(0.80, 0.5, alternative="greater")
    two = sample_size_for_power(0.80, 0.5, alternative="two-sided")
    assert two >= one


@pytest.mark.parametrize("tp", [0.0, 1.0, -0.1, 1.2])
def test_sample_size_for_power_rejects_bad_target(tp):
    with pytest.raises(ValueError, match="target_power"):
        sample_size_for_power(tp, 0.5)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_sample_size_for_power_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        sample_size_for_power(0.80, 0.5, alpha=alpha)


def test_sample_size_for_power_rejects_bad_alternative():
    with pytest.raises(ValueError, match="alternative"):
        sample_size_for_power(0.80, 0.5, alternative="bogus")


def test_sample_size_for_power_rejects_nonfinite_effect():
    with pytest.raises(ValueError, match="finite"):
        sample_size_for_power(0.80, float("inf"))
