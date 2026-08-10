"""Paired comparison and multiple-testing deflation — Algorithm Gate I.

When the experiment harness pits robust federated belief-sharing against the
naive project log-linear pool (the documented categorical Eq. 7
specialization), the headline claim is a
*paired* one: across matched scenarios (same seed, same contamination), does the
robust aggregator lower per-agent surprise / raise accuracy? Two methods are
needed to make that claim survive scrutiny.

* :func:`paired_test` — the matched-pairs Wilcoxon signed-rank test
  (``scipy.stats.wilcoxon``) plus the matched-pairs **rank-biserial** effect
  size ``r = (T+ - T-) / (T+ + T-)``. It does not assume Gaussian paired
  differences, but it is still a signed-rank test of paired differences rather
  than an assumption-free equality-of-means test.

* :func:`bh_fdr` — the Benjamini-Hochberg (1995) step-up procedure that controls
  the false-discovery rate of the *family* of comparisons (one per
  metric/horizon), returning the rejection mask and the monotone BH q-values.
  This is the multiple-testing deflation: with many simultaneous robust-vs-naive
  contrasts, an uncorrected ``p < 0.05`` sweep would manufacture false wins.

* :func:`rank_stability` — mean Spearman rank correlation of agent-influence
  rankings across cross-validation folds, the stability diagnostic for the
  robust weighting (a robust aggregator that re-ranks agents arbitrarily fold to
  fold is not trustworthy).

* :func:`power_analysis` / :func:`sample_size_for_power` — observed-effect
  planning diagnostics for the paired Wilcoxon at a given effect size and sample
  size. They are not confirmatory evidence and do not turn nested simulation
  trials into independent scientific replicates.

Pure ``numpy`` / ``scipy.stats`` only; no active-inference imports — this module
sits at the analysis tier above the FedGVI core.
"""

from __future__ import annotations

import warnings
from numbers import Integral

import numpy as np
from scipy.stats import ConstantInputWarning, norm, rankdata, spearmanr, wilcoxon

ArrayF = np.ndarray

# Conventional |d|-equivalent thresholds for the secondary display label.
_EFFECT_THRESHOLDS = ((0.2, "small"), (0.5, "medium"), (0.8, "large"))

# Pitman asymptotic relative efficiency of the Wilcoxon signed-rank test against
# the one-sample/paired t-test under a normal shift alternative: 3/pi ~= 0.9549.
# The Wilcoxon needs the effective sample size n * ARE_WILCOXON to match the
# t-test's noncentrality, so its power is computed on that deflated n. This is
# the standard, deterministic large-sample correction (no simulation needed).
_ARE_WILCOXON: float = 3.0 / np.pi


def _positive_integer(value: object, name: str) -> int:
    """Validate a count without silently truncating a non-integral value.

    Counts arrive from both typed Python calls and JSON/YAML configuration.  A
    call such as ``n_boot=2.5`` must not silently become two bootstrap draws,
    and ``True`` must not become one draw merely because ``bool`` subclasses
    ``int``.  Keeping this check local to the statistics boundary makes the
    numerical routines fail closed before allocating arrays or reporting a
    misleading sample size.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def paired_test(a, b) -> dict:
    """Wilcoxon signed-rank test on the matched pairs ``(a_i, b_i)``.

    Returns a dict with:

    * ``statistic`` — the Wilcoxon test statistic (min of the signed-rank sums).
    * ``pvalue`` — two-sided p-value.
    * ``effect_size`` — matched-pairs rank-biserial correlation
      ``r = (T+ - T-) / (T+ + T-)`` in ``[-1, 1]``; ``+1`` when every
      difference ``b_i - a_i`` is positive, ``-1`` when every one is negative,
      ``0.0`` when the signed ranks cancel or all pairs are tied.

    Raises ``ValueError`` if the inputs differ in length, are empty, or contain
    only zero differences (Wilcoxon is undefined — there is nothing to rank).
    """
    a_ = np.asarray(a, dtype=np.float64).ravel()
    b_ = np.asarray(b, dtype=np.float64).ravel()
    if a_.size == 0 or b_.size == 0:
        raise ValueError("paired_test requires non-empty samples")
    if a_.size != b_.size:
        raise ValueError("paired_test requires equal-length paired samples")
    if not np.all(np.isfinite(a_)) or not np.all(np.isfinite(b_)):
        raise ValueError("paired_test requires finite samples")

    diff = b_ - a_
    nonzero = diff[diff != 0.0]
    if nonzero.size == 0:
        raise ValueError("all paired differences are zero; Wilcoxon is undefined")

    res = wilcoxon(a_, b_)

    abs_ranks = rankdata(np.abs(nonzero))
    t_plus = float(abs_ranks[nonzero > 0].sum())
    t_minus = float(abs_ranks[nonzero < 0].sum())
    total = t_plus + t_minus
    effect_size = (t_plus - t_minus) / total if total > 0 else 0.0

    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "effect_size": float(effect_size),
    }


def bh_fdr(pvalues, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg (1995) step-up FDR control over a family of p-values.

    Returns a dict with:

    * ``rejected`` — boolean ``ndarray`` (same order as the input) marking the
      hypotheses rejected at family-wise FDR ``alpha``.
    * ``qvalues`` — the monotone BH-adjusted p-values (q-values) clipped to
      ``[0, 1]``, in the input order.
    * ``pvalues`` — the raw (uncorrected) input p-values as a ``float64``
      ``ndarray`` in the input order, exposed alongside the q-values so callers
      can report both the uncorrected and FDR-adjusted significance.

    The rejection rule finds the largest rank ``k`` with ``p_(k) <= (k/m) alpha``
    and rejects every hypothesis with that p-value or smaller. ``alpha`` must lie
    in ``(0, 1]``; p-values must lie in ``[0, 1]``.
    """
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("bh_fdr requires a non-empty p-value family")
    if not np.all(np.isfinite(p)):
        raise ValueError("p-values must be finite")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p-values must lie in [0, 1]")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")

    m = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    ranks = np.arange(1, m + 1)

    # Rejection: largest k with p_(k) <= (k/m) * alpha.
    below = ranked <= (ranks / m) * alpha
    rejected = np.zeros(m, dtype=bool)
    if below.any():
        kmax = int(np.max(np.nonzero(below)[0]))
        rejected[order[: kmax + 1]] = True

    # BH q-values: enforce monotonicity from the largest p-value downward.
    raw_q = ranked * m / ranks
    monotone = np.minimum.accumulate(raw_q[::-1])[::-1]
    qvalues = np.empty(m, dtype=np.float64)
    qvalues[order] = np.clip(monotone, 0.0, 1.0)

    return {"rejected": rejected, "qvalues": qvalues, "pvalues": p}


def rank_stability(score_matrix) -> float:
    """Mean Spearman rank correlation of agent rankings across folds.

    ``score_matrix`` is ``(n_folds, n_agents)``: row ``f`` holds the per-agent
    influence scores estimated on fold ``f``. Returns the mean Spearman
    correlation over all ``C(n_folds, 2)`` fold pairs — ``1.0`` when every fold
    induces the identical agent ranking, lower as rankings disagree.

    Requires at least two folds and at least two agents. A pair whose
    correlation is undefined (a constant row -> ``nan``) contributes ``0.0``.
    """
    mat = np.asarray(score_matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("score_matrix must be 2-D (n_folds, n_agents)")
    n_folds, n_agents = mat.shape
    if n_folds < 2:
        raise ValueError("rank_stability requires at least two folds")
    if n_agents < 2:
        raise ValueError("rank_stability requires at least two agents")
    if not np.all(np.isfinite(mat)):
        raise ValueError("score_matrix must contain finite values")

    correlations: list[float] = []
    with warnings.catch_warnings():
        # a constant fold makes Spearman undefined (nan); we map that to 0.0.
        warnings.simplefilter("ignore", ConstantInputWarning)
        for i in range(n_folds):
            for j in range(i + 1, n_folds):
                rho = spearmanr(mat[i], mat[j]).statistic
                correlations.append(0.0 if np.isnan(rho) else float(rho))
    return float(np.mean(correlations))


def _z_power(effect_size: float, n: float, alpha: float, alternative: str) -> float:
    """Normal-approximation power for a paired Wilcoxon at a given ``effect_size``.

    Uses the noncentral-normal approximation to the matched-pairs t-test with
    the Wilcoxon's Pitman ARE (``3/pi``) deflating the effective sample size.
    The noncentrality is ``delta = effect_size * sqrt(n * ARE)``; the rejection
    threshold is the standard-normal critical value at ``alpha`` (one-sided for
    ``'greater'`` / ``'less'``, two-sided for ``'two-sided'``). Returns a power
    in ``[0, 1]``.
    """
    ncp = float(effect_size) * np.sqrt(float(n) * _ARE_WILCOXON)
    if alternative == "two-sided":
        z = norm.ppf(1.0 - alpha / 2.0)
        # Power against a two-sided rejection region (both tails).
        return float(norm.cdf(abs(ncp) - z) + norm.cdf(-abs(ncp) - z))
    if alternative == "greater":
        z = norm.ppf(1.0 - alpha)
        return float(norm.cdf(ncp - z))
    # 'less': power accrues in the lower tail.
    z = norm.ppf(1.0 - alpha)
    return float(norm.cdf(-ncp - z))


def power_analysis(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    alternative: str = "greater",
) -> dict:
    """Statistical power of the paired Wilcoxon signed-rank test.

    Computes the observed-effect design power to detect a standardized paired shift of
    magnitude ``effect_size`` (here a rank-biserial-derived d-equivalent) with
    ``n`` matched pairs at significance ``alpha``, plus the prospective sample
    size ``n_for_80_power`` needed to reach 80% power at the same effect and
    alpha. Deterministic: a closed-form noncentral-normal approximation to the
    paired t-test, deflated by the Wilcoxon's Pitman asymptotic relative
    efficiency (``3/pi ~= 0.955``) so the reported power is conservative under
    the normal-shift approximation. This is a planning calculation conditional
    on the observed effect size, not independent evidence for the verdict.

    Returns a dict with:

    * ``power`` — power in ``[0, 1]`` at the supplied ``n``.
    * ``n_for_80_power`` — the smallest integer ``n`` reaching power ``>= 0.80``
      at this ``effect_size`` / ``alpha`` (via :func:`sample_size_for_power`);
      capped at a large finite ceiling when the effect is vanishingly small.
    * ``effect_size``, ``n``, ``alpha``, ``alternative`` — the inputs, echoed.

    ``n`` must be a positive integer, ``alpha`` in ``(0, 1)``, and
    ``alternative`` one of ``'greater'``, ``'less'``, ``'two-sided'``.
    """
    n_ = _positive_integer(n, "n")
    if not np.isfinite(float(effect_size)):
        raise ValueError("effect_size must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative must be 'greater', 'less' or 'two-sided'")
    power = _z_power(float(effect_size), n_, alpha, alternative)
    n80 = sample_size_for_power(
        target_power=0.80,
        effect_size=float(effect_size),
        alpha=alpha,
        alternative=alternative,
    )
    return {
        "power": float(np.clip(power, 0.0, 1.0)),
        "n_for_80_power": int(n80),
        "effect_size": float(effect_size),
        "n": n_,
        "alpha": float(alpha),
        "alternative": str(alternative),
    }


#: Sample-size search ceiling: a vanishing effect can require an unbounded n, so
#: the prospective-n helper caps its monotone search here and reports the cap.
_MAX_SAMPLE_SIZE: int = 100_000


def _wilcoxon_min_feasible_n(alpha: float, alternative: str) -> int:
    """Smallest ``n`` at which a paired signed-rank test can reject at ``alpha``.

    With ``n`` pairs the most extreme attainable p-value is ``2**-n`` one-sided
    (``2**(1-n)`` two-sided), so no sample below this floor can ever reject —
    regardless of effect size. The normal-approximation inversion in
    :func:`sample_size_for_power` is invalid below it (a saturated effect
    otherwise "recommends" an infeasible n of 1).
    """
    import math

    if alternative == "two-sided":
        return max(1, math.ceil(1.0 - math.log2(alpha)))
    return max(1, math.ceil(-math.log2(alpha)))


def sample_size_for_power(
    target_power: float,
    effect_size: float,
    alpha: float = 0.05,
    alternative: str = "greater",
) -> int:
    """Smallest ``n`` whose paired-Wilcoxon power reaches ``target_power``.

    Inverts :func:`power_analysis`'s power curve by a monotone integer search:
    power rises monotonically with ``n`` (the noncentrality grows as
    ``sqrt(n)``), so the first ``n`` clearing ``target_power`` is the answer.
    For a two-sided test the sign is irrelevant and ``|effect_size|`` is used.
    For a directional test the sign must agree with the alternative: a positive
    effect is expected for ``greater`` and a negative effect for ``less``. A
    wrong-signed effect has no finite prospective sample size under this
    directional design and returns the search ceiling
    :data:`_MAX_SAMPLE_SIZE`. Returns that same ceiling when an aligned effect
    is too small to reach the target within the ceiling.

    The result is floored at :func:`_wilcoxon_min_feasible_n`: a signed-rank
    test with fewer pairs cannot reject at ``alpha`` at any effect size, so a
    saturated effect yields the smallest *feasible* n, never an infeasible 1.

    ``target_power`` must lie in ``(0, 1)``; ``alpha`` in ``(0, 1)``;
    ``alternative`` as in :func:`power_analysis`. A zero ``effect_size`` can
    never exceed ``alpha`` power and returns the ceiling.
    """
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must lie in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative must be 'greater', 'less' or 'two-sided'")
    if not np.isfinite(float(effect_size)):
        raise ValueError("effect_size must be finite")

    effect = float(effect_size)
    if alternative == "greater" and effect < 0.0:
        return _MAX_SAMPLE_SIZE
    if alternative == "less" and effect > 0.0:
        return _MAX_SAMPLE_SIZE
    eff = abs(effect)
    if eff == 0.0:
        return _MAX_SAMPLE_SIZE

    # Closed-form seed from the noncentral-normal inversion, then a short
    # integer climb to absorb the continuity / ARE rounding. For 'greater' the
    # required noncentrality is z_{1-alpha} + z_{power}; deflate by the ARE.
    z_alpha = norm.ppf(1.0 - (alpha / 2.0 if alternative == "two-sided" else alpha))
    z_power = norm.ppf(target_power)
    n_seed = ((z_alpha + z_power) / eff) ** 2 / _ARE_WILCOXON
    n_floor = _wilcoxon_min_feasible_n(alpha, alternative)
    n = max(n_floor, int(np.floor(n_seed)))
    while n <= _MAX_SAMPLE_SIZE:
        if _z_power(eff, n, alpha, "greater" if alternative != "two-sided" else "two-sided") >= target_power:
            return int(n)
        n += 1
    return _MAX_SAMPLE_SIZE


def bootstrap_ci(
    samples,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``samples``.

    Resamples ``samples`` with replacement ``n_boot`` times, takes the mean of
    each resample, and returns the central ``(1 - alpha)`` percentile interval
    ``(lo, hi)`` of those bootstrap means. With the default ``alpha = 0.05`` this
    is the 95% CI (the 2.5th and 97.5th percentiles).

    Determinism is the caller's responsibility: pass an explicit
    ``np.random.default_rng(seed)`` to get a reproducible interval. When ``rng``
    is ``None`` a fresh default generator is used (non-reproducible) — the
    experiment harness always threads its seeded generator in.

    A single distinct sample value collapses every resample to that value, so
    ``lo == hi`` equals that value (a degenerate but valid interval). ``alpha``
    must lie in ``(0, 1)`` and ``n_boot`` must be a positive integer; the input
    must be non-empty.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("bootstrap_ci requires non-empty samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("bootstrap_ci requires finite samples")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    n_boot_ = _positive_integer(n_boot, "n_boot")
    if rng is None:
        rng = np.random.default_rng()

    idx = rng.integers(0, x.size, size=(n_boot_, x.size))
    boot_means = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def per_group_test(groups) -> list[dict]:
    """Run :func:`paired_test` independently on each ``(a, b)`` group.

    ``groups`` is an iterable of ``(a, b)`` paired-sample tuples — one per
    subgroup / contamination rate / horizon. Returns a list (input order) of the
    :func:`paired_test` result dicts (``statistic``, ``pvalue``, ``effect_size``),
    so a caller can build a family of contrasts and then deflate the collected
    p-values with :func:`bh_fdr`. Raises ``ValueError`` on an empty ``groups`` or
    a malformed (non-pair) entry; the per-pair validation of :func:`paired_test`
    still applies to every group.
    """
    materialised = list(groups)
    if not materialised:
        raise ValueError("per_group_test requires at least one group")
    results: list[dict] = []
    for entry in materialised:
        pair = tuple(entry)
        if len(pair) != 2:
            raise ValueError("each group must be an (a, b) pair of samples")
        a, b = pair
        results.append(paired_test(a, b))
    return results


def d_equivalent_from_rank_biserial(r: float) -> float:
    """Return the monotone ``d``-equivalent display transform of rank-biserial ``r``.

    This is ``2 r / sqrt(1 - r^2)``. It is a secondary, rank-biserial-derived
    display transform, not a raw paired Cohen's ``d`` estimator. The primary
    effect is the rank-biserial value returned by :func:`paired_test`, together
    with the paired mean difference and its interval. ``r`` must lie in
    ``[-1, 1]``; the boundary values return signed infinity and are handled by
    report writers with an explicit saturation marker.
    """
    r_ = float(r)
    if not -1.0 <= r_ <= 1.0:
        raise ValueError("rank-biserial r must lie in [-1, 1]")
    if r_ == 1.0:
        return float("inf")
    if r_ == -1.0:
        return float("-inf")
    return float(2.0 * r_ / np.sqrt(1.0 - r_ * r_))


def cohens_d_from_rank_biserial(r: float) -> float:
    """Deprecated compatibility alias for :func:`d_equivalent_from_rank_biserial`.

    The old name is retained for additive v0.x compatibility, but the returned
    value must not be described as an independently estimated Cohen's ``d``.
    """
    warnings.warn(
        "cohens_d_from_rank_biserial is deprecated; use "
        "d_equivalent_from_rank_biserial",
        DeprecationWarning,
        stacklevel=2,
    )
    return d_equivalent_from_rank_biserial(r)


def interpret_effect_size(d: float) -> str:
    """Label a secondary ``d``-equivalent display value by magnitude.

    Returns ``'negligible'`` for ``|d| < 0.2``, ``'small'`` for
    ``0.2 <= |d| < 0.5``, ``'medium'`` for ``0.5 <= |d| < 0.8`` and ``'large'``
    for ``|d| >= 0.8`` (including infinite ``d``). Only the magnitude matters —
    the sign of ``d`` does not change the label.
    """
    mag = abs(float(d))
    label = "negligible"
    for threshold, name in _EFFECT_THRESHOLDS:
        if mag >= threshold:
            label = name
    return label


def minimum_detectable_effect(
    standard_deviation: float,
    n: int,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> float:
    """Approximate a two-sided normal-mean minimum detectable effect.

    This is a planning diagnostic based on the seed-level standard deviation;
    it is not a replacement for the paired Wilcoxon test or evidence of a
    real-world effect. The approximation is
    ``(z_(1-alpha/2) + z_(target_power)) * sd / sqrt(n)``.
    """
    sd = float(standard_deviation)
    if not np.isfinite(sd) or sd < 0.0:
        raise ValueError("standard_deviation must be finite and non-negative")
    n_ = _positive_integer(n, "n")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must lie in (0, 1)")
    critical = norm.ppf(1.0 - alpha / 2.0) + norm.ppf(target_power)
    return float(critical * sd / np.sqrt(n_))


def summary_statistics(
    values,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
    target_power: float = 0.80,
) -> dict:
    """Summarize independent simulation replicates with MCSE and MDE.

    ``values`` must contain one scalar per independent seed or other declared
    Monte Carlo unit. Nested clients, episodes, or trajectories must be
    reduced before this function is called to avoid pseudoreplication.

    A one-value input is accepted for deterministic reports; its ``mcse`` and
    ``mde`` are ``0.0`` for compatibility and should be treated as unavailable,
    not as evidence of zero uncertainty. The returned
    ``uncertainty_available`` flag makes that distinction machine-readable;
    callers must still ensure that the values represent the declared
    independent Monte Carlo unit.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("summary_statistics requires non-empty values")
    if not np.all(np.isfinite(x)):
        raise ValueError("summary_statistics requires finite values")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    n_boot_ = _positive_integer(n_boot, "n_boot")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must lie in (0, 1)")
    n = int(x.size)
    std = float(x.std(ddof=1)) if n >= 2 else 0.0
    mcse = float(std / np.sqrt(n)) if n >= 2 else 0.0
    ci_lo, ci_hi = bootstrap_ci(x, alpha=alpha, n_boot=n_boot_, rng=rng)
    mde = (
        minimum_detectable_effect(std, n, alpha=alpha, target_power=target_power)
        if n >= 2
        else 0.0
    )
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": std,
        "mcse": mcse,
        "mde": float(mde),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "min": float(x.min()),
        "max": float(x.max()),
        "n": n,
        "uncertainty_available": bool(n >= 2),
        "ci_method": "percentile_bootstrap",
    }


def multiseed_summary(
    values,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng_seed: int = 0,
) -> dict:
    """Summary statistics for a vector of per-seed scalar values.

    Designed for the pattern: run an experiment for K seeds, collect the
    target metric for each seed into an array, then call this function to get
    the mean, median, standard deviation, MCSE, MDE, percentile-bootstrap CI,
    min, max, and n in a single dict — ready for manuscript-token emission and
    cross-study comparison plots.

    Args:
        values: 1-D array-like of per-seed scalar values (float or int).
        alpha: Two-tailed confidence level for the bootstrap CI
            (default 0.05 yields a 95% CI).
        n_boot: Number of bootstrap resamples (default 2000).
        rng_seed: Integer seed for the bootstrap generator, ensuring
            reproducible intervals across separate calls.

    Returns:
        Dict with the fields above, where ``mcse`` is the standard error of the
        mean and ``mde`` is the two-sided normal-approximation minimum detectable
        effect at 80% power. For one value, ``mcse`` and ``mde`` are ``0.0`` but
        are not inferentially available.

    Raises:
        ValueError: if ``values`` is empty, ``alpha`` is outside ``(0, 1)``,
            or ``n_boot`` is not a positive integer.
    """
    return summary_statistics(
        values,
        alpha=alpha,
        n_boot=n_boot,
        rng=np.random.default_rng(rng_seed),
    )
