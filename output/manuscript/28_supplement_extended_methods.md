# Supplement: extended methods for scoped generalization {#sec:supp-extended}

This supplement documents three method extensions that broaden the toolkit
without changing the categorical federated claims of the main text. Each answers
a "does it generalize?" question raised by a specific main-text result and each
connects back to it: the additional contamination models, gallery, and onset
sweep stress-test the robustness verdict of [@sec:results-robustness] beyond its
single confident-wrong mechanism; the Gaussian divergence bridge points toward
the continuous-state direction of [@sec:future-continuous]; and greedy
multi-hypothesis reduction extends the emergence study of
[@sec:results-emergence] from one pruned state to a family. Each is tested and
isolated; none participates in the headline robustness verdict, so the main-text
claims stand or fall without them.

## Continuous-state divergence bridge for Gaussian beliefs {#sec:supp-gaussian}

Every robustness claim in the main text is discrete-categorical, matching
Friston's worked example. To show the divergence family carries over to the
Gaussian beliefs a continuous-state active-inference extension would use,
`divergences.py` adds closed forms for 1-D Gaussians: the Kullback–Leibler
divergence

$$
\mathrm{KL}\!\big(\mathcal N(\mu_q,\sigma_q^2)\,\|\,\mathcal N(\mu_p,\sigma_p^2)\big)
= \tfrac12\!\left[\tfrac{\sigma_q^2}{\sigma_p^2}
+ \tfrac{(\mu_p-\mu_q)^2}{\sigma_p^2} - 1
+ \log\tfrac{\sigma_p^2}{\sigma_q^2}\right],
$$ {#eq:gaussian-kl}

and the $\alpha$-Rényi divergence with interpolated variance
$\sigma_\alpha^2 = \alpha\sigma_p^2 + (1-\alpha)\sigma_q^2$,

$$
D_\alpha\!\big(\mathcal N_q\,\|\,\mathcal N_p\big)
= \frac{\alpha(\mu_q-\mu_p)^2}{2\sigma_\alpha^2}
- \frac{1}{2(\alpha-1)}\log\frac{\sigma_\alpha^2}{\sigma_q^{2(1-\alpha)}\sigma_p^{2\alpha}}.
$$ {#eq:gaussian-renyi}

As in the categorical case ([@eq:renyi-limit] and Lemma \ref{lem:renyi-kl-limit}),
[@eq:gaussian-renyi] recovers [@eq:gaussian-kl] in the
$\alpha\to1$ limit, and the closed form is returned exactly inside a small band
around one. These functions are **out of scope** for the federated experiments —
they are wired into no aggregation rule or sweep — and exist purely as the
explicitly-scoped bridge toward continuous active inference.

## Additional contamination models for the robustness surface {#sec:supp-contamination}

`contamination.py` extends the confident-wrong, label-noise, and uniform models
with two mechanisms that probe different attack surfaces, both honoring the
identity anchor (rate zero returns the belief unchanged):

- **Byzantine targeted** — a *multiplicative* log-odds tilt toward an
  adversary-chosen state, $s' \propto s\cdot\exp(\text{rate}\cdot\text{tilt}\cdot e_{\text{target}})$.
  Unlike the additive convex mixes, the corruption compounds with the belief's
  own shape, so the non-target states keep their relative order — the canonical
  targeted poisoning of a product-of-experts pool.
- **Drift** — a *slowly-moving* bias that grows linearly across communication
  rounds via a phase $\phi = \text{round}/(\text{rounds}-1)$, so the first round
  is clean and the bias creeps in. This is the stealthy sentinel whose
  miscalibration only becomes confident late, defeating any one-shot screen.

Both are exercised in the contamination tests; the headline sweep continues to
use the confident-wrong model so the verdict is comparable to the main text.

### Contamination gallery by corruption mechanism {#sec:supp-gallery}

To check the robust-beats-naive result is not an artifact of the single
confident-wrong mechanism — and not of a single lucky seed —
`experiments.run_contamination_gallery` re-runs the paired comparison
($n = 24$ trials across 64 independent seeds,
contamination strength 0.60) under every model. For each mechanism
it selects one robust method by pooled mean consensus accuracy **for
descriptive gallery display only**, then reports that displayed member's
robust-minus-naive difference, 95% seed bootstrap interval, and
*win fraction* — the fraction of seeds in which the displayed member beats
naive. This is not the selection-free inferential surface; the all-method
review grid below serves that role.

| Mechanism | Class | Naive | Best robust | Mean diff | 95% CI | Win frac. | Reliable |
|---|---|---|---|---|---|---|---|
| byzantine | directional | 0.6306 | 0.6599 (beta) | 0.0293 | [0.0124, 0.0472] | 0.62 | No |
| confident wrong | directional | 0.9836 | 0.9878 (AR) | 0.0043 | [0.0040, 0.0046] | 1.00 | Yes |
| drift | directional | 0.9836 | 0.9878 (AR) | 0.0043 | [0.0040, 0.0046] | 1.00 | Yes |
| label noise | entropy | 0.9982 | 0.9931 (AR) | -0.0051 | [-0.0052, -0.0051] | 0.00 | No |
| uniform | entropy | 0.9985 | 0.9947 (AR) | -0.0038 | [-0.0038, -0.0038] | 0.00 | No |

: Seed-aggregated descriptive display of robust-vs-naive accuracy under each
contamination mechanism ($n = 24$ trials ×
64 seeds at strength 0.60). `Reliable` is a
display flag for the pooled-selected member: it is `Yes` only when that
member beats naive in at least 0.95 of seeds *and*
its displayed difference CI excludes zero. This is an across-seed screen, not
one lucky seed, a p-value, or selection-free post-selection inference.
{#tbl:contamination-gallery}

The contamination summary [@tbl:contamination-gallery] is a descriptive
sensitivity screen, deliberately narrower than "robust always wins." The pooled
display member has a positive all-seed and interval pattern under
confident wrong, drift — the *additive* directional attacks. The full set
of directional mechanisms
is confident wrong, byzantine, drift; the **byzantine** attack is directional too,
but its *multiplicative* log-odds tilt escalates faster: at this strength it
sits near a veto cliff where the naive pool is already badly degraded and the
displayed robust advantage does not hold across seeds (its win fraction is well
below the 0.95 display bar and its difference CI
straddles zero), so we do *not* claim it. The **entropy** attacks
(label noise, uniform)
raise entropy or inject noise without a fixed wrong target, so the
product-of-experts is not pulled off the truth and there is nothing to beat — the
robust members stay close rather than winning (naive undegraded by entropy
attacks: Yes). [@fig:contamination-gallery] draws
all mechanisms with their win fractions. This is the honest scope of this
configured gallery: its displayed members separate from naive under the
declared *sustained additive* directional contamination, stay close under the
declared entropy attacks, and lose the displayed advantage against the tested
multiplicative adversary near the veto regime. These finite cells do not
establish the same ordering for every attack strength or world, and they do not
turn a pooled display selection into selection-free inference.

![Seed-aggregated mean consensus accuracy. Source relation: original project contamination diagnostic; estimand: true-state accuracy fraction by attack mechanism; uncertainty: the bars show 95% seed-level bootstrap confidence intervals for the pooled-selected display member, while the adjacent table reports its conditional paired difference interval. $q(\text{true state})$ for the naive log-linear pool versus the robust method selected once by pooled mean under each contamination mechanism ($n = 24$ trials × 64 seeds at strength 0.60). The x-axis is the contamination mechanism; the y-axis is mean consensus accuracy. Each group has two bars: naive log-linear pooling and the pooled display member for that mechanism. The robust bar is drawn in full color only where the across-seed win fraction (annotated above the group) clears the 0.95 display bar — confident wrong, drift; the byzantine mechanism and entropy attacks are muted because they do not clear that descriptive screen. The in-figure summary gives the display-flag count across mechanisms and reminds readers that the labels are win fractions, not p-values. The bars are means over 64 seeds; the selected method is shown above each bar. This is a descriptive pooled-selection graphic, not selection-free post-selection inference; the all-method review grid supplies the latter surface.](../output/figures/contamination_gallery.png){#fig:contamination-gallery width=85%}

### Robustness onset by corruption mechanism {#sec:supp-onset}

The gallery fixes one contamination strength; `experiments.run_robustness_onset`
maps the *rate dependence* ($n = 24$ trials × 64
seeds per rate). For each directional mechanism it reports the **descriptive
onset rate** — the smallest rate at which the pooled-selected display member's
win fraction reaches 0.95 — and that member's versus naive
accuracy at the worst (highest) swept rate. These display summaries are not
selection-free inference; the all-method review grid is the inferential
surface:

| Mechanism | Onset rate | Naive @ worst | Robust @ worst | Robust method @ worst |
|---|---|---|---|---|
| byzantine | 0.4 | 0.0165 | 0.0000 | beta |
| confident wrong | 0.6 | 0.6676 | 0.7772 | beta |
| drift | 0.6 | 0.6676 | 0.7772 | beta |

: Per-mechanism descriptive onset and worst-rate accuracy
($n = 24$ trials × 64 seeds). The onset rate is
where the pooled display method reaches the displayed win-fraction rule
(win fraction ≥ 0.95); it is not a per-seed selection,
selection-free inferential result, or universal crossover claim.
{#tbl:robustness-onset}

The mechanism-specific onset thresholds are collected in [@tbl:robustness-onset].

The rate dependence sharpens the gallery's snapshot, and [@fig:robustness-onset]
draws it. The additive confident-wrong and drift attacks degrade the naive pool
gradually; past their onset rate the robust member stays above it through to the
worst rate. The multiplicative byzantine attack is qualitatively different: it
opens an *early* robustness window — robust overtakes at a lower onset rate — but
then escalates to the veto cliff where naive and robust both collapse, so its
worst-rate accuracy is near zero for both. This is the rate-resolved form of the
honest verdict: robustness is sustained against additive directional
contamination and only transient against a multiplicative one.

![Naive (dashed) versus the pooled display method. Source relation: original project robustness-onset diagnostic; estimand: mean consensus accuracy fraction by attack rate; uncertainty: shaded 95% seed-level bootstrap confidence intervals conditional on the pooled-selected display member. Mean consensus accuracy (solid, robust method selected once by pooled mean across seeds at each rate; dashed, naive) as the contamination rate rises, one panel per directional mechanism ($n = 24$ trials × 64 seeds per rate). The x-axis is the contamination rate; the y-axis is mean consensus accuracy. The dotted vertical line marks the descriptive onset rate (pooled robust win fraction ≥ 0.95), and each panel's inset reports that onset plus the final pooled robust-minus-naive gap at the largest swept rate. Confident-wrong and drift show a sustained displayed contrast past onset; byzantine shows a transient display window before both aggregators lose consensus accuracy at the highest corruption rates. The plotted values are seed-aggregated means with shaded bootstrap intervals; the companion table carries the displayed onset, worst-rate values, and selected method. This pooled-selection display is not selection-free post-selection inference; the all-method review grid supplies that inferential surface.](../output/figures/robustness_onset.png){#fig:robustness-onset width=95%}

### Conditional world and attack-geometry grid {#sec:supp-conditional-world}

The finite MAJ-1 characterization is now extended across
40 preregistered world/scenario cells: two hidden-state
locations, two observability levels, five attack mechanisms, and two adversarial
weight settings. The independent unit is the seeded world/scenario row; each
cell averages 24 nested trials over 64
seeds before the matched contrast is formed. The primary estimand is naive
true-state error minus robust true-state error, so a positive value means the
robust heuristic assigns more true-state mass in that finite cell. The
robustness-zero control is pass, and the report remains
explicitly labelled `conditional_finite_grid`. The resulting conditional
surface is shown in [@fig:conditional-world].

![Conditional-world robustness grid. Source relation: original project finite-grid generalization of the MAJ-1 characterization; estimand: naive true-state error minus robust true-state error; uncertainty: each heatmap cell is a seed-level mean with a 95% seed bootstrap interval in the source report, while the right panel shows finite-grid min/max span rather than a confidence interval; independent unit: seeded world/scenario row. The x-axis is the declared hidden-state and observability cell; the y-axis is the attack mechanism. The left panel varies hidden state and observability across columns and attack mechanism across rows; the right panel summarizes the finite-grid span by attack. Positive values favour robust true-state mass, negative values favour naive pooling, and zero is the recovery/no-contrast reference. This is conditional evidence over a declared finite grid, not a theorem, breakdown bound, or universal attack result.](../output/figures/conditional_world.png){#fig:conditional-world width=95%}

### Source-bound robustness review grid {#sec:supp-review-grid}

The red-team review adds a bounded, selection-free stress surface that joins the
existing conditional-world cells to the existing directional rate profiles. It
uses 160 deterministic seed replicates and
24 trials nested within each seed/cell. The finite attack
union is clean, confident wrong, permutation, byzantine, drift, label noise, uniform; the rate-resolved directional mechanisms are
confident wrong, byzantine, drift, with entropy controls uniform, label noise.
The registered rate set is $\{0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$. The independent unit is
seed within a declared scenario or rate cell, and the nesting rule is:
n_trials nested within each seed/cell; no trial is promoted to an independent world. This is a source-bound simulation review, not
an external-data benchmark or a claim that cells sharing design structure are
independent.

The payload is explicitly selection-free source payload; every configured non-KLD method is reported at every directional rate and no winner is used for inference and the statistics
surface is selection-free. It reports seed-level contrasts, paired
Wilcoxon/rank-biserial results, percentile bootstrap intervals, MCSE, an
observed-effect MDE, and BH-adjusted rate families. BH ownership is
BH is applied within each attack-mechanism × method rate family; cells sharing design structure are not treated as independent families. The precision plan targets maximum MCSE
0.0100 and observed maximum MCSE
0.0066 across 96. Every
configured robust method is retained as a rate-profile curve and inferential
member; no method is selected per seed, rate, or pooled mean for this review
grid. The all-method display does not close the open calibration or server-theory
questions.

The rendered diagnostic is shown in [@fig:robustness-review-grid].

![Expanded source-bound robustness review grid. Source relation: original project finite simulation review diagnostic composed from the existing conditional-world and onset mechanisms; estimand: seed-level robust-minus-naive true-state probability-mass contrast; uncertainty: the right-panel shaded bands are percentile bootstrap intervals over independent seeds for every configured robust method, while the second line in each left-panel cell is half the finite-grid min--max span, not a confidence interval; replication unit: configured seed, with trials nested within seed and cell. The x-axis is the declared adversarial-weight setting in the left panel and the contamination rate in the right panel; the y-axis is the seed-level robust-minus-naive true-state mass contrast in both panels. The left panel summarizes conditional attack cells, and the right panel shows every configured directional method's signed rate profile over the registered rates. Positive values favour robust true-state mass, negative values favour naive pooling, and zero is the recovery/no-contrast reference. No method or curve is selected by pooled mean for this grid; all displayed intervals and comparisons are selection-free. This visualization is conditional finite-grid evidence and does not claim a universal winner, breakdown bound, causal effect, or independence across shared design cells.](../output/figures/robustness_review_grid.png){#fig:robustness-review-grid width=95%}

### Proper scores and calibration controls {#sec:supp-belief-quality}

Argmax accuracy does not distinguish a cautious posterior from an overconfident
one. The scoring extension therefore pre-registers the paired seed-level
categorical log-score difference between naive and robust consensus beliefs as
its primary belief-quality estimand. The Brier score and equal-width expected
calibration error are secondary diagnostics; higher log score and lower Brier/ECE
are better. Agents and trials remain nested within the seed.

The report also includes three negative controls: an oracle, a uniform belief,
and a confidently wrong belief. Their expected score ordering is checked before
any method contrast is interpreted: oracle > uniform > confidently wrong under
the clipped log score. The control ordering gate is
`pass`, and the confidently-wrong-versus-uniform control is
`pass`, using 64 independent
seeds and 24 nested trials per seed. The diagnostic is shown
in [@fig:belief-quality].

![Proper scoring and calibration controls. Source relation: original project belief-quality diagnostic; estimand: categorical log score as the primary measure, with Brier score and reliability error as secondary diagnostics; uncertainty: 95% seed bootstrap confidence intervals for control log scores; independent unit: seed, with trials nested within seed. The x-axis is the control type in the left panel and mean confidence in the right panel; the y-axis is mean categorical log score in the left panel and empirical accuracy in the right panel. The left panel compares oracle, uniform, and confidently-wrong controls on the higher-is-better log-score scale. The right panel plots mean confidence against empirical accuracy for the same controls and a perfect-calibration diagonal. The controls are negative checks on score implementation, not evidence for decision optimality, distribution-shift calibration, or robustness outside the tested finite world.](../output/figures/belief_quality.png){#fig:belief-quality width=90%}

## Greedy multi-hypothesis model reduction beyond the main BMR study {#sec:supp-greedy-bmr}

The single-step reduction of [@sec:method-learning] scores one candidate reduced
prior. `bayesian_model_reduction.py` adds `greedy_reduce`, which performs
structure learning over a *family* of redundant states: starting from the full
prior, it scores pruning each not-yet-pruned state against the current reduced
prior, accepts the single prune with the largest positive free-energy gain, and
repeats until no remaining prune improves model evidence. Every accepted step has
a strictly positive incremental $\Delta F$, so the cumulative evidence is
monotone-increasing and the search recovers the sparse generative model the data
support — a state with genuine evidence yields $\Delta F < 0$ when pruned and is
kept. This is the multi-state analogue of the emergence result of
[@sec:results-emergence], and is verified directly in the model-reduction tests.
