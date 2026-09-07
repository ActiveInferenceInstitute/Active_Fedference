# Supplement: extended methods for scoped generalization {#sec:supp-extended}

This supplement documents three method extensions that broaden the toolkit
without changing the categorical federated claims of the main text. Each answers
a "does it generalize?" question raised by a specific main-text result and each
connects back to it.

The additional contamination models, gallery, and onset
sweep stress-test the robustness verdict of [@sec:results-robustness] beyond its
single confident-wrong mechanism; the Gaussian divergence bridge points toward
the continuous-state direction of [@sec:future-continuous]; and greedy
multi-hypothesis reduction extends the configured BMR sign control of
[@sec:results-emergence] from one pruned state to a family.

Each is tested and
isolated; none participates in the headline robustness verdict, so the main-text
claims stand or fall without them.

## Continuous-state divergence bridge for Gaussian beliefs {#sec:supp-gaussian}

Every robustness claim in the main text is discrete-categorical, matching
Friston's worked example. To show the divergence family carries over to the
Gaussian beliefs a continuous-state active-inference extension would use,
`divergences.py` adds closed forms for 1-D Gaussians: the Kullback–Leibler
divergence

$$
\begin{aligned}
&\mathrm{KL}\!\big(
  \mathcal N(\mu_q,\sigma_q^2)\,\|\,\mathcal N(\mu_p,\sigma_p^2)\big)\\
&\quad = \tfrac12\!\left[
\begin{aligned}
&\tfrac{\sigma_q^2}{\sigma_p^2}
 + \tfrac{(\mu_p-\mu_q)^2}{\sigma_p^2}\\
&\quad - 1 + \log\tfrac{\sigma_p^2}{\sigma_q^2}
\end{aligned}
\right].
\end{aligned}
$$ {#eq:gaussian-kl}

and the $\alpha$-Rényi divergence with interpolated variance
$\sigma_\alpha^2 = \alpha\sigma_p^2 + (1-\alpha)\sigma_q^2$,

$$
\begin{aligned}
&D_\alpha\!\big(\mathcal N_q\,\|\,\mathcal N_p\big)\\
&\quad = \frac{\alpha(\mu_q-\mu_p)^2}{2\sigma_\alpha^2}\\
&\qquad - \frac{1}{2(\alpha-1)}
\log\frac{\sigma_\alpha^2}
{\sigma_q^{2(1-\alpha)}\sigma_p^{2\alpha}}.
\end{aligned}
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

**Byzantine targeted.** A *multiplicative* log-odds tilt toward an
adversary-chosen state,
$s' \propto s\cdot\exp(\text{rate}\cdot\text{tilt}\cdot e_{\text{target}})$.
Unlike the additive convex mixes, the corruption compounds with the belief's
own shape, so the non-target states keep their relative order — the canonical
targeted poisoning of a product-of-experts pool.

**Drift.** A *slowly-moving* bias grows linearly across communication rounds
via a phase $\phi = \text{round}/(\text{rounds}-1)$, so the first round is clean
and the bias creeps in. This is the stealthy sentinel whose miscalibration only
becomes confident late, defeating any one-shot screen.

Both are exercised in the contamination tests; the headline sweep continues to
use the confident-wrong model so the verdict is comparable to the main text.

### Contamination gallery by corruption mechanism {#sec:supp-gallery}

To check the robust-beats-naive result is not an artifact of the single
confident-wrong mechanism — and not of a single lucky seed — the
`run_contamination_gallery` function in the `experiments` module re-runs the
paired comparison
($n = {{GALLERY_N_TRIALS}}$ trials across {{GALLERY_N_SEEDS}} independent seeds,
contamination strength {{GALLERY_RATE}}) under every model.

For each mechanism
it selects one robust method by pooled mean consensus accuracy **for
descriptive gallery display only**, then reports that displayed member's
robust-minus-naive difference, {{CI_PERCENT}}% seed bootstrap interval, and
*win fraction* — the fraction of seeds in which the displayed member beats
naive. This is not the selection-free inferential surface; the main-text
all-method review grid in [@sec:results-review-grid] serves that role.

| Mechanism | Class | Naive mean | Display robust mean (preset) |
|---|---|---|---|
{{GALLERY_OPERATING_POINT_TABLE_ROWS}}

: Operating-point projection of the seed-aggregated contamination gallery,
keyed by mechanism. Means reduce {{GALLERY_N_TRIALS}} trials within each of
{{GALLERY_N_SEEDS}} independent seeds at strength {{GALLERY_RATE}}; the preset
in parentheses is selected once by pooled mean for descriptive display. Joining
this projection to [@tbl:contamination-gallery-contrast] on mechanism
reconstructs every source row exactly and does not create selection-free
inference.
{#tbl:contamination-gallery}

| Mechanism | Mean robust − naive | {{CI_PERCENT}}% CI | Win fraction | Reliable |
|---|---|---|---|---|
{{GALLERY_CONTRAST_DISPLAY_TABLE_ROWS}}

: Contrast-and-display projection of the same gallery, keyed by mechanism.
Intervals resample the {{GALLERY_N_SEEDS}} seed-level differences after nested
trial reduction. `Reliable` is `Yes` only when the pooled-selected display
member beats naive in at least {{GALLERY_RELIABLE_WIN_FRACTION}} of seeds and
its displayed difference interval excludes zero. It is a descriptive screen,
not a p-value or selection-free post-selection inference; read it with
[@tbl:contamination-gallery] to recover the complete source row.
{#tbl:contamination-gallery-contrast}

The paired contamination summaries [@tbl:contamination-gallery;
@tbl:contamination-gallery-contrast] form one descriptive
sensitivity screen, deliberately narrower than "robust always wins." The pooled
display member has a positive all-seed and interval pattern under
{{GALLERY_RELIABLE_KINDS}} — the *additive* directional attacks.

The full set
of directional mechanisms
is {{GALLERY_DIRECTIONAL_KINDS}}; the **byzantine** attack is directional too,
but its *multiplicative* log-odds tilt escalates faster: at this strength it
sits near a veto cliff where the naive pool is already badly degraded and the
displayed robust advantage does not hold across seeds (its win fraction is well
below the {{GALLERY_RELIABLE_WIN_FRACTION}} display bar and its difference CI
straddles zero), so we do *not* claim it.

The **entropy** attacks
({{GALLERY_ENTROPY_KINDS}})
raise entropy or inject noise without a fixed wrong target, so the
product-of-experts is not pulled off the truth and there is nothing to beat — the
robust members stay close rather than winning (naive undegraded by entropy
attacks: {{GALLERY_ENTROPY_NAIVE_ROBUST}}).

[@fig:contamination-gallery] draws
all mechanisms with their win fractions. This is the honest scope of this
configured gallery: its displayed members separate from naive under the
declared *sustained additive* directional contamination, stay close under the
declared entropy attacks, and lose the displayed advantage against the tested
multiplicative adversary near the veto regime.

These finite cells do not
establish the same ordering for every attack strength or world, and they do not
turn a pooled display selection into selection-free inference.

![Seed-aggregated mean consensus accuracy. Source relation: original project contamination diagnostic; estimand: true-state accuracy fraction by attack mechanism; uncertainty: the bars show {{CI_PERCENT}}% seed-level bootstrap confidence intervals for the pooled-selected display member, while the adjacent table reports its conditional paired difference interval. $q(\text{true state})$ for the reference log-linear pool versus the server preset selected once by pooled mean under each contamination mechanism ($n = {{GALLERY_N_TRIALS}}$ trials × {{GALLERY_N_SEEDS}} seeds at strength {{GALLERY_RATE}}). The x-axis is the contamination mechanism; the y-axis is mean consensus accuracy. Each group has two bars: an open, directly labeled reference-log-pool bar and a hatched selected-server-preset bar whose direct annotation names the preset and its across-seed win fraction. The preset bar is drawn in full color only where that win fraction clears the {{GALLERY_RELIABLE_WIN_FRACTION}} display bar — {{GALLERY_RELIABLE_KINDS}}; the byzantine mechanism and entropy attacks are muted because they do not clear that descriptive screen. The in-figure summary gives the display-flag count across mechanisms and reminds readers that the labels are win fractions, not p-values. Bars are means over {{GALLERY_N_SEEDS}} independent configured seeds, with {{GALLERY_N_TRIALS}} matched trials nested within each seed. This is a descriptive pooled-selection graphic, not selection-free post-selection inference; the all-method review grid supplies the latter surface.](../output/figures/contamination_gallery.png){#fig:contamination-gallery width=85% data-slide-manifest="../output/figures/contamination_gallery.slides.json"}

### Robustness onset by corruption mechanism {#sec:supp-onset}

The gallery fixes one contamination strength; `experiments.run_robustness_onset`
maps the *rate dependence* ($n = {{ONSET_N_TRIALS}}$ trials × {{ONSET_N_SEEDS}}
seeds per rate).

For each directional mechanism it reports the **descriptive
onset rate** — the smallest rate at which the pooled-selected display member's
win fraction reaches {{ONSET_WIN_FRACTION}} — and that member's versus naive
accuracy at the worst (highest) swept rate. These display summaries are not
selection-free inference; the main-text all-method review grid in
[@sec:results-review-grid] is the inferential surface:

| Mechanism | Onset rate | Naive @ worst | Robust @ worst | Robust method @ worst |
|---|---|---|---|---|
{{ONSET_TABLE_ROWS}}

: Per-mechanism descriptive onset and worst-rate accuracy
($n = {{ONSET_N_TRIALS}}$ trials × {{ONSET_N_SEEDS}} seeds). The onset rate is
where the pooled display method reaches the displayed win-fraction rule
(win fraction ≥ {{ONSET_WIN_FRACTION}}); it is not a per-seed selection,
selection-free inferential result, or universal crossover claim.
{#tbl:robustness-onset}

The mechanism-specific onset thresholds are collected in [@tbl:robustness-onset].

The rate dependence sharpens the gallery's snapshot, and [@fig:robustness-onset]
draws it. The additive confident-wrong and drift attacks degrade the naive pool
gradually; past their onset rate the robust member stays above it through to the
worst rate.

The multiplicative byzantine attack is qualitatively different: it
opens an *early* robustness window — robust overtakes at a lower onset rate — but
then escalates to the veto cliff where naive and robust both collapse, so its
worst-rate accuracy is near zero for both.

This is the rate-resolved bounded interpretation: the displayed contrast is
sustained against the two additive directional mechanisms and only transient
against the multiplicative mechanism.

![Reference log pool and pooled-selected server preset across attack rates. Source relation: original project robustness-onset diagnostic; estimand: mean consensus accuracy fraction by attack rate; uncertainty: shaded {{CI_PERCENT}}% percentile-bootstrap intervals over configured seeds, conditional on the pooled-selected display member. Each panel reads left to right as contamination increases: filled circles with a solid line identify the reference log pool, while open squares with a dashed line identify the pooled display server preset selected once by pooled mean across seeds at each rate. The x-axis is contamination rate and the y-axis is mean consensus accuracy. Each of {{ONSET_N_SEEDS}} independent configured seeds contains {{ONSET_N_TRIALS}} nested trials per rate; trials are not promoted to independent replicates. A dark dotted vertical rule marks the descriptive onset rate (pooled preset win fraction ≥ {{ONSET_WIN_FRACTION}}), and the inset reports that onset plus the terminal preset-minus-reference gap. Confident-wrong and drift retain a displayed contrast after onset; Byzantine contamination produces a transient window before both methods lose consensus accuracy at the largest rates. The companion table gives the displayed onset, terminal values, and selected preset. This pooled-selection figure is descriptive, not selection-free post-selection inference; the all-method review grid supplies the selection-free comparative surface.](../output/figures/robustness_onset.png){#fig:robustness-onset width=95% data-slide-manifest="../output/figures/robustness_onset.slides.json"}

### Conditional world and attack-geometry grid {#sec:supp-conditional-world}

The finite MAJ-1 characterization is now extended across
{{CONDITIONAL_N_SCENARIOS}} preregistered world/scenario cells: two hidden-state
locations, two observability levels, five attack mechanisms, and two adversarial
weight settings.

The independent unit is the seeded world/scenario row; each
cell averages {{CONDITIONAL_N_TRIALS}} nested trials over {{CONDITIONAL_N_SEEDS}}
seeds before the matched contrast is formed. The primary estimand is naive
true-state error minus robust true-state error, so a positive value means the
robust heuristic assigns more true-state mass in that finite cell.

The
robustness-zero control is {{CONDITIONAL_ZERO_CONTROL}}, and the report remains
explicitly labelled `{{CONDITIONAL_CLAIM_STATUS}}`. The resulting conditional
surface is shown in [@fig:conditional-world].

![Conditional-world robustness grid. Source relation: original project finite-grid generalization of the MAJ-1 characterization; estimand: naive true-state error minus robust true-state error in probability-mass units. In Panel A, the x-axis indexes hidden-state and observability cells and the y-axis indexes attack mechanisms; every heatmap cell prints its signed seed-level mean, so sign and magnitude remain available without colour. The source report retains a {{CI_PERCENT}}% seed-bootstrap interval for each cell. In Panel B, the x-axis is the signed contrast and the y-axis again lists attacks. Its point is the mean over all declared finite-grid cells for that attack, and the asymmetric capped whiskers extend to the observed cell minimum and maximum. These capped min/max spans are finite-grid ranges, not confidence intervals and not symmetric mean-plus-or-minus errors. A dark-neutral dotted zero rule marks no method contrast. Positive values favour robust true-state mass; negative values favour naive pooling. The independent unit is the seeded world/scenario row, with {{CONDITIONAL_N_TRIALS}} trials nested within each row. This is conditional evidence over a declared finite grid, not a theorem, breakdown bound, universal attack result, or estimate of performance beyond the registered worlds.](../output/figures/conditional_world.png){#fig:conditional-world width=95% data-slide-manifest="../output/figures/conditional_world.slides.json"}

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
`{{QUALITY_CONTROL_ORDER}}`, and the confidently-wrong-versus-uniform control is
`{{QUALITY_CONFIDENT_WRONG_CONTROL}}`, using {{QUALITY_N_SEEDS}} independent
seeds and {{QUALITY_N_TRIALS}} nested trials per seed. The diagnostic is shown
in [@fig:belief-quality].

![Proper-score and reliability controls. Source relation: original project belief-quality diagnostic; displayed estimands: categorical log score in nats and binned mean-confidence versus empirical-accuracy coordinates in fractions. The x-axis is control type in Panel A and mean confidence in Panel B; the y-axis is mean categorical log score and empirical accuracy, respectively. Open circles identify oracle controls, open diamonds identify uniform controls, and filled crosses identify confidently-wrong controls; the reliability panel retains the same marker-and-dash identities, direct endpoint labels, and a dark dotted perfect-calibration rule. Panel A's capped whiskers are {{CI_PERCENT}}% percentile-bootstrap intervals across independent configured seeds. Trials are nested within seed, and the reliability coordinates display no separate interval. Brier score and expected calibration error are retained as report-only secondary diagnostics and are not plotted in this two-panel figure. The ordered controls test score and reliability implementation on the configured finite world; they do not establish decision optimality, calibration under distribution shift, or robustness beyond the tested conditions.](../output/figures/belief_quality.png){#fig:belief-quality width=90% data-slide-manifest="../output/figures/belief_quality.slides.json"}

## Greedy multi-hypothesis model reduction beyond the main BMR study {#sec:supp-greedy-bmr}

The single-step reduction of [@sec:method-learning] scores one candidate reduced
prior. `bayesian_model_reduction.py` adds `greedy_reduce`, which performs
structure learning over a *family* of redundant states: starting from the full
prior, it scores pruning each not-yet-pruned state against the current reduced
prior, accepts the single prune with the largest positive free-energy gain, and
repeats until no remaining prune improves model evidence.

Every accepted step has
a strictly positive incremental $\Delta F$, so the cumulative evidence is
monotone-increasing and the search recovers the sparse generative model the data
support — a state with genuine evidence yields $\Delta F < 0$ when pruned and is
kept.

This is the multi-state analogue of the configured BMR sign-control result of
[@sec:results-emergence], and is verified directly in the model-reduction tests.
