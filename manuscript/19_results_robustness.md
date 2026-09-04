## Contamination sweep: regime-dependent server behavior under declared attacks {#sec:results-robustness}

This experiment compares server presets in one active-inference colony. Its
legacy labels borrow FedGVI client-divergence vocabulary
[@mildner2025fedgvi], but the presets are not client losses or divergence
theorems. `KLD` denotes the standard log-linear pool; every other label denotes
a `robust_aggregate` heuristic constant in
{{SWEEP_SERVER_OPERATING_POINTS}}. The estimand is consensus mass
$q(\text{true state})$ under the declared contamination mechanism.

The colony contains {{SWEEP_N_AGENTS}} sentinels, of which
{{SWEEP_N_CONTAMINATED}} mix their broadcasts toward a confident-wrong delta at
each contamination rate. The comparisons remain conditional on this fixed
world, attack target, and preset grid.

The evidence map in [@fig:evidence-replication-map] locates these comparisons in
the conditional heuristic-server lane. It keeps the fixed-world matched trials
distinct from the seed-level review below and prevents either result from
inheriting client-loss or variational-server guarantees.

As the contamination rate rises across $\{{{SWEEP_RATES}}\}$, the **standard**
(`KLD`) consensus accuracy degrades monotonically:

| Contamination rate | KLD | RKL | AR | beta | rcce |
|---|---|---|---|---|---|
{{SWEEP_RATE_TABLE_ROWS}}

: Consensus accuracy $q(\text{true state})$ by contamination rate and
configured server operating point ($n = 1$ deterministic sweep per cell).
`KLD` is the standard log-linear pool ([@eq:log-linear-pool]); the other columns
use the fixed heuristic constants listed above in the same seeded colony. This
table is descriptive; inferential paired evidence appears below.
{#tbl:robustness_sweep}

As the saboteurs capture more belief mass, `KLD` falls monotonically while at
least one non-reference preset remains above the stated accuracy threshold.
**Reference trend:** standard accuracy degrades monotonically with rate,
recorded as {{SWEEP_NAIVE_DEGRADES}}.

**Worst-rate display check:** at rate {{SWEEP_WORST_RATE}}, at least one
non-reference preset remains at or above {{SWEEP_ACCURACY_THRESHOLD}}, recorded
as {{SWEEP_ROBUST_ABOVE_THRESHOLD}}. Standard accuracy is
{{SWEEP_NAIVE_ACCURACY}}, the highest pooled robust mean is
{{SWEEP_BEST_ROBUST_ACCURACY}}, and the display method is
{{SWEEP_WORST_RATE_BEST_METHOD}}.

The rate trend above is one deterministic sweep per cell. The paired profile
reruns each rate over $n_{\rm trial} = {{SWEEP_N_TRIALS}}$ matched trials nested
within the fixed seeded world. Each non-reference preset is compared with the
standard pool, and p-values are BH-adjusted within method
[@benjamini1995controlling]. This is a finite, rate-resolved diagnostic, not a
continuous-family result.

| Server preset | Rate | Rank-biserial-derived $d$-equivalent | Label |
|---|---|---|---|
{{SWEEP_PAIRED_BY_RATE_EFFECT_TABLE_ROWS}}

: Effect-size projection of the per-contamination-rate standard-versus-preset
paired tests, keyed by `(server preset, rate)`. Each cell uses
{{SWEEP_N_TRIALS}} matched trial replicates nested within the fixed seeded world.
Joining this projection to [@tbl:paired-by-rate-inference] on the displayed key
reconstructs every source row exactly; the label and $d$-equivalent do not add an
independent world-level estimand.
{#tbl:paired-by-rate}

| Server preset | Rate | Raw p | q | Reject |
|---|---|---|---|---|
{{SWEEP_PAIRED_BY_RATE_INFERENCE_TABLE_ROWS}}

: Inference projection of the same per-rate matched-pairs Wilcoxon tests
[@wilcoxon1945individual; @fay2010wilcoxon], keyed by `(server preset, rate)` and
BH-deflated within each preset's rate family. `Reject` is the report-owned
family decision. The {{SWEEP_N_TRIALS}} trials are nested within one fixed
seeded world, not {{SWEEP_N_TRIALS}} independent worlds; this table must be read
with [@tbl:paired-by-rate] to recover the complete source row.
{#tbl:paired-by-rate-inference}

The `KLD` baseline is excluded from both projections because it is the standard reference, not a
self-contrast. The displayed $d$-equivalent is a rank-biserial-derived
transform, not raw Cohen's $d$; the signed-saturation marker flags contrasts
where the rank-biserial correlation saturates at $\pm1$. These contrasts
decorate the server-side `robust_aggregate` heuristic only.

![Consensus accuracy: probability mass assigned to the true hidden state. The plotted estimand is $q(\text{true state})$. Source relation: original project robustness extension; estimand: true-state probability mass; uncertainty: matched-trial percentile-bootstrap intervals over configured trials. The x-axis is saboteur convex-mix contamination rate over $\{{{SWEEP_RATES}}\}$; the y-axis is consensus probability mass on the true state. Curves compare the standard `KLD` log-linear pool with `robust_aggregate` server presets ${{SWEEP_SERVER_OPERATING_POINTS}}$ for ${{SWEEP_N_AGENTS}}$ agents, including ${{SWEEP_N_CONTAMINATED}}$ saboteurs; they are not named client losses. Distinct markers, dashes, direct labels, and the dark dotted threshold at {{SWEEP_ACCURACY_THRESHOLD}} duplicate color. The independent replication unit is a matched synthetic trial within the fixed seeded true state and attack target, with {{SWEEP_N_TRIALS}} trials per rate. Curves show trial means with {{CI_PERCENT}}% percentile-bootstrap confidence intervals. At the largest rate, the standard pool reaches {{SWEEP_PROFILE_NAIVE_ACCURACY}} and the within-sweep highest pooled robust mean reaches {{SWEEP_PROFILE_BEST_ROBUST_ACCURACY}}. That operating point is a disclosed display selection: low-rate robust means can match or trail the standard pool, and individual presets can fall below the threshold. The truncated linear y-axis enlarges the declared threshold region. This fixed-world sweep does not establish a universal method ranking, alternate-world generalization, a client-loss theorem, bounded influence, or Byzantine tolerance; the selection-free seed-level review is reported separately.](../output/figures/robustness_sweep.png){#fig:robustness-sweep width=80%}

### Selection-free all-method robustness review {#sec:results-review-grid}

Figure [@fig:robustness-review-grid] is the principal comparative robustness
surface. It joins the conditional-world cells to the directional rate profiles
while retaining every configured non-reference server preset. It uses
{{REVIEW_GRID_N_SEEDS}} deterministic seed replicates and
{{REVIEW_GRID_N_TRIALS}} trials nested within each seed and cell.

Read the portrait surface from top to bottom: Panel A is the full-width
conditional heatmap, and Panels B–D stack the three rate profiles. Their elbow
leaders terminate in reserved right-side label lanes so curve identity remains
separate from the plotted endpoints.

The finite attack union is {{REVIEW_GRID_ATTACKS}}. The rate-resolved directional
mechanisms are {{REVIEW_GRID_DIRECTIONAL}}, with entropy controls
{{REVIEW_GRID_ENTROPY_CONTROLS}} and registered rates
$\{{{REVIEW_GRID_RATES}}\}$. The independent unit is
{{REVIEW_GRID_INDEPENDENT_UNIT}}; the nesting rule is
{{REVIEW_GRID_TRIAL_STRUCTURE}}.

The payload is {{REVIEW_GRID_SELECTION_STATUS}}, and its statistics surface is
{{REVIEW_GRID_STATS_STATUS}}. It retains seed-level contrasts, paired
Wilcoxon/rank-biserial results, percentile-bootstrap intervals, MCSE, an
observed-effect MDE, and BH-adjusted rate families. BH ownership is
{{REVIEW_GRID_BH_OWNERSHIP}}.

The precision plan targets maximum MCSE {{REVIEW_GRID_TARGET_MAX_MCSE}}; the
observed maximum is {{REVIEW_GRID_OBSERVED_MAX_MCSE}} across
{{REVIEW_GRID_SIGNED_CELLS}}. No method is selected by seed, rate, or pooled
mean. Mixed and reversed cells remain visible, and the grid does not close the
open calibration, external-data, or server-theory questions.

The source-generated exact-value fallback calls the same finite grouping helper
as Panel A. For every displayed attack-by-weight cell it records the printed
grouped mean and the printed half min–max span, preventing the numeric fallback
from drifting from the renderer.

![The robustness review retains every server preset and preserves mixed signs across the attack grid. Source relation: original project conditional simulation review joining registered conditional-world and directional-rate reports; study status: selection-free, all-method finite-grid evidence; estimand: seed-level robust-minus-standard true-state probability-mass contrast. Read the portrait figure from top to bottom: Panel A is a full-width heatmap of conditional attack cells by adversarial-weight setting; Panels B–D stack confident-wrong, Byzantine, and drift rate profiles. The x-axis is adversarial weight in Panel A and contamination rate in Panels B–D; the y-axis is contrast in probability units. Printed cell signs, distinct markers and dashes, and elbow leaders into reserved right-side endpoint-label lanes duplicate color; the dark dotted rule marks zero. The independent replication unit is the configured seed ($n={{REVIEW_GRID_N_SEEDS}}$), with {{REVIEW_GRID_N_TRIALS}} trials plus agents and conditions nested within each cell. Rate-profile bands are {{CI_PERCENT}}% percentile-bootstrap intervals over seeds. The second heatmap line is half the finite-grid min–max span, not a confidence interval. No curve is selected by pooled mean. The figure supports only mechanism- and grid-conditional contrasts; it does not establish a universal winner, bounded influence, Byzantine tolerance, a causal effect, independent shared-design cells, or generalization beyond the registered worlds.](../output/figures/robustness_review_grid.png){#fig:robustness-review-grid width=98%}

### Conditional matched-trial comparison at the declared rate {#sec:results-verdict}

At contamination rate {{SWEEP_VERDICT_RATE}}, each of
{{SWEEP_N_TRIALS}} matched trials redraws the heterogeneous healthy colony while
holding the seeded world's true state and attack target fixed. Each
non-reference server preset is compared with the standard pool using the paired
Wilcoxon signed-rank test [@wilcoxon1945individual]. Benjamini–Hochberg FDR
adjustment applies within the declared family [@benjamini1995controlling].

The generated `Wins` field is true only when the adjusted null is rejected and
the paired effect is positive. It is conditional bookkeeping for this fixed
world and rate, not a universal method ranking.

| Server preset | Effect size | Raw p | q | Wins |
|---|---|---|---|---|
{{SWEEP_VERDICT_TABLE_ROWS}}

: Per-preset paired comparison against the standard pool at rate {{SWEEP_VERDICT_RATE}} ($n_{\rm trial} = {{SWEEP_N_TRIALS}}$ matched trial replicates, signed-rank test under the paired-difference assumptions of [@sec:methods-paired]). This is a fixed-world conditional diagnostic; seed-level inference is reported separately in [@sec:results-review-grid]. Effect size is the matched-pairs rank-biserial correlation; positive values mean the non-reference preset exceeds the standard pool. A `Wins` value is true only when the BH-adjusted null is rejected *and* the effect is positive, with FDR scoped to this declared family. {#tbl:robustness_verdict}

| Method | Derived $d_{\rm eq}$ | Label | Mean $\Delta$ accuracy | {{CI_PERCENT}}% CI |
|---|---|---|---|---|
{{SWEEP_VERDICT_EFFECT_ESTIMATE_TABLE_ROWS}}

: Effect estimates at rate {{SWEEP_VERDICT_RATE}} ($n_{\rm trial} = {{SWEEP_N_TRIALS}}$ matched trial replicates). Source relation and study status: these rows are generated from the typed primary-sweep report and constitute a conditional fixed-world diagnostic. Each row gives the rank-biserial-derived $d$-equivalent and display label, followed by the mean non-reference-minus-standard accuracy difference in accuracy-proportion units and its {{CI_PERCENT}}% percentile-bootstrap CI. The matched trial is the resampling unit; all trials remain nested within the one seeded world. Row order is preserved in [@tbl:verdict-inference-power]. The signed-saturation marker replaces a misleading finite literal where the rank-biserial transform diverges. These estimates do not identify an independently replicated world effect, calibration result, or universal server ranking. {#tbl:verdict-effect-estimates}

| Method | Raw p | q | Power | Target $n_{\rm trial}$ | Reject |
|---|---|---|---|---|---|
{{SWEEP_VERDICT_INFERENCE_POWER_TABLE_ROWS}}

: Inference and planning diagnostics for the same ordered method rows and matched trials as [@tbl:verdict-effect-estimates]. Source relation and study status: the typed primary-sweep report supplies every cell for this conditional fixed-world comparison. Raw paired-Wilcoxon p-values are accompanied by BH-deflated q-values and the resulting rejection indicator for the declared method family. Observed-effect design power uses $\alpha = {{SWEEP_POWER_ALPHA}}$ and alternative `{{SWEEP_POWER_ALTERNATIVE}}`; the prospective $n_{\rm trial}$ targets power {{SWEEP_TARGET_POWER}}. The matched trial is the inferential and planning unit within one seeded world. Power and target sample size are conditional approximations derived from the observed effect, not independent confirmation, calibration, a guarantee of future power, or evidence beyond the declared world and rate. {#tbl:verdict-inference-power}

| Method | n | Mean acc. @ verdict rate | {{CI_PERCENT}}% CI |
|---|---|---|---|
{{SWEEP_ACCURACY_AT_VERDICT_TABLE_ROWS}}

: Per-method consensus accuracy at the verdict rate {{SWEEP_VERDICT_RATE}} with {{CI_PERCENT}}% percentile-bootstrap CI, including the standard `KLD` baseline. The CI is conditional on the seeded matched-trial design and resamples trials, not alternate world models. The standard pool sits at {{SWEEP_NAIVE_VERDICT_ACCURACY_MEAN}} ($[{{SWEEP_NAIVE_VERDICT_ACCURACY_CI_LO}}, {{SWEEP_NAIVE_VERDICT_ACCURACY_CI_HI}}]$); the highest pooled robust mean among the configured members is {{SWEEP_BEST_VERDICT_ACCURACY_MEAN}} ($[{{SWEEP_BEST_VERDICT_ACCURACY_CI_LO}}, {{SWEEP_BEST_VERDICT_ACCURACY_CI_HI}}]$). {#tbl:accuracy-at-verdict}

**Reference mean.** Naive-pool mean accuracy at the verdict rate is
{{SWEEP_NAIVE_VERDICT_RATE_MEAN}} (per-trial mean over {{SWEEP_N_TRIALS}}
trials; the bootstrap CI is in [@tbl:accuracy-at-verdict]).

**Positive contrasts.** At least one non-reference preset is a BH-rejected
positive contrast: {{SWEEP_ANY_ROBUST_WINS}}.

**Headline display.** The method is {{SWEEP_HEADLINE_METHOD}} (tied set:
{{SWEEP_HEADLINE_TIE_SET}}), with rank-biserial-derived $d$-equivalent
{{SWEEP_BEST_D_EQUIVALENT}} ({{SWEEP_BEST_EFFECT_LABEL}}), mean accuracy
difference {{SWEEP_BEST_MEAN_ACC_DIFF}}
($[{{SWEEP_BEST_MEAN_ACC_DIFF_CI_LO}}, {{SWEEP_BEST_MEAN_ACC_DIFF_CI_HI}}]$),
raw $p = {{SWEEP_BEST_RAW_PVALUE_MATH}}$, and
$q = {{SWEEP_BEST_QVALUE_MATH}}$.

Its observed-effect design power is {{SWEEP_BEST_POWER}}. Prospective $n$ for
power {{SWEEP_TARGET_POWER}} is {{SWEEP_BEST_N_FOR_TARGET_POWER}}.

**Display rule.** Under {{SWEEP_HEADLINE_SELECTION_RULE}} (tie-break:
{{SWEEP_HEADLINE_TIE_BREAK}}), observed-effect design power is
{{SWEEP_HEADLINE_POWER}} at the run's $n_{\rm trial} = {{SWEEP_N_TRIALS}}$. A
confirmatory replication should budget
$n_{\rm trial} = {{SWEEP_HEADLINE_N_FOR_TARGET_POWER}}$ for power
{{SWEEP_TARGET_POWER}} (prospective $n = {{SWEEP_PROSPECTIVE_N}}$).

The standard pool degrades and some non-reference presets have positive
conditional contrasts. The statistics module produces the paired tests,
multiple-testing adjustment, bootstrap intervals, and planning analysis; the
prose does not promote them beyond their matched-trial estimand.

The headline label is a deterministic display choice, not a unique scientific winner. The complete tied set is {{SWEEP_HEADLINE_TIE_SET}}, the largest paired mean-difference method is {{SWEEP_LARGEST_MEAN_DIFFERENCE_METHOD}}, and the worst-rate pooled display method is {{SWEEP_WORST_RATE_BEST_METHOD}}. These may differ because they answer different descriptive questions.

#### Server-side heuristic axis: conditional contrasts without theorem transfer

The comparison above belongs to the heuristic server axis defined in
[@sec:robustness-axes].
[@fig:robust-weights] visualizes the `robust_aggregate` divergence-reweighting
that down-weights agents at pooling time. Its positive formal property is the
naive-recovery limit of Theorem \ref{thm:belief-sharing-recovery}
([@eq:robust-identity], the
{{RECOVERY_AGGREGATE_MAXDIFF}} residual of [@sec:results-recovery]), and it
has a scoped no-go result for a declared separable objective class, not an
objective certificate.

The heuristic carries no bounded-influence guarantee. The effect sizes,
intervals, and planning quantities characterize its conditional contrasts; they
do not certify the per-agent generalized-Bayes claim, which remains separately
source-conditional in [@sec:results-baseline].

![Server-side influence weights assigned by `robust_aggregate`. Source relation: original project server-side diagnostic; estimand: normalized pooling weight; uncertainty: deterministic single-run display. Divergence-reweighting weights for each of the ${{SWEEP_N_AGENTS}}$ agents at the verdict contamination rate ${{SWEEP_VERDICT_RATE}}$ (the convex-mix strength applied to each saboteur's belief — distinct from the *count* of contaminated agents, ${{SWEEP_N_CONTAMINATED}}$ of ${{SWEEP_N_AGENTS}}$, reported in the in-figure box). x-axis: agent index, zero-based ($a0$ upward), with each agent's role (honest / adversary) shown beneath its label; y-axis: normalized pooling weight, with weights summing to one and the dotted reference marking the equal-weight pool ($1/n$). The ${{SWEEP_N_CONTAMINATED}}$ contaminated agents use cross-hatching and direct role labels; downward arrows mark suppression below the equal-weight reference. No resampling interval applies to this deterministic run. This server-heuristic display establishes neither the per-client bounded-influence theorem, Byzantine tolerance, nor a universal influence bound.](../output/figures/robust_influence_weights.png){#fig:robust-weights width=80%}

### Variational aggregator: conservative objective-backed weight control {#sec:results-variational}

The heuristic has the higher point accuracy in the configured verdict cell. The
variational aggregator of [@sec:method-variational], derived in
[@sec:supp-variational], is its objective-backed complement. Each exact block
update does not increase the stated free energy [@eq:agg-free-energy], and a
converged fixed point is coordinatewise stationary. Two deterministic runs show
its weight behavior at robustness {{VARIATIONAL_ROBUSTNESS}}.

First, the descent. On a contaminated colony the free energy falls monotonically
from {{VARIATIONAL_F_INITIAL}} to {{VARIATIONAL_F_FINAL}} (a drop of
{{VARIATIONAL_DELTA_F}} over {{VARIATIONAL_ITERATIONS}} block-coordinate
iterations, converged: {{VARIATIONAL_CONVERGED}}), with a largest single-step
*increase* of {{VARIATIONAL_MAX_ASCENT}} — machine zero, the numerical witness
of the descent theorem ([@sec:supp-theorem]).

![Variational free energy $F(q, a)$ as a function of block-coordinate descent iteration. Source relation: original project objective-descent diagnostic; estimand: free energy in nats by iteration; uncertainty: none for the deterministic seeded run. The trace is a single `variational_aggregate` fusion of a ${{SWEEP_N_AGENTS}}$-agent contaminated colony (robustness ${{VARIATIONAL_ROBUSTNESS}}$). The x-axis is block-coordinate iteration number and the y-axis is $F(q, a)$ in nats. Open triangles joined by a dash-dot path identify the variational trajectory, a filled triangle marks the terminal iterate, enlarged open triangles mark the endpoints of the largest observed descent step, and a dark dotted horizontal rule marks the final stationary level; these shapes and line patterns duplicate color. The curve is monotone non-increasing across all recorded iterations (largest single-step increase: ${{VARIATIONAL_MAX_ASCENT_MATH}}$ nats, at machine precision), and the implementation reports converged status {{VARIATIONAL_CONVERGED}} at $F={{VARIATIONAL_F_FINAL}}$ nats. Iterations are ordered states of one deterministic run, not independent replicates, so no resampling interval applies. This trace verifies descent only for the executed path; it does not certify a global optimum, exhaustive basin search, or the separate `robust_aggregate` heuristic.](../output/figures/aggregation_descent.png){#fig:aggregation-descent width=80%}

Second, the effective-weight response. As one agent is drifted from healthy toward a
confident-wrong delta, its normalized influence falls from
{{VARIATIONAL_INFLUENCE_CLEAN}} to below {{VARIATIONAL_INFLUENCE_DIVERGED}} — a factor
of {{VARIATIONAL_INFLUENCE_DROP_FACTOR}} below the fixed
{{VARIATIONAL_NAIVE_INFLUENCE}} the naive log-linear pool grants every agent
regardless of how wrong it is. The gap between the falling variational curve and
the flat naive line is the empirical redescending weight response, drawn.

![Normalized influence weight of one probed agent. Source relation: original project variational-server diagnostic; estimand: the probed agent's normalized server weight as its posterior drifts toward a confident-wrong delta. The x-axis is outlier drift from consensus at zero to the wrong-state delta at one; the y-axis is normalized influence weight. The open-triangle dash-dot path is `variational_aggregate`; the circle-solid horizontal reference is the naive log-linear pool. Direct endpoint labels name both rules and values, shaded separation shows their tested-path gap, and an adversarial X with an arrow marks the first displayed point below one-half of the naive weight. As drift becomes extreme, the variational weight falls below ${{VARIATIONAL_INFLUENCE_DIVERGED}}$, whereas the naive pool remains fixed at $1/n={{VARIATIONAL_NAIVE_INFLUENCE}}$. This deterministic seeded sweep contains $n={{SWEEP_N_AGENTS}}$ agents in one configured colony; drift points are ordered evaluations, not independent replications, so no confidence interval or error band applies. The pattern is redescending normalized weight along this path, while the algebraic property bounds raw effective weight; the figure is not an estimator-level B-robustness proof and does not transfer the property to the sharper `robust_aggregate` heuristic.](../output/figures/bounded_influence.png){#fig:bounded-influence width=80%}

The honest trade is conservatism: because $F$ carries the $-H(q)$ entropy term,
its consensus is the maximum-entropy distribution consistent with the weighted
cross-entropies, deliberately flatter than the product-of-experts. The
variational aggregator therefore does *not* win the peak-accuracy verdict of
[@sec:results-verdict] — that remains the sharp heuristic's role — and the two
are reported as complements, never conflated: rigor-with-conservatism on one
side, accuracy-without-an-objective on the other.
