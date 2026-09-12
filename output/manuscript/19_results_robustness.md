## Contamination sweep: regime-dependent server behavior under declared attacks {#sec:results-robustness}

This experiment compares server presets in one active-inference colony. Its
legacy labels borrow FedGVI client-divergence vocabulary
[@mildner2025fedgvi], but the presets are not client losses or divergence
theorems. `KLD` denotes the standard log-linear pool; every other label denotes
a `robust_aggregate` heuristic constant in
KLD (c=0.00), RKL (c=1.50), AR (c=1.30), beta (c=1.70), rcce (c=1.60). The estimand is consensus mass
$q(\text{true state})$ under the declared contamination mechanism.

The colony contains 7 sentinels, of which
2 mix their broadcasts toward a confident-wrong delta at
each contamination rate. The comparisons remain conditional on this fixed
world, attack target, and preset grid.

The evidence map in [@fig:evidence-replication-map] locates these comparisons in
the conditional heuristic-server lane. It keeps the fixed-world matched trials
distinct from the seed-level review below and prevents either result from
inheriting client-loss or variational-server guarantees.

As the contamination rate rises across $\{0, 0.225, 0.45, 0.675, 0.9\}$, the **standard**
(`KLD`) consensus accuracy degrades monotonically:

| Contamination rate | KLD | RKL | AR | beta | rcce |
|---|---|---|---|---|---|
| 0 | 1.000 | 0.997 | 0.998 | 0.996 | 0.996 |
| 0.225 | 0.999 | 0.993 | 0.995 | 0.990 | 0.992 |
| 0.45 | 0.995 | 0.987 | 0.990 | 0.984 | 0.986 |
| 0.675 | 0.975 | 0.985 | 0.989 | 0.981 | 0.983 |
| 0.9 | 0.693 | 0.984 | 0.988 | 0.980 | 0.982 |

: Consensus accuracy $q(\text{true state})$ by contamination rate and
configured server operating point ($n = 1$ deterministic sweep per cell).
`KLD` is the standard log-linear pool ([@eq:log-linear-pool]); the other columns
use the fixed heuristic constants listed above in the same seeded colony. This
table is descriptive; inferential paired evidence appears below.
{#tbl:robustness_sweep}

As the saboteurs capture more belief mass, `KLD` falls monotonically while at
least one non-reference preset remains above the stated accuracy threshold.
**Reference trend:** standard accuracy degrades monotonically with rate,
recorded as Yes.

**Worst-rate display check:** at rate 0.900, at least one
non-reference preset remains at or above 0.50, recorded
as Yes. Standard accuracy is
0.6928; the highest robust consensus mass in this
single-world mechanistic sweep is 0.9880,
attained by AR.

The rate trend above is one deterministic sweep per cell. The paired profile
reruns each rate over $n_{\rm trial} = 960$ matched trials nested
within the fixed seeded world. Each non-reference preset is compared with the
standard pool, and p-values are BH-adjusted within method
[@benjamini1995controlling]. This is a finite, rate-resolved diagnostic, not a
continuous-family result.

| Server preset | Rate | Rank-biserial-derived $d$-equivalent | Label |
|---|---|---|---|
| RKL | 0 | saturated (r=-1) | large |
| RKL | 0.225 | saturated (r=-1) | large |
| RKL | 0.45 | saturated (r=-1) | large |
| RKL | 0.675 | 13.79 | large |
| RKL | 0.9 | -0.37 | small |
| AR | 0 | saturated (r=-1) | large |
| AR | 0.225 | saturated (r=-1) | large |
| AR | 0.45 | -303.73 | large |
| AR | 0.675 | 160.07 | large |
| AR | 0.9 | -1.57 | large |
| beta | 0 | saturated (r=-1) | large |
| beta | 0.225 | saturated (r=-1) | large |
| beta | 0.45 | saturated (r=-1) | large |
| beta | 0.675 | 2.81 | large |
| beta | 0.9 | 0.61 | medium |
| rcce | 0 | saturated (r=-1) | large |
| rcce | 0.225 | saturated (r=-1) | large |
| rcce | 0.45 | saturated (r=-1) | large |
| rcce | 0.675 | 5.67 | large |
| rcce | 0.9 | 0.14 | negligible |

: Effect-size projection of the per-contamination-rate standard-versus-preset
paired tests, keyed by `(server preset, rate)`. Each cell uses
960 matched trial replicates nested within the fixed seeded world.
Joining this projection to [@tbl:paired-by-rate-inference] on the displayed key
reconstructs every source row exactly; the label and $d$-equivalent do not add an
independent world-level estimand.
{#tbl:paired-by-rate}

| Server preset | Rate | Raw p | q | Reject |
|---|---|---|---|---|
| RKL | 0 | 1.11e-158 | 1.85e-158 | Yes |
| RKL | 0.225 | 1.11e-158 | 1.85e-158 | Yes |
| RKL | 0.45 | 1.11e-158 | 1.85e-158 | Yes |
| RKL | 0.675 | 1.88e-155 | 2.35e-155 | Yes |
| RKL | 0.9 | 1.06e-06 | 1.06e-06 | Yes |
| AR | 0 | 1.11e-158 | 1.47e-158 | Yes |
| AR | 0.225 | 1.11e-158 | 1.47e-158 | Yes |
| AR | 0.45 | 1.13e-158 | 1.47e-158 | Yes |
| AR | 0.675 | 1.17e-158 | 1.47e-158 | Yes |
| AR | 0.9 | 1.62e-61 | 1.62e-61 | Yes |
| beta | 0 | 1.11e-158 | 1.85e-158 | Yes |
| beta | 0.225 | 1.11e-158 | 1.85e-158 | Yes |
| beta | 0.45 | 1.11e-158 | 1.85e-158 | Yes |
| beta | 0.675 | 6.29e-106 | 7.86e-106 | Yes |
| beta | 0.9 | 6.04e-15 | 6.04e-15 | Yes |
| rcce | 0 | 1.11e-158 | 1.85e-158 | Yes |
| rcce | 0.225 | 1.11e-158 | 1.85e-158 | Yes |
| rcce | 0.45 | 1.11e-158 | 1.85e-158 | Yes |
| rcce | 0.675 | 2.37e-141 | 2.96e-141 | Yes |
| rcce | 0.9 | 6.54e-02 | 6.54e-02 | No |

: Inference projection of the same per-rate matched-pairs Wilcoxon tests
[@wilcoxon1945individual; @fay2010wilcoxon], keyed by `(server preset, rate)` and
BH-deflated within each preset's rate family. `Reject` is the report-owned
family decision. The 960 trials are nested within one fixed
seeded world, not 960 independent worlds; this table must be read
with [@tbl:paired-by-rate] to recover the complete source row.
{#tbl:paired-by-rate-inference}

The `KLD` baseline is excluded from both projections because it is the standard reference, not a
self-contrast. The displayed $d$-equivalent is a rank-biserial-derived
transform, not raw Cohen's $d$; the signed-saturation marker flags contrasts
where the rank-biserial correlation saturates at $\pm1$. These contrasts
decorate the server-side `robust_aggregate` heuristic only.

![Consensus accuracy: probability mass assigned to the true hidden state. The plotted estimand is $q(\text{true state})$. Source relation: original project robustness extension; estimand: true-state probability mass; uncertainty: matched-trial percentile-bootstrap intervals over configured trials. The x-axis is saboteur convex-mix contamination rate over $\{0, 0.225, 0.45, 0.675, 0.9\}$; the y-axis is consensus probability mass on the true state. Curves compare the standard `KLD` log-linear pool with `robust_aggregate` server presets $KLD (c=0.00), RKL (c=1.50), AR (c=1.30), beta (c=1.70), rcce (c=1.60)$ for $7$ agents, including $2$ saboteurs; they are not named client losses. Distinct markers, dashes, direct labels, and the dark dotted threshold at 0.50 duplicate color. The independent replication unit is a matched synthetic trial within the fixed seeded true state and attack target, with 960 trials per rate. Curves show trial means with 95% percentile-bootstrap confidence intervals. At the largest rate, the standard pool reaches 0.6697 and the within-sweep highest pooled robust mean reaches 0.7857. That operating point is a disclosed display selection: low-rate robust means can match or trail the standard pool, and individual presets can fall below the threshold. The truncated linear y-axis enlarges the declared threshold region. This fixed-world sweep does not establish a universal method ranking, alternate-world generalization, a client-loss theorem, bounded influence, or Byzantine tolerance; the selection-free seed-level review is reported separately.](../output/figures/robustness_sweep.png){#fig:robustness-sweep width=80% data-slide-manifest="../output/figures/robustness_sweep.slides.json"}

### Selection-free all-method robustness review {#sec:results-review-grid}

The comparison in [@fig:robustness-review-grid] is the principal comparative robustness
surface. It joins the conditional-world cells to the directional rate profiles
while retaining every configured non-reference server preset. It uses
160 deterministic seed replicates and
24 trials nested within each seed and cell.

Read the portrait surface from top to bottom: Panel A is the full-width
conditional heatmap, and Panels B–D stack the three rate profiles. Their elbow
leaders terminate in reserved right-side label lanes so curve identity remains
separate from the plotted endpoints.

The finite attack union is clean, confident wrong, permutation, byzantine, drift, label noise, uniform. The rate-resolved directional
mechanisms are confident wrong, byzantine, drift, with entropy controls
uniform, label noise and registered rates
$\{0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$. The independent unit is
seed within a declared scenario or rate cell; the nesting rule is
n_trials nested within each seed/cell; no trial is promoted to an independent world.

The payload is selection-free source payload; every configured non-KLD method is reported at every directional rate and no winner is used for inference, and its statistics surface is
selection-free. It retains seed-level contrasts, paired
Wilcoxon/rank-biserial results, percentile-bootstrap intervals, MCSE, an
observed-effect MDE, and BH-adjusted rate families. BH ownership is
BH is applied within each attack-mechanism × method rate family; cells sharing design structure are not treated as independent families.

The precision plan targets maximum MCSE 0.0100; the
observed maximum is 0.0066 across
96. No method is selected by seed, rate, or pooled
mean. Mixed and reversed cells remain visible, and the grid does not close the
open calibration, external-data, or server-theory questions.

The source-generated exact-value fallback calls the same finite grouping helper
as Panel A. For every displayed attack-by-weight cell it records the printed
grouped mean and the printed half min–max span, preventing the numeric fallback
from drifting from the renderer.

![The robustness review retains every server preset and preserves mixed signs across the attack grid. Source relation: original project conditional simulation review joining registered conditional-world and directional-rate reports; study status: selection-free, all-method finite-grid evidence; estimand: seed-level robust-minus-standard true-state probability-mass contrast. Read the portrait figure from top to bottom: Panel A is a full-width heatmap of conditional attack cells by adversarial-weight setting; Panels B–D stack confident-wrong, Byzantine, and drift rate profiles. The x-axis is adversarial weight in Panel A and contamination rate in Panels B–D; the y-axis is contrast in probability units. Printed cell signs, distinct markers and dashes, and elbow leaders into reserved right-side endpoint-label lanes duplicate color; the dark dotted rule marks zero. The independent replication unit is the configured seed ($n=160$), with 24 trials plus agents and conditions nested within each cell. Rate-profile bands are 95% percentile-bootstrap intervals over seeds. The second heatmap line is half the finite-grid min–max span, not a confidence interval. No curve is selected by pooled mean. The figure supports only mechanism- and grid-conditional contrasts; it does not establish a universal winner, bounded influence, Byzantine tolerance, a causal effect, independent shared-design cells, or generalization beyond the registered worlds.](../output/figures/robustness_review_grid.png){#fig:robustness-review-grid data-slide-manifest="../output/figures/robustness_review_grid.slides.json" width=98%}

### Conditional matched-trial comparison at the declared rate {#sec:results-verdict}

At contamination rate 0.800, each of
960 matched trials redraws the heterogeneous healthy colony while
holding the seeded world's true state and attack target fixed. Each
non-reference server preset is compared with the standard pool using the paired
Wilcoxon signed-rank test [@wilcoxon1945individual]. Benjamini–Hochberg FDR
adjustment applies within the declared family [@benjamini1995controlling].

The generated `Wins` field is true only when the adjusted null is rejected and
the paired effect is positive. It is conditional bookkeeping for this fixed
world and rate, not a universal method ranking.

| Server preset | Effect size | Raw p | q | Wins |
|---|---|---|---|---|
| AR | 1.0000 | 1.11e-158 | 1.11e-158 | Yes |
| RKL | 1.0000 | 1.11e-158 | 1.11e-158 | Yes |
| beta | 1.0000 | 1.11e-158 | 1.11e-158 | Yes |
| rcce | 1.0000 | 1.11e-158 | 1.11e-158 | Yes |

: Per-preset paired comparison against the standard pool at rate 0.800 ($n_{\rm trial} = 960$ matched trial replicates, signed-rank test under the paired-difference assumptions of [@sec:methods-paired]). This is a fixed-world conditional diagnostic; seed-level inference is reported separately in [@sec:results-review-grid]. Effect size is the matched-pairs rank-biserial correlation; positive values mean the non-reference preset exceeds the standard pool. A `Wins` value is true only when the BH-adjusted null is rejected *and* the effect is positive, with FDR scoped to this declared family. {#tbl:robustness_verdict}

| Method | Derived $d_{\rm eq}$ | Label | Mean $\Delta$ accuracy | 95% CI |
|---|---|---|---|---|
| AR | saturated (r=+1) | large | 0.0846 | [0.0821, 0.0873] |
| RKL | saturated (r=+1) | large | 0.0809 | [0.0783, 0.0834] |
| beta | saturated (r=+1) | large | 0.0764 | [0.0738, 0.0790] |
| rcce | saturated (r=+1) | large | 0.0787 | [0.0761, 0.0814] |

: Effect estimates at rate 0.800 ($n_{\rm trial} = 960$ matched trial replicates). Source relation and study status: these rows are generated from the typed primary-sweep report and constitute a conditional fixed-world diagnostic. Each row gives the rank-biserial-derived $d$-equivalent and display label, followed by the mean non-reference-minus-standard accuracy difference in accuracy-proportion units and its 95% percentile-bootstrap CI. The matched trial is the resampling unit; all trials remain nested within the one seeded world. Row order is preserved in [@tbl:verdict-inference-power]. The signed-saturation marker replaces a misleading finite literal where the rank-biserial transform diverges. These estimates do not identify an independently replicated world effect, calibration result, or universal server ranking. {#tbl:verdict-effect-estimates}

| Method | Raw p | q | Power | Target $n_{\rm trial}$ | Reject |
|---|---|---|---|---|---|
| AR | 1.11e-158 | 1.11e-158 | 1.0000 | 5 | Yes |
| RKL | 1.11e-158 | 1.11e-158 | 1.0000 | 5 | Yes |
| beta | 1.11e-158 | 1.11e-158 | 1.0000 | 5 | Yes |
| rcce | 1.11e-158 | 1.11e-158 | 1.0000 | 5 | Yes |

: Inference and planning diagnostics for the same ordered method rows and matched trials as [@tbl:verdict-effect-estimates]. Source relation and study status: the typed primary-sweep report supplies every cell for this conditional fixed-world comparison. Raw paired-Wilcoxon p-values are accompanied by BH-deflated q-values and the resulting rejection indicator for the declared method family. Observed-effect design power uses $\alpha = 0.05$ and alternative `greater`; the prospective $n_{\rm trial}$ targets power 0.80. The matched trial is the inferential and planning unit within one seeded world. Power and target sample size are conditional approximations derived from the observed effect, not independent confirmation, calibration, a guarantee of future power, or evidence beyond the declared world and rate. {#tbl:verdict-inference-power}

| Method | n | Mean acc. @ verdict rate | 95% CI |
|---|---|---|---|
| KLD | 960 | 0.9021 | [0.8993, 0.9049] |
| RKL | 960 | 0.9829 | [0.9827, 0.9832] |
| AR | 960 | 0.9867 | [0.9865, 0.9869] |
| beta | 960 | 0.9785 | [0.9782, 0.9787] |
| rcce | 960 | 0.9808 | [0.9806, 0.9810] |

: Per-method consensus accuracy at the verdict rate 0.800 with 95% percentile-bootstrap CI, including the standard `KLD` baseline. The CI is conditional on the seeded matched-trial design and resamples trials, not alternate world models. The standard pool sits at 0.9021 ($[0.8993, 0.9049]$); the headline display method RKL has pooled mean 0.9829 ($[0.9827, 0.9832]$). {#tbl:accuracy-at-verdict}

**Reference mean.** Naive-pool mean accuracy at the verdict rate is
0.9021 (per-trial mean over 960
trials; the bootstrap CI is in [@tbl:accuracy-at-verdict]).

**Positive contrasts.** At least one non-reference preset is a BH-rejected
positive contrast: Yes.

**Headline display.** The method is RKL (tied set:
RKL, AR, beta, rcce), with rank-biserial-derived $d$-equivalent
saturated (r=+1) (large), mean accuracy
difference 0.0809
($[0.0783, 0.0834]$),
raw $p = 1.11 \times 10^{-158}$, and
$q = 1.11 \times 10^{-158}$.

Its observed-effect design power is 1.0000. Prospective $n$ for
power 0.80 is 5.

**Display rule.** Under largest positive rank-biserial effect_size; stable method order tie-break (tie-break:
first robust method in divergences order), observed-effect design power is
1.0000 at the run's $n_{\rm trial} = 960$. A
confirmatory replication should budget
$n_{\rm trial} = 5$ for power
0.80 (prospective $n = 7$).

The standard pool degrades and some non-reference presets have positive
conditional contrasts. The statistics module produces the paired tests,
multiple-testing adjustment, bootstrap intervals, and planning analysis; the
prose does not promote them beyond their matched-trial estimand.

The headline label is a deterministic display choice, not a unique scientific winner. The complete tied set is RKL, AR, beta, rcce, the largest paired mean-difference method is AR, and the worst-rate pooled display method is beta. These may differ because they answer different descriptive questions.

#### Server-side heuristic axis: conditional contrasts without theorem transfer

The comparison above belongs to the heuristic server axis defined in
[@sec:robustness-axes].
[@fig:robust-weights] visualizes the `robust_aggregate` divergence-reweighting
that down-weights agents at pooling time. Its positive formal property is the
naive-recovery limit of Theorem \ref{thm:belief-sharing-recovery}
([@eq:robust-identity], the
0 residual of [@sec:results-recovery]), and it
has a scoped no-go result for a declared separable objective class, not an
objective certificate.

The heuristic carries no bounded-influence guarantee. The effect sizes,
intervals, and planning quantities characterize its conditional contrasts; they
do not certify the per-agent generalized-Bayes claim, which remains separately
source-conditional in [@sec:results-baseline].

![Server-side influence weights assigned by `robust_aggregate`. Source relation: original project server-side diagnostic; estimand: normalized pooling weight; uncertainty: deterministic single-run display. Divergence-reweighting weights for each of the $7$ agents at the verdict contamination rate $0.800$ (the convex-mix strength applied to each saboteur's belief — distinct from the *count* of contaminated agents, $2$ of $7$, reported in the in-figure box). x-axis: agent index, zero-based ($a0$ upward), with each agent's role (honest / adversary) shown beneath its label; y-axis: normalized pooling weight, with weights summing to one and the dotted reference marking the equal-weight pool ($1/n$). The $2$ contaminated agents use cross-hatching and direct role labels; downward arrows mark suppression below the equal-weight reference. No resampling interval applies to this deterministic run. This server-heuristic display establishes neither the per-client bounded-influence theorem, Byzantine tolerance, nor a universal influence bound.](../output/figures/robust_influence_weights.png){#fig:robust-weights data-slide-manifest="../output/figures/robust_influence_weights.slides.json" width=80%}

### Variational aggregator: conservative objective-backed weight control {#sec:results-variational}

The heuristic has the higher point accuracy in the configured verdict cell. The
variational aggregator of [@sec:method-variational], derived in
[@sec:supp-variational], is its objective-backed complement. Each exact block
update does not increase the stated free energy [@eq:agg-free-energy], and a
converged fixed point is coordinatewise stationary. Two deterministic runs show
its weight behavior at robustness 1.50.

First, the descent. On a contaminated colony the free energy falls monotonically
from 3.2458 to 2.3780 (a drop of
0.8678 over 11 block-coordinate
iterations, converged: Yes), with a largest single-step
*increase* of 8.88e-16 — machine zero, the numerical witness
of the descent theorem ([@sec:supp-theorem]).

![Variational free energy $F(q, a)$ as a function of block-coordinate descent iteration. Source relation: original project objective-descent diagnostic; estimand: free energy in nats by iteration; uncertainty: none for the deterministic seeded run. The trace is a single `variational_aggregate` fusion of a $7$-agent contaminated colony (robustness $1.50$). The x-axis is block-coordinate iteration number and the y-axis is $F(q, a)$ in nats. Open triangles joined by a dash-dot path identify the variational trajectory, a filled triangle marks the terminal iterate, enlarged open triangles mark the endpoints of the largest observed descent step, and a dark dotted horizontal rule marks the final stationary level; these shapes and line patterns duplicate color. The curve is monotone non-increasing across all recorded iterations (largest single-step increase: $8.88 \times 10^{-16}$ nats, at machine precision), and the implementation reports converged status Yes at $F=2.3780$ nats. Iterations are ordered states of one deterministic run, not independent replicates, so no resampling interval applies. This trace verifies descent only for the executed path; it does not certify a global optimum, exhaustive basin search, or the separate `robust_aggregate` heuristic.](../output/figures/aggregation_descent.png){#fig:aggregation-descent data-slide-manifest="../output/figures/aggregation_descent.slides.json" width=80%}

Second, the effective-weight response. As one agent is drifted from healthy toward a
confident-wrong delta, its normalized influence falls from
0.143 to below 0.001 — a factor
of 267.1 below the fixed
0.143 the naive log-linear pool grants every agent
regardless of how wrong it is. The gap between the falling variational curve and
the flat naive line is the empirical redescending weight response, drawn.

![Normalized influence weight of one probed agent. Source relation: original project variational-server diagnostic; estimand: the probed agent's normalized server weight as its posterior drifts toward a confident-wrong delta. The x-axis is outlier drift from consensus at zero to the wrong-state delta at one; the y-axis is normalized influence weight. The open-triangle dash-dot path is `variational_aggregate`; the circle-solid horizontal reference is the naive log-linear pool. Direct endpoint labels name both rules and values, shaded separation shows their tested-path gap, and an adversarial X with an arrow marks the first displayed point below one-half of the naive weight. As drift becomes extreme, the variational weight falls below $0.001$, whereas the naive pool remains fixed at $1/n=0.143$. This deterministic seeded sweep contains $n=7$ agents in one configured colony; drift points are ordered evaluations, not independent replications, so no confidence interval or error band applies. The pattern is redescending normalized weight along this path, while the algebraic property bounds raw effective weight; the figure is not an estimator-level B-robustness proof and does not transfer the property to the sharper `robust_aggregate` heuristic.](../output/figures/bounded_influence.png){#fig:bounded-influence data-slide-manifest="../output/figures/bounded_influence.slides.json" width=80%}

The honest trade is conservatism: because $F$ carries the $-H(q)$ entropy term,
its consensus is the maximum-entropy distribution consistent with the weighted
cross-entropies, deliberately flatter than the product-of-experts. The
variational aggregator therefore does *not* win the peak-accuracy verdict of
[@sec:results-verdict] — that remains the sharp heuristic's role — and the two
are reported as complements, never conflated: rigor-with-conservatism on one
side, accuracy-without-an-objective on the other.
