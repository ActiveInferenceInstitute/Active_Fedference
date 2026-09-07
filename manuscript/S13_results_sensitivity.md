## Signed accuracy contrasts over acuity and colony size {#sec:results-sensitivity}

Studies 1–7 fix specific parameter configurations (sensor acuity and colony
size) to isolate mechanistic claims. Study 8 asks how the communicating-minus-
isolated and hierarchical-minus-flat accuracy contrasts vary across a
systematic two-dimensional sweep. This estimand-first formulation makes sign
changes visible without presuming that either architecture is uniformly
advantageous.

We sweep sensor acuity $\kappa \in {{SENS_ACUITY_GRID}}$ and
colony size $n \in {{SENS_COLONY_SIZE_GRID}}$, evaluating two systems:

* **Belief sharing** (Study 1 architecture) — accuracy gap = communicating
  minus isolated mean accuracy;
* **Hierarchical POMDP** (Study 6 architecture) — accuracy gap = hierarchical
  minus flat location accuracy.

Each cell averages $n_{\text{trials}} = {{SENS_N_TRIALS}}$ independent trials to reduce
Monte-Carlo noise at the cell level.

The resulting heatmaps ([@fig:sensitivity-heatmap]) show the accuracy gap for
both systems as a function of acuity and colony size. The signed scale is
centered on zero: brown cells encode positive gaps, blue cells encode negative
gaps, and the neutral midpoint encodes zero. Every cell also prints its signed
value, so the direction remains explicit without color. Hatching identifies
only the declared $|\mathrm{gap}| \le {{SENS_NOISE_FLOOR}}$ display band; it is
not an unreliability flag, confidence interval, significance test, or proof of
zero effect.

Across the grid the following patterns hold:

### Pattern 1: low-to-moderate acuity

**Positive communicating-minus-isolated contrasts occur at low-to-moderate
acuity.** The accuracy gap peaks in the second-lowest acuity row, where
observations carry some signal but no single sentinel resolves the location
alone. It remains positive across the two lowest-acuity rows for colonies of at
least four agents in the declared grid.

Near ceiling acuity the gap approaches zero because isolated agents already
solve most trials in the declared task.

### Pattern 2: a colony-size floor

**Colony size acts through a floor, not a smooth slope.** The two-agent column
shows an exactly zero belief-sharing gap at every acuity. Under self-exclusion
("agents do not hear themselves"), each member hears exactly one incoming
belief, so the heard consensus adds no pooled evidence. Colonies of four or
more show the positive low-acuity contrast within this grid.

### Pattern 3: a conditional hierarchical contrast

**The hierarchical contrast changes with acuity and colony size.** Negative
hierarchical-minus-flat location-accuracy gaps occur across the lower-acuity
rows, including colonies larger than two agents. The higher-acuity rows are
near zero outside the two-agent column; positive cells are confined to that
column in this grid. This conditional pattern does not establish a general
hierarchical accuracy advantage, and the per-study paired location contrast
remains a separate estimate under its own trial and seed budget.

The full parameter grid and protocol details are in the supplement
([@sec:supp-sensitivity]). A native-unit cross-study overview of the headline
metrics across all {{N_STUDIES}} studies is shown in
[@fig:cross-study-summary].

Read the page-compatible composition down the full left column and then down
the right: Panel A contains the six accuracy rows, Panel B at upper right the
two information or free-energy rows, and Panel C at lower right the single
recovery row. Each study appears once, in its applicable native-unit panel.

![Two-panel sensitivity heatmap of belief-sharing and hierarchical accuracy gaps.
Source relation: original project sensitivity diagnostic; estimand: per-cell
accuracy gaps (fractions) as functions of acuity and colony size; uncertainty:
deterministic per-cell means over the declared trials, with no resampling
interval. Two vertically stacked heatmaps of the Study 8 parameter sensitivity sweep.
Upper panel: y-axis indexes sensor acuity ({{SENS_ACUITY_MIN}}–{{SENS_ACUITY_MAX}}, {{SENS_N_ACUITY_LEVELS}} levels); x-axis
indexes colony size ({{SENS_COLONY_SIZE_MIN}}–{{SENS_COLONY_SIZE_MAX}} agents, {{SENS_N_COLONY_SIZES}} levels); color encodes the
belief-sharing accuracy gap (communicating minus isolated mean accuracy).
Lower panel: identical axes; color encodes the hierarchical POMDP location
accuracy gap (hierarchical minus flat).
The zero-centered blue–neutral–brown scale encodes negative, zero, and positive
gaps, respectively; printed signed values provide the non-color encoding.
Diagonal hatching marks the declared $|\mathrm{gap}| \le {{SENS_NOISE_FLOOR}}$
display band. The hatch is not an unreliability designation, confidence
interval, significance test, or proof of zero effect. Cell values are
deterministic per-cell means over {{SENS_N_TRIALS}} trials; trials are nested
within their configured cell, and no resampling interval is shown. The sweep
protocol is detailed in the sensitivity supplement, and the finite grid does
not establish a general communication or hierarchy effect outside the declared
acuity and colony-size settings.
](../output/figures/sensitivity_heatmap.png){#fig:sensitivity-heatmap width=90% data-slide-manifest="../output/figures/sensitivity_heatmap.slides.json"}

![Native-unit cross-study summary of the headline signed contrasts. Source
relation: original-project synthesis from a separate harmonized seed-level
rerun; status: conditional overview, not a pooled meta-analysis. Panel A, read
down the left column, shows six signed accuracy contrasts in fractions. Panel B
at upper right shows two information or free-energy contrasts in nats. Panel C
at lower right shows one parameter-recovery fit in unitless $R^2$. Each x-axis
reports the signed study-level mean in its panel's native unit; each y-axis lists
only that panel's studies. Horizontal bars are means, whiskers are
{{CI_PERCENT}}% percentile-bootstrap intervals, and direct signed labels give
the means. Forward, cross, and dotted hatches distinguish positive, negative,
and near-zero estimands; the dark dotted rule marks zero without color. The
independent and resampling unit is the seed
($n={{CROSS_STUDY_N_SEEDS}}$) within each configured study, with observations,
agents, trials, and ordered steps nested. These intervals belong only to the
harmonized rerun, including rows whose primary figure is a deterministic
single-posterior diagnostic; they do not retrofit uncertainty to those primary
figures. The Study 4 row is the within-run display-selected maximum across
non-reference server presets at the declared worst rate, neither a preselected method nor an inferential winner.
Native units are preserved: the figure neither
ranks nor pools studies nor implies a shared effect or universal benefit.
](../output/figures/cross_study_summary.png){#fig:cross-study-summary width=95% data-slide-manifest="../output/figures/cross_study_summary.slides.json"}
