## Parameter sensitivity of federation benefit {#sec:results-sensitivity}

Studies 1–7 fix specific parameter configurations (sensor acuity, colony size)
to isolate mechanistic claims. A natural question is whether the federation
benefit is **robust to those choices** or is an artifact of a narrow operating
point. Study 8 addresses this with a systematic 2-D sensitivity sweep over
sensor acuity and colony size.

We sweep sensor acuity $\kappa \in {{SENS_ACUITY_GRID}}$ and
colony size $n \in {{SENS_COLONY_SIZE_GRID}}$, evaluating two systems:

* **Belief sharing** (Study 1 architecture) — accuracy gap = communicating
  minus isolated mean accuracy;
* **Hierarchical POMDP** (Study 6 architecture) — accuracy gap = hierarchical
  minus flat location accuracy.

Each cell averages $n_{\text{trials}} = {{SENS_N_TRIALS}}$ independent trials to reduce
Monte-Carlo noise at the cell level.

The resulting heatmaps ([@fig:sensitivity-heatmap]) show the accuracy gap for
both systems as a function of acuity and colony size. Green cells indicate that
federation benefits the colony; red cells indicate that the chosen configuration
yields no benefit or a slight deficit. The symmetric RdYlGn colormap is centered
on zero so the sign of the benefit is immediately legible.

Across the grid the following patterns hold:

1. **The belief-sharing benefit lives at low-to-moderate acuity.** The
   accuracy gap peaks in the second-lowest acuity row — where individual
   observations carry some signal but no single sentinel resolves the location
   alone, so pooled evidence pays most — and remains uniformly positive across
   the two lowest-acuity rows for colonies of at least four agents. Near
   ceiling acuity the gap shrinks toward zero: single agents already solve the
   task, so federation has nothing left to add.

2. **Colony size acts through a floor, not a smooth slope.** The two-agent
   column shows an exactly zero belief-sharing gap at every acuity: under
   self-exclusion ("agents do not hear themselves"), each member of a two-agent
   colony hears exactly one incoming belief, so the heard consensus adds no
   pooled evidence. Colonies of four or more realize the low-acuity benefit.

3. **The hierarchical gap is near zero across most of the grid.** The
   hierarchical-minus-flat location-accuracy gap is approximately zero over
   most cells, with a few strongly negative low-acuity cells and small positive
   cells confined to the two-agent column — consistent with the per-study
   finding that the hierarchical architecture matches rather than beats the
   flat baseline on location accuracy.

The full parameter grid and protocol details are in the supplement
([@sec:supp-sensitivity]). A native-unit cross-study overview of the headline
metrics across all {{N_STUDIES}} studies is shown in
[@fig:cross-study-summary].

![Source relation: original project sensitivity diagnostic; estimand: per-cell
accuracy gaps (fractions) as functions of acuity and colony size; uncertainty:
deterministic per-cell means over the declared trials, with no resampling
interval. Two-panel heatmap (1×2) of the Study 8 parameter sensitivity sweep.
Left panel: y-axis indexes sensor acuity ({{SENS_ACUITY_MIN}}–{{SENS_ACUITY_MAX}}, {{SENS_N_ACUITY_LEVELS}} levels); x-axis
indexes colony size ({{SENS_COLONY_SIZE_MIN}}–{{SENS_COLONY_SIZE_MAX}} agents, {{SENS_N_COLONY_SIZES}} levels); color encodes the
belief-sharing accuracy gap (communicating minus isolated mean accuracy).
Right panel: identical axes; color encodes the hierarchical POMDP location
accuracy gap (hierarchical minus flat).
Color scale: RdYlGn symmetric around zero — green denotes federation benefit,
red denotes deficit. Diagonal hatching marks cells with $|\mathrm{gap}| \le {{SENS_NOISE_FLOOR}}$
(unreliable, near-zero benefit). Cell values are deterministic per-cell means
over {{SENS_N_TRIALS}} trials; no resampling error band is shown — the sweep
protocol is detailed in the sensitivity supplement.
](../output/figures/sensitivity_heatmap.png){#fig:sensitivity-heatmap width=90%}

![Source relation: original project cross-study summary; estimand: grouped
study-level means in native units (accuracy fractions, nats, or $R^2$), never a
cross-unit ranking; uncertainty: seed-level bootstrap confidence intervals.
Horizontal native-unit facet chart summarizing the key federation benefit metric for each
of the {{N_STUDIES}} studies (Studies 1–{{N_STUDIES}}). x-axis indexes benefit value in
metric-specific units (accuracy gain for Studies 1, 4, 5, 6, 7, 8; KL reduction
for Study 2; ΔF for Study 3; $R^2$ for Study 9). y-axis lists the {{N_STUDIES}} studies (one row each),
ordered from Study 1 at the top to Study 9 at the bottom within each native-unit
facet. Each mark shows the mean over {{CROSS_STUDY_N_SEEDS}} independent seeds
with intervals spanning the {{CI_PERCENT}} % bootstrap confidence interval.
There is no cross-unit ranking: zero is a within-unit reference only, and the
facet labels carry the units. Consistent with the per-study results, Studies 5
(moving world, EFE) and 6 (2-level hierarchical) sit at approximately zero
within their respective units.
](../output/figures/cross_study_summary.png){#fig:cross-study-summary width=75%}
