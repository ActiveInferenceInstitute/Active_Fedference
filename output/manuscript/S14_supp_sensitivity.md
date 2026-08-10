## Supplement: parameter-sensitivity methods {#sec:supp-sensitivity}

This supplement documents how the sensitivity grid of [@sec:results-sensitivity]
is generated and — importantly for a reader trying to reconcile numbers across
studies — where its two sweeps use *different* seeding and trial budgets. It
answers two questions the main section leaves implicit: exactly which seed drives
each cell (so any cell is independently reproducible), and why the cross-study
summary's sensitivity row is not directly comparable, at matched trial counts, to
the standalone heatmap.

### Experimental protocol for grid sensitivity

The sensitivity sweep is implemented in
`fedference.experiments.run_belief_sharing_sensitivity` and
`fedference.experiments.run_hierarchical_sensitivity`. Each function accepts a
tuple of acuity values and a tuple of colony sizes. In the **belief-sharing**
sweep every (acuity, colony-size) cell averages $n_{\text{trials}}$ independent
trials, each seeded via a deterministic formula:

$$
\text{seed}_{\text{cell}} = \text{seed}_{\text{base}} + i \cdot 10^5 + j \cdot 10^3 + t
$$ {#eq:sensitivity-seed}

The deterministic seed rule [@eq:sensitivity-seed] makes every grid cell and trial independently
reproducible from the base seed.

where $i$ indexes acuity, $j$ indexes colony size, and $t$ indexes the trial
within a cell. For the belief-sharing sweep this guarantees that:

1. no two cells share a trial seed (no correlation between cells);
2. re-running with the same `seed_base` is bit-identical (reproducibility);
3. different `seed_base` values produce independent replicates (robustness
   checking).

The **hierarchical** sweep uses a simpler protocol: every cell calls
`run_hierarchical_world` once with the same base seed (its internal trials are
seeded by that run), so hierarchical cells share the base seed rather than the
per-cell formula above.

### Grid parameters for acuity and colony size

| Parameter | Values |
|-----------|--------|
| Sensor acuity $\kappa$ | \{0.40, 0.55, 0.70, 0.85, 0.95\} |
| Colony size $n$ | \{2, 4, 6, 8, 10\} |
| Trials per cell | 20 |
| Base seed | 0 |

The 5×5 = 25 cells per system are run with `seed_base = 0` by default;
`generate_sensitivity_heatmap` accepts a `seed` argument to override this.

### Belief-sharing condition in the sensitivity grid

Each trial in the belief-sharing sweep:

1. Draws a random true state and one noisy observation per agent
   (same protocol as `run_belief_sharing`).
2. Runs one belief-sharing round with `communicate=True` (communicating) and
   `communicate=False` (isolated).
3. Records `mean_accuracy` for each condition.
4. The cell value is the average over `n_trials` of this gap.

### Hierarchical POMDP condition in the sensitivity grid

Each cell in the hierarchical sweep calls `run_hierarchical_world` once with
the cell's acuity and colony size, passing the constant base seed (not the
per-cell formula, which applies only to the belief-sharing sweep). The
returned `location_accuracy_gap` (hierarchical minus flat) becomes the cell
value.

### Figure rendering for sensitivity summaries

`generate_sensitivity_heatmap` assembles the two grids into a 1×2 matplotlib
`imshow` figure with RdYlGn colormap, symmetric bounds at
$\pm\max(|\text{gap}|)$, per-cell numeric annotations, and a per-panel colorbar
labeled "Accuracy gap (hierarchical/comm. − baseline)". The figure is written to
`output/figures/sensitivity_heatmap.png`.

### Cross-study summary construction

`generate_cross_study_summary` runs a 128-seed ($n_{\text{seeds}} = 128$) ensemble
over Studies 1--9 and reports the mean ± 95 % bootstrap CI of the key
federation-benefit metric for each study. The metric definitions are:

The robustness row uses 40 matched trials per seed and rate;
the trial-level observations are reduced within seed before the cross-study
summary is formed. This preserves the seed as the independent Monte Carlo unit.

The Study 8 row below uses 3 trials per cell —
smaller than the full-resolution 20-trial `Trials per cell` grid
documented above for the standalone sensitivity heatmap figure, a deliberate
runtime budget for the per-seed cross-study loop rather than an oversight — so
the two are not directly comparable at matched trial counts.

| Study | Metric |
|-------|--------|
| 1 — Belief sharing | Accuracy gain: communicating − isolated |
| 2 — Language acquisition | KL reduction: initial − final |
| 3 — Emergence (BMR) | $\Delta F$ for redundant pruning |
| 4 — Robustness sweep | Accuracy gain: pooled display robust method − naive at worst contamination rate |
| 5 — Moving world (EFE) | Accuracy gain: EFE-guided − isolated |
| 6 — Hierarchical POMDP (2-level) | Location accuracy gap: hierarchical − flat |
| 7 — 3-level POMDP | Location accuracy gap: 3-level − flat |
| 8 — Parameter sensitivity | Mean accuracy gap across the sensitivity grid |
| 9 — Parameter recovery | $R^2$ for acuity identifiability |

Bootstrap CIs use 5000 resamples (default `n_boot` in `fedference.statistics.bootstrap_ci`).
