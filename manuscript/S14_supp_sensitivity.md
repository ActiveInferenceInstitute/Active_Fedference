## Supplement: parameter-sensitivity methods {#sec:supp-sensitivity}

This supplement documents how the sensitivity grid of [@sec:results-sensitivity]
is generated and — importantly for a reader trying to reconcile numbers across
studies — where its two sweeps use *different* seeding and trial budgets. It
answers two questions the main section leaves implicit: exactly which seed
drives each cell, so every cell is deterministically reproducible in isolation;
and why the cross-study summary's sensitivity row is not directly comparable,
at matched trial counts, to the standalone heatmap.

### Experimental protocol for grid sensitivity

The sensitivity sweep uses `run_belief_sharing_sensitivity` and
`run_hierarchical_sensitivity` from the `fedference.experiments` module. Each
function accepts a tuple of acuity values and a tuple of colony sizes. In the **belief-sharing**
sweep every (acuity, colony-size) cell averages $n_{\text{trials}}$ independent
trials, each seeded via a deterministic formula:

$$
\text{seed}_{\text{cell}} = \text{seed}_{\text{base}} + i \cdot 10^5 + j \cdot 10^3 + t
$$ {#eq:sensitivity-seed}

where $i$ indexes acuity, $j$ indexes colony size, and $t$ indexes the trial
within a cell. The deterministic rule [@eq:sensitivity-seed] makes every grid cell and trial
reproducible in isolation from the base seed. For the belief-sharing sweep, no
two cells share a trial seed, so the design introduces no seed reuse between
cells.

Re-running with the same `seed_base` is bit-identical. Changing `seed_base`
produces an independent replicate for robustness checking.

The **hierarchical** sweep uses a simpler protocol: every cell calls
`run_hierarchical_world` once with the same base seed (its internal trials are
seeded by that run), so hierarchical cells share the base seed rather than the
per-cell formula above.

### Grid parameters for acuity and colony size

| Parameter | Values |
|-----------|--------|
| Sensor acuity $\kappa$ | {{SENS_ACUITY_GRID}} |
| Colony size $n$ | {{SENS_COLONY_SIZE_GRID}} |
| Trials per cell | {{SENS_N_TRIALS}} |
| Base seed | {{SENS_SEED_BASE}} |

The {{SENS_N_ACUITY_LEVELS}}×{{SENS_N_COLONY_SIZES}} = {{SENS_N_CELLS}} cells per system are run with `seed_base = 0` by default;
`generate_sensitivity_heatmap` accepts a `seed` argument to override this.

### Belief-sharing condition in the sensitivity grid

Each trial follows four labeled operations. **Draw:** sample one true state and
one noisy observation per agent using the `run_belief_sharing` protocol.
**Compare:** run one belief-sharing round with `communicate=True` and the
matched isolated condition with `communicate=False`. **Record:** retain
`mean_accuracy` for both conditions. **Reduce:** define the cell value as the
mean communicating-minus-isolated gap across `n_trials`.

### Hierarchical POMDP condition in the sensitivity grid

Each cell in the hierarchical sweep calls `run_hierarchical_world` once with
the cell's acuity and colony size, passing the constant base seed (not the
per-cell formula, which applies only to the belief-sharing sweep). The
returned `location_accuracy_gap` (hierarchical minus flat) becomes the cell
value.

### Figure rendering for sensitivity summaries

`generate_sensitivity_heatmap` assembles the two grids into a 1×2 matplotlib
`imshow` figure with a color-vision-deficiency-safer blue–neutral–brown signed
palette and symmetric bounds at
$\pm\max(|\text{gap}|)$, per-cell numeric annotations, and a per-panel colorbar
labeled "Accuracy gap (hierarchical/comm. − baseline)". The figure is written
as `sensitivity_heatmap.png` under `output/figures/`. Negative gaps map toward
blue, positive gaps toward brown, and zero to the neutral midpoint; exact signed cell
labels duplicate that encoding. Hatching marks only the declared
$|\mathrm{gap}| \le {{SENS_NOISE_FLOOR}}$ display band. It is not an
unreliability flag, confidence interval, significance test, or proof of zero
effect.

### Cross-study summary construction

`generate_cross_study_summary` performs a separate harmonized
{{CROSS_STUDY_N_SEEDS}}-seed
($n_{\text{seeds}} = {{CROSS_STUDY_N_SEEDS}}$) rerun over Studies 1--9 and
reports the mean with a {{CI_PERCENT}}% seed-level percentile-bootstrap
interval for each headline signed contrast or estimand. These intervals belong
to that rerun, including for rows whose primary figure is a deterministic
single-posterior or fixed-configuration diagnostic; they are not intervals on
the primary deterministic display. The definitions are:

The robustness row uses {{CROSS_STUDY_N_TRIALS}} matched trials per seed and rate;
the trial-level observations are reduced within seed before the cross-study
summary is formed. Within each seed, the row then takes the display-selected
maximum non-reference server-preset contrast at the declared worst rate. This
preserves the seed as the independent Monte Carlo unit but makes Study 4 a
within-run display-selected summary, not a preselected method or inferential
winner.

The Study 8 row below uses {{CROSS_STUDY_SENS_N_TRIALS}} trials per cell —
smaller than the full-resolution {{SENS_N_TRIALS}}-trial `Trials per cell` grid
documented above for the standalone sensitivity heatmap figure, a deliberate
runtime budget for the per-seed cross-study loop rather than an oversight — so
the two are not directly comparable at matched trial counts.

| Study | Metric |
|-------|--------|
| 1 — Belief sharing | Accuracy contrast: communicating − isolated |
| 2 — Language acquisition | KL reduction: initial − final |
| 3 — Emergence (BMR) | $\Delta F$ for redundant pruning |
| 4 — Robustness sweep | Accuracy contrast: within-run display-selected maximum server-preset contrast − naive at worst contamination rate |
| 5 — Moving world (EFE) | Accuracy contrast: EFE-guided − isolated |
| 6 — Hierarchical POMDP (2-level) | Location accuracy gap: hierarchical − flat |
| 7 — 3-level POMDP | Location accuracy gap: 3-level − flat |
| 8 — Parameter sensitivity | Mean accuracy gap across the sensitivity grid |
| 9 — Parameter recovery | $R^2$ for acuity identifiability |

Bootstrap CIs use {{BOOTSTRAP_N_BOOT}} resamples. Their default `n_boot` is
defined by `bootstrap_ci` in the `fedference.statistics` module.
