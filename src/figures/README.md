# `src/figures/` — Active Fedference figures

Figure builders for the manuscript-backed outputs.

This package contains importable Matplotlib figure workflows used by scripts and
its tests. Keep figure-specific styling and data preparation here; core inference
and statistics remain in `src/fedference/`.

Manuscript figures use PNG as the embedded surface. Generators that call
`save_figure` / `save_figure_pair` also emit a byte-stable sibling PDF for
archival/vector use. All generators render through the shared `apply_style()`
rcParams and honor the legibility floors in `_common.py`
(`MIN_QUANTITATIVE_FONT_SIZE` 9.5 pt, `MIN_SCHEMATIC_FONT_SIZE` 8.5 pt) at
`FIGURE_EXPORT_DPI` (220). A caption must identify whether a figure is formal,
mechanistic, deterministic, or data-bearing; data-bearing captions name the
estimand, replication/resampling unit, and uncertainty disposition. Those same
provenance fields are declared per generator in `_metadata.py`
(`FIGURE_METADATA`), including a concise `alt_text` for every generator, and
the figure registry, captions, and manuscript embed must agree on the exact
filename. Figures use color-independent cues for meaningful group distinctions,
and captions/HTML alternatives must remain
useful without color or visual inspection; see the
[publication accessibility contract](../../docs/manuscript/accessibility.md).

## Files

| File | Role |
| --- | --- |
| `__init__.py` | Public figure-builder exports. |
| `_common.py` | Shared plotting helpers, palette constants, font-size floors, and figure savers. |
| `_metadata.py` | Data-only provenance registry (`FIGURE_METADATA`): status, source relation, estimand, unit, uncertainty, replication unit, and concise `alt_text` per generator. |
| `aggregation_descent.py` | Variational aggregation descent figure. |
| `bounded_influence.py` | Legacy-named variational redescending-weight diagnostic; not an estimator-level B-robustness figure. |
| `belief_heatmap.py` | Belief-ensemble consensus visualisation. |
| `belief_quality.py` | Proper-score and calibration sensitivity summary. |
| `bnn_robustness.py` | FedGVI logistic-regression robustness curve with seed-level intervals. |
| `contamination_gallery.py` | Adversarial attack regime gallery. |
| `complexity_scaling.py` | Declared dense orders and machine-scoped measured scaling. |
| `conditional_world.py` | Conditional-shift generalization and sensitivity diagnostics. |
| `cross_study_summary.py` | Study-level benchmark summary figure. |
| `descent_comparison.py` | Multi-start / single-start comparison. |
| `disjoint_fov_world.py` | Disjoint field-of-view communication scenario. |
| `efe_decomposition.py` | EFE decomposition plot. |
| `emergence_bmr.py` | Model-reduction emergence figure. |
| `free_energy_comparison.py` | Communication versus isolation comparisons. |
| `graphical_abstract.py` | Manuscript cover schematic. |
| `generative_model_schema.py` | Formal temporal, hierarchical, and factorial categorical-model schematic. |
| `heuristic_breakdown.py` | `robust_aggregate` influence-weight characterization (recovery-limit diagnostic). |
| `hierarchical_bmr.py` | Hierarchical Bayesian-model-reduction per-level surprise figure. |
| `hierarchical_pomdp.py` | Hierarchical world dynamics figures. |
| `language_kl_decay.py` | Language acquisition KL learning curves. |
| `moving_world.py` | Static world / moving world comparison. |
| `message_passing.py` | Belief-sharing message path and three-axis claim-ownership schematic. |
| `parameter_recovery.py` | Parameter recovery scatter/accuracy figure. |
| `pomdp_loop.py` | Hidden-state, observation, action, and federation-loop schematic. |
| `robust_influence_weights.py` | Server-side heuristic reweighting diagnostics. |
| `robustness_onset.py` | Accuracy onset versus contamination mechanism. |
| `robustness_sweep.py` | Contamination sweep figure. |
| `sensitivity_heatmap.py` | 2D sensitivity heatmap (acuity x colony size). |
| `system_overview.py` | High-level architecture overview figure. |

## See also

- [`../README.md`](../README.md) — project source overview
- [`../AGENTS.md`](../AGENTS.md) — source-layer editing rules
