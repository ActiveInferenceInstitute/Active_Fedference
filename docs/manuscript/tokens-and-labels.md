# Tokens and labels

Active Fedference uses two registries. **Do not duplicate full tables in
`docs/`** — edit the authoritative sources.

## `{{TOKEN}}` protocol (numbers in prose)

| Source | Role |
| --- | --- |
| [`src/manuscript_variables.py`](../../src/manuscript_variables.py) | Public entry point (re-export shim): `generate_variables()` builds the mapping |
| [`src/manuscript_vars/`](../../src/manuscript_vars/generate.py) | Implementation package: `generate.py`, `tokens.py`, `loaders.py`, `render.py` |
| [`../../manuscript/AGENTS.md`](../../manuscript/AGENTS.md) | Authoring workflow and token groups |
| `output/data/manuscript_variables.json` | Generated mapping (disposable) |

### Token groups (summary)

| Group | Examples | Origin |
| --- | --- | --- |
| Provenance | `ISC_TOTAL`, `ISC_PASSED`, `TEST_COUNT`, `COVERAGE_PERCENT`, environment versions, `CONFIG_HASH` | ISC values are parsed from `ISA.md`; final test/coverage and environment values come from the fresh successful validation receipt |
| Config | `EXPERIMENT_SEED`, `N_AGENTS`, `N_SEEDS` | `manuscript/config.yaml` |
| Belief sharing | `BELIEF_SHARING_MEAN_F_COMMUNICATE`, `BELIEF_SHARING_FREE_ENERGY_GAP` | `output/reports/belief_sharing.json` |
| Language | `LANGUAGE_FINAL_KL`, `LANGUAGE_MONOTONE` | `output/reports/language_acquisition.json` |
| Emergence | `EMERGENCE_DELTA_F_REDUNDANT`, `EMERGENCE_DELTA_F_SUPPORTED` | `output/reports/emergence.json` |
| Robustness sweep | `SWEEP_BEST_QVALUE`, `SWEEP_ANY_ROBUST_WINS`, `SWEEP_HEADLINE_POWER`, per-rate p/q tokens | `output/reports/robustness_sweep.json` |
| Recovery residuals | `RECOVERY_*` (each with a `*_MATH` sibling — see below) | Deterministic checks in `src/manuscript_vars/` |
| Variational diagnostics | `VARIATIONAL_F_INITIAL`, `VARIATIONAL_DELTA_F`, `VARIATIONAL_INFLUENCE_DROP_FACTOR`, `VARIATIONAL_CAPTURE_GAP` | `output/reports/variational_aggregation.json` |
| Contamination gallery | `GALLERY_RATE`, `GALLERY_RELIABLE_KINDS`, `GALLERY_TABLE_ROWS` | `output/reports/contamination_gallery.json` |
| Robustness onset | `ONSET_WIN_FRACTION`, `ONSET_TABLE_ROWS` | `output/reports/robustness_onset.json` |
| Tempered aggregation (V1) | `TEMPERED_LAMBDA_STAR`, `TEMPERED_HONEST_EXIT_SENTENCE`, `TEMPERED_ENTROPY_WEIGHT_DEFAULT` | Computed at token-generation time |
| Federation transport (V3) | `FEDERATION_N_WORKERS`, `FEDERATION_BIT_IDENTICAL`, `FEDERATION_TRANSPORT` | Compile-time constants |
| Moving sentinel world (V4) | `MOVING_ACC_COMMUNICATING`, `MOVING_FE_GAP_COMMUNICATING`, `MOVING_ACC_EFE`, `MOVING_ACC_ISOLATED`, `MOVING_N_TRIALS`, `MOVING_N_STEPS`, `MOVING_N_POSITIONS`, `MOVING_N_AGENTS`; **multi-seed CI**: `MOVING_N_SEEDS`, `MOVING_ACC_EFE_MEAN`, `MOVING_ACC_EFE_CI_LO`, `MOVING_ACC_EFE_CI_HI`, `MOVING_ACC_ISO_MEAN`, `MOVING_ACC_ISO_CI_LO`, `MOVING_ACC_ISO_CI_HI`, `MOVING_ACC_COMM_MEAN`, `MOVING_ACC_COMM_CI_LO`, `MOVING_ACC_COMM_CI_HI`, `MOVING_FE_GAP_EFE_MEAN`, `MOVING_FE_GAP_EFE_CI_LO`, `MOVING_FE_GAP_EFE_CI_HI`, `MOVING_WILCOX_PVALUE`, `MOVING_EFFECT_SIZE`, `MOVING_EFFECT_LABEL` | `output/reports/moving_world.json` (point estimates) + `_moving_world_variables()` (CI) |
| Hierarchical POMDP (V2, Study 6) | `HIER_LOC_ACC_FLAT`, `HIER_LOC_ACC_HIER`, `HIER_LOC_ACC_GAP`, `HIER_CTX_ACC`, `HIER_N_LOCATIONS`, `HIER_N_CONTEXTS`, `HIER_N_AGENTS`, `HIER_N_TRIALS`, `HIER_ACUITY`, `HIER_N_ITERS`, `HIER_SEED`, `HIER_ALERT_CENTER_MASS`, `HIER_CTX_PERSIST`; **multi-seed CI**: `HIER_N_SEEDS`, `HIER_LOC_ACC_HIER_MEAN`, `HIER_LOC_ACC_HIER_CI_LO`, `HIER_LOC_ACC_HIER_CI_HI`, `HIER_LOC_ACC_FLAT_MEAN`, `HIER_LOC_ACC_FLAT_CI_LO`, `HIER_LOC_ACC_FLAT_CI_HI`, `HIER_LOC_ACC_GAP_MEAN`, `HIER_LOC_ACC_GAP_CI_LO`, `HIER_LOC_ACC_GAP_CI_HI`, `HIER_WILCOX_PVALUE`, `HIER_EFFECT_SIZE`, `HIER_EFFECT_LABEL` | Strictly loaded from `output/reports/hierarchical_world.json` |
| 3-level POMDP (V2, Study 7) | `NLEVEL3_LOC_ACC_FLAT`, `NLEVEL3_LOC_ACC_3LEVEL`, `NLEVEL3_LOC_ACC_GAP`, `NLEVEL3_CTX_ACC`, `NLEVEL3_META_CTX_ACC`, `NLEVEL3_N_LEVELS`, `NLEVEL3_N_LOCATIONS`, `NLEVEL3_N_CONTEXTS`, `NLEVEL3_N_META_CONTEXTS`, `NLEVEL3_N_AGENTS`, `NLEVEL3_N_TRIALS`, `NLEVEL3_ACUITY`, `NLEVEL3_N_ITERS`, `NLEVEL3_SEED`; **multi-seed CI**: `NLEVEL3_N_SEEDS`, `NLEVEL3_LOC_ACC_3LEVEL_MEAN`, `NLEVEL3_LOC_ACC_3LEVEL_CI_LO`, `NLEVEL3_LOC_ACC_3LEVEL_CI_HI`, `NLEVEL3_LOC_ACC_FLAT_MEAN`, `NLEVEL3_LOC_ACC_FLAT_CI_LO`, `NLEVEL3_LOC_ACC_FLAT_CI_HI`, `NLEVEL3_LOC_ACC_GAP_MEAN`, `NLEVEL3_LOC_ACC_GAP_CI_LO`, `NLEVEL3_LOC_ACC_GAP_CI_HI`, `NLEVEL3_WILCOX_PVALUE`, `NLEVEL3_EFFECT_SIZE`, `NLEVEL3_EFFECT_LABEL` | Strictly loaded from `output/reports/nlevel3_world.json` |
| Parameter recovery | `PARAM_RECOVERY_MEAN_ABS_ERROR`, `PARAM_RECOVERY_R_SQUARED`, `PARAM_RECOVERY_N_TRIALS`, `PARAM_RECOVERY_N_OBSERVATIONS`, `PARAM_RECOVERY_INTERVAL_PERCENT`, `PARAM_RECOVERY_ACUITY_GRID` | `output/reports/parameter_recovery.json` |
| Computational complexity | `COMPLEXITY_*` symbolic orders, benchmark grids, machine metadata, and observed log-log slopes | `output/reports/complexity_scaling.json` |
| Sensitivity sweep config | `SENS_N_TRIALS`, `SENS_SEED_BASE`, `SENS_N_ACUITY_LEVELS`, `SENS_N_COLONY_SIZES`, `SENS_N_CELLS`, `SENS_NOISE_FLOOR`, `BOOTSTRAP_N_BOOT` | Shared constants in `src/experiment_config.py` and the token loader; the noise floor is a visualization convention, not a significance threshold |

The robustness verdict tokens (`SWEEP_*`) reflect **computed** Wilcoxon + BH-FDR
results — never hand-authored (ISC-30). Power tokens are observed-effect
design-planning quantities for the server-side heuristic contrast; they do not
certify the per-client beta/rcce FedGVI guarantee.

The variational tokens (`VARIATIONAL_*`) report objective descent and the
redescending-weight diagnostic; `VARIATIONAL_CAPTURE_GAP` is the headline multi-start
vs single-start descent-comparison number. The `GALLERY_*` tokens carry the
seed-robust contamination-gallery verdict: `GALLERY_RELIABLE_KINDS` names only the
attack mechanisms with a statistically reliable robust advantage. The `ONSET_*`
tokens surface the per-mechanism onset rate (smallest contamination rate at which
robust reliably wins).

The `HIER_*` and `NLEVEL3_*` tokens are strict loads from
`output/reports/hierarchical_world.json` and `output/reports/nlevel3_world.json`
(Phase 1 outputs); hydration raises `FileNotFoundError` if either report is
missing rather than rerunning the study or synthesizing values.

### `*_MATH` sibling tokens (scientific notation in math contexts)

Small magnitudes are emitted twice. The plain token carries the `.2e` string
(for example `2.33e-80`) for prose and tables; a `*_MATH` sibling carries the
same value as LaTeX scientific notation (`M \times 10^{E}`, mantissa to two
decimals, exponent normalized to a plain int) for use inside `$...$` spans,
where the bare `e`-form would typeset as a malformed product.
`_format_residual_math()` in
[`src/manuscript_vars/loaders.py`](../../src/manuscript_vars/loaders.py)
implements the conversion; non-positive inputs render as `0`.

Sibling-emitting groups: every `RECOVERY_*` residual (each key gets a
`{KEY}_MATH` twin), `RECOVERY_OFFSWITCH_Q_MATH`, `VARIATIONAL_MAX_ASCENT_MATH`,
`SWEEP_BEST_QVALUE_MATH`, and `SWEEP_BEST_RAW_PVALUE_MATH`. Rule: inside math
mode use the `_MATH` sibling; outside math mode use the plain token. Never
hand-format scientific notation in manuscript prose.

### Hydration command

```bash
# A pre-test rendered-input pass, only when needed by the full suite.
uv run python scripts/z_generate_manuscript_variables.py --provisional-validation
uv run --extra dev python scripts/validate_test_coverage.py
# Final non-draft hydration reads the receipt and fails before writing if stale.
uv run python scripts/z_generate_manuscript_variables.py
```

`output/data/test_coverage_receipt.json` records the successful command,
collected/passed/failed/skipped counts, achieved and required coverage,
environment, source-owned documentation and release metadata, manuscript-source/ISC and dependency-lock digests, and the fresh
analysis-stage input/output digests. It records matching pre- and post-suite
snapshots and refuses to issue a receipt if any of those bound inputs drift
while the full suite is running. It is a direct hydration input rather than a pipeline dependency, so
the pre-test render does not create a dependency cycle. `--allow-draft` is for
incomplete draft fixtures; `--provisional-validation` is the explicit
non-draft pre-test escape hatch and must not be used for the final manuscript.

### Unresolved token check

```bash
if rg -n '\{\{[A-Z][A-Z0-9_]*\}\}' output/manuscript/; then
  echo UNRESOLVED
  exit 1
else
  echo OK
fi
```

## Cross-reference labels (equations, figures, sections)

| Source | Role |
| --- | --- |
| [`../../manuscript/SYNTAX.md`](../../manuscript/SYNTAX.md) | Canonical `{#eq:}`, `{#fig:}`, `{#tbl:}`, `{#sec:}`, `{#prop:}`, `{#thm:}`, `{#lem:}`, `{#cor:}`, `{#def:}` registry |
| Manuscript `.md` files | Use `[@fig:label]` in prose; `{#fig:label}` on figure lines |

Adding a figure:

1. Implement generator in `src/figures/`.
2. Register label in `SYNTAX.md`.
3. Reference in the appropriate results section with a full caption.
4. Add any new numeric claims as tokens in `manuscript_variables.py` first.

## Repository-wide semantics

Shared Pandoc/LaTeX rules for this standalone repository live in the local
manuscript guide and registry:

- [`../../manuscript/AGENTS.md`](../../manuscript/AGENTS.md) — editing workflow,
  token protocol, figure protocol, and theorem conventions.
- [`../../manuscript/SYNTAX.md`](../../manuscript/SYNTAX.md) — canonical label
  strings and project-specific syntax constraints.

`SYNTAX.md` takes precedence for label strings.

## See also

- [`rendering_pipeline.md`](rendering_pipeline.md)
- [`../core/experiments-and-artifacts.md`](../core/experiments-and-artifacts.md)
