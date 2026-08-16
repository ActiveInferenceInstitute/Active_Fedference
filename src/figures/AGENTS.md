# `src/figures/` - figure-generation agent guide

## Purpose

Importable figure-generation package for Active Fedference.

## Rules

- Keep scripts as thin wrappers; figure assembly belongs in this package.
- Do not move core domain math into figure modules.
- Use real project outputs or typed config inputs in tests; avoid mocks.
- Write generated files only from explicit orchestration paths.
- Import shared style, palette, paths, and savers from `_common.py`; do not
  re-declare colors or rcParams per module.
- Encode method/status differences with line style, marker, hatch, direct
  label, or panel structure as well as color. The manuscript caption supplies
  the current HTML alternative/long description and must remain intelligible
  without the image; follow the
  [publication accessibility contract](../../docs/manuscript/accessibility.md).
- Inspect every changed figure at native PNG scale and in its final PDF/HTML
  placement. Check title/axis/legend collisions, clipped annotations, grayscale
  distinction, near-zero/signed scales, and whether intervals visibly match the
  caption's declared replication unit. The metadata registry also supplies a
  concise `alt_text` for the tagged PDF and non-visual readers; it must describe
  the encodings and claim boundary without merely repeating the title. A
  passing generator test is not a
  substitute for final-context visual inspection.

## Legibility floors

Every generator draws through the `apply_style()` rcParams in `_common.py`, and
no rendered text may fall below the shared floors declared there:
`MIN_QUANTITATIVE_FONT_SIZE` (9.5 pt) for data-bearing labels and
`MIN_SCHEMATIC_FONT_SIZE` (8.5 pt) for schematic node labels, both exported so
`tests/figures/test_palette.py` can gate drift. Figures export at
`FIGURE_EXPORT_DPI` (220). Save through `save_figure` / `save_figure_pair`, which
emit a deterministic PNG plus a byte-stable sibling PDF (creation metadata is
suppressed so repeated renders stay reproducible).

## Metadata contract

`_metadata.py` is the data-only provenance registry: one entry per generator in
`FIGURE_METADATA`, keyed by module name, recording `status`, `source_relation`,
literature anchors (`source_figure` / `source_equation` / `source_citation`),
and — for data-bearing figures — `estimand`, `unit`, `uncertainty`, and
`replication_unit`, plus a concise `alt_text` for every figure. Figure modules
stay pure plotting functions; the meaning of a rendered artifact lives here.
Adding a generator means adding its name to
`_GENERATORS` and giving it a contract entry, so the registry, captions, and the
`figure_registry.json` payload stay in agreement. Terminology tracks the
manuscript's three axes: keep server-side heuristic figures (e.g.
`robust_influence_weights`) described as recovery-limit diagnostics and reserve
the source-conditional bounded-influence claim for client-side FedGVI. Describe
variational-server figures as raw effective-weight bounds or empirical
redescending-weight diagnostics, never as estimator-level B-robustness.

## See Also

- [`README.md`](README.md) - quick reference
- [`../AGENTS.md`](../AGENTS.md) - source-layer contract
- [`../../docs/manuscript/accessibility.md`](../../docs/manuscript/accessibility.md) - color-independent and reader-surface contract
