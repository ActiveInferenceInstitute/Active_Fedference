# `src/fedference_cli/` — package-local rules

This package is the installed process adapter for Active Fedference. Keep it
modular, evidence-bound, and compatible with the public `fedference` entry
point.

## Required boundaries

- Keep `__init__.py` the installed entry-point facade and export only `main`.
  `_report_fallbacks` remains internal to `_support.py`; do not reexport it or
  put parser, runner, receipt, or numerical logic in the facade.
- Keep parser grammar and process error mapping in `_parser.py`.
- Keep registry-to-runner dispatch in `_commands.py`.
- Keep read-only GNN validation in `_models.py`; delegate parsing and native configuration checks to the domain loaders. Never run simulations to report admission.
- Keep atomic writes, project-root/output isolation, validation, registry
  summaries, and receipt construction in `_support.py`.
- Keep research algorithms in `src/fedference/` and schemas in
  `src/analysis/`; do not duplicate aggregation math in this package.
- Preserve explicit output paths, empty-directory checks, configuration hashes,
  dataset hashes, Git/tree state, lockfile digests, and fallback receipts.
- Do not write into the committed `output/` reviewer snapshot from a CLI run.

## Change recipe

1. Declare the evidence contract in `fedference.research_registry`.
2. Implement and test the reusable operation in the domain/experiment layer.
3. Add the smallest possible command adapter and parser surface.
4. Add zero-mock tests covering success, malformed inputs, output isolation,
   receipt verification, and the relevant negative control.
5. Run the package lint/type/test slice, then the full source and publication
   gates before changing generated artifacts.

The package map and examples live in [`README.md`](README.md). The cross-layer
rules and extension recipes live in
[`docs/development/modularity.md`](../../docs/development/modularity.md).
