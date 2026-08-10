# `src/analysis/` — Active Fedference orchestration layer

This package owns deterministic workflow orchestration for the project experiments and artifact generation.

## Files

| File | Role |
| --- | --- |
| `__init__.py` | Public module exports for the analysis layer. |
| `workflow.py` | Pipeline entrypoint for source-mechanism analogues, figure generation, and report materialisation. |
| `report_schemas.py` | Typed report/figure-registry schemas and the runtime validator enforced at the write boundary. |

## Workflow contract

- **Inputs:** `experiment_config.py` is the single source of experiment truth.
- **Execution:** Scripts call into `workflow` functions; this package stays importable and testable.
- **Output:** Writes JSON and figure files into `output/` under deterministic, test-backed filenames.
- **Determinism:** All analyses are keyed by explicit `seed` values from config.

## Schema and validation boundary

Every report payload written to `output/reports/*.json` and the
`output/figures/figure_registry.json` payload is typed and centrally validated:

- **What is validated:** `report_schemas.py` declares one `TypedDict` per report
  (documentation for readers and mypy) plus a shallow runtime schema table of
  required/optional top-level fields and their types. Declared degradations are
  part of the schema — `bnn_torch` accepts either the full `status: "ok"` payload
  or the single-key `{"status": "skipped: ..."}` payload written when the PyTorch
  optional extra is unavailable.
- **Where:** validation happens once, at the single write boundary.
  `workflow._write_json(payload, path, schema="<name>")` calls
  `report_schemas.validate_report` *before* anything touches disk; a malformed
  payload raises `ReportSchemaError` naming the report and field, and no file
  (or directory) is created. Additionally, each figure generator's consumed
  report fields are declared in `report_schemas.FIGURE_DEPENDENCY_CONTRACTS`
  and checked via `check_figure_contract(generator, report_name, report)` in
  `workflow.py` immediately before the figure call, so a missing input fails
  with a named report+field error instead of an anonymous `KeyError`.
- **How to add a schema when adding a report:** define the `TypedDict` and the
  `_REPORT_SCHEMAS` entry in `report_schemas.py`, pass `schema="<name>"` at the
  new `_write_json` call site in `workflow.py`, declare any figure dependencies
  in `FIGURE_DEPENDENCY_CONTRACTS`, and rely on
  `tests/analysis/test_report_schemas.py` — its accept/reject tests are derived
  mechanically from the schema table, so every declared schema is covered
  automatically (add report-specific shape tests only for bespoke variants such
  as `bnn_torch`).

Schemas describe existing payload shapes; they never add, rename, or drop
report keys. Validation is a software-quality boundary, not evidence about any
scientific quantity.

## See also

- [`../README.md`](../README.md) — project source overview
- [`../AGENTS.md`](../AGENTS.md) — source-layer editing rules
