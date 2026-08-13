# `src/analysis/` - project execution orchestration guide

## Purpose

Importable analysis orchestration for Active Fedference.

## Rules

- Keep orchestration code in `workflow.py`; avoid placing domain math here.
- Keep the canonical Stage-02 path declaration in `artifacts.py`; validators
  consume it instead of scanning `output/` or duplicating filename lists.
- Keep CLI parsing in `scripts/`; this package should expose callable workflow
  functions.
- Keep generated file writes explicit and covered by tests.
- Preserve deterministic experiment inputs from `experiment_config.py`.

## Schema and validation boundary

- All report writes go through `workflow._write_json(payload, path, schema="<name>")`,
  which validates against `report_schemas.py` before anything lands on disk.
  Never bypass it with a bare `json.dump`, and never add a `_write_json` call
  without a `schema=` name.
- Figure input contracts live in `report_schemas.FIGURE_DEPENDENCY_CONTRACTS`
  and are enforced via `check_figure_contract(...)` before each figure call.
- Adding a report: declare its `TypedDict` + `_REPORT_SCHEMAS` entry in
  `report_schemas.py`, pass `schema="<name>"` at the write site, declare figure
  dependencies, and extend `tests/analysis/test_report_schemas.py` for any
  bespoke shape (the generic accept/reject tests cover table-declared schemas
  automatically).
- Validation failures must stay loud: `ReportSchemaError` names the report and
  field; do not catch-and-continue at the write boundary.
- Schemas describe existing shapes only — a schema change that renames or drops
  a consumed report key is out of scope for this layer.

## See Also

- [`README.md`](README.md) - quick reference
- [`../AGENTS.md`](../AGENTS.md) - source-layer contract
