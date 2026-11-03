# `src/analysis/schemas/` — report-family schemas

Typed report shapes and per-family validators owned by the canonical
`analysis.report_schemas` boundary. One module per report family; `__init__.py`
marks the package and carries no exports.

`analysis/report_schemas.py` imports these modules, assembles the runtime schema
table and exposes `validate_report` plus the figure-dependency contracts.
Validation runs once, at `workflow._write_json(payload, path, schema=...)`,
before anything touches disk.

## Families

The modules cover aggregation diagnostics, application integrity, belief
sharing, BNN robustness/validation/Torch, contamination, evidence replication,
figure dependencies and registries, hierarchical BMR, learning, observer worlds,
parameter recovery, presentation, primitives, records, registry, review
grid/validation/values, robustness sweep, sensitivity, source-render provenance,
study summaries and worlds.

## Adding a schema

1. Add the module and its `TypedDict` shapes plus `_check_<family>` validator.
2. Wire the import and `_REPORT_SCHEMAS` entry in
   [`../report_schemas.py`](../report_schemas.py).
3. Pass `schema="<name>"` at the write site in
   [`../workflow.py`](../workflow.py).
4. Declare consumed fields in `FIGURE_DEPENDENCY_CONTRACTS` when a figure reads
   the report.
5. Rely on
   [`../../../tests/analysis/test_report_schemas.py`](../../../tests/analysis/test_report_schemas.py),
   which derives accept/reject coverage from the table.

Schemas describe existing payload shapes. They never rename, add or drop a
consumed key.
