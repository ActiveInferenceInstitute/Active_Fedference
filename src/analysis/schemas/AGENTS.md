# AGENTS.md — `src/analysis/schemas/`

- This package contains schema declarations and pure validators only. No file
  I/O, orchestration, plotting or domain math.
- Keep one module per report family and import it through
  `analysis.report_schemas`; do not let callers import a schema module directly
  as a second public path.
- Schemas describe existing shapes. A change that renames or drops a key
  consumed by a figure or report is a contract change, not a schema cleanup;
  update the producer, writer, figure dependencies and tests together.
- Validation failures must name the report and field. Do not catch and continue.
- Adding a schema requires the `_REPORT_SCHEMAS` entry, a `schema=` write site
  and accept/reject tests.
- Parent rules: [../../AGENTS.md](../../AGENTS.md). The analysis-level contract
  is [../AGENTS.md](../AGENTS.md).
