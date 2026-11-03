# AGENTS.md — `src/fedference/experiments/`

- Return payloads; never write files, create output directories or mutate shared
  state. Persistence belongs to `analysis/workflow.py`.
- Use the configured seed budget from `ExperimentConfig`. Do not add a
  module-level seed or a global `np.random` call.
- Do not validate report shapes here. The write boundary validates against
  `analysis.report_schemas`; duplicating that check creates two sources of truth.
- Keep claim labels accurate. Do not upgrade a seeded characterization or a
  heuristic result into a universal guarantee.
- A new family needs a mirrored test, a producer entry and a schema-backed
  report name before it can be registered.
- Tests are zero-mock and seeded; do not shrink a study to a trivial size just
  to make a slow test pass without documenting the reduced scope.
- Parent rules: [../AGENTS.md](../AGENTS.md).
