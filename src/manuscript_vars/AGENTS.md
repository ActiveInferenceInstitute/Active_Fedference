# AGENTS.md — `src/manuscript_vars/`

- Never hardcode a numeric value in manuscript prose. Add or reuse a token and
  let this package resolve it.
- Generation reads analysis reports, configuration and the fresh coverage
  receipt. It must not run the test suite during rendering or invent a value
  when an input is missing.
- Keep token names and groups stable; a rename ripples into every manuscript
  section and the syntax registry. Update
  [`../../manuscript/SYNTAX.md`](../../manuscript/SYNTAX.md) when a label
  changes.
- Keep output deterministic: sorted groups, stable formatting, no timestamps or
  machine paths.
- This package hydrates Markdown and the token map. It does not build PDFs,
  slides or web assets.
- Add provenance and table tests for every new token class.
- Parent rules: [../AGENTS.md](../AGENTS.md).
