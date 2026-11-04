# AGENTS.md — `tests/browser/`

- The spec is consumed by exact path from
  [`../test_html_browser_accessibility.py`](../test_html_browser_accessibility.py).
  A new spec must be wired into that driver and the rendered page list; it is
  not discovered automatically.
- Keep the Playwright version pinned in the driver. Do not add a floating
  dependency or a second browser install path.
- The check reads local rendered files only. Do not add a remote URL, telemetry
  or a network fixture.
- A passing browser run is rendered-surface QA, not an accessibility
  conformance claim.
- Keep the spec deterministic: no animations, no wall-clock waits, explicit
  viewports and reflow checks.
- Parent rules: [../AGENTS.md](../AGENTS.md). Render contract:
  [`../../docs/manuscript/rendering_pipeline.md`](../../docs/manuscript/rendering_pipeline.md).
