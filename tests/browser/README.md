# `tests/browser/`

One browser spec: [html_zoom_overflow.spec.cjs](html_zoom_overflow.spec.cjs).

It is driven by
[`../test_html_browser_accessibility.py`](../test_html_browser_accessibility.py),
which passes the rendered reader pages through `FEDFERENCE_HTML_PAGES` and
checks desktop visibility, initial zoom and live reflow for every page. The
driver installs the pinned `@playwright/test@1.62.1` and Chromium into
`.tmp/playwright-node` when they are absent.

Prerequisites: a rendered `output/web/` tree (`index.html` plus
`manuscript__*.html`) and Node/npm on the runner. The test is marked
`publication`, so select it explicitly:

```bash
uv run --locked --extra dev pytest tests/test_html_browser_accessibility.py -m publication -q
```

This is local rendered-surface QA, not a WCAG or PDF/UA conformance check.
