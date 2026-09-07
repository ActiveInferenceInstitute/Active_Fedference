"""Real-browser figure visibility, zoom, and overflow checks for the HTML reader."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.publication

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = _ROOT / "tests" / "browser" / "html_zoom_overflow.spec.cjs"
_WEB = _ROOT / "output" / "web"
_NODE_PROJECT = _ROOT / ".tmp" / "playwright-node"
_PLAYWRIGHT = _NODE_PROJECT / "node_modules" / ".bin" / "playwright"


def test_every_rendered_html_surface_contains_zoom_overflow_locally() -> None:
    """Check desktop figure edges and 200%/400% reflow on every reader page."""

    index = _WEB / "index.html"
    manuscript_pages = sorted(_WEB.glob("manuscript__*.html"))
    assert index.is_file(), "the rendered reader index is required for browser reflow QA"
    assert manuscript_pages, "rendered per-section manuscript pages are required for browser reflow QA"
    html_pages = [index, *manuscript_pages]
    environment = os.environ.copy()
    environment["FEDFERENCE_HTML_PAGES"] = json.dumps([str(page.resolve()) for page in html_pages])
    environment["NODE_PATH"] = str(_NODE_PROJECT / "node_modules")
    environment["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = "1"
    if not _PLAYWRIGHT.is_file():
        installed = subprocess.run(
            [
                "npm",
                "install",
                "--prefix",
                str(_NODE_PROJECT),
                "--no-package-lock",
                "--no-save",
                "@playwright/test@1.62.1",
            ],
            cwd=_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert installed.returncode == 0, (
            "could not provision the pinned Playwright test runner\n"
            f"stdout:\n{installed.stdout}\n"
            f"stderr:\n{installed.stderr}"
        )
    browser_install = subprocess.run(
        [str(_PLAYWRIGHT), "install", "chromium"],
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert browser_install.returncode == 0, (
        "could not provision pinned Chromium for the rendered-browser probe\n"
        f"stdout:\n{browser_install.stdout}\n"
        f"stderr:\n{browser_install.stderr}"
    )
    completed = subprocess.run(
        [
            str(_PLAYWRIGHT),
            "test",
            str(_SPEC),
            "--browser=chromium",
            "--workers=1",
            "--reporter=line",
            "--output=.tmp/playwright-results",
        ],
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        "rendered-browser accessibility probe failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
