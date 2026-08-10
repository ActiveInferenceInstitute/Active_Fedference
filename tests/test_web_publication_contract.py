from __future__ import annotations

from pathlib import Path

import pytest

from publication.web_package import (
    mirror_web_figures,
    normalize_web_xrefs,
    sanitize_machine_paths,
    validate_web_package,
)

pytestmark = pytest.mark.publication


def _write_web_fixture(root: Path) -> None:
    figures = root / "output" / "figures"
    web = root / "output" / "web"
    figures.mkdir(parents=True)
    web.mkdir(parents=True)
    (figures / "system_overview.png").write_bytes(b"png")
    (web / "index.html").write_text(
        '<!doctype html><html lang="en"><head><title>Index</title></head><body>'
        '<a class="skip-link" href="#main-content">Skip to main content</a>'
        '<main id="main-content" tabindex="-1"><h2 data-number="3.3" '
        'id="sec:robustness-axes">Robustness</h2>'
        '<span class="citation" data-cites="sec:robustness-axes">'
        "[@sec:robustness-axes]</span>"
        '<figure><a class="figure-full-size-link" href="../figures/system_overview.png" '
        'aria-label="Open full-size figure"><img src="../figures/system_overview.png" '
        'alt="System overview"></a><figcaption>System overview.</figcaption></figure>'
        "</main></body></html>",
        encoding="utf-8",
    )
    (web / "manuscript__example.html").write_text(
        '<!doctype html><html lang="en"><head><title>Example</title></head><body>'
        '<a class="skip-link" href="#main-content">Skip to main content</a>'
        '<main id="main-content" tabindex="-1">See [sec:robustness-axes] and '
        'Theorem~<span class="math inline">'
        r"\(\ref{thm:recovery}\)</span>.</main></body></html>",
        encoding="utf-8",
    )


def test_prepare_web_package_mirrors_figures_and_normalizes_xrefs(tmp_path: Path) -> None:
    _write_web_fixture(tmp_path)

    copied = mirror_web_figures(tmp_path)
    replacements = normalize_web_xrefs(tmp_path)
    result = validate_web_package(tmp_path)

    assert (tmp_path / "output" / "web" / "figures" / "system_overview.png") in copied
    assert replacements == 3
    assert result.ok
    assert result.assets_checked == 1
    assert "[@sec:robustness-axes]" not in (
        tmp_path / "output" / "web" / "index.html"
    ).read_text(encoding="utf-8")
    individual = (tmp_path / "output" / "web" / "manuscript__example.html").read_text(
        encoding="utf-8"
    )
    assert 'href="index.html#sec:robustness-axes"' in individual
    assert ">Section 3.3</a>" in individual
    assert "Theorem~" not in individual
    assert '<span class="xref">Theorem recovery</span>' in individual


def test_sanitize_machine_paths_makes_text_outputs_clone_independent(tmp_path: Path) -> None:
    output = tmp_path / "output" / "logs"
    output.mkdir(parents=True)
    log = output / "render.log"
    log.write_text(
        "/private/tmp/render-work/output/file\n"
        "/Users/mini/Documents/project/output/file\n"
        "/Volumes/blue/project/output/file\n",
        encoding="utf-8",
    )
    changed = sanitize_machine_paths(tmp_path)
    assert changed == (log,)
    assert log.read_text(encoding="utf-8") == (
        "<tmp>/output/file\n<home>/Documents/project/output/file\n"
        "<volume>/project/output/file\n"
    )


def test_validate_web_package_reports_missing_assets(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        '<html><body><img src="../figures/missing.png"></body></html>',
        encoding="utf-8",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert result.missing_assets


def test_validate_web_package_reports_raw_crossrefs(tmp_path: Path) -> None:
    _write_web_fixture(tmp_path)
    mirror_web_figures(tmp_path)

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert result.raw_xrefs


def test_validate_web_package_catches_renderer_style_xrefs(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        "<html><body>See [thm:belief-sharing-recovery].</body></html>",
        encoding="utf-8",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert result.raw_xrefs


def test_validate_web_package_reports_broken_internal_fragment(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        '<html><body><a href="#sec:missing">missing</a></body></html>',
        encoding="utf-8",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert result.broken_xrefs


def test_validate_web_package_reports_leaked_figure_markdown(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        "<html><body>caption](../figures/plot.png){#fig:plot}</body></html>",
        encoding="utf-8",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert result.malformed_markup


def test_validate_web_package_reports_unresolved_manuscript_token(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        "<html><body>Tests: {{TEST_COUNT}}</body></html>",
        encoding="utf-8",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("unresolved manuscript token" in issue for issue in result.malformed_markup)


def test_validate_web_package_reports_accessibility_contract_violations(
    tmp_path: Path,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        "<html><head><title> </title></head><body>"
        '<div id="duplicate"></div><div id="duplicate"></div>'
        '<figure><a class="figure-full-size-link" href="plot.png">'
        '<img alt=""></a></figure></body></html>',
        encoding="utf-8",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    issues = "\n".join(result.accessibility_issues)
    assert "non-empty html lang" in issues
    assert "non-empty document title" in issues
    assert "main element" in issues
    assert ".skip-link" in issues
    assert "lack non-empty alt text" in issues
    assert "lack figcaption" in issues
    assert "lack aria-label" in issues
    assert "duplicate id(s): duplicate" in issues


def test_validate_web_package_reports_unreadable_html(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_bytes(b"\xff\xfe")

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("accessibility parse failed" in issue for issue in result.accessibility_issues)
    assert any("unreadable HTML" in issue for issue in result.malformed_markup)


def test_validate_web_package_requires_focusable_skip_target(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        '<html lang="en"><head><title>Example</title></head><body>'
        '<a class="skip-link" href="#main-content">Skip to main content</a>'
        '<main id="main-content">content</main></body></html>',
        encoding="utf-8",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "skip target must be focusable" in issue
        for issue in result.accessibility_issues
    )


def test_validate_web_package_requires_skip_link_first(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        '<html lang="en"><head><title>Example</title></head><body>'
        '<a href="other.html">Other</a>'
        '<a class="skip-link" href="#main-content">Skip to main content</a>'
        '<main id="main-content" tabindex="-1">content</main></body></html>',
        encoding="utf-8",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "skip link must be the first interactive element" in issue
        for issue in result.accessibility_issues
    )
