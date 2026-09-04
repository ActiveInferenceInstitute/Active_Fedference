from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import publication.web_package as web_package
from analysis.visual_contracts import COMPLEX_FIGURE_GENERATORS
from publication.web_package import (
    _asset_path,
    _local_href_target,
    mirror_web_figures,
    normalize_web_xrefs,
    sanitize_machine_paths,
    validate_web_package,
)

pytestmark = pytest.mark.publication

_COMPLEX_TARGET_GENERATOR = "application_integrity_flow"
_COMPLEX_TARGET_LABEL = "fig:test-application-integrity-flow"
_COMPLEX_TARGET_DETAILS_ID = "fig-test-application-integrity-flow-long-description"
_COMPLEX_TARGET_DESCRIPTION = (
    "Read the three panels in order.\n\nThis source-owned map carries no broader empirical claim."
)


def _complex_label(generator: str) -> str:
    return f"fig:test-{generator.replace('_', '-')}"


def _complex_description(generator: str) -> str:
    if generator == _COMPLEX_TARGET_GENERATOR:
        return _COMPLEX_TARGET_DESCRIPTION
    return (
        f"Read the source-bound {generator.replace('_', ' ')} figure in order.\n\n"
        "This accessibility fixture carries no broader empirical claim."
    )


def _complex_registry_entry(
    generator: str,
    *,
    omit_long_description: bool = False,
) -> dict[str, str]:
    label = _complex_label(generator)
    entry = {
        "label": label,
        "filename": f"{generator}.png",
        "path": f"output/figures/{generator}.png",
        "source_manuscript": "manuscript/26_reproducibility.md",
        "caption": f"Source-bound {generator.replace('_', ' ')} test caption.",
        "generated_by": generator,
        "status": "test fixture",
        "source_relation": "source-owned accessibility fixture",
        "source_figure": "none",
        "source_equation": "none",
        "source_citation": "none",
        "estimand": "accessibility association",
        "unit": "HTML element",
        "uncertainty": "none",
        "replication_unit": "not applicable",
        "alt_text": f"Concise alternative for {generator.replace('_', ' ')}.",
    }
    if not omit_long_description:
        entry["long_description"] = _complex_description(generator)
    return entry


def _write_complex_registry(
    root: Path,
    *,
    omit_generator: str | None = None,
    omit_long_description: str | None = None,
) -> None:
    figures = root / "output" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    for generator in sorted(COMPLEX_FIGURE_GENERATORS):
        if generator == omit_generator:
            continue
        (figures / f"{generator}.png").write_bytes(b"png")
        entries.append(
            _complex_registry_entry(
                generator,
                omit_long_description=generator == omit_long_description,
            )
        )
    payload = {
        "schema_version": "1.2",
        "generated_by": "analysis.workflow.run_analysis_pipeline",
        "figures": sorted(entries, key=lambda entry: entry["label"]),
    }
    (figures / "figure_registry.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _complex_figure_markup(
    *,
    target_overrides: dict[str, str | bool] | None = None,
    omit_generator: str | None = None,
) -> str:
    overrides = target_overrides or {}
    figures: list[str] = []
    for index, generator in enumerate(sorted(COMPLEX_FIGURE_GENERATORS), start=1):
        if generator == omit_generator:
            continue
        label = _complex_label(generator)
        details_id = f"{label.replace(':', '-')}-long-description"
        description = _complex_description(generator)
        summary = f"Detailed description of figure {generator.replace('_', ' ')}"
        details_label = label
        initially_open = False
        duplicate_summary = False
        collapse_description_paragraphs = False
        details_before_caption = False
        aria_details = details_id
        if generator == _COMPLEX_TARGET_GENERATOR:
            aria_details = str(overrides.get("aria_details", aria_details))
            details_id = str(overrides.get("details_id", details_id))
            details_label = str(overrides.get("details_label", details_label))
            summary = str(overrides.get("summary", summary))
            description = str(overrides.get("description", description))
            initially_open = bool(overrides.get("initially_open", False))
            duplicate_summary = bool(overrides.get("duplicate_summary", False))
            collapse_description_paragraphs = bool(overrides.get("collapse_description_paragraphs", False))
            details_before_caption = bool(overrides.get("details_before_caption", False))
        open_attribute = " open" if initially_open else ""
        summary_html = f"<summary>{summary}</summary>"
        if duplicate_summary:
            summary_html += "<summary>Duplicate summary</summary>"
        description_parts = description.split("\n\n")
        if collapse_description_paragraphs:
            description_parts = [" ".join(description_parts)]
        description_html = "".join(f"<p>{paragraph}</p>" for paragraph in description_parts)
        details_html = (
            f'<details id="{details_id}" class="figure-long-description" '
            f'data-figure-label="{details_label}"{open_attribute}>{summary_html}'
            f"{description_html}</details>"
        )
        caption_html = f"<figcaption>Figure {index}: {generator.replace('_', ' ')}.</figcaption>"
        caption_and_details = (
            details_html + caption_html if details_before_caption else caption_html + details_html
        )
        figures.append(
            f'<figure id="{label}">'
            f'<a class="figure-full-size-link" href="../figures/{generator}.png" '
            f'aria-label="Open full-size Figure {index}, {generator.replace("_", " ")}.">'
            f'<img src="../figures/{generator}.png" '
            f'alt="Concise alternative for {generator.replace("_", " ")}." '
            f'aria-details="{aria_details}"></a>'
            f"{caption_and_details}</figure>"
        )
    return "".join(figures)


def _install_complex_reader_contract(
    root: Path,
    *,
    target_overrides: dict[str, str | bool] | None = None,
    omit_registry_generator: str | None = None,
    omit_registry_long_description: str | None = None,
    omit_html_generator: str | None = None,
) -> str:
    _write_complex_registry(
        root,
        omit_generator=omit_registry_generator,
        omit_long_description=omit_registry_long_description,
    )
    return _complex_figure_markup(
        target_overrides=target_overrides,
        omit_generator=omit_html_generator,
    )


def _write_web_fixture(root: Path) -> None:
    figures = root / "output" / "figures"
    web = root / "output" / "web"
    figures.mkdir(parents=True)
    web.mkdir(parents=True)
    (figures / "system_overview.png").write_bytes(b"png")
    complex_figures = _install_complex_reader_contract(root)
    (web / "index.html").write_text(
        '<!doctype html><html lang="en"><head><title>Index</title></head><body>'
        '<a class="skip-link" href="#main-content">Skip to main content</a>'
        '<main id="main-content" tabindex="-1"><h2 data-number="3.3" '
        'id="sec:robustness-axes">Robustness</h2>'
        '<span class="citation" data-cites="sec:robustness-axes">'
        "[@sec:robustness-axes]</span>"
        '<figure><a class="figure-full-size-link" href="../figures/system_overview.png" '
        'aria-label="Open full-size figure, System overview."><img src="../figures/system_overview.png" '
        'alt="System overview"></a><figcaption>System overview.</figcaption></figure>'
        f"{complex_figures}"
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
    assert result.assets_checked == len(COMPLEX_FIGURE_GENERATORS) + 1
    assert "[@sec:robustness-axes]" not in (tmp_path / "output" / "web" / "index.html").read_text(
        encoding="utf-8"
    )
    individual = (tmp_path / "output" / "web" / "manuscript__example.html").read_text(encoding="utf-8")
    assert 'href="index.html#sec:robustness-axes"' in individual
    assert ">Section 3.3</a>" in individual
    assert "Theorem~" not in individual
    assert '<span class="xref">Theorem recovery</span>' in individual


def test_sanitize_machine_paths_makes_text_outputs_clone_independent(tmp_path: Path) -> None:
    output = tmp_path / "output" / "logs"
    output.mkdir(parents=True)
    log = output / "render.log"
    temporary_fixture = Path("/", "private", "tmp", "render-work", "output", "file")
    home_fixture = Path("/", "Users", "test-user", "workspace", "project", "output", "file")
    volume_fixture = Path("/", "Volumes", "test-volume", "project", "output", "file")
    log.write_text(
        f"{temporary_fixture}\n{home_fixture}\n{volume_fixture}\n",
        encoding="utf-8",
    )
    changed = sanitize_machine_paths(tmp_path)
    assert changed == (log,)
    assert log.read_text(encoding="utf-8") == (
        "<tmp>/output/file\n<home>/workspace/project/output/file\n<volume>/project/output/file\n"
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


def _write_valid_link_page(
    root: Path,
    links: str,
    *,
    target_overrides: dict[str, str | bool] | None = None,
    omit_registry_generator: str | None = None,
    omit_registry_long_description: str | None = None,
    omit_html_generator: str | None = None,
) -> Path:
    web = root / "output" / "web"
    web.mkdir(parents=True, exist_ok=True)
    complex_figures = _install_complex_reader_contract(
        root,
        target_overrides=target_overrides,
        omit_registry_generator=omit_registry_generator,
        omit_registry_long_description=omit_registry_long_description,
        omit_html_generator=omit_html_generator,
    )
    page = web / "index.html"
    page.write_text(
        '<!doctype html><html lang="en"><head><title>Links</title></head><body>'
        '<a class="skip-link" href="#main-content">Skip to main content</a>'
        f'<main id="main-content" tabindex="-1">{links}{complex_figures}</main></body></html>',
        encoding="utf-8",
    )
    mirror_web_figures(root)
    return page


def test_validate_web_package_reports_missing_local_href(tmp_path: Path) -> None:
    _write_valid_link_page(
        tmp_path,
        '<a href="missing.json">Missing</a>'
        "<a href='also-missing.json'>Also missing</a>"
        "<a href=third-missing.json>Third missing</a>",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("missing.json (missing local target)" in issue for issue in result.broken_xrefs)
    assert any("also-missing.json (missing local target)" in issue for issue in result.broken_xrefs)
    assert any("third-missing.json (missing local target)" in issue for issue in result.broken_xrefs)


def test_validate_web_package_accepts_existing_local_href_and_output_figure(
    tmp_path: Path,
) -> None:
    page = _write_valid_link_page(
        tmp_path,
        '<a href="evidence.json">Evidence</a><a href="../figures/plot.png">Full-size figure</a>',
    )
    (page.parent / "evidence.json").write_text('{"status":"ok"}\n', encoding="utf-8")
    figures = tmp_path / "output" / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "plot.png").write_bytes(b"png")

    result = validate_web_package(tmp_path)

    assert result.ok
    assert not result.broken_xrefs


def test_validate_web_package_accepts_external_mailto_tel_and_fragment_hrefs(
    tmp_path: Path,
) -> None:
    _write_valid_link_page(
        tmp_path,
        '<span id="details">Details</span>'
        '<a href="#details">Fragment</a>'
        '<a href="https://example.org/reference">External</a>'
        '<a href="mailto:maintainer@example.org">Email</a>'
        '<a href="tel:+12025550123">Telephone</a>',
    )

    result = validate_web_package(tmp_path)

    assert result.ok
    assert not result.broken_xrefs


def test_validate_web_package_accepts_explicit_pandoc_markdown_fragment(tmp_path: Path) -> None:
    _write_valid_link_page(
        tmp_path,
        '<a href="../figures/figure_exact_values.md#fig-values-belief-quality">Exact values</a>',
    )
    figures = tmp_path / "output" / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "figure_exact_values.md").write_text(
        "## Belief quality {#fig-values-belief-quality}\n",
        encoding="utf-8",
    )

    result = validate_web_package(tmp_path)

    assert result.ok
    assert not result.broken_xrefs


@pytest.mark.parametrize(
    ("href", "message"),
    [
        ("../../private.txt", "leaves the shipped output tree"),
        ("javascript:alert(1)", "unsupported href scheme"),
        ("//example.org/path", "protocol-relative href is not allowed"),
        ("/absolute.txt", "absolute local href is not allowed"),
    ],
)
def test_validate_web_package_rejects_unsafe_local_hrefs(
    tmp_path: Path,
    href: str,
    message: str,
) -> None:
    (tmp_path / "private.txt").write_text("private\n", encoding="utf-8")
    _write_valid_link_page(tmp_path, f'<a href="{href}">Unsafe</a>')

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(message in issue for issue in result.broken_xrefs)


def test_reader_surface_manuscript_links_use_canonical_public_repository_urls() -> None:
    root = Path(__file__).resolve().parents[1]
    repository_base = "https://github.com/ActiveInferenceInstitute/Active_Fedference/blob/main"

    experimental_design = (root / "manuscript" / "12_methods_experimental_design.md").read_text(
        encoding="utf-8"
    )
    references = (root / "manuscript" / "99_references.md").read_text(encoding="utf-8")

    assert f"]({repository_base}/manuscript/config.yaml)" in experimental_design
    assert f"]({repository_base}/manuscript/references.bib)" in references


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


def test_validate_web_package_rejects_literal_markup_in_accessible_name(
    tmp_path: Path,
) -> None:
    _write_valid_link_page(
        tmp_path,
        '<div aria-label="Broken <span class="citation">Citation</span>">Text</div>',
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("aria-label contains literal HTML markup" in issue for issue in result.malformed_markup)


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


def test_validate_web_package_rejects_generic_full_size_aria_label(
    tmp_path: Path,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "plot.png").write_bytes(b"png")
    _write_valid_link_page(
        tmp_path,
        '<figure><a class="figure-full-size-link" href="plot.png" '
        'aria-label="Open full-size figure"><img src="plot.png" alt="Plot"></a>'
        "<figcaption>Specific plot.</figcaption></figure>",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("lack a contextual Figure number or title" in issue for issue in result.accessibility_issues)


def _write_minimal_reader_without_registry(root: Path, *, body_prefix: str = "") -> None:
    web = root / "output" / "web"
    web.mkdir(parents=True, exist_ok=True)
    (web / "index.html").write_text(
        '<!doctype html><html lang="en"><head><title>Reader</title></head><body>'
        f"{body_prefix}"
        '<a class="skip-link" href="#main-content">Skip to main content</a>'
        '<main id="main-content" tabindex="-1">Reader content.</main></body></html>',
        encoding="utf-8",
    )


def test_validate_web_package_fails_closed_without_figure_registry(tmp_path: Path) -> None:
    _write_minimal_reader_without_registry(tmp_path)

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("required figure registry is missing" in issue for issue in result.accessibility_issues)


def test_validate_web_package_fails_closed_on_unreadable_figure_registry(tmp_path: Path) -> None:
    _write_minimal_reader_without_registry(tmp_path)
    figures = tmp_path / "output" / "figures"
    figures.mkdir(parents=True)
    (figures / "figure_registry.json").write_text("{not-json\n", encoding="utf-8")

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("figure registry is unreadable" in issue for issue in result.accessibility_issues)


def test_validate_web_package_fails_closed_on_invalid_figure_registry_schema(tmp_path: Path) -> None:
    _write_valid_link_page(tmp_path, "")
    registry = tmp_path / "output" / "figures" / "figure_registry.json"
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["schema_version"] = "0.0"
    registry.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("figure registry schema is invalid" in issue for issue in result.accessibility_issues)


def test_validate_web_package_requires_every_declared_complex_generator(tmp_path: Path) -> None:
    _write_valid_link_page(
        tmp_path,
        "",
        omit_registry_generator=_COMPLEX_TARGET_GENERATOR,
        omit_html_generator=_COMPLEX_TARGET_GENERATOR,
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        f"required complex figure {_COMPLEX_TARGET_GENERATOR!r} is absent" in issue
        for issue in result.accessibility_issues
    )


def test_validate_web_package_requires_declared_complex_long_description(tmp_path: Path) -> None:
    _write_valid_link_page(
        tmp_path,
        "",
        omit_registry_long_description=_COMPLEX_TARGET_GENERATOR,
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "figure registry schema is invalid" in issue
        and "complex figure requires a structured long_description" in issue
        for issue in result.accessibility_issues
    )


def _write_complex_figure_fixture(
    root: Path,
    *,
    aria_details: str = _COMPLEX_TARGET_DETAILS_ID,
    details_id: str = _COMPLEX_TARGET_DETAILS_ID,
    details_label: str = _COMPLEX_TARGET_LABEL,
    summary: str = "Detailed description of figure application integrity flow",
    description: str = _COMPLEX_TARGET_DESCRIPTION,
    details_before_caption: bool = False,
    initially_open: bool = False,
    duplicate_summary: bool = False,
    collapse_description_paragraphs: bool = False,
) -> None:
    _write_valid_link_page(
        root,
        "",
        target_overrides={
            "aria_details": aria_details,
            "details_id": details_id,
            "details_label": details_label,
            "summary": summary,
            "description": description,
            "details_before_caption": details_before_caption,
            "initially_open": initially_open,
            "duplicate_summary": duplicate_summary,
            "collapse_description_paragraphs": collapse_description_paragraphs,
        },
    )


def test_validate_web_package_accepts_registry_bound_long_description(
    tmp_path: Path,
) -> None:
    _write_complex_figure_fixture(tmp_path)

    result = validate_web_package(tmp_path)

    assert result.ok
    assert not result.accessibility_issues


def test_validate_web_package_requires_complex_details_after_caption(
    tmp_path: Path,
) -> None:
    _write_complex_figure_fixture(tmp_path, details_before_caption=True)

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "labeled details disclosure must follow figcaption in DOM child order" in issue
        for issue in result.accessibility_issues
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"aria_details": "fig-test-wrong-long-description"},
            "image aria-details must reference",
        ),
        (
            {"details_id": "fig-test-wrong-long-description"},
            "details id must equal",
        ),
        (
            {"details_label": "fig:wrong"},
            "data-figure-label must match",
        ),
        ({"summary": " "}, "details summary must equal"),
        (
            {"summary": "Detailed description of figure a different map"},
            "details summary must equal",
        ),
        (
            {"description": "A different figure description."},
            "details text does not match",
        ),
    ],
)
def test_validate_web_package_rejects_misassociated_complex_description(
    tmp_path: Path,
    overrides: dict[str, str],
    message: str,
) -> None:
    _write_complex_figure_fixture(tmp_path, **overrides)

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(message in issue for issue in result.accessibility_issues)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"initially_open": True}, "details disclosure must be initially collapsed"),
        ({"duplicate_summary": True}, "must contain exactly one summary"),
        (
            {"collapse_description_paragraphs": True},
            "structured long description must contain at least two paragraphs",
        ),
    ],
)
def test_validate_web_package_requires_keyboard_ready_complex_disclosure(
    tmp_path: Path,
    overrides: dict[str, bool],
    message: str,
) -> None:
    _write_complex_figure_fixture(tmp_path, **overrides)

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(message in issue for issue in result.accessibility_issues)


def test_validate_web_package_requires_every_registry_complex_figure(
    tmp_path: Path,
) -> None:
    _write_complex_figure_fixture(tmp_path)
    page = tmp_path / "output" / "web" / "index.html"
    page.write_text(
        '<html lang="en"><head><title>Empty</title></head><body>'
        '<a class="skip-link" href="#main-content">Skip to main content</a>'
        '<main id="main-content" tabindex="-1">No figure.</main></body></html>',
        encoding="utf-8",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        f"registry-declared complex figure {_COMPLEX_TARGET_LABEL}" in issue
        for issue in result.accessibility_issues
    )


def test_validate_web_package_reports_unreadable_html(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_bytes(b"\xff\xfe")

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("accessibility parse failed" in issue for issue in result.accessibility_issues)
    assert any("unreadable HTML" in issue for issue in result.malformed_markup)


@pytest.mark.parametrize("tabindex", [None, "0"])
def test_validate_web_package_requires_focusable_skip_target(
    tmp_path: Path,
    tabindex: str | None,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    tabindex_attribute = "" if tabindex is None else f' tabindex="{tabindex}"'
    (web / "index.html").write_text(
        '<html lang="en"><head><title>Example</title></head><body>'
        '<a class="skip-link" href="#main-content">Skip to main content</a>'
        f'<main id="main-content"{tabindex_attribute}>content</main></body></html>',
        encoding="utf-8",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("skip target must use tabindex='-1'" in issue for issue in result.accessibility_issues)


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
        "skip link must be the first interactive element" in issue for issue in result.accessibility_issues
    )


def test_validate_web_package_treats_summary_as_interactive_before_skip_link(
    tmp_path: Path,
) -> None:
    _write_minimal_reader_without_registry(
        tmp_path,
        body_prefix="<details><summary>Earlier control</summary><p>Content.</p></details>",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "skip link must be the first interactive element" in issue for issue in result.accessibility_issues
    )


def test_sanitize_machine_paths_handles_absent_and_mixed_output_safely(
    tmp_path: Path,
) -> None:
    assert sanitize_machine_paths(tmp_path) == ()

    output = tmp_path / "output"
    output.mkdir()
    unchanged = output / "unchanged.txt"
    unchanged.write_text("portable\n", encoding="utf-8")
    log = output / "render.log"
    log.write_text("portable\n\n", encoding="utf-8")
    unreadable_text = output / "binary.json"
    unreadable_text.write_bytes(b"\xff\xfe")
    ignored_binary = output / "figure.png"
    machine_path = Path("/", "Users", "test-user", "private").as_posix().encode()
    ignored_binary.write_bytes(machine_path)
    symlink = output / "linked.log"
    symlink.symlink_to(log)

    assert sanitize_machine_paths(tmp_path) == (log,)
    assert log.read_text(encoding="utf-8") == "portable\n"
    assert unchanged.read_text(encoding="utf-8") == "portable\n"
    assert unreadable_text.read_bytes() == b"\xff\xfe"
    assert ignored_binary.read_bytes() == machine_path


def test_mirror_web_figures_rejects_missing_source_and_removes_stale_tree(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="missing source figures directory"):
        mirror_web_figures(tmp_path)

    figures = tmp_path / "output" / "figures"
    figures.mkdir(parents=True)
    (figures / "nested").mkdir()
    (figures / "nested" / "current.svg").write_text("<svg/>\n", encoding="utf-8")
    destination = tmp_path / "output" / "web" / "figures"
    (destination / "stale-dir").mkdir(parents=True)
    (destination / "stale-dir" / "stale.txt").write_text("stale\n", encoding="utf-8")

    copied = mirror_web_figures(tmp_path)

    current = destination / "nested" / "current.svg"
    assert copied == (current,)
    assert current.read_text(encoding="utf-8") == "<svg/>\n"
    assert not (destination / "stale-dir").exists()


def test_normalize_web_xrefs_resolves_numbered_surfaces_and_citations(
    tmp_path: Path,
) -> None:
    web = tmp_path / "output" / "web"
    pdf = tmp_path / "output" / "pdf"
    manuscript = tmp_path / "output" / "manuscript"
    web.mkdir(parents=True)
    pdf.mkdir(parents=True)
    manuscript.mkdir(parents=True)
    (manuscript / "references.bib").write_text(
        "@article{single,\n  author = {Ada Lovelace},\n  year = {1843},\n}\n"
        "@article{pair,\n  author = {Hopper, Grace and Alan Turing},\n  year = {1950},\n}\n"
        "@article{group,\n  author = {One, A and Two, B and Three, C},\n  year = {2026},\n}\n"
        "@article{incomplete,\n  title = {No author or year},\n}\n",
        encoding="utf-8",
    )
    (pdf / "_combined_manuscript.aux").write_text(
        "\\newlabel{thm:health}{{2}{7}}\n\\newlabel{sec:ignored}{{9}{9}}\n",
        encoding="utf-8",
    )
    (web / "index.html").write_text(
        '<h2 data-number="3" id="sec:method">Method</h2>'
        '<figure id="fig:result"><figcaption>Figure 4: Result.</figcaption></figure>'
        '<table id="tbl:result"><caption>Table 5: Result.</caption></table>'
        '<span id="eq:pool">x \\qquad{(6)}</span>'
        '<div aria-label="See [tbl:result] and [single]">Attribute test.</div>'
        '<script>const citation = "[single]";</script>'
        '<script type="text/plain"><div aria-label="[single]"></div></script>'
        '<style>.example::after { content: "[tbl:result]"; }</style>'
        '<!-- <div aria-label="[single]"></div> -->'
        '<span class="citation" data-cites="sec:method">[@sec:method]</span> '
        "[fig:result] [tbl:result] [eq:pool] [thm:health] "
        "[single] [pair] [group] [unknown]",
        encoding="utf-8",
    )

    replacements = normalize_web_xrefs(tmp_path)
    rendered = (web / "index.html").read_text(encoding="utf-8")

    assert replacements == 10
    assert ">Section 3</a>" in rendered
    assert ">Figure 4</a>" in rendered
    assert ">Table 5</a>" in rendered
    assert ">Eq. (6)</a>" in rendered
    assert '<span class="xref">Theorem 2</span>' in rendered
    assert "(Lovelace 1843)" in rendered
    assert "(Hopper and Turing 1950)" in rendered
    assert "(One et al. 2026)" in rendered
    assert "[unknown]" in rendered
    assert 'aria-label="See Table 5 and (Lovelace 1843)"' in rendered
    assert 'aria-label="See <' not in rendered
    assert '<script>const citation = "[single]";</script>' in rendered
    assert '<script type="text/plain"><div aria-label="[single]"></div></script>' in rendered
    assert '<style>.example::after { content: "[tbl:result]"; }</style>' in rendered
    assert '<!-- <div aria-label="[single]"></div> -->' in rendered


def test_normalize_web_xrefs_preserves_single_and_double_quoted_attribute_markup(
    tmp_path: Path,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    single_quoted = (
        "data-single='<span class=\"citation\" data-cites=\"sec:method\">"
        "[@sec:method]</span>'"
    )
    double_quoted = (
        'data-double="<span class=\'citation\' data-cites=\'sec:method\'>'
        '[@sec:method]</span>"'
    )
    (web / "index.html").write_text(
        '<h2 data-number="3" id="sec:method">Method</h2>'
        f"<div {single_quoted} {double_quoted}>Attribute templates.</div>"
        '<div data-aria-label="[sec:method]">Named data attribute.</div>'
        '<div aria-label="See [sec:method]">Double quoted name.</div>'
        "<div aria-label='See [sec:method]'>Single quoted name.</div>"
        "<span data-cites='sec:method' class='citation'>[@sec:method]</span>",
        encoding="utf-8",
    )

    replacements = normalize_web_xrefs(tmp_path)
    rendered = (web / "index.html").read_text(encoding="utf-8")

    assert replacements == 3
    assert single_quoted in rendered
    assert double_quoted in rendered
    assert 'data-aria-label="[sec:method]"' in rendered
    assert 'aria-label="See Section 3"' in rendered
    assert "aria-label='See Section 3'" in rendered
    assert '<a class="xref" href="#sec:method">Section 3</a>' in rendered


def test_normalize_and_validate_share_protected_context_boundaries(
    tmp_path: Path,
) -> None:
    page = _write_valid_link_page(
        tmp_path,
        '<h2 data-number="3" id="sec:method">Method</h2>'
        '<!-- aria-label="Comment [sec:method]" and [sec:method] -->'
        '<ScRiPt data-probe=">">const ref = "[sec:method]";</sCrIpT>'
        '<STYLE>.probe::after { content: "[sec:method]"; }</style>'
        '<TeXtArEa>[sec:method]</tExTaReA>'
        '<div data-note="[sec:method]" aria-label="See [sec:method]">'
        "Visible [sec:method]</div>",
    )
    original = page.read_text(encoding="utf-8").replace(
        "<title>Links</title>",
        "<TiTlE>Links [sec:method]</tItLe>",
    )
    page.write_text(original, encoding="utf-8")

    replacements = normalize_web_xrefs(tmp_path)
    rendered = page.read_text(encoding="utf-8")
    result = validate_web_package(tmp_path)

    assert replacements == 2
    assert '<!-- aria-label="Comment [sec:method]" and [sec:method] -->' in rendered
    assert '<ScRiPt data-probe=">">const ref = "[sec:method]";</sCrIpT>' in rendered
    assert '<STYLE>.probe::after { content: "[sec:method]"; }</style>' in rendered
    assert '<TeXtArEa>[sec:method]</tExTaReA>' in rendered
    assert '<TiTlE>Links [sec:method]</tItLe>' in rendered
    assert 'data-note="[sec:method]"' in rendered
    assert 'aria-label="See Section 3"' in rendered
    assert '<a class="xref" href="#sec:method">Section 3</a>' in rendered
    assert result.ok
    assert not result.raw_xrefs


@pytest.mark.parametrize(
    "unterminated",
    [
        '<!-- protected [sec:method]',
        '<ScRiPt data-probe=">">const ref = "[sec:method]";',
        '<STYLE data-probe=">">[sec:method]',
        '<TiTlE data-probe=">">[sec:method]',
        '<TeXtArEa data-probe=">">[sec:method]',
    ],
)
def test_normalize_web_xrefs_rejects_unterminated_protected_context_before_writing(
    tmp_path: Path,
    unterminated: str,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    index = web / "index.html"
    broken = web / "z-broken.html"
    index_text = (
        '<h2 data-number="3" id="sec:method">Method</h2>'
        "Visible [sec:method]"
    )
    index.write_text(index_text, encoding="utf-8")
    broken.write_text(unterminated, encoding="utf-8")

    with pytest.raises(ValueError, match="unterminated"):
        normalize_web_xrefs(tmp_path)

    assert index.read_text(encoding="utf-8") == index_text
    assert broken.read_text(encoding="utf-8") == unterminated


def test_validate_web_package_reports_unterminated_protected_context(
    tmp_path: Path,
) -> None:
    _write_minimal_reader_without_registry(
        tmp_path,
        body_prefix='<script data-probe=">">const ref = "[sec:method]";',
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "unterminated <script> raw-text element" in issue
        for issue in result.malformed_markup
    )


def test_normalize_web_xrefs_avoids_nested_links_in_anchor_and_button(
    tmp_path: Path,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        '<h2 data-number="3" id="sec:method">Method</h2>'
        '<a href="#sec:method">[sec:method]</a>'
        '<button>[sec:method]</button>'
        '<a href="#other"><span class="citation" '
        'data-cites="sec:method">[@sec:method]</span></a>',
        encoding="utf-8",
    )

    replacements = normalize_web_xrefs(tmp_path)
    rendered = (web / "index.html").read_text(encoding="utf-8")

    assert replacements == 3
    assert '<a href="#sec:method"><span class="xref">Section 3</span></a>' in rendered
    assert '<button><span class="xref">Section 3</span></button>' in rendered
    assert '<a href="#other"><span class="xref">Section 3</span></a>' in rendered
    assert '<a href="#sec:method"><a ' not in rendered
    assert "<button><a " not in rendered


def test_normalize_web_xrefs_has_wrapper_and_table_page_target_parity(
    tmp_path: Path,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    reference = (
        '<div class="wrapper"><table><tbody><tr><td>'
        '<span class="citation" data-cites="sec:method">[@sec:method]</span>'
        "</td></tr></tbody></table></div>"
    )
    (web / "index.html").write_text(
        '<h2 data-number="3" id="sec:method">Method</h2>' + reference,
        encoding="utf-8",
    )
    (web / "manuscript__table.html").write_text(reference, encoding="utf-8")

    replacements = normalize_web_xrefs(tmp_path)
    combined = (web / "index.html").read_text(encoding="utf-8")
    section = (web / "manuscript__table.html").read_text(encoding="utf-8")

    assert replacements == 2
    assert '<a class="xref" href="#sec:method">Section 3</a>' in combined
    assert '<a class="xref" href="index.html#sec:method">Section 3</a>' in section


def test_normalize_web_xrefs_uses_page_relative_root_index_targets(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    nested = web / "nested" / "deeper"
    nested.mkdir(parents=True)
    (web / "index.html").write_text(
        '<h2 data-number="3" id="sec:method">Method</h2>',
        encoding="utf-8",
    )
    page = nested / "page.html"
    nested_index = nested / "index.html"
    page.write_text("See [sec:method].", encoding="utf-8")
    nested_index.write_text("See [sec:method].", encoding="utf-8")

    replacements = normalize_web_xrefs(tmp_path)

    assert replacements == 2
    assert 'href="../../index.html#sec:method"' in page.read_text(encoding="utf-8")
    assert 'href="../../index.html#sec:method"' in nested_index.read_text(encoding="utf-8")


def test_normalize_web_xrefs_rejects_symlinked_html_without_writing_target(
    tmp_path: Path,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    outside = tmp_path / "outside.html"
    original = '<h2 data-number="3" id="sec:method">Method</h2>See [sec:method].'
    outside.write_text(original, encoding="utf-8")
    (web / "index.html").symlink_to(outside)

    with pytest.raises(ValueError, match="must not be a symlink"):
        normalize_web_xrefs(tmp_path)

    assert outside.read_text(encoding="utf-8") == original


def test_validate_web_package_rejects_symlinked_html(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    outside = tmp_path / "outside.html"
    outside.write_text("<html></html>", encoding="utf-8")
    (web / "index.html").symlink_to(outside)

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("HTML input must not be a symlink" in issue for issue in result.malformed_markup)
    assert outside.read_text(encoding="utf-8") == "<html></html>"


def test_normalize_web_xrefs_preflights_every_destination_before_writing(
    tmp_path: Path,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    index = web / "index.html"
    first = web / "a.html"
    blocked = web / "z.html"
    index.write_text('<h2 data-number="3" id="sec:method">Method</h2>', encoding="utf-8")
    first.write_text("First [sec:method].", encoding="utf-8")
    blocked.write_text("Blocked [sec:method].", encoding="utf-8")
    blocked.chmod(0o444)
    try:
        with pytest.raises(PermissionError, match="is not writable"):
            normalize_web_xrefs(tmp_path)
    finally:
        blocked.chmod(0o644)

    assert first.read_text(encoding="utf-8") == "First [sec:method]."
    assert blocked.read_text(encoding="utf-8") == "Blocked [sec:method]."
    assert not tuple(web.glob(".*.xref-*.tmp"))


def test_normalize_web_xrefs_rechecks_each_destination_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    first = web / "a.html"
    index = web / "index.html"
    later = web / "z.html"
    first_original = b"First [sec:method]."
    later_original = b"Later [sec:method]."
    concurrent_edit = b"Concurrent edit that must be preserved."
    first.write_bytes(first_original)
    index.write_text(
        '<h2 data-number="3" id="sec:method">Method</h2>',
        encoding="utf-8",
    )
    later.write_bytes(later_original)

    real_link = os.link
    forward_links = 0

    def link_with_concurrent_edit(
        source: str | Path,
        destination: str | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal forward_links
        source_path = Path(source)
        real_link(source, destination, follow_symlinks=follow_symlinks)
        if "xref-replacement" in source_path.name:
            forward_links += 1
            if forward_links == 1:
                later.write_bytes(concurrent_edit)

    monkeypatch.setattr(web_package.os, "link", link_with_concurrent_edit)

    with pytest.raises(
        RuntimeError,
        match=r"z\.html: HTML input changed during normalization",
    ):
        normalize_web_xrefs(tmp_path)

    assert first.read_bytes() == first_original
    assert later.read_bytes() == concurrent_edit
    assert not tuple(web.glob(".*.xref-*.tmp"))


def test_normalize_web_xrefs_rolls_back_when_replace_raises_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    page = web / "index.html"
    original = b'<h2 data-number="3" id="sec:method">Method</h2>See [sec:method].'
    page.write_bytes(original)

    real_link = os.link
    interrupted = False

    def link_then_interrupt(
        source: str | Path,
        destination: str | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal interrupted
        source_path = Path(source)
        real_link(source, destination, follow_symlinks=follow_symlinks)
        if not interrupted and "xref-replacement" in source_path.name:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(web_package.os, "link", link_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        normalize_web_xrefs(tmp_path)

    assert page.read_bytes() == original
    assert not tuple(web.glob(".*.xref-*.tmp"))


def test_normalize_web_xrefs_preserves_concurrent_edit_and_recovery_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    first = web / "a.html"
    later = web / "b.html"
    index = web / "index.html"
    first_original = b"First [sec:method]."
    later_original = b"Later [sec:method]."
    concurrent_edit = b"External edit after the first commit."
    first.write_bytes(first_original)
    later.write_bytes(later_original)
    index.write_text(
        '<h2 data-number="3" id="sec:method">Method</h2>',
        encoding="utf-8",
    )

    real_link = os.link

    def link_then_edit_or_fail(
        source: str | Path,
        destination: str | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if "xref-replacement" in source_path.name and destination_path == later:
            raise OSError("forced second commit failure")
        real_link(source, destination, follow_symlinks=follow_symlinks)
        if "xref-replacement" in source_path.name and destination_path == first:
            first.write_bytes(concurrent_edit)

    monkeypatch.setattr(web_package.os, "link", link_then_edit_or_fail)

    with pytest.raises(
        RuntimeError,
        match="destination changed concurrently after commit",
    ):
        normalize_web_xrefs(tmp_path)

    assert first.read_bytes() == concurrent_edit
    assert later.read_bytes() == later_original
    replacement_recovery = tuple(web.glob(".a.html.xref-replacement-*.tmp"))
    rollback_recovery = tuple(web.glob(".a.html.xref-rollback-*.tmp"))
    assert len(replacement_recovery) == 1
    assert len(rollback_recovery) == 1
    assert replacement_recovery[0].read_bytes() == concurrent_edit
    assert rollback_recovery[0].read_bytes() == first_original
    assert set(web.glob(".*.xref-*.tmp")) == {
        replacement_recovery[0],
        rollback_recovery[0],
    }


def test_normalize_web_xrefs_captures_edit_after_freshness_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    page = web / "index.html"
    original = b'<h2 data-number="3" id="sec:method">Method</h2>See [sec:method].'
    concurrent_edit = b"Edit injected between freshness check and capture."
    page.write_bytes(original)

    real_replace = os.replace
    injected = False

    def edit_before_capture(source: str | Path, destination: str | Path) -> None:
        nonlocal injected
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not injected
            and source_path == page
            and "xref-rollback" in destination_path.name
        ):
            injected = True
            page.write_bytes(concurrent_edit)
        real_replace(source, destination)

    monkeypatch.setattr(web_package.os, "replace", edit_before_capture)

    with pytest.raises(RuntimeError, match="HTML input changed during normalization"):
        normalize_web_xrefs(tmp_path)

    assert page.read_bytes() == concurrent_edit
    assert not tuple(web.glob(".*.xref-*.tmp"))


@pytest.mark.parametrize("interruption_stage", ["rollback_move", "rollback_link"])
def test_normalize_web_xrefs_retains_recovery_after_rollback_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption_stage: str,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    first = web / "a.html"
    later = web / "b.html"
    index = web / "index.html"
    first_original = b"First [sec:method]."
    later_original = b"Later [sec:method]."
    first.write_bytes(first_original)
    later.write_bytes(later_original)
    index.write_text(
        '<h2 data-number="3" id="sec:method">Method</h2>',
        encoding="utf-8",
    )

    real_replace = os.replace
    real_link = os.link
    interrupted = False

    def replace_with_rollback_interrupt(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        nonlocal interrupted
        source_path = Path(source)
        destination_path = Path(destination)
        real_replace(source, destination)
        if (
            not interrupted
            and interruption_stage == "rollback_move"
            and source_path == first
            and "xref-replacement" in destination_path.name
        ):
            interrupted = True
            raise KeyboardInterrupt

    def link_with_failure_or_interrupt(
        source: str | Path,
        destination: str | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal interrupted
        source_path = Path(source)
        destination_path = Path(destination)
        if "xref-replacement" in source_path.name and destination_path == later:
            raise OSError("force rollback after first commit")
        real_link(source, destination, follow_symlinks=follow_symlinks)
        if (
            not interrupted
            and interruption_stage == "rollback_link"
            and "xref-rollback" in source_path.name
            and destination_path == first
        ):
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(web_package.os, "replace", replace_with_rollback_interrupt)
    monkeypatch.setattr(web_package.os, "link", link_with_failure_or_interrupt)

    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        normalize_web_xrefs(tmp_path)

    assert interrupted
    assert first.read_bytes() == first_original
    assert later.read_bytes() == later_original
    replacement_recovery = tuple(web.glob(".a.html.xref-replacement-*.tmp"))
    rollback_recovery = tuple(web.glob(".a.html.xref-rollback-*.tmp"))
    assert len(replacement_recovery) == 1
    assert len(rollback_recovery) == 1
    assert rollback_recovery[0].read_bytes() == first_original


def test_normalize_web_xrefs_preserves_crlf_outside_replacement(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    page = web / "index.html"
    page.write_bytes(
        b'<h2 data-number="3" id="sec:method">Method</h2>\r\n'
        b"See [sec:method].\r\n"
    )

    normalize_web_xrefs(tmp_path)

    rendered = page.read_bytes()
    assert rendered.count(b"\r\n") == 2
    assert b'href="#sec:method"' in rendered


def test_normalize_web_xrefs_uses_only_actual_id_attributes(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    page = web / "index.html"
    page.write_text(
        "<h2 id='sec:method' data-number='3'>Method</h2>"
        "<script>const fake = 'id=\"sec:fake\"';</script>"
        "See [sec:method] and [sec:fake].",
        encoding="utf-8",
    )

    normalize_web_xrefs(tmp_path)
    rendered = page.read_text(encoding="utf-8")

    assert '<a class="xref" href="#sec:method">Section 3</a>' in rendered
    assert '<span class="xref">Section fake</span>' in rendered
    assert 'href="#sec:fake"' not in rendered


def test_validate_web_package_ignores_fragment_ids_in_protected_text(tmp_path: Path) -> None:
    _write_valid_link_page(
        tmp_path,
        "<script>const fake = 'id=\"sec:fake\"';</script>"
        '<a href="#sec:fake">Fake target</a>',
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("#sec:fake (missing fragment)" in issue for issue in result.broken_xrefs)


@pytest.mark.parametrize(
    "tag",
    ["iframe", "noembed", "noframes", "xmp"],
)
def test_normalize_web_xrefs_preserves_all_bounded_raw_text_elements(
    tmp_path: Path,
    tag: str,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    page = web / "index.html"
    original = (
        '<h2 data-number="3" id="sec:method">Method</h2>'
        f"<{tag}>[sec:method]</{tag}>"
    )
    page.write_text(original, encoding="utf-8")

    assert normalize_web_xrefs(tmp_path) == 0
    assert page.read_text(encoding="utf-8") == original


def test_normalize_web_xrefs_preserves_plaintext_through_eof(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    page = web / "index.html"
    original = (
        '<h2 data-number="3" id="sec:method">Method</h2>'
        "<plaintext>[sec:method]</plaintext>[sec:method]"
    )
    page.write_text(original, encoding="utf-8")

    assert normalize_web_xrefs(tmp_path) == 0
    assert page.read_text(encoding="utf-8") == original


def test_normalize_web_xrefs_preserves_structural_reference_whitespace(
    tmp_path: Path,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    page = web / "index.html"
    page.write_text(
        '<h2 data-number="3" id="sec:method">Method</h2>'
        'See<span class="citation" data-cites="sec:method"> '
        "[@sec:method] </span>next.",
        encoding="utf-8",
    )

    normalize_web_xrefs(tmp_path)
    rendered = page.read_text(encoding="utf-8")

    assert "See <a" in rendered
    assert "</a> next." in rendered


def test_normalize_web_xrefs_resolves_entity_encoded_visible_references(
    tmp_path: Path,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    page = web / "index.html"
    page.write_text(
        '<h2 data-number="3" id="sec:method">Method</h2>'
        '<div aria-label="See &#91;sec:method&#93;">'
        "See &lbrack;sec:method&rbrack;.</div>",
        encoding="utf-8",
    )

    assert normalize_web_xrefs(tmp_path) == 2
    rendered = page.read_text(encoding="utf-8")

    assert 'aria-label="See Section 3"' in rendered
    assert '<a class="xref" href="#sec:method">Section 3</a>' in rendered


def test_normalize_web_xrefs_fails_closed_without_web_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing web output directory"):
        normalize_web_xrefs(tmp_path)


def test_validate_web_package_scopes_text_scans_away_from_protected_contexts(
    tmp_path: Path,
) -> None:
    _write_valid_link_page(
        tmp_path,
        '<script>const source = \'src="missing.png" {{FAKE_TOKEN}}\';</script>'
        '<style>.probe::after { content: "](/figures/a.png){#fig:x"; }</style>'
        "<!-- {{FAKE_TOKEN}} -->",
    )

    result = validate_web_package(tmp_path)

    assert result.ok
    assert not result.missing_assets
    assert not result.malformed_markup


@pytest.mark.parametrize(
    "markup",
    [
        '<iframe src="javascript:alert(1)" title="probe"></iframe>',
        '<script src="data:text/javascript,alert(1)"></script>',
        '<img src="data:text/html,<svg onload=alert(1)>" alt="probe">',
        '<video poster="file:///private/preview.png"></video>',
        '<img srcset="blob:https://example.org/id 2x" alt="probe">',
    ],
)
def test_validate_web_package_rejects_unsafe_resource_schemes(
    tmp_path: Path,
    markup: str,
) -> None:
    _write_valid_link_page(tmp_path, markup)

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("unsupported resource scheme" in issue for issue in result.malformed_markup)
    assert any(":1:" in issue for issue in result.malformed_markup)


@pytest.mark.parametrize(
    ("markup", "expected_issue"),
    [
        (
            '<img src="../../outside.png" alt="probe">',
            "leaves the shipped output tree",
        ),
        ('<img src="/inside.png" alt="probe">', "absolute resource URL is not allowed"),
        (
            '<img src="\nhttps://cdn.example.org/image.png" alt="probe">',
            "ASCII control character",
        ),
    ],
)
def test_validate_web_package_rejects_noncanonical_resource_paths(
    tmp_path: Path,
    markup: str,
    expected_issue: str,
) -> None:
    (tmp_path / "outside.png").write_bytes(b"outside")
    _write_valid_link_page(tmp_path, markup)

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(expected_issue in issue for issue in result.malformed_markup)


def test_validate_web_package_rejects_symlinked_resource_target(tmp_path: Path) -> None:
    _write_valid_link_page(tmp_path, '<img src="alias.png" alt="probe">')
    web = tmp_path / "output" / "web"
    (web / "real.png").write_bytes(b"real")
    (web / "alias.png").symlink_to(web / "real.png")

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "resource target has a symlinked path component" in issue
        for issue in result.malformed_markup
    )


def test_validate_web_package_requires_integrity_for_external_scripts(tmp_path: Path) -> None:
    _write_valid_link_page(
        tmp_path,
        '<script src="https://cdn.example.org/library.js"></script>',
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "external script must declare a valid SHA-2 integrity value" in issue
        for issue in result.malformed_markup
    )


def test_validate_web_package_uses_first_duplicate_script_integrity_attribute(
    tmp_path: Path,
) -> None:
    _write_valid_link_page(
        tmp_path,
        '<script src="https://cdn.example.org/library.js" integrity="invalid" '
        'integrity="sha384-YWJjZA==" crossorigin="anonymous"></script>',
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "external script must declare a valid SHA-2 integrity value" in issue
        for issue in result.malformed_markup
    )


@pytest.mark.parametrize(
    ("markup", "expected_ok"),
    [
        (
            '<script src="local.js" src="javascript:alert(1)"></script>',
            True,
        ),
        (
            '<script src="javascript:alert(1)" src="local.js"></script>',
            False,
        ),
        (
            '<video poster="local.png" poster="javascript:alert(1)"></video>',
            True,
        ),
        (
            '<video poster="javascript:alert(1)" poster="local.png"></video>',
            False,
        ),
    ],
)
def test_validate_web_package_uses_first_duplicate_resource_attribute(
    tmp_path: Path,
    markup: str,
    expected_ok: bool,
) -> None:
    page = _write_valid_link_page(tmp_path, markup)
    (page.parent / "local.js").write_text("// local\n", encoding="utf-8")
    (page.parent / "local.png").write_bytes(b"png")

    result = validate_web_package(tmp_path)

    assert result.ok is expected_ok
    if not expected_ok:
        assert any("unsupported resource scheme" in issue for issue in result.malformed_markup)


def test_validate_web_package_keeps_non_ascii_resource_name_distinct(
    tmp_path: Path,
) -> None:
    _write_valid_link_page(
        tmp_path,
        '<script ſrc="local.js" src="https://cdn.example.org/library.js"></script>',
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "external script must declare a valid SHA-2 integrity value" in issue
        for issue in result.malformed_markup
    )


def test_validate_web_package_does_not_treat_non_ascii_space_as_attribute_separator(
    tmp_path: Path,
) -> None:
    integrity = f"sha384-{'A' * 64}"
    _write_valid_link_page(
        tmp_path,
        '<script src="https://cdn.example.org/library.js"'
        "\u00a0"
        f'integrity="{integrity}" '
        'crossorigin="anonymous"></script>',
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "external script must declare a valid SHA-2 integrity value" in issue
        for issue in result.malformed_markup
    )


def test_validate_web_package_rejects_external_document_base_script_bypass(
    tmp_path: Path,
) -> None:
    page = _write_valid_link_page(
        tmp_path,
        '<base href="https://cdn.example.org/">'
        '<script src="local.js"></script>',
    )
    (page.parent / "local.js").write_text("// local decoy\n", encoding="utf-8")

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "document <base> elements are not allowed because they alter resource resolution"
        in issue
        for issue in result.malformed_markup
    )


@pytest.mark.parametrize(
    "base_markup",
    [
        '<base href="./assets/">',
        '<base href="./assets/" href="https://cdn.example.org/">',
        '<base href="https://cdn.example.org/" href="./assets/">',
        "<base>",
        "<base href>",
        '<base/href="https://cdn.example.org/">',
        '<BASE HREF="./assets/">',
    ],
)
def test_validate_web_package_rejects_every_document_base_variant(
    tmp_path: Path,
    base_markup: str,
) -> None:
    _write_valid_link_page(tmp_path, base_markup)

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("document <base> elements are not allowed" in issue for issue in result.malformed_markup)


def test_validate_web_package_ignores_base_like_protected_text(tmp_path: Path) -> None:
    _write_valid_link_page(
        tmp_path,
        '<script>const example = \'<base href="https://cdn.example.org/">\';</script>'
        '<!-- <base href="https://cdn.example.org/"> -->',
    )

    result = validate_web_package(tmp_path)

    assert result.ok


@pytest.mark.parametrize(
    ("markup", "expected_issue"),
    [
        (
            '<![CDATA[><base href="https://evil.example/">]]>'
            '<script src="local.js"></script>',
            "CDATA sections are not allowed in static text/html",
        ),
        (
            '<!--x--!><base href="https://evil.example/"><!-- -->'
            '<script src="local.js"></script>',
            "document <base> elements are not allowed",
        ),
        (
            '<script>0</script ignored><base href="https://evil.example/">'
            '<script src="local.js"></script>',
            "document <base> elements are not allowed",
        ),
        (
            '<div data=x"><base href="https://evil.example/">">'
            '<script src="local.js"></script>',
            "document <base> elements are not allowed",
        ),
        (
            '<script>0</script data=x"><base href="https://evil.example/>">'
            '<script src="local.js"></script>',
            "document <base> elements are not allowed",
        ),
        (
            '<noscript><base href="https://evil.example/"></noscript>'
            '<script src="local.js"></script>',
            "<noscript> elements are not allowed",
        ),
    ],
)
def test_validate_web_package_rejects_ambiguous_protected_context_bypasses(
    tmp_path: Path,
    markup: str,
    expected_issue: str,
) -> None:
    page = _write_valid_link_page(tmp_path, markup)
    (page.parent / "local.js").write_text("// local decoy\n", encoding="utf-8")

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(expected_issue in issue for issue in result.malformed_markup)


@pytest.mark.parametrize(
    ("markup", "expected_issue"),
    [
        (
            '<script =src="local.js" '
            'src="https://cdn.example.org/library.js"></script>',
            "external script must declare a valid SHA-2 integrity value",
        ),
        (
            '<script src="https://cdn.example.org/library.js" '
            f'=integrity="sha384-{"A" * 64}" integrity="invalid" '
            'crossorigin="anonymous"></script>',
            "external script must declare a valid SHA-2 integrity value",
        ),
        (
            '<script src="https://cdn.example.org/library.js" '
            f'integrity="sha384-{"A" * 64}" '
            '=crossorigin="anonymous" crossorigin="use-credentials"></script>',
            "external script with integrity must use crossorigin='anonymous'",
        ),
    ],
)
def test_validate_web_package_preserves_unexpected_leading_equals_in_attribute_names(
    tmp_path: Path,
    markup: str,
    expected_issue: str,
) -> None:
    page = _write_valid_link_page(tmp_path, markup)
    (page.parent / "local.js").write_text("// local decoy\n", encoding="utf-8")

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(expected_issue in issue for issue in result.malformed_markup)


def test_validate_web_package_preserves_effective_srcset_candidates(
    tmp_path: Path,
) -> None:
    page = _write_valid_link_page(
        tmp_path,
        '<img srcset="one.png 1x, two.png 2x" '
        'srcset="javascript:alert(1) 1x" alt="set">',
    )
    (page.parent / "one.png").write_bytes(b"one")
    (page.parent / "two.png").write_bytes(b"two")

    result = validate_web_package(tmp_path)

    assert result.ok
    assert result.assets_checked == len(COMPLEX_FIGURE_GENERATORS) + 2


@pytest.mark.parametrize(
    "integrity",
    [
        f"sha256-{'A' * 43}",
        f"sha256-{'A' * 43}=",
        f"sha384-{'A' * 64}",
        f"sha512-{'A' * 86}",
        f"sha512-{'A' * 86}==",
    ],
)
def test_validate_web_package_accepts_integrity_bound_https_script(
    tmp_path: Path,
    integrity: str,
) -> None:
    _write_valid_link_page(
        tmp_path,
        '<script src="https://cdn.example.org/library.js" '
        f'integrity="{integrity}" crossorigin="anonymous"></script>',
    )

    result = validate_web_package(tmp_path)

    assert result.ok


def test_validate_web_package_accepts_valid_mixed_sri_metadata(tmp_path: Path) -> None:
    integrity = (
        f"  sha256-{'A' * 43}=\t"
        f"sha384-{'A' * 64}\n"
        f"sha512-{'A' * 86}==  "
    )
    _write_valid_link_page(
        tmp_path,
        '<script src="https://cdn.example.org/library.js" '
        f'integrity="{integrity}" crossorigin="anonymous"></script>',
    )

    result = validate_web_package(tmp_path)

    assert result.ok


@pytest.mark.parametrize(
    "integrity",
    [
        "sha256-YWJjZA==",
        "sha384-YWJjZA==",
        "sha512-YWJjZA==",
        f"sha384-{'A' * 64}=",
        "sha384-not!base64",
    ],
)
def test_validate_web_package_rejects_invalid_sri_digest_bytes(
    tmp_path: Path,
    integrity: str,
) -> None:
    _write_valid_link_page(
        tmp_path,
        '<script src="https://cdn.example.org/library.js" '
        f'integrity="{integrity}" crossorigin="anonymous"></script>',
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "external script must declare a valid SHA-2 integrity value" in issue
        for issue in result.malformed_markup
    )


@pytest.mark.parametrize(
    "integrity",
    [
        f"sha384-{'A' * 64} sha256-YWJjZA==",
        f"sha384-{'A' * 64} sha1-{'A' * 27}=",
        f"sha384-{'A' * 64} sha512-{'A' * 86}===",
        f"sha384-{'A' * 64}\N{NO-BREAK SPACE}sha256-{'A' * 43}=",
        f"sha384-{'A' * 64} sha256-{'A' * 43}=?options",
    ],
)
def test_validate_web_package_rejects_invalid_mixed_sri_metadata(
    tmp_path: Path,
    integrity: str,
) -> None:
    _write_valid_link_page(
        tmp_path,
        '<script src="https://cdn.example.org/library.js" '
        f'integrity="{integrity}" crossorigin="anonymous"></script>',
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "external script must declare a valid SHA-2 integrity value" in issue
        for issue in result.malformed_markup
    )


@pytest.mark.parametrize(
    "crossorigin_attribute",
    ["crossorigin", 'crossorigin=""', "crossorigin=''", 'crossorigin="ANONYMOUS"'],
)
def test_validate_web_package_accepts_anonymous_crossorigin_spellings(
    tmp_path: Path,
    crossorigin_attribute: str,
) -> None:
    _write_valid_link_page(
        tmp_path,
        '<script src="https://cdn.example.org/library.js" '
        f'integrity="sha384-{"A" * 64}" {crossorigin_attribute}></script>',
    )

    result = validate_web_package(tmp_path)

    assert result.ok


@pytest.mark.parametrize(
    "crossorigin_attribute",
    ["", 'crossorigin="use-credentials"', 'crossorigin="other"'],
)
def test_validate_web_package_rejects_nonanonymous_crossorigin(
    tmp_path: Path,
    crossorigin_attribute: str,
) -> None:
    _write_valid_link_page(
        tmp_path,
        '<script src="https://cdn.example.org/library.js" '
        f'integrity="sha384-{"A" * 64}" {crossorigin_attribute}></script>',
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any(
        "external script with integrity must use crossorigin='anonymous'" in issue
        for issue in result.malformed_markup
    )


def test_validate_web_package_parses_single_quoted_unquoted_and_srcset_resources(
    tmp_path: Path,
) -> None:
    _write_valid_link_page(
        tmp_path,
        "<img src='missing-single.png' alt='single'>"
        "<img src=missing-unquoted.png alt=unquoted>"
        '<img srcset="missing-one.png 1x, missing-two.png 2x" alt="set">',
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    missing = "\n".join(result.missing_assets)
    assert "missing-single.png" in missing
    assert "missing-unquoted.png" in missing
    assert "missing-one.png" in missing
    assert "missing-two.png" in missing


def test_validate_web_package_rejects_embedded_iframe_srcdoc(tmp_path: Path) -> None:
    _write_valid_link_page(
        tmp_path,
        '<iframe title="probe" srcdoc="&lt;script&gt;alert(1)&lt;/script&gt;"></iframe>',
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("iframe srcdoc is not allowed" in issue for issue in result.malformed_markup)


def test_asset_and_href_resolution_cover_external_rooted_and_unsafe_paths(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    web = output / "web"
    nested = web / "nested"
    nested.mkdir(parents=True)
    page = nested / "page.html"
    page.write_text("<html></html>\n", encoding="utf-8")
    (web / "asset.png").write_bytes(b"png")

    assert _asset_path(web, page, "https://example.org/image.png") is None
    assert _asset_path(web, page, "#fragment") is None
    assert _asset_path(web, page, "?download=1") is None
    assert _asset_path(web, page, "/asset.png") == web / "asset.png"
    assert _asset_path(web, page, "../../../asset.png") == web / "asset.png"

    assert _local_href_target(output, page, "https://example.org") == (None, None)
    assert _local_href_target(output, page, "unsafe%00name")[1] == ("local href contains an unsafe path")
    assert _local_href_target(output, page, "unsafe\\name")[1] == ("local href contains an unsafe path")


def test_validation_reports_unclosed_figure_and_unreadable_fragment_target(
    tmp_path: Path,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    unreadable = web / "directory-target"
    unreadable.mkdir()
    (web / "index.html").write_text(
        '<!doctype html><html lang="en"><head><title>Edges</title></head><body>'
        '<a class="skip-link" href="#main-content">Skip to main content</a>'
        '<main id="main-content" tabindex="0">'
        '<a href="directory-target#section">Unreadable fragment</a>'
        '<figure><img alt="Described image">'
        "</main></body></html>",
        encoding="utf-8",
    )

    result = validate_web_package(tmp_path)

    assert not result.ok
    assert any("lack figcaption" in issue for issue in result.accessibility_issues)
    assert any("fragment target is unreadable" in issue for issue in result.broken_xrefs)
