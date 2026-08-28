from __future__ import annotations

from pathlib import Path

import pytest

from publication.web_package import (
    _asset_path,
    _local_href_target,
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


def _write_valid_link_page(root: Path, links: str) -> Path:
    web = root / "output" / "web"
    web.mkdir(parents=True)
    page = web / "index.html"
    page.write_text(
        '<!doctype html><html lang="en"><head><title>Links</title></head><body>'
        '<a class="skip-link" href="#main-content">Skip to main content</a>'
        f'<main id="main-content" tabindex="-1">{links}</main></body></html>',
        encoding="utf-8",
    )
    return page


def test_validate_web_package_reports_missing_local_href(tmp_path: Path) -> None:
    _write_valid_link_page(
        tmp_path,
        '<a href="missing.json">Missing</a>'
        '<a href=\'also-missing.json\'>Also missing</a>'
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
        '<a href="evidence.json">Evidence</a>'
        '<a href="../figures/plot.png">Full-size figure</a>',
    )
    (page.parent / "evidence.json").write_text('{"status":"ok"}\n', encoding="utf-8")
    figures = tmp_path / "output" / "figures"
    figures.mkdir()
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
        "\\newlabel{thm:health}{{2}{7}}\n"
        "\\newlabel{sec:ignored}{{9}{9}}\n",
        encoding="utf-8",
    )
    (web / "index.html").write_text(
        '<h2 data-number="3" id="sec:method">Method</h2>'
        '<figure id="fig:result"><figcaption>Figure 4: Result.</figcaption></figure>'
        '<table id="tbl:result"><caption>Table 5: Result.</caption></table>'
        '<span id="eq:pool">x \\qquad{(6)}</span>'
        '<span class="citation" data-cites="sec:method">[@sec:method]</span> '
        '[fig:result] [tbl:result] [eq:pool] [thm:health] '
        '[single] [pair] [group] [unknown]',
        encoding="utf-8",
    )

    replacements = normalize_web_xrefs(tmp_path)
    rendered = (web / "index.html").read_text(encoding="utf-8")

    assert replacements == 8
    assert '>Section 3</a>' in rendered
    assert '>Figure 4</a>' in rendered
    assert '>Table 5</a>' in rendered
    assert '>Eq. (6)</a>' in rendered
    assert '<span class="xref">Theorem 2</span>' in rendered
    assert "(Lovelace 1843)" in rendered
    assert "(Hopper and Turing 1950)" in rendered
    assert "(One et al. 2026)" in rendered
    assert "[unknown]" in rendered


def test_normalize_web_xrefs_fails_closed_without_web_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing web output directory"):
        normalize_web_xrefs(tmp_path)


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
    assert _local_href_target(output, page, "unsafe%00name")[1] == (
        "local href contains an unsafe path"
    )
    assert _local_href_target(output, page, "unsafe\\name")[1] == (
        "local href contains an unsafe path"
    )


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
