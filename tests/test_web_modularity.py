"""Byte custody and adverse-input checks for the HTML publication boundary."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import pytest

from publication.web_package import normalize_web_xrefs, validate_web_package
from tests.test_web_publication_contract import (
    _observe_filesystem_calls,
    _write_valid_link_page,
)


class _LinkFacts(HTMLParser):
    """Accumulate link destinations from parsed HTML, independently of normalization."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href":
                    if value is not None:
                        self.links.append(value)
                    break


def test_html_bytes_and_browser_links_unchanged(tmp_path: Path) -> None:
    # Given: CRLF, encoded references, protected text and a nested page.
    page = _write_valid_link_page(
        tmp_path,
        '<h2 id="sec:method" data-number="3">Method</h2>\r\n'
        "<p>See &#91;sec:method&#93;.</p>\r\n"
        '<script>const sample = "[sec:method]";</script>',
    )
    nested = page.parent / "nested" / "page.html"
    nested.parent.mkdir()
    nested_original = b"<p>See [sec:method].</p>\r\n"
    nested.write_bytes(nested_original)
    original = page.read_bytes()

    # When: the public boundary normalizes the actual files.
    replacements = normalize_web_xrefs(tmp_path)

    # Then: only declared reference bytes change; parsed destinations resolve.
    expected = original.replace(b"&#91;sec:method&#93;", b'<a class="xref" href="#sec:method">Section 3</a>')
    assert replacements == 2
    assert page.read_bytes() == expected
    assert nested.read_bytes() == (
        b'<p>See <a class="xref" href="../index.html#sec:method">Section 3</a>.</p>\r\n'
    )
    facts = _LinkFacts()
    facts.feed(nested.read_bytes().decode())
    facts.close()
    assert facts.links == ["../index.html#sec:method"]
    target, fragment = facts.links[0].split("#")
    assert (nested.parent / target).resolve() == page.resolve()
    assert f'id="{fragment}"'.encode() in page.read_bytes()
    findings = validate_web_package(tmp_path)
    assert findings.broken_xrefs == ()
    assert findings.raw_xrefs == ()
    assert findings.malformed_markup == ()
    print(f"normalized_files=2 replacements={replacements} links={facts.links}")


@pytest.mark.parametrize("case", ["escape", "resource_symlink", "html_symlink", "changed_original"])
def test_resource_escape_and_changed_original(tmp_path: Path, case: str) -> None:
    # Given: a real unsafe path or a concurrent edit at the capture syscall.
    outside = tmp_path / "private.png"
    outside.write_bytes(b"private bytes")
    resource = "../../private.png" if case == "escape" else "alias.png"
    page = _write_valid_link_page(tmp_path, f'<img src="{resource}" alt="probe">')
    original = page.read_bytes()
    if case in {"escape", "resource_symlink"}:
        if case == "resource_symlink":
            (page.parent / "alias.png").symlink_to(outside)
        # When: the canonical validator inspects the unsafe resource.
        result = validate_web_package(tmp_path)
        # Then: exact findings reject it and validation writes nothing.
        reason = (
            "resource target leaves the shipped output tree"
            if case == "escape"
            else "resource target has a symlinked path component"
        )
        assert result.malformed_markup == (f"{page}:1: {resource} ({reason})",)
        assert not result.ok
        assert page.read_bytes() == original
        assert outside.read_bytes() == b"private bytes"
        print(result.malformed_markup)
        return
    if case == "html_symlink":
        alias = page.parent / "alias.html"
        alias.symlink_to(page)
        # When: normalization preflights a symlinked HTML input.
        with pytest.raises(ValueError, match="HTML input must not be a symlink") as caught:
            normalize_web_xrefs(tmp_path)
        # Then: the actual target is retained byte for byte.
        assert page.read_bytes() == original
        assert alias.is_symlink()
        print(str(caught.value))
        return

    page.write_bytes(b'<h2 id="sec:method" data-number="3">Method</h2>[sec:method]')
    concurrent = b"Caller changed the original after the freshness check."
    injected = False

    def edit_before_capture(source: Path, destination: Path, event: str) -> None:
        nonlocal injected
        if event == "c_call" and source == page and "xref-rollback" in destination.name:
            injected = True
            page.write_bytes(concurrent)

    # When: real bytes change immediately before the real capture syscall.
    with _observe_filesystem_calls(replace=edit_before_capture):
        with pytest.raises(RuntimeError, match="HTML input changed during normalization") as changed:
            normalize_web_xrefs(tmp_path)
    # Then: the write guard refuses the replacement and restores the caller's edit.
    assert injected
    assert page.read_bytes() == concurrent
    assert not tuple(page.parent.glob(".*.xref-*.tmp"))
    print(str(changed.value))


@pytest.mark.parametrize(
    "prefix",
    [
        "<div a / b = \"one\" c = 'two' d = bare e / >text</div>",
        "<div a = >text</div>",
        "<!-->",
        "<!--->",
        "<!---[sec:method]-->",
        "<!-- [sec:method] --x-->",
        "<!-- [sec:method] --!x-->",
        "<!-- [sec:method] --!-x-->",
        "<p>literal < comparison</p>",
    ],
)
def test_markup_state_boundaries_preserve_original_bytes(tmp_path: Path, prefix: str) -> None:
    # Given: valid tokenizer recovery states adjacent to a transformable reference.
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    page = web / "index.html"
    heading = '<h2 id="sec:method" data-number="3">Method</h2>'
    page.write_bytes((heading + prefix + "[sec:method]").encode())

    # When: normalization traverses the real lexical states.
    count = normalize_web_xrefs(tmp_path)

    # Then: protected contexts and original markup survive exactly.
    assert count == 1
    assert (
        page.read_bytes() == (heading + prefix + '<a class="xref" href="#sec:method">Section 3</a>').encode()
    )


@pytest.mark.parametrize("malformed", ["<!---", "<!-- x --", "<div a = 'unclosed", "<!DOCTYPE"])
def test_unbounded_markup_rejects_batch_before_writes(tmp_path: Path, malformed: str) -> None:
    # Given: an earlier eligible document and a later unbounded context.
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    first = web / "a.html"
    later = web / "z.html"
    first.write_bytes(b"[sec:method]")
    later.write_bytes(malformed.encode())

    # When: the boundary parses the complete batch.
    with pytest.raises(ValueError, match="unterminated HTML"):
        normalize_web_xrefs(tmp_path)

    # Then: neither earlier nor malformed source bytes are committed.
    assert first.read_bytes() == b"[sec:method]"
    assert later.read_bytes() == malformed.encode()
    assert not tuple(web.glob(".*.xref-*.tmp"))


@pytest.mark.parametrize(
    ("resource", "reason"),
    [
        ("", "resource URL is empty"),
        ("//example.org/a.png", "protocol-relative resource URL is not allowed"),
        ("bad%00name", "resource URL contains an unsafe path"),
        ("#fragment", "resource URL has no path"),
        ("folder", "resource target is not a regular file"),
    ],
)
def test_resource_boundary_rejects_nonfile_fetches(tmp_path: Path, resource: str, reason: str) -> None:
    # Given: a reader with a resource that cannot identify an admitted file.
    page = _write_valid_link_page(tmp_path, f'<img src="{resource}" alt="probe">')
    (page.parent / "folder").mkdir()
    original = page.read_bytes()

    # When: the single resource validator evaluates the fetch.
    result = validate_web_package(tmp_path)

    # Then: the resource is rejected with its exact diagnostic and no write.
    assert result.malformed_markup == (f"{page}:1: {resource} ({reason})",)
    assert not result.ok
    assert page.read_bytes() == original


@pytest.mark.parametrize("placement", ["web_symlink", "output_escape", "fifo"])
def test_html_path_custody_rejects_nonregular_or_unconfined_input(tmp_path: Path, placement: str) -> None:
    # Given: an outside original and an inadmissible HTML entry.
    import os

    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "index.html"
    target.write_bytes(b"[sec:method]")
    output = root / "output"
    if placement == "output_escape":
        output.symlink_to(outside, target_is_directory=True)
        (outside / "web").mkdir()
    else:
        output.mkdir()
        if placement == "web_symlink":
            (output / "web").symlink_to(outside, target_is_directory=True)
        else:
            (output / "web").mkdir()
            os.mkfifo(output / "web" / "pipe.html")

    # When: normalization inspects the filesystem before reading HTML.
    with pytest.raises(ValueError):
        normalize_web_xrefs(root)

    # Then: no outside original or FIFO is rewritten.
    assert target.read_bytes() == b"[sec:method]"
    assert not tuple(root.rglob(".*.xref-*.tmp"))


@pytest.mark.parametrize(
    ("markup", "key", "label"),
    [
        ('<h2 id="sec:plain">Unnumbered</h2>', "sec:plain", "Section plain"),
        (
            '<figure id="tbl:wrong"><figcaption>Figure 9: Wrong</figcaption></figure>',
            "tbl:wrong",
            "Table wrong",
        ),
        (
            '<figure id="fig:nested"><div><figcaption>Figure 9: Nested</figcaption></div></figure>',
            "fig:nested",
            "Figure nested",
        ),
        ('<figure id="fig:plain"><figcaption>Unnumbered</figcaption></figure>', "fig:plain", "Figure plain"),
        ('<span id="eq:plain"><span>Unnumbered</span></span>', "eq:plain", "Eq. plain"),
        ('<div id="sec:plain">Unnumbered</div>', "sec:plain", "Section plain"),
    ],
)
def test_reference_labels_do_not_invent_numbers(tmp_path: Path, markup: str, key: str, label: str) -> None:
    # Given: IDs with no admissible numbered caption or heading.
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    page = web / "index.html"
    page.write_text(f"{markup}[{key}]", encoding="utf-8")

    # When: normalization labels the cross-reference.
    count = normalize_web_xrefs(tmp_path)

    # Then: the valid target remains linked with the unnumbered fallback.
    assert count == 1
    assert page.read_text(encoding="utf-8") == f'{markup}<a class="xref" href="#{key}">{label}</a>'
