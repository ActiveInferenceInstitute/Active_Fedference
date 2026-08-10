from __future__ import annotations

import posixpath
import re
import shutil
from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_XREF_PREFIXES = "sec|eq|fig|tbl|prop|thm|lem|cor|def"
_RAW_XREF_RE = re.compile(rf"\[@?(?P<key>(?:{_XREF_PREFIXES}):[^\]]+)\]")
_CROSSREF_SPAN_RE = re.compile(
    rf'<span class="citation"\s+data-cites="(?P<key>(?:{_XREF_PREFIXES}):[^"]+)">'
    r"\s*\[@?(?P=key)\]\s*</span>",
    re.DOTALL,
)
_LATEX_REF_HTML_RE = re.compile(
    r"(?:(?:Theorems?|Lemmas?|Propositions?|Corollaries|Definitions?|Sections?|Equations?)~?\s*)?"
    r'<span\s+class="math inline">\\\(\\ref\{(?P<key>[^}]+)\}\\\)</span>'
)
_DUPLICATE_XREF_KIND_RE = re.compile(
    r"(?:Theorem|Lemma|Proposition|Corollary|Definition|Section|Equation)~"
    r'(?=<span class="xref">)'
)
_BIB_ENTRY_RE = re.compile(
    r"^@(?!comment\b)\w+\s*\{\s*(?P<key>[^,\s]+),(?P<body>.*?)\n\}",
    re.DOTALL | re.MULTILINE,
)
_BIB_FIELD_RE = re.compile(r"^\s*(?P<field>\w+)\s*=\s*(?P<value>.+?),?\s*$")
_RAW_CITATION_RE = re.compile(
    r"\[(?P<keys>[A-Za-z][A-Za-z0-9_-]*(?:\s*;\s*[A-Za-z][A-Za-z0-9_-]*)*)\]"
)
_SRC_RE = re.compile(r"\bsrc=\"(?P<url>[^\"]+)\"")
_HREF_RE = re.compile(r"\bhref=\"(?P<url>[^\"]+)\"")
_LEAKED_FIGURE_RE = re.compile(r"\]\([^)]*figures/[^)]*\)\{#fig:")
_UNRESOLVED_TOKEN_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")
_PUBLICATION_TEXT_SUFFIXES = frozenset(
    {".csv", ".html", ".json", ".jsonl", ".log", ".md", ".svg", ".tex", ".txt", ".yaml", ".yml"}
)
_MACHINE_PATH_RE = re.compile(
    r"(?P<prefix>/private/tmp|/tmp|/Users|/home|/Volumes)/[^/\s\"'<>]+"
)


def sanitize_machine_paths(project_root: str | Path | None = None) -> tuple[Path, ...]:
    """Replace local home, temporary, and volume prefixes in text artifacts.

    TeX and renderer logs can contain absolute paths even though the published
    PDF/HTML does not depend on them.  Sanitising the committed text surfaces
    keeps reviewer snapshots clone-independent and prevents local workspace
    names from leaking into a release.  Binary figures and PDFs are untouched.
    """
    root = _root(project_root)
    output_dir = root / "output"
    if not output_dir.is_dir() or output_dir.is_symlink():
        return ()
    replacements = {
        "/private/tmp": "<tmp>",
        "/tmp": "<tmp>",
        "/Users": "<home>",
        "/home": "<home>",
        "/Volumes": "<volume>",
    }
    changed: list[Path] = []
    for path in sorted(output_dir.rglob("*")):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix.casefold() not in _PUBLICATION_TEXT_SUFFIXES
        ):
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        sanitized = _MACHINE_PATH_RE.sub(lambda match: replacements[match.group("prefix")], original)
        if path.suffix.casefold() == ".log":
            sanitized = sanitized.rstrip("\n") + "\n"
        if sanitized == original:
            continue
        path.write_text(sanitized, encoding="utf-8")
        changed.append(path)
    return tuple(changed)


@dataclass(frozen=True)
class WebPackageValidation:
    """Asset, reference, markup, and accessibility result for generated HTML."""

    html_files: int
    assets_checked: int
    missing_assets: tuple[str, ...]
    raw_xrefs: tuple[str, ...]
    broken_xrefs: tuple[str, ...] = ()
    malformed_markup: tuple[str, ...] = ()
    accessibility_issues: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether HTML exists and every package/accessibility check passes."""
        return (
            not self.missing_assets
            and not self.raw_xrefs
            and not self.broken_xrefs
            and not self.malformed_markup
            and not self.accessibility_issues
            and self.html_files > 0
        )


class _AccessibilityParser(HTMLParser):
    """Collect bounded structural accessibility facts from one HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.html_languages: list[str] = []
        self.titles: list[str] = []
        self._title_chunks: list[str] | None = None
        self.main_count = 0
        self.main_content_id_count = 0
        self.focusable_main_content_count = 0
        self.skip_link_count = 0
        self.first_interactive_seen = False
        self.first_interactive_is_skip = False
        self.missing_image_alt = 0
        self.unlabelled_full_size_links = 0
        self.duplicate_ids: set[str] = set()
        self._ids: set[str] = set()
        self._figure_stack: list[list[bool]] = []
        self.figures_missing_captions = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        classes = set((attributes.get("class") or "").split())
        is_skip_link = (
            tag == "a"
            and "skip-link" in classes
            and attributes.get("href") == "#main-content"
        )
        is_interactive = (
            (tag == "a" and bool(attributes.get("href")))
            or tag in {"button", "select", "textarea"}
            or (
                tag == "input"
                and (attributes.get("type") or "text").lower() != "hidden"
            )
        )
        if is_interactive and not self.first_interactive_seen:
            self.first_interactive_seen = True
            self.first_interactive_is_skip = is_skip_link
        if element_id:
            if element_id in self._ids:
                self.duplicate_ids.add(element_id)
            self._ids.add(element_id)

        if tag == "html":
            self.html_languages.append((attributes.get("lang") or "").strip())
        elif tag == "title":
            self._title_chunks = []
        elif tag == "main":
            self.main_count += 1
            if element_id == "main-content":
                self.main_content_id_count += 1
                if attributes.get("tabindex") in ("-1", "0"):
                    self.focusable_main_content_count += 1
        elif tag == "a":
            if is_skip_link:
                self.skip_link_count += 1
            if (
                "figure-full-size-link" in classes
                and not (attributes.get("aria-label") or "").strip()
            ):
                self.unlabelled_full_size_links += 1
        elif tag == "figure":
            self._figure_stack.append([False, False])
        elif tag == "img":
            if not (attributes.get("alt") or "").strip():
                self.missing_image_alt += 1
            if self._figure_stack:
                self._figure_stack[-1][0] = True
        elif tag == "figcaption" and self._figure_stack:
            self._figure_stack[-1][1] = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._title_chunks is not None:
            self.titles.append("".join(self._title_chunks).strip())
            self._title_chunks = None
        elif tag == "figure" and self._figure_stack:
            has_image, has_caption = self._figure_stack.pop()
            if has_image and not has_caption:
                self.figures_missing_captions += 1

    def handle_data(self, data: str) -> None:
        if self._title_chunks is not None:
            self._title_chunks.append(data)

    def close(self) -> None:
        super().close()
        while self._figure_stack:
            has_image, has_caption = self._figure_stack.pop()
            if has_image and not has_caption:
                self.figures_missing_captions += 1


def _accessibility_issues_in(path: Path, text: str) -> tuple[str, ...]:
    """Return reader-facing HTML accessibility contract violations."""
    parser = _AccessibilityParser()
    parser.feed(text)
    parser.close()

    issues: list[str] = []
    if len(parser.html_languages) != 1 or not parser.html_languages[0]:
        issues.append(f"{path}: expected one non-empty html lang attribute")
    if len(parser.titles) != 1 or not parser.titles[0]:
        issues.append(f"{path}: expected one non-empty document title")
    if parser.main_count != 1 or parser.main_content_id_count != 1:
        issues.append(f"{path}: expected one main element with id='main-content'")
    elif parser.focusable_main_content_count != 1:
        issues.append(
            f"{path}: main-content skip target must be focusable with tabindex"
        )
    if parser.skip_link_count != 1:
        issues.append(
            f"{path}: expected one .skip-link targeting '#main-content'"
        )
    elif not parser.first_interactive_is_skip:
        issues.append(f"{path}: skip link must be the first interactive element")
    if parser.missing_image_alt:
        issues.append(
            f"{path}: {parser.missing_image_alt} image(s) lack non-empty alt text"
        )
    if parser.figures_missing_captions:
        issues.append(
            f"{path}: {parser.figures_missing_captions} image figure(s) lack figcaption"
        )
    if parser.unlabelled_full_size_links:
        issues.append(
            f"{path}: {parser.unlabelled_full_size_links} full-size figure link(s) "
            "lack aria-label"
        )
    if parser.duplicate_ids:
        issues.append(
            f"{path}: duplicate id(s): {', '.join(sorted(parser.duplicate_ids))}"
        )
    return tuple(issues)


def _root(project_root: str | Path | None = None) -> Path:
    return Path(project_root) if project_root is not None else _PROJECT_ROOT


def _web_dir(project_root: Path) -> Path:
    return project_root / "output" / "web"


def mirror_web_figures(project_root: str | Path | None = None) -> tuple[Path, ...]:
    """Mirror every generated figure into the web package, removing stale files."""
    root = _root(project_root)
    source_dir = root / "output" / "figures"
    destination_dir = _web_dir(root) / "figures"
    if not source_dir.exists():
        raise FileNotFoundError(f"missing source figures directory: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    source_files = {
        path.relative_to(source_dir): path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file()
    }

    for existing in sorted(destination_dir.rglob("*"), reverse=True):
        if existing.is_file() and existing.relative_to(destination_dir) not in source_files:
            existing.unlink()
        elif existing.is_dir():
            try:
                existing.rmdir()
            except OSError:
                pass

    copied: list[Path] = []
    for relative_path, source_path in source_files.items():
        destination_path = destination_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied.append(destination_path)
    return tuple(copied)


def _xref_label(key: str, labels: dict[str, str]) -> str:
    if key in labels:
        return labels[key]
    prefix, value = key.split(":", 1)
    labels = {
        "sec": "Section",
        "eq": "Eq.",
        "fig": "Figure",
        "tbl": "Table",
        "prop": "Proposition",
        "thm": "Theorem",
        "lem": "Lemma",
        "cor": "Corollary",
        "def": "Definition",
    }
    return f"{labels[prefix]} {value}"


def _xref_labels(web_dir: Path) -> dict[str, str]:
    """Extract display labels from the numbered combined HTML manuscript."""
    index = web_dir / "index.html"
    if not index.exists():
        return {}
    text = index.read_text(encoding="utf-8")
    labels: dict[str, str] = {}
    for match in re.finditer(
        rf'<h[1-6][^>]*data-number="(?P<number>[^"]+)"[^>]*'
        rf'id="(?P<key>(?:{_XREF_PREFIXES}):[^"]+)"',
        text,
    ):
        labels[match.group("key")] = f"Section {match.group('number')}"
    for match in re.finditer(
        r'<(?P<tag>figure|table) id="(?P<key>(?:fig|tbl):[^"]+)">'
        r'.*?<(?:figcaption|caption)>(?P<kind>Figure|Table)\s+'
        r'(?P<number>\d+):',
        text,
        re.DOTALL,
    ):
        labels[match.group("key")] = f"{match.group('kind')} {match.group('number')}"
    for match in re.finditer(
        r'<span id="(?P<key>eq:[^"]+)">.*?\\qquad\{\((?P<number>\d+)\)\}',
        text,
        re.DOTALL,
    ):
        labels[match.group("key")] = f"Eq. ({match.group('number')})"
    aux = web_dir.parent / "pdf" / "_combined_manuscript.aux"
    if aux.exists():
        prefix_labels = {
            "eq": "Eq.",
            "prop": "Proposition",
            "thm": "Theorem",
            "lem": "Lemma",
            "cor": "Corollary",
            "def": "Definition",
        }
        for match in re.finditer(
            rf"\\newlabel\{{(?P<key>(?:{_XREF_PREFIXES}):[^}}]+)\}}"
            r"\{\{(?P<number>[^}]+)\}",
            aux.read_text(encoding="utf-8"),
        ):
            key = match.group("key")
            prefix = key.split(":", 1)[0]
            if prefix in prefix_labels:
                number = match.group("number")
                label = f"{prefix_labels[prefix]} {number}"
                labels.setdefault(key, label)
    return labels


def _xref_ids(web_dir: Path) -> set[str]:
    index = web_dir / "index.html"
    if not index.exists():
        return set()
    return set(re.findall(r'\bid="([^"]+)"', index.read_text(encoding="utf-8")))


def _clean_bib_value(value: str) -> str:
    stripped = value.strip().rstrip(",").strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith('"') and stripped.endswith('"')
    ):
        stripped = stripped[1:-1]
    stripped = re.sub(r"\\[`'\"^~=.]\{?([A-Za-z])\}?", r"\1", stripped)
    return stripped.replace("{", "").replace("}", "").replace("\\&", "&").strip()


def _bib_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in body.splitlines():
        match = _BIB_FIELD_RE.match(line)
        if match:
            fields[match.group("field").lower()] = _clean_bib_value(
                match.group("value")
            )
    return fields


def _surname(author: str) -> str:
    name = author.strip()
    if "," in name:
        return name.split(",", 1)[0].strip()
    parts = name.split()
    return parts[-1] if parts else name


def _author_year_label(author_field: str, year: str) -> str:
    authors = [part.strip() for part in re.split(r"\s+and\s+", author_field) if part.strip()]
    if not authors:
        return year
    if len(authors) == 1:
        author_text = _surname(authors[0])
    elif len(authors) == 2:
        author_text = f"{_surname(authors[0])} and {_surname(authors[1])}"
    else:
        author_text = f"{_surname(authors[0])} et al."
    return f"{author_text} {year}"


def _citation_labels(project_root: Path) -> dict[str, str]:
    references = project_root / "manuscript" / "references.bib"
    if not references.exists():
        references = project_root / "output" / "manuscript" / "references.bib"
    if not references.exists():
        return {}

    labels: dict[str, str] = {}
    text = references.read_text(encoding="utf-8")
    for entry in _BIB_ENTRY_RE.finditer(text):
        fields = _bib_fields(entry.group("body"))
        author = fields.get("author")
        year = fields.get("year")
        if author and year:
            labels[entry.group("key").strip()] = _author_year_label(author, year)
    return labels


def normalize_web_xrefs(project_root: str | Path | None = None) -> int:
    """Replace raw citation/cross-reference markup with self-contained HTML links."""
    root = _root(project_root)
    web_dir = _web_dir(root)
    if not web_dir.exists():
        raise FileNotFoundError(f"missing web output directory: {web_dir}")

    citation_labels = _citation_labels(root)
    xref_labels = _xref_labels(web_dir)
    xref_ids = _xref_ids(web_dir)
    replacements = 0
    for html_path in sorted(web_dir.rglob("*.html")):
        text = html_path.read_text(encoding="utf-8")

        def replace_span(match: re.Match[str]) -> str:
            key = match.group("key")
            label = escape(_xref_label(key, xref_labels))
            if key not in xref_ids:
                return f'<span class="xref">{label}</span>'
            target = f"#{key}" if html_path.name == "index.html" else f"index.html#{key}"
            return (
                f'<a class="xref" href="{escape(target, quote=True)}">'
                f"{label}</a>"
            )

        updated, span_count = _CROSSREF_SPAN_RE.subn(replace_span, text)

        def replace_latex_ref(match: re.Match[str]) -> str:
            key = match.group("key")
            label = escape(_xref_label(key, xref_labels))
            if key not in xref_ids:
                return f'<span class="xref">{label}</span>'
            target = f"#{key}" if html_path.name == "index.html" else f"index.html#{key}"
            return f'<a class="xref" href="{escape(target, quote=True)}">{label}</a>'

        updated, latex_count = _LATEX_REF_HTML_RE.subn(replace_latex_ref, updated)
        updated, duplicate_kind_count = _DUPLICATE_XREF_KIND_RE.subn("", updated)

        def replace_raw(match: re.Match[str]) -> str:
            key = match.group("key")
            label = escape(_xref_label(key, xref_labels))
            if key not in xref_ids:
                return f'<span class="xref">{label}</span>'
            target = f"#{key}" if html_path.name == "index.html" else f"index.html#{key}"
            return (
                f'<a class="xref" href="{escape(target, quote=True)}">'
                f"{label}</a>"
            )

        updated, raw_count = _RAW_XREF_RE.subn(replace_raw, updated)
        citation_count = 0

        def replace_citation(match: re.Match[str]) -> str:
            nonlocal citation_count
            keys = [key.strip() for key in match.group("keys").split(";")]
            if not keys or any(key not in citation_labels for key in keys):
                return match.group(0)
            citation_count += 1
            label = "; ".join(citation_labels[key] for key in keys)
            return f'<span class="citation">({escape(label)})</span>'

        updated = _RAW_CITATION_RE.sub(replace_citation, updated)
        if updated != text:
            html_path.write_text(updated, encoding="utf-8")
        replacements += (
            span_count
            + latex_count
            + duplicate_kind_count
            + raw_count
            + citation_count
        )
    return replacements


def _is_external(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.scheme) or url.startswith("#")


def _asset_path(web_dir: Path, html_path: Path, url: str) -> Path | None:
    if _is_external(url):
        return None
    clean_url = unquote(url.split("#", 1)[0].split("?", 1)[0])
    if not clean_url:
        return None
    if clean_url.startswith("/"):
        relative = clean_url.lstrip("/")
    else:
        html_relative_dir = html_path.relative_to(web_dir).parent.as_posix()
        relative = posixpath.normpath(posixpath.join(html_relative_dir, clean_url))
        while relative.startswith("../"):
            relative = relative[3:]
    return web_dir / relative


def _raw_xrefs_in(path: Path) -> tuple[str, ...]:
    offenders: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if _RAW_XREF_RE.search(line) or r"\ref{" in line:
            offenders.append(f"{path}:{line_number}")
    return tuple(offenders)


def validate_web_package(project_root: str | Path | None = None) -> WebPackageValidation:
    """Check generated HTML assets, links, markup, and accessibility structure."""
    root = _root(project_root)
    web_dir = _web_dir(root)
    if not web_dir.exists():
        return WebPackageValidation(
            html_files=0,
            assets_checked=0,
            missing_assets=(f"missing web output directory: {web_dir}",),
            raw_xrefs=(),
            broken_xrefs=(),
            malformed_markup=(),
            accessibility_issues=(),
        )

    html_files = sorted(web_dir.rglob("*.html"))
    missing: list[str] = []
    raw_xrefs: list[str] = []
    broken_xrefs: list[str] = []
    malformed_markup: list[str] = []
    accessibility_issues: list[str] = []
    assets_checked = 0
    for html_path in html_files:
        try:
            text = html_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            accessibility_issues.append(
                f"{html_path}: accessibility parse failed: {exc}"
            )
            malformed_markup.append(f"{html_path}: unreadable HTML: {exc}")
            continue
        accessibility_issues.extend(_accessibility_issues_in(html_path, text))
        raw_xrefs.extend(_raw_xrefs_in(html_path))
        if _LEAKED_FIGURE_RE.search(text):
            malformed_markup.append(f"{html_path}: leaked Markdown figure syntax")
        if _UNRESOLVED_TOKEN_RE.search(text):
            malformed_markup.append(f"{html_path}: unresolved manuscript token")
        for match in _SRC_RE.finditer(text):
            asset = _asset_path(web_dir, html_path, match.group("url"))
            if asset is None:
                continue
            assets_checked += 1
            if not asset.exists():
                missing.append(f"{html_path}: {match.group('url')} -> {asset}")
        for match in _HREF_RE.finditer(text):
            url = match.group("url")
            if url.startswith("#"):
                target_file = html_path
                fragment = url[1:]
            elif "#" in url and not urlparse(url).scheme:
                file_part, fragment = url.split("#", 1)
                target_file = html_path.parent / file_part
            else:
                continue
            if not target_file.exists():
                broken_xrefs.append(f"{html_path}: {url} (missing target file)")
                continue
            target_text = target_file.read_text(encoding="utf-8")
            if fragment and f'id="{fragment}"' not in target_text:
                broken_xrefs.append(f"{html_path}: {url} (missing fragment)")

    return WebPackageValidation(
        html_files=len(html_files),
        assets_checked=assets_checked,
        missing_assets=tuple(sorted(set(missing))),
        raw_xrefs=tuple(sorted(set(raw_xrefs))),
        broken_xrefs=tuple(sorted(set(broken_xrefs))),
        malformed_markup=tuple(sorted(set(malformed_markup))),
        accessibility_issues=tuple(sorted(set(accessibility_issues))),
    )
