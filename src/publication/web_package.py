from __future__ import annotations

import base64
import binascii
import json
import os
import posixpath
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

from analysis.report_schemas import ReportSchemaError, validate_report
from analysis.visual_contracts import COMPLEX_FIGURE_GENERATORS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_XREF_PREFIXES = "sec|eq|fig|tbl|prop|thm|lem|cor|def"
_LEFT_BRACKET_PATTERN = (
    r"(?:\[|&(?:lbrack|lsqb);|&#0*91(?:;|(?![0-9]))|&#[xX]0*5[bB](?:;|(?![0-9A-Fa-f])))"
)
_RIGHT_BRACKET_PATTERN = (
    r"(?:\]|&(?:rbrack|rsqb);|&#0*93(?:;|(?![0-9]))|&#[xX]0*5[dD](?:;|(?![0-9A-Fa-f])))"
)
_RAW_XREF_RE = re.compile(
    rf"{_LEFT_BRACKET_PATTERN}@?(?P<key>(?:{_XREF_PREFIXES}):.+?){_RIGHT_BRACKET_PATTERN}"
)
_LATEX_REF_TEXT_RE = re.compile(
    rf"\\\(\\ref\{{(?P<key>(?:{_XREF_PREFIXES}):[^}}]+)\}}\\\)"
)
_LATEX_KIND_SUFFIX_RE = re.compile(
    r"(?:Theorems?|Lemmas?|Propositions?|Corollaries|Definitions?|Sections?|Equations?)~?\s*$"
)
_DUPLICATE_XREF_KIND_SUFFIX_RE = re.compile(
    r"(?:Theorem|Lemma|Proposition|Corollary|Definition|Section|Equation)~\s*$"
)
_BIB_ENTRY_RE = re.compile(
    r"^@(?!comment\b)\w+\s*\{\s*(?P<key>[^,\s]+),(?P<body>.*?)\n\}",
    re.DOTALL | re.MULTILINE,
)
_BIB_FIELD_RE = re.compile(r"^\s*(?P<field>\w+)\s*=\s*(?P<value>.+?),?\s*$")
_RAW_CITATION_RE = re.compile(
    rf"{_LEFT_BRACKET_PATTERN}"
    r"(?P<keys>[A-Za-z][A-Za-z0-9_-]*(?:\s*;\s*[A-Za-z][A-Za-z0-9_-]*)*)"
    rf"{_RIGHT_BRACKET_PATTERN}"
)
_LEAKED_FIGURE_RE = re.compile(r"\]\([^)]*figures/[^)]*\)\{#fig:")
_UNRESOLVED_TOKEN_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")
_PUBLICATION_TEXT_SUFFIXES = frozenset(
    {".csv", ".html", ".json", ".jsonl", ".log", ".md", ".svg", ".tex", ".txt", ".yaml", ".yml"}
)
_MACHINE_PATH_RE = re.compile(r"(?P<prefix>/private/tmp|/tmp|/Users|/home|/Volumes)/[^/\s\"'<>]+")
_ALLOWED_EXTERNAL_HREF_SCHEMES = frozenset({"http", "https", "mailto", "tel"})
_CONTEXTUAL_FULL_SIZE_LABEL_RE = re.compile(
    r"Open full[- ]size (?:Figure\s+\d+|figure),\s*(?P<context>.+)",
    re.IGNORECASE,
)
_MARKDOWN_FRAGMENT_RE_TEMPLATE = r"\{{#{fragment}(?:\s|\}})"
_START_TAG_RE = re.compile(
    r"<(?P<name>[A-Za-z][A-Za-z0-9:._-]*)(?=[ \t\n\f\r/>])"
)
_END_TAG_RE = re.compile(r"</(?P<name>[A-Za-z][A-Za-z0-9:._-]*)(?=[ \t\n\f\r>])")
_PROTECTED_RAW_ELEMENTS = frozenset(
    {
        "iframe",
        "noembed",
        "noframes",
        "script",
        "style",
        "textarea",
        "title",
        "xmp",
    }
)
_SCRIPTING_DEPENDENT_ELEMENTS = frozenset({"noscript"})
_PLAINTEXT_ELEMENTS = frozenset({"plaintext"})
_VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_NO_NESTED_LINK_ANCESTORS = frozenset({"a", "button"})
_ALLOWED_EXTERNAL_RESOURCE_SCHEMES = frozenset({"https"})
_RESOURCE_ATTRIBUTES = frozenset({"poster", "src", "srcset"})
_SRI_DIGEST_BYTES = {"sha256": 32, "sha384": 48, "sha512": 64}
_SRI_VALUE_RE = re.compile(
    r"(?P<algorithm>sha(?:256|384|512))-(?P<digest>[A-Za-z0-9+/]+={0,2})"
)
_HTML_ASCII_WHITESPACE = " \t\n\f\r"
_HTML_ASCII_WHITESPACE_RE = re.compile(r"[ \t\n\f\r]+")
_HTML_ASCII_LOWER_TRANSLATION = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
)


def _html_ascii_lower(value: str) -> str:
    """Apply HTML's ASCII-only case normalization."""
    return value.translate(_HTML_ASCII_LOWER_TRANSLATION)


def _is_html_ascii_whitespace(character: str) -> bool:
    """Return whether one character is HTML tokenizer whitespace."""
    return character in _HTML_ASCII_WHITESPACE


@dataclass(frozen=True)
class _ComplexFigureRequirement:
    """Source-registry contract for one complex HTML figure."""

    filename: str
    generator: str
    label: str
    details_id: str
    long_description: str


@dataclass(frozen=True)
class _HTMLAttribute:
    """One source-preserving start-tag attribute."""

    name: str
    value: str | None
    value_start: int | None
    value_end: int | None
    quote: str | None


@dataclass(frozen=True)
class _HTMLToken:
    """One source-preserving HTML token used by normalization and validation."""

    kind: str
    raw: str
    start: int
    tag_name: str = ""
    attributes: tuple[_HTMLAttribute, ...] = ()
    self_closing: bool = False


@dataclass(frozen=True)
class _HTMLAttributeOccurrence:
    """One decoded attribute value with its owning element and source offset."""

    tag_name: str
    attribute_name: str
    value: str
    start: int
    attributes: tuple[_HTMLAttribute, ...]


@dataclass(frozen=True)
class _HTMLDocument:
    """One path-confined HTML source captured before any normalization write."""

    path: Path
    original: bytes
    text: str
    tokens: tuple[_HTMLToken, ...]
    source_stat: os.stat_result


@dataclass(frozen=True)
class _StagedHTMLWrite:
    """Same-directory replacement and rollback files for one HTML document."""

    document: _HTMLDocument
    replacement_path: Path
    replacement_bytes: bytes
    replacement_stat: os.stat_result
    rollback_path: Path


class _HTMLTokenizationError(ValueError):
    """Raised when a protected HTML context is not safely bounded."""


class _HTMLRollbackError(RuntimeError):
    """Raised when a failed batch leaves durable recovery files behind."""

    def __init__(self, message: str, retained_paths: frozenset[Path]) -> None:
        super().__init__(message)
        self.retained_paths = retained_paths


def _attribute_values(token: _HTMLToken) -> dict[str, str | None]:
    """Return decoded attributes using HTML's first-duplicate-wins rule."""
    values: dict[str, str | None] = {}
    for attribute in token.attributes:
        values.setdefault(attribute.name, attribute.value)
    return values


@dataclass
class _FigureAccessibilityFacts:
    """Markup facts collected for one rendered ``figure`` element."""

    figure_id: str
    has_image: bool = False
    has_caption: bool = False
    caption_count: int = 0
    caption_details_dom_order: list[str] = field(default_factory=list)
    image_sources: list[str] = field(default_factory=list)
    image_aria_details: list[str] = field(default_factory=list)
    details_count: int = 0
    details_id: str = ""
    details_classes: set[str] = field(default_factory=set)
    details_figure_label: str = ""
    details_expanded_initially: bool = False
    summary_count: int = 0
    description_paragraph_count: int = 0
    summary_chunks: list[str] = field(default_factory=list)
    description_chunks: list[str] = field(default_factory=list)
    details_open: bool = False
    summary_open: bool = False


def _is_contextual_full_size_label(value: str) -> bool:
    match = _CONTEXTUAL_FULL_SIZE_LABEL_RE.fullmatch(value)
    return bool(match and re.search(r"\w", match.group("context"), re.UNICODE))


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
        self.generic_full_size_links = 0
        self.duplicate_ids: set[str] = set()
        self._ids: set[str] = set()
        self._figure_stack: list[_FigureAccessibilityFacts] = []
        self.figure_facts: list[_FigureAccessibilityFacts] = []
        self.figures_missing_captions = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        classes = set((attributes.get("class") or "").split())
        is_skip_link = tag == "a" and "skip-link" in classes and attributes.get("href") == "#main-content"
        tabindex = attributes.get("tabindex")
        try:
            has_sequential_tabindex = tabindex is not None and int(tabindex) >= 0
        except ValueError:
            has_sequential_tabindex = False
        is_interactive = (
            (tag == "a" and bool(attributes.get("href")))
            or tag in {"button", "embed", "iframe", "object", "select", "summary", "textarea"}
            or (tag == "input" and (attributes.get("type") or "text").lower() != "hidden")
            or (tag in {"audio", "video"} and "controls" in attributes)
            or has_sequential_tabindex
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
                if attributes.get("tabindex") == "-1":
                    self.focusable_main_content_count += 1
        elif tag == "a":
            if is_skip_link:
                self.skip_link_count += 1
            if "figure-full-size-link" in classes:
                aria_label = (attributes.get("aria-label") or "").strip()
                if not aria_label:
                    self.unlabelled_full_size_links += 1
                elif not _is_contextual_full_size_label(aria_label):
                    self.generic_full_size_links += 1
        elif tag == "figure":
            self._figure_stack.append(_FigureAccessibilityFacts(figure_id=(element_id or "").strip()))
        elif tag == "img":
            if not (attributes.get("alt") or "").strip():
                self.missing_image_alt += 1
            if self._figure_stack:
                facts = self._figure_stack[-1]
                facts.has_image = True
                facts.image_sources.append((attributes.get("src") or "").strip())
                facts.image_aria_details.append((attributes.get("aria-details") or "").strip())
        elif tag == "figcaption" and self._figure_stack:
            facts = self._figure_stack[-1]
            facts.has_caption = True
            facts.caption_count += 1
            facts.caption_details_dom_order.append("figcaption")
        elif tag == "details" and self._figure_stack:
            facts = self._figure_stack[-1]
            facts.details_count += 1
            facts.caption_details_dom_order.append("details")
            facts.details_id = (element_id or "").strip()
            facts.details_classes = classes
            facts.details_figure_label = (attributes.get("data-figure-label") or "").strip()
            facts.details_expanded_initially = "open" in attributes
            facts.details_open = True
        elif tag == "summary" and self._figure_stack:
            facts = self._figure_stack[-1]
            if facts.details_open:
                facts.summary_count += 1
                facts.summary_open = True
        elif tag == "p" and self._figure_stack:
            facts = self._figure_stack[-1]
            if facts.details_open and not facts.summary_open:
                facts.description_paragraph_count += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._title_chunks is not None:
            self.titles.append("".join(self._title_chunks).strip())
            self._title_chunks = None
        elif tag == "summary" and self._figure_stack:
            self._figure_stack[-1].summary_open = False
        elif tag == "details" and self._figure_stack:
            self._figure_stack[-1].details_open = False
        elif tag == "figure" and self._figure_stack:
            facts = self._figure_stack.pop()
            self.figure_facts.append(facts)
            if facts.has_image and not facts.has_caption:
                self.figures_missing_captions += 1

    def handle_data(self, data: str) -> None:
        if self._title_chunks is not None:
            self._title_chunks.append(data)
        if self._figure_stack:
            facts = self._figure_stack[-1]
            if facts.details_open:
                if facts.summary_open:
                    facts.summary_chunks.append(data)
                else:
                    facts.description_chunks.append(data)

    def close(self) -> None:
        super().close()
        while self._figure_stack:
            facts = self._figure_stack.pop()
            self.figure_facts.append(facts)
            if facts.has_image and not facts.has_caption:
                self.figures_missing_captions += 1


class _HrefCollector(HTMLParser):
    """Collect real HTML ``href`` attributes independent of quoting style."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del tag
        for name, value in attrs:
            if name.casefold() == "href" and value is not None:
                self.hrefs.append(value)


def _hrefs_in(text: str) -> tuple[str, ...]:
    parser = _HrefCollector()
    parser.feed(text)
    parser.close()
    return tuple(parser.hrefs)


def _normalize_accessibility_text(value: str) -> str:
    """Normalize renderer whitespace without weakening textual identity."""
    return " ".join(value.split())


def _figure_filename(raw_src: str) -> str:
    parsed = urlparse(unescape(raw_src))
    return Path(unquote(parsed.path)).name


def _complex_figure_issues(
    path: Path,
    parser: _AccessibilityParser,
    requirements_by_filename: dict[str, _ComplexFigureRequirement],
) -> tuple[tuple[str, ...], set[str]]:
    """Validate complex figure disclosure identity and exact association."""
    issues: list[str] = []
    observed: set[str] = set()
    requirements_by_label = {
        requirement.label: requirement for requirement in requirements_by_filename.values()
    }
    for facts in parser.figure_facts:
        candidates = {
            requirement.label: requirement
            for source in facts.image_sources
            if (requirement := requirements_by_filename.get(_figure_filename(source)))
        }
        if facts.figure_id in requirements_by_label:
            requirement = requirements_by_label[facts.figure_id]
            candidates[requirement.label] = requirement
        if not candidates:
            continue
        if len(candidates) != 1:
            issues.append(
                f"{path}: complex figure element ambiguously contains {', '.join(sorted(candidates))}"
            )
            continue

        requirement = next(iter(candidates.values()))
        observed.add(requirement.label)
        prefix = f"{path}: {requirement.label}"
        if facts.figure_id != requirement.label:
            issues.append(f"{prefix} figure id must equal the registry label")
        matching_sources = [
            source for source in facts.image_sources if _figure_filename(source) == requirement.filename
        ]
        if len(matching_sources) != 1:
            issues.append(f"{prefix} must contain exactly one registered image {requirement.filename}")
        if len(facts.image_aria_details) != 1 or (facts.image_aria_details[0] != requirement.details_id):
            issues.append(f"{prefix} image aria-details must reference {requirement.details_id!r}")
        if facts.details_count != 1:
            issues.append(f"{prefix} must contain exactly one details disclosure")
        if facts.caption_count != 1:
            issues.append(f"{prefix} must contain exactly one figcaption")
        if facts.caption_details_dom_order != ["figcaption", "details"]:
            issues.append(f"{prefix} labeled details disclosure must follow figcaption in DOM child order")
        if facts.details_id != requirement.details_id:
            issues.append(f"{prefix} details id must equal {requirement.details_id!r}")
        if "figure-long-description" not in facts.details_classes:
            issues.append(f"{prefix} details must use the figure-long-description class")
        if facts.details_figure_label != requirement.label:
            issues.append(f"{prefix} details data-figure-label must match the registry label")
        if facts.details_expanded_initially:
            issues.append(f"{prefix} details disclosure must be initially collapsed")
        if facts.summary_count != 1:
            issues.append(f"{prefix} details disclosure must contain exactly one summary")
        if facts.description_paragraph_count < 2:
            issues.append(f"{prefix} structured long description must contain at least two paragraphs")
        summary = _normalize_accessibility_text(" ".join(facts.summary_chunks))
        expected_summary = f"Detailed description of figure {requirement.generator.replace('_', ' ')}"
        if summary.casefold() != expected_summary.casefold():
            issues.append(f"{prefix} details summary must equal {expected_summary!r}")
        observed_description = _normalize_accessibility_text(" ".join(facts.description_chunks))
        expected_description = _normalize_accessibility_text(requirement.long_description)
        if observed_description != expected_description:
            issues.append(f"{prefix} details text does not match the registry long description")
    return tuple(issues), observed


def _accessibility_issues_in(
    path: Path,
    text: str,
    requirements_by_filename: dict[str, _ComplexFigureRequirement] | None = None,
) -> tuple[tuple[str, ...], set[str]]:
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
        issues.append(f"{path}: main-content skip target must use tabindex='-1'")
    if parser.skip_link_count != 1:
        issues.append(f"{path}: expected one .skip-link targeting '#main-content'")
    elif not parser.first_interactive_is_skip:
        issues.append(f"{path}: skip link must be the first interactive element")
    if parser.missing_image_alt:
        issues.append(f"{path}: {parser.missing_image_alt} image(s) lack non-empty alt text")
    if parser.figures_missing_captions:
        issues.append(f"{path}: {parser.figures_missing_captions} image figure(s) lack figcaption")
    if parser.unlabelled_full_size_links:
        issues.append(f"{path}: {parser.unlabelled_full_size_links} full-size figure link(s) lack aria-label")
    if parser.generic_full_size_links:
        issues.append(
            f"{path}: {parser.generic_full_size_links} full-size figure link(s) "
            "lack a contextual Figure number or title in aria-label"
        )
    if parser.duplicate_ids:
        issues.append(f"{path}: duplicate id(s): {', '.join(sorted(parser.duplicate_ids))}")
    complex_issues, observed = _complex_figure_issues(
        path,
        parser,
        requirements_by_filename or {},
    )
    issues.extend(complex_issues)
    return tuple(issues), observed


def _complex_figure_requirements(
    project_root: Path,
) -> tuple[dict[str, _ComplexFigureRequirement], tuple[str, ...]]:
    """Load complex-figure requirements from the generated source registry."""
    registry_path = project_root / "output" / "figures" / "figure_registry.json"
    if not registry_path.exists():
        return {}, (f"{registry_path}: required figure registry is missing",)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, (f"{registry_path}: figure registry is unreadable: {exc}",)
    if not isinstance(payload, dict) or not isinstance(payload.get("figures"), list):
        return {}, (f"{registry_path}: figure registry has no figures list",)
    try:
        validate_report("figure_registry", payload)
    except ReportSchemaError as exc:
        return {}, (f"{registry_path}: figure registry schema is invalid: {exc}",)

    requirements: dict[str, _ComplexFigureRequirement] = {}
    labels: set[str] = set()
    observed_complex_generators: set[str] = set()
    issues: list[str] = []
    for index, raw in enumerate(payload["figures"]):
        if not isinstance(raw, dict):
            continue
        generator = raw.get("generated_by")
        long_description_value = str(raw.get("long_description", "")).strip()
        is_required_complex = generator in COMPLEX_FIGURE_GENERATORS
        if is_required_complex:
            observed_complex_generators.add(str(generator))
            if not long_description_value:
                issues.append(f"{registry_path}: complex figure entry {index} lacks long_description")
                continue
        elif not long_description_value:
            continue
        required_values = {
            key: raw.get(key) for key in ("filename", "generated_by", "label", "long_description")
        }
        if any(not isinstance(value, str) or not value.strip() for value in required_values.values()):
            issues.append(f"{registry_path}: complex figure entry {index} has malformed identity fields")
            continue
        filename = str(required_values["filename"])
        generator = str(required_values["generated_by"])
        label = str(required_values["label"])
        long_description = str(required_values["long_description"])
        if filename in requirements or label in labels:
            issues.append(f"{registry_path}: duplicate complex figure filename or label at entry {index}")
            continue
        labels.add(label)
        requirements[filename] = _ComplexFigureRequirement(
            filename=filename,
            generator=generator,
            label=label,
            details_id=f"{label.replace(':', '-')}-long-description",
            long_description=long_description,
        )
    for missing_generator in sorted(COMPLEX_FIGURE_GENERATORS - observed_complex_generators):
        issues.append(f"{registry_path}: required complex figure {missing_generator!r} is absent")
    return requirements, tuple(issues)


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
        path.relative_to(source_dir): path for path in sorted(source_dir.rglob("*")) if path.is_file()
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


def _element_text(
    tokens: tuple[_HTMLToken, ...],
    start_index: int,
) -> tuple[str, int] | None:
    """Return decoded text and end-token index for one actual start token."""
    start = tokens[start_index]
    if start.kind != "start" or start.tag_name in _VOID_ELEMENTS:
        return None
    depth = 1
    chunks: list[str] = []
    for index in range(start_index + 1, len(tokens)):
        token = tokens[index]
        if token.kind == "start" and token.tag_name == start.tag_name:
            depth += 1
        elif token.kind == "end" and token.tag_name == start.tag_name:
            depth -= 1
            if depth == 0:
                return "".join(chunks), index
        elif token.kind == "text":
            chunks.append(unescape(token.raw))
    return None


def _direct_caption_text(
    tokens: tuple[_HTMLToken, ...],
    container_index: int,
    caption_tag: str,
) -> str | None:
    """Return text from the first direct caption child of a figure or table."""
    container = tokens[container_index]
    depth = 0
    for index in range(container_index + 1, len(tokens)):
        token = tokens[index]
        if token.kind == "end" and token.tag_name == container.tag_name and depth == 0:
            return None
        if token.kind == "start":
            if depth == 0 and token.tag_name == caption_tag:
                extracted = _element_text(tokens, index)
                return None if extracted is None else extracted[0]
            if token.tag_name not in _VOID_ELEMENTS:
                depth += 1
        elif token.kind == "end" and depth:
            depth -= 1
    return None


def _xref_labels(
    web_dir: Path,
    *,
    index_text: str | None = None,
    index_tokens: tuple[_HTMLToken, ...] | None = None,
) -> dict[str, str]:
    """Extract display labels only from actual numbered index elements."""
    index = web_dir / "index.html"
    if index_tokens is None:
        if not index.exists() or index.is_symlink():
            return {}
        text = index.read_bytes().decode("utf-8") if index_text is None else index_text
        index_tokens = _tokenize_html(text)

    labels: dict[str, str] = {}
    for token_index, token in enumerate(index_tokens):
        if token.kind != "start":
            continue
        attributes = _attribute_values(token)
        key = attributes.get("id") or ""
        if re.fullmatch(rf"(?:{_XREF_PREFIXES}):.+", key) is None:
            continue
        if re.fullmatch(r"h[1-6]", token.tag_name):
            number = attributes.get("data-number")
            if number:
                labels[key] = f"Section {number}"
        elif token.tag_name in {"figure", "table"}:
            expected_prefix = "fig:" if token.tag_name == "figure" else "tbl:"
            if not key.startswith(expected_prefix):
                continue
            caption_tag = "figcaption" if token.tag_name == "figure" else "caption"
            caption = _direct_caption_text(index_tokens, token_index, caption_tag)
            caption_match = (
                None
                if caption is None
                else re.match(r"\s*(?P<kind>Figure|Table)\s+(?P<number>\d+):", caption)
            )
            if caption_match is not None:
                labels[key] = (
                    f"{caption_match.group('kind')} {caption_match.group('number')}"
                )
        elif token.tag_name == "span" and key.startswith("eq:"):
            equation = _element_text(index_tokens, token_index)
            equation_match = (
                None
                if equation is None
                else re.search(r"\\qquad\{\((?P<number>\d+)\)\}", equation[0])
            )
            if equation_match is not None:
                labels[key] = f"Eq. ({equation_match.group('number')})"
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


def _xref_ids(tokens: tuple[_HTMLToken, ...]) -> set[str]:
    """Return IDs belonging to actual start tags, excluding protected bytes."""
    return {
        element_id
        for token in tokens
        if token.kind == "start"
        if (element_id := _attribute_values(token).get("id"))
    }


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
            fields[match.group("field").lower()] = _clean_bib_value(match.group("value"))
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


def _line_number_at(text: str, offset: int) -> int:
    prefix = text[:offset]
    return prefix.count("\n") + prefix.count("\r") - prefix.count("\r\n") + 1


def _markup_end(html: str, start: int, name_end: int) -> int:
    """Return a tag end using HTML's distinct attribute-value states."""
    state = "before_attribute_name"
    cursor = name_end
    while cursor < len(html):
        character = html[cursor]
        if state == "before_attribute_name":
            if _is_html_ascii_whitespace(character):
                pass
            elif character == "/":
                state = "self_closing_start_tag"
            elif character == ">":
                return cursor + 1
            else:
                state = "attribute_name"
        elif state == "self_closing_start_tag":
            if character == ">":
                return cursor + 1
            state = "before_attribute_name"
            continue
        elif state == "attribute_name":
            if _is_html_ascii_whitespace(character):
                state = "after_attribute_name"
            elif character == "/":
                state = "self_closing_start_tag"
            elif character == "=":
                state = "before_attribute_value"
            elif character == ">":
                return cursor + 1
        elif state == "after_attribute_name":
            if _is_html_ascii_whitespace(character):
                pass
            elif character == "/":
                state = "self_closing_start_tag"
            elif character == "=":
                state = "before_attribute_value"
            elif character == ">":
                return cursor + 1
            else:
                state = "attribute_name"
        elif state == "before_attribute_value":
            if _is_html_ascii_whitespace(character):
                pass
            elif character == '"':
                state = "double_quoted_attribute_value"
            elif character == "'":
                state = "single_quoted_attribute_value"
            elif character == ">":
                return cursor + 1
            else:
                state = "unquoted_attribute_value"
        elif state == "double_quoted_attribute_value":
            if character == '"':
                state = "after_quoted_attribute_value"
        elif state == "single_quoted_attribute_value":
            if character == "'":
                state = "after_quoted_attribute_value"
        elif state == "after_quoted_attribute_value":
            if _is_html_ascii_whitespace(character):
                state = "before_attribute_name"
            elif character == "/":
                state = "self_closing_start_tag"
            elif character == ">":
                return cursor + 1
            else:
                state = "before_attribute_name"
                continue
        elif _is_html_ascii_whitespace(character):
            state = "before_attribute_name"
        elif character == ">":
            return cursor + 1
        cursor += 1
    raise _HTMLTokenizationError(
        f"unterminated HTML tag at line {_line_number_at(html, start)}"
    )


def _html_comment_end(html: str, start: int) -> int:
    """Return the end of an HTML comment using its browser closing states."""
    state = "start"
    cursor = start + 4
    while cursor < len(html):
        character = html[cursor]
        if state == "start":
            if character == "-":
                state = "start_dash"
            elif character == ">":
                return cursor + 1
            else:
                state = "comment"
        elif state == "start_dash":
            if character == "-":
                state = "end"
            elif character == ">":
                return cursor + 1
            else:
                state = "comment"
        elif state == "comment":
            if character == "-":
                state = "end_dash"
        elif state == "end_dash":
            state = "end" if character == "-" else "comment"
        elif state == "end":
            if character == ">":
                return cursor + 1
            if character == "!":
                state = "end_bang"
            elif character != "-":
                state = "comment"
        else:
            if character == ">":
                return cursor + 1
            state = "end_dash" if character == "-" else "comment"
        cursor += 1
    raise _HTMLTokenizationError(
        f"unterminated HTML comment at line {_line_number_at(html, start)}"
    )


def _start_tag_attributes(raw: str, name_end: int) -> tuple[_HTMLAttribute, ...]:
    """Parse attribute identities and source spans without reserializing a tag."""
    attributes: list[_HTMLAttribute] = []
    cursor = name_end
    while cursor < len(raw):
        while cursor < len(raw) and _is_html_ascii_whitespace(raw[cursor]):
            cursor += 1
        if cursor >= len(raw) or raw[cursor] == ">":
            break
        if raw[cursor] == "/" and raw[cursor + 1 :].lstrip(
            _HTML_ASCII_WHITESPACE
        ).startswith(">"):
            break

        name_start = cursor
        if raw[cursor] == "=":
            # HTML's unexpected-equals-sign recovery starts an attribute whose
            # name includes this leading equals sign; it must not be rebound
            # to a later protected attribute such as ``src`` or ``integrity``.
            cursor += 1
        while (
            cursor < len(raw)
            and not _is_html_ascii_whitespace(raw[cursor])
            and raw[cursor] not in "=/>"
        ):
            cursor += 1
        if cursor == name_start:
            cursor += 1
            continue
        name = _html_ascii_lower(raw[name_start:cursor])
        while cursor < len(raw) and _is_html_ascii_whitespace(raw[cursor]):
            cursor += 1
        if cursor >= len(raw) or raw[cursor] != "=":
            attributes.append(_HTMLAttribute(name, None, None, None, None))
            continue

        cursor += 1
        while cursor < len(raw) and _is_html_ascii_whitespace(raw[cursor]):
            cursor += 1
        if cursor >= len(raw) or raw[cursor] == ">":
            attributes.append(_HTMLAttribute(name, "", cursor, cursor, None))
            continue
        quote = raw[cursor] if raw[cursor] in {'"', "'"} else None
        if quote is not None:
            cursor += 1
            value_start = cursor
            while cursor < len(raw) and raw[cursor] != quote:
                cursor += 1
            value_end = cursor
            cursor += int(cursor < len(raw))
        else:
            value_start = cursor
            while (
                cursor < len(raw)
                and not _is_html_ascii_whitespace(raw[cursor])
                and raw[cursor] != ">"
            ):
                cursor += 1
            value_end = cursor
        raw_value = raw[value_start:value_end]
        attributes.append(
            _HTMLAttribute(
                name=name,
                value=unescape(raw_value),
                value_start=value_start,
                value_end=value_end,
                quote=quote,
            )
        )
    return tuple(attributes)


def _tokenize_html(html: str) -> tuple[_HTMLToken, ...]:
    """Tokenize HTML while preserving source bytes and bounding raw contexts.

    This tokenizer intentionally handles only the distinctions required by the
    publication transform. It never reparses attribute contents as markup, and
    it refuses to continue past an unterminated comment or raw-text element.
    """
    tokens: list[_HTMLToken] = []
    cursor = 0
    text_start = 0

    def emit_text(end: int) -> None:
        nonlocal text_start
        if end > text_start:
            tokens.append(_HTMLToken("text", html[text_start:end], text_start))

    while cursor < len(html):
        markup_start = html.find("<", cursor)
        if markup_start < 0:
            break

        if html.startswith("<!--", markup_start):
            emit_text(markup_start)
            token_end = _html_comment_end(html, markup_start)
            tokens.append(
                _HTMLToken("protected", html[markup_start:token_end], markup_start)
            )
            cursor = token_end
            text_start = token_end
            continue

        if html.startswith("<![CDATA[", markup_start):
            emit_text(markup_start)
            raise _HTMLTokenizationError(
                "CDATA sections are not allowed in static text/html at line "
                f"{_line_number_at(html, markup_start)}"
            )

        end_match = _END_TAG_RE.match(html, markup_start)
        start_match = _START_TAG_RE.match(html, markup_start)
        is_declaration = html.startswith("<!", markup_start) or html.startswith(
            "<?", markup_start
        )
        if end_match is None and start_match is None and not is_declaration:
            cursor = markup_start + 1
            continue

        emit_text(markup_start)
        if is_declaration:
            declaration_end = html.find(">", markup_start + 2)
            if declaration_end < 0:
                raise _HTMLTokenizationError(
                    "unterminated HTML declaration at line "
                    f"{_line_number_at(html, markup_start)}"
                )
            token_end = declaration_end + 1
            raw = html[markup_start:token_end]
            tokens.append(_HTMLToken("markup", raw, markup_start))
            cursor = token_end
            text_start = token_end
            continue
        matched_tag = end_match if end_match is not None else start_match
        assert matched_tag is not None
        token_end = _markup_end(html, markup_start, matched_tag.end())
        raw = html[markup_start:token_end]
        if end_match is not None:
            tokens.append(
                _HTMLToken(
                    "end",
                    raw,
                    markup_start,
                    tag_name=_html_ascii_lower(end_match.group("name")),
                )
            )
            cursor = token_end
            text_start = token_end
            continue

        assert start_match is not None
        tag_name = _html_ascii_lower(start_match.group("name"))
        if tag_name in _SCRIPTING_DEPENDENT_ELEMENTS:
            raise _HTMLTokenizationError(
                "<noscript> elements are not allowed because their parsing depends on "
                "browser scripting state at line "
                f"{_line_number_at(html, markup_start)}"
            )
        relative_name_end = start_match.end() - markup_start
        attributes = _start_tag_attributes(raw, relative_name_end)
        self_closing = re.search(r"/[ \t\n\f\r]*>$", raw) is not None
        tokens.append(
            _HTMLToken(
                "start",
                raw,
                markup_start,
                tag_name=tag_name,
                attributes=attributes,
                self_closing=self_closing,
            )
        )
        # In text/html, a trailing slash does not make raw-text/RCDATA elements
        # self-closing. Protect their contents according to tokenizer state so
        # reference-looking bytes can never be rewritten as ordinary prose.
        if tag_name not in _PROTECTED_RAW_ELEMENTS | _PLAINTEXT_ELEMENTS:
            cursor = token_end
            text_start = token_end
            continue

        # The HTML plaintext state consumes every remaining byte, including any
        # apparent closing tag.
        if tag_name in _PLAINTEXT_ELEMENTS:
            if token_end < len(html):
                tokens.append(_HTMLToken("protected", html[token_end:], token_end))
            cursor = len(html)
            text_start = len(html)
            break

        closing_start = re.compile(
            rf"</{re.escape(tag_name)}(?=[ \t\n\f\r/>])",
            re.IGNORECASE | re.ASCII,
        ).search(
            html,
            token_end,
        )
        if closing_start is None:
            raise _HTMLTokenizationError(
                f"unterminated <{tag_name}> raw-text element at line "
                f"{_line_number_at(html, markup_start)}"
            )
        closing_end = _markup_end(
            html,
            closing_start.start(),
            closing_start.end(),
        )
        if closing_start.start() > token_end:
            tokens.append(
                _HTMLToken("protected", html[token_end : closing_start.start()], token_end)
            )
        tokens.append(
            _HTMLToken(
                "end",
                html[closing_start.start() : closing_end],
                closing_start.start(),
                tag_name=tag_name,
            )
        )
        cursor = closing_end
        text_start = closing_end

    emit_text(len(html))
    return tuple(tokens)


def _normalize_aria_label_in_tag(
    token: _HTMLToken,
    *,
    xref_labels: dict[str, str],
    citation_labels: dict[str, str],
) -> tuple[str, int]:
    """Normalize only actual ``aria-label`` values, preserving all other bytes."""
    replacements: list[tuple[int, int, str]] = []
    replacement_count = 0
    for attribute in token.attributes:
        if (
            attribute.name != "aria-label"
            or attribute.value_start is None
            or attribute.value_end is None
        ):
            continue
        value = token.raw[attribute.value_start : attribute.value_end]

        def replace_xref(match: re.Match[str]) -> str:
            return escape(_xref_label(match.group("key"), xref_labels), quote=True)

        value, xref_count = _RAW_XREF_RE.subn(replace_xref, value)
        replacement_count += xref_count

        def replace_citation(match: re.Match[str]) -> str:
            nonlocal replacement_count
            keys = [key.strip() for key in match.group("keys").split(";")]
            if not keys or any(key not in citation_labels for key in keys):
                return match.group(0)
            replacement_count += 1
            label = "; ".join(citation_labels[key] for key in keys)
            return f"({escape(label, quote=True)})"

        value = _RAW_CITATION_RE.sub(replace_citation, value)
        if xref_count or value != token.raw[attribute.value_start : attribute.value_end]:
            replacement = value if attribute.quote is not None else f'"{value}"'
            replacements.append((attribute.value_start, attribute.value_end, replacement))

    normalized = token.raw
    for start, end, replacement in reversed(replacements):
        normalized = f"{normalized[:start]}{replacement}{normalized[end:]}"
    return normalized, replacement_count


def _structural_reference_at(
    tokens: tuple[_HTMLToken, ...],
    index: int,
) -> tuple[str, str, str, str] | None:
    """Recognize a renderer reference only from three actual HTML tokens."""
    if index + 2 >= len(tokens):
        return None
    start, content, end = tokens[index : index + 3]
    if (
        start.kind != "start"
        or start.tag_name != "span"
        or start.self_closing
        or content.kind != "text"
        or end.kind != "end"
        or end.tag_name != "span"
    ):
        return None
    attributes = _attribute_values(start)
    attribute_names = [attribute.name for attribute in start.attributes]
    classes = set((attributes.get("class") or "").split())
    data_cites = attributes.get("data-cites")
    stripped = content.raw.strip()
    leading = content.raw[: len(content.raw) - len(content.raw.lstrip())]
    trailing = content.raw[len(content.raw.rstrip()) :]
    raw_match = _RAW_XREF_RE.fullmatch(stripped)
    if (
        len(attribute_names) == 2
        and set(attribute_names) == {"class", "data-cites"}
        and classes == {"citation"}
        and raw_match is not None
        and data_cites == raw_match.group("key")
    ):
        return "crossref", raw_match.group("key"), leading, trailing
    latex_match = _LATEX_REF_TEXT_RE.fullmatch(stripped)
    if attribute_names == ["class"] and classes == {"math", "inline"} and latex_match is not None:
        return "latex", latex_match.group("key"), leading, trailing
    return None


def _xref_markup(
    key: str,
    *,
    xref_labels: dict[str, str],
    xref_ids: set[str],
    target_prefix: str,
    allow_link: bool,
) -> str:
    label = escape(_xref_label(key, xref_labels))
    if key not in xref_ids or not allow_link:
        return f'<span class="xref">{label}</span>'
    target = f"{target_prefix}#{key}"
    return f'<a class="xref" href="{escape(target, quote=True)}">{label}</a>'


def _pop_element(stack: list[str], tag_name: str) -> None:
    for index in range(len(stack) - 1, -1, -1):
        if stack[index] == tag_name:
            del stack[index:]
            return


def _normalize_html_tokens(
    tokens: tuple[_HTMLToken, ...],
    *,
    xref_labels: dict[str, str],
    citation_labels: dict[str, str],
    xref_ids: set[str],
    target_prefix: str,
) -> tuple[str, int]:
    """Normalize transformable text while retaining source HTML token boundaries."""
    output: list[str] = []
    element_stack: list[str] = []
    replacements = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        structural_reference = _structural_reference_at(tokens, index)
        if structural_reference is not None:
            reference_kind, key, leading, trailing = structural_reference
            if output and index > 0 and tokens[index - 1].kind == "text":
                if reference_kind == "latex":
                    output[-1] = _LATEX_KIND_SUFFIX_RE.sub("", output[-1])
                else:
                    output[-1], duplicate_count = _DUPLICATE_XREF_KIND_SUFFIX_RE.subn(
                        "",
                        output[-1],
                    )
                    replacements += duplicate_count
            output.append(
                leading
                + _xref_markup(
                    key,
                    xref_labels=xref_labels,
                    xref_ids=xref_ids,
                    target_prefix=target_prefix,
                    allow_link=not any(
                        ancestor in _NO_NESTED_LINK_ANCESTORS for ancestor in element_stack
                    ),
                )
                + trailing
            )
            replacements += 1
            index += 3
            continue

        if token.kind == "start":
            normalized_tag, attribute_count = _normalize_aria_label_in_tag(
                token,
                xref_labels=xref_labels,
                citation_labels=citation_labels,
            )
            output.append(normalized_tag)
            replacements += attribute_count
            # The self-closing flag is ignored for non-void elements in
            # text/html, so keep them on the ancestor stack until a real end tag.
            if token.tag_name not in _VOID_ELEMENTS:
                element_stack.append(token.tag_name)
            index += 1
            continue
        if token.kind == "end":
            output.append(token.raw)
            _pop_element(element_stack, token.tag_name)
            index += 1
            continue
        if token.kind != "text":
            output.append(token.raw)
            index += 1
            continue

        allow_link = not any(
            ancestor in _NO_NESTED_LINK_ANCESTORS for ancestor in element_stack
        )

        def replace_raw(match: re.Match[str]) -> str:
            return _xref_markup(
                match.group("key"),
                xref_labels=xref_labels,
                xref_ids=xref_ids,
                target_prefix=target_prefix,
                allow_link=allow_link,
            )

        normalized_text, raw_count = _RAW_XREF_RE.subn(replace_raw, token.raw)
        replacements += raw_count

        changed_citations = 0

        def replace_citation(match: re.Match[str]) -> str:
            nonlocal changed_citations
            keys = [key.strip() for key in match.group("keys").split(";")]
            if not keys or any(key not in citation_labels for key in keys):
                return match.group(0)
            changed_citations += 1
            label = "; ".join(citation_labels[key] for key in keys)
            return f'<span class="citation">({escape(label)})</span>'

        normalized_text = _RAW_CITATION_RE.sub(replace_citation, normalized_text)
        replacements += changed_citations
        output.append(normalized_text)
        index += 1
    return "".join(output), replacements


def _safe_html_paths(project_root: Path, web_dir: Path) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Enumerate regular, non-symlinked HTML files confined to the project."""
    issues: list[str] = []
    if web_dir.is_symlink():
        return (), (f"{web_dir}: web output directory must not be a symlink",)
    try:
        project_resolved = project_root.resolve(strict=True)
        web_resolved = web_dir.resolve(strict=True)
        web_resolved.relative_to(project_resolved)
    except (OSError, ValueError) as exc:
        return (), (f"{web_dir}: web output directory is not confined to the project: {exc}",)

    paths: list[Path] = []
    try:
        candidates = sorted(web_dir.rglob("*.html"))
    except OSError as exc:
        return (), (f"{web_dir}: cannot enumerate HTML files: {exc}",)
    for path in candidates:
        try:
            relative = path.relative_to(web_dir)
            source_stat = path.lstat()
        except (OSError, ValueError) as exc:
            issues.append(f"{path}: cannot inspect HTML input: {exc}")
            continue
        if stat.S_ISLNK(source_stat.st_mode):
            issues.append(f"{path}: HTML input must not be a symlink")
            continue
        if not stat.S_ISREG(source_stat.st_mode):
            issues.append(f"{path}: HTML input must be a regular file")
            continue
        parent = web_dir
        parent_symlink = False
        for part in relative.parts[:-1]:
            parent /= part
            if parent.is_symlink():
                issues.append(f"{path}: HTML input has a symlinked parent directory")
                parent_symlink = True
                break
        if parent_symlink:
            continue
        try:
            path.resolve(strict=True).relative_to(web_resolved)
        except (OSError, ValueError):
            issues.append(f"{path}: HTML input resolves outside the web output directory")
            continue
        paths.append(path)
    return tuple(paths), tuple(sorted(set(issues)))


def _xref_target_prefix(web_dir: Path, html_path: Path) -> str:
    """Return the correct POSIX path from one page to the combined index."""
    root_index = web_dir / "index.html"
    if html_path == root_index:
        return ""
    relative_parent = html_path.relative_to(web_dir).parent.as_posix()
    return posixpath.relpath("index.html", start=relative_parent or ".")


def _stage_bytes(path: Path, payload: bytes, mode: int, purpose: str) -> Path:
    """Durably stage bytes beside their destination for an atomic replacement."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.xref-{purpose}-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        os.fchmod(descriptor, stat.S_IMODE(mode))
        with os.fdopen(descriptor, "wb") as handle:
            descriptor_open = False
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor_open:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _document_is_unchanged(document: _HTMLDocument) -> bool:
    """Check identity and bytes immediately before replacing a captured file."""
    try:
        current = document.path.lstat()
    except OSError:
        return False
    expected = document.source_stat
    return (
        stat.S_ISREG(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and current.st_dev == expected.st_dev
        and current.st_ino == expected.st_ino
        and current.st_size == expected.st_size
        and current.st_mtime_ns == expected.st_mtime_ns
        and document.path.read_bytes() == document.original
    )


def _path_is_staged_replacement(item: _StagedHTMLWrite, path: Path) -> bool:
    """Check that ``path`` still holds the exact file installed by this batch."""
    try:
        current = path.lstat()
        payload = path.read_bytes()
    except OSError:
        return False
    expected = item.replacement_stat
    return (
        stat.S_ISREG(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and current.st_dev == expected.st_dev
        and current.st_ino == expected.st_ino
        and stat.S_IMODE(current.st_mode) == stat.S_IMODE(expected.st_mode)
        and current.st_size == expected.st_size
        and current.st_mtime_ns == expected.st_mtime_ns
        and payload == item.replacement_bytes
    )


def _path_is_captured_original(item: _StagedHTMLWrite, path: Path) -> bool:
    """Check that a commit capture still contains the caller's source file."""
    try:
        current = path.lstat()
        payload = path.read_bytes()
    except OSError:
        return False
    expected = item.document.source_stat
    return (
        stat.S_ISREG(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and current.st_dev == expected.st_dev
        and current.st_ino == expected.st_ino
        and stat.S_IMODE(current.st_mode) == stat.S_IMODE(expected.st_mode)
        and current.st_size == expected.st_size
        and current.st_mtime_ns == expected.st_mtime_ns
        and payload == item.document.original
    )


def _paths_share_regular_file(left: Path, right: Path) -> bool:
    """Return whether two paths are hard links to the same regular file."""
    try:
        left_stat = left.lstat()
        right_stat = right.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(left_stat.st_mode)
        and stat.S_ISREG(right_stat.st_mode)
        and not stat.S_ISLNK(left_stat.st_mode)
        and not stat.S_ISLNK(right_stat.st_mode)
        and left_stat.st_dev == right_stat.st_dev
        and left_stat.st_ino == right_stat.st_ino
    )


def _link_no_clobber(source: Path, destination: Path) -> tuple[bool, BaseException | None]:
    """Link only into an absent destination and observe post-syscall interrupts."""
    try:
        os.link(source, destination, follow_symlinks=False)
    except BaseException as exc:
        return _paths_share_regular_file(source, destination), exc
    return True, None


def _restore_committed_html(
    item: _StagedHTMLWrite,
) -> tuple[str | None, frozenset[Path]]:
    """Restore one commit without replacing a concurrently changed destination.

    Move the current destination into the consumed replacement staging name,
    then use a no-clobber hard link to reinstall either the original bytes or,
    when the destination changed, the concurrent bytes. Any incomplete or
    concurrency-preserving outcome retains both available recovery files.
    """
    retained_paths = frozenset({item.replacement_path, item.rollback_path})
    try:
        os.replace(item.document.path, item.replacement_path)
    except BaseException as exc:
        restored = False
        link_exception: BaseException | None = None
        if not os.path.lexists(item.document.path):
            restore_source = (
                item.rollback_path
                if _path_is_staged_replacement(item, item.replacement_path)
                else item.replacement_path
            )
            restored, link_exception = _link_no_clobber(
                restore_source,
                item.document.path,
            )
        return (
            f"{item.document.path}: rollback move was interrupted or failed: {exc}; "
            f"destination restored without clobber={restored}; "
            f"restoration signal={link_exception}; current bytes retained at "
            f"{item.replacement_path}; original bytes retained at {item.rollback_path}",
            retained_paths,
        )

    try:
        destination_changed = not _path_is_staged_replacement(
            item,
            item.replacement_path,
        )
    except BaseException as exc:
        return (
            f"{item.document.path}: rollback inspection was interrupted: {exc}; "
            f"current bytes retained at {item.replacement_path}; "
            f"original bytes retained at {item.rollback_path}",
            retained_paths,
        )
    restore_source = item.replacement_path if destination_changed else item.rollback_path
    restored, link_exception = _link_no_clobber(
        restore_source,
        item.document.path,
    )
    if not restored or link_exception is not None:
        return (
            f"{item.document.path}: no-clobber restoration was interrupted or failed: "
            f"{link_exception}; restored={restored}; "
            f"current bytes retained at {item.replacement_path}; "
            f"original bytes retained at {item.rollback_path}",
            retained_paths,
        )

    if destination_changed:
        return (
            f"{item.document.path}: destination changed concurrently after commit; "
            f"concurrent bytes were preserved at the destination and {item.replacement_path}; "
            f"original bytes retained at {item.rollback_path}",
            retained_paths,
        )
    return None, frozenset()


def _commit_html_writes(staged: list[_StagedHTMLWrite]) -> None:
    """Commit with no-clobber links and restore originals after a later failure."""
    committed: list[_StagedHTMLWrite] = []
    try:
        for item in staged:
            if not _document_is_unchanged(item.document):
                raise RuntimeError(
                    f"{item.document.path}: HTML input changed during normalization"
                )

            # Atomically capture the current destination first. A change in the
            # former check-to-replace window is now moved into ``rollback_path``
            # and compared before any replacement can be installed.
            try:
                os.replace(item.document.path, item.rollback_path)
            except BaseException as exc:
                restored = False
                link_exception: BaseException | None = None
                if not os.path.lexists(item.document.path):
                    restored, link_exception = _link_no_clobber(
                        item.rollback_path,
                        item.document.path,
                    )
                raise _HTMLRollbackError(
                    f"{item.document.path}: forward capture was interrupted or failed: {exc}; "
                    f"destination restored without clobber={restored}; "
                    f"restoration signal={link_exception}; recovery files retained at "
                    f"{item.replacement_path} and {item.rollback_path}",
                    frozenset({item.replacement_path, item.rollback_path}),
                ) from exc

            try:
                captured_original = _path_is_captured_original(
                    item,
                    item.rollback_path,
                )
            except BaseException as exc:
                raise _HTMLRollbackError(
                    f"{item.document.path}: forward capture inspection was interrupted: {exc}; "
                    f"recovery files retained at {item.replacement_path} and "
                    f"{item.rollback_path}",
                    frozenset({item.replacement_path, item.rollback_path}),
                ) from exc
            if not captured_original:
                restored, link_exception = _link_no_clobber(
                    item.rollback_path,
                    item.document.path,
                )
                if restored and link_exception is None:
                    raise RuntimeError(
                        f"{item.document.path}: HTML input changed during normalization"
                    )
                raise _HTMLRollbackError(
                    f"{item.document.path}: changed input could not be restored cleanly; "
                    f"restored={restored}; restoration signal={link_exception}; "
                    f"recovery files retained at {item.replacement_path} and "
                    f"{item.rollback_path}",
                    frozenset({item.replacement_path, item.rollback_path}),
                ) from link_exception

            installed, link_exception = _link_no_clobber(
                item.replacement_path,
                item.document.path,
            )
            if not installed:
                restored, restore_exception = _link_no_clobber(
                    item.rollback_path,
                    item.document.path,
                )
                if restored and restore_exception is None and link_exception is not None:
                    raise link_exception
                raise _HTMLRollbackError(
                    f"{item.document.path}: no-clobber forward install failed; "
                    f"signal={link_exception}; original restored={restored}; "
                    f"restoration signal={restore_exception}; recovery files retained at "
                    f"{item.replacement_path} and {item.rollback_path}",
                    frozenset({item.replacement_path, item.rollback_path}),
                ) from link_exception
            committed.append(item)
            item.replacement_path.unlink()
            if link_exception is not None:
                raise link_exception

        for item in committed:
            if not _path_is_staged_replacement(item, item.document.path):
                raise RuntimeError(
                    f"{item.document.path}: HTML output changed during normalization"
                )
    except BaseException as exc:
        rollback_failures: list[str] = []
        retained_paths: set[Path] = set()
        if isinstance(exc, _HTMLRollbackError):
            rollback_failures.append(str(exc))
            retained_paths.update(exc.retained_paths)
        for item in reversed(committed):
            rollback_failure, item_retained_paths = _restore_committed_html(item)
            if rollback_failure is not None:
                rollback_failures.append(rollback_failure)
                retained_paths.update(item_retained_paths)
        if rollback_failures:
            if isinstance(exc, _HTMLRollbackError) and len(rollback_failures) == 1:
                raise
            raise _HTMLRollbackError(
                "HTML normalization failed and rollback was incomplete: "
                + "; ".join(rollback_failures),
                frozenset(retained_paths),
            ) from exc
        raise


def normalize_web_xrefs(project_root: str | Path | None = None) -> int:
    """Atomically replace visible references with self-contained HTML links."""
    root = _root(project_root)
    web_dir = _web_dir(root)
    if not web_dir.exists():
        raise FileNotFoundError(f"missing web output directory: {web_dir}")

    html_paths, path_issues = _safe_html_paths(root, web_dir)
    if path_issues:
        raise ValueError("; ".join(path_issues))

    citation_labels = _citation_labels(root)
    documents: list[_HTMLDocument] = []
    for html_path in html_paths:
        source_stat = html_path.lstat()
        if not stat.S_IMODE(source_stat.st_mode) & 0o222:
            raise PermissionError(f"{html_path}: HTML input is not writable")
        original = html_path.read_bytes()
        try:
            text = original.decode("utf-8")
            tokens = _tokenize_html(text)
        except (UnicodeError, _HTMLTokenizationError) as exc:
            raise ValueError(f"{html_path}: {exc}") from exc
        documents.append(
            _HTMLDocument(
                path=html_path,
                original=original,
                text=text,
                tokens=tokens,
                source_stat=source_stat,
            )
        )

    index_document = next(
        (document for document in documents if document.path == web_dir / "index.html"),
        None,
    )
    index_tokens = () if index_document is None else index_document.tokens
    xref_labels = _xref_labels(
        web_dir,
        index_text=None if index_document is None else index_document.text,
        index_tokens=index_tokens,
    )
    xref_ids = _xref_ids(index_tokens)

    normalized_documents: list[tuple[_HTMLDocument, bytes]] = []
    replacements = 0
    for document in documents:
        updated, document_count = _normalize_html_tokens(
            document.tokens,
            xref_labels=xref_labels,
            citation_labels=citation_labels,
            xref_ids=xref_ids,
            target_prefix=_xref_target_prefix(web_dir, document.path),
        )
        normalized_documents.append((document, updated.encode("utf-8")))
        replacements += document_count

    staged: list[_StagedHTMLWrite] = []
    temporary_paths: set[Path] = set()
    retained_paths: set[Path] = set()
    try:
        for document, updated_bytes in normalized_documents:
            if updated_bytes == document.original:
                continue
            replacement_path = _stage_bytes(
                document.path,
                updated_bytes,
                document.source_stat.st_mode,
                "replacement",
            )
            temporary_paths.add(replacement_path)
            replacement_stat = replacement_path.lstat()
            rollback_path = _stage_bytes(
                document.path,
                document.original,
                document.source_stat.st_mode,
                "rollback",
            )
            temporary_paths.add(rollback_path)
            staged.append(
                _StagedHTMLWrite(
                    document=document,
                    replacement_path=replacement_path,
                    replacement_bytes=updated_bytes,
                    replacement_stat=replacement_stat,
                    rollback_path=rollback_path,
                )
            )
        try:
            _commit_html_writes(staged)
        except _HTMLRollbackError as exc:
            retained_paths.update(exc.retained_paths)
            raise
    finally:
        for temporary_path in temporary_paths - retained_paths:
            temporary_path.unlink(missing_ok=True)
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


def _local_href_target(
    output_dir: Path,
    html_path: Path,
    raw_url: str,
) -> tuple[Path | None, str | None]:
    """Resolve one link target within the shipped output tree.

    Generated pages live in ``output/web`` while full-resolution figures live
    in the sibling ``output/figures`` directory, so a confined ``../figures``
    link is valid. Absolute paths, malformed URL paths, unsupported schemes,
    missing targets, and symlink/path traversal outside ``output`` fail closed.
    ``None`` as the target denotes an accepted external link.
    """

    url = unescape(raw_url)
    parsed = urlparse(url)
    scheme = parsed.scheme.casefold()
    if scheme:
        if scheme in _ALLOWED_EXTERNAL_HREF_SCHEMES:
            return None, None
        return None, f"unsupported href scheme: {scheme}"
    if parsed.netloc:
        return None, "protocol-relative href is not allowed"

    decoded_path = unquote(parsed.path)
    if "\x00" in decoded_path or "\\" in decoded_path:
        return None, "local href contains an unsafe path"
    if decoded_path.startswith("/"):
        return None, "absolute local href is not allowed"

    candidate = html_path if not decoded_path else html_path.parent / decoded_path
    try:
        target = candidate.resolve(strict=True)
    except OSError:
        return None, "missing local target"
    try:
        target.relative_to(output_dir.resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        return None, "local href leaves the shipped output tree"
    return target, None


def _fragment_exists(
    target_file: Path,
    target_text: str,
    fragment: str,
    *,
    html_ids: set[str] | None = None,
) -> bool:
    """Recognize explicit HTML or Pandoc-Markdown fragment identifiers."""
    if target_file.suffix.casefold() in {".md", ".markdown"}:
        escaped_fragment = re.escape(fragment)
        pattern = _MARKDOWN_FRAGMENT_RE_TEMPLATE.format(fragment=escaped_fragment)
        return re.search(pattern, target_text, flags=re.IGNORECASE) is not None
    if target_file.suffix.casefold() in {".htm", ".html"}:
        if html_ids is not None:
            return fragment in html_ids
        try:
            return fragment in _xref_ids(_tokenize_html(target_text))
        except _HTMLTokenizationError:
            return False
    return False


def _attribute_occurrences(
    tokens: tuple[_HTMLToken, ...],
    names: frozenset[str],
) -> tuple[_HTMLAttributeOccurrence, ...]:
    """Collect browser-effective attributes without inspecting protected text.

    The HTML tokenizer keeps the first duplicate attribute and ignores later
    copies. Retain that rule here while leaving every URL candidate inside the
    effective ``srcset`` value available to ``_srcset_urls``.
    """
    occurrences: list[_HTMLAttributeOccurrence] = []
    for token in tokens:
        if token.kind != "start":
            continue
        seen: set[str] = set()
        for attribute in token.attributes:
            if attribute.name not in names or attribute.name in seen:
                continue
            seen.add(attribute.name)
            occurrences.append(
                _HTMLAttributeOccurrence(
                    tag_name=token.tag_name,
                    attribute_name=attribute.name,
                    value="" if attribute.value is None else attribute.value,
                    start=(
                        token.start
                        if attribute.value_start is None
                        else token.start + attribute.value_start
                    ),
                    attributes=token.attributes,
                )
            )
    return tuple(occurrences)


def _srcset_urls(occurrence: _HTMLAttributeOccurrence) -> tuple[tuple[str, int], ...]:
    """Extract source candidates; unsafe data URLs are rejected before comma ambiguity matters."""
    candidates: list[tuple[str, int]] = []
    for match in re.finditer(
        r"(?:^|,)[ \t\n\f\r]*(?P<url>[^ \t\n\f\r,]+)",
        occurrence.value,
    ):
        candidates.append((match.group("url"), occurrence.start + match.start("url")))
    return tuple(candidates)


def _resource_urls(
    tokens: tuple[_HTMLToken, ...],
) -> tuple[tuple[_HTMLAttributeOccurrence, str, int], ...]:
    """Return URLs from actual resource-bearing attributes with source offsets."""
    resources: list[tuple[_HTMLAttributeOccurrence, str, int]] = []
    for occurrence in _attribute_occurrences(tokens, _RESOURCE_ATTRIBUTES):
        if occurrence.attribute_name == "srcset":
            resources.extend(
                (occurrence, url, offset) for url, offset in _srcset_urls(occurrence)
            )
        else:
            resources.append((occurrence, occurrence.value, occurrence.start))
    return tuple(resources)


def _document_base_issues(
    path: Path,
    text: str,
    tokens: tuple[_HTMLToken, ...],
) -> tuple[str, ...]:
    """Reject document bases so every resource is resolved from its owning page.

    Even a local-looking ``src`` can become an external fetch when a browser
    applies ``<base href>``. Publication pages have no need for that ambient
    resolution state, so rejecting the element itself is simpler and safer
    than attempting to reproduce every browser URL-resolution edge case.
    """
    return tuple(
        f"{path}:{_line_number_at(text, token.start)}: "
        "document <base> elements are not allowed because they alter resource resolution"
        for token in tokens
        if token.kind == "start" and token.tag_name == "base"
    )


def _resource_target(
    web_dir: Path,
    html_path: Path,
    raw_url: str,
) -> tuple[Path | None, str | None]:
    """Resolve a publication resource or return an explicit safety failure."""
    decoded_url = unescape(raw_url)
    if any(
        ord(character) < 0x20 or ord(character) == 0x7F
        for character in decoded_url
    ):
        return None, "resource URL contains an ASCII control character"
    url = decoded_url.strip()
    if not url:
        return None, "resource URL is empty"
    parsed = urlparse(url)
    scheme = parsed.scheme.casefold()
    if scheme:
        if scheme in _ALLOWED_EXTERNAL_RESOURCE_SCHEMES:
            return None, None
        return None, f"unsupported resource scheme: {scheme}"
    if parsed.netloc or url.startswith("//"):
        return None, "protocol-relative resource URL is not allowed"
    decoded_path = unquote(parsed.path)
    if "\x00" in decoded_path or "\\" in decoded_path:
        return None, "resource URL contains an unsafe path"
    if decoded_path.startswith("/"):
        return None, "absolute resource URL is not allowed"
    if not decoded_path:
        return None, "resource URL has no path"

    # Resolve the URL from the page that actually owns the attribute.  The
    # shipped package deliberately permits sibling ``output/figures`` assets,
    # but nothing outside ``output``.  Do not reuse ``_asset_path`` here: its
    # legacy leading-``..`` normalization is a packaging convenience, not a
    # security boundary, and can change which file the browser would request.
    asset = html_path.parent / decoded_path
    try:
        output_resolved = web_dir.parent.resolve(strict=True)
        asset = Path(os.path.abspath(os.fspath(asset)))
        relative = asset.relative_to(output_resolved)
    except (OSError, RuntimeError, ValueError):
        return None, "resource target leaves the shipped output tree"

    inspected = output_resolved
    for part in relative.parts:
        inspected /= part
        try:
            component_stat = inspected.lstat()
        except FileNotFoundError:
            break
        except OSError:
            return None, "resource target cannot be inspected"
        if stat.S_ISLNK(component_stat.st_mode):
            return None, "resource target has a symlinked path component"
    try:
        asset.resolve(strict=False).relative_to(output_resolved)
    except (OSError, RuntimeError, ValueError):
        return None, "resource target leaves the shipped output tree"
    if asset.exists():
        try:
            asset_stat = asset.stat()
        except OSError:
            return None, "resource target cannot be inspected"
        if not stat.S_ISREG(asset_stat.st_mode):
            return None, "resource target is not a regular file"
    return asset, None


def _valid_sri_token(token: str) -> bool:
    """Return whether one SRI token is canonical and has its declared SHA size."""
    match = _SRI_VALUE_RE.fullmatch(token)
    if match is None:
        return False
    digest = match.group("digest")
    unpadded_digest = digest.rstrip("=")
    canonical_padding = "=" * (-len(unpadded_digest) % 4)
    canonical_digest = f"{unpadded_digest}{canonical_padding}"
    # Preserve the prior allowance for omitted Base64 padding, while rejecting
    # explicitly supplied non-canonical padding and non-zero discarded bits.
    if digest not in {unpadded_digest, canonical_digest}:
        return False
    try:
        decoded_digest = base64.b64decode(canonical_digest, validate=True)
    except (binascii.Error, ValueError):
        return False
    return (
        len(decoded_digest) == _SRI_DIGEST_BYTES[match.group("algorithm")]
        and base64.b64encode(decoded_digest).decode("ascii").rstrip("=") == unpadded_digest
    )


def _valid_sri_metadata(integrity: str) -> bool:
    """Validate every token in a whitespace-separated SRI metadata value."""
    stripped = integrity.strip(_HTML_ASCII_WHITESPACE)
    if not stripped:
        return False
    return all(
        _valid_sri_token(token)
        for token in _HTML_ASCII_WHITESPACE_RE.split(stripped)
    )


def _external_script_issue(occurrence: _HTMLAttributeOccurrence, url: str) -> str | None:
    """Require integrity-bound HTTPS for the external script dependency surface."""
    if occurrence.tag_name != "script" or urlparse(url.strip()).scheme.casefold() != "https":
        return None
    # HTML keeps the first duplicate attribute.  Mirror that behavior so a
    # later well-formed ``integrity`` cannot mask an earlier invalid one.
    attributes: dict[str, str | None] = {}
    for attribute in occurrence.attributes:
        if attribute.name != "src":
            attributes.setdefault(attribute.name, attribute.value)
    integrity = attributes.get("integrity") or ""
    if not _valid_sri_metadata(integrity):
        return "external script must declare a valid SHA-2 integrity value"
    if "crossorigin" not in attributes:
        return "external script with integrity must use crossorigin='anonymous'"
    crossorigin = attributes["crossorigin"]
    if crossorigin is not None and _html_ascii_lower(crossorigin) not in {"", "anonymous"}:
        return "external script with integrity must use crossorigin='anonymous'"
    return None


def _pattern_lines_in_text_tokens(
    path: Path,
    text: str,
    tokens: tuple[_HTMLToken, ...],
    pattern: re.Pattern[str],
    message: str,
    *,
    include_attributes: bool = False,
) -> tuple[str, ...]:
    """Report a pattern only where it represents transformable/public markup."""
    lines: set[int] = set()
    for token in tokens:
        if token.kind == "text":
            for match in pattern.finditer(token.raw):
                lines.add(_line_number_at(text, token.start + match.start()))
        elif include_attributes and token.kind == "start":
            for attribute in token.attributes:
                if attribute.value_start is None or attribute.value_end is None:
                    continue
                raw_value = token.raw[attribute.value_start : attribute.value_end]
                for match in pattern.finditer(raw_value):
                    lines.add(
                        _line_number_at(text, token.start + attribute.value_start + match.start())
                    )
    return tuple(f"{path}:{line}: {message}" for line in sorted(lines))


def _raw_xrefs_in(
    path: Path,
    text: str | None = None,
    tokens: tuple[_HTMLToken, ...] | None = None,
) -> tuple[str, ...]:
    """Find unresolved references only in transformable textual contexts."""
    document = path.read_text(encoding="utf-8") if text is None else text
    parsed_tokens = _tokenize_html(document) if tokens is None else tokens
    offender_lines: set[int] = set()

    def inspect(value: str, absolute_start: int) -> None:
        for match in _RAW_XREF_RE.finditer(value):
            offender_lines.add(_line_number_at(document, absolute_start + match.start()))
        for match in re.finditer(r"\\ref\{", value):
            offender_lines.add(_line_number_at(document, absolute_start + match.start()))

    for token in parsed_tokens:
        if token.kind == "text":
            inspect(token.raw, token.start)
        elif token.kind == "start":
            for attribute in token.attributes:
                if (
                    attribute.name == "aria-label"
                    and attribute.value_start is not None
                    and attribute.value_end is not None
                ):
                    inspect(
                        token.raw[attribute.value_start : attribute.value_end],
                        token.start + attribute.value_start,
                    )
    return tuple(f"{path}:{line_number}" for line_number in sorted(offender_lines))


def _accessible_name_markup_in(
    path: Path,
    text: str,
    tokens: tuple[_HTMLToken, ...] | None = None,
) -> tuple[str, ...]:
    """Reject literal element markup injected into an ``aria-label`` value."""
    offenders: list[str] = []
    parsed_tokens = _tokenize_html(text) if tokens is None else tokens
    for token in parsed_tokens:
        if token.kind != "start":
            continue
        for attribute in token.attributes:
            if (
                attribute.name != "aria-label"
                or attribute.value_start is None
                or attribute.value_end is None
            ):
                continue
            value = token.raw[attribute.value_start : attribute.value_end]
            if "<" in value or ">" in value:
                line_number = _line_number_at(text, token.start + attribute.value_start)
                offenders.append(f"{path}:{line_number}: aria-label contains literal HTML markup")
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

    html_files, html_path_issues = _safe_html_paths(root, web_dir)
    missing: list[str] = []
    raw_xrefs: list[str] = []
    broken_xrefs: list[str] = []
    malformed_markup: list[str] = []
    malformed_markup.extend(html_path_issues)
    accessibility_issues: list[str] = []
    complex_requirements, registry_issues = _complex_figure_requirements(root)
    accessibility_issues.extend(registry_issues)
    observed_complex_labels: set[str] = set()
    assets_checked = 0
    target_text_cache: dict[Path, str] = {}
    target_read_errors: dict[Path, str] = {}
    target_html_ids: dict[Path, set[str]] = {}
    for html_path in html_files:
        try:
            text = html_path.read_bytes().decode("utf-8")
        except (OSError, UnicodeError) as exc:
            accessibility_issues.append(f"{html_path}: accessibility parse failed: {exc}")
            malformed_markup.append(f"{html_path}: unreadable HTML: {exc}")
            continue
        try:
            tokens = _tokenize_html(text)
        except _HTMLTokenizationError as exc:
            malformed_markup.append(f"{html_path}: {exc}")
            continue
        page_issues, page_complex_labels = _accessibility_issues_in(
            html_path,
            text,
            complex_requirements,
        )
        accessibility_issues.extend(page_issues)
        observed_complex_labels.update(page_complex_labels)
        raw_xrefs.extend(_raw_xrefs_in(html_path, text, tokens))
        malformed_markup.extend(_accessible_name_markup_in(html_path, text, tokens))
        malformed_markup.extend(_document_base_issues(html_path, text, tokens))
        malformed_markup.extend(
            _pattern_lines_in_text_tokens(
                html_path,
                text,
                tokens,
                _LEAKED_FIGURE_RE,
                "leaked Markdown figure syntax",
            )
        )
        malformed_markup.extend(
            _pattern_lines_in_text_tokens(
                html_path,
                text,
                tokens,
                _UNRESOLVED_TOKEN_RE,
                "unresolved manuscript token",
                include_attributes=True,
            )
        )
        for srcdoc in _attribute_occurrences(tokens, frozenset({"srcdoc"})):
            line_number = _line_number_at(text, srcdoc.start)
            malformed_markup.append(
                f"{html_path}:{line_number}: embedded iframe srcdoc is not allowed"
            )
        for occurrence, url, resource_offset in _resource_urls(tokens):
            line_number = _line_number_at(text, resource_offset)
            asset, resource_issue = _resource_target(web_dir, html_path, url)
            if resource_issue is not None:
                malformed_markup.append(
                    f"{html_path}:{line_number}: {url} ({resource_issue})"
                )
                continue
            script_issue = _external_script_issue(occurrence, url)
            if script_issue is not None:
                malformed_markup.append(
                    f"{html_path}:{line_number}: {url} ({script_issue})"
                )
                continue
            if asset is not None:
                assets_checked += 1
            if asset is not None and not asset.exists():
                missing.append(f"{html_path}:{line_number}: {url} -> {asset}")
        for href in _attribute_occurrences(tokens, frozenset({"href"})):
            url = href.value
            line_number = _line_number_at(text, href.start)
            parsed = urlparse(unescape(url))
            target_file, target_issue = _local_href_target(
                web_dir.parent,
                html_path,
                url,
            )
            if target_issue is not None:
                broken_xrefs.append(f"{html_path}:{line_number}: {url} ({target_issue})")
                continue
            if target_file is None:
                continue
            fragment = unquote(parsed.fragment)
            if fragment:
                target_key = target_file.resolve()
                if target_key not in target_text_cache and target_key not in target_read_errors:
                    try:
                        target_text_cache[target_key] = target_file.read_bytes().decode("utf-8")
                    except (OSError, UnicodeError) as exc:
                        target_read_errors[target_key] = str(exc)
                    else:
                        if target_file.suffix.casefold() in {".htm", ".html"}:
                            try:
                                target_html_ids[target_key] = _xref_ids(
                                    _tokenize_html(target_text_cache[target_key])
                                )
                            except _HTMLTokenizationError:
                                target_html_ids[target_key] = set()
                if target_key in target_read_errors:
                    broken_xrefs.append(
                        f"{html_path}:{line_number}: {url} "
                        f"(fragment target is unreadable: {target_read_errors[target_key]})"
                    )
                else:
                    if not _fragment_exists(
                        target_file,
                        target_text_cache[target_key],
                        fragment,
                        html_ids=target_html_ids.get(target_key),
                    ):
                        broken_xrefs.append(
                            f"{html_path}:{line_number}: {url} (missing fragment)"
                        )

    required_complex_labels = {requirement.label for requirement in complex_requirements.values()}
    for missing_label in sorted(required_complex_labels - observed_complex_labels):
        accessibility_issues.append(
            f"{web_dir}: registry-declared complex figure {missing_label} has no "
            "associated labeled details long description"
        )

    return WebPackageValidation(
        html_files=len(html_files),
        assets_checked=assets_checked,
        missing_assets=tuple(sorted(set(missing))),
        raw_xrefs=tuple(sorted(set(raw_xrefs))),
        broken_xrefs=tuple(sorted(set(broken_xrefs))),
        malformed_markup=tuple(sorted(set(malformed_markup))),
        accessibility_issues=tuple(sorted(set(accessibility_issues))),
    )
