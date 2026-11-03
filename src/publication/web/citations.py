"""Citations for source-preserving HTML publication."""
from __future__ import annotations

import re
from pathlib import Path

_BIB_ENTRY_RE = re.compile(
    r"^@(?!comment\b)\w+\s*\{\s*(?P<key>[^,\s]+),(?P<body>.*?)\n\}",
    re.DOTALL | re.MULTILINE,
)


_BIB_FIELD_RE = re.compile(r"^\s*(?P<field>\w+)\s*=\s*(?P<value>.+?),?\s*$")


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
