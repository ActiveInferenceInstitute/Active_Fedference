"""Markup findings for source-preserving HTML publication."""
from __future__ import annotations

import re
from pathlib import Path

from publication.web.lexical import _line_number_at
from publication.web.models import _HTMLToken
from publication.web.reference_patterns import _RAW_XREF_RE
from publication.web.tokenizer import _tokenize_html

_LEAKED_FIGURE_RE = re.compile(r"\]\([^)]*figures/[^)]*\)\{#fig:")


_UNRESOLVED_TOKEN_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")


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
