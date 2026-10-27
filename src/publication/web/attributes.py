"""Attributes for source-preserving HTML publication."""
from __future__ import annotations

from html import unescape

from publication.web.lexical import _HTML_ASCII_WHITESPACE, _html_ascii_lower, _is_html_ascii_whitespace
from publication.web.models import _HTMLAttribute, _HTMLAttributeOccurrence, _HTMLToken


def _attribute_values(token: _HTMLToken) -> dict[str, str | None]:
    """Return decoded attributes using HTML's first-duplicate-wins rule."""
    values: dict[str, str | None] = {}
    for attribute in token.attributes:
        values.setdefault(attribute.name, attribute.value)
    return values


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
