"""Reference markup for source-preserving HTML publication."""
from __future__ import annotations

import re
from html import escape

from publication.web.attributes import _attribute_values
from publication.web.models import _HTMLToken
from publication.web.reference_labels import _xref_label
from publication.web.reference_patterns import (
    _DUPLICATE_XREF_KIND_SUFFIX_RE,
    _LATEX_KIND_SUFFIX_RE,
    _LATEX_REF_TEXT_RE,
    _RAW_CITATION_RE,
    _RAW_XREF_RE,
)
from publication.web.tokenizer import _VOID_ELEMENTS

_NO_NESTED_LINK_ANCESTORS = frozenset({"a", "button"})


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
