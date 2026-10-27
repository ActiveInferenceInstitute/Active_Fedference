"""Reference labels for source-preserving HTML publication."""
from __future__ import annotations

import re
from html import unescape
from pathlib import Path

from publication.web.attributes import _attribute_values
from publication.web.models import _HTMLToken
from publication.web.reference_patterns import _XREF_PREFIXES
from publication.web.tokenizer import _VOID_ELEMENTS, _tokenize_html


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
