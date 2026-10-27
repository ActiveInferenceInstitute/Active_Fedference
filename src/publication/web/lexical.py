"""Lexical for source-preserving HTML publication."""
from __future__ import annotations

import re

from publication.web.models import _HTMLTokenizationError

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
