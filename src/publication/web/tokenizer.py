"""Tokenizer for source-preserving HTML publication."""
from __future__ import annotations

import re

from publication.web.attributes import _start_tag_attributes
from publication.web.lexical import _html_ascii_lower, _html_comment_end, _line_number_at, _markup_end
from publication.web.models import _HTMLToken, _HTMLTokenizationError

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
