"""Links for source-preserving HTML publication."""
from __future__ import annotations

import posixpath
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

from publication.web.models import _HTMLTokenizationError
from publication.web.reference_labels import _xref_ids
from publication.web.tokenizer import _tokenize_html

_ALLOWED_EXTERNAL_HREF_SCHEMES = frozenset({"http", "https", "mailto", "tel"})


_MARKDOWN_FRAGMENT_RE_TEMPLATE = r"\{{#{fragment}(?:\s|\}})"


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
