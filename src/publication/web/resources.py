"""Resources for source-preserving HTML publication."""
from __future__ import annotations

import os
import re
import stat
from html import unescape
from pathlib import Path
from urllib.parse import unquote, urlparse

from publication.web.attributes import _attribute_occurrences
from publication.web.lexical import _line_number_at
from publication.web.models import _HTMLAttributeOccurrence, _HTMLToken

_ALLOWED_EXTERNAL_RESOURCE_SCHEMES = frozenset({"https"})


_RESOURCE_ATTRIBUTES = frozenset({"poster", "src", "srcset"})


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
