"""Script integrity for source-preserving HTML publication."""
from __future__ import annotations

import base64
import binascii
import re
from urllib.parse import urlparse

from publication.web.lexical import _HTML_ASCII_WHITESPACE, _HTML_ASCII_WHITESPACE_RE, _html_ascii_lower
from publication.web.models import _HTMLAttributeOccurrence

_SRI_DIGEST_BYTES = {"sha256": 32, "sha384": 48, "sha512": 64}


_SRI_VALUE_RE = re.compile(
    r"(?P<algorithm>sha(?:256|384|512))-(?P<digest>[A-Za-z0-9+/]+={0,2})"
)


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
