"""Canonical JSON and file byte digests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value using canonical key and separator order."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()



def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of ``path`` without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

