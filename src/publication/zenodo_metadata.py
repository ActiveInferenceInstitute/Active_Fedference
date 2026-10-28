"""Canonical comparisons for caller-owned Zenodo purpose metadata."""

from __future__ import annotations

from typing import Any, Mapping

from publication.zenodo_errors import ZenodoError
from publication.zenodo_models import ZenodoMetadataSnapshot

_SERVER_OWNED_METADATA_FIELDS = frozenset({"doi", "prereserve_doi"})


def _comparable_metadata(payload: Mapping[str, Any]) -> ZenodoMetadataSnapshot:
    """Canonicalize caller-owned metadata, excluding explicit server fields."""
    return ZenodoMetadataSnapshot.from_mapping(
        {
            key: value
            for key, value in payload.items()
            if key not in _SERVER_OWNED_METADATA_FIELDS
        }
    )


def _request_metadata_comparison(payload: Mapping[str, Any]) -> ZenodoMetadataSnapshot:
    """Compare a request with the observed legacy API software representation.

    Zenodo returns the MIT license as ``mit-license`` and supplies publisher
    ``Zenodo`` when software metadata omits it. Admit only these known
    equivalents; retain explicit publisher values and every other field.
    This projection never changes the request, stored server snapshot, or
    strict inherited-purpose comparison used for linked-version recovery.
    """
    comparable = _comparable_metadata(payload).as_dict()
    if comparable.get("license") in ("MIT", "mit-license"):
        comparable["license"] = "MIT"
    if comparable.get("upload_type") == "software":
        comparable.setdefault("imprint_publisher", "Zenodo")
    return ZenodoMetadataSnapshot.from_mapping(comparable)


def _require_metadata_match(
    expected: Mapping[str, Any],
    actual: ZenodoMetadataSnapshot,
    *,
    operation: str,
) -> None:
    """Fail when Zenodo did not retain the complete requested metadata."""
    expected_snapshot = _request_metadata_comparison(expected)
    actual_snapshot = _request_metadata_comparison(actual.as_dict())
    if expected_snapshot.canonical_json != actual_snapshot.canonical_json:
        raise ZenodoError(
            f"Zenodo {operation} metadata does not match the canonical request"
        )
