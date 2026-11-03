"""Immutable public Zenodo file, metadata and deposition representations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from publication.zenodo_errors import ZenodoError

_ZenodoLinkedVersionShape = Literal[
    "legacy_inherited", "current_separate_record", "current_inherited_files"
]


@dataclass(frozen=True)
class ZenodoFile:
    """The server-side identity and checksum of one deposition file."""

    id: str
    filename: str
    filesize: int
    checksum: str


@dataclass(frozen=True)
class ZenodoMetadataSnapshot:
    """Immutable, canonical copy of one deposition's server metadata.

    The legacy deposition API returns the complete editable metadata mapping.
    Keeping its canonical JSON rather than a mutable ``dict`` lets release
    receipts bind the exact draft purpose while preserving the frozen result
    contract. The adapter never adds a bearer token or local env-file path to
    this server-owned metadata.
    """

    canonical_json: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ZenodoMetadataSnapshot:
        """Validate and freeze one JSON-compatible metadata mapping."""
        try:
            canonical = json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ZenodoError("Zenodo returned non-JSON deposition metadata") from exc
        return cls(canonical)

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical semantic metadata object."""
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible metadata mapping."""
        payload = json.loads(self.canonical_json)
        if not isinstance(payload, dict):  # pragma: no cover - constructor guarantees an object
            raise ZenodoError("canonical Zenodo metadata is not an object")
        return payload


@dataclass(frozen=True)
class ZenodoDeposition:
    """Stable subset of a Zenodo deposition response."""

    id: int
    record_id: int | None
    concept_record_id: int | None
    concept_doi: str | None
    state: str
    doi: str | None
    reserved_doi: str | None
    reserved_record_id: int | None
    html_url: str | None
    self_url: str | None
    bucket_url: str | None
    publish_url: str | None
    metadata: ZenodoMetadataSnapshot
    files: tuple[ZenodoFile, ...]
    created_utc: str | None = None
    linked_version_shape: _ZenodoLinkedVersionShape | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a token- and local-path-free inspection summary."""
        return {
            "id": self.id,
            "record_id": self.record_id,
            "concept_record_id": self.concept_record_id,
            "concept_doi": self.concept_doi,
            "state": self.state,
            "doi": self.doi,
            "reserved_doi": self.reserved_doi,
            "reserved_record_id": self.reserved_record_id,
            "html_url": self.html_url,
            "created_utc": self.created_utc,
            "linked_version_shape": self.linked_version_shape,
            "metadata_sha256": self.metadata.sha256,
            "metadata": self.metadata.as_dict(),
            "files": [
                {
                    "filename": file.filename,
                    "filesize": file.filesize,
                    "checksum": file.checksum,
                }
                for file in self.files
            ],
        }
