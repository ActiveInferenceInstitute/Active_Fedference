"""Admission of untrusted Zenodo identifiers, metadata, timestamps and files."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Mapping

from publication.identifiers import normalize_doi
from publication.zenodo_errors import ZenodoError
from publication.zenodo_models import ZenodoDeposition, ZenodoFile, ZenodoMetadataSnapshot

_RFC3339_TIMESTAMP_RE = re.compile(
    r"\A(?P<date_time>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?P<fraction>\.\d{1,6})?(?P<offset>Z|[+-]\d{2}:\d{2})\Z"
)


def _optional_text(value: object, field: str) -> str | None:
    """Normalize an optional server text field and reject typed corruption."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ZenodoError(f"Zenodo returned an invalid {field}")
    return value


def _optional_utc_timestamp(value: object, field: str) -> str | None:
    """Parse one optional strict RFC 3339 wire timestamp and canonicalize UTC."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ZenodoError(f"Zenodo returned an invalid {field}")
    match = _RFC3339_TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise ZenodoError(f"Zenodo returned an invalid {field}")
    offset = match.group("offset")
    if offset != "Z":
        if int(offset[1:3]) > 23 or int(offset[4:6]) > 59:
            raise ZenodoError(f"Zenodo returned an invalid {field}")
    fraction = match.group("fraction")
    # Python 3.10 accepts only three or six fractional digits here, while the
    # validated RFC 3339 boundary permits every precision from one through six.
    normalized_fraction = (
        f".{fraction[1:].ljust(6, '0')}" if fraction is not None else ""
    )
    normalized_offset = "+00:00" if offset == "Z" else offset
    iso_value = (
        f"{match.group('date_time')}{normalized_fraction}{normalized_offset}"
    )
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError as exc:
        raise ZenodoError(f"Zenodo returned an invalid {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ZenodoError(f"Zenodo returned an invalid {field}")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _integer_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ZenodoError(f"invalid Zenodo deposition id: {value!r}")
    return value


def _optional_integer_id(value: object, field: str) -> int | None:
    """Normalize optional legacy integer-or-decimal-string identifiers."""
    if value is None:
        return None
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    try:
        return _integer_id(value)
    except ZenodoError as exc:
        raise ZenodoError(f"Zenodo returned an invalid {field}") from exc


def _required_doi(value: object, field: str) -> str:
    """Normalize a required DOI while retaining a boundary-specific error."""
    try:
        normalized = normalize_doi(value)
    except ValueError as exc:
        raise ZenodoError(f"Zenodo {field} is invalid") from exc
    if normalized is None:
        raise ZenodoError(f"Zenodo {field} is required")
    return normalized


def _doi_for_record(value: object, record_id: int, field: str) -> str:
    """Require one production/sandbox Zenodo DOI to name *record_id*."""
    normalized = _required_doi(value, field)
    if not normalized.casefold().endswith(f"/zenodo.{record_id}"):
        raise ZenodoError(f"Zenodo {field} does not identify record {record_id}")
    return normalized


def _file_record(payload: object) -> ZenodoFile:
    if not isinstance(payload, Mapping):
        raise ZenodoError("Zenodo returned a malformed file record")
    try:
        file_id = payload["id"]
        filename = payload["filename"]
        filesize = payload["filesize"]
        checksum = payload["checksum"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ZenodoError("Zenodo returned a malformed file record") from exc
    if (
        not isinstance(file_id, str)
        or not file_id.strip()
        or not isinstance(filename, str)
        or not filename.strip()
        or isinstance(filesize, bool)
        or not isinstance(filesize, (int, float))
        or not math.isfinite(filesize)
        or not float(filesize).is_integer()
        or filesize < 0
        or not isinstance(checksum, str)
        or not checksum.strip()
    ):
        raise ZenodoError("Zenodo returned an invalid file record")
    # Zenodo's upload response currently serializes ``filesize`` as a JSON
    # number with a ``.0`` suffix, while deposition listings use an integer.
    # Accept only an exact non-negative integral float so this wire-format
    # variation does not weaken the typed boundary.
    return ZenodoFile(file_id, filename, int(filesize), checksum)


def _deposition(payload: object) -> ZenodoDeposition:
    if not isinstance(payload, Mapping):
        raise ZenodoError("Zenodo returned a malformed deposition")
    try:
        deposition_id = _integer_id(payload["id"])
        state = payload["state"]
        metadata = payload.get("metadata", {})
        links = payload.get("links", {})
        files_payload = payload.get("files", [])
    except (KeyError, TypeError, ValueError) as exc:
        raise ZenodoError("Zenodo returned a malformed deposition") from exc
    if not isinstance(state, str) or not state.strip():
        raise ZenodoError("Zenodo returned an invalid deposition state")
    if not isinstance(metadata, Mapping) or not isinstance(links, Mapping):
        raise ZenodoError("Zenodo returned malformed deposition metadata or links")
    record_id = _optional_integer_id(payload.get("record_id"), "record id")
    if record_id is not None and record_id != deposition_id:
        raise ZenodoError("Zenodo deposition id and record id disagree")
    concept_record_id = _optional_integer_id(
        payload.get("conceptrecid"),
        "concept record id",
    )
    reserved = metadata.get("prereserve_doi")
    if reserved is not None and not isinstance(reserved, Mapping):
        raise ZenodoError("Zenodo returned malformed DOI reservation metadata")
    reserved_doi = (
        _optional_text(reserved.get("doi"), "reserved DOI")
        if isinstance(reserved, Mapping)
        else None
    )
    reserved_record_id = (
        _optional_integer_id(reserved.get("recid"), "reserved record id")
        if isinstance(reserved, Mapping)
        else None
    )
    if reserved_record_id is not None and reserved_record_id != deposition_id:
        raise ZenodoError("Zenodo reserved record id and deposition id disagree")
    if reserved_doi is not None and reserved_record_id is not None:
        _doi_for_record(reserved_doi, reserved_record_id, "reserved DOI")
    concept_doi = _optional_text(payload.get("conceptdoi"), "concept DOI")
    if concept_doi is not None and concept_record_id is not None:
        _doi_for_record(concept_doi, concept_record_id, "concept DOI")
    doi = _optional_text(payload.get("doi"), "DOI")
    if doi is not None and record_id is not None:
        _doi_for_record(doi, record_id, "record DOI")
    if not isinstance(files_payload, list):
        raise ZenodoError("Zenodo returned malformed deposition files")
    files = tuple(_file_record(item) for item in files_payload)
    return ZenodoDeposition(
        id=deposition_id,
        record_id=record_id,
        concept_record_id=concept_record_id,
        concept_doi=concept_doi,
        state=state,
        doi=doi,
        reserved_doi=reserved_doi,
        reserved_record_id=reserved_record_id,
        html_url=_optional_text(links.get("html"), "HTML link"),
        self_url=_optional_text(links.get("self"), "self link"),
        bucket_url=_optional_text(links.get("bucket"), "bucket link"),
        publish_url=_optional_text(links.get("publish"), "publish link"),
        metadata=ZenodoMetadataSnapshot.from_mapping(metadata),
        files=files,
    )
