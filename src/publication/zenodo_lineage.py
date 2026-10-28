"""Record-lineage and historical/current linked-version admission."""

from __future__ import annotations

from dataclasses import replace

from publication.zenodo_admission import _doi_for_record, _optional_utc_timestamp, _required_doi
from publication.zenodo_errors import ZenodoError
from publication.zenodo_metadata import _comparable_metadata, _require_metadata_match
from publication.zenodo_models import (
    ZenodoDeposition,
    ZenodoFile,
    ZenodoMetadataSnapshot,
    _ZenodoLinkedVersionShape,
)


def _file_set_identity(files: tuple[ZenodoFile, ...]) -> tuple[tuple[str, int, str], ...]:
    """Return an order-independent semantic file identity."""
    return tuple(
        sorted(
            (
                file.filename,
                file.filesize,
                file.checksum.removeprefix("md5:").casefold(),
            )
            for file in files
        )
    )


def _require_same_record_line(
    expected: ZenodoDeposition,
    actual: ZenodoDeposition,
    *,
    operation: str,
) -> None:
    """Bind a response/refetch to the same record and concept lineage."""
    if actual.id != expected.id or actual.record_id != expected.record_id:
        raise ZenodoError(f"Zenodo {operation} changed the record identity")
    if actual.concept_record_id != expected.concept_record_id:
        raise ZenodoError(f"Zenodo {operation} changed the concept identity")
    expected_concept_doi = _required_doi(expected.concept_doi, "concept DOI")
    actual_concept_doi = _required_doi(actual.concept_doi, "concept DOI")
    if actual_concept_doi != expected_concept_doi:
        raise ZenodoError(f"Zenodo {operation} changed the concept identity")


def _require_current_separate_record_metadata(
    source: ZenodoDeposition,
    draft: ZenodoDeposition,
) -> None:
    """Validate the current Zenodo new-version metadata normalization.

    Current Zenodo creates a separate empty draft. It retains caller-purpose
    metadata, resets ``publication_date`` to the draft's UTC creation date,
    and may omit the source version. No other purpose-metadata drift is safe.
    """
    if draft.created_utc is None:
        raise ZenodoError(
            "Zenodo separate-record linked draft has no parseable creation timestamp"
        )
    expected = _comparable_metadata(source.metadata.as_dict()).as_dict()
    actual = _comparable_metadata(draft.metadata.as_dict()).as_dict()

    missing = object()
    expected_version = expected.pop("version", missing)
    actual_has_version = "version" in actual
    actual_version = actual.pop("version", None)
    if actual_has_version and (
        expected_version is missing or actual_version != expected_version
    ):
        raise ZenodoError("Zenodo linked draft version does not match the published source")

    expected.pop("publication_date", None)
    actual_publication_date = actual.pop("publication_date", None)
    creation_date = draft.created_utc[:10]
    if actual_publication_date != creation_date:
        raise ZenodoError(
            "Zenodo linked draft publication_date does not match its UTC creation date"
        )

    expected_snapshot = ZenodoMetadataSnapshot.from_mapping(expected)
    actual_snapshot = ZenodoMetadataSnapshot.from_mapping(actual)
    if expected_snapshot.canonical_json != actual_snapshot.canonical_json:
        raise ZenodoError(
            "Zenodo linked draft purpose metadata differs from the published source"
        )


def _require_linked_version(
    source: ZenodoDeposition,
    draft: ZenodoDeposition,
    *,
    created: object,
) -> tuple[_ZenodoLinkedVersionShape, str | None]:
    """Prove a linked version's identity, normalized metadata and file inheritance."""
    if source.record_id != source.id or draft.record_id != draft.id:
        raise ZenodoError("Zenodo linked version has incomplete record identity")
    if source.concept_record_id is None:
        raise ZenodoError("Zenodo source record has no concept identity")
    source_doi = _doi_for_record(source.doi, source.id, "source record DOI")
    _doi_for_record(
        source.concept_doi,
        source.concept_record_id,
        "source concept DOI",
    )
    if draft.concept_record_id != source.concept_record_id:
        raise ZenodoError("Zenodo linked draft belongs to a different concept record")
    if _required_doi(draft.concept_doi, "draft concept DOI") != _required_doi(
        source.concept_doi,
        "source concept DOI",
    ):
        raise ZenodoError("Zenodo linked draft belongs to a different concept DOI")
    if draft.reserved_record_id != draft.id:
        raise ZenodoError("Zenodo linked draft reservation does not identify the draft record")
    reserved_doi = _doi_for_record(draft.reserved_doi, draft.id, "draft reserved DOI")
    if draft.doi is not None and _required_doi(draft.doi, "draft DOI") != reserved_doi:
        raise ZenodoError("Zenodo linked draft DOI and reservation disagree")
    if source_doi == reserved_doi:
        raise ZenodoError("Zenodo linked draft reused the published source DOI")

    source_metadata = _comparable_metadata(source.metadata.as_dict())
    draft_metadata = _comparable_metadata(draft.metadata.as_dict())
    metadata_is_inherited = source_metadata.canonical_json == draft_metadata.canonical_json
    files_are_inherited = _file_set_identity(source.files) == _file_set_identity(draft.files)
    if metadata_is_inherited and files_are_inherited:
        try:
            created_utc = _optional_utc_timestamp(created, "creation timestamp")
        except ZenodoError:
            # The legacy shape is proven by exact metadata and file inheritance;
            # its optional creation timestamp is informative rather than part of
            # that proof. Preserve a parseable value without making unrelated
            # legacy wire spelling a compatibility gate.
            created_utc = None
        return "legacy_inherited", created_utc

    created_utc = _optional_utc_timestamp(created, "creation timestamp")
    draft = replace(draft, created_utc=created_utc)

    current_shape_error: ZenodoError | None = None
    try:
        _require_current_separate_record_metadata(source, draft)
    except ZenodoError as exc:
        current_shape_error = exc
    else:
        if draft.files and not files_are_inherited:
            raise ZenodoError(
                "Zenodo separate-record linked draft must have an empty file set "
                "or the exact inherited file set"
            )
        if draft.files:
            return "current_inherited_files", created_utc
        return "current_separate_record", created_utc

    if metadata_is_inherited:
        raise ZenodoError(
            "Zenodo linked draft file set is neither the exact inherited set nor "
            "the empty separate-record set"
        )
    assert current_shape_error is not None  # guarded by the successful-shape return
    raise current_shape_error


def _require_published_snapshot(
    expected: ZenodoDeposition,
    actual: ZenodoDeposition,
    *,
    expected_doi: str,
    operation: str,
) -> None:
    """Validate one irreversible-action response or immediate refetch."""
    _require_same_record_line(expected, actual, operation=operation)
    if actual.state != "done":
        raise ZenodoError(f"Zenodo {operation} did not return a published record")
    if _required_doi(actual.doi, "published DOI") != expected_doi:
        raise ZenodoError(f"Zenodo {operation} changed the published DOI")
    if _file_set_identity(actual.files) != _file_set_identity(expected.files):
        raise ZenodoError(f"Zenodo {operation} changed the verified file set")
    _require_metadata_match(
        expected.metadata.as_dict(),
        actual.metadata,
        operation=operation,
    )
