"""Phase-specific ordinary draft and published metadata-only updates."""

from __future__ import annotations

from typing import Any, Mapping

from publication.zenodo_admission import _deposition, _doi_for_record, _integer_id, _required_doi
from publication.zenodo_errors import ZenodoError
from publication.zenodo_lineage import _file_set_identity, _require_same_record_line
from publication.zenodo_metadata import _comparable_metadata, _require_metadata_match
from publication.zenodo_models import ZenodoDeposition
from publication.zenodo_operations import _ZenodoOperations


class _ZenodoMetadataOperations(_ZenodoOperations):
    """Ordinary draft updates and explicit published metadata-only edits."""

    def update_metadata(self, deposition_id: int, metadata: Mapping[str, Any]) -> ZenodoDeposition:
        """Replace editable deposition metadata without publishing it."""
        deposition = self._editable_deposition(deposition_id)
        payload_metadata = dict(metadata)
        requested_doi = payload_metadata.pop("doi", None)
        normalized_requested_doi = _required_doi(
            requested_doi,
            "requested metadata DOI",
        )
        normalized_reserved_doi = _required_doi(
            deposition.reserved_doi,
            "draft reserved DOI",
        )
        if normalized_requested_doi != normalized_reserved_doi:
            raise ZenodoError(
                "Zenodo draft reserved DOI does not match the requested metadata DOI"
            )
        _comparable_metadata(payload_metadata)
        payload = self._request(
            "PUT",
            f"deposit/depositions/{_integer_id(deposition_id)}",
            payload={"metadata": payload_metadata},
        )
        updated = _deposition(payload)
        _require_same_record_line(deposition, updated, operation="metadata update response")
        refetched = self.get_deposition(deposition.id)
        _require_same_record_line(deposition, refetched, operation="metadata update refetch")
        if refetched.state != "unsubmitted":
            raise ZenodoError("Zenodo metadata update no longer targets an unsubmitted draft")
        if _required_doi(refetched.reserved_doi, "draft reserved DOI") != normalized_reserved_doi:
            raise ZenodoError("Zenodo metadata update changed the draft DOI reservation")
        _require_metadata_match(
            payload_metadata,
            refetched.metadata,
            operation="updated draft",
        )
        return refetched

    def edit_published_metadata(
        self, deposition_id: int, metadata: Mapping[str, Any]
    ) -> ZenodoDeposition:
        """Open and update a published record's metadata-only edit draft.

        Zenodo permits metadata corrections on a published record without
        changing its DOI.  This is deliberately separate from
        :meth:`update_metadata`, which is reserved for ordinary unsubmitted
        drafts and never unlocks a published deposition.  Files are not
        touched by this operation.
        """
        source = self.get_deposition(deposition_id)
        if source.state not in {"done", "inprogress"}:
            raise ZenodoError(
                f"Zenodo deposition {source.id} is {source.state!r}; "
                "published metadata edits require the latest published record"
            )
        if source.record_id != source.id or source.concept_record_id is None:
            raise ZenodoError("Zenodo published metadata edit has incomplete record identity")
        source_doi = _doi_for_record(source.doi, source.id, "published record DOI")
        source_concept_doi = _doi_for_record(
            source.concept_doi,
            source.concept_record_id,
            "published concept DOI",
        )
        payload_metadata = dict(metadata)
        requested_doi = _required_doi(
            payload_metadata.pop("doi", None),
            "requested metadata-edit DOI",
        )
        if requested_doi != source_doi:
            raise ZenodoError(
                "Zenodo published DOI does not match the requested metadata-edit DOI"
            )
        payload_metadata.setdefault("access_right", "open")
        _comparable_metadata(payload_metadata)
        if source.state == "done":
            payload = self._request(
                "POST",
                f"deposit/depositions/{_integer_id(deposition_id)}/actions/edit",
            )
            editable = _deposition(payload)
            _require_same_record_line(source, editable, operation="metadata-edit response")
            if editable.state != "inprogress":
                raise ZenodoError("Zenodo metadata-edit action did not open an editable draft")
        else:
            # Re-running the explicit operation against its already-open edit
            # draft must update the draft rather than attempt a second action.
            editable = source
        if _required_doi(editable.doi, "metadata-edit DOI") != source_doi:
            raise ZenodoError("Zenodo metadata-edit draft changed the published DOI")
        payload = self._request(
            "PUT",
            f"deposit/depositions/{_integer_id(deposition_id)}",
            payload={"metadata": payload_metadata},
        )
        updated = _deposition(payload)
        _require_same_record_line(editable, updated, operation="metadata-edit update response")
        refetched = self.get_deposition(editable.id)
        _require_same_record_line(editable, refetched, operation="metadata-edit update refetch")
        if refetched.state != "inprogress":
            raise ZenodoError("Zenodo metadata-edit draft is no longer editable")
        if _required_doi(refetched.doi, "metadata-edit DOI") != source_doi:
            raise ZenodoError("Zenodo metadata-edit draft changed the published DOI")
        if _file_set_identity(refetched.files) != _file_set_identity(source.files):
            raise ZenodoError("Zenodo metadata-only edit changed the record files")
        _require_metadata_match(
            payload_metadata,
            refetched.metadata,
            operation="published metadata edit",
        )
        self._metadata_edit_identities[refetched.id] = (
            refetched.record_id,
            refetched.concept_record_id,
            source_concept_doi,
            source_doi,
        )
        return refetched
