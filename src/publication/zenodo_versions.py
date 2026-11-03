"""New-version and existing-draft recovery over the canonical HTTP transport."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping
from urllib.parse import urlencode

from publication.zenodo_admission import _deposition, _doi_for_record, _integer_id
from publication.zenodo_errors import ZenodoError, _ZenodoHTTPError
from publication.zenodo_lineage import (
    _file_set_identity,
    _require_linked_version,
    _require_published_snapshot,
)
from publication.zenodo_models import ZenodoDeposition
from publication.zenodo_operations import _ZenodoOperations
from publication.zenodo_response import _is_existing_draft_response

_DEPOSITION_LIST_PAGE_SIZE = 100


_FULL_DEPOSITION_FIELDS = frozenset(
    {"id", "record_id", "conceptrecid", "conceptdoi", "state", "metadata", "links", "files"}
)


def _latest_draft_link(
    payload: Mapping[Any, Any],
    *,
    operation: str,
    required: bool,
) -> str | None:
    """Read one authoritative ``latest_draft`` link without following it."""
    links = payload.get("links")
    if not isinstance(links, Mapping):
        raise ZenodoError(f"Zenodo {operation} has malformed links")
    latest_draft = links.get("latest_draft")
    if latest_draft is None and not required:
        return None
    if (
        not isinstance(latest_draft, str)
        or not latest_draft
        or latest_draft != latest_draft.strip()
    ):
        raise ZenodoError(f"Zenodo {operation} has no valid latest_draft link")
    return latest_draft


class _ZenodoVersionOperations(_ZenodoOperations):
    """Origin-bound linked-version creation and historical draft recovery."""

    @staticmethod
    def _validate_linked_draft_object(
        source: ZenodoDeposition,
        draft: ZenodoDeposition,
        *,
        created: object,
    ) -> ZenodoDeposition:
        """Apply the complete linked-draft contract to one full object."""
        if draft.state != "unsubmitted" or draft.reserved_doi is None:
            raise ZenodoError(
                "Zenodo latest_draft is not an unsubmitted draft with a reserved DOI"
            )
        linked_version_shape, created_utc = _require_linked_version(
            source,
            draft,
            created=created,
        )
        return replace(
            draft,
            created_utc=created_utc,
            linked_version_shape=linked_version_shape,
        )

    def _validated_linked_draft_id(
        self,
        source: ZenodoDeposition,
        draft_id: int,
    ) -> ZenodoDeposition:
        """Fetch and fully validate one distinct draft deposition id."""
        draft, payload = self._get_deposition_response(draft_id)
        return self._validate_linked_draft_object(
            source,
            draft,
            created=payload.get("created"),
        )

    def _validated_linked_draft(
        self,
        source: ZenodoDeposition,
        latest_draft: str,
    ) -> ZenodoDeposition:
        """Resolve and fully revalidate one authoritative latest-draft link."""
        return self._validated_linked_draft_id(
            source,
            self._latest_draft_id(latest_draft),
        )

    def _listed_existing_draft(self, source: ZenodoDeposition) -> ZenodoDeposition:
        """Recover exactly one full concept-linked draft from a bounded listing."""
        if source.concept_record_id is None:
            raise ZenodoError("Zenodo source record has no concept identity")
        query = urlencode(
            (
                ("q", f"conceptrecid:{source.concept_record_id}"),
                ("status", "draft"),
                ("sort", "mostrecent"),
                ("page", "1"),
                ("size", str(_DEPOSITION_LIST_PAGE_SIZE)),
                ("all_versions", "true"),
            )
        )
        payload = self._request("GET", f"deposit/depositions?{query}")
        if not isinstance(payload, list):
            raise ZenodoError("Zenodo draft listing response is malformed")
        if len(payload) >= _DEPOSITION_LIST_PAGE_SIZE:
            raise ZenodoError("Zenodo draft listing may be truncated")

        candidates: list[tuple[ZenodoDeposition, Mapping[Any, Any]]] = []
        for item in payload:
            if not isinstance(item, Mapping) or not _FULL_DEPOSITION_FIELDS.issubset(
                item.keys()
            ):
                raise ZenodoError("Zenodo draft listing contains a partial deposition")
            deposition = _deposition(item)
            if deposition.id == source.id:
                continue
            if deposition.concept_record_id != source.concept_record_id:
                continue
            if deposition.state != "unsubmitted":
                raise ZenodoError(
                    "Zenodo concept-linked listing candidate is not unsubmitted"
                )
            candidates.append((deposition, item))

        if len(candidates) != 1:
            raise ZenodoError(
                "Zenodo draft listing did not return exactly one distinct "
                "concept-linked draft"
            )
        listed_deposition, listed_payload = candidates[0]
        listed = self._validate_linked_draft_object(
            source,
            listed_deposition,
            created=listed_payload.get("created"),
        )
        refetched = self._validated_linked_draft_id(source, listed.id)
        if (
            listed.metadata.canonical_json != refetched.metadata.canonical_json
            or _file_set_identity(listed.files) != _file_set_identity(refetched.files)
            or listed.created_utc != refetched.created_utc
            or listed.reserved_doi != refetched.reserved_doi
            or listed.reserved_record_id != refetched.reserved_record_id
        ):
            raise ZenodoError("Zenodo listed draft changed during validation")
        return refetched

    def new_version(self, deposition_id: int) -> ZenodoDeposition:
        """Create or retrieve the unpublished next version of a published record.

        A published source may already expose its authoritative
        ``links.latest_draft``. Otherwise the ``newversion`` action supplies
        that link. After Zenodo's exact already-exists response, recovery
        requires a fresh, unchanged source and either its distinct draft link
        or exactly one full concept-linked draft from a bounded listing.
        Every path resolves and fully validates the draft before returning it.
        """
        source, source_payload = self._get_deposition_response(deposition_id)
        if source.state != "done":
            raise ZenodoError(
                f"Zenodo deposition {source.id} is {source.state!r}; "
                "new versions require the latest published record"
            )
        latest_draft = _latest_draft_link(
            source_payload,
            operation="published source",
            required=False,
        )
        if latest_draft is not None:
            latest_draft_id = self._latest_draft_id(latest_draft)
            if latest_draft_id != source.id:
                return self._validated_linked_draft_id(source, latest_draft_id)

        try:
            action_payload = self._request(
                "POST",
                f"deposit/depositions/{_integer_id(deposition_id)}/actions/newversion",
            )
        except _ZenodoHTTPError as exc:
            if not _is_existing_draft_response(exc):
                raise
            recovered_source, recovered_payload = self._get_deposition_response(
                deposition_id
            )
            _require_published_snapshot(
                source,
                recovered_source,
                expected_doi=_doi_for_record(source.doi, source.id, "source record DOI"),
                operation="existing-draft source refetch",
            )
            latest_draft = _latest_draft_link(
                recovered_payload,
                operation="existing-draft source refetch",
                required=False,
            )
            source = recovered_source
            if latest_draft is not None:
                latest_draft_id = self._latest_draft_id(latest_draft)
                if latest_draft_id != source.id:
                    return self._validated_linked_draft_id(
                        source,
                        latest_draft_id,
                    )
            return self._listed_existing_draft(source)
        else:
            if not isinstance(action_payload, Mapping):
                raise ZenodoError("Zenodo new-version response is malformed")
            latest_draft = _latest_draft_link(
                action_payload,
                operation="new-version response",
                required=True,
            )
            assert latest_draft is not None  # required=True guarantees a string
            latest_draft_id = self._latest_draft_id(latest_draft)
            if latest_draft_id == source.id:
                raise ZenodoError(
                    "Zenodo new-version response linked the published source "
                    "instead of a distinct draft"
                )
            return self._validated_linked_draft_id(source, latest_draft_id)
