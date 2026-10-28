"""Explicit publication phases over verified deposition identities."""

from __future__ import annotations

from pathlib import Path

from publication.zenodo_admission import _deposition, _integer_id, _required_doi
from publication.zenodo_errors import ZenodoError
from publication.zenodo_files import _pdf_path, _ZenodoFileOperations
from publication.zenodo_lineage import _require_published_snapshot
from publication.zenodo_models import ZenodoDeposition


class _ZenodoPublicationOperations(_ZenodoFileOperations):
    """Explicit irreversible operations with phase-specific admission."""

    def publish_verified_pdf(
        self,
        deposition_id: int,
        pdf_path: str | Path,
        *,
        remote_filename: str | None = None,
    ) -> ZenodoDeposition:
        """Verify the exact one-PDF draft and then publish it.

        The ordinary :meth:`publish` operation remains available to explicit
        lower-level callers.  Release orchestration uses this stricter surface
        so an inherited or already-present unrelated file blocks the
        irreversible action. The response and an immediate refetch are also
        checked; separate GET and POST requests cannot eliminate a concurrent
        service-level race.
        """
        path = _pdf_path(pdf_path)
        deposition = self._editable_deposition(deposition_id)
        self._verify_pdf_record(
            deposition,
            path,
            remote_filename=remote_filename,
            require_exact_file_set=True,
        )
        expected_doi = _required_doi(deposition.reserved_doi, "draft reserved DOI")
        payload = self._request(
            "POST",
            f"deposit/depositions/{_integer_id(deposition_id)}/actions/publish",
        )
        published = _deposition(payload)
        _require_published_snapshot(
            deposition,
            published,
            expected_doi=expected_doi,
            operation="publish response",
        )
        refetched = self.get_deposition(deposition.id)
        _require_published_snapshot(
            deposition,
            refetched,
            expected_doi=expected_doi,
            operation="post-publish refetch",
        )
        return refetched

    def publish(self, deposition_id: int) -> ZenodoDeposition:
        """Publish a deposition explicitly; callers must gate this action."""
        self._editable_deposition(deposition_id)
        payload = self._request("POST", f"deposit/depositions/{_integer_id(deposition_id)}/actions/publish")
        return _deposition(payload)

    def publish_metadata_edit(self, deposition_id: int) -> ZenodoDeposition:
        """Publish an already-updated metadata-only edit draft."""
        expected_identity = self._metadata_edit_identities.get(
            _integer_id(deposition_id)
        )
        if expected_identity is None:
            raise ZenodoError(
                "Zenodo metadata edit must be opened and verified by this client before publication"
            )
        deposition = self.get_deposition(deposition_id)
        if deposition.state != "inprogress":
            raise ZenodoError(
                f"Zenodo deposition {deposition.id} is {deposition.state!r}; "
                "only an open metadata edit can be published"
            )
        actual_identity = (
            deposition.record_id,
            deposition.concept_record_id,
            _required_doi(deposition.concept_doi, "metadata-edit concept DOI"),
            _required_doi(deposition.doi, "metadata-edit DOI"),
        )
        if actual_identity != expected_identity:
            raise ZenodoError("Zenodo metadata-edit identity changed before publication")
        payload = self._request(
            "POST",
            f"deposit/depositions/{_integer_id(deposition_id)}/actions/publish",
        )
        published = _deposition(payload)
        _require_published_snapshot(
            deposition,
            published,
            expected_doi=expected_identity[3],
            operation="metadata-edit publish response",
        )
        refetched = self.get_deposition(deposition.id)
        _require_published_snapshot(
            deposition,
            refetched,
            expected_doi=expected_identity[3],
            operation="metadata-edit post-publish refetch",
        )
        del self._metadata_edit_identities[deposition.id]
        return refetched
