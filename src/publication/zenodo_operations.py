"""Shared reversible deposition operations and phase guards."""

from __future__ import annotations

from typing import Any, Mapping

from publication.zenodo_admission import _deposition, _integer_id
from publication.zenodo_errors import ZenodoError
from publication.zenodo_models import ZenodoDeposition
from publication.zenodo_transport import ZenodoTransport


class _ZenodoOperations(ZenodoTransport):
    """Shared retrieval, draft-state admission and reversible DOI reservation."""

    def _editable_deposition(self, deposition_id: int) -> ZenodoDeposition:
        """Return a draft and fail closed for irreversible/published records."""
        deposition = self.get_deposition(deposition_id)
        if deposition.state != "unsubmitted":
            raise ZenodoError(
                f"Zenodo deposition {deposition.id} is {deposition.state!r}; "
                "only an unsubmitted draft is editable"
            )
        return deposition

    def _get_deposition_response(
        self,
        deposition_id: int,
    ) -> tuple[ZenodoDeposition, Mapping[Any, Any]]:
        """Retrieve both the typed deposition and its server response mapping."""
        payload = self._request(
            "GET",
            f"deposit/depositions/{_integer_id(deposition_id)}",
        )
        deposition = _deposition(payload)
        if not isinstance(payload, Mapping):  # pragma: no cover - _deposition rejects it
            raise ZenodoError("Zenodo returned a malformed deposition")
        return deposition, payload

    def get_deposition(self, deposition_id: int) -> ZenodoDeposition:
        """Retrieve one deposition and normalize its public fields."""
        deposition, _payload = self._get_deposition_response(deposition_id)
        return deposition

    def reserve_doi(self, metadata: Mapping[str, Any]) -> ZenodoDeposition:
        """Create an unsubmitted draft and reserve a DOI without publishing."""
        payload_metadata = dict(metadata)
        payload_metadata.pop("doi", None)
        payload_metadata.setdefault("access_right", "open")
        payload_metadata["prereserve_doi"] = True
        payload = self._request("POST", "deposit/depositions", payload={"metadata": payload_metadata})
        return _deposition(payload)
