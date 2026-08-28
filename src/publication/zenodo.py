"""Typed, composable boundary adapter for a Zenodo deposition.

The adapter intentionally has no project-specific paths and never prints or
stores bearer tokens.  Callers supply metadata, a deposition id, and files;
the thin CLI in ``scripts/zenodo_release.py`` supplies the release workflow.
Publication remains an explicit action rather than an implicit side effect of
reserving a DOI or uploading a file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from publication.identifiers import normalize_doi

DEFAULT_ZENODO_API = "https://zenodo.org/api"
DEFAULT_TOKEN_ENV_NAMES: tuple[str, ...] = (
    "ZENODO_PROD_TOKEN",
    "ZENODO_TOKEN",
    "ZENODO_API_TOKEN",
)
_SERVER_OWNED_METADATA_FIELDS = frozenset({"doi", "prereserve_doi"})
_ZenodoLinkedVersionShape = Literal["legacy_inherited", "current_separate_record"]
_RFC3339_TIMESTAMP_RE = re.compile(
    r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})\Z"
)
_ALREADY_EXISTS_RESPONSE = {"status": 400, "message": "A draft already exists."}
_DEPOSITION_LIST_PAGE_SIZE = 100
_FULL_DEPOSITION_FIELDS = frozenset(
    {"id", "record_id", "conceptrecid", "conceptdoi", "state", "metadata", "links", "files"}
)


class ZenodoError(RuntimeError):
    """Raised when Zenodo rejects or cannot complete a request."""


class _ZenodoHTTPError(ZenodoError):
    """Typed HTTP failure with no retained response body or credential echo."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        exact_already_exists: bool,
    ) -> None:
        super().__init__(message)
        self.status = status
        self._exact_already_exists = exact_already_exists


class _RejectRedirects(HTTPRedirectHandler):
    """Prevent bearer credentials from following any HTTP redirect."""

    def redirect_request(
        self,
        _request: Request,
        _file_pointer: Any,
        _code: int,
        _message: str,
        _headers: Any,
        _new_url: str,
    ) -> None:
        return None


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
    if not isinstance(value, str) or _RFC3339_TIMESTAMP_RE.fullmatch(value) is None:
        raise ZenodoError(f"Zenodo returned an invalid {field}")
    if not value.endswith("Z"):
        offset = value[-6:]
        if int(offset[1:3]) > 23 or int(offset[4:6]) > 59:
            raise ZenodoError(f"Zenodo returned an invalid {field}")
    iso_value = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError as exc:
        raise ZenodoError(f"Zenodo returned an invalid {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ZenodoError(f"Zenodo returned an invalid {field}")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_existing_draft_response(error: _ZenodoHTTPError) -> bool:
    """Recognize only Zenodo's exact production already-exists response."""
    return error.status == 400 and error._exact_already_exists


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


def _api_base(value: str) -> str:
    """Validate an API endpoint before a bearer token can be sent to it."""
    if not isinstance(value, str) or not value.strip():
        raise ZenodoError("Zenodo API base must be a non-empty URL")
    normalized = value.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise ZenodoError("Zenodo API base is not a valid URL") from exc
    if parsed.scheme not in {"https", "http"} or not parsed.netloc or hostname is None:
        raise ZenodoError("Zenodo API base must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ZenodoError("Zenodo API base must not contain URL credentials")
    if parsed.query or parsed.fragment:
        raise ZenodoError("Zenodo API base must not contain a query or fragment")
    # Plain HTTP is only acceptable for a loopback test server.  This prevents
    # an accidental production token leak when a caller misconfigures the API.
    if parsed.scheme == "http" and hostname.casefold() not in {"localhost", "127.0.0.1", "::1"}:
        raise ZenodoError("Zenodo API base must use HTTPS outside loopback test endpoints")
    return normalized


def _pdf_path(value: str | Path) -> Path:
    """Resolve an existing regular PDF path for upload or checksum work."""
    path = Path(value)
    if not path.is_file() or path.suffix.casefold() != ".pdf":
        raise ZenodoError(f"PDF file does not exist or is not a PDF: {path}")
    return path


def _safe_upload_filename(filename: str) -> str:
    """Reject multipart names that could break the Content-Disposition header."""
    if (
        not filename
        or Path(filename).name != filename
        or any(character in filename for character in '\"\r\n')
        or len(filename) > 255
    ):
        raise ZenodoError("PDF filename is not safe for multipart upload")
    return filename


def _parse_env_value(value: str) -> str:
    """Parse the small dotenv subset needed for a bearer token."""
    normalized = value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] in {"'", '"'}:
        normalized = normalized[1:-1]
    return normalized


def token_from_env_file(
    env_file: str | Path,
    *,
    names: tuple[str, ...] = DEFAULT_TOKEN_ENV_NAMES,
) -> tuple[str, str]:
    """Read the first configured Zenodo token from a dotenv-style file.

    The returned tuple is ``(token, variable_name)`` so a caller can record
    which credential slot was selected without ever logging the secret.
    """
    path = Path(env_file)
    if not path.is_file():
        raise ZenodoError("Zenodo env file does not exist")
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ZenodoError("Zenodo env file could not be read") from exc
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw_value = line.partition("=")
        if separator and key.strip() in names:
            values[key.strip()] = _parse_env_value(raw_value)
    for name in names:
        if values.get(name):
            return values[name], name
    raise ZenodoError(f"Zenodo env file has none of: {', '.join(names)}")


def token_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    names: tuple[str, ...] = DEFAULT_TOKEN_ENV_NAMES,
) -> tuple[str, str]:
    """Read the first configured Zenodo token from process environment."""
    source = os.environ if environ is None else environ
    for name in names:
        value = source.get(name, "").strip()
        if value:
            return value, name
    raise ZenodoError(f"environment has none of: {', '.join(names)}")


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


def _comparable_metadata(payload: Mapping[str, Any]) -> ZenodoMetadataSnapshot:
    """Canonicalize caller-owned metadata, excluding explicit server fields."""
    return ZenodoMetadataSnapshot.from_mapping(
        {
            key: value
            for key, value in payload.items()
            if key not in _SERVER_OWNED_METADATA_FIELDS
        }
    )


def _require_metadata_match(
    expected: Mapping[str, Any],
    actual: ZenodoMetadataSnapshot,
    *,
    operation: str,
) -> None:
    """Fail when Zenodo did not retain the complete requested metadata."""
    expected_snapshot = _comparable_metadata(expected)
    actual_snapshot = _comparable_metadata(actual.as_dict())
    if expected_snapshot.canonical_json != actual_snapshot.canonical_json:
        raise ZenodoError(
            f"Zenodo {operation} metadata does not match the canonical request"
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
) -> _ZenodoLinkedVersionShape:
    """Prove that *draft* is one of the two safe linked-version shapes."""
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
        return "legacy_inherited"

    current_shape_error: ZenodoError | None = None
    try:
        _require_current_separate_record_metadata(source, draft)
    except ZenodoError as exc:
        current_shape_error = exc
    else:
        if draft.files:
            raise ZenodoError(
                "Zenodo separate-record linked draft must have an empty file set"
            )
        return "current_separate_record"

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
        created_utc=_optional_utc_timestamp(payload.get("created"), "creation timestamp"),
        metadata=ZenodoMetadataSnapshot.from_mapping(metadata),
        files=files,
    )


def _multipart_body(field_name: str, filename: str, content: bytes) -> tuple[bytes, str]:
    safe_filename = _safe_upload_filename(filename)
    boundary = f"----active-fedference-{uuid.uuid4().hex}"
    marker = boundary.encode("ascii")
    chunks = [
        b"--" + marker + b"\r\n",
        f'Content-Disposition: form-data; name="name"\r\n\r\n{safe_filename}\r\n'.encode("utf-8"),
        b"--" + marker + b"\r\n",
        (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{safe_filename}"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode("utf-8"),
        content,
        b"\r\n--" + marker + b"--\r\n",
    ]
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class ZenodoClient:
    """Small standard-library Zenodo REST client for release boundaries."""

    def __init__(
        self,
        token: str,
        *,
        api_base: str = DEFAULT_ZENODO_API,
        timeout: float = 60.0,
    ) -> None:
        if not token or not token.strip():
            raise ZenodoError("Zenodo token must be non-empty")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ZenodoError("Zenodo timeout must be a finite positive number")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ZenodoError("Zenodo timeout must be a finite positive number")
        self._token = token.strip()
        self._api_base = _api_base(api_base)
        self._timeout = float(timeout)
        self._opener = build_opener(_RejectRedirects())
        self._metadata_edit_identities: dict[
            int,
            tuple[int | None, int | None, str | None, str],
        ] = {}

    def _editable_deposition(self, deposition_id: int) -> ZenodoDeposition:
        """Return a draft and fail closed for irreversible/published records."""
        deposition = self.get_deposition(deposition_id)
        if deposition.state != "unsubmitted":
            raise ZenodoError(
                f"Zenodo deposition {deposition.id} is {deposition.state!r}; "
                "only an unsubmitted draft is editable"
            )
        return deposition

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: object | None = None,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> object:
        if payload is not None and body is not None:
            raise ZenodoError("request cannot contain both JSON and raw bodies")
        request_body = body
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        if payload is not None:
            request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif content_type is not None:
            headers["Content-Type"] = content_type
        request = Request(
            f"{self._api_base}/{path.lstrip('/')}",
            data=request_body,
            headers=headers,
            method=method.upper(),
        )
        sanitized_http_error: _ZenodoHTTPError | None = None
        try:
            with self._opener.open(request, timeout=self._timeout) as response:  # noqa: S310
                raw = response.read()
        except HTTPError as exc:
            error_bytes = exc.read()
            decoded_error = error_bytes.decode("utf-8", errors="replace")
            detail = decoded_error.replace(
                self._token,
                "<redacted>",
            )
            exact_already_exists = False
            try:
                parsed_error = json.loads(decoded_error)
            except json.JSONDecodeError:
                pass
            else:
                exact_already_exists = (
                    exc.code == 400
                    and isinstance(parsed_error, Mapping)
                    and dict(parsed_error) == _ALREADY_EXISTS_RESPONSE
                )
            sanitized_http_error = _ZenodoHTTPError(
                f"Zenodo HTTP {exc.code}: {detail[:500]}",
                status=exc.code,
                exact_already_exists=exact_already_exists,
            )
        except URLError as exc:
            reason = str(exc.reason).replace(self._token, "<redacted>")
            raise ZenodoError(f"Zenodo request failed: {reason}") from exc
        if sanitized_http_error is not None:
            # Raise only after leaving the HTTPError handler.  Raising inside
            # that block—even with ``from None``—would retain the original
            # response object through ``__context__`` and could preserve a
            # credential echoed by an untrusted server.
            raise sanitized_http_error
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ZenodoError("Zenodo returned non-JSON content") from exc

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

    def _latest_draft_id(self, latest_draft: str) -> int:
        """Extract an id only from this client's exact API origin and path."""
        try:
            parsed = urlsplit(latest_draft)
            api = urlsplit(self._api_base)
            parsed_port = parsed.port
            api_port = api.port
        except ValueError as exc:
            raise ZenodoError("Zenodo latest_draft link is invalid") from exc

        parsed_scheme = parsed.scheme.casefold()
        api_scheme = api.scheme.casefold()
        if (
            parsed_scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.hostname is None
            or api.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ZenodoError("Zenodo latest_draft link is invalid")

        def effective_port(scheme: str, port: int | None) -> int:
            return port if port is not None else (443 if scheme == "https" else 80)

        if (
            parsed_scheme != api_scheme
            or parsed.hostname.casefold() != api.hostname.casefold()
            or effective_port(parsed_scheme, parsed_port)
            != effective_port(api_scheme, api_port)
        ):
            raise ZenodoError(
                "Zenodo latest_draft link is outside the configured API origin"
            )

        api_path = api.path.rstrip("/")
        expected_prefix = f"{api_path}/deposit/depositions/"
        match = re.fullmatch(
            rf"{re.escape(expected_prefix)}([1-9]\d*)/?",
            parsed.path,
        )
        if match is None:
            raise ZenodoError(
                "Zenodo latest_draft link is outside the configured API path"
            )
        return _integer_id(int(match.group(1)))

    def get_deposition(self, deposition_id: int) -> ZenodoDeposition:
        """Retrieve one deposition and normalize its public fields."""
        deposition, _payload = self._get_deposition_response(deposition_id)
        return deposition

    @staticmethod
    def _validate_linked_draft_object(
        source: ZenodoDeposition,
        draft: ZenodoDeposition,
    ) -> ZenodoDeposition:
        """Apply the complete linked-draft contract to one full object."""
        if draft.state != "unsubmitted" or draft.reserved_doi is None:
            raise ZenodoError(
                "Zenodo latest_draft is not an unsubmitted draft with a reserved DOI"
            )
        linked_version_shape = _require_linked_version(source, draft)
        return replace(draft, linked_version_shape=linked_version_shape)

    def _validated_linked_draft_id(
        self,
        source: ZenodoDeposition,
        draft_id: int,
    ) -> ZenodoDeposition:
        """Fetch and fully validate one distinct draft deposition id."""
        return self._validate_linked_draft_object(
            source,
            self.get_deposition(draft_id),
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

        candidates: list[ZenodoDeposition] = []
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
            candidates.append(deposition)

        if len(candidates) != 1:
            raise ZenodoError(
                "Zenodo draft listing did not return exactly one distinct "
                "concept-linked draft"
            )
        listed = self._validate_linked_draft_object(source, candidates[0])
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

    def reserve_doi(self, metadata: Mapping[str, Any]) -> ZenodoDeposition:
        """Create an unsubmitted draft and reserve a DOI without publishing."""
        payload_metadata = dict(metadata)
        payload_metadata.pop("doi", None)
        payload_metadata.setdefault("access_right", "open")
        payload_metadata["prereserve_doi"] = True
        payload = self._request("POST", "deposit/depositions", payload={"metadata": payload_metadata})
        return _deposition(payload)

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

    def delete_file(self, deposition_id: int, file_id: str) -> None:
        """Delete one file from an editable deposition."""
        self._editable_deposition(deposition_id)
        if not isinstance(file_id, str) or not file_id.strip() or "/" in file_id:
            raise ZenodoError("Zenodo file id is unsafe")
        self._request(
            "DELETE",
            f"deposit/depositions/{_integer_id(deposition_id)}/files/{quote(file_id, safe='')}",
        )

    def upload_pdf(
        self,
        deposition_id: int,
        pdf_path: str | Path,
        *,
        replace_existing: bool = False,
    ) -> ZenodoFile:
        """Upload one PDF, returning the server checksum and file identity."""
        path = _pdf_path(pdf_path)
        content = path.read_bytes()
        expected_md5 = hashlib.md5(content).hexdigest()  # noqa: S324 - Zenodo's file API uses MD5
        deposition = self._editable_deposition(deposition_id)
        existing = [file for file in deposition.files if file.filename == path.name]
        if existing:
            if len(existing) == 1 and existing[0].filesize == len(content) and existing[0].checksum in {
                expected_md5,
                f"md5:{expected_md5}",
            }:
                return existing[0]
            if not replace_existing:
                raise ZenodoError(
                    f"Zenodo deposition already contains a different file named {path.name!r}; "
                    "inspect or replace the draft explicitly"
                )
            for record in existing:
                self.delete_file(deposition.id, record.id)
        body, content_type = _multipart_body("file", path.name, content)
        payload = self._request(
            "POST",
            f"deposit/depositions/{_integer_id(deposition_id)}/files",
            body=body,
            content_type=content_type,
        )
        return _file_record(payload)

    @staticmethod
    def _verify_pdf_record(
        deposition: ZenodoDeposition,
        pdf_path: str | Path,
        *,
        remote_filename: str | None,
        require_exact_file_set: bool,
    ) -> ZenodoFile:
        """Verify local bytes against one already-fetched deposition."""
        path = _pdf_path(pdf_path)
        expected_filename = _safe_upload_filename(remote_filename or path.name)
        if Path(expected_filename).suffix.casefold() != ".pdf":
            raise ZenodoError("verified Zenodo filename must use a .pdf extension")
        expected_md5 = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - Zenodo's file API uses MD5
        matches = [file for file in deposition.files if file.filename == expected_filename]
        if len(matches) != 1:
            raise ZenodoError(f"Zenodo deposition has {len(matches)} files named {expected_filename!r}")
        if require_exact_file_set and len(deposition.files) != 1:
            raise ZenodoError(
                "Zenodo deposition must contain exactly the verified PDF before publication"
            )
        record = matches[0]
        checksums = {expected_md5, f"md5:{expected_md5}"}
        if record.filesize != path.stat().st_size or record.checksum not in checksums:
            raise ZenodoError(f"Zenodo PDF checksum or size mismatch for {expected_filename}")
        return record

    def verify_pdf(
        self,
        deposition_id: int,
        pdf_path: str | Path,
        *,
        remote_filename: str | None = None,
        require_exact_file_set: bool = False,
    ) -> ZenodoFile:
        """Verify PDF bytes, optionally against a distinct server-side filename.

        A repository may use an informative copy name while Zenodo retains the
        canonical manuscript filename.  The alternate name is explicit so a
        checksum cannot silently match an unrelated deposition file.
        """
        path = _pdf_path(pdf_path)
        deposition = self.get_deposition(deposition_id)
        return self._verify_pdf_record(
            deposition,
            path,
            remote_filename=remote_filename,
            require_exact_file_set=require_exact_file_set,
        )

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


__all__ = [
    "DEFAULT_TOKEN_ENV_NAMES",
    "DEFAULT_ZENODO_API",
    "ZenodoClient",
    "ZenodoDeposition",
    "ZenodoError",
    "ZenodoFile",
    "ZenodoMetadataSnapshot",
    "token_from_environment",
    "token_from_env_file",
]
