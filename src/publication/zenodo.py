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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

DEFAULT_ZENODO_API = "https://zenodo.org/api"
DEFAULT_TOKEN_ENV_NAMES: tuple[str, ...] = (
    "ZENODO_PROD_TOKEN",
    "ZENODO_TOKEN",
    "ZENODO_API_TOKEN",
)


class ZenodoError(RuntimeError):
    """Raised when Zenodo rejects or cannot complete a request."""


@dataclass(frozen=True)
class ZenodoFile:
    """The server-side identity and checksum of one deposition file."""

    id: str
    filename: str
    filesize: int
    checksum: str


@dataclass(frozen=True)
class ZenodoDeposition:
    """Stable subset of a Zenodo deposition response."""

    id: int
    state: str
    doi: str | None
    reserved_doi: str | None
    html_url: str | None
    self_url: str | None
    bucket_url: str | None
    publish_url: str | None
    files: tuple[ZenodoFile, ...]


def _optional_text(value: object, field: str) -> str | None:
    """Normalize an optional server text field and reject typed corruption."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ZenodoError(f"Zenodo returned an invalid {field}")
    return value


def _api_base(value: str) -> str:
    """Validate an API endpoint before a bearer token can be sent to it."""
    if not isinstance(value, str) or not value.strip():
        raise ZenodoError("Zenodo API base must be a non-empty URL")
    normalized = value.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname
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
        raise ZenodoError(f"Zenodo env file does not exist: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
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


def _integer_id(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ZenodoError(f"invalid Zenodo deposition id: {value!r}")
    return value


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
    reserved = metadata.get("prereserve_doi")
    if reserved is not None and not isinstance(reserved, Mapping):
        raise ZenodoError("Zenodo returned malformed DOI reservation metadata")
    reserved_doi = (
        _optional_text(reserved.get("doi"), "reserved DOI")
        if isinstance(reserved, Mapping)
        else None
    )
    if not isinstance(files_payload, list):
        raise ZenodoError("Zenodo returned malformed deposition files")
    files = tuple(_file_record(item) for item in files_payload)
    return ZenodoDeposition(
        id=deposition_id,
        state=state,
        doi=_optional_text(payload.get("doi"), "DOI"),
        reserved_doi=reserved_doi,
        html_url=_optional_text(links.get("html"), "HTML link"),
        self_url=_optional_text(links.get("self"), "self link"),
        bucket_url=_optional_text(links.get("bucket"), "bucket link"),
        publish_url=_optional_text(links.get("publish"), "publish link"),
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
        try:
            with urlopen(request, timeout=self._timeout) as response:  # noqa: S310 - URL is caller-configured
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ZenodoError(f"Zenodo HTTP {exc.code}: {detail[:500]}") from exc
        except URLError as exc:
            raise ZenodoError(f"Zenodo request failed: {exc.reason}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ZenodoError("Zenodo returned non-JSON content") from exc

    def get_deposition(self, deposition_id: int) -> ZenodoDeposition:
        """Retrieve one deposition and normalize its public fields."""
        return _deposition(self._request("GET", f"deposit/depositions/{_integer_id(deposition_id)}"))

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

        Zenodo returns the original record from the ``newversion`` action and
        exposes the draft only through ``links.latest_draft``.  Resolve that
        link explicitly so the caller cannot accidentally edit the immutable
        published record or create an unrelated deposition.
        """
        source = self.get_deposition(deposition_id)
        if source.state != "done":
            raise ZenodoError(
                f"Zenodo deposition {source.id} is {source.state!r}; "
                "new versions require the latest published record"
            )
        payload = self._request(
            "POST",
            f"deposit/depositions/{_integer_id(deposition_id)}/actions/newversion",
        )
        if not isinstance(payload, Mapping):
            raise ZenodoError("Zenodo new-version response is malformed")
        links = payload.get("links")
        latest_draft = links.get("latest_draft") if isinstance(links, Mapping) else None
        if not isinstance(latest_draft, str) or not latest_draft.strip():
            raise ZenodoError("Zenodo new-version response has no latest_draft link")
        draft_path = urlsplit(latest_draft).path.rstrip("/")
        draft_id_text = draft_path.rsplit("/", 1)[-1]
        try:
            draft_id = int(draft_id_text)
        except ValueError as exc:
            raise ZenodoError("Zenodo latest_draft link has an invalid deposition id") from exc
        return self.get_deposition(_integer_id(draft_id))

    def update_metadata(self, deposition_id: int, metadata: Mapping[str, Any]) -> ZenodoDeposition:
        """Replace editable deposition metadata without publishing it."""
        self._editable_deposition(deposition_id)
        payload_metadata = dict(metadata)
        payload_metadata.pop("doi", None)
        payload = self._request(
            "PUT",
            f"deposit/depositions/{_integer_id(deposition_id)}",
            payload={"metadata": payload_metadata},
        )
        return _deposition(payload)

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
        if source.state not in {"done", "inprogress", "unsubmitted"}:
            raise ZenodoError(
                f"Zenodo deposition {source.id} is {source.state!r}; "
                "published metadata edits require the latest published record"
            )
        if source.state == "done":
            payload = self._request(
                "POST",
                f"deposit/depositions/{_integer_id(deposition_id)}/actions/edit",
            )
            editable = _deposition(payload)
            if editable.id != source.id:
                raise ZenodoError("Zenodo metadata-edit response changed the deposition id")
            if editable.state == "done":
                raise ZenodoError("Zenodo metadata-edit action did not open an editable draft")
        else:
            # Re-running the explicit operation against its already-open edit
            # draft must update the draft rather than attempt a second action.
            editable = source
        payload_metadata = dict(metadata)
        # The DOI remains owned by the published record and must not be sent as
        # a replacement metadata field during an edit.
        payload_metadata.pop("doi", None)
        payload_metadata.setdefault("access_right", "open")
        updated = self._request(
            "PUT",
            f"deposit/depositions/{_integer_id(deposition_id)}",
            payload={"metadata": payload_metadata},
        )
        return _deposition(updated)

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

    def verify_pdf(
        self,
        deposition_id: int,
        pdf_path: str | Path,
        *,
        remote_filename: str | None = None,
    ) -> ZenodoFile:
        """Verify PDF bytes, optionally against a distinct server-side filename.

        A repository may use an informative copy name while Zenodo retains the
        canonical manuscript filename.  The alternate name is explicit so a
        checksum cannot silently match an unrelated deposition file.
        """
        path = _pdf_path(pdf_path)
        expected_filename = _safe_upload_filename(remote_filename or path.name)
        expected_md5 = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - Zenodo's file API uses MD5
        deposition = self.get_deposition(deposition_id)
        matches = [file for file in deposition.files if file.filename == expected_filename]
        if len(matches) != 1:
            raise ZenodoError(f"Zenodo deposition has {len(matches)} files named {expected_filename!r}")
        record = matches[0]
        checksums = {expected_md5, f"md5:{expected_md5}"}
        if record.filesize != path.stat().st_size or record.checksum not in checksums:
            raise ZenodoError(f"Zenodo PDF checksum or size mismatch for {expected_filename}")
        return record

    def publish(self, deposition_id: int) -> ZenodoDeposition:
        """Publish a deposition explicitly; callers must gate this action."""
        self._editable_deposition(deposition_id)
        payload = self._request("POST", f"deposit/depositions/{_integer_id(deposition_id)}/actions/publish")
        return _deposition(payload)

    def publish_metadata_edit(self, deposition_id: int) -> ZenodoDeposition:
        """Publish an already-updated metadata-only edit draft."""
        deposition = self.get_deposition(deposition_id)
        if deposition.state == "done":
            raise ZenodoError(
                f"Zenodo deposition {deposition.id} is already published; "
                "open a metadata edit before publishing"
            )
        payload = self._request(
            "POST",
            f"deposit/depositions/{_integer_id(deposition_id)}/actions/publish",
        )
        return _deposition(payload)


__all__ = [
    "DEFAULT_TOKEN_ENV_NAMES",
    "DEFAULT_ZENODO_API",
    "ZenodoClient",
    "ZenodoDeposition",
    "ZenodoError",
    "ZenodoFile",
    "token_from_environment",
    "token_from_env_file",
]
