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
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
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
        file_id = str(payload["id"])
        filename = str(payload["filename"])
        filesize = int(payload["filesize"])
        checksum = str(payload["checksum"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ZenodoError("Zenodo returned a malformed file record") from exc
    if not file_id or not filename or filesize < 0 or not checksum:
        raise ZenodoError("Zenodo returned an invalid file record")
    return ZenodoFile(file_id, filename, filesize, checksum)


def _deposition(payload: object) -> ZenodoDeposition:
    if not isinstance(payload, Mapping):
        raise ZenodoError("Zenodo returned a malformed deposition")
    try:
        deposition_id = _integer_id(int(payload["id"]))
        state = str(payload["state"])
        metadata = payload.get("metadata", {})
        links = payload.get("links", {})
        files_payload = payload.get("files", [])
    except (KeyError, TypeError, ValueError) as exc:
        raise ZenodoError("Zenodo returned a malformed deposition") from exc
    if not isinstance(metadata, Mapping) or not isinstance(links, Mapping):
        raise ZenodoError("Zenodo returned malformed deposition metadata or links")
    reserved = metadata.get("prereserve_doi")
    reserved_doi = str(reserved.get("doi")) if isinstance(reserved, Mapping) and reserved.get("doi") else None
    files = tuple(_file_record(item) for item in files_payload) if isinstance(files_payload, list) else ()
    return ZenodoDeposition(
        id=deposition_id,
        state=state,
        doi=str(payload["doi"]) if payload.get("doi") else None,
        reserved_doi=reserved_doi,
        html_url=str(links["html"]) if links.get("html") else None,
        self_url=str(links["self"]) if links.get("self") else None,
        bucket_url=str(links["bucket"]) if links.get("bucket") else None,
        publish_url=str(links["publish"]) if links.get("publish") else None,
        files=files,
    )


def _multipart_body(field_name: str, filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = f"----active-fedference-{uuid.uuid4().hex}"
    marker = boundary.encode("ascii")
    chunks = [
        b"--" + marker + b"\r\n",
        f'Content-Disposition: form-data; name="name"\r\n\r\n{filename}\r\n'.encode("utf-8"),
        b"--" + marker + b"\r\n",
        (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
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
        self._token = token.strip()
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout

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

    def update_metadata(self, deposition_id: int, metadata: Mapping[str, Any]) -> ZenodoDeposition:
        """Replace editable deposition metadata without publishing it."""
        payload = self._request(
            "PUT",
            f"deposit/depositions/{_integer_id(deposition_id)}",
            payload={"metadata": dict(metadata)},
        )
        return _deposition(payload)

    def upload_pdf(self, deposition_id: int, pdf_path: str | Path) -> ZenodoFile:
        """Upload one PDF, returning the server checksum and file identity."""
        path = Path(pdf_path)
        if not path.is_file() or path.suffix.casefold() != ".pdf":
            raise ZenodoError(f"PDF file does not exist or is not a PDF: {path}")
        content = path.read_bytes()
        body, content_type = _multipart_body("file", path.name, content)
        payload = self._request(
            "POST",
            f"deposit/depositions/{_integer_id(deposition_id)}/files",
            body=body,
            content_type=content_type,
        )
        return _file_record(payload)

    def verify_pdf(self, deposition_id: int, pdf_path: str | Path) -> ZenodoFile:
        """Verify the uploaded PDF filename, size, and MD5 checksum."""
        path = Path(pdf_path)
        expected_md5 = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - Zenodo's file API uses MD5
        deposition = self.get_deposition(deposition_id)
        matches = [file for file in deposition.files if file.filename == path.name]
        if len(matches) != 1:
            raise ZenodoError(f"Zenodo deposition has {len(matches)} files named {path.name!r}")
        record = matches[0]
        checksums = {expected_md5, f"md5:{expected_md5}"}
        if record.filesize != path.stat().st_size or record.checksum not in checksums:
            raise ZenodoError(f"Zenodo PDF checksum or size mismatch for {path.name}")
        return record

    def publish(self, deposition_id: int) -> ZenodoDeposition:
        """Publish a deposition explicitly; callers must gate this action."""
        payload = self._request("POST", f"deposit/depositions/{_integer_id(deposition_id)}/actions/publish")
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
