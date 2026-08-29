from __future__ import annotations

import hashlib
import io
import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from publication.zenodo import (
    _JSON_RESPONSE_BODY_LIMIT,
    ZenodoClient,
    ZenodoDeposition,
    ZenodoError,
    ZenodoFile,
    ZenodoMetadataSnapshot,
    _deposition,
    _json_response_without_response_body,
    _multipart_body,
    token_from_env_file,
    token_from_environment,
)
from scripts.zenodo_release import _summary
from scripts.zenodo_release import main as zenodo_main


class _ZenodoHandler(BaseHTTPRequestHandler):
    expected_content = b""
    uploaded = False
    extra_file = False
    add_extra_after_publish = False
    ignore_metadata_update = False
    metadata_payload: dict[str, Any] | None = None
    state = "unsubmitted"
    publish_calls = 0
    authorization_headers: list[str] = []
    put_payloads: list[dict[str, Any]] = []

    def log_message(self, _format: str, *_args: object) -> None:
        return

    @classmethod
    def _deposition(cls, *, published: bool | None = None) -> dict[str, Any]:
        content = cls.expected_content
        files = []
        if cls.uploaded:
            files.append(
                {
                    "id": "file-1",
                    "filename": "paper.pdf",
                    "filesize": len(content),
                    "checksum": hashlib.md5(content).hexdigest(),  # noqa: S324 - Zenodo contract
                }
            )
        if cls.extra_file:
            files.append(
                {
                    "id": "unexpected-file",
                    "filename": "unexpected.txt",
                    "filesize": 7,
                    "checksum": "md5:00000000000000000000000000000000",
                }
            )
        metadata = (
            dict(cls.metadata_payload)
            if cls.metadata_payload is not None
            else {
                "title": "Release draft",
                "description": "Complete paper abstract.",
                "version": "1.1.0",
                "publication_date": "2026-08-27",
                "upload_type": "software",
                "access_right": "open",
                "license": "MIT",
                "creators": [
                    {
                        "name": "Friedman, Daniel Ari",
                        "orcid": "0000-0001-6232-9096",
                        "affiliation": "Active Inference Institute",
                    }
                ],
                "related_identifiers": [
                    {
                        "identifier": "https://github.com/example/repository",
                        "relation": "isSupplementTo",
                    }
                ],
            }
        )
        metadata["prereserve_doi"] = {"doi": "10.5281/zenodo.7", "recid": 7}
        state = "done" if published is True else cls.state
        return {
            "id": 7,
            "record_id": 7,
            "conceptrecid": "6",
            "conceptdoi": "10.5281/zenodo.6",
            "state": state,
            # Ordinary deposition operations do not depend on creation time.
            # This deliberately non-profile spelling proves that only linked
            # separate-record validation applies the strict RFC 3339 parser.
            "created": "2026-08-28 01:30:00Z",
            **({"doi": "10.5281/zenodo.7"} if state == "done" else {}),
            "metadata": metadata,
            "links": {
                "html": "http://example.test/deposit/7",
                "self": "http://example.test/api/deposit/depositions/7",
                "bucket": "http://example.test/api/files/bucket-7",
                "publish": "http://example.test/api/deposit/depositions/7/actions/publish",
            },
            "files": files,
        }

    def _write_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        size = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(size)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        self.authorization_headers.append(self.headers.get("Authorization", ""))
        if self.path == "/api/deposit/depositions/7":
            self._write_json(self._deposition())
            return
        self._write_json({"error": "not found"}, status=404)

    def do_PUT(self) -> None:  # noqa: N802 - stdlib handler API
        self.authorization_headers.append(self.headers.get("Authorization", ""))
        payload = json.loads(self._read_body().decode("utf-8"))
        self.put_payloads.append(payload)
        if not type(self).ignore_metadata_update:
            type(self).metadata_payload = dict(payload["metadata"])
        self._write_json(self._deposition())

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        self.authorization_headers.append(self.headers.get("Authorization", ""))
        if self.path == "/api/deposit/depositions":
            payload = json.loads(self._read_body().decode("utf-8"))
            assert payload["metadata"]["prereserve_doi"] is True
            self._write_json(self._deposition())
            return
        if self.path.endswith("/files"):
            body = self._read_body()
            assert b"Content-Type: application/pdf" in body
            assert re.search(rb'filename="paper\.pdf"', body)
            type(self).uploaded = True
            self._write_json(type(self)._deposition()["files"][0])
            return
        if self.path.endswith("/actions/publish"):
            type(self).publish_calls += 1
            type(self).state = "done"
            response = self._deposition(published=True)
            self._write_json(response)
            if type(self).add_extra_after_publish:
                type(self).extra_file = True
            return
        self._write_json({"error": "not found"}, status=404)


class _NewVersionHandler(BaseHTTPRequestHandler):
    """A real loopback HTTP boundary for the Zenodo new-version action."""

    api_base = ""
    authorization_headers: list[str] = []
    new_version_calls = 0
    draft_state = "unsubmitted"
    include_reserved_doi = True
    draft_concept_record_id = 6
    draft_concept_doi = "10.5281/zenodo.6"
    reserved_record_id = 8
    reserved_doi = "10.5281/zenodo.8"
    draft_version: str | None = "1.0.4"
    draft_publication_date: str | None = "2026-08-20"
    draft_created: object | None = "2026-08-27T18:30:00-07:00"
    draft_extra_metadata: dict[str, Any] = {}
    source_files: list[dict[str, Any]] = []
    draft_files: list[dict[str, Any]] = []
    draft_exists = False
    expose_source_latest_draft = True
    source_latest_draft: str | None = None
    post_mode = "success"
    listing_mode = "single"
    listing_paths: list[str] = []

    @classmethod
    def reset(cls) -> None:
        """Restore one exact legacy-inheritance response for each test."""
        cls.api_base = ""
        cls.authorization_headers = []
        cls.new_version_calls = 0
        cls.draft_state = "unsubmitted"
        cls.include_reserved_doi = True
        cls.draft_concept_record_id = 6
        cls.draft_concept_doi = "10.5281/zenodo.6"
        cls.reserved_record_id = 8
        cls.reserved_doi = "10.5281/zenodo.8"
        cls.draft_version = "1.0.4"
        cls.draft_publication_date = "2026-08-20"
        cls.draft_created = "2026-08-27T18:30:00-07:00"
        cls.draft_extra_metadata = {}
        cls.draft_exists = False
        cls.expose_source_latest_draft = True
        cls.source_latest_draft = None
        cls.post_mode = "success"
        cls.listing_mode = "single"
        cls.listing_paths = []
        cls.source_files = [
            {
                "id": "published-file",
                "filename": "paper.pdf",
                "filesize": 1,
                "checksum": "md5:old",
            }
        ]
        cls.draft_files = [
            {
                "id": "inherited",
                "filename": "paper.pdf",
                "filesize": 1,
                "checksum": "md5:old",
            }
        ]

    @classmethod
    def _draft_link(cls, deposition_id: int) -> str:
        """Return one absolute loopback link on the configured fixture API."""
        assert cls.api_base
        return f"{cls.api_base}/deposit/depositions/{deposition_id}"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _write_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._write_raw_json(body, status=status)

    def _write_raw_json(self, body: bytes, status: int = 200) -> None:
        """Write exact JSON bytes so malformed-service shapes need no encoder."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_truncated_json(self, body: bytes, status: int = 400) -> None:
        """Advertise more bytes than sent to exercise a real failed body read."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body) + 100))
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    @classmethod
    def _published_source(cls) -> dict[str, Any]:
        links = {
            "html": "http://example.test/records/7",
            "self": "http://example.test/api/deposit/depositions/7",
        }
        if cls.source_latest_draft is not None:
            links["latest_draft"] = cls.source_latest_draft
        elif cls.draft_exists and cls.expose_source_latest_draft:
            links["latest_draft"] = cls._draft_link(8)
        return {
            "id": 7,
            "record_id": 7,
            "conceptrecid": "6",
            "conceptdoi": "10.5281/zenodo.6",
            "state": "done",
            "doi": "10.5281/zenodo.7",
            "metadata": {
                "title": "Published source",
                "description": "Inherited abstract.",
                "version": "1.0.4",
                "publication_date": "2026-08-20",
                "upload_type": "software",
            },
            "links": links,
            "files": [dict(file) for file in cls.source_files],
        }

    @classmethod
    def _draft(cls) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "title": "Published source",
            "description": "Inherited abstract.",
            "upload_type": "software",
        }
        if cls.draft_version is not None:
            metadata["version"] = cls.draft_version
        if cls.draft_publication_date is not None:
            metadata["publication_date"] = cls.draft_publication_date
        metadata.update(cls.draft_extra_metadata)
        if cls.include_reserved_doi:
            metadata["prereserve_doi"] = {
                "doi": cls.reserved_doi,
                "recid": cls.reserved_record_id,
            }
        draft = {
            "id": 8,
            "record_id": 8,
            "conceptrecid": str(cls.draft_concept_record_id),
            "conceptdoi": cls.draft_concept_doi,
            "state": cls.draft_state,
            "metadata": metadata,
            "links": {
                "html": "http://example.test/records/8",
                "self": "http://example.test/api/deposit/depositions/8",
            },
            "files": [dict(file) for file in cls.draft_files],
        }
        if cls.draft_created is not None:
            draft["created"] = cls.draft_created
        return draft

    @classmethod
    def _other_draft(cls, *, concept_record_id: int = 6) -> dict[str, Any]:
        draft = cls._draft()
        draft["id"] = 9
        draft["record_id"] = 9
        draft["conceptrecid"] = str(concept_record_id)
        draft["conceptdoi"] = f"10.5281/zenodo.{concept_record_id}"
        metadata = dict(draft["metadata"])
        metadata["prereserve_doi"] = {"doi": "10.5281/zenodo.9", "recid": 9}
        draft["metadata"] = metadata
        return draft

    @classmethod
    def _listing(cls) -> object:
        if cls.listing_mode == "malformed":
            return {"not": "an array"}
        if cls.listing_mode == "partial":
            return [{"id": 8, "state": "unsubmitted"}]
        if cls.listing_mode == "truncated":
            return [cls._published_source() for _index in range(100)]
        if cls.listing_mode == "zero":
            return [cls._published_source()]
        if cls.listing_mode == "multiple":
            return [cls._published_source(), cls._draft(), cls._other_draft()]
        if cls.listing_mode == "wrong_concept":
            return [cls._published_source(), cls._other_draft(concept_record_id=99)]
        return [cls._published_source(), cls._draft()]

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        type(self).authorization_headers.append(self.headers.get("Authorization", ""))
        if self.path.startswith("/api/deposit/depositions?"):
            type(self).listing_paths.append(self.path)
            self._write_json(type(self)._listing())
            return
        if self.path == "/api/deposit/depositions/7":
            self._write_json(self._published_source())
            return
        if self.path == "/api/deposit/depositions/8":
            self._write_json(type(self)._draft())
            return
        self._write_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        type(self).authorization_headers.append(self.headers.get("Authorization", ""))
        if self.path == "/api/deposit/depositions/7/actions/newversion":
            type(self).new_version_calls += 1
            if type(self).post_mode == "unrelated_error":
                self._write_json(
                    {"status": 400, "message": "The record cannot be versioned."},
                    status=400,
                )
                return
            if type(self).post_mode == "already_exists_extra":
                self._write_json(
                    {
                        "status": 400,
                        "message": "A draft already exists.",
                        "unexpected": True,
                    },
                    status=400,
                )
                return
            if type(self).post_mode == "already_exists_wrong_status":
                self._write_json(
                    {"status": 409, "message": "A draft already exists."},
                    status=409,
                )
                return
            if type(self).post_mode == "echo_token":
                authorization = self.headers.get("Authorization", "")
                self._write_json(
                    {"status": 400, "message": authorization.removeprefix("Bearer ")},
                    status=400,
                )
                return
            if type(self).post_mode == "deeply_nested_echo":
                authorization = self.headers.get("Authorization", "")
                token_json = json.dumps(
                    authorization.removeprefix("Bearer ")
                ).encode("utf-8")
                body = (
                    b'{"status":400,"message":'
                    + (b"[" * 10_000)
                    + token_json
                    + (b"]" * 10_000)
                    + b"}"
                )
                self._write_raw_json(body, status=400)
                return
            if type(self).post_mode == "truncated_echo":
                authorization = self.headers.get("Authorization", "")
                token_json = json.dumps(
                    authorization.removeprefix("Bearer ")
                ).encode("utf-8")
                self._write_truncated_json(
                    b'{"status":400,"message":' + token_json,
                )
                return
            if type(self).post_mode == "deeply_nested_success":
                authorization = self.headers.get("Authorization", "")
                token_json = json.dumps(
                    authorization.removeprefix("Bearer ")
                ).encode("utf-8")
                body = (b"[" * 10_000) + token_json + (b"]" * 10_000)
                self._write_raw_json(body, status=201)
                return
            if type(self).post_mode == "truncated_success":
                authorization = self.headers.get("Authorization", "")
                token_json = json.dumps(
                    authorization.removeprefix("Bearer ")
                ).encode("utf-8")
                self._write_truncated_json(
                    b'{"message":' + token_json,
                    status=201,
                )
                return
            if type(self).post_mode == "invalid_utf8_success":
                self._write_raw_json(b'{"message":"test-token-\xff"}', status=201)
                return
            if type(self).post_mode == "semantic_echo_success":
                authorization = self.headers.get("Authorization", "")
                token = authorization.removeprefix("Bearer ")
                self._write_json(
                    {
                        f"unexpected-{token}": {
                            "nested": [f"prefix-{token}-suffix"],
                        },
                    },
                    status=201,
                )
                return
            if type(self).post_mode == "already_exists" or type(self).draft_exists:
                type(self).draft_exists = True
                self._write_json(
                    {"status": 400, "message": "A draft already exists."},
                    status=400,
                )
                return
            type(self).draft_exists = True
            self._write_json(
                {"links": {"latest_draft": type(self)._draft_link(8)}}
            )
            return
        self._write_json({"error": "not found"}, status=404)


@pytest.fixture(autouse=True)
def _reset_new_version_handler() -> None:
    """Keep mutable loopback service shapes independent across tests."""
    _NewVersionHandler.reset()


@pytest.fixture
def new_version_client() -> Iterator[ZenodoClient]:
    """Serve the mutable new-version fixture over the real HTTP adapter."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NewVersionHandler)
    _NewVersionHandler.api_base = f"http://127.0.0.1:{server.server_port}/api"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ZenodoClient(
            "test-token",
            api_base=_NewVersionHandler.api_base,
            timeout=5.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _PublishedMetadataEditHandler(BaseHTTPRequestHandler):
    """Loopback boundary for Zenodo's metadata-only published-record edit."""

    authorization_headers: list[str] = []
    edit_calls = 0
    publish_calls = 0
    put_payloads: list[dict[str, Any]] = []
    metadata_payload: dict[str, Any] | None = None
    state = "done"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _write_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        size = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(size)

    @classmethod
    def _deposition(cls) -> dict[str, Any]:
        metadata = dict(cls.metadata_payload or {"description": "Original abstract."})
        metadata["prereserve_doi"] = {"doi": "10.5281/zenodo.7", "recid": 7}
        return {
            "id": 7,
            "record_id": 7,
            "conceptrecid": "6",
            "conceptdoi": "10.5281/zenodo.6",
            "state": cls.state,
            "doi": "10.5281/zenodo.7",
            "metadata": metadata,
            "links": {
                "html": "http://example.test/records/7",
                "self": "http://example.test/api/deposit/depositions/7",
            },
            "files": [],
        }

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        type(self).authorization_headers.append(self.headers.get("Authorization", ""))
        if self.path == "/api/deposit/depositions/7":
            self._write_json(type(self)._deposition())
            return
        self._write_json({"error": "not found"}, status=404)

    def do_PUT(self) -> None:  # noqa: N802 - stdlib handler API
        type(self).authorization_headers.append(self.headers.get("Authorization", ""))
        payload = json.loads(self._read_body().decode("utf-8"))
        type(self).put_payloads.append(payload)
        type(self).metadata_payload = dict(payload["metadata"])
        self._write_json(type(self)._deposition())

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        type(self).authorization_headers.append(self.headers.get("Authorization", ""))
        if self.path == "/api/deposit/depositions/7/actions/edit":
            type(self).edit_calls += 1
            type(self).state = "inprogress"
            self._write_json(type(self)._deposition(), status=201)
            return
        if self.path == "/api/deposit/depositions/7/actions/publish":
            type(self).publish_calls += 1
            type(self).state = "done"
            self._write_json(type(self)._deposition(), status=202)
            return
        self._write_json({"error": "not found"}, status=404)


def test_token_sources_and_missing_token(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("export ZENODO_TOKEN='secret-token'\n", encoding="utf-8")
    assert token_from_env_file(env_file) == ("secret-token", "ZENODO_TOKEN")
    assert token_from_environment({"ZENODO_PROD_TOKEN": "prod-token"}) == (
        "prod-token",
        "ZENODO_PROD_TOKEN",
    )
    missing = tmp_path / "sensitive" / "missing.env"
    with pytest.raises(ZenodoError, match="does not exist") as exc_info:
        token_from_env_file(missing)
    assert str(missing) not in str(exc_info.value)
    with pytest.raises(ZenodoError, match="none of"):
        token_from_environment({})


def test_client_rejects_unsafe_endpoint_and_timeout() -> None:
    with pytest.raises(ZenodoError, match="HTTPS"):
        ZenodoClient("test-token", api_base="http://example.test/api")
    for endpoint in (
        "not-a-url",
        "ftp://zenodo.org/api",
        "https://user:password@zenodo.org/api",
        "https://zenodo.org/api?token=leak",
        "https://zenodo.org/api#fragment",
    ):
        with pytest.raises(ZenodoError):
            ZenodoClient("test-token", api_base=endpoint)
    with pytest.raises(ZenodoError, match="non-empty"):
        ZenodoClient("   ")
    with pytest.raises(ZenodoError, match="finite positive"):
        ZenodoClient("test-token", timeout=0)
    with pytest.raises(ZenodoError, match="finite positive"):
        ZenodoClient("test-token", timeout=float("inf"))
    with pytest.raises(ZenodoError, match="finite positive"):
        ZenodoClient("test-token", timeout=True)
    with pytest.raises(ZenodoError, match="finite positive"):
        ZenodoClient("test-token", timeout=float("nan"))


def test_client_never_forwards_bearer_token_across_redirects() -> None:
    captured_headers: list[str | None] = []

    class _SinkHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            captured_headers.append(self.headers.get("Authorization"))
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class _RedirectHandler(BaseHTTPRequestHandler):
        target = ""

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            self.send_response(302)
            self.send_header("Location", type(self).target)
            self.end_headers()

    sink = ThreadingHTTPServer(("127.0.0.1", 0), _SinkHandler)
    redirect = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    _RedirectHandler.target = f"http://127.0.0.1:{sink.server_port}/capture"
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (sink, redirect)
    ]
    for thread in threads:
        thread.start()
    try:
        client = ZenodoClient(
            "sentinel-secret",
            api_base=f"http://127.0.0.1:{redirect.server_port}/api",
            timeout=5.0,
        )
        with pytest.raises(ZenodoError, match="HTTP 302") as exc_info:
            client.get_deposition(7)
        assert "sentinel-secret" not in str(exc_info.value)
        assert captured_headers == []
    finally:
        for server in (redirect, sink):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)


def test_file_and_env_boundaries_reject_unsafe_inputs(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ZENODO_PROD_TOKEN=\nZENODO_TOKEN=\"fallback\"\n", encoding="utf-8")
    assert token_from_env_file(env_file) == ("fallback", "ZENODO_TOKEN")
    pdf = tmp_path / "not-a-pdf.txt"
    pdf.write_text("not a PDF", encoding="utf-8")
    client = ZenodoClient("test-token", api_base="http://127.0.0.1:1/api")
    with pytest.raises(ZenodoError, match="does not exist"):
        client.upload_pdf(7, pdf)
    with pytest.raises(ZenodoError, match="does not exist"):
        client.verify_pdf(7, tmp_path / "missing.pdf")
    for filename in ("../paper.pdf", "paper\n.pdf", "a" * 256 + ".pdf"):
        with pytest.raises(ZenodoError, match="safe"):
            _multipart_body("file", filename, b"%PDF")


def test_response_and_multipart_boundaries_fail_closed() -> None:
    with pytest.raises(ZenodoError, match="malformed deposition"):
        _deposition(None)
    malformed_files = {
        "id": 7,
        "state": "unsubmitted",
        "metadata": {},
        "links": {},
        "files": {},
    }
    with pytest.raises(ZenodoError, match="malformed deposition files"):
        _deposition(malformed_files)
    with pytest.raises(ZenodoError, match="invalid file record"):
        _deposition(
            {
                "id": 7,
                "state": "unsubmitted",
                "metadata": {},
                "links": {},
                "files": [{"id": 1, "filename": "paper.pdf", "filesize": 1, "checksum": "bad"}],
            }
        )
    integral_float = _deposition(
        {
            "id": 7,
            "state": "unsubmitted",
            "metadata": {},
            "links": {},
            "files": [{"id": "file-1", "filename": "paper.pdf", "filesize": 1.0, "checksum": "bad"}],
        }
    )
    assert integral_float.files[0].filesize == 1
    for filesize in (-1.0, 1.25, float("inf")):
        with pytest.raises(ZenodoError, match="invalid file record"):
            _deposition(
                {
                    "id": 7,
                    "state": "unsubmitted",
                    "metadata": {},
                    "links": {},
                    "files": [
                        {
                            "id": "file-1",
                            "filename": "paper.pdf",
                            "filesize": filesize,
                            "checksum": "bad",
                        }
                    ],
                }
            )
    for payload in (
        {"id": True, "state": "unsubmitted"},
        {"id": 7, "state": ""},
        {"id": 7, "record_id": 8, "state": "unsubmitted"},
        {"id": 7, "record_id": "not-an-id", "state": "unsubmitted"},
        {"id": 7, "conceptrecid": False, "state": "unsubmitted"},
        {"id": 7, "conceptdoi": 3, "state": "unsubmitted"},
        {"id": 7, "state": "unsubmitted", "metadata": []},
        {"id": 7, "state": "unsubmitted", "metadata": {"bad": float("nan")}},
        {"id": 7, "state": "unsubmitted", "links": []},
        {
            "id": 7,
            "state": "unsubmitted",
            "metadata": {"prereserve_doi": "not-a-mapping"},
        },
        {
            "id": 7,
            "state": "unsubmitted",
            "metadata": {"prereserve_doi": {"doi": 3}},
        },
        {"id": 7, "state": "unsubmitted", "doi": 3},
    ):
        with pytest.raises(ZenodoError):
            _deposition(payload)
    with pytest.raises(ZenodoError, match="safe"):
        _multipart_body("file", 'unsafe\"name.pdf', b"%PDF")


def test_zenodo_deposition_preserves_the_former_positional_constructor() -> None:
    metadata = ZenodoMetadataSnapshot.from_mapping({"title": "Legacy caller"})
    files = (ZenodoFile("file-1", "paper.pdf", 1, "md5:old"),)

    deposition = ZenodoDeposition(
        7,
        7,
        6,
        "10.5281/zenodo.6",
        "done",
        "10.5281/zenodo.7",
        None,
        None,
        "https://example.test/records/7",
        "https://example.test/api/deposit/depositions/7",
        None,
        None,
        metadata,
        files,
    )

    assert deposition.metadata is metadata
    assert deposition.files is files
    assert deposition.created_utc is None
    assert deposition.linked_version_shape is None


def test_zenodo_deposition_preserves_the_former_keyword_constructor() -> None:
    metadata = ZenodoMetadataSnapshot.from_mapping({"title": "Legacy caller"})
    deposition = ZenodoDeposition(
        id=7,
        record_id=7,
        concept_record_id=6,
        concept_doi="10.5281/zenodo.6",
        state="done",
        doi="10.5281/zenodo.7",
        reserved_doi=None,
        reserved_record_id=None,
        html_url=None,
        self_url=None,
        bucket_url=None,
        publish_url=None,
        metadata=metadata,
        files=(),
    )

    assert deposition.metadata is metadata
    assert deposition.files == ()
    assert deposition.created_utc is None
    assert deposition.linked_version_shape is None


def test_client_round_trip_uses_typed_draft_boundary(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nActive Fedference\n")
    _ZenodoHandler.expected_content = pdf.read_bytes()
    _ZenodoHandler.uploaded = False
    _ZenodoHandler.extra_file = False
    _ZenodoHandler.add_extra_after_publish = False
    _ZenodoHandler.ignore_metadata_update = False
    _ZenodoHandler.metadata_payload = None
    _ZenodoHandler.state = "unsubmitted"
    _ZenodoHandler.publish_calls = 0
    _ZenodoHandler.authorization_headers = []
    _ZenodoHandler.put_payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ZenodoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_base = f"http://127.0.0.1:{server.server_port}/api"
        client = ZenodoClient("test-token", api_base=api_base, timeout=5.0)
        reserved = client.reserve_doi({"title": "Draft", "doi": "10.5281/zenodo/old"})
        assert reserved.id == 7
        assert reserved.reserved_doi == "10.5281/zenodo.7"
        assert reserved.record_id == 7
        assert reserved.concept_record_id == 6
        assert reserved.concept_doi == "10.5281/zenodo.6"
        assert reserved.created_utc is None
        summary = reserved.as_dict()
        assert summary["metadata"]["title"] == "Release draft"
        assert summary["metadata_sha256"] == reserved.metadata.sha256
        assert len(summary["metadata_sha256"]) == 64
        mutable_metadata = reserved.metadata.as_dict()
        mutable_metadata["title"] = "locally changed"
        assert reserved.metadata.as_dict()["title"] == "Release draft"

        with pytest.raises(ZenodoError, match="requested metadata DOI is required"):
            client.update_metadata(7, {"title": "Missing release identity"})
        with pytest.raises(ZenodoError, match="reserved DOI does not match"):
            client.update_metadata(
                7,
                {"title": "Wrong draft", "doi": "10.5281/zenodo.999"},
            )
        assert _ZenodoHandler.put_payloads == []

        updated = client.update_metadata(
            7,
            {"title": "Updated", "doi": "10.5281/zenodo.7"},
        )
        assert updated.state == "unsubmitted"
        assert updated.created_utc is None
        assert updated.metadata.as_dict()["title"] == "Updated"
        assert _ZenodoHandler.put_payloads == [{"metadata": {"title": "Updated"}}]

        uploaded = client.upload_pdf(7, pdf)
        assert uploaded.filename == "paper.pdf"
        assert uploaded.checksum == hashlib.md5(pdf.read_bytes()).hexdigest()  # noqa: S324
        assert client.verify_pdf(7, pdf) == uploaded
        assert client.verify_pdf(7, pdf, remote_filename="paper.pdf") == uploaded

        published = client.publish_verified_pdf(7, pdf)
        assert published.state == "done"
        assert published.doi == "10.5281/zenodo.7"
        assert published.created_utc is None
        assert _ZenodoHandler.publish_calls == 1
        assert _ZenodoHandler.authorization_headers
        assert set(_ZenodoHandler.authorization_headers) == {"Bearer test-token"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        _ZenodoHandler.state = "unsubmitted"
        _ZenodoHandler.metadata_payload = None


def test_metadata_update_requires_canonical_post_put_refetch(
    tmp_path: Path,
) -> None:
    _ZenodoHandler.expected_content = b""
    _ZenodoHandler.uploaded = False
    _ZenodoHandler.extra_file = False
    _ZenodoHandler.add_extra_after_publish = False
    _ZenodoHandler.ignore_metadata_update = True
    _ZenodoHandler.metadata_payload = None
    _ZenodoHandler.state = "unsubmitted"
    _ZenodoHandler.publish_calls = 0
    _ZenodoHandler.authorization_headers = []
    _ZenodoHandler.put_payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ZenodoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ZenodoClient(
            "test-token",
            api_base=f"http://127.0.0.1:{server.server_port}/api",
            timeout=5.0,
        )
        with pytest.raises(ZenodoError, match="canonical request"):
            client.update_metadata(
                7,
                {"title": "Updated", "doi": "https://doi.org/10.5281/zenodo.7"},
            )
        assert _ZenodoHandler.put_payloads == [{"metadata": {"title": "Updated"}}]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        _ZenodoHandler.ignore_metadata_update = False


def test_new_version_resolves_latest_draft_link() -> None:
    _NewVersionHandler.authorization_headers = []
    _NewVersionHandler.new_version_calls = 0
    _NewVersionHandler.draft_state = "unsubmitted"
    _NewVersionHandler.include_reserved_doi = True
    _NewVersionHandler.draft_concept_record_id = 6
    _NewVersionHandler.draft_concept_doi = "10.5281/zenodo.6"
    _NewVersionHandler.reserved_record_id = 8
    _NewVersionHandler.reserved_doi = "10.5281/zenodo.8"
    _NewVersionHandler.draft_version = "1.0.4"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NewVersionHandler)
    _NewVersionHandler.api_base = f"http://127.0.0.1:{server.server_port}/api"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_base = f"http://127.0.0.1:{server.server_port}/api"
        client = ZenodoClient("test-token", api_base=api_base, timeout=5.0)
        result = client.new_version(7)
        assert result.id == 8
        assert result.state == "unsubmitted"
        assert result.reserved_doi == "10.5281/zenodo.8"
        assert result.reserved_record_id == 8
        assert result.record_id == 8
        assert result.concept_record_id == 6
        assert result.concept_doi == "10.5281/zenodo.6"
        assert result.metadata.as_dict()["title"] == "Published source"
        assert result.metadata.as_dict()["version"] == "1.0.4"
        assert result.files[0].filename == "paper.pdf"
        assert result.created_utc == "2026-08-28T01:30:00Z"
        assert result.linked_version_shape == "legacy_inherited"
        cli_summary = _summary(result, source_deposition_id=7)
        assert cli_summary["source_deposition_id"] == 7
        assert cli_summary["metadata"] == result.metadata.as_dict()
        encoded_summary = json.dumps(cli_summary, sort_keys=True)
        assert "test-token" not in encoded_summary
        assert "env-file" not in encoded_summary
        assert _NewVersionHandler.new_version_calls == 1
        assert _NewVersionHandler.authorization_headers == [
            "Bearer test-token",
            "Bearer test-token",
            "Bearer test-token",
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_legacy_linked_version_does_not_require_a_creation_timestamp(
    new_version_client: ZenodoClient,
) -> None:
    _NewVersionHandler.draft_created = None

    result = new_version_client.new_version(7)

    assert result.linked_version_shape == "legacy_inherited"
    assert result.created_utc is None


def test_legacy_linked_version_ignores_an_unneeded_creation_timestamp_spelling(
    new_version_client: ZenodoClient,
) -> None:
    _NewVersionHandler.draft_created = "2026-08-28 01:30:00Z"

    result = new_version_client.new_version(7)

    assert result.linked_version_shape == "legacy_inherited"
    assert result.created_utc is None


@pytest.mark.parametrize("draft_version", [None, "1.0.4"])
def test_new_version_accepts_the_current_empty_separate_record_shape(
    new_version_client: ZenodoClient,
    draft_version: str | None,
) -> None:
    _NewVersionHandler.draft_version = draft_version
    _NewVersionHandler.draft_publication_date = "2026-08-28"
    _NewVersionHandler.draft_files = []

    result = new_version_client.new_version(7)

    assert result.linked_version_shape == "current_separate_record"
    assert result.created_utc == "2026-08-28T01:30:00Z"
    assert result.files == ()
    assert result.metadata.as_dict()["publication_date"] == "2026-08-28"
    if draft_version is None:
        assert "version" not in result.metadata.as_dict()
    else:
        assert result.metadata.as_dict()["version"] == "1.0.4"


@pytest.mark.parametrize(
    ("created", "publication_date", "created_utc"),
    [
        ("2026-08-28T01:30:00Z", "2026-08-28", "2026-08-28T01:30:00Z"),
        (
            "2026-08-28T01:30:00.1+00:00",
            "2026-08-28",
            "2026-08-28T01:30:00.100000Z",
        ),
        (
            "2026-08-28T01:30:00.12Z",
            "2026-08-28",
            "2026-08-28T01:30:00.120000Z",
        ),
        (
            "2026-08-28T01:30:00.123+00:00",
            "2026-08-28",
            "2026-08-28T01:30:00.123000Z",
        ),
        (
            "2026-08-28T01:30:00.1234Z",
            "2026-08-28",
            "2026-08-28T01:30:00.123400Z",
        ),
        (
            "2026-08-28T01:30:00.12345+00:00",
            "2026-08-28",
            "2026-08-28T01:30:00.123450Z",
        ),
        (
            "2026-08-28T01:30:00.123456Z",
            "2026-08-28",
            "2026-08-28T01:30:00.123456Z",
        ),
        (
            "2026-08-27T18:30:00-07:00",
            "2026-08-28",
            "2026-08-28T01:30:00Z",
        ),
        (
            "2026-08-28T05:00:00+05:30",
            "2026-08-27",
            "2026-08-27T23:30:00Z",
        ),
    ],
)
def test_current_linked_version_accepts_only_strict_rfc3339_creation_timestamps(
    new_version_client: ZenodoClient,
    created: str,
    publication_date: str,
    created_utc: str,
) -> None:
    _NewVersionHandler.draft_version = None
    _NewVersionHandler.draft_publication_date = publication_date
    _NewVersionHandler.draft_created = created
    _NewVersionHandler.draft_files = []

    result = new_version_client.new_version(7)

    assert result.linked_version_shape == "current_separate_record"
    assert result.created_utc == created_utc


@pytest.mark.parametrize("publication_date", [None, "2026-08-27", "2026-08-28T00:00:00Z"])
def test_new_version_rejects_an_arbitrary_or_missing_normalized_publication_date(
    new_version_client: ZenodoClient,
    publication_date: str | None,
) -> None:
    _NewVersionHandler.draft_version = None
    _NewVersionHandler.draft_publication_date = publication_date
    _NewVersionHandler.draft_files = []

    with pytest.raises(ZenodoError, match="publication_date"):
        new_version_client.new_version(7)


@pytest.mark.parametrize(
    "created",
    [
        None,
        123,
        "2026-08-28",
        "2026-08-28 09:30:00Z",
        "2026-08-28_09:30:00Z",
        "20260828T093000Z",
        "2026-W35-5T09:30:00Z",
        "2026-08-28T09:30:00",
        "2026-08-28t09:30:00z",
        "2026-08-28T09:30Z",
        "2026-08-28T09:30:00+2400",
        "2026-08-28T09:30:00+24:00",
        "2026-08-28T09:30:00+01:60",
        "2026-02-30T09:30:00Z",
        "2026-13-28T09:30:00Z",
        " 2026-08-28T09:30:00Z",
        "2026-08-28T09:30:00Z ",
        "2026-08-28T09:30:00.1234567Z",
        "2026-08-28T09:30:00,123Z",
    ],
)
def test_new_version_rejects_a_missing_or_invalid_server_creation_timestamp(
    new_version_client: ZenodoClient,
    created: object | None,
) -> None:
    _NewVersionHandler.draft_version = None
    _NewVersionHandler.draft_publication_date = "2026-08-28"
    _NewVersionHandler.draft_created = created
    _NewVersionHandler.draft_files = []

    with pytest.raises(ZenodoError, match="creation timestamp"):
        new_version_client.new_version(7)


def test_new_version_rejects_any_other_purpose_metadata_drift(
    new_version_client: ZenodoClient,
) -> None:
    _NewVersionHandler.draft_version = None
    _NewVersionHandler.draft_publication_date = "2026-08-28"
    _NewVersionHandler.draft_extra_metadata = {"notes": "unexpected drift"}
    _NewVersionHandler.draft_files = []

    with pytest.raises(ZenodoError, match="purpose metadata"):
        new_version_client.new_version(7)


@pytest.mark.parametrize("file_shape", ["partial", "unrelated", "normalized_with_file"])
def test_new_version_rejects_partial_unrelated_or_mixed_file_shapes(
    new_version_client: ZenodoClient,
    file_shape: str,
) -> None:
    if file_shape == "partial":
        _NewVersionHandler.source_files.append(
            {
                "id": "published-data",
                "filename": "data.json",
                "filesize": 2,
                "checksum": "md5:data",
            }
        )
    elif file_shape == "unrelated":
        _NewVersionHandler.draft_files = [
            {
                "id": "unrelated",
                "filename": "unrelated.txt",
                "filesize": 3,
                "checksum": "md5:unrelated",
            }
        ]
    else:
        _NewVersionHandler.draft_version = None
        _NewVersionHandler.draft_publication_date = "2026-08-28"

    with pytest.raises(ZenodoError, match="file set"):
        new_version_client.new_version(7)


def test_new_version_reuses_the_same_valid_current_draft_idempotently(
    new_version_client: ZenodoClient,
) -> None:
    _NewVersionHandler.draft_version = None
    _NewVersionHandler.draft_publication_date = "2026-08-28"
    _NewVersionHandler.draft_files = []

    first = new_version_client.new_version(7)
    second = new_version_client.new_version(7)

    assert first.as_dict() == second.as_dict()
    assert first.id == second.id == 8
    assert first.linked_version_shape == second.linked_version_shape
    assert _NewVersionHandler.new_version_calls == 1


def test_new_version_recovers_the_exact_production_already_exists_response(
    new_version_client: ZenodoClient,
) -> None:
    _NewVersionHandler.post_mode = "already_exists"
    _NewVersionHandler.draft_version = None
    _NewVersionHandler.draft_publication_date = "2026-08-28"
    _NewVersionHandler.draft_files = []

    result = new_version_client.new_version(7)

    assert result.id == 8
    assert result.linked_version_shape == "current_separate_record"
    assert _NewVersionHandler.new_version_calls == 1
    assert len(_NewVersionHandler.authorization_headers) == 4


def test_source_self_link_does_not_preempt_a_successful_new_version_post(
    new_version_client: ZenodoClient,
) -> None:
    _NewVersionHandler.source_latest_draft = _NewVersionHandler._draft_link(7)

    result = new_version_client.new_version(7)

    assert result.id == 8
    assert result.linked_version_shape == "legacy_inherited"
    assert _NewVersionHandler.new_version_calls == 1
    assert _NewVersionHandler.listing_paths == []


def test_source_latest_draft_accepts_one_exact_trailing_slash(
    new_version_client: ZenodoClient,
) -> None:
    _NewVersionHandler.source_latest_draft = (
        f"{_NewVersionHandler._draft_link(8)}/"
    )

    result = new_version_client.new_version(7)

    assert result.id == 8
    assert result.linked_version_shape == "legacy_inherited"
    assert _NewVersionHandler.new_version_calls == 0


@pytest.mark.parametrize(
    "link_case",
    [
        "foreign_origin",
        "hostname_mismatch",
        "protocol_relative",
        "missing_netloc",
        "malformed_scheme",
        "relative",
        "scheme_mismatch",
        "arbitrary_prefix",
        "query",
        "fragment",
        "credentials",
        "port_mismatch",
        "two_trailing_slashes",
        "zero_id",
        "leading_zero_id",
        "missing_id",
    ],
)
def test_latest_draft_link_is_bound_to_the_exact_configured_api(
    new_version_client: ZenodoClient,
    link_case: str,
) -> None:
    api_base = _NewVersionHandler.api_base
    parsed = urlsplit(api_base)
    assert parsed.hostname == "127.0.0.1"
    assert parsed.port is not None
    origin = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
    valid = _NewVersionHandler._draft_link(8)
    links = {
        "foreign_origin": "https://example.test/api/deposit/depositions/8",
        "hostname_mismatch": (
            f"{parsed.scheme}://localhost:{parsed.port}/api/deposit/depositions/8"
        ),
        "protocol_relative": f"//{parsed.netloc}/api/deposit/depositions/8",
        "missing_netloc": f"{parsed.scheme}:///api/deposit/depositions/8",
        "malformed_scheme": (
            f"zenodo+{parsed.scheme}://{parsed.netloc}/api/deposit/depositions/8"
        ),
        "relative": "/api/deposit/depositions/8",
        "scheme_mismatch": (
            f"https://{parsed.hostname}:{parsed.port}/api/deposit/depositions/8"
        ),
        "arbitrary_prefix": f"{origin}/prefix/api/deposit/depositions/8",
        "query": f"{valid}?access_token=not-allowed",
        "fragment": f"{valid}#not-allowed",
        "credentials": (
            f"{parsed.scheme}://user:password@{parsed.netloc}"
            "/api/deposit/depositions/8"
        ),
        "port_mismatch": (
            f"{parsed.scheme}://{parsed.hostname}:{parsed.port + 1}"
            "/api/deposit/depositions/8"
        ),
        "two_trailing_slashes": f"{valid}//",
        "zero_id": f"{api_base}/deposit/depositions/0",
        "leading_zero_id": f"{api_base}/deposit/depositions/08",
        "missing_id": f"{api_base}/deposit/depositions/",
    }
    _NewVersionHandler.source_latest_draft = links[link_case]

    with pytest.raises(ZenodoError, match="latest_draft"):
        new_version_client.new_version(7)

    assert _NewVersionHandler.new_version_calls == 0
    assert len(_NewVersionHandler.authorization_headers) == 1


def test_exact_existing_error_with_source_self_link_recovers_from_listing(
    new_version_client: ZenodoClient,
) -> None:
    _NewVersionHandler.source_latest_draft = _NewVersionHandler._draft_link(7)
    _NewVersionHandler.post_mode = "already_exists"
    _NewVersionHandler.draft_version = None
    _NewVersionHandler.draft_publication_date = "2026-08-28"
    _NewVersionHandler.draft_files = []

    result = new_version_client.new_version(7)

    assert result.id == 8
    assert result.linked_version_shape == "current_separate_record"
    assert _NewVersionHandler.new_version_calls == 1
    assert _NewVersionHandler.listing_paths == [
        "/api/deposit/depositions?"
        "q=conceptrecid%3A6&status=draft&sort=mostrecent&page=1&size=100&all_versions=true"
    ]


@pytest.mark.parametrize(
    "post_mode",
    ["unrelated_error", "already_exists_extra", "already_exists_wrong_status"],
)
def test_new_version_does_not_swallow_unrelated_or_near_match_http_errors(
    new_version_client: ZenodoClient,
    post_mode: str,
) -> None:
    _NewVersionHandler.post_mode = post_mode

    with pytest.raises(ZenodoError, match="Zenodo HTTP"):
        new_version_client.new_version(7)

    assert _NewVersionHandler.new_version_calls == 1
    assert len(_NewVersionHandler.authorization_headers) == 2


@pytest.mark.parametrize(
    "post_mode",
    ["echo_token", "deeply_nested_echo", "truncated_echo"],
)
def test_http_error_retains_no_echoed_token_or_raw_payload(
    new_version_client: ZenodoClient,
    post_mode: str,
) -> None:
    _NewVersionHandler.post_mode = post_mode

    with pytest.raises(ZenodoError, match="Zenodo HTTP") as caught:
        new_version_client.new_version(7)

    error = caught.value
    inspection_surfaces = "\n".join(
        (
            repr(error),
            str(error),
            repr(error.args),
            repr(vars(error)),
            repr(error.__dict__),
        )
    )
    assert "test-token" not in inspection_surfaces
    assert '"echo": "test-token"' not in inspection_surfaces
    assert "response_payload" not in error.__dict__
    assert set(error.__dict__) == {"status", "_exact_already_exists"}
    assert error.__dict__["_exact_already_exists"] is False
    assert error.__cause__ is None
    assert error.__context__ is None

    adapter_path = Path(__file__).resolve().parents[1] / "src" / "publication" / "zenodo.py"
    adapter_frames = []
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if Path(frame.f_code.co_filename).resolve() == adapter_path:
            adapter_frames.append((frame.f_code.co_name, dict(frame.f_locals)))
        traceback = traceback.tb_next

    assert adapter_frames
    forbidden_local_names = {
        "body",
        "content_type",
        "decoded_error",
        "detail",
        "error_bytes",
        "headers",
        "parsed_error",
        "payload",
        "request",
        "request_body",
    }
    for _frame_name, frame_locals in adapter_frames:
        assert forbidden_local_names.isdisjoint(frame_locals)
        assert all("test-token" not in repr(value) for value in frame_locals.values())


@pytest.mark.parametrize(
    "post_mode",
    ["deeply_nested_success", "truncated_success", "invalid_utf8_success"],
)
def test_malformed_success_response_retains_no_raw_payload(
    new_version_client: ZenodoClient,
    post_mode: str,
) -> None:
    _NewVersionHandler.post_mode = post_mode

    with pytest.raises(ZenodoError, match="malformed or oversized JSON") as caught:
        new_version_client.new_version(7)

    error = caught.value
    assert error.__cause__ is None
    assert error.__context__ is None
    adapter_path = Path(__file__).resolve().parents[1] / "src" / "publication" / "zenodo.py"
    adapter_frames = []
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if Path(frame.f_code.co_filename).resolve() == adapter_path:
            adapter_frames.append((frame.f_code.co_name, dict(frame.f_locals)))
        traceback = traceback.tb_next

    assert adapter_frames
    forbidden_local_names = {
        "body",
        "content_type",
        "headers",
        "payload",
        "raw",
        "request",
        "request_body",
        "response",
        "response_payload",
    }
    for _frame_name, frame_locals in adapter_frames:
        assert forbidden_local_names.isdisjoint(frame_locals)
        assert all("test-token" not in repr(value) for value in frame_locals.values())


def test_semantically_malformed_success_with_token_echo_fails_closed(
    new_version_client: ZenodoClient,
) -> None:
    _NewVersionHandler.post_mode = "semantic_echo_success"

    with pytest.raises(ZenodoError, match="echoed bearer credential") as caught:
        new_version_client.new_version(7)

    error = caught.value
    assert error.__cause__ is None
    assert error.__context__ is None
    adapter_path = Path(__file__).resolve().parents[1] / "src" / "publication" / "zenodo.py"
    adapter_frames = []
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if Path(frame.f_code.co_filename).resolve() == adapter_path:
            adapter_frames.append((frame.f_code.co_name, dict(frame.f_locals)))
        traceback = traceback.tb_next

    assert adapter_frames
    assert all(
        "test-token" not in repr(value)
        for _frame_name, frame_locals in adapter_frames
        for value in frame_locals.values()
    )


@pytest.mark.parametrize(
    ("token", "payload"),
    [
        ("id", {"id": 7, "record_id": 7}),
        ("1.0.4", {"metadata": {"version": "1.0.4"}}),
        ("secret", {"secret-key": {"nested": ["prefix-secret-suffix"]}}),
        ("<redacted>", {"value": "<redacted>"}),
    ],
)
def test_success_response_token_echo_is_rejected_without_semantic_rewrite(
    token: str,
    payload: object,
) -> None:
    canonical_before = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    response = io.BytesIO(canonical_before.encode("utf-8"))

    parsed, error = _json_response_without_response_body(response, token)

    assert parsed is None
    assert isinstance(error, ZenodoError)
    assert str(error) == "Zenodo response echoed bearer credential"
    assert json.dumps(payload, sort_keys=True, separators=(",", ":")) == canonical_before


def test_oversized_success_response_read_is_bounded_and_fails_closed() -> None:
    response = io.BytesIO(b"x" * (_JSON_RESPONSE_BODY_LIMIT + 2))

    payload, error = _json_response_without_response_body(response, "test-token")

    assert payload is None
    assert isinstance(error, ZenodoError)
    assert str(error) == "Zenodo returned malformed or oversized JSON content"
    assert response.tell() == _JSON_RESPONSE_BODY_LIMIT + 1


@pytest.mark.parametrize(
    ("listing_mode", "message"),
    [
        ("zero", "exactly one distinct"),
        ("multiple", "exactly one distinct"),
        ("truncated", "truncated"),
        ("malformed", "malformed"),
        ("partial", "partial deposition"),
        ("wrong_concept", "exactly one distinct"),
    ],
)
def test_existing_draft_listing_recovery_fails_closed(
    new_version_client: ZenodoClient,
    listing_mode: str,
    message: str,
) -> None:
    _NewVersionHandler.post_mode = "already_exists"
    _NewVersionHandler.source_latest_draft = _NewVersionHandler._draft_link(7)
    _NewVersionHandler.listing_mode = listing_mode

    with pytest.raises(ZenodoError, match=message):
        new_version_client.new_version(7)

    assert _NewVersionHandler.new_version_calls == 1
    assert len(_NewVersionHandler.listing_paths) == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("state", "not an unsubmitted draft"),
        ("lineage", "different concept record"),
        ("reservation", "reserved DOI"),
        ("metadata", "purpose metadata"),
        ("files", "empty file set"),
    ],
)
def test_source_link_recovery_revalidates_every_draft_boundary(
    new_version_client: ZenodoClient,
    mutation: str,
    message: str,
) -> None:
    _NewVersionHandler.draft_version = None
    _NewVersionHandler.draft_publication_date = "2026-08-28"
    _NewVersionHandler.draft_files = []
    first = new_version_client.new_version(7)
    assert first.linked_version_shape == "current_separate_record"

    if mutation == "state":
        _NewVersionHandler.draft_state = "done"
    elif mutation == "lineage":
        _NewVersionHandler.draft_concept_record_id = 9
        _NewVersionHandler.draft_concept_doi = "10.5281/zenodo.9"
    elif mutation == "reservation":
        _NewVersionHandler.reserved_doi = "10.5281/zenodo.999"
    elif mutation == "metadata":
        _NewVersionHandler.draft_extra_metadata = {"unexpected": True}
    else:
        _NewVersionHandler.draft_files = [
            {
                "id": "unrelated",
                "filename": "unrelated.txt",
                "filesize": 1,
                "checksum": "md5:unrelated",
            }
        ]

    with pytest.raises(ZenodoError, match=message):
        new_version_client.new_version(7)

    assert _NewVersionHandler.new_version_calls == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "draft_concept_record_id": 9,
                "draft_concept_doi": "10.5281/zenodo.9",
            },
            "different concept record",
        ),
        ({"reserved_record_id": 9}, "reserved record id"),
        ({"reserved_doi": "10.5281/zenodo.999"}, "reserved DOI"),
        ({"draft_version": "9.9.9"}, "version"),
    ],
)
def test_new_version_binds_concept_reservation_and_inherited_purpose(
    changes: dict[str, object],
    message: str,
) -> None:
    _NewVersionHandler.authorization_headers = []
    _NewVersionHandler.new_version_calls = 0
    _NewVersionHandler.draft_state = "unsubmitted"
    _NewVersionHandler.include_reserved_doi = True
    _NewVersionHandler.draft_concept_record_id = 6
    _NewVersionHandler.draft_concept_doi = "10.5281/zenodo.6"
    _NewVersionHandler.reserved_record_id = 8
    _NewVersionHandler.reserved_doi = "10.5281/zenodo.8"
    _NewVersionHandler.draft_version = "1.0.4"
    for attribute, value in changes.items():
        setattr(_NewVersionHandler, attribute, value)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NewVersionHandler)
    _NewVersionHandler.api_base = f"http://127.0.0.1:{server.server_port}/api"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ZenodoClient(
            "test-token",
            api_base=f"http://127.0.0.1:{server.server_port}/api",
            timeout=5.0,
        )
        with pytest.raises(ZenodoError, match=message):
            client.new_version(7)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    ("draft_state", "include_reserved_doi"),
    [("done", True), ("unsubmitted", False)],
)
def test_new_version_rejects_a_noneditable_or_unreserved_latest_draft(
    draft_state: str,
    include_reserved_doi: bool,
) -> None:
    _NewVersionHandler.authorization_headers = []
    _NewVersionHandler.new_version_calls = 0
    _NewVersionHandler.draft_state = draft_state
    _NewVersionHandler.include_reserved_doi = include_reserved_doi
    _NewVersionHandler.draft_concept_record_id = 6
    _NewVersionHandler.draft_concept_doi = "10.5281/zenodo.6"
    _NewVersionHandler.reserved_record_id = 8
    _NewVersionHandler.reserved_doi = "10.5281/zenodo.8"
    _NewVersionHandler.draft_version = "1.0.4"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NewVersionHandler)
    _NewVersionHandler.api_base = f"http://127.0.0.1:{server.server_port}/api"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ZenodoClient(
            "test-token",
            api_base=f"http://127.0.0.1:{server.server_port}/api",
            timeout=5.0,
        )
        with pytest.raises(
            ZenodoError,
            match="not an unsubmitted draft with a reserved DOI",
        ):
            client.new_version(7)
        assert _NewVersionHandler.new_version_calls == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_publish_verified_pdf_rejects_an_unexpected_inherited_file(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nActive Fedference\n")
    _ZenodoHandler.expected_content = pdf.read_bytes()
    _ZenodoHandler.uploaded = True
    _ZenodoHandler.extra_file = True
    _ZenodoHandler.add_extra_after_publish = False
    _ZenodoHandler.ignore_metadata_update = False
    _ZenodoHandler.metadata_payload = None
    _ZenodoHandler.state = "unsubmitted"
    _ZenodoHandler.publish_calls = 0
    _ZenodoHandler.authorization_headers = []
    _ZenodoHandler.put_payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ZenodoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ZenodoClient(
            "test-token",
            api_base=f"http://127.0.0.1:{server.server_port}/api",
            timeout=5.0,
        )
        with pytest.raises(
            ZenodoError,
            match="exactly the verified PDF",
        ):
            client.publish_verified_pdf(7, pdf)
        assert _ZenodoHandler.publish_calls == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        _ZenodoHandler.extra_file = False


def test_publish_refetch_detects_a_post_request_file_change(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nActive Fedference\n")
    _ZenodoHandler.expected_content = pdf.read_bytes()
    _ZenodoHandler.uploaded = True
    _ZenodoHandler.extra_file = False
    _ZenodoHandler.add_extra_after_publish = True
    _ZenodoHandler.ignore_metadata_update = False
    _ZenodoHandler.metadata_payload = None
    _ZenodoHandler.state = "unsubmitted"
    _ZenodoHandler.publish_calls = 0
    _ZenodoHandler.authorization_headers = []
    _ZenodoHandler.put_payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ZenodoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ZenodoClient(
            "test-token",
            api_base=f"http://127.0.0.1:{server.server_port}/api",
            timeout=5.0,
        )
        with pytest.raises(ZenodoError, match="post-publish refetch changed"):
            client.publish_verified_pdf(7, pdf)
        assert _ZenodoHandler.publish_calls == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        _ZenodoHandler.add_extra_after_publish = False
        _ZenodoHandler.extra_file = False
        _ZenodoHandler.state = "unsubmitted"


def test_published_metadata_edit_preserves_doi_and_does_not_touch_files() -> None:
    _PublishedMetadataEditHandler.authorization_headers = []
    _PublishedMetadataEditHandler.edit_calls = 0
    _PublishedMetadataEditHandler.publish_calls = 0
    _PublishedMetadataEditHandler.put_payloads = []
    _PublishedMetadataEditHandler.metadata_payload = None
    _PublishedMetadataEditHandler.state = "done"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PublishedMetadataEditHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_base = f"http://127.0.0.1:{server.server_port}/api"
        client = ZenodoClient("test-token", api_base=api_base, timeout=5.0)
        with pytest.raises(ZenodoError, match="does not match"):
            client.edit_published_metadata(
                7,
                {
                    "description": "Wrong record.",
                    "doi": "10.5281/zenodo.999",
                },
            )
        assert _PublishedMetadataEditHandler.put_payloads == []
        edited = client.edit_published_metadata(
            7,
            {
                "description": "The complete paper abstract.",
                "doi": "https://doi.org/10.5281/zenodo.7",
            },
        )
        assert edited.state == "inprogress"
        assert edited.doi == "10.5281/zenodo.7"
        assert _PublishedMetadataEditHandler.edit_calls == 1
        assert _PublishedMetadataEditHandler.put_payloads == [
            {"metadata": {"description": "The complete paper abstract.", "access_right": "open"}}
        ]

        published = client.publish_metadata_edit(7)
        assert published.state == "done"
        assert published.doi == "10.5281/zenodo.7"
        assert _PublishedMetadataEditHandler.publish_calls == 1
        assert set(_PublishedMetadataEditHandler.authorization_headers) == {"Bearer test-token"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_metadata_edit_mode_cannot_publish_an_ordinary_unsubmitted_draft(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nActive Fedference\n")
    _ZenodoHandler.expected_content = pdf.read_bytes()
    _ZenodoHandler.uploaded = True
    _ZenodoHandler.extra_file = True
    _ZenodoHandler.add_extra_after_publish = False
    _ZenodoHandler.ignore_metadata_update = False
    _ZenodoHandler.metadata_payload = None
    _ZenodoHandler.state = "unsubmitted"
    _ZenodoHandler.publish_calls = 0
    _ZenodoHandler.authorization_headers = []
    _ZenodoHandler.put_payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ZenodoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ZenodoClient(
            "test-token",
            api_base=f"http://127.0.0.1:{server.server_port}/api",
            timeout=5.0,
        )
        with pytest.raises(ZenodoError, match="published metadata edits require"):
            client.edit_published_metadata(
                7,
                {"title": "Not a metadata edit", "doi": "10.5281/zenodo.7"},
            )
        with pytest.raises(ZenodoError, match="opened and verified by this client"):
            client.publish_metadata_edit(7)
        assert _ZenodoHandler.put_payloads == []
        assert _ZenodoHandler.publish_calls == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        _ZenodoHandler.extra_file = False


def test_cli_inspects_a_linked_version_over_real_loopback_http(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env_file = tmp_path / "private-token.env"
    env_file.write_text("ZENODO_PROD_TOKEN=cli-secret\n", encoding="utf-8")
    _NewVersionHandler.authorization_headers = []
    _NewVersionHandler.new_version_calls = 0
    _NewVersionHandler.draft_state = "unsubmitted"
    _NewVersionHandler.include_reserved_doi = True
    _NewVersionHandler.draft_concept_record_id = 6
    _NewVersionHandler.draft_concept_doi = "10.5281/zenodo.6"
    _NewVersionHandler.reserved_record_id = 8
    _NewVersionHandler.reserved_doi = "10.5281/zenodo.8"
    _NewVersionHandler.draft_version = None
    _NewVersionHandler.draft_publication_date = "2026-08-28"
    _NewVersionHandler.draft_files = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NewVersionHandler)
    _NewVersionHandler.api_base = f"http://127.0.0.1:{server.server_port}/api"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        exit_code = zenodo_main(
            [
                "--project-root",
                str(tmp_path),
                "--env-file",
                str(env_file),
                "--new-version-of",
                "7",
            ],
            api_base=f"http://127.0.0.1:{server.server_port}/api",
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        summary = json.loads(captured.out)
        assert summary["id"] == 8
        assert summary["source_deposition_id"] == 7
        assert summary["token_source"] == "ZENODO_PROD_TOKEN"
        assert summary["created_utc"] == "2026-08-28T01:30:00Z"
        assert summary["linked_version_shape"] == "current_separate_record"
        assert summary["files"] == []
        assert "version" not in summary["metadata"]
        assert "cli-secret" not in captured.out + captured.err
        assert str(env_file) not in captured.out + captured.err
        assert _NewVersionHandler.new_version_calls == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cli_combines_exact_pdf_verification_and_publication_over_loopback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env_file = tmp_path / "private-token.env"
    env_file.write_text("ZENODO_PROD_TOKEN=cli-secret\n", encoding="utf-8")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nActive Fedference\n")
    _ZenodoHandler.expected_content = pdf.read_bytes()
    _ZenodoHandler.uploaded = True
    _ZenodoHandler.extra_file = False
    _ZenodoHandler.add_extra_after_publish = False
    _ZenodoHandler.ignore_metadata_update = False
    _ZenodoHandler.metadata_payload = None
    _ZenodoHandler.state = "unsubmitted"
    _ZenodoHandler.publish_calls = 0
    _ZenodoHandler.authorization_headers = []
    _ZenodoHandler.put_payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ZenodoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        exit_code = zenodo_main(
            [
                "--project-root",
                str(tmp_path),
                "--env-file",
                str(env_file),
                "--deposition-id",
                "7",
                "--verify",
                str(pdf),
                "--publish",
                "--confirm-publish",
            ],
            api_base=f"http://127.0.0.1:{server.server_port}/api",
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        summary = json.loads(captured.out)
        assert summary["state"] == "done"
        assert summary["doi"] == "10.5281/zenodo.7"
        assert summary["token_source"] == "ZENODO_PROD_TOKEN"
        assert summary["files"] == [
            {
                "filename": "paper.pdf",
                "filesize": pdf.stat().st_size,
                "checksum": hashlib.md5(pdf.read_bytes()).hexdigest(),  # noqa: S324
            }
        ]
        assert "cli-secret" not in captured.out + captured.err
        assert str(env_file) not in captured.out + captured.err
        assert _ZenodoHandler.publish_calls == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        _ZenodoHandler.state = "unsubmitted"


@pytest.mark.parametrize(
    "mutation_flags",
    [
        ["--update-metadata"],
        ["--upload", "paper.pdf"],
        ["--verify", "paper.pdf"],
        ["--verify", "paper.pdf", "--publish", "--confirm-publish"],
    ],
)
def test_cli_new_version_operation_rejects_follow_on_draft_mutation(
    mutation_flags: list[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        zenodo_main(["--new-version-of", "7", *mutation_flags])
    assert exc_info.value.code == 2
