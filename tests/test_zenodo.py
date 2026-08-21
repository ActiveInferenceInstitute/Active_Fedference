from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from publication.zenodo import (
    ZenodoClient,
    ZenodoError,
    _deposition,
    _multipart_body,
    token_from_env_file,
    token_from_environment,
)


class _ZenodoHandler(BaseHTTPRequestHandler):
    expected_content = b""
    uploaded = False
    authorization_headers: list[str] = []
    put_payloads: list[dict[str, Any]] = []

    def log_message(self, _format: str, *_args: object) -> None:
        return

    @classmethod
    def _deposition(cls, *, published: bool = False) -> dict[str, Any]:
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
        return {
            "id": 7,
            "state": "done" if published else "unsubmitted",
            **({"doi": "10.5281/zenodo.7"} if published else {}),
            "metadata": {"prereserve_doi": {"doi": "10.5281/zenodo.7", "recid": 7}},
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
            self._write_json(self._deposition(published=True))
            return
        self._write_json({"error": "not found"}, status=404)


class _NewVersionHandler(BaseHTTPRequestHandler):
    """A real loopback HTTP boundary for the Zenodo new-version action."""

    authorization_headers: list[str] = []
    new_version_calls = 0

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _write_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _published_source() -> dict[str, Any]:
        return {
            "id": 7,
            "state": "done",
            "doi": "10.5281/zenodo.7",
            "metadata": {},
            "links": {
                "html": "http://example.test/records/7",
                "self": "http://example.test/api/deposit/depositions/7",
            },
            "files": [],
        }

    @staticmethod
    def _draft() -> dict[str, Any]:
        return {
            "id": 8,
            "state": "unsubmitted",
            "metadata": {"prereserve_doi": {"doi": "10.5281/zenodo.8", "recid": 8}},
            "links": {
                "html": "http://example.test/records/8",
                "self": "http://example.test/api/deposit/depositions/8",
            },
            "files": [
                {"id": "inherited", "filename": "paper.pdf", "filesize": 1, "checksum": "md5:old"}
            ],
        }

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        type(self).authorization_headers.append(self.headers.get("Authorization", ""))
        if self.path == "/api/deposit/depositions/7":
            self._write_json(self._published_source())
            return
        if self.path == "/api/deposit/depositions/8":
            self._write_json(self._draft())
            return
        self._write_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        type(self).authorization_headers.append(self.headers.get("Authorization", ""))
        if self.path == "/api/deposit/depositions/7/actions/newversion":
            type(self).new_version_calls += 1
            self._write_json(
                {"links": {"latest_draft": "http://zenodo.test/api/deposit/depositions/8"}}
            )
            return
        self._write_json({"error": "not found"}, status=404)


class _PublishedMetadataEditHandler(BaseHTTPRequestHandler):
    """Loopback boundary for Zenodo's metadata-only published-record edit."""

    authorization_headers: list[str] = []
    edit_calls = 0
    publish_calls = 0
    put_payloads: list[dict[str, Any]] = []
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
        return {
            "id": 7,
            "state": cls.state,
            "doi": "10.5281/zenodo.7",
            "metadata": {"prereserve_doi": {"doi": "10.5281/zenodo.7", "recid": 7}},
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
    with pytest.raises(ZenodoError, match="does not exist"):
        token_from_env_file(tmp_path / "missing.env")
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
        {"id": 7, "state": "unsubmitted", "metadata": []},
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


def test_client_round_trip_uses_typed_draft_boundary(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nActive Fedference\n")
    _ZenodoHandler.expected_content = pdf.read_bytes()
    _ZenodoHandler.uploaded = False
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

        updated = client.update_metadata(7, {"title": "Updated"})
        assert updated.state == "unsubmitted"
        assert _ZenodoHandler.put_payloads == [{"metadata": {"title": "Updated"}}]

        uploaded = client.upload_pdf(7, pdf)
        assert uploaded.filename == "paper.pdf"
        assert uploaded.checksum == hashlib.md5(pdf.read_bytes()).hexdigest()  # noqa: S324
        assert client.verify_pdf(7, pdf) == uploaded
        assert client.verify_pdf(7, pdf, remote_filename="paper.pdf") == uploaded

        published = client.publish(7)
        assert published.state == "done"
        assert published.doi == "10.5281/zenodo.7"
        assert _ZenodoHandler.authorization_headers
        assert set(_ZenodoHandler.authorization_headers) == {"Bearer test-token"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_new_version_resolves_latest_draft_link() -> None:
    _NewVersionHandler.authorization_headers = []
    _NewVersionHandler.new_version_calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NewVersionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_base = f"http://127.0.0.1:{server.server_port}/api"
        client = ZenodoClient("test-token", api_base=api_base, timeout=5.0)
        result = client.new_version(7)
        assert result.id == 8
        assert result.state == "unsubmitted"
        assert result.reserved_doi == "10.5281/zenodo.8"
        assert result.files[0].filename == "paper.pdf"
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


def test_published_metadata_edit_preserves_doi_and_does_not_touch_files() -> None:
    _PublishedMetadataEditHandler.authorization_headers = []
    _PublishedMetadataEditHandler.edit_calls = 0
    _PublishedMetadataEditHandler.publish_calls = 0
    _PublishedMetadataEditHandler.put_payloads = []
    _PublishedMetadataEditHandler.state = "done"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PublishedMetadataEditHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_base = f"http://127.0.0.1:{server.server_port}/api"
        client = ZenodoClient("test-token", api_base=api_base, timeout=5.0)
        edited = client.edit_published_metadata(
            7,
            {"description": "The complete paper abstract.", "doi": "10.5281/zenodo/other"},
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
