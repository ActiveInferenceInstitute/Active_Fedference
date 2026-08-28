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

    authorization_headers: list[str] = []
    new_version_calls = 0
    draft_state = "unsubmitted"
    include_reserved_doi = True
    draft_concept_record_id = 6
    draft_concept_doi = "10.5281/zenodo.6"
    reserved_record_id = 8
    reserved_doi = "10.5281/zenodo.8"
    draft_version = "1.0.4"

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
            "record_id": 7,
            "conceptrecid": "6",
            "conceptdoi": "10.5281/zenodo.6",
            "state": "done",
            "doi": "10.5281/zenodo.7",
            "metadata": {
                "title": "Published source",
                "description": "Inherited abstract.",
                "version": "1.0.4",
            },
            "links": {
                "html": "http://example.test/records/7",
                "self": "http://example.test/api/deposit/depositions/7",
            },
            "files": [
                {
                    "id": "published-file",
                    "filename": "paper.pdf",
                    "filesize": 1,
                    "checksum": "md5:old",
                }
            ],
        }

    @classmethod
    def _draft(cls) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "title": "Published source",
            "description": "Inherited abstract.",
            "version": cls.draft_version,
        }
        if cls.include_reserved_doi:
            metadata["prereserve_doi"] = {
                "doi": cls.reserved_doi,
                "recid": cls.reserved_record_id,
            }
        return {
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
            self._write_json(type(self)._draft())
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
        ({"draft_version": "9.9.9"}, "canonical request"),
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
    _NewVersionHandler.draft_version = "1.0.4"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NewVersionHandler)
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
def test_cli_new_version_operation_is_inspection_only(
    mutation_flags: list[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        zenodo_main(["--new-version-of", "7", *mutation_flags])
    assert exc_info.value.code == 2
