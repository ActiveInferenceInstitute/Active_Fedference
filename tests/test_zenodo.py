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

        published = client.publish(7)
        assert published.state == "done"
        assert published.doi == "10.5281/zenodo.7"
        assert _ZenodoHandler.authorization_headers
        assert set(_ZenodoHandler.authorization_headers) == {"Bearer test-token"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
