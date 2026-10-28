"""Real HTTP characterization of the Zenodo phase and credential boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TypeAlias

import pytest

from publication.zenodo import ZenodoClient, ZenodoError, ZenodoFile
from scripts.zenodo_release import _summary

_JSON: TypeAlias = (
    None | bool | int | float | str | list["_JSON"] | dict[str, "_JSON"]
)
_TOKEN = "task17-dummy-bearer-never-a-real-credential"
_PDF = b"%PDF-1.7\nTask17 loopback characterization\n"
_MD5 = hashlib.md5(_PDF).hexdigest()  # noqa: S324 - Zenodo wire contract
_METADATA: dict[str, _JSON] = {"title": "Pinned draft", "access_right": "open"}


@dataclass
class _Service:
    """Mutable state belongs to one bounded, real loopback server."""

    api_base: str = ""
    mode: str = "normal"
    state: str = "unsubmitted"
    uploaded: bool = False
    metadata: dict[str, _JSON] = field(default_factory=lambda: dict(_METADATA))
    requests: list[dict[str, _JSON]] = field(default_factory=list)
    redirect_target: str = ""
    version_shape: str = "legacy_inherited"
    file_checksum: str = f"md5:{_MD5}"
    response_overrides: dict[int, dict[str, _JSON]] = field(default_factory=dict)

    def deposition(self, identifier: int = 7) -> dict[str, _JSON]:
        metadata = dict(self.metadata)
        state = self.state
        files: list[_JSON] = [self.file_record()] if self.uploaded else []
        if identifier == 8:
            state = "unsubmitted"
            if self.version_shape != "legacy_inherited":
                metadata["publication_date"] = "2026-08-28"
                if self.version_shape == "current_separate_record":
                    files = []
        metadata["prereserve_doi"] = {
            "doi": f"10.5281/zenodo.{identifier}", "recid": identifier,
        }
        return {
            "id": identifier, "record_id": identifier,
            "conceptrecid": "6", "conceptdoi": "10.5281/zenodo.6",
            "state": state, "metadata": metadata,
            "doi": f"10.5281/zenodo.{identifier}" if state in {"done", "inprogress"} else None,
            "created": "2026-08-27T18:30:00-07:00",
            "links": {
                "self": f"{self.api_base}/deposit/depositions/{identifier}",
                "html": f"http://example.test/records/{identifier}",
            },
            "files": files,
        }

    def file_record(self) -> dict[str, _JSON]:
        return {
            "id": "fixture-file", "filename": "paper.pdf",
            "filesize": len(_PDF), "checksum": self.file_checksum,
        }


def _sanitized(text: str, service: _Service) -> str:
    """Normalize only credentials, ephemeral origin, and multipart nonce."""
    safe = text.replace(_TOKEN, "<dummy-bearer>")
    if service.api_base:
        safe = safe.replace(service.api_base, "<api>")
    if service.redirect_target:
        safe = safe.replace(service.redirect_target, "<redirect-target>")
    return re.sub(r"----active-fedference-[0-9a-f]{32}", "<boundary>", safe)


@contextmanager
def _loopback(service: _Service) -> Iterator[str]:
    """Serve a fixture with bounded connections and prove complete teardown."""

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(3)

        def log_message(self, _format: str, *_args: str) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802 - stdlib interface
            self.respond()

        def do_POST(self) -> None:  # noqa: N802 - stdlib interface
            self.respond()

        def do_PUT(self) -> None:  # noqa: N802 - stdlib interface
            self.respond()

        def do_DELETE(self) -> None:  # noqa: N802 - stdlib interface
            self.respond()

        def respond(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload: _JSON = service.deposition()
            status = 200
            location = ""
            if service.mode == "redirect":
                status, location, payload = 302, service.redirect_target, {}
            elif service.mode == "echo_key":
                payload = {f"prefix-{_TOKEN}": {"nested": [7]}}
            elif service.mode == "echo_value":
                payload = {"nested": [None, {"echo": f"prefix-{_TOKEN}-suffix"}]}
            elif service.mode == "echo_error":
                status, payload = 400, {"error": _TOKEN}
            elif self.path.endswith("/actions/edit"):
                service.state = "inprogress"
                payload = service.deposition()
            elif self.path.endswith("/actions/publish"):
                service.state = "done"
                payload = service.deposition()
            elif self.path.endswith("/actions/newversion"):
                payload = {"links": {"latest_draft": f"{service.api_base}/deposit/depositions/8"}}
            elif self.path.endswith("/8"):
                payload = service.deposition(8)
            elif self.command == "DELETE":
                service.uploaded = False
                status, payload = 204, None
            elif self.command == "PUT":
                decoded = json.loads(body)
                assert isinstance(decoded, dict) and isinstance(decoded["metadata"], dict)
                service.metadata = decoded["metadata"]
                payload = service.deposition()
            elif self.path.endswith("/files"):
                service.uploaded = True
                service.file_checksum = f"md5:{_MD5}"
                payload = service.file_record()
                status = 201
            elif self.command == "POST" and self.path.endswith("/depositions"):
                status = 201
            if isinstance(payload, dict):
                payload.update(service.response_overrides.get(len(service.requests) + 1, {}))
            raw = b'{"broken":' if service.mode == "malformed" else json.dumps(payload).encode()
            if status == 204:
                raw = b""
            service.requests.append({
                "method": self.command, "path": self.path,
                "authorized": self.headers.get("Authorization") == f"Bearer {_TOKEN}",
                "request_body": _sanitized(body.decode(), service),
                "content_type": _sanitized(self.headers.get("Content-Type", ""), service),
                "status": status, "response_body": _sanitized(raw.decode(), service),
            })
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if location:
                self.send_header("Location", location)
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_port
    service.api_base = f"http://127.0.0.1:{port}/api"
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield service.api_base
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        with socket.socket() as probe:
            probe.settimeout(1)
            closed = probe.connect_ex(("127.0.0.1", port)) != 0
        print("HTTP_RECEIPT " + json.dumps(service.requests, sort_keys=True))
        print("TEARDOWN " + json.dumps({
            "thread_stopped": not thread.is_alive(), "port_closed": closed,
        }, sort_keys=True))
        print("LISTENER_RECEIPT " + json.dumps({
            "host": "127.0.0.1", "port": port, "pid": os.getpid(),
            "server_thread": thread.ident, "thread_stopped": not thread.is_alive(),
            "port_closed": closed,
        }, sort_keys=True))
        assert not thread.is_alive(), "owned HTTP service thread did not stop"
        assert closed, "owned HTTP listener is still reachable"


def _assert_safe_failure(error: ZenodoError) -> None:
    """Inspect every focused owner's exception frames, not just its message."""
    traceback = error.__traceback__
    frames = []
    while traceback is not None:
        frame = traceback.tb_frame
        origin = Path(frame.f_code.co_filename)
        if origin.parent.name == "publication" and origin.stem.startswith("zenodo"):
            frames.append(frame)
        traceback = traceback.tb_next
    assert frames
    forbidden = {
        "body", "content_type", "headers", "raw", "request", "request_body",
        "error_bytes", "parsed_error",
    }
    for frame in frames:
        assert forbidden.isdisjoint(frame.f_locals)
        assert frame.f_locals.get("response_payload") is None
        assert all(_TOKEN not in repr(value) for value in frame.f_locals.values())
    assert error.__cause__ is None
    assert error.__context__ is None
    assert _TOKEN not in repr((error, error.args, vars(error)))
    print("SAFE_FAILURE " + json.dumps({
        "message": str(error), "chain_empty": True,
        "frames": [frame.f_code.co_name for frame in frames],
    }, sort_keys=True))


def test_real_http_draft_verify_sequence(tmp_path: Path) -> None:
    # Given: a real draft service and known PDF bytes, with an explicit origin.
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_PDF)
    service = _Service()
    request = {"title": "Pinned draft", "doi": "10.5281/zenodo.999"}
    with _loopback(service) as api:
        client = ZenodoClient(_TOKEN, api_base=api, timeout=3)
        # When: reserve, retrieve, bind metadata, upload, then verify the bytes.
        reserved = client.reserve_doi(request)
        retrieved = client.get_deposition(7)
        updated = client.update_metadata(7, {"title": "Updated draft", "doi": "10.5281/zenodo.7"})
        uploaded = client.upload_pdf(7, pdf)
        verified = client.verify_pdf(7, pdf, require_exact_file_set=True)
        # Then: the actual wire requests, metadata, DOI and file identities agree.
        assert request == {"title": "Pinned draft", "doi": "10.5281/zenodo.999"}
        assert reserved == retrieved
        assert reserved.id == reserved.record_id == reserved.reserved_record_id == 7
        assert reserved.concept_record_id == 6
        assert reserved.reserved_doi == "10.5281/zenodo.7"
        assert reserved.concept_doi == "10.5281/zenodo.6"
        assert updated.metadata.as_dict() == {
            "title": "Updated draft", "prereserve_doi": {"doi": "10.5281/zenodo.7", "recid": 7},
        }
        assert uploaded == verified == ZenodoFile("fixture-file", "paper.pdf", len(_PDF), f"md5:{_MD5}")
        assert [(r["method"], r["path"]) for r in service.requests] == [
            ("POST", "/api/deposit/depositions"),
            ("GET", "/api/deposit/depositions/7"),
            ("GET", "/api/deposit/depositions/7"),
            ("PUT", "/api/deposit/depositions/7"),
            ("GET", "/api/deposit/depositions/7"),
            ("GET", "/api/deposit/depositions/7"),
            ("POST", "/api/deposit/depositions/7/files"),
            ("GET", "/api/deposit/depositions/7"),
        ]
        assert all(r["authorized"] is True for r in service.requests)
        assert service.requests[0]["request_body"] == (
            '{"metadata": {"title": "Pinned draft", "access_right": "open", "prereserve_doi": true}}'
        )
        assert service.requests[3]["request_body"] == '{"metadata": {"title": "Updated draft"}}'
        expected_multipart = (
            '--<boundary>\r\nContent-Disposition: form-data; name="name"\r\n\r\npaper.pdf\r\n'
            '--<boundary>\r\nContent-Disposition: form-data; name="file"; filename="paper.pdf"\r\n'
            'Content-Type: application/pdf\r\n\r\n'
        ) + _PDF.decode() + "\r\n--<boundary>--\r\n"
        assert service.requests[6]["request_body"] == expected_multipart
        assert service.requests[6]["content_type"] == "multipart/form-data; boundary=<boundary>"
        print("SUMMARY " + json.dumps(_summary(updated), sort_keys=True))


@pytest.mark.parametrize("mode", ["redirect", "echo_key", "echo_value", "echo_error", "malformed"])
def test_redirect_token_echo_and_unapproved_publish(tmp_path: Path, mode: str) -> None:
    # Given: two actual HTTP origins and a server-controlled adverse response.
    service, sink = _Service(mode=mode), _Service()
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_PDF)
    with _loopback(sink) as target, _loopback(service) as api:
        service.redirect_target = f"{target}/deposit/depositions/7"
        client = ZenodoClient(_TOKEN, api_base=api, timeout=3)
        # When: untrusted HTTP is admitted, followed by unauthorized phases.
        with pytest.raises(ZenodoError) as caught:
            client.get_deposition(7)
        # Then: every adverse wire class fails without retaining raw credentials.
        expected = {
            "redirect": "Zenodo HTTP 302",
            "echo_key": "Zenodo response echoed bearer credential",
            "echo_value": "Zenodo response echoed bearer credential",
            "echo_error": "Zenodo HTTP 400",
            "malformed": "Zenodo returned malformed or oversized JSON content",
        }
        assert str(caught.value) == expected[mode]
        _assert_safe_failure(caught.value)
        assert sink.requests == [], "redirect target received a request or bearer"
        assert len(service.requests) == 1
        service.mode = "normal"
        with pytest.raises(ZenodoError, match="opened and verified by this client"):
            client.publish_metadata_edit(7)
        assert len(service.requests) == 1
        with pytest.raises(ZenodoError, match="does not exist"):
            client.verify_pdf(7, tmp_path / "missing.pdf")
        assert len(service.requests) == 1
        with pytest.raises(ZenodoError, match="has 0 files named"):
            client.publish_verified_pdf(7, pdf)
        service.state = "done"
        with pytest.raises(ZenodoError, match="only an unsubmitted draft is editable"):
            client.publish(7)
        assert len(service.requests) == 3
        assert all(r["method"] == "GET" for r in service.requests)


@pytest.mark.parametrize("shape", [
    "legacy_inherited", "current_separate_record", "current_inherited_files",
])
def test_real_http_linked_version_identity(shape: str) -> None:
    # Given: actual historical/current server representations of a linked version.
    service = _Service(
        state="done", uploaded=True, version_shape=shape,
        metadata={"title": "Published source", "version": "1.0.4", "publication_date": "2026-08-20"},
    )
    with _loopback(service) as api:
        # When: the client creates and admits a linked draft.
        result = ZenodoClient(_TOKEN, api_base=api, timeout=3).new_version(7)
        # Then: the new DOI, lineage, metadata and inherited/empty files are bound.
        assert result.id == result.record_id == result.reserved_record_id == 8
        assert result.concept_record_id == 6
        assert result.concept_doi == "10.5281/zenodo.6"
        assert result.reserved_doi == "10.5281/zenodo.8"
        assert result.linked_version_shape == shape
        assert result.created_utc == "2026-08-28T01:30:00Z"
        expected_date = "2026-08-20" if shape == "legacy_inherited" else "2026-08-28"
        assert result.metadata.as_dict() == {
            "title": "Published source", "version": "1.0.4",
            "publication_date": expected_date,
            "prereserve_doi": {"doi": "10.5281/zenodo.8", "recid": 8},
        }
        expected_files = () if shape == "current_separate_record" else (
            ZenodoFile("fixture-file", "paper.pdf", len(_PDF), f"md5:{_MD5}"),
        )
        assert result.files == expected_files
        assert [(r["method"], r["path"]) for r in service.requests] == [
            ("GET", "/api/deposit/depositions/7"),
            ("POST", "/api/deposit/depositions/7/actions/newversion"),
            ("GET", "/api/deposit/depositions/8"),
        ]
        print("SUMMARY " + json.dumps(_summary(result, source_deposition_id=7), sort_keys=True))


def test_cli_wrapper_real_process_loopback_success_and_denied_publish(tmp_path: Path) -> None:
    # Given: a real service and isolated subprocess environment with only a dummy bearer.
    root = Path(__file__).resolve().parents[1]
    service = _Service()
    environment = {
        "PATH": os.defpath, "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
        "ZENODO_PROD_TOKEN": _TOKEN, "TMPDIR": str(tmp_path),
    }
    program = (
        "import sys; from scripts.zenodo_release import main; "
        "raise SystemExit(main(sys.argv[2:], api_base=sys.argv[1]))"
    )
    with _loopback(service) as api:
        # When: the actual wrapper inspects a draft, then rejects unconfirmed publish.
        success = subprocess.run(
            [sys.executable, "-c", program, api, "--deposition-id", "7"],
            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15,
        )
        denied = subprocess.run(
            [sys.executable, "-c", program, api, "--deposition-id", "7", "--publish"],
            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15,
        )
        # Then: stdout is a valid record; authorization failure issues no request.
        assert success.returncode == 0, success.stderr
        summary = json.loads(success.stdout)
        assert summary["id"] == 7
        assert summary["state"] == "unsubmitted"
        assert summary["reserved_doi"] == "10.5281/zenodo.7"
        expected_metadata: dict[str, _JSON] = dict(_METADATA)
        expected_metadata["prereserve_doi"] = {"doi": "10.5281/zenodo.7", "recid": 7}
        assert summary["metadata"] == expected_metadata
        assert success.stderr == ""
        assert denied.returncode == 2
        assert denied.stdout == ""
        assert "--publish requires --confirm-publish" in denied.stderr
        assert _TOKEN not in success.stdout + success.stderr + denied.stdout + denied.stderr
        assert len(service.requests) == 1
        print("CLI_RECEIPT " + json.dumps({
            "success_status": success.returncode, "success_stdout": summary,
            "success_stderr": success.stderr,
            "denied_status": denied.returncode, "denied_stdout": denied.stdout,
            "denied_stderr": denied.stderr,
        }, sort_keys=True))


def test_real_http_pdf_idempotence_and_explicit_replacement(tmp_path: Path) -> None:
    # Given: a draft already containing the exact known PDF.
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_PDF)
    service = _Service(uploaded=True)
    with _loopback(service) as api:
        client = ZenodoClient(_TOKEN, api_base=api, timeout=3)
        # When: reuse the exact upload, reject drift, then explicitly replace it.
        reused = client.upload_pdf(7, pdf)
        assert reused == ZenodoFile("fixture-file", "paper.pdf", len(_PDF), f"md5:{_MD5}")
        assert [(r["method"], r["path"]) for r in service.requests] == [
            ("GET", "/api/deposit/depositions/7"),
        ]
        service.file_checksum = "md5:00000000000000000000000000000000"
        with pytest.raises(ZenodoError, match="checksum or size mismatch"):
            client.verify_pdf(7, pdf)
        with pytest.raises(ZenodoError, match="must use a .pdf extension"):
            client.verify_pdf(7, pdf, remote_filename="paper.txt")
        with pytest.raises(ZenodoError, match="already contains a different file"):
            client.upload_pdf(7, pdf)
        assert all(r["method"] == "GET" for r in service.requests)
        replaced = client.upload_pdf(7, pdf, replace_existing=True)
        # Then: deletion and upload happen only on explicit replacement.
        assert replaced == reused
        assert client.verify_pdf(7, pdf, require_exact_file_set=True) == reused
        assert [r["method"] for r in service.requests] == [
            "GET", "GET", "GET", "GET", "GET", "GET", "DELETE", "POST", "GET",
        ]
        assert service.requests[6]["path"] == "/api/deposit/depositions/7/files/fixture-file"
        assert service.requests[6]["status"] == 204
        assert service.requests[6]["response_body"] == ""
        with pytest.raises(ZenodoError, match="file id is unsafe"):
            client.delete_file(7, "../unsafe")
        assert [r["method"] for r in service.requests].count("DELETE") == 1


def test_real_http_transport_rejects_dual_body_before_request() -> None:
    # Given: an actual local service that records every outbound request.
    service = _Service()
    with _loopback(service) as api:
        client = ZenodoClient(_TOKEN, api_base=api, timeout=3)
        # When: the private transport is asked to encode two competing bodies.
        with pytest.raises(ZenodoError, match="both JSON and raw bodies"):
            client._request("POST", "deposit/depositions", payload={}, body=b"conflicting")
        # Then: rejection happens before any HTTP or authorization header leaves.
        assert service.requests == []


def test_closed_loopback_transport_returns_body_free_failure() -> None:
    # Given: a local endpoint whose server has demonstrably closed.
    service = _Service()
    with _loopback(service) as api:
        client = ZenodoClient(_TOKEN, api_base=api, timeout=3)
    # When: connect to that closed, explicitly configured loopback endpoint.
    with pytest.raises(ZenodoError, match="Zenodo request failed") as caught:
        client.get_deposition(7)
    # Then: the transport error has no body, token or original exception chain.
    _assert_safe_failure(caught.value)
    assert service.requests == []


@pytest.mark.parametrize("mode", ["echo_key", "echo_value"])
def test_real_http_token_echo_never_reaches_exception_frames(mode: str) -> None:
    # Given: a successful HTTP response containing a nested dummy credential.
    service = _Service(mode=mode)
    with _loopback(service) as api:
        client = ZenodoClient(_TOKEN, api_base=api, timeout=3)
        # When: the actual JSON response crosses the transport boundary.
        with pytest.raises(ZenodoError) as caught:
            client.get_deposition(7)
        # Then: inspect frames before the finding so a bypass exposes retained data.
        _assert_safe_failure(caught.value)
        assert str(caught.value) == "Zenodo response echoed bearer credential"
        assert len(service.requests) == 1


@pytest.mark.parametrize(("fault", "finding"), [
    ("record", "metadata update response changed the record identity"),
    ("concept", "metadata update response changed the concept identity"),
    ("state", "metadata update no longer targets an unsubmitted draft"),
    ("reservation", "metadata update changed the draft DOI reservation"),
])
def test_real_http_draft_metadata_admission_stops_at_first_drift(
    fault: str, finding: str,
) -> None:
    # Given: a real draft whose update response or refetch changes one binding.
    overrides: dict[int, dict[str, _JSON]] = {}
    if fault == "record":
        overrides[2] = {
            "id": 8, "record_id": 8,
            "metadata": {
                "title": "Updated draft",
                "prereserve_doi": {"doi": "10.5281/zenodo.8", "recid": 8},
            },
        }
    elif fault == "concept":
        overrides[2] = {"conceptrecid": "9", "conceptdoi": "10.5281/zenodo.9"}
    elif fault == "state":
        overrides[3] = {"state": "done"}
    else:
        overrides[3] = {"metadata": {
            "title": "Updated draft",
            "prereserve_doi": {"doi": "10.5072/zenodo.7", "recid": 7},
        }}
    service = _Service(response_overrides=overrides)
    with _loopback(service) as api:
        client = ZenodoClient(_TOKEN, api_base=api, timeout=3)
        # When: update the metadata under the original reserved DOI.
        with pytest.raises(ZenodoError, match=finding) as caught:
            client.update_metadata(7, {"title": "Updated draft", "doi": "10.5281/zenodo.7"})
        # Then: the response is rejected without follow-on mutation or publication.
        methods = [r["method"] for r in service.requests]
        assert methods == (["GET", "PUT"] if fault in {"record", "concept"} else ["GET", "PUT", "GET"])
        assert all("/actions/" not in str(r["path"]) for r in service.requests)
        print("PHASE_FINDING " + json.dumps({"fault": fault, "finding": str(caught.value)}, sort_keys=True))


@pytest.mark.parametrize(("fault", "finding"), [
    ("source_identity", "published metadata edit has incomplete record identity"),
    ("edit_state", "metadata-edit action did not open an editable draft"),
    ("refetch_state", "metadata-edit draft is no longer editable"),
    ("refetch_doi", "metadata-edit draft changed the published DOI"),
    ("refetch_files", "metadata-only edit changed the record files"),
])
def test_real_http_metadata_edit_admission_does_not_authorize_publish(
    fault: str, finding: str,
) -> None:
    # Given: a published record with one adversarial phase-specific response.
    overrides: dict[int, dict[str, _JSON]] = {}
    if fault == "source_identity":
        overrides[1] = {"conceptrecid": None}
    elif fault == "edit_state":
        overrides[2] = {"state": "done"}
    elif fault == "refetch_state":
        overrides[4] = {"state": "done"}
    elif fault == "refetch_doi":
        overrides[4] = {"doi": "10.5072/zenodo.7"}
    else:
        overrides[4] = {"files": []}
    service = _Service(state="done", uploaded=True, response_overrides=overrides)
    with _loopback(service) as api:
        client = ZenodoClient(_TOKEN, api_base=api, timeout=3)
        # When: explicitly open/update the metadata-only edit.
        with pytest.raises(ZenodoError, match=finding) as caught:
            client.edit_published_metadata(7, {"title": "Edited draft", "doi": "10.5281/zenodo.7"})
        # Then: failure never registers an edit identity that can authorize publish.
        expected = {
            "source_identity": ["GET"],
            "edit_state": ["GET", "POST"],
            "refetch_state": ["GET", "POST", "PUT", "GET"],
            "refetch_doi": ["GET", "POST", "PUT", "GET"],
            "refetch_files": ["GET", "POST", "PUT", "GET"],
        }
        assert [r["method"] for r in service.requests] == expected[fault]
        count = len(service.requests)
        with pytest.raises(ZenodoError, match="opened and verified by this client"):
            client.publish_metadata_edit(7)
        assert len(service.requests) == count
        assert not any(str(r["path"]).endswith("/actions/publish") for r in service.requests)
        print("PHASE_FINDING " + json.dumps({"fault": fault, "finding": str(caught.value)}, sort_keys=True))


def test_real_http_metadata_edit_reentry_and_publication_consumes_authorization() -> None:
    # Given: a published PDF and an explicit metadata-only edit request.
    service = _Service(state="done", uploaded=True)
    request = {"title": "Edited draft", "doi": "10.5281/zenodo.7"}
    with _loopback(service) as api:
        client = ZenodoClient(_TOKEN, api_base=api, timeout=3)
        # When: open, re-enter, and explicitly publish the verified edit.
        first = client.edit_published_metadata(7, request)
        second = client.edit_published_metadata(7, request)
        published = client.publish_metadata_edit(7)
        # Then: re-entry uses the existing edit and publication consumes its authority.
        assert first == second
        assert first.state == "inprogress"
        assert published.state == "done"
        assert published.doi == "10.5281/zenodo.7"
        assert published.files == first.files == (
            ZenodoFile("fixture-file", "paper.pdf", len(_PDF), f"md5:{_MD5}"),
        )
        assert published.metadata.as_dict() == {
            "title": "Edited draft", "access_right": "open",
            "prereserve_doi": {"doi": "10.5281/zenodo.7", "recid": 7},
        }
        assert request == {"title": "Edited draft", "doi": "10.5281/zenodo.7"}
        assert [r["method"] for r in service.requests] == [
            "GET", "POST", "PUT", "GET", "GET", "PUT", "GET", "GET", "POST", "GET",
        ]
        actions = [r["path"] for r in service.requests if r["method"] == "POST"]
        assert actions == [
            "/api/deposit/depositions/7/actions/edit",
            "/api/deposit/depositions/7/actions/publish",
        ]
        with pytest.raises(ZenodoError, match="opened and verified by this client"):
            client.publish_metadata_edit(7)
        assert len(service.requests) == 10
        print("SUMMARY " + json.dumps(_summary(published), sort_keys=True))


@pytest.mark.parametrize(("fault", "finding"), [
    ("state", "only an open metadata edit can be published"),
    ("concept", "metadata-edit identity changed before publication"),
])
def test_real_http_metadata_edit_publish_rechecks_current_authority(
    fault: str, finding: str,
) -> None:
    # Given: an admitted edit with a server response that changes before publish.
    service = _Service(state="done", uploaded=True)
    with _loopback(service) as api:
        client = ZenodoClient(_TOKEN, api_base=api, timeout=3)
        client.edit_published_metadata(7, {"title": "Edited draft", "doi": "10.5281/zenodo.7"})
        service.response_overrides[5] = (
            {"state": "done"} if fault == "state"
            else {"conceptrecid": "9", "conceptdoi": "10.5281/zenodo.9"}
        )
        # When: explicitly request publication under the previously admitted identity.
        with pytest.raises(ZenodoError, match=finding) as caught:
            client.publish_metadata_edit(7)
        # Then: current authority is rechecked and no publication request is issued.
        assert [r["method"] for r in service.requests] == ["GET", "POST", "PUT", "GET", "GET"]
        assert not any(str(r["path"]).endswith("/actions/publish") for r in service.requests)
        print("PHASE_FINDING " + json.dumps({"fault": fault, "finding": str(caught.value)}, sort_keys=True))


@pytest.mark.parametrize(("fault", "finding"), [
    ("state", "metadata-edit publish response did not return a published record"),
    ("doi", "metadata-edit publish response changed the published DOI"),
    ("files", "metadata-edit publish response changed the verified file set"),
])
def test_real_http_metadata_edit_publication_response_is_not_trusted(
    fault: str, finding: str,
) -> None:
    # Given: verified metadata and bytes; the explicit publish response may drift.
    service = _Service(state="done", uploaded=True)
    with _loopback(service) as api:
        client = ZenodoClient(_TOKEN, api_base=api, timeout=3)
        client.edit_published_metadata(7, {"title": "Edited draft", "doi": "10.5281/zenodo.7"})
        if fault == "state":
            service.response_overrides[6] = {"state": "inprogress"}
        elif fault == "doi":
            service.response_overrides[6] = {"doi": "10.5072/zenodo.7"}
        else:
            service.response_overrides[6] = {"files": []}
        # When: explicitly publish only to the dummy loopback service.
        with pytest.raises(ZenodoError, match=finding) as caught:
            client.publish_metadata_edit(7)
        # Then: fail on the untrusted result; do not retry or claim rollback.
        assert [r["method"] for r in service.requests] == ["GET", "POST", "PUT", "GET", "GET", "POST"]
        assert sum(str(r["path"]).endswith("/actions/publish") for r in service.requests) == 1
        assert service.state == "done"
        print("PHASE_FINDING " + json.dumps({"fault": fault, "finding": str(caught.value)}, sort_keys=True))
