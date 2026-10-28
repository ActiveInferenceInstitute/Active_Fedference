"""Bearer HTTP transport with endpoint admission and unconditional redirect refusal."""

from __future__ import annotations

import json
import math
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from publication.zenodo_admission import _integer_id
from publication.zenodo_errors import ZenodoError, _ZenodoHTTPError
from publication.zenodo_response import (
    _http_error_without_response_body,
    _json_response_without_response_body,
)

DEFAULT_ZENODO_API = "https://zenodo.org/api"


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


class ZenodoTransport:
    """HTTP/auth safety shared by the explicit deposition phases."""

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
        sanitized_transport_error: ZenodoError | None = None
        response_payload: object | None = None
        response_error: ZenodoError | None = None
        response: Any | None = None
        try:
            with self._opener.open(request, timeout=self._timeout) as response:  # noqa: S310
                response_payload, response_error = _json_response_without_response_body(
                    response,
                    self._token,
                )
        except HTTPError as exc:
            try:
                sanitized_http_error = _http_error_without_response_body(exc)
            finally:
                headers.clear()
                del request, request_body, body, payload, headers, content_type
        except URLError:
            try:
                sanitized_transport_error = ZenodoError("Zenodo request failed")
            finally:
                headers.clear()
                del request, request_body, body, payload, headers, content_type
        if sanitized_http_error is not None:
            # Raise only after leaving the HTTPError handler.  Raising inside
            # that block—even with ``from None``—would retain the original
            # response object through ``__context__`` and could preserve a
            # credential echoed by an untrusted server.  Request/header/body
            # locals were removed unconditionally in the handler because
            # exception tracebacks retain the raising frame.
            raise sanitized_http_error
        if sanitized_transport_error is not None:
            raise sanitized_transport_error
        if response_error is not None or response_payload is None:
            safe_response_error = response_error or ZenodoError(
                "Zenodo returned malformed or oversized JSON content"
            )
            headers.clear()
            del request, request_body, body, payload, headers, content_type
            del response, response_payload
            raise safe_response_error
        return response_payload

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
