"""Bounded HTTP/JSON response admission without retained credential echoes."""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import HTTPError

from publication.zenodo_errors import ZenodoError, _ZenodoHTTPError

_ALREADY_EXISTS_RESPONSE = {"status": 400, "message": "A draft already exists."}


_HTTP_ERROR_BODY_LIMIT = 64 * 1024


_JSON_RESPONSE_BODY_LIMIT = 16 * 1024 * 1024


def _http_error_without_response_body(error: HTTPError) -> _ZenodoHTTPError:
    """Reduce an untrusted HTTP response to one safe, body-free exception.

    Reads are bounded and every ordinary read/parse failure degrades to a
    generic status-only error.  The helper returns normally, so its frame is
    absent from the traceback when the caller raises the returned exception;
    its ``finally`` block also scrubs untrusted bytes if a process-level
    interruption escapes.  No server-supplied diagnostic text is retained.
    """
    status = int(error.code)
    error_bytes = b""
    parsed_error: object | None = None
    exact_already_exists = False
    try:
        error_bytes = error.read(_HTTP_ERROR_BODY_LIMIT + 1)
        if len(error_bytes) <= _HTTP_ERROR_BODY_LIMIT:
            parsed_error = json.loads(error_bytes)
            exact_already_exists = (
                status == 400
                and isinstance(parsed_error, Mapping)
                and dict(parsed_error) == _ALREADY_EXISTS_RESPONSE
            )
    except Exception:
        exact_already_exists = False
    finally:
        try:
            try:
                error.close()
            except Exception:
                pass
        finally:
            del error, error_bytes, parsed_error
    return _ZenodoHTTPError(
        f"Zenodo HTTP {status}",
        status=status,
        exact_already_exists=exact_already_exists,
    )


def _json_contains_token_echo(value: object, token: str) -> bool:
    """Return whether a JSON key or value contains the bearer token."""
    if isinstance(value, str):
        return token in value
    if isinstance(value, list):
        return any(_json_contains_token_echo(item, token) for item in value)
    if isinstance(value, dict):
        return any(
            (isinstance(key, str) and token in key)
            or _json_contains_token_echo(item, token)
            for key, item in value.items()
        )
    return False


def _json_response_without_response_body(
    response: Any,
    token: str,
) -> tuple[object | None, ZenodoError | None]:
    """Return parsed bounded JSON or a generic body-free failure.

    All server-controlled reads and parsing occur in a helper that returns
    normally.  Consequently, malformed, truncated, invalid-encoding, deeply
    nested, and oversized responses cannot attach their bytes to the later
    public exception traceback.
    """
    raw = b""
    parsed: object | None = None
    try:
        raw = response.read(_JSON_RESPONSE_BODY_LIMIT + 1)
        if not isinstance(raw, bytes) or len(raw) > _JSON_RESPONSE_BODY_LIMIT:
            return None, ZenodoError("Zenodo returned malformed or oversized JSON content")
        if not raw:
            return {}, None
        parsed = json.loads(raw)
        if _json_contains_token_echo(parsed, token):
            return None, ZenodoError("Zenodo response echoed bearer credential")
        return parsed, None
    except Exception:
        return None, ZenodoError("Zenodo returned malformed or oversized JSON content")
    finally:
        del response, raw, parsed, token


def _is_existing_draft_response(error: _ZenodoHTTPError) -> bool:
    """Recognize only Zenodo's exact production already-exists response."""
    return error.status == 400 and error._exact_already_exists
