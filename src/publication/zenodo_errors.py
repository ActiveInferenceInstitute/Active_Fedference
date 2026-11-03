"""Canonical, body-free errors for the Zenodo boundary."""

from __future__ import annotations


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
