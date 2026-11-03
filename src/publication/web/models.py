"""Models for source-preserving HTML publication."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _HTMLAttribute:
    """One source-preserving start-tag attribute."""

    name: str
    value: str | None
    value_start: int | None
    value_end: int | None
    quote: str | None


@dataclass(frozen=True)
class _HTMLToken:
    """One source-preserving HTML token used by normalization and validation."""

    kind: str
    raw: str
    start: int
    tag_name: str = ""
    attributes: tuple[_HTMLAttribute, ...] = ()
    self_closing: bool = False


@dataclass(frozen=True)
class _HTMLAttributeOccurrence:
    """One decoded attribute value with its owning element and source offset."""

    tag_name: str
    attribute_name: str
    value: str
    start: int
    attributes: tuple[_HTMLAttribute, ...]


@dataclass(frozen=True)
class _HTMLDocument:
    """One path-confined HTML source captured before any normalization write."""

    path: Path
    original: bytes
    text: str
    tokens: tuple[_HTMLToken, ...]
    source_stat: os.stat_result


@dataclass(frozen=True)
class _StagedHTMLWrite:
    """Same-directory replacement and rollback files for one HTML document."""

    document: _HTMLDocument
    replacement_path: Path
    replacement_bytes: bytes
    replacement_stat: os.stat_result
    rollback_path: Path


class _HTMLTokenizationError(ValueError):
    """Raised when a protected HTML context is not safely bounded."""


class _HTMLRollbackError(RuntimeError):
    """Raised when a failed batch leaves durable recovery files behind."""

    def __init__(self, message: str, retained_paths: frozenset[Path]) -> None:
        super().__init__(message)
        self.retained_paths = retained_paths


@dataclass(frozen=True)
class WebPackageValidation:
    """Asset, reference, markup, and accessibility result for generated HTML."""

    html_files: int
    assets_checked: int
    missing_assets: tuple[str, ...]
    raw_xrefs: tuple[str, ...]
    broken_xrefs: tuple[str, ...] = ()
    malformed_markup: tuple[str, ...] = ()
    accessibility_issues: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether HTML exists and every package/accessibility check passes."""
        return (
            not self.missing_assets
            and not self.raw_xrefs
            and not self.broken_xrefs
            and not self.malformed_markup
            and not self.accessibility_issues
            and self.html_files > 0
        )
