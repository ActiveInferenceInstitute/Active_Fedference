"""Pure helpers for validating and rendering persistent identifiers.

Publication metadata, manuscript tokens, and the Zenodo adapter all consume
the same DOI contract.  Keeping normalization here prevents one surface from
silently accepting a DOI URL while another emits a second, malformed value.
"""

from __future__ import annotations

import re

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]*$")
_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "doi:",
)
_PLACEHOLDERS = {"", "(forthcoming)", "forthcoming", "none", "null"}


def normalize_doi(value: object, *, allow_placeholder: bool = False) -> str | None:
    """Return a bare DOI or ``None`` for an explicit unreleased placeholder.

    ``allow_placeholder`` is useful for pre-release metadata fixtures.  Any
    non-placeholder value is validated strictly so a manuscript cannot render
    a DOI-shaped typo or an arbitrary URL as publication metadata.
    """
    if value is None:
        return None
    normalized = str(value).strip()
    lowered = normalized.casefold()
    if lowered in _PLACEHOLDERS:
        if allow_placeholder:
            return None
        raise ValueError("publication DOI is not assigned")
    for prefix in _DOI_PREFIXES:
        if lowered.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    if not _DOI_RE.fullmatch(normalized):
        raise ValueError(f"invalid DOI: {value!r}")
    return normalized


def doi_url(value: object, *, allow_placeholder: bool = False) -> str | None:
    """Return the canonical resolver URL for a DOI value."""
    normalized = normalize_doi(value, allow_placeholder=allow_placeholder)
    return f"https://doi.org/{normalized}" if normalized is not None else None


def manuscript_pdf_filename(version: object, doi: object) -> str:
    """Return the canonical informative top-level manuscript PDF filename.

    The release identity is derived from the package version and the reserved
    bare DOI, so clean-checkout and README/release tooling do not need a
    version-specific hard-coded filename. Characters that are unsafe or
    ambiguous in a filename are replaced deterministically in the DOI slug.
    """
    normalized_version = str(version).strip()
    if not _VERSION_RE.fullmatch(normalized_version):
        raise ValueError(f"invalid package version for manuscript filename: {version!r}")
    normalized_doi = normalize_doi(doi)
    if normalized_doi is None:
        raise ValueError("manuscript PDF filename requires an assigned DOI")
    doi_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", normalized_doi.replace("/", "-"))
    return f"Active_Fedference_Research_Manuscript_v{normalized_version}_Zenodo_{doi_slug}.pdf"


__all__ = ["doi_url", "manuscript_pdf_filename", "normalize_doi"]
