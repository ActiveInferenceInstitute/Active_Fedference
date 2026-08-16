from __future__ import annotations

import pytest

from publication.identifiers import doi_url, manuscript_pdf_filename, normalize_doi


def test_normalize_doi_accepts_common_resolver_forms() -> None:
    doi = "10.5281/zenodo.21864004"
    assert normalize_doi(doi) == doi
    assert normalize_doi(f"https://doi.org/{doi}") == doi
    assert normalize_doi(f"doi:{doi}") == doi
    assert doi_url(f"http://doi.org/{doi}") == f"https://doi.org/{doi}"


def test_normalize_doi_handles_explicit_unreleased_placeholder() -> None:
    assert normalize_doi("(forthcoming)", allow_placeholder=True) is None
    with pytest.raises(ValueError, match="not assigned"):
        normalize_doi("(forthcoming)")


def test_normalize_doi_rejects_malformed_identifiers() -> None:
    for value in ("https://example.com/paper", "10.12", "not-a-doi"):
        with pytest.raises(ValueError, match="invalid DOI"):
            normalize_doi(value)


def test_manuscript_pdf_filename_is_version_and_doi_bound() -> None:
    assert manuscript_pdf_filename("1.0.3", "10.5281/zenodo.21934992") == (
        "Active_Fedference_Research_Manuscript_v1.0.3_Zenodo_10.5281-zenodo.21934992.pdf"
    )
    assert manuscript_pdf_filename("1.0.3", "https://doi.org/10.5281/zenodo.21934992") == (
        "Active_Fedference_Research_Manuscript_v1.0.3_Zenodo_10.5281-zenodo.21934992.pdf"
    )
    with pytest.raises(ValueError, match="not assigned"):
        manuscript_pdf_filename("1.0.3", "forthcoming")
