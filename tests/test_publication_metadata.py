"""Canonical metadata emitter tests (MED-2) — no mocks, real files.

The emitter is the single writer of CITATION.cff/.zenodo.json/codemeta.json;
these tests pin (a) purity and cross-surface consistency, (b) idempotency,
(c) drift DETECTION (proof-of-detection: a tampered surface must be flagged),
and (d) that the real repository state is emitter-consistent, so hand-edits
to generated files can never land silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from publication.identifiers import normalize_doi, publication_identity_sentence
from publication.metadata import (
    GENERATED_SURFACES,
    build_metadata,
    check_metadata,
    validate_publication_lifecycle,
    write_metadata,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_packaging_identity(
    root: Path,
    *,
    version: str,
    doi: str | None,
) -> None:
    doi_url = f'\nDOI = "https://doi.org/{doi}"' if doi is not None else ""
    (root / "pyproject.toml").write_text(
        '[project]\nname = "proj"\n'
        f'version = "{version}"\n'
        f"[project.urls]{doi_url}\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        "version = 1\n"
        "[[package]]\n"
        'name = "proj"\n'
        f'version = "{version}"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )


def _make_project(tmp_path: Path) -> Path:
    (tmp_path / "manuscript").mkdir()
    config = {
        "paper": {
            "title": "T",
            "subtitle": "A complete subtitle",
            "version": "9.9.9",
            "date": "2026-02-02",
        },
        "authors": [
            {
                "name": "Ada Q Lovelace",
                "orcid": "0000-0000-0000-0001",
                "affiliation": "Analytical Engine Society",
                "email": "ada@example.com",
            }
        ],
        "publication": {
            "doi": "10.5281/zenodo.12345",
            "github_repository": "https://github.com/example/proj",
            "year": "2026",
            "software_name": "Proj",
            "date_created": "2026-01-01",
            "date_released": "2026-02-02",
            "abstract": "Line one\nfolds  into one   line.",
            "description": "Short description.",
            "github_description": "Shorter still.",
            "related_identifiers": [{"relation": "cites", "identifier": "https://doi.org/10.1/x"}],
        },
        "keywords": ["alpha", "beta"],
        "metadata": {"license": "MIT"},
    }
    (tmp_path / "manuscript" / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_packaging_identity(
        tmp_path,
        version="9.9.9",
        doi="10.5281/zenodo.12345",
    )
    return tmp_path


def test_build_emits_all_three_surfaces_from_config(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    out = build_metadata(root)
    assert set(out) == set(GENERATED_SURFACES)
    cff = yaml.safe_load(out["CITATION.cff"])
    zen = json.loads(out[".zenodo.json"])
    cm = json.loads(out["codemeta.json"])
    # Version comes from pyproject (the packaging source of truth), everywhere.
    assert cff["version"] == zen["version"] == cm["version"] == "9.9.9"
    # Software citation surfaces use the package name; Zenodo identifies the
    # deposited manuscript with its complete paper title.
    assert cff["title"] == cm["name"] == "Proj"
    assert zen["title"] == "T: A complete subtitle"
    assert cff["authors"][0]["orcid"] == zen["creators"][0]["orcid"]
    assert cm["author"][0]["@id"].endswith(zen["creators"][0]["orcid"])
    assert cff["license"] == zen["license"] == "MIT"
    assert zen["access_right"] == "open"
    assert cm["license"].endswith("/MIT")
    # Name split: everything before the last space is given names.
    assert cff["authors"][0]["given-names"] == "Ada Q"
    assert cff["authors"][0]["family-names"] == "Lovelace"
    assert zen["creators"][0]["name"] == "Lovelace, Ada Q"
    # Folded abstract collapses to one normalized line.
    assert cff["abstract"] == "Line one folds into one line."
    assert zen["description"] == cff["abstract"]
    assert cm["abstract"] == cff["abstract"]
    # Related identifiers pass through with the stated direction.
    assert zen["related_identifiers"] == [{"relation": "cites", "identifier": "https://doi.org/10.1/x"}]


def test_unreleased_metadata_omits_release_date_claims(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paper"]["version"] = "9.9.9.dev0"
    config["publication"]["doi"] = ""
    config["publication"]["doi_status"] = "(forthcoming)"
    config["publication"]["date_released"] = None
    config["publication"]["abstract"] = "Lead. {{PUBLICATION_IDENTITY_SENTENCE}}"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_packaging_identity(root, version="9.9.9.dev0", doi=None)
    out = build_metadata(root)
    cff = yaml.safe_load(out["CITATION.cff"])
    zen = json.loads(out[".zenodo.json"])
    codemeta = json.loads(out["codemeta.json"])
    assert "date-released" not in cff
    assert "publication_date" not in zen
    assert "dateModified" not in codemeta
    assert "identifiers" not in cff
    assert "doi" not in zen
    assert "identifier" not in codemeta
    assert cff["abstract"] == f"Lead. {publication_identity_sentence(None)}"
    assert "N/A" not in cff["abstract"]
    assert zen["description"] == codemeta["abstract"] == cff["abstract"]


def test_complete_development_lifecycle_uses_real_emitted_surfaces(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paper"]["version"] = "9.9.9.dev0"
    config["publication"].update(
        {"doi": "", "doi_status": "(forthcoming)", "date_released": None}
    )
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_packaging_identity(root, version="9.9.9.dev0", doi=None)

    assert write_metadata(root) == list(GENERATED_SURFACES)
    lifecycle = validate_publication_lifecycle(root)

    assert lifecycle.state == "development"
    assert lifecycle.version == "9.9.9.dev0"
    assert lifecycle.doi is None
    assert lifecycle.date_released is None
    assert lifecycle.canonical_pdf is None


def test_complete_final_lifecycle_uses_real_emitted_surfaces_and_pdf(
    tmp_path: Path,
) -> None:
    root = _make_project(tmp_path)
    assert write_metadata(root) == list(GENERATED_SURFACES)
    expected_pdf = (
        root
        / "Active_Fedference_Research_Manuscript_v9.9.9_"
        "Zenodo_10.5281-zenodo.12345.pdf"
    )
    expected_pdf.write_bytes(b"final lifecycle PDF fixture\n")

    lifecycle = validate_publication_lifecycle(root)

    assert lifecycle.state == "final"
    assert lifecycle.version == "9.9.9"
    assert lifecycle.doi == "10.5281/zenodo.12345"
    assert lifecycle.date_released == "2026-02-02"
    assert lifecycle.canonical_pdf == expected_pdf.name


@pytest.mark.parametrize("paper_date", [None, "", "2026-02-01", "2026-02-30"])
def test_final_lifecycle_requires_exact_manuscript_release_date(
    tmp_path: Path,
    paper_date: str | None,
) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if paper_date is None:
        config["paper"].pop("date")
    else:
        config["paper"]["date"] = paper_date
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    expected = (
        "paper.date must be YYYY-MM-DD or empty"
        if paper_date == "2026-02-30"
        else "paper.date to exactly match publication.date_released"
    )
    with pytest.raises(ValueError, match=expected):
        validate_publication_lifecycle(root, require_canonical_pdf=False)


def test_development_and_final_project_doi_url_rules_are_fail_closed(
    tmp_path: Path,
) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paper"]["version"] = "9.9.9.dev0"
    config["publication"].update(
        {"doi": "", "doi_status": "(forthcoming)", "date_released": None}
    )
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_packaging_identity(
        root,
        version="9.9.9.dev0",
        doi="10.5281/zenodo.12345",
    )
    with pytest.raises(ValueError, match="must not define a DOI project URL"):
        build_metadata(root)

    config["paper"]["version"] = "9.9.9"
    config["publication"].update(
        {"doi": "10.5281/zenodo.12345", "date_released": "2026-02-02"}
    )
    config["publication"].pop("doi_status", None)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_packaging_identity(root, version="9.9.9", doi=None)
    with pytest.raises(ValueError, match=r"requires exactly one \[project.urls\]\.DOI"):
        build_metadata(root)


def test_lifecycle_rejects_lock_project_url_surface_and_pdf_drift(
    tmp_path: Path,
) -> None:
    root = _make_project(tmp_path)
    write_metadata(root)

    (root / "uv.lock").write_text(
        "version = 1\n"
        "[[package]]\n"
        'name = "proj"\n'
        'version = "9.9.8"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="uv.lock root package version"):
        validate_publication_lifecycle(root)

    _write_packaging_identity(root, version="9.9.9", doi="10.5281/zenodo.99999")
    with pytest.raises(ValueError, match=r"\[project.urls\]\.DOI"):
        validate_publication_lifecycle(root)

    _write_packaging_identity(root, version="9.9.9", doi="10.5281/zenodo.12345")
    (root / "CITATION.cff").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generated publication metadata is stale"):
        validate_publication_lifecycle(root)

    write_metadata(root)
    with pytest.raises(ValueError, match="configured manuscript PDF is missing"):
        validate_publication_lifecycle(root)


def test_unreleased_metadata_rejects_placeholder_in_doi_field(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paper"]["version"] = "9.9.9.dev0"
    config["publication"]["doi"] = "(forthcoming)"
    config["publication"]["doi_status"] = "(forthcoming)"
    config["publication"]["date_released"] = None
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_packaging_identity(root, version="9.9.9.dev0", doi=None)

    with pytest.raises(ValueError, match="publication.doi to be the empty string"):
        build_metadata(root)


@pytest.mark.parametrize("doi_status", [None, "forthcoming", " (forthcoming) "])
def test_unreleased_metadata_requires_exact_plain_doi_status(
    tmp_path: Path,
    doi_status: str | None,
) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paper"]["version"] = "9.9.9.dev0"
    config["publication"]["doi"] = ""
    config["publication"]["doi_status"] = doi_status
    config["publication"]["date_released"] = None
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_packaging_identity(root, version="9.9.9.dev0", doi=None)

    with pytest.raises(ValueError, match="doi_status to be exactly"):
        build_metadata(root)


def test_unreleased_metadata_requires_literal_null_release_date(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paper"]["version"] = "9.9.9.dev0"
    config["publication"]["doi"] = ""
    config["publication"]["doi_status"] = "(forthcoming)"
    config["publication"]["date_released"] = ""
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_packaging_identity(root, version="9.9.9.dev0", doi=None)

    with pytest.raises(ValueError, match="date_released to be null"):
        build_metadata(root)


def test_development_metadata_rejects_release_identity(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paper"]["version"] = "9.9.9.dev0"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_packaging_identity(root, version="9.9.9.dev0", doi=None)

    try:
        build_metadata(root)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "development metadata" in str(exc)


def test_final_metadata_rejects_missing_release_identity(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["publication"]["doi"] = ""
    config["publication"]["doi_status"] = "(forthcoming)"
    config["publication"]["date_released"] = None
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    try:
        build_metadata(root)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "final release metadata" in str(exc)


@pytest.mark.parametrize("doi_status", (None, "", "(forthcoming)"))
def test_final_metadata_requires_doi_status_key_to_be_absent(
    tmp_path: Path,
    doi_status: str | None,
) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["publication"]["doi_status"] = doi_status
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="must remove publication.doi_status"):
        build_metadata(root)


def test_metadata_rejects_package_and_manuscript_version_drift(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paper"]["version"] = "9.9.8"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    try:
        build_metadata(root)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "paper.version must exactly match" in str(exc)


def test_assigned_doi_is_emitted_on_all_surfaces(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["publication"]["doi"] = "https://doi.org/10.5281/zenodo.12345"
    config["publication"]["abstract"] = "Lead. {{PUBLICATION_IDENTITY_SENTENCE}}"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    out = build_metadata(root)
    cff = yaml.safe_load(out["CITATION.cff"])
    zen = json.loads(out[".zenodo.json"])
    codemeta = json.loads(out["codemeta.json"])
    assert cff["identifiers"] == [{"type": "doi", "value": "10.5281/zenodo.12345"}]
    assert zen["doi"] == "10.5281/zenodo.12345"
    assert codemeta["identifier"] == "https://doi.org/10.5281/zenodo.12345"
    expected_identity = publication_identity_sentence("10.5281/zenodo.12345")
    expected_identity = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", expected_identity)
    assert cff["abstract"] == f"Lead. {expected_identity}"
    assert zen["description"] == codemeta["abstract"] == cff["abstract"]
    assert "development manuscript" not in cff["abstract"]


@pytest.mark.parametrize(
    "invalid_date",
    (
        "not-a-date",
        "20260202",
        "2026-W35-4",
        "２０２６-０８-２７",
        "2026-2-2",
    ),
)
def test_invalid_release_date_fails_loudly(
    tmp_path: Path,
    invalid_date: str,
) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["publication"]["date_released"] = invalid_date
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    try:
        build_metadata(root)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "date_released must be YYYY-MM-DD or null" in str(exc)


def test_write_then_check_is_consistent_and_idempotent(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    assert sorted(check_metadata(root)) == sorted(GENERATED_SURFACES)  # missing = drift
    written = write_metadata(root)
    assert sorted(written) == sorted(GENERATED_SURFACES)
    assert check_metadata(root) == []
    first = {rel: (root / rel).read_text() for rel in GENERATED_SURFACES}
    write_metadata(root)
    assert {rel: (root / rel).read_text() for rel in GENERATED_SURFACES} == first


def test_check_detects_a_tampered_surface(tmp_path: Path) -> None:
    """Proof-of-detection: a hand-edit to any generated file must be flagged."""
    root = _make_project(tmp_path)
    write_metadata(root)
    path = root / ".zenodo.json"
    data = json.loads(path.read_text())
    data["version"] = "0.0.0-tampered"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    assert check_metadata(root) == [".zenodo.json"]


def test_zenodo_uses_full_paper_identity_and_abstract(tmp_path: Path) -> None:
    """Zenodo must not regress to the abbreviated package metadata fields."""
    root = _make_project(tmp_path)
    config = yaml.safe_load((root / "manuscript" / "config.yaml").read_text(encoding="utf-8"))
    output = build_metadata(root)
    zenodo = json.loads(output[".zenodo.json"])
    abstract = " ".join(str(config["publication"]["abstract"]).split())
    assert zenodo["title"] == "T: A complete subtitle"
    assert zenodo["description"] == abstract
    assert zenodo["description"] != "Short description."


def test_missing_publication_block_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "config.yaml").write_text(
        yaml.safe_dump({"paper": {"title": "T"}}), encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text('version = "1.0.0"\n', encoding="utf-8")
    try:
        build_metadata(tmp_path)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "publication" in str(exc)


def test_missing_paper_title_fails_loudly(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paper"]["title"] = ""
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    try:
        build_metadata(root)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "paper block must contain a non-empty title" in str(exc)


def test_malformed_experiment_block_uses_the_shared_config_boundary(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["experiment"] = []
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    try:
        build_metadata(root)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "experiment block must be a mapping" in str(exc)


def test_real_repository_surfaces_are_emitter_consistent() -> None:
    """The shipped CITATION.cff/.zenodo.json/codemeta.json must be exactly what
    the config emits — hand-edits to generated surfaces cannot land silently."""
    assert check_metadata(_PROJECT_ROOT) == []


def test_zenodo_description_is_the_hydrated_manuscript_abstract() -> None:
    """The deposited description must be the paper abstract, not a package blurb."""
    config = yaml.safe_load(
        (_PROJECT_ROOT / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    )
    assigned_doi = normalize_doi(
        config["publication"].get("doi"),
        allow_placeholder=True,
    )
    doi_token = assigned_doi or "N/A"
    source = (_PROJECT_ROOT / "manuscript" / "00_abstract.md").read_text(encoding="utf-8")
    abstract = source.split("# Abstract", 1)[1].split("**Keywords:**", 1)[0]
    abstract = re.sub(r"\s*\{#[^}]+\}", "", abstract, count=1)
    abstract = abstract.replace(
        "{{PUBLICATION_IDENTITY_SENTENCE}}",
        publication_identity_sentence(assigned_doi),
    )
    abstract = abstract.replace("{{PUBLICATION_DOI}}", doi_token)
    abstract = abstract.replace(
        "{{PUBLICATION_DOI_URL}}",
        f"https://doi.org/{doi_token}",
    )
    abstract = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", abstract)
    abstract = re.sub(r"`([^`]+)`", r"\1", abstract)
    expected = " ".join(abstract.split())

    zenodo = json.loads(build_metadata(_PROJECT_ROOT)[".zenodo.json"])
    assert zenodo["description"] == expected
    assert "Keywords:" not in zenodo["description"]
    assert "tested, reproducible research package" not in zenodo["description"]
    assert "{{PUBLICATION_" not in zenodo["description"]
    assert "[@" not in zenodo["description"]


def test_repository_metadata_does_not_expose_manuscript_cross_reference_syntax() -> None:
    """Standalone discovery metadata must remain readable outside Pandoc."""

    emitted = build_metadata(_PROJECT_ROOT)
    for relative_path in ("CITATION.cff", ".zenodo.json", "codemeta.json"):
        assert "[@" not in emitted[relative_path], relative_path


def test_real_abstract_hypothetical_final_identity_never_retains_development_prose(
    tmp_path: Path,
) -> None:
    """Exercise the shipped abstract/config through the future final lifecycle."""
    doi = "10.5281/zenodo.99999999"
    config = yaml.safe_load(
        (_PROJECT_ROOT / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    )
    config["paper"]["version"] = "1.1.0"
    config["paper"]["date"] = "2026-09-01"
    config["publication"]["doi"] = doi
    config["publication"].pop("doi_status", None)
    config["publication"]["date_released"] = "2026-09-01"
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    _write_packaging_identity(
        tmp_path,
        version="1.1.0",
        doi=doi,
    )

    final_metadata = json.loads(build_metadata(tmp_path)[".zenodo.json"])
    source = (_PROJECT_ROOT / "manuscript" / "00_abstract.md").read_text(encoding="utf-8")
    source_abstract = source.split("# Abstract", 1)[1].split("**Keywords:**", 1)[0]
    source_abstract = re.sub(r"\s*\{#[^}]+\}", "", source_abstract, count=1)
    source_abstract = source_abstract.replace(
        "{{PUBLICATION_IDENTITY_SENTENCE}}",
        publication_identity_sentence(doi),
    )
    source_abstract = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", source_abstract)
    source_abstract = re.sub(r"`([^`]+)`", r"\1", source_abstract)

    assert final_metadata["description"] == " ".join(source_abstract.split())
    assert "This final-release manuscript" in final_metadata["description"]
    assert doi in final_metadata["description"]
    assert "development manuscript" not in final_metadata["description"]
