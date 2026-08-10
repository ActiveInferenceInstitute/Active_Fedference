"""Canonical metadata emitter tests (MED-2) — no mocks, real files.

The emitter is the single writer of CITATION.cff/.zenodo.json/codemeta.json;
these tests pin (a) purity and cross-surface consistency, (b) idempotency,
(c) drift DETECTION (proof-of-detection: a tampered surface must be flagged),
and (d) that the real repository state is emitter-consistent, so hand-edits
to generated files can never land silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from publication.metadata import (
    GENERATED_SURFACES,
    build_metadata,
    check_metadata,
    write_metadata,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_project(tmp_path: Path) -> Path:
    (tmp_path / "manuscript").mkdir()
    config = {
        "paper": {"title": "T", "version": "1.0"},
        "authors": [
            {
                "name": "Ada Q Lovelace",
                "orcid": "0000-0000-0000-0001",
                "affiliation": "Analytical Engine Society",
                "email": "ada@example.com",
            }
        ],
        "publication": {
            "doi": "(forthcoming)",
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
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "proj"\nversion = "9.9.9"\n', encoding="utf-8"
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
    # Same title/author/orcid/license on every surface.
    assert cff["title"] == zen["title"] == cm["name"] == "Proj"
    assert cff["authors"][0]["orcid"] == zen["creators"][0]["orcid"]
    assert cm["author"][0]["@id"].endswith(zen["creators"][0]["orcid"])
    assert cff["license"] == zen["license"] == "MIT"
    assert cm["license"].endswith("/MIT")
    # Name split: everything before the last space is given names.
    assert cff["authors"][0]["given-names"] == "Ada Q"
    assert cff["authors"][0]["family-names"] == "Lovelace"
    assert zen["creators"][0]["name"] == "Lovelace, Ada Q"
    # Folded abstract collapses to one normalized line.
    assert cff["abstract"] == "Line one folds into one line."
    # Related identifiers pass through with the stated direction.
    assert zen["related_identifiers"] == [{"relation": "cites", "identifier": "https://doi.org/10.1/x"}]


def test_unreleased_metadata_omits_release_date_claims(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["publication"]["date_released"] = None
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = build_metadata(root)
    cff = yaml.safe_load(out["CITATION.cff"])
    zen = json.loads(out[".zenodo.json"])
    codemeta = json.loads(out["codemeta.json"])
    assert "date-released" not in cff
    assert "publication_date" not in zen
    assert "dateModified" not in codemeta


def test_assigned_doi_is_emitted_on_all_surfaces(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["publication"]["doi"] = "https://doi.org/10.5281/zenodo.12345"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    out = build_metadata(root)
    cff = yaml.safe_load(out["CITATION.cff"])
    zen = json.loads(out[".zenodo.json"])
    codemeta = json.loads(out["codemeta.json"])
    assert cff["identifiers"] == [{"type": "doi", "value": "10.5281/zenodo.12345"}]
    assert zen["doi"] == "10.5281/zenodo.12345"
    assert codemeta["identifier"] == "https://doi.org/10.5281/zenodo.12345"


def test_invalid_release_date_fails_loudly(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    config_path = root / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["publication"]["date_released"] = "not-a-date"
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


def test_real_repository_surfaces_are_emitter_consistent() -> None:
    """The shipped CITATION.cff/.zenodo.json/codemeta.json must be exactly what
    the config emits — hand-edits to generated surfaces cannot land silently."""
    assert check_metadata(_PROJECT_ROOT) == []
