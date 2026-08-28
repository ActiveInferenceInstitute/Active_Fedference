"""Canonical metadata emitter (MED-2): one config source, three file surfaces.

``manuscript/config.yaml`` (the ``paper:``/``authors:``/``publication:``/
``keywords:``/``metadata:`` blocks) plus the package version in
``pyproject.toml`` are the single source of truth for publication metadata.
This module deterministically emits the three generated surfaces —
``CITATION.cff``, ``.zenodo.json``, and ``codemeta.json`` — so a change made
once in config propagates everywhere and the surfaces can never silently
drift (the drift class this replaces was hand-maintenance of five copies).

Contract:

* :func:`build_metadata` — pure: returns the exact text content of each
  generated file, derived only from config + pyproject.
* :func:`check_metadata` — read-only: diff each on-disk surface against the
  emitted content; returns the list of drifted paths (empty == consistent).
* :func:`write_metadata` — the ONLY writer, and only when explicitly called
  (``scripts/emit_metadata.py --write``); check mode never mutates anything.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI lane
    import tomli as tomllib  # type: ignore[no-redef]

import yaml

from experiment_config import load_manuscript_config
from publication.identifiers import (
    doi_url,
    manuscript_pdf_filename,
    normalize_doi,
    publication_identity_sentence,
)

#: The generated surfaces, relative to the project root.
GENERATED_SURFACES: tuple[str, ...] = ("CITATION.cff", ".zenodo.json", "codemeta.json")

_PROJECT_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}(?:\.dev[0-9]+)?$")
_DEVELOPMENT_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}\.dev[0-9]+$")
_ISO_RELEASE_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_DEVELOPMENT_DOI_STATUS = "(forthcoming)"


@dataclass(frozen=True)
class PublicationLifecycle:
    """Validated identity shared by packaging and publication surfaces."""

    state: Literal["development", "final"]
    version: str
    doi: str | None
    date_released: str | None
    canonical_pdf: str | None


def _project_root(project_root: Path | None) -> Path:
    # metadata.py lives at <root>/src/publication/metadata.py.
    return project_root or Path(__file__).resolve().parent.parent.parent


def _load_config(root: Path) -> dict[str, Any]:
    config_path = root / "manuscript" / "config.yaml"
    if not config_path.exists():
        raise ValueError("config.yaml is missing")
    data = load_manuscript_config(root)
    publication = data.get("publication")
    if not isinstance(publication, dict) or not publication:
        raise ValueError("config.yaml publication block must be a non-empty mapping")
    paper = data.get("paper")
    if not isinstance(paper, dict) or not str(paper.get("title", "")).strip():
        raise ValueError("config.yaml paper block must contain a non-empty title")
    authors = data.get("authors")
    if (
        not isinstance(authors, list)
        or not authors
        or any(not isinstance(author, dict) for author in authors)
    ):
        raise ValueError("config.yaml authors must be a non-empty list of mappings")
    keywords = data.get("keywords", [])
    if not isinstance(keywords, list) or any(not isinstance(keyword, str) for keyword in keywords):
        raise ValueError("config.yaml keywords must be a list of strings")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("config.yaml metadata block must be a mapping")
    return data


def _toml_document(path: Path) -> dict[str, Any]:
    """Load a TOML document with one stable error boundary."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required TOML file is missing or unsafe: {path}")
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid TOML file {path}: {exc}") from exc
    if not isinstance(document, dict):  # pragma: no cover - tomllib always returns a dict
        raise ValueError(f"TOML root must be a mapping: {path}")
    return document


def _project_metadata(root: Path) -> dict[str, Any]:
    document = _toml_document(root / "pyproject.toml")
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml must contain a [project] table")
    return project


def package_version(root: Path) -> str:
    """Software version from pyproject.toml (the packaging source of truth)."""
    value = _project_metadata(root).get("version")
    if not isinstance(value, str) or not value:
        raise ValueError("pyproject.toml [project] has no version field")
    return value


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _validate_locked_version(root: Path, project: dict[str, Any], version: str) -> None:
    """Require the editable root entry in uv.lock to use the package version."""
    raw_name = project.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("pyproject.toml [project] has no name field")
    expected_name = _normalized_distribution_name(raw_name)
    lock = _toml_document(root / "uv.lock")
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock must contain package entries")
    editable_matches: list[dict[str, Any]] = []
    for item in packages:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        source = item.get("source")
        if (
            isinstance(name, str)
            and _normalized_distribution_name(name) == expected_name
            and isinstance(source, dict)
            and source.get("editable") == "."
        ):
            editable_matches.append(item)
    if len(editable_matches) != 1:
        raise ValueError(
            "uv.lock must contain exactly one editable root package entry for "
            f"{raw_name!r}"
        )
    locked_version = editable_matches[0].get("version")
    if locked_version != version:
        raise ValueError(
            "uv.lock root package version must exactly match pyproject.toml "
            f"({locked_version!r} != {version!r})"
        )


def _validate_project_doi_url(
    project: dict[str, Any],
    *,
    is_development: bool,
    doi: str | None,
) -> None:
    urls = project.get("urls", {})
    if not isinstance(urls, dict):
        raise ValueError("pyproject.toml [project.urls] must be a mapping")
    doi_keys = [key for key in urls if isinstance(key, str) and key.casefold() == "doi"]
    if is_development:
        if doi_keys:
            raise ValueError(
                "unreleased development metadata must not define a DOI project URL"
            )
        return
    if doi is None:  # validate_publication_identity reports this first
        raise ValueError("final release metadata requires an assigned DOI")
    if doi_keys != ["DOI"]:
        raise ValueError(
            "final release metadata requires exactly one [project.urls].DOI entry"
        )
    expected = doi_url(doi)
    if urls["DOI"] != expected:
        raise ValueError(
            "[project.urls].DOI must exactly match the assigned publication DOI "
            f"({urls['DOI']!r} != {expected!r})"
        )


def _one_line(text: str) -> str:
    """Collapse a folded YAML scalar to a single normalized line."""
    return " ".join(str(text).split())


_INLINE_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_METADATA_TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


def _publication_abstract(value: object, doi: str | None) -> str:
    """Return the reader-facing paper abstract for citation metadata.

    The paper source uses the same DOI tokens as the hydrated manuscript.  The
    generated metadata must resolve those tokens against the current DOI, and
    must store the rendered text rather than Markdown link/code delimiters.
    Keeping this normalization here makes the Zenodo ``description`` field
    semantically identical to the abstract printed in the paper while still
    allowing a future DOI reservation to flow through every surface.
    """
    text = str(value)
    replacements = {
        "{{PUBLICATION_IDENTITY_SENTENCE}}": publication_identity_sentence(doi),
        "{{PUBLICATION_DOI}}": doi or "N/A",
        "{{PUBLICATION_DOI_URL}}": doi_url(doi) or "N/A",
    }
    for token, replacement in replacements.items():
        text = text.replace(token, replacement)
    unresolved = sorted(set(_METADATA_TOKEN_RE.findall(text)))
    if unresolved:
        raise ValueError(
            "publication.abstract contains unresolved manuscript tokens: "
            + ", ".join(unresolved)
        )
    # The rendered paper displays the link label and code contents, not the
    # source-only Markdown delimiters.  Zenodo's abstract is plain metadata,
    # so store the same visible text there.
    text = _INLINE_LINK_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    return _one_line(text)


def _full_paper_title(paper: dict[str, Any]) -> str:
    """Return the reader-facing paper title used for a Zenodo deposition.

    The software citation title intentionally remains ``publication.software_name``.
    Zenodo stores the deposited research artifact, so its title must identify the
    complete manuscript rather than the short package name.  The subtitle is
    joined here instead of being duplicated in a second metadata-only config
    field, keeping the manuscript title block authoritative.
    """
    title = _one_line(str(paper["title"]))
    subtitle = _one_line(str(paper.get("subtitle", "")))
    return f"{title}: {subtitle}" if subtitle else title


def _author_email(author: dict[str, Any]) -> str | None:
    email = str(author.get("email", "")).strip()
    return email or None


def _release_date(value: object) -> str | None:
    """Return an ISO release date, or ``None`` for an unreleased project."""
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if not _ISO_RELEASE_DATE_RE.fullmatch(normalized):
        raise ValueError("publication.date_released must be YYYY-MM-DD or null")
    try:
        date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("publication.date_released must be YYYY-MM-DD or null") from exc
    return normalized


def validate_publication_identity(
    package_version: str,
    paper: dict[str, Any],
    publication: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Validate and return the release date and DOI for one lifecycle state.

    Active Fedference metadata has two fail-closed states. Development versions
    use a canonical ``X.Y.Z.devN`` identity, an exact empty DOI field, the plain
    ``(forthcoming)`` display status, and a null release date. Keeping status out
    of the DOI field prevents a renderer from fabricating a resolver link for an
    unassigned identifier. Final ``X.Y.Z`` versions have an assigned DOI/date
    and no development status. The manuscript and package versions must agree
    so a generated citation cannot describe different code.
    """
    paper_version = str(paper.get("version", "")).strip()
    if paper_version != package_version:
        raise ValueError(
            "paper.version must exactly match the pyproject.toml package version "
            f"({paper_version!r} != {package_version!r})"
        )
    if not _PROJECT_VERSION_RE.fullmatch(package_version):
        raise ValueError(
            "project version must be X.Y.Z or an unreleased X.Y.Z.devN version"
        )

    raw_doi = publication.get("doi")
    raw_doi_status = publication.get("doi_status")
    raw_date_released = publication.get("date_released")
    date_released = _release_date(raw_date_released)
    doi = normalize_doi(raw_doi, allow_placeholder=True)
    is_development = _DEVELOPMENT_VERSION_RE.fullmatch(package_version) is not None
    if is_development:
        if raw_doi != "":
            raise ValueError(
                "unreleased development metadata requires publication.doi to be "
                "the empty string; put (forthcoming) in publication.doi_status"
            )
        if raw_doi_status != _DEVELOPMENT_DOI_STATUS:
            raise ValueError(
                "unreleased development metadata requires publication.doi_status "
                f"to be exactly {_DEVELOPMENT_DOI_STATUS!r}"
            )
        if raw_date_released is not None:
            raise ValueError(
                "unreleased development metadata requires publication.date_released "
                "to be null"
            )
        assert doi is None and date_released is None
    elif doi is None or date_released is None:
        raise ValueError(
            "final release metadata requires an assigned DOI and date_released"
        )
    elif "doi_status" in publication:
        raise ValueError(
            "final release metadata must remove publication.doi_status after "
            "assigning the DOI"
        )
    return date_released, doi


def build_metadata(project_root: Path | None = None) -> dict[str, str]:
    """Return ``{relative_path: exact_file_content}`` for every surface."""
    root = _project_root(project_root)
    cfg = _load_config(root)
    pub = cfg["publication"]
    authors = cfg["authors"]
    paper = cfg["paper"]
    keywords = [str(k) for k in cfg.get("keywords", [])]
    metadata_cfg = cfg.get("metadata", {})
    license_id = str(metadata_cfg.get("license", "MIT"))
    access_right = str(metadata_cfg.get("access_right", "open"))
    project = _project_metadata(root)
    version_value = project.get("version")
    if not isinstance(version_value, str) or not version_value:
        raise ValueError("pyproject.toml [project] has no version field")
    version = version_value
    name = _one_line(pub["software_name"])
    description = _one_line(pub["description"])
    zenodo_title = _full_paper_title(paper)
    repo = str(pub["github_repository"])
    date_created = str(pub["date_created"])
    date_released, doi = validate_publication_identity(version, paper, pub)
    is_development = _DEVELOPMENT_VERSION_RE.fullmatch(version) is not None
    _validate_project_doi_url(project, is_development=is_development, doi=doi)
    _validate_locked_version(root, project, version)
    doi_resolver = doi_url(doi)
    abstract = _publication_abstract(pub["abstract"], doi)
    related = [
        {"relation": str(r["relation"]), "identifier": str(r["identifier"])}
        for r in pub.get("related_identifiers", [])
    ]

    cff: dict[str, Any] = {
        "cff-version": "1.2.0",
        "title": name,
        "type": "software",
        "message": ("If you use this software, please cite it using the metadata from this file."),
        "abstract": abstract,
        "version": version,
        "license": license_id,
        "repository-code": repo,
        "keywords": keywords,
        **({"identifiers": [{"type": "doi", "value": doi}]} if doi is not None else {}),
        **({"date-released": date_released} if date_released is not None else {}),
        "authors": [
            {
                "given-names": a["name"].rsplit(" ", 1)[0],
                "family-names": a["name"].rsplit(" ", 1)[1],
                "orcid": a["orcid"],
                "affiliation": a["affiliation"],
                **({"email": email} if (email := _author_email(a)) else {}),
            }
            for a in authors
        ],
    }
    cff_text = (
        "# GENERATED by src/publication/metadata.py from manuscript/config.yaml.\n"
        "# Do not hand-edit: run `uv run python scripts/emit_metadata.py --write`.\n"
        + yaml.safe_dump(cff, sort_keys=True, allow_unicode=True, default_flow_style=False)
    )

    zenodo: dict[str, Any] = {
        # Zenodo's API calls the UI's abstract field ``description``.  Keep it
        # equal to the full canonical abstract, not the short software
        # description used by the package metadata surfaces.
        "title": zenodo_title,
        "upload_type": "software",
        "description": abstract,
        "version": version,
        "license": license_id,
        "access_right": access_right,
        "keywords": keywords,
        **({"publication_date": date_released} if date_released is not None else {}),
        "creators": [
            {
                "name": "{}, {}".format(a["name"].rsplit(" ", 1)[1], a["name"].rsplit(" ", 1)[0]),
                "orcid": a["orcid"],
                "affiliation": a["affiliation"],
            }
            for a in authors
        ],
        "related_identifiers": related,
        **({"doi": doi} if doi is not None else {}),
    }
    zenodo_text = json.dumps(zenodo, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    codemeta: dict[str, Any] = {
        "@context": "https://doi.org/10.5063/schema/codemeta-2.0",
        "@type": "SoftwareSourceCode",
        "name": name,
        "description": description,
        "abstract": abstract,
        "version": version,
        "dateCreated": date_created,
        "codeRepository": repo,
        "license": f"https://spdx.org/licenses/{license_id}",
        "keywords": keywords,
        **({"dateModified": date_released} if date_released is not None else {}),
        "programmingLanguage": [{"@type": "ComputerLanguage", "name": "Python", "version": "3.10+"}],
        "author": [
            {
                "@id": f"https://orcid.org/{a['orcid']}",
                "@type": "Person",
                "givenName": a["name"].rsplit(" ", 1)[0],
                "familyName": a["name"].rsplit(" ", 1)[1],
                "affiliation": {"@type": "Organization", "name": a["affiliation"]},
                **({"email": email} if (email := _author_email(a)) else {}),
            }
            for a in authors
        ],
        **({"identifier": doi_resolver} if doi_resolver is not None else {}),
    }
    codemeta_text = json.dumps(codemeta, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    return {
        "CITATION.cff": cff_text,
        ".zenodo.json": zenodo_text,
        "codemeta.json": codemeta_text,
    }


def validate_publication_lifecycle(
    project_root: Path | None = None,
    *,
    require_generated_metadata: bool = True,
    require_canonical_pdf: bool = True,
) -> PublicationLifecycle:
    """Validate one complete development or final publication identity.

    The check reads real project/config/lock files.  In both states it binds
    ``pyproject.toml``, ``manuscript/config.yaml``, ``uv.lock``, and the three
    generated metadata surfaces.  A development identity intentionally has no
    canonical DOI-named PDF requirement; a final identity requires the exact
    version/DOI-derived top-level PDF.
    """
    root = _project_root(project_root).resolve()
    cfg = _load_config(root)
    paper = cfg["paper"]
    publication = cfg["publication"]
    project = _project_metadata(root)
    version_value = project.get("version")
    if not isinstance(version_value, str) or not version_value:
        raise ValueError("pyproject.toml [project] has no version field")
    version = version_value
    date_released, doi = validate_publication_identity(version, paper, publication)
    is_development = _DEVELOPMENT_VERSION_RE.fullmatch(version) is not None
    _validate_project_doi_url(project, is_development=is_development, doi=doi)
    _validate_locked_version(root, project, version)

    canonical_pdf = None if doi is None else manuscript_pdf_filename(version, doi)
    if require_generated_metadata:
        expected = build_metadata(root)
        drifted = [
            relative
            for relative, content in expected.items()
            if (
                not (path := root / relative).is_file()
                or path.is_symlink()
                or path.read_text(encoding="utf-8") != content
            )
        ]
        if drifted:
            raise ValueError(
                "generated publication metadata is stale: " + ", ".join(drifted)
            )
    if require_canonical_pdf and canonical_pdf is not None:
        pdf_path = root / canonical_pdf
        if not pdf_path.is_file() or pdf_path.is_symlink():
            raise ValueError(f"configured manuscript PDF is missing or unsafe: {canonical_pdf}")

    return PublicationLifecycle(
        state="development" if is_development else "final",
        version=version,
        doi=doi,
        date_released=date_released,
        canonical_pdf=canonical_pdf,
    )


def check_metadata(project_root: Path | None = None) -> list[str]:
    """Read-only drift check: return the surfaces whose on-disk content differs.

    A missing surface counts as drifted. An empty return means all generated
    surfaces are byte-identical to what the config would emit.
    """
    root = _project_root(project_root)
    expected = build_metadata(root)
    drifted: list[str] = []
    for rel, content in expected.items():
        path = root / rel
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            drifted.append(rel)
    return drifted


def write_metadata(project_root: Path | None = None) -> list[str]:
    """Write every generated surface; return the paths written (explicit only)."""
    root = _project_root(project_root)
    expected = build_metadata(root)
    written: list[str] = []
    for rel, content in expected.items():
        (root / rel).write_text(content, encoding="utf-8")
        written.append(rel)
    return written


__all__ = [
    "GENERATED_SURFACES",
    "PublicationLifecycle",
    "build_metadata",
    "check_metadata",
    "package_version",
    "validate_publication_identity",
    "validate_publication_lifecycle",
    "write_metadata",
]
