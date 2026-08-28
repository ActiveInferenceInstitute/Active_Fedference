"""Tests for the clean-checkout and clone-correctness probe."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from publication.clean_checkout import (
    HISTORICAL_RELEASE_PDF_LEDGER_PATH,
    IMMUTABLE_RELEASE_PDFS,
    REQUIRED_TRACKED_PATHS,
    historical_release_pdf_findings,
    inspect_clean_checkout,
)
from publication.metadata import write_metadata

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HERMETIC_GIT = ("git", "-c", "core.fsmonitor=false", "-c", "core.untrackedcache=false")
GENERATED_METADATA_PATHS = ("CITATION.cff", ".zenodo.json", "codemeta.json")


def test_validation_receipt_chain_is_required_for_clean_checkout() -> None:
    """Keep final-hydration evidence present in every clone-correct candidate."""
    assert {
        "LICENSE",
        "src/publication/validation_receipt.py",
        "scripts/validate_test_coverage.py",
        "tests/test_validation_receipt.py",
        "output/data/analysis_execution.json",
        "output/data/test_coverage_receipt.json",
        "src/fedference_cli/README.md",
        "src/fedference_cli/_commands.py",
        "src/fedference_cli/_parser.py",
        "src/fedference_cli/_support.py",
    } <= set(REQUIRED_TRACKED_PATHS)


def test_all_immutable_release_pdfs_are_required_for_clean_checkout() -> None:
    assert IMMUTABLE_RELEASE_PDFS == (
        "Active_Fedference_Research_Manuscript_v0.1.0_Zenodo_10.5281-zenodo.21864004.pdf",
        "Active_Fedference_Research_Manuscript_v1.0.1_Zenodo_10.5281-zenodo.21919307.pdf",
        "Active_Fedference_Research_Manuscript_v1.0.2_Zenodo_10.5281-zenodo.21934992.pdf",
        "Active_Fedference_Research_Manuscript_v1.0.3_Zenodo_10.5281-zenodo.21969756.pdf",
        "Active_Fedference_Research_Manuscript_v1.0.4_Zenodo_10.5281-zenodo.21972644.pdf",
    )
    assert set(IMMUTABLE_RELEASE_PDFS) <= set(REQUIRED_TRACKED_PATHS)
    assert HISTORICAL_RELEASE_PDF_LEDGER_PATH in REQUIRED_TRACKED_PATHS


def test_real_historical_release_pdf_ledger_matches_all_shipped_bytes() -> None:
    assert historical_release_pdf_findings(_PROJECT_ROOT) == ()


def _write_historical_pdf_fixtures(root: Path) -> None:
    digests: dict[str, str] = {}
    for filename in IMMUTABLE_RELEASE_PDFS:
        payload = f"historical release PDF fixture: {filename}\n".encode()
        (root / filename).write_bytes(payload)
        digests[filename] = hashlib.sha256(payload).hexdigest()
    ledger_path = root / HISTORICAL_RELEASE_PDF_LEDGER_PATH
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps({"schema_version": "1.0", "sha256": digests}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _init_clean_repo(root: Path) -> None:
    subprocess.run([*_HERMETIC_GIT, "init", "-q", str(root)], check=True, capture_output=True, text=True)
    for relative in REQUIRED_TRACKED_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tracked\n", encoding="utf-8")
    _write_historical_pdf_fixtures(root)
    (root / "manuscript" / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "paper": {"title": "Fixture paper", "version": "0.0.0.dev0"},
                "authors": [
                    {
                        "name": "Ada Lovelace",
                        "orcid": "0000-0000-0000-0001",
                        "affiliation": "Analytical Engine Society",
                    }
                ],
                "publication": {
                    "doi": "",
                    "doi_status": "(forthcoming)",
                    "date_released": None,
                    "software_name": "Clean-checkout fixture",
                    "github_repository": "https://example.invalid/repository",
                    "date_created": "2026-08-01",
                    "abstract": "Fixture abstract.",
                    "description": "Fixture description.",
                },
                "metadata": {"license": "MIT"},
            }
        ),
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "active-fedference"\nversion = "0.0.0.dev0"\n'
        "[project.urls]\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        "version = 1\n"
        "[[package]]\n"
        'name = "active-fedference"\n'
        'version = "0.0.0.dev0"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )
    write_metadata(root)
    subprocess.run([*_HERMETIC_GIT, "-C", str(root), "add", "."], check=True, capture_output=True, text=True)
    subprocess.run(
        [
            *_HERMETIC_GIT,
            "-c",
            "user.name=clean-checkout-test",
            "-c",
            "user.email=clean-checkout-test@example.invalid",
            "-C",
            str(root),
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _commit_publication_identity(
    root: Path,
    *,
    version: str,
    doi: str,
    date_released: str | None,
    doi_status: str | None = None,
) -> None:
    """Install and commit the minimum lifecycle sources for the checkout probe."""
    config_path = root / "manuscript" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paper"]["version"] = version
    publication = config["publication"]
    publication["doi"] = doi
    publication["date_released"] = date_released
    publication.pop("doi_status", None)
    if doi_status is not None:
        publication["doi_status"] = doi_status
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    final_doi_url = (
        f'DOI = "https://doi.org/{doi}"\n'
        if ".dev" not in version and doi
        else ""
    )
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "active-fedference"\nversion = "{version}"\n'
        "[project.urls]\n"
        f"{final_doi_url}",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        "version = 1\n"
        "[[package]]\n"
        'name = "active-fedference"\n'
        f'version = "{version}"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )
    try:
        write_metadata(root)
    except ValueError:
        # Invalid lifecycle cases deliberately retain the prior generated
        # bytes so the real checkout validator can report the source error.
        pass
    subprocess.run(
        [
            *_HERMETIC_GIT,
            "-C",
            str(root),
            "add",
            "manuscript/config.yaml",
            "pyproject.toml",
            "uv.lock",
            *GENERATED_METADATA_PATHS,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            *_HERMETIC_GIT,
            "-c",
            "user.name=clean-checkout-test",
            "-c",
            "user.email=clean-checkout-test@example.invalid",
            "-C",
            str(root),
            "commit",
            "-qm",
            "add publication identity",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_clean_tracking_probe_passes_in_a_real_temporary_git_repo(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    report = inspect_clean_checkout(tmp_path, check_imports=False)
    assert report.ok
    assert report.tracked_files >= len(REQUIRED_TRACKED_PATHS)


def test_unreleased_checkout_does_not_require_a_version_pdf(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    _commit_publication_identity(
        tmp_path,
        version="1.1.0.dev0",
        doi="",
        date_released=None,
        doi_status="(forthcoming)",
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert report.ok


def test_development_checkout_rejects_placeholder_in_doi_field(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    _commit_publication_identity(
        tmp_path,
        version="1.1.0.dev0",
        doi="(forthcoming)",
        date_released=None,
        doi_status="(forthcoming)",
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert not report.ok
    assert any("publication.doi to be the empty string" in item for item in report.findings)


def test_development_checkout_rejects_missing_doi_status(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    _commit_publication_identity(
        tmp_path,
        version="1.1.0.dev0",
        doi="",
        date_released=None,
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert not report.ok
    assert any("doi_status to be exactly" in item for item in report.findings)


def test_development_checkout_rejects_a_deleted_immutable_release_pdf(
    tmp_path: Path,
) -> None:
    _init_clean_repo(tmp_path)
    removed = IMMUTABLE_RELEASE_PDFS[2]
    (tmp_path / removed).unlink()
    subprocess.run(
        [*_HERMETIC_GIT, "-C", str(tmp_path), "add", "-u"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            *_HERMETIC_GIT,
            "-c",
            "user.name=clean-checkout-test",
            "-c",
            "user.email=clean-checkout-test@example.invalid",
            "-C",
            str(tmp_path),
            "commit",
            "-qm",
            "delete immutable release PDF",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert not report.ok
    assert any(removed in item for item in report.findings)


def test_development_checkout_rejects_mutated_immutable_release_pdf(
    tmp_path: Path,
) -> None:
    _init_clean_repo(tmp_path)
    mutated = IMMUTABLE_RELEASE_PDFS[-1]
    (tmp_path / mutated).write_bytes(b"substituted historical PDF bytes\n")
    subprocess.run(
        [*_HERMETIC_GIT, "-C", str(tmp_path), "add", mutated],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            *_HERMETIC_GIT,
            "-c",
            "user.name=clean-checkout-test",
            "-c",
            "user.email=clean-checkout-test@example.invalid",
            "-C",
            str(tmp_path),
            "commit",
            "-qm",
            "substitute immutable release PDF",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert not report.ok
    assert any(
        "historical release PDF SHA-256 mismatch" in item and mutated in item
        for item in report.findings
    )


def test_development_checkout_rejects_tampered_historical_pdf_ledger(
    tmp_path: Path,
) -> None:
    _init_clean_repo(tmp_path)
    ledger_path = tmp_path / HISTORICAL_RELEASE_PDF_LEDGER_PATH
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    target = IMMUTABLE_RELEASE_PDFS[1]
    ledger["sha256"][target] = "0" * 64
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    subprocess.run(
        [*_HERMETIC_GIT, "-C", str(tmp_path), "add", HISTORICAL_RELEASE_PDF_LEDGER_PATH],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            *_HERMETIC_GIT,
            "-c",
            "user.name=clean-checkout-test",
            "-c",
            "user.email=clean-checkout-test@example.invalid",
            "-C",
            str(tmp_path),
            "commit",
            "-qm",
            "tamper with historical PDF ledger",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert not report.ok
    assert any(
        "historical release PDF SHA-256 mismatch" in item and target in item
        for item in report.findings
    )


def test_checkout_requires_the_authoritative_identity_sources(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    config_path = tmp_path / "manuscript" / "config.yaml"
    config_path.unlink()
    subprocess.run(
        [*_HERMETIC_GIT, "-C", str(tmp_path), "add", "-u"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            *_HERMETIC_GIT,
            "-c",
            "user.name=clean-checkout-test",
            "-c",
            "user.email=clean-checkout-test@example.invalid",
            "-C",
            str(tmp_path),
            "commit",
            "-qm",
            "remove publication identity",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert not report.ok
    assert any("manuscript/config.yaml" in item for item in report.findings)


def test_released_checkout_requires_its_version_doi_pdf(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    _commit_publication_identity(
        tmp_path,
        version="1.1.0",
        doi="10.5281/zenodo.12345",
        date_released="2026-08-25",
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert not report.ok
    assert any("Active_Fedference_Research_Manuscript_v1.1.0" in item for item in report.findings)


def test_released_checkout_accepts_new_exact_pdf_without_replacing_history(
    tmp_path: Path,
) -> None:
    _init_clean_repo(tmp_path)
    _commit_publication_identity(
        tmp_path,
        version="1.1.0",
        doi="10.5281/zenodo.12345",
        date_released="2026-08-25",
    )
    new_pdf = (
        "Active_Fedference_Research_Manuscript_v1.1.0_"
        "Zenodo_10.5281-zenodo.12345.pdf"
    )
    (tmp_path / new_pdf).write_bytes(b"new final release PDF fixture\n")
    subprocess.run(
        [*_HERMETIC_GIT, "-C", str(tmp_path), "add", new_pdf],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            *_HERMETIC_GIT,
            "-c",
            "user.name=clean-checkout-test",
            "-c",
            "user.email=clean-checkout-test@example.invalid",
            "-C",
            str(tmp_path),
            "commit",
            "-qm",
            "add exact final release PDF",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert report.ok
    assert all((tmp_path / name).is_file() for name in IMMUTABLE_RELEASE_PDFS)


def test_final_checkout_rejects_stale_development_doi_status(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    _commit_publication_identity(
        tmp_path,
        version="1.1.0",
        doi="10.5281/zenodo.12345",
        date_released="2026-08-25",
        doi_status="(forthcoming)",
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert not report.ok
    assert any("must remove publication.doi_status" in item for item in report.findings)


def test_final_checkout_rejects_a_missing_doi_and_release_date(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    _commit_publication_identity(
        tmp_path,
        version="1.1.0",
        doi="",
        date_released=None,
        doi_status="(forthcoming)",
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert not report.ok
    assert any("final release metadata requires" in item for item in report.findings)


def test_development_checkout_rejects_release_identity(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    _commit_publication_identity(
        tmp_path,
        version="1.1.0.dev0",
        doi="10.5281/zenodo.12345",
        date_released="2026-08-25",
        doi_status="(forthcoming)",
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert not report.ok
    assert any("development metadata" in item for item in report.findings)


def test_checkout_rejects_package_manuscript_version_drift(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    _commit_publication_identity(
        tmp_path,
        version="1.1.0.dev0",
        doi="",
        date_released=None,
        doi_status="(forthcoming)",
    )
    config_path = tmp_path / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["paper"]["version"] = "1.1.1.dev0"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    subprocess.run(
        [*_HERMETIC_GIT, "-C", str(tmp_path), "add", "manuscript/config.yaml"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            *_HERMETIC_GIT,
            "-c",
            "user.name=clean-checkout-test",
            "-c",
            "user.email=clean-checkout-test@example.invalid",
            "-C",
            str(tmp_path),
            "commit",
            "-qm",
            "introduce identity drift",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = inspect_clean_checkout(tmp_path, check_imports=False)

    assert not report.ok
    assert any("paper.version must exactly match" in item for item in report.findings)


def test_dirty_checkout_is_reported(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    report = inspect_clean_checkout(tmp_path, check_imports=False)
    assert not report.ok
    assert any("worktree is dirty" in finding for finding in report.findings)


def test_missing_required_tracking_is_reported(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    (tmp_path / "src" / "publication" / "pipeline_freshness.py").unlink()
    subprocess.run(
        [*_HERMETIC_GIT, "-C", str(tmp_path), "add", "-u"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            *_HERMETIC_GIT,
            "-c",
            "user.name=clean-checkout-test",
            "-c",
            "user.email=clean-checkout-test@example.invalid",
            "-C",
            str(tmp_path),
            "commit",
            "-qm",
            "remove",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = inspect_clean_checkout(tmp_path, check_imports=False)
    assert not report.ok
    assert any("pipeline_freshness.py" in finding for finding in report.findings)


def test_import_probe_failure_is_reported(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    report = inspect_clean_checkout(tmp_path, check_imports=True)
    assert not report.ok
    assert any("import probe failed" in finding for finding in report.findings)


def test_import_probe_does_not_accept_packages_from_inherited_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incomplete candidate cannot borrow imports from a neighbouring checkout."""
    _init_clean_repo(tmp_path)
    external = tmp_path.parent / "external_packages"
    for package in ("analysis", "fedference", "figures", "publication"):
        path = external / package
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").write_text("SOURCE = 'external'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(external))

    report = inspect_clean_checkout(tmp_path, check_imports=True)

    assert not report.ok
    assert any("package import probe failed" in finding for finding in report.findings)
