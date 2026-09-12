"""Regression tests for the release-facing script preflights.

The scripts remain thin orchestrators, but the bundle boundary itself must
reject stale generated metadata and an unvalidated render tree. These tests
exercise those checks with real temporary files rather than substituting test
doubles for the publication validators.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from publication.clean_checkout import (
    HISTORICAL_RELEASE_PDF_LEDGER_PATH,
    IMMUTABLE_RELEASE_PDFS,
)
from publication.metadata import write_metadata

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str) -> ModuleType:
    path = _PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{name.removesuffix('.py')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata_project(root: Path) -> Path:
    (root / "manuscript").mkdir()
    (root / "manuscript" / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "paper": {
                    "title": "Preflight Fixture Paper",
                    "subtitle": "A Complete Fixture Subtitle",
                    "version": "0.0.1.dev0",
                },
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
                    "software_name": "Preflight fixture",
                    "github_repository": "https://example.invalid/repository",
                    "date_created": "2026-08-02",
                    "date_released": None,
                    "abstract": "Fixture abstract.",
                    "description": "Fixture description.",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "preflight-fixture"\nversion = "0.0.1.dev0"\n'
        "[project.urls]\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        "version = 1\n"
        "[[package]]\n"
        'name = "preflight-fixture"\n'
        'version = "0.0.1.dev0"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )
    write_metadata(root)
    return root


def _historical_pdf_project(root: Path) -> Path:
    digests: dict[str, str] = {}
    for filename in IMMUTABLE_RELEASE_PDFS:
        payload = f"historical release PDF preflight fixture: {filename}\n".encode()
        (root / filename).write_bytes(payload)
        digests[filename] = hashlib.sha256(payload).hexdigest()
    ledger_path = root / HISTORICAL_RELEASE_PDF_LEDGER_PATH
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps({"schema_version": "1.0", "sha256": digests}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return root


def test_release_preflight_rejects_stale_generated_metadata(tmp_path: Path) -> None:
    root = _metadata_project(tmp_path)
    build_release = _load_script("build_release.py")
    build_release._require_current_metadata(root)
    (root / "CITATION.cff").write_text("stale\n", encoding="utf-8")

    with pytest.raises(ValueError, match="generated publication metadata is stale: CITATION.cff"):
        build_release._require_current_metadata(root)


def test_release_preflight_rejects_historical_pdf_substitution(tmp_path: Path) -> None:
    root = _historical_pdf_project(tmp_path)
    build_release = _load_script("build_release.py")
    build_release._require_immutable_release_pdfs(root)
    mutated = IMMUTABLE_RELEASE_PDFS[0]
    (root / mutated).write_bytes(b"substituted historical PDF bytes\n")

    with pytest.raises(
        ValueError,
        match=f"historical release PDF validation failed:.*{mutated}",
    ):
        build_release._require_immutable_release_pdfs(root)


def test_release_preflight_rejects_unvalidated_rendered_surfaces(tmp_path: Path) -> None:
    build_release = _load_script("build_release.py")

    with pytest.raises(ValueError, match="rendered surface validation failed"):
        build_release._require_current_rendered_surfaces(tmp_path)


def test_render_receipt_cli_refuses_an_unvalidated_render_tree(tmp_path: Path) -> None:
    recorder = _load_script("record_pipeline_stage.py")

    with pytest.raises(ValueError, match="cannot record render before rendered-surface validation passes"):
        recorder.main(
            [
                "render",
                "--project-root",
                str(tmp_path),
                "--template-root",
                str(tmp_path / "missing-template"),
            ]
        )


def test_render_receipt_cli_requires_producer_logs(tmp_path: Path) -> None:
    recorder = _load_script("record_pipeline_stage.py")
    root = tmp_path / "producer-log-fixture"
    pdf_dir = root / "output" / "pdf"
    slides_dir = root / "output" / "slides"
    pdf_dir.mkdir(parents=True)
    slides_dir.mkdir(parents=True)

    for filename in ("active_fedference_combined.pdf", "_combined_manuscript.tex"):
        shutil.copy2(_PROJECT_ROOT / "output" / "pdf" / filename, pdf_dir / filename)
    for filename in ("00_abstract_slides.pdf", "00_abstract_slides.tex"):
        shutil.copy2(_PROJECT_ROOT / "output" / "slides" / filename, slides_dir / filename)
    shutil.copytree(_PROJECT_ROOT / "output" / "web", root / "output" / "web")

    with pytest.raises(ValueError) as exc_info:
        recorder.main(
            [
                "render",
                "--project-root",
                str(root),
                "--template-root",
                str(tmp_path / "missing-template"),
            ]
        )
    detail = str(exc_info.value)
    assert "missing combined manuscript logs" in detail
    assert "missing generated slide logs" in detail
