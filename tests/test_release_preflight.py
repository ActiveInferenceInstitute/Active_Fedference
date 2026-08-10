"""Regression tests for the release-facing script preflights.

The scripts remain thin orchestrators, but the bundle boundary itself must
reject stale generated metadata and an unvalidated render tree. These tests
exercise those checks with real temporary files rather than substituting test
doubles for the publication validators.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml

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
                "authors": [
                    {
                        "name": "Ada Lovelace",
                        "orcid": "0000-0000-0000-0001",
                        "affiliation": "Analytical Engine Society",
                    }
                ],
                "publication": {
                    "software_name": "Preflight fixture",
                    "github_repository": "https://example.invalid/repository",
                    "date_created": "2026-08-02",
                    "abstract": "Fixture abstract.",
                    "description": "Fixture description.",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "preflight-fixture"\nversion = "0.0.1"\n', encoding="utf-8"
    )
    write_metadata(root)
    return root


def test_release_preflight_rejects_stale_generated_metadata(tmp_path: Path) -> None:
    root = _metadata_project(tmp_path)
    build_release = _load_script("build_release.py")
    build_release._require_current_metadata(root)
    (root / "CITATION.cff").write_text("stale\n", encoding="utf-8")

    with pytest.raises(ValueError, match="generated publication metadata is stale: CITATION.cff"):
        build_release._require_current_metadata(root)


def test_release_preflight_rejects_unvalidated_rendered_surfaces(tmp_path: Path) -> None:
    build_release = _load_script("build_release.py")

    with pytest.raises(ValueError, match="rendered surface validation failed"):
        build_release._require_current_rendered_surfaces(tmp_path)


def test_render_receipt_cli_refuses_an_unvalidated_render_tree(tmp_path: Path) -> None:
    recorder = _load_script("record_pipeline_stage.py")

    with pytest.raises(ValueError, match="cannot record render before rendered-surface validation passes"):
        recorder.main(["render", "--project-root", str(tmp_path)])
