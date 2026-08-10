"""Tests for the clean-checkout and clone-correctness probe."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from publication.clean_checkout import REQUIRED_TRACKED_PATHS, inspect_clean_checkout

_HERMETIC_GIT = ("git", "-c", "core.fsmonitor=false", "-c", "core.untrackedcache=false")


def test_validation_receipt_chain_is_required_for_clean_checkout() -> None:
    """Keep final-hydration evidence present in every clone-correct candidate."""
    assert {
        "src/publication/validation_receipt.py",
        "scripts/validate_test_coverage.py",
        "tests/test_validation_receipt.py",
        "output/data/analysis_execution.json",
        "output/data/test_coverage_receipt.json",
    } <= set(REQUIRED_TRACKED_PATHS)


def _init_clean_repo(root: Path) -> None:
    subprocess.run([*_HERMETIC_GIT, "init", "-q", str(root)], check=True, capture_output=True, text=True)
    for relative in REQUIRED_TRACKED_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tracked\n", encoding="utf-8")
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


def test_clean_tracking_probe_passes_in_a_real_temporary_git_repo(tmp_path: Path) -> None:
    _init_clean_repo(tmp_path)
    report = inspect_clean_checkout(tmp_path, check_imports=False)
    assert report.ok
    assert report.tracked_files >= len(REQUIRED_TRACKED_PATHS)


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
