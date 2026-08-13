"""Tests for shared script checkout-root resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from project_paths import ENV_PROJECT_ROOT_VAR, resolve_script_project_root


def _make_project_root(path: Path) -> Path:
    (path / "manuscript").mkdir(parents=True)
    (path / "manuscript" / "config.yaml").write_text("paper: {}\n", encoding="utf-8")
    return path


def test_explicit_root_takes_precedence_over_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_root = _make_project_root(tmp_path / "env")
    explicit_root = tmp_path / "explicit"
    explicit_root.mkdir()
    monkeypatch.setenv(ENV_PROJECT_ROOT_VAR, str(env_root))

    assert resolve_script_project_root(tmp_path / "default", explicit_root) == explicit_root.resolve()


def test_environment_root_is_used_when_explicit_root_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_root = _make_project_root(tmp_path / "env")
    monkeypatch.setenv(ENV_PROJECT_ROOT_VAR, str(env_root))

    assert resolve_script_project_root(tmp_path / "default") == env_root.resolve()


def test_explicit_root_is_allowed_for_incomplete_validator_fixture(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()

    assert resolve_script_project_root(tmp_path / "default", fixture_root) == fixture_root.resolve()


def test_invalid_explicit_root_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="explicit project root"):
        resolve_script_project_root(tmp_path / "default", tmp_path / "missing")
