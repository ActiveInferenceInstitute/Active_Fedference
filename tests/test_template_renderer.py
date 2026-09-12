"""Real-Git tests for the source-locked Template renderer boundary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from publication.template_renderer import (
    TEMPLATE_RENDERER_LOCK_SCHEMA,
    TEMPLATE_RENDERER_REPOSITORY,
    inspect_template_renderer_checkout,
    load_template_renderer_lock,
    renderer_receipt_findings,
    require_locked_template_renderer,
)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _init_repository(root: Path, *, origin: str = "https://github.com/docxology/template.git") -> str:
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Renderer Test")
    _git(root, "config", "user.email", "renderer@example.invalid")
    (root / "infrastructure").mkdir()
    (root / "infrastructure" / "renderer.py").write_text("RENDERER = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "test: initialize renderer")
    _git(root, "remote", "add", "origin", origin)
    return _git(root, "rev-parse", "HEAD^{commit}")


def _write_lock(
    project: Path,
    commit: str,
    *,
    extra: str = "",
    repository: str = TEMPLATE_RENDERER_REPOSITORY,
) -> None:
    config = project / "manuscript" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "rendering:\n"
        "  template_renderer:\n"
        f'    schema_version: "{TEMPLATE_RENDERER_LOCK_SCHEMA}"\n'
        f'    repository: "{repository}"\n'
        f'    git_commit: "{commit}"\n'
        '    git_tree_state: "clean"\n'
        f"{extra}",
        encoding="utf-8",
    )


def test_exact_clean_renderer_succeeds_without_persisting_local_details(tmp_path: Path) -> None:
    renderer = tmp_path / "template"
    project = tmp_path / "project"
    commit = _init_repository(renderer, origin="git@github.com:docxology/template.git")
    _write_lock(project, commit)

    identity = require_locked_template_renderer(project, renderer)

    assert identity == inspect_template_renderer_checkout(renderer)
    assert identity == load_template_renderer_lock(project)
    serialized = json.dumps(identity.as_dict(), sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "github.com" not in serialized
    assert identity.repository == TEMPLATE_RENDERER_REPOSITORY


@pytest.mark.parametrize(
    ("commit", "extra", "repository", "message"),
    (
        ("abc123", "", TEMPLATE_RENDERER_REPOSITORY, "40 lowercase hexadecimal"),
        ("A" * 40, "", TEMPLATE_RENDERER_REPOSITORY, "40 lowercase hexadecimal"),
        ("0" * 40, "    unexpected: true\n", TEMPLATE_RENDERER_REPOSITORY, "fields mismatch"),
        ("0" * 40, "", "someone/else", "repository must be"),
    ),
)
def test_renderer_lock_rejects_noncanonical_values(
    tmp_path: Path,
    commit: str,
    extra: str,
    repository: str,
    message: str,
) -> None:
    _write_lock(tmp_path, commit, extra=extra, repository=repository)
    with pytest.raises(ValueError, match=message):
        load_template_renderer_lock(tmp_path)


def test_missing_renderer_lock_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "config.yaml").write_text("rendering: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing rendering.template_renderer"):
        load_template_renderer_lock(tmp_path)


def test_wrong_commit_and_wrong_origin_fail(tmp_path: Path) -> None:
    renderer = tmp_path / "template"
    project = tmp_path / "project"
    commit = _init_repository(renderer)
    _write_lock(project, "0" * 40)
    with pytest.raises(ValueError, match="does not match"):
        require_locked_template_renderer(project, renderer)

    _write_lock(project, commit)
    _git(renderer, "remote", "set-url", "origin", "https://github.com/someone/else.git")
    with pytest.raises(ValueError, match="wrong origin"):
        require_locked_template_renderer(project, renderer)


@pytest.mark.parametrize("kind", ("staged", "unstaged", "untracked"))
def test_dirty_renderer_states_fail(tmp_path: Path, kind: str) -> None:
    renderer = tmp_path / "template"
    _init_repository(renderer)
    tracked = renderer / "infrastructure" / "renderer.py"
    if kind == "staged":
        tracked.write_text("RENDERER = 2\n", encoding="utf-8")
        _git(renderer, "add", str(tracked.relative_to(renderer)))
    elif kind == "unstaged":
        tracked.write_text("RENDERER = 2\n", encoding="utf-8")
    else:
        (renderer / "untracked.txt").write_text("local\n", encoding="utf-8")

    with pytest.raises(ValueError, match="completely clean"):
        inspect_template_renderer_checkout(renderer)


def test_checkout_subdirectory_is_rejected(tmp_path: Path) -> None:
    renderer = tmp_path / "template"
    _init_repository(renderer)
    with pytest.raises(ValueError, match="exact Git checkout root"):
        inspect_template_renderer_checkout(renderer / "infrastructure")


def test_dirty_submodule_is_rejected(tmp_path: Path) -> None:
    component = tmp_path / "component"
    _init_repository(component, origin="https://github.com/docxology/template.git")
    renderer = tmp_path / "template"
    _init_repository(renderer)
    subprocess.run(
        (
            "git",
            "-C",
            str(renderer),
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(component),
            "vendor/component",
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _git(renderer, "commit", "-q", "-am", "test: add component")
    (renderer / "vendor" / "component" / "infrastructure" / "renderer.py").write_text(
        "RENDERER = 3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="completely clean"):
        inspect_template_renderer_checkout(renderer)


def test_renderer_receipt_findings_are_stable_and_lock_bound(tmp_path: Path) -> None:
    project = tmp_path / "project"
    commit = "1" * 40
    _write_lock(project, commit)
    valid = {
        "schema_version": TEMPLATE_RENDERER_LOCK_SCHEMA,
        "repository": TEMPLATE_RENDERER_REPOSITORY,
        "git_commit": commit,
        "git_tree_state": "clean",
    }
    assert renderer_receipt_findings(project, valid) == []
    assert renderer_receipt_findings(project, "legacy renderer label") == [
        "render receipt renderer must be a mapping"
    ]
    assert renderer_receipt_findings(project, {**valid, "git_commit": "2" * 40}) == [
        "render receipt renderer does not match the committed renderer lock"
    ]
