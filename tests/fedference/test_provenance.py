"""Runtime and explicit-source provenance contracts."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from fedference import __version__
from fedference.evidence_contracts.source_verification import _git_revision as evidence_git_revision
from fedference.provenance import (
    RuntimeProvenance,
    SourceProvenance,
    active_fedference_checkout_version,
    collect_source_provenance,
    runtime_provenance,
)
from fedference_cli._support import _git_revision as cli_git_revision
from fedference_cli._support import _resolve_project_root, _validate_output_dir

HERMETIC_GIT = ("git", "-c", "core.fsmonitor=false", "-c", "core.untrackedcache=false")


def _write_checkout_identity(root, *, version: str = __version__) -> None:
    package = root / "src" / "fedference"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "aggregation.py").write_text("\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "active_fedference"\n'
        f'version = "{version}"\n',
        encoding="utf-8",
    )


def test_runtime_provenance_round_trips_without_local_install_url() -> None:
    provenance = runtime_provenance()
    assert provenance.distribution_name == "active_fedference"
    assert provenance.distribution_version == __version__
    assert provenance.python_version
    assert {"numpy", "scipy", "matplotlib", "pyyaml", "typing-extensions"}.issubset(
        provenance.dependency_versions
    )
    payload = provenance.as_dict()
    assert RuntimeProvenance.from_dict(payload).as_dict() == payload
    serialized = json.dumps(payload)
    assert "file://" not in serialized
    assert "direct_url" not in serialized


def test_explicit_git_provenance_is_path_independent_and_hash_bound(tmp_path) -> None:
    source = tmp_path / "source"
    clone = tmp_path / "clone"
    source.mkdir()
    _write_checkout_identity(source)
    (source / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (source / "tracked.txt").write_text("source\n", encoding="utf-8")
    for command in (
        (*HERMETIC_GIT, "init", "-q"),
        (*HERMETIC_GIT, "config", "user.email", "test@example.com"),
        (*HERMETIC_GIT, "config", "user.name", "Provenance Test"),
        (*HERMETIC_GIT, "add", "."),
        (*HERMETIC_GIT, "commit", "-qm", "fixture"),
    ):
        subprocess.run(command, cwd=source, check=True)
    subprocess.run(("git", "clone", "-q", str(source), str(clone)), check=True)
    original = collect_source_provenance(source)
    moved = collect_source_provenance(clone)
    assert original.source_kind == "git_checkout"
    assert active_fedference_checkout_version(source) == __version__
    assert original.git_tree_state == "clean"
    assert original.git_commit == moved.git_commit
    assert original.uv_lock_sha256 == moved.uv_lock_sha256
    assert SourceProvenance.from_dict(original.as_dict()).as_dict() == original.as_dict()
    assert str(source) not in json.dumps(original.as_dict())

    (source / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    dirty = collect_source_provenance(source)
    assert dirty.git_tree_state == "dirty"


def test_non_git_source_never_invents_a_revision(tmp_path) -> None:
    provenance = collect_source_provenance(tmp_path)
    assert provenance.source_kind in {"installed_archive", "unavailable"}
    assert provenance.git_commit is None
    assert provenance.git_tree_state is None
    assert provenance.uv_lock_sha256 is None


def test_unrelated_git_repository_is_not_accepted_as_project_source(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "other_project"\nversion = "1.0"\n',
        encoding="utf-8",
    )
    package = tmp_path / "src" / "fedference"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "aggregation.py").write_text("\n", encoding="utf-8")
    subprocess.run((*HERMETIC_GIT, "init", "-q"), cwd=tmp_path, check=True)
    assert active_fedference_checkout_version(tmp_path) is None
    provenance = collect_source_provenance(tmp_path)
    assert provenance.source_kind in {"installed_archive", "unavailable"}
    assert provenance.git_commit is None


@pytest.fixture
def git_source_checkout(tmp_path: Path) -> tuple[Path, str]:
    """Create an independent Git store; no canonical index is involved."""
    source = tmp_path / "source"
    source.mkdir()
    _write_checkout_identity(source)
    (source / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (source / "tracked.txt").write_text("unchanged source\n", encoding="utf-8")
    (source / "changed.txt").write_text("original\n", encoding="utf-8")
    for arguments in (
        ("init", "-q"),
        ("config", "user.email", "test@example.invalid"),
        ("config", "user.name", "Provenance Test"),
        ("add", "."),
        ("-c", "commit.gpgsign=false", "-c", f"core.hooksPath={os.devnull}", "commit", "-qm", "fixture"),
    ):
        subprocess.run((*HERMETIC_GIT, "-C", str(source), *arguments), check=True, capture_output=True)
    commit = subprocess.run(
        (*HERMETIC_GIT, "-C", str(source), "rev-parse", "HEAD"),
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return source, commit


_SOURCE_OBSERVERS = (
    pytest.param(collect_source_provenance, id="application-provenance"),
    pytest.param(cli_git_revision, id="cli-receipt"),
    pytest.param(evidence_git_revision, id="evidence-verification"),
)


@pytest.mark.parametrize("observer", _SOURCE_OBSERVERS)
@pytest.mark.parametrize("optional_locks", (None, "1"), ids=("unset-lock-policy", "hostile-lock-policy"))
@pytest.mark.parametrize("state", ("clean", "staged", "unstaged", "untracked"))
def test_source_observation_preserves_index_bytes(
    git_source_checkout: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    observer: Callable[[Path], SourceProvenance | tuple[str, str]],
    optional_locks: str | None,
    state: str,
) -> None:
    source, commit = git_source_checkout
    if state in {"staged", "unstaged"}:
        (source / "changed.txt").write_text("edited\n", encoding="utf-8")
        if state == "staged":
            subprocess.run(
                (*HERMETIC_GIT, "-C", str(source), "add", "changed.txt"),
                check=True, capture_output=True,
            )
    elif state == "untracked":
        (source / "untracked.txt").write_text("new\n", encoding="utf-8")
    if optional_locks is None:
        monkeypatch.delenv("GIT_OPTIONAL_LOCKS", raising=False)
    else:
        monkeypatch.setenv("GIT_OPTIONAL_LOCKS", optional_locks)

    tracked = source / "tracked.txt"
    original = tracked.read_bytes()
    stat = tracked.stat()
    os.utime(tracked, ns=(stat.st_atime_ns, stat.st_mtime_ns - 2_000_000_000))
    index = source / ".git" / "index"
    before = index.read_bytes()
    observed = observer(source)
    expected_state = "clean" if state == "clean" else "dirty"
    if isinstance(observed, SourceProvenance):
        assert observed.source_kind == "git_checkout"
        assert observed.git_commit == commit
        assert observed.git_tree_state == expected_state
    else:
        assert observed == (commit, expected_state)
    assert index.read_bytes() == before
    assert tracked.read_bytes() == original
    assert os.environ.get("GIT_OPTIONAL_LOCKS") == optional_locks

    # Positive control: this exact stat-drift fixture really permits an
    # ordinary status observation to refresh the fixture's own index.
    refreshed = subprocess.run(
        (*HERMETIC_GIT, "-C", str(source), "status", "--porcelain=v1", "--untracked-files=normal"),
        check=True, capture_output=True, text=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "1"},
    )
    assert bool(refreshed.stdout) == (state != "clean")
    assert index.read_bytes() != before


@pytest.mark.parametrize("observer", _SOURCE_OBSERVERS)
@pytest.mark.parametrize("state", ("missing", "non-git", "unborn"))
def test_unavailable_source_observation_preserves_caller_policy(
    tmp_path: Path,
    observer: Callable[[Path], SourceProvenance | tuple[str, str]],
    state: str,
) -> None:
    source = tmp_path / "source"
    if state != "missing":
        source.mkdir()
        _write_checkout_identity(source)
    if state == "unborn":
        subprocess.run((*HERMETIC_GIT, "-C", str(source), "init", "-q"), check=True, capture_output=True)
    observed = observer(source)
    if isinstance(observed, SourceProvenance):
        assert observed.source_kind in {"installed_archive", "unavailable"}
        assert observed.git_commit is None
        assert observed.git_tree_state is None
        assert observed.uv_lock_sha256 is None
        assert any("explicitly selected project root" in warning for warning in observed.warnings)
    else:
        assert observed == ("unavailable", "unavailable")


@pytest.mark.parametrize("observer", _SOURCE_OBSERVERS)
def test_source_observation_does_not_hide_missing_git_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observer: Callable[[Path], SourceProvenance | tuple[str, str]],
) -> None:
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))
    with pytest.raises(FileNotFoundError):
        observer(tmp_path)


def test_cli_discovery_preserves_checkout_and_output_policies(
    git_source_checkout: tuple[Path, str], tmp_path: Path,
) -> None:
    source, _ = git_source_checkout
    index = source / ".git" / "index"
    before = index.read_bytes()
    assert _resolve_project_root(source / "src" / "fedference") == source.resolve()
    outside = tmp_path / "caller-output"
    assert _validate_output_dir(str(outside), source) == outside
    assert not outside.exists()
    with pytest.raises(ValueError, match="committed reviewer output"):
        _validate_output_dir(str(source / "output" / "run"), tmp_path)
    assert not (source / "output").exists()
    assert index.read_bytes() == before
