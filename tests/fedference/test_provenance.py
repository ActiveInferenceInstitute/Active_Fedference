"""Runtime and explicit-source provenance contracts."""

from __future__ import annotations

import json
import subprocess

from fedference import __version__
from fedference.provenance import (
    RuntimeProvenance,
    SourceProvenance,
    active_fedference_checkout_version,
    collect_source_provenance,
    runtime_provenance,
)

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
