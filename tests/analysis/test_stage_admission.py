"""Malformed workspace and ownership inputs fail before starting any producer."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from analysis.stage_io import StageError, digest, encode, read_json
from analysis.stage_process import inherit_workspace_lock, request_cancel, workspace_lock, writer_held
from analysis.stage_runner import run_stage
from analysis.stage_workspace import admit_workspace, prepare_workspace
from tests.analysis.stage_fixtures import prepared as prepared


@pytest.mark.parametrize(
    "stage,schema,rss",
    [
        ("unknown", "belief_sharing", 1024),
        ("belief_sharing", "unknown", 1024),
        ("belief_sharing", "belief_sharing", 0),
        ("belief_sharing", "belief_sharing", True),
    ],
)
def test_invalid_stage_or_rss_starts_no_attempt(
    prepared: tuple[Path, Path],
    stage: str,
    schema: str,
    rss: int,
) -> None:
    _, workspace = prepared
    with pytest.raises(StageError, match="stage/schema|max_rss_mb"):
        run_stage(workspace, stage, schema, max_seconds=5, max_rss_mb=rss)
    assert not (workspace / "attempts").exists()


@pytest.mark.parametrize("status,token", [("complete", "a" * 32), ("running", "../outside"), ("running", 12)])
def test_cancel_rejects_nonrunning_or_unconfined_token(
    prepared: tuple[Path, Path],
    status: str,
    token: object,
) -> None:
    _, workspace = prepared
    (workspace / "active.json").write_bytes(encode({"status": status, "attempt": token}))
    with pytest.raises(StageError, match="no running|invalid attempt"):
        request_cancel(workspace)
    assert not list(workspace.glob("cancel-*"))


def test_lock_observation_and_inheritance_require_the_actual_private_inode(
    prepared: tuple[Path, Path],
) -> None:
    _, workspace = prepared
    assert not writer_held(workspace)
    other = workspace / "other-lock"
    other.write_bytes(b"retained")
    descriptor = os.open(other, os.O_RDWR)
    try:
        with workspace_lock(workspace):
            with pytest.raises(StageError, match="did not inherit"):
                inherit_workspace_lock(workspace, descriptor)
    finally:
        os.close(descriptor)
    assert not writer_held(workspace)
    os.link(workspace / "writer.lock", workspace / "retained-lock-link")
    with pytest.raises(StageError, match="private regular"):
        writer_held(workspace)


@pytest.mark.parametrize("damage", ["root", "profile", "digest", "missing", "inventory"])
def test_workspace_refuses_unbound_source_before_reuse(prepared: tuple[Path, Path], damage: str) -> None:
    _, workspace = prepared
    if damage == "root":
        alias = workspace.with_name("alias")
        alias.symlink_to(workspace, target_is_directory=True)
        with pytest.raises(StageError, match="canonical"):
            admit_workspace(alias)
        return
    path = workspace / "workspace.json"
    manifest = read_json(path)
    if damage == "profile":
        manifest["profile"] = "unregistered"
    elif damage == "digest":
        manifest["files"]["src/analysis/stage_io.py"] = "invalid digest"
    elif damage == "missing":
        target = workspace / "source/src/analysis/stage_io.py"
        target.rename(workspace / "retained-stage-io.py")
    else:
        (workspace / "source/src/unregistered.py").write_text("value = 1\n")
    manifest["identity"] = digest(encode({k: v for k, v in manifest.items() if k != "identity"}))
    path.write_bytes(encode(manifest))
    with pytest.raises(StageError, match="profile|digest|missing|inventory"):
        admit_workspace(workspace)


@pytest.mark.parametrize("profile", ["smoke-invalid", ""])
def test_preparation_refuses_an_unknown_profile_without_creating_workspace(
    prepared: tuple[Path, Path],
    profile: str,
) -> None:
    project, workspace = prepared
    successor = workspace.with_name("not-created")
    with pytest.raises(StageError, match="profile"):
        prepare_workspace(project, successor, profile=profile)
    assert not successor.exists()
