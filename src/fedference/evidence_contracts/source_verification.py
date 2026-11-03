"""Live checkout comparison for evidence receipts."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..provenance import (
    collect_source_provenance,
)
from .application_schema import ApplicationReceipt
from .schema import GitTreeState


def _git_revision(project_root: Path) -> tuple[str, GitTreeState]:
    """Return the live full commit and tree state without invoking shell hooks."""
    commit = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedcache=false",
            "-C",
            str(project_root),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0 or not commit.stdout.strip():
        return "unavailable", "unavailable"
    status = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedcache=false",
            "-C",
            str(project_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    state: GitTreeState = (
        "unavailable"
        if status.returncode != 0
        else ("dirty" if status.stdout else "clean")
    )
    return commit.stdout.strip(), state



def _application_source_findings(
    receipt: ApplicationReceipt,
    *,
    require_clean_git: bool,
    project_root: str | Path | None,
) -> tuple[str, ...]:
    """Return only checkout/source-equivalence findings."""
    findings: list[str] = []
    if require_clean_git and receipt.source_provenance.source_kind != "git_checkout":
        findings.append("application receipt is not bound to an explicit Git checkout")
    if require_clean_git and receipt.source_provenance.git_tree_state != "clean":
        findings.append(
            "receipt Git tree state is "
            f"{receipt.source_provenance.git_tree_state!r}, not 'clean'"
        )
    if project_root is not None:
        live = collect_source_provenance(project_root)
        recorded = receipt.source_provenance
        if live.source_kind != "git_checkout":
            findings.append(
                f"live Git source is unavailable: {Path(project_root).resolve()}"
            )
        elif recorded.source_kind != "git_checkout":
            findings.append("receipt does not bind an explicit Git checkout")
        else:
            if live.git_commit != recorded.git_commit:
                findings.append(
                    "live Git commit does not match receipt: "
                    f"{live.git_commit} != {recorded.git_commit}"
                )
            if live.git_tree_state != recorded.git_tree_state:
                findings.append(
                    "live Git tree state does not match receipt: "
                    f"{live.git_tree_state!r} != {recorded.git_tree_state!r}"
                )
            if live.uv_lock_sha256 != recorded.uv_lock_sha256:
                findings.append("live uv.lock digest does not match receipt")
            if require_clean_git and live.git_tree_state != "clean":
                findings.append(
                    f"live Git tree state is {live.git_tree_state!r}, not 'clean'"
                )
            if recorded.git_tree_state != "clean":
                findings.append(
                    "receipt binds an unhashed dirty tree; live source equivalence "
                    "cannot be verified"
                )
    return tuple(dict.fromkeys(findings))

