"""Fail-closed identity checks for the external Template renderer checkout.

The publication repository owns the renderer *expectation* while the Template
repository owns the renderer implementation.  This module joins those two
planes without persisting a machine-local checkout path, raw remote URL, branch
name, or uncommitted-diff digest.  A release-facing render is admissible only
when an explicitly supplied checkout is the canonical repository, is at the
exact committed revision, and has no staged, unstaged, untracked, or submodule
changes.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

import yaml

TEMPLATE_RENDERER_LOCK_SCHEMA: Final[str] = "template-renderer-lock-v1"
TEMPLATE_RENDERER_REPOSITORY: Final[str] = "docxology/template"
_LOCK_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_version", "repository", "git_commit", "git_tree_state"}
)
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class TemplateRendererIdentity:
    """Canonical, path-free identity for one clean Template checkout."""

    schema_version: str
    repository: str
    git_commit: str
    git_tree_state: str

    def as_dict(self) -> dict[str, str]:
        """Return the stable JSON-compatible receipt representation."""
        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "git_commit": self.git_commit,
            "git_tree_state": self.git_tree_state,
        }


def _identity_from_mapping(payload: object, *, label: str) -> TemplateRendererIdentity:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    keys = {str(key) for key in payload}
    if keys != _LOCK_FIELDS:
        missing = sorted(_LOCK_FIELDS - keys)
        extra = sorted(keys - _LOCK_FIELDS)
        raise ValueError(f"{label} fields mismatch: missing={missing}, extra={extra}")

    values: dict[str, str] = {}
    for field in sorted(_LOCK_FIELDS):
        value = payload.get(field)
        if not isinstance(value, str):
            raise ValueError(f"{label}.{field} must be a string")
        values[field] = value

    if values["schema_version"] != TEMPLATE_RENDERER_LOCK_SCHEMA:
        raise ValueError(f"{label}.schema_version is unsupported")
    if values["repository"] != TEMPLATE_RENDERER_REPOSITORY:
        raise ValueError(f"{label}.repository must be {TEMPLATE_RENDERER_REPOSITORY!r}")
    if not _FULL_COMMIT.fullmatch(values["git_commit"]):
        raise ValueError(f"{label}.git_commit must be 40 lowercase hexadecimal characters")
    if values["git_tree_state"] != "clean":
        raise ValueError(f"{label}.git_tree_state must be 'clean'")
    return TemplateRendererIdentity(**values)


def load_template_renderer_lock(project_root: Path) -> TemplateRendererIdentity:
    """Load and strictly validate ``rendering.template_renderer``."""
    config_path = Path(project_root).resolve() / "manuscript" / "config.yaml"
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("could not read manuscript renderer lock") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("manuscript/config.yaml top level must be a mapping")
    rendering = payload.get("rendering")
    if not isinstance(rendering, Mapping):
        raise ValueError("manuscript/config.yaml rendering block must be a mapping")
    if "template_renderer" not in rendering:
        raise ValueError("manuscript/config.yaml is missing rendering.template_renderer")
    return _identity_from_mapping(
        rendering["template_renderer"],
        label="manuscript/config.yaml rendering.template_renderer",
    )


def _run_git(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Template renderer checkout failed Git validation") from exc
    return completed.stdout


def _canonical_repository(remote: str) -> str:
    candidate = remote.strip()
    patterns = (
        re.compile(r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$", re.I),
        re.compile(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", re.I),
        re.compile(
            r"^ssh://git@github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$",
            re.I,
        ),
    )
    for pattern in patterns:
        match = pattern.fullmatch(candidate)
        if match is not None:
            return f"{match.group('owner')}/{match.group('repo')}"
    raise ValueError("Template renderer origin is not a supported canonical GitHub remote")


def inspect_template_renderer_checkout(template_root: Path) -> TemplateRendererIdentity:
    """Inspect an explicit Template checkout and return a path-free identity."""
    root = Path(template_root).expanduser().resolve()
    top_level_bytes = _run_git(root, "rev-parse", "--show-toplevel")
    try:
        top_level = Path(top_level_bytes.decode("utf-8").strip()).resolve()
    except UnicodeDecodeError as exc:
        raise ValueError("Template renderer Git top-level path is not UTF-8") from exc
    if top_level != root:
        raise ValueError("Template renderer path must be the exact Git checkout root")

    commit = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    if not _FULL_COMMIT.fullmatch(commit):
        raise ValueError("Template renderer HEAD is not a canonical full commit")

    remote = _run_git(root, "remote", "get-url", "origin").decode("utf-8").strip()
    repository = _canonical_repository(remote)
    if repository.casefold() != TEMPLATE_RENDERER_REPOSITORY.casefold():
        raise ValueError("Template renderer checkout has the wrong origin repository")

    status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise ValueError("Template renderer checkout must be completely clean")

    return TemplateRendererIdentity(
        schema_version=TEMPLATE_RENDERER_LOCK_SCHEMA,
        repository=TEMPLATE_RENDERER_REPOSITORY,
        git_commit=commit,
        git_tree_state="clean",
    )


def require_locked_template_renderer(
    project_root: Path,
    template_root: Path,
) -> TemplateRendererIdentity:
    """Require the runtime checkout to equal the source-owned renderer lock."""
    expected = load_template_renderer_lock(project_root)
    observed = inspect_template_renderer_checkout(template_root)
    if observed != expected:
        raise ValueError(
            "Template renderer checkout does not match the committed renderer lock: "
            f"expected {expected.git_commit}, observed {observed.git_commit}"
        )
    return observed


def renderer_receipt_findings(project_root: Path, payload: object) -> list[str]:
    """Return stable findings for a persisted renderer identity payload."""
    try:
        recorded = _identity_from_mapping(payload, label="render receipt renderer")
    except ValueError as exc:
        return [str(exc)]
    try:
        expected = load_template_renderer_lock(project_root)
    except ValueError as exc:
        return [str(exc)]
    if recorded != expected:
        return ["render receipt renderer does not match the committed renderer lock"]
    return []


__all__ = [
    "TEMPLATE_RENDERER_LOCK_SCHEMA",
    "TEMPLATE_RENDERER_REPOSITORY",
    "TemplateRendererIdentity",
    "inspect_template_renderer_checkout",
    "load_template_renderer_lock",
    "renderer_receipt_findings",
    "require_locked_template_renderer",
]
