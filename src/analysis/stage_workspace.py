"""Freeze and admit exact analysis inputs before execution or import."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from analysis.stage_identity import output_baseline, runtime_identity, source_names, validate_source_tree
from analysis.stage_io import StageError, atomic_bytes, confined, digest, encode, read_json
from publication.pipeline_freshness import capture_analysis_input_snapshot


def prepare_workspace(project: Path, workspace: Path, *, profile: str) -> dict[str, Any]:
    """Freeze source/config/seed inputs under a short caller-held repository lease."""
    project, workspace = project.resolve(strict=True), workspace.absolute()
    if profile not in {"smoke", "publication"}:
        raise StageError("profile must be smoke or publication")
    if workspace.resolve().is_relative_to(project) or project.is_relative_to(workspace.resolve()):
        raise StageError("workspace must be independent of the project")
    if workspace.resolve() != workspace:
        raise StageError("workspace path must not traverse symlinks")
    validate_source_tree(project)
    snapshot = capture_analysis_input_snapshot(project)
    names = source_names(project, snapshot)
    hashes = {name: digest(confined(project, name).read_bytes()) for name in sorted(names)}
    workspace.mkdir()  # A failed/partial prior preparation is never silently replaced.
    for name, expected in hashes.items():
        data = confined(project, name).read_bytes()
        if digest(data) != expected:
            raise StageError(f"source changed during freeze: {name}")
        atomic_bytes(confined(workspace, f"source/{name}"), data)
    if capture_analysis_input_snapshot(project) != snapshot:
        raise StageError("analysis inputs changed during freeze")
    payload = {
        "schema": 1,
        "profile": profile,
        "files": hashes,
        "output_baseline": output_baseline(project),
        "analysis_snapshot": snapshot,
        "runtime": runtime_identity(),
    }
    # Explicit seed/config identity is also covered by exact config bytes.
    payload["identity"] = digest(encode(payload))
    atomic_bytes(workspace / "workspace.json", encode(payload))
    return payload


def admit_workspace(workspace: Path, project: Path | None = None) -> dict[str, Any]:
    """Revalidate the complete identity before compute, reuse or import."""
    if workspace.is_symlink() or workspace.resolve() != workspace.absolute():
        raise StageError("workspace must be a real canonical directory")
    manifest = read_json(confined(workspace, "workspace.json"))
    if set(manifest) != {
        "schema",
        "profile",
        "files",
        "analysis_snapshot",
        "runtime",
        "identity",
        "output_baseline",
    }:
        raise StageError("invalid workspace fields")
    if type(manifest["schema"]) is not int or manifest["schema"] != 1:
        raise StageError("unsupported workspace schema")
    if (
        not isinstance(manifest["profile"], str)
        or manifest["profile"] not in {"smoke", "publication"}
        or not isinstance(manifest["files"], dict)
    ):
        raise StageError("invalid workspace profile/files")
    if not isinstance(manifest["output_baseline"], dict):
        raise StageError("invalid output baseline")
    for name, value in manifest["output_baseline"].items():
        if (
            not name.startswith("output/")
            or not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
        ):
            raise StageError("invalid output baseline entry")
        confined(workspace / "source", name)
    identity = {key: value for key, value in manifest.items() if key != "identity"}
    if digest(encode(identity)) != manifest["identity"]:
        raise StageError("workspace identity mismatch")
    if encode(runtime_identity()) != encode(manifest["runtime"]):
        raise StageError("runtime identity changed")
    validate_source_tree(workspace / "source")
    if project is not None:
        validate_source_tree(project)
    for name, expected in manifest["files"].items():
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise StageError("invalid source digest")
        for root in (workspace / "source",) if project is None else (workspace / "source", project):
            try:
                actual = digest(confined(root, name).read_bytes())
            except OSError as exc:
                raise StageError(f"source input missing: {name}") from exc
            if actual != expected:
                raise StageError(f"source input changed: {name}")
    source = workspace / "source"
    snapshot = capture_analysis_input_snapshot(source)
    if snapshot != manifest["analysis_snapshot"]:
        raise StageError("source inventory changed")
    if source_names(source, snapshot) != set(manifest["files"]):
        raise StageError("source resource inventory changed")
    if project is not None and capture_analysis_input_snapshot(project) != manifest["analysis_snapshot"]:
        raise StageError("destination inputs changed")
    if project is not None and source_names(project, manifest["analysis_snapshot"]) != set(manifest["files"]):
        raise StageError("destination resource inventory changed")
    return manifest
