"""Assemble figures only from admitted report checkpoints and import the result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis.stage_checkpoint import preserve_import_target
from analysis.stage_io import StageError, atomic_bytes, confined, digest
from analysis.stage_workspace import admit_workspace


def bundle_files(root: Path) -> dict[str, str]:
    """Use the existing producer's declared artifact boundary, not a second roster."""
    from analysis.artifacts import expected_artifacts
    from publication.pipeline_freshness import PIPELINE_STAGES

    spec = next(stage for stage in PIPELINE_STAGES if stage.name == "analysis")
    files = {
        path.relative_to(root).as_posix(): digest(path.read_bytes())
        for pattern in spec.output_patterns
        for path in root.glob(pattern)
        if path.is_file()
    }
    # Manuscript variables belong to downstream hydration, not this producer.
    for path in expected_artifacts(root).values():
        if path.name != "manuscript_variables.json" and path.relative_to(root).as_posix() not in files:
            raise StageError(f"analysis artifact missing: {path.name}")
    for name in files:
        if any(part.startswith(".") for part in Path(name).parts) or Path(name).suffix not in {
            ".json",
            ".md",
            ".png",
            ".pdf",
        }:
            raise StageError(f"recovery/nonartifact residue in analysis outputs: {name}")
        confined(root, name)
    return dict(sorted(files.items()))


def validate_bundle(workspace: Path, payload: dict[str, Any]) -> None:
    """Rehash the complete source-owned output inventory, including added/removed files."""
    if set(payload) != {"files"} or payload["files"] != bundle_files(workspace / "source"):
        raise StageError("analysis bundle inventory/digest mismatch")


def assemble_bundle(workspace: Path) -> dict[str, Any]:
    """Generate derivatives without rerunning any completed scientific report."""
    from analysis.workflow import run_analysis_pipeline

    manifest = admit_workspace(workspace)
    root = workspace / "source"
    run_analysis_pipeline(project_root=root, profile=manifest["profile"], checkpoints=workspace)
    return {"files": bundle_files(root)}


def import_bundle(workspace: Path, project: Path) -> dict[str, Any]:
    """Caller holds a short destination lease; partial imports cannot mint acceptance."""
    from analysis.stage_checkpoint import load_checkpoint
    from publication.pipeline_freshness import record_publication_analysis_stage

    manifest = admit_workspace(workspace, project)
    payload = load_checkpoint(workspace, "figures", "analysis_bundle", manifest["identity"])
    # Admit every target before making the first replacement.
    targets = {
        name: preserve_import_target(workspace, project, name, expected, manifest)
        for name, expected in payload["files"].items()
    }
    for name, expected in payload["files"].items():
        data = confined(workspace / "source", name).read_bytes()
        if digest(data) != expected:
            raise StageError(f"bundle changed during import: {name}")
        atomic_bytes(targets[name], data)
    admit_workspace(workspace, project)
    if manifest["profile"] == "publication":
        record_publication_analysis_stage(project)
    return {"imported": len(payload["files"]), "profile": manifest["profile"], "scientific_acceptance": False}
