"""Validate complete report checkpoints and preserve guarded imports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from analysis.report_schemas import validate_report
from analysis.stage_io import StageError, atomic_bytes, confined, digest, encode, read_json
from analysis.stage_workspace import admit_workspace


def preserve_import_target(
    workspace: Path, project: Path, name: str, expected: str, manifest: dict[str, Any]
) -> Path:
    """Refuse concurrent output edits and retain the previous admitted bytes."""
    target = confined(project, name)
    old = target.read_bytes() if target.exists() else None
    actual = digest(old) if old is not None else None
    if actual != expected and actual != manifest["output_baseline"].get(name):
        raise StageError(f"destination output changed: {name}")
    if old is not None and actual != expected:
        backup = confined(workspace, f"prior-outputs/{actual}.bin")
        if backup.exists() and backup.read_bytes() != old:
            raise StageError("prior-output evidence corruption")
        if not backup.exists():
            atomic_bytes(backup, old)
    return target


def load_checkpoint(workspace: Path, stage: str, schema: str, identity: str) -> dict[str, Any]:
    """A filename alone is never a reusable checkpoint."""
    folder = confined(workspace, f"completed/{stage}")
    marker = read_json(confined(folder, "complete.json"))
    if (
        type(marker.get("schema")) is not int
        or set(marker) != {"schema", "stage", "identity", "report_sha256", "result_sha256"}
        or marker
        != {
            "schema": 1,
            "stage": stage,
            "identity": identity,
            "report_sha256": marker.get("report_sha256"),
            "result_sha256": marker.get("result_sha256"),
        }
    ):
        raise StageError(f"completion identity mismatch: {stage}")
    report = confined(folder, "report.json")
    result_file = confined(folder, "result.json")
    if digest(result_file.read_bytes()) != marker["result_sha256"]:
        raise StageError(f"checkpoint result digest mismatch: {stage}")
    result = read_json(result_file)
    elapsed = result.get("elapsed_seconds")
    if (
        result.get("status") != "complete"
        or type(result.get("returncode")) is not int
        or result["returncode"] != 0
        or result.get("identity") != identity
        or result.get("stage") != stage
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise StageError(f"invalid completed process receipt: {stage}")
    if digest(report.read_bytes()) != marker["report_sha256"]:
        raise StageError(f"checkpoint digest mismatch: {stage}")
    payload = read_json(report)
    if schema == "analysis_bundle":
        from analysis.stage_bundle import validate_bundle

        validate_bundle(workspace, payload)
    else:
        validate_report(schema, payload)
    return payload


def import_report(workspace: Path, project: Path, stage: str, schema: str) -> Path:
    """Import one checked report under a short repository lease; no analysis receipt."""
    manifest = admit_workspace(workspace, project)
    payload: Mapping[str, object] = load_checkpoint(workspace, stage, schema, manifest["identity"])
    from analysis.workflow import _write_json

    destination = preserve_import_target(
        workspace, project, f"output/reports/{stage}.json", digest(encode(payload)), manifest
    )
    return _write_json(payload, destination, schema=schema)
