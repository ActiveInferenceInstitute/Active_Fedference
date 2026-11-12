"""Explicit receipt file loading and persistence."""

from __future__ import annotations

from pathlib import Path

from ._atomic import publish_json
from .application_schema import ApplicationReceipt
from .hashing import sha256_file
from .run_schema import RunReceipt
from .schema import ArtifactRecord
from .strict_json import _loads_json_strict


def make_artifact_record(
    name: str,
    path: str | Path,
    *,
    root: str | Path,
) -> ArtifactRecord:
    """Create a receipt record for an existing file below ``root``."""
    root_path = Path(root).resolve()
    artifact_path = Path(path).resolve()
    try:
        relative = artifact_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("artifact must be below the receipt root") from exc
    if not artifact_path.is_file():
        raise ValueError(f"artifact is not a file: {artifact_path}")
    return ArtifactRecord(
        name=name,
        path=relative.as_posix(),
        sha256=sha256_file(artifact_path),
        bytes=artifact_path.stat().st_size,
    )



def write_run_receipt(path: str | Path, receipt: RunReceipt) -> Path:
    """Exclusively publish a canonical run receipt."""
    return publish_json(path, receipt.as_dict())



def write_application_receipt(
    path: str | Path,
    receipt: ApplicationReceipt,
) -> Path:
    """Exclusively publish one canonical application receipt."""
    if not isinstance(receipt, ApplicationReceipt):
        raise ValueError("receipt must be an ApplicationReceipt")
    return publish_json(path, receipt.as_dict())



def load_run_receipt(path: str | Path) -> RunReceipt:
    """Load and validate a JSON run receipt."""
    try:
        raw = _loads_json_strict(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid run receipt file: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("run receipt must be a JSON object")
    return RunReceipt.from_dict(raw)



def load_application_receipt(path: str | Path) -> ApplicationReceipt:
    """Load and validate one strict application receipt."""
    try:
        raw = _loads_json_strict(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid application receipt file: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("application receipt must be a JSON object")
    return ApplicationReceipt.from_dict(raw)



def load_receipt(path: str | Path) -> RunReceipt | ApplicationReceipt:
    """Auto-detect and load a research or application receipt."""
    try:
        raw = _loads_json_strict(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid receipt file: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("receipt must be a JSON object")
    if raw.get("receipt_type") == "application":
        return ApplicationReceipt.from_dict(raw)
    return RunReceipt.from_dict(raw)

