"""Explicit receipt file loading and persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

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
    """Atomically persist a canonical run receipt."""
    receipt_path = Path(path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=receipt_path.parent,
            prefix=f".{receipt_path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(
                receipt.as_dict(),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, receipt_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return receipt_path



def write_application_receipt(
    path: str | Path,
    receipt: ApplicationReceipt,
) -> Path:
    """Atomically persist one canonical application receipt."""
    if not isinstance(receipt, ApplicationReceipt):
        raise ValueError("receipt must be an ApplicationReceipt")
    receipt_path = Path(path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=receipt_path.parent,
            prefix=f".{receipt_path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(
                receipt.as_dict(),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, receipt_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return receipt_path



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

