"""Shared receipt field invariants and artifact records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

EVIDENCE_SCHEMA_VERSION = "1.2"



LEGACY_EVIDENCE_SCHEMA_VERSION = "1.1"



APPLICATION_RECEIPT_SCHEMA_VERSION = "1.0"



ExperimentState = Literal["planned", "active", "complete", "external"]



RunStatus = Literal["completed", "failed", "partial"]



GitTreeState = Literal["clean", "dirty", "unavailable"]



SolverStatus = Literal[
    "nominal",
    "converged_with_fallback",
    "not_converged",
    "not_converged_with_fallback",
]



ApplicationVerificationScope = Literal["all", "artifacts", "source", "solver"]



def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")



def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True



def _require_sha256(value: str, name: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")



def _parse_utc_timestamp(value: str, name: str) -> datetime:
    _require_text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must carry an explicit UTC offset")
    return parsed



@dataclass(frozen=True)
class ArtifactRecord:
    """One output file bound into a run receipt."""

    name: str
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _require_text(self.name, "artifact name")
        _require_text(self.path, "artifact path")
        relative = Path(self.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path must be relative and remain below the receipt root")
        _require_sha256(self.sha256, "artifact sha256")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise ValueError("artifact bytes must be a non-negative integer")

