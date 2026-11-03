"""Labeled aggregation receipt representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from ..provenance import (
    RuntimeProvenance,
    SourceProvenance,
)
from .schema import (
    APPLICATION_RECEIPT_SCHEMA_VERSION,
    ArtifactRecord,
    SolverStatus,
    _parse_utc_timestamp,
    _require_sha256,
)


@dataclass(frozen=True)
class ApplicationReceipt:
    """Integrity receipt for one labeled categorical aggregation operation.

    ``status='completed'`` means that execution and artifact writing completed;
    solver health remains a separate, mandatory field and is not a scientific
    validity or downstream-decision claim.
    """

    request_sha256: str
    config_fingerprint: str
    solver_status: SolverStatus
    fallback_events: tuple[str, ...]
    outputs: tuple[ArtifactRecord, ...]
    runtime_provenance: RuntimeProvenance
    source_provenance: SourceProvenance
    started_at_utc: str
    completed_at_utc: str
    status: Literal["completed"] = "completed"
    receipt_type: Literal["application"] = "application"
    operation: Literal["labeled_categorical_aggregation"] = (
        "labeled_categorical_aggregation"
    )
    schema_version: str = APPLICATION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != APPLICATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {APPLICATION_RECEIPT_SCHEMA_VERSION!r}"
            )
        if self.receipt_type != "application":
            raise ValueError("receipt_type must be 'application'")
        if self.operation != "labeled_categorical_aggregation":
            raise ValueError("operation is not recognized")
        if self.status != "completed":
            raise ValueError("application receipt status must be 'completed'")
        _require_sha256(self.request_sha256, "request_sha256")
        _require_sha256(self.config_fingerprint, "config_fingerprint")
        if self.solver_status not in (
            "nominal",
            "converged_with_fallback",
            "not_converged",
            "not_converged_with_fallback",
        ):
            raise ValueError("solver_status is not recognized")
        if not isinstance(self.fallback_events, (tuple, list)) or any(
            not isinstance(event, str) or not event.strip()
            for event in self.fallback_events
        ):
            raise ValueError("fallback_events must contain non-empty strings")
        fallbacks = tuple(self.fallback_events)
        status_has_fallback = self.solver_status in (
            "converged_with_fallback",
            "not_converged_with_fallback",
        )
        if status_has_fallback != bool(fallbacks):
            raise ValueError("solver_status and fallback_events are inconsistent")
        if not isinstance(self.outputs, (tuple, list)) or any(
            not isinstance(record, ArtifactRecord) for record in self.outputs
        ):
            raise ValueError("outputs must contain ArtifactRecord values")
        outputs = tuple(self.outputs)
        if tuple(record.name for record in outputs) != ("request", "result"):
            raise ValueError(
                "application receipts must bind exactly request and result artifacts"
            )
        if len({record.path for record in outputs}) != 2:
            raise ValueError("application receipt artifact paths must be unique")
        if tuple(record.path for record in outputs) != ("request.json", "result.json"):
            raise ValueError(
                "application receipt artifacts must be request.json and result.json"
            )
        if not isinstance(self.runtime_provenance, RuntimeProvenance):
            raise ValueError("runtime_provenance must be a RuntimeProvenance")
        if not isinstance(self.source_provenance, SourceProvenance):
            raise ValueError("source_provenance must be a SourceProvenance")
        started = _parse_utc_timestamp(self.started_at_utc, "started_at_utc")
        completed = _parse_utc_timestamp(self.completed_at_utc, "completed_at_utc")
        if completed < started:
            raise ValueError("completed_at_utc must not precede started_at_utc")
        object.__setattr__(self, "request_sha256", self.request_sha256.lower())
        object.__setattr__(self, "config_fingerprint", self.config_fingerprint.lower())
        object.__setattr__(self, "fallback_events", fallbacks)
        object.__setattr__(self, "outputs", outputs)

    def as_dict(self) -> dict[str, Any]:
        """Return the complete strict application-receipt representation."""
        return {
            "receipt_type": self.receipt_type,
            "schema_version": self.schema_version,
            "operation": self.operation,
            "status": self.status,
            "runtime_provenance": self.runtime_provenance.as_dict(),
            "source_provenance": self.source_provenance.as_dict(),
            "request_sha256": self.request_sha256,
            "config_fingerprint": self.config_fingerprint,
            "solver_status": self.solver_status,
            "fallback_events": list(self.fallback_events),
            "outputs": [asdict(record) for record in self.outputs],
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ApplicationReceipt:
        """Decode a fail-closed application receipt."""
        required = {
            "receipt_type",
            "schema_version",
            "operation",
            "status",
            "runtime_provenance",
            "source_provenance",
            "request_sha256",
            "config_fingerprint",
            "solver_status",
            "fallback_events",
            "outputs",
            "started_at_utc",
            "completed_at_utc",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("application receipt fields do not match schema")
        if not isinstance(raw["fallback_events"], list):
            raise ValueError("fallback_events must be a list")
        outputs_raw = raw["outputs"]
        if not isinstance(outputs_raw, list) or any(
            not isinstance(item, dict) for item in outputs_raw
        ):
            raise ValueError("outputs must be a list")
        try:
            return cls(
                receipt_type=raw["receipt_type"],
                schema_version=raw["schema_version"],
                operation=raw["operation"],
                status=raw["status"],
                runtime_provenance=RuntimeProvenance.from_dict(
                    raw["runtime_provenance"]
                ),
                source_provenance=SourceProvenance.from_dict(
                    raw["source_provenance"]
                ),
                request_sha256=raw["request_sha256"],
                config_fingerprint=raw["config_fingerprint"],
                solver_status=raw["solver_status"],
                fallback_events=tuple(raw["fallback_events"]),
                outputs=tuple(ArtifactRecord(**item) for item in outputs_raw),
                started_at_utc=raw["started_at_utc"],
                completed_at_utc=raw["completed_at_utc"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid application receipt: {exc}") from exc

