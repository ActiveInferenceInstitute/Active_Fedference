"""Versioned research run receipt representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import Any, Mapping

from ..provenance import (
    RuntimeProvenance,
    runtime_provenance,
)
from .schema import (
    EVIDENCE_SCHEMA_VERSION,
    LEGACY_EVIDENCE_SCHEMA_VERSION,
    ArtifactRecord,
    GitTreeState,
    RunStatus,
    _parse_utc_timestamp,
    _require_sha256,
    _require_text,
)


@dataclass(frozen=True)
class RunReceipt:
    """Content-bound receipt for one executed experiment profile."""

    run_id: str
    experiment_id: str
    experiment_version: str
    profile: str
    git_commit: str
    git_tree_state: GitTreeState
    environment_lock_sha256: str
    config_sha256: str
    dataset_sha256: Mapping[str, str]
    seeds: tuple[int, ...]
    device: str
    backend: str
    fallbacks: tuple[str, ...]
    checkpoints: tuple[str, ...]
    outputs: tuple[ArtifactRecord, ...]
    status: RunStatus
    started_at_utc: str
    completed_at_utc: str
    runtime_provenance: RuntimeProvenance = dataclass_field(
        default_factory=runtime_provenance
    )
    schema_version: str = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_sha256, Mapping):
            raise ValueError("dataset_sha256 must be a mapping")
        for name in ("seeds", "fallbacks", "checkpoints", "outputs"):
            if not isinstance(getattr(self, name), (tuple, list)):
                raise ValueError(f"{name} must be a sequence")
        object.__setattr__(
            self,
            "dataset_sha256",
            MappingProxyType(dict(self.dataset_sha256)),
        )
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "fallbacks", tuple(self.fallbacks))
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        for name in (
            "run_id",
            "experiment_id",
            "experiment_version",
            "profile",
            "device",
            "backend",
        ):
            _require_text(getattr(self, name), name)
        if self.git_commit != "unavailable" and (
            len(self.git_commit) != 40
            or any(character not in "0123456789abcdefABCDEF" for character in self.git_commit)
        ):
            raise ValueError("git_commit must be a full 40-character commit hash or 'unavailable'")
        if self.git_tree_state not in ("clean", "dirty", "unavailable"):
            raise ValueError("git_tree_state is not recognized")
        if self.git_commit == "unavailable" and self.git_tree_state != "unavailable":
            raise ValueError("an unavailable git commit requires git_tree_state='unavailable'")
        if self.schema_version not in (
            LEGACY_EVIDENCE_SCHEMA_VERSION,
            EVIDENCE_SCHEMA_VERSION,
        ):
            raise ValueError(
                "schema_version must be "
                f"{LEGACY_EVIDENCE_SCHEMA_VERSION!r} or {EVIDENCE_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.runtime_provenance, RuntimeProvenance):
            raise ValueError("runtime_provenance must be a RuntimeProvenance")
        if self.schema_version == LEGACY_EVIDENCE_SCHEMA_VERSION:
            object.__setattr__(
                self,
                "runtime_provenance",
                RuntimeProvenance.unavailable(
                    warning="runtime provenance is unavailable in legacy schema 1.1"
                ),
            )
        _require_sha256(self.environment_lock_sha256, "environment_lock_sha256")
        _require_sha256(self.config_sha256, "config_sha256")
        for dataset_id, digest in self.dataset_sha256.items():
            _require_text(dataset_id, "dataset id")
            _require_sha256(digest, f"dataset_sha256[{dataset_id!r}]")
        if not self.seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds
        ):
            raise ValueError("seeds must contain non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique within a run receipt")
        if self.status not in ("completed", "failed", "partial"):
            raise ValueError("status is not recognized")
        if any(not isinstance(value, str) or not value.strip() for value in self.fallbacks):
            raise ValueError("fallbacks must contain non-empty strings")
        if any(not isinstance(value, str) or not value.strip() for value in self.checkpoints):
            raise ValueError("checkpoints must contain non-empty strings")
        if any(not isinstance(record, ArtifactRecord) for record in self.outputs):
            raise ValueError("outputs must contain ArtifactRecord values")
        if self.status == "completed" and not self.outputs:
            raise ValueError("completed receipts must declare at least one output")
        output_names = [record.name for record in self.outputs]
        output_paths = [record.path for record in self.outputs]
        if len(set(output_names)) != len(output_names):
            raise ValueError("output artifact names must be unique")
        if len(set(output_paths)) != len(output_paths):
            raise ValueError("output artifact paths must be unique")
        if self.status == "completed" and output_names.count("config") != 1:
            raise ValueError("completed receipts must declare exactly one 'config' artifact")
        started = _parse_utc_timestamp(self.started_at_utc, "started_at_utc")
        completed = _parse_utc_timestamp(self.completed_at_utc, "completed_at_utc")
        if completed < started:
            raise ValueError("completed_at_utc must not precede started_at_utc")

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""
        payload = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "experiment_version": self.experiment_version,
            "profile": self.profile,
            "git_commit": self.git_commit,
            "git_tree_state": self.git_tree_state,
            "environment_lock_sha256": self.environment_lock_sha256,
            "config_sha256": self.config_sha256,
            "dataset_sha256": dict(sorted(self.dataset_sha256.items())),
            "seeds": list(self.seeds),
            "device": self.device,
            "backend": self.backend,
            "fallbacks": list(self.fallbacks),
            "checkpoints": list(self.checkpoints),
            "outputs": [asdict(record) for record in self.outputs],
            "status": self.status,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "schema_version": self.schema_version,
        }
        if self.schema_version == EVIDENCE_SCHEMA_VERSION:
            payload["runtime_provenance"] = self.runtime_provenance.as_dict()
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RunReceipt:
        """Fail closed while constructing a receipt from decoded JSON."""
        required = {
            "run_id",
            "experiment_id",
            "experiment_version",
            "profile",
            "git_commit",
            "git_tree_state",
            "environment_lock_sha256",
            "config_sha256",
            "dataset_sha256",
            "seeds",
            "device",
            "backend",
            "fallbacks",
            "checkpoints",
            "outputs",
            "status",
            "started_at_utc",
            "completed_at_utc",
            "schema_version",
        }
        schema_version = raw.get("schema_version")
        if schema_version == EVIDENCE_SCHEMA_VERSION:
            required.add("runtime_provenance")
        elif schema_version != LEGACY_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("run receipt schema_version is not supported")
        if set(raw) != required:
            missing = sorted(required - set(raw))
            extra = sorted(set(raw) - required)
            raise ValueError(f"run receipt fields do not match schema; missing={missing}, extra={extra}")
        outputs_raw = raw["outputs"]
        if not isinstance(outputs_raw, list) or any(not isinstance(item, dict) for item in outputs_raw):
            raise ValueError("outputs must be a list")
        if not isinstance(raw["dataset_sha256"], dict):
            raise ValueError("dataset_sha256 must be an object")
        for field in ("seeds", "fallbacks", "checkpoints"):
            if not isinstance(raw[field], list):
                raise ValueError(f"{field} must be a list")
        try:
            outputs = tuple(ArtifactRecord(**item) for item in outputs_raw)
            datasets = dict(raw["dataset_sha256"])
            provenance = (
                RuntimeProvenance.from_dict(raw["runtime_provenance"])
                if schema_version == EVIDENCE_SCHEMA_VERSION
                else RuntimeProvenance.unavailable(
                    warning="runtime provenance is unavailable in legacy schema 1.1"
                )
            )
            return cls(
                run_id=raw["run_id"],
                experiment_id=raw["experiment_id"],
                experiment_version=raw["experiment_version"],
                profile=raw["profile"],
                git_commit=raw["git_commit"],
                git_tree_state=raw["git_tree_state"],
                environment_lock_sha256=raw["environment_lock_sha256"],
                config_sha256=raw["config_sha256"],
                dataset_sha256=datasets,
                seeds=tuple(raw["seeds"]),
                device=raw["device"],
                backend=raw["backend"],
                fallbacks=tuple(raw["fallbacks"]),
                checkpoints=tuple(raw["checkpoints"]),
                outputs=outputs,
                status=raw["status"],
                started_at_utc=raw["started_at_utc"],
                completed_at_utc=raw["completed_at_utc"],
                runtime_provenance=provenance,
                schema_version=raw["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid run receipt: {exc}") from exc

