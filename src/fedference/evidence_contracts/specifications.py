"""Research specification declarations and report contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from .schema import ExperimentState, _require_sha256, _require_text


def validate_evidence_report(payload: Mapping[str, object]) -> None:
    """Validate the typed top-level contract for an executable run report.

    Analysis reports have per-file schemas in ``analysis.report_schemas``.
    CLI evidence reports intentionally remain lane-specific, but every lane
    must still declare its status, estimand, independent unit, and no-claim
    boundary before the report can be bound into a receipt.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("evidence report must be a mapping")
    required = ("status", "primary_estimand", "independent_unit", "no_claim")
    for field in required:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"evidence report field {field!r} must be a non-empty string")
    rows = payload.get("rows")
    if rows is not None and not isinstance(rows, list):
        raise ValueError("evidence report field 'rows' must be a list when present")
    controls = payload.get("negative_controls")
    if controls is not None and not isinstance(controls, Mapping):
        raise ValueError("evidence report field 'negative_controls' must be a mapping when present")



@dataclass(frozen=True)
class SourceReference:
    """Pinned scholarly or implementation source used by an experiment."""

    source_id: str
    title: str
    url: str
    role: str
    doi: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_id", "title", "url", "role"):
            _require_text(getattr(self, name), name)
        if self.doi is not None:
            _require_text(self.doi, "doi")
        if self.revision is not None:
            _require_text(self.revision, "revision")



@dataclass(frozen=True)
class DatasetSpec:
    """Legally and byte-level reproducible external dataset declaration."""

    dataset_id: str
    name: str
    source_url: str
    doi: str
    license: str
    archive_sha256: str
    archive_member: str
    file_format: Literal["csv", "arff"]
    n_rows: int
    n_features: int
    n_classes: int
    has_missing_values: bool
    preprocessing: tuple[str, ...]
    schema: tuple[str, ...]
    split_policy: str
    split_seed: int | None = None
    split_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.preprocessing, (tuple, list)):
            raise ValueError("preprocessing must be a sequence")
        if not isinstance(self.schema, (tuple, list)):
            raise ValueError("schema must be a sequence")
        object.__setattr__(self, "preprocessing", tuple(self.preprocessing))
        object.__setattr__(self, "schema", tuple(self.schema))
        for name in (
            "dataset_id",
            "name",
            "source_url",
            "doi",
            "license",
            "archive_member",
            "split_policy",
        ):
            _require_text(getattr(self, name), name)
        _require_sha256(self.archive_sha256, "archive_sha256")
        if self.file_format not in ("csv", "arff"):
            raise ValueError("file_format must be 'csv' or 'arff'")
        for name in ("n_rows", "n_features", "n_classes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.has_missing_values, bool):
            raise ValueError("has_missing_values must be a boolean")
        if not self.preprocessing or any(
            not isinstance(step, str) or not step.strip() for step in self.preprocessing
        ):
            raise ValueError("preprocessing must contain non-empty steps")
        if not self.schema or any(not isinstance(field, str) or not field.strip() for field in self.schema):
            raise ValueError("schema must contain non-empty field declarations")
        if self.split_seed is not None and (
            isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int) or self.split_seed < 0
        ):
            raise ValueError("split_seed must be a non-negative integer or None")
        if self.split_sha256 is not None:
            _require_sha256(self.split_sha256, "split_sha256")
            if self.split_seed is None:
                raise ValueError("split_sha256 requires a declared split_seed")



@dataclass(frozen=True)
class ExperimentSpec:
    """Decision-complete declaration for one research experiment family."""

    experiment_id: str
    version: str
    title: str
    state: ExperimentState
    source_ids: tuple[str, ...]
    primary_estimand: str
    independent_unit: str
    falsifier: str
    no_claim: str
    profiles: tuple[str, ...]
    smallest_effect_of_interest: str
    mcse_stopping_target: str
    maximum_budget: str
    comparison_family: str
    confirmatory_ready: bool = False
    runner: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_ids, (tuple, list)):
            raise ValueError("source_ids must be a sequence")
        if not isinstance(self.profiles, (tuple, list)):
            raise ValueError("profiles must be a sequence")
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        object.__setattr__(self, "profiles", tuple(self.profiles))
        for name in (
            "experiment_id",
            "version",
            "title",
            "primary_estimand",
            "independent_unit",
            "falsifier",
            "no_claim",
            "smallest_effect_of_interest",
            "mcse_stopping_target",
            "maximum_budget",
            "comparison_family",
        ):
            _require_text(getattr(self, name), name)
        if self.state not in ("planned", "active", "complete", "external"):
            raise ValueError("state is not recognized")
        if (
            not self.source_ids
            or any(not isinstance(source_id, str) or not source_id.strip() for source_id in self.source_ids)
            or len(set(self.source_ids)) != len(self.source_ids)
        ):
            raise ValueError("source_ids must be non-empty and unique")
        if (
            not self.profiles
            or any(not isinstance(profile, str) or not profile.strip() for profile in self.profiles)
            or len(set(self.profiles)) != len(self.profiles)
        ):
            raise ValueError("profiles must be non-empty and unique")
        if not isinstance(self.confirmatory_ready, bool):
            raise ValueError("confirmatory_ready must be a boolean")
        if self.runner is not None:
            _require_text(self.runner, "runner")

