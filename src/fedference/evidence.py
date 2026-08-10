"""Versioned research specifications and evidence receipts.

The scientific extension lanes in Active Fedference have different proof and
evaluation obligations. This module provides one small, dependency-free
contract for declaring those obligations and binding completed runs to their
configuration, datasets, device, and output bytes. It does not decide whether a
scientific claim is true; it makes the evidence needed for that decision
explicit and machine-verifiable.

The contract follows the source/provenance boundary used throughout the
project: FedGVI (Mildner et al., 2025) and Friston et al. (2024) are source
bundles, while repository experiments must record their own estimand,
independent unit, falsifier, and no-claim outcome.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

EVIDENCE_SCHEMA_VERSION = "1.1"
ExperimentState = Literal["planned", "active", "complete", "external"]
RunStatus = Literal["completed", "failed", "partial"]
GitTreeState = Literal["clean", "dirty", "unavailable"]


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


def _reject_json_constant(value: str) -> Any:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _loads_json_strict(value: str) -> Any:
    """Decode standards-compliant JSON without Python's NaN extensions."""
    return json.loads(value, parse_constant=_reject_json_constant)


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value using canonical key and separator order."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of ``path`` without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EVIDENCE_SCHEMA_VERSION!r}")
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
        return {
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
                schema_version=raw["schema_version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid run receipt: {exc}") from exc


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


def load_run_receipt(path: str | Path) -> RunReceipt:
    """Load and validate a JSON run receipt."""
    try:
        raw = _loads_json_strict(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid run receipt file: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("run receipt must be a JSON object")
    return RunReceipt.from_dict(raw)


def _git_revision(project_root: Path) -> tuple[str, GitTreeState]:
    """Return the live full commit and tree state without invoking shell hooks."""
    commit = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedcache=false",
            "-C",
            str(project_root),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0 or not commit.stdout.strip():
        return "unavailable", "unavailable"
    status = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedcache=false",
            "-C",
            str(project_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    state: GitTreeState = (
        "unavailable"
        if status.returncode != 0
        else ("dirty" if status.stdout else "clean")
    )
    return commit.stdout.strip(), state


def verify_run_receipt(
    receipt: RunReceipt,
    *,
    root: str | Path,
    require_clean_git: bool = False,
    project_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return exact provenance/artifact findings; empty means verified.

    Development receipts may bind a dirty tree and still verify their config and
    output bytes. Publication callers should pass ``require_clean_git=True`` and
    ``project_root`` so both the recorded and live source tree, commit, and
    environment lock are checked.
    """
    findings: list[str] = []
    if receipt.status != "completed":
        findings.append(f"run status is {receipt.status!r}, not 'completed'")
    if require_clean_git and receipt.git_tree_state != "clean":
        findings.append(f"git tree state is {receipt.git_tree_state!r}, not 'clean'")
    if project_root is not None:
        source_root = Path(project_root).resolve()
        live_commit, live_state = _git_revision(source_root)
        if live_commit == "unavailable" or live_state == "unavailable":
            findings.append(f"live Git state is unavailable: {source_root}")
        else:
            if live_commit != receipt.git_commit:
                findings.append(
                    "live Git commit does not match receipt: "
                    f"{live_commit} != {receipt.git_commit}"
                )
            if live_state != receipt.git_tree_state:
                findings.append(
                    "live Git tree state does not match receipt: "
                    f"{live_state!r} != {receipt.git_tree_state!r}"
                )
            if require_clean_git and live_state != "clean":
                findings.append(f"live Git tree state is {live_state!r}, not 'clean'")
            if receipt.git_tree_state != "clean":
                findings.append(
                    "receipt binds an unhashed dirty tree; live source equivalence "
                    "cannot be verified"
                )
        lock_path = source_root / "uv.lock"
        if not lock_path.is_file():
            findings.append(f"live environment lock is missing: {lock_path}")
        elif sha256_file(lock_path) != receipt.environment_lock_sha256:
            findings.append("live environment lock digest does not match receipt")
    root_path = Path(root).resolve()
    for artifact in receipt.outputs:
        path = (root_path / artifact.path).resolve()
        try:
            path.relative_to(root_path)
        except ValueError:
            findings.append(f"artifact escapes receipt root: {artifact.path}")
            continue
        if not path.is_file():
            findings.append(f"missing artifact: {artifact.path}")
            continue
        if path.stat().st_size != artifact.bytes:
            findings.append(f"artifact byte-size mismatch: {artifact.path}")
        if sha256_file(path) != artifact.sha256:
            findings.append(f"artifact digest mismatch: {artifact.path}")
    config_records = [record for record in receipt.outputs if record.name == "config"]
    if len(config_records) == 1:
        config_path = (root_path / config_records[0].path).resolve()
        try:
            config_path.relative_to(root_path)
        except ValueError:
            pass
        else:
            if config_path.is_file():
                try:
                    config = _loads_json_strict(
                        config_path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    findings.append(f"config artifact is not valid JSON: {config_records[0].path}")
                else:
                    if canonical_sha256(config) != receipt.config_sha256:
                        findings.append(f"configuration hash mismatch: {config_records[0].path}")
    return tuple(findings)


__all__ = [
    "ArtifactRecord",
    "DatasetSpec",
    "EVIDENCE_SCHEMA_VERSION",
    "ExperimentSpec",
    "ExperimentState",
    "GitTreeState",
    "RunReceipt",
    "RunStatus",
    "SourceReference",
    "canonical_sha256",
    "load_run_receipt",
    "make_artifact_record",
    "sha256_file",
    "validate_evidence_report",
    "verify_run_receipt",
    "write_run_receipt",
]
