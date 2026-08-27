"""Runtime and source provenance for application and research receipts.

The application boundary may run from an explicit Git checkout or from an
installed wheel.  These records distinguish those cases without retaining a
machine-local PEP 610 URL and without inventing a source revision when the
installed archive does not carry one.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

_DISTRIBUTION_NAME = "active_fedference"
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_REQUIREMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")

SourceKind = Literal["git_checkout", "installed_archive", "unavailable"]


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 hex digest or None")
    return value.lower()


def _warnings(values: object) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError("warnings must be a sequence of non-empty strings")
    return tuple(values)


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def active_fedference_checkout_version(project_root: str | Path) -> str | None:
    """Return the declared version for a positively identified source checkout.

    Identity requires both the ``[project]`` distribution name and the core
    source layout. This prevents an unrelated Git repository from being bound
    into an Active Fedference application receipt merely because it has a
    commit and a lockfile.
    """
    root = Path(project_root).resolve()
    pyproject = root / "pyproject.toml"
    required_sources = (
        root / "src" / "fedference" / "__init__.py",
        root / "src" / "fedference" / "aggregation.py",
    )
    if not pyproject.is_file() or not all(path.is_file() for path in required_sources):
        return None
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    project_match = re.search(
        r"(?ms)^\[project\]\s*$\n(?P<body>.*?)(?=^\[|\Z)",
        text,
    )
    if project_match is None:
        return None
    body = project_match.group("body")
    name_match = re.search(
        r"(?m)^\s*name\s*=\s*['\"](?P<value>[^'\"]+)['\"]\s*(?:#.*)?$",
        body,
    )
    version_match = re.search(
        r"(?m)^\s*version\s*=\s*['\"](?P<value>[^'\"]+)['\"]\s*(?:#.*)?$",
        body,
    )
    if name_match is None or version_match is None:
        return None
    if _canonical_distribution_name(name_match.group("value")) != "active-fedference":
        return None
    version = version_match.group("value")
    return version if version.strip() == version and version else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_sha256(distribution: metadata.Distribution) -> tuple[str | None, tuple[str, ...]]:
    """Read only the archive digest from PEP 610 metadata, never its URL."""
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        return None, ()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None, ("installed distribution has malformed PEP 610 metadata",)
    if not isinstance(payload, dict):
        return None, ("installed distribution has malformed PEP 610 metadata",)
    archive = payload.get("archive_info")
    if not isinstance(archive, dict):
        return None, ()
    hashes = archive.get("hashes")
    candidate: object = hashes.get("sha256") if isinstance(hashes, dict) else None
    if candidate is None:
        legacy_hash = archive.get("hash")
        if isinstance(legacy_hash, str) and legacy_hash.startswith("sha256="):
            candidate = legacy_hash.removeprefix("sha256=")
    if candidate is None:
        return None, ()
    try:
        return _optional_sha256(candidate, "PEP 610 archive sha256"), ()
    except ValueError:
        return None, ("installed distribution has an invalid PEP 610 archive digest",)


@dataclass(frozen=True)
class RuntimeProvenance:
    """Installed distribution, interpreter, and declared dependency versions."""

    distribution_name: str
    distribution_version: str
    python_version: str
    dependency_versions: Mapping[str, str]
    installed_archive_sha256: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.distribution_name, "distribution_name")
        _require_text(self.distribution_version, "distribution_version")
        _require_text(self.python_version, "python_version")
        if not isinstance(self.dependency_versions, Mapping):
            raise ValueError("dependency_versions must be a mapping")
        dependencies: dict[str, str] = {}
        for name, version in self.dependency_versions.items():
            key = _require_text(name, "dependency name")
            dependencies[key] = _require_text(version, f"dependency version for {key!r}")
        object.__setattr__(
            self,
            "dependency_versions",
            MappingProxyType(dict(sorted(dependencies.items()))),
        )
        object.__setattr__(
            self,
            "installed_archive_sha256",
            _optional_sha256(self.installed_archive_sha256, "installed_archive_sha256"),
        )
        object.__setattr__(self, "warnings", _warnings(self.warnings))

    @classmethod
    def unavailable(cls, *, warning: str) -> RuntimeProvenance:
        """Construct explicit unavailable provenance for a legacy receipt."""
        return cls(
            distribution_name=_DISTRIBUTION_NAME,
            distribution_version="unavailable",
            python_version="unavailable",
            dependency_versions={},
            warnings=(warning,),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the strict JSON-compatible representation."""
        return {
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
            "python_version": self.python_version,
            "dependency_versions": dict(self.dependency_versions),
            "installed_archive_sha256": self.installed_archive_sha256,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RuntimeProvenance:
        """Decode one fail-closed runtime-provenance object."""
        required = {
            "distribution_name",
            "distribution_version",
            "python_version",
            "dependency_versions",
            "installed_archive_sha256",
            "warnings",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("runtime provenance fields do not match schema")
        return cls(
            distribution_name=raw["distribution_name"],
            distribution_version=raw["distribution_version"],
            python_version=raw["python_version"],
            dependency_versions=raw["dependency_versions"],
            installed_archive_sha256=raw["installed_archive_sha256"],
            warnings=raw["warnings"],
        )


def runtime_provenance() -> RuntimeProvenance:
    """Collect installed runtime versions without retaining local source URLs."""
    warnings: list[str] = []
    dependencies: dict[str, str] = {}
    archive_sha256: str | None = None
    try:
        distribution = metadata.distribution(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return RuntimeProvenance(
            distribution_name=_DISTRIBUTION_NAME,
            distribution_version="unavailable",
            python_version=platform.python_version(),
            dependency_versions={},
            warnings=("active_fedference distribution metadata is unavailable",),
        )
    for requirement in distribution.requires or ():
        if "extra ==" in requirement or "extra==" in requirement:
            continue
        match = _REQUIREMENT_NAME_PATTERN.match(requirement)
        if match is None:
            warnings.append(f"could not parse declared runtime requirement: {requirement}")
            continue
        name = _canonical_distribution_name(match.group(0))
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = "unavailable"
            warnings.append(f"declared runtime dependency version is unavailable: {name}")
    archive_sha256, archive_warnings = _archive_sha256(distribution)
    warnings.extend(archive_warnings)
    return RuntimeProvenance(
        distribution_name=_DISTRIBUTION_NAME,
        distribution_version=distribution.version,
        python_version=platform.python_version(),
        dependency_versions=dependencies,
        installed_archive_sha256=archive_sha256,
        warnings=tuple(dict.fromkeys(warnings)),
    )


@dataclass(frozen=True)
class SourceProvenance:
    """Caller-selected Git source or installed-archive provenance."""

    source_kind: SourceKind
    git_commit: str | None = None
    git_tree_state: Literal["clean", "dirty"] | None = None
    uv_lock_sha256: str | None = None
    installed_archive_sha256: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source_kind not in ("git_checkout", "installed_archive", "unavailable"):
            raise ValueError("source_kind is not recognized")
        if self.git_commit is not None and (
            not isinstance(self.git_commit, str)
            or re.fullmatch(r"[0-9a-fA-F]{40}", self.git_commit) is None
        ):
            raise ValueError("git_commit must be a full Git commit or None")
        if self.git_tree_state not in (None, "clean", "dirty"):
            raise ValueError("git_tree_state must be clean, dirty, or None")
        lock_digest = _optional_sha256(self.uv_lock_sha256, "uv_lock_sha256")
        archive_digest = _optional_sha256(
            self.installed_archive_sha256,
            "installed_archive_sha256",
        )
        if self.source_kind == "git_checkout" and (
            self.git_commit is None or self.git_tree_state is None
        ):
            raise ValueError("git_checkout provenance requires commit and tree state")
        if self.source_kind != "git_checkout" and any(
            value is not None for value in (self.git_commit, self.git_tree_state, lock_digest)
        ):
            raise ValueError("non-Git provenance cannot declare Git fields")
        if self.source_kind == "installed_archive" and archive_digest is None:
            raise ValueError("installed_archive provenance requires an archive digest")
        if self.source_kind == "unavailable" and archive_digest is not None:
            raise ValueError("unavailable provenance cannot declare an archive digest")
        object.__setattr__(self, "git_commit", self.git_commit.lower() if self.git_commit else None)
        object.__setattr__(self, "uv_lock_sha256", lock_digest)
        object.__setattr__(self, "installed_archive_sha256", archive_digest)
        object.__setattr__(self, "warnings", _warnings(self.warnings))

    def as_dict(self) -> dict[str, Any]:
        """Return the strict JSON-compatible representation."""
        return {
            "source_kind": self.source_kind,
            "git_commit": self.git_commit,
            "git_tree_state": self.git_tree_state,
            "uv_lock_sha256": self.uv_lock_sha256,
            "installed_archive_sha256": self.installed_archive_sha256,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SourceProvenance:
        """Decode one fail-closed source-provenance object."""
        required = {
            "source_kind",
            "git_commit",
            "git_tree_state",
            "uv_lock_sha256",
            "installed_archive_sha256",
            "warnings",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("source provenance fields do not match schema")
        return cls(
            source_kind=raw["source_kind"],
            git_commit=raw["git_commit"],
            git_tree_state=raw["git_tree_state"],
            uv_lock_sha256=raw["uv_lock_sha256"],
            installed_archive_sha256=raw["installed_archive_sha256"],
            warnings=raw["warnings"],
        )


def _git_source(project_root: Path) -> SourceProvenance | None:
    root_result = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedcache=false",
            "-C",
            str(project_root),
            "rev-parse",
            "--show-toplevel",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if root_result.returncode != 0 or not root_result.stdout.strip():
        return None
    root = Path(root_result.stdout.strip()).resolve()
    if active_fedference_checkout_version(root) is None:
        return None
    commit_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status_result = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedcache=false",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if (
        commit_result.returncode != 0
        or re.fullmatch(r"[0-9a-fA-F]{40}", commit_result.stdout.strip()) is None
        or status_result.returncode != 0
    ):
        return None
    warnings: list[str] = []
    lock_path = root / "uv.lock"
    lock_digest = _sha256_file(lock_path) if lock_path.is_file() else None
    if lock_digest is None:
        warnings.append("uv.lock is unavailable in the selected Git checkout")
    return SourceProvenance(
        source_kind="git_checkout",
        git_commit=commit_result.stdout.strip(),
        git_tree_state="dirty" if status_result.stdout else "clean",
        uv_lock_sha256=lock_digest,
        warnings=tuple(warnings),
    )


def collect_source_provenance(project_root: str | Path | None = None) -> SourceProvenance:
    """Collect explicit Git provenance, then archive provenance, else unavailable."""
    warnings: list[str] = []
    if project_root is not None:
        source = _git_source(Path(project_root).resolve())
        if source is not None:
            return source
        warnings.append(
            "the explicitly selected project root is not a readable "
            "Active Fedference Git checkout"
        )
    runtime = runtime_provenance()
    if runtime.installed_archive_sha256 is not None:
        return SourceProvenance(
            source_kind="installed_archive",
            installed_archive_sha256=runtime.installed_archive_sha256,
            warnings=tuple(warnings),
        )
    warnings.append("installed archive provenance is unavailable")
    return SourceProvenance(
        source_kind="unavailable",
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = [
    "RuntimeProvenance",
    "SourceKind",
    "SourceProvenance",
    "active_fedference_checkout_version",
    "collect_source_provenance",
    "runtime_provenance",
]
