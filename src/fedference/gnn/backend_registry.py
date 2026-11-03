"""Typed admission for named GNN backend runtimes (Friston et al., 2024)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from ._documents import UPSTREAM_REVISION, _json, _mapping
from .admission import GNNAdmissionError
from .types import JSONValue

RTOL: Final = 1e-5
ATOL: Final = 1e-6
Qualification: TypeAlias = Literal[
    "fresh-provisioning-and-execution", "existing-runtime-execution"
]


@dataclass(frozen=True, slots=True)
class BackendProfile:
    """One admitted runtime identity, separate from its serialization."""

    python: str
    distributions: Mapping[str, str]
    qualification: Qualification


@dataclass(frozen=True, slots=True)
class BackendRegistry:
    """Immutable admitted views and an isolated JSON projection."""

    _document: dict[str, JSONValue] = field(repr=False)
    source_files: Mapping[str, str]
    profiles: Mapping[str, BackendProfile]

    def as_dict(self) -> dict[str, JSONValue]:
        """Return writable serialization without exposing admitted state."""
        return deepcopy(self._document)


def read_backend_document(path: Path) -> dict[str, JSONValue]:
    """Read one strict backend JSON boundary with its existing finding code."""
    try:
        return _mapping(_json(path.read_text(encoding="utf-8")), str(path))
    except (OSError, UnicodeError) as error:
        raise GNNAdmissionError("GNN_BACKEND_OUTPUT", str(path)) from error


def load_backend_registry(path: Path) -> BackendRegistry:
    """Parse registry semantics once and retain their admitted field types."""
    registry = read_backend_document(path)
    common: dict[str, JSONValue] = {
        "schema": "fedference.gnn-backend/2",
        "revision": UPSTREAM_REVISION,
        "agent": "pymdp.agent.Agent",
        "dtype": "float32",
        "device": "cpu",
        "batch_size": 1,
        "policy_horizon": 1,
        "rtol": RTOL,
        "atol": ATOL,
    }
    if set(registry) != set(common) | {"source_files", "profiles"} or json.dumps(
        {key: registry.get(key) for key in common}, sort_keys=True
    ) != json.dumps(common, sort_keys=True):
        raise GNNAdmissionError("GNN_BACKEND_LOCK", "registry fields/schema/semantics")
    files = registry["source_files"]
    profiles = registry["profiles"]
    if not isinstance(files, dict) or not files:
        raise GNNAdmissionError("GNN_BACKEND_LOCK", "source closure")
    if not isinstance(profiles, dict) or not profiles:
        raise GNNAdmissionError("GNN_BACKEND_LOCK", "profiles")
    source_files: dict[str, str] = {}
    for relative, digest in files.items():
        if not isinstance(digest, str):
            raise GNNAdmissionError("GNN_BACKEND_LOCK", "source hash")
        source_files[relative] = digest
    identities: set[str] = set()
    admitted: dict[str, BackendProfile] = {}
    for name, profile in profiles.items():
        if (not name or not isinstance(profile, dict)
                or set(profile) != {"python", "distributions", "qualification"}):
            raise GNNAdmissionError("GNN_BACKEND_LOCK", "profile fields")
        qualification: Qualification
        match profile["qualification"]:
            case "fresh-provisioning-and-execution":
                qualification = "fresh-provisioning-and-execution"
            case "existing-runtime-execution":
                qualification = "existing-runtime-execution"
            case _:
                raise GNNAdmissionError("GNN_BACKEND_LOCK", "profile qualification")
        version = profile["python"]
        distributions = profile["distributions"]
        if not isinstance(version, str) or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is None:
            raise GNNAdmissionError("GNN_BACKEND_LOCK", "profile python")
        if not isinstance(distributions, dict) or not distributions:
            raise GNNAdmissionError("GNN_BACKEND_LOCK", "profile distributions")
        releases: dict[str, str] = {}
        for distribution, release in distributions.items():
            if (re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", distribution) is None
                    or not isinstance(release, str)
                    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", release) is None):
                raise GNNAdmissionError("GNN_BACKEND_LOCK", "distribution name/version")
            releases[distribution] = release
        identity = json.dumps({"python": version, "distributions": distributions}, sort_keys=True)
        if identity in identities:
            raise GNNAdmissionError("GNN_BACKEND_LOCK", "ambiguous runtime profiles")
        identities.add(identity)
        admitted[name] = BackendProfile(version, MappingProxyType(releases), qualification)
    return BackendRegistry(registry, MappingProxyType(source_files), MappingProxyType(admitted))
