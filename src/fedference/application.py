"""Labeled categorical aggregation for caller-owned application data.

This module adds state and agent identity to the existing mathematical
aggregation boundary.  It deliberately delegates fusion to
``aggregate_result`` and does not infer a winning state or downstream decision.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from ._validation import as_pmf_matrix
from .aggregation import AggregationConfig, AggregationResult, aggregate_result
from .evidence import canonical_sha256

ArrayF = np.ndarray
APPLICATION_SCHEMA_VERSION = "1.0"
APPLICATION_OPERATION = "labeled_categorical_aggregation"
_CONFIG_FIELDS = {
    "method",
    "robustness",
    "entropy_weight",
    "max_iter",
    "tol",
    "multistart",
}


def _require_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    return value


def _numeric_vector(values: object, *, name: str) -> ArrayF:
    if isinstance(values, np.ndarray):
        raw = values
    else:
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            raise ValueError(f"{name} must be a one-dimensional numeric sequence")
        raw = np.asarray(tuple(values), dtype=object)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.size == 0:
        raise ValueError(f"{name} must be non-empty")
    object_values = np.asarray(raw, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in object_values):
        raise ValueError(f"{name} must not contain booleans")
    if any(not isinstance(value, Real) for value in object_values):
        raise ValueError(f"{name} must contain only numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if float(result.sum()) <= 0.0:
        raise ValueError(f"{name} must have a positive finite sum")
    result = result.copy()
    result.setflags(write=False)
    return result


def _base_weight(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("base_weight must be a finite non-negative number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("base_weight must be a finite non-negative number")
    return result


def _config_from_dict(raw: object) -> AggregationConfig:
    if not isinstance(raw, Mapping) or set(raw) != _CONFIG_FIELDS:
        raise ValueError("aggregation_config fields do not match schema")
    try:
        return AggregationConfig(
            method=raw["method"],
            robustness=raw["robustness"],
            entropy_weight=raw["entropy_weight"],
            max_iter=raw["max_iter"],
            tol=raw["tol"],
            multistart=raw["multistart"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid aggregation_config: {exc}") from exc


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _loads_request(value: str) -> Mapping[str, Any]:
    raw = json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(raw, Mapping):
        raise ValueError("labeled aggregation request must be a JSON object")
    return raw


@dataclass(frozen=True)
class AgentPosterior:
    """One identified caller-supplied categorical posterior and base weight."""

    agent_id: str
    posterior: ArrayF
    base_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _require_identifier(self.agent_id, name="agent_id"))
        object.__setattr__(
            self,
            "posterior",
            _numeric_vector(self.posterior, name=f"posterior for {self.agent_id!r}"),
        )
        object.__setattr__(self, "base_weight", _base_weight(self.base_weight))

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical semantic agent record."""
        return {
            "agent_id": self.agent_id,
            "posterior": self.posterior.tolist(),
            "base_weight": self.base_weight,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AgentPosterior:
        """Decode one strict agent record, expanding an omitted weight to one."""
        if not isinstance(raw, Mapping):
            raise ValueError("agent must be an object")
        required = {"agent_id", "posterior"}
        allowed = required | {"base_weight"}
        if not required.issubset(raw) or not set(raw).issubset(allowed):
            raise ValueError("agent fields do not match schema")
        return cls(
            agent_id=raw["agent_id"],
            posterior=raw["posterior"],
            base_weight=raw.get("base_weight", 1.0),
        )


@dataclass(frozen=True)
class LabeledAggregationRequest:
    """Validated labeled input envelope for one categorical aggregation."""

    state_labels: tuple[str, ...]
    agents: tuple[AgentPosterior, ...]
    config: AggregationConfig
    schema_version: str = APPLICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != APPLICATION_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {APPLICATION_SCHEMA_VERSION!r}")
        if not isinstance(self.state_labels, (tuple, list)):
            raise ValueError("state_labels must be a sequence")
        labels = tuple(
            _require_identifier(value, name="state label") for value in self.state_labels
        )
        if len(labels) < 2:
            raise ValueError("state_labels must contain at least two states")
        if len(set(labels)) != len(labels):
            raise ValueError("state_labels must be unique")
        if not isinstance(self.agents, (tuple, list)):
            raise ValueError("agents must be a sequence")
        agents = tuple(self.agents)
        if not agents or any(not isinstance(agent, AgentPosterior) for agent in agents):
            raise ValueError("agents must contain AgentPosterior values")
        agent_ids = tuple(agent.agent_id for agent in agents)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agent_id values must be unique")
        if any(agent.posterior.size != len(labels) for agent in agents):
            raise ValueError("every posterior length must match state_labels")
        if sum(agent.base_weight for agent in agents) <= 0.0:
            raise ValueError("base weights must contain at least one positive value")
        if not isinstance(self.config, AggregationConfig):
            raise ValueError("config must be an AggregationConfig")
        object.__setattr__(self, "state_labels", labels)
        object.__setattr__(self, "agents", agents)

    @property
    def request_sha256(self) -> str:
        """SHA-256 over the canonical semantic request."""
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical semantic request, including default weights."""
        return {
            "schema_version": self.schema_version,
            "state_labels": list(self.state_labels),
            "agents": [agent.as_dict() for agent in self.agents],
            "aggregation_config": self.config.as_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LabeledAggregationRequest:
        """Decode one strict request mapping without coercing its structure."""
        required = {"schema_version", "state_labels", "agents", "aggregation_config"}
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("labeled aggregation request fields do not match schema")
        if not isinstance(raw["state_labels"], list):
            raise ValueError("state_labels must be a list")
        if not isinstance(raw["agents"], list):
            raise ValueError("agents must be a list")
        return cls(
            state_labels=tuple(raw["state_labels"]),
            agents=tuple(AgentPosterior.from_dict(agent) for agent in raw["agents"]),
            config=_config_from_dict(raw["aggregation_config"]),
            schema_version=raw["schema_version"],
        )

    @classmethod
    def from_json(cls, value: str) -> LabeledAggregationRequest:
        """Decode strict JSON, including duplicate-key and non-finite rejection."""
        return cls.from_dict(_loads_request(value))


@dataclass(frozen=True)
class LabeledAggregationResult:
    """Labeled consensus and complete solver diagnostics; no decision semantics."""

    state_labels: tuple[str, ...]
    agent_ids: tuple[str, ...]
    normalized_local_posteriors: ArrayF
    config: AggregationConfig
    request_sha256: str
    aggregation: AggregationResult
    software_version: str
    schema_version: str = APPLICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != APPLICATION_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {APPLICATION_SCHEMA_VERSION!r}")
        labels = tuple(_require_identifier(value, name="state label") for value in self.state_labels)
        agent_ids = tuple(_require_identifier(value, name="agent_id") for value in self.agent_ids)
        if len(set(labels)) != len(labels) or len(set(agent_ids)) != len(agent_ids):
            raise ValueError("state labels and agent ids must be unique")
        matrix = np.asarray(self.normalized_local_posteriors)
        if matrix.ndim != 2 or matrix.shape != (len(agent_ids), len(labels)):
            raise ValueError("normalized_local_posteriors shape must match agents and states")
        matrix = np.asarray(matrix, dtype=np.float64)
        if (
            not np.all(np.isfinite(matrix))
            or np.any(matrix < 0.0)
            or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("normalized_local_posteriors must contain finite probability rows")
        if not isinstance(self.config, AggregationConfig):
            raise ValueError("config must be an AggregationConfig")
        if not isinstance(self.aggregation, AggregationResult):
            raise ValueError("aggregation must be an AggregationResult")
        if self.aggregation.consensus.size != len(labels):
            raise ValueError("aggregation consensus must match state_labels")
        raw_weights = self.aggregation.raw_effective_weights
        if raw_weights is None or raw_weights.size != len(agent_ids) or (
            self.aggregation.normalized_effective_weights.size != len(agent_ids)
        ):
            raise ValueError("aggregation weights must match agent_ids")
        if (
            not isinstance(self.request_sha256, str)
            or len(self.request_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in self.request_sha256)
        ):
            raise ValueError("request_sha256 must be a SHA-256 digest")
        _require_identifier(self.software_version, name="software_version")
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "state_labels", labels)
        object.__setattr__(self, "agent_ids", agent_ids)
        object.__setattr__(self, "normalized_local_posteriors", matrix)
        object.__setattr__(self, "request_sha256", self.request_sha256.lower())

    def as_dict(self) -> dict[str, Any]:
        """Return exactly the public labeled-result JSON contract."""
        return {
            "schema_version": self.schema_version,
            "operation": APPLICATION_OPERATION,
            "software_version": self.software_version,
            "request_sha256": self.request_sha256,
            "state_labels": list(self.state_labels),
            "agent_ids": list(self.agent_ids),
            "normalized_local_posteriors": self.normalized_local_posteriors.tolist(),
            "aggregation_config": self.config.as_dict(),
            "config_fingerprint": self.config.fingerprint,
            "aggregation": self.aggregation.as_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LabeledAggregationResult:
        """Decode the exact result contract and recheck all content bindings."""
        required = {
            "schema_version",
            "operation",
            "software_version",
            "request_sha256",
            "state_labels",
            "agent_ids",
            "normalized_local_posteriors",
            "aggregation_config",
            "config_fingerprint",
            "aggregation",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("labeled aggregation result fields do not match schema")
        if raw["operation"] != APPLICATION_OPERATION:
            raise ValueError(f"operation must be {APPLICATION_OPERATION!r}")
        for field_name in (
            "state_labels",
            "agent_ids",
            "normalized_local_posteriors",
        ):
            if not isinstance(raw[field_name], list):
                raise ValueError(f"{field_name} must be a list")
        config = _config_from_dict(raw["aggregation_config"])
        if raw["config_fingerprint"] != config.fingerprint:
            raise ValueError("config_fingerprint does not match aggregation_config")
        if not isinstance(raw["aggregation"], Mapping):
            raise ValueError("aggregation must be an object")
        return cls(
            state_labels=tuple(raw["state_labels"]),
            agent_ids=tuple(raw["agent_ids"]),
            normalized_local_posteriors=np.asarray(
                raw["normalized_local_posteriors"],
            ),
            config=config,
            request_sha256=raw["request_sha256"],
            aggregation=AggregationResult.from_dict(raw["aggregation"]),
            software_version=raw["software_version"],
            schema_version=raw["schema_version"],
        )

    @classmethod
    def from_json(cls, value: str) -> LabeledAggregationResult:
        """Decode strict result JSON, including duplicate-key rejection."""
        return cls.from_dict(_loads_request(value))


def aggregate_labeled(request: LabeledAggregationRequest) -> LabeledAggregationResult:
    """Aggregate one labeled request through the canonical domain dispatcher once."""
    if not isinstance(request, LabeledAggregationRequest):
        raise ValueError("request must be a LabeledAggregationRequest")
    normalized = as_pmf_matrix(
        (agent.posterior for agent in request.agents),
        name="agents.posterior",
    )
    aggregation = aggregate_result(
        (agent.posterior for agent in request.agents),
        config=request.config,
        base_weights=(agent.base_weight for agent in request.agents),
    )
    return LabeledAggregationResult(
        state_labels=request.state_labels,
        agent_ids=tuple(agent.agent_id for agent in request.agents),
        normalized_local_posteriors=normalized,
        config=request.config,
        request_sha256=request.request_sha256,
        aggregation=aggregation,
        software_version=metadata.version("active_fedference"),
    )


def load_labeled_aggregation_request(path: str | Path) -> LabeledAggregationRequest:
    """Load strict labeled-request JSON from an explicit caller-owned path."""
    try:
        value = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read labeled aggregation request: {path}") from exc
    try:
        return LabeledAggregationRequest.from_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid labeled aggregation request: {exc}") from exc


__all__ = [
    "APPLICATION_OPERATION",
    "APPLICATION_SCHEMA_VERSION",
    "AgentPosterior",
    "LabeledAggregationRequest",
    "LabeledAggregationResult",
    "aggregate_labeled",
    "load_labeled_aggregation_request",
]
