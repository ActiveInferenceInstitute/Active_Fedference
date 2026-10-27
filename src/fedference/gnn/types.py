"""Immutable canonical GNN IR; categorical semantics follow Friston et al. (2024).

These values are representations, not execution certificates. Use admission to
construct validated models. Tuples retain numeric values without normalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

ModelKind: TypeAlias = Literal[
    "categorical", "moving-observer", "hierarchical", "hybrid-gaussian", "site-factor"
]

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class Tensor:
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Provenance:
    source: str
    asset: str
    parser: str
    backend: str | None
    native_result: str | None


@dataclass(frozen=True, slots=True)
class Metadata:
    name: str
    capabilities: tuple[str, ...]
    schedule: tuple[str, ...]
    provenance: Provenance
    schema: Literal["fedference.gnn/1"] = "fedference.gnn/1"
    precision: Literal["float64-atol-1e-12-rtol-0"] = "float64-atol-1e-12-rtol-0"


@dataclass(frozen=True, slots=True)
class Categorical:
    states: tuple[str, ...]
    observations: tuple[str, ...]
    actions: tuple[str, ...]
    A: Tensor
    B: Tensor
    C: Tensor
    D: Tensor


@dataclass(frozen=True, slots=True)
class CategoricalModel:
    metadata: Metadata
    categorical: Categorical
    kind: Literal["categorical"] = field(default="categorical", init=False)


@dataclass(frozen=True, slots=True)
class MovingObserverModel:
    metadata: Metadata
    threat: Categorical
    positions: tuple[str, ...]
    position_transition: Tensor
    position_prior: Tensor
    observation_map: Tensor
    kind: Literal["moving-observer"] = field(default="moving-observer", init=False)


@dataclass(frozen=True, slots=True)
class Layer:
    """Topologically ordered parent; conditioned prior replaces D when conditioned."""

    name: str
    categorical: Categorical
    parent: str | None
    conditioned_prior: Tensor | None


@dataclass(frozen=True, slots=True)
class HierarchicalModel:
    metadata: Metadata
    layers: tuple[Layer, ...]
    kind: Literal["hierarchical"] = field(default="hierarchical", init=False)


@dataclass(frozen=True, slots=True)
class HybridGaussianModel:
    """Mean uses coordinate units; covariance entry (i,j) uses units[i]*units[j]."""

    metadata: Metadata
    categorical: Categorical
    coordinates: tuple[str, ...]
    units: tuple[str, ...]
    mean: Tensor
    covariance: Tensor
    kind: Literal["hybrid-gaussian"] = field(default="hybrid-gaussian", init=False)


@dataclass(frozen=True, slots=True)
class SiteFactorModel:
    """Signed Gaussian increments (Mildner et al., 2025), not PMFs/cavities.

    Precision units are inverse coordinate-unit products; information units are
    inverse coordinate units. Properness is required of the combined Gaussian,
    not an individual signed site. Admission does not perform that combination.
    """

    metadata: Metadata
    coordinates: tuple[str, ...]
    units: tuple[str, ...]
    precision_increment: Tensor
    information_increment: Tensor
    protocol: Literal["gaussian-natural-additive/1"] = "gaussian-natural-additive/1"
    kind: Literal["site-factor"] = field(default="site-factor", init=False)


Model: TypeAlias = (
    CategoricalModel | MovingObserverModel | HierarchicalModel | HybridGaussianModel | SiteFactorModel
)
