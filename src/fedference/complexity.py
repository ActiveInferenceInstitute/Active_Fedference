"""Implementation-derived complexity contracts for the categorical core.

The expressions in this module describe the dominant dense-array work performed
by the implementation, rather than claiming a hardware-independent FLOP count.
They make the relevant dimensions explicit:

* ``N`` — number of agents;
* ``S`` — number of categorical states;
* ``I`` — solver-iteration budget; and
* ``M`` — number of independent observation modalities.

The aggregation code materializes an ``(N, S)`` belief matrix and, for the
iterative rules, retains a per-iteration history.  The reported memory orders
therefore describe the current implementation's peak/returned storage, not an
ideal streaming implementation.  This is a source-bound accounting layer for
Active Fedference's categorical specialization of Friston et al. (2024) and
Mildner et al. (2025); it is not a general complexity theorem for other FedGVI
implementations or hardware backends.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Final


def _positive_int(value: object, *, name: str) -> int:
    """Validate and return a strictly positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _ordered_sizes(values: object, *, name: str) -> tuple[int, ...]:
    """Validate a strictly increasing, non-empty benchmark size grid."""
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a list or tuple of positive integers")
    sizes = tuple(_positive_int(value, name=f"{name} item") for value in values)
    if len(sizes) < 2:
        raise ValueError(f"{name} must contain at least two sizes")
    if sizes != tuple(sorted(set(sizes))):
        raise ValueError(f"{name} must be strictly increasing with no duplicates")
    return sizes


@dataclass(frozen=True)
class ComplexityBenchmarkConfig:
    """Seeded, bounded grid for the machine scaling experiment.

    The benchmark is intentionally modest: it measures the dense categorical
    kernels used by the release without turning a publication regeneration into
    a stress test.  ``repeats`` controls repeated wall-clock observations;
    their median is plotted and the min--max span is retained in the report.
    """

    agent_sizes: tuple[int, ...] = (4, 8, 16, 32, 64)
    state_sizes: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    sharing_agent_sizes: tuple[int, ...] = (4, 8, 16, 32)
    modality_sizes: tuple[int, ...] = (1, 2, 4, 8)
    fixed_agent_count: int = 256
    fixed_state_count: int = 64
    inference_state_count: int = 16384
    observation_count: int = 4
    repeats: int = 5
    warmups: int = 1
    max_iter: int = 6
    variational_starts: int = 3
    seed: int = 20260728

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_sizes", _ordered_sizes(self.agent_sizes, name="agent_sizes"))
        object.__setattr__(self, "state_sizes", _ordered_sizes(self.state_sizes, name="state_sizes"))
        object.__setattr__(
            self,
            "sharing_agent_sizes",
            _ordered_sizes(self.sharing_agent_sizes, name="sharing_agent_sizes"),
        )
        object.__setattr__(self, "modality_sizes", _ordered_sizes(self.modality_sizes, name="modality_sizes"))
        for name in (
            "fixed_agent_count",
            "fixed_state_count",
            "inference_state_count",
            "observation_count",
            "max_iter",
            "variational_starts",
        ):
            _positive_int(getattr(self, name), name=name)
        if isinstance(self.repeats, bool) or not isinstance(self.repeats, int) or self.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if isinstance(self.warmups, bool) or not isinstance(self.warmups, int) or self.warmups < 0:
            raise ValueError("warmups must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "ComplexityBenchmarkConfig":
        """Build a benchmark config from the optional YAML mapping."""
        raw = dict(values or {})
        defaults = cls()

        def _mapping_int(value: object, *, name: str) -> int:
            """Coerce only an integer-valued mapping scalar; never truncate."""
            if isinstance(value, bool):
                raise ValueError(f"complexity.{name} must be an integer")
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value.strip(), 10)
                except ValueError:
                    pass
            raise ValueError(f"complexity.{name} must be an integer")

        def _grid(key: str, default: tuple[int, ...]) -> tuple[int, ...]:
            value = raw.get(key, default)
            if isinstance(value, int) and not isinstance(value, bool):
                return (value,)
            if isinstance(value, (list, tuple)):
                return tuple(_mapping_int(item, name=f"{key} item") for item in value)
            raise ValueError(f"complexity.{key} must be a list or tuple of integers")

        def _scalar_int(key: str, default: int) -> int:
            value = raw.get(key, default)
            return _mapping_int(value, name=key)

        return cls(
            agent_sizes=_grid("agent_sizes", defaults.agent_sizes),
            state_sizes=_grid("state_sizes", defaults.state_sizes),
            sharing_agent_sizes=_grid("sharing_agent_sizes", defaults.sharing_agent_sizes),
            modality_sizes=_grid("modality_sizes", defaults.modality_sizes),
            fixed_agent_count=_scalar_int("fixed_agent_count", defaults.fixed_agent_count),
            fixed_state_count=_scalar_int("fixed_state_count", defaults.fixed_state_count),
            inference_state_count=_scalar_int("inference_state_count", defaults.inference_state_count),
            observation_count=_scalar_int("observation_count", defaults.observation_count),
            repeats=_scalar_int("repeats", defaults.repeats),
            warmups=_scalar_int("warmups", defaults.warmups),
            max_iter=_scalar_int("max_iter", defaults.max_iter),
            variational_starts=_scalar_int("variational_starts", defaults.variational_starts),
            seed=_scalar_int("seed", defaults.seed),
        )

    def for_smoke(self) -> "ComplexityBenchmarkConfig":
        """Return a smaller real benchmark for repeated smoke-profile runs."""
        return replace(
            self,
            agent_sizes=self.agent_sizes[:3],
            state_sizes=self.state_sizes[:3],
            sharing_agent_sizes=self.sharing_agent_sizes[:3],
            modality_sizes=self.modality_sizes[:3],
            fixed_agent_count=min(self.fixed_agent_count, 32),
            fixed_state_count=min(self.fixed_state_count, 32),
            inference_state_count=min(self.inference_state_count, 256),
            repeats=min(self.repeats, 2),
            warmups=0,
            max_iter=min(self.max_iter, 3),
        )

    def as_dict(self) -> dict[str, object]:
        """Return JSON-safe benchmark settings."""
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class ComplexitySpec:
    """One source-bound asymptotic accounting row."""

    operation: str
    time_order: str
    memory_order: str
    work_formula: str
    workspace_formula: str
    source_modules: tuple[str, ...]
    notes: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation for the report."""
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class ComplexityEstimate:
    """Concrete dominant-work proxy for one operation and parameter setting."""

    operation: str
    parameters: dict[str, int]
    work_units: int
    workspace_units: int
    time_order: str
    memory_order: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation for the report."""
        return asdict(self)


_SPECS: Final[tuple[ComplexitySpec, ...]] = (
    ComplexitySpec(
        "log_linear_pool",
        "Theta(N S)",
        "Theta(N S)",
        "N S",
        "N S",
        ("src/fedference/aggregation.py",),
        (
            "The implementation normalizes and materializes all belief rows before one "
            "weighted log-matrix product."
        ),
    ),
    ComplexitySpec(
        "robust_aggregate",
        "Theta(I N S)",
        "Theta(N S + I S)",
        "I N S",
        "N S + I S",
        ("src/fedference/aggregation.py", "src/fedference/divergences.py"),
        (
            "Each solver iteration evaluates one divergence per agent and one weighted "
            "log-matrix product; history is returned."
        ),
    ),
    ComplexitySpec(
        "variational_aggregate",
        "Theta(B I N S)",
        "Theta(N S + I S + B S)",
        "B I N S",
        "N S + I S + B S",
        ("src/fedference/aggregation.py",),
        (
            "B is the number of starts (three in the default multistart path); each "
            "start performs cross-entropies and a log-matrix product."
        ),
    ),
    ComplexitySpec(
        "aggregation_free_energy",
        "Theta(N S)",
        "Theta(N S)",
        "N S",
        "N S",
        ("src/fedference/aggregation.py",),
        "The objective evaluates per-agent cross-entropies over the dense belief matrix.",
    ),
    ComplexitySpec(
        "share_round_naive",
        "Theta(N^2 S)",
        "Theta(N S)",
        "N^2 S",
        "N S",
        ("src/fedference/belief_sharing.py", "src/fedference/aggregation.py"),
        (
            "With self-exclusion, one global pool plus N leave-one-out pools are "
            "computed; this is the naive project log-pool / qualified Eq. 7 bridge path."
        ),
    ),
    ComplexitySpec(
        "share_round_robust",
        "Theta(I N^2 S)",
        "Theta(N S + I S)",
        "I N^2 S",
        "N S + I S",
        ("src/fedference/belief_sharing.py", "src/fedference/aggregation.py"),
        (
            "The same leave-one-out fan-out calls the iterative robust server rule; I "
            "is its per-call solver budget."
        ),
    ),
    ComplexitySpec(
        "infer_states",
        "Theta(M S)",
        "Theta(S)",
        "M S",
        "S",
        ("src/fedference/belief_updating.py",),
        "Independent modality log-likelihood messages are accumulated over the shared state vector.",
    ),
    ComplexitySpec(
        "federation_server_round",
        "Theta(N log N + I N S)",
        "Theta(N S)",
        "N log N + I N S",
        "N S",
        (
            "src/fedference/federation/server.py",
            "src/fedference/federation/transport.py",
            "src/fedference/aggregation.py",
        ),
        (
            "Compute excludes queue/network wait; it includes worker-id sorting, belief "
            "deserialization, one robust server aggregation, and one result serialization. "
            "If broadcast bytes are counted per recipient, the outgoing volume is "
            "Theta(N S + N^2) because agent weights are included in every result."
        ),
    ),
)

_SPEC_BY_OPERATION: Final[dict[str, ComplexitySpec]] = {spec.operation: spec for spec in _SPECS}


def complexity_catalog() -> tuple[ComplexitySpec, ...]:
    """Return the immutable catalog of implementation-derived complexity rows."""
    return _SPECS


def estimate_complexity(
    operation: str,
    *,
    n_agents: int,
    n_states: int,
    n_modalities: int = 1,
    iterations: int = 1,
    n_starts: int = 3,
    exclude_self: bool = True,
) -> ComplexityEstimate:
    """Calculate a concrete dominant-work proxy for a catalogued operation.

    ``work_units`` are dimensionless ``agent-state`` (or ``modality-state``)
    interaction units. They are useful for checking scaling directions and are
    deliberately not presented as measured FLOPs or seconds.
    """
    if operation not in _SPEC_BY_OPERATION:
        known = ", ".join(sorted(_SPEC_BY_OPERATION))
        raise ValueError(f"unknown complexity operation {operation!r}; expected one of {known}")
    n = _positive_int(n_agents, name="n_agents")
    s = _positive_int(n_states, name="n_states")
    m = _positive_int(n_modalities, name="n_modalities")
    i = _positive_int(iterations, name="iterations")
    b = _positive_int(n_starts, name="n_starts")

    if operation == "log_linear_pool":
        work = n * s
        workspace = n * s
        parameters = {"N": n, "S": s}
    elif operation == "robust_aggregate":
        work = i * n * s
        workspace = n * s + i * s
        parameters = {"N": n, "S": s, "I": i}
    elif operation == "variational_aggregate":
        work = b * i * n * s
        workspace = n * s + i * s + b * s
        parameters = {"N": n, "S": s, "I": i, "B": b}
    elif operation == "aggregation_free_energy":
        work = n * s
        workspace = n * s
        parameters = {"N": n, "S": s}
    elif operation == "share_round_naive":
        fanout = n * n if exclude_self and n > 1 else n
        work = fanout * s
        workspace = n * s
        parameters = {"N": n, "S": s, "exclude_self": int(exclude_self)}
    elif operation == "share_round_robust":
        fanout = n * n if exclude_self and n > 1 else n
        work = i * fanout * s
        workspace = n * s + i * s
        parameters = {"N": n, "S": s, "I": i, "exclude_self": int(exclude_self)}
    elif operation == "infer_states":
        work = m * s
        workspace = s
        parameters = {"M": m, "S": s}
    else:
        # The server performs an N log N worker-id sort and an I N S robust
        # aggregation.  ``max(2, N)`` keeps log2 defined for the validated N.
        import math

        work = int(round(n * math.log2(max(2, n)))) + i * n * s
        workspace = n * s
        parameters = {"N": n, "S": s, "I": i}

    spec = _SPEC_BY_OPERATION[operation]
    return ComplexityEstimate(
        operation=operation,
        parameters=parameters,
        work_units=int(work),
        workspace_units=int(workspace),
        time_order=spec.time_order,
        memory_order=spec.memory_order,
    )


__all__ = [
    "ComplexityBenchmarkConfig",
    "ComplexityEstimate",
    "ComplexitySpec",
    "complexity_catalog",
    "estimate_complexity",
]
