"""Seeded machine-scaling experiments for the categorical implementation.

This module measures the public code paths used by aggregation, belief sharing,
and state inference. Inputs are generated once per grid point from
``np.random.default_rng(seed)`` and reused for repeated timings. The benchmark
therefore has deterministic inputs and an auditable grid, while the measured
wall-clock values remain machine- and load-dependent diagnostics rather than
scientific effect estimates.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from ..aggregation import log_linear_pool, robust_aggregate, variational_aggregate
from ..belief_sharing import share_round
from ..belief_updating import infer_states
from ..complexity import (
    ComplexityBenchmarkConfig,
    complexity_catalog,
    estimate_complexity,
)

_ROBUST_SHARING_ITERATIONS = 32


def _local_posteriors(seed: int, n_agents: int, n_states: int) -> np.ndarray:
    """Create deterministic positive local posteriors for one grid point."""
    rng = np.random.default_rng(seed)
    return np.asarray(rng.dirichlet(np.ones(n_states), size=n_agents), dtype=np.float64)


def _likelihoods(
    seed: int,
    n_modalities: int,
    n_observations: int,
    n_states: int,
) -> tuple[np.ndarray, ...]:
    """Create deterministic per-modality likelihood matrices."""
    rng = np.random.default_rng(seed)
    return tuple(
        np.asarray(rng.dirichlet(np.ones(n_states), size=n_observations), dtype=np.float64)
        for _ in range(n_modalities)
    )


def _digest_arrays(arrays: Sequence[np.ndarray]) -> str:
    """Hash benchmark inputs so the report carries an input provenance receipt."""
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _timed_call(
    function: Callable[[], object],
    *,
    repeats: int,
    warmups: int,
) -> list[float]:
    """Run one real callable repeatedly and return wall-clock samples."""
    for _ in range(warmups):
        function()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        elapsed = time.perf_counter() - start
        if not np.isfinite(elapsed) or elapsed <= 0.0:
            raise RuntimeError(f"non-positive or non-finite benchmark duration: {elapsed!r}")
        samples.append(float(elapsed))
    return samples


def _log_slope(sizes: Sequence[int], medians: Sequence[float]) -> float:
    """Fit the descriptive log-log slope used by the scaling figure."""
    x = np.log(np.asarray(sizes, dtype=np.float64))
    y = np.log(np.asarray(medians, dtype=np.float64))
    slope = float(np.polyfit(x, y, 1)[0])
    if not np.isfinite(slope):
        raise RuntimeError("log-log benchmark slope is non-finite")
    return slope


def _measurement(
    *,
    method: str,
    axis: str,
    sizes: Sequence[int],
    samples: Sequence[Sequence[float]],
    expected_exponent: float,
    work_units: Sequence[int],
    parameters: dict[str, int],
    input_digests: Sequence[str],
    repeats: int,
    warmups: int,
    note: str,
) -> dict[str, object]:
    """Assemble one JSON-safe scaling measurement row."""
    medians = [float(np.median(values)) for values in samples]
    minima = [float(np.min(values)) for values in samples]
    maxima = [float(np.max(values)) for values in samples]
    return {
        "method": method,
        "axis": axis,
        "sizes": [int(size) for size in sizes],
        "samples_seconds": [[float(value) for value in row] for row in samples],
        "median_seconds": medians,
        "min_seconds": minima,
        "max_seconds": maxima,
        "work_units": [int(value) for value in work_units],
        "expected_exponent": float(expected_exponent),
        "observed_log_log_slope": _log_slope(sizes, medians),
        "parameters": dict(parameters),
        "input_digests": list(input_digests),
        "repeats": int(repeats),
        "warmups": int(warmups),
        "fit_method": "ordinary least squares on log(size) versus log(median wall time)",
        "note": note,
    }


def _machine_metadata() -> dict[str, object]:
    """Return portable environment facts needed to interpret timing diagnostics.

    The executable is recorded by basename rather than an absolute path so the
    generated report never embeds a contributor's home directory. Environment
    locks and run receipts carry the stronger interpreter provenance.
    """
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": int(os.cpu_count() or 1),
        "python_implementation": platform.python_implementation(),
        "timer": "time.perf_counter",
        "sys_executable": Path(sys.executable).name,
    }


def _aggregation_call(
    method: str,
    local_posteriors: np.ndarray,
    *,
    max_iter: int,
) -> Callable[[], object]:
    """Return the real aggregation callable for one timed path."""
    if method == "log_linear_pool":
        return lambda: log_linear_pool(local_posteriors)
    if method == "robust_aggregate":
        return lambda: robust_aggregate(
            local_posteriors,
            robustness=1.0,
            max_iter=max_iter,
            tol=0.0,
        )
    if method == "variational_aggregate":
        return lambda: variational_aggregate(
            local_posteriors,
            robustness=1.0,
            max_iter=max_iter,
            tol=0.0,
            multistart=True,
        )
    raise ValueError(f"unknown aggregation benchmark method {method!r}")


def _aggregation_measurements(config: ComplexityBenchmarkConfig) -> list[dict[str, object]]:
    """Measure N- and S-scaling for the three public aggregation methods."""
    measurements: list[dict[str, object]] = []
    methods = (
        ("log_linear_pool", 1.0),
        ("robust_aggregate", 1.0),
        ("variational_aggregate", 1.0),
    )
    for method, expected in methods:
        samples: list[list[float]] = []
        digests: list[str] = []
        work: list[int] = []
        for index, n_agents in enumerate(config.agent_sizes):
            local_posteriors = _local_posteriors(
                config.seed + 10_000 + index, n_agents, config.fixed_state_count
            )
            samples.append(
                _timed_call(
                    _aggregation_call(method, local_posteriors, max_iter=config.max_iter),
                    repeats=config.repeats,
                    warmups=config.warmups,
                )
            )
            digests.append(_digest_arrays((local_posteriors,)))
            work.append(
                estimate_complexity(
                    method,
                    n_agents=n_agents,
                    n_states=config.fixed_state_count,
                    iterations=config.max_iter,
                    n_starts=config.variational_starts,
                ).work_units
            )
        measurements.append(
            _measurement(
                method=method,
                axis="agents",
                sizes=config.agent_sizes,
                samples=samples,
                expected_exponent=expected,
                work_units=work,
                parameters={
                    "S": config.fixed_state_count,
                    "I": config.max_iter,
                    "B": config.variational_starts if method == "variational_aggregate" else 1,
                },
                input_digests=digests,
                repeats=config.repeats,
                warmups=config.warmups,
                note="N-scaling with fixed state cardinality; I and B are held constant.",
            )
        )

        samples = []
        digests = []
        work = []
        for index, n_states in enumerate(config.state_sizes):
            local_posteriors = _local_posteriors(
                config.seed + 20_000 + index, config.fixed_agent_count, n_states
            )
            samples.append(
                _timed_call(
                    _aggregation_call(method, local_posteriors, max_iter=config.max_iter),
                    repeats=config.repeats,
                    warmups=config.warmups,
                )
            )
            digests.append(_digest_arrays((local_posteriors,)))
            work.append(
                estimate_complexity(
                    method,
                    n_agents=config.fixed_agent_count,
                    n_states=n_states,
                    iterations=config.max_iter,
                    n_starts=config.variational_starts,
                ).work_units
            )
        measurements.append(
            _measurement(
                method=method,
                axis="states",
                sizes=config.state_sizes,
                samples=samples,
                expected_exponent=expected,
                work_units=work,
                parameters={
                    "N": config.fixed_agent_count,
                    "I": config.max_iter,
                    "B": config.variational_starts if method == "variational_aggregate" else 1,
                },
                input_digests=digests,
                repeats=config.repeats,
                warmups=config.warmups,
                note="S-scaling with fixed agent count; I and B are held constant.",
            )
        )
    return measurements


def _sharing_measurements(config: ComplexityBenchmarkConfig) -> list[dict[str, object]]:
    """Measure naive and iterative-robust quadratic leave-one-out fan-out."""
    measurements: list[dict[str, object]] = []
    for method, operation, iterations in (
        ("naive", "share_round_naive", 1),
        ("robust", "share_round_robust", _ROBUST_SHARING_ITERATIONS),
    ):
        samples: list[list[float]] = []
        digests: list[str] = []
        work: list[int] = []
        for index, n_agents in enumerate(config.sharing_agent_sizes):
            local_posteriors = _local_posteriors(
                config.seed + 30_000 + index, n_agents, config.fixed_state_count
            )

            def _share() -> object:
                return share_round(
                    local_posteriors,
                    method=method,
                    robustness=1.0,
                    exclude_self=True,
                )

            samples.append(
                _timed_call(
                    _share,
                    repeats=config.repeats,
                    warmups=config.warmups,
                )
            )
            digests.append(_digest_arrays((local_posteriors,)))
            work.append(
                estimate_complexity(
                    operation,
                    n_agents=n_agents,
                    n_states=config.fixed_state_count,
                    iterations=iterations,
                    exclude_self=True,
                ).work_units
            )
        measurements.append(
            _measurement(
                method=f"share_round_{method}",
                axis="agents",
                sizes=config.sharing_agent_sizes,
                samples=samples,
                expected_exponent=2.0,
                work_units=work,
                parameters={
                    "S": config.fixed_state_count,
                    "I": iterations,
                    "exclude_self": 1,
                },
                input_digests=digests,
                repeats=config.repeats,
                warmups=config.warmups,
                note=(
                    "N-scaling of one self-excluding sharing round; one global plus N "
                    "leave-one-out pools."
                ),
            )
        )
    return measurements


def _inference_measurement(config: ComplexityBenchmarkConfig) -> dict[str, object]:
    """Measure modality scaling for the real one-step state-inference path."""
    samples: list[list[float]] = []
    digests: list[str] = []
    work: list[int] = []
    n_states = config.inference_state_count
    log_prior = np.log(np.full(n_states, 1.0 / n_states, dtype=np.float64))
    for index, n_modalities in enumerate(config.modality_sizes):
        matrices = _likelihoods(
            config.seed + 40_000 + index,
            n_modalities,
            config.observation_count,
            n_states,
        )
        observations = tuple(0 for _ in range(n_modalities))

        def _infer() -> object:
            return infer_states(matrices, observations, log_prior)

        samples.append(
            _timed_call(
                _infer,
                repeats=config.repeats,
                warmups=config.warmups,
            )
        )
        digests.append(_digest_arrays(matrices))
        work.append(
            estimate_complexity(
                "infer_states",
                n_agents=1,
                n_states=n_states,
                n_modalities=n_modalities,
            ).work_units
        )
    return _measurement(
        method="infer_states",
        axis="modalities",
        sizes=config.modality_sizes,
        samples=samples,
        expected_exponent=1.0,
        work_units=work,
        parameters={"S": n_states, "n_observations": config.observation_count},
        input_digests=digests,
        repeats=config.repeats,
        warmups=config.warmups,
        note="M-scaling with fixed state cardinality and observation-row count.",
    )


def _analytic_rows(config: ComplexityBenchmarkConfig) -> list[dict[str, object]]:
    """Attach representative concrete work proxies to the symbolic catalog."""
    rows: list[dict[str, object]] = []
    for spec in complexity_catalog():
        if spec.operation == "infer_states":
            estimate = estimate_complexity(
                spec.operation,
                n_agents=1,
                n_states=config.inference_state_count,
                n_modalities=max(config.modality_sizes),
            )
        elif spec.operation == "federation_server_round":
            estimate = estimate_complexity(
                spec.operation,
                n_agents=config.fixed_agent_count,
                n_states=config.fixed_state_count,
                iterations=config.max_iter,
            )
        elif spec.operation == "share_round_robust":
            estimate = estimate_complexity(
                spec.operation,
                n_agents=config.fixed_agent_count,
                n_states=config.fixed_state_count,
                iterations=_ROBUST_SHARING_ITERATIONS,
                exclude_self=True,
            )
        else:
            estimate = estimate_complexity(
                spec.operation,
                n_agents=config.fixed_agent_count,
                n_states=config.fixed_state_count,
                iterations=config.max_iter,
                n_starts=config.variational_starts,
                exclude_self=True,
            )
        rows.append(
            {
                **spec.as_dict(),
                "representative_estimate": estimate.as_dict(),
            }
        )
    return rows


def run_complexity_scaling(
    config: ComplexityBenchmarkConfig | None = None,
) -> dict[str, object]:
    """Run and return the source-bound complexity calculation and measurements.

    The report deliberately separates ``analytic_specs`` from ``measurements``:
    the former are implementation-derived asymptotic orders, while the latter
    are repeated wall-clock observations on the current machine. No confidence
    interval or cross-machine performance claim is attached to the timing
    slopes.
    """
    cfg = config or ComplexityBenchmarkConfig()
    measurements = _aggregation_measurements(cfg)
    measurements.extend(_sharing_measurements(cfg))
    measurements.append(_inference_measurement(cfg))
    return {
        "schema_version": "1.0",
        "status": "ok",
        "claim_boundary": (
            "Analytic orders are derived from the current dense NumPy implementation. "
            "Timing slopes are descriptive, machine-specific diagnostics on seeded inputs; "
            "they are not FLOP counts, inferential statistics, or cross-machine guarantees."
        ),
        "seed": int(cfg.seed),
        "machine": _machine_metadata(),
        "benchmark": cfg.as_dict(),
        "analytic_specs": _analytic_rows(cfg),
        "measurements": measurements,
    }


__all__ = ["run_complexity_scaling"]
