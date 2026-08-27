"""Minimal deterministic use of the canonical Active Fedference API."""

from __future__ import annotations

import json

import numpy as np

from fedference import AggregationConfig, aggregate_result


def _rounded(values: np.ndarray) -> list[float]:
    """Return a stable, reader-sized representation of a numeric vector."""
    return [round(float(value), 6) for value in values]


def main() -> None:
    """Aggregate two categorical posteriors and expose validation behavior."""
    local_posteriors = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.30, 0.10],
        ],
        dtype=np.float64,
    )
    base_weights = np.asarray([1.0, 2.0], dtype=np.float64)
    config = AggregationConfig(method="naive")
    result = aggregate_result(
        local_posteriors,
        config=config,
        base_weights=base_weights,
    )

    try:
        aggregate_result(
            [[0.70, 0.30], [0.40, -0.10]],
            config=config,
        )
    except ValueError as exc:
        validation_error = {"message": str(exc), "type": type(exc).__name__}
    else:  # pragma: no cover - the public validation contract must fail closed
        raise AssertionError("negative probabilities were accepted")

    payload = {
        "config": config.as_dict(),
        "example": "minimal_aggregation",
        "input": {
            "base_weights": _rounded(base_weights),
            "local_posteriors": [_rounded(row) for row in local_posteriors],
        },
        "result": {
            "consensus": _rounded(result.consensus),
            "converged": result.converged,
            "fallback_events": list(result.fallback_events),
            "iterations": result.iterations,
            "normalized_effective_weights": _rounded(result.normalized_effective_weights),
            "raw_effective_weights": _rounded(result.raw_effective_weights),
            "sums_to_one": bool(np.isclose(result.consensus.sum(), 1.0)),
        },
        "validation_error": validation_error,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
