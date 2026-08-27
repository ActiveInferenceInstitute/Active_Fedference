"""Compare the three declared server aggregation rules on fixed beliefs."""

from __future__ import annotations

import json

import numpy as np

from fedference import AggregationConfig, aggregate_result


def _rounded(values: np.ndarray) -> list[float]:
    """Return stable six-decimal output without changing the computation."""
    return [round(float(value), 6) for value in values]


def main() -> None:
    """Run naive, robust-heuristic, and variational rules side by side."""
    local_posteriors = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.30, 0.10],
            [0.10, 0.10, 0.80],
        ],
        dtype=np.float64,
    )
    configs = (
        AggregationConfig(method="naive"),
        AggregationConfig(
            method="robust",
            robustness=1.5,
            max_iter=64,
            tol=1e-9,
        ),
        AggregationConfig(
            method="variational",
            robustness=1.5,
            entropy_weight=1.0,
            max_iter=64,
            tol=1e-9,
            multistart=True,
        ),
    )

    methods: dict[str, object] = {}
    consensus_by_method: dict[str, np.ndarray] = {}
    for config in configs:
        result = aggregate_result(local_posteriors, config=config)
        consensus_by_method[config.method] = result.consensus
        methods[config.method] = {
            "config_fingerprint": config.fingerprint,
            "consensus": _rounded(result.consensus),
            "converged": result.converged,
            "fallback_events": list(result.fallback_events),
            "iterations": result.iterations,
            "normalized_effective_weights": _rounded(result.normalized_effective_weights),
        }

    zero_robustness_configs = (
        AggregationConfig(
            method="robust",
            robustness=0.0,
            max_iter=64,
            tol=1e-9,
        ),
        AggregationConfig(
            method="variational",
            robustness=0.0,
            entropy_weight=1.0,
            max_iter=64,
            tol=1e-9,
            multistart=True,
        ),
    )
    zero_robustness_recovery: dict[str, object] = {}
    for config in zero_robustness_configs:
        result = aggregate_result(local_posteriors, config=config)
        zero_robustness_recovery[config.method] = {
            "bit_identical_to_naive": bool(
                np.array_equal(result.consensus, consensus_by_method["naive"])
            ),
            "consensus": _rounded(result.consensus),
            "entropy_weight": config.entropy_weight,
            "robustness": config.robustness,
        }

    payload = {
        "example": "compare_aggregation_methods",
        "input": [_rounded(row) for row in local_posteriors],
        "methods": methods,
        "scope": {
            "naive": "categorical posterior-log-potential specialization",
            "robust": "project server heuristic",
            "variational": "objective-backed conservative server rule",
        },
        "zero_robustness_recovery": zero_robustness_recovery,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
