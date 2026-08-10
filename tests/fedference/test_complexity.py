"""Tests for implementation-derived complexity calculations."""

from __future__ import annotations

import pytest

from fedference.complexity import (
    ComplexityBenchmarkConfig,
    complexity_catalog,
    estimate_complexity,
)


def test_catalog_covers_core_and_federation_paths() -> None:
    operations = {spec.operation for spec in complexity_catalog()}
    assert operations == {
        "log_linear_pool",
        "robust_aggregate",
        "variational_aggregate",
        "aggregation_free_energy",
        "share_round_naive",
        "share_round_robust",
        "infer_states",
        "federation_server_round",
    }


def test_aggregation_work_proxies_follow_declared_orders() -> None:
    naive = estimate_complexity("log_linear_pool", n_agents=4, n_states=8)
    robust = estimate_complexity(
        "robust_aggregate",
        n_agents=4,
        n_states=8,
        iterations=3,
    )
    variational = estimate_complexity(
        "variational_aggregate",
        n_agents=4,
        n_states=8,
        iterations=3,
        n_starts=2,
    )
    assert naive.work_units == 32
    assert robust.work_units == 3 * naive.work_units
    assert variational.work_units == 2 * robust.work_units
    assert robust.workspace_units == 4 * 8 + 3 * 8
    assert variational.time_order == "Theta(B I N S)"


def test_leave_one_out_and_inference_proxies() -> None:
    sharing = estimate_complexity(
        "share_round_naive",
        n_agents=5,
        n_states=7,
        exclude_self=True,
    )
    sharing_no_exclusion = estimate_complexity(
        "share_round_naive",
        n_agents=5,
        n_states=7,
        exclude_self=False,
    )
    inference = estimate_complexity(
        "infer_states",
        n_agents=1,
        n_states=11,
        n_modalities=4,
    )
    server = estimate_complexity(
        "federation_server_round",
        n_agents=8,
        n_states=11,
        iterations=2,
    )
    assert sharing.work_units == 25 * 7
    assert sharing_no_exclusion.work_units == 5 * 7
    assert inference.work_units == 4 * 11
    assert server.work_units > 2 * 8 * 11


def test_unknown_operation_and_invalid_dimensions_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown complexity operation"):
        estimate_complexity("missing", n_agents=2, n_states=2)
    with pytest.raises(ValueError, match="n_agents"):
        estimate_complexity("log_linear_pool", n_agents=0, n_states=2)
    with pytest.raises(ValueError, match="iterations"):
        estimate_complexity("robust_aggregate", n_agents=2, n_states=2, iterations=0)


def test_benchmark_config_yaml_mapping_smoke_reduction_and_validation() -> None:
    config = ComplexityBenchmarkConfig.from_mapping(
        {
            "agent_sizes": [2, 4, 8],
            "state_sizes": [4, 8, 16],
            "sharing_agent_sizes": [2, 4, 8],
            "modality_sizes": [1, 2, 4],
            "fixed_agent_count": 4,
            "fixed_state_count": 8,
            "inference_state_count": 8,
            "observation_count": 3,
            "repeats": 2,
            "warmups": 1,
            "max_iter": 3,
            "variational_starts": 2,
            "seed": 11,
        }
    )
    assert config.as_dict()["agent_sizes"] == [2, 4, 8]
    assert config.seed == 11
    smoke = config.for_smoke()
    assert smoke.repeats == 2
    assert smoke.warmups == 0
    assert smoke.max_iter == 3
    assert smoke.fixed_agent_count == 4
    assert smoke.fixed_state_count == 8
    assert smoke.inference_state_count == 8
    with pytest.raises(ValueError, match="strictly increasing"):
        ComplexityBenchmarkConfig(agent_sizes=(4, 2, 8))
    with pytest.raises(ValueError, match="at least two sizes"):
        ComplexityBenchmarkConfig(modality_sizes=(1,))
    with pytest.raises(ValueError, match="must be an integer"):
        ComplexityBenchmarkConfig.from_mapping({"max_iter": 1.5})
