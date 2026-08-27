"""Strict labeled application API and solver-health contracts."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import fedference.application as application_module
from fedference import (
    AgentPosterior,
    AggregationConfig,
    AggregationResult,
    LabeledAggregationRequest,
    LabeledAggregationResult,
    __version__,
    aggregate_labeled,
)
from fedference.aggregation import aggregation_free_energy


def _raw_request() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "state_labels": ["clear", "warning", "critical"],
        "agents": [
            {"agent_id": "agent-0", "posterior": [7.0, 2.0, 1.0]},
            {
                "agent_id": "agent-1",
                "posterior": [1.0, 3.0, 6.0],
                "base_weight": 2.0,
            },
        ],
        "aggregation_config": {
            "method": "robust",
            "robustness": 1.5,
            "entropy_weight": 1.0,
            "max_iter": 64,
            "tol": 1e-9,
            "multistart": True,
        },
    }


def test_labeled_request_preserves_masses_and_result_round_trips_exactly() -> None:
    request = LabeledAggregationRequest.from_dict(_raw_request())
    assert request.agents[0].posterior.tolist() == [7.0, 2.0, 1.0]
    assert request.agents[0].base_weight == 1.0
    assert request.as_dict()["agents"][0]["base_weight"] == 1.0
    result = aggregate_labeled(request)
    assert result.software_version == __version__
    assert np.allclose(result.normalized_local_posteriors.sum(axis=1), 1.0)
    assert result.state_labels == ("clear", "warning", "critical")
    assert result.agent_ids == ("agent-0", "agent-1")
    raw = result.as_dict()
    assert set(raw) == {
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
    assert not ({"winner", "decision", "accepted", "threshold"} & set(raw))
    assert LabeledAggregationResult.from_dict(raw).as_dict() == raw
    assert LabeledAggregationResult.from_json(json.dumps(raw)).as_dict() == raw
    with pytest.raises(ValueError, match="config_fingerprint"):
        LabeledAggregationResult.from_dict({**raw, "config_fingerprint": "0" * 64})
    inconsistent = json.loads(json.dumps(raw))
    inconsistent["aggregation"]["solver_status"] = "not_converged"
    with pytest.raises(ValueError, match="solver_status"):
        LabeledAggregationResult.from_dict(inconsistent)


def test_aggregate_labeled_delegates_exactly_once_without_reimplementing_math(
) -> None:
    request = LabeledAggregationRequest.from_dict(_raw_request())
    dispatcher_code = application_module.aggregate_result.__code__
    call_count = 0

    def observe_real_dispatch(frame, event, _arg) -> None:
        nonlocal call_count
        if event == "call" and frame.f_code is dispatcher_code:
            call_count += 1

    previous_profiler = sys.getprofile()
    sys.setprofile(observe_real_dispatch)
    try:
        result = aggregate_labeled(request)
    finally:
        sys.setprofile(previous_profiler)

    direct = application_module.aggregate_result(
        (agent.posterior for agent in request.agents),
        config=request.config,
        base_weights=(agent.base_weight for agent in request.agents),
    )
    assert call_count == 1
    assert result.aggregation.as_dict() == direct.as_dict()


def test_semantic_request_hash_ignores_formatting_and_expands_default_weight() -> None:
    compact = json.dumps(_raw_request(), separators=(",", ":"))
    reordered = json.dumps(_raw_request(), indent=4, sort_keys=True)
    first = LabeledAggregationRequest.from_json(compact)
    second = LabeledAggregationRequest.from_json(reordered)
    assert first.request_sha256 == second.request_sha256
    explicit = _raw_request()
    explicit["agents"][0]["base_weight"] = 1.0
    assert LabeledAggregationRequest.from_dict(explicit).request_sha256 == first.request_sha256


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update(extra=True), "fields do not match"),
        (lambda raw: raw["state_labels"].append("clear"), "unique"),
        (lambda raw: raw["state_labels"].__setitem__(0, " clear"), "whitespace"),
        (lambda raw: raw["agents"].append(dict(raw["agents"][0])), "agent_id values"),
        (lambda raw: raw["agents"][0].update(agent_id="agent-0 "), "whitespace"),
        (lambda raw: raw["agents"][0].update(posterior=[1.0, 0.0]), "match state_labels"),
        (lambda raw: raw["agents"][0].update(posterior=[[1.0, 0.0, 0.0]]), "one-dimensional"),
        (lambda raw: raw["agents"][0].update(posterior=[1.0, -1.0, 1.0]), "non-negative"),
        (lambda raw: raw["agents"][0].update(posterior=[0.0, 0.0, 0.0]), "positive"),
        (lambda raw: raw["agents"][0].update(posterior=[True, 0.0, 1.0]), "booleans"),
        (lambda raw: raw["agents"][0].update(extra=1), "agent fields"),
        (
            lambda raw: [agent.update(base_weight=0.0) for agent in raw["agents"]],
            "at least one positive",
        ),
        (
            lambda raw: raw["aggregation_config"].update(extra=1),
            "aggregation_config fields",
        ),
    ],
)
def test_labeled_request_rejects_malformed_semantics(mutate, message: str) -> None:
    raw = _raw_request()
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        LabeledAggregationRequest.from_dict(raw)


def test_labeled_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    duplicate = json.dumps(_raw_request()).replace(
        '"schema_version": "1.0",',
        '"schema_version": "1.0", "schema_version": "1.0",',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        LabeledAggregationRequest.from_json(duplicate)
    nonfinite = json.dumps(_raw_request()).replace("7.0", "NaN", 1)
    with pytest.raises(ValueError, match="non-finite"):
        LabeledAggregationRequest.from_json(nonfinite)


@pytest.mark.parametrize(
    ("converged", "fallbacks", "expected"),
    [
        (True, (), "nominal"),
        (True, ("fallback",), "converged_with_fallback"),
        (False, (), "not_converged"),
        (False, ("fallback",), "not_converged_with_fallback"),
    ],
)
def test_aggregation_result_solver_status_and_json_contract(
    converged: bool,
    fallbacks: tuple[str, ...],
    expected: str,
) -> None:
    result = AggregationResult(
        consensus=np.asarray([0.6, 0.4]),
        raw_effective_weights=np.asarray([0.25, 0.5]),
        normalized_effective_weights=np.asarray([1.0 / 3.0, 2.0 / 3.0]),
        iterations=1,
        converged=converged,
        history=[np.asarray([0.5, 0.5]), np.asarray([0.6, 0.4])],
        free_energy_history=[1.0],
        fallback_events=fallbacks,
    )
    assert result.solver_status == expected
    assert AggregationResult.from_dict(result.as_dict()).as_dict() == result.as_dict()
    json.dumps(result.as_dict(), allow_nan=False)
    assert not result.consensus.flags.writeable
    assert all(not iterate.flags.writeable for iterate in result.history)


def test_public_aggregation_vectors_reject_implicit_flattening() -> None:
    kwargs = {
        "local_posteriors": [[0.7, 0.3], [0.2, 0.8]],
        "base_weights": [1.0, 1.0],
        "robustness": 1.0,
    }
    with pytest.raises(ValueError, match="consensus must be one-dimensional"):
        aggregation_free_energy(
            consensus_posterior=np.asarray([[0.5, 0.5]]),
            raw_effective_weights=[1.0, 1.0],
            **kwargs,
        )
    with pytest.raises(ValueError, match="raw_effective_weights must be one-dimensional"):
        aggregation_free_energy(
            consensus_posterior=[0.5, 0.5],
            raw_effective_weights=np.asarray([[1.0], [1.0]]),
            **kwargs,
        )
    for field_name in (
        "consensus",
        "raw_effective_weights",
        "normalized_effective_weights",
    ):
        values = {
            "consensus": np.asarray([0.5, 0.5]),
            "raw_effective_weights": np.asarray([1.0, 1.0]),
            "normalized_effective_weights": np.asarray([0.5, 0.5]),
            "iterations": 0,
            "converged": True,
        }
        values[field_name] = np.asarray([[0.5, 0.5]])
        with pytest.raises(ValueError, match="one-dimensional"):
            AggregationResult(**values)


def test_labeled_result_rejects_weight_agent_mismatch() -> None:
    aggregation = AggregationResult(
        consensus=np.asarray([0.5, 0.5]),
        raw_effective_weights=np.asarray([1.0]),
        normalized_effective_weights=np.asarray([1.0]),
        iterations=0,
        converged=True,
    )
    with pytest.raises(ValueError, match="weights must match agent_ids"):
        LabeledAggregationResult(
            state_labels=("a", "b"),
            agent_ids=("agent-0", "agent-1"),
            normalized_local_posteriors=np.asarray([[0.5, 0.5], [0.5, 0.5]]),
            config=AggregationConfig(),
            request_sha256="a" * 64,
            aggregation=aggregation,
            software_version=__version__,
        )


def test_agent_posterior_arrays_are_copied_and_read_only() -> None:
    source = np.asarray([2.0, 1.0])
    agent = AgentPosterior("agent", source)
    source[0] = 99.0
    assert agent.posterior.tolist() == [2.0, 1.0]
    with pytest.raises(ValueError):
        agent.posterior[0] = 1.0
