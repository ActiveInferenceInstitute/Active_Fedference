"""Task52: source-pinned numerics and executable rejection of retired routes."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pytest

from fedference import (
    AggregationConfig,
    AggregationResult,
    SharingRoundResult,
    aggregate_result,
    share_round_result,
)
from fedference.aggregation import log_linear_pool, robust_aggregate, variational_aggregate
from fedference.aggregation_comparators import (
    ComparatorResult,
    clr_geometric_median_pool,
    linear_opinion_pool,
)
from fedference.benchmark import load_tabular_csv, run_benchmark
from fedference.bnn_baseline import fed_gvi_logreg

PIN = json.loads((Path(__file__).parent / "fixtures/api_migration/domain-characterization.json").read_text())


PinnedValue: TypeAlias = (
    AggregationResult
    | SharingRoundResult
    | ComparatorResult
    | np.ndarray
    | np.generic
    | None
    | bool
    | int
    | float
    | str
    | list["PinnedValue"]
    | tuple["PinnedValue", ...]
    | dict[str, "PinnedValue"]
)


def _assert_pinned(actual: PinnedValue, expected: PinnedValue) -> None:
    """Compare all stored diagnostic fields, including solver histories exactly."""
    if dataclasses.is_dataclass(actual):
        actual = {field.name: getattr(actual, field.name) for field in dataclasses.fields(actual)}
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_pinned(actual[key], expected[key])
    elif isinstance(actual, (np.ndarray, list, tuple)):
        assert isinstance(expected, (np.ndarray, list, tuple))
        assert len(actual) == len(expected)
        for value, pinned in zip(actual, expected, strict=True):
            _assert_pinned(value, pinned)
    else:
        assert actual == expected


@pytest.mark.parametrize("index", range(6))
def test_canonical_result_paths_preserve_numeric_controls(index: int) -> None:
    # Given pre-migration real outputs for every method and recipient policy.
    case = PIN["cases"][index]
    config = AggregationConfig(**case["config"])
    # When canonical dispatch and sharing run on exactly the pinned inputs.
    direct = aggregate_result(case["rows"], config=config, base_weights=case["weights"])
    shared = share_round_result(
        case["rows"],
        config=config,
        base_weights=case["weights"],
        exclude_self=case["exclude_self"],
        true_state=0,
    )
    # Then every numeric and diagnostic field matches the frozen characterization.
    _assert_pinned(direct, case["result"])
    _assert_pinned(shared, case["sharing"])


@pytest.mark.parametrize("iterations", (32, 64))
def test_iteration_sensitive_solver_budget_is_preserved(iterations: int) -> None:
    case = PIN["iteration_sensitive"]
    result = aggregate_result(
        case["rows"], config=AggregationConfig(method="robust", robustness=1.0, max_iter=iterations, tol=1e-9)
    )
    _assert_pinned(result, case[str(iterations)])
    assert not np.array_equal(case["32"]["consensus"], case["64"]["consensus"])


def test_six_substantive_scientific_controls_remain() -> None:
    rows = np.array([[0.8, 0.2], [0.5, 0.5]])
    weights = np.array([2.0, 1.0])
    controls = PIN["scientific_controls"]
    pool = log_linear_pool(rows, base_weights=weights)
    np.testing.assert_allclose(pool, [16 / 17, 1 / 17], rtol=1e-14, atol=0)
    _assert_pinned(pool, controls["log_linear_pool"])
    _assert_pinned(
        robust_aggregate(rows, base_weights=weights, robustness=0).consensus, controls["robust_aggregate"]
    )
    _assert_pinned(
        variational_aggregate(rows, base_weights=weights, robustness=0).consensus,
        controls["variational_aggregate"],
    )
    _assert_pinned(linear_opinion_pool(rows, base_weights=weights), controls["linear_opinion_pool"])
    _assert_pinned(
        clr_geometric_median_pool(rows, base_weights=weights), controls["clr_geometric_median_pool"]
    )
    _assert_pinned(fed_gvi_logreg(n_clients=2, n_per=12, seed=52), controls["fed_gvi_logreg"])


@pytest.mark.parametrize("index", range(3))
def test_explicit_benchmark_inputs_preserve_reports(index: int) -> None:
    resource = resources.files("fedference").joinpath("data", "synthetic_tabular.csv")
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == PIN["dataset_sha256"]
    with resources.as_file(resource) as path:
        features, labels = load_tabular_csv(path)
    case = PIN["benchmark"][index]
    report = run_benchmark(features, labels, seed=0, **case["kwargs"])
    report.update(dataset=resource.name, n_rows=int(features.shape[0]), n_features=int(features.shape[1]))
    assert report == case["report"]


def test_missing_truth_is_none_and_naive_influence_is_normalized() -> None:
    rows = np.array([[0.8, 0.2], [0.5, 0.5]])
    result = share_round_result(rows, base_weights=[2.0, 1.0], exclude_self=False)
    assert result.mean_accuracy is None
    assert result.mean_surprise is None
    np.testing.assert_array_equal(result.normalized_effective_weights, [2 / 3, 1 / 3])
    np.testing.assert_allclose(result.consensus, [16 / 17, 1 / 17])


@pytest.mark.parametrize(
    ("module", "symbol"),
    (
        ("fedference.aggregation", "aggregate"),
        ("fedference.aggregation", "friston_belief_share"),
        ("fedference.belief_sharing", "share_round"),
        ("fedference.belief_sharing", "SharingDiagnostics"),
        ("fedference.benchmark", "run_tabular_benchmark"),
        *(
            (module, name)
            for module in ("fedference", "src")
            for name in ("aggregate", "friston_belief_share", "share_round", "SharingDiagnostics")
        ),
    ),
)
def test_removed_wrappers_and_selector_keywords_reject(module: str, symbol: str) -> None:
    # Given the actual importable module, not source-string matching.
    imported = importlib.import_module(module)
    # When an obsolete consumer resolves or imports the removed name.
    with pytest.raises(AttributeError):
        getattr(imported, symbol)
    with pytest.raises(ImportError):
        exec(f"from {module} import {symbol}", {})


@pytest.mark.parametrize("function", (aggregate_result, share_round_result))
@pytest.mark.parametrize("keyword", ("method", "robustness", "max_iter", "tol", "beliefs", "weights"))
def test_canonical_selectors_reject_duplicate_keyword_routes(
    function: Callable[..., AggregationResult | SharingRoundResult],
    keyword: str,
) -> None:
    rows = [[0.8, 0.2], [0.5, 0.5]]
    with pytest.raises(TypeError, match="unexpected keyword"):
        function(rows, config=AggregationConfig(), **{keyword: "naive"})


@pytest.mark.parametrize(
    ("contents", "message"),
    (
        (b"f0,f0,label\n1,2,0\n", "unique feature names"),
        (b"f0,label\n", "no data rows"),
        (b"f0,label\nnan,0\n", "features must be finite"),
        (b"\xff", "cannot read"),
    ),
)
def test_explicit_csv_inputs_reject_invalid_data(
    tmp_path: Path,
    contents: bytes,
    message: str,
) -> None:
    # Given real caller-owned CSV bytes.
    path = tmp_path / "invalid.csv"
    path.write_bytes(contents)
    # When parsing explicit input, reject it before benchmark fitting.
    with pytest.raises(ValueError, match=message):
        load_tabular_csv(path)


def test_explicit_csv_missing_input_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        load_tabular_csv(tmp_path / "missing.csv")


def test_explicit_csv_loader_skips_blank_rows_and_preserves_values(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_text("f0,label\n\n1.25,0\n2.5,1\n")
    features, labels = load_tabular_csv(path)
    np.testing.assert_array_equal(features, [[1.25], [2.5]])
    np.testing.assert_array_equal(labels, [0, 1])


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("consensus", [True, False], "numeric non-boolean"),
        ("normalized_effective_weights", [True, False], "numeric non-boolean"),
        ("normalized_effective_weights", [], "finite and non-empty"),
        ("normalized_effective_weights", [-0.1, 1.1], "probability vector"),
        ("raw_effective_weights", [True, False], "numeric non-boolean"),
        ("raw_effective_weights", [1.0], "must match"),
        ("iterations", -1, "non-negative integer"),
        ("converged", 1, "must be a boolean"),
        ("history", None, "must be a sequence"),
        ("history", [[1.0]], "match consensus shape"),
        ("history", [[True, False]], "numeric non-boolean"),
        ("history", [[0.2, 0.2]], "finite probability vector"),
        ("free_energy_history", None, "must be a sequence"),
        ("free_energy_history", [True], "only numeric"),
        ("free_energy_history", [float("inf")], "only finite"),
        ("fallback_events", [""], "non-empty strings"),
    ),
)
def test_rich_aggregation_rejects_invalid_diagnostics(
    field: str,
    invalid: PinnedValue,
    message: str,
) -> None:
    # Given a real canonical result with valid probabilities and weights.
    valid = aggregate_result([[0.8, 0.2], [0.5, 0.5]])
    # When malformed diagnostics cross its public result boundary.
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(valid, **{field: invalid})


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("shared_posteriors", [], "non-empty finite matrix"),
        ("recipient_aggregations", (), "one result per shared posterior"),
        ("exclude_self", 1, "must be a boolean"),
    ),
)
def test_rich_sharing_rejects_invalid_diagnostics(
    field: str,
    invalid: PinnedValue,
    message: str,
) -> None:
    valid = share_round_result([[0.8, 0.2], [0.5, 0.5]])
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(valid, **{field: invalid})


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"labels": ["a"] * 8}, "numeric integers"),
        ({"labels": [0.5, 1.0] * 4}, "finite integers"),
        ({"labels": [0, 1]}, "aligned rows"),
        ({"features": [[float("nan")]] * 8}, "finite values"),
        ({"labels": [0, 2] * 4}, "contiguous non-negative"),
        ({"n_clients": 0}, "n_clients must be positive"),
        ({"n_contaminated": 3}, "n_contaminated must lie"),
    ),
)
def test_explicit_benchmark_inputs_reject_invalid_arrays_and_controls(
    overrides: dict[str, PinnedValue],
    message: str,
) -> None:
    # Given caller-provided numeric arrays and one malformed explicit input.
    arguments: dict[str, PinnedValue] = {
        "features": np.arange(8, dtype=np.float64).reshape(8, 1),
        "labels": np.asarray([0, 1] * 4),
        "n_clients": 2,
    }
    arguments.update(overrides)
    # When crossing the benchmark's array/configuration boundary.
    with pytest.raises(ValueError, match=message):
        run_benchmark(**arguments)
