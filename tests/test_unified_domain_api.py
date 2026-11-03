"""Task52: source-pinned numerics and executable rejection of retired routes."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import sys
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from types import ModuleType
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
    """Pin structure exactly and float64 values within cross-libm roundoff."""
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
    elif isinstance(expected, float):
        assert isinstance(actual, (float, np.floating))
        assert np.isfinite(actual) and np.isfinite(expected)
        # The retained macOS fixture and Linux libm differ in their last bits.
        # No absolute slack: an expected zero must still be exactly zero.
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0, equal_nan=False)
    elif isinstance(expected, bool):
        assert isinstance(actual, (bool, np.bool_))
        assert bool(actual) is expected
    elif isinstance(expected, int):
        assert isinstance(actual, (int, np.integer)) and not isinstance(actual, (bool, np.bool_))
        assert actual == expected
    else:
        assert actual == expected


@pytest.mark.parametrize(
    ("actual", "expected"),
    (
        (0.02940496133272336, 0.029404961332723373),
        (1.0814907801671152, 1.0814907801671154),
        (0.0188324550232095, 0.018832455023209495),
        (0.01883245459245099, 0.018832454592450972),
        (0.2413399864597576, 0.24133998645975765),
        (0.007811403127670776, 0.007811403127670778),
        (0.004944319068453697, 0.004944319068453693),
        (-0.00962311143579847, -0.009623111435798474),
    ),
)
def test_frozen_characterization_accepts_observed_float64_platform_roundoff(
    actual: float, expected: float
) -> None:
    """Replay Linux values against the unchanged macOS characterization bytes."""
    _assert_pinned({"history": [actual]}, {"history": [expected]})


@pytest.mark.parametrize(
    ("actual", "expected"),
    (
        (0.25 + 1e-12, 0.25),
        (1e-300, 0.0),
        (float("nan"), 0.25),
        (float("inf"), float("inf")),
        (True, 1),
        (1, True),
        (1.0, 1),
        ({"iterations": 33}, {"iterations": 32}),
        ({"converged": False}, {"converged": True}),
        ({"fallback_events": ["changed"]}, {"fallback_events": []}),
        ({"history": [0.25]}, {"history": [0.25, 0.75]}),
        ({"extra": 0.25}, {"history": [0.25]}),
    ),
)
def test_frozen_characterization_rejects_semantic_or_structural_drift(
    actual: PinnedValue, expected: PinnedValue
) -> None:
    with pytest.raises(AssertionError):
        _assert_pinned(actual, expected)


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
    reference = _sharing_reference().share_round_result(
        case["rows"], config=config, base_weights=case["weights"],
        exclude_self=case["exclude_self"], true_state=0,
    )
    _assert_exact_report(shared, reference)


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


REFERENCE_SHA256 = "4408a7a80bfbe34ec58a62fce61f3424a33fa79bca08530726b97e251ceea0f7"
REFERENCE_PATH = Path(__file__).parent / "fixtures/api_migration/benchmark-3ec62f68.py.txt"


SHARING_REFERENCE_SHA256 = "7ae27fae8cf4dfa4e1a9b08aa9c5581260fdc3c298087fff033fad1cf68820e6"
SHARING_REFERENCE_PATH = REFERENCE_PATH.with_name("sharing-3ec62f68.py.txt")


def _reference_module(path: Path, expected_sha256: str, name: str) -> ModuleType:
    """Load only an immutable original under its own test-only module name."""
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclass annotation resolution needs a registered module while it loads.
    # This private name never replaces a production dependency or implementation.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous
    return module


def _benchmark_reference(path: Path = REFERENCE_PATH) -> ModuleType:
    module = _reference_module(path, REFERENCE_SHA256, "fedference._benchmark_migration_reference")
    assert module.run_benchmark is not run_benchmark
    assert module.aggregate_result is aggregate_result
    return module


def _sharing_reference(path: Path = SHARING_REFERENCE_PATH) -> ModuleType:
    module = _reference_module(path, SHARING_REFERENCE_SHA256, "fedference._sharing_migration_reference")
    assert module.share_round_result is not share_round_result
    assert module.aggregate_result is aggregate_result
    return module


def _assert_exact_report(actual: object, expected: object) -> None:
    """Retain exact report types, keys, values and integer diagnostics."""
    if dataclasses.is_dataclass(actual):
        assert dataclasses.is_dataclass(expected)
        actual, expected = dataclasses.asdict(actual), dataclasses.asdict(expected)
    assert type(actual) is type(expected)
    if isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected, strict=True)
    elif isinstance(actual, dict):
        assert isinstance(expected, dict) and actual.keys() == expected.keys()
        for key in actual:
            _assert_exact_report(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert isinstance(expected, (tuple, list)) and len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            _assert_exact_report(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("index", range(3))
def test_explicit_benchmark_inputs_preserve_reports(index: int) -> None:
    resource = resources.files("fedference").joinpath("data", "synthetic_tabular.csv")
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == PIN["dataset_sha256"]
    case = PIN["benchmark"][index]
    with resources.as_file(resource) as path:
        features, labels = load_tabular_csv(path)
        reference = _benchmark_reference().run_tabular_benchmark(path, seed=0, **case["kwargs"])
    report = run_benchmark(features, labels, seed=0, **case["kwargs"])
    report.update(dataset=resource.name, n_rows=int(features.shape[0]), n_features=int(features.shape[1]))
    # The JSON remains historical Mac evidence. Iteration counts are compared
    # exactly to the original boundary executed in this same runtime, not to a
    # cross-platform numerical reproducibility claim it never established.
    _assert_exact_report(report, reference)


def test_benchmark_reference_and_historical_record_keep_source_provenance(tmp_path: Path) -> None:
    metadata = json.loads((REFERENCE_PATH.parent / "benchmark-reference.json").read_text())
    assert metadata["revision"] == "3ec62f68f08b03dc22323fa4e8a67b0c87be894e"
    assert metadata["path"] == "src/fedference/benchmark.py"
    assert metadata["sha256"] == REFERENCE_SHA256
    historical = REFERENCE_PATH.parent / "domain-characterization.json"
    assert hashlib.sha256(historical.read_bytes()).hexdigest() == metadata["historical_json_sha256"]
    changed = tmp_path / "changed.py.txt"
    changed.write_bytes(REFERENCE_PATH.read_bytes() + b"\n# changed reference\n")
    with pytest.raises(AssertionError):
        _benchmark_reference(changed)
    assert _benchmark_reference().run_benchmark.__module__.startswith("fedference._benchmark_migration")


@pytest.mark.parametrize(
    ("actual", "expected"),
    ((16, 15), (13, 15), (0.25 + 1e-15, 0.25), (True, 1), (1.0, 1),
     ([0.25], [0.25, 0.75]), ({"extra": 1}, {"iterations": 1})),
)
def test_same_runtime_migration_oracle_rejects_changed_report(actual: object, expected: object) -> None:
    with pytest.raises(AssertionError):
        _assert_exact_report(actual, expected)


@pytest.mark.parametrize("field", ("seed", "robustness", "n_contaminated"))
def test_same_runtime_migration_oracle_rejects_wrong_current_parameters(field: str) -> None:
    resource = resources.files("fedference").joinpath("data", "synthetic_tabular.csv")
    case = PIN["benchmark"][2]
    kwargs = dict(case["kwargs"], seed=0)
    with resources.as_file(resource) as path:
        features, labels = load_tabular_csv(path)
        reference = _benchmark_reference().run_tabular_benchmark(path, **kwargs)
    kwargs[field] = {"seed": 1, "robustness": 2.25, "n_contaminated": 3}[field]
    changed = run_benchmark(features, labels, **kwargs)
    changed.update(dataset=resource.name, n_rows=int(features.shape[0]), n_features=int(features.shape[1]))
    with pytest.raises(AssertionError):
        _assert_exact_report(changed, reference)


def test_sharing_reference_has_original_provenance_and_rejects_changed_bytes(tmp_path: Path) -> None:
    metadata = json.loads((SHARING_REFERENCE_PATH.parent / "sharing-reference.json").read_text())
    assert metadata["revision"] == "3ec62f68f08b03dc22323fa4e8a67b0c87be894e"
    assert metadata["path"] == "src/fedference/belief_sharing.py"
    assert metadata["sha256"] == SHARING_REFERENCE_SHA256
    changed = tmp_path / "sharing.py.txt"
    changed.write_bytes(SHARING_REFERENCE_PATH.read_bytes() + b"\n# changed\n")
    with pytest.raises(AssertionError):
        _sharing_reference(changed)
    assert "fedference._sharing_migration_reference" not in sys.modules
    assert _sharing_reference().share_round_result.__module__ == "fedference._sharing_migration_reference"
    assert "fedference._sharing_migration_reference" not in sys.modules


@pytest.mark.parametrize("field", ("exclude_self", "true_state"))
def test_sharing_migration_refuses_changed_current_controls(field: str) -> None:
    case = PIN["cases"][4]
    kwargs = {"config": AggregationConfig(**case["config"]), "base_weights": case["weights"],
              "exclude_self": case["exclude_self"], "true_state": 0}
    reference = _sharing_reference().share_round_result(case["rows"], **kwargs)
    kwargs[field] = not case["exclude_self"] if field == "exclude_self" else 1
    changed = share_round_result(case["rows"], **kwargs)
    with pytest.raises(AssertionError):
        _assert_exact_report(changed, reference)


@pytest.mark.parametrize("kind", ("iterations", "array", "mean_accuracy", "dtype"))
def test_sharing_migration_refuses_changed_complete_diagnostics(kind: str) -> None:
    case = PIN["cases"][4]
    original = share_round_result(
        case["rows"], config=AggregationConfig(**case["config"]), base_weights=case["weights"],
        exclude_self=case["exclude_self"], true_state=0,
    )
    if kind == "iterations":
        global_result = dataclasses.replace(original.global_aggregation,
                                             iterations=original.global_aggregation.iterations + 1)
        changed = dataclasses.replace(original, global_aggregation=global_result)
    elif kind == "mean_accuracy":
        assert original.mean_accuracy is not None
        changed = dataclasses.replace(
            original, mean_accuracy=float(np.nextafter(original.mean_accuracy, 1.0))
        )
    elif kind == "array":
        shared = original.shared_posteriors.copy()
        shared[0] = np.roll(shared[0], 1)
        changed = dataclasses.replace(original, shared_posteriors=shared)
    else:
        with pytest.raises(AssertionError):
            _assert_exact_report(np.array([0.5], dtype=np.float32), np.array([0.5], dtype=np.float64))
        return
    with pytest.raises(AssertionError):
        _assert_exact_report(changed, original)


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
