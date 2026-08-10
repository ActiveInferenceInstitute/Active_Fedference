"""External tabular-benchmark harness (MAJ-6) — no mocks, real CSV + real fits.

Honest bindings, no overclaim:

1. Recovery identity — at robustness=0 the federated consensus is bit-identical
   to the naive log-linear pool (the guaranteed corner), for every run.
2. Clean data sanity — with no contaminated clients the well-separated dataset
   is classified perfectly by both naive and robust pools.
3. Contamination stress (measured, NOT a law) — under a heavy adversarial
   majority (4 of 5 clients label-shuffled) the robust pool shows a small
   MEASURED accuracy edge over the naive pool at the pinned seed; the harness
   makes no claim that robust always wins (MAJ-1 shows it has a finite
   breakdown point).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from fedference.benchmark import (
    load_tabular_csv,
    run_benchmark,
    run_external_benchmark_pack,
    run_tabular_benchmark,
    summarize_external_benchmark_rows,
)

_DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic_tabular.csv"


def test_dataset_loads_with_expected_shape() -> None:
    features, labels = load_tabular_csv(_DATA)
    assert features.shape == (150, 4)
    assert set(np.unique(labels).tolist()) == {0, 1, 2}


def test_repository_and_packaged_synthetic_fixtures_are_identical() -> None:
    packaged = resources.files("fedference").joinpath("data", "synthetic_tabular.csv")
    repository_bytes = _DATA.read_bytes()
    assert packaged.read_bytes() == repository_bytes
    assert b"\r" not in repository_bytes


def test_recovery_identity_holds_and_accuracies_valid() -> None:
    r = run_tabular_benchmark(seed=0)
    assert r["recovery_identity_holds"] is True
    assert 0.0 <= r["naive_accuracy"] <= 1.0
    assert 0.0 <= r["robust_accuracy"] <= 1.0
    assert r["dataset"] == "synthetic_tabular.csv"
    assert r["n_rows"] == 150
    assert r["n_features"] == 4


def test_clean_data_classified_perfectly() -> None:
    r = run_tabular_benchmark(seed=0, n_contaminated=0)
    assert r["naive_accuracy"] == 1.0
    assert r["robust_accuracy"] == 1.0
    assert r["robust_minus_naive"] == 0.0


def test_heavy_contamination_shows_measured_robust_edge() -> None:
    """Pinned measured fact at seed 0: with 4/5 clients label-shuffled, the
    robust pool holds slightly higher held-out accuracy than the naive pool.
    This is a MEASUREMENT, not a guarantee — asserted only at this setting."""
    r = run_tabular_benchmark(seed=0, n_contaminated=4, contamination_rate=1.0, robustness=2.5)
    assert r["naive_accuracy"] == 0.8222222222222222
    assert r["robust_accuracy"] == 0.8444444444444444
    assert r["robust_minus_naive"] > 0.0
    assert r["recovery_identity_holds"] is True


def test_run_benchmark_is_deterministic() -> None:
    features, labels = load_tabular_csv(_DATA)
    a = run_benchmark(features, labels, seed=3, n_contaminated=3)
    b = run_benchmark(features, labels, seed=3, n_contaminated=3)
    assert a == b


def test_benchmark_reports_proper_scores_calibration_and_split_receipt() -> None:
    features, labels = load_tabular_csv(_DATA)
    report = run_benchmark(
        features,
        labels,
        seed=4,
        n_contaminated=2,
        contamination_rate=0.5,
    )
    for method in ("naive", "robust", "variational"):
        assert np.isfinite(report[f"{method}_log_score"])
        assert 0.0 <= report[f"{method}_ece"] <= 1.0
        assert 0.0 <= report[f"{method}_accuracy"] <= 1.0
        assert 0 <= report[f"{method}_fallback_predictions"] <= report["n_test"]
        assert 0 <= report[f"{method}_nonconverged_predictions"] <= report["n_test"]
        assert report[f"{method}_max_iterations"] >= 0
    assert report["robust_minus_naive_log_score"] == pytest.approx(
        report["robust_log_score"] - report["naive_log_score"]
    )
    assert len(report["split_sha256"]) == 64
    assert "training-split" in report["preprocessing"]
    assert sum(report["train_class_counts"]) + sum(report["test_class_counts"]) == 150
    assert report["contaminated_client_ids"] == [0, 1]


def test_pack_rejects_duplicate_or_invalid_nested_units(tmp_path) -> None:
    with pytest.raises(ValueError, match="dataset_ids must be unique"):
        run_external_benchmark_pack(
            cache_dir=tmp_path,
            dataset_ids=("uci-banknote", "uci-banknote"),
        )
    with pytest.raises(ValueError, match="seeds must be unique"):
        run_external_benchmark_pack(cache_dir=tmp_path, seeds=(1, 1))
    with pytest.raises(ValueError, match="non-negative integers"):
        run_external_benchmark_pack(cache_dir=tmp_path, seeds=(-1,))


def test_external_summary_keeps_dataset_as_the_independent_unit() -> None:
    rows = [
        {
            "dataset_id": "uci-a",
            "robust_minus_naive_log_score": 0.1,
            "variational_minus_naive_log_score": -0.1,
        },
        {
            "dataset_id": "uci-a",
            "robust_minus_naive_log_score": 0.3,
            "variational_minus_naive_log_score": 0.1,
        },
        {
            "dataset_id": "uci-b",
            "robust_minus_naive_log_score": -0.2,
            "variational_minus_naive_log_score": 0.0,
        },
    ]
    summary = summarize_external_benchmark_rows(rows)
    assert summary["uci-a"]["n_seed_runs"] == 2
    assert summary["uci-a"]["robust_minus_naive_log_score_mean"] == pytest.approx(0.2)
    assert summary["uci-b"]["robust_minus_naive_log_score_mcse"] == 0.0


@pytest.mark.parametrize(
    ("contents", "message"),
    (
        ("", "empty"),
        ("f0,f1\n1,2\n", "exactly one 'label'"),
        ("f0,label\n1,0\n2\n", "has 1 fields"),
        ("f0,label\nnot-a-number,0\n", "non-numeric"),
        ("f0,label\n1,0.5\n", "finite integer"),
    ),
)
def test_csv_loader_rejects_malformed_inputs(tmp_path, contents, message) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_tabular_csv(path)


def test_benchmark_rejects_boolean_or_invalid_numeric_controls() -> None:
    features, labels = load_tabular_csv(_DATA)
    with pytest.raises(ValueError, match="contamination_rate"):
        run_benchmark(features, labels, contamination_rate=True)
    with pytest.raises(ValueError, match="test_fraction"):
        run_benchmark(features, labels, test_fraction=True)
    with pytest.raises(ValueError, match="robustness"):
        run_benchmark(features, labels, robustness=np.nan)
