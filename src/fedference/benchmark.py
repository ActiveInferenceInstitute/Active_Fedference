"""External tabular-benchmark stress harness (MAJ-6 slice).

Applies the federated belief-sharing machinery to a real tabular classification
task read from a CSV, instead of the synthetic sentinel world. Each of several
clients fits a class-conditional Gaussian (Gaussian naive Bayes) on its own data
shard and emits, for each held-out point, a posterior pmf over the classes; those
per-client pmfs are federated with the stable server rules. Held-out log score
is primary; accuracy and expected calibration error are secondary.

Two honest bindings:

* **Recovery identity** — at ``robustness = 0`` the federated consensus is
  bit-identical to the naive log-linear pool (the proven corner), so the harness
  cannot silently change the aggregation math.
* **Contamination stress** — when a fraction of clients have training labels
  replaced uniformly at random (``label_noise``), every method is compared on
  the same train/test split and client shards.

The compatibility path remains exercised on the bundled synthetic CSV. The
registered path downloads and verifies three pinned UCI archives into a caller-
provided cache and records archive/member/split provenance. That executable
path is not itself the confirmatory evidence pack: pilots, locked comparisons,
dataset-level inference, negative controls, manuscript artifacts, and a release
receipt remain open.
"""

from __future__ import annotations

import csv
import hashlib
from importlib import resources
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from .aggregation import AggregationConfig, aggregate_result, log_linear_pool, robust_aggregate
from .evidence import canonical_sha256
from .external_data import fetch_external_dataset
from .research_registry import DATASET_SPECS

ArrayF = np.ndarray


class _BenchmarkMetrics(TypedDict):
    """Predictive metrics plus solver-health counts at held-out-point grain."""

    accuracy: float
    log_score: float
    ece: float
    fallback_predictions: int
    nonconverged_predictions: int
    max_iterations: int


def load_tabular_csv(path: str | Path) -> tuple[ArrayF, ArrayF]:
    """Load a numeric-feature + integer-``label`` CSV into ``(features, labels)``."""
    rows: list[list[float]] = []
    labels: list[int] = []
    try:
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError("tabular CSV is empty") from exc
            if header.count("label") != 1:
                raise ValueError("tabular CSV must contain exactly one 'label' column")
            if len(header) < 2 or len(set(header)) != len(header):
                raise ValueError("tabular CSV header must contain unique feature names")
            label_idx = header.index("label")
            feat_idx = [i for i in range(len(header)) if i != label_idx]
            for line_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        f"tabular CSV row {line_number} has {len(row)} fields; expected {len(header)}"
                    )
                features = [float(row[i]) for i in feat_idx]
                label_value = float(row[label_idx])
                if not np.isfinite(label_value) or label_value != np.floor(label_value):
                    raise ValueError(f"tabular CSV row {line_number} label must be a finite integer")
                rows.append(features)
                labels.append(int(label_value))
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read tabular CSV: {path}") from exc
    except ValueError as exc:
        if str(exc).startswith("tabular CSV"):
            raise
        raise ValueError(f"tabular CSV contains a non-numeric value: {path}") from exc
    if not rows:
        raise ValueError("tabular CSV contains no data rows")
    features_array = np.asarray(rows, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    if not np.all(np.isfinite(features_array)):
        raise ValueError("tabular CSV features must be finite")
    return features_array, labels_array


def _gaussian_nb_fit(x: ArrayF, y: ArrayF, n_classes: int, *, var_floor: float = 1e-3) -> dict[str, ArrayF]:
    """Fit a diagonal-covariance Gaussian naive Bayes on one shard."""
    global_mean = x.mean(axis=0)
    global_variance = np.maximum(x.var(axis=0), var_floor)
    means = np.repeat(global_mean[None, :], n_classes, axis=0)
    variances = np.repeat(global_variance[None, :], n_classes, axis=0)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    priors = (counts + 1.0) / (x.shape[0] + n_classes)
    for c in range(n_classes):
        rows = x[y == c]
        if rows.shape[0] > 0:
            means[c] = rows.mean(axis=0)
            variances[c] = np.maximum(rows.var(axis=0), var_floor)
    return {"means": means, "variances": variances, "priors": priors}


def _gaussian_nb_pmf(model: dict[str, ArrayF], point: ArrayF) -> ArrayF:
    """Posterior pmf over classes for one point under a fitted GNB."""
    means, variances, priors = model["means"], model["variances"], model["priors"]
    log_lik = -0.5 * (
        np.log(2.0 * np.pi * variances).sum(axis=1) + (((point - means) ** 2) / variances).sum(axis=1)
    )
    log_post = np.log(np.clip(priors, 1e-12, None)) + log_lik
    log_post -= log_post.max()
    p = np.exp(log_post)
    return p / p.sum()


def _shard_indices(n: int, n_clients: int, rng: np.random.Generator) -> list[ArrayF]:
    perm = rng.permutation(n)
    return [np.sort(s) for s in np.array_split(perm, n_clients)]


def _expected_calibration_error(
    probabilities: ArrayF,
    labels: ArrayF,
    *,
    n_bins: int = 10,
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(selected):
            error += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return error


def run_benchmark(
    features: ArrayF,
    labels: ArrayF,
    *,
    n_clients: int = 5,
    n_contaminated: int = 2,
    robustness: float = 1.5,
    contamination_rate: float = 1.0,
    test_fraction: float = 0.3,
    entropy_weight: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Federated tabular classification with contaminated clients.

    Splits the data into a train/test partition, shards the train set across
    ``n_clients`` Gaussian-NB clients, shuffles the labels of ``n_contaminated``
    clients (uniform random-replacement ``label_noise`` at
    ``contamination_rate``), and for each test point
    federates the per-client class pmfs with both the naive pool and
    ``robust_aggregate``. Reports held-out accuracy for naive vs robust, plus the
    recovery-identity check at ``robustness = 0``.
    """
    x = np.asarray(features, dtype=np.float64)
    raw_y = np.asarray(labels)
    if not np.issubdtype(raw_y.dtype, np.number):
        raise ValueError("labels must be numeric integers")
    numeric_y = np.asarray(raw_y, dtype=np.float64).ravel()
    if not np.all(np.isfinite(numeric_y)) or not np.all(numeric_y == np.floor(numeric_y)):
        raise ValueError("labels must contain only finite integers")
    y = numeric_y.astype(np.int64)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 4:
        raise ValueError("features and labels must define at least four aligned rows")
    if not np.all(np.isfinite(x)):
        raise ValueError("features must contain only finite values")
    if (
        np.any(y < 0)
        or np.unique(y).size < 2
        or not np.array_equal(np.unique(y), np.arange(int(y.max()) + 1))
    ):
        raise ValueError("labels must be contiguous non-negative integers")
    if (
        isinstance(n_clients, bool)
        or not isinstance(n_clients, (int, np.integer))
        or n_clients < 1
        or n_clients >= x.shape[0]
    ):
        raise ValueError("n_clients must be positive and smaller than the row count")
    if (
        isinstance(n_contaminated, bool)
        or not isinstance(n_contaminated, (int, np.integer))
        or not 0 <= n_contaminated <= n_clients
    ):
        raise ValueError("n_contaminated must lie in [0, n_clients]")
    if (
        isinstance(contamination_rate, bool)
        or not isinstance(contamination_rate, (int, float, np.integer, np.floating))
        or not np.isfinite(contamination_rate)
        or not 0.0 <= contamination_rate <= 1.0
    ):
        raise ValueError("contamination_rate must lie in [0, 1]")
    if (
        isinstance(test_fraction, bool)
        or not isinstance(test_fraction, (int, float, np.integer, np.floating))
        or not np.isfinite(test_fraction)
        or not 0.0 < test_fraction < 1.0
    ):
        raise ValueError("test_fraction must lie in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    robust_config = AggregationConfig(method="robust", robustness=robustness)
    variational_config = AggregationConfig(
        method="variational",
        robustness=robustness,
        entropy_weight=entropy_weight,
    )

    rng = np.random.default_rng(seed)
    n = x.shape[0]
    n_classes = int(y.max()) + 1
    perm = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    if n_test < 1 or n - n_test < n_clients:
        raise ValueError("split leaves too few train/test rows")
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_te, y_te = x[test_idx], y[test_idx]
    train_mean = x_tr.mean(axis=0)
    train_scale = x_tr.std(axis=0)
    train_scale = np.where(train_scale > 1e-12, train_scale, 1.0)
    x_tr = (x_tr - train_mean) / train_scale
    x_te = (x_te - train_mean) / train_scale

    shards = _shard_indices(x_tr.shape[0], n_clients, rng)
    contaminated = set(range(n_contaminated))
    models = []
    for cid, idx in enumerate(shards):
        xs, ys = x_tr[idx], y_tr[idx].copy()
        if cid in contaminated:
            flip = rng.random(ys.shape[0]) < contamination_rate
            ys[flip] = rng.integers(0, n_classes, size=int(flip.sum()))
        models.append(_gaussian_nb_fit(xs, ys, n_classes))

    def _metrics(config: AggregationConfig) -> _BenchmarkMetrics:
        probabilities: list[ArrayF] = []
        fallback_predictions = 0
        nonconverged_predictions = 0
        max_iterations = 0
        for j in range(x_te.shape[0]):
            local_posteriors = np.vstack(
                [_gaussian_nb_pmf(m, x_te[j]) for m in models]
            )
            result = aggregate_result(local_posteriors, config=config)
            probabilities.append(result.consensus)
            fallback_predictions += int(bool(result.fallback_events))
            nonconverged_predictions += int(not result.converged)
            max_iterations = max(max_iterations, result.iterations)
        matrix = np.vstack(probabilities)
        row_ids = np.arange(y_te.shape[0])
        return {
            "accuracy": float(np.mean(matrix.argmax(axis=1) == y_te)),
            "log_score": float(np.mean(np.log(np.clip(matrix[row_ids, y_te], 1e-12, None)))),
            "ece": _expected_calibration_error(matrix, y_te),
            "fallback_predictions": fallback_predictions,
            "nonconverged_predictions": nonconverged_predictions,
            "max_iterations": max_iterations,
        }

    naive = _metrics(AggregationConfig(method="naive"))
    robust = _metrics(robust_config)
    variational = _metrics(variational_config)

    # Recovery identity: at robustness=0 the consensus is the naive pool exactly.
    sample = np.vstack([_gaussian_nb_pmf(m, x_te[0]) for m in models])
    recovery_ok = bool(
        np.array_equal(robust_aggregate(sample, robustness=0.0).consensus, log_linear_pool(sample))
    )

    return {
        "naive_accuracy": naive["accuracy"],
        "robust_accuracy": robust["accuracy"],
        "variational_accuracy": variational["accuracy"],
        "robust_minus_naive": robust["accuracy"] - naive["accuracy"],
        "variational_minus_naive": variational["accuracy"] - naive["accuracy"],
        "naive_log_score": naive["log_score"],
        "robust_log_score": robust["log_score"],
        "variational_log_score": variational["log_score"],
        "robust_minus_naive_log_score": robust["log_score"] - naive["log_score"],
        "variational_minus_naive_log_score": (variational["log_score"] - naive["log_score"]),
        "naive_ece": naive["ece"],
        "robust_ece": robust["ece"],
        "variational_ece": variational["ece"],
        "naive_fallback_predictions": naive["fallback_predictions"],
        "robust_fallback_predictions": robust["fallback_predictions"],
        "variational_fallback_predictions": variational["fallback_predictions"],
        "naive_nonconverged_predictions": naive["nonconverged_predictions"],
        "robust_nonconverged_predictions": robust["nonconverged_predictions"],
        "variational_nonconverged_predictions": variational["nonconverged_predictions"],
        "naive_max_iterations": naive["max_iterations"],
        "robust_max_iterations": robust["max_iterations"],
        "variational_max_iterations": variational["max_iterations"],
        "n_clients": int(n_clients),
        "n_contaminated": int(n_contaminated),
        "n_classes": int(n_classes),
        "n_test": int(n_test),
        "robustness": float(robustness),
        "entropy_weight": float(entropy_weight),
        "contamination_rate": float(contamination_rate),
        "train_class_counts": np.bincount(y_tr, minlength=n_classes).astype(int).tolist(),
        "test_class_counts": np.bincount(y_te, minlength=n_classes).astype(int).tolist(),
        "contaminated_client_ids": sorted(contaminated),
        "recovery_identity_holds": recovery_ok,
        "seed": int(seed),
        "split_sha256": canonical_sha256(
            {
                "seed": int(seed),
                "test_fraction": float(test_fraction),
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "input_sha256": hashlib.sha256(x.tobytes(order="C") + y.tobytes(order="C")).hexdigest(),
            }
        ),
        "preprocessing": "training-split z-score; zero scales replaced by one",
    }


def run_tabular_benchmark(path: str | Path | None = None, *, seed: int = 0, **kwargs: Any) -> dict[str, Any]:
    """Load the bundled synthetic tabular CSV (or a user CSV) and stress it."""
    if path is None:
        resource = resources.files("fedference").joinpath("data", "synthetic_tabular.csv")
        with resources.as_file(resource) as bundled_path:
            features, labels = load_tabular_csv(bundled_path)
        dataset_name = resource.name
    else:
        features, labels = load_tabular_csv(path)
        dataset_name = Path(path).name
    report = run_benchmark(features, labels, seed=seed, **kwargs)
    report["dataset"] = dataset_name
    report["n_rows"] = int(features.shape[0])
    report["n_features"] = int(features.shape[1])
    return report


def run_external_dataset_benchmark(
    dataset_id: str,
    *,
    cache_dir: str | Path,
    seed: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fetch, verify, and run one registered UCI benchmark."""
    dataset = fetch_external_dataset(dataset_id, cache_dir=cache_dir)
    report = run_benchmark(dataset.features, dataset.labels, seed=seed, **kwargs)
    report.update(
        {
            "dataset_id": dataset.spec.dataset_id,
            "dataset": dataset.spec.name,
            "dataset_doi": dataset.spec.doi,
            "dataset_license": dataset.spec.license,
            "dataset_source_url": dataset.spec.source_url,
            "dataset_archive_sha256": dataset.archive_sha256,
            "dataset_member_sha256": dataset.member_sha256,
            "n_rows": int(dataset.features.shape[0]),
            "n_features": int(dataset.features.shape[1]),
            "label_mapping": dict(dataset.label_mapping),
            "declared_preprocessing": list(dataset.spec.preprocessing),
            "independent_unit": "dataset; seeds are nested",
        }
    )
    return report


def run_external_benchmark_pack(
    *,
    cache_dir: str | Path,
    seeds: tuple[int, ...] = (0,),
    dataset_ids: tuple[str, ...] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the preregistered three-dataset pack without pooling nested seeds."""
    selected = tuple(spec.dataset_id for spec in DATASET_SPECS) if dataset_ids is None else dataset_ids
    if not selected:
        raise ValueError("dataset_ids must be non-empty")
    if any(not isinstance(dataset_id, str) or not dataset_id for dataset_id in selected):
        raise ValueError("dataset_ids must contain non-empty strings")
    if len(set(selected)) != len(selected):
        raise ValueError("dataset_ids must be unique")
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    rows = [
        run_external_dataset_benchmark(
            dataset_id,
            cache_dir=cache_dir,
            seed=seed,
            **kwargs,
        )
        for dataset_id in selected
        for seed in seeds
    ]
    summaries = summarize_external_benchmark_rows(rows)
    return {
        "status": "ok",
        "datasets": list(selected),
        "seeds": [int(seed) for seed in seeds],
        "independent_unit": "dataset; seed-level results remain nested",
        "primary_estimand": "contamination-conditioned held-out log-score difference",
        "dataset_level_inference": summaries,
        "negative_controls": {
            "all_recovery_identity_holds": all(bool(row["recovery_identity_holds"]) for row in rows),
            "all_split_receipts_present": all(
                isinstance(row.get("split_sha256"), str) and len(row["split_sha256"]) == 64 for row in rows
            ),
            "all_archive_receipts_present": all(
                isinstance(row.get("dataset_archive_sha256"), str)
                and len(row["dataset_archive_sha256"]) == 64
                for row in rows
            ),
        },
        "no_claim": "external datasets are conditional stress tests, not deployment or universality evidence",
        "rows": rows,
    }


def summarize_external_benchmark_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, object]]:
    """Summarize seed-level rows without treating seeds as datasets."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list")
    if any(not isinstance(row, dict) or not isinstance(row.get("dataset_id"), str) for row in rows):
        raise ValueError("rows must contain dataset_id mappings")
    summaries: dict[str, dict[str, object]] = {}
    for dataset_id in dict.fromkeys(str(row["dataset_id"]) for row in rows):
        selected = [row for row in rows if row["dataset_id"] == dataset_id]
        robust_effects = np.asarray([float(row["robust_minus_naive_log_score"]) for row in selected])
        variational_effects = np.asarray(
            [float(row["variational_minus_naive_log_score"]) for row in selected]
        )
        summaries[dataset_id] = {
            "n_seed_runs": int(len(selected)),
            "robust_minus_naive_log_score_mean": float(np.mean(robust_effects)),
            "robust_minus_naive_log_score_mcse": float(
                np.std(robust_effects, ddof=1) / np.sqrt(robust_effects.size)
                if robust_effects.size > 1
                else 0.0
            ),
            "variational_minus_naive_log_score_mean": float(np.mean(variational_effects)),
            "variational_minus_naive_log_score_mcse": float(
                np.std(variational_effects, ddof=1) / np.sqrt(variational_effects.size)
                if variational_effects.size > 1
                else 0.0
            ),
            "independent_unit": "dataset; seeds remain nested",
        }
    return summaries


__all__ = [
    "load_tabular_csv",
    "run_benchmark",
    "run_external_benchmark_pack",
    "run_external_dataset_benchmark",
    "run_tabular_benchmark",
    "summarize_external_benchmark_rows",
]
