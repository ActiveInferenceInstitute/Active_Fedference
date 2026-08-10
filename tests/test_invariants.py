"""Tests for the Active Fedference numerical invariants.

No mocks: every check runs the real seeded FedGVI / active-inference core and
pins explicit numeric properties (pmf mass, the robustness-zero recovery gap,
the EFE identity residual, and the KL learning curve).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from invariants import (
    InvariantResult,
    all_invariants,
    check_efe_identity,
    check_kl_monotonicity,
    check_pmf_normalization,
    check_robust_recovers_naive,
    write_invariants_report,
)


def test_invariant_result_defaults() -> None:
    r = InvariantResult(name="x", kind="equal", actual=1.0)
    assert r.passed is True
    assert r.expected is None
    assert r.extra == {}


def test_pmf_normalisation_passes_and_sums_to_one() -> None:
    results = check_pmf_normalization(seed=1)
    assert len(results) == 4
    assert all(r.passed for r in results)
    sum_checks = [r for r in results if r.kind == "equal"]
    for r in sum_checks:
        assert abs(r.actual - 1.0) <= 1e-9


def test_robust_recovers_naive_is_bit_identical() -> None:
    (result,) = check_robust_recovers_naive(seed=2)
    assert result.passed
    # robustness=0 must reproduce the naive log-linear pool to machine precision.
    assert result.actual <= 1e-12


def test_efe_identity_residual_is_zero() -> None:
    (result,) = check_efe_identity(seed=3)
    assert result.passed
    assert result.actual <= 1e-9


def test_kl_monotonicity_declines() -> None:
    (result,) = check_kl_monotonicity(seed=4)
    assert result.passed
    traj = result.actual
    assert traj[0] > traj[-1]
    # Monotone non-increasing within tolerance.
    assert all(traj[i] >= traj[i + 1] - 1e-9 for i in range(len(traj) - 1))


def test_all_invariants_aggregates_and_passes() -> None:
    results = all_invariants(seed=0)
    assert isinstance(results, list)
    # 4 pmf + 1 recovery + 1 efe + 1 kl = 7 witness records.
    assert len(results) == 7
    assert all(isinstance(r, InvariantResult) for r in results)
    assert all(r.passed for r in results)


def test_pmf_normalisation_nonnegative_entries() -> None:
    results = check_pmf_normalization(seed=5, robustness=2.0)
    nonneg = [r for r in results if r.kind == "nonneg"]
    assert len(nonneg) == 2
    for r in nonneg:
        assert r.actual >= 0.0
        assert r.passed


def test_results_are_deterministic_across_calls() -> None:
    a = check_efe_identity(seed=7)[0].actual
    b = check_efe_identity(seed=7)[0].actual
    assert np.isclose(a, b)


def test_write_invariants_report_serialises_passing_witnesses(tmp_path: Path) -> None:
    out_path, all_passed = write_invariants_report(tmp_path, seed=0)
    assert all_passed is True
    assert out_path == tmp_path / "output" / "reports" / "invariants.json"
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["seed"] == 0
    assert payload["all_passed"] is True
    assert payload["n_invariants"] == 7
    assert len(payload["invariants"]) == 7
    names = {rec["name"] for rec in payload["invariants"]}
    assert "robust_equals_naive_at_robustness_zero" in names
    assert "efe_decomposition_identity" in names
    # The robustness-zero recovery gap is the locked FedGVI<->Friston identity:
    # robust_aggregate(robustness=0) == log_linear_pool, bit-identical.
    recovery = next(
        r for r in payload["invariants"]
        if r["name"] == "robust_equals_naive_at_robustness_zero"
    )
    assert recovery["actual"] == 0.0
    assert recovery["passed"] is True
