"""Minimal hybrid position/velocity tracking task and recovery controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fedference.hybrid_tracking import (
    HybridTrackingConfig,
    run_hybrid_tracking,
    run_hybrid_tracking_comparison,
)


def test_hybrid_tracking_is_seeded_finite_and_closed_loop() -> None:
    config = HybridTrackingConfig(horizon=8, n_agents=3)
    first = run_hybrid_tracking(4, method="robust", config=config)
    second = run_hybrid_tracking(4, method="robust", config=config)
    assert first == second
    for field in (
        "on_policy_mean_log_score",
        "position_rmse",
        "known_context_component_rmse",
        "mean_control_cost",
        "mean_predictive_risk_surrogate",
    ):
        assert np.isfinite(first[field])
    assert "on-policy" in first["primary_estimand"]
    assert "held-out prediction" in first["no_claim"]
    assert len(first["trajectory"]) == 8
    assert {row["context"] for row in first["trajectory"]} == {0, 1}
    for row in first["trajectory"]:
        assert row["predicted_position"] == pytest.approx(
            row["uncontrolled_predicted_position"] + row["acceleration"]
        )


def test_zero_robustness_recovers_naive_episode_exactly() -> None:
    config = HybridTrackingConfig(horizon=8, n_agents=3, robustness=0.0)
    naive = run_hybrid_tracking(9, method="naive", config=config)
    robust_zero = run_hybrid_tracking(9, method="robust", config=config)
    assert naive["trajectory"] == robust_zero["trajectory"]
    assert naive["on_policy_mean_log_score"] == robust_zero["on_policy_mean_log_score"]
    assert naive["position_rmse"] == robust_zero["position_rmse"]


def test_outlier_control_changes_only_declared_contamination_path() -> None:
    config = HybridTrackingConfig(horizon=8, n_agents=3)
    clean = run_hybrid_tracking(
        2,
        method="robust",
        contaminate_one_agent=False,
        config=config,
    )
    contaminated = run_hybrid_tracking(
        2,
        method="robust",
        contaminate_one_agent=True,
        config=config,
    )
    assert clean["contaminate_one_agent"] is False
    assert contaminated["contaminate_one_agent"] is True
    assert clean["trajectory"] != contaminated["trajectory"]


def test_tracking_config_rejects_singular_or_invalid_controls() -> None:
    with pytest.raises(ValueError, match="position_observation_var"):
        HybridTrackingConfig(position_observation_var=0.0)
    with pytest.raises(ValueError, match="context_accuracy"):
        HybridTrackingConfig(context_accuracy=0.5)
    with pytest.raises(ValueError, match="method"):
        run_hybrid_tracking(method="unknown")
    with pytest.raises(ValueError, match="n_agents"):
        HybridTrackingConfig(n_agents=True)
    with pytest.raises(ValueError, match="seed"):
        run_hybrid_tracking(seed=True)
    with pytest.raises(ValueError, match="contaminate_one_agent"):
        run_hybrid_tracking(contaminate_one_agent=1)
    invalid: Any = {"horizon": 8}
    with pytest.raises(ValueError, match="HybridTrackingConfig"):
        run_hybrid_tracking(config=invalid)


def test_hybrid_comparison_includes_locked_controls_and_singular_negative() -> None:
    report = run_hybrid_tracking_comparison(3, config=HybridTrackingConfig(horizon=6, n_agents=3))
    assert report["method_order"] == [
        "naive",
        "robust",
        "discrete-only",
        "continuous-only",
        "oracle-context",
    ]
    assert set(report["methods"]) == set(report["method_order"])
    assert report["singular_covariance_control"]["status"] == "rejected"
    assert "held-out posterior-predictive" in report["primary_estimand"]
