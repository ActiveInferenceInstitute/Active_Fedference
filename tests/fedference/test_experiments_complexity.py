"""Tests for the real seeded complexity-scaling experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from fedference.complexity import ComplexityBenchmarkConfig
from fedference.experiments.complexity import run_complexity_scaling


def _small_config() -> ComplexityBenchmarkConfig:
    return ComplexityBenchmarkConfig(
        agent_sizes=(2, 4, 8),
        state_sizes=(4, 8, 16),
        sharing_agent_sizes=(2, 4, 8),
        modality_sizes=(1, 2, 4),
        fixed_agent_count=4,
        fixed_state_count=8,
        inference_state_count=8,
        observation_count=3,
        repeats=1,
        warmups=0,
        max_iter=2,
        variational_starts=3,
        seed=17,
    )


def test_complexity_report_has_analytic_and_measured_layers() -> None:
    report = run_complexity_scaling(_small_config())
    assert report["status"] == "ok"
    assert "machine-specific diagnostics" in report["claim_boundary"]
    assert len(report["analytic_specs"]) == 8
    measurements = report["measurements"]
    assert len(measurements) == 9
    keys = {(row["method"], row["axis"]) for row in measurements}
    assert ("log_linear_pool", "agents") in keys
    assert ("variational_aggregate", "states") in keys
    assert ("share_round_naive", "agents") in keys
    assert ("share_round_robust", "agents") in keys
    assert ("infer_states", "modalities") in keys


def test_complexity_report_is_seeded_and_records_repeat_samples() -> None:
    config = _small_config()
    first = run_complexity_scaling(config)
    second = run_complexity_scaling(config)
    first_rows = {(row["method"], row["axis"]): row for row in first["measurements"]}
    second_rows = {(row["method"], row["axis"]): row for row in second["measurements"]}
    for key, row in first_rows.items():
        other = second_rows[key]
        assert row["input_digests"] == other["input_digests"]
        assert row["sizes"] == other["sizes"]
        assert all(np.isfinite(value) and value > 0.0 for value in row["median_seconds"])
        assert len(row["samples_seconds"]) == len(row["sizes"])
        assert all(len(samples) == 1 for samples in row["samples_seconds"])
        assert np.isfinite(float(row["observed_log_log_slope"]))


def test_complexity_report_records_machine_timer_and_work_units() -> None:
    report = run_complexity_scaling(_small_config())
    machine = report["machine"]
    assert machine["timer"] == "time.perf_counter"
    assert isinstance(machine["python"], str)
    assert isinstance(machine["numpy"], str)
    assert machine["sys_executable"] == Path(sys.executable).name
    assert "/" not in str(machine["sys_executable"])
    assert "\\" not in str(machine["sys_executable"])
    for row in report["measurements"]:
        assert all(int(value) > 0 for value in row["work_units"])
        assert row["fit_method"].startswith("ordinary least squares")
    robust_sharing = next(
        row
        for row in report["measurements"]
        if row["method"] == "share_round_robust" and row["axis"] == "agents"
    )
    assert robust_sharing["parameters"]["I"] == 32
