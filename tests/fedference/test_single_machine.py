"""Single-machine research-lane pilots and their evidence boundaries."""

from __future__ import annotations

import pytest

from fedference.single_machine import (
    _synthetic_bnn_data,
    run_calibration_pilot,
    run_fedgvi_bnn_pilot,
)


def test_calibration_pilot_is_disjoint_and_retains_complete_candidate_scores() -> None:
    report = run_calibration_pilot(seed=0, profile="smoke")
    assert report["status"] == "pilot"
    assert len(report["candidate_scores"]) == 3
    assert report["overlap_negative_control"]["status"] == "rejected"
    assert report["locked_evaluation"]["config_fingerprint"] == report["selected_config_fingerprint"]


def test_single_machine_pilots_reject_invalid_profiles_seeds_and_data_controls() -> None:
    with pytest.raises(ValueError, match="accept only"):
        run_calibration_pilot(profile="confirmatory")
    with pytest.raises(ValueError, match="non-negative"):
        run_calibration_pilot(seed=-1)
    with pytest.raises(ValueError, match="invalid synthetic"):
        _synthetic_bnn_data(0, n_clients=0, n_per_class=1, contamination=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        run_fedgvi_bnn_pilot(seed=-1)


@pytest.mark.requires_torch
def test_portable_bnn_pilot_binds_cavity_checkpoint_and_pvi_controls() -> None:
    pytest.importorskip("torch")
    report = run_fedgvi_bnn_pilot(seed=0, profile="smoke", requested_device="cpu")
    assert report["status"] == "pilot"
    assert report["negative_controls"]["checkpoint_resume"] is True
    assert all(row["checkpoint_resume_equivalent"] for row in report["rows"])
    assert {row["contamination"] for row in report["rows"]} == {0.0, 0.6}
