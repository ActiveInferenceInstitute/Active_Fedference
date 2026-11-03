"""Exact seeded parameter-recovery parity and per-call model construction."""

from __future__ import annotations

import hashlib
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from fedference.experiments.parameter_recovery import run_parameter_recovery

REFERENCE = Path(__file__).parents[1] / "fixtures/performance/parameter-recovery-7d343a2.py.txt"
REFERENCE_SHA256 = "f37d0635d72d43f5505f17f1e9fc7d08dc5a6299f097a8b13fe5168d050818f5"


def _reference(path: Path = REFERENCE) -> Callable[..., dict[str, Any]]:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256
    namespace = runpy.run_path(str(path))
    function = namespace["run_parameter_recovery"]
    assert callable(function)
    return function


@pytest.mark.parametrize(
    "seed,trials,observations,resolution", [(0, 3, 9, 7), (11, 4, 11, 9), (23, 5, 20, 20)]
)
def test_parameter_recovery_preserves_original_complete_report(
    seed: int, trials: int, observations: int, resolution: int,
) -> None:
    kwargs = {"seed": seed, "n_trials": trials, "n_observations": observations,
              "fit_resolution": resolution}
    assert run_parameter_recovery(**kwargs) == _reference()(**kwargs)


def test_parameter_recovery_reference_rejects_changed_bytes(tmp_path: Path) -> None:
    changed = tmp_path / "changed.py.txt"
    changed.write_bytes(REFERENCE.read_bytes() + b"\n# changed\n")
    with pytest.raises(AssertionError):
        _reference(changed)


@pytest.mark.parametrize("trials,resolution", [(2, 3), (4, 5)])
def test_parameter_recovery_builds_each_fit_model_once_per_call(trials: int, resolution: int) -> None:
    from fedference.gnn.loader import load_sentinel

    calls = 0

    def observe(frame: FrameType, event: str, value: object) -> None:
        nonlocal calls
        if event == "call" and frame.f_code is load_sentinel.__code__:
            calls += 1

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        for invocation in range(2):
            report = run_parameter_recovery(
                seed=19, acuity_grid=(0.6, 0.9), n_trials=trials,
                n_observations=5, fit_resolution=resolution,
            )
            assert report["n_independent_trials"] == 2 * trials
            assert calls == (invocation + 1) * (2 + resolution)
    finally:
        sys.setprofile(previous)
