"""Tests for the parameter-recovery figure generator (no mocks).

Real data is constructed and passed directly to
:func:`figures.parameter_recovery.generate_parameter_recovery`; no patching.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from figures.parameter_recovery import generate_parameter_recovery

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n: int = 4, seed: int = 0):
    """Return consistent (true, recovered, ci_lo, ci_hi, abs_error) for n points."""
    rng = np.random.default_rng(seed)
    true_acuity = list(np.linspace(0.60, 0.90, n))
    noise = rng.normal(0, 0.02, size=n)
    recovered = [float(np.clip(t + d, 0.51, 0.99)) for t, d in zip(true_acuity, noise)]
    half_ci = list(np.abs(rng.normal(0.04, 0.005, size=n)))
    ci_lo = [r - h for r, h in zip(recovered, half_ci)]
    ci_hi = [r + h for r, h in zip(recovered, half_ci)]
    abs_error = [abs(r - t) for r, t in zip(recovered, true_acuity)]
    return true_acuity, recovered, ci_lo, ci_hi, abs_error


# ---------------------------------------------------------------------------
# Happy path — four acuity levels
# ---------------------------------------------------------------------------

def test_generate_parameter_recovery_happy_path(tmp_path: Path) -> None:
    """call with 4 acuity levels: file exists and has valid PNG header."""
    ta, ra, ci_lo, ci_hi, ae = _make_data(n=4)
    out = generate_parameter_recovery(
        ta, ra, ci_lo, ci_hi, ae,
        r_squared=0.99,
        mean_abs_error=float(np.mean(ae)),
        n_trials=3,
        n_observations=10,
        project_root=tmp_path,
    )
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:8] == _PNG_MAGIC


# ---------------------------------------------------------------------------
# Custom filename
# ---------------------------------------------------------------------------

def test_generate_parameter_recovery_custom_filename(tmp_path: Path) -> None:
    """Output path uses the supplied filename."""
    ta, ra, ci_lo, ci_hi, ae = _make_data(n=4)
    custom = "my_recovery_plot.png"
    out = generate_parameter_recovery(
        ta, ra, ci_lo, ci_hi, ae,
        project_root=tmp_path,
        filename=custom,
    )
    assert out.name == custom
    assert out.exists()
    assert out.read_bytes()[:8] == _PNG_MAGIC


# ---------------------------------------------------------------------------
# Empty input raises ValueError
# ---------------------------------------------------------------------------

def test_generate_parameter_recovery_empty_raises(tmp_path: Path) -> None:
    """Empty true_acuity list raises ValueError."""
    with pytest.raises(ValueError):
        generate_parameter_recovery(
            [], [], [], [], [],
            project_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# Mismatched lengths raise ValueError
# ---------------------------------------------------------------------------

def test_generate_parameter_recovery_length_mismatch_raises(tmp_path: Path) -> None:
    """Arrays of inconsistent length raise ValueError."""
    ta, ra, ci_lo, ci_hi, ae = _make_data(n=4)
    # Make recovered one element shorter — mismatch with true_acuity
    with pytest.raises(ValueError):
        generate_parameter_recovery(
            ta, ra[:-1], ci_lo, ci_hi, ae,
            project_root=tmp_path,
        )
