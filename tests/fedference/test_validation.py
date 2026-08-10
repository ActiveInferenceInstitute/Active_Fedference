"""Shared finite-simplex validation tests (ISC-156)."""

from __future__ import annotations

import numpy as np
import pytest

from fedference._validation import as_nonnegative_weights, as_pmf, as_pmf_matrix


def test_as_pmf_rejects_nonfinite_and_negative_mass():
    with pytest.raises(ValueError, match="finite"):
        as_pmf([0.5, np.nan])
    with pytest.raises(ValueError, match="finite"):
        as_pmf([0.5, np.inf])
    with pytest.raises(ValueError, match="non-negative"):
        as_pmf([0.6, -0.1])
    with pytest.raises(ValueError, match="positive finite sum"):
        as_pmf([0.0, 0.0])


def test_as_pmf_matrix_rejects_empty_and_ragged_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        as_pmf_matrix([])
    with pytest.raises(ValueError, match="equal-length"):
        as_pmf_matrix([[0.5, 0.5], [1.0]])
    with pytest.raises(ValueError, match="vector rows"):
        as_pmf_matrix([0.5, 0.5])


def test_as_nonnegative_weights_rejects_nonfinite_and_zero_sum():
    with pytest.raises(ValueError, match="finite"):
        as_nonnegative_weights([1.0, np.nan], 2)
    with pytest.raises(ValueError, match="positive"):
        as_nonnegative_weights([0.0, 0.0], 2)


def test_valid_simplex_inputs_are_normalized_without_repairing_invalid_mass():
    assert np.allclose(as_pmf([2.0, 1.0]), [2.0 / 3.0, 1.0 / 3.0])
    assert np.allclose(as_pmf_matrix([[2.0, 1.0], [1.0, 3.0]]).sum(axis=1), 1.0)
    assert np.allclose(as_nonnegative_weights(None, 3), 1.0)
