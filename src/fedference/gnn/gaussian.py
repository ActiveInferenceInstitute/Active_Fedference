"""Full Gaussian conditioning, separate from native tracking role mapping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .admission import GNNAdmissionError, dumps
from .scientific_admission import _array
from .types import HybridGaussianModel


@dataclass(frozen=True, slots=True)
class GaussianPosterior:
    """Multivariate conditioning result in the model coordinate units."""

    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]



def gaussian_condition(
    model: HybridGaussianModel,
    observation: tuple[float, ...],
    observation_covariance: tuple[tuple[float, ...], ...],
) -> GaussianPosterior:
    """Condition a full Gaussian on a same-coordinate noisy observation.

    This is a conjugate arithmetic operation, separate from the limited native
    tracking configuration adapter. Every covariance entry participates.
    """
    dumps(model)
    n = len(model.coordinates)
    y = _array(list(observation), (n,), "observation")
    r = _array([list(row) for row in observation_covariance], (n, n), "observation_covariance")
    if not np.array_equal(r, r.T):
        raise GNNAdmissionError("GNN_SYMMETRY", "observation covariance")
    try:
        np.linalg.cholesky(r)
    except np.linalg.LinAlgError as error:
        raise GNNAdmissionError("GNN_COVARIANCE", "observation covariance") from error
    mean = np.array(model.mean.values)
    covariance = np.array(model.covariance.values).reshape(n, n)
    try:
        with np.errstate(over="raise", invalid="raise"):
            scale = max(float(np.max(np.abs(covariance))), float(np.max(np.abs(r))))
            scaled_prior, scaled_noise = covariance / scale, r / scale
            innovation = scaled_prior + scaled_noise
            gain = np.linalg.solve(innovation, scaled_prior).T
            prior_gain = np.linalg.solve(innovation, scaled_noise).T
            updated = prior_gain @ mean + gain @ y
            posterior = prior_gain @ covariance @ prior_gain.T + gain @ r @ gain.T
            posterior = posterior * 0.5 + posterior.T * 0.5
        if not np.isfinite(updated).all() or not np.isfinite(posterior).all():
            raise GNNAdmissionError("GNN_COVARIANCE", "unrepresentable posterior")
        np.linalg.cholesky(posterior)
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        raise GNNAdmissionError("GNN_COVARIANCE", "posterior arithmetic") from error
    return GaussianPosterior(
        tuple(float(value) for value in updated),
        tuple(tuple(float(value) for value in row) for row in posterior),
    )
