"""FedGVI/PVI diagonal-Gaussian site-factor server protocol.

This module implements the protocol shape used by the FedGVI source lane:
global posterior natural parameters are the prior plus client site factors; a
client trains against its cavity (global posterior minus its old site); and the
server replaces that site with ``client_posterior - cavity``. It deliberately
does not call this operation moment matching.

The primitive is model-agnostic and NumPy-only. A PyTorch BNN client may flatten
its mean-field parameters into :class:`DiagonalGaussian`, perform its local GVI
update, and return the updated posterior. Scientific parity still depends on the
source loss, divergence, optimization schedule, and data partition.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .evidence import canonical_sha256

ArrayF = np.ndarray
UpdateSchedule = Literal["sequential", "parallel"]


def _vector(value: ArrayF, *, name: str) -> ArrayF:
    result = np.array(value, dtype=np.float64, copy=True).ravel()
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DiagonalGaussian:
    """Normalized diagonal Gaussian posterior parameters."""

    mean: ArrayF
    variance: ArrayF

    def __post_init__(self) -> None:
        mean = _vector(self.mean, name="mean")
        variance = _vector(self.variance, name="variance")
        if mean.shape != variance.shape:
            raise ValueError("mean and variance must have the same shape")
        if np.any(variance <= 0.0):
            raise ValueError("variance must be strictly positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)

    @property
    def precision(self) -> ArrayF:
        """Natural precision parameter."""
        return 1.0 / self.variance

    @property
    def precision_mean(self) -> ArrayF:
        """Natural precision-times-mean parameter."""
        return self.mean / self.variance

    @classmethod
    def from_natural(
        cls,
        precision_mean: ArrayF,
        precision: ArrayF,
    ) -> DiagonalGaussian:
        """Construct a normalized posterior from positive natural precision."""
        eta = _vector(precision_mean, name="precision_mean")
        tau = _vector(precision, name="precision")
        if eta.shape != tau.shape:
            raise ValueError("natural parameter vectors must have the same shape")
        if np.any(tau <= 0.0):
            raise ValueError("posterior precision must be strictly positive")
        return cls(mean=eta / tau, variance=1.0 / tau)


@dataclass(frozen=True)
class GaussianSiteFactor:
    """Possibly unnormalized client factor in Gaussian natural coordinates."""

    precision_mean: ArrayF
    precision: ArrayF

    def __post_init__(self) -> None:
        eta = _vector(self.precision_mean, name="site precision_mean")
        tau = _vector(self.precision, name="site precision")
        if eta.shape != tau.shape:
            raise ValueError("site natural parameter vectors must have the same shape")
        object.__setattr__(self, "precision_mean", eta)
        object.__setattr__(self, "precision", tau)

    @classmethod
    def zeros(cls, dimension: int) -> GaussianSiteFactor:
        """Return the neutral factor for one client."""
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        return cls(np.zeros(dimension), np.zeros(dimension))


@dataclass(frozen=True)
class FedGVIServerState:
    """Prior, client sites, and round counter for resumable FedGVI updates."""

    prior: DiagonalGaussian
    sites: tuple[GaussianSiteFactor, ...]
    round_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.prior, DiagonalGaussian):
            raise ValueError("prior must be a DiagonalGaussian")
        if not isinstance(self.sites, (tuple, list)):
            raise ValueError("sites must be a sequence of GaussianSiteFactor values")
        object.__setattr__(self, "sites", tuple(self.sites))
        if not self.sites:
            raise ValueError("sites must contain at least one client factor")
        if any(not isinstance(site, GaussianSiteFactor) for site in self.sites):
            raise ValueError("sites must contain GaussianSiteFactor values")
        dimension = self.prior.mean.size
        if any(site.precision.size != dimension for site in self.sites):
            raise ValueError("every site must match the prior dimension")
        if (
            isinstance(self.round_index, bool)
            or not isinstance(self.round_index, int)
            or self.round_index < 0
        ):
            raise ValueError("round_index must be a non-negative integer")
        # Validate that the current aggregate is a normalized Gaussian.
        self.posterior()
        for client_id in range(len(self.sites)):
            try:
                self.cavity(client_id)
            except ValueError as exc:
                raise ValueError(f"site {client_id} leaves a non-normalizable cavity") from exc

    @classmethod
    def initialize(
        cls,
        prior: DiagonalGaussian,
        *,
        n_clients: int,
    ) -> FedGVIServerState:
        """Initialize neutral client sites at the declared prior."""
        if isinstance(n_clients, bool) or not isinstance(n_clients, int) or n_clients <= 0:
            raise ValueError("n_clients must be a positive integer")
        sites = tuple(GaussianSiteFactor.zeros(prior.mean.size) for _ in range(n_clients))
        return cls(prior=prior, sites=sites)

    def _natural(self) -> tuple[ArrayF, ArrayF]:
        precision_mean = self.prior.precision_mean.copy()
        precision = self.prior.precision.copy()
        for site in self.sites:
            precision_mean += site.precision_mean
            precision += site.precision
        return precision_mean, precision

    def posterior(self) -> DiagonalGaussian:
        """Return the normalized global posterior ``prior × product(sites)``."""
        precision_mean, precision = self._natural()
        return DiagonalGaussian.from_natural(precision_mean, precision)

    def cavity(self, client_id: int) -> DiagonalGaussian:
        """Return the global posterior with ``client_id``'s old site removed."""
        if (
            isinstance(client_id, bool)
            or not isinstance(client_id, (int, np.integer))
            or not 0 <= int(client_id) < len(self.sites)
        ):
            raise ValueError("client_id is outside the site table")
        client_id = int(client_id)
        precision_mean, precision = self._natural()
        site = self.sites[client_id]
        return DiagonalGaussian.from_natural(
            precision_mean - site.precision_mean,
            precision - site.precision,
        )

    def replace_site(
        self,
        client_id: int,
        client_posterior: DiagonalGaussian,
    ) -> FedGVIServerState:
        """Replace one site using ``new posterior - old cavity`` natural parameters."""
        if not isinstance(client_posterior, DiagonalGaussian):
            raise ValueError("client_posterior must be a DiagonalGaussian")
        if client_posterior.mean.shape != self.prior.mean.shape:
            raise ValueError("client_posterior must match the server parameter dimension")
        cavity = self.cavity(client_id)
        site = GaussianSiteFactor(
            precision_mean=(client_posterior.precision_mean - cavity.precision_mean),
            precision=client_posterior.precision - cavity.precision,
        )
        sites = list(self.sites)
        sites[client_id] = site
        return FedGVIServerState(
            prior=self.prior,
            sites=tuple(sites),
            round_index=self.round_index,
        )

    def advance_round(
        self,
        client_posteriors: Mapping[int, DiagonalGaussian],
        *,
        schedule: UpdateSchedule = "parallel",
    ) -> FedGVIServerState:
        """Apply a complete declared client update set and increment the round."""
        expected = set(range(len(self.sites)))
        if any(
            isinstance(client_id, bool) or not isinstance(client_id, (int, np.integer))
            for client_id in client_posteriors
        ):
            raise ValueError("client ids must be integers")
        if set(client_posteriors) != expected:
            raise ValueError("client_posteriors must contain exactly one update per client")
        if any(not isinstance(posterior, DiagonalGaussian) for posterior in client_posteriors.values()):
            raise ValueError("client updates must be DiagonalGaussian values")
        if any(
            posterior.mean.shape != self.prior.mean.shape
            for posterior in client_posteriors.values()
        ):
            raise ValueError("every client update must match the server parameter dimension")
        if schedule not in ("sequential", "parallel"):
            raise ValueError("schedule must be 'sequential' or 'parallel'")
        if schedule == "sequential":
            updated = self
            for client_id in sorted(client_posteriors):
                updated = updated.replace_site(client_id, client_posteriors[client_id])
        else:
            new_sites: list[GaussianSiteFactor] = []
            for client_id in range(len(self.sites)):
                cavity = self.cavity(client_id)
                posterior = client_posteriors[client_id]
                new_sites.append(
                    GaussianSiteFactor(
                        precision_mean=(posterior.precision_mean - cavity.precision_mean),
                        precision=posterior.precision - cavity.precision,
                    )
                )
            updated = FedGVIServerState(
                prior=self.prior,
                sites=tuple(new_sites),
                round_index=self.round_index,
            )
        return FedGVIServerState(
            prior=updated.prior,
            sites=updated.sites,
            round_index=self.round_index + 1,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe checkpoint data."""
        return {
            "schema_version": "1.0",
            "round_index": self.round_index,
            "prior": {
                "mean": self.prior.mean.tolist(),
                "variance": self.prior.variance.tolist(),
            },
            "sites": [
                {
                    "precision_mean": site.precision_mean.tolist(),
                    "precision": site.precision.tolist(),
                }
                for site in self.sites
            ],
        }

    @property
    def fingerprint(self) -> str:
        """Canonical state hash for checkpoint/resume receipts."""
        return canonical_sha256(self.as_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FedGVIServerState:
        """Load a strict schema-v1 server checkpoint."""
        if set(raw) != {"schema_version", "round_index", "prior", "sites"}:
            raise ValueError("FedGVI checkpoint fields do not match schema")
        if raw["schema_version"] != "1.0":
            raise ValueError("unsupported FedGVI checkpoint schema")
        prior_raw = raw["prior"]
        sites_raw = raw["sites"]
        if not isinstance(prior_raw, Mapping) or not isinstance(sites_raw, list):
            raise ValueError("FedGVI checkpoint prior/sites are malformed")
        if set(prior_raw) != {"mean", "variance"}:
            raise ValueError("FedGVI checkpoint prior fields do not match schema")
        if not all(isinstance(prior_raw[field], list) for field in prior_raw):
            raise ValueError("FedGVI checkpoint prior values must be arrays")
        if (
            isinstance(raw["round_index"], bool)
            or not isinstance(raw["round_index"], int)
            or raw["round_index"] < 0
        ):
            raise ValueError("FedGVI checkpoint round_index is malformed")
        for site in sites_raw:
            if not isinstance(site, Mapping) or set(site) != {
                "precision_mean",
                "precision",
            }:
                raise ValueError("FedGVI checkpoint site fields do not match schema")
            if not all(isinstance(site[field], list) for field in site):
                raise ValueError("FedGVI checkpoint site values must be arrays")
        try:
            prior = DiagonalGaussian(
                mean=np.asarray(prior_raw["mean"], dtype=np.float64),
                variance=np.asarray(prior_raw["variance"], dtype=np.float64),
            )
            sites = tuple(
                GaussianSiteFactor(
                    precision_mean=np.asarray(site["precision_mean"], dtype=np.float64),
                    precision=np.asarray(site["precision"], dtype=np.float64),
                )
                for site in sites_raw
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("FedGVI checkpoint natural parameters are malformed") from exc
        return cls(prior=prior, sites=sites, round_index=raw["round_index"])


def save_server_checkpoint(
    path: str | Path,
    state: FedGVIServerState,
) -> Path:
    """Atomically write one round-level server checkpoint."""
    if not isinstance(state, FedGVIServerState):
        raise ValueError("state must be a FedGVIServerState")
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=checkpoint.parent,
            prefix=f".{checkpoint.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(
                state.as_dict(),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, checkpoint)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return checkpoint


def load_server_checkpoint(path: str | Path) -> FedGVIServerState:
    """Load and validate a round-level server checkpoint."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid FedGVI checkpoint: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("FedGVI checkpoint must be a JSON object")
    return FedGVIServerState.from_dict(raw)


__all__ = [
    "DiagonalGaussian",
    "FedGVIServerState",
    "GaussianSiteFactor",
    "UpdateSchedule",
    "load_server_checkpoint",
    "save_server_checkpoint",
]
