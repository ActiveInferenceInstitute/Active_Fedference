"""Signed-site native cavity/replacement arithmetic (Mildner et al., 2025).

This adapter does not implement the paper's client optimization protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bnn_fedgvi import DiagonalGaussian, FedGVIServerState, GaussianSiteFactor
from ..protocol_parity import ParityRow, ProtocolParityMatrix
from .admission import GNNAdmissionError
from .scientific_admission import ScientificProgram, _array, _native
from .types import SiteFactorModel


@dataclass(frozen=True, slots=True)
class SiteRun:
    """Real cavity/replacement states and bounded protocol parity."""

    initial: FedGVIServerState
    cavity: DiagonalGaussian
    updated: FedGVIServerState
    parity: ProtocolParityMatrix
    model_digest: str



def run_sites(program: ScientificProgram) -> SiteRun:
    """Run signed diagonal prior × sites, proper cavity, and factor replacement."""
    native = _native(program)
    model = program.model
    if not isinstance(model, SiteFactorModel):
        raise GNNAdmissionError("GNN_KIND", "site runner")
    n = len(model.coordinates)
    sites = []
    for factor in (model, *program.peers):
        precision = np.array(factor.precision_increment.values).reshape(n, n)
        if not np.array_equal(precision, np.diag(np.diag(precision))):
            raise GNNAdmissionError("GNN_CAPABILITY", "non-diagonal site")
        sites.append(GaussianSiteFactor(np.array(factor.information_increment.values), np.diag(precision)))
    prior_mean = _array(native["prior_mean"], (n,), "prior mean")
    prior_covariance = _array(native["prior_covariance"], (n, n), "prior covariance")
    if not np.array_equal(prior_covariance, np.diag(np.diag(prior_covariance))):
        raise GNNAdmissionError("GNN_CAPABILITY", "non-diagonal prior")
    prior = DiagonalGaussian(prior_mean, np.diag(prior_covariance))
    client_means = _array(native["client_means"], (len(sites), n), "client means")
    client_covariances = _array(native["client_covariances"], (len(sites), n, n), "client covariances")
    clients = {}
    for index, covariance in enumerate(client_covariances):
        if not np.array_equal(covariance, np.diag(np.diag(covariance))):
            raise GNNAdmissionError("GNN_CAPABILITY", "non-diagonal client")
        clients[index] = DiagonalGaussian(client_means[index], np.diag(covariance))
    state = FedGVIServerState(prior, tuple(sites))
    cavity = state.cavity(0)
    updated = state.advance_round(clients, schedule="parallel")
    parity = ProtocolParityMatrix(
        "gnn-diagonal-site-smoke",
        "fedgvi-mildner-2025",
        "native signed-site arithmetic",
        (
            ParityRow(
                "server update",
                "cavity and site replacement",
                "cavity and site replacement",
                "matched",
                "fedference.bnn_fedgvi.FedGVIServerState",
            ),
            ParityRow(
                "client objective and optimization",
                "source loss and schedule",
                "declared Gaussian client posterior; no optimization",
                "unresolved",
                "GNN native client parameters; arithmetic smoke only",
            ),
        ),
    )
    return SiteRun(state, cavity, updated, parity, program.digest)
