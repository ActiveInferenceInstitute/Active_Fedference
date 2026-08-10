"""FedGVI site-factor, cavity, replacement, and checkpoint protocol."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from fedference.bnn_fedgvi import (
    DiagonalGaussian,
    FedGVIServerState,
    GaussianSiteFactor,
    load_server_checkpoint,
    save_server_checkpoint,
)


def _prior() -> DiagonalGaussian:
    return DiagonalGaussian(mean=np.zeros(2), variance=np.ones(2))


def test_neutral_sites_recover_prior_and_cavity() -> None:
    state = FedGVIServerState.initialize(_prior(), n_clients=2)
    assert np.array_equal(state.posterior().mean, _prior().mean)
    assert np.array_equal(state.posterior().variance, _prior().variance)
    for client_id in range(2):
        cavity = state.cavity(client_id)
        assert np.array_equal(cavity.mean, _prior().mean)
        assert np.array_equal(cavity.variance, _prior().variance)


def test_site_replacement_reconstructs_returned_client_posterior() -> None:
    state = FedGVIServerState.initialize(_prior(), n_clients=2)
    client = DiagonalGaussian(
        mean=np.asarray([0.5, -0.25]),
        variance=np.asarray([0.5, 0.25]),
    )
    updated = state.replace_site(0, client)
    assert np.allclose(updated.posterior().mean, client.mean)
    assert np.allclose(updated.posterior().variance, client.variance)
    assert np.allclose(updated.cavity(0).mean, state.posterior().mean)


def test_parallel_round_combines_sites_and_sequential_round_replaces_in_order() -> None:
    state = FedGVIServerState.initialize(_prior(), n_clients=2)
    clients = {
        0: DiagonalGaussian(mean=np.ones(2) * 0.5, variance=np.ones(2) * 0.5),
        1: DiagonalGaussian(mean=np.ones(2) * -0.5, variance=np.ones(2) * 0.5),
    }
    parallel = state.advance_round(clients, schedule="parallel")
    assert parallel.round_index == 1
    assert np.allclose(parallel.posterior().mean, np.zeros(2))
    assert np.allclose(parallel.posterior().variance, np.ones(2) / 3.0)

    sequential = state.advance_round(clients, schedule="sequential")
    assert sequential.round_index == 1
    assert np.allclose(sequential.posterior().mean, clients[1].mean)
    assert np.allclose(sequential.posterior().variance, clients[1].variance)


def test_checkpoint_resume_is_exact(tmp_path) -> None:
    state = FedGVIServerState.initialize(_prior(), n_clients=2)
    clients = {
        0: DiagonalGaussian(mean=np.ones(2) * 0.2, variance=np.ones(2) * 0.8),
        1: DiagonalGaussian(mean=np.ones(2) * -0.1, variance=np.ones(2) * 0.9),
    }
    advanced = state.advance_round(clients)
    path = save_server_checkpoint(tmp_path / "round-1.json", advanced)
    resumed = load_server_checkpoint(path)
    assert resumed.fingerprint == advanced.fingerprint
    assert np.array_equal(resumed.posterior().mean, advanced.posterior().mean)
    assert np.array_equal(resumed.posterior().variance, advanced.posterior().variance)


def test_checkpoint_writer_rejects_non_state_payload(tmp_path) -> None:
    invalid_state: Any = {"round_index": 1}
    with pytest.raises(ValueError, match="FedGVIServerState"):
        save_server_checkpoint(tmp_path / "invalid.json", invalid_state)


def test_protocol_rejects_incomplete_rounds_and_invalid_cavities() -> None:
    state = FedGVIServerState.initialize(_prior(), n_clients=2)
    posterior = DiagonalGaussian(mean=np.zeros(2), variance=np.ones(2))
    with pytest.raises(ValueError, match="exactly one update"):
        state.advance_round({0: posterior})
    with pytest.raises(ValueError, match="schedule"):
        state.advance_round({0: posterior, 1: posterior}, schedule="unknown")
    with pytest.raises(ValueError, match="client_id"):
        state.cavity(2)
    with pytest.raises(ValueError, match="client ids"):
        state.advance_round({False: posterior, 1: posterior})
    with pytest.raises(ValueError, match="DiagonalGaussian"):
        state.advance_round({0: posterior, 1: "not-a-posterior"})

    with pytest.raises(ValueError, match="non-normalizable cavity"):
        FedGVIServerState(
            prior=DiagonalGaussian(mean=np.zeros(1), variance=np.ones(1)),
            sites=(
                GaussianSiteFactor(
                    precision_mean=np.zeros(1),
                    precision=np.asarray([-2.0]),
                ),
                GaussianSiteFactor(
                    precision_mean=np.zeros(1),
                    precision=np.asarray([3.0]),
                ),
            ),
        )


def test_site_replacement_rejects_silent_numpy_dimension_broadcast() -> None:
    """A scalar-width client posterior must not broadcast across server parameters."""
    state = FedGVIServerState.initialize(
        DiagonalGaussian(mean=np.zeros(3), variance=np.ones(3)),
        n_clients=2,
    )
    wrong_dimension = DiagonalGaussian(mean=np.zeros(1), variance=np.ones(1))
    correct_dimension = DiagonalGaussian(mean=np.zeros(3), variance=np.ones(3))
    with pytest.raises(ValueError, match="server parameter dimension"):
        state.replace_site(0, wrong_dimension)
    with pytest.raises(ValueError, match="server parameter dimension"):
        state.advance_round(
            {0: wrong_dimension, 1: correct_dimension},
            schedule="parallel",
        )


def test_protocol_parameters_are_immutable_owned_arrays() -> None:
    source_mean = np.asarray([0.0, 1.0])
    posterior = DiagonalGaussian(source_mean, np.ones(2))
    source_mean[0] = 99.0
    assert posterior.mean[0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        posterior.mean[0] = 1.0
    with pytest.raises(ValueError, match="GaussianSiteFactor"):
        FedGVIServerState(prior=_prior(), sites=("not-a-site",))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("round_index", True),
        ("round_index", "1"),
        ("prior", {"mean": [0.0, 0.0]}),
        (
            "sites",
            [{"precision_mean": [0.0, 0.0], "precision": "not-an-array"}],
        ),
    ),
)
def test_checkpoint_decoder_rejects_malformed_nested_fields(
    tmp_path,
    field,
    value,
) -> None:
    raw = FedGVIServerState.initialize(_prior(), n_clients=1).as_dict()
    raw[field] = value
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        load_server_checkpoint(path)
