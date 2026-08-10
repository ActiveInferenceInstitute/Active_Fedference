"""Versioned transport envelope and configured federation paths."""

from __future__ import annotations

import json
import queue
import struct
from io import BytesIO

import numpy as np
import pytest

from fedference.aggregation import (
    AggregationConfig,
    aggregate_result,
)
from fedference.federation import FederationServer, FederationWorker, run_socket_round
from fedference.federation.transport import (
    PROTOCOL_VERSION,
    deserialize_belief,
    deserialize_envelope,
    deserialize_result,
    serialize_envelope,
)


def test_protocol_envelope_round_trip_and_digest_tamper_detection() -> None:
    config = AggregationConfig(method="variational", robustness=1.5)
    wire = serialize_envelope(
        b"payload",
        message_type="belief",
        round_id="round-7",
        worker_id=2,
        aggregation_config_hash=config.fingerprint,
        authentication="hmac-sha256",
    )
    envelope, payload = deserialize_envelope(wire)
    assert envelope.protocol_version == PROTOCOL_VERSION
    assert envelope.round_id == "round-7"
    assert envelope.worker_id == 2
    assert envelope.aggregation_config_hash == config.fingerprint
    assert payload == b"payload"
    tampered = wire[:-1] + bytes([wire[-1] ^ 1])
    with pytest.raises(ValueError, match="digest mismatch"):
        deserialize_envelope(tampered)


def test_protocol_envelope_rejects_unknown_version() -> None:
    config = AggregationConfig()
    wire = serialize_envelope(
        b"x",
        message_type="belief",
        round_id="round",
        worker_id=0,
        aggregation_config_hash=config.fingerprint,
    )
    header_length = struct.unpack(">I", wire[:4])[0]
    header = json.loads(wire[4 : 4 + header_length])
    header["protocol_version"] = PROTOCOL_VERSION + 1
    changed = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    altered = struct.pack(">I", len(changed)) + changed + wire[4 + header_length :]
    with pytest.raises(ValueError, match="unsupported protocol_version"):
        deserialize_envelope(altered)
    with pytest.raises(ValueError, match="worker_id"):
        serialize_envelope(
            b"x",
            message_type="belief",
            round_id="round",
            worker_id=1.5,
            aggregation_config_hash=config.fingerprint,
        )


def test_result_transport_rejects_malformed_or_non_probability_archives() -> None:
    malformed = BytesIO()
    np.savez(malformed, consensus=np.asarray([0.5, 0.5]), unexpected=np.asarray([1.0]))
    with pytest.raises(ValueError, match="fields do not match"):
        deserialize_result(malformed.getvalue())

    invalid_weights = BytesIO()
    np.savez(
        invalid_weights,
        consensus=np.asarray([0.5, 0.5]),
        agent_weights=np.asarray([1.0, 1.0]),
    )
    with pytest.raises(ValueError, match="agent_weights must sum to one"):
        deserialize_result(invalid_weights.getvalue())

    wrong_dtype = BytesIO()
    np.savez(
        wrong_dtype,
        consensus=np.asarray([0.5, 0.5], dtype=np.float32),
        agent_weights=np.asarray([0.5, 0.5]),
    )
    with pytest.raises(ValueError, match="float64"):
        deserialize_result(wrong_dtype.getvalue())


def test_belief_transport_rejects_non_vector_and_non_probability_payloads() -> None:
    for value, message in (
        (np.asarray([[0.5, 0.5]]), "non-empty vector"),
        (np.asarray([0.5, 0.5], dtype=np.float32), "float64"),
        (np.asarray([0.5, -0.5]), "finite and non-negative"),
        (np.asarray([0.5, 0.6]), "sum to one"),
    ):
        payload = BytesIO()
        np.save(payload, value)
        with pytest.raises(ValueError, match=message):
            deserialize_belief(payload.getvalue())


def test_queue_server_dispatches_variational_configuration() -> None:
    beliefs = [
        np.asarray([0.7, 0.2, 0.1]),
        np.asarray([0.6, 0.3, 0.1]),
        np.asarray([0.1, 0.1, 0.8]),
    ]
    config = AggregationConfig(
        method="variational",
        robustness=1.5,
        entropy_weight=0.8,
    )
    requests: queue.Queue = queue.Queue()
    responses: dict[int, queue.Queue] = {
        index: queue.Queue() for index in range(len(beliefs))
    }
    workers = [FederationWorker(index, requests, responses[index]) for index in range(len(beliefs))]
    for worker, belief in zip(workers, beliefs, strict=True):
        worker.send_belief(belief)
    server = FederationServer(n_workers=len(beliefs), config=config)
    consensus = server.run_round(requests, responses)
    reference = aggregate_result(beliefs, config=config).consensus
    assert np.array_equal(consensus, reference)
    for worker in workers:
        assert np.array_equal(worker.receive_consensus(), reference)


def test_socket_round_carries_config_hash_and_variational_result() -> None:
    beliefs = [
        np.asarray([0.7, 0.2, 0.1]),
        np.asarray([0.6, 0.3, 0.1]),
        np.asarray([0.1, 0.1, 0.8]),
    ]
    config = AggregationConfig(method="variational", robustness=1.5)
    result = run_socket_round(
        beliefs,
        config=config,
        round_id="configured-round",
        auth_key="test-key",
    )
    assert result["bit_identical"] is True
    assert result["protocol_version"] == PROTOCOL_VERSION
    assert result["round_id"] == "configured-round"
    assert result["aggregation_config_hash"] == config.fingerprint
    assert result["replay_valid"] is True
    assert np.array_equal(
        result["consensus"],
        aggregate_result(beliefs, config=config).consensus,
    )


def test_config_and_legacy_robustness_conflicts_fail() -> None:
    config = AggregationConfig(method="robust", robustness=1.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        FederationServer(1, robustness=1.0, config=config)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_socket_round(
            [np.asarray([0.5, 0.5])],
            robustness=1.0,
            config=config,
        )
