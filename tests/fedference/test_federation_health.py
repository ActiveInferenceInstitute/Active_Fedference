"""Rich federation diagnostics and structured replay findings."""

from __future__ import annotations

import hashlib
import json
import queue
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fedference import (
    AggregationConfig,
    SharingDiagnostics,
    SharingRoundResult,
    aggregate_result,
    share_round,
    share_round_result,
)
from fedference.federation import (
    MultiprocessRoundResult,
    ReplayValidationResult,
    SocketRoundResult,
    aggregation_config_from_replay,
    inspect_socket_replay,
    load_socket_replay,
    run_multiprocess_round,
    run_multiprocess_round_result,
    run_socket_round,
    run_socket_round_result,
    validate_socket_replay,
)
from fedference.federation.server import FederationServer
from fedference.federation.transport import (
    PROTOCOL_VERSION,
    serialize_envelope,
    serialize_result,
)
from fedference.federation.worker import FederationWorker

BELIEFS = (
    np.asarray([0.70, 0.20, 0.10]),
    np.asarray([0.60, 0.30, 0.10]),
    np.asarray([0.10, 0.10, 0.80]),
    np.asarray([0.65, 0.25, 0.10]),
)
CONFIG = AggregationConfig(method="robust", robustness=1.5, max_iter=32)
NONUNIT_MASSES = tuple(
    belief * scale
    for belief, scale in zip(BELIEFS, (10.0, 2.5, 7.0, 4.0), strict=True)
)


@pytest.fixture(scope="module")
def socket_result() -> SocketRoundResult:
    """Produce one real socket result reused by read-only replay tests."""
    return run_socket_round_result(
        list(BELIEFS),
        config=CONFIG,
        round_id="structured-replay",
    )


@pytest.fixture(scope="module")
def one_worker_socket_result() -> SocketRoundResult:
    """Produce a one-worker replay for equality-confusion metadata probes."""
    return run_socket_round_result(
        [BELIEFS[0]],
        config=CONFIG,
        round_id="structured-replay-one-worker",
    )


def test_sharing_round_result_retains_every_aggregation_diagnostic() -> None:
    result = share_round_result(
        BELIEFS,
        config=CONFIG,
        exclude_self=True,
        true_state=0,
    )
    direct = aggregate_result(BELIEFS, config=CONFIG)
    assert isinstance(result, SharingRoundResult)
    assert len(result.recipient_aggregations) == len(BELIEFS)
    assert np.array_equal(result.consensus, direct.consensus)
    assert result.global_aggregation.solver_status == direct.solver_status
    assert result.true_state == 0
    assert result.mean_accuracy is not None
    assert result.mean_surprise is not None
    assert result.shared_posteriors.flags.writeable is False
    assert result.global_aggregation.consensus.flags.writeable is False
    assert all(
        aggregation.consensus.flags.writeable is False
        for aggregation in result.recipient_aggregations
    )
    with pytest.raises(ValueError, match="read-only"):
        result.shared_posteriors[0, 0] = 0.0


def test_share_round_preserves_historical_shape_and_nan_behavior() -> None:
    rich = share_round_result(BELIEFS, method="naive", exclude_self=False)
    legacy = share_round(BELIEFS, method="naive", exclude_self=False)
    assert isinstance(legacy, SharingDiagnostics)
    assert np.array_equal(legacy.consensus, rich.consensus)
    assert np.array_equal(legacy.shared_posteriors, rich.shared_posteriors)
    assert legacy.normalized_effective_weights is None
    assert np.isnan(legacy.mean_accuracy)
    assert np.isnan(legacy.mean_surprise)
    assert rich.mean_accuracy is None
    assert rich.mean_surprise is None


def test_server_rich_round_retains_result_but_wire_payload_is_unchanged() -> None:
    requests: queue.Queue[Any] = queue.Queue()
    responses: dict[int, queue.Queue[Any]] = {
        worker_id: queue.Queue() for worker_id in range(len(BELIEFS))
    }
    workers = [
        FederationWorker(worker_id, requests, responses[worker_id])
        for worker_id in range(len(BELIEFS))
    ]
    for worker, belief in zip(workers, BELIEFS, strict=True):
        worker.send_belief(belief)
    server = FederationServer(n_workers=len(BELIEFS), config=CONFIG)
    result = server.run_round_result(requests, responses)
    direct = aggregate_result(BELIEFS, config=CONFIG)
    assert result.solver_status == direct.solver_status
    assert np.array_equal(result.consensus, direct.consensus)
    for worker in workers:
        assert np.array_equal(worker.receive_consensus(), result.consensus)


@pytest.mark.integration
def test_multiprocess_rich_result_is_ordered_and_legacy_is_an_array() -> None:
    result = run_multiprocess_round_result(BELIEFS, config=CONFIG, timeout=5.0)
    legacy = run_multiprocess_round(BELIEFS, config=CONFIG, timeout=5.0)
    assert isinstance(result, MultiprocessRoundResult)
    assert isinstance(legacy, np.ndarray)
    assert result.config == CONFIG
    assert result.n_workers == len(BELIEFS)
    assert result.bit_identical is True
    assert np.array_equal(result.consensus, legacy)
    for worker_consensus in result.worker_consensuses:
        assert np.array_equal(worker_consensus, result.consensus)
        assert worker_consensus.flags.writeable is False


def test_socket_rich_result_and_legacy_dictionary_contract(
    socket_result: SocketRoundResult,
) -> None:
    legacy = run_socket_round(BELIEFS, config=CONFIG, round_id="legacy-projection")
    assert isinstance(socket_result, SocketRoundResult)
    assert socket_result.protocol_version == PROTOCOL_VERSION == 1
    assert socket_result.bit_identical is True
    assert socket_result.replay_validation.integrity_valid is True
    assert len(socket_result.worker_consensuses) == len(BELIEFS)
    assert all(
        not consensus.flags.writeable
        for consensus in socket_result.worker_consensuses
    )
    assert set(legacy) == {
        "consensus",
        "in_process",
        "bit_identical",
        "worker_consensuses",
        "n_workers",
        "port",
        "authenticated",
        "protocol_version",
        "round_id",
        "aggregation_config",
        "aggregation_config_hash",
        "replay",
        "replay_path",
        "replay_valid",
    }
    assert np.array_equal(legacy["consensus"], socket_result.consensus)


def test_protocol_v1_result_and_envelope_fixtures_remain_byte_identical() -> None:
    result_payload = serialize_result(
        np.asarray([0.75, 0.25], dtype=np.float64),
        np.asarray([0.5, 0.5], dtype=np.float64),
    )
    envelope = serialize_envelope(
        result_payload,
        message_type="result",
        round_id="fixture-round",
        worker_id=None,
        aggregation_config_hash=CONFIG.fingerprint,
        authentication="none",
    )
    assert len(result_payload) == 562
    assert hashlib.sha256(result_payload).hexdigest() == (
        "a3690e2c28203a461a0c08b37be712111f5b7e7db607e965503e16b289585820"
    )
    assert len(envelope) == 857
    assert hashlib.sha256(envelope).hexdigest() == (
        "236853b31db0b3ea0bbd1cf407b8cf7860b97e2b0d04514c740a7079c73ff3eb"
    )


def test_nonunit_positive_masses_are_bit_identical_across_every_boundary() -> None:
    direct = aggregate_result(NONUNIT_MASSES, config=CONFIG)
    sharing = share_round_result(
        NONUNIT_MASSES,
        config=CONFIG,
        exclude_self=False,
    )

    request_queue: queue.Queue = queue.Queue()
    response_queues: dict[int, queue.Queue] = {
        worker_id: queue.Queue() for worker_id in range(len(NONUNIT_MASSES))
    }
    workers = [
        FederationWorker(worker_id, request_queue, response_queues[worker_id])
        for worker_id in range(len(NONUNIT_MASSES))
    ]
    for worker, belief in zip(workers, NONUNIT_MASSES, strict=True):
        worker.send_belief(belief)
    server = FederationServer(n_workers=len(NONUNIT_MASSES), config=CONFIG)
    server_result = server.run_round_result(request_queue, response_queues)

    process = run_multiprocess_round_result(NONUNIT_MASSES, config=CONFIG)
    socket = run_socket_round_result(
        list(NONUNIT_MASSES),
        config=CONFIG,
        round_id="nonunit-positive-masses",
    )

    assert np.array_equal(sharing.consensus, direct.consensus)
    assert np.array_equal(server_result.consensus, direct.consensus)
    assert np.array_equal(process.consensus, direct.consensus)
    assert np.array_equal(socket.consensus, direct.consensus)
    assert process.bit_identical is True
    assert socket.bit_identical is True
    assert socket.replay_validation.integrity_valid is True
    assert validate_socket_replay(
        list(socket.replay),
        list(NONUNIT_MASSES),
        socket.consensus,
        config=CONFIG,
    )


def test_replay_uses_recorded_config_and_partial_flags_are_assertions(
    socket_result: SocketRoundResult,
) -> None:
    recorded = aggregation_config_from_replay(list(socket_result.replay))
    inspected = inspect_socket_replay(
        list(socket_result.replay),
        list(BELIEFS),
        socket_result.consensus,
    )
    asserted = inspect_socket_replay(
        list(socket_result.replay),
        list(BELIEFS),
        socket_result.consensus,
        config_assertions={"method": "robust", "robustness": 1.5},
    )
    mismatch = inspect_socket_replay(
        list(socket_result.replay),
        list(BELIEFS),
        socket_result.consensus,
        config_assertions={"method": "naive"},
    )
    assert recorded == CONFIG
    assert inspected.integrity_valid is True
    assert inspected.recorded_config == CONFIG
    assert asserted.integrity_valid is True
    assert mismatch.integrity_valid is False
    assert mismatch.findings[0].code == "configuration.mismatch"
    assert validate_socket_replay(
        list(socket_result.replay),
        list(BELIEFS),
        socket_result.consensus,
    )


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("event_order", "event.order_mismatch"),
        ("event_fields", "event.fields_mismatch"),
        ("protocol", "protocol.mismatch"),
        ("round", "round.mismatch"),
        ("configuration", "configuration.mismatch"),
        ("worker_order", "worker.order_mismatch"),
        ("belief", "belief.mismatch"),
        ("frame", "frame.mismatch"),
        ("payload", "payload.mismatch"),
        ("envelope", "envelope.mismatch"),
        ("consensus", "consensus.recomputed_mismatch"),
    ),
)
def test_structured_replay_integrity_finding_codes(
    socket_result: SocketRoundResult,
    case: str,
    expected_code: str,
) -> None:
    replay = deepcopy(list(socket_result.replay))
    consensus = socket_result.consensus
    received = next(event for event in replay if event["event"] == "belief_received")
    aggregate = next(event for event in replay if event["event"] == "aggregate")
    broadcast = next(
        event for event in replay if event["event"] == "consensus_broadcast"
    )
    if case == "event_order":
        replay[1], replay[-1] = replay[-1], replay[1]
    elif case == "event_fields":
        replay[0]["unexpected"] = True
    elif case == "protocol":
        replay[0]["protocol_version"] = PROTOCOL_VERSION + 1
    elif case == "round":
        aggregate["round_id"] = "different-round"
    elif case == "configuration":
        replay[0]["aggregation_config_hash"] = "0" * 64
    elif case == "worker_order":
        aggregate["worker_order"] = list(reversed(aggregate["worker_order"]))
    elif case == "belief":
        received["belief_sha256"] = "0" * 64
    elif case == "frame":
        received["frame_bytes"] += 1
    elif case == "payload":
        broadcast["payload_sha256"] = "0" * 64
    elif case == "envelope":
        broadcast["envelope_sha256"] = "0" * 64
    elif case == "consensus":
        consensus = np.asarray([0.0, 1.0, 0.0])
        aggregate["consensus_sha256"] = hashlib.sha256(
            _serialize_belief_fixture(consensus)
        ).hexdigest()
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(case)
    verdict = inspect_socket_replay(replay, list(BELIEFS), consensus)
    assert verdict.integrity_valid is False
    assert verdict.findings == (
        verdict.findings[0],
    )
    assert verdict.findings[0].category == "integrity"
    assert verdict.findings[0].code == expected_code


@pytest.mark.parametrize(
    "event_name",
    (
        "server_listen",
        "belief_received",
        "aggregate",
        "consensus_broadcast",
    ),
)
@pytest.mark.parametrize("invalid", (True, "1", 1.0))
def test_replay_protocol_version_requires_an_exact_integer(
    socket_result: SocketRoundResult,
    event_name: str,
    invalid: object,
) -> None:
    replay = deepcopy(list(socket_result.replay))
    event = next(event for event in replay if event["event"] == event_name)
    event["protocol_version"] = invalid
    verdict = inspect_socket_replay(replay, list(BELIEFS), socket_result.consensus)
    assert [finding.code for finding in verdict.findings] == ["protocol.mismatch"]


@pytest.mark.parametrize("invalid", (True, "1", 1.0))
def test_replay_worker_count_requires_an_exact_integer(
    one_worker_socket_result: SocketRoundResult,
    invalid: object,
) -> None:
    replay = deepcopy(list(one_worker_socket_result.replay))
    replay[0]["n_workers"] = invalid
    verdict = inspect_socket_replay(
        replay,
        [BELIEFS[0]],
        one_worker_socket_result.consensus,
    )
    assert [finding.code for finding in verdict.findings] == [
        "event.fields_mismatch"
    ]


@pytest.mark.parametrize(
    "event_name",
    ("belief_received", "consensus_broadcast"),
)
@pytest.mark.parametrize("invalid", (False, "0", 0.0, [], {}))
def test_replay_worker_ids_reject_nonintegers_before_set_construction(
    socket_result: SocketRoundResult,
    event_name: str,
    invalid: object,
) -> None:
    replay = deepcopy(list(socket_result.replay))
    event = next(
        event
        for event in replay
        if event["event"] == event_name and event["worker_id"] == 0
    )
    event["worker_id"] = invalid
    verdict = inspect_socket_replay(replay, list(BELIEFS), socket_result.consensus)
    assert [finding.code for finding in verdict.findings] == [
        "worker.order_mismatch"
    ]


@pytest.mark.parametrize(
    ("event_name", "field", "expected_code"),
    (
        ("belief_received", "frame_bytes", "frame.mismatch"),
        ("consensus_broadcast", "payload_bytes", "payload.mismatch"),
        ("consensus_broadcast", "envelope_bytes", "envelope.mismatch"),
    ),
)
@pytest.mark.parametrize("invalid_kind", ("boolean", "string", "float"))
def test_replay_byte_counts_require_exact_integers(
    socket_result: SocketRoundResult,
    event_name: str,
    field: str,
    expected_code: str,
    invalid_kind: str,
) -> None:
    replay = deepcopy(list(socket_result.replay))
    event = next(event for event in replay if event["event"] == event_name)
    original = event[field]
    replacements = {
        "boolean": True,
        "string": str(original),
        "float": float(original),
    }
    event[field] = replacements[invalid_kind]
    verdict = inspect_socket_replay(replay, list(BELIEFS), socket_result.consensus)
    assert [finding.code for finding in verdict.findings] == [expected_code]


def _serialize_belief_fixture(belief: np.ndarray) -> bytes:
    """Use the public serializer without exposing it in every test import."""
    from fedference.federation.transport import serialize_belief

    return serialize_belief(belief)


def test_empty_and_malformed_replays_have_distinct_stable_codes() -> None:
    empty = inspect_socket_replay([], list(BELIEFS), BELIEFS[0])
    malformed = inspect_socket_replay([42], list(BELIEFS), BELIEFS[0])  # type: ignore[list-item]
    assert empty.findings[0].code == "replay.empty"
    assert malformed.findings[0].code == "replay.malformed"


def test_nonconvergence_and_fallback_are_solver_findings_not_integrity_failures() -> None:
    nonconverged = run_socket_round_result(
        list(BELIEFS),
        config=AggregationConfig(
            method="robust",
            robustness=1.5,
            max_iter=1,
            tol=0.0,
        ),
        round_id="nonconverged-round",
    )
    fallback = run_socket_round_result(
        list(BELIEFS),
        config=AggregationConfig(
            method="robust",
            robustness=1e9,
            max_iter=5,
        ),
        round_id="fallback-round",
    )
    assert nonconverged.replay_validation.integrity_valid is True
    assert nonconverged.replay_validation.solver_status == "not_converged"
    assert [
        finding.code for finding in nonconverged.replay_validation.findings
    ] == ["solver.nonconvergence"]
    assert fallback.replay_validation.integrity_valid is True
    assert fallback.replay_validation.solver_status == "not_converged_with_fallback"
    assert {
        finding.code for finding in fallback.replay_validation.findings
    } == {"solver.nonconvergence", "solver.fallback"}
    assert validate_socket_replay(
        list(fallback.replay),
        list(BELIEFS),
        fallback.consensus,
    )


@pytest.mark.parametrize(
    ("config", "expected_status"),
    (
        (
            AggregationConfig(
                method="robust",
                robustness=1.5,
                max_iter=1,
                tol=0.0,
            ),
            "not_converged",
        ),
        (
            AggregationConfig(
                method="robust",
                robustness=1e9,
                max_iter=5,
            ),
            "not_converged_with_fallback",
        ),
    ),
)
def test_solver_health_is_visible_in_direct_and_every_sharing_result(
    config: AggregationConfig,
    expected_status: str,
) -> None:
    direct = aggregate_result(BELIEFS, config=config)
    shared = share_round_result(
        BELIEFS,
        config=config,
        exclude_self=True,
    )
    assert direct.solver_status == expected_status
    assert shared.global_aggregation.solver_status == expected_status
    assert [
        result.solver_status for result in shared.recipient_aggregations
    ] == [expected_status] * len(BELIEFS)
    if expected_status.endswith("with_fallback"):
        assert direct.fallback_events
        assert shared.global_aggregation.fallback_events
        assert all(
            result.fallback_events
            for result in shared.recipient_aggregations
        )
    else:
        assert not direct.fallback_events
        assert not shared.global_aggregation.fallback_events
        assert all(
            not result.fallback_events
            for result in shared.recipient_aggregations
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("config", "expected_status"),
    (
        (
            AggregationConfig(
                method="robust",
                robustness=1.5,
                max_iter=1,
                tol=0.0,
            ),
            "not_converged",
        ),
        (
            AggregationConfig(
                method="robust",
                robustness=1e9,
                max_iter=5,
            ),
            "not_converged_with_fallback",
        ),
    ),
)
def test_solver_health_is_visible_in_multiprocess_server_result(
    config: AggregationConfig,
    expected_status: str,
) -> None:
    result = run_multiprocess_round_result(
        BELIEFS,
        config=config,
        timeout=5.0,
    )
    assert result.server_aggregation.solver_status == expected_status
    assert result.bit_identical is True
    assert all(
        np.array_equal(worker_consensus, result.consensus)
        for worker_consensus in result.worker_consensuses
    )
    if expected_status.endswith("with_fallback"):
        assert result.server_aggregation.fallback_events
    else:
        assert not result.server_aggregation.fallback_events


def test_replay_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    replay_path = tmp_path / "duplicate.json"
    replay_path.write_text(
        '[{"event":"server_listen","event":"aggregate"}]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid socket replay"):
        load_socket_replay(replay_path)


def test_replay_verdict_has_stable_json_shape(
    socket_result: SocketRoundResult,
) -> None:
    payload = socket_result.replay_validation.as_dict()
    assert set(payload) == {
        "integrity_valid",
        "solver_status",
        "nominal_solver",
        "aggregation_config",
        "aggregation_config_hash",
        "findings",
    }
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert isinstance(socket_result.replay_validation, ReplayValidationResult)


@pytest.mark.parametrize(
    "consensus",
    (
        np.asarray([-0.1, 0.6, 0.5]),
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.7, 0.2, 0.2]),
        np.asarray([[0.7, 0.2, 0.1]]),
        np.asarray([[0.7], [0.2], [0.1]]),
        np.asarray([[[0.7, 0.2, 0.1]]]),
        np.asarray([True, False, False]),
        np.asarray(["0.7", "0.2", "0.1"]),
        np.asarray([0.7, None, 0.3], dtype=object),
    ),
)
def test_replay_inspection_rejects_malformed_consensus_before_hashing(
    socket_result: SocketRoundResult,
    consensus: np.ndarray,
) -> None:
    verdict = inspect_socket_replay(
        list(socket_result.replay),
        list(BELIEFS),
        consensus,
    )
    assert verdict.integrity_valid is False
    assert verdict.solver_status is None
    assert [finding.code for finding in verdict.findings] == ["input.malformed"]
    assert "consensus" in verdict.findings[0].message


@pytest.mark.parametrize(
    "operation",
    (
        lambda: share_round_result([np.asarray([[0.5, 0.5]])]),
        lambda: run_multiprocess_round_result([np.asarray([[0.5, 0.5]])]),
        lambda: run_socket_round_result([np.asarray([[0.5, 0.5]])]),
    ),
)
def test_rich_federation_boundaries_reject_implicit_matrix_flattening(
    operation: Any,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        operation()
