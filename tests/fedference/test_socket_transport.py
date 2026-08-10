"""Loopback-TCP federation transport (MAJ-4) — no mocks, real sockets.

Real TCP connections over loopback with length-prefixed framing. The socket
round must return a consensus BIT-IDENTICAL (atol=0) to the in-process
robust_aggregate — any framing bug (short read, wrong byte order, truncation)
breaks the exact equality. All workers must receive the same broadcast.
"""

from __future__ import annotations

import socket
import sqlite3
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import Any

import numpy as np
import pytest

from fedference.aggregation import robust_aggregate
from fedference.federation import (
    PersistentReplayGuard as PublicPersistentReplayGuard,
)
from fedference.federation.socket_transport import (
    _MAX_FRAME_BYTES,
    PersistentReplayGuard,
    ReplayGuard,
    _recv_framed,
    _send_framed,
    _validate_loopback_host,
    load_socket_replay,
    run_socket_round,
    save_socket_replay,
    validate_socket_replay,
)

_BELIEFS = [
    np.array([0.70, 0.20, 0.10]),
    np.array([0.60, 0.30, 0.10]),
    np.array([0.10, 0.10, 0.80]),  # an outlier
    np.array([0.65, 0.25, 0.10]),
]


def test_persistent_replay_guard_is_exported_from_federation_namespace() -> None:
    assert PublicPersistentReplayGuard is PersistentReplayGuard


@pytest.mark.parametrize("robustness", [0.0, 1.5])
def test_socket_round_is_bit_identical_to_in_process(robustness: float) -> None:
    out = run_socket_round(_BELIEFS, robustness=robustness)
    reference = robust_aggregate(_BELIEFS, robustness=robustness).consensus
    assert out["bit_identical"] is True
    assert np.array_equal(out["consensus"], reference)  # atol = 0, exact
    assert out["n_workers"] == len(_BELIEFS)
    assert out["port"] > 1024  # a real ephemeral port was bound


def test_every_worker_receives_the_same_broadcast() -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5)
    reference = robust_aggregate(_BELIEFS, robustness=1.5).consensus
    assert len(out["worker_consensuses"]) == len(_BELIEFS)
    for consensus in out["worker_consensuses"].values():
        assert np.array_equal(consensus, reference)


def test_replay_log_reconstructs_consensus() -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5)
    assert out["replay_valid"] is True
    assert validate_socket_replay(out["replay"], _BELIEFS, out["consensus"], robustness=1.5)
    assert [event["event"] for event in out["replay"]].count("belief_received") == len(_BELIEFS)
    assert [event["event"] for event in out["replay"]].count("consensus_broadcast") == len(_BELIEFS)


def test_replay_log_persists_and_reloads_from_disk(tmp_path) -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5, replay_path=tmp_path / "round.json")
    persisted = load_socket_replay(tmp_path / "round.json")
    assert out["replay_path"] == str(tmp_path / "round.json")
    assert persisted == out["replay"]
    assert validate_socket_replay(persisted, _BELIEFS, out["consensus"], robustness=1.5)


def test_save_socket_replay_creates_parent_directory(tmp_path) -> None:
    out = run_socket_round(_BELIEFS, robustness=0.0)
    replay_path = save_socket_replay(tmp_path / "nested" / "round.json", out["replay"])
    assert replay_path.exists()
    assert load_socket_replay(replay_path) == out["replay"]


def test_load_socket_replay_rejects_non_event_list(tmp_path) -> None:
    replay_path = tmp_path / "bad.json"
    replay_path.write_text('{"event": "aggregate"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="event objects"):
        load_socket_replay(replay_path)
    replay_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid socket replay"):
        load_socket_replay(replay_path)
    replay_path.write_text('[{"event": "aggregate", "value": NaN}]', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid socket replay"):
        load_socket_replay(replay_path)


def test_replay_log_rejects_wrong_consensus() -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5)
    wrong = np.array([0.0, 1.0, 0.0])
    assert not validate_socket_replay(out["replay"], _BELIEFS, wrong, robustness=1.5)


def test_replay_log_rejects_tampered_belief_digest() -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5)
    replay = [dict(event) for event in out["replay"]]
    replay[1]["belief_sha256"] = "0" * 64
    assert not validate_socket_replay(replay, _BELIEFS, out["consensus"], robustness=1.5)


def test_replay_log_rejects_tampered_consensus_digest() -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5)
    replay = [dict(event) for event in out["replay"]]
    aggregate = next(event for event in replay if event["event"] == "aggregate")
    aggregate["consensus_sha256"] = "0" * 64
    assert not validate_socket_replay(replay, _BELIEFS, out["consensus"], robustness=1.5)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("n_workers", 999),
        ("authentication", "tampered"),
    ),
)
def test_replay_log_rejects_tampered_listen_metadata(field, value) -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5)
    replay = [dict(event) for event in out["replay"]]
    listen = next(event for event in replay if event["event"] == "server_listen")
    listen[field] = value
    assert not validate_socket_replay(
        replay,
        _BELIEFS,
        out["consensus"],
        robustness=1.5,
    )


def test_replay_log_rejects_tampered_frame_size() -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5)
    replay = [dict(event) for event in out["replay"]]
    replay[1]["frame_bytes"] += 1
    assert not validate_socket_replay(replay, _BELIEFS, out["consensus"], robustness=1.5)


def test_replay_log_rejects_inconsistent_broadcast_digest() -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5)
    replay = [dict(event) for event in out["replay"]]
    broadcast = next(event for event in replay if event["event"] == "consensus_broadcast")
    broadcast["payload_sha256"] = "0" * 64
    assert not validate_socket_replay(replay, _BELIEFS, out["consensus"], robustness=1.5)


def test_authenticated_socket_round_preserves_bit_identity() -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5, auth_key="shared-test-key")
    assert out["authenticated"] is True
    assert out["bit_identical"] is True
    assert out["replay"][0]["authenticated"] is True
    assert out["replay_valid"] is True


def test_authenticated_frame_rejects_wrong_key() -> None:
    left, right = socket.socketpair()
    try:
        _send_framed(left, b"payload", auth_key=b"correct")
        with pytest.raises(PermissionError, match="authentication tag"):
            _recv_framed(right, auth_key=b"wrong")
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("declared_length", [0, _MAX_FRAME_BYTES + 1])
def test_frame_length_guard_rejects_empty_or_oversized_frames(
    declared_length: int,
) -> None:
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack(">I", declared_length))
        with pytest.raises(ValueError, match="frame length"):
            _recv_framed(right)
    finally:
        left.close()
        right.close()


def test_auth_key_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="auth_key"):
        run_socket_round(_BELIEFS, auth_key=b"")
    with pytest.raises(ValueError, match="bytes"):
        run_socket_round(_BELIEFS, auth_key=123)


def test_single_worker_round() -> None:
    out = run_socket_round([np.array([0.5, 0.5])], robustness=1.5)
    assert out["bit_identical"] is True
    assert out["n_workers"] == 1


def test_rejects_empty_beliefs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_socket_round([], robustness=0.0)


def test_socket_round_rejects_non_configuration_objects() -> None:
    invalid: Any = {"method": "naive"}
    with pytest.raises(ValueError, match="config must be an AggregationConfig"):
        run_socket_round(_BELIEFS, config=invalid)
    assert not validate_socket_replay([], _BELIEFS, _BELIEFS[0], config=invalid)


def test_reordering_agnostic_consensus_holds() -> None:
    """The server orders beliefs by worker id, so socket arrival order must not
    change the fused consensus (matches the in-process ordering guarantee)."""
    out = run_socket_round(_BELIEFS, robustness=1.5)
    reordered = run_socket_round(list(reversed(_BELIEFS)), robustness=1.5)
    # Same multiset of beliefs -> the robust pool is order-independent up to the
    # id sort; both must match their own in-process reference exactly.
    assert out["bit_identical"] and reordered["bit_identical"]


def test_replay_rejects_duplicate_worker_events_and_config_tamper() -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5)
    duplicate = [dict(event) for event in out["replay"]]
    received = next(event for event in duplicate if event["event"] == "belief_received")
    duplicate.append(dict(received))
    assert not validate_socket_replay(
        duplicate,
        _BELIEFS,
        out["consensus"],
        robustness=1.5,
    )

    changed = [dict(event) for event in out["replay"]]
    aggregate = next(event for event in changed if event["event"] == "aggregate")
    aggregate["aggregation_config"] = dict(aggregate["aggregation_config"])
    aggregate["aggregation_config"]["robustness"] = 9.0
    assert not validate_socket_replay(
        changed,
        _BELIEFS,
        out["consensus"],
        robustness=1.5,
    )


def test_replay_rejects_unknown_fields_and_out_of_order_events() -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5)
    extra = [dict(event) for event in out["replay"]]
    extra[0]["unversioned"] = "field"
    assert not validate_socket_replay(
        extra,
        _BELIEFS,
        out["consensus"],
        robustness=1.5,
    )

    reordered = [dict(event) for event in out["replay"]]
    reordered[1], reordered[-1] = reordered[-1], reordered[1]
    assert not validate_socket_replay(
        reordered,
        _BELIEFS,
        out["consensus"],
        robustness=1.5,
    )

    changed_auth = [dict(event) for event in out["replay"]]
    received = next(
        event for event in changed_auth if event["event"] == "belief_received"
    )
    received["authentication"] = "tampered"
    assert not validate_socket_replay(
        changed_auth,
        _BELIEFS,
        out["consensus"],
        robustness=1.5,
    )


@pytest.mark.parametrize(
    "worker_order",
    (
        [0, 1, 2, []],
        [0, 1, 2, "3"],
        [False, 1, 2, 3],
    ),
)
def test_malformed_replay_worker_orders_fail_closed(worker_order) -> None:
    out = run_socket_round(_BELIEFS, robustness=1.5)
    replay = [dict(event) for event in out["replay"]]
    aggregate = next(event for event in replay if event["event"] == "aggregate")
    aggregate["worker_order"] = worker_order
    assert not validate_socket_replay(
        replay,
        _BELIEFS,
        out["consensus"],
        robustness=1.5,
    )


def test_shared_replay_guard_rejects_round_id_reuse() -> None:
    guard = ReplayGuard()
    first = run_socket_round(
        _BELIEFS,
        robustness=1.5,
        round_id="guarded-round",
        replay_guard=guard,
    )
    assert first["bit_identical"]
    with pytest.raises(ValueError, match="replayed socket round"):
        run_socket_round(
            _BELIEFS,
            robustness=1.5,
            round_id="guarded-round",
            replay_guard=guard,
        )
    second = run_socket_round(
        _BELIEFS,
        robustness=1.5,
        round_id="guarded-round-2",
        replay_guard=guard,
    )
    assert second["bit_identical"]


def test_persistent_replay_guard_rejects_reuse_after_reopen(tmp_path) -> None:
    database = tmp_path / "replay" / "rounds.sqlite3"
    first_guard = PersistentReplayGuard(database)
    first = run_socket_round(
        _BELIEFS,
        robustness=1.5,
        round_id="durable-round",
        replay_guard=first_guard,
    )
    assert first["bit_identical"]
    assert database.exists()

    restarted_guard = PersistentReplayGuard(database)
    with pytest.raises(ValueError, match="replayed socket round"):
        run_socket_round(
            _BELIEFS,
            robustness=1.5,
            round_id="durable-round",
            replay_guard=restarted_guard,
        )
    second = run_socket_round(
        _BELIEFS,
        robustness=1.5,
        round_id="durable-round-2",
        replay_guard=restarted_guard,
    )
    assert second["bit_identical"]


def test_persistent_replay_guard_rejects_unsafe_paths(tmp_path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="must be a file"):
        PersistentReplayGuard(directory)

    target = tmp_path / "target.sqlite3"
    target.touch()
    link = tmp_path / "link.sqlite3"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        PersistentReplayGuard(link)


def test_persistent_replay_guard_claim_is_atomic_across_instances(tmp_path) -> None:
    database = tmp_path / "rounds.sqlite3"
    guards = (PersistentReplayGuard(database), PersistentReplayGuard(database))
    barrier = threading.Barrier(2)

    def claim(guard: PersistentReplayGuard) -> str:
        barrier.wait()
        try:
            guard.claim("contended-round")
        except ValueError as exc:
            assert "replayed socket round" in str(exc)
            return "rejected"
        return "claimed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(claim, guards))

    assert outcomes == ["claimed", "rejected"]


def test_persistent_replay_guard_rejects_unknown_or_corrupt_schema(tmp_path) -> None:
    future_database = tmp_path / "future.sqlite3"
    with closing(sqlite3.connect(future_database)) as connection, connection:
        connection.execute("PRAGMA user_version = 2")
    with pytest.raises(ValueError, match="unsupported persistent replay schema"):
        PersistentReplayGuard(future_database)

    corrupt_database = tmp_path / "corrupt.sqlite3"
    corrupt_database.write_bytes(b"not a sqlite database")
    with pytest.raises(ValueError, match="invalid persistent replay guard database"):
        PersistentReplayGuard(corrupt_database)


@pytest.mark.parametrize("host", ("", "0.0.0.0", "192.0.2.1"))
def test_socket_round_rejects_non_loopback_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        run_socket_round(_BELIEFS, host=host)


def test_socket_round_accepts_localhost_alias() -> None:
    assert _validate_loopback_host("localhost").startswith("127.")
    out = run_socket_round(_BELIEFS, host="localhost", round_id="localhost-round")
    assert out["bit_identical"]


@pytest.mark.parametrize("bad_round_id", ("", "   ", 7))
def test_round_id_and_timeout_types_fail_before_socket_use(bad_round_id) -> None:
    with pytest.raises(ValueError, match="round_id"):
        run_socket_round(_BELIEFS, round_id=bad_round_id)
    with pytest.raises(ValueError, match="timeout"):
        run_socket_round(_BELIEFS, timeout=True)
