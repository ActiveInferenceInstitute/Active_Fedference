"""Loopback-TCP federation transport (MAJ-4 slice).

The existing federation transport fuses beliefs over in-process queues or spawned
OS processes. This module adds a REAL network path: workers connect to the server
over loopback TCP sockets and exchange the same lossless float64 ``.npy`` /
``.npz`` payloads through length-prefixed framing. Because the wire format is
byte-preserving, the socket-federated consensus is **bit-identical** (``atol=0``)
to the matching configured in-process call—any framing bug (short read, wrong
byte order, truncation) breaks that exact identity, which the end-to-end test
asserts.

Scope (kept honest): this is a single-machine loopback-TCP adapter demonstrating
a genuine network transport with real socket framing, optional HMAC frame
authentication, versioned configuration-bound envelopes, and deterministic
digest-verified replay validation. It is NOT a Docker/mTLS emulator or
physical cross-host federation with discovery and fault tolerance — those
remain MAJ-4A/4B. The value here is a real, tested wire protocol proving
the FederationServer contract is transport-agnostic down to raw sockets.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import socket
import sqlite3
import struct
import tempfile
import threading
import warnings
from contextlib import closing
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ..aggregation import AggregationConfig, aggregate_result
from .transport import (
    PROTOCOL_VERSION,
    deserialize_belief,
    deserialize_envelope,
    deserialize_result,
    serialize_belief,
    serialize_envelope,
    serialize_result,
)

#: Framing: a 4-byte big-endian unsigned length prefix precedes each payload.
_LEN = struct.Struct(">I")
_AUTH_TAG_BYTES = hashlib.sha256().digest_size
_MAX_FRAME_BYTES = 128 * 1024 * 1024
_REPLAY_FIELDS: dict[str, frozenset[str]] = {
    "server_listen": frozenset(
        {
            "event",
            "host",
            "port",
            "n_workers",
            "authenticated",
            "authentication",
            "protocol_version",
            "round_id",
            "aggregation_config_hash",
        }
    ),
    "belief_received": frozenset(
        {
            "event",
            "worker_id",
            "protocol_version",
            "round_id",
            "aggregation_config_hash",
            "authentication",
            "frame_sha256",
            "belief_sha256",
            "frame_bytes",
        }
    ),
    "aggregate": frozenset(
        {
            "event",
            "worker_order",
            "method",
            "robustness",
            "aggregation_config",
            "aggregation_config_hash",
            "protocol_version",
            "round_id",
            "authentication",
            "consensus_sha256",
        }
    ),
    "consensus_broadcast": frozenset(
        {
            "event",
            "worker_id",
            "protocol_version",
            "round_id",
            "aggregation_config_hash",
            "authentication",
            "payload_sha256",
            "payload_bytes",
            "envelope_sha256",
            "envelope_bytes",
        }
    ),
}


class ReplayGuard:
    """In-memory guard against round-id reuse within one running process.

    Share one instance across successive :func:`run_socket_round` calls.
    Use :class:`PersistentReplayGuard` when claims must survive a local process
    restart. Multi-host replay-domain design remains part of MAJ-4A/4B.
    """

    def __init__(self) -> None:
        self._round_ids: set[str] = set()
        self._lock = threading.Lock()

    def claim(self, round_id: str) -> None:
        """Atomically reserve a non-empty round id or reject its reuse."""
        if not isinstance(round_id, str) or not round_id.strip():
            raise ValueError("round_id must be a non-empty string")
        with self._lock:
            if round_id in self._round_ids:
                raise ValueError(f"replayed socket round id: {round_id!r}")
            self._round_ids.add(round_id)


class PersistentReplayGuard(ReplayGuard):
    """SQLite-backed round-id guard that survives process restarts.

    The caller owns the database path and must place it on storage shared by
    every server process that participates in the same replay-protection
    domain. SQLite's primary-key constraint and ``BEGIN IMMEDIATE`` transaction
    make one round-id claim atomic across threads and local processes.

    This is a durable protocol primitive, not a deployment claim: certificate
    identity, key distribution, multi-host shared state, retention, and
    disaster recovery remain responsibilities of the MAJ-4A emulator/runtime.
    """

    _SCHEMA_VERSION = 1

    def __init__(self, path: str | PathLike[str]) -> None:
        replay_path = Path(path)
        if replay_path.exists() and replay_path.is_dir():
            raise ValueError("persistent replay guard path must be a file")
        if replay_path.is_symlink():
            raise ValueError("persistent replay guard path must not be a symlink")
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = replay_path
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=10.0,
            isolation_level=None,
        )
        try:
            connection.execute("PRAGMA synchronous = FULL")
        except sqlite3.Error:
            connection.close()
            raise
        return connection

    def _initialize(self) -> None:
        try:
            with closing(self._connect()) as connection, connection:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version not in (0, self._SCHEMA_VERSION):
                    raise ValueError(
                        f"unsupported persistent replay schema version: {version}"
                    )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS claimed_rounds "
                    "(round_id TEXT PRIMARY KEY NOT NULL CHECK(length(trim(round_id)) > 0))"
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
        except sqlite3.Error as exc:
            raise ValueError(
                f"invalid persistent replay guard database: {self.path}"
            ) from exc

    def claim(self, round_id: str) -> None:
        """Atomically persist a non-empty round id or reject its reuse."""
        if not isinstance(round_id, str) or not round_id.strip():
            raise ValueError("round_id must be a non-empty string")
        with self._lock:
            try:
                with closing(self._connect()) as connection, connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        "INSERT INTO claimed_rounds(round_id) VALUES (?)",
                        (round_id,),
                    )
                    connection.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"replayed socket round id: {round_id!r}") from exc
            except sqlite3.Error as exc:
                raise ValueError(
                    f"persistent replay guard failed for {self.path}"
                ) from exc


def _validate_loopback_host(host: str) -> str:
    """Resolve and return one numeric IPv4 loopback address.

    Returning the checked numeric address prevents a hostname from being
    resolved a second time between validation and ``socket.bind``.
    """
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty loopback hostname or address")
    try:
        resolved = {
            ipaddress.ip_address(address[0])
            for _family, _type, _proto, _canonname, address in socket.getaddrinfo(
                host,
                0,
                family=socket.AF_INET,
                type=socket.SOCK_STREAM,
            )
        }
    except (ValueError, socket.gaierror) as exc:
        raise ValueError(f"host must resolve to IPv4 loopback: {host!r}") from exc
    if not resolved or any(not address.is_loopback for address in resolved):
        raise ValueError(
            "run_socket_round is loopback-only; non-loopback hosts require "
            "the future mTLS multi-node adapter"
        )
    return str(min(resolved, key=int))


def _coerce_auth_key(auth_key: bytes | str | None) -> bytes | None:
    if auth_key is None:
        return None
    if not isinstance(auth_key, (bytes, str)):
        raise ValueError("auth_key must be bytes, text, or None")
    key = auth_key.encode("utf-8") if isinstance(auth_key, str) else auth_key
    if not key:
        raise ValueError("auth_key must be non-empty when provided")
    return key


def _payload_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pack_authenticated(payload: bytes, auth_key: bytes | str | None) -> bytes:
    key = _coerce_auth_key(auth_key)
    if key is None:
        return payload
    tag = hmac.new(key, payload, hashlib.sha256).digest()
    return tag + payload


def _unpack_authenticated(wire_payload: bytes, auth_key: bytes | str | None) -> bytes:
    key = _coerce_auth_key(auth_key)
    if key is None:
        return wire_payload
    if len(wire_payload) < _AUTH_TAG_BYTES:
        raise PermissionError("authenticated socket frame is shorter than its tag")
    tag = wire_payload[:_AUTH_TAG_BYTES]
    payload = wire_payload[_AUTH_TAG_BYTES:]
    expected = hmac.new(key, payload, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise PermissionError("invalid socket frame authentication tag")
    return payload


def _send_framed(sock: socket.socket, payload: bytes, *, auth_key: bytes | str | None = None) -> None:
    wire_payload = _pack_authenticated(payload, auth_key)
    if not wire_payload or len(wire_payload) > _MAX_FRAME_BYTES:
        raise ValueError("socket frame length is outside the allowed range")
    sock.sendall(_LEN.pack(len(wire_payload)) + wire_payload)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise ``ConnectionError`` on early close."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"socket closed with {remaining} of {n} bytes unread")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_framed(sock: socket.socket, *, auth_key: bytes | str | None = None) -> bytes:
    (length,) = _LEN.unpack(_recv_exactly(sock, _LEN.size))
    if length <= 0 or length > _MAX_FRAME_BYTES:
        raise ValueError("socket frame length is outside the allowed range")
    return _unpack_authenticated(_recv_exactly(sock, length), auth_key)


def _worker(
    host: str,
    port: int,
    worker_id: int,
    belief: np.ndarray,
    out: dict[int, np.ndarray],
    auth_key: bytes | None,
    round_id: str,
    aggregation_config_hash: str,
    timeout: float,
    errors: dict[int, Exception],
) -> None:
    """One worker: connect, send its framed belief, receive the framed consensus."""
    try:
        authentication = "hmac-sha256" if auth_key is not None else "none"
        belief_payload = serialize_belief(belief)
        envelope = serialize_envelope(
            belief_payload,
            message_type="belief",
            round_id=round_id,
            worker_id=worker_id,
            aggregation_config_hash=aggregation_config_hash,
            authentication=authentication,
        )
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            _send_framed(sock, envelope, auth_key=auth_key)
            result_envelope, result_payload = deserialize_envelope(_recv_framed(sock, auth_key=auth_key))
            if (
                result_envelope.message_type != "result"
                or result_envelope.round_id != round_id
                or result_envelope.aggregation_config_hash != aggregation_config_hash
                or result_envelope.authentication != authentication
                or result_envelope.worker_id is not None
            ):
                raise ValueError("consensus protocol envelope does not match the round")
            result = deserialize_result(result_payload)
        out[worker_id] = result["consensus"]
    except Exception as exc:
        errors[worker_id] = exc


def _validate_socket_replay(
    replay: list[dict[str, Any]],
    local_posteriors: list[np.ndarray] | np.ndarray | None,
    consensus: np.ndarray,
    *,
    robustness: float | None = None,
    config: AggregationConfig | None = None,
) -> bool:
    """Validate that a socket replay log reconstructs the reported consensus.

    The replay log intentionally stores event metadata and payload digests rather
    than raw beliefs. Validation therefore recomputes the aggregation from the
    caller-provided beliefs using the worker order captured by the server.
    """
    if not isinstance(replay, list) or not replay or any(not isinstance(event, dict) for event in replay):
        return False
    if config is not None and not isinstance(config, AggregationConfig):
        return False
    if config is not None and robustness is not None:
        return False
    resolved_config = config or AggregationConfig(
        method="robust",
        robustness=0.0 if robustness is None else robustness,
        max_iter=32,
    )
    if local_posteriors is None:
        return False
    belief_list = [
        np.asarray(posterior, dtype=np.float64)
        for posterior in local_posteriors
    ]
    n_workers = len(belief_list)
    expected_event_order = (
        ["server_listen"]
        + ["belief_received"] * n_workers
        + ["aggregate"]
        + ["consensus_broadcast"] * n_workers
    )
    if [event.get("event") for event in replay] != expected_event_order:
        return False
    if any(
        event.get("event") not in _REPLAY_FIELDS
        or set(event) != _REPLAY_FIELDS[str(event.get("event"))]
        for event in replay
    ):
        return False
    listen_events = [event for event in replay if event.get("event") == "server_listen"]
    if len(listen_events) != 1:
        return False
    listen = listen_events[0]
    round_id = listen.get("round_id")
    authenticated = listen.get("authenticated")
    port = listen.get("port")
    if (
        listen.get("protocol_version") != PROTOCOL_VERSION
        or listen.get("aggregation_config_hash") != resolved_config.fingerprint
        or listen.get("n_workers") != len(belief_list)
        or not isinstance(listen.get("host"), str)
        or not str(listen.get("host")).strip()
        or isinstance(port, bool)
        or not isinstance(port, int)
        or not 0 < port <= 65535
        or not isinstance(round_id, str)
        or not round_id.strip()
        or not isinstance(authenticated, bool)
    ):
        return False
    authentication = "hmac-sha256" if authenticated else "none"
    if listen.get("authentication") != authentication:
        return False
    aggregate_events = [event for event in replay if event.get("event") == "aggregate"]
    if len(aggregate_events) != 1:
        return False
    worker_order = aggregate_events[0].get("worker_order")
    if (
        not isinstance(worker_order, list)
        or any(isinstance(worker_id, bool) or not isinstance(worker_id, int) for worker_id in worker_order)
        or sorted(worker_order) != list(range(len(belief_list)))
    ):
        return False
    consensus_digest = aggregate_events[0].get("consensus_sha256")
    if consensus_digest != _payload_digest(serialize_belief(np.asarray(consensus, dtype=np.float64))):
        return False
    if (
        aggregate_events[0].get("aggregation_config_hash") != resolved_config.fingerprint
        or aggregate_events[0].get("aggregation_config") != resolved_config.as_dict()
        or aggregate_events[0].get("protocol_version") != PROTOCOL_VERSION
        or aggregate_events[0].get("round_id") != round_id
        or aggregate_events[0].get("authentication") != authentication
    ):
        return False
    received_rows = [event for event in replay if event.get("event") == "belief_received"]
    broadcast_rows = [event for event in replay if event.get("event") == "consensus_broadcast"]
    received_ids = {event.get("worker_id") for event in received_rows}
    broadcast_ids = {event.get("worker_id") for event in broadcast_rows}
    expected_ids = set(range(len(belief_list)))
    if (
        len(received_rows) != len(expected_ids)
        or len(broadcast_rows) != len(expected_ids)
        or received_ids != expected_ids
        or broadcast_ids != expected_ids
    ):
        return False
    received_events = {
        int(event["worker_id"]): event
        for event in replay
        if event.get("event") == "belief_received" and event.get("worker_id") in expected_ids
    }
    for worker_id, belief in enumerate(belief_list):
        event = received_events.get(worker_id)
        if event is None:
            return False
        belief_payload = serialize_belief(belief)
        frame = serialize_envelope(
            belief_payload,
            message_type="belief",
            round_id=round_id,
            worker_id=worker_id,
            aggregation_config_hash=resolved_config.fingerprint,
            authentication=authentication,
        )
        if (
            event.get("protocol_version") != PROTOCOL_VERSION
            or event.get("round_id") != round_id
            or event.get("aggregation_config_hash") != resolved_config.fingerprint
            or event.get("authentication") != authentication
            or event.get("belief_sha256") != _payload_digest(belief_payload)
        ):
            return False
        if event.get("frame_sha256") != _payload_digest(frame):
            return False
        if event.get("frame_bytes") != len(frame):
            return False
    ordered = [belief_list[int(worker_id)] for worker_id in worker_order]
    reference_result = aggregate_result(ordered, config=resolved_config)
    reference = reference_result.consensus
    result_payload = serialize_result(
        reference_result.consensus,
        reference_result.normalized_effective_weights,
    )
    result_envelope = serialize_envelope(
        result_payload,
        message_type="result",
        round_id=round_id,
        worker_id=None,
        aggregation_config_hash=resolved_config.fingerprint,
        authentication=authentication,
    )
    for event in broadcast_rows:
        if (
            event.get("protocol_version") != PROTOCOL_VERSION
            or event.get("round_id") != round_id
            or event.get("aggregation_config_hash") != resolved_config.fingerprint
            or event.get("authentication") != authentication
            or event.get("payload_sha256") != _payload_digest(result_payload)
            or event.get("payload_bytes") != len(result_payload)
            or event.get("envelope_sha256") != _payload_digest(result_envelope)
            or event.get("envelope_bytes") != len(result_envelope)
        ):
            return False
    return bool(np.array_equal(reference, np.asarray(consensus, dtype=np.float64)))


def validate_socket_replay(
    replay: list[dict[str, Any]],
    local_posteriors: list[np.ndarray] | np.ndarray | None = None,
    consensus: np.ndarray | None = None,
    *,
    robustness: float | None = None,
    config: AggregationConfig | None = None,
    **legacy: object,
) -> bool:
    """Validate a digest-only replay, returning ``False`` for malformed input."""
    if "beliefs" in legacy:
        if local_posteriors is not None:
            return False
        local_posteriors = legacy.pop("beliefs")  # type: ignore[assignment]
        warnings.warn(
            "beliefs is deprecated; use local_posteriors",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy or local_posteriors is None or consensus is None:
        return False
    try:
        return _validate_socket_replay(
            replay,
            local_posteriors,
            consensus,
            robustness=robustness,
            config=config,
        )
    except (
        IndexError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return False


def save_socket_replay(path: str | PathLike[str], replay: list[dict[str, Any]]) -> Path:
    """Persist a digest-only socket replay log as deterministic JSON."""
    replay_path = Path(path)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=replay_path.parent,
            prefix=f".{replay_path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(
                replay,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, replay_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return replay_path


def load_socket_replay(path: str | PathLike[str]) -> list[dict[str, Any]]:
    """Load a persisted digest-only socket replay log."""
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant is not allowed: {value}")

    try:
        loaded = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid socket replay file: {path}") from exc
    if not isinstance(loaded, list) or not all(isinstance(event, dict) for event in loaded):
        raise ValueError("socket replay file must contain a list of event objects")
    return loaded


def run_socket_round(
    local_posteriors: list[np.ndarray] | np.ndarray | None = None,
    *,
    robustness: float | None = None,
    config: AggregationConfig | None = None,
    round_id: str = "round-0",
    host: str = "127.0.0.1",
    timeout: float = 10.0,
    auth_key: bytes | str | None = None,
    replay_path: str | PathLike[str] | None = None,
    replay_guard: ReplayGuard | None = None,
    **legacy: object,
) -> dict[str, Any]:
    """Run one federation round over real loopback TCP sockets.

    Spins up a server socket on an ephemeral port, launches one worker thread per
    belief (each opening its own TCP connection), collects all beliefs, fuses
    them with the configured public aggregation dispatcher, and broadcasts the
    serialized consensus back to every worker.

    ``auth_key`` enables per-frame HMAC-SHA256 authentication with a pre-shared
    key. It protects the loopback frame integrity test from accidental or
    adversarial payload mutation, but it is not TLS, key exchange, or cross-host
    deployment security.

    ``replay_path`` persists the digest-only replay log to deterministic JSON
    after the round completes. The file still omits raw beliefs; validation
    verifies the stored belief, frame, consensus, and broadcast digests against
    caller-provided beliefs and consensus.

    ``replay_guard`` optionally rejects reuse of a round id across calls.
    :class:`ReplayGuard` scopes that claim to one process;
    :class:`PersistentReplayGuard` persists it across local process restarts.
    Neither choice establishes a multi-host replay domain.

    Returns ``consensus`` (the fused pmf), ``in_process`` (the reference
    configured in-process consensus), ``bit_identical`` (``np.array_equal`` of the
    two), ``n_workers``, each worker's received consensus for a broadcast check,
    and a digest-only ``replay`` log that can reconstruct and digest-verify the
    round.
    """
    if "beliefs" in legacy:
        if local_posteriors is not None:
            raise TypeError(
                "local_posteriors and deprecated beliefs cannot both be supplied"
            )
        local_posteriors = legacy.pop("beliefs")  # type: ignore[assignment]
        warnings.warn(
            "beliefs is deprecated; use local_posteriors",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(legacy))}")
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    belief_list = [
        np.asarray(posterior, dtype=np.float64)
        for posterior in local_posteriors
    ]
    n = len(belief_list)
    if n == 0:
        raise ValueError("local_posteriors must be non-empty")
    if config is not None and not isinstance(config, AggregationConfig):
        raise ValueError("config must be an AggregationConfig or None")
    if config is not None and robustness is not None:
        raise ValueError("config and compatibility robustness are mutually exclusive")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float, np.integer, np.floating))
        or not np.isfinite(timeout)
        or timeout <= 0.0
    ):
        raise ValueError("timeout must be finite and positive")
    resolved_config = config or AggregationConfig(
        method="robust",
        robustness=0.0 if robustness is None else robustness,
        max_iter=32,
    )
    if not isinstance(round_id, str) or not round_id.strip():
        raise ValueError("round_id must be a non-empty string")
    host = _validate_loopback_host(host)
    key = _coerce_auth_key(auth_key)
    if replay_guard is not None:
        if not isinstance(replay_guard, ReplayGuard):
            raise ValueError("replay_guard must be a ReplayGuard")
        replay_guard.claim(round_id)
    authentication = "hmac-sha256" if key is not None else "none"
    replay: list[dict[str, Any]] = []

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, 0))
    server.listen(n)
    server.settimeout(timeout)
    port = server.getsockname()[1]
    replay.append(
        {
            "event": "server_listen",
            "host": host,
            "port": int(port),
            "n_workers": n,
            "authenticated": key is not None,
            "authentication": authentication,
            "protocol_version": PROTOCOL_VERSION,
            "round_id": round_id,
            "aggregation_config_hash": resolved_config.fingerprint,
        }
    )

    received: dict[int, np.ndarray] = {}
    worker_errors: dict[int, Exception] = {}
    threads = [
        threading.Thread(
            target=_worker,
            args=(
                host,
                port,
                wid,
                belief_list[wid],
                received,
                key,
                round_id,
                resolved_config.fingerprint,
                timeout,
                worker_errors,
            ),
        )
        for wid in range(n)
    ]
    for t in threads:
        t.start()

    # Server side: accept n connections, read each framed (worker_id + belief).
    conns: list[tuple[int, socket.socket]] = []
    beliefs_by_id: dict[int, np.ndarray] = {}
    try:
        for _ in range(n):
            conn, _addr = server.accept()
            adopted = False
            try:
                conn.settimeout(timeout)
                frame = _recv_framed(conn, auth_key=key)
                envelope, belief_payload = deserialize_envelope(frame)
                worker_id = envelope.worker_id
                if (
                    envelope.message_type != "belief"
                    or envelope.round_id != round_id
                    or envelope.aggregation_config_hash != resolved_config.fingerprint
                    or envelope.authentication != authentication
                    or worker_id is None
                ):
                    raise ValueError("belief protocol envelope does not match the round")
                if worker_id in beliefs_by_id:
                    raise ValueError(f"duplicate worker id in socket round: {worker_id}")
                beliefs_by_id[worker_id] = deserialize_belief(belief_payload)
                replay.append(
                    {
                        "event": "belief_received",
                        "worker_id": int(worker_id),
                        "protocol_version": envelope.protocol_version,
                        "round_id": envelope.round_id,
                        "aggregation_config_hash": envelope.aggregation_config_hash,
                        "authentication": envelope.authentication,
                        "frame_sha256": _payload_digest(frame),
                        "belief_sha256": _payload_digest(belief_payload),
                        "frame_bytes": len(frame),
                    }
                )
                conns.append((worker_id, conn))
                adopted = True
            finally:
                if not adopted:
                    conn.close()
        expected_worker_ids = set(range(n))
        if set(beliefs_by_id) != expected_worker_ids:
            raise ValueError(f"socket round worker ids must be exactly {sorted(expected_worker_ids)}")
        worker_order = sorted(beliefs_by_id)
        ordered = [beliefs_by_id[wid] for wid in worker_order]
        result = aggregate_result(ordered, config=resolved_config)
        replay.append(
            {
                "event": "aggregate",
                "worker_order": [int(wid) for wid in worker_order],
                "method": resolved_config.method,
                "robustness": float(resolved_config.robustness),
                "aggregation_config": resolved_config.as_dict(),
                "aggregation_config_hash": resolved_config.fingerprint,
                "protocol_version": PROTOCOL_VERSION,
                "round_id": round_id,
                "authentication": authentication,
                "consensus_sha256": _payload_digest(serialize_belief(result.consensus)),
            }
        )
        result_payload = serialize_result(
            result.consensus, result.normalized_effective_weights
        )
        payload = serialize_envelope(
            result_payload,
            message_type="result",
            round_id=round_id,
            worker_id=None,
            aggregation_config_hash=resolved_config.fingerprint,
            authentication=authentication,
        )
        for wid, conn in conns:
            _send_framed(conn, payload, auth_key=key)
            replay.append(
                {
                    "event": "consensus_broadcast",
                    "worker_id": int(wid),
                    "protocol_version": PROTOCOL_VERSION,
                    "round_id": round_id,
                    "aggregation_config_hash": resolved_config.fingerprint,
                    "authentication": authentication,
                    "payload_sha256": _payload_digest(result_payload),
                    "payload_bytes": len(result_payload),
                    "envelope_sha256": _payload_digest(payload),
                    "envelope_bytes": len(payload),
                }
            )
    finally:
        for _wid, conn in conns:
            conn.close()
        server.close()
        for t in threads:
            t.join(timeout=timeout)

    alive = [index for index, thread in enumerate(threads) if thread.is_alive()]
    if alive:
        raise TimeoutError(f"socket worker threads did not finish: {alive}")
    if worker_errors:
        worker_id = min(worker_errors)
        error = worker_errors[worker_id]
        raise RuntimeError(f"socket worker {worker_id} failed: {type(error).__name__}: {error}") from error
    if set(received) != set(range(n)):
        raise RuntimeError("not every socket worker received the consensus")
    reference = aggregate_result(belief_list, config=resolved_config).consensus
    replay_file = save_socket_replay(replay_path, replay) if replay_path is not None else None
    return {
        "consensus": result.consensus,
        "in_process": reference,
        "bit_identical": bool(np.array_equal(result.consensus, reference)),
        "worker_consensuses": received,
        "n_workers": n,
        "port": int(port),
        "authenticated": key is not None,
        "protocol_version": PROTOCOL_VERSION,
        "round_id": round_id,
        "aggregation_config": resolved_config.as_dict(),
        "aggregation_config_hash": resolved_config.fingerprint,
        "replay": replay,
        "replay_path": str(replay_file) if replay_file is not None else None,
        "replay_valid": validate_socket_replay(
            replay,
            belief_list,
            result.consensus,
            config=resolved_config,
        ),
    }


__all__ = [
    "PersistentReplayGuard",
    "ReplayGuard",
    "load_socket_replay",
    "run_socket_round",
    "save_socket_replay",
    "validate_socket_replay",
]
