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
from collections.abc import Mapping
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .._validation import as_pmf_matrix
from ..aggregation import (
    AggregationConfig,
    AggregationMethod,
    AggregationResult,
    aggregate_result,
)
from .transport import (
    PROTOCOL_VERSION,
    _serialization_input_vector,
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


def _is_exact_int(value: object) -> bool:
    """Return whether *value* has the exact integer type emitted by JSON.

    Replay metadata is a typed protocol surface, so values that merely compare
    equal to integers (notably booleans and integral floats) must not satisfy
    integer fields. Requiring ``type(value) is int`` also rejects strings and
    unhashable containers before worker identifiers reach set construction.
    """

    return type(value) is int


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


ReplayFindingCategory = Literal["integrity", "solver"]


@dataclass(frozen=True)
class ReplayFinding:
    """One stable, machine-readable socket replay finding."""

    category: ReplayFindingCategory
    code: str
    message: str

    def __post_init__(self) -> None:
        if self.category not in ("integrity", "solver"):
            raise ValueError("replay finding category must be 'integrity' or 'solver'")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("replay finding code must be non-empty")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("replay finding message must be non-empty")

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-compatible finding representation."""
        return {
            "category": self.category,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ReplayValidationResult:
    """Integrity and solver-health verdict for a digest-only socket replay."""

    integrity_valid: bool
    findings: tuple[ReplayFinding, ...]
    recorded_config: AggregationConfig | None = None
    aggregation: AggregationResult | None = None

    def __post_init__(self) -> None:
        findings = tuple(self.findings)
        if any(not isinstance(finding, ReplayFinding) for finding in findings):
            raise ValueError("findings must contain ReplayFinding values")
        expected_valid = not any(
            finding.category == "integrity" for finding in findings
        )
        if self.integrity_valid is not expected_valid:
            raise ValueError("integrity_valid must agree with integrity findings")
        object.__setattr__(self, "findings", findings)

    @property
    def solver_status(self) -> str | None:
        """Return the recomputed aggregation status, when integrity reached it."""
        if self.aggregation is None:
            return None
        status = getattr(self.aggregation, "solver_status", None)
        if isinstance(status, str):
            return status
        if self.aggregation.converged:
            return (
                "converged_with_fallback"
                if self.aggregation.fallback_events
                else "nominal"
            )
        return (
            "not_converged_with_fallback"
            if self.aggregation.fallback_events
            else "not_converged"
        )

    @property
    def nominal_solver(self) -> bool:
        """Whether replayed aggregation used the nominal converged path."""
        return self.solver_status == "nominal"

    def as_dict(self) -> dict[str, Any]:
        """Return the stable JSON-facing replay verdict."""
        config = self.recorded_config
        return {
            "integrity_valid": self.integrity_valid,
            "solver_status": self.solver_status,
            "nominal_solver": self.nominal_solver,
            "aggregation_config": None if config is None else config.as_dict(),
            "aggregation_config_hash": None if config is None else config.fingerprint,
            "findings": [finding.as_dict() for finding in self.findings],
        }


def _replay_failure(
    code: str,
    message: str,
    *,
    recorded_config: AggregationConfig | None = None,
    aggregation: AggregationResult | None = None,
) -> ReplayValidationResult:
    return ReplayValidationResult(
        integrity_valid=False,
        findings=(ReplayFinding("integrity", code, message),),
        recorded_config=recorded_config,
        aggregation=aggregation,
    )


def aggregation_config_from_replay(
    replay: list[dict[str, Any]],
) -> AggregationConfig:
    """Recover and strictly validate the configuration recorded by a replay."""
    if not isinstance(replay, list) or not replay:
        raise ValueError("socket replay must be a non-empty event list")
    aggregate_events = [
        event
        for event in replay
        if isinstance(event, dict) and event.get("event") == "aggregate"
    ]
    if len(aggregate_events) != 1:
        raise ValueError("socket replay must contain exactly one aggregate event")
    aggregate_event = aggregate_events[0]
    if set(aggregate_event) != _REPLAY_FIELDS["aggregate"]:
        raise ValueError("socket replay aggregate fields do not match schema")
    raw = aggregate_event.get("aggregation_config")
    expected_fields = {
        "method",
        "robustness",
        "entropy_weight",
        "max_iter",
        "tol",
        "multistart",
    }
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise ValueError("recorded aggregation configuration does not match schema")
    config = AggregationConfig(
        method=raw["method"],
        robustness=raw["robustness"],
        entropy_weight=raw["entropy_weight"],
        max_iter=raw["max_iter"],
        tol=raw["tol"],
        multistart=raw["multistart"],
    )
    recorded_robustness = aggregate_event.get("robustness")
    if (
        aggregate_event.get("method") != config.method
        or isinstance(recorded_robustness, bool)
        or not isinstance(recorded_robustness, (int, float))
        or not np.isfinite(recorded_robustness)
        or float(recorded_robustness) != config.robustness
        or aggregate_event.get("aggregation_config_hash") != config.fingerprint
    ):
        raise ValueError("recorded aggregation configuration bindings disagree")
    return config


def _assert_recorded_config(
    recorded: AggregationConfig,
    *,
    robustness: float | None,
    config: AggregationConfig | None,
    config_assertions: Mapping[str, object] | None,
) -> ReplayValidationResult | None:
    if config is not None and not isinstance(config, AggregationConfig):
        raise ValueError("config must be an AggregationConfig or None")
    supplied_modes = sum(
        value is not None for value in (robustness, config, config_assertions)
    )
    if supplied_modes > 1:
        raise ValueError(
            "robustness, config, and config_assertions are mutually exclusive"
        )
    expected: AggregationConfig | None = config
    if robustness is not None:
        expected = AggregationConfig(
            method="robust",
            robustness=robustness,
            max_iter=32,
        )
    if expected is not None and expected.fingerprint != recorded.fingerprint:
        return _replay_failure(
            "configuration.mismatch",
            "supplied aggregation configuration does not match the recorded configuration",
            recorded_config=recorded,
        )
    if config_assertions is not None:
        if not isinstance(config_assertions, Mapping):
            raise ValueError("config_assertions must be a mapping or None")
        unknown = set(config_assertions) - set(recorded.as_dict())
        if unknown:
            raise ValueError(
                "unknown aggregation configuration assertion(s): "
                + ", ".join(sorted(unknown))
            )
        candidate_data: dict[str, object] = dict(recorded.as_dict())
        candidate_data.update(dict(config_assertions))
        candidate = AggregationConfig(
            method=cast(AggregationMethod, candidate_data["method"]),
            robustness=cast(float, candidate_data["robustness"]),
            entropy_weight=cast(float, candidate_data["entropy_weight"]),
            max_iter=cast(int, candidate_data["max_iter"]),
            tol=cast(float, candidate_data["tol"]),
            multistart=cast(bool, candidate_data["multistart"]),
        )
        mismatched = [
            name
            for name in config_assertions
            if candidate.as_dict()[name] != recorded.as_dict()[name]
        ]
        if mismatched:
            return _replay_failure(
                "configuration.mismatch",
                "supplied aggregation assertion(s) do not match the replay: "
                + ", ".join(sorted(mismatched)),
                recorded_config=recorded,
            )
    return None


def inspect_socket_replay(
    replay: list[dict[str, Any]],
    local_posteriors: list[np.ndarray] | np.ndarray | None = None,
    consensus: np.ndarray | None = None,
    *,
    robustness: float | None = None,
    config: AggregationConfig | None = None,
    config_assertions: Mapping[str, object] | None = None,
    **legacy: object,
) -> ReplayValidationResult:
    """Inspect replay integrity and report solver health separately.

    With no configuration assertion, the validated configuration embedded in
    the replay is authoritative.  ``config`` supplies a complete assertion;
    ``config_assertions`` supplies CLI-friendly partial assertions.  The
    compatibility ``robustness`` keyword asserts the historical robust socket
    configuration with ``max_iter=32``.
    """
    if "beliefs" in legacy:
        if local_posteriors is not None:
            return _replay_failure(
                "input.malformed",
                "local_posteriors and deprecated beliefs cannot both be supplied",
            )
        local_posteriors = legacy.pop("beliefs")  # type: ignore[assignment]
        warnings.warn(
            "beliefs is deprecated; use local_posteriors",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        return _replay_failure(
            "input.malformed",
            "unexpected replay validation argument(s): "
            + ", ".join(sorted(legacy)),
        )
    if replay == []:
        return _replay_failure("replay.empty", "socket replay is empty")
    if not isinstance(replay, list) or any(
        not isinstance(event, dict) for event in replay
    ):
        return _replay_failure(
            "replay.malformed",
            "socket replay must be a non-empty list of event objects",
        )
    if local_posteriors is None or consensus is None:
        return _replay_failure(
            "input.malformed",
            "local_posteriors and consensus are required",
        )
    try:
        raw_local_posteriors = list(local_posteriors)
        as_pmf_matrix(raw_local_posteriors, name="local_posteriors")
        belief_list = [
            np.asarray(posterior, dtype=np.float64)
            for posterior in raw_local_posteriors
        ]
        reported_consensus = _serialization_input_vector(
            consensus,
            name="consensus",
            require_unit_sum=True,
        )
    except (TypeError, ValueError) as exc:
        return _replay_failure(
            "input.malformed",
            f"belief or consensus arrays are invalid: {exc}",
        )
    if (
        not belief_list
        or any(
            belief.ndim != 1
            or belief.size == 0
            or not np.all(np.isfinite(belief))
            for belief in belief_list
        )
    ):
        return _replay_failure(
            "input.malformed",
            "beliefs and consensus must be non-empty finite vectors",
        )
    n_workers = len(belief_list)
    expected_event_order = (
        ["server_listen"]
        + ["belief_received"] * n_workers
        + ["aggregate"]
        + ["consensus_broadcast"] * n_workers
    )
    if [event.get("event") for event in replay] != expected_event_order:
        return _replay_failure(
            "event.order_mismatch",
            "socket replay event order or event count does not match the worker count",
        )
    if any(
        event.get("event") not in _REPLAY_FIELDS
        or set(event) != _REPLAY_FIELDS[str(event.get("event"))]
        for event in replay
    ):
        return _replay_failure(
            "event.fields_mismatch",
            "one or more replay event fields do not match the protocol-v1 schema",
        )
    try:
        recorded_config = aggregation_config_from_replay(replay)
    except (TypeError, ValueError) as exc:
        return _replay_failure("configuration.mismatch", str(exc))
    assertion_failure = _assert_recorded_config(
        recorded_config,
        robustness=robustness,
        config=config,
        config_assertions=config_assertions,
    )
    if assertion_failure is not None:
        return assertion_failure

    listen = replay[0]
    listen_protocol_version = listen.get("protocol_version")
    if (
        not _is_exact_int(listen_protocol_version)
        or listen_protocol_version != PROTOCOL_VERSION
    ):
        return _replay_failure(
            "protocol.mismatch",
            f"recorded protocol version is not {PROTOCOL_VERSION}",
            recorded_config=recorded_config,
        )
    round_id = listen.get("round_id")
    if not isinstance(round_id, str) or not round_id.strip():
        return _replay_failure(
            "round.mismatch",
            "recorded round id is invalid",
            recorded_config=recorded_config,
        )
    authenticated = listen.get("authenticated")
    host = listen.get("host")
    port = listen.get("port")
    if (
        not _is_exact_int(listen.get("n_workers"))
        or listen.get("n_workers") != n_workers
        or not isinstance(host, str)
        or not host.strip()
        or isinstance(port, bool)
        or not isinstance(port, int)
        or not 0 < port <= 65535
        or not isinstance(authenticated, bool)
    ):
        return _replay_failure(
            "event.fields_mismatch",
            "server_listen metadata is invalid",
            recorded_config=recorded_config,
        )
    try:
        recorded_host = ipaddress.ip_address(host)
    except ValueError:
        return _replay_failure(
            "event.fields_mismatch",
            "server_listen host is not a numeric IPv4 loopback address",
            recorded_config=recorded_config,
        )
    if (
        not isinstance(recorded_host, ipaddress.IPv4Address)
        or not recorded_host.is_loopback
        or host != str(recorded_host)
    ):
        return _replay_failure(
            "event.fields_mismatch",
            "server_listen host is not the canonical IPv4 loopback address",
            recorded_config=recorded_config,
        )
    authentication = "hmac-sha256" if authenticated else "none"
    if listen.get("authentication") != authentication:
        return _replay_failure(
            "event.fields_mismatch",
            "server authentication metadata is inconsistent",
            recorded_config=recorded_config,
        )
    if listen.get("aggregation_config_hash") != recorded_config.fingerprint:
        return _replay_failure(
            "configuration.mismatch",
            "server configuration fingerprint does not match the recorded configuration",
            recorded_config=recorded_config,
        )

    aggregate_event = replay[n_workers + 1]
    aggregate_protocol_version = aggregate_event.get("protocol_version")
    if (
        not _is_exact_int(aggregate_protocol_version)
        or aggregate_protocol_version != PROTOCOL_VERSION
    ):
        return _replay_failure(
            "protocol.mismatch",
            "aggregate event protocol version is inconsistent",
            recorded_config=recorded_config,
        )
    if aggregate_event.get("round_id") != round_id:
        return _replay_failure(
            "round.mismatch",
            "aggregate event round id is inconsistent",
            recorded_config=recorded_config,
        )
    if aggregate_event.get("authentication") != authentication:
        return _replay_failure(
            "event.fields_mismatch",
            "aggregate event authentication metadata is inconsistent",
            recorded_config=recorded_config,
        )
    worker_order = aggregate_event.get("worker_order")
    if (
        not isinstance(worker_order, list)
        or any(
            not _is_exact_int(worker_id)
            for worker_id in worker_order
        )
        or worker_order != list(range(n_workers))
    ):
        return _replay_failure(
            "worker.order_mismatch",
            "aggregate worker order is not the exact ordered worker-id set",
            recorded_config=recorded_config,
        )
    if (
        aggregate_event.get("aggregation_config_hash")
        != recorded_config.fingerprint
        or aggregate_event.get("aggregation_config") != recorded_config.as_dict()
        or aggregate_event.get("method") != recorded_config.method
        or aggregate_event.get("robustness") != recorded_config.robustness
    ):
        return _replay_failure(
            "configuration.mismatch",
            "aggregate event configuration bindings are inconsistent",
            recorded_config=recorded_config,
        )
    reported_payload = serialize_belief(reported_consensus)
    if aggregate_event.get("consensus_sha256") != _payload_digest(reported_payload):
        return _replay_failure(
            "payload.mismatch",
            "reported consensus digest does not match the aggregate event",
            recorded_config=recorded_config,
        )

    received_rows = replay[1 : n_workers + 1]
    broadcast_rows = replay[n_workers + 2 :]
    received_worker_ids = [event.get("worker_id") for event in received_rows]
    broadcast_worker_ids = [event.get("worker_id") for event in broadcast_rows]
    expected_ids = set(range(n_workers))
    if (
        len(received_rows) != n_workers
        or len(broadcast_rows) != n_workers
        or any(not _is_exact_int(worker_id) for worker_id in received_worker_ids)
        or any(not _is_exact_int(worker_id) for worker_id in broadcast_worker_ids)
    ):
        return _replay_failure(
            "worker.order_mismatch",
            "replay worker ids must be exact non-Boolean integers",
            recorded_config=recorded_config,
        )
    received_ids = set(received_worker_ids)
    broadcast_ids = set(broadcast_worker_ids)
    if received_ids != expected_ids or broadcast_ids != expected_ids:
        return _replay_failure(
            "worker.order_mismatch",
            "replay does not contain exactly one receive and broadcast per worker",
            recorded_config=recorded_config,
        )
    received_events = {
        int(event["worker_id"]): event
        for event in received_rows
        if event.get("worker_id") in expected_ids
    }
    for worker_id, belief in enumerate(belief_list):
        event = received_events[worker_id]
        event_protocol_version = event.get("protocol_version")
        if (
            not _is_exact_int(event_protocol_version)
            or event_protocol_version != PROTOCOL_VERSION
        ):
            return _replay_failure(
                "protocol.mismatch",
                f"belief event for worker {worker_id} has the wrong protocol version",
                recorded_config=recorded_config,
            )
        if event.get("round_id") != round_id:
            return _replay_failure(
                "round.mismatch",
                f"belief event for worker {worker_id} has the wrong round id",
                recorded_config=recorded_config,
            )
        if event.get("aggregation_config_hash") != recorded_config.fingerprint:
            return _replay_failure(
                "configuration.mismatch",
                f"belief event for worker {worker_id} has the wrong configuration hash",
                recorded_config=recorded_config,
            )
        if event.get("authentication") != authentication:
            return _replay_failure(
                "event.fields_mismatch",
                f"belief event for worker {worker_id} has inconsistent authentication metadata",
                recorded_config=recorded_config,
            )
        belief_payload = serialize_belief(belief)
        if event.get("belief_sha256") != _payload_digest(belief_payload):
            return _replay_failure(
                "belief.mismatch",
                f"belief digest mismatch for worker {worker_id}",
                recorded_config=recorded_config,
            )
        frame = serialize_envelope(
            belief_payload,
            message_type="belief",
            round_id=round_id,
            worker_id=worker_id,
            aggregation_config_hash=recorded_config.fingerprint,
            authentication=authentication,
        )
        frame_bytes = event.get("frame_bytes")
        if (
            event.get("frame_sha256") != _payload_digest(frame)
            or not _is_exact_int(frame_bytes)
            or frame_bytes != len(frame)
        ):
            return _replay_failure(
                "frame.mismatch",
                f"belief frame digest or size mismatch for worker {worker_id}",
                recorded_config=recorded_config,
            )

    ordered = [belief_list[worker_id] for worker_id in worker_order]
    try:
        reference_result = aggregate_result(ordered, config=recorded_config)
    except (IndexError, OverflowError, TypeError, ValueError) as exc:
        return _replay_failure(
            "input.malformed",
            f"beliefs cannot be recomputed under the recorded configuration: {exc}",
            recorded_config=recorded_config,
        )
    result_payload = serialize_result(
        reference_result.consensus,
        reference_result.normalized_effective_weights,
    )
    result_envelope = serialize_envelope(
        result_payload,
        message_type="result",
        round_id=round_id,
        worker_id=None,
        aggregation_config_hash=recorded_config.fingerprint,
        authentication=authentication,
    )
    for event in broadcast_rows:
        worker_id = int(event["worker_id"])
        event_protocol_version = event.get("protocol_version")
        if (
            not _is_exact_int(event_protocol_version)
            or event_protocol_version != PROTOCOL_VERSION
        ):
            return _replay_failure(
                "protocol.mismatch",
                f"broadcast event for worker {worker_id} has the wrong protocol version",
                recorded_config=recorded_config,
                aggregation=reference_result,
            )
        if event.get("round_id") != round_id:
            return _replay_failure(
                "round.mismatch",
                f"broadcast event for worker {worker_id} has the wrong round id",
                recorded_config=recorded_config,
                aggregation=reference_result,
            )
        if event.get("aggregation_config_hash") != recorded_config.fingerprint:
            return _replay_failure(
                "configuration.mismatch",
                f"broadcast event for worker {worker_id} has the wrong configuration hash",
                recorded_config=recorded_config,
                aggregation=reference_result,
            )
        if event.get("authentication") != authentication:
            return _replay_failure(
                "event.fields_mismatch",
                f"broadcast event for worker {worker_id} has inconsistent authentication metadata",
                recorded_config=recorded_config,
                aggregation=reference_result,
            )
        payload_bytes = event.get("payload_bytes")
        if (
            event.get("payload_sha256") != _payload_digest(result_payload)
            or not _is_exact_int(payload_bytes)
            or payload_bytes != len(result_payload)
        ):
            return _replay_failure(
                "payload.mismatch",
                f"result payload digest or size mismatch for worker {worker_id}",
                recorded_config=recorded_config,
                aggregation=reference_result,
            )
        envelope_bytes = event.get("envelope_bytes")
        if (
            event.get("envelope_sha256") != _payload_digest(result_envelope)
            or not _is_exact_int(envelope_bytes)
            or envelope_bytes != len(result_envelope)
        ):
            return _replay_failure(
                "envelope.mismatch",
                f"result envelope digest or size mismatch for worker {worker_id}",
                recorded_config=recorded_config,
                aggregation=reference_result,
            )
    if not np.array_equal(reference_result.consensus, reported_consensus):
        return _replay_failure(
            "consensus.recomputed_mismatch",
            "reported consensus is not bit-identical to the recomputed consensus",
            recorded_config=recorded_config,
            aggregation=reference_result,
        )
    findings: list[ReplayFinding] = []
    if not reference_result.converged:
        findings.append(
            ReplayFinding(
                "solver",
                "solver.nonconvergence",
                "recomputed aggregation did not converge within its recorded budget",
            )
        )
    if reference_result.fallback_events:
        findings.append(
            ReplayFinding(
                "solver",
                "solver.fallback",
                "recomputed aggregation used one or more numerical fallbacks",
            )
        )
    return ReplayValidationResult(
        integrity_valid=True,
        findings=tuple(findings),
        recorded_config=recorded_config,
        aggregation=reference_result,
    )


def _validate_socket_replay(
    replay: list[dict[str, Any]],
    local_posteriors: list[np.ndarray] | np.ndarray | None,
    consensus: np.ndarray,
    *,
    robustness: float | None = None,
    config: AggregationConfig | None = None,
) -> bool:
    """Internal Boolean compatibility projection of :func:`inspect_socket_replay`."""
    return inspect_socket_replay(
        replay,
        local_posteriors,
        consensus,
        robustness=robustness,
        config=config,
    ).integrity_valid


def validate_socket_replay(
    replay: list[dict[str, Any]],
    local_posteriors: list[np.ndarray] | np.ndarray | None = None,
    consensus: np.ndarray | None = None,
    *,
    robustness: float | None = None,
    config: AggregationConfig | None = None,
    **legacy: object,
) -> bool:
    """Validate replay integrity, returning ``False`` for malformed input.

    Solver nonconvergence and fallback are intentionally not integrity
    failures.  Use :func:`inspect_socket_replay` when numerical health must be
    enforced or reported.
    """
    if "beliefs" in legacy:
        if local_posteriors is not None:
            return False
        local_posteriors = legacy.pop("beliefs")  # type: ignore[assignment]
        warnings.warn(
            "beliefs is deprecated; use local_posteriors",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        return False
    try:
        return inspect_socket_replay(
            replay,
            local_posteriors,
            consensus,
            robustness=robustness,
            config=config,
        ).integrity_valid
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

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is not allowed: {key!r}")
            result[key] = value
        return result

    try:
        loaded = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid socket replay file: {path}") from exc
    if not isinstance(loaded, list) or not all(isinstance(event, dict) for event in loaded):
        raise ValueError("socket replay file must contain a list of event objects")
    return loaded


@dataclass(frozen=True)
class SocketRoundResult:
    """Complete local and transport diagnostics for one loopback round."""

    server_aggregation: AggregationResult
    in_process_aggregation: AggregationResult
    worker_consensuses: tuple[np.ndarray, ...]
    n_workers: int
    port: int
    authenticated: bool
    protocol_version: int
    round_id: str
    aggregation_config: AggregationConfig
    aggregation_config_hash: str
    replay: tuple[dict[str, Any], ...]
    replay_path: str | None
    replay_validation: ReplayValidationResult
    bit_identical: bool

    def __post_init__(self) -> None:
        if self.n_workers != len(self.worker_consensuses) or self.n_workers <= 0:
            raise ValueError("worker_consensuses must contain one result per worker")
        if not isinstance(self.aggregation_config, AggregationConfig):
            raise ValueError("aggregation_config must be an AggregationConfig")
        if self.aggregation_config_hash != self.aggregation_config.fingerprint:
            raise ValueError("aggregation_config_hash does not match aggregation_config")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("protocol_version does not match protocol v1")
        if not isinstance(self.authenticated, bool) or not isinstance(
            self.bit_identical, bool
        ):
            raise ValueError("authenticated and bit_identical must be booleans")
        frozen_workers: list[np.ndarray] = []
        for consensus in self.worker_consensuses:
            copied = np.asarray(consensus, dtype=np.float64).copy()
            if copied.ndim != 1 or copied.size == 0 or not np.all(np.isfinite(copied)):
                raise ValueError("worker consensuses must be non-empty finite vectors")
            copied.setflags(write=False)
            frozen_workers.append(copied)
        object.__setattr__(self, "worker_consensuses", tuple(frozen_workers))
        object.__setattr__(self, "replay", tuple(deepcopy(list(self.replay))))

    @property
    def consensus(self) -> np.ndarray:
        """Return the consensus broadcast by the socket server."""
        return self.server_aggregation.consensus

    @property
    def in_process(self) -> np.ndarray:
        """Return the matching in-process reference consensus."""
        return self.in_process_aggregation.consensus


def run_socket_round_result(
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
) -> SocketRoundResult:
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

    Returns complete server and in-process aggregation diagnostics, ordered
    worker consensuses, transport metadata, and a structured replay verdict.
    :func:`run_socket_round` retains the historical dictionary projection.
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
    raw_local_posteriors = list(local_posteriors)
    n = len(raw_local_posteriors)
    if n == 0:
        raise ValueError("local_posteriors must be non-empty")
    as_pmf_matrix(raw_local_posteriors, name="local_posteriors")
    # Preserve caller mass in protocol-v1 belief frames. The receiving server
    # performs the single canonical normalization through aggregate_result.
    belief_list = [
        np.asarray(posterior, dtype=np.float64)
        for posterior in raw_local_posteriors
    ]
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
    reference_result = aggregate_result(belief_list, config=resolved_config)
    replay_file = save_socket_replay(replay_path, replay) if replay_path is not None else None
    replay_validation = inspect_socket_replay(
        replay,
        belief_list,
        result.consensus,
        config=resolved_config,
    )
    return SocketRoundResult(
        server_aggregation=result,
        in_process_aggregation=reference_result,
        worker_consensuses=tuple(received[worker_id] for worker_id in range(n)),
        n_workers=n,
        port=int(port),
        authenticated=key is not None,
        protocol_version=PROTOCOL_VERSION,
        round_id=round_id,
        aggregation_config=resolved_config,
        aggregation_config_hash=resolved_config.fingerprint,
        replay=tuple(replay),
        replay_path=str(replay_file) if replay_file is not None else None,
        replay_validation=replay_validation,
        bit_identical=bool(
            np.array_equal(result.consensus, reference_result.consensus)
        ),
    )


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
    """Run one loopback round and return the protocol-v1 compatibility mapping."""
    rich = run_socket_round_result(
        local_posteriors,
        robustness=robustness,
        config=config,
        round_id=round_id,
        host=host,
        timeout=timeout,
        auth_key=auth_key,
        replay_path=replay_path,
        replay_guard=replay_guard,
        **legacy,
    )
    return {
        "consensus": rich.consensus,
        "in_process": rich.in_process,
        "bit_identical": rich.bit_identical,
        "worker_consensuses": {
            worker_id: consensus
            for worker_id, consensus in enumerate(rich.worker_consensuses)
        },
        "n_workers": rich.n_workers,
        "port": rich.port,
        "authenticated": rich.authenticated,
        "protocol_version": rich.protocol_version,
        "round_id": rich.round_id,
        "aggregation_config": rich.aggregation_config.as_dict(),
        "aggregation_config_hash": rich.aggregation_config_hash,
        "replay": deepcopy(list(rich.replay)),
        "replay_path": rich.replay_path,
        "replay_valid": rich.replay_validation.integrity_valid,
    }


__all__ = [
    "PersistentReplayGuard",
    "ReplayFinding",
    "ReplayGuard",
    "ReplayValidationResult",
    "SocketRoundResult",
    "aggregation_config_from_replay",
    "inspect_socket_replay",
    "load_socket_replay",
    "run_socket_round",
    "run_socket_round_result",
    "save_socket_replay",
    "validate_socket_replay",
]
