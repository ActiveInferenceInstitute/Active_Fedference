"""Federation transport layer for belief sharing.

This subpackage realizes the V3 federation transport: workers serialize their
categorical beliefs as lossless IEEE-754 float64 NumPy bytes, push them over a
queue to a :class:`FederationServer`, which fuses them through the same
configured aggregation interface used in process and broadcasts the consensus
back. Because serialization is a lossless float64 round trip, the federated
consensus is **bit-identical** to the matching configured in-process call —
federation is transport, not new math.

This retires the direct in-process serialization caveat: the transport is now
genuine at the queue boundary (threading-safe queues, byte-level serialization),
the optional process helper uses the same server/worker protocol inside one
machine, and the loopback-TCP helper adds versioned envelopes, bounded framing,
optional HMAC frame integrity, and persisted digest-verified replay validation.
A SQLite-backed replay guard can retain claimed round identifiers across local
process restarts. A separate transport adapter is required for true
multi-machine deployment; that evidence lane requires mTLS and physical
cross-host receipts.
"""

from __future__ import annotations

from fedference.federation.process import run_multiprocess_round
from fedference.federation.server import FederationServer
from fedference.federation.socket_transport import (
    PersistentReplayGuard,
    ReplayGuard,
    load_socket_replay,
    run_socket_round,
    save_socket_replay,
    validate_socket_replay,
)
from fedference.federation.transport import (
    PROTOCOL_VERSION,
    ProtocolEnvelope,
    deserialize_belief,
    deserialize_envelope,
    deserialize_result,
    serialize_belief,
    serialize_envelope,
    serialize_result,
)
from fedference.federation.worker import FederationWorker

__all__ = [
    "FederationServer",
    "FederationWorker",
    "PROTOCOL_VERSION",
    "PersistentReplayGuard",
    "ProtocolEnvelope",
    "ReplayGuard",
    "deserialize_envelope",
    "load_socket_replay",
    "run_multiprocess_round",
    "run_socket_round",
    "save_socket_replay",
    "validate_socket_replay",
    "serialize_belief",
    "serialize_envelope",
    "deserialize_belief",
    "serialize_result",
    "deserialize_result",
]
