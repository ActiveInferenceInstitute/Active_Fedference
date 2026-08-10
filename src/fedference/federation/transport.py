"""Lossless serialization for federation transport.

Beliefs and aggregation results are serialized via numpy's native ``.npy`` /
``.npz`` formats at float64 precision. The float64 round-trip is exact (IEEE-754
bit-preserving), which is what makes a federated consensus bit-identical to its
matching configured in-process aggregation call. ``allow_pickle=False`` on
every load keeps the wire format to plain numeric arrays—no arbitrary-object
deserialization.
"""

from __future__ import annotations

import hashlib
import json
import struct
import warnings
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Literal

import numpy as np

PROTOCOL_VERSION = 1
_ENVELOPE_HEADER_LENGTH = struct.Struct(">I")
_MAX_ENVELOPE_HEADER_BYTES = 64 * 1024
MessageType = Literal["belief", "result"]


@dataclass(frozen=True)
class ProtocolEnvelope:
    """Versioned metadata binding for a serialized federation payload."""

    protocol_version: int
    message_type: MessageType
    round_id: str
    worker_id: int | None
    aggregation_config_hash: str
    payload_sha256: str
    authentication: str


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def serialize_envelope(
    payload: bytes,
    *,
    message_type: MessageType,
    round_id: str,
    worker_id: int | None,
    aggregation_config_hash: str,
    authentication: str = "none",
) -> bytes:
    """Bind payload bytes to deterministic, versioned protocol metadata."""
    if message_type not in ("belief", "result"):
        raise ValueError("message_type must be 'belief' or 'result'")
    if not isinstance(round_id, str) or not round_id.strip():
        raise ValueError("round_id must be non-empty")
    if worker_id is not None and (
        isinstance(worker_id, bool) or not isinstance(worker_id, int) or worker_id < 0
    ):
        raise ValueError("worker_id must be a non-negative integer or None")
    if not _is_sha256(aggregation_config_hash):
        raise ValueError("aggregation_config_hash must be a SHA-256 hex digest")
    if not isinstance(authentication, str) or not authentication.strip():
        raise ValueError("authentication must be non-empty")
    envelope = ProtocolEnvelope(
        protocol_version=PROTOCOL_VERSION,
        message_type=message_type,
        round_id=round_id,
        worker_id=worker_id,
        aggregation_config_hash=aggregation_config_hash,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        authentication=authentication,
    )
    header = json.dumps(asdict(envelope), sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(header) > _MAX_ENVELOPE_HEADER_BYTES:
        raise ValueError("protocol envelope header is too large")
    return _ENVELOPE_HEADER_LENGTH.pack(len(header)) + header + payload


def deserialize_envelope(data: bytes) -> tuple[ProtocolEnvelope, bytes]:
    """Validate and unpack bytes produced by :func:`serialize_envelope`."""
    if len(data) < _ENVELOPE_HEADER_LENGTH.size:
        raise ValueError("protocol envelope is shorter than its header length")
    (header_length,) = _ENVELOPE_HEADER_LENGTH.unpack(data[: _ENVELOPE_HEADER_LENGTH.size])
    if header_length <= 0 or header_length > _MAX_ENVELOPE_HEADER_BYTES:
        raise ValueError("protocol envelope header length is invalid")
    header_end = _ENVELOPE_HEADER_LENGTH.size + header_length
    if len(data) < header_end:
        raise ValueError("protocol envelope header is truncated")
    try:
        raw = json.loads(data[_ENVELOPE_HEADER_LENGTH.size : header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("protocol envelope header is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("protocol envelope header must be an object")
    required = {
        "protocol_version",
        "message_type",
        "round_id",
        "worker_id",
        "aggregation_config_hash",
        "payload_sha256",
        "authentication",
    }
    if set(raw) != required:
        raise ValueError("protocol envelope header fields do not match schema")
    if raw["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported protocol_version {raw['protocol_version']!r}; expected {PROTOCOL_VERSION}"
        )
    if raw["message_type"] not in ("belief", "result"):
        raise ValueError("protocol envelope message_type is invalid")
    if not isinstance(raw["round_id"], str) or not raw["round_id"].strip():
        raise ValueError("protocol envelope round_id is invalid")
    worker_id = raw["worker_id"]
    if worker_id is not None and (
        isinstance(worker_id, bool) or not isinstance(worker_id, int) or worker_id < 0
    ):
        raise ValueError("protocol envelope worker_id is invalid")
    if not _is_sha256(raw["aggregation_config_hash"]):
        raise ValueError("protocol envelope aggregation_config_hash is invalid")
    if not _is_sha256(raw["payload_sha256"]):
        raise ValueError("protocol envelope payload_sha256 is invalid")
    if not isinstance(raw["authentication"], str) or not raw["authentication"].strip():
        raise ValueError("protocol envelope authentication is invalid")
    payload = data[header_end:]
    if hashlib.sha256(payload).hexdigest() != raw["payload_sha256"]:
        raise ValueError("protocol envelope payload digest mismatch")
    envelope = ProtocolEnvelope(
        protocol_version=raw["protocol_version"],
        message_type=raw["message_type"],
        round_id=raw["round_id"],
        worker_id=worker_id,
        aggregation_config_hash=raw["aggregation_config_hash"],
        payload_sha256=raw["payload_sha256"],
        authentication=raw["authentication"],
    )
    return envelope, payload


def serialize_belief(belief: np.ndarray) -> bytes:
    """Lossless numpy float64 serialization of a 1-D pmf belief array."""
    buf = BytesIO()
    np.save(buf, np.asarray(belief, dtype=np.float64))
    return buf.getvalue()


def _transport_probability_vector(
    values: object,
    *,
    name: str,
) -> np.ndarray:
    """Validate the exact float64 one-dimensional wire probability schema."""
    if not isinstance(values, np.ndarray):
        raise ValueError(f"serialized {name} must be a NumPy array")
    if values.dtype.kind != "f" or values.dtype.itemsize != 8:
        raise ValueError(f"serialized {name} must use float64")
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"serialized {name} must be a non-empty vector")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"serialized {name} must be finite and non-negative")
    if not np.isclose(float(result.sum()), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"serialized {name} must sum to one")
    return result


def deserialize_belief(data: bytes) -> np.ndarray:
    """Deserialize and validate one exact float64 categorical belief."""
    try:
        loaded = np.load(BytesIO(data), allow_pickle=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("serialized belief is not a valid NumPy array") from exc
    if not isinstance(loaded, np.ndarray):
        if hasattr(loaded, "close"):
            loaded.close()
        raise ValueError("serialized belief must be one NumPy array")
    return _transport_probability_vector(loaded, name="belief")


def serialize_result(
    consensus: np.ndarray,
    normalized_effective_weights: np.ndarray | None = None,
    **legacy: object,
) -> bytes:
    """Serialize consensus and normalized influence losslessly.

    The on-wire field remains ``agent_weights`` for protocol compatibility;
    callers use ``normalized_effective_weights`` in new code.
    """
    if "agent_weights" in legacy:
        if normalized_effective_weights is not None:
            raise TypeError(
                "normalized_effective_weights and deprecated agent_weights cannot both be supplied"
            )
        normalized_effective_weights = legacy.pop("agent_weights")  # type: ignore[assignment]
        warnings.warn(
            "agent_weights is deprecated; use normalized_effective_weights",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(legacy))}")
    if normalized_effective_weights is None:
        raise TypeError("normalized_effective_weights is required")
    buf = BytesIO()
    np.savez(
        buf,
        consensus=np.asarray(consensus, dtype=np.float64),
        # Preserve the version-1 federation wire key exactly.
        agent_weights=np.asarray(normalized_effective_weights, dtype=np.float64),
    )
    return buf.getvalue()


def deserialize_result(data: bytes) -> dict[str, np.ndarray]:
    """Deserialize and validate a result serialized by :func:`serialize_result`."""
    with np.load(BytesIO(data), allow_pickle=False) as npz:
        if set(npz.files) != {"consensus", "agent_weights"}:
            raise ValueError("serialized result fields do not match the transport schema")
        consensus = _transport_probability_vector(
            npz["consensus"],
            name="result consensus",
        )
        agent_weights = _transport_probability_vector(
            npz["agent_weights"],
            name="result agent_weights",
        )
    return {"consensus": consensus, "agent_weights": agent_weights}


__all__ = [
    "MessageType",
    "PROTOCOL_VERSION",
    "ProtocolEnvelope",
    "deserialize_belief",
    "deserialize_envelope",
    "deserialize_result",
    "serialize_belief",
    "serialize_envelope",
    "serialize_result",
]
