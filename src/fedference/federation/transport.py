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
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from numbers import Real
from typing import Literal

import numpy as np

PROTOCOL_VERSION = 1
_ENVELOPE_HEADER_LENGTH = struct.Struct(">I")
_MAX_ENVELOPE_HEADER_BYTES = 64 * 1024
MessageType = Literal["belief", "result"]


class _ProtocolV1ZipInfo(zipfile.ZipInfo):
    """Keep protocol-v1 NPZ headers identical across supported Python versions.

    ``numpy.savez`` requests ZIP64 headers even for small members.  CPython
    3.10 writes actual 32-bit sizes into those local headers, while 3.11+
    writes the ZIP64 sentinel values and version 4.5.  Protocol v1 already
    records the latter bytes, so normalize the 3.10 writer to that existing
    representation without changing member names, array bytes, or framing.
    """

    def FileHeader(self, zip64: bool | None = None) -> bytes:  # noqa: N802
        if zip64:
            self.extract_version = max(zipfile.ZIP64_VERSION, self.extract_version)
            self.create_version = max(zipfile.ZIP64_VERSION, self.create_version)
        header = bytearray(super().FileHeader(zip64))
        if zip64:
            struct.pack_into("<H", header, 4, zipfile.ZIP64_VERSION)
            struct.pack_into("<L", header, 18, 0xFFFFFFFF)
            struct.pack_into("<L", header, 22, 0xFFFFFFFF)
        return bytes(header)


def _serialize_protocol_v1_npz(
    consensus: np.ndarray,
    agent_weights: np.ndarray,
) -> bytes:
    """Write the two fixed protocol-v1 NPZ members in canonical order."""
    buf = BytesIO()
    with zipfile.ZipFile(
        buf,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name, values in (
            ("consensus", consensus),
            ("agent_weights", agent_weights),
        ):
            info = _ProtocolV1ZipInfo(
                f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.create_system = 3
            info.external_attr = 0o600 << 16
            info.compress_type = zipfile.ZIP_STORED
            with archive.open(info, mode="w", force_zip64=True) as member:
                np.save(member, values, allow_pickle=False)
    return buf.getvalue()


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


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate member names."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate protocol envelope field: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    """Reject the non-standard NaN/Infinity spellings accepted by json.loads."""
    raise ValueError(f"non-finite protocol envelope JSON constant: {value}")


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
        raw = json.loads(
            data[_ENVELOPE_HEADER_LENGTH.size : header_end].decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
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
    if (
        type(raw["protocol_version"]) is not int
        or raw["protocol_version"] != PROTOCOL_VERSION
    ):
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


def _serialization_input_vector(
    values: object,
    *,
    name: str,
    require_unit_sum: bool,
) -> np.ndarray:
    """Validate a caller vector without normalizing or coercing its semantics."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric vector") from exc
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in np.asarray(values, dtype=object)
    ):
        raise ValueError(f"{name} must contain only numeric non-boolean values")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    total = float(result.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite mass")
    if require_unit_sum and not np.isclose(total, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"{name} must sum to one")
    return result


def serialize_belief(belief: np.ndarray) -> bytes:
    """Lossless numpy float64 serialization of a 1-D pmf belief array."""
    validated = _serialization_input_vector(
        belief,
        name="belief",
        require_unit_sum=False,
    )
    buf = BytesIO()
    np.save(buf, validated)
    return buf.getvalue()


def _transport_probability_vector(
    values: object,
    *,
    name: str,
    require_unit_sum: bool = True,
) -> np.ndarray:
    """Validate one exact float64 categorical vector on the wire."""
    if not isinstance(values, np.ndarray):
        raise ValueError(f"serialized {name} must be a NumPy array")
    if values.dtype.kind != "f" or values.dtype.itemsize != 8:
        raise ValueError(f"serialized {name} must use float64")
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"serialized {name} must be a non-empty vector")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"serialized {name} must be finite and non-negative")
    total = float(result.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"serialized {name} must have positive finite mass")
    if require_unit_sum and not np.isclose(total, 1.0, rtol=0.0, atol=1e-9):
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
    # Protocol-v1 belief frames preserve caller-supplied positive masses.  The
    # server's canonical aggregate_result boundary normalizes each row exactly
    # once; pre-normalizing here can perturb already normalized float64 bytes.
    return _transport_probability_vector(
        loaded,
        name="belief",
        require_unit_sum=False,
    )


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
    validated_consensus = _serialization_input_vector(
        consensus,
        name="result consensus",
        require_unit_sum=True,
    )
    validated_weights = _serialization_input_vector(
        normalized_effective_weights,
        name="result agent_weights",
        require_unit_sum=True,
    )
    return _serialize_protocol_v1_npz(validated_consensus, validated_weights)


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
