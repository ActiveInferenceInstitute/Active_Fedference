"""Optional PyTorch/MPS runtime contract for the FedGVI BNN lane.

Importing this module requires the ``torch`` package extra; the default
NumPy/SciPy core does not import it. Device fallback is always explicit and
receipt-bearing—an unavailable MPS request never silently becomes a CPU run.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from .research_registry import BNN_PROTOCOL_PROFILES


@dataclass(frozen=True)
class TorchDeviceReceipt:
    """Resolved device and any explicit portability fallback."""

    requested: str
    resolved: str
    torch_version: str
    mps_available: bool
    deterministic_algorithms: bool
    fallback: str | None


def configure_torch_determinism(seed: int) -> None:
    """Seed Torch and request deterministic algorithms where implemented."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_torch_device(
    requested: str = "cpu",
    *,
    allow_cpu_fallback: bool = False,
) -> tuple[torch.device, TorchDeviceReceipt]:
    """Resolve ``cpu``, ``mps``, or ``auto`` without a silent fallback."""
    if not isinstance(requested, str):
        raise ValueError("requested device must be 'cpu', 'mps', or 'auto'")
    if not isinstance(allow_cpu_fallback, bool):
        raise ValueError("allow_cpu_fallback must be a boolean")
    normalized = requested.strip().lower()
    if normalized not in ("cpu", "mps", "auto"):
        raise ValueError("requested device must be 'cpu', 'mps', or 'auto'")
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    fallback: str | None = None
    if normalized == "auto":
        resolved = "mps" if mps_available else "cpu"
    elif normalized == "mps" and not mps_available:
        if not allow_cpu_fallback:
            raise RuntimeError("MPS was requested but is unavailable")
        resolved = "cpu"
        fallback = "requested mps was unavailable; explicit CPU fallback enabled"
    else:
        resolved = normalized
    receipt = TorchDeviceReceipt(
        requested=normalized,
        resolved=resolved,
        torch_version=str(torch.__version__),
        mps_available=mps_available,
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        fallback=fallback,
    )
    return torch.device(resolved), receipt


def bnn_protocol_profile(name: str) -> dict[str, object]:
    """Return a defensive copy of one declared BNN execution profile."""
    if not isinstance(name, str) or not name:
        raise ValueError("profile name must be a non-empty string")
    try:
        return copy.deepcopy(BNN_PROTOCOL_PROFILES[name])
    except KeyError as exc:
        raise KeyError(f"unknown BNN protocol profile {name!r}") from exc


__all__ = [
    "TorchDeviceReceipt",
    "bnn_protocol_profile",
    "configure_torch_determinism",
    "resolve_torch_device",
]
