"""Optional Torch device/profile contract with real CPU/MPS execution."""

from __future__ import annotations

import pytest
import torch

from fedference.bnn_variational_torch import VariationalMLP
from fedference.torch_bnn import (
    bnn_protocol_profile,
    configure_torch_determinism,
    resolve_torch_device,
)

pytestmark = pytest.mark.requires_torch


def test_device_resolution_never_silently_falls_back() -> None:
    device, receipt = resolve_torch_device("auto")
    assert device.type in ("cpu", "mps")
    assert receipt.resolved == device.type
    assert receipt.fallback is None
    if receipt.mps_available:
        mps, mps_receipt = resolve_torch_device("mps")
        assert mps.type == "mps"
        assert mps_receipt.fallback is None
    else:
        with pytest.raises(RuntimeError, match="unavailable"):
            resolve_torch_device("mps")
        cpu, fallback = resolve_torch_device("mps", allow_cpu_fallback=True)
        assert cpu.type == "cpu"
        assert fallback.fallback is not None
    with pytest.raises(ValueError, match="requested device"):
        resolve_torch_device(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="allow_cpu_fallback"):
        resolve_torch_device("cpu", allow_cpu_fallback=1)  # type: ignore[arg-type]


def test_variational_model_executes_on_resolved_backend() -> None:
    configure_torch_determinism(7)
    device, receipt = resolve_torch_device("auto")
    model = VariationalMLP(2, hidden_dim=3, output_dim=2, seed=7).to(device)
    inputs = torch.asarray([[0.1, 0.2], [0.3, -0.2]], dtype=torch.float32).to(device)
    prediction = model.predict_proba(inputs, n_samples=2)
    assert prediction.device.type == receipt.resolved
    assert prediction.shape == (2, 2)
    assert torch.allclose(
        prediction.sum(dim=1),
        torch.ones(2, device=device),
        atol=1e-5,
    )
    assert torch.isfinite(model.kl_to_prior())


def test_protocol_profiles_are_defensive_copies() -> None:
    profile = bnn_protocol_profile("source_5090")
    profile["executed_locally"] = True
    profile["seeds"].append(999)
    assert bnn_protocol_profile("source_5090")["executed_locally"] is False
    assert 999 not in bnn_protocol_profile("source_5090")["seeds"]
    with pytest.raises(KeyError, match="unknown BNN"):
        bnn_protocol_profile("missing")
    with pytest.raises(ValueError, match="non-empty"):
        bnn_protocol_profile("")  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", (True, -1, 1.5, "7"))
def test_determinism_seed_rejects_coercive_values(seed) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        configure_torch_determinism(seed)
