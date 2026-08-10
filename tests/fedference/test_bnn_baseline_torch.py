"""Tests for the PyTorch point-mass MLP FedGVI complement.

All tests use real tensors and real training loops (no mocks). Determinism
is enforced via the model's private CPU generator and a fixed ``seed``
argument to ``federated_bnn_round``.
"""

import pytest
import torch

pytestmark = [pytest.mark.requires_torch, pytest.mark.slow]


@pytest.mark.requires_torch
def test_point_mass_mlp_forward_valid_simplex():
    """forward() returns valid probability simplex (sums to 1)."""
    import numpy as np

    from fedference.bnn_baseline_torch import PointMassMLP

    bnn = PointMassMLP(input_dim=4, hidden_dim=8, output_dim=3, seed=42)
    x = torch.randn(10, 4)
    out = bnn.forward(x)
    assert out.shape == (10, 3)
    assert np.allclose(out.detach().numpy().sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.requires_torch
def test_point_mass_mlp_is_composable_module():
    """The point-mass complement exposes standard module parameters/state."""
    from fedference.bnn_baseline_torch import PointMassMLP

    bnn = PointMassMLP(input_dim=4, hidden_dim=8, output_dim=3, seed=42)
    assert isinstance(bnn, torch.nn.Module)
    assert len(tuple(bnn.parameters())) == 4
    assert set(bnn.state_dict()) == {"_W1", "_b1", "_W2", "_b2"}


@pytest.mark.requires_torch
def test_point_mass_beta_zero_recovers_nll():
    """The Torch complement obeys the same beta -> 0 limit as the NumPy loss."""
    from fedference.bnn_baseline_torch import PointMassMLP

    bnn = PointMassMLP(input_dim=2, hidden_dim=3, output_dim=3, seed=0, beta=0.0)
    probs = torch.tensor([[0.8, 0.15, 0.05], [0.1, 0.2, 0.7]])
    y = torch.eye(3)[torch.tensor([0, 2])]
    expected = -torch.log(torch.tensor([0.8, 0.7]))
    assert torch.allclose(bnn.beta_loss(probs, y), expected, atol=1e-7)
    assert torch.isfinite(bnn.beta_loss(probs, y, beta=1e-9)).all()


@pytest.mark.requires_torch
def test_point_mass_mlp_fit_decreases_loss():
    """Training loss decreases over 100 steps on a linearly-separable dataset."""
    from fedference.bnn_baseline_torch import PointMassMLP

    bnn = PointMassMLP(input_dim=2, hidden_dim=4, output_dim=2, seed=0)
    x = torch.cat([torch.randn(10, 2) + 2, torch.randn(10, 2) - 2])
    y = torch.zeros(20, 2)
    y[:10, 0] = 1.0
    y[10:, 1] = 1.0
    hist = bnn.fit(x, y, n_steps=100, lr=0.05)
    assert hist[-1] < hist[0], f"Loss did not decrease: {hist[0]:.4f} -> {hist[-1]:.4f}"


@pytest.mark.requires_torch
def test_federated_bnn_round_deterministic():
    """federated_bnn_round is bit-identical across two calls with the same seed."""
    import numpy as np

    from fedference.bnn_baseline_torch import federated_bnn_round

    rng = np.random.default_rng(0)
    n, d, k = 3, 4, 2
    data = [
        (
            rng.standard_normal((10, d)).astype("float32"),
            np.eye(k)[rng.integers(0, k, 10)].astype("float32"),
        )
        for _ in range(n)
    ]
    r1 = federated_bnn_round(data, seed=7, n_steps=20)
    r2 = federated_bnn_round(data, seed=7, n_steps=20)
    assert np.allclose(r1["consensus"], r2["consensus"], atol=1e-6)


@pytest.mark.requires_torch
def test_federated_bnn_consensus_valid_simplex():
    """federated_bnn_round consensus sums to 1 and has non-negative entries."""
    import numpy as np

    from fedference.bnn_baseline_torch import federated_bnn_round

    rng = np.random.default_rng(1)
    data = [
        (
            rng.standard_normal((8, 3)).astype("float32"),
            np.eye(2)[rng.integers(0, 2, 8)].astype("float32"),
        )
        for _ in range(4)
    ]
    r = federated_bnn_round(data, seed=0, n_steps=20)
    assert np.isclose(r["consensus"].sum(), 1.0, atol=1e-5)
    assert (r["consensus"] >= 0).all()


@pytest.mark.requires_torch
def test_federated_bnn_round_rejects_empty_client_data():
    """federated_bnn_round raises ValueError for an empty client_data list."""
    import pytest

    from fedference.bnn_baseline_torch import federated_bnn_round

    with pytest.raises(ValueError, match="client_data must be a non-empty sequence"):
        federated_bnn_round([], seed=0, n_steps=5)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("seed", True, "seed"),
        ("hidden_dim", 2.5, "hidden_dim"),
        ("n_steps", True, "n_steps"),
        ("robustness", "0.2", "robustness"),
        ("beta", False, "beta"),
    ),
)
def test_federated_bnn_round_rejects_coercive_controls(
    keyword,
    value,
    message,
):
    import numpy as np

    from fedference.bnn_baseline_torch import federated_bnn_round

    data = [(np.ones((2, 2), dtype=np.float32), np.eye(2, dtype=np.float32))]
    with pytest.raises(ValueError, match=message):
        federated_bnn_round(data, **{keyword: value})


def test_federated_bnn_round_rejects_malformed_client_shards():
    import numpy as np

    from fedference.bnn_baseline_torch import federated_bnn_round

    with pytest.raises(ValueError, match="one-hot"):
        federated_bnn_round(
            [(np.ones((2, 2)), np.asarray([[0.5, 0.5], [0.0, 1.0]]))],
        )
    with pytest.raises(ValueError, match="share feature and class dimensions"):
        federated_bnn_round(
            [
                (np.ones((2, 2)), np.eye(2)),
                (np.ones((2, 3)), np.eye(2)),
            ],
        )


@pytest.mark.requires_torch
def test_run_bnn_torch_experiment_is_executed_and_deterministic():
    """The end-to-end BNN complement returns executed, deterministic tokens."""
    from fedference.bnn_baseline_torch import run_bnn_torch_experiment

    # Small, fast configuration — still a real training + fusion run.
    report = run_bnn_torch_experiment(
        seed=0,
        n_per=30,
        hidden_dim=8,
        n_steps=25,
        contamination_levels=(0.0, 0.3),
    )
    assert report["status"] == "ok"
    # Accuracy curves are present, in [0, 1], one entry per contamination level.
    for curve in report["accuracy_by_config"].values():
        assert len(curve) == 2
        assert all(0.0 <= a <= 1.0 for a in curve)
    # Consensus is a valid simplex at every fused test point.
    assert report["consensus_max_simplex_deviation"] < 1e-6
    # Determinism: the reported flag is honest — a rerun matches bit-for-bit.
    assert report["deterministic"] is True
    rerun = run_bnn_torch_experiment(
        seed=0,
        n_per=30,
        hidden_dim=8,
        n_steps=25,
        contamination_levels=(0.0, 0.3),
    )
    assert rerun["robust_accuracy"] == report["robust_accuracy"]
    assert rerun["standard_accuracy"] == report["standard_accuracy"]


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("n_clients", True, "n_clients"),
        ("n_per", 0, "n_per"),
        ("contamination_levels", (), "contamination_levels"),
        ("contamination_levels", (0.0, 1.1), "at most 1"),
    ),
)
def test_run_bnn_torch_experiment_rejects_invalid_controls(
    keyword,
    value,
    message,
):
    from fedference.bnn_baseline_torch import run_bnn_torch_experiment

    with pytest.raises(ValueError, match=message):
        run_bnn_torch_experiment(**{keyword: value})
