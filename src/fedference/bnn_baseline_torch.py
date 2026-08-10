"""FedGVI neural classification baseline — PyTorch implementation.

This module provides a deterministic point-estimate MLP trained with a
recentered density-power beta-loss (Basu et al., 1998; Fujisawa & Eguchi,
2008) federated objective, mirroring the FedGVI client
per-example loss of :mod:`fedference.bnn_baseline` (which uses numpy-based
logistic regression with rcce/nll) but in the neural-network setting.

Architecture: Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, output_dim)
-> Softmax. All parameters are plain ``torch.nn.Parameter`` — there is NO
variational distribution over weights and no sampling: in GVI terms this is
the point-mass (Dirac) variational family.

Beta-loss objective (the same recentered categorical density-power loss as
:func:`fedference.losses.beta_loss`):

    L_beta(p, y) = -(p_y^beta - 1) / beta
                   + (sum_k p_k^(beta+1) - 1) / (beta+1)

where ``y`` is one-hot and ``p`` is the softmax output. At ``beta -> 0``
this approaches the standard cross-entropy (NLL), recovering the non-robust
Friston regime. The finite-beta loss is bounded on the categorical simplex
and is the implementation used by the NumPy client-side experiments.

Note on the (absent) KL term: a genuine mean-field ELBO would include a
KL(q||prior) term over a weight distribution. With a point-mass family there
is no such term — the loss is exactly the per-sample beta-loss, keeping this
baseline minimal and directly comparable to the numpy bnn_baseline. The beta-loss itself already
provides robustness to contaminated clients — which is the load-bearing
claim — independently of the prior regularizer.

Federated round: each client trains a fresh point-mass MLP from a random init, produces
predictions on a shared probe point (the grand-mean of all client data), and
the server calls :func:`fedference.aggregation.robust_aggregate` on the
per-client softmax predictions to form the consensus belief.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fedference.aggregation import robust_aggregate

# ---- Module-level defaults (surfaced as manuscript tokens) ------------------
# Values live in the torch-free fedference.bnn_defaults module so the
# manuscript-variable generator (which must not import torch) consumes the
# same constants; re-exported here for backward compatibility.
from fedference.bnn_defaults import (
    BNN_BETA_DEFAULT,
    BNN_HIDDEN_DIM_DEFAULT,
    BNN_N_CLIENTS_DEFAULT,
    BNN_N_STEPS_DEFAULT,
    BNN_ROBUSTNESS_DEFAULT,
)
from fedference.torch_bnn import (
    configure_torch_determinism,
    resolve_torch_device,
)

TEMPERED_ENTROPY_WEIGHT_DEFAULT: float = 1.0  # mirror from aggregation

ArrayF = np.ndarray
_EPS = 1e-12


def _integer_control(
    value: object,
    *,
    name: str,
    minimum: int,
) -> int:
    """Return one strict integer control, excluding boolean coercion."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def _real_control(
    value: object,
    *,
    name: str,
    lower: float,
    strictly_greater: bool,
) -> float:
    """Return one finite real control without accepting strings or booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        relation = "positive" if strictly_greater and lower == 0.0 else "non-negative"
        raise ValueError(f"{name} must be finite and {relation}")
    result = float(value)
    invalid_bound = result <= lower if strictly_greater else result < lower
    if not np.isfinite(result) or invalid_bound:
        relation = "positive" if strictly_greater and lower == 0.0 else "non-negative"
        raise ValueError(f"{name} must be finite and {relation}")
    return result


def _validate_client_data(client_data: object) -> tuple[int, int]:
    """Validate real, finite, shape-compatible client classification shards."""
    if not isinstance(client_data, (list, tuple)) or not client_data:
        raise ValueError("client_data must be a non-empty sequence")
    dimensions: tuple[int, int] | None = None
    for client_id, item in enumerate(client_data):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"client {client_id} data must be an (x, y_onehot) pair")
        x_raw = np.asarray(item[0])
        y_raw = np.asarray(item[1])
        for name, values in (("features", x_raw), ("labels", y_raw)):
            if (
                np.issubdtype(values.dtype, np.bool_)
                or not np.issubdtype(values.dtype, np.number)
                or np.issubdtype(values.dtype, np.complexfloating)
            ):
                raise ValueError(f"client {client_id} {name} must be real numeric values")
        x = np.asarray(x_raw, dtype=np.float64)
        y = np.asarray(y_raw, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] == 0:
            raise ValueError(f"client {client_id} arrays must be non-empty matrices")
        if x.shape[0] != y.shape[0] or x.shape[1] == 0 or y.shape[1] < 2:
            raise ValueError(f"client {client_id} feature and label shapes are incompatible")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError(f"client {client_id} arrays must be finite")
        if np.any((y != 0.0) & (y != 1.0)) or not np.all(y.sum(axis=1) == 1.0):
            raise ValueError(f"client {client_id} labels must be one-hot encoded")
        client_dimensions = (x.shape[1], y.shape[1])
        if dimensions is None:
            dimensions = client_dimensions
        elif client_dimensions != dimensions:
            raise ValueError("all client shards must share feature and class dimensions")
    assert dimensions is not None
    return dimensions


class PointMassMLP(nn.Module):
    """Deterministic point-estimate MLP with a beta-loss FedGVI objective.

    Architecture: Linear(input_dim, hidden_dim) -> ReLU ->
    Linear(hidden_dim, output_dim) -> Softmax.

    Parameters are plain ``torch.nn.Parameter`` instances — a point-mass (Dirac)
    variational family: there is no distribution over weights and no sampling. No
    KL term is included in the loss (see module docstring). The genuine
    mean-field variational family (Gaussian q(w) with a KL term) lives in
    :class:`fedference.bnn_variational_torch.VariationalMLP`, which recovers this
    class exactly as its posterior variance goes to zero.

    Args:
        input_dim: number of input features.
        hidden_dim: number of hidden units.
        output_dim: number of output classes.
        seed: private CPU-generator seed used for parameter initialization.
        beta: non-negative beta-loss exponent. At ``beta = 0`` the loss is
            evaluated as NLL; small positive beta values use the same stable
            NLL branch. Default: ``BNN_BETA_DEFAULT``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seed: int = 0,
        beta: float = BNN_BETA_DEFAULT,
    ) -> None:
        super().__init__()
        input_dim = _integer_control(input_dim, name="input_dim", minimum=1)
        hidden_dim = _integer_control(hidden_dim, name="hidden_dim", minimum=1)
        output_dim = _integer_control(output_dim, name="output_dim", minimum=1)
        seed = _integer_control(seed, name="seed", minimum=0)
        self._gen = torch.Generator().manual_seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.beta = _real_control(
            beta,
            name="beta",
            lower=0.0,
            strictly_greater=False,
        )

        # Linear layer 1: input_dim -> hidden_dim
        self._W1 = nn.Parameter(torch.randn(hidden_dim, input_dim, generator=self._gen) * 0.1)
        self._b1 = nn.Parameter(torch.zeros(hidden_dim))
        # Linear layer 2: hidden_dim -> output_dim
        self._W2 = nn.Parameter(torch.randn(output_dim, hidden_dim, generator=self._gen) * 0.1)
        self._b2 = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns softmax probabilities.

        Args:
            x: input tensor of shape ``(n_samples, input_dim)``.

        Returns:
            Probability tensor of shape ``(n_samples, output_dim)`` summing
            to 1 along ``dim=1``.
        """
        h = torch.relu(x @ self._W1.T + self._b1)
        logits = h @ self._W2.T + self._b2
        return torch.softmax(logits, dim=1)

    def beta_loss(
        self,
        probs: torch.Tensor,
        y_onehot: torch.Tensor,
        beta: float | None = None,
    ) -> torch.Tensor:
        """Per-sample recentered density-power beta-loss.

        .. math::

            L_\\beta(p, y) = -\\frac{p_y^\\beta - 1}{\\beta}
                + \\frac{\\sum_k p_k^{\\beta+1} - 1}{\\beta+1}

        This is the same loss as :func:`fedference.losses.beta_loss`. At
        ``beta = 0`` (and for sufficiently small positive beta) it is
        evaluated as cross-entropy (NLL), avoiding the removable singularity.

        Args:
            probs: softmax probabilities, shape ``(n_samples, output_dim)``.
            y_onehot: one-hot labels, shape ``(n_samples, output_dim)``.
            beta: exponent; defaults to ``self.beta``.

        Returns:
            Per-sample loss tensor, shape ``(n_samples,)``.
        """
        b = (
            _real_control(
                beta,
                name="beta",
                lower=0.0,
                strictly_greater=False,
            )
            if beta is not None
            else self.beta
        )
        # Clamp to avoid log(0); the clamp is below the precision relevant to
        # the returned loss while keeping gradients finite at the boundary.
        p = torch.clamp(probs, min=_EPS)
        p_true = (p * y_onehot).sum(dim=1)
        if b < 1e-8:
            return -torch.log(p_true)
        data_term = -(p_true**b - 1.0) / b
        norm_term = (p.pow(b + 1.0).sum(dim=1) - 1.0) / (b + 1.0)
        return data_term + norm_term

    def fit(
        self,
        x: torch.Tensor,
        y_onehot: torch.Tensor,
        n_steps: int = BNN_N_STEPS_DEFAULT,
        lr: float = 0.01,
    ) -> list[float]:
        """Train using Adam on the mean beta-loss.

        Note: no KL term is added — see module docstring for rationale.

        Args:
            x: input tensor, shape ``(n_samples, input_dim)``.
            y_onehot: one-hot labels, shape ``(n_samples, output_dim)``.
            n_steps: number of gradient steps.
            lr: Adam learning rate.

        Returns:
            List of mean loss values, one per step (length ``n_steps``).
        """
        n_steps = _integer_control(n_steps, name="n_steps", minimum=0)
        lr = _real_control(lr, name="lr", lower=0.0, strictly_greater=True)
        device = self._W1.device
        x = x.to(device)
        y_onehot = y_onehot.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history: list[float] = []
        for _ in range(n_steps):
            optimizer.zero_grad()
            probs = self.forward(x)
            loss = self.beta_loss(probs, y_onehot).mean()
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        return history

    def predict_proba(self, x: torch.Tensor) -> ArrayF:
        """Return softmax probabilities as a numpy array.

        Args:
            x: input tensor, shape ``(n_samples, input_dim)``.

        Returns:
            NumPy array of shape ``(n_samples, output_dim)`` with rows
            summing to 1.
        """
        x = x.to(self._W1.device)
        with torch.no_grad():
            probs = self.forward(x)
        return probs.detach().cpu().numpy()


def federated_bnn_round(
    client_data: list[tuple[ArrayF, ArrayF]],
    seed: int = 0,
    hidden_dim: int = BNN_HIDDEN_DIM_DEFAULT,
    n_steps: int = BNN_N_STEPS_DEFAULT,
    robustness: float = BNN_ROBUSTNESS_DEFAULT,
    beta: float = BNN_BETA_DEFAULT,
    device: str = "cpu",
) -> dict:
    """One round of federated point-mass MLP training.

    Each client independently trains a fresh :class:`PointMassMLP` from a
    seeded random initialization. After training, every client predicts on a
    shared probe point (the grand-mean of all client inputs). The server fuses
    the per-client belief vectors with
    :func:`fedference.aggregation.robust_aggregate`.

    Args:
        client_data: list of ``(x_np, y_onehot_np)`` tuples — float32 arrays
            of shapes ``(n_i, d)`` and ``(n_i, k)`` respectively.
        seed: base RNG seed; client ``n`` uses ``seed + n`` for its model init.
        hidden_dim: hidden layer width for every client model.
        n_steps: gradient steps per client.
        robustness: robustness parameter passed to ``robust_aggregate``.
        beta: beta-loss exponent for every client model.

    Returns:
        Dict with keys:

        * ``"consensus"`` — numpy array, the server consensus belief (sums to 1).
        * ``"client_predictions"`` — list of per-client softmax vectors on
          the probe point.
        * ``"n_clients"`` — number of clients.
        * ``"robustness"`` — the robustness value used.
        * ``"beta"`` — the beta value used.
    """
    seed = _integer_control(seed, name="seed", minimum=0)
    hidden_dim = _integer_control(hidden_dim, name="hidden_dim", minimum=1)
    n_steps = _integer_control(n_steps, name="n_steps", minimum=0)
    robustness = _real_control(
        robustness,
        name="robustness",
        lower=0.0,
        strictly_greater=False,
    )
    beta = _real_control(beta, name="beta", lower=0.0, strictly_greater=False)
    input_dim, output_dim = _validate_client_data(client_data)
    configure_torch_determinism(seed)
    torch_device, device_receipt = resolve_torch_device(device)

    n_clients = len(client_data)

    # Shared probe point: grand-mean of all client inputs.
    all_x = np.concatenate([x for x, _ in client_data], axis=0)
    probe_np = all_x.mean(axis=0, keepdims=True).astype(np.float32)  # (1, d)
    probe = torch.from_numpy(probe_np).to(torch_device)

    client_predictions: list[ArrayF] = []
    for n, (x_np, y_np) in enumerate(client_data):
        bnn = PointMassMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seed=seed + n,
            beta=beta,
        ).to(torch_device)
        x_t = torch.from_numpy(x_np.astype(np.float32)).to(torch_device)
        y_t = torch.from_numpy(y_np.astype(np.float32)).to(torch_device)
        bnn.fit(x_t, y_t, n_steps=n_steps)
        pred = bnn.predict_proba(probe)[0]  # shape (output_dim,)
        client_predictions.append(pred)

    result = robust_aggregate(client_predictions, robustness=robustness)
    return {
        "consensus": result.consensus,
        "client_predictions": client_predictions,
        "n_clients": n_clients,
        "robustness": robustness,
        "beta": beta,
        "device": device_receipt.resolved,
        "device_receipt": {
            "requested": device_receipt.requested,
            "resolved": device_receipt.resolved,
            "torch_version": device_receipt.torch_version,
            "mps_available": device_receipt.mps_available,
            "deterministic_algorithms": device_receipt.deterministic_algorithms,
            "fallback": device_receipt.fallback,
        },
    }


# ---- Executed experiment: held-out accuracy vs contamination -----------------

#: Standard client uses a small beta (approaches cross-entropy / NLL); the robust
#: client uses ``BNN_BETA_DEFAULT``. Kept module-level so the manuscript tokens and
#: the experiment share one source of truth.
BNN_STANDARD_BETA: float = 0.05
_BNN_TORCH_CONTAMINATION_LEVELS: tuple[float, ...] = (0.0, 0.2, 0.4)


def _consensus_accuracy(
    client_data: list[tuple[ArrayF, ArrayF]],
    x_test: ArrayF,
    y_test: ArrayF,
    *,
    seed: int,
    hidden_dim: int,
    n_steps: int,
    robustness: float,
    beta: float,
    device: torch.device,
) -> tuple[float, float]:
    """Train a point-mass MLP per shard, fuse per-test-point predictions, score accuracy.

    Every client trains on its own (contaminated) shard, predicts softmax
    probabilities on the shared clean test set, and the server fuses the
    per-client predictions at each test point with
    :func:`fedference.aggregation.robust_aggregate`. Returns
    ``(accuracy, max_simplex_deviation)`` where the second value is the largest
    ``|sum(consensus) - 1|`` observed over the test set (a validity check).
    """
    output_dim = int(client_data[0][1].shape[1])
    input_dim = int(client_data[0][0].shape[1])
    # Per-client predictions on the full test set: list of (n_test, k) arrays.
    per_client: list[ArrayF] = []
    for n, (x_np, y_np) in enumerate(client_data):
        bnn = PointMassMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seed=seed + n,
            beta=beta,
        ).to(device)
        bnn.fit(
            torch.from_numpy(x_np.astype(np.float32)).to(device),
            torch.from_numpy(y_np.astype(np.float32)).to(device),
            n_steps=n_steps,
        )
        per_client.append(bnn.predict_proba(torch.from_numpy(x_test.astype(np.float32)).to(device)))
    n_test = x_test.shape[0]
    correct = 0
    max_dev = 0.0
    for i in range(n_test):
        point_preds = [pc[i] for pc in per_client]
        consensus = robust_aggregate(point_preds, robustness=robustness).consensus
        max_dev = max(max_dev, abs(float(consensus.sum()) - 1.0))
        if int(np.argmax(consensus)) == int(y_test[i]):
            correct += 1
    return correct / n_test, max_dev


def run_bnn_torch_experiment(
    seed: int = 0,
    *,
    n_clients: int = BNN_N_CLIENTS_DEFAULT,
    n_per: int = 80,
    hidden_dim: int = BNN_HIDDEN_DIM_DEFAULT,
    n_steps: int = BNN_N_STEPS_DEFAULT,
    robustness: float = BNN_ROBUSTNESS_DEFAULT,
    beta: float = BNN_BETA_DEFAULT,
    contamination_levels: tuple[float, ...] = _BNN_TORCH_CONTAMINATION_LEVELS,
    device: str = "cpu",
) -> dict:
    """Run the PyTorch point-mass MLP FedGVI complement end-to-end.

    Mirrors the numpy logistic-regression baseline
    (:func:`fedference.bnn_baseline.fed_gvi_logreg`) in the neural-network
    setting: a federated colony of deterministic point-mass MLPs is trained under
    per-client label contamination, and the server fuses their predictions with
    :func:`fedference.aggregation.robust_aggregate`. Every number returned is an
    executed result — the report the manuscript tokens read.

    Returns a JSON-serializable dict with per-config held-out accuracy across the
    contamination grid, a consensus-simplex validity check, and a determinism
    check (two identical seeded runs produce bit-identical consensus accuracy).
    """
    from fedference.bnn_baseline import contaminate, make_blobs

    seed = _integer_control(seed, name="seed", minimum=0)
    n_clients = _integer_control(n_clients, name="n_clients", minimum=1)
    n_per = _integer_control(n_per, name="n_per", minimum=1)
    hidden_dim = _integer_control(hidden_dim, name="hidden_dim", minimum=1)
    n_steps = _integer_control(n_steps, name="n_steps", minimum=0)
    robustness = _real_control(
        robustness,
        name="robustness",
        lower=0.0,
        strictly_greater=False,
    )
    beta = _real_control(beta, name="beta", lower=0.0, strictly_greater=False)
    if not isinstance(contamination_levels, (tuple, list)) or not contamination_levels:
        raise ValueError("contamination_levels must be a non-empty sequence")
    levels = [
        _real_control(
            contamination,
            name="contamination level",
            lower=0.0,
            strictly_greater=False,
        )
        for contamination in contamination_levels
    ]
    if any(contamination > 1.0 for contamination in levels):
        raise ValueError("contamination levels must be at most 1")
    configure_torch_determinism(seed)
    torch_device, device_receipt = resolve_torch_device(device)

    def _build(contamination: float) -> tuple[list, ArrayF, ArrayF]:
        rng = np.random.default_rng(seed)
        clients: list[tuple[ArrayF, ArrayF]] = []
        for _ in range(n_clients):
            x, y_clean = make_blobs(n_per, rng=rng)
            y = contaminate(x, y_clean, contamination, rng=rng)
            clients.append((x.astype(np.float32), np.eye(2, dtype=np.float32)[y]))
        x_test, y_test = make_blobs(4 * n_per, rng=rng)
        return clients, x_test, y_test

    standard: list[float] = []
    robust: list[float] = []
    max_simplex_dev = 0.0
    for c in levels:
        clients, x_test, y_test = _build(c)
        acc_s, dev_s = _consensus_accuracy(
            clients,
            x_test,
            y_test,
            seed=seed,
            hidden_dim=hidden_dim,
            n_steps=n_steps,
            robustness=robustness,
            beta=BNN_STANDARD_BETA,
            device=torch_device,
        )
        acc_r, dev_r = _consensus_accuracy(
            clients,
            x_test,
            y_test,
            seed=seed,
            hidden_dim=hidden_dim,
            n_steps=n_steps,
            robustness=robustness,
            beta=beta,
            device=torch_device,
        )
        standard.append(acc_s)
        robust.append(acc_r)
        max_simplex_dev = max(max_simplex_dev, dev_s, dev_r)

    # Determinism: rerun the highest-contamination robust config; must match.
    clients, x_test, y_test = _build(levels[-1])
    acc_r2, _ = _consensus_accuracy(
        clients,
        x_test,
        y_test,
        seed=seed,
        hidden_dim=hidden_dim,
        n_steps=n_steps,
        robustness=robustness,
        beta=beta,
        device=torch_device,
    )
    deterministic = bool(abs(acc_r2 - robust[-1]) < 1e-12)

    return {
        "status": "ok",
        "torch_version": str(torch.__version__),
        "device": device_receipt.resolved,
        "device_receipt": {
            "requested": device_receipt.requested,
            "resolved": device_receipt.resolved,
            "mps_available": device_receipt.mps_available,
            "deterministic_algorithms": device_receipt.deterministic_algorithms,
            "fallback": device_receipt.fallback,
        },
        "contamination_levels": levels,
        "accuracy_by_config": {
            "beta->0 (standard)": standard,
            f"beta={beta} (robust)": robust,
        },
        "reported_contamination": float(levels[-1]),
        "standard_accuracy": float(standard[-1]),
        "robust_accuracy": float(robust[-1]),
        "consensus_max_simplex_deviation": float(max_simplex_dev),
        "deterministic": deterministic,
        "n_clients": int(n_clients),
        "hidden_dim": int(hidden_dim),
        "n_steps": int(n_steps),
        "beta": float(beta),
        "robustness": float(robustness),
        "seed": int(seed),
    }
