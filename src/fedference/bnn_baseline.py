"""FedGVI logistic-regression baseline — the small anchor experiment.

This module is a minimal, fully deterministic synthetic complement: under its
declared Gaussian-blob and label-flip mechanism, it compares a *robust*
generalized-variational client loss (Mildner et al., 2025, FedGVI,
arXiv:2502.00846) with a standard negative-log-likelihood (NLL) client. It is
not an external-data result, a deployment guarantee, or a reconstruction of the
FedGVI source protocol.

The setup is a federated 2-class logistic regression — the conjugate Bernoulli
analogue of the categorical generalized-Bayes update in
:mod:`fedference.generalized_bayes`. Each of ``n_clients`` clients owns a private
split of synthetic 2-D Gaussian-blob data, optionally with a fraction of its
labels flipped (``contamination``). A client runs a few gradient steps of the
chosen per-example loss and ships its weight vector to the server. The server
fuses the client weight factors by averaging their natural parameters. This is
a deliberately limited mean-field Gaussian / FedAvg weight-space analogue of
the project's log-linear pooling construction; it does not reconstruct
Friston et al. (2024) Eq. 7 or its complete message-passing protocol.

Loss / robustness mechanism (the load-bearing maths):

* ``loss='nll'`` — per-example NLL. Gradient w.r.t. the logit is the standard
  ``(p - y)``; a confidently *mislabelled* point produces a huge gradient and
  drags the estimator toward the wrong boundary.
* ``loss='rcce'`` — robust categorical cross-entropy / generalized cross-entropy
  ``L_{q_loss} = (1 - p_y^q_loss) / q_loss`` (Zhang & Sabuncu, 2018), the categorical loss in
  :func:`fedference.losses.rcce`. Its logit-gradient is the NLL gradient scaled
  by ``p_y^q_loss`` — the model's own confidence in the *given* label. A contaminated
  point (low ``p_y``) is therefore down-weighted by ``p_y^q_loss``, so flipped labels
  cannot dominate in this declared synthetic regime. As ``loss_param = q_loss
  -> 0``, the scale ``p_y^q_loss -> 1`` and RCCE collapses back to NLL
  (standard Bayes), the non-robust project client
  recovery limit.

The ``divergence`` argument (``'KLD'`` by default) names the regularizer that
pulls each client weight toward the shared prior, mirroring FedGVI's
``--server_div``; ``'KLD'`` gives the L2 (Gaussian-prior) shrinkage used here.
"""

from __future__ import annotations

import numpy as np

ArrayF = np.ndarray
_EPS = 1e-12


def _sigmoid(z: ArrayF) -> ArrayF:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def make_blobs(n_per: int, *, rng: np.random.Generator, separation: float = 2.2) -> tuple[ArrayF, ArrayF]:
    """Synthetic 2-D, 2-class Gaussian blobs.

    Returns ``(X, y)`` with ``X`` shape ``(2 * n_per, 2)`` and binary ``y``.
    Class 0 is centered at ``(-separation/2, 0)`` and class 1 at
    ``(+separation/2, 0)``; both unit-variance. Deterministic given ``rng``.
    """
    if n_per <= 0:
        raise ValueError("n_per must be positive")
    half = separation / 2.0
    c0 = rng.normal(loc=(-half, 0.0), scale=1.0, size=(n_per, 2))
    c1 = rng.normal(loc=(+half, 0.0), scale=1.0, size=(n_per, 2))
    x = np.vstack([c0, c1])
    y = np.concatenate([np.zeros(n_per, dtype=np.int64), np.ones(n_per, dtype=np.int64)])
    perm = rng.permutation(x.shape[0])
    return x[perm], y[perm]


def contaminate(x: ArrayF, y: ArrayF, fraction: float, *, rng: np.random.Generator) -> ArrayF:
    """Flip a ``fraction`` of binary labels, biased toward high-leverage outliers.

    A point is selected for flipping with probability proportional to its
    distance from the decision boundary (``|x_0|``), so the corruption
    concentrates on *confidently classifiable* points — the high-leverage
    contamination regime FedGVI's robust losses are designed to survive (a
    flipped outlier produces an enormous NLL gradient but a small RCCE one). The
    selection is without replacement and fully determined by ``rng``.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("contamination fraction must lie in [0, 1]")
    x_arr = np.asarray(x, dtype=np.float64)
    y_out = np.asarray(y, dtype=np.int64).copy()
    if x_arr.shape[0] != y_out.shape[0]:
        raise ValueError(
            f"x and y must have the same number of rows; "
            f"got x.shape[0]={x_arr.shape[0]} vs y.shape[0]={y_out.shape[0]}"
        )
    n_flip = int(round(fraction * y_out.shape[0]))
    if n_flip == 0:
        return y_out
    leverage = np.abs(x_arr[:, 0]) + _EPS
    probs = leverage / leverage.sum()
    idx = rng.choice(y_out.shape[0], size=n_flip, replace=False, p=probs)
    y_out[idx] = 1 - y_out[idx]
    return y_out


def _loss_grad_scale(p_true: ArrayF, loss: str, loss_param: float) -> ArrayF:
    """Per-example multiplier applied to the NLL logit-gradient ``(p - y)``.

    * ``nll``  -> 1 (standard gradient).
    * ``rcce`` -> ``p_true ** q_loss`` (the robust down-weight;
      ``q_loss = loss_param``). ``q_loss = 0`` recovers the NLL scale of 1, matching
      :func:`fedference.losses.rcce`'s limit.
    """
    if loss == "nll":
        return np.ones_like(p_true)
    if loss == "rcce":
        q_loss = float(loss_param)
        if not 0.0 <= q_loss <= 1.0:
            raise ValueError("rcce loss_param (q_loss) must lie in [0, 1]")
        return np.clip(p_true, _EPS, None) ** q_loss
    raise ValueError(f"unknown loss {loss!r}; choose from 'nll', 'rcce'")


def _client_update(
    x: ArrayF,
    y: ArrayF,
    w0: ArrayF,
    *,
    loss: str,
    loss_param: float,
    steps: int,
    lr: float,
    l2: float,
) -> ArrayF:
    """A few full-batch gradient steps of the chosen loss from prior ``w0``.

    ``w`` has length ``n_features + 1`` (last entry is the bias). The ``l2`` term
    is the KL/Gaussian-prior regularizer shrinking ``w`` toward ``w0`` — the
    FedGVI ``divergence`` acting in weight space.
    """
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    w = w0.astype(np.float64).copy()
    yf = y.astype(np.float64)
    n = xb.shape[0]
    for _ in range(steps):
        p1 = _sigmoid(xb @ w)  # P(y = 1)
        p_true = np.where(yf > 0.5, p1, 1.0 - p1)
        scale = _loss_grad_scale(p_true, loss, loss_param)
        grad_data = xb.T @ (scale * (p1 - yf)) / n
        grad = grad_data + l2 * (w - w0)
        w = w - lr * grad
    return w


def _accuracy(x: ArrayF, y: ArrayF, w: ArrayF) -> float:
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    pred = (_sigmoid(xb @ w) >= 0.5).astype(np.int64)
    return float(np.mean(pred == y.astype(np.int64)))


def fed_gvi_logreg(
    n_clients: int = 5,
    n_per: int = 80,
    contamination: float = 0.0,
    loss: str = "nll",
    loss_param: float = 0.0,
    divergence: str = "KLD",
    seed: int = 0,
) -> dict:
    """Run the federated logistic-regression baseline and return its result.

    Args:
        n_clients: number of federated clients (each gets a fresh data split).
        n_per: points per class per client (client split size ``2 * n_per``).
        contamination: fraction of each client's labels to flip (label noise).
        loss: ``'nll'`` (standard) or ``'rcce'`` (robust, FedGVI client loss).
        loss_param: the RCCE robustness ``q_loss`` in ``[0, 1]`` (ignored for NLL).
        divergence: weight-space regularizer name (``'KLD'`` -> Gaussian/L2;
            ``'AR'`` -> a slightly heavier shrinkage). Mirrors FedGVI server_div.
        seed: deterministic RNG seed (uses ``np.random.default_rng``).

    Returns:
        ``dict`` with ``test_accuracy`` in ``[0, 1]`` (server consensus weights
        on a clean held-out set) and ``weights`` (the aggregated weight vector,
        length ``n_features + 1``).
    """
    if n_clients <= 0:
        raise ValueError("n_clients must be positive")
    rng = np.random.default_rng(seed)

    # Weight-space regularizer strength selected by the named divergence.
    div = divergence.upper()
    if div == "KLD":
        l2 = 0.05
    elif div == "AR":
        l2 = 0.10
    else:
        raise ValueError(f"unknown divergence {divergence!r}; choose 'KLD' or 'AR'")

    n_features = 2
    prior = np.zeros(n_features + 1, dtype=np.float64)  # shared shrinkage prior

    client_weights = []
    for _ in range(n_clients):
        x, y_clean = make_blobs(n_per, rng=rng)
        y = contaminate(x, y_clean, contamination, rng=rng)
        w = _client_update(
            x,
            y,
            prior,
            loss=loss,
            loss_param=loss_param,
            steps=200,
            lr=0.8,
            l2=l2,
        )
        client_weights.append(w)

    # Server fuses client weight factors by averaging natural parameters:
    # a limited mean-field Gaussian / FedAvg analogue, not an Eq. 7
    # source-protocol reconstruction.
    consensus = np.mean(np.vstack(client_weights), axis=0)

    # Clean held-out test set from the same generative process.
    x_test, y_test = make_blobs(4 * n_per, rng=rng)
    acc = _accuracy(x_test, y_test, consensus)

    return {"test_accuracy": acc, "weights": consensus}
