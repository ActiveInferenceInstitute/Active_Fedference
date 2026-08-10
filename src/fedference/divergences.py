"""Divergences over categorical distributions for generalized variational inference.

These are the *regularizing* divergences ``D(q || cavity)`` of the FedGVI
objective (Mildner et al., 2025, arXiv:2502.00846). In the discrete
active-inference setting every belief is a categorical pmf (a 1-D, non-negative
array summing to one), so the divergence family reduces to closed forms.

The single most important property — exercised by the test-suite — is that the
robust members **recover the Kullback-Leibler divergence in a limit**:

* ``renyi_divergence(q, p, alpha) -> kl_divergence(q, p)`` as ``alpha -> 1``.

KL is the divergence that makes generalized Bayes collapse to *standard* Bayes.
Separately, under the project's shared-support, posterior-log-potential, and
fixed-weight bridge assumptions, its categorical log-linear/product-of-experts
pool specializes Friston et al. (2024) Eq. 7's message-combination term. That
is not a reconstruction of the complete source protocol. Everything robust is a
controlled departure from the project-local recovery points.
"""

from __future__ import annotations

import numpy as np

from ._validation import as_pmf

ArrayF = np.ndarray
_EPS = 1e-12


def _as_pmf(p: ArrayF) -> ArrayF:
    """Validate and normalize a finite categorical probability vector."""
    return as_pmf(p)


def kl_divergence(q: ArrayF, p: ArrayF) -> float:
    """Return ``KL(q || p) = sum_k q_k log(q_k / p_k)`` in nats (>= 0)."""
    q_ = _as_pmf(q)
    p_ = _as_pmf(p)
    return float(np.sum(q_ * (np.log(q_) - np.log(p_))))


def reverse_kl(q: ArrayF, p: ArrayF) -> float:
    """Return the reverse KL ``KL(p || q)`` — FedGVI's ``RKL`` client divergence."""
    return kl_divergence(p, q)


def renyi_divergence(q: ArrayF, p: ArrayF, alpha: float) -> float:
    """Return the standard Rényi divergence ``D_alpha(q || p)``.

    ``(1 / (alpha - 1)) * log sum_k q_k^alpha p_k^(1 - alpha)``.

    This is the conventional Rényi family. FedGVI's Alpha-Rényi regularizer
    has an additional factor ``1 / alpha``; use
    :func:`alpha_renyi_divergence` for that objective. The limit ``alpha -> 1``
    is :func:`kl_divergence`. We switch to the KL closed form inside a small
    band around 1 for numerical stability, so the recovery is exact rather
    than merely asymptotic.
    """
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if abs(alpha - 1.0) < 1e-6:
        return kl_divergence(q, p)
    q_ = _as_pmf(q)
    p_ = _as_pmf(p)
    summand = np.sum(q_**alpha * p_ ** (1.0 - alpha))
    return float(np.log(max(summand, _EPS)) / (alpha - 1.0))


def alpha_renyi_divergence(q: ArrayF, p: ArrayF, alpha: float) -> float:
    r"""Return FedGVI's Alpha-Rényi divergence.

    FedGVI defines

    .. math::

        D_{AR}^{(\alpha)}(q\|p) =
        \frac{1}{\alpha(\alpha-1)}
        \log \sum_k q_k^\alpha p_k^{1-\alpha}.

    It is therefore the standard Rényi divergence divided by ``alpha``. The
    ``alpha -> 1`` limit is KL; ``alpha`` must be strictly positive.
    """
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and strictly positive")
    if abs(alpha - 1.0) < 1e-6:
        return kl_divergence(q, p)
    return float(renyi_divergence(q, p, alpha) / alpha)


def total_variation(q: ArrayF, p: ArrayF) -> float:
    """Return total-variation distance ``0.5 * sum_k |q_k - p_k|`` in [0, 1]."""
    q_ = _as_pmf(q)
    p_ = _as_pmf(p)
    return float(0.5 * np.sum(np.abs(q_ - p_)))


# ---------------------------------------------------------------------------
# Continuous-state (1-D Gaussian) divergences — the explicitly-scoped bridge
# toward continuous active inference. OUT OF SCOPE for the categorical federated
# experiments (Friston's worked example and every robustness claim are discrete);
# these closed forms exist so the same divergence family (KL with a Renyi
# generalization that recovers it as alpha -> 1) is shown to carry over to the
# Gaussian beliefs a continuous-state extension would use. They are NOT wired
# into :func:`divergence` or any categorical experiment; the
# ``continuous_recovery`` module exercises them for the 1-D Gaussian-mean
# recovery-limit slice.
# ---------------------------------------------------------------------------


def gaussian_kl(mu_q: float, var_q: float, mu_p: float, var_p: float) -> float:
    """Closed-form ``KL(N(mu_q, var_q) || N(mu_p, var_p))`` for 1-D Gaussians.

    ``0.5 * [ var_q/var_p + (mu_p - mu_q)^2 / var_p - 1 + log(var_p / var_q) ]``
    (nats, ``>= 0``, zero iff the two Gaussians coincide). Variances must be
    strictly positive. The continuous-state analogue of :func:`kl_divergence`.
    """
    if var_q <= 0.0 or var_p <= 0.0:
        raise ValueError("variances must be strictly positive")
    mq, vq, mp, vp = float(mu_q), float(var_q), float(mu_p), float(var_p)
    return float(0.5 * (vq / vp + (mp - mq) ** 2 / vp - 1.0 + np.log(vp / vq)))


def gaussian_renyi(
    mu_q: float, var_q: float, mu_p: float, var_p: float, alpha: float
) -> float:
    """Closed-form standard Rényi divergence between two 1-D Gaussians.

    ``D_alpha(N_q || N_p) = alpha (mu_q - mu_p)^2 / (2 var_alpha)
        - 1/(2 (alpha - 1)) log( var_alpha / (var_q^(1-alpha) var_p^alpha) )``
    with the interpolated variance ``var_alpha = alpha var_p + (1 - alpha) var_q``.
    Defined for ``alpha > 0``, ``alpha != 1`` (with ``var_alpha > 0``); the
    ``alpha -> 1`` limit is :func:`gaussian_kl`, so — as in the categorical case —
    we return the KL closed form inside a small band around 1 for an exact,
    not merely asymptotic, recovery.
    """
    if var_q <= 0.0 or var_p <= 0.0:
        raise ValueError("variances must be strictly positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if abs(alpha - 1.0) < 1e-6:
        return gaussian_kl(mu_q, var_q, mu_p, var_p)
    mq, vq, mp, vp = float(mu_q), float(var_q), float(mu_p), float(var_p)
    var_alpha = alpha * vp + (1.0 - alpha) * vq
    if var_alpha <= 0.0:
        raise ValueError(
            "interpolated variance is non-positive; Renyi divergence diverges "
            "for this (alpha, var_q, var_p)"
        )
    quad = alpha * (mq - mp) ** 2 / (2.0 * var_alpha)
    logterm = np.log(var_alpha / (vq ** (1.0 - alpha) * vp**alpha))
    return float(quad - logterm / (2.0 * (alpha - 1.0)))


def gaussian_alpha_renyi(
    mu_q: float, var_q: float, mu_p: float, var_p: float, alpha: float
) -> float:
    """Return the Gaussian Alpha-Rényi divergence used by FedGVI.

    This is :func:`gaussian_renyi` divided by ``alpha``, with the KL limit at
    ``alpha = 1``. It is provided as a clearly named continuous-state bridge;
    the categorical experiment harness does not dispatch to it.
    """
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and strictly positive")
    if abs(alpha - 1.0) < 1e-6:
        return gaussian_kl(mu_q, var_q, mu_p, var_p)
    return float(gaussian_renyi(mu_q, var_q, mu_p, var_p, alpha) / alpha)


_DIVERGENCES = {
    "KLD": kl_divergence,
    "RKL": reverse_kl,
    "TV": total_variation,
}

# Labels used as robustness-method identifiers in the experiment harness that do
# NOT correspond to a categorical divergence computable here: they label FedGVI
# client objectives whose robustness effect is expressed as a server-side
# down-weighting strength via ``_divergence_to_robustness`` in experiments.py.
# The dispatcher surfaces a clear message rather than a cryptic KeyError.
_EXPERIMENT_ONLY_LABELS = frozenset({"BETA", "RCCE"})


def divergence(name: str, q: ArrayF, p: ArrayF, param: float | None = None) -> float:
    """Dispatch a named categorical divergence (``KLD``, ``RKL``, ``AR``, ``TV``).

    ``AR`` (alpha-Renyi) requires ``param`` (the alpha); the others ignore it.
    Mirrors FedGVI's ``--client_div`` / ``--server_div`` string arguments.

    Note: ``beta`` and ``rcce`` are valid FedGVI *client* divergence labels used
    by the experiment harness (routed via ``_divergence_to_robustness``), but they
    are not categorical divergences computable directly on two pmfs — they label
    robust server-weighting strengths, not closed-form ``D(q||p)`` formulas.
    Passing them here raises ``ValueError`` with an explanatory message.
    """
    key = name.upper()
    if key == "AR":
        if param is None:
            raise ValueError("alpha-Renyi divergence requires param (alpha)")
        return alpha_renyi_divergence(q, p, param)
    if key in _EXPERIMENT_ONLY_LABELS:
        raise ValueError(
            f"{name!r} is a FedGVI client-objective label, not a categorical divergence. "
            "Use fedference.experiments._divergence_to_robustness to map it to a "
            "robustness strength for robust_aggregate / share_round."
        )
    if key not in _DIVERGENCES:
        raise ValueError(
            f"unknown divergence {name!r}; valid choices: "
            f"{sorted(_DIVERGENCES)} plus 'AR' (requires param=alpha)"
        )
    return _DIVERGENCES[key](q, p)
