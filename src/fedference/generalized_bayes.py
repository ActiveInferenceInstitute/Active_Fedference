"""Generalized Bayesian posterior updates in the conjugate categorical setting.

The generalized Bayes / Gibbs posterior (Bissiri et al., 2016; the inference
engine FedGVI federates) is

    q*(s)  =  argmin_q  E_q[ sum_i L(s; o_i) ]  +  (1/tau) * D(q || pi_0(s))

where ``L`` is a (robust) loss, ``pi_0`` the prior, ``tau`` the learning rate and
``D`` a divergence. For ``D = KL`` this has the closed form

    q*(s)  proportional to  pi_0(s) * exp( - tau * sum_i L(s; o_i) )

which, with ``L = NLL``, is *exactly* standard Bayes
``q*(s) ∝ pi_0(s) prod_i p(o_i | s)``. That identity is the hinge of the whole
project: it is what lets robust federated inference degrade gracefully to the
non-robust belief-sharing of Friston et al. (2024).

We also expose the EP/PVI **cavity** operation (remove one factor in
natural-parameter space), since FedGVI's client updates are computed against a
cavity rather than the full posterior.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import brentq

from ._validation import as_pmf
from .divergences import alpha_renyi_divergence, renyi_divergence  # noqa: F401

ArrayF = np.ndarray


def _log_pmf(p: ArrayF) -> ArrayF:
    return np.log(as_pmf(p))


def softmax(logits: ArrayF) -> ArrayF:
    """Numerically stable softmax returning a categorical pmf."""
    z = np.asarray(logits, dtype=np.float64).ravel()
    if z.size == 0:
        raise ValueError("softmax requires at least one logit")
    if not np.all(np.isfinite(z)):
        raise ValueError("softmax logits must be finite")
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _alpha_renyi_posterior(
    log_prior: ArrayF,
    loss_by_state: ArrayF,
    *,
    alpha: float,
    tau: float,
) -> ArrayF:
    """Solve the finite-support Alpha-Rényi generalized-Bayes objective.

    For ``D_AR`` as defined in :func:`fedference.divergences.alpha_renyi_divergence`,
    the stationarity equation has the form ``q_i ∝ pi_i (t - b_i)`` raised to
    ``1 / (alpha - 1)``. The scalar ``t`` is obtained from the normalization
    constraint. This one-dimensional solve is exact up to floating-point error
    and avoids conflating the objective with an ad-hoc power-softmax heuristic.
    """
    prior = softmax(log_prior)
    b = tau * np.asarray(loss_by_state, dtype=np.float64)
    b = b - float(np.mean(b))  # only differences matter; improve root stability
    exponent = 1.0 / (alpha - 1.0)
    target = 1.0 / abs(alpha - 1.0)

    def ratio(t: float, active: ArrayF) -> float:
        active_b = b[active]
        h = t - active_b if alpha > 1.0 else active_b - t
        if np.any(h <= 0.0):
            return 0.0 if alpha < 1.0 else np.inf
        log_weights = np.log(prior[active]) + exponent * np.log(h)
        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        return float(np.dot(weights, h) / np.sum(weights))

    span = max(float(np.ptp(b)), 1.0)
    epsilon = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(b))))
    if alpha > 1.0:
        # For alpha > 1 the optimum may lie on a face of the simplex.  The
        # active states are the lowest-loss prefix; states with b_i >= t have
        # a valid zero-mass KKT solution. Enumerate those prefixes so a large
        # alpha cannot force an invalid all-states interior solve.
        order = np.argsort(b)
        root: float | None = None
        active: ArrayF | None = None
        for count in range(1, b.size + 1):
            candidate = order[:count]
            lower = float(b[candidate[-1]] + epsilon)
            if count < b.size:
                gap = float(b[order[count]] - b[candidate[-1]])
                if gap <= 2.0 * epsilon:
                    continue
                upper = float(b[order[count]] - epsilon)
            else:
                upper = float(b[candidate[-1]] + span + target + 1.0)
                while ratio(upper, candidate) < target:
                    upper = float(b[candidate[-1]] + 2.0 * (upper - b[candidate[-1]]))
                    if not np.isfinite(upper):
                        raise FloatingPointError(
                            "could not bracket Alpha-Rényi posterior root"
                        )
            if ratio(lower, candidate) <= target <= ratio(upper, candidate):
                root = brentq(
                    lambda value, indices=candidate: ratio(value, indices) - target,
                    lower,
                    upper,
                )
                active = candidate
                break
        if root is None or active is None:
            raise FloatingPointError("could not bracket Alpha-Rényi posterior root")
        t = root
        h = t - b[active]
        log_weights = np.log(prior[active]) + exponent * np.log(h)
        weights = np.zeros_like(prior)
        weights[active] = np.exp(log_weights - float(np.max(log_weights)))
        return weights / np.sum(weights)
    else:
        upper = float(np.min(b) - epsilon)
        lower = float(np.min(b) - span - target - 1.0)
        all_states = np.arange(b.size)
        while ratio(lower, all_states) < target:
            lower = float(np.min(b) - 2.0 * (np.min(b) - lower))
            if not np.isfinite(lower):
                raise FloatingPointError("could not bracket Alpha-Rényi posterior root")
        active = all_states

    t = brentq(lambda value: ratio(value, active) - target, lower, upper)
    h = b - t
    log_weights = np.log(prior) + exponent * np.log(h)
    return softmax(log_weights)


def generalized_posterior(
    log_prior: ArrayF,
    loss_by_state: ArrayF | None = None,
    *,
    tau: float = 1.0,
    divergence: str = "KLD",
    alpha: float = 1.0,
    **legacy: object,
) -> ArrayF:
    """Closed-form generalized posterior over hidden states.

    ``log_prior`` : length-``n_s`` log prior (need not be normalized).
    ``loss_by_state`` : length-``n_s`` accumulated loss per state (see
                    :func:`fedference.losses.loss_vector`; the canonical API
                    term is ``loss_by_state``).
    ``tau`` : the learning-rate scalar :math:`\tau` tempering the data term.
    The old ``learning_rate`` keyword is retained as a warned compatibility
    adapter; it is not the federation influence weight ``w_n``.

    For ``divergence='KLD'`` the minimizer is the tempered softmax
    ``softmax(log_prior - tau * loss_by_state)``. For ``divergence='AR'`` (FedGVI's
    Alpha-Rényi divergence), the finite categorical minimizer is obtained by a
    one-dimensional normalization solve; it is not generally the commonly
    used power-softmax shortcut. At ``alpha = 1`` it is identical to the KL
    form, preserving the limit.
    """
    if "loss_vec" in legacy:
        if loss_by_state is not None:
            raise TypeError("loss_by_state and deprecated loss_vec cannot both be supplied")
        loss_by_state = legacy.pop("loss_vec")  # type: ignore[assignment]
        warnings.warn(
            "loss_vec is deprecated; use loss_by_state",
            DeprecationWarning,
            stacklevel=2,
        )
    used_legacy_learning_rate = "learning_rate" in legacy
    if used_legacy_learning_rate:
        if tau != 1.0:
            raise TypeError(
                "tau and deprecated learning_rate cannot both be supplied"
            )
        tau = legacy.pop("learning_rate")  # type: ignore[assignment]
        warnings.warn(
            "learning_rate is deprecated; use tau",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        names = ", ".join(sorted(legacy))
        raise TypeError(f"unexpected keyword argument(s): {names}")
    if loss_by_state is None:
        raise TypeError("loss_by_state is required")
    lp = np.asarray(log_prior, dtype=np.float64).ravel()
    lv = np.asarray(loss_by_state, dtype=np.float64).ravel()
    if lp.shape != lv.shape:
        raise ValueError("log_prior and loss_by_state must have the same shape")
    if lp.size == 0:
        raise ValueError("log_prior and loss_by_state must be non-empty")
    if not np.all(np.isfinite(lp)) or not np.all(np.isfinite(lv)):
        raise ValueError("log_prior and loss_by_state must be finite")
    if not np.isfinite(tau) or tau < 0.0:
        parameter_name = "learning_rate" if used_legacy_learning_rate else "tau"
        raise ValueError(f"{parameter_name} must be finite and non-negative")
    key = divergence.upper()
    if key == "AR":
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be finite and strictly positive")
        if abs(alpha - 1.0) < 1e-6:
            return softmax(lp - tau * lv)
        return _alpha_renyi_posterior(lp, lv, alpha=alpha, tau=tau)
    if key != "KLD":
        raise ValueError("unknown generalized-Bayes divergence; choose KLD or AR")
    return softmax(lp - tau * lv)


def cavity(
    global_posterior: ArrayF | None = None,
    site_factor: ArrayF | None = None,
    **legacy: object,
) -> ArrayF:
    """Return the cavity ``q_{-n}`` with a site factor ``t_n`` removed.

    In natural-parameter (log) space this is subtraction:
    ``log q_{-n} = log q - log t_n``. The cavity is a normalized posterior
    proportional to the global posterior divided by the local site factor.
    ``posterior``/``factor`` remain deprecated compatibility keywords.
    """
    if "posterior" in legacy:
        if global_posterior is not None:
            raise TypeError("global_posterior and deprecated posterior cannot both be supplied")
        global_posterior = legacy.pop("posterior")  # type: ignore[assignment]
        warnings.warn(
            "posterior is deprecated; use global_posterior",
            DeprecationWarning,
            stacklevel=2,
        )
    if "factor" in legacy:
        if site_factor is not None:
            raise TypeError("site_factor and deprecated factor cannot both be supplied")
        site_factor = legacy.pop("factor")  # type: ignore[assignment]
        warnings.warn(
            "factor is deprecated; use site_factor",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        names = ", ".join(sorted(legacy))
        raise TypeError(f"unexpected keyword argument(s): {names}")
    if global_posterior is None or site_factor is None:
        raise TypeError("global_posterior and site_factor are required")
    log_cav = _log_pmf(global_posterior) - _log_pmf(site_factor)
    return softmax(log_cav)


def update_factor(
    old_site_factor: ArrayF | None = None,
    old_global_posterior: ArrayF | None = None,
    new_global_posterior: ArrayF | None = None,
    **legacy: object,
) -> ArrayF:
    """Return the refreshed local factor ``t_i`` after a client update.

    ``log t_n^new = log t_n^old + log q^new - log q^old`` (the PVI factor
    update), renormalized to a pmf for transport.
    """
    for old_name, canonical, target in (
        ("old_factor", "old_site_factor", "old_site_factor"),
        ("old_posterior", "old_global_posterior", "old_global_posterior"),
        ("new_posterior", "new_global_posterior", "new_global_posterior"),
    ):
        if old_name not in legacy:
            continue
        if target == "old_site_factor" and old_site_factor is not None:
            raise TypeError(f"{canonical} and deprecated {old_name} cannot both be supplied")
        if target == "old_global_posterior" and old_global_posterior is not None:
            raise TypeError(f"{canonical} and deprecated {old_name} cannot both be supplied")
        if target == "new_global_posterior" and new_global_posterior is not None:
            raise TypeError(f"{canonical} and deprecated {old_name} cannot both be supplied")
        value = legacy.pop(old_name)
        if target == "old_site_factor":
            old_site_factor = value  # type: ignore[assignment]
        elif target == "old_global_posterior":
            old_global_posterior = value  # type: ignore[assignment]
        else:
            new_global_posterior = value  # type: ignore[assignment]
        warnings.warn(
            f"{old_name} is deprecated; use {canonical}",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        names = ", ".join(sorted(legacy))
        raise TypeError(f"unexpected keyword argument(s): {names}")
    if old_site_factor is None or old_global_posterior is None or new_global_posterior is None:
        raise TypeError("old_site_factor, old_global_posterior, and new_global_posterior are required")
    log_t = _log_pmf(old_site_factor) + _log_pmf(new_global_posterior) - _log_pmf(old_global_posterior)
    return softmax(log_t)
