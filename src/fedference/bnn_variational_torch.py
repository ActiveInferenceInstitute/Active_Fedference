"""Genuine mean-field variational MLP — the faithful FedGVI variational family (MAJ-2 slice).

The point-mass baseline (:class:`fedference.bnn_baseline_torch.PointMassMLP`) is
a deterministic net: no distribution over weights. This module adds the real
mean-field variational family the FedGVI framing calls for — each weight carries
a diagonal Gaussian ``q(w) = N(mu, sigma^2)`` with ``sigma = softplus(rho)`` — a
reparameterized forward (``w = mu + sigma * eps``), a **closed-form** diagonal
Gaussian KL to an ``N(0, prior_var)`` prior (built on the tested
:func:`fedference.divergences.gaussian_kl`), and an ELBO objective combining a
Monte-Carlo beta-loss data term with the KL.

Two recovery limits keep it honest, each bound to an INDEPENDENT reference:

* ``sigma -> 0`` recovers the point-mass net exactly: with the variance driven to
  zero the reparameterized forward equals ``PointMassMLP.forward`` on the same
  ``mu`` weights (the mean network).
* ``kl_weight = 1`` with ``beta -> 0`` recovers the standard Bayes-by-backprop
  ELBO — a mean NLL data term plus the closed-form KL.

Scope (kept honest): this is the variational-family + ELBO primitive. The
model-agnostic site/cavity/factor-replacement server state lives separately in
``bnn_fedgvi.py``. The cavity-conditioned source client optimizer,
protocol-scale vision runs, confirmatory contamination sweep, and any "beats
NLL" claim remain open under MAJ-2A/2B. The data term is a Monte-Carlo estimate:
it is an "MC ELBO estimate", never an exact ELBO.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from .bnn_defaults import (
    BNN_BETA_DEFAULT,
    BNN_HIDDEN_DIM_DEFAULT,
)
from .bnn_fedgvi import DiagonalGaussian
from .divergences import gaussian_kl

if TYPE_CHECKING:
    from .bnn_baseline_torch import PointMassMLP

_EPS = 1e-12
#: Prior variance of the weight prior N(0, prior_var) used in the KL term.
PRIOR_VAR_DEFAULT: float = 1.0


def _integer_control(
    value: object,
    *,
    name: str,
    minimum: int,
) -> int:
    """Return one strict integer control, excluding boolean coercion."""
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def _real_control(
    value: object,
    *,
    name: str,
    positive: bool,
) -> float:
    """Return one finite real control without accepting strings or booleans."""
    qualifier = "positive" if positive else "non-negative"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and {qualifier}")
    result = float(value)
    invalid_bound = result <= 0.0 if positive else result < 0.0
    if not math.isfinite(result) or invalid_bound:
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


class VariationalMLP(nn.Module):
    """Mean-field variational MLP: diagonal-Gaussian ``q(w)`` over every weight.

    Architecture mirrors :class:`PointMassMLP`
    (Linear -> ReLU -> Linear -> Softmax); every weight/bias ``w`` is replaced by
    a variational pair ``(mu_w, rho_w)`` with ``sigma_w = softplus(rho_w)``.

    Args:
        input_dim / hidden_dim / output_dim: layer sizes.
        seed: private CPU-generator seed for parameter initialization and
            reparameterization draws.
        beta: beta-loss exponent for the data term.
        prior_var: variance of the ``N(0, prior_var)`` weight prior in the KL.
        init_rho: initial ``rho`` for every weight (``softplus(init_rho)`` is the
            initial posterior std). A large negative value makes the net nearly
            deterministic at initialization.
    """

    _gen: torch.Generator
    _mu_W1: nn.Parameter
    _rho_W1: nn.Parameter
    _mu_b1: nn.Parameter
    _rho_b1: nn.Parameter
    _mu_W2: nn.Parameter
    _rho_W2: nn.Parameter
    _mu_b2: nn.Parameter
    _rho_b2: nn.Parameter
    _mus: list[nn.Parameter]
    _rhos: list[nn.Parameter]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = BNN_HIDDEN_DIM_DEFAULT,
        output_dim: int = 2,
        *,
        seed: int = 0,
        beta: float = BNN_BETA_DEFAULT,
        prior_var: float = PRIOR_VAR_DEFAULT,
        init_rho: float = -3.0,
    ) -> None:
        super().__init__()
        input_dim = _integer_control(input_dim, name="input_dim", minimum=1)
        hidden_dim = _integer_control(hidden_dim, name="hidden_dim", minimum=1)
        output_dim = _integer_control(output_dim, name="output_dim", minimum=1)
        seed = _integer_control(seed, name="seed", minimum=0)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.beta = _real_control(beta, name="beta", positive=False)
        self.prior_var = _real_control(prior_var, name="prior_var", positive=True)
        if isinstance(init_rho, bool) or not isinstance(init_rho, Real):
            raise ValueError("init_rho must be finite")
        init_rho = float(init_rho)
        if not math.isfinite(init_rho):
            raise ValueError("init_rho must be finite")
        self._gen = torch.Generator().manual_seed(seed)

        def _mu(shape: tuple[int, ...]) -> nn.Parameter:
            return nn.Parameter(torch.randn(*shape, generator=self._gen) * 0.1)

        def _rho(shape: tuple[int, ...]) -> nn.Parameter:
            return nn.Parameter(torch.full(shape, init_rho))

        self._mu_W1, self._rho_W1 = _mu((hidden_dim, input_dim)), _rho((hidden_dim, input_dim))
        self._mu_b1, self._rho_b1 = nn.Parameter(torch.zeros(hidden_dim)), _rho((hidden_dim,))
        self._mu_W2, self._rho_W2 = _mu((output_dim, hidden_dim)), _rho((output_dim, hidden_dim))
        self._mu_b2, self._rho_b2 = nn.Parameter(torch.zeros(output_dim)), _rho((output_dim,))

        self._mus = [self._mu_W1, self._mu_b1, self._mu_W2, self._mu_b2]
        self._rhos = [self._rho_W1, self._rho_b1, self._rho_W2, self._rho_b2]

    # -- weight sampling -----------------------------------------------------
    def _sigma(self, rho: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(rho)

    def _sample_weight(self, mu: torch.Tensor, rho: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
        if deterministic:
            return mu
        # Keep one private CPU generator for reproducible parameter draws, then
        # transfer the draw to the selected backend. MPS does not currently
        # accept a CPU generator directly in a device-local ``torch.randn``.
        eps = torch.randn(mu.shape, generator=self._gen, dtype=mu.dtype).to(mu.device)
        return mu + self._sigma(rho) * eps

    def forward(self, x: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        """Reparameterized forward. ``deterministic`` uses the mean weights
        (``sigma`` ignored), which — at ``sigma -> 0`` — equals the mean network."""
        if not isinstance(deterministic, bool):
            raise ValueError("deterministic must be a boolean")
        w1 = self._sample_weight(self._mu_W1, self._rho_W1, deterministic=deterministic)
        b1 = self._sample_weight(self._mu_b1, self._rho_b1, deterministic=deterministic)
        w2 = self._sample_weight(self._mu_W2, self._rho_W2, deterministic=deterministic)
        b2 = self._sample_weight(self._mu_b2, self._rho_b2, deterministic=deterministic)
        h = torch.relu(x @ w1.T + b1)
        logits = h @ w2.T + b2
        return torch.softmax(logits, dim=1)

    # -- objective -----------------------------------------------------------
    def kl_to_prior(self) -> torch.Tensor:
        """Closed-form ``sum_w KL(N(mu_w, sigma_w^2) || N(0, prior_var))``.

        Uses the analytic diagonal-Gaussian KL — the same formula as the tested
        :func:`fedference.divergences.gaussian_kl` — summed over every weight.
        ``>= 0`` (Gibbs), and ``0`` iff every ``mu_w = 0`` and ``sigma_w^2 =
        prior_var``.
        """
        device = self._mus[0].device
        # MPS does not implement float64 kernels. Retain the historical
        # float64 analytic path on CPU and use the model dtype on MPS.
        dtype = torch.float64 if device.type == "cpu" else self._mus[0].dtype
        total = torch.zeros((), dtype=dtype, device=device)
        prior = torch.tensor(self.prior_var, dtype=dtype, device=device)
        for mu, rho in zip(self._mus, self._rhos, strict=True):
            var = self._sigma(rho) ** 2
            m = mu.to(dtype=dtype)
            v = var.to(dtype=dtype)
            # 0.5 * [ v/pv + m^2/pv - 1 + log(pv/v) ], elementwise, summed.
            total = (
                total + 0.5 * (v / self.prior_var + m**2 / self.prior_var - 1.0 + torch.log(prior / v)).sum()
            )
        return total

    def to_diagonal_gaussian(self) -> DiagonalGaussian:
        """Export the mean-field parameters in FedGVI natural-state format.

        The export is deliberately explicit: the neural family remains a
        PyTorch object, while the server protocol consumes an immutable
        NumPy ``DiagonalGaussian``.  This is the bridge that makes cavity and
        site-factor updates auditable rather than an unrelated neural
        aggregation shortcut.
        """
        means = torch.cat([mu.detach().reshape(-1) for mu in self._mus])
        variances = torch.cat(
            [(self._sigma(rho).detach() ** 2).reshape(-1) for rho in self._rhos]
        )
        return DiagonalGaussian(
            mean=means.to(device="cpu", dtype=torch.float64).numpy(),
            variance=variances.to(device="cpu", dtype=torch.float64).numpy(),
        )

    def load_diagonal_gaussian(self, posterior: DiagonalGaussian) -> None:
        """Load a server/cavity Gaussian into the model's ``mu`` and ``rho``.

        ``rho`` is the inverse-softplus parameterisation of the standard
        deviation.  Loading is shape-checked and performed without creating a
        gradient edge to the server state.
        """
        if not isinstance(posterior, DiagonalGaussian):
            raise ValueError("posterior must be a DiagonalGaussian")
        expected = sum(parameter.numel() for parameter in self._mus)
        if posterior.mean.size != expected:
            raise ValueError("posterior dimension does not match the VariationalMLP")
        device = self._mus[0].device
        dtype = self._mus[0].dtype
        means = torch.as_tensor(
            np.array(posterior.mean, dtype=np.float64, copy=True),
            dtype=dtype,
            device=device,
        )
        std = torch.as_tensor(
            np.array(np.sqrt(posterior.variance), dtype=np.float64, copy=True),
            dtype=dtype,
            device=device,
        )
        if not torch.all(torch.isfinite(means)) or not torch.all(torch.isfinite(std)):
            raise ValueError("posterior parameters must be finite")
        if torch.any(std <= 0.0):
            raise ValueError("posterior standard deviations must be positive")

        def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
            # ``value + log(1 - exp(-value))`` avoids exp overflow for wide
            # cavities while retaining precision for the small variances used
            # by the portable pilot.
            return value + torch.log(-torch.expm1(-value))

        offset = 0
        with torch.no_grad():
            for mu, rho in zip(self._mus, self._rhos, strict=True):
                size = mu.numel()
                mu.copy_(means[offset : offset + size].reshape_as(mu))
                rho.copy_(inverse_softplus(std[offset : offset + size]).reshape_as(rho))
                offset += size

    def kl_to_reference(self, reference: DiagonalGaussian) -> torch.Tensor:
        """Return ``KL(q || reference)`` for a diagonal Gaussian cavity."""
        if not isinstance(reference, DiagonalGaussian):
            raise ValueError("reference must be a DiagonalGaussian")
        if reference.mean.size != sum(parameter.numel() for parameter in self._mus):
            raise ValueError("reference dimension does not match the VariationalMLP")
        device = self._mus[0].device
        dtype = self._mus[0].dtype
        ref_mean = torch.as_tensor(
            np.array(reference.mean, dtype=np.float64, copy=True),
            dtype=dtype,
            device=device,
        )
        ref_var = torch.as_tensor(
            np.array(reference.variance, dtype=np.float64, copy=True),
            dtype=dtype,
            device=device,
        )
        means = torch.cat([mu.reshape(-1) for mu in self._mus])
        variances = torch.cat([(self._sigma(rho) ** 2).reshape(-1) for rho in self._rhos])
        return 0.5 * (
            torch.log(ref_var / variances)
            + (variances + (means - ref_mean) ** 2) / ref_var
            - 1.0
        ).sum()

    def fit_from_cavity(
        self,
        cavity: DiagonalGaussian,
        x: torch.Tensor,
        y_onehot: torch.Tensor,
        *,
        n_steps: int = 100,
        lr: float = 0.01,
        n_mc: int = 4,
        kl_weight: float = 1.0,
        beta: float | None = None,
    ) -> list[float]:
        """Fit one client against a supplied FedGVI cavity.

        The local objective is an MC beta-loss plus ``KL(q || cavity)``.  The
        resulting posterior can therefore be returned to
        :class:`fedference.bnn_fedgvi.FedGVIServerState`, which replaces the
        client's site by posterior-minus-cavity natural parameters.
        """
        if not isinstance(cavity, DiagonalGaussian):
            raise ValueError("cavity must be a DiagonalGaussian")
        n_steps = _integer_control(n_steps, name="n_steps", minimum=0)
        lr = _real_control(lr, name="lr", positive=True)
        n_mc = _integer_control(n_mc, name="n_mc", minimum=1)
        kl_weight = _real_control(kl_weight, name="kl_weight", positive=False)
        self.load_diagonal_gaussian(cavity)
        device = self._mus[0].device
        x = x.to(device)
        y_onehot = y_onehot.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history: list[float] = []
        for _ in range(n_steps):
            optimizer.zero_grad()
            data_terms = []
            for _ in range(n_mc):
                probs = self.forward(x, deterministic=False)
                data_terms.append(self.beta_loss(probs, y_onehot, beta=beta).mean())
            data = torch.stack(data_terms).mean()
            loss = data + kl_weight * self.kl_to_reference(cavity).to(data.dtype)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach().cpu()))
        return history

    def beta_loss(
        self, probs: torch.Tensor, y_onehot: torch.Tensor, beta: float | None = None
    ) -> torch.Tensor:
        """Per-sample recentered density-power beta-loss.

        This is identical to :meth:`PointMassMLP.beta_loss` and to
        :func:`fedference.losses.beta_loss`; the removable beta-zero limit is
        evaluated as NLL.
        """
        b = _real_control(beta, name="beta", positive=False) if beta is not None else self.beta
        p = torch.clamp(probs, min=_EPS)
        p_true = (p * y_onehot).sum(dim=1)
        if b < 1e-8:
            return -torch.log(p_true)
        data_term = -(p_true**b - 1.0) / b
        norm_term = (p.pow(b + 1.0).sum(dim=1) - 1.0) / (b + 1.0)
        return data_term + norm_term

    def elbo(
        self,
        x: torch.Tensor,
        y_onehot: torch.Tensor,
        *,
        n_mc: int = 4,
        kl_weight: float = 1.0,
        beta: float | None = None,
    ) -> torch.Tensor:
        """Negative ELBO (a loss to minimize): MC beta-loss data term + KL.

        ``mean over MC samples of mean-over-batch beta-loss  +  kl_weight * KL``.
        A Monte-Carlo estimate of the data term (the KL is exact).
        """
        n_mc = _integer_control(n_mc, name="n_mc", minimum=1)
        kl_weight = _real_control(kl_weight, name="kl_weight", positive=False)
        data_terms = []
        for _ in range(n_mc):
            probs = self.forward(x, deterministic=False)
            data_terms.append(self.beta_loss(probs, y_onehot, beta=beta).mean())
        data = torch.stack(data_terms).mean()
        return data + kl_weight * self.kl_to_prior().to(data.dtype)

    def fit(
        self,
        x: torch.Tensor,
        y_onehot: torch.Tensor,
        *,
        n_steps: int = 100,
        lr: float = 0.01,
        n_mc: int = 4,
        kl_weight: float = 1.0,
    ) -> list[float]:
        """Train by minimizing the negative ELBO with Adam. Returns the loss trace."""
        n_steps = _integer_control(n_steps, name="n_steps", minimum=0)
        lr = _real_control(lr, name="lr", positive=True)
        n_mc = _integer_control(n_mc, name="n_mc", minimum=1)
        kl_weight = _real_control(kl_weight, name="kl_weight", positive=False)
        device = self._mus[0].device
        x = x.to(device)
        y_onehot = y_onehot.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history: list[float] = []
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = self.elbo(x, y_onehot, n_mc=n_mc, kl_weight=kl_weight)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        return history

    def copy_means_into(self, point_mass: PointMassMLP) -> None:
        """Copy this net's mean weights into a PointMassMLP (for the sigma->0
        recovery test — the mean network must equal the deterministic net)."""
        with torch.no_grad():
            point_mass._W1.copy_(self._mu_W1)  # type: ignore[attr-defined]
            point_mass._b1.copy_(self._mu_b1)  # type: ignore[attr-defined]
            point_mass._W2.copy_(self._mu_W2)  # type: ignore[attr-defined]
            point_mass._b2.copy_(self._mu_b2)  # type: ignore[attr-defined]

    def predict_proba(
        self,
        x: torch.Tensor,
        *,
        n_samples: int = 50,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Return a posterior-predictive mean on the model's current device."""
        n_samples = _integer_control(n_samples, name="n_samples", minimum=1)
        if not isinstance(deterministic, bool):
            raise ValueError("deterministic must be a boolean")
        x = x.to(self._mus[0].device)
        with torch.no_grad():
            if deterministic:
                return self.forward(x, deterministic=True)
            samples = [self.forward(x, deterministic=False) for _ in range(n_samples)]
        return torch.stack(samples).mean(dim=0)


def gaussian_kl_reference(mu: float, var: float, prior_var: float = PRIOR_VAR_DEFAULT) -> float:
    """One-weight KL via the tested categorical-sibling :func:`gaussian_kl` — the
    independent reference the module's summed KL is checked against."""
    return gaussian_kl(mu, var, 0.0, prior_var)


__all__ = ["VariationalMLP", "PRIOR_VAR_DEFAULT", "gaussian_kl_reference"]
