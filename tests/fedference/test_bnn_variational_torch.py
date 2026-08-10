"""Mean-field VariationalMLP recovery limits (MAJ-2) — no mocks, real torch.

Torch-gated (``requires_torch``). Every identity binds to an INDEPENDENT
reference so nothing is green-by-construction:

1. sigma -> 0 recovers the deterministic point-mass net EXACTLY (the mean
   network), checked against a separate PointMassMLP holding the same mu.
2. The closed-form summed KL equals the tested categorical-sibling
   :func:`gaussian_kl` summed per weight (independent formula).
3. Gibbs: KL == 0 iff q == prior (mu=0, var=prior_var); KL > 0 otherwise.
4. ELBO decomposition: with the MC draws held fixed, elbo(kl_weight=1) -
   elbo(kl_weight=0) equals exactly the closed-form KL — the data term and the
   KL term are correctly separated.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from fedference.bnn_baseline_torch import PointMassMLP  # noqa: E402
from fedference.bnn_fedgvi import DiagonalGaussian  # noqa: E402
from fedference.bnn_variational_torch import (  # noqa: E402
    PRIOR_VAR_DEFAULT,
    VariationalMLP,
    gaussian_kl_reference,
)

pytestmark = [pytest.mark.requires_torch, pytest.mark.slow]


def test_sigma_to_zero_recovers_point_mass_net() -> None:
    v = VariationalMLP(4, 8, 3, seed=0, init_rho=-30.0)  # softplus(-30) ~ 0 -> deterministic
    pm = PointMassMLP(4, 8, 3, seed=1)
    v.copy_means_into(pm)
    x = torch.randn(5, 4, generator=torch.Generator().manual_seed(7))
    out_v = v.forward(x, deterministic=False)  # sigma ~ 0 -> equals the mean net
    out_pm = pm.forward(x)
    assert torch.allclose(out_v, out_pm, atol=1e-5)
    # The deterministic path is exactly the mean net.
    assert torch.allclose(v.forward(x, deterministic=True), out_pm, atol=1e-12)


def test_variational_mlp_is_composable_module() -> None:
    """The variational complement exposes standard module parameters/state."""
    v = VariationalMLP(4, 8, 3, seed=0)
    assert isinstance(v, torch.nn.Module)
    assert len(tuple(v.parameters())) == 8
    assert set(v.state_dict()) == {
        "_mu_W1",
        "_rho_W1",
        "_mu_b1",
        "_rho_b1",
        "_mu_W2",
        "_rho_W2",
        "_mu_b2",
        "_rho_b2",
    }


def test_variational_mlp_rejects_invalid_configuration() -> None:
    """Invalid dimensions, prior variance, and optimizer controls fail closed."""
    with pytest.raises(ValueError, match="positive integer"):
        VariationalMLP(0, 8, 3)
    with pytest.raises(ValueError, match="prior_var"):
        VariationalMLP(4, 8, 3, prior_var=0.0)
    v = VariationalMLP(4, 8, 3)
    x = torch.ones(2, 4)
    y = torch.eye(3)[torch.tensor([0, 1])]
    with pytest.raises(ValueError, match="n_steps"):
        v.fit(x, y, n_steps=-1)
    with pytest.raises(ValueError, match="lr"):
        v.fit(x, y, lr=0.0)
    with pytest.raises(ValueError, match="n_mc"):
        v.elbo(x, y, n_mc=0)
    with pytest.raises(ValueError, match="seed"):
        VariationalMLP(4, 8, 3, seed=True)
    with pytest.raises(ValueError, match="beta"):
        VariationalMLP(4, 8, 3, beta="0.2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="init_rho"):
        VariationalMLP(4, 8, 3, init_rho=False)


@pytest.mark.parametrize(
    ("method", "kwargs", "message"),
    (
        ("elbo", {"n_mc": True}, "n_mc"),
        ("elbo", {"kl_weight": "1"}, "kl_weight"),
        ("fit", {"n_steps": True}, "n_steps"),
        ("fit", {"lr": "0.1"}, "lr"),
        ("predict_proba", {"n_samples": True}, "n_samples"),
        ("predict_proba", {"deterministic": "yes"}, "deterministic"),
    ),
)
def test_variational_mlp_rejects_coercive_runtime_controls(
    method,
    kwargs,
    message,
) -> None:
    v = VariationalMLP(2, 3, 2)
    x = torch.ones(2, 2)
    y = torch.eye(2)
    callable_method = getattr(v, method)
    with pytest.raises(ValueError, match=message):
        if method in {"elbo", "fit"}:
            callable_method(x, y, **kwargs)
        else:
            callable_method(x, **kwargs)


def test_variational_beta_zero_recovers_nll() -> None:
    v = VariationalMLP(2, 3, 3, seed=0, beta=0.0)
    probs = torch.tensor([[0.8, 0.15, 0.05], [0.1, 0.2, 0.7]])
    y = torch.eye(3)[torch.tensor([0, 2])]
    expected = -torch.log(torch.tensor([0.8, 0.7]))
    assert torch.allclose(v.beta_loss(probs, y), expected, atol=1e-7)
    assert torch.isfinite(v.beta_loss(probs, y, beta=1e-9)).all()


def test_closed_form_kl_matches_independent_per_weight_reference() -> None:
    v = VariationalMLP(3, 5, 2, seed=0, init_rho=-1.0)
    klsum = v.kl_to_prior().detach().item()
    ref = 0.0
    for mu, rho in zip(v._mus, v._rhos, strict=True):
        var = torch.nn.functional.softplus(rho) ** 2
        for m, vv in zip(mu.flatten().tolist(), var.flatten().tolist()):
            ref += gaussian_kl_reference(m, vv)
    assert abs(klsum - ref) < 1e-6
    assert klsum >= 0.0


def test_kl_is_zero_at_the_prior_and_positive_off_it() -> None:
    v = VariationalMLP(2, 3, 2, seed=0)
    # Set q == prior: every mu = 0 and softplus(rho) = sqrt(prior_var).
    target_sigma = math.sqrt(PRIOR_VAR_DEFAULT)
    rho_star = math.log(math.expm1(target_sigma))  # softplus(rho_star) == target_sigma
    with torch.no_grad():
        for mu in v._mus:
            mu.zero_()
        for rho in v._rhos:
            rho.fill_(rho_star)
    assert abs(v.kl_to_prior().detach().item()) < 1e-9
    # Perturb one mean off zero -> KL strictly positive.
    with torch.no_grad():
        v._mu_W1[0, 0] += 0.5
    assert v.kl_to_prior().detach().item() > 1e-6


def test_elbo_separates_data_and_kl_terms_exactly() -> None:
    v = VariationalMLP(3, 4, 2, seed=0, init_rho=-1.0)
    x = torch.randn(6, 3, generator=torch.Generator().manual_seed(3))
    y = torch.zeros(6, 2)
    y[torch.arange(6), torch.randint(0, 2, (6,), generator=torch.Generator().manual_seed(4))] = 1.0
    kl = v.kl_to_prior().detach().item()
    # Reset the reparameterization generator before each call so the MC data
    # term is IDENTICAL; the elbo difference is then exactly kl_weight * KL.
    v._gen.manual_seed(11)
    e0 = v.elbo(x, y, n_mc=3, kl_weight=0.0).detach().item()
    v._gen.manual_seed(11)
    e1 = v.elbo(x, y, n_mc=3, kl_weight=1.0).detach().item()
    assert abs((e1 - e0) - kl) < 1e-6
    # kl_weight=0 is a pure MC data term -> a finite beta-loss value.
    assert math.isfinite(e0)


def test_fit_reduces_the_elbo_on_separable_data() -> None:
    v = VariationalMLP(2, 6, 2, seed=0, init_rho=-2.0, beta=0.2)
    g = torch.Generator().manual_seed(0)
    n = 40
    x0 = torch.randn(n // 2, 2, generator=g) + torch.tensor([2.0, 2.0])
    x1 = torch.randn(n // 2, 2, generator=g) + torch.tensor([-2.0, -2.0])
    x = torch.cat([x0, x1])
    y = torch.zeros(n, 2)
    y[: n // 2, 0] = 1.0
    y[n // 2 :, 1] = 1.0
    hist = v.fit(x, y, n_steps=120, lr=0.02, n_mc=2, kl_weight=1e-3)
    # The optimizer makes real progress (start-to-end decrease on a smoothed view).
    assert min(hist[-10:]) < hist[0]


def test_cavity_export_load_and_fit_are_protocol_bound() -> None:
    v = VariationalMLP(2, 3, 2, seed=0, init_rho=-1.5)
    exported = v.to_diagonal_gaussian()
    clone = VariationalMLP(2, 3, 2, seed=9, init_rho=-4.0)
    clone.load_diagonal_gaussian(exported)
    assert clone.to_diagonal_gaussian().mean.tolist() == pytest.approx(exported.mean.tolist())
    assert clone.to_diagonal_gaussian().variance.tolist() == pytest.approx(exported.variance.tolist())
    assert clone.kl_to_reference(exported).detach().item() < 1e-6
    x = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    y = torch.eye(2)
    history = clone.fit_from_cavity(
        DiagonalGaussian(exported.mean, exported.variance),
        x,
        y,
        n_steps=2,
        n_mc=1,
        beta=0.2,
    )
    assert len(history) == 2
    assert all(math.isfinite(value) for value in history)


def test_cavity_protocol_rejects_wrong_dimension() -> None:
    v = VariationalMLP(2, 3, 2, seed=0)
    with pytest.raises(ValueError, match="dimension"):
        v.load_diagonal_gaussian(DiagonalGaussian([0.0], [1.0]))
