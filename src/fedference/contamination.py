"""Adversarial / miscalibrated sentinel models — the contamination source.

FedGVI's robust client mechanism (Mildner et al., 2025, arXiv:2502.00846) is
motivated by misspecification or contamination under the source theorem's
assumptions. This module constructs bad broadcasts for finite server-side
experiments. It is the experimental complement of
:mod:`fedference.aggregation`'s ``robust_aggregate``: contaminate a belief here,
feed it to the server rules, and measure whether influence is suppressed in
the declared regime. Suppression is an outcome, not an invariant.

In Friston et al. (2024) every sentinel reports a posterior over a shared latent
(e.g. a predator's location). A *healthy* sentinel reports a well-calibrated
categorical; a contaminated one reports something corrupted. We model three
canonical corruptions, each parameterised by a single ``rate`` in ``[0, 1]`` so
that the experiments sweep one knob:

* ``confident_wrong`` — convex-mix the belief toward a one-hot spike on a wrong
  state: ``(1 - rate) * belief + rate * onehot(wrong_state)``. This is the
  adversarial sentinel that is *sure* and *mistaken* — exactly the agent robust
  aggregation must reject. At ``rate = 1`` it is a pure delta on ``wrong_state``.

* ``label_noise`` — convex-mix the belief toward a fixed noisy categorical drawn
  once from a Dirichlet(1) (a random but valid pmf): the miscalibrated sentinel
  whose report is partly random. ``(1 - rate) * belief + rate * noise``.

* ``uniform`` — flatten toward the maximum-entropy uniform pmf:
  ``(1 - rate) * belief + rate * (1/n_s)``. The apathetic / saturated sentinel
  that has lost all information. At ``rate = 1`` it reports uniform.

* ``byzantine`` — a *multiplicative* targeted boost in log space:
  ``softmax(log belief + rate * tilt * onehot(target_state))``. Unlike the
  additive convex mixes above, a coordinated Byzantine adversary tilts the
  log-odds toward a chosen ``target_state``, so the corruption compounds with the
  belief's own shape rather than overwriting it — the canonical targeted poisoning
  attack against a product-of-experts pool.

* ``drift`` — a *slowly-moving* bias that grows across communication rounds:
  ``(1 - rate * phase) * belief + rate * phase * onehot(target_state)`` with
  ``phase = round_index / (n_rounds - 1)``. The first round is clean
  (``phase = 0``) and the bias creeps in linearly — the stealthy sentinel whose
  miscalibration only becomes confident late, defeating any one-shot screen.

All five share the contract that makes the limit clean and gives the suite its
anchor (**ISC-26**): at ``rate = 0`` every kind returns the input belief
unchanged (``drift`` additionally returns it unchanged on the first round), so
contamination is a strict, continuous departure from the uncorrupted Friston
belief-share — never a discontinuity.
"""

from __future__ import annotations

import numpy as np

ArrayF = np.ndarray
_EPS = 1e-12

_KINDS = ("confident_wrong", "label_noise", "uniform", "byzantine", "drift")

#: Log-odds tilt strength for the multiplicative ``byzantine`` attack at
#: ``rate = 1``; large enough to dominate a soft product-of-experts pool, finite
#: so the attack stays a continuous departure rather than a hard one-hot veto.
_BYZANTINE_TILT: float = 8.0


def _as_pmf(belief: ArrayF) -> ArrayF:
    """Coerce ``belief`` to a clipped, renormalized 1-D probability vector."""
    arr: ArrayF = np.asarray(belief, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("belief is empty")
    if np.any(arr < -1e-9):
        raise ValueError("belief has negative entries")
    arr = np.clip(arr, _EPS, None)  # clip floors every entry, so sum > 0 always
    return arr / arr.sum()


def contaminate(
    belief: ArrayF,
    *,
    kind: str = "confident_wrong",
    rate: float = 0.0,
    rng: np.random.Generator,
    wrong_state: int | None = None,
    target_state: int | None = None,
    round_index: int = 0,
    n_rounds: int = 1,
) -> ArrayF:
    """Corrupt a sentinel belief into an adversarial / miscalibrated report.

    Parameters
    ----------
    belief : 1-D categorical pmf (need not be exactly normalized; coerced).
    kind : one of ``'confident_wrong'``, ``'label_noise'``, ``'uniform'``,
        ``'byzantine'`` or ``'drift'``.
    rate : contamination strength in ``[0, 1]``. ``rate = 0`` returns the input
        belief unchanged (ISC-26); ``rate = 1`` is the fully-corrupted limit.
    rng : a ``np.random.Generator`` (e.g. ``np.random.default_rng(seed)``).
        Required and used only by ``label_noise`` to draw the noisy categorical;
        passed explicitly so every contamination is reproducible.
    wrong_state : index of the wrong state for ``confident_wrong``. If ``None``
        the argmax-avoiding default is the state with the *least* current mass.
    target_state : adversary-chosen state for ``byzantine`` / ``drift``. If
        ``None`` it defaults to the least-mass state (the same argmax-avoiding
        choice as ``wrong_state``), so a single ``wrong_state`` argument can drive
        all targeted kinds.
    round_index, n_rounds : the communication round and round budget for
        ``drift``. The drift phase is ``round_index / (n_rounds - 1)`` (``0`` for
        a single round), so ``round_index = 0`` is always clean and the bias
        grows linearly across rounds. Ignored by the other kinds.

    Returns
    -------
    A fresh normalized pmf of the same length as ``belief``.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose one of {_KINDS}")
    if not (0.0 <= rate <= 1.0):
        raise ValueError("rate must lie in [0, 1]")
    if rng is None:  # explicit: determinism contract forbids a global default
        raise ValueError("rng is required (pass np.random.default_rng(seed))")
    if n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")
    if not 0 <= round_index < max(n_rounds, 1):
        raise ValueError("round_index must lie in [0, n_rounds)")

    p = _as_pmf(belief)
    n_s = p.shape[0]

    # ISC-26 anchor: identity at rate == 0, regardless of kind.
    if rate == 0.0:
        return p

    def _resolve(idx: int | None) -> int:
        chosen = int(np.argmin(p)) if idx is None else int(idx)
        if not (0 <= chosen < n_s):
            raise ValueError("wrong_state out of range (target/wrong_state)")
        return chosen

    # --- multiplicative targeted attack (compounds with the belief shape) ----
    if kind == "byzantine":
        tilt = rate * _BYZANTINE_TILT
        log_p = np.log(p)
        log_p[_resolve(target_state)] += tilt
        log_p -= log_p.max()
        boosted = np.exp(log_p)
        return boosted / boosted.sum()

    # --- additive convex mixes (confident_wrong / label_noise / uniform / drift)
    if kind == "confident_wrong":
        target = np.zeros(n_s, dtype=np.float64)
        target[_resolve(wrong_state)] = 1.0
        effective = rate
    elif kind == "label_noise":
        target = rng.dirichlet(np.ones(n_s))
        effective = rate
    elif kind == "drift":
        phase = round_index / (n_rounds - 1) if n_rounds > 1 else 0.0
        effective = rate * phase
        if effective == 0.0:  # first round (or single round) is clean
            return p
        target = np.zeros(n_s, dtype=np.float64)
        target[_resolve(target_state)] = 1.0
    else:  # uniform
        target = np.full(n_s, 1.0 / n_s, dtype=np.float64)
        effective = rate

    mixed = (1.0 - effective) * p + effective * target
    mixed = np.clip(mixed, _EPS, None)
    return mixed / mixed.sum()
