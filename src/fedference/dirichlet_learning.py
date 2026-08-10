"""Conjugate Dirichlet likelihood learning (Friston et al., 2024, Eqs. 9-12).

Active-inference agents learn the parameters of their generative model, not just
plan with them. The likelihood matrix ``A`` (shape ``(n_o, n_s)``, columns indexed
by hidden state and summing to one) is given a Dirichlet prior with concentration
``a`` over each column, updated conjugately by accumulating observation-state
co-occurrence counts (Friston et al., 2024, *Federated inference and belief
sharing*):

    a  <-  a + counts                         (Eq. 9-11, conjugate Dirichlet update)
    E[A] = a / sum_o(a)                        (column-normalized expected likelihood)

We drive the update with the *expected sufficient statistics* under the true model
(a fixed count batch ``count_scale * target_A`` per step), optionally jittered by a
seeded RNG. As the concentrations accumulate, the expected likelihood ``E[A]``
converges to the data-generating ``target_A``; we measure convergence by the
per-column KL divergence summed over hidden states,

    KL(target || learned) = sum_s sum_o A[o, s] * ln( A[o, s] / E[A][o, s] ),

which decreases monotonically toward zero (the standard Bayes / KL fixed point).

**The eta forgetting hyperprior (Eq. 12).** Friston et al. temper the accumulating
counts with a forgetting / decay hyperprior so the agent does not become infinitely
confident: before each conjugate addition the running concentration is decayed by a
factor that drives the *total* concentration mass toward the asymptote ``eta``.
Concretely, with total mass ``T = sum(a)`` over the whole matrix, a decay
``a <- a * (eta - batch_total) / T`` (clamped to keep ``a`` positive) applied before
adding the new batch makes the post-update total a fixed point at ``eta``: once
``sum(a) == eta`` the decay exactly cancels the incoming batch mass, so the sum of
counts saturates at ``eta`` rather than growing without bound. With ``eta = None``
the classical unbounded accumulation is recovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.float64]

DEFAULT_PRIOR_CONCENTRATION: float = 1.0
DEFAULT_COUNT_SCALE: float = 10.0
# A learned likelihood is "converged" when KL(true || learned) drops below this.
CONVERGENCE_KL_ATOL: float = 1e-2
_EPS = 1e-12


@dataclass(frozen=True)
class DirichletLearningResult:
    """Trajectory of a Dirichlet likelihood-learning run.

    ``kl_trajectory[k]`` is ``KL(target A || learned A)`` after ``k`` count
    batches have been applied (``num_steps + 1`` points): index 0 is the prior
    (largest KL), the last index is the state after the final batch, and the
    sequence is monotonically non-increasing. ``expected_a`` is the final learned (column-
    normalized) likelihood; ``concentration_totals[k]`` is ``sum(a)`` at step k.
    """

    kl_trajectory: tuple[float, ...]
    expected_a: ArrayF
    concentration: ArrayF
    concentration_totals: tuple[float, ...]
    num_states: int
    num_obs: int
    count_scale: float
    prior_concentration: float
    eta: float | None = None
    _meta: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def final_kl(self) -> float:
        """KL to the target likelihood after the final count batch (last point)."""
        return self.kl_trajectory[-1]

    @property
    def initial_kl(self) -> float:
        """KL to the target likelihood at the flat-prior start (first point)."""
        return self.kl_trajectory[0]

    @property
    def total_count(self) -> float:
        """Total Dirichlet concentration mass ``sum(a)`` after the final update."""
        return float(np.sum(self.concentration))

    @property
    def is_monotone_decreasing(self) -> bool:
        """True if the KL trajectory never rises (within 1e-12) batch to batch."""
        return all(
            self.kl_trajectory[i] >= self.kl_trajectory[i + 1] - 1e-12
            for i in range(len(self.kl_trajectory) - 1)
        )

    @property
    def steps_to_converge(self) -> int:
        """First recorded step whose KL is below :data:`CONVERGENCE_KL_ATOL`.

        Returns ``len(kl_trajectory)`` if convergence is never reached within the
        recorded horizon (the contract is explicit and total).
        """
        for index, value in enumerate(self.kl_trajectory):
            if value < CONVERGENCE_KL_ATOL:
                return index
        return len(self.kl_trajectory)


def _as_likelihood(target_a: Any) -> ArrayF:
    """Coerce ``target_a`` to a column-normalized ``(n_o, n_s)`` float matrix."""
    arr = np.asarray(target_a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("target_A must be a 2-D (n_o, n_s) likelihood matrix")
    if arr.size == 0:
        raise ValueError("target_A is empty")
    if np.any(arr < -1e-9):
        raise ValueError("target_A has negative entries")
    arr = np.clip(arr, 0.0, None)
    totals = arr.sum(axis=0, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("every column of target_A must have positive mass")
    return arr / totals


def expected_likelihood(concentration: ArrayF) -> ArrayF:
    """Expected likelihood ``E[A] = a / sum_o(a)`` (column-normalized).

    The Dirichlet posterior mean of each likelihood column is its concentration
    vector normalized over outcomes.
    """
    a = np.asarray(concentration, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("concentration must be a 2-D (n_o, n_s) matrix")
    column_totals = a.sum(axis=0, keepdims=True)
    if np.any(column_totals <= 0.0):
        raise ValueError("every concentration column must have positive mass")
    return a / column_totals


def _kl_columns(target_a: ArrayF, learned_a: ArrayF) -> float:
    """``sum_s KL(target_a[:, s] || learned_a[:, s])`` in nats (>= 0).

    The learned likelihood always has full support (Dirichlet prior > 0), so the
    divergence is finite; ``target_a`` zeros contribute nothing (0 * log 0 == 0).

    Vectorized: avoids Python loops over states and observations, computing the
    full KL sum in a single NumPy expression.
    """
    # Mask out zero entries in target to handle 0 * log 0 = 0 correctly.
    mask = target_a > 0.0
    log_ratio = np.where(mask, np.log(np.where(mask, target_a, 1.0) / learned_a), 0.0)
    return float(np.sum(target_a * log_ratio))


def learn_likelihood(
    target_A: Any,
    num_steps: int,
    *,
    count_scale: float = DEFAULT_COUNT_SCALE,
    prior_concentration: float = DEFAULT_PRIOR_CONCENTRATION,
    eta: float | None = None,
    rng: np.random.Generator | None = None,
) -> DirichletLearningResult:
    """Learn the likelihood ``A`` via conjugate Dirichlet updates (Eqs. 9-12).

    Parameters
    ----------
    target_A:
        Data-generating likelihood, shape ``(n_o, n_s)`` (columns are per-state
        outcome pmfs). Re-normalized on entry.
    num_steps:
        Number of conjugate count batches to apply (>= 1). One KL value is
        recorded per step, *before* that step's batch is applied (index 0 is the
        prior), giving a length-``num_steps`` trajectory.
    count_scale:
        Mass of the expected-sufficient-statistics batch ``count_scale * target_A``
        added each step. Must be positive.
    prior_concentration:
        Flat Dirichlet prior placed on every entry of ``a``. Must be positive.
    eta:
        Forgetting hyperprior (Eq. 12). When set (> 0), the running concentration
        is decayed before each conjugate addition so the *total* count mass
        ``sum(a)`` saturates at ``eta`` instead of growing without bound. When
        ``None`` the classical unbounded accumulation is used.
    rng:
        Optional seeded ``numpy.random.Generator`` for deterministic multiplicative
        jitter on each count batch (the empirical sufficient statistics fluctuate
        around their expectation). When ``None`` the exact expected counts are used,
        making the run fully deterministic.

    Returns
    -------
    DirichletLearningResult
        With ``kl_trajectory`` (monotone non-increasing toward 0) and
        ``expected_a`` (the final learned likelihood).
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if count_scale <= 0.0:
        raise ValueError("count_scale must be positive")
    if prior_concentration <= 0.0:
        raise ValueError("prior_concentration must be positive")
    if eta is not None and eta <= 0.0:
        raise ValueError("eta must be positive when set")

    true_a = _as_likelihood(target_A)
    n_o, n_s = true_a.shape
    base_batch = count_scale * true_a  # expected sufficient statistics
    batch_total = float(np.sum(base_batch))

    if eta is not None and eta <= batch_total:
        raise ValueError(
            f"eta ({eta}) must exceed the per-step batch mass ({batch_total}) "
            "so a positive decayed total exists"
        )

    concentration: ArrayF = np.full((n_o, n_s), float(prior_concentration), dtype=np.float64)

    kls: list[float] = []
    totals: list[float] = []
    for _ in range(num_steps):
        learned = expected_likelihood(concentration)
        kls.append(_kl_columns(true_a, learned))
        totals.append(float(np.sum(concentration)))

        # Draw this step's count batch (deterministic unless an RNG is given).
        if rng is None:
            batch = base_batch
        else:
            jitter = rng.uniform(0.95, 1.05, size=base_batch.shape)
            batch = base_batch * jitter

        if eta is None:
            concentration = concentration + batch
        else:
            # Eq. 12 forgetting: decay the running mass toward the asymptote so
            # the post-addition total is a fixed point at eta. With deterministic
            # batches sum(batch) == batch_total, so once sum(a) == eta the decay
            # exactly offsets the incoming mass and the total stays at eta.
            current_total = float(np.sum(concentration))
            target_pre_add = max(eta - float(np.sum(batch)), _EPS)
            decay = target_pre_add / current_total
            concentration = concentration * decay + batch

    # Record the post-final-batch point: without it, ``final_kl`` silently
    # reports the state BEFORE the last batch (off-by-one caught in review).
    kls.append(_kl_columns(true_a, expected_likelihood(concentration)))
    totals.append(float(np.sum(concentration)))

    return DirichletLearningResult(
        kl_trajectory=tuple(kls),
        expected_a=expected_likelihood(concentration),
        concentration=concentration,
        concentration_totals=tuple(totals),
        num_states=int(n_s),
        num_obs=int(n_o),
        count_scale=float(count_scale),
        prior_concentration=float(prior_concentration),
        eta=None if eta is None else float(eta),
        _meta={"deterministic": rng is None},
    )
