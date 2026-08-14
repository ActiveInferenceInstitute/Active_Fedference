"""Numerical invariants for the Active Fedference source-mechanism analogues.

Pure-compute checks (zero I/O, no ``infrastructure.*`` imports — layer
contract) that validate the *actual* numerical behavior of the locked FedGVI /
active-inference core. Each ``check_*`` function returns a list of
:class:`InvariantResult` witness records; :func:`all_invariants` runs the lot.
The analysis workflow serialises these to JSON and the test suite asserts on
them directly.

The four invariants pin the mathematical hinges of the project:

* **pmf normalization** — every consensus the federated server emits
  (:func:`fedference.aggregation.log_linear_pool` and
  :func:`fedference.aggregation.robust_aggregate`) is a categorical pmf:
  non-negative and summing to one. The log-linear pool is the documented
  categorical Eq. 7 specialization, not a complete source protocol.
* **robust == naive at robustness 0** — ``robust_aggregate(..., robustness=0)``
  is bit-identical to the naive project log-linear pool. This is the defining
  project-local identity of :mod:`fedference.aggregation`; the separate
  Friston comparison is a categorical specialization, not a full protocol
  reconstruction.
* **EFE identity** — the closed-form Expected Free Energy decomposition
  (:func:`fedference.expected_free_energy.decompose`, Friston et al. 2024
  Eq. 2) satisfies ``(risk + ambiguity) + (pragmatic + epistemic) == 0`` to
  floating-point tolerance.
* **KL monotonicity** — as an agent accumulates conjugate Dirichlet counts
  (:func:`fedference.dirichlet_learning.learn_likelihood`, related to Friston
  Eq. 12 / Fig. 7) the divergence ``KL(true A || learned A)`` declines monotonically:
  the agent "acquires the language" of its shared world.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fedference.aggregation import log_linear_pool, robust_aggregate
from fedference.dirichlet_learning import learn_likelihood
from fedference.expected_free_energy import EFE_IDENTITY_ATOL, decompose
from fedference.pomdp import N_LOCATIONS, build_sentinel_world

ArrayF = np.ndarray
_EPS = 1e-12


@dataclass
class InvariantResult:
    """Witness record for one numerical invariant.

    ``kind`` in {``"equal"``, ``"le"``, ``"ge"``, ``"nonneg"``,
    ``"monotone_decreasing"``}; ``passed`` is the boolean verdict derived from
    ``actual`` / ``expected`` / ``tol``.
    """

    name: str
    kind: str
    actual: Any
    expected: Any = None
    tol: float = 1e-9
    description: str = ""
    passed: bool = True
    extra: dict = field(default_factory=dict)


def _colony(seed: int, n_agents: int, n_states: int) -> ArrayF:
    """Build a seeded colony of soft heterogeneous beliefs peaked on a truth."""
    rng = np.random.default_rng(seed)
    true_state = int(rng.integers(0, n_states))
    colony = np.empty((n_agents, n_states), dtype=np.float64)
    for n in range(n_agents):
        conf = float(np.clip(0.5 + rng.uniform(-0.1, 0.1), 0.05, 0.95))
        belief = np.full(n_states, (1.0 - conf) / (n_states - 1), dtype=np.float64)
        belief[true_state] = conf
        colony[n] = belief
    return colony


def check_pmf_normalization(
    seed: int = 0, *, n_agents: int = 7, robustness: float = 1.3
) -> list[InvariantResult]:
    """Every fused consensus is a valid categorical pmf (non-negative, sums to 1)."""
    n_states = int(N_LOCATIONS)
    colony = _colony(seed, n_agents, n_states)
    naive = log_linear_pool(colony)
    robust = robust_aggregate(colony, robustness=robustness).consensus

    out: list[InvariantResult] = []
    for label, consensus in (("naive", naive), ("robust", robust)):
        mass = float(consensus.sum())
        min_entry = float(consensus.min())
        out.append(
            InvariantResult(
                name=f"consensus_sums_to_one_{label}",
                kind="equal",
                actual=mass,
                expected=1.0,
                tol=1e-9,
                description=f"{label} consensus mass == 1 (categorical pmf)",
                passed=abs(mass - 1.0) <= 1e-9,
            )
        )
        out.append(
            InvariantResult(
                name=f"consensus_nonnegative_{label}",
                kind="nonneg",
                actual=min_entry,
                expected=0.0,
                tol=0.0,
                description=f"{label} consensus has no negative entries",
                passed=min_entry >= 0.0,
            )
        )
    return out


def check_robust_recovers_naive(
    seed: int = 0, *, n_agents: int = 7
) -> list[InvariantResult]:
    """``robust_aggregate(robustness=0)`` is bit-identical to ``log_linear_pool``."""
    n_states = int(N_LOCATIONS)
    colony = _colony(seed, n_agents, n_states)
    naive = log_linear_pool(colony)
    robust0 = robust_aggregate(colony, robustness=0.0).consensus
    gap = float(np.max(np.abs(naive - robust0)))
    return [
        InvariantResult(
            name="robust_equals_naive_at_robustness_zero",
            kind="equal",
            actual=gap,
            expected=0.0,
            tol=1e-12,
            description=(
                "max |log_linear_pool - robust_aggregate(robustness=0)| == 0 "
                "(project-local zero-robustness recovery; the Eq. 7 relation "
                "is a documented categorical specialization)"
            ),
            passed=gap <= 1e-12,
        )
    ]


def check_efe_identity(seed: int = 0) -> list[InvariantResult]:
    """EFE decomposition identity: ``(risk+ambiguity)+(pragmatic+epistemic)==0``."""
    rng = np.random.default_rng(seed)
    world = build_sentinel_world(rng, acuity=0.8)
    A = np.asarray(world["A"][0], dtype=np.float64)  # type: ignore[index]
    B = np.asarray(world["B"][0], dtype=np.float64)  # type: ignore[index]
    C = np.asarray(world["C"][0], dtype=np.float64).ravel()  # type: ignore[index]
    prior = np.asarray(world["D"][0], dtype=np.float64).ravel()  # type: ignore[index]
    terms = decompose(A, B, C, prior, policy=(0, 2, 1))
    residual = float(terms.identity_residual)
    return [
        InvariantResult(
            name="efe_decomposition_identity",
            kind="equal",
            actual=abs(residual),
            expected=0.0,
            tol=EFE_IDENTITY_ATOL,
            description=(
                "(risk + ambiguity) + (pragmatic_value + epistemic_value) == 0 "
                "(Friston et al. 2024 Eq. 2, closed form)"
            ),
            passed=abs(residual) <= EFE_IDENTITY_ATOL,
        )
    ]


def check_kl_monotonicity(
    seed: int = 0, *, num_steps: int = 24, n_states: int = 4
) -> list[InvariantResult]:
    """KL(true A || learned A) declines monotonically as Dirichlet counts accrue."""
    rng = np.random.default_rng(seed)
    target = np.full((n_states, n_states), 0.05, dtype=np.float64)
    np.fill_diagonal(target, 1.0)
    target = target / target.sum(axis=0, keepdims=True)
    result = learn_likelihood(target, num_steps, count_scale=8.0, rng=rng)
    traj = [float(v) for v in result.kl_trajectory]
    monotone = all(traj[i] >= traj[i + 1] - 1e-9 for i in range(len(traj) - 1))
    declined = bool(traj and traj[0] > traj[-1])
    return [
        InvariantResult(
            name="kl_learning_curve_monotone_decreasing",
            kind="monotone_decreasing",
            actual=traj,
            tol=1e-9,
            description=(
                "KL(true A || learned A) is monotone non-increasing and strictly "
                "declines overall (source-mechanism analogue to Friston Eq. 12 / "
                "Fig. 7 language acquisition)"
            ),
            passed=monotone and declined,
        )
    ]


def all_invariants(seed: int = 0) -> list[InvariantResult]:
    """Every fedference invariant the analysis report should display."""
    out: list[InvariantResult] = []
    out.extend(check_pmf_normalization(seed))
    out.extend(check_robust_recovers_naive(seed))
    out.extend(check_efe_identity(seed))
    out.extend(check_kl_monotonicity(seed))
    return out


def write_invariants_report(
    project_root: Path, *, seed: int = 0
) -> tuple[Path, bool]:
    """Run :func:`all_invariants` and serialise the witnesses to JSON.

    Writes ``output/reports/invariants.json`` (a list of witness records plus an
    ``all_passed`` summary). Pure stdlib I/O — no ``infrastructure.*`` import, so
    the layer contract of this module is preserved. Returns the written path and
    the overall pass verdict for the calling script's exit code.
    """
    results = all_invariants(seed)
    all_passed = all(r.passed for r in results)
    payload = {
        "seed": int(seed),
        "all_passed": bool(all_passed),
        "n_invariants": len(results),
        "invariants": [asdict(r) for r in results],
    }
    reports = project_root / "output" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out_path = reports / "invariants.json"
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{out_path.name}.", suffix=".tmp", dir=reports)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, out_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return out_path, all_passed


__all__ = [
    "InvariantResult",
    "all_invariants",
    "check_efe_identity",
    "check_kl_monotonicity",
    "check_pmf_normalization",
    "check_robust_recovers_naive",
    "write_invariants_report",
]
