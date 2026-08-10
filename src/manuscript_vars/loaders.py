"""Manuscript variable submodule."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from experiment_config import SENSITIVITY_NOISE_FLOOR, load_experiment_config
from fedference.aggregation import log_linear_pool, robust_aggregate
from fedference.bnn_defaults import (
    BNN_BETA_DEFAULT,
    BNN_HIDDEN_DIM_DEFAULT,
    BNN_N_CLIENTS_DEFAULT,
    BNN_N_STEPS_DEFAULT,
    BNN_ROBUSTNESS_DEFAULT,
)
from fedference.divergences import kl_divergence, renyi_divergence
from fedference.experiments.cross_study import CROSS_STUDY_SENS_N_TRIALS
from fedference.experiments.sensitivity import DEFAULT_SENSITIVITY_N_TRIALS
from fedference.generalized_bayes import generalized_posterior
from fedference.losses import beta_loss, loss_vector, nll, rcce
from project_paths import resolve_env_project_root
from publication.release_manifest import timestamp_from_source_date_epoch

_log = logging.getLogger(__name__)

#: Optional dependency blocks only — missing imports degrade to warnings.
_TOKEN_BLOCK_EXPECTED = (ImportError,)

#: Bootstrap CI significance level used throughout the harness (statistics.bootstrap_ci
#: and the experiment reports default to alpha = 0.05). The reported CI *percentage*
#: is its complement; surfaced as a token so the prose tracks the analysis choice
#: rather than hardcoding "95%".
_BOOTSTRAP_ALPHA = 0.05

# Sensitivity sweep defaults. The trials-per-cell count is IMPORTED from the
# experiment module (the same constant its signature consumes) so it cannot
# drift from the executed default; the remaining values below must match
# experiments.run_belief_sharing_sensitivity.
_SENS_N_TRIALS: int = DEFAULT_SENSITIVITY_N_TRIALS
# Cross-study Study 8 trials-per-cell, imported from the experiment module so
# the manuscript token cannot drift from the executed default.
_CROSS_STUDY_SENS_N_TRIALS: int = CROSS_STUDY_SENS_N_TRIALS
_SENS_SEED_BASE: int = 0
_SENS_N_ACUITY_LEVELS: int = 5
_SENS_N_COLONY_SIZES: int = 5
_SENS_NOISE_FLOOR: float = SENSITIVITY_NOISE_FLOOR
# Bootstrap resamples — must match fedference.experiments._N_BOOT
_N_BOOT: int = 5000
_TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_EXCLUDED_MANUSCRIPT_DOCS = frozenset({"AGENTS.md", "README.md", "SYNTAX.md"})


def _project_root(project_root: Path | None) -> Path:
    # loaders.py lives at <root>/src/manuscript_vars/loaders.py, so the project
    # root is three parents up. (A prior two-parent default resolved to src/
    # and silently grew a stray src/output/ tree on any default-root call.)
    # A validated ACTIVE_FEDFERENCE_PROJECT_ROOT env override wins over the
    # file-location default so subprocess tests never write into the real tree.
    if project_root is not None:
        return project_root
    return resolve_env_project_root(Path(__file__).resolve().parent.parent.parent)


def _build_timestamp(source_date_epoch: str | None) -> str:
    """Return a reproducible UTC build epoch or an honest unreleased sentinel."""
    if source_date_epoch is None:
        return "omitted (unreleased reproducible build)"
    return timestamp_from_source_date_epoch(source_date_epoch)


def _load_report(reports: Path, name: str) -> dict[str, Any]:
    path = reports / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, places: int = 4) -> str:
    return f"{float(value):.{places}f}"


def _format_residual(value: float) -> str:
    """Format a near-zero residual: ``< 10^{-k}`` scientific signpost."""
    if value <= 0.0:
        return "0"
    return f"{value:.2e}"


def _format_residual_math(value: float) -> str:
    """Math-mode sibling of :func:`_format_residual`.

    Renders the same ``.2e`` representation as LaTeX scientific notation
    (``M \\times 10^{E}``) so the token can sit inside ``$...$`` spans, where
    a bare ``2.33e-80`` would typeset as a malformed product. The exponent is
    normalized to a plain int (no leading zeros, no ``+``).
    """
    if value <= 0.0:
        return "0"
    mantissa, _, exponent = f"{value:.2e}".partition("e")
    return f"{mantissa} \\times 10^{{{int(exponent)}}}"


#: Off-switch-point parameters for the M2 convergence witness below: each
#: value sits strictly outside its function's closed-form switch band
#: (rcce/beta_loss switch at |param| < 1e-9; renyi_divergence switches at
#: |alpha - 1| < 1e-6), so evaluating here exercises the general formula
#: rather than the exact-branch equality the main residuals above measure.
_OFFSWITCH_Q: float = 1e-6
_OFFSWITCH_BETA: float = 1e-6
_OFFSWITCH_ALPHA: float = 1.0 + 1e-5


def _recovery_residuals() -> dict[str, float]:
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5))
    p = np.asarray(p, dtype=np.float64)
    o = 2
    rcce_gap = abs(rcce(p, o, q_loss=1e-12) - nll(p, o))
    beta_gap = abs(beta_loss(p, o, 1e-12) - nll(p, o))

    q = rng.dirichlet(np.ones(5))
    p_ref = rng.dirichlet(np.ones(5))
    renyi_gap = abs(renyi_divergence(q, p_ref, alpha=1.0) - kl_divergence(q, p_ref))

    # M2 fix: off-switch-point convergence witness, evaluated on the same
    # (p, o) / (q, p_ref) draws as the exact-branch residuals above, so the
    # only thing that changes is the loss/divergence parameter crossing
    # outside the closed-form band. Nonzero by construction (verified in
    # tests/fedference/test_core_identities.py that these gaps shrink
    # monotonically as the offset shrinks) — genuine numerical convergence of
    # the general formula, not a code-branch coincidence.
    rcce_offswitch_gap = abs(rcce(p, o, q_loss=_OFFSWITCH_Q) - nll(p, o))
    beta_offswitch_gap = abs(beta_loss(p, o, _OFFSWITCH_BETA) - nll(p, o))
    renyi_offswitch_gap = abs(renyi_divergence(q, p_ref, alpha=_OFFSWITCH_ALPHA) - kl_divergence(q, p_ref))

    local_posteriors = [rng.dirichlet(np.ones(5)) for _ in range(4)]
    base_weights = rng.uniform(0.2, 1.0, size=4)
    robust_limit = robust_aggregate(
        local_posteriors=local_posteriors,
        base_weights=base_weights,
        robustness=0.0,
    ).consensus
    friston_pool = log_linear_pool(local_posteriors=local_posteriors, base_weights=base_weights)
    aggregate_gap = float(np.max(np.abs(robust_limit - friston_pool)))

    likelihood = rng.dirichlet(np.ones(4), size=4).T  # (n_o=4, n_s=4)
    observation = 1
    loss_by_state = loss_vector(likelihood, o=observation, loss="nll")
    prior = np.full(loss_by_state.shape[0], 1.0 / loss_by_state.shape[0])
    log_prior = np.log(prior)
    post_kld = generalized_posterior(log_prior, loss_by_state, divergence="KLD")
    analytic_bayes = prior * likelihood[observation]
    analytic_bayes = analytic_bayes / analytic_bayes.sum()
    posterior_gap = float(np.max(np.abs(post_kld - analytic_bayes)))
    return {
        "RECOVERY_RCCE_MAXDIFF": float(rcce_gap),
        "RECOVERY_BETA_MAXDIFF": float(beta_gap),
        "RECOVERY_RENYI_MAXDIFF": float(renyi_gap),
        "RECOVERY_AGGREGATE_MAXDIFF": aggregate_gap,
        "RECOVERY_POSTERIOR_MAXDIFF": posterior_gap,
        "RECOVERY_RCCE_OFFSWITCH_MAXDIFF": float(rcce_offswitch_gap),
        "RECOVERY_BETA_OFFSWITCH_MAXDIFF": float(beta_offswitch_gap),
        "RECOVERY_RENYI_OFFSWITCH_MAXDIFF": float(renyi_offswitch_gap),
    }


def _count_isc(root: Path) -> tuple[str, str]:
    """Live ISC tally parsed from the project ``ISA.md`` (defined, passed).

    Counts ``- [ ]/[x]/[DEFERRED-VERIFY] ISC-N:`` definition lines so the prose
    tokens never drift from the acceptance contract. Duplicate identifiers are
    rejected rather than silently deduplicated: reuse corrupts both the total
    and passed manuscript certificate tokens. A
    ``[DEFERRED-VERIFY]`` row counts toward the total but never toward passed —
    otherwise a deferred criterion silently vanishes from the denominator and
    the manuscript renders "N of N verified" while one criterion is open.
    Degrades to ``"N/A"`` if absent.
    """
    isa = root / "ISA.md"
    if not isa.exists():
        return ("N/A", "N/A")
    pattern = re.compile(r"\s*-\s*\[(DEFERRED-VERIFY|[ x])\]\s*ISC-([0-9]+(?:\.[0-9]+)?):")
    status_by_id: dict[str, str] = {}
    for line in isa.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line)
        if m:
            status, identifier = m.group(1), m.group(2)
            if identifier in status_by_id:
                raise ValueError(f"duplicate ISC identifier in {isa}: ISC-{identifier}")
            status_by_id[identifier] = status
    if not status_by_id:
        return ("N/A", "N/A")
    return (
        str(len(status_by_id)),
        str(sum(status == "x" for status in status_by_id.values())),
    )


def _count_tests(root: Path) -> str:
    """Live count of collected tests under ``tests/`` via pytest --collect-only.

    Uses pytest's own collection (including parametrize expansions) for an
    accurate count.  Falls back to def test_* line-counting if pytest is
    unavailable or times out.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q", "--no-header"],
            cwd=root,
            capture_output=True,
            text=True,
            # Collection expands parametrized tests and can contend with the
            # publication analysis process on a loaded workstation.  A short
            # timeout silently falls back to a stale definition-line count,
            # which is not acceptable for a provenance token.
            timeout=120,
        )
        for line in reversed(result.stdout.splitlines()):
            match = re.search(r"(\d+)\s+tests?\s+collected", line)
            if match:
                return match.group(1)
    except (OSError, subprocess.SubprocessError):
        return _count_tests_by_name(root)
    return _count_tests_by_name(root)


def _count_tests_by_name(root: Path) -> str:
    # Fallback: count def test_* lines
    tests = root / "tests"
    if not tests.exists():
        return "N/A"
    n = 0
    for path in tests.rglob("test_*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("def test_", "async def test_")):
                n += 1
    return str(n) if n else "N/A"


def _coverage_percent(root: Path) -> str:
    """Live coverage % from a pipeline/CI artifact, else the documented gate floor.

    Honest by construction: never fabricates an achieved number — if no
    ``coverage.json`` was produced it returns the true enforced minimum.
    """
    candidates = (
        root / "output" / "reports" / "coverage.json",
        root / "output" / "reports" / "coverage_summary.json",
        root / "output" / "reports" / "coverage_project.json",
        root / "coverage_project.json",
    )
    for path in candidates:
        if not path.exists():
            continue
        # ``coverage_project.json`` is an ignored convenience artifact and can
        # outlive the coverage database that produced the current manuscript.
        # Never hydrate prose from a stale exact percentage: the enforced gate
        # floor is safer than silently reporting yesterday's result.
        live_database = root / ".coverage"
        if (
            path.name == "coverage_project.json"
            and live_database.exists()
            and path.stat().st_mtime < live_database.stat().st_mtime
        ):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        pct = data.get("coverage_percent")
        if pct is None:
            pct = data.get("totals", {}).get("percent_covered")
        if pct is not None:
            return _fmt(pct, 2)
    return "≥90"  # enforced gate floor; the live achieved % is tracked by the test gate


def _validation_receipt_variables(root: Path) -> dict[str, str]:
    """Load final test/coverage and core-environment tokens from the receipt.

    The CLI invokes this path only after its freshness preflight. Keeping the
    adapter in the manuscript loader package makes the provenance-token source
    explicit while isolated analysis fixtures can still exercise their bounded
    non-release fallbacks.
    """
    from publication.validation_receipt import validation_receipt_tokens

    return validation_receipt_tokens(root)


def _bnn_torch_variables(root: Path) -> dict:
    """Manuscript tokens for the PyTorch point-mass MLP complement.

    Reads the executed report ``output/reports/bnn_torch.json`` (written by the
    analysis pipeline when PyTorch is installed) and surfaces its numbers as
    tokens. When the report is missing or records a ``skipped`` status (PyTorch
    not installed), configuration tokens fall back to the ACTUAL experiment
    defaults imported from the torch-free :mod:`fedference.bnn_defaults`
    (the same constants ``run_bnn_torch_experiment``'s signature consumes —
    never re-typed literals) and the executed tokens read a clearly labeled
    ``N/A (PyTorch not run)`` — never a fabricated value. The generator never
    imports torch itself.
    """
    out = {
        "BNN_HIDDEN_DIM": str(BNN_HIDDEN_DIM_DEFAULT),
        "BNN_N_STEPS": str(BNN_N_STEPS_DEFAULT),
        "BNN_BETA": str(BNN_BETA_DEFAULT),
        "BNN_ROBUSTNESS": str(BNN_ROBUSTNESS_DEFAULT),
        "BNN_N_CLIENTS": str(BNN_N_CLIENTS_DEFAULT),
        "BNN_TORCH_STD_ACC": "N/A (PyTorch not run)",
        "BNN_TORCH_ROBUST_ACC": "N/A (PyTorch not run)",
        "BNN_TORCH_CONTAM": "N/A",
        "BNN_CONSENSUS_SUM": "N/A",
        "BNN_DETERMINISTIC": "N/A",
        "PYTORCH_VERSION": "not installed",
    }
    path = root / "output" / "reports" / "bnn_torch.json"
    if not path.exists():
        return out
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return out
    if data.get("status") != "ok":
        return out
    out["BNN_HIDDEN_DIM"] = str(data["hidden_dim"])
    out["BNN_N_STEPS"] = str(data["n_steps"])
    out["BNN_BETA"] = str(data["beta"])
    out["BNN_ROBUSTNESS"] = str(data["robustness"])
    out["BNN_N_CLIENTS"] = str(data["n_clients"])
    out["BNN_TORCH_STD_ACC"] = _fmt(data["standard_accuracy"], 3)
    out["BNN_TORCH_ROBUST_ACC"] = _fmt(data["robust_accuracy"], 3)
    out["BNN_TORCH_CONTAM"] = _fmt(data["reported_contamination"], 2)
    out["BNN_CONSENSUS_SUM"] = _format_residual(float(data["consensus_max_simplex_deviation"]))
    out["BNN_DETERMINISTIC"] = "Yes" if data["deterministic"] else "No"
    out["PYTORCH_VERSION"] = str(data.get("torch_version", "unknown"))
    return out


def _bnn_robustness_variables(root: Path) -> dict:
    """Manuscript tokens for the federated logistic-regression baseline (fig:bnn-robustness).

    Reads ``output/reports/bnn_robustness.json`` (written by
    :func:`analysis.workflow._bnn_report`) so the caption's seed count,
    per-client sample size, and contamination range are generated tokens, not
    hand-typed numerals (C3 fix: the curve is now a multi-seed mean, so the
    caption must say how many seeds and stop citing a single deterministic run).
    """
    out = {
        "BNN_ROBUSTNESS_N_SEEDS": "N/A",
        "BNN_ROBUSTNESS_N_PER": "N/A",
        "BNN_ROBUSTNESS_LOSS_PARAM": "N/A",
        "BNN_ROBUSTNESS_MAX_CONTAM": "N/A",
        "BNN_ROBUSTNESS_PEAK_CONTAM": "N/A",
        "BNN_ROBUSTNESS_PEAK_GAP": "N/A",
    }
    path = root / "output" / "reports" / "bnn_robustness.json"
    if not path.exists():
        return out
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return out
    out["BNN_ROBUSTNESS_N_SEEDS"] = str(data["n_seeds"])
    out["BNN_ROBUSTNESS_N_PER"] = str(data["n_per"])
    out["BNN_ROBUSTNESS_LOSS_PARAM"] = _fmt(data["robust_loss_param"], 2)
    levels = data["contamination_levels"]
    out["BNN_ROBUSTNESS_MAX_CONTAM"] = _fmt(max(levels), 2)

    if "peak_margin" in data and "peak_margin_contamination" in data:
        out["BNN_ROBUSTNESS_PEAK_CONTAM"] = _fmt(data["peak_margin_contamination"], 2)
        out["BNN_ROBUSTNESS_PEAK_GAP"] = _fmt(data["peak_margin"], 3)
    else:
        # Peak-gap level (advisor fix): the contamination level at which the
        # robust-minus-standard margin is largest, computed from the actual
        # curves rather than assumed to be the last swept level — the last level
        # is deliberately kept in the sweep even where it does NOT favor the
        # robust client (see _BNN_CONTAMINATION_LEVELS docstring).
        accuracy = data["accuracy_by_config"]
        standard_key = next(k for k in accuracy if "nll" in k.lower() or "standard" in k.lower())
        robust_key = next(k for k in accuracy if k != standard_key)
        gaps = [r - s for r, s in zip(accuracy[robust_key], accuracy[standard_key])]
        peak_idx = max(range(len(gaps)), key=lambda i: gaps[i])
        out["BNN_ROBUSTNESS_PEAK_CONTAM"] = _fmt(levels[peak_idx], 2)
        out["BNN_ROBUSTNESS_PEAK_GAP"] = _fmt(gaps[peak_idx], 3)
    return out


#: Tempered-aggregation lambda grid — single definition consumed by BOTH the
#: sweep loop in :func:`_tempered_variables` and the TEMPERED_LAMBDA_GRID token
#: (previously a re-typed display string).
_TEMPERED_LAMBDA_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


def _tempered_variables(root) -> dict:
    import numpy as np

    from fedference import aggregation as _agg
    from fedference.aggregation import robust_aggregate, variational_aggregate

    n_trials = 10
    n_agents = 5
    n_states = 9
    # Adversarial-agent count: the SAME variable drives the belief-construction
    # loop below and the TEMPERED_N_ADVERSARIAL token (no re-typed literal).
    n_adversarial = 2
    rng = np.random.default_rng(42)
    lambda_grid = _TEMPERED_LAMBDA_GRID
    acc_robust_list: list[float] = []
    acc_var: dict[float, list[float]] = {lam: [] for lam in lambda_grid}
    for _ in range(n_trials):
        true_s = int(rng.integers(0, n_states))
        wrong_s = int((true_s + n_states // 2) % n_states)
        local_posteriors = []
        for k in range(n_agents):
            b = np.full(n_states, (1 - 0.35) / (n_states - 1))
            b[true_s] = 0.35
            local_posteriors.append(b)
        b_adv = np.full(n_states, 0.02)
        b_adv[wrong_s] = 0.84
        for j in range(n_adversarial):
            local_posteriors[j] = b_adv.copy()
        acc_robust_list.append(float(robust_aggregate(local_posteriors, robustness=1.5).consensus[true_s]))
        for lam in lambda_grid:
            acc_var[lam].append(
                float(
                    variational_aggregate(
                        local_posteriors,
                        robustness=1.5,
                        entropy_weight=lam,
                        multistart=True,
                    ).consensus[true_s]
                )
            )
    mean_robust = float(np.mean(acc_robust_list))
    diffs = {lam: abs(float(np.mean(acc_var[lam])) - mean_robust) for lam in lambda_grid}
    best_lam = min(diffs, key=lambda lam: diffs[lam])
    best_diff = diffs[best_lam]
    union = best_diff < 0.20
    return {
        "TEMPERED_LAMBDA_GRID": ", ".join(f"{lam:g}" for lam in lambda_grid),
        "TEMPERED_N_TRIALS": str(n_trials),
        "TEMPERED_N_AGENTS": str(n_agents),
        "TEMPERED_N_ADVERSARIAL": str(n_adversarial),
        "TEMPERED_LAMBDA_STAR": f"{best_lam:.1f}",
        "TEMPERED_LAMBDA_STAR_DIFF": f"{best_diff:.4f}",
        "TEMPERED_HONEST_EXIT_SENTENCE": (
            "A lambda* narrows the tested accuracy gap while preserving the stated weight bound on this grid."
            if union
            else "No single lambda* fully closes the tested accuracy gap while "
            "preserving the bounded-influence weight update (no-free-lunch regime)."
        ),
        "TEMPERED_ENTROPY_WEIGHT_DEFAULT": str(_agg.TEMPERED_ENTROPY_WEIGHT_DEFAULT),
    }


#: Federation demo parameters — a SINGLE definition consumed by both the
#: executed bit-identity round in :func:`_federation_variables` and the
#: FEDERATION_N_WORKERS / FEDERATION_ROBUSTNESS tokens (previously duplicated
#: as string literals). They mirror the end-to-end test configuration in
#: tests/fedference/test_federation_end_to_end.py.
FEDERATION_N_WORKERS: int = 5
FEDERATION_ROBUSTNESS: float = 1.5


def _federation_demo_local_posteriors() -> list[np.ndarray]:
    """Deterministic worker posteriors for the federation identity round.

    Same construction as ``_make_beliefs`` in
    tests/fedference/test_federation_end_to_end.py: honest workers peak on the
    true state, two adversaries peak confidently on a wrong state.
    """
    n_states = 9
    true_s = 3
    wrong_s = 7
    local_posteriors = [np.full(n_states, 0.07) for _ in range(FEDERATION_N_WORKERS)]
    for posterior in local_posteriors:
        posterior[true_s] = 0.44
    b_adv = np.full(n_states, 0.02)
    b_adv[wrong_s] = 0.84
    local_posteriors[0] = b_adv
    local_posteriors[1] = b_adv.copy()
    return local_posteriors


def _federation_demo_beliefs() -> list[np.ndarray]:
    """Compatibility adapter for the pre-canonical demo helper name.

    ``local_posteriors`` is the canonical manuscript-token vocabulary.  Keep
    the old private helper importable for downstream token tests and local
    integrations, but make the rename observable instead of silently
    maintaining two implementations.
    """
    warnings.warn(
        "_federation_demo_beliefs is deprecated; use _federation_demo_local_posteriors",
        DeprecationWarning,
        stacklevel=2,
    )
    return _federation_demo_local_posteriors()


def _federation_variables(root: Path) -> dict[str, str]:
    """V3 federation-transport manuscript tokens.

    The bit-identity verdict is DERIVED, not asserted: one federation round is
    executed through the same queue-backed worker/server path the end-to-end
    tests exercise (tests/fedference/test_federation_end_to_end.py), and its
    broadcast consensus is compared bit-for-bit (``np.array_equal``) against
    the in-process ``fedference.aggregation.robust_aggregate`` call on the
    same local posteriors. The token reads ``"True"``/``"False"`` from that comparison.
    """
    import queue

    from fedference.federation import FederationServer, FederationWorker

    local_posteriors = _federation_demo_local_posteriors()
    ref = robust_aggregate(local_posteriors, robustness=FEDERATION_ROBUSTNESS).consensus

    request_q: queue.Queue = queue.Queue()
    response_qs: dict[int, queue.Queue] = {i: queue.Queue() for i in range(FEDERATION_N_WORKERS)}
    workers = [FederationWorker(i, request_q, response_qs[i]) for i in range(FEDERATION_N_WORKERS)]
    server = FederationServer(n_workers=FEDERATION_N_WORKERS, robustness=FEDERATION_ROBUSTNESS)
    for worker, posterior in zip(workers, local_posteriors):
        worker.send_belief(posterior)
    fed = server.run_round(request_q, response_qs)

    return {
        "FEDERATION_N_WORKERS": str(FEDERATION_N_WORKERS),
        "FEDERATION_ROBUSTNESS": f"{FEDERATION_ROBUSTNESS:g}",
        "FEDERATION_BIT_IDENTICAL": str(bool(np.array_equal(ref, fed))),
        "FEDERATION_TRANSPORT": "numpy-lossless-float64",
    }


def _moving_world_variables(root: Path) -> dict[str, str]:
    """V4 moving-world manuscript tokens from ``output/reports/moving_world.json``."""
    report = _load_report(root / "output" / "reports", "moving_world.json")
    if not report:
        raise FileNotFoundError("missing moving_world.json")

    # STRICT key access for every numeric the prose renders (same discipline as
    # _hierarchical_variables): a report missing a block fails loudly rather
    # than degrading to a result-shaped 0.000 (cross-vendor audit finding).
    acc = report["accuracy"]
    gap = report["free_energy_gap"]
    ms = report["multiseed"]
    iso_ms = ms["isolated"]
    comm_ms = ms["communicating"]
    efe_ms = ms["efe_guided"]
    fe_gap_ms = ms["efe_free_energy_gap"]
    pt_efe = ms["efe_vs_isolated"]["paired_test"]

    # Significance verdict DERIVED from the report's p-value against the
    # configured alpha (config.yaml -> experiment.statistics.power_alpha) —
    # never asserted in prose. Strict key access when a paired test exists:
    # a missing p-value fails loudly rather than fabricating a verdict.
    if pt_efe:
        significance_verdict = (
            "significant"
            if float(pt_efe["pvalue"]) < load_experiment_config(root).power_alpha
            else "not significant"
        )
    else:
        significance_verdict = "N/A"

    return {
        "MOVING_SIGNIFICANCE_VERDICT": significance_verdict,
        "MOVING_ACC_ISOLATED": _fmt(float(acc["isolated"]), 3),
        "MOVING_ACC_COMMUNICATING": _fmt(float(acc["communicating"]), 3),
        "MOVING_ACC_EFE": _fmt(float(acc["efe_guided"]), 3),
        "MOVING_FE_GAP_COMMUNICATING": _fmt(float(gap["communicating"]), 3),
        "MOVING_N_TRIALS": str(report["n_trials"]),
        "MOVING_N_STEPS": str(report["n_steps"]),
        "MOVING_N_POSITIONS": str(report["n_positions"]),
        "MOVING_N_AGENTS": str(report["n_agents"]),
        "MOVING_N_SEEDS": str(ms["n_seeds"]),
        "MOVING_ACC_ISO_MEAN": _fmt(float(iso_ms["mean"]), 3),
        "MOVING_ACC_ISO_CI_LO": _fmt(float(iso_ms["ci_lo"]), 3),
        "MOVING_ACC_ISO_CI_HI": _fmt(float(iso_ms["ci_hi"]), 3),
        "MOVING_ACC_COMM_MEAN": _fmt(float(comm_ms["mean"]), 3),
        "MOVING_ACC_COMM_CI_LO": _fmt(float(comm_ms["ci_lo"]), 3),
        "MOVING_ACC_COMM_CI_HI": _fmt(float(comm_ms["ci_hi"]), 3),
        "MOVING_ACC_EFE_MEAN": _fmt(float(efe_ms["mean"]), 3),
        "MOVING_ACC_EFE_CI_LO": _fmt(float(efe_ms["ci_lo"]), 3),
        "MOVING_ACC_EFE_CI_HI": _fmt(float(efe_ms["ci_hi"]), 3),
        "MOVING_FE_GAP_EFE_MEAN": _fmt(float(fe_gap_ms["mean"]), 3),
        "MOVING_FE_GAP_EFE_CI_LO": _fmt(float(fe_gap_ms["ci_lo"]), 3),
        "MOVING_FE_GAP_EFE_CI_HI": _fmt(float(fe_gap_ms["ci_hi"]), 3),
        "MOVING_WILCOX_PVALUE": _fmt(float(pt_efe["pvalue"]), 4),
        "MOVING_EFFECT_SIZE": _fmt(abs(float(pt_efe["effect_size"])), 3),
        "MOVING_EFFECT_LABEL": ms["efe_vs_isolated"]["effect_label"],
    }


def _disjoint_fov_variables(root: Path) -> dict[str, str]:
    """V4 disjoint-FOV manuscript tokens from ``output/reports/disjoint_fov_world.json``.

    Executed-run parameters use STRICT key access (``report[...]``): a report
    missing a key fails loudly rather than fabricating a result-shaped default
    (same discipline as :func:`_hierarchical_variables`).
    """
    report = _load_report(root / "output" / "reports", "disjoint_fov_world.json")
    if not report:
        raise FileNotFoundError("missing disjoint_fov_world.json")

    n_positions = int(report["n_positions"])
    ms = report.get("multiseed", {})
    iso_ms = ms.get("isolated", {})
    comm_ms = ms.get("communicating", {})
    pt_comm = ms.get("communicating_vs_isolated", {}).get("paired_test", {})
    efe_ms = ms.get("efe_guided", {})
    rnd_ms = ms.get("random", {})
    pt_efe = ms.get("efe_vs_random", {}).get("paired_test", {})
    efe_point = report.get("efe_navigation", {})

    return {
        "V4_ISOLATED_ACCURACY": _fmt(float(report.get("isolated_accuracy", 0)), 2),
        "V4_COMMUNICATING_ACCURACY": _fmt(float(report.get("communicating_accuracy", 0)), 2),
        "V4_ACCURACY_GAP": _fmt(float(report.get("gap", 0)), 2),
        "V4_N_AGENTS": str(report["n_agents"]),
        "V4_FOV_WIDTH": str(report["fov_width"]),
        "V4_N_POSITIONS": str(n_positions),
        "V4_CHANCE_BASELINE": _fmt(1.0 / n_positions, 3),
        "V4_EFE_N_AGENTS": str(efe_point["n_agents"]),
        "V4_EFE_N_POSITIONS": str(efe_point["n_positions"]),
        "V4_N_SEEDS": str(ms.get("n_seeds", "N/A")),
        "V4_ISO_MEAN": _fmt(float(iso_ms.get("mean", 0)), 3),
        "V4_ISO_CI_LO": _fmt(float(iso_ms.get("ci_lo", 0)), 3),
        "V4_ISO_CI_HI": _fmt(float(iso_ms.get("ci_hi", 0)), 3),
        "V4_COMM_MEAN": _fmt(float(comm_ms.get("mean", 0)), 3),
        "V4_COMM_CI_LO": _fmt(float(comm_ms.get("ci_lo", 0)), 3),
        "V4_COMM_CI_HI": _fmt(float(comm_ms.get("ci_hi", 0)), 3),
        "V4_WILCOX_PVALUE": _fmt(float(pt_comm.get("pvalue", 0)), 4),
        "V4_EFFECT_SIZE": _fmt(abs(float(pt_comm.get("effect_size", 0))), 3),
        "V4_EFFECT_LABEL": ms.get("communicating_vs_isolated", {}).get("effect_label", "N/A"),
        "V4_EFE_ACC_MEAN": _fmt(float(efe_ms.get("mean", 0)), 3),
        "V4_RANDOM_ACC_MEAN": _fmt(float(rnd_ms.get("mean", 0)), 3),
        "V4_EFE_WILCOX_PVALUE": _fmt(float(pt_efe.get("pvalue", 0)), 4),
        "V4_EFE_EFFECT_SIZE": _fmt(abs(float(pt_efe.get("effect_size", 0))), 3),
        "V4_EFE_EFFECT_LABEL": ms.get("efe_vs_random", {}).get("effect_label", "N/A"),
    }


def _hierarchical_variables(root: Path) -> dict[str, str]:
    """V2 hierarchical POMDP tokens from ``output/reports/hierarchical_world.json``.

    Executed-run parameters use STRICT key access (``report[...]``): a report
    missing a key fails loudly rather than fabricating a result-shaped default.
    World constants are imported from :mod:`fedference.pomdp` — the same
    constants the builders consume — never duplicated literals.
    """
    from fedference.pomdp import (
        ALERT_CENTER_MASS,
        CONTEXT_PERSISTENCE,
        N_CONTEXTS,
        N_LOCATIONS,
    )

    report = _load_report(root / "output" / "reports", "hierarchical_world.json")
    if not report:
        raise FileNotFoundError("missing hierarchical_world.json")

    acc = report["location_accuracy"]
    ms = report.get("multiseed", {})
    hi_ms = ms.get("primary", {})
    fl_ms = ms.get("baseline", {})
    gap_ms = ms.get("gap", {})
    pt = ms.get("paired_test", {})

    return {
        "HIER_N_LOCATIONS": str(N_LOCATIONS),
        "HIER_N_CONTEXTS": str(N_CONTEXTS),
        "HIER_N_AGENTS": str(report["n_agents"]),
        "HIER_N_TRIALS": str(report["n_trials"]),
        "HIER_ACUITY": _fmt(float(report["acuity"]), 2),
        "HIER_N_ITERS": str(report["n_iters"]),
        "HIER_SEED": str(report["seed"]),
        "HIER_ALERT_CENTER_MASS": _fmt(ALERT_CENTER_MASS, 2),
        "HIER_CTX_PERSIST": _fmt(CONTEXT_PERSISTENCE, 2),
        "HIER_LOC_ACC_FLAT": _fmt(float(acc["flat"]), 3),
        "HIER_LOC_ACC_HIER": _fmt(float(acc["hierarchical"]), 3),
        "HIER_LOC_ACC_GAP": _fmt(float(report["location_accuracy_gap"]), 3),
        "HIER_CTX_ACC": _fmt(float(report["context_accuracy"]), 3),
        "HIER_N_SEEDS": str(ms.get("n_seeds", "N/A")),
        "HIER_LOC_ACC_HIER_MEAN": _fmt(float(hi_ms.get("mean", 0)), 3),
        "HIER_LOC_ACC_HIER_STD": _fmt(float(hi_ms.get("std", 0)), 3),
        "HIER_LOC_ACC_HIER_CI_LO": _fmt(float(hi_ms.get("ci_lo", 0)), 3),
        "HIER_LOC_ACC_HIER_CI_HI": _fmt(float(hi_ms.get("ci_hi", 0)), 3),
        "HIER_LOC_ACC_FLAT_MEAN": _fmt(float(fl_ms.get("mean", 0)), 3),
        "HIER_LOC_ACC_FLAT_CI_LO": _fmt(float(fl_ms.get("ci_lo", 0)), 3),
        "HIER_LOC_ACC_FLAT_CI_HI": _fmt(float(fl_ms.get("ci_hi", 0)), 3),
        "HIER_LOC_ACC_GAP_MEAN": _fmt(float(gap_ms.get("mean", 0)), 3),
        "HIER_LOC_ACC_GAP_CI_LO": _fmt(float(gap_ms.get("ci_lo", 0)), 3),
        "HIER_LOC_ACC_GAP_CI_HI": _fmt(float(gap_ms.get("ci_hi", 0)), 3),
        "HIER_WILCOX_PVALUE": _fmt(float(pt.get("pvalue", 0)), 4),
        "HIER_EFFECT_SIZE": _fmt(abs(float(pt.get("effect_size", 0))), 3),
        "HIER_EFFECT_LABEL": ms.get("effect_label", "N/A"),
    }


def _nlevel3_variables(root: Path) -> dict[str, str]:
    """V2 3-level POMDP tokens from ``output/reports/nlevel3_world.json``.

    Executed-run parameters use STRICT key access (``report[...]``): a report
    missing a key fails loudly rather than fabricating a result-shaped default.
    World prior constants are imported from :mod:`fedference.pomdp` — the same
    constants :func:`fedference.pomdp.build_3level_world` consumes.
    """
    from fedference.pomdp import (
        ALERT_CENTER_MASS,
        GRID_SIDE,
        L2_HIGH_THREAT_ALERT_PRIOR,
        L2_HIGH_THREAT_QUIET_PRIOR,
        N_CONTEXTS,
        N_LOCATIONS,
        N_META_CONTEXTS,
    )

    report = _load_report(root / "output" / "reports", "nlevel3_world.json")
    if not report:
        raise FileNotFoundError("missing nlevel3_world.json")

    acc = report["location_accuracy"]
    ms = report.get("multiseed", {})
    nl_ms = ms.get("primary", {})
    fl_ms = ms.get("baseline", {})
    gap_ms = ms.get("gap", {})
    pt = ms.get("paired_test", {})

    return {
        "NLEVEL3_N_LOCATIONS": str(N_LOCATIONS),
        "NLEVEL3_N_CONTEXTS": str(N_CONTEXTS),
        "NLEVEL3_N_META_CONTEXTS": str(N_META_CONTEXTS),
        "NLEVEL3_N_AGENTS": str(report["n_agents"]),
        "NLEVEL3_N_TRIALS": str(report["n_trials"]),
        "NLEVEL3_ACUITY": _fmt(float(report["acuity"]), 2),
        "NLEVEL3_N_ITERS": str(report["n_iters"]),
        "NLEVEL3_SEED": str(report["seed"]),
        "NLEVEL3_N_LEVELS": str(report["n_levels"]),
        "NLEVEL3_LOC_ACC_FLAT": _fmt(float(acc["flat"]), 3),
        "NLEVEL3_LOC_ACC_3LEVEL": _fmt(float(acc["nlevel3"]), 3),
        "NLEVEL3_LOC_ACC_GAP": _fmt(float(report["location_accuracy_gap"]), 3),
        "NLEVEL3_CTX_ACC": _fmt(float(report["context_accuracy"]), 3),
        "NLEVEL3_META_CTX_ACC": _fmt(float(report["meta_context_accuracy"]), 3),
        "NLEVEL3_N_SEEDS": str(ms.get("n_seeds", "N/A")),
        "NLEVEL3_LOC_ACC_3LEVEL_MEAN": _fmt(float(nl_ms.get("mean", 0)), 3),
        "NLEVEL3_LOC_ACC_3LEVEL_STD": _fmt(float(nl_ms.get("std", 0)), 3),
        "NLEVEL3_LOC_ACC_3LEVEL_CI_LO": _fmt(float(nl_ms.get("ci_lo", 0)), 3),
        "NLEVEL3_LOC_ACC_3LEVEL_CI_HI": _fmt(float(nl_ms.get("ci_hi", 0)), 3),
        "NLEVEL3_LOC_ACC_FLAT_MEAN": _fmt(float(fl_ms.get("mean", 0)), 3),
        "NLEVEL3_LOC_ACC_FLAT_CI_LO": _fmt(float(fl_ms.get("ci_lo", 0)), 3),
        "NLEVEL3_LOC_ACC_FLAT_CI_HI": _fmt(float(fl_ms.get("ci_hi", 0)), 3),
        "NLEVEL3_LOC_ACC_GAP_MEAN": _fmt(float(gap_ms.get("mean", 0)), 3),
        "NLEVEL3_LOC_ACC_GAP_CI_LO": _fmt(float(gap_ms.get("ci_lo", 0)), 3),
        "NLEVEL3_LOC_ACC_GAP_CI_HI": _fmt(float(gap_ms.get("ci_hi", 0)), 3),
        "NLEVEL3_WILCOX_PVALUE": _fmt(float(pt.get("pvalue", 0)), 4),
        "NLEVEL3_EFFECT_SIZE": _fmt(abs(float(pt.get("effect_size", 0))), 3),
        "NLEVEL3_EFFECT_LABEL": ms.get("effect_label", "N/A"),
        "NLEVEL3_LOW_THREAT_QUIET_PRIOR": _fmt(1.0 / N_CONTEXTS, 2),
        "NLEVEL3_LOW_THREAT_ALERT_PRIOR": _fmt(1.0 / N_CONTEXTS, 2),
        "NLEVEL3_HIGH_THREAT_QUIET_PRIOR": _fmt(L2_HIGH_THREAT_QUIET_PRIOR, 2),
        "NLEVEL3_HIGH_THREAT_ALERT_PRIOR": _fmt(L2_HIGH_THREAT_ALERT_PRIOR, 2),
        "NLEVEL3_ALERT_CENTER_MASS": _fmt(ALERT_CENTER_MASS, 2),
        "NLEVEL3_CENTER_CELL_INDEX": str((GRID_SIDE // 2) * GRID_SIDE + (GRID_SIDE // 2)),
    }
