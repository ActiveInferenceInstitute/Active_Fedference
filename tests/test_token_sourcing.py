"""Regression tests: manuscript tokens are SOURCED (derived), never asserted.

Guards the fixes for three green-by-construction / duplicated-literal defect
classes in ``src/manuscript_vars/loaders.py``:

* ``FEDERATION_BIT_IDENTICAL`` was a hardcoded ``"True"`` — it must now be
  COMPUTED from one executed federation round compared bit-for-bit against the
  in-process ``robust_aggregate`` call on the same beliefs.
* ``TEMPERED_LAMBDA_GRID`` was a re-typed display string — it must now be
  joined from the same grid the sweep loop executes.
* The HIER_/NLEVEL3_ world-constant tokens duplicated literals that live in
  ``fedference.pomdp`` — the tokens must read the SAME named constants the
  world builders consume, and those constants must actually appear in the
  built worlds' tensors.

No mocks (project policy): every check runs the real code paths.
"""

from __future__ import annotations

import inspect
import json
import queue
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiment_config import SENSITIVITY_NOISE_FLOOR
from fedference.aggregation import robust_aggregate
from fedference.bnn_defaults import (
    BNN_BETA_DEFAULT,
    BNN_HIDDEN_DIM_DEFAULT,
    BNN_N_CLIENTS_DEFAULT,
    BNN_N_STEPS_DEFAULT,
    BNN_ROBUSTNESS_DEFAULT,
)
from fedference.experiments.sensitivity import (
    DEFAULT_SENSITIVITY_N_TRIALS,
    run_belief_sharing_sensitivity,
)
from fedference.federation import FederationServer, FederationWorker
from fedference.pomdp import (
    ALERT_CENTER_MASS,
    CONTEXT_PERSISTENCE,
    CONTEXT_SWITCH_PROB,
    GRID_SIDE,
    L2_HIGH_THREAT_ALERT_PRIOR,
    L2_HIGH_THREAT_QUIET_PRIOR,
    build_3level_world,
    build_hierarchical_world,
)
from manuscript_vars.loaders import (
    _SENS_N_TRIALS,
    _SENS_NOISE_FLOOR,
    _TEMPERED_LAMBDA_GRID,
    FEDERATION_N_WORKERS,
    FEDERATION_ROBUSTNESS,
    _bnn_robustness_variables,
    _bnn_torch_variables,
    _count_isc,
    _count_tests,
    _count_tests_by_name,
    _federation_demo_beliefs,
    _federation_variables,
    _hierarchical_variables,
    _nlevel3_variables,
    _tempered_variables,
)
from manuscript_vars.tokens import _complexity_variables, _review_grid_variables, _sweep_variables

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPORTS = _PROJECT_ROOT / "output" / "reports"
_CENTER = (GRID_SIDE // 2) * GRID_SIDE + (GRID_SIDE // 2)


# ---- (a) FEDERATION_BIT_IDENTICAL is computed, not asserted -----------------


def test_federation_bit_identical_token_matches_direct_recomputation() -> None:
    tokens = _federation_variables(_PROJECT_ROOT)

    # Direct recomputation through the same worker/server queue transport on
    # the same deterministic beliefs.
    beliefs = _federation_demo_beliefs()
    ref = robust_aggregate(beliefs, robustness=FEDERATION_ROBUSTNESS).consensus
    request_q: queue.Queue = queue.Queue()
    response_qs: dict[int, queue.Queue] = {i: queue.Queue() for i in range(FEDERATION_N_WORKERS)}
    workers = [FederationWorker(i, request_q, response_qs[i]) for i in range(FEDERATION_N_WORKERS)]
    server = FederationServer(n_workers=FEDERATION_N_WORKERS, robustness=FEDERATION_ROBUSTNESS)
    for worker, belief in zip(workers, beliefs):
        worker.send_belief(belief)
    fed = server.run_round(request_q, response_qs)

    assert tokens["FEDERATION_BIT_IDENTICAL"] == str(bool(np.array_equal(ref, fed)))
    # The transport is in fact lossless (same invariant as the end-to-end tests).
    assert tokens["FEDERATION_BIT_IDENTICAL"] == "True"
    # Parameter tokens come from the SAME constants the executed round used.
    assert tokens["FEDERATION_N_WORKERS"] == str(FEDERATION_N_WORKERS)
    assert tokens["FEDERATION_ROBUSTNESS"] == f"{FEDERATION_ROBUSTNESS:g}"


# ---- (b) TEMPERED_LAMBDA_GRID is joined from the executed grid --------------


def test_tempered_tokens_derive_from_the_executed_grid() -> None:
    tokens = _tempered_variables(_PROJECT_ROOT)
    assert tokens["TEMPERED_LAMBDA_GRID"] == ", ".join(f"{lam:g}" for lam in _TEMPERED_LAMBDA_GRID)
    # lambda* is selected from (and therefore a member of) the executed grid.
    assert float(tokens["TEMPERED_LAMBDA_STAR"]) in {round(lam, 1) for lam in _TEMPERED_LAMBDA_GRID}
    # The adversarial count is a consumed integer within the colony size.
    n_adv = int(tokens["TEMPERED_N_ADVERSARIAL"])
    assert 1 <= n_adv < int(tokens["TEMPERED_N_AGENTS"])


# ---- (c) pomdp constants are the ones the built worlds actually contain -----


def test_pomdp_constants_appear_in_the_built_world_tensors() -> None:
    hier = build_hierarchical_world()

    p_alert = np.asarray(hier["L1_priors_given_context"][1])
    assert p_alert[_CENTER] == ALERT_CENTER_MASS
    off_mass = (1.0 - ALERT_CENTER_MASS) / (p_alert.size - 1)
    assert np.all(np.delete(p_alert, _CENTER) == off_mass)

    trans = np.asarray(hier["L2_transition"])
    assert trans[0, 0] == CONTEXT_PERSISTENCE
    assert trans[1, 1] == CONTEXT_PERSISTENCE
    assert trans[0, 1] == CONTEXT_SWITCH_PROB
    assert trans[1, 0] == CONTEXT_SWITCH_PROB

    w3 = build_3level_world()
    l2_high = np.asarray(w3["L2_priors_given_l3"][1])
    assert l2_high[0] == L2_HIGH_THREAT_QUIET_PRIOR
    assert l2_high[1] == L2_HIGH_THREAT_ALERT_PRIOR
    p_alert_3 = np.asarray(w3["L1_priors_given_l2"][1])
    assert p_alert_3[_CENTER] == ALERT_CENTER_MASS


def test_world_constant_tokens_read_the_pomdp_constants() -> None:
    hier_tokens = _hierarchical_variables(_PROJECT_ROOT)
    assert hier_tokens["HIER_ALERT_CENTER_MASS"] == f"{ALERT_CENTER_MASS:.2f}"
    assert hier_tokens["HIER_CTX_PERSIST"] == f"{CONTEXT_PERSISTENCE:.2f}"

    n3_tokens = _nlevel3_variables(_PROJECT_ROOT)
    assert n3_tokens["NLEVEL3_ALERT_CENTER_MASS"] == f"{ALERT_CENTER_MASS:.2f}"
    assert n3_tokens["NLEVEL3_HIGH_THREAT_QUIET_PRIOR"] == f"{L2_HIGH_THREAT_QUIET_PRIOR:.2f}"
    assert n3_tokens["NLEVEL3_HIGH_THREAT_ALERT_PRIOR"] == f"{L2_HIGH_THREAT_ALERT_PRIOR:.2f}"


# ---- Single-definition defaults (sensitivity + BNN fallbacks) ---------------


def test_sensitivity_token_constant_is_the_experiment_default() -> None:
    assert _SENS_N_TRIALS == DEFAULT_SENSITIVITY_N_TRIALS
    assert _SENS_NOISE_FLOOR == SENSITIVITY_NOISE_FLOOR
    sig = inspect.signature(run_belief_sharing_sensitivity)
    assert sig.parameters["n_trials"].default == DEFAULT_SENSITIVITY_N_TRIALS


def test_bnn_fallback_tokens_are_the_experiment_defaults(tmp_path: Path) -> None:
    # tmp_path has no bnn_torch.json, so the fallback branch is exercised.
    out = _bnn_torch_variables(tmp_path)
    assert out["BNN_HIDDEN_DIM"] == str(BNN_HIDDEN_DIM_DEFAULT)
    assert out["BNN_N_STEPS"] == str(BNN_N_STEPS_DEFAULT)
    assert out["BNN_BETA"] == str(BNN_BETA_DEFAULT)
    assert out["BNN_ROBUSTNESS"] == str(BNN_ROBUSTNESS_DEFAULT)
    assert out["BNN_N_CLIENTS"] == str(BNN_N_CLIENTS_DEFAULT)

    # When torch is importable, the experiment signature must consume the very
    # same torch-free constants the fallback tokens read.
    torch = pytest.importorskip("torch")
    assert torch is not None
    from fedference.bnn_baseline_torch import run_bnn_torch_experiment

    sig = inspect.signature(run_bnn_torch_experiment)
    assert sig.parameters["hidden_dim"].default == BNN_HIDDEN_DIM_DEFAULT
    assert sig.parameters["n_steps"].default == BNN_N_STEPS_DEFAULT
    assert sig.parameters["beta"].default == BNN_BETA_DEFAULT
    assert sig.parameters["robustness"].default == BNN_ROBUSTNESS_DEFAULT
    assert sig.parameters["n_clients"].default == BNN_N_CLIENTS_DEFAULT


def test_loader_fallbacks_and_counting_paths(tmp_path: Path) -> None:
    assert _count_isc(tmp_path) == ("N/A", "N/A")
    (tmp_path / "ISA.md").write_text("not an ISC row\n", encoding="utf-8")
    assert _count_isc(tmp_path) == ("N/A", "N/A")
    (tmp_path / "ISA.md").write_text(
        "- [x] ISC-1: passed\n- [ ] ISC-2: open\n- [DEFERRED-VERIFY] ISC-3: deferred\n",
        encoding="utf-8",
    )
    assert _count_isc(tmp_path) == ("3", "1")

    (tmp_path / "ISA.md").write_text(
        "- [x] ISC-1: first definition\n- [ ] ISC-1: reused definition\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate ISC identifier"):
        _count_isc(tmp_path)

    assert _count_tests_by_name(tmp_path) == "N/A"
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_dummy.py").write_text(
        "def test_one():\n    pass\n\nasync def test_two():\n    pass\n",
        encoding="utf-8",
    )
    assert _count_tests_by_name(tmp_path) == "2"
    assert _count_tests(tmp_path) == "2"


def test_project_isa_frontmatter_matches_the_unique_live_tally() -> None:
    """The prose certificate cannot silently diverge from its acceptance rows."""
    text = (_PROJECT_ROOT / "ISA.md").read_text(encoding="utf-8")
    _, frontmatter, _ = text.split("---", maxsplit=2)
    metadata = yaml.safe_load(frontmatter)
    assert isinstance(metadata, dict)
    total, passed = _count_isc(_PROJECT_ROOT)
    assert metadata["progress"] == f"{passed}/{total}"


def test_loader_invalid_reports_and_source_bound_token_fallbacks(tmp_path: Path) -> None:
    reports = tmp_path / "output" / "reports"
    reports.mkdir(parents=True)
    (reports / "bnn_torch.json").write_text("{not-json", encoding="utf-8")
    assert _bnn_torch_variables(tmp_path)["PYTORCH_VERSION"] == "not installed"

    (reports / "bnn_robustness.json").write_text("{not-json", encoding="utf-8")
    assert _bnn_robustness_variables(tmp_path)["BNN_ROBUSTNESS_N_SEEDS"] == "N/A"
    (reports / "bnn_robustness.json").write_text(
        json.dumps(
            {
                "n_seeds": 4,
                "n_per": 8,
                "robust_loss_param": 0.3,
                "contamination_levels": [0.0, 0.5],
                "accuracy_by_config": {
                    "nll": [0.9, 0.4],
                    "rcce": [0.8, 0.7],
                },
            }
        ),
        encoding="utf-8",
    )
    out = _bnn_robustness_variables(tmp_path)
    assert out["BNN_ROBUSTNESS_PEAK_CONTAM"] == "0.50"
    assert out["BNN_ROBUSTNESS_PEAK_GAP"] == "0.300"


def test_token_flatteners_fail_closed_on_malformed_optional_blocks() -> None:
    assert _review_grid_variables({})["REVIEW_GRID_N_SEEDS"] == "N/A"
    review_tokens = _review_grid_variables(
        {
            "n_seeds": 16,
            "n_trials": 8,
            "rates": [0.0, 0.5],
            "attack_mechanisms": ["clean"],
            "directional_mechanisms": ["drift"],
            "entropy_controls": ["uniform"],
            "selection_status": "selection-free",
            "independent_unit": "seed",
            "trial_structure": "nested trials",
            "statistics": {"selection_free": True, "bh_family_ownership": "per-method"},
            "precision_plan": {
                "target_max_mcse": 0.01,
                "observed_max_mcse": 0.0084,
                "n_signed_method_rate_cells": 96,
            },
        }
    )
    assert review_tokens["REVIEW_GRID_TARGET_MAX_MCSE"] == "0.0100"
    assert review_tokens["REVIEW_GRID_OBSERVED_MAX_MCSE"] == "0.0084"
    assert review_tokens["REVIEW_GRID_SIGNED_CELLS"] == "96"
    smoke_precision_tokens = _review_grid_variables(
        {
            "precision_plan": {
                # Diagnostic smoke runs intentionally do not enforce the
                # publication stopping rule, but the manuscript must retain
                # the pre-run registered design target.
                "target_max_mcse": None,
                "observed_max_mcse": 0.0625,
                "n_signed_method_rate_cells": 48,
            }
        },
        configured_target_max_mcse=0.0125,
    )
    assert smoke_precision_tokens["REVIEW_GRID_TARGET_MAX_MCSE"] == "0.0125"
    with pytest.raises(ValueError, match="validation-backed hydration"):
        _review_grid_variables(
            {"precision_plan": {"target_max_mcse": None}},
            configured_target_max_mcse=0.0125,
            require_reported_target=True,
        )
    with pytest.raises(ValueError, match="disagrees"):
        _review_grid_variables(
            {"precision_plan": {"target_max_mcse": 0.02}},
            configured_target_max_mcse=0.0125,
        )
    malformed_review_tokens = _review_grid_variables({"precision_plan": {"target_max_mcse": True}})
    assert malformed_review_tokens["REVIEW_GRID_TARGET_MAX_MCSE"] == "N/A"
    assert _complexity_variables({})["COMPLEXITY_AGENT_GRID"] == "N/A"
    assert (
        _complexity_variables({"benchmark": [], "machine": {}, "analytic_specs": [], "measurements": []})[
            "COMPLEXITY_AGENT_GRID"
        ]
        == "N/A"
    )
    malformed = _complexity_variables(
        {
            "benchmark": {
                "agent_sizes": "bad",
                "state_sizes": [],
                "sharing_agent_sizes": [],
                "modality_sizes": [],
            },
            "machine": {},
            "analytic_specs": [
                None,
                {"operation": "unknown"},
                {"operation": "log_linear_pool", "time_order": "Theta", "memory_order": "Theta"},
            ],
            "measurements": [
                None,
                {
                    "method": "share_round_robust",
                    "axis": "agents",
                    "parameters": "not-a-dict",
                    "observed_log_log_slope": None,
                },
                {
                    "method": "log_linear_pool",
                    "axis": "agents",
                    "observed_log_log_slope": None,
                },
            ],
        }
    )
    assert malformed["COMPLEXITY_AGENT_GRID"] == "N/A"

    sweep = json.loads(
        (_PROJECT_ROOT / "output" / "reports" / "robustness_sweep.json").read_text(encoding="utf-8")
    )
    worst_key = f"{float(sweep['worst_rate']):g}"
    sweep["per_rate_summary"] = {worst_key: {"methods": {"KLD": {"mean": 0.5}}}}
    assert _sweep_variables(sweep)["SWEEP_PROFILE_BEST_ROBUST_ACCURACY"] == "N/A"
