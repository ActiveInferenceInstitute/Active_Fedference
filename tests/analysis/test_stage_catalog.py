"""Real seeded catalog selection and source-derived completeness contracts."""

from __future__ import annotations

import ast
import inspect
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from analysis import visual_contracts, workflow
from analysis.producers import belief, bnn, diagnostics, robustness, worlds
from analysis.report_schemas import validate_report
from analysis.stage_catalog import STAGE_SCHEMAS, report_producers
from experiment_config import ExperimentConfig
from fedference import experiments
from fedference.complexity import ComplexityBenchmarkConfig


def test_catalog_covers_every_original_report_in_workflow_order(tmp_path: Path) -> None:
    tree = ast.parse(inspect.getsource(workflow.run_analysis_pipeline))
    writes = sorted(
        (
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "_write_json"
        ),
        key=lambda node: node.lineno,
    )
    schemas: list[str] = []
    for write in writes:
        for keyword in write.keywords:
            if keyword.arg == "schema" and isinstance(keyword.value, ast.Constant):
                if keyword.value.value == "figure_exact_values":
                    assert list(STAGE_SCHEMAS) == schemas
                    assert list(STAGE_SCHEMAS.values()) == schemas
                    producers = report_producers(ExperimentConfig(), tmp_path / "absent", smoke=False)
                    assert list(producers) == schemas
                    assert all(callable(producer) for producer in producers.values())
                    return
                assert isinstance(keyword.value.value, str)
                schemas.append(keyword.value.value)
    pytest.fail("workflow must retain the report-to-exact-values boundary")


@pytest.mark.parametrize("smoke", [False, True])
def test_catalog_construction_does_not_read_project_configuration(tmp_path: Path, smoke: bool) -> None:
    # No project exists; optional BNN configuration access must remain deferred.
    root = tmp_path / "absent"
    catalog = report_producers(ExperimentConfig(n_seeds=2, n_agents=3), root, smoke=smoke)
    assert len(catalog) == 27
    assert not root.exists()
    assert catalog["hierarchical_world"] is not catalog["nlevel3_world"]
    assert catalog["cross_study_summary"] is not catalog["disjoint_fov_world"]
    # Even malformed actual configuration is untouched during catalog construction.
    manuscript = root / "manuscript"
    manuscript.mkdir(parents=True)
    path = manuscript / "config.yaml"
    path.write_text("experiment: [invalid mapping]\n", encoding="utf-8")
    before = path.read_bytes()
    assert list(report_producers(ExperimentConfig(), root, smoke=smoke)) == list(catalog)
    assert path.read_bytes() == before
    assert not (root / "output").exists()


@pytest.mark.parametrize("smoke", [False, True])
def test_selected_seeded_reports_match_existing_producers(tmp_path: Path, smoke: bool) -> None:
    config = ExperimentConfig(n_agents=3, n_seeds=2)
    producers = report_producers(config, tmp_path, smoke=smoke)
    expected = {
        "belief_sharing": workflow._belief_sharing_report(config),
        "variational_aggregation": workflow._variational_aggregation_report(config),
    }
    for name, direct in expected.items():
        payload = producers[name]()
        assert payload == direct
        validate_report(name, payload)
    assert expected["belief_sharing"]["n_agents"] == 3
    assert expected["belief_sharing"]["n_seeds"] == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name,direct", [
    ("application_integrity_flow", workflow.application_integrity_flow_contract),
    ("evidence_replication_map", workflow.evidence_replication_map_contract),
    ("source_render_provenance", workflow.source_render_provenance_contract),
])
def test_static_contracts_are_selected_without_output(
    name: str, direct: Callable[[], dict[str, Any]], tmp_path: Path,
) -> None:
    catalog = report_producers(ExperimentConfig(n_seeds=2), tmp_path, smoke=True)
    payload = catalog[name]()
    assert payload == direct()
    validate_report(name, payload)
    assert list(tmp_path.iterdir()) == []



def _input_contract(config: ExperimentConfig, root: Path) -> dict[
    str, tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]
]:
    """Independent expected bindings, not extracted from either implementation."""
    smoke_complexity = ComplexityBenchmarkConfig(
        agent_sizes=(2, 3, 4), state_sizes=(2, 3, 4), sharing_agent_sizes=(2, 3, 4),
        modality_sizes=(1, 2, 3), fixed_agent_count=4, fixed_state_count=4,
        inference_state_count=8, observation_count=2, repeats=2, warmups=0,
        max_iter=3, variational_starts=1, seed=17,
    )
    return {
        "application_integrity_flow": (visual_contracts.application_integrity_flow_contract, (), {}),
        "evidence_replication_map": (visual_contracts.evidence_replication_map_contract, (), {}),
        "source_render_provenance": (visual_contracts.source_render_provenance_contract, (), {}),
        "belief_sharing": (belief._belief_sharing_report, (config,), {}),
        "language_acquisition": (belief._language_report, (config,), {}),
        "emergence": (belief._emergence_report, (config,), {}),
        "robustness_sweep": (robustness._robustness_report, (config,), {}),
        "hierarchical_bmr": (experiments.run_hierarchical_bmr, (), {}),
        "heuristic_characterization": (experiments.run_heuristic_characterization, (0,), {}),
        "efe_decomposition": (belief._efe_terms, (config,), {}),
        "robust_influence_weights": (robustness._influence_weights, (config,), {}),
        "bnn_robustness": (bnn._bnn_report, (config,), {"smoke": True}),
        "bnn_torch": (workflow._configured_bnn_torch_report, (config,), {"project_root": root}),
        "variational_aggregation": (robustness._variational_aggregation_report, (config,), {}),
        "contamination_gallery": (robustness._contamination_gallery_report, (config,), {"smoke": True}),
        "robustness_onset": (robustness._robustness_onset_report, (config,), {"smoke": True}),
        "conditional_world": (worlds._conditional_world_report, (config,), {}),
        "robustness_review_grid": (robustness._review_grid_report, (config,), {"smoke": True}),
        "belief_quality": (diagnostics._belief_quality_report, (config,), {}),
        "complexity_scaling": (experiments.run_complexity_scaling, (smoke_complexity,), {}),
        "sensitivity": (diagnostics._sensitivity_report, (), {"seed": 0, "n_trials": 5}),
        "moving_world": (worlds._moving_world_report, (config,), {}),
        "parameter_recovery": (worlds._parameter_recovery_report, (config,), {"smoke": True}),
        "hierarchical_world": (
            experiments.hierarchical_world_report, (0,), {"n_trials": 2, "n_multiseed": 2},
        ),
        "nlevel3_world": (experiments.nlevel3_world_report, (0,), {"n_trials": 2, "n_multiseed": 2}),
        "cross_study_summary": (experiments.summarize_cross_study, (0,), {"n_seeds": 2, "n_trials": 2}),
        "disjoint_fov_world": (experiments.disjoint_fov_report, (0,), {"n_multiseed": 2}),
    }


def _observe_call(
    selected: Callable[[], Any], target: Callable[..., Any],
) -> tuple[Any, list[dict[str, Any]]]:
    """Observe the real target's call-entry locals without replacing any operation."""
    observed: list[dict[str, Any]] = []
    parameters = tuple(inspect.signature(target).parameters)
    code = getattr(target, "__code__")
    previous = sys.getprofile()

    def record(frame: FrameType, event: str, argument: Any) -> None:
        if event == "call" and frame.f_code is code:
            observed.append({name: frame.f_locals[name] for name in parameters})
            # Once entry is observed, leave numerical execution unprofiled.
            sys.setprofile(previous)

    try:
        sys.setprofile(record)
        result = selected()
    finally:
        sys.setprofile(previous)
    assert sys.getprofile() is previous
    return result, observed


def _assert_binding(
    observed: list[dict[str, Any]], target: Callable[..., Any],
    args: tuple[Any, ...], kwargs: dict[str, Any],
) -> None:
    expected = inspect.signature(target).bind(*args, **kwargs)
    expected.apply_defaults()
    assert observed == [dict(expected.arguments)]


@pytest.mark.parametrize("stage", tuple(STAGE_SCHEMAS))
def test_real_catalog_calls_preserve_independent_inputs(stage: str, tmp_path: Path) -> None:
    complexity = ComplexityBenchmarkConfig(
        agent_sizes=(2, 3, 4, 5), state_sizes=(2, 3, 4, 5), sharing_agent_sizes=(2, 3, 4, 5),
        modality_sizes=(1, 2, 3, 4), fixed_agent_count=4, fixed_state_count=4,
        inference_state_count=8, observation_count=2, repeats=3, warmups=1,
        max_iter=4, variational_starts=1, seed=17,
    )
    config = ExperimentConfig(
        n_agents=3, n_seeds=2, replicate_seeds=2, n_trials=2, cross_study_n_trials=2,
        conditional_world_n_seeds=2, conditional_world_n_trials=1,
        review_grid_n_seeds=2, review_grid_n_trials=2, review_grid_rates=(0.0,),
        gallery_n_seeds=2, gallery_n_trials=2, onset_n_seeds=2, onset_n_trials=2,
        bnn_n_seeds=2, bnn_n_per=2, contamination_rates=(0.0,), complexity=complexity,
    )
    # The optional Torch producer consumes a real, explicitly bounded source config.
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    (manuscript / "config.yaml").write_text(
        "experiment:\n  bnn_torch:\n    n_clients: 2\n    n_per: 2\n    hidden_dim: 2\n"
        "    n_steps: 1\n    contamination_levels: [0.0]\n", encoding="utf-8",
    )
    contracts = _input_contract(config, tmp_path)
    assert set(contracts) == set(STAGE_SCHEMAS)
    target, args, kwargs = contracts[stage]
    payload, observed = _observe_call(report_producers(config, tmp_path, smoke=True)[stage], target)
    _assert_binding(observed, target, args, kwargs)
    validate_report(stage, payload)
    assert not (tmp_path / "output").exists()


def test_call_contract_accepts_real_helpers_and_partials_and_rejects_wrong_seed() -> None:
    target = experiments.run_hierarchical_bmr

    def helper() -> dict[str, Any]:
        return target(seed=0)

    for selected in (helper, partial(target, seed=0)):
        payload, observed = _observe_call(selected, target)
        _assert_binding(observed, target, (), {"seed": 0})
        validate_report("hierarchical_bmr", payload)
    wrong_payload, wrong = _observe_call(partial(target, seed=1), target)
    validate_report("hierarchical_bmr", wrong_payload)
    with pytest.raises(AssertionError):
        _assert_binding(wrong, target, (), {"seed": 0})


def test_observation_restores_profile_after_real_producer_failure() -> None:
    previous = sys.getprofile()
    with pytest.raises(ValueError, match="n_iters must be a positive integer"):
        _observe_call(partial(experiments.run_hierarchical_bmr, n_iters=0), experiments.run_hierarchical_bmr)
    assert sys.getprofile() is previous
