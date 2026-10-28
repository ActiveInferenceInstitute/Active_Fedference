"""Characterized real smoke producer output and fail-closed ownership."""

from __future__ import annotations

import ast
import inspect
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from typing_extensions import assert_never

from analysis import workflow
from analysis.artifacts import ANALYSIS_REPORT_FILENAMES, expected_artifacts
from analysis.producers import belief, bnn, diagnostics, robustness, worlds
from analysis.report_schemas import ReportSchemaError, validate_report
from experiment_config import ExperimentConfig, load_experiment_config
from fedference.experiments import run_belief_sharing
from figures._presentation import validate_presentation_files
from tests.analysis.test_workflow import _make_project

pytestmark = [pytest.mark.slow, pytest.mark.publication]

# Artifact order characterized by the pre-extraction real CLI smoke run.
_ARTIFACT_ORDER = (
    "application_integrity_flow_report",
    "evidence_replication_map_report",
    "source_render_provenance_report",
    "belief_sharing_report",
    "language_report",
    "emergence_report",
    "robustness_report",
    "hierarchical_bmr_report",
    "heuristic_characterization_report",
    "efe_report",
    "influence_weights_report",
    "bnn_report",
    "bnn_torch_report",
    "variational_aggregation_report",
    "contamination_gallery_report",
    "robustness_onset_report",
    "conditional_world_report",
    "review_grid_report",
    "belief_quality_report",
    "complexity_scaling_report",
    "sensitivity_report",
    "moving_world_report",
    "parameter_recovery_report",
    "hierarchical_world_report",
    "nlevel3_world_report",
    "cross_study_report",
    "disjoint_fov_report",
    "figure_exact_values",
    "figure_exact_values_markdown",
    "application_integrity_flow",
    "evidence_replication_map",
    "source_render_provenance",
    "graphical_abstract",
    "system_overview",
    "generative_model_schema",
    "message_passing",
    "pomdp_loop",
    "belief_heatmap",
    "free_energy_comparison",
    "robustness_sweep",
    "language_kl_decay",
    "emergence_bmr",
    "hierarchical_bmr",
    "heuristic_breakdown",
    "efe_decomposition",
    "robust_influence_weights",
    "bnn_robustness",
    "aggregation_descent",
    "bounded_influence",
    "contamination_gallery",
    "descent_comparison",
    "robustness_onset",
    "conditional_world",
    "robustness_review_grid",
    "belief_quality",
    "complexity_scaling",
    "moving_world",
    "parameter_recovery",
    "sensitivity_heatmap",
    "cross_study_summary",
    "hierarchical_pomdp",
    "disjoint_fov_world",
    "figure_registry",
    "analysis_execution",
)


def _assert_source_origins() -> Path:
    root = Path(__file__).resolve().parents[2]
    expected = Path(os.environ.get("TASK15_EXPECT_SOURCE_ROOT", str(root))).resolve()
    assert root == expected
    for module in (workflow, belief, bnn, diagnostics, robustness, worlds):
        assert module.__file__ is not None
        origin = Path(module.__file__).resolve()
        assert origin.is_relative_to(expected / "src"), origin
        print(f"imported {module.__name__}: {origin}")
    for function, directory in ((run_belief_sharing, "src"), (_make_project, "tests")):
        origin = Path(inspect.getfile(function)).resolve()
        assert origin.is_relative_to(expected / directory), origin
        print(f"imported {function.__module__}.{function.__name__}: {origin}")
    conftest_origins = []
    for name, loaded in tuple(sys.modules.items()):
        if name == "conftest" or name.endswith(".conftest"):
            assert loaded is not None
            filename = loaded.__file__
            assert filename is not None
            origin = Path(filename).resolve()
            assert origin.is_relative_to(expected), origin
            conftest_origins.append(str(origin))
            print(f"imported {name}: {origin}")
    assert conftest_origins
    return root


def test_producer_outputs_and_inventory_unchanged(tmp_path: Path) -> None:
    # Given: real family outputs from the unchanged explicit smoke inputs.
    _assert_source_origins()
    _make_project(tmp_path)
    configured = load_experiment_config(tmp_path)
    assert configured.seeds == (0, 1, 2, 3)
    # These are the existing smoke controls needed by extracted producers.
    # Family-owned smoke branches clamp their remaining nested budgets.
    config = replace(
        configured,
        n_trials=6,
        conditional_world_n_seeds=4,
        conditional_world_n_trials=4,
    )
    try:
        from fedference.bnn_baseline_torch import run_bnn_torch_experiment
    except ImportError as exc:
        expected_torch = {"status": f"skipped: PyTorch unavailable ({exc})"}
    else:
        expected_torch = bnn._bnn_torch_report(
            config,
            options=workflow._bnn_torch_options(tmp_path),
            run_experiment=run_bnn_torch_experiment,
        )
    expected = {
        "belief_sharing.json": belief._belief_sharing_report(config),
        "language_acquisition.json": belief._language_report(config),
        "emergence.json": belief._emergence_report(config),
        "robustness_sweep.json": robustness._robustness_report(config),
        "efe_decomposition.json": belief._efe_terms(config),
        "robust_influence_weights.json": robustness._influence_weights(config),
        "bnn_robustness.json": bnn._bnn_report(config, smoke=True),
        "bnn_torch.json": expected_torch,
        "variational_aggregation.json": robustness._variational_aggregation_report(config),
        "contamination_gallery.json": robustness._contamination_gallery_report(config, smoke=True),
        "robustness_onset.json": robustness._robustness_onset_report(config, smoke=True),
        "conditional_world.json": worlds._conditional_world_report(config),
        "robustness_review_grid.json": robustness._review_grid_report(config, smoke=True),
        "belief_quality.json": diagnostics._belief_quality_report(config),
        "sensitivity.json": diagnostics._sensitivity_report(seed=0, n_trials=5),
        "moving_world.json": worlds._moving_world_report(config),
        "parameter_recovery.json": worlds._parameter_recovery_report(config, smoke=True),
    }
    # When: canonical orchestration consumes every real producer.
    paths = workflow.run_analysis_pipeline(project_root=tmp_path)
    # Then: producer payloads reach their unique report files byte-for-byte.
    assert tuple(paths) == _ARTIFACT_ORDER
    assert len(paths.values()) == len(set(paths.values()))
    reports = tmp_path / "output/reports"
    assert sorted(p.name for p in reports.glob("*.json")) == list(ANALYSIS_REPORT_FILENAMES)
    for filename, payload in expected.items():
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        assert (reports / filename).read_bytes() == encoded.encode(), filename
    for filename in ANALYSIS_REPORT_FILENAMES:
        validate_report(Path(filename).stem, json.loads((reports / filename).read_text()))
    actual = {p.relative_to(tmp_path) for p in (tmp_path / "output").rglob("*") if p.is_file()}
    declared = {p.relative_to(tmp_path) for p in expected_artifacts(tmp_path).values()}
    declared.remove(Path("output/data/manuscript_variables.json"))
    for manifest in (tmp_path / "output/figures").glob("*.slides.json"):
        declared.update(p.relative_to(tmp_path) for p in validate_presentation_files(manifest))
    assert actual == declared
    assert json.loads(paths["analysis_execution"].read_text()) == {
        "schema_version": 2,
        "configured_profile": "smoke",
        "effective_profile": "smoke",
        "producer": "analysis.workflow.run_analysis_pipeline",
    }


@pytest.mark.parametrize("case", ["malformed-new", "malformed-existing", "duplicate-key", "duplicate-path"])
def test_malformed_report_and_duplicate_artifact(
    tmp_path: Path,
    case: Literal["malformed-new", "malformed-existing", "duplicate-key", "duplicate-path"],
) -> None:
    # Given: an accepted real report and an independently tracked owner.
    _assert_source_origins()
    _make_project(tmp_path)
    report = belief._belief_sharing_report(load_experiment_config(tmp_path))
    accepted = tmp_path / "accepted.json"
    workflow._write_json(report, accepted, schema="belief_sharing")
    original = accepted.read_bytes()
    paths = {"accepted": accepted}
    candidate, key, payload = accepted, "candidate", report
    error: type[Exception] = workflow.DuplicateArtifactError
    match case:
        case "malformed-new":
            candidate, payload, error = tmp_path / "new/nested/report.json", {}, ReportSchemaError
        case "malformed-existing":
            payload, paths, error = {}, {}, ReportSchemaError
        case "duplicate-key":
            key, candidate = "accepted", tmp_path / "other.json"
        case "duplicate-path":
            candidate = accepted.parent / "nested/../accepted.json"
            (accepted.parent / "nested").mkdir()
        case unreachable:
            assert_never(unreachable)
    prior = paths.copy()
    # When/Then: refusal precedes accepted replacement, parent creation or claim.
    with pytest.raises(error):
        workflow._write_json(payload, candidate, schema="belief_sharing", artifact=(key, paths))
    assert accepted.read_bytes() == original
    assert paths == prior
    if candidate.resolve() != accepted.resolve():
        assert not candidate.exists()
        if case == "malformed-new":
            assert not candidate.parent.exists()
    assert not list(tmp_path.rglob(".*.tmp"))


def test_family_imports_and_write_ownership_are_direct() -> None:
    # Given: independent families with canonical function objects.
    root = _assert_source_origins()
    for module in (belief, bnn, diagnostics, robustness, worlds):
        assert module.__file__ is not None
        tree = ast.parse(Path(module.__file__).read_text())
        assert not any(
            isinstance(node, ast.ImportFrom) and node.module and "workflow" in node.module
            for node in ast.walk(tree)
        )
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and hasattr(workflow, node.name):
                assert getattr(workflow, node.name) is getattr(module, node.name)
    # When: inspect actual canonical report write sites.
    tree = ast.parse((root / "src/analysis/workflow.py").read_text())
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_json"
        and any(keyword.arg == "schema" for keyword in node.keywords)
    ]
    # Then: every schema-validated report participates in unique ownership.
    assert len(writes) == len(ANALYSIS_REPORT_FILENAMES) + 2
    assert all(any(keyword.arg == "artifact" for keyword in node.keywords) for node in writes)


@pytest.mark.parametrize(("n_agents", "n_seeds"), [(3, 2), (7, 3)])
def test_seeded_family_consumes_configured_inputs(n_agents: int, n_seeds: int) -> None:
    # Given: distinct configurations and independent domain results.
    _assert_source_origins()
    config = ExperimentConfig(n_agents=n_agents, n_seeds=n_seeds)
    expected = [
        float(run_belief_sharing(seed, communicate=True, n_agents=n_agents)["mean_free_energy"])
        for seed in range(n_seeds)
    ]
    # When: the extracted family consumes the explicit configuration.
    report = belief._belief_sharing_report(config)
    # Then: every configured seed and colony size reaches the domain operation.
    assert report["communicating_free_energy"] == expected
    assert report["n_seeds"] == n_seeds
    assert report["n_agents"] == n_agents


@pytest.mark.parametrize(
    "knob",
    [
        "n_clients: true",
        "n_per: 0",
        "hidden_dim: 1.5",
        "robustness: true",
        "robustness: bad",
        "robustness: -0.1",
        "beta: .inf",
        "contamination_levels: []",
        "contamination_levels: [-0.1]",
        "contamination_levels: [true]",
    ],
)
def test_torch_producer_rejects_invalid_source_options(tmp_path: Path, knob: str) -> None:
    # Given: a real source profile with one invalid optional Torch knob.
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    (manuscript / "config.yaml").write_text("experiment:\n  bnn_torch:\n    " + knob + "\n")
    # When/Then: malformed controls fail before experiment use.
    with pytest.raises(ValueError, match="bnn_torch"):
        workflow._bnn_torch_options(tmp_path)


def test_torch_options_and_profiles_retain_explicit_defaults(tmp_path: Path) -> None:
    # Given: absent optional roots and a complete explicit Torch profile.
    assert workflow._bnn_torch_options(None) == {}
    assert workflow._bnn_torch_options(tmp_path) == {}
    assert workflow.resolve_analysis_profile(None) == "publication"
    assert workflow.resolve_analysis_profile(tmp_path) == "publication"
    _make_project(tmp_path)
    options = {
        "n_clients": 2,
        "n_per": 8,
        "hidden_dim": 4,
        "n_steps": 3,
        "robustness": 0.7,
        "beta": 0.2,
        "contamination_levels": [0.0, 0.5],
    }
    (tmp_path / "manuscript/config.yaml").write_text(
        json.dumps({"experiment": {"analysis_profile": "smoke", "bnn_torch": options}})
    )
    # When: consume the explicit source options.
    observed = workflow._bnn_torch_options(tmp_path)
    # Then: retain architecture and both separately configured scientific controls.
    assert observed == {**options, "contamination_levels": (0.0, 0.5)}
    assert workflow.resolve_analysis_profile(tmp_path) == "smoke"


def test_standalone_figure_generator_ownership_excludes_support_modules(tmp_path: Path) -> None:
    # Given: a real standalone generator and real shared support source.
    root = _assert_source_origins()
    package = tmp_path / "src/figures"
    package.mkdir(parents=True)
    for name in ("belief_heatmap.py", "_common.py"):
        shutil.copyfile(root / "src/figures" / name, package / name)
    # When: the registry resolves standalone ownership from source modules.
    observed = workflow._figure_generators_by_filename(tmp_path)
    # Then: both formats belong to the generator; support files own no figure.
    assert observed == {
        "belief_heatmap.png": "belief_heatmap",
        "belief_heatmap.pdf": "belief_heatmap",
    }


@pytest.mark.requires_torch
@pytest.mark.parametrize("source_state", ["invalid", "missing"])
def test_torch_producer_uses_validated_options_after_source_changes(
    tmp_path: Path, source_state: Literal["invalid", "missing"]
) -> None:
    # Given: a real executed report and options loaded at the workflow boundary.
    from fedference.bnn_baseline_torch import run_bnn_torch_experiment

    _assert_source_origins()
    _make_project(tmp_path)
    config = load_experiment_config(tmp_path)
    options = workflow._bnn_torch_options(tmp_path)
    assert options["n_steps"] == 3
    expected = run_bnn_torch_experiment(seed=config.seeds[0], **options)
    source = tmp_path / "manuscript/config.yaml"
    match source_state:
        case "invalid":
            source.write_text("experiment:\n  bnn_torch:\n    n_steps: 0\n")
        case "missing":
            source.unlink()
        case unreachable:
            assert_never(unreachable)
    # When: the pure producer consumes the previously validated typed options.
    actual = bnn._bnn_torch_report(config, options=options, run_experiment=run_bnn_torch_experiment)
    # Then: source-file changes cannot alter the executed report.
    assert actual == expected
    validate_report("bnn_torch", actual)
