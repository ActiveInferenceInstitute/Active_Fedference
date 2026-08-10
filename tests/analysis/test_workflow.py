"""Tests for the thin analysis-pipeline orchestrator.

No mocks: the pipeline runs the real fedference experiments into a throwaway
``tmp_path`` project root, then we assert the JSON reports and PNG figures
exist and carry the expected headline properties.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from analysis import workflow
from analysis.workflow import main, resolve_analysis_profile, run_analysis_pipeline
from experiment_config import ExperimentConfig

pytestmark = [pytest.mark.slow, pytest.mark.publication]


def test_default_project_root_is_repo_root() -> None:
    # The default-root helper resolves to the project root (two levels up).
    assert workflow._project_root() == workflow._PROJECT_ROOT
    assert (workflow._project_root() / "src" / "analysis" / "workflow.py").exists()


def test_project_root_env_override_is_validated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A valid ACTIVE_FEDFERENCE_PROJECT_ROOT (existing dir with
    # manuscript/config.yaml) redirects the resolved root; invalid values fail
    # loudly — never a silent fallback to the real tree (masking discipline).
    _make_project(tmp_path)
    monkeypatch.setenv("ACTIVE_FEDFERENCE_PROJECT_ROOT", str(tmp_path))
    assert workflow._project_root() == tmp_path.resolve()

    monkeypatch.setenv("ACTIVE_FEDFERENCE_PROJECT_ROOT", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="not an existing directory"):
        workflow._project_root()

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("ACTIVE_FEDFERENCE_PROJECT_ROOT", str(empty))
    with pytest.raises(RuntimeError, match="config.yaml"):
        workflow._project_root()

    monkeypatch.delenv("ACTIVE_FEDFERENCE_PROJECT_ROOT")
    assert workflow._project_root() == workflow._PROJECT_ROOT


def test_bnn_torch_profile_is_explicitly_loaded_and_validated(tmp_path: Path) -> None:
    _make_project(tmp_path)
    options = workflow._bnn_torch_options(tmp_path)
    assert options["n_steps"] == 3
    assert options["contamination_levels"] == (0.0, 0.5)

    (tmp_path / "manuscript" / "config.yaml").write_text(
        yaml.safe_dump({"experiment": {"bnn_torch": {"n_steps": 0}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="n_steps"):
        workflow._bnn_torch_options(tmp_path)


def test_analysis_profile_is_explicit_and_validated(tmp_path: Path) -> None:
    _make_project(tmp_path)
    assert resolve_analysis_profile(tmp_path) == "smoke"
    assert resolve_analysis_profile(tmp_path, override="publication") == "publication"
    (tmp_path / "manuscript" / "config.yaml").write_text(
        yaml.safe_dump({"experiment": {"analysis_profile": "unknown"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="analysis_profile"):
        resolve_analysis_profile(tmp_path)

    with pytest.raises(ValueError, match="analysis profile"):
        resolve_analysis_profile(tmp_path, override="invalid")


def _make_project(root: Path) -> None:
    manuscript = root / "manuscript"
    manuscript.mkdir(parents=True, exist_ok=True)
    config = {
        "paper": {"version": "0.1"},
        "authors": [{"name": "Test Author"}],
        "keywords": ["fedgvi", "active inference"],
        "experiment": {
            "analysis_profile": "smoke",
            "n_agents": 5,
            "n_seeds": 4,
            "replicate_seeds": 4,
            "cross_study_n_trials": 4,
            "contamination_rates": [0.0, 0.45, 0.9],
            "divergences": ["KLD", "RKL", "beta"],
            "bnn_torch": {
                "n_clients": 2,
                "n_per": 8,
                "hidden_dim": 4,
                "n_steps": 3,
                "contamination_levels": [0.0, 0.5],
            },
        },
    }
    (manuscript / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_run_analysis_pipeline_writes_reports_and_figures(tmp_path: Path) -> None:
    _make_project(tmp_path)
    paths = run_analysis_pipeline(project_root=tmp_path)

    expected = {
        # reports
        "belief_sharing_report",
        "language_report",
        "emergence_report",
        "robustness_report",
        "efe_report",
        "influence_weights_report",
        "bnn_report",
        "hierarchical_world_report",
        "nlevel3_world_report",
        "cross_study_report",
        "disjoint_fov_report",
        "moving_world_report",
        "conditional_world_report",
        "belief_quality_report",
        # figures
        "belief_heatmap",
        "free_energy_comparison",
        "robustness_sweep",
        "language_kl_decay",
        "emergence_bmr",
        "efe_decomposition",
        "robust_influence_weights",
        "bnn_robustness",
        "cross_study_summary",
        "hierarchical_pomdp",
        "disjoint_fov_world",
        "conditional_world",
        "belief_quality",
        "figure_registry",
        "analysis_execution",
    }
    assert expected <= set(paths)
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0

    conditional = json.loads(paths["conditional_world_report"].read_text())
    assert len(conditional["grid"]) == 40
    assert conditional["controls"]["robustness_zero_recovers_log_pool"] is True
    quality = json.loads(paths["belief_quality_report"].read_text())
    assert quality["controls"]["oracle_best_log_score"] is True
    assert quality["controls"]["confident_wrong_worse_than_uniform"] is True
    execution = json.loads(paths["analysis_execution"].read_text())
    assert execution == {
        "configured_profile": "smoke",
        "effective_profile": "smoke",
        "producer": "analysis.workflow.run_analysis_pipeline",
        "schema_version": 2,
    }


def test_world_reports_include_multiseed_block(tmp_path: Path) -> None:
    _make_project(tmp_path)
    paths = run_analysis_pipeline(project_root=tmp_path)
    for key in (
        "hierarchical_world_report",
        "nlevel3_world_report",
        "moving_world_report",
        "disjoint_fov_report",
    ):
        report = json.loads(paths[key].read_text())
        assert "multiseed" in report, f"{key} must carry multi-seed statistics"
        assert int(report["multiseed"]["n_seeds"]) >= 2


def test_cross_study_report_has_nine_studies(tmp_path: Path) -> None:
    _make_project(tmp_path)
    paths = run_analysis_pipeline(project_root=tmp_path)
    report = json.loads(paths["cross_study_report"].read_text())
    assert len(report["studies"]) == 9
    assert all("mean" in study and "ci_lo" in study for study in report["studies"])
    robustness = next(study for study in report["studies"] if study["study"] == 4)
    assert robustness["within_seed_n_trials"] == 4
    assert "Trial-mean" in robustness["metric"]
    assert "reduced within each seed" in robustness["estimand"]


def test_variational_report_descent_comparison_shows_real_capture() -> None:
    # The descent-comparison figure must show a GENUINE single-start capture, not
    # two identical curves (audit w9p4z6iv4 found capture_gap == 0 with the old
    # full-colony construction). On the capture-prone 2-honest + near-vertex-liar
    # colony the single-start final F must sit strictly above the multi-start one.
    report = workflow._variational_aggregation_report(ExperimentConfig())
    assert report["capture_gap"] > 0.05
    assert report["single_start_final_f"] > report["multi_start_final_f"]


def test_belief_sharing_report_shows_communication_helps(tmp_path: Path) -> None:
    _make_project(tmp_path)
    paths = run_analysis_pipeline(project_root=tmp_path)
    report = json.loads(paths["belief_sharing_report"].read_text())
    # Two heads beat one: communicating colony has lower mean free energy.
    assert report["communicating_mean"] < report["incommunicado_mean"]
    assert report["communication_helps"] is True
    assert report["free_energy_gap"] > 0.0
    assert len(report["communicating_free_energy"]) == 4


def test_robustness_report_has_method_rate_grid(tmp_path: Path) -> None:
    _make_project(tmp_path)
    paths = run_analysis_pipeline(project_root=tmp_path)
    report = json.loads(paths["robustness_report"].read_text())
    grid = report["accuracy_by_method_and_rate"]
    assert set(grid) == {"KLD", "RKL", "beta"}
    # Naive accuracy at the worst rate is recorded and is a probability.
    worst = report["accuracy_by_method_and_rate"]["KLD"]["0.9"]
    assert 0.0 <= worst <= 1.0
    assert report["n_agents"] == 5
    assert report["n_contaminated"] >= 1


def test_language_report_kl_declines(tmp_path: Path) -> None:
    _make_project(tmp_path)
    paths = run_analysis_pipeline(project_root=tmp_path)
    report = json.loads(paths["language_report"].read_text())
    assert report["final_kl"] < report["initial_kl"]


def test_figures_are_png(tmp_path: Path) -> None:
    _make_project(tmp_path)
    paths = run_analysis_pipeline(project_root=tmp_path)
    for key in (
        "belief_heatmap",
        "free_energy_comparison",
        "robustness_sweep",
        "language_kl_decay",
        "emergence_bmr",
        "efe_decomposition",
        "robust_influence_weights",
        "bnn_robustness",
    ):
        assert paths[key].suffix == ".png"
        assert paths[key].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_pipeline_writes_source_bound_validator_compatible_figure_registry(tmp_path: Path) -> None:
    _make_project(tmp_path)
    (tmp_path / "manuscript" / "16_results_belief_sharing.md").write_text(
        "See [@fig:belief-heatmap].\n\n"
        "![Belief heatmap caption for {{BELIEF_SHARING_N_AGENTS}} agents.]"
        "(../output/figures/belief_heatmap.png)"
        "{#fig:belief-heatmap width=80%}\n",
        encoding="utf-8",
    )
    paths = run_analysis_pipeline(project_root=tmp_path)
    registry = json.loads(paths["figure_registry"].read_text())
    figures = {item["label"]: item for item in registry["figures"]}

    assert set(figures) == {"fig:belief-heatmap"}
    assert figures["fig:belief-heatmap"]["filename"] == "belief_heatmap.png"
    assert figures["fig:belief-heatmap"]["generated_by"] == "belief_heatmap"
    assert figures["fig:belief-heatmap"]["source_manuscript"] == "manuscript/16_results_belief_sharing.md"
    assert figures["fig:belief-heatmap"]["caption"] == (
        "Belief heatmap caption for {{BELIEF_SHARING_N_AGENTS}} agents."
    )
    assert (tmp_path / "output" / "figures" / figures["fig:belief-heatmap"]["filename"]).exists()


def test_emergence_report_signs(tmp_path: Path) -> None:
    _make_project(tmp_path)
    paths = run_analysis_pipeline(project_root=tmp_path)
    report = json.loads(paths["emergence_report"].read_text())
    # Pruning the redundant column wins; pruning a supported column is rejected.
    assert report["delta_F_redundant"] > 0.0 > report["delta_F_supported"]
    assert report["convergence"] is True


def test_efe_report_identity_holds(tmp_path: Path) -> None:
    _make_project(tmp_path)
    paths = run_analysis_pipeline(project_root=tmp_path)
    report = json.loads(paths["efe_report"].read_text())
    # risk + ambiguity == -(pragmatic + epistemic) to floating-point tolerance.
    lhs = report["risk"] + report["ambiguity"]
    rhs = -(report["pragmatic_value"] + report["epistemic_value"])
    assert abs(lhs - rhs) < 1e-9
    assert abs(report["identity_residual"]) < 1e-9
    assert report["prior_type"] == "uniform diagnostic prior"
    assert report["epistemic_value"] > 0.0


def test_influence_weights_downweight_saboteur(tmp_path: Path) -> None:
    _make_project(tmp_path)
    paths = run_analysis_pipeline(project_root=tmp_path)
    report = json.loads(paths["influence_weights_report"].read_text())
    weights = report["agent_weights"]
    contaminated = report["contaminated_indices"]
    healthy = [w for i, w in enumerate(weights) if i not in contaminated]
    saboteur = [weights[i] for i in contaminated]
    # The robust pooling heuristic gives saboteurs strictly less influence.
    assert max(saboteur) < min(healthy)


def test_bnn_report_robust_holds_under_contamination(tmp_path: Path) -> None:
    _make_project(tmp_path)
    paths = run_analysis_pipeline(project_root=tmp_path)
    report = json.loads(paths["bnn_report"].read_text())
    configs = report["accuracy_by_config"]
    assert set(configs) == {"nll / KLD (standard)", "rcce / AR (robust)"}
    levels = report["contamination_levels"]
    for curve in configs.values():
        assert len(curve) == len(levels)
        assert all(0.0 <= v <= 1.0 for v in curve)
    # C3 fix: the robust client must show a genuine, reproducible margin over
    # the standard client SOMEWHERE in the sweep — not both curves
    # overlapping throughout (the original defect). The honest shape is
    # peak-then-reconverge: separation opens up in the moderate-to-high
    # contamination range and both curves collapse together again at the
    # extreme (0.4) endpoint, so this asserts the genuine mid-range margin
    # exists rather than requiring robust >= standard at every level
    # (including the terminal point, where advisor review confirmed there is
    # no principled reason to expect or require it).
    standard = configs["nll / KLD (standard)"]
    robust = configs["rcce / AR (robust)"]
    gaps = [r - s for r, s in zip(robust, standard)]
    assert max(gaps) > 0.025, (
        "robust client must show a real (not noise-level) margin over "
        "standard somewhere in the contamination sweep"
    )
    assert report["peak_margin"] == pytest.approx(max(gaps))
    assert report["peak_margin_contamination"] in levels
    # Temporary-project fixtures deliberately use the bounded smoke profile;
    # the shipped manuscript configuration remains publication-scale.  Keep
    # this assertion sensitive to both contracts rather than treating the
    # smoke budget as evidence that a publication run was executed.
    profile = (
        yaml.safe_load((tmp_path / "manuscript" / "config.yaml").read_text(encoding="utf-8"))
        .get("experiment", {})
        .get("analysis_profile", "publication")
    )
    minimum_seeds = 4 if profile == "smoke" else 20
    assert report["n_seeds"] >= minimum_seeds, "curve must be averaged over the configured multi-seed budget"
    assert report["robust_loss_param"] == pytest.approx(1.0)
    assert set(report["accuracy_ci_by_config"]) == set(configs)
    assert all(len(intervals) == len(levels) for intervals in report["accuracy_ci_by_config"].values())
    assert 0.4 in levels, (
        "the sweep must include the highest contamination level even though "
        "it doesn't favor the robust client — dropping the unfavorable point "
        "would be cherry-picking, not honest reporting"
    )


def test_pipeline_accepts_explicit_config(tmp_path: Path) -> None:
    _make_project(tmp_path)
    cfg = ExperimentConfig(n_agents=4, n_seeds=3, divergences=("KLD", "RKL"))
    paths = run_analysis_pipeline(config=cfg, project_root=tmp_path)
    report = json.loads(paths["belief_sharing_report"].read_text())
    assert report["n_agents"] == 4
    assert len(report["communicating_free_energy"]) == 3


def test_publication_profile_rejects_a_noncanonical_explicit_config(tmp_path: Path) -> None:
    """A publication sidecar cannot attest a caller-supplied reduced budget."""
    _make_project(tmp_path)
    cfg = ExperimentConfig(n_agents=4, n_seeds=3, divergences=("KLD", "RKL"))

    with pytest.raises(ValueError, match="canonical manuscript/config.yaml"):
        run_analysis_pipeline(config=cfg, project_root=tmp_path, profile="publication")


def test_main_prints_paths_and_returns_mapping(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _make_project(tmp_path)
    paths = main(project_root=tmp_path)
    out = capsys.readouterr().out
    assert "belief_sharing_report:" in out
    assert "robustness_sweep:" in out
    assert set(paths)  # non-empty
