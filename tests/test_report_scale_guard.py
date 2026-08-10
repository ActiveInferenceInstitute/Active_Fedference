"""Publication-scale guard for committed analysis reports (smoke-contamination tripwire).

The analysis workflow has a ``smoke`` profile that clamps the seeded budgets
(``n_seeds -> 4``, ``replicate_seeds -> 4``, ``n_trials -> 6``,
``cross_study_n_trials -> 4`` in ``src/analysis/workflow.py``) so temporary
projects can run the same real code paths cheaply. Three separate incidents
have now seen smoke-scale reports leak into the committed publication
``output/reports/`` tree. This gate is the tripwire that would have caught all
three: it asserts the committed reports carry the *publication-scale* n-fields
declared by ``manuscript/config.yaml`` -> ``experiment:`` (via
``load_experiment_config``), not the smoke clamps.

Expected values are derived from the config, never hardcoded, with one
deliberate exception: the smoke clamp ceilings themselves are pinned as floors
so a quietly smoke-scaled *config* cannot make the guard pass vacuously.

Root cause of the incident class (fixed at the ingestion point):
``tests/test_scripts_smoke.py`` ran ``scripts/02_run_analysis.py --profile
smoke`` and ``scripts/z_generate_manuscript_variables.py`` as real
subprocesses, and the scripts resolved the project root from their own file
location — so every full-suite run silently overwrote the committed
``output/reports/`` (n_seeds 240 -> 4), ``output/data/manuscript_variables.json``
and ``output/manuscript/`` with smoke-scale values. The fix: the pipeline
scripts and ``analysis.workflow._project_root`` honor a validated
``ACTIVE_FEDFERENCE_PROJECT_ROOT`` env override (``src/project_paths.py``),
and the smoke tests point it at a temporary scaffold. This guard remains the
order-dependent tripwire for any FUTURE writer that bypasses the override —
it detects contamination after the fact; the override prevents it at the
source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiment_config import load_experiment_config

_ROOT = Path(__file__).resolve().parent.parent
_REPORTS = _ROOT / "output" / "reports"

# The smoke-profile clamp ceilings from src/analysis/workflow.py. Publication
# budgets must sit strictly above them or the guard could pass on smoke output.
_SMOKE_N_SEEDS = 4
_SMOKE_REPLICATE_SEEDS = 4
_SMOKE_N_TRIALS = 6
_SMOKE_CROSS_STUDY_N_TRIALS = 4
_SMOKE_CELL_N_SEEDS = 4
_SMOKE_CELL_N_TRIALS = 4
_SMOKE_BNN_N_PER = 20

pytestmark = pytest.mark.publication


def _declared_profile() -> str:
    config_path = _ROOT / "manuscript" / "config.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = data.get("experiment", {}) or {}
    return str(experiment.get("analysis_profile", "publication"))


def _load_report(name: str) -> dict:
    path = _REPORTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet")
    return json.loads(path.read_text(encoding="utf-8"))


def _publication_config():
    if _declared_profile() != "publication":
        pytest.skip("project config does not declare the publication profile")
    cfg = load_experiment_config(_ROOT)
    # Guard the guard: the declared publication budgets must exceed the smoke
    # clamps, or a smoke-scaled config would make every check below vacuous.
    assert cfg.n_seeds > _SMOKE_N_SEEDS, f"config n_seeds={cfg.n_seeds} is within the smoke clamp"
    assert cfg.replicate_seeds > _SMOKE_REPLICATE_SEEDS, (
        f"config replicate_seeds={cfg.replicate_seeds} is within the smoke clamp"
    )
    assert cfg.n_trials > _SMOKE_N_TRIALS, f"config n_trials={cfg.n_trials} is within the smoke clamp"
    assert cfg.cross_study_n_trials > _SMOKE_CROSS_STUDY_N_TRIALS, (
        f"config cross_study_n_trials={cfg.cross_study_n_trials} is within the smoke clamp"
    )
    for label, value, ceiling in (
        ("conditional_world_n_seeds", cfg.conditional_world_n_seeds, _SMOKE_CELL_N_SEEDS),
        ("conditional_world_n_trials", cfg.conditional_world_n_trials, _SMOKE_CELL_N_TRIALS),
        ("review_grid_n_seeds", cfg.review_grid_n_seeds, _SMOKE_CELL_N_SEEDS),
        ("review_grid_n_trials", cfg.review_grid_n_trials, _SMOKE_CELL_N_TRIALS),
        ("gallery_n_seeds", cfg.gallery_n_seeds, 2),
        ("gallery_n_trials", cfg.gallery_n_trials, _SMOKE_CELL_N_TRIALS),
        ("onset_n_seeds", cfg.onset_n_seeds, 2),
        ("onset_n_trials", cfg.onset_n_trials, _SMOKE_CELL_N_TRIALS),
        ("bnn_n_seeds", cfg.bnn_n_seeds, _SMOKE_N_SEEDS),
        ("bnn_n_per", cfg.bnn_n_per, _SMOKE_BNN_N_PER),
    ):
        assert value > ceiling, f"config {label}={value} is within the smoke clamp"
    return cfg


def test_belief_sharing_report_is_publication_scale() -> None:
    cfg = _publication_config()
    report = _load_report("belief_sharing.json")
    assert report["n_seeds"] == cfg.n_seeds, (
        f"belief_sharing.json n_seeds={report['n_seeds']} != declared "
        f"publication n_seeds={cfg.n_seeds} — smoke-scale contamination?"
    )
    assert report["n_agents"] == cfg.n_agents
    # The per-seed arrays must actually carry that many entries.
    assert len(report["communicating_free_energy"]) == cfg.n_seeds
    assert len(report["incommunicado_free_energy"]) == cfg.n_seeds


def test_robustness_sweep_report_is_publication_scale() -> None:
    cfg = _publication_config()
    report = _load_report("robustness_sweep.json")
    per_divergence = report["accuracy_at_verdict_rate"]
    assert set(per_divergence) == set(cfg.divergences)
    for divergence, cell in per_divergence.items():
        assert cell["n"] == cfg.n_trials, (
            f"robustness_sweep.json accuracy_at_verdict_rate[{divergence!r}].n="
            f"{cell['n']} != declared publication n_trials={cfg.n_trials} — "
            "smoke-scale contamination?"
        )


def test_cross_study_summary_report_is_publication_scale() -> None:
    cfg = _publication_config()
    report = _load_report("cross_study_summary.json")
    assert report["n_seeds"] == cfg.replicate_seeds, (
        f"cross_study_summary.json n_seeds={report['n_seeds']} != declared "
        f"publication replicate_seeds={cfg.replicate_seeds} — smoke-scale contamination?"
    )
    assert report["n_trials"] == cfg.cross_study_n_trials, (
        f"cross_study_summary.json n_trials={report['n_trials']} != declared "
        f"publication cross_study_n_trials={cfg.cross_study_n_trials} — "
        "smoke-scale contamination?"
    )


def test_seed_nested_reports_are_publication_scale() -> None:
    cfg = _publication_config()
    for name in ("conditional_world.json", "belief_quality.json"):
        report = _load_report(name)
        assert report["n_seeds"] == cfg.conditional_world_n_seeds
        assert report["n_trials"] == cfg.conditional_world_n_trials
    review = _load_report("robustness_review_grid.json")
    assert review["n_seeds"] == cfg.review_grid_n_seeds
    assert review["n_trials"] == cfg.review_grid_n_trials
    assert review["precision_plan"]["target_max_mcse"] == pytest.approx(cfg.review_grid_target_max_mcse)
    assert review["precision_plan"]["target_met"] is True


def test_gallery_onset_and_bnn_reports_are_publication_scale() -> None:
    cfg = _publication_config()
    gallery = _load_report("contamination_gallery.json")
    assert gallery["n_seeds"] == cfg.gallery_n_seeds
    assert gallery["n_trials"] == cfg.gallery_n_trials
    onset = _load_report("robustness_onset.json")
    assert onset["n_seeds"] == cfg.onset_n_seeds
    assert onset["n_trials"] == cfg.onset_n_trials
    bnn = _load_report("bnn_robustness.json")
    assert bnn["n_seeds"] == cfg.bnn_n_seeds
    assert bnn["n_per"] == cfg.bnn_n_per
