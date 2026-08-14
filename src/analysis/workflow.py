"""Thin orchestrator for the Active Fedference analysis pipeline.

Wires the locked :mod:`fedference.experiments` categorical source-mechanism
analogues of Friston et al. (2024), *Federated inference and belief sharing* (Neurosci. Biobehav. Rev.
156:105500) to disk: it runs the experiments across the configured seeds, writes
JSON result reports to ``output/reports/`` and the registered figures to
``output/figures/``, and returns/prints the artifact paths. **All numerics live
in the imported core modules** (``fedference.*``) and figure modules
(:mod:`figures`); this file only aggregates, serialises and plots — the thin
orchestrator contract. No ``infrastructure.*`` imports (layer contract).

Three-robustness-axes honesty: the logistic-regression robustness figure
exercises the per-client FedGVI generalized-Bayes update (rcce loss + AR
regularizer), which carries the client-side robustness claim; the
influence-weights figure exercises the server-side ``robust_aggregate`` pooling
*heuristic*, which only owns the naive-recovery limit; and the variational
diagnostics exercise the conservative objective-backed server rule. The three
are sourced from different core calls and never conflated.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

from experiment_config import ExperimentConfig, load_experiment_config, load_manuscript_config
from fedference.aggregation import log_linear_pool
from fedference.belief_updating import infer_states
from fedference.experiments import (
    disjoint_fov_report,
    hierarchical_world_report,
    moving_world_report,
    nlevel3_world_report,
    run_belief_quality_sensitivity,
    run_belief_sharing,
    run_bnn_robustness_report,
    run_complexity_scaling,
    run_conditional_world_generalization,
    run_contamination_gallery,
    run_efe_decomposition_report,
    run_emergence,
    run_heuristic_characterization,
    run_hierarchical_bmr,
    run_influence_weights_report,
    run_parameter_recovery,
    run_review_grid,
    run_robustness_onset,
    run_robustness_sweep,
    run_variational_aggregation_report,
    summarize_cross_study,
    summarize_language_acquisition,
)
from fedference.pomdp import N_LOCATIONS, build_sentinel_world
from figures import (
    generate_aggregation_descent,
    generate_belief_heatmap,
    generate_belief_quality,
    generate_bnn_robustness,
    generate_bounded_influence,
    generate_complexity_scaling,
    generate_conditional_world,
    generate_contamination_gallery,
    generate_cross_study_summary,
    generate_descent_comparison,
    generate_disjoint_fov_figure,
    generate_efe_decomposition,
    generate_emergence_bmr,
    generate_free_energy_comparison,
    generate_generative_model_schema,
    generate_graphical_abstract,
    generate_heuristic_breakdown,
    generate_hierarchical_bmr,
    generate_hierarchical_pomdp,
    generate_language_kl_decay,
    generate_message_passing,
    generate_moving_world,
    generate_parameter_recovery,
    generate_pomdp_loop,
    generate_robust_influence_weights,
    generate_robustness_onset,
    generate_robustness_review_grid,
    generate_robustness_sweep,
    generate_sensitivity_heatmap,
)
from figures._metadata import figure_metadata
from project_paths import resolve_env_project_root
from publication.pipeline_freshness import capture_analysis_input_snapshot

from . import report_schemas

#: Drift levels documented in diagnostics module; kept for figure-registry metadata.
_INFLUENCE_DRIFTS: tuple[float, ...] = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.99)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_MANUSCRIPT_FIGURE_PATTERN = re.compile(
    r"!\[(?P<caption>.*?)\]\((?P<path>[^)]+)\)\{#(?P<label>fig:[-A-Za-z0-9_.:]+)(?P<attrs>[^}]*)\}",
    re.DOTALL,
)
_SKIP_MANUSCRIPT_DOCS = frozenset({"AGENTS.md", "README.md", "SYNTAX.md"})


def _project_root() -> Path:
    # ACTIVE_FEDFERENCE_PROJECT_ROOT (validated) wins; otherwise the repo root.
    return resolve_env_project_root(_PROJECT_ROOT)


def _reports_dir(project_root: Path) -> Path:
    out = project_root / "output" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(
    payload: Mapping[str, object],
    path: Path,
    *,
    schema: str | None = None,
) -> Path:
    if schema is not None:
        report_schemas.validate_report(schema, payload)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def _normalise_caption(caption: str) -> str:
    return " ".join(caption.split())


def _declared_manuscript_figures(project_root: Path) -> list[dict[str, str]]:
    source_manuscript = project_root / "manuscript"
    if not source_manuscript.exists():
        return []
    # Figure ownership and captions are an analysis input contract. Read only
    # the source tree: consulting an older hydrated tree would make analysis
    # outputs depend on a downstream stage and could promote stale tokens.
    source_files = sorted(source_manuscript.glob("*.md"))
    figures: list[dict[str, str]] = []
    for path in source_files:
        if path.name in _SKIP_MANUSCRIPT_DOCS:
            continue
        text = path.read_text(encoding="utf-8")
        source_path = source_manuscript / path.name
        for match in _MANUSCRIPT_FIGURE_PATTERN.finditer(text):
            figure_path = Path(match.group("path"))
            figures.append(
                {
                    "label": match.group("label"),
                    "filename": figure_path.name,
                    "path": str(Path("output") / "figures" / figure_path.name),
                    "source_manuscript": str(source_path.relative_to(project_root)),
                    "caption": _normalise_caption(match.group("caption")),
                }
            )
    return figures


def _figure_generators_by_filename(project_root: Path) -> dict[str, str]:
    """Map each ``output/figures/*.png`` to the :mod:`figures` submodule that draws it.

    The analysis pipeline (this module) produces most figures and records their
    artifact path, but a few figures — the standalone introductory / cover
    schematics — are generated by their own :mod:`figures` submodule outside the
    analysis pass. Scanning ``src/figures`` for the matching module keeps the
    registry honest about *which* generator produced every embedded figure,
    instead of falling back to ``"preexisting_figure"`` for the ones the pipeline
    did not return a path for.
    """
    figures_pkg = project_root / "src" / "figures"
    mapping: dict[str, str] = {}
    if not figures_pkg.is_dir():
        return mapping
    for mod in sorted(figures_pkg.glob("*.py")):
        if mod.name in ("__init__.py", "_common.py"):
            continue
        stem = mod.stem
        # Generators write both a .png (embedded) and often a sibling .pdf.
        for ext in (".png", ".pdf"):
            mapping[f"{stem}{ext}"] = stem
    return mapping


def _write_figure_registry(project_root: Path, artifact_paths: dict[str, Path]) -> Path:
    # Pipeline-produced figures (keyed by the .png they write, since the
    # manuscript embeds the PNG even when the generator returns the .pdf path).
    generated_by_filename = {
        path.name: artifact_key
        for artifact_key, path in artifact_paths.items()
        if path.parent.name == "figures"
    }
    # Fall back to the :mod:`figures` submodule that owns the filename, so
    # standalone generators (system_overview, graphical_abstract, moving_world)
    # are credited rather than labelled "preexisting_figure".
    submodule_by_filename = _figure_generators_by_filename(project_root)

    def _resolve_generated_by(filename: str) -> str:
        if filename in generated_by_filename:
            return generated_by_filename[filename]
        if filename in submodule_by_filename:
            return submodule_by_filename[filename]
        return "preexisting_figure"

    figures: list[dict[str, str]] = []
    seen: set[str] = set()
    for figure in _declared_manuscript_figures(project_root):
        label = figure["label"]
        if label in seen:
            continue
        seen.add(label)
        filename = figure["filename"]
        figure_path = project_root / "output" / "figures" / filename
        if not figure_path.exists():
            raise FileNotFoundError(f"Declared manuscript figure is missing: {figure_path}")
        generator = _resolve_generated_by(filename)
        if generator == "preexisting_figure":
            raise ValueError(f"no generator metadata owner for embedded figure {filename}")
        figures.append(
            {
                **figure,
                "generated_by": generator,
                **figure_metadata(generator),
            }
        )
    return _write_json(
        {
            "schema_version": "1.1",
            "generated_by": "analysis.workflow.run_analysis_pipeline",
            "figures": sorted(figures, key=lambda item: item["label"]),
        },
        project_root / "output" / "figures" / "figure_registry.json",
        schema="figure_registry",
    )


def _belief_sharing_report(config: ExperimentConfig) -> dict:
    """Aggregate per-seed belief-sharing free energies (communicating vs not)."""
    communicating: list[float] = []
    incommunicado: list[float] = []
    for seed in config.seeds:
        comm = run_belief_sharing(seed, communicate=True, n_agents=config.n_agents)
        incom = run_belief_sharing(seed, communicate=False, n_agents=config.n_agents)
        communicating.append(float(comm["mean_free_energy"]))
        incommunicado.append(float(incom["mean_free_energy"]))
    comm_mean = float(np.mean(communicating))
    incom_mean = float(np.mean(incommunicado))
    return {
        "communicating_free_energy": communicating,
        "incommunicado_free_energy": incommunicado,
        "communicating_mean": comm_mean,
        "incommunicado_mean": incom_mean,
        "free_energy_gap": incom_mean - comm_mean,
        "communication_helps": bool(comm_mean < incom_mean),
        "n_agents": config.n_agents,
        "n_seeds": config.n_seeds,
    }


def _language_report(config: ExperimentConfig) -> dict:
    """Seed-level language-acquisition trajectory summary for publication."""
    return summarize_language_acquisition(config.seeds)


def _emergence_report(config: ExperimentConfig) -> dict:
    """Bayesian-model-reduction emergence contrast at the first configured seed."""
    seed = config.seeds[0]
    result = run_emergence(seed)
    return dict(result)


def _robustness_report(config: ExperimentConfig) -> dict:
    """Robustness sweep at the first seed, using the configured rates/divergences.

    ``run_robustness_sweep`` does not echo back ``n_agents`` / ``n_contaminated``,
    so we record the colony shape we asked for alongside its output (the
    manuscript prose reports both).
    """
    seed = config.seeds[0]
    n_contaminated = max(1, config.n_agents // 3)
    report = run_robustness_sweep(
        seed,
        rates=config.contamination_rates,
        divergences=config.divergences,
        n_agents=config.n_agents,
        n_contaminated=n_contaminated,
        n_trials=config.n_trials,
        fdr_alpha=config.fdr_alpha,
        power_alpha=config.power_alpha,
        power_alternative=config.power_alternative,
        target_power=config.target_power,
    )
    report["n_agents"] = config.n_agents
    report["n_contaminated"] = n_contaminated
    return report


def _efe_terms(config: ExperimentConfig) -> dict:
    """Closed-form EFE decomposition of one sentinel policy."""
    return run_efe_decomposition_report(config.seeds[0])


def _influence_weights(config: ExperimentConfig) -> dict:
    """Server-side robust pooling influence weights on a contaminated colony."""
    return run_influence_weights_report(config.seeds[0], n_agents=config.n_agents)


def _contamination_gallery_report(config: ExperimentConfig, *, smoke: bool = False) -> dict:
    """Descriptive pooled-method gallery across the declared mechanisms.

    Runs :func:`fedference.experiments.run_contamination_gallery` over
    confident-wrong, byzantine, drift, uniform, and label-noise mechanisms. The
    gallery chooses a pooled display method from these same seed-level results,
    so it is not the selection-free inferential surface and cannot establish a
    mechanism-wide or universal robustness conclusion. The review grid retains
    every configured method and owns the all-method signed comparisons.
    """
    n_contaminated = max(1, config.n_agents // 3)
    # The frozen config owns every stochastic budget. Smoke uses the same real
    # code path with explicitly smaller units for temporary-project tests.
    return run_contamination_gallery(
        config.seeds[0],
        n_agents=config.n_agents,
        n_contaminated=n_contaminated,
        divergences=config.divergences,
        n_trials=min(config.gallery_n_trials, 4) if smoke else config.gallery_n_trials,
        n_seeds=min(config.gallery_n_seeds, 2) if smoke else config.gallery_n_seeds,
    )


def _robustness_onset_report(config: ExperimentConfig, *, smoke: bool = False) -> dict:
    """Per-mechanism robustness-onset curves for the pooled display contrast.

    Maps the rate dependence the single-strength gallery cannot: for each
    directional mechanism, the naive and pooled-display accuracy curves over a rate
    grid plus the onset rate. Its seed and nested-trial budgets are frozen in
    :class:`ExperimentConfig`, so the publication profile cannot silently use
    the former hard-coded small grid.
    """
    n_contaminated = max(1, config.n_agents // 3)
    return run_robustness_onset(
        config.seeds[0],
        n_agents=config.n_agents,
        n_contaminated=n_contaminated,
        divergences=config.divergences,
        n_trials=min(config.onset_n_trials, 4) if smoke else config.onset_n_trials,
        n_seeds=min(config.onset_n_seeds, 2) if smoke else config.onset_n_seeds,
    )


def _variational_aggregation_report(config: ExperimentConfig) -> dict:
    """Diagnostics for the objective-backed variational aggregator."""
    return run_variational_aggregation_report(config.seeds[0], n_agents=config.n_agents)


def _bnn_report(config: ExperimentConfig, *, smoke: bool = False) -> dict:
    """Logistic-regression held-out accuracy vs label contamination."""
    if smoke:
        return run_bnn_robustness_report(
            config.seeds[0],
            n_seeds=min(config.bnn_n_seeds, 4),
            n_per=min(config.bnn_n_per, 20),
        )
    return run_bnn_robustness_report(
        config.seeds[0],
        n_seeds=config.bnn_n_seeds,
        n_per=config.bnn_n_per,
    )


class BnnTorchOptions(TypedDict, total=False):
    """Validated optional keyword arguments for the Torch complement."""

    n_clients: int
    n_per: int
    hidden_dim: int
    n_steps: int
    robustness: float
    beta: float
    contamination_levels: tuple[float, ...]


_HERMETIC_GIT = ("git", "-c", "core.fsmonitor=false", "-c", "core.untrackedcache=false")


def _bnn_torch_options(project_root: Path | None) -> BnnTorchOptions:
    """Read the optional executed Torch-lane profile from project config.

    The main :class:`ExperimentConfig` intentionally remains the source of
    truth for the categorical studies. The Torch complement has additional
    architecture/training knobs, so it accepts a separate explicit
    ``experiment.bnn_torch`` block. This makes smoke fixtures cheap without
    silently changing the publication profile or hiding a test-only branch.
    """
    if project_root is None:
        return {}
    path = project_root / "manuscript" / "config.yaml"
    if not path.exists():
        return {}
    data = load_manuscript_config(project_root)
    block = data["experiment"].get("bnn_torch", {})
    if not isinstance(block, Mapping):  # defensive: the shared loader enforces this
        raise ValueError("experiment.bnn_torch must be a mapping when provided")

    def _integer(value: object, *, key: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"experiment.bnn_torch.{key} must be an integer")
        return value

    def _real(value: object, *, key: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"experiment.bnn_torch.{key} must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"experiment.bnn_torch.{key} must be finite")
        return number

    options: dict[str, object] = {}
    for key in ("n_clients", "n_per", "hidden_dim", "n_steps"):
        if key in block:
            value = _integer(block[key], key=key)
            if value < 1:
                raise ValueError(f"experiment.bnn_torch.{key} must be >= 1")
            options[key] = value
    for key in ("robustness", "beta"):
        if key in block:
            value_real = _real(block[key], key=key)
            if value_real < 0.0:
                raise ValueError(f"experiment.bnn_torch.{key} must be >= 0")
            options[key] = value_real
    if "contamination_levels" in block:
        raw_levels = block["contamination_levels"]
        if not isinstance(raw_levels, (list, tuple)):
            raise ValueError("experiment.bnn_torch.contamination_levels must be a list of numbers")
        levels = tuple(_real(value, key="contamination_levels[]") for value in raw_levels)
        if not levels or any(not 0.0 <= value <= 1.0 for value in levels):
            raise ValueError("experiment.bnn_torch.contamination_levels must be non-empty in [0, 1]")
        options["contamination_levels"] = levels
    return cast(BnnTorchOptions, options)


def resolve_analysis_profile(project_root: Path | None, *, override: str | None = None) -> str:
    """Return the effective workflow budget profile for a project root.

    ``publication`` is the default and preserves the declared study budgets.
    ``smoke`` is a lower-budget real execution profile used by repeated
    temporary-project tests; it is never used by the shipped configuration.
    """
    if override is not None:
        profile = str(override)
    elif project_root is None:
        profile = "publication"
    else:
        path = project_root / "manuscript" / "config.yaml"
        if not path.exists():
            profile = "publication"
        else:
            data = load_manuscript_config(project_root)
            profile = str(data["experiment"].get("analysis_profile", "publication"))
    if profile not in {"publication", "smoke"}:
        raise ValueError("analysis_profile/analysis profile must be 'publication' or 'smoke'")
    return profile


def _bnn_torch_report(config: ExperimentConfig, *, project_root: Path | None = None) -> dict:
    """PyTorch point-mass MLP FedGVI complement — executed when torch is present.

    Runs :func:`fedference.bnn_baseline_torch.run_bnn_torch_experiment`, whose
    every number is an executed result. When PyTorch is not installed the report
    records a ``skipped`` status so the manuscript tokens degrade to a clearly
    labeled ``N/A`` rather than a fabricated value.
    """
    try:
        from fedference.bnn_baseline_torch import run_bnn_torch_experiment
    except ImportError as exc:  # PyTorch optional-extra not installed
        return {"status": f"skipped: PyTorch unavailable ({exc})"}
    options = _bnn_torch_options(project_root)
    return dict(run_bnn_torch_experiment(seed=config.seeds[0], **options))


def _heatmap_inputs(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build a seeded colony of posteriors plus their consensus for the heatmap."""
    rng = np.random.default_rng(config.seeds[0])
    world = build_sentinel_world(rng, acuity=0.6)
    A = np.asarray(world["A"][0], dtype=np.float64)  # type: ignore[index]
    n_s = int(N_LOCATIONS)
    true_state = int(rng.integers(0, n_s))
    log_prior = np.log(np.full(n_s, 1.0 / n_s))
    local_posteriors = np.empty((config.n_agents, n_s), dtype=np.float64)
    for n in range(config.n_agents):
        column = A[:, true_state]
        o = int(rng.choice(n_s, p=column / column.sum()))
        local_posteriors[n] = infer_states(A, o, log_prior)
    consensus = log_linear_pool(local_posteriors=local_posteriors)
    return local_posteriors, consensus


def _moving_world_report(config: ExperimentConfig) -> dict:
    """Moving-world experiment: isolated vs communicating vs EFE-guided (V4)."""
    seed = config.seeds[0]
    return moving_world_report(
        seed,
        n_trials=config.n_trials,
        n_multiseed=config.replicate_seeds,
    )


def _parameter_recovery_report(config: ExperimentConfig, *, smoke: bool = False) -> dict:
    """Parameter recovery: validate generative-model identifiability (Study 9)."""
    seed = config.seeds[0]
    return dict(
        run_parameter_recovery(
            seed,
            n_trials=config.n_trials,
            n_observations=24 if smoke else 200,
            fit_resolution=20 if smoke else 80,
        )
    )


def _conditional_world_report(config: ExperimentConfig) -> dict:
    """Conditional world/attack geometry grid with seed-level contrasts."""
    return run_conditional_world_generalization(
        config.seeds[0],
        n_seeds=config.conditional_world_n_seeds,
        n_trials=config.conditional_world_n_trials,
        n_agents=config.n_agents,
        robustness=1.5,
    )


def _review_grid_report(config: ExperimentConfig, *, smoke: bool = False) -> dict:
    """Run the bounded source-bound review grid with explicit seed nesting."""
    return run_review_grid(
        config.seeds[0],
        n_seeds=min(config.review_grid_n_seeds, 4) if smoke else config.review_grid_n_seeds,
        n_trials=min(config.review_grid_n_trials, 4) if smoke else config.review_grid_n_trials,
        n_agents=config.n_agents,
        robustness=1.5,
        rates=config.review_grid_rates,
        divergences=config.divergences,
        fdr_alpha=config.fdr_alpha,
        power_alpha=config.power_alpha,
        planning_alternative=config.power_alternative,
        # Smoke verifies the same producer but intentionally does not claim to
        # meet the publication precision stopping rule.
        target_max_mcse=None if smoke else config.review_grid_target_max_mcse,
    )


def _belief_quality_report(config: ExperimentConfig) -> dict:
    """Proper-score and reliability diagnostics on the conditional subset."""
    return run_belief_quality_sensitivity(
        config.seeds[0],
        n_seeds=config.conditional_world_n_seeds,
        n_trials=config.conditional_world_n_trials,
        n_agents=config.n_agents,
        robustness=1.5,
    )


def run_analysis_pipeline(
    config: ExperimentConfig | None = None,
    *,
    project_root: Path | None = None,
    profile: str | None = None,
) -> dict[str, Path]:
    """Run every fedference experiment, write reports + figures, return paths.

    Args:
        config: Experiment configuration; loaded from ``manuscript/config.yaml``
            when omitted.
        project_root: Output root override (defaults to the project root).
        profile: Optional explicit budget profile. ``publication`` is the
            default; ``smoke`` uses the same real code paths with bounded
            budgets for repeated verification runs.

    Returns:
        Mapping of artifact label -> written path (reports and figures).
    """
    root = Path(project_root or _project_root()).resolve()
    configured_profile = resolve_analysis_profile(root)
    effective_profile = resolve_analysis_profile(root, override=profile)
    canonical_config = load_experiment_config(root)
    if config is not None and effective_profile == "publication" and config != canonical_config:
        raise ValueError(
            "publication analysis requires the canonical manuscript/config.yaml ExperimentConfig; "
            "an explicit non-equivalent config cannot mint publication evidence"
        )
    cfg = config or canonical_config
    # Publication-scale analysis is long-running. Capture its declared inputs
    # before it begins so the subsequent receipt cannot silently attest a
    # source/configuration edit that occurred during the run. Smoke runs never
    # mint a publication receipt and may use deliberately minimal test roots.
    analysis_input_snapshot = (
        capture_analysis_input_snapshot(root) if effective_profile == "publication" else None
    )
    smoke = effective_profile == "smoke"
    if smoke:
        # An explicitly supplied ExperimentConfig can carry publication-scale
        # values even when the temporary project declares smoke. Clamp only
        # the nested workflow budgets; the same real code paths still execute.
        cfg = replace(
            cfg,
            n_seeds=min(cfg.n_seeds, 4),
            replicate_seeds=min(cfg.replicate_seeds, 4),
            n_trials=min(cfg.n_trials, 6),
            cross_study_n_trials=min(cfg.cross_study_n_trials, 4),
            conditional_world_n_seeds=min(cfg.conditional_world_n_seeds, 4),
            conditional_world_n_trials=min(cfg.conditional_world_n_trials, 4),
            review_grid_n_seeds=min(cfg.review_grid_n_seeds, 4),
            review_grid_n_trials=min(cfg.review_grid_n_trials, 4),
            gallery_n_seeds=min(cfg.gallery_n_seeds, 2),
            gallery_n_trials=min(cfg.gallery_n_trials, 4),
            onset_n_seeds=min(cfg.onset_n_seeds, 2),
            onset_n_trials=min(cfg.onset_n_trials, 4),
            bnn_n_seeds=min(cfg.bnn_n_seeds, 4),
            bnn_n_per=min(cfg.bnn_n_per, 20),
        )
    reports = _reports_dir(root)

    paths: dict[str, Path] = {}

    # Timing accumulators — written to output/data/stage_timings.json at the end.
    _timings: dict[str, float] = {
        "total": 0.0,
        "moving_world": 0.0,
        "hierarchical": 0.0,
        "sensitivity": 0.0,
        "parameter_recovery": 0.0,
        "complexity": 0.0,
    }
    _pipeline_start = time.time()

    # --- Reports -----------------------------------------------------------
    bs_report = _belief_sharing_report(cfg)
    paths["belief_sharing_report"] = _write_json(
        bs_report, reports / "belief_sharing.json", schema="belief_sharing"
    )

    lang_report = _language_report(cfg)
    paths["language_report"] = _write_json(
        lang_report,
        reports / "language_acquisition.json",
        schema="language_acquisition",
    )

    emergence_report = _emergence_report(cfg)
    paths["emergence_report"] = _write_json(emergence_report, reports / "emergence.json", schema="emergence")

    rob_report = _robustness_report(cfg)
    paths["robustness_report"] = _write_json(
        rob_report, reports / "robustness_sweep.json", schema="robustness_sweep"
    )

    hier_bmr_report = run_hierarchical_bmr()
    paths["hierarchical_bmr_report"] = _write_json(
        hier_bmr_report, reports / "hierarchical_bmr.json", schema="hierarchical_bmr"
    )

    heuristic_report = run_heuristic_characterization(cfg.seeds[0])
    paths["heuristic_characterization_report"] = _write_json(
        heuristic_report,
        reports / "heuristic_characterization.json",
        schema="heuristic_characterization",
    )

    efe_report = _efe_terms(cfg)
    paths["efe_report"] = _write_json(
        efe_report, reports / "efe_decomposition.json", schema="efe_decomposition"
    )

    weights_report = _influence_weights(cfg)
    paths["influence_weights_report"] = _write_json(
        weights_report,
        reports / "robust_influence_weights.json",
        schema="robust_influence_weights",
    )

    bnn_report = _bnn_report(cfg, smoke=smoke)
    paths["bnn_report"] = _write_json(bnn_report, reports / "bnn_robustness.json", schema="bnn_robustness")

    bnn_torch_report = _bnn_torch_report(cfg, project_root=root)
    paths["bnn_torch_report"] = _write_json(bnn_torch_report, reports / "bnn_torch.json", schema="bnn_torch")

    var_report = _variational_aggregation_report(cfg)
    paths["variational_aggregation_report"] = _write_json(
        var_report,
        reports / "variational_aggregation.json",
        schema="variational_aggregation",
    )

    gallery_report = _contamination_gallery_report(cfg, smoke=smoke)
    paths["contamination_gallery_report"] = _write_json(
        gallery_report,
        reports / "contamination_gallery.json",
        schema="contamination_gallery",
    )

    onset_report = _robustness_onset_report(cfg, smoke=smoke)
    paths["robustness_onset_report"] = _write_json(
        onset_report, reports / "robustness_onset.json", schema="robustness_onset"
    )

    conditional_report = _conditional_world_report(cfg)
    paths["conditional_world_report"] = _write_json(
        conditional_report,
        reports / "conditional_world.json",
        schema="conditional_world",
    )

    review_grid_report = _review_grid_report(cfg, smoke=smoke)
    paths["review_grid_report"] = _write_json(
        review_grid_report,
        reports / "robustness_review_grid.json",
        schema="robustness_review_grid",
    )

    quality_report = _belief_quality_report(cfg)
    paths["belief_quality_report"] = _write_json(
        quality_report,
        reports / "belief_quality.json",
        schema="belief_quality",
    )

    _t0 = time.time()
    complexity_config = cfg.complexity.for_smoke() if smoke else cfg.complexity
    complexity_report = run_complexity_scaling(complexity_config)
    _timings["complexity"] = time.time() - _t0
    paths["complexity_scaling_report"] = _write_json(
        complexity_report,
        reports / "complexity_scaling.json",
        schema="complexity_scaling",
    )

    _t0 = time.time()
    moving_report = _moving_world_report(cfg)
    _timings["moving_world"] = time.time() - _t0
    paths["moving_world_report"] = _write_json(
        moving_report, reports / "moving_world.json", schema="moving_world"
    )

    _t0 = time.time()
    param_rec_report = _parameter_recovery_report(cfg, smoke=smoke)
    _timings["parameter_recovery"] = time.time() - _t0
    paths["parameter_recovery_report"] = _write_json(
        param_rec_report,
        reports / "parameter_recovery.json",
        schema="parameter_recovery",
    )

    _t0 = time.time()
    hier_report = hierarchical_world_report(
        cfg.seeds[0], n_trials=cfg.n_trials, n_multiseed=cfg.replicate_seeds
    )
    nl3_report = nlevel3_world_report(cfg.seeds[0], n_trials=cfg.n_trials, n_multiseed=cfg.replicate_seeds)
    cross_report = summarize_cross_study(
        cfg.seeds[0],
        n_seeds=cfg.replicate_seeds,
        n_trials=cfg.cross_study_n_trials,
    )
    disjoint_report = disjoint_fov_report(0, n_multiseed=cfg.replicate_seeds)
    _timings["hierarchical"] = time.time() - _t0
    paths["hierarchical_world_report"] = _write_json(
        hier_report, reports / "hierarchical_world.json", schema="hierarchical_world"
    )
    paths["nlevel3_world_report"] = _write_json(
        nl3_report, reports / "nlevel3_world.json", schema="nlevel3_world"
    )
    paths["cross_study_report"] = _write_json(
        cross_report, reports / "cross_study_summary.json", schema="cross_study_summary"
    )
    paths["disjoint_fov_report"] = _write_json(
        disjoint_report, reports / "disjoint_fov_world.json", schema="disjoint_fov_world"
    )

    # --- Figures (consume the report dicts above) --------------------------
    paths["graphical_abstract"] = generate_graphical_abstract(project_root=root)
    paths["generative_model_schema"] = generate_generative_model_schema(project_root=root)
    paths["message_passing"] = generate_message_passing(project_root=root)
    paths["pomdp_loop"] = generate_pomdp_loop(project_root=root)

    local_posteriors, consensus = _heatmap_inputs(cfg)
    paths["belief_heatmap"] = generate_belief_heatmap(local_posteriors, consensus, project_root=root)
    report_schemas.check_figure_contract("free_energy_comparison", "belief_sharing", bs_report)
    paths["free_energy_comparison"] = generate_free_energy_comparison(
        bs_report["incommunicado_free_energy"],
        bs_report["communicating_free_energy"],
        project_root=root,
    )
    report_schemas.check_figure_contract("robustness_sweep", "robustness_sweep", rob_report)
    paths["robustness_sweep"] = generate_robustness_sweep(
        rob_report["accuracy_by_method_and_rate"],
        cfg.contamination_rates,
        accuracy_threshold=rob_report.get("accuracy_threshold"),
        rate_summary=rob_report.get("per_rate_summary"),
        project_root=root,
    )
    report_schemas.check_figure_contract("language_kl_decay", "language_acquisition", lang_report)
    paths["language_kl_decay"] = generate_language_kl_decay(
        lang_report["kl_trajectory"],
        trajectory_ci=(lang_report["trajectory_ci_lo"], lang_report["trajectory_ci_hi"]),
        monotone_decreasing=lang_report.get("monotone_decreasing"),
        n_seeds=lang_report.get("n_seeds"),
        project_root=root,
    )
    report_schemas.check_figure_contract("emergence_bmr", "emergence", emergence_report)
    paths["emergence_bmr"] = generate_emergence_bmr(
        emergence_report["delta_F_redundant"],
        emergence_report["delta_F_supported"],
        convergence=emergence_report.get("convergence"),
        project_root=root,
    )
    report_schemas.check_figure_contract("hierarchical_bmr", "hierarchical_bmr", hier_bmr_report)
    paths["hierarchical_bmr"] = generate_hierarchical_bmr(
        hier_bmr_report["degenerate"],
        hier_bmr_report["informative"],
        project_root=root,
    )
    report_schemas.check_figure_contract(
        "heuristic_breakdown", "heuristic_characterization", heuristic_report
    )
    paths["heuristic_breakdown"] = generate_heuristic_breakdown(heuristic_report, project_root=root)
    report_schemas.check_figure_contract("efe_decomposition", "efe_decomposition", efe_report)
    paths["efe_decomposition"] = generate_efe_decomposition(
        efe_report["risk"],
        efe_report["ambiguity"],
        efe_report["pragmatic_value"],
        efe_report["epistemic_value"],
        project_root=root,
    )
    report_schemas.check_figure_contract(
        "robust_influence_weights", "robust_influence_weights", weights_report
    )
    paths["robust_influence_weights"] = generate_robust_influence_weights(
        weights_report["normalized_effective_weights"],
        weights_report["contaminated_indices"],
        project_root=root,
    )
    report_schemas.check_figure_contract("bnn_robustness", "bnn_robustness", bnn_report)
    paths["bnn_robustness"] = generate_bnn_robustness(
        bnn_report["accuracy_by_config"],
        bnn_report["contamination_levels"],
        accuracy_ci_by_config=bnn_report.get("accuracy_ci_by_config"),
        project_root=root,
    )
    report_schemas.check_figure_contract("aggregation_descent", "variational_aggregation", var_report)
    paths["aggregation_descent"] = generate_aggregation_descent(
        var_report["free_energy_history"],
        converged=var_report["converged"],
        project_root=root,
    )
    report_schemas.check_figure_contract("bounded_influence", "variational_aggregation", var_report)
    paths["bounded_influence"] = generate_bounded_influence(
        var_report["drifts"],
        var_report["variational_influence"],
        var_report["naive_influence"],
        project_root=root,
    )
    report_schemas.check_figure_contract("contamination_gallery", "contamination_gallery", gallery_report)
    paths["contamination_gallery"] = generate_contamination_gallery(
        gallery_report["by_kind"], project_root=root
    )
    report_schemas.check_figure_contract("descent_comparison", "variational_aggregation", var_report)
    paths["descent_comparison"] = generate_descent_comparison(
        var_report["single_start_history"],
        var_report["multi_start_history"],
        project_root=root,
    )
    report_schemas.check_figure_contract("robustness_onset", "robustness_onset", onset_report)
    paths["robustness_onset"] = generate_robustness_onset(onset_report["by_kind"], project_root=root)
    report_schemas.check_figure_contract("conditional_world", "conditional_world", conditional_report)
    paths["conditional_world"] = generate_conditional_world(conditional_report, project_root=root)
    report_schemas.check_figure_contract(
        "robustness_review_grid", "robustness_review_grid", review_grid_report
    )
    paths["robustness_review_grid"] = generate_robustness_review_grid(review_grid_report, project_root=root)
    report_schemas.check_figure_contract("belief_quality", "belief_quality", quality_report)
    paths["belief_quality"] = generate_belief_quality(quality_report, project_root=root)
    report_schemas.check_figure_contract("complexity_scaling", "complexity_scaling", complexity_report)
    paths["complexity_scaling"] = generate_complexity_scaling(complexity_report, project_root=root)
    report_schemas.check_figure_contract("moving_world", "moving_world", moving_report)
    paths["moving_world"] = generate_moving_world(moving_report, project_root=root)
    report_schemas.check_figure_contract("parameter_recovery", "parameter_recovery", param_rec_report)
    paths["parameter_recovery"] = generate_parameter_recovery(
        param_rec_report["true_acuity"],
        param_rec_report["recovered_acuity"],
        param_rec_report["recovered_acuity_ci_lo"],
        param_rec_report["recovered_acuity_ci_hi"],
        param_rec_report["abs_error"],
        r_squared=param_rec_report.get("r_squared"),
        mean_abs_error=param_rec_report.get("mean_abs_error"),
        n_trials=param_rec_report.get("n_trials"),
        n_observations=param_rec_report.get("n_observations"),
        project_root=root,
    )
    _t0 = time.time()
    paths["sensitivity_heatmap"] = generate_sensitivity_heatmap(
        project_root=root, n_trials=5 if smoke else 20
    )
    _timings["sensitivity"] = time.time() - _t0
    report_schemas.check_figure_contract("cross_study_summary", "cross_study_summary", cross_report)
    paths["cross_study_summary"] = generate_cross_study_summary(cross_report, project_root=root)
    _t0 = time.time()
    report_schemas.check_figure_contract("hierarchical_pomdp", "hierarchical_world", hier_report)
    report_schemas.check_figure_contract("hierarchical_pomdp", "nlevel3_world", nl3_report)
    paths["hierarchical_pomdp"] = generate_hierarchical_pomdp(hier_report, nl3_report, project_root=root)
    _timings["hierarchical"] += time.time() - _t0
    report_schemas.check_figure_contract("disjoint_fov_world", "disjoint_fov_world", disjoint_report)
    paths["disjoint_fov_world"] = Path(generate_disjoint_fov_figure(disjoint_report, project_root=root))
    paths["figure_registry"] = _write_figure_registry(root, paths)

    # --- Write stage timings -----------------------------------------------
    _timings["total"] = time.time() - _pipeline_start
    _timings_path = root / "output" / "data" / "stage_timings.json"
    _write_json(_timings, _timings_path)
    # This producer-owned sidecar records the exact budget selection and, for a
    # publication run, the pre-run input snapshot that wrote the report tree.
    # A publication receipt must consume this artifact rather than infer the
    # profile from a CLI spelling or a mutable configuration after the run.
    paths["analysis_execution"] = _write_json(
        {
            "schema_version": 2,
            "configured_profile": configured_profile,
            "effective_profile": effective_profile,
            "producer": "analysis.workflow.run_analysis_pipeline",
            **(
                {"analysis_input_snapshot": analysis_input_snapshot}
                if analysis_input_snapshot is not None
                else {}
            ),
        },
        root / "output" / "data" / "analysis_execution.json",
    )

    return paths


def main(project_root: Path | None = None) -> dict[str, Path]:
    """Run the analysis pipeline and print every artifact path to stdout."""
    paths = run_analysis_pipeline(project_root=project_root)
    for label, path in paths.items():
        print(f"{label}: {path}")
    return paths


__all__ = ["main", "resolve_analysis_profile", "run_analysis_pipeline"]
