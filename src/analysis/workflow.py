"""Thin orchestrator for the Active Fedference analysis pipeline.

Wires the locked :mod:`fedference.experiments` categorical source-mechanism
analogues of Friston et al. (2024), *Federated inference and belief sharing* (Neurosci. Biobehav. Rev.
156:105500) to disk: it runs the experiments across the configured seeds, writes
JSON result reports to ``output/reports/`` and the registered figures to
``output/figures/``, and returns/prints the artifact paths. **All numerics live
in the imported core modules** (``fedference.*``) and figure modules
(:mod:`figures`); this file only aggregates, serialises and plots — the thin
orchestrator contract. No ``infrastructure.*`` imports (layer contract).

Three-robustness-axes honesty: the exploratory point-estimate logistic-regression
figure compares RCCE and NLL gradients under two explicitly reported L2
coefficients. Its legacy ``AR`` argument is only a compatibility selector for
the stronger coefficient; this proxy does not compute an Alpha-Renyi objective
or establish the source-conditional client-side theorem. The
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
from typing import Any, Callable, cast

from experiment_config import ExperimentConfig, load_experiment_config, load_manuscript_config
from fedference.experiments import (
    disjoint_fov_report,
    hierarchical_world_report,
    nlevel3_world_report,
    run_complexity_scaling,
    run_heuristic_characterization,
    run_hierarchical_bmr,
    summarize_cross_study,
)
from figures import (
    generate_aggregation_descent,
    generate_application_integrity_flow,
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
    generate_evidence_replication_map,
    generate_free_energy_comparison,
    generate_frontmatter_execution,
    generate_frontmatter_model,
    generate_frontmatter_robustness,
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
    generate_source_render_provenance,
    generate_system_overview,
)
from figures._metadata import figure_metadata
from project_paths import resolve_env_project_root
from publication.pipeline_freshness import capture_analysis_input_snapshot

from . import report_schemas
from .figure_exact_values import (
    build_figure_exact_values,
    render_figure_exact_values_markdown,
)
from .producers.belief import (
    _belief_sharing_report as _belief_sharing_report,
)
from .producers.belief import (
    _efe_terms as _efe_terms,
)
from .producers.belief import (
    _emergence_report as _emergence_report,
)
from .producers.belief import (
    _heatmap_inputs as _heatmap_inputs,
)
from .producers.belief import (
    _language_report as _language_report,
)
from .producers.bnn import (
    BnnTorchOptions as BnnTorchOptions,
)
from .producers.bnn import (
    _bnn_report as _bnn_report,
)
from .producers.bnn import (
    _bnn_torch_report as _bnn_torch_report,
)
from .producers.diagnostics import (
    _belief_quality_report as _belief_quality_report,
)
from .producers.diagnostics import (
    _sensitivity_report as _sensitivity_report,
)
from .producers.robustness import (
    _contamination_gallery_report as _contamination_gallery_report,
)
from .producers.robustness import (
    _influence_weights as _influence_weights,
)
from .producers.robustness import (
    _review_grid_report as _review_grid_report,
)
from .producers.robustness import (
    _robustness_onset_report as _robustness_onset_report,
)
from .producers.robustness import (
    _robustness_report as _robustness_report,
)
from .producers.robustness import (
    _variational_aggregation_report as _variational_aggregation_report,
)
from .producers.worlds import (
    _conditional_world_report as _conditional_world_report,
)
from .producers.worlds import (
    _moving_world_report as _moving_world_report,
)
from .producers.worlds import (
    _parameter_recovery_report as _parameter_recovery_report,
)
from .visual_contracts import (
    application_integrity_flow_contract,
    evidence_replication_map_contract,
    source_render_provenance_contract,
)

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


class DuplicateArtifactError(ValueError):
    """A report producer attempted to reuse an accepted artifact key or path."""

    def __init__(self, artifact_key: str, path: Path) -> None:
        self.artifact_key = artifact_key
        self.path = path
        super().__init__(f"duplicate producer artifact {artifact_key!r}: {path}")


def _write_json(
    payload: Mapping[str, object],
    path: Path,
    *,
    schema: str | None = None,
    artifact: tuple[str, Mapping[str, Path]] | None = None,
) -> Path:
    # Claim within this pipeline before validation or any accepted replacement.
    # The caller records the returned path only after a successful write.
    if artifact is not None:
        key, produced = artifact
        if key in produced or any(path.resolve() == prior.resolve() for prior in produced.values()):
            raise DuplicateArtifactError(key, path)
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


def _write_text(text: str, path: Path) -> Path:
    """Atomically persist a deterministic UTF-8 text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
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
    from figures._frontmatter import validate_frontmatter

    validate_frontmatter(project_root, required=False)
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
            from analysis.artifacts import PRESENTATION_FIGURE_STEMS

            selectors = re.findall(r'\bdata-slide-manifest="([^"]+)"', match.group("attrs"))
            presentation: dict[str, str] = {}
            if figure_path.stem in PRESENTATION_FIGURE_STEMS:
                manifest = f"output/figures/{figure_path.stem}.slides.json"
                if selectors != [f"../{manifest}"]:
                    raise ValueError(
                        f"figure {figure_path.stem} requires its exact source-owned presentation selector"
                    )
                presentation["presentation_manifest"] = manifest
            elif selectors:
                raise ValueError(f"figure {figure_path.stem} has an undeclared presentation selector")
            figures.append(
                {
                    "label": match.group("label"),
                    "filename": figure_path.name,
                    "path": str(Path("output") / "figures" / figure_path.name),
                    "source_manuscript": str(source_path.relative_to(project_root)),
                    "caption": _normalise_caption(match.group("caption")),
                    **presentation,
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
    from figures._metadata import FIGURE_SUPPORT_MODULES

    for mod in sorted(figures_pkg.glob("*.py")):
        if mod.stem in FIGURE_SUPPORT_MODULES:
            continue
        stem = mod.stem
        # Generators write both a .png (embedded) and often a sibling .pdf.
        for ext in (".png", ".pdf"):
            mapping[f"{stem}{ext}"] = stem
    return mapping


def _require_unique_declared_figures(figures: list[dict[str, str]]) -> None:
    """Reject ambiguous label or artifact ownership before reading outputs."""
    seen_labels: set[str] = set()
    seen_filenames: set[str] = set()
    for figure in figures:
        label = figure["label"]
        filename = figure["filename"]
        if label in seen_labels:
            raise ValueError(f"duplicate manuscript figure label {label!r}")
        if filename in seen_filenames:
            raise ValueError(f"duplicate manuscript figure filename {filename!r}")
        seen_labels.add(label)
        seen_filenames.add(filename)


def _write_figure_registry(project_root: Path, artifact_paths: dict[str, Path]) -> Path:
    # Pipeline-produced figures (keyed by the .png they write, since the
    # manuscript embeds the PNG even when the generator returns the .pdf path).
    generated_by_filename = {
        path.name: artifact_key
        for artifact_key, path in artifact_paths.items()
        if path.parent.name == "figures"
    }
    # Fall back to the :mod:`figures` submodule that owns the filename so any
    # explicitly supported standalone invocation remains attributable rather
    # than being labelled ``preexisting_figure``.
    submodule_by_filename = _figure_generators_by_filename(project_root)

    def _resolve_generated_by(filename: str) -> str:
        if filename in generated_by_filename:
            return generated_by_filename[filename]
        if filename in submodule_by_filename:
            return submodule_by_filename[filename]
        return "preexisting_figure"

    declared_figures = _declared_manuscript_figures(project_root)
    _require_unique_declared_figures(declared_figures)

    exact_values_path = artifact_paths.get("figure_exact_values")
    exact_values_markdown = artifact_paths.get("figure_exact_values_markdown")
    if exact_values_path is None or exact_values_markdown is None:
        raise ValueError("figure registry requires both exact-value fallback artifacts")
    exact_payload = json.loads(exact_values_path.read_text(encoding="utf-8"))
    report_schemas.validate_report("figure_exact_values", exact_payload)
    exact_identifiers = {
        str(table["identifier"])
        for table in exact_payload["tables"]
        if isinstance(table, Mapping) and isinstance(table.get("identifier"), str)
    }

    figures: list[dict[str, str]] = []
    for figure in declared_figures:
        label = figure["label"]
        filename = figure["filename"]
        figure_path = project_root / "output" / "figures" / filename
        if not figure_path.exists():
            raise FileNotFoundError(f"Declared manuscript figure is missing: {figure_path}")
        generator = _resolve_generated_by(filename)
        if generator == "preexisting_figure":
            raise ValueError(f"no generator metadata owner for embedded figure {filename}")
        metadata = figure_metadata(generator)
        if "presentation_manifest" in figure:
            from figures._presentation import validate_presentation_files

            validate_presentation_files(project_root / figure["presentation_manifest"])
        fallback = metadata.get("exact_value_fallback")
        if fallback and fallback not in exact_identifiers:
            raise ValueError(f"figure {label} references unknown exact-value fallback {fallback!r}")
        figures.append({**figure, "generated_by": generator, **metadata})
    used_identifiers = {
        str(figure["exact_value_fallback"]) for figure in figures if figure.get("exact_value_fallback")
    }
    return _write_json(
        {
            "schema_version": "1.2",
            "generated_by": "analysis.workflow.run_analysis_pipeline",
            "exact_value_artifact": {
                "json_path": str(exact_values_path.relative_to(project_root)),
                "markdown_path": str(exact_values_markdown.relative_to(project_root)),
                "identifiers": sorted(used_identifiers),
            },
            "figures": sorted(figures, key=lambda item: item["label"]),
        },
        project_root / "output" / "figures" / "figure_registry.json",
        schema="figure_registry",
        artifact=("figure_registry", artifact_paths),
    )


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


def _configured_bnn_torch_report(config: ExperimentConfig, *, project_root: Path | None) -> dict:
    """Admit the optional backend before reading its source-owned configuration."""
    try:
        from fedference.bnn_baseline_torch import run_bnn_torch_experiment
    except ImportError as exc:  # PyTorch optional-extra not installed
        return {"status": f"skipped: PyTorch unavailable ({exc})"}
    options = _bnn_torch_options(project_root)
    return _bnn_torch_report(config, options=options, run_experiment=run_bnn_torch_experiment)


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


def effective_analysis_config(
    root: Path,
    profile: str,
    config: ExperimentConfig | None = None,
) -> ExperimentConfig:
    """Resolve unchanged scientific budgets for direct and staged producers."""
    effective_profile = resolve_analysis_profile(root, override=profile)
    canonical_config = load_experiment_config(root)
    if config is not None and effective_profile == "publication" and config != canonical_config:
        raise ValueError(
            "publication analysis requires the canonical manuscript/config.yaml ExperimentConfig; "
            "an explicit non-equivalent config cannot mint publication evidence"
        )
    cfg = config or canonical_config
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
    return cfg


def run_analysis_pipeline(
    config: ExperimentConfig | None = None,
    *,
    project_root: Path | None = None,
    profile: str | None = None,
    checkpoints: Path | None = None,
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
    cfg = effective_analysis_config(root, effective_profile, config)
    if effective_profile == "publication" and checkpoints is None:
        raise ValueError(
            "publication compute requires external report checkpoints; monolithic execution is refused"
        )
    analysis_input_snapshot = (
        capture_analysis_input_snapshot(root) if effective_profile == "publication" else None
    )
    smoke = effective_profile == "smoke"
    loaded: dict[str, dict[str, Any]] = {}
    if checkpoints is not None:
        from analysis.stage_catalog import STAGE_SCHEMAS
        from analysis.stage_checkpoint import load_checkpoint
        from analysis.stage_io import StageError
        from analysis.stage_workspace import admit_workspace

        if root != (checkpoints / "source").resolve():
            raise StageError("checkpoint assembly must run in the workspace's frozen source")

        manifest = admit_workspace(checkpoints, root)
        if manifest["profile"] != effective_profile or config is not None:
            raise StageError("checkpoint profile/configuration mismatch")
        # Admit the complete report set before writing or rendering anything.
        loaded = {
            name: load_checkpoint(checkpoints, name, schema, manifest["identity"])
            for name, schema in STAGE_SCHEMAS.items()
        }

    def _completed(name: str, producer: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
        return loaded[name] if checkpoints is not None else dict(producer())

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
    application_flow_report = _completed(
        "application_integrity_flow", lambda: application_integrity_flow_contract()
    )
    paths["application_integrity_flow_report"] = _write_json(
        application_flow_report,
        reports / "application_integrity_flow.json",
        schema="application_integrity_flow",
        artifact=("application_integrity_flow_report", paths),
    )
    evidence_map_report = _completed("evidence_replication_map", lambda: evidence_replication_map_contract())
    paths["evidence_replication_map_report"] = _write_json(
        evidence_map_report,
        reports / "evidence_replication_map.json",
        schema="evidence_replication_map",
        artifact=("evidence_replication_map_report", paths),
    )
    source_render_report = _completed("source_render_provenance", lambda: source_render_provenance_contract())
    paths["source_render_provenance_report"] = _write_json(
        source_render_report,
        reports / "source_render_provenance.json",
        schema="source_render_provenance",
        artifact=("source_render_provenance_report", paths),
    )

    bs_report = _completed("belief_sharing", lambda: _belief_sharing_report(cfg))
    paths["belief_sharing_report"] = _write_json(
        bs_report,
        reports / "belief_sharing.json",
        schema="belief_sharing",
        artifact=("belief_sharing_report", paths),
    )

    lang_report = _completed("language_acquisition", lambda: _language_report(cfg))
    paths["language_report"] = _write_json(
        lang_report,
        reports / "language_acquisition.json",
        schema="language_acquisition",
        artifact=("language_report", paths),
    )

    emergence_report = _completed("emergence", lambda: _emergence_report(cfg))
    paths["emergence_report"] = _write_json(
        emergence_report, reports / "emergence.json", schema="emergence", artifact=("emergence_report", paths)
    )

    rob_report = _completed("robustness_sweep", lambda: _robustness_report(cfg))
    paths["robustness_report"] = _write_json(
        rob_report,
        reports / "robustness_sweep.json",
        schema="robustness_sweep",
        artifact=("robustness_report", paths),
    )

    hier_bmr_report = _completed("hierarchical_bmr", lambda: run_hierarchical_bmr())
    paths["hierarchical_bmr_report"] = _write_json(
        hier_bmr_report,
        reports / "hierarchical_bmr.json",
        schema="hierarchical_bmr",
        artifact=("hierarchical_bmr_report", paths),
    )

    heuristic_report = _completed(
        "heuristic_characterization", lambda: run_heuristic_characterization(cfg.seeds[0])
    )
    paths["heuristic_characterization_report"] = _write_json(
        heuristic_report,
        reports / "heuristic_characterization.json",
        schema="heuristic_characterization",
        artifact=("heuristic_characterization_report", paths),
    )

    efe_report = _completed("efe_decomposition", lambda: _efe_terms(cfg))
    paths["efe_report"] = _write_json(
        efe_report,
        reports / "efe_decomposition.json",
        schema="efe_decomposition",
        artifact=("efe_report", paths),
    )

    weights_report = _completed("robust_influence_weights", lambda: _influence_weights(cfg))
    paths["influence_weights_report"] = _write_json(
        weights_report,
        reports / "robust_influence_weights.json",
        schema="robust_influence_weights",
        artifact=("influence_weights_report", paths),
    )

    bnn_report = _completed("bnn_robustness", lambda: _bnn_report(cfg, smoke=smoke))
    paths["bnn_report"] = _write_json(
        bnn_report, reports / "bnn_robustness.json", schema="bnn_robustness", artifact=("bnn_report", paths)
    )

    bnn_torch_report = _completed("bnn_torch", lambda: _configured_bnn_torch_report(cfg, project_root=root))
    paths["bnn_torch_report"] = _write_json(
        bnn_torch_report, reports / "bnn_torch.json", schema="bnn_torch", artifact=("bnn_torch_report", paths)
    )

    var_report = _completed("variational_aggregation", lambda: _variational_aggregation_report(cfg))
    paths["variational_aggregation_report"] = _write_json(
        var_report,
        reports / "variational_aggregation.json",
        schema="variational_aggregation",
        artifact=("variational_aggregation_report", paths),
    )

    gallery_report = _completed(
        "contamination_gallery", lambda: _contamination_gallery_report(cfg, smoke=smoke)
    )
    paths["contamination_gallery_report"] = _write_json(
        gallery_report,
        reports / "contamination_gallery.json",
        schema="contamination_gallery",
        artifact=("contamination_gallery_report", paths),
    )

    onset_report = _completed("robustness_onset", lambda: _robustness_onset_report(cfg, smoke=smoke))
    paths["robustness_onset_report"] = _write_json(
        onset_report,
        reports / "robustness_onset.json",
        schema="robustness_onset",
        artifact=("robustness_onset_report", paths),
    )

    conditional_report = _completed("conditional_world", lambda: _conditional_world_report(cfg))
    paths["conditional_world_report"] = _write_json(
        conditional_report,
        reports / "conditional_world.json",
        schema="conditional_world",
        artifact=("conditional_world_report", paths),
    )

    review_grid_report = _completed("robustness_review_grid", lambda: _review_grid_report(cfg, smoke=smoke))
    paths["review_grid_report"] = _write_json(
        review_grid_report,
        reports / "robustness_review_grid.json",
        schema="robustness_review_grid",
        artifact=("review_grid_report", paths),
    )

    quality_report = _completed("belief_quality", lambda: _belief_quality_report(cfg))
    paths["belief_quality_report"] = _write_json(
        quality_report,
        reports / "belief_quality.json",
        schema="belief_quality",
        artifact=("belief_quality_report", paths),
    )

    _t0 = time.time()
    complexity_config = cfg.complexity.for_smoke() if smoke else cfg.complexity
    complexity_report = _completed("complexity_scaling", lambda: run_complexity_scaling(complexity_config))
    _timings["complexity"] = time.time() - _t0
    paths["complexity_scaling_report"] = _write_json(
        complexity_report,
        reports / "complexity_scaling.json",
        schema="complexity_scaling",
        artifact=("complexity_scaling_report", paths),
    )

    _t0 = time.time()
    sensitivity_report = _completed(
        "sensitivity", lambda: _sensitivity_report(seed=cfg.seeds[0], n_trials=5 if smoke else 20)
    )
    _timings["sensitivity"] = time.time() - _t0
    paths["sensitivity_report"] = _write_json(
        sensitivity_report,
        reports / "sensitivity.json",
        schema="sensitivity",
        artifact=("sensitivity_report", paths),
    )

    _t0 = time.time()
    moving_report = _completed("moving_world", lambda: _moving_world_report(cfg))
    _timings["moving_world"] = time.time() - _t0
    paths["moving_world_report"] = _write_json(
        moving_report,
        reports / "moving_world.json",
        schema="moving_world",
        artifact=("moving_world_report", paths),
    )

    _t0 = time.time()
    param_rec_report = _completed("parameter_recovery", lambda: _parameter_recovery_report(cfg, smoke=smoke))
    _timings["parameter_recovery"] = time.time() - _t0
    paths["parameter_recovery_report"] = _write_json(
        param_rec_report,
        reports / "parameter_recovery.json",
        schema="parameter_recovery",
        artifact=("parameter_recovery_report", paths),
    )

    _t0 = time.time()
    hier_report = _completed(
        "hierarchical_world",
        lambda: hierarchical_world_report(
            cfg.seeds[0], n_trials=cfg.n_trials, n_multiseed=cfg.replicate_seeds
        ),
    )
    nl3_report = _completed(
        "nlevel3_world",
        lambda: nlevel3_world_report(cfg.seeds[0], n_trials=cfg.n_trials, n_multiseed=cfg.replicate_seeds),
    )
    cross_report = _completed(
        "cross_study_summary",
        lambda: summarize_cross_study(
            cfg.seeds[0],
            n_seeds=cfg.replicate_seeds,
            n_trials=cfg.cross_study_n_trials,
        ),
    )
    disjoint_report = _completed(
        "disjoint_fov_world", lambda: disjoint_fov_report(0, n_multiseed=cfg.replicate_seeds)
    )
    _timings["hierarchical"] = time.time() - _t0
    paths["hierarchical_world_report"] = _write_json(
        hier_report,
        reports / "hierarchical_world.json",
        schema="hierarchical_world",
        artifact=("hierarchical_world_report", paths),
    )
    paths["nlevel3_world_report"] = _write_json(
        nl3_report,
        reports / "nlevel3_world.json",
        schema="nlevel3_world",
        artifact=("nlevel3_world_report", paths),
    )
    paths["cross_study_report"] = _write_json(
        cross_report,
        reports / "cross_study_summary.json",
        schema="cross_study_summary",
        artifact=("cross_study_report", paths),
    )
    paths["disjoint_fov_report"] = _write_json(
        disjoint_report,
        reports / "disjoint_fov_world.json",
        schema="disjoint_fov_world",
        artifact=("disjoint_fov_report", paths),
    )

    exact_values = build_figure_exact_values(
        {
            "belief_quality": quality_report,
            "bnn_robustness": bnn_report,
            "complexity_scaling": complexity_report,
            "conditional_world": conditional_report,
            "cross_study_summary": cross_report,
            "robustness_review_grid": review_grid_report,
            "sensitivity": sensitivity_report,
        }
    )
    paths["figure_exact_values"] = _write_json(
        exact_values,
        root / "output" / "figures" / "figure_exact_values.json",
        schema="figure_exact_values",
        artifact=("figure_exact_values", paths),
    )
    paths["figure_exact_values_markdown"] = _write_text(
        render_figure_exact_values_markdown(exact_values),
        root / "output" / "figures" / "figure_exact_values.md",
    )

    # --- Figures (consume the report dicts above) --------------------------
    report_schemas.check_figure_contract(
        "application_integrity_flow", "application_integrity_flow", application_flow_report
    )
    paths["application_integrity_flow"] = generate_application_integrity_flow(
        application_flow_report,
        project_root=root,
    )
    report_schemas.check_figure_contract(
        "evidence_replication_map", "evidence_replication_map", evidence_map_report
    )
    paths["evidence_replication_map"] = generate_evidence_replication_map(
        evidence_map_report,
        project_root=root,
    )
    report_schemas.check_figure_contract(
        "source_render_provenance", "source_render_provenance", source_render_report
    )
    paths["source_render_provenance"] = generate_source_render_provenance(
        source_render_report,
        project_root=root,
    )
    paths["graphical_abstract"] = generate_graphical_abstract(project_root=root)
    paths["frontmatter_execution"] = generate_frontmatter_execution(project_root=root)
    paths["frontmatter_model"] = generate_frontmatter_model(project_root=root)
    paths["frontmatter_robustness"] = generate_frontmatter_robustness(project_root=root)
    paths["system_overview"] = generate_system_overview(project_root=root)
    paths["generative_model_schema"] = generate_generative_model_schema(project_root=root)
    paths["message_passing"] = generate_message_passing(project_root=root)
    paths["pomdp_loop"] = generate_pomdp_loop(project_root=root)

    local_posteriors, consensus = _heatmap_inputs(cfg)
    paths["belief_heatmap"] = generate_belief_heatmap(local_posteriors, consensus, project_root=root)
    report_schemas.check_figure_contract("free_energy_comparison", "belief_sharing", bs_report)
    paths["free_energy_comparison"] = generate_free_energy_comparison(
        bs_report["incommunicado_free_energy"],
        bs_report["communicating_free_energy"],
        paired_difference_mean=bs_report["paired_difference_mean"],
        paired_difference_ci=bs_report["paired_difference_ci"],
        difference_definition=bs_report["difference_definition"],
        analysis_unit=bs_report["analysis_unit"],
        replication_unit=bs_report["replication_unit"],
        project_root=root,
    )
    report_schemas.check_figure_contract("robustness_sweep", "robustness_sweep", rob_report)
    paths["robustness_sweep"] = generate_robustness_sweep(
        rob_report["accuracy_by_method_and_rate"],
        cfg.contamination_rates,
        accuracy_threshold=rob_report.get("accuracy_threshold"),
        rate_summary=rob_report.get("per_rate_summary"),
        server_robustness_by_label=rob_report.get("server_robustness_by_label"),
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
        surprise_tol=hier_bmr_report["surprise_tol"],
        decision_rule=hier_bmr_report["decision_rule"],
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
        selection_disclosure=bnn_report["selection_disclosure"],
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
    report_schemas.check_figure_contract("sensitivity_heatmap", "sensitivity", sensitivity_report)
    paths["sensitivity_heatmap"] = generate_sensitivity_heatmap(
        project_root=root,
        report=sensitivity_report,
    )
    report_schemas.check_figure_contract("cross_study_summary", "cross_study_summary", cross_report)
    paths["cross_study_summary"] = generate_cross_study_summary(cross_report, project_root=root)
    _t0 = time.time()
    report_schemas.check_figure_contract("hierarchical_pomdp", "hierarchical_world", hier_report)
    report_schemas.check_figure_contract("hierarchical_pomdp", "nlevel3_world", nl3_report)
    # The caption's HIER_SEED belongs to the executed report. Bind the
    # posterior/consensus diagnostic to that seed instead of the demo default.
    paths["hierarchical_pomdp"] = generate_hierarchical_pomdp(
        hier_report, nl3_report, project_root=root, seed=int(hier_report["seed"])
    )
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
