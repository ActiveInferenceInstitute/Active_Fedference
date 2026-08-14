"""Frozen experiment configuration for the Active Fedference project.

Single loader for ``manuscript/config.yaml`` -> ``experiment:`` block. The
parameters here drive the categorical source-mechanism analogues of Friston et al. (2024),
*Federated inference and belief sharing* (Neurosci. Biobehav. Rev. 156:105500),
run by :mod:`fedference.experiments`: the colony size and noisy-sensor grid of
Fig. 1/4, the contamination rates and FedGVI client divergences of the
robustness sweep (Fig. 5 / FedGVI client weighting, Mildner et al. 2025), and
the seed budget consumed by the paired/BH-FDR robustness report.

Pure ``numpy`` / ``pyyaml``; no ``infrastructure.*`` imports (layer contract).
Used by analysis, figures, and manuscript variable generation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fedference.complexity import ComplexityBenchmarkConfig

#: Default FedGVI client divergences, project log-linear-pool baseline ``KLD``
#: first (Fig. 5 / FedGVI client weighting). Under the separately documented
#: categorical posterior-log-potential assumptions, that baseline specializes
#: Friston Eq. 7's message-combination term; it is not the complete protocol.
_DEFAULT_DIVERGENCES: tuple[str, ...] = ("KLD", "RKL", "AR", "beta", "rcce")
#: Default contamination-rate sweep, kept below the ``rate = 1`` pure-veto cliff
#: where every pooling rule collapses to zero accuracy.
_DEFAULT_CONTAMINATION_RATES: tuple[float, ...] = (0.0, 0.225, 0.45, 0.675, 0.9)
_DEFAULT_REVIEW_GRID_RATES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# Keep the dataclass fallback and the shipped publication configuration on the
# same source-bound contract.  A temporary or minimal project may omit
# ``manuscript/config.yaml``; falling back to an older diagnostic budget in
# that case would silently produce a different evidence design.
_DEFAULT_N_SEEDS = 480
_DEFAULT_REPLICATE_SEEDS = 128
_DEFAULT_N_TRIALS = 960
_DEFAULT_CROSS_STUDY_N_TRIALS = 40
_DEFAULT_CONDITIONAL_WORLD_N_SEEDS = 64
_DEFAULT_CONDITIONAL_WORLD_N_TRIALS = 24
_DEFAULT_REVIEW_GRID_N_SEEDS = 160
_DEFAULT_REVIEW_GRID_N_TRIALS = 24
_DEFAULT_REVIEW_GRID_TARGET_MAX_MCSE = 0.01
_DEFAULT_GALLERY_N_SEEDS = 64
_DEFAULT_GALLERY_N_TRIALS = 24
_DEFAULT_ONSET_N_SEEDS = 64
_DEFAULT_ONSET_N_TRIALS = 24
_DEFAULT_BNN_N_SEEDS = 64
_DEFAULT_BNN_N_PER = 200
#: Default sensitivity-sweep grids — shared by experiments, figures, and tokens.
DEFAULT_SENSITIVITY_ACUITY: tuple[float, ...] = (0.40, 0.55, 0.70, 0.85, 0.95)
DEFAULT_SENSITIVITY_COLONY_SIZES: tuple[int, ...] = (2, 4, 6, 8, 10)
#: Display/annotation threshold for cells whose accuracy gap is treated as
#: practically negligible in the sensitivity heatmap.  This is a visualization
#: convention, not a statistical test or a claim that the gap is zero.
SENSITIVITY_NOISE_FLOOR: float = 0.05


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen Active Fedference parameters from ``config.yaml`` -> ``experiment:``.

    Attributes:
        n_agents: Colony size (sentinels broadcasting beliefs, Friston Fig. 1).
        n_locations: Hidden-state cardinality of the shared latent factor
            (creature location); the project uses a source-inspired 3x3 grid
            (9 locations), so the default matches :data:`fedference.pomdp.N_LOCATIONS`.
        contamination_rates: Convex-mix rates toward a confident-wrong delta
            swept in the robustness experiment.
        divergences: FedGVI client divergence labels; ``"KLD"`` is the naive
            Friston pool, the rest are robust pools.
        n_seeds: Number of independent seeds for the across-seed studies
            (belief sharing). Raised to a defensible default so the across-seed
            colony means and their bootstrap CIs carry real sample size.
        replicate_seeds: Independent seed budget for the multi-study overview
            and structural extension reports. This is separate from ``n_seeds``
            so the expensive cross-study suite is explicit rather than silently
            truncated to a handful of seeds.
        n_trials: Paired-trial budget of the robustness verdict (the Wilcoxon/
            BH-FDR panel and per-condition bootstrap CIs). Distinct from
            ``n_seeds`` so the verdict can be powered independently of the
            across-seed study budget; defaults above the seed budget.
        robustness: Robust-aggregation strength used for the EFE / aggregation
            invariant checks; ``0.0`` recovers the naive pool exactly.
        fdr_alpha: Benjamini–Hochberg family-wise FDR level for the robustness
            verdict panel (:func:`fedference.statistics.bh_fdr`); consumed by
            the executed sweep so the manuscript's reported level is the level
            the test actually ran at.
        power_alpha: Significance level for the headline robust-vs-naive power
            analysis (:func:`fedference.statistics.power_analysis`).
        power_alternative: Directional alternative for that power analysis
            (``"greater"`` = robust accuracy exceeds naive).
        target_power: Prospective-power target for the sample-size justification
            (:func:`fedference.statistics.sample_size_for_power`).
        cross_study_n_trials: Matched trials per rate in the cross-study summary.
        conditional_world_n_seeds: Seed budget for the conditional-world and
            proper-score extensions; kept separate from the larger core sweep.
        conditional_world_n_trials: Nested trials per conditional-world cell.
        gallery_n_seeds: Independent seed budget for the multi-mechanism
            contamination gallery.
        gallery_n_trials: Nested trials per gallery seed/mechanism cell.
        onset_n_seeds: Independent seed budget for the directional
            contamination-rate profiles.
        onset_n_trials: Nested trials per onset seed/rate cell.
        bnn_n_seeds: Independent synthetic-data seeds for the NumPy BNN
            complement.
        bnn_n_per: Examples per class per client in that BNN complement.
        complexity: Seeded grid and timing budget for the implementation-derived
            complexity report and scaling figure.
    """

    n_agents: int = 7
    n_locations: int = 9
    contamination_rates: tuple[float, ...] = _DEFAULT_CONTAMINATION_RATES
    divergences: tuple[str, ...] = _DEFAULT_DIVERGENCES
    n_seeds: int = _DEFAULT_N_SEEDS
    replicate_seeds: int = _DEFAULT_REPLICATE_SEEDS
    n_trials: int = _DEFAULT_N_TRIALS
    cross_study_n_trials: int = _DEFAULT_CROSS_STUDY_N_TRIALS
    conditional_world_n_seeds: int = _DEFAULT_CONDITIONAL_WORLD_N_SEEDS
    conditional_world_n_trials: int = _DEFAULT_CONDITIONAL_WORLD_N_TRIALS
    review_grid_n_seeds: int = _DEFAULT_REVIEW_GRID_N_SEEDS
    review_grid_n_trials: int = _DEFAULT_REVIEW_GRID_N_TRIALS
    review_grid_rates: tuple[float, ...] = _DEFAULT_REVIEW_GRID_RATES
    review_grid_target_max_mcse: float = _DEFAULT_REVIEW_GRID_TARGET_MAX_MCSE
    gallery_n_seeds: int = _DEFAULT_GALLERY_N_SEEDS
    gallery_n_trials: int = _DEFAULT_GALLERY_N_TRIALS
    onset_n_seeds: int = _DEFAULT_ONSET_N_SEEDS
    onset_n_trials: int = _DEFAULT_ONSET_N_TRIALS
    bnn_n_seeds: int = _DEFAULT_BNN_N_SEEDS
    bnn_n_per: int = _DEFAULT_BNN_N_PER
    robustness: float = 0.0
    fdr_alpha: float = 0.05
    power_alpha: float = 0.05
    power_alternative: str = "greater"
    target_power: float = 0.80
    complexity: ComplexityBenchmarkConfig = field(default_factory=ComplexityBenchmarkConfig)

    def __post_init__(self) -> None:
        integer_fields = (
            "n_agents",
            "n_locations",
            "n_seeds",
            "replicate_seeds",
            "n_trials",
            "cross_study_n_trials",
            "conditional_world_n_seeds",
            "conditional_world_n_trials",
            "review_grid_n_seeds",
            "review_grid_n_trials",
            "gallery_n_seeds",
            "gallery_n_trials",
            "onset_n_seeds",
            "onset_n_trials",
            "bnn_n_seeds",
            "bnn_n_per",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")

        real_fields = (
            "review_grid_target_max_mcse",
            "robustness",
            "fdr_alpha",
            "power_alpha",
            "target_power",
        )
        for name in real_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number")

        if self.n_agents < 2:
            raise ValueError("n_agents must be >= 2 (belief sharing needs a colony)")
        if self.n_locations < 2:
            raise ValueError("n_locations must be >= 2")
        if not self.contamination_rates:
            raise ValueError("contamination_rates must be non-empty")
        if any(
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(float(rate))
            or not 0.0 <= rate <= 1.0
            for rate in self.contamination_rates
        ):
            raise ValueError("contamination_rates must lie in [0, 1]")
        if not self.divergences or any(
            not isinstance(divergence, str) or not divergence for divergence in self.divergences
        ):
            raise ValueError("divergences must be a non-empty tuple of non-empty strings")
        if len(set(self.divergences)) != len(self.divergences):
            raise ValueError("divergences must not contain duplicate labels")
        if self.divergences.count("KLD") != 1:
            raise ValueError("divergences must include the naive baseline 'KLD' exactly once")
        if self.n_seeds < 2:
            raise ValueError("n_seeds must be >= 2 for a paired test")
        if self.replicate_seeds < 2:
            raise ValueError("replicate_seeds must be >= 2 for multi-seed reports")
        if self.n_trials < 2:
            raise ValueError("n_trials must be >= 2 for a paired test")
        if self.cross_study_n_trials < 2:
            raise ValueError("cross_study_n_trials must be >= 2")
        if self.conditional_world_n_seeds < 2:
            raise ValueError("conditional_world_n_seeds must be >= 2")
        if self.conditional_world_n_trials < 1:
            raise ValueError("conditional_world_n_trials must be >= 1")
        if self.review_grid_n_seeds < 2:
            raise ValueError("review_grid_n_seeds must be >= 2")
        if self.review_grid_n_trials < 2:
            raise ValueError("review_grid_n_trials must be >= 2")
        if (
            not self.review_grid_rates
            or any(
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate))
                or not 0.0 <= rate <= 1.0
                for rate in self.review_grid_rates
            )
            or tuple(self.review_grid_rates) != tuple(sorted(self.review_grid_rates))
            or len(set(self.review_grid_rates)) != len(self.review_grid_rates)
        ):
            raise ValueError("review_grid_rates must be non-empty, unique, sorted, finite, and lie in [0, 1]")
        if (
            isinstance(self.review_grid_target_max_mcse, bool)
            or not isinstance(self.review_grid_target_max_mcse, (int, float))
            or not math.isfinite(self.review_grid_target_max_mcse)
            or self.review_grid_target_max_mcse <= 0.0
        ):
            raise ValueError("review_grid_target_max_mcse must be finite and > 0")
        if self.gallery_n_seeds < 2 or self.gallery_n_trials < 2:
            raise ValueError("gallery_n_seeds and gallery_n_trials must both be >= 2")
        if self.onset_n_seeds < 2 or self.onset_n_trials < 2:
            raise ValueError("onset_n_seeds and onset_n_trials must both be >= 2")
        if self.bnn_n_seeds < 2:
            raise ValueError("bnn_n_seeds must be >= 2")
        if self.bnn_n_per < 1:
            raise ValueError("bnn_n_per must be >= 1")
        if self.robustness < 0.0:
            raise ValueError("robustness must be non-negative")
        if not 0.0 < self.fdr_alpha < 1.0:
            raise ValueError("fdr_alpha must lie in (0, 1)")
        if not 0.0 < self.power_alpha < 1.0:
            raise ValueError("power_alpha must lie in (0, 1)")
        if self.power_alternative not in ("greater", "less", "two-sided"):
            raise ValueError("power_alternative must be 'greater', 'less' or 'two-sided'")
        if not 0.0 < self.target_power < 1.0:
            raise ValueError("target_power must lie in (0, 1)")

    @property
    def seeds(self) -> tuple[int, ...]:
        """Deterministic seed list ``0 .. n_seeds-1`` for reproducible runs."""
        return tuple(range(self.n_seeds))

    @property
    def robust_divergences(self) -> tuple[str, ...]:
        """The non-``KLD`` (robust) divergence labels."""
        return tuple(d for d in self.divergences if d != "KLD")


def _strict_int(value: Any, *, field: str) -> int:
    """Read an integer YAML scalar without truncating or string-coercing it."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"manuscript/config.yaml {field} must be an integer")
    return value


def _strict_float(value: Any, *, field: str) -> float:
    """Read a finite numeric YAML scalar without accepting text or booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"manuscript/config.yaml {field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"manuscript/config.yaml {field} must be finite")
    return number


def _coerce_float_tuple(values: Any, default: tuple[float, ...], *, field: str) -> tuple[float, ...]:
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        return tuple(_strict_float(value, field=f"{field}[]") for value in values)
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        return (_strict_float(values, field=field),)
    raise ValueError(f"manuscript/config.yaml {field} must be a number or list of numbers")


def _coerce_str_tuple(values: Any, default: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        if any(not isinstance(value, str) for value in values):
            raise ValueError(f"manuscript/config.yaml {field} must contain only strings")
        return tuple(values)
    if isinstance(values, str):
        return (values,)
    raise ValueError(f"manuscript/config.yaml {field} must be a string or list of strings")


def load_manuscript_config(project_root: Path | None = None) -> dict[str, Any]:
    """Load and validate the canonical manuscript YAML structure.

    Every workflow reader must use this boundary rather than parsing the YAML
    independently.  A malformed top-level or ``experiment`` block can
    otherwise be coerced into an empty mapping by ``yaml.safe_load(...) or
    {}``, causing a publication run to silently use defaults.  Known nested
    experiment blocks are validated here as well, including the optional Torch
    profile.

    Args:
        project_root: Directory containing ``manuscript/config.yaml``; defaults
            to the project root (two levels above this file).

    Returns:
        A mutable plain-dict copy suitable for read-only workflow access.
    """
    root = project_root or Path(__file__).resolve().parent.parent
    config_path = root / "manuscript" / "config.yaml"
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ValueError("manuscript/config.yaml top level must be a mapping")

    result = dict(data)
    raw_experiment = result.get("experiment", {})
    if raw_experiment is None:
        raw_experiment = {}
    if not isinstance(raw_experiment, Mapping):
        raise ValueError("manuscript/config.yaml experiment block must be a mapping")
    experiment = dict(raw_experiment)
    for name in (
        "belief_sharing",
        "robustness_sweep",
        "statistics",
        "complexity",
        "bnn_torch",
    ):
        value = experiment.get(name, {})
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError(f"manuscript/config.yaml experiment.{name} block must be a mapping")
        experiment[name] = dict(value)
    result["experiment"] = experiment
    return result


def load_experiment_config(project_root: Path | None = None) -> ExperimentConfig:
    """Load the canonical publication config, failing closed on malformed YAML.

    A missing or empty configuration uses the same publication-budget defaults
    as :class:`ExperimentConfig`; it must not silently downgrade to a legacy
    diagnostic budget.

    Args:
        project_root: Directory containing ``manuscript/config.yaml``; defaults
            to the project root (two levels above this file).

    Returns:
        A frozen :class:`ExperimentConfig`.
    """
    root = project_root or Path(__file__).resolve().parent.parent
    if not (root / "manuscript" / "config.yaml").exists():
        return ExperimentConfig()

    data = load_manuscript_config(root)
    exp: Mapping[str, Any] = data["experiment"]
    # The shipped config nests per-study sub-blocks (belief_sharing, emergence,
    # robustness_sweep, ...); flat keys at the ``experiment:`` top level take
    # precedence so the same loader serves both a flat and a nested config.
    raw_sharing = exp.get("belief_sharing", {})
    raw_sweep = exp.get("robustness_sweep", {})
    raw_stats = exp.get("statistics", {})
    raw_complexity = exp.get("complexity", {})
    raw_sharing = {} if raw_sharing is None else raw_sharing
    raw_sweep = {} if raw_sweep is None else raw_sweep
    raw_stats = {} if raw_stats is None else raw_stats
    raw_complexity = {} if raw_complexity is None else raw_complexity
    for name, value in (
        ("belief_sharing", raw_sharing),
        ("robustness_sweep", raw_sweep),
        ("statistics", raw_stats),
        ("complexity", raw_complexity),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"manuscript/config.yaml {name} block must be a mapping")
    sharing = raw_sharing
    sweep = raw_sweep
    stats = raw_stats
    complexity: Mapping[str, object] = raw_complexity

    n_agents = exp.get("n_agents", sweep.get("n_agents", sharing.get("n_agents", 7)))
    contamination = exp.get("contamination_rates", sweep.get("rates"))
    divergences = exp.get("divergences", sweep.get("divergences"))
    n_seeds = exp.get("n_seeds", sharing.get("n_seeds", _DEFAULT_N_SEEDS))
    n_trials = exp.get("n_trials", sweep.get("n_trials", _DEFAULT_N_TRIALS))

    return ExperimentConfig(
        n_agents=_strict_int(n_agents, field="experiment.n_agents"),
        n_locations=_strict_int(exp.get("n_locations", 9), field="experiment.n_locations"),
        contamination_rates=_coerce_float_tuple(
            contamination, _DEFAULT_CONTAMINATION_RATES, field="experiment.contamination_rates"
        ),
        divergences=_coerce_str_tuple(divergences, _DEFAULT_DIVERGENCES, field="experiment.divergences"),
        n_seeds=_strict_int(n_seeds, field="experiment.n_seeds"),
        replicate_seeds=_strict_int(
            exp.get("replicate_seeds", _DEFAULT_REPLICATE_SEEDS), field="experiment.replicate_seeds"
        ),
        n_trials=_strict_int(n_trials, field="experiment.n_trials"),
        cross_study_n_trials=_strict_int(
            exp.get("cross_study_n_trials", _DEFAULT_CROSS_STUDY_N_TRIALS),
            field="experiment.cross_study_n_trials",
        ),
        conditional_world_n_seeds=_strict_int(
            exp.get("conditional_world_n_seeds", _DEFAULT_CONDITIONAL_WORLD_N_SEEDS),
            field="experiment.conditional_world_n_seeds",
        ),
        conditional_world_n_trials=_strict_int(
            exp.get("conditional_world_n_trials", _DEFAULT_CONDITIONAL_WORLD_N_TRIALS),
            field="experiment.conditional_world_n_trials",
        ),
        review_grid_n_seeds=_strict_int(
            exp.get("review_grid_n_seeds", _DEFAULT_REVIEW_GRID_N_SEEDS),
            field="experiment.review_grid_n_seeds",
        ),
        review_grid_n_trials=_strict_int(
            exp.get("review_grid_n_trials", _DEFAULT_REVIEW_GRID_N_TRIALS),
            field="experiment.review_grid_n_trials",
        ),
        review_grid_rates=_coerce_float_tuple(
            exp.get("review_grid_rates"), _DEFAULT_REVIEW_GRID_RATES, field="experiment.review_grid_rates"
        ),
        review_grid_target_max_mcse=_strict_float(
            exp.get("review_grid_target_max_mcse", _DEFAULT_REVIEW_GRID_TARGET_MAX_MCSE),
            field="experiment.review_grid_target_max_mcse",
        ),
        gallery_n_seeds=_strict_int(
            exp.get("gallery_n_seeds", _DEFAULT_GALLERY_N_SEEDS), field="experiment.gallery_n_seeds"
        ),
        gallery_n_trials=_strict_int(
            exp.get("gallery_n_trials", _DEFAULT_GALLERY_N_TRIALS), field="experiment.gallery_n_trials"
        ),
        onset_n_seeds=_strict_int(
            exp.get("onset_n_seeds", _DEFAULT_ONSET_N_SEEDS), field="experiment.onset_n_seeds"
        ),
        onset_n_trials=_strict_int(
            exp.get("onset_n_trials", _DEFAULT_ONSET_N_TRIALS), field="experiment.onset_n_trials"
        ),
        bnn_n_seeds=_strict_int(exp.get("bnn_n_seeds", _DEFAULT_BNN_N_SEEDS), field="experiment.bnn_n_seeds"),
        bnn_n_per=_strict_int(exp.get("bnn_n_per", _DEFAULT_BNN_N_PER), field="experiment.bnn_n_per"),
        robustness=_strict_float(exp.get("robustness", 0.0), field="experiment.robustness"),
        fdr_alpha=_strict_float(
            stats.get("fdr_alpha", exp.get("fdr_alpha", 0.05)), field="experiment.statistics.fdr_alpha"
        ),
        power_alpha=_strict_float(
            stats.get("power_alpha", exp.get("power_alpha", 0.05)), field="experiment.statistics.power_alpha"
        ),
        power_alternative=_power_alternative(stats, exp),
        target_power=_strict_float(
            stats.get("target_power", exp.get("target_power", 0.80)),
            field="experiment.statistics.target_power",
        ),
        complexity=ComplexityBenchmarkConfig.from_mapping(complexity),
    )


def _power_alternative(stats: Mapping[str, Any], experiment: Mapping[str, Any]) -> str:
    """Read the declared power direction without stringifying malformed YAML."""
    value = stats.get("power_alternative", experiment.get("power_alternative", "greater"))
    if not isinstance(value, str):
        raise ValueError("manuscript/config.yaml experiment.statistics.power_alternative must be a string")
    return value


__all__ = ["ExperimentConfig", "load_experiment_config", "load_manuscript_config"]
