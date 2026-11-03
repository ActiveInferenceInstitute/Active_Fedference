"""Contamination report contracts."""

from __future__ import annotations

from typing import TypedDict

from typing_extensions import NotRequired

from analysis.schemas.primitives import SchemaDefinition


class ContaminationGalleryCell(TypedDict):
    """One seed-aggregated contamination-gallery mechanism cell."""

    best_robust_method: str
    diff_ci: list[float]
    directional: bool
    mean_diff: float
    naive_ci: list[float]
    naive_mean: float
    reliably_beats: bool
    robust_ci: list[float]
    robust_mean: float
    win_fraction: float


class ContaminationGalleryReport(TypedDict):
    """Top-level payload written to ``contamination_gallery.json``."""

    by_kind: dict[str, ContaminationGalleryCell]
    directional_kinds: list[str]
    entropy_kinds: list[str]
    entropy_naive_robust: bool
    kinds: list[str]
    n_agents: int
    n_contaminated: int
    n_seeds: int
    n_trials: int
    rate: float
    reliable_kinds: list[str]
    reliable_win_fraction: float
    seed: int


class RobustnessOnsetCell(TypedDict):
    """One pooled-method, seed-aggregated robustness-onset rate cell."""

    best_robust_method_by_rate: list[str]
    naive_ci: list[list[float]]
    naive_curve: list[float]
    onset_rate: float | None
    rates: list[float]
    robust_ci: list[list[float]]
    robust_curve: list[float]
    win_curve: list[float]
    by_rate: NotRequired[dict[str, object]]


class RobustnessOnsetReport(TypedDict):
    """Top-level payload written to ``robustness_onset.json``."""

    by_kind: dict[str, RobustnessOnsetCell]
    kinds: list[str]
    n_agents: int
    n_contaminated: int
    n_seeds: int
    n_trials: int
    onset_win_fraction: float
    rates: list[float]
    seed: int


_CONTAMINATION_GALLERY_SCHEMA = SchemaDefinition(
    required={
        "by_kind": "dict",
        "directional_kinds": "list",
        "entropy_kinds": "list",
        "entropy_naive_robust": "bool",
        "kinds": "list",
        "n_agents": "int",
        "n_contaminated": "int",
        "n_seeds": "int",
        "n_trials": "int",
        "rate": "number",
        "reliable_kinds": "list",
        "reliable_win_fraction": "number",
        "seed": "int",
    }
)


_ROBUSTNESS_ONSET_SCHEMA = SchemaDefinition(
    required={
        "by_kind": "dict",
        "kinds": "list",
        "n_agents": "int",
        "n_contaminated": "int",
        "n_seeds": "int",
        "n_trials": "int",
        "onset_win_fraction": "number",
        "rates": "list",
        "seed": "int",
    }
)
