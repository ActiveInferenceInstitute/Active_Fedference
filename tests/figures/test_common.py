"""Tests for the shared figure-styling helpers in ``figures._common``.

No mocks: ``apply_style`` mutates the real matplotlib rcParams (headless Agg)
and ``figures_dir`` creates a real directory under ``tmp_path``.

Split out of the former flat ``tests/test_figures.py`` to mirror
``src/figures/_common.py`` under the three-tree discipline. Logic unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from figures import apply_style, figures_dir
from figures._common import (
    MIN_QUANTITATIVE_FONT_SIZE,
    contrast_ratio,
    contrasting_text_color,
    plt,
    require_contrast,
    save_figure,
    semantic_style,
    validate_figure_text,
)


def test_apply_style_and_figures_dir(tmp_path: Path) -> None:
    apply_style()
    out = figures_dir(tmp_path)
    assert out.exists()
    assert out == tmp_path / "output" / "figures"


def test_semantic_style_lookup_fails_closed() -> None:
    assert semantic_style("naive").marker == "o"
    assert semantic_style("heuristic_robust").marker == "s"
    assert semantic_style("variational").marker == "^"
    with pytest.raises(ValueError, match="unknown semantic style"):
        semantic_style("invented")


def test_contrast_helpers_enforce_normal_and_large_text_thresholds() -> None:
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert require_contrast("#000000", "#FFFFFF") >= 4.5
    with pytest.raises(ValueError, match="below the required 4.5"):
        require_contrast("#777777", "#FFFFFF")
    assert require_contrast("#777777", "#FFFFFF", large_text=True) >= 3.0
    assert contrasting_text_color("#101820") == "#FFFFFF"
    assert contrasting_text_color("#228b8d") == "#000000"


def test_shared_save_boundary_rejects_visible_text_below_floor(tmp_path: Path) -> None:
    apply_style()
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "undersized", fontsize=MIN_QUANTITATIVE_FONT_SIZE - 0.5)
    with pytest.raises(ValueError, match="visible figure text below"):
        save_figure(fig, tmp_path / "too-small.png")
    assert not (tmp_path / "too-small.png").exists()
    plt.close(fig)


def test_visible_text_at_declared_floor_is_valid() -> None:
    apply_style()
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "at floor", fontsize=MIN_QUANTITATIVE_FONT_SIZE)
    validate_figure_text(fig, minimum_font_size=MIN_QUANTITATIVE_FONT_SIZE)
    plt.close(fig)


def test_shared_save_boundary_rejects_low_effective_contrast(tmp_path: Path) -> None:
    apply_style()
    fig, ax = plt.subplots()
    ax.text(
        0.5,
        0.5,
        "low contrast",
        color="#888888",
        fontsize=MIN_QUANTITATIVE_FONT_SIZE,
    )
    with pytest.raises(ValueError, match="insufficient effective contrast"):
        save_figure(fig, tmp_path / "low-contrast.png")
    assert not (tmp_path / "low-contrast.png").exists()
    plt.close(fig)


def test_effective_contrast_resolves_text_bbox_and_lower_patch() -> None:
    apply_style()
    fig, (bbox_axis, patch_axis) = plt.subplots(1, 2)
    bbox_axis.text(
        0.5,
        0.5,
        "boxed",
        color="white",
        fontsize=MIN_QUANTITATIVE_FONT_SIZE,
        bbox={"facecolor": "#1A1A2E", "edgecolor": "none", "alpha": 1.0},
    )
    patch_axis.bar([0.5], [1.0], width=0.8, color="#1A1A2E")
    patch_axis.text(
        0.5,
        0.5,
        "on bar",
        color="white",
        fontsize=MIN_QUANTITATIVE_FONT_SIZE,
        ha="center",
    )
    validate_figure_text(fig, minimum_font_size=MIN_QUANTITATIVE_FONT_SIZE)
    plt.close(fig)


def test_effective_contrast_composites_translucent_text() -> None:
    apply_style()
    fig, ax = plt.subplots()
    ax.text(
        0.5,
        0.5,
        "translucent",
        color="black",
        alpha=0.2,
        fontsize=MIN_QUANTITATIVE_FONT_SIZE,
    )
    with pytest.raises(ValueError, match="insufficient effective contrast"):
        validate_figure_text(fig, minimum_font_size=MIN_QUANTITATIVE_FONT_SIZE)
    plt.close(fig)
