"""Tests for the shared figure-styling helpers in ``figures._common``.

No mocks: ``apply_style`` mutates the real matplotlib rcParams (headless Agg)
and ``figures_dir`` creates a real directory under ``tmp_path``.

Split out of the former flat ``tests/test_figures.py`` to mirror
``src/figures/_common.py`` under the three-tree discipline. Logic unchanged.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import numpy as np
import pytest

from figures import apply_style, figures_dir
from figures._common import (
    DEFAULT_MANUSCRIPT_WIDTH_FRACTION,
    MIN_EFFECTIVE_MANUSCRIPT_FONT_SIZE,
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


def test_shared_save_boundary_rejects_text_too_small_after_page_scaling(
    tmp_path: Path,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(14.0, 4.0))
    ax.text(0.5, 0.5, "native floor but page-scale illegible", fontsize=10.0)
    with pytest.raises(ValueError, match="at declared manuscript scale"):
        save_figure(fig, tmp_path / "too-wide.png")
    assert not (tmp_path / "too-wide.png").exists()
    plt.close(fig)


def test_declared_narrow_embed_uses_its_actual_page_scale() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    required_native = max(
        MIN_QUANTITATIVE_FONT_SIZE,
        MIN_EFFECTIVE_MANUSCRIPT_FONT_SIZE / 0.75,
    )
    ax.text(0.5, 0.5, "narrow embed", fontsize=required_native)
    validate_figure_text(
        fig,
        minimum_font_size=MIN_QUANTITATIVE_FONT_SIZE,
        manuscript_width_fraction=0.75,
    )
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


@pytest.mark.parametrize(
    ("native_font_size", "font_weight"),
    ((18.0, "normal"), (14.0, "bold")),
)
def test_large_text_contrast_exception_uses_page_scaled_size(
    native_font_size: float,
    font_weight: str,
) -> None:
    """Native-large text must remain large after its declared page scaling."""

    apply_style()
    fig = plt.figure(figsize=(10.0, 4.0))
    fig.text(
        0.5,
        0.5,
        "native-large but ordinary-sized in the manuscript",
        color="#777777",
        fontsize=native_font_size,
        fontweight=font_weight,
        ha="center",
    )

    assert 3.0 <= contrast_ratio("#777777", "#FFFFFF") < 4.5
    with pytest.raises(
        ValueError,
        match=r"insufficient effective contrast.*required 4\.5:1",
    ):
        validate_figure_text(fig, minimum_font_size=MIN_QUANTITATIVE_FONT_SIZE)
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


@pytest.mark.parametrize("origin", ("upper", "lower"))
@pytest.mark.parametrize("extent", (None, (10.0, 11.0, 20.0, 22.0)))
def test_effective_contrast_samples_the_painted_heatmap_cell(
    origin: str,
    extent: tuple[float, float, float, float] | None,
) -> None:
    """Upper/lower origins and explicit extents map text to the right row."""

    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    values = np.asarray([[0.0], [1.0]])
    image = ax.imshow(values, cmap="gray", vmin=0.0, vmax=1.0, origin=origin, extent=extent)
    left, right, bottom, top = (float(value) for value in image.get_extent())
    x = (left + right) / 2.0
    lower_center = bottom + (top - bottom) * 0.25
    upper_center = bottom + (top - bottom) * 0.75
    row_centers = (upper_center, lower_center) if origin == "upper" else (lower_center, upper_center)
    ax.text(x, row_centers[0], "dark cell", color="white", fontsize=MIN_QUANTITATIVE_FONT_SIZE, ha="center")
    ax.text(x, row_centers[1], "light cell", color="black", fontsize=MIN_QUANTITATIVE_FONT_SIZE, ha="center")
    ax.axis("off")

    validate_figure_text(fig, minimum_font_size=MIN_QUANTITATIVE_FONT_SIZE)
    plt.close(fig)


@pytest.mark.parametrize("origin", ("upper", "lower"))
@pytest.mark.parametrize("extent", (None, (10.0, 11.0, 20.0, 22.0)))
def test_effective_contrast_rejects_wrong_text_for_painted_heatmap_cell(
    origin: str,
    extent: tuple[float, float, float, float] | None,
) -> None:
    """The negative control fails on the actual dark row for every geometry."""

    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    values = np.asarray([[0.0], [1.0]])
    image = ax.imshow(values, cmap="gray", vmin=0.0, vmax=1.0, origin=origin, extent=extent)
    left, right, bottom, top = (float(value) for value in image.get_extent())
    x = (left + right) / 2.0
    lower_center = bottom + (top - bottom) * 0.25
    upper_center = bottom + (top - bottom) * 0.75
    dark_row_center = upper_center if origin == "upper" else lower_center
    ax.text(
        x,
        dark_row_center,
        "black on black",
        color="black",
        fontsize=MIN_QUANTITATIVE_FONT_SIZE,
        ha="center",
    )
    ax.axis("off")

    with pytest.raises(ValueError, match="insufficient effective contrast"):
        validate_figure_text(fig, minimum_font_size=MIN_QUANTITATIVE_FONT_SIZE)
    plt.close(fig)


_MANUSCRIPT_FIGURE_RE = re.compile(
    r"\]\([^)]*?/figures/(?P<generator>[A-Za-z0-9_]+)\.(?:png|pdf)\)"
    r"\{#fig:[^}\s]+(?P<attributes>[^}]*)\}",
    re.DOTALL,
)
_WIDTH_RE = re.compile(r"\bwidth=(?P<percent>[0-9]+(?:\.[0-9]+)?)%")


def _manuscript_embed_widths(root: Path) -> dict[str, float]:
    """Return one exact canonical embed fraction for every manuscript figure."""

    widths: dict[str, float] = {}
    sections = sorted((root / "manuscript").glob("[0-9]*.md")) + sorted(
        (root / "manuscript").glob("S[0-9]*.md")
    )
    for section in sections:
        source = section.read_text(encoding="utf-8")
        for match in _MANUSCRIPT_FIGURE_RE.finditer(source):
            width = _WIDTH_RE.search(match.group("attributes"))
            assert width is not None, f"{section.name}: figure embed omits width"
            generator = match.group("generator")
            fraction = float(width.group("percent")) / 100.0
            if generator in widths:
                assert widths[generator] == fraction, (
                    f"{generator}: inconsistent manuscript embed widths"
                )
            widths[generator] = fraction
    return widths


def _declared_boundary_width(
    module_name: str,
    call: ast.Call,
) -> float:
    width_keyword = next(
        (keyword.value for keyword in call.keywords if keyword.arg == "manuscript_width_fraction"),
        None,
    )
    if width_keyword is None:
        return DEFAULT_MANUSCRIPT_WIDTH_FRACTION
    if isinstance(width_keyword, ast.Constant) and isinstance(width_keyword.value, int | float):
        return float(width_keyword.value)
    if isinstance(width_keyword, ast.Name):
        module = importlib.import_module(f"figures.{module_name}")
        return float(getattr(module, width_keyword.id))
    pytest.fail(
        f"{module_name}: manuscript_width_fraction must be a numeric literal or module constant"
    )


def test_every_manuscript_embed_matches_its_effective_font_boundary() -> None:
    """Bind all embeds, including narrow and full-width figures, to saver QA."""

    root = Path(__file__).resolve().parents[2]
    widths = _manuscript_embed_widths(root)
    generator_names = {
        path.stem
        for path in (root / "src" / "figures").glob("*.py")
        if path.name not in {"__init__.py", "_common.py", "_metadata.py"}
    }
    assert set(widths) == generator_names

    for module_name, embedded_fraction in sorted(widths.items()):
        source_path = root / "src" / "figures" / f"{module_name}.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        calls_by_name: dict[str, list[ast.Call]] = {
            "save": [],
            "validate": [],
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id in {"save_figure", "save_figure_pair"}:
                calls_by_name["save"].append(node)
            elif node.func.id == "validate_figure_text":
                calls_by_name["validate"].append(node)
        boundary_calls = calls_by_name["save"] or calls_by_name["validate"]
        assert len(boundary_calls) == 1, (
            f"{module_name}: expected one saver or direct text-validation boundary"
        )
        declared_fraction = _declared_boundary_width(module_name, boundary_calls[0])
        assert declared_fraction == embedded_fraction, (
            f"{module_name}: manuscript embeds at {embedded_fraction:.0%}, "
            f"but effective-font QA declares {declared_fraction:.0%}"
        )
