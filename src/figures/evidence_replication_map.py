"""Evidence-class, estimand, and independent-replication-unit map."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ._common import (
    COLOR_ARROW,
    COLOR_DEEP,
    COLOR_EDGE_PANEL,
    COLOR_PANEL_BG,
    COLOR_PANEL_FAIL,
    COLOR_PANEL_GOOD,
    COLOR_PANEL_NOTE,
    COLOR_WHITE,
    apply_style,
    figures_dir,
    plt,
    save_figure_pair,
    semantic_style,
)

MANUSCRIPT_WIDTH_FRACTION = 0.98


def _records(report: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    value = report.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return list(value)


def _text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record field {key!r} must be a non-empty string")
    return value


def _wrapped(value: str, width: int) -> str:
    return textwrap.fill(value, width=width, break_long_words=False, break_on_hyphens=False)


def _display(lane: Mapping[str, object]) -> Mapping[str, object]:
    value = lane.get("display")
    if not isinstance(value, Mapping):
        raise ValueError("each evidence lane must provide a display mapping")
    return value


_CLASS_STYLE = {
    "formal_executable": "evidence_formal",
    "source_conditional": "evidence_source_conditional",
    "conditional_empirical": "evidence_conditional_empirical",
    "scoped_implementation": "evidence_scoped",
    "open": "evidence_open",
}

_CLASS_CODE = {
    "formal_executable": "FE",
    "source_conditional": "SC",
    "conditional_empirical": "CE",
    "scoped_implementation": "SI",
    "open": "O",
}

_CLASS_SHORT_LABEL = {
    "formal_executable": "Formal /\nexecutable",
    "source_conditional": "Source-\nconditional",
    "conditional_empirical": "Conditional\nempirical",
    "scoped_implementation": "Scoped\nimpl.",
    "open": "Open",
}

_NESTING_SHORT_LABEL = {
    "study": "Study / run",
    "seed": "Seed (if present)",
    "trial": "Matched trial",
    "agent": "Agent / role",
    "step": "Step / state",
}

_INDEPENDENT_UNIT_HEADER = "Declared\nindependent unit"
_NESTING_PANEL_TITLE = (
    "Nesting grammar — arrows mean contains, pairs, or repeats; use only row-declared levels"
)


def _column_bounds(
    x: float,
    width: float,
    fractions: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("matrix column fractions must sum to one")
    bounds: list[tuple[float, float]] = []
    left = x
    for fraction in fractions:
        cell_width = width * fraction
        bounds.append((left, cell_width))
        left += cell_width
    return tuple(bounds)


def _header_row(
    ax: "plt.Axes",
    bounds: Sequence[tuple[float, float]],
    labels: Sequence[str],
    *,
    y: float,
    height: float,
    font_size: float = 10.7,
) -> None:
    if len(bounds) != len(labels):
        raise ValueError("matrix headers must match the declared columns")
    for (x, width), label in zip(bounds, labels, strict=True):
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=COLOR_DEEP,
                edgecolor=COLOR_WHITE,
                linewidth=1.0,
                zorder=0,
            )
        )
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",
            color=COLOR_WHITE,
            linespacing=1.0,
            zorder=2,
        )


def _class_key(
    ax: "plt.Axes",
    classes: Sequence[Mapping[str, object]],
    *,
    y: float,
) -> None:
    """Draw a compact coded key so class names need not repeat in every lane."""

    gap = 0.008
    x0 = 0.035
    width = (0.93 - gap * (len(classes) - 1)) / len(classes)
    for index, record in enumerate(classes):
        identifier = _text(record, "id")
        if (
            identifier not in _CLASS_STYLE
            or identifier not in _CLASS_CODE
            or identifier not in _CLASS_SHORT_LABEL
        ):
            raise ValueError(f"unknown evidence class {identifier!r}")
        style = semantic_style(_CLASS_STYLE[identifier])
        x = x0 + index * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                0.041,
                boxstyle="round,pad=0.003,rounding_size=0.004",
                facecolor=COLOR_WHITE,
                edgecolor=style.keyline,
                linewidth=1.2,
                zorder=0,
            )
        )
        ax.plot(
            [x + 0.020],
            [y + 0.0205],
            marker=style.marker,
            markersize=9.0,
            markerfacecolor=COLOR_WHITE,
            markeredgecolor=style.keyline,
            markeredgewidth=1.2,
            linestyle="none",
            zorder=2,
        )
        ax.text(
            x + 0.040,
            y + 0.0205,
            f"{_CLASS_CODE[identifier]}  {_CLASS_SHORT_LABEL[identifier]}",
            ha="left",
            va="center",
            fontsize=15.5,
            fontweight="bold",
            color=COLOR_DEEP,
            linespacing=0.95,
            zorder=2,
        )


def _matrix_rows(
    ax: "plt.Axes",
    lanes: Sequence[Mapping[str, object]],
    *,
    bounds: Sequence[tuple[float, float]],
    fields: Sequence[str],
    wraps: Sequence[int],
    top: float,
    row_height: float,
    owner_matrix: bool,
) -> None:
    """Draw one full-width half of the row-coded evidence matrix."""

    if len(bounds) != len(fields) or len(fields) != len(wraps):
        raise ValueError("evidence matrix fields, columns, and wrap widths must agree")
    for index, lane in enumerate(lanes):
        evidence_class = _text(lane, "evidence_class")
        if evidence_class not in _CLASS_STYLE or evidence_class not in _CLASS_CODE:
            raise ValueError(f"unknown evidence class {evidence_class!r}")
        style = semantic_style(_CLASS_STYLE[evidence_class])
        display = _display(lane)
        y = top - (index + 1) * row_height
        background = COLOR_WHITE if index % 2 == 0 else COLOR_PANEL_BG
        for x, width in bounds:
            ax.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    row_height,
                    facecolor=background,
                    edgecolor=COLOR_EDGE_PANEL,
                    linewidth=0.85,
                    zorder=0,
                )
            )

        first_x, _first_width = bounds[0]
        ax.add_patch(
            Rectangle(
                (first_x, y),
                0.006,
                row_height,
                facecolor=style.keyline,
                edgecolor=style.keyline,
                linewidth=0.0,
                zorder=1,
            )
        )
        for column_index, ((cell_x, cell_width), field, wrap_width) in enumerate(
            zip(bounds, fields, wraps, strict=True)
        ):
            if field == "headline":
                value = _text(lane, field)
            elif field == "class_code":
                value = _CLASS_CODE[evidence_class]
            else:
                value = _text(display, field)
            left_pad = 0.035 if column_index == 0 else 0.010
            if column_index == 0:
                ax.text(
                    cell_x + 0.018,
                    y + row_height / 2,
                    str(index + 1),
                    ha="center",
                    va="center",
                    fontsize=11.0,
                    fontweight="bold",
                    color=style.keyline,
                    zorder=3,
                )
            wrapped_value = _wrapped(value, wrap_width)
            if len(wrapped_value.splitlines()) > 3:
                raise ValueError(
                    f"evidence matrix field {field!r} in row {index + 1} exceeds three lines; "
                    "reflow or tighten its display wording"
                )
            ax.text(
                cell_x + left_pad,
                y + row_height / 2,
                wrapped_value,
                ha="left" if field != "class_code" else "center",
                va="center",
                fontsize=11.2 if field == "headline" else (10.8 if owner_matrix else 10.4),
                fontweight="bold" if field in {"headline", "class_code"} else "normal",
                color=style.keyline if field == "class_code" else COLOR_DEEP,
                linespacing=0.92 if owner_matrix else 0.84,
                zorder=2,
            )


def _nesting_strip(
    ax: "plt.Axes",
    nesting: Sequence[Mapping[str, object]],
    edges: Sequence[Mapping[str, object]],
    *,
    y: float = 0.145,
    height: float = 0.046,
    font_size: float = 10.8,
) -> None:
    expected_pairs = [
        (_text(nesting[index], "id"), _text(nesting[index + 1], "id"))
        for index in range(len(nesting) - 1)
    ]
    observed_pairs = [(_text(edge, "source"), _text(edge, "target")) for edge in edges]
    if observed_pairs != expected_pairs:
        raise ValueError("nesting edges must connect the five declared levels in order")

    x0 = 0.035
    box_width = 0.156
    gap = 0.037
    fills = (COLOR_PANEL_GOOD, COLOR_PANEL_NOTE, COLOR_PANEL_BG, COLOR_PANEL_NOTE, COLOR_PANEL_BG)
    xs = tuple(x0 + index * (box_width + gap) for index in range(len(nesting)))
    for x, record, fill in zip(xs, nesting, fills, strict=True):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                box_width,
                height,
                boxstyle="round,pad=0.003,rounding_size=0.005",
                facecolor=fill,
                edgecolor=COLOR_EDGE_PANEL,
                linewidth=1.0,
                zorder=0,
            )
        )
        ax.text(
            x + box_width / 2,
            y + height / 2,
            _NESTING_SHORT_LABEL[_text(record, "id")],
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",
            color=COLOR_DEEP,
            linespacing=1.0,
            zorder=2,
        )

    for index, _edge in enumerate(edges):
        start = xs[index] + box_width
        end = xs[index + 1]
        ax.add_patch(
            FancyArrowPatch(
                (start + 0.003, y + height / 2),
                (end - 0.003, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=9,
                color=COLOR_ARROW,
                linewidth=1.1,
                zorder=1,
            )
        )


def _no_transfer_strip(ax: "plt.Axes", no_claims: Sequence[str]) -> None:
    x0 = 0.035
    width = 0.93
    row_height = 0.024
    gap = 0.003
    for index, claim in enumerate(no_claims):
        y = 0.005 + (2 - index) * (row_height + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x0, y),
                width,
                row_height,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                facecolor=COLOR_PANEL_FAIL,
                edgecolor=semantic_style("adversarial").keyline,
                linewidth=1.0,
                zorder=0,
            )
        )
        ax.text(
            x0 + 0.010,
            y + row_height / 2,
            f"NT{index + 1}  {claim}",
            ha="left",
            va="center",
            fontsize=15.5,
            color=COLOR_DEEP,
            linespacing=0.95,
            zorder=2,
        )


def _lane_card(
    ax: "plt.Axes",
    lane: Mapping[str, object],
    *,
    index: int,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    """Draw one large-type headline-result card with all evidence fields."""
    evidence_class = _text(lane, "evidence_class")
    if evidence_class not in _CLASS_STYLE or evidence_class not in _CLASS_CODE:
        raise ValueError(f"unknown evidence class {evidence_class!r}")
    style = semantic_style(_CLASS_STYLE[evidence_class])
    display = _display(lane)
    background = COLOR_WHITE if index % 2 == 0 else COLOR_PANEL_BG
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.003,rounding_size=0.004",
            facecolor=background,
            edgecolor=COLOR_EDGE_PANEL,
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, y),
            0.006,
            height,
            facecolor=style.keyline,
            edgecolor=style.keyline,
            linewidth=0.0,
            zorder=1,
        )
    )
    ax.plot(
        [x + 0.020],
        [y + height - 0.020],
        marker=style.marker,
        markersize=8.5,
        markerfacecolor=COLOR_WHITE,
        markeredgecolor=style.keyline,
        markeredgewidth=1.4,
        linestyle="none",
        zorder=3,
    )
    headline = _wrapped(_text(lane, "headline"), 38)
    if len(headline.splitlines()) > 2:
        raise ValueError(f"evidence headline in row {index + 1} exceeds two lines")
    ax.text(
        x + 0.038,
        y + height - 0.010,
        f"{index + 1} · {headline}  [{_CLASS_CODE[evidence_class]}]",
        ha="left",
        va="top",
        fontsize=16.0,
        fontweight="bold",
        color=COLOR_DEEP,
        linespacing=0.82,
        zorder=3,
    )
    fields = (
        ("E", "estimand_unit"),
        ("R", "replication"),
        ("N", "nesting"),
        ("+", "permitted"),
        ("−", "prohibited"),
    )
    body_parts = [f"{prefix} · {_text(display, key)}" for prefix, key in fields]
    body = "\n".join(_wrapped(part, 43) for part in body_parts)
    if len(body.splitlines()) > 7:
        raise ValueError(
            f"evidence card row {index + 1} exceeds seven body lines; tighten display wording"
        )
    ax.text(
        x + 0.016,
        y + height - 0.047,
        body,
        ha="left",
        va="top",
        fontsize=15.5,
        color=COLOR_DEEP,
        linespacing=0.76,
        zorder=2,
    )


def _headline_table(
    ax: "plt.Axes",
    lanes: Sequence[Mapping[str, object]],
    *,
    top: float,
    row_height: float,
) -> None:
    """Draw the complete fourteen-row evidence ledger at large effective type."""
    fractions = (0.215, 0.18, 0.14, 0.17, 0.135, 0.16)
    bounds = _column_bounds(0.020, 0.960, fractions)
    _header_row(
        ax,
        bounds,
        (
            "Result family\n+ class",
            "Estimand\n+ unit",
            _INDEPENDENT_UNIT_HEADER,
            "Nested\nstructure",
            "May\nsupport",
            "Does not\nsupport",
        ),
        y=top,
        height=0.040,
        font_size=15.5,
    )
    wraps = (21, 17, 13, 16, 12, 16)
    fields = ("headline", "estimand_unit", "replication", "nesting", "permitted", "prohibited")
    for index, lane in enumerate(lanes):
        evidence_class = _text(lane, "evidence_class")
        if evidence_class not in _CLASS_STYLE or evidence_class not in _CLASS_CODE:
            raise ValueError(f"unknown evidence class {evidence_class!r}")
        style = semantic_style(_CLASS_STYLE[evidence_class])
        display = _display(lane)
        y = top - (index + 1) * row_height
        background = COLOR_WHITE if index % 2 == 0 else COLOR_PANEL_BG
        for column_index, ((cell_x, cell_width), field, wrap_width) in enumerate(
            zip(bounds, fields, wraps, strict=True)
        ):
            ax.add_patch(
                Rectangle(
                    (cell_x, y),
                    cell_width,
                    row_height,
                    facecolor=background,
                    edgecolor=COLOR_EDGE_PANEL,
                    linewidth=0.8,
                    zorder=0,
                )
            )
            value = _text(lane, "headline") if field == "headline" else _text(display, field)
            if field == "headline":
                value = f"{index + 1} · {value} [{_CLASS_CODE[evidence_class]}]"
            wrapped = _wrapped(value, wrap_width)
            if len(wrapped.splitlines()) > 3:
                raise ValueError(
                    f"evidence table field {field!r} in row {index + 1} exceeds three lines; "
                    "tighten its source-owned display wording"
                )
            left_pad = 0.016 if column_index == 0 else 0.008
            if column_index == 0:
                ax.add_patch(
                    Rectangle(
                        (cell_x, y),
                        0.006,
                        row_height,
                        facecolor=style.keyline,
                        edgecolor=style.keyline,
                        linewidth=0.0,
                        zorder=1,
                    )
                )
            ax.text(
                cell_x + left_pad,
                y + row_height / 2,
                wrapped,
                ha="left",
                va="center",
                fontsize=15.5,
                fontweight="bold" if field == "headline" else "normal",
                color=COLOR_DEEP,
                linespacing=0.78,
                zorder=2,
            )


def generate_evidence_replication_map(
    report: Mapping[str, object],
    *,
    project_root: Path | None = None,
    filename: str = "evidence_replication_map.png",
) -> Path:
    """Render an aligned evidence ledger plus nesting and no-transfer strips."""
    classes = _records(report, "evidence_classes")
    lanes = _records(report, "lanes")
    nesting = _records(report, "nesting")
    nesting_edges = _records(report, "nesting_edges")
    no_claims_raw = report.get("no_claims")
    if (
        not isinstance(no_claims_raw, list)
        or any(not isinstance(item, str) or not item for item in no_claims_raw)
        or len(no_claims_raw) != 3
    ):
        raise ValueError("no_claims must contain the three non-transfer statements")
    no_claims = list(no_claims_raw)
    class_labels = {_text(record, "id"): _text(record, "label") for record in classes}
    if len(class_labels) != len(classes):
        raise ValueError("evidence class ids must be unique")
    if len(lanes) != 14 or len(nesting) != 5 or len(nesting_edges) != 4:
        raise ValueError("the evidence map requires fourteen lanes and a five-level nesting chain")

    apply_style()
    plt.rcParams["figure.autolayout"] = False
    # A broad, near-square ledger holds fourteen complete result-family rows.
    # Native 15.5-point text remains at or above 7 points at the declared 98%
    # manuscript width; completeness never comes from shrinking typography.
    fig, ax = plt.subplots(figsize=(14.0, 13.5), facecolor=COLOR_WHITE)
    fig.set_layout_engine("none")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_facecolor(COLOR_WHITE)

    fig.text(
        0.5,
        0.992,
        "Headline-result evidence classes and replication units",
        ha="center",
        va="top",
        fontsize=22.0,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    _class_key(ax, classes, y=0.925)
    ax.text(
        0.020,
        0.902,
        _NESTING_PANEL_TITLE,
        ha="left",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    _nesting_strip(ax, nesting, nesting_edges, y=0.858, height=0.030, font_size=15.5)
    _headline_table(ax, lanes, top=0.803, row_height=0.0505)
    _no_transfer_strip(ax, no_claims)

    fig.subplots_adjust(left=0.010, right=0.990, top=0.998, bottom=0.010)
    return save_figure_pair(
        fig,
        figures_dir(project_root) / filename,
        manuscript_width_fraction=MANUSCRIPT_WIDTH_FRACTION,
    )


__all__ = ["generate_evidence_replication_map"]
