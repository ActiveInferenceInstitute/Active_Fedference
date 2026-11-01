"""GNN consumption and evidence boundaries; Friston et al. (2024) specialization.

Locators: fedference.gnn.loader/native/backend; examples/06_gnn_sentinel.py;
docs/manuscript/accessibility.md. No backend is qualified by this schematic.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from ._common import figures_dir, save_figure_pair
from ._frontmatter import arrow, canvas, emit_readings, label, node

READING = (
    "A GNN model document is admitted through a pinned parser, or the packaged sentinel uses its "
    "source-bound cache. Validate dimensions, axis order, support, and model matrices before consumption.",
    "Native execution consumes GNN-derived A, B, C, D values through native_world and native_posterior. "
    "The sentinel exchange example compares extracted matrices against the native model.",
    "Optional upstream execution requires an explicit source root, qualified runtime and backend "
    "contract, followed by real execution and parity checks. Dashed arrows mark prerequisites, "
    "not a backend support or qualification claim.",
    "Receipts bind declared sources, inputs and outputs. Integrity evidence does not establish "
    "scientific validity. Parser success is not execution; matrix parity is not protocol replication; "
    "scientific conclusions still require appropriate models, controls and scoped evidence.",
)


def build_figure() -> Figure:
    """Branch admitted model data into native and conditional upstream routes."""
    fig, ax = canvas("A model file must reach real execution", "GNN CONSUMPTION  /  EXECUTION  /  EVIDENCE")
    node(ax, 12, 75, 76, 11, "GNN document → admitted model\nPinned parser or source-bound sentinel cache")
    label(ax, 0, 71, "Check dimensions, axis order, support and A / B / C / D.", size=11)
    arrow(ax, (39, 65), (24, 58))
    arrow(ax, (61, 65), (77, 58), dashed=True)
    node(ax, 0, 45, 47, 13, "Native consumption\nnative_world / native_posterior")
    node(ax, 53, 45, 47, 13, "Optional upstream\nExplicit root + qualified runtime")
    label(ax, 0, 42, "GNN-derived matrices\nenter inference", size=11)
    label(ax, 53, 42, "Real run + parity required\nNo backend certified here", size=11)
    arrow(ax, (24, 34), (24, 27))
    arrow(ax, (77, 34), (77, 27), dashed=True)
    node(ax, 0, 17, 100, 10, "Retain source / input / output digests and execution receipts")
    label(ax, 0, 13, "Integrity ≠ scientific validity", bold=True)
    label(
        ax,
        0,
        7,
        "Parser success ≠ execution. Matrix parity ≠ protocol replication.\n"
        "Scientific conclusions require models, controls and scoped evidence.",
        size=10.5,
    )
    return fig


def generate_frontmatter_execution(*, project_root: Path | None = None) -> Path:
    """Generate the execution/evidence pair and complete reading panels."""
    path = save_figure_pair(
        build_figure(),
        figures_dir(project_root) / "frontmatter_execution.png",
        manuscript_width_fraction=1.0,
    )
    return emit_readings(path, READING)
