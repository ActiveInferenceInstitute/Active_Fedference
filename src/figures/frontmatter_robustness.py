"""Distinct robustness owners; Mildner et al. (2025) is client conditional.

Locators: manuscript/06_methods_divergences_and_losses.md and
manuscript/07_methods_aggregation.md; fedference.aggregation and fedference.fedgvi.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from ._common import COLOR_MULTI_1, COLOR_MULTI_2, COLOR_NAIVE, figures_dir, save_figure_pair
from ._frontmatter import arrow, canvas, emit_readings, label, node

READING = (
    "Client FedGVI: generalized variational updates combine a divergence and a loss. Bounded-loss "
    "influence claims require the source's matching model and regularity assumptions. "
    "KL/NLL with beta zero recovers the project-local Bayes update.",
    "Heuristic server: robust_aggregate discounts reports by disagreement with the emerging consensus. "
    "At robustness zero it exactly returns log_linear_pool. Recovery only; no transferred client theorem.",
    "Variational server: variational_aggregate alternates consensus and raw effective-weight updates "
    "on a stated objective. Descent and raw-weight bounds do not prove estimator B-robustness.",
    "Choose one server rule after the client update. Guarantees stay with their own mathematical "
    "construction. No universal robustness or source protocol replication follows from this diagram.",
)


def build_figure() -> Figure:
    """Place the conditional client owner above two distinct server branches."""
    fig, ax = canvas(
        "Robustness has distinct claim owners", "CLIENT UPDATE  /  SERVER CHOICE  /  CLAIM BOUNDARY"
    )
    label(ax, 0, 86, "A  Client FedGVI", bold=True, color=COLOR_MULTI_1)
    node(ax, 0, 71, 100, 10, "Private observations → divergence + loss → posterior qₙ(s)")
    label(
        ax,
        0,
        68,
        "Bounded-loss influence: source-conditional assumptions.\n"
        "KL / NLL / β = 0: project-local Bayes recovery.",
        size=11,
    )
    arrow(ax, (50, 58), (50, 53))
    label(ax, 22, 52, "Choose a server rule", bold=True)
    arrow(ax, (44, 48), (24, 44))
    arrow(ax, (56, 48), (76, 44))
    node(ax, 0, 34, 47, 10, "H  Heuristic pool")
    node(ax, 53, 34, 47, 10, "V  Variational server")
    label(ax, 0, 30, "robust_aggregate\nDisagreement reweighting\nRecovery only", size=11, color=COLOR_NAIVE)
    label(
        ax,
        53,
        30,
        "variational_aggregate\nObjective + block descent\nRaw-weight bounds",
        size=11,
        color=COLOR_MULTI_2,
    )
    label(ax, 0, 15, "robustness = 0\nexactly log_linear_pool", size=10.5)
    label(ax, 53, 15, "Positive c and entropy weight\nNo estimator B-robustness", size=10.5)
    label(ax, 0, 4, "No client theorem transfers to either server rule.", bold=True)
    return fig


def generate_frontmatter_robustness(*, project_root: Path | None = None) -> Path:
    """Generate the three-axis conceptual pair and complete reading panels."""
    path = save_figure_pair(
        build_figure(),
        figures_dir(project_root) / "frontmatter_robustness.png",
        manuscript_width_fraction=1.0,
    )
    return emit_readings(path, READING)
