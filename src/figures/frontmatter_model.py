"""Model and belief flow: categorical Friston et al. (2024) specialization.

Locators: manuscript/09_methods_generative_model.md; manuscript/08_methods_belief_sharing.md;
fedference.pomdp; fedference.belief_sharing. No full source-protocol reconstruction.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from ._common import figures_dir, save_figure_pair
from ._frontmatter import arrow, canvas, emit_readings, label, node

READING = (
    "Generative model: D supplies initial state beliefs. B maps state and action to the next state. "
    "A maps hidden states to observations. C supplies log preferences over outcomes.",
    "Observe outcomes, infer a local categorical posterior, evaluate policies with expected free energy, "
    "then act and observe again. Solid arrows distinguish the closed perception and action cycle.",
    "Share posterior beliefs, not raw observations. Dashed message edges take admitted posteriors to "
    "fusion and return recipient-specific cavity beliefs, excluding the recipient's own message.",
    "The Eq. 7 bridge assumes shared positive support, posterior log potentials, and fixed weights. "
    "It specializes categorical combination; it does not reconstruct the complete source protocol.",
)


def build_figure() -> Figure:
    """Draw a model graph above an explicit perception/action/sharing cycle."""
    fig, ax = canvas(
        "From a world model to shared beliefs", "MODEL  /  PERCEPTION  /  ACTION  /  COMMUNICATION"
    )
    label(ax, 0, 86, "A  Generative model", bold=True)
    node(ax, 0, 70, 25, 11, "Initial belief D")
    node(ax, 38, 70, 24, 11, "Hidden state s")
    node(ax, 75, 70, 25, 11, "Outcome o")
    arrow(ax, (25, 75.5), (38, 75.5))
    arrow(ax, (62, 75.5), (75, 75.5))
    label(ax, 65, 80, "A", bold=True)
    node(ax, 38, 50, 24, 11, "Next state s′")
    arrow(ax, (50, 70), (50, 61))
    label(ax, 54, 67, "B + action", size=10.5)
    label(ax, 0, 61, "C: preferred outcomes\n(log preferences)", size=10.5)
    label(ax, 0, 48, "B  Local cycle and belief exchange", bold=True)
    node(ax, 0, 31, 27, 9, "Observe")
    node(ax, 37, 31, 26, 9, "Infer q(s)")
    node(ax, 73, 31, 27, 9, "Evaluate / act")
    arrow(ax, (27, 35.5), (37, 35.5))
    arrow(ax, (63, 35.5), (73, 35.5))
    arrow(ax, (87, 40), (87, 43))
    arrow(ax, (87, 43), (13, 43))
    arrow(ax, (13, 43), (13, 40))
    label(ax, 28, 42, "new outcome after action", size=10.5)
    node(ax, 33, 5, 34, 9, "Fuse peer beliefs")
    arrow(ax, (42, 31), (42, 14), dashed=True)
    arrow(ax, (58, 14), (58, 31), dashed=True)
    label(ax, 0, 17, "broadcast q(s)", size=10.5)
    label(ax, 68, 17, "cavity return", size=10.5)
    label(ax, 0, 2, "Shared support and fixed weights; categorical Eq. 7 bridge only.", size=10.5)
    return fig


def generate_frontmatter_model(*, project_root: Path | None = None) -> Path:
    """Generate the conceptual pair plus source-owned reading panels."""
    path = save_figure_pair(
        build_figure(),
        figures_dir(project_root) / "frontmatter_model.png",
        manuscript_width_fraction=1.0,
    )
    return emit_readings(path, READING)
