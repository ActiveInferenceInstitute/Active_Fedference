"""Moving sentinel world figure (V4): three-panel condition comparison.

Headless (``Agg``) matplotlib only; no ``infrastructure.*`` imports (layer
contract). Shares the project palette via :mod:`figures._common`.
"""

from __future__ import annotations

from pathlib import Path

from figures._common import (
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_NAIVE,
    COLOR_ROBUST,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure_pair,
)


def generate_moving_world(
    results, project_root, *, filename: str = "moving_world.png"
) -> Path:
    """Three-panel bar chart for the moving sentinel world.

    Left panel: consensus accuracy. Center panel: free-energy gap (nats). Right
    panel: steps-to-consensus. Each panel compares the isolated
    (``COLOR_NAIVE``), communicating (``COLOR_ROBUST``) and EFE-guided
    (``COLOR_MUTED``) conditions; all axes are labelled.  A stats box in the
    accuracy panel shows the communication gain and EFE gain over isolated.
    """
    apply_style()
    # Embedded at width=80% (~5.2 in): large fonts so effective text >= 7 pt.
    _FS_TICK, _FS_LABEL, _FS_TITLE, _FS_ANN = 13, 14, 14, 11
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    conditions = ["isolated", "communicating", "efe_guided"]
    colors = [COLOR_NAIVE, COLOR_ROBUST, COLOR_MUTED]
    labels = ["Isolated", "Comm.", "EFE-guided"]

    acc = results["accuracy"]
    gap = results["free_energy_gap"]
    steps = results["n_steps_to_consensus"]

    for ax, data, ylabel, title in zip(
        axes,
        [acc, gap, steps],
        [
            "Accuracy (fraction correct)",
            "Free-energy gap (nats)",
            "Steps to consensus",
        ],
        ["Accuracy", "Free-energy gap", "Steps to consensus"],
    ):
        vals = [float(data.get(c, 0)) for c in conditions]
        ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_ylabel(ylabel, labelpad=5)
        ax.set_title(title, fontsize=_FS_TITLE)
        lo, hi = min(vals), max(vals)
        if lo < 0:
            # Signed range with a zero reference line so negative gaps render.
            ax.set_ylim(lo * 1.35, max(hi, 0.0) + 0.18 * (max(hi, 0.0) - lo))
            ax.axhline(y=0, color=COLOR_GRID, linewidth=0.9)
        else:
            ax.set_ylim(0, max(hi * 1.2, 0.01) + 0.01)
        ax.tick_params(axis="x", rotation=15, labelsize=_FS_TICK)
        ax.tick_params(axis="y", labelsize=_FS_TICK)
        ax.xaxis.label.set_size(_FS_LABEL)
        ax.yaxis.label.set_size(_FS_LABEL)
        ax.set_xlabel("Condition", labelpad=5)

    # --- per-bar value annotations (gap + steps panels) ---
    # The gap panel's isolated bar is exactly 0 and the steps panel's bars are
    # near-identical: print the values so ties read as measured, not broken.
    for ax, data, fmt in ((axes[1], gap, "{:+.2f}"), (axes[2], steps, "{:.2f}")):
        vals = [float(data.get(c, 0)) for c in conditions]
        y_lo, y_hi = ax.get_ylim()
        pad = 0.03 * (y_hi - y_lo)
        for i, v in enumerate(vals):
            va = "bottom" if v >= 0 else "top"
            offset = pad if v >= 0 else -pad
            ax.text(
                i, v + offset, fmt.format(v),
                ha="center", va=va, fontsize=_FS_ANN, color=COLOR_AXIS,
            )

    # --- stats box in accuracy panel ---
    iso_acc = float(acc.get("isolated", 0))
    comm_acc = float(acc.get("communicating", 0))
    efe_acc = float(acc.get("efe_guided", 0))
    comm_gain = comm_acc - iso_acc
    efe_gain = efe_acc - iso_acc
    annotate_stats_box(
        axes[0],
        f"comm gain = {comm_gain:+.3f}\nefe gain  = {efe_gain:+.3f}",
        loc="upper left",
        fontsize=_FS_ANN,
    )

    fig.suptitle("Moving sentinel world: 3 conditions", fontsize=_FS_TITLE + 1)

    out = figures_dir(Path(project_root))
    out_path = out / filename
    # The manuscript embeds the PNG (like every sibling figure); always retain
    # a PDF companion for archival/vector use.  A caller may request either
    # extension, but the canonical no-override path is the manuscript PNG.
    png_path = out_path if out_path.suffix.lower() == ".png" else out_path.with_suffix(".png")
    save_figure_pair(fig, png_path)
    return png_path if out_path.suffix.lower() == ".png" else out_path


__all__ = ["generate_moving_world"]
