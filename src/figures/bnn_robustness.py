"""Logistic-regression robustness figure: held-out accuracy vs label contamination.

Draws the federated logistic-regression baseline
(:func:`fedference.bnn_baseline.fed_gvi_logreg`) held-out test accuracy as a
function of the per-client label-contamination fraction, for two FedGVI client
configurations:

* **standard** — ``loss="nll"`` with the ``KLD`` (Gaussian/L2) weight-space
  regularizer: the non-robust baseline, which degrades as labels are flipped;
* **robust** — ``loss="rcce"`` (robust client loss) with the ``AR`` regularizer:
  the FedGVI generalized-Bayes client objective, which opens a reproducible
  mid-range margin before the terminal contamination cliff.

This figure is the small weight-space anchor for the per-client FedGVI axis.
The contrast it shows is a client-update property (the rcce loss + AR
regularizer act inside each client), distinct from the server-side pooling
heuristic of ``robust_influence_weights``. Pure ``matplotlib`` (Agg); the
accuracy grid comes from the analysis workflow, this module only draws.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.ticker as _ticker

from ._common import (
    COLOR_NAIVE,
    COLOR_ROBUST,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def generate_bnn_robustness(
    accuracy_by_config: Mapping[str, Sequence[float]],
    contamination_levels: Sequence[float],
    *,
    accuracy_ci_by_config: Mapping[str, Sequence[Sequence[float]]] | None = None,
    project_root: Path | None = None,
    filename: str = "bnn_robustness.png",
) -> Path:
    """Render held-out accuracy curves vs contamination for client configurations.

    Args:
        accuracy_by_config: ``{config_label: [accuracy per contamination level]}``.
            Labels containing ``"nll"`` or ``"standard"`` are drawn as the naive
            baseline; any other label is drawn as a robust curve.
        contamination_levels: Per-client label-flip fractions, in order.
        accuracy_ci_by_config: Optional ``{config_label: [[lo, hi], ...]}``
            intervals for shaded uncertainty bands.
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If inputs are empty or a curve length mismatches the levels.
    """
    if not accuracy_by_config:
        raise ValueError("accuracy_by_config must be non-empty")
    levels = [float(c) for c in contamination_levels]
    if not levels:
        raise ValueError("contamination_levels must be non-empty")
    for label, curve in accuracy_by_config.items():
        if len(curve) != len(levels):
            raise ValueError(
                f"curve for {label!r} has length {len(curve)}, expected {len(levels)}"
            )
    if accuracy_ci_by_config is not None:
        for label, intervals in accuracy_ci_by_config.items():
            if label not in accuracy_by_config:
                raise ValueError(f"CI supplied for unknown curve {label!r}")
            if len(intervals) != len(levels):
                raise ValueError(
                    f"CI for {label!r} has length {len(intervals)}, expected {len(levels)}"
                )
            if any(len(pair) != 2 for pair in intervals):
                raise ValueError(f"CI for {label!r} must contain [lo, hi] pairs")

    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for label, curve in accuracy_by_config.items():
        is_naive = "nll" in label.lower() or "standard" in label.lower()
        # Robust curve drawn on top so it stays visible where the curves coincide.
        color = COLOR_NAIVE if is_naive else COLOR_ROBUST
        if accuracy_ci_by_config is not None and label in accuracy_ci_by_config:
            intervals = accuracy_ci_by_config[label]
            lo = [float(pair[0]) for pair in intervals]
            hi = [float(pair[1]) for pair in intervals]
            ax.fill_between(
                levels,
                lo,
                hi,
                color=color,
                alpha=0.14,
                linewidth=0.0,
                zorder=1,
            )
        ax.plot(
            levels,
            [float(v) for v in curve],
            marker="o",
            linewidth=2.4 if is_naive else 1.8,
            color=color,
            label=label,
            zorder=2 if is_naive else 3,
        )
    ax.set_xlabel("per-client label contamination fraction", labelpad=6)
    ax.set_ylabel("held-out test accuracy", labelpad=6)
    ax.yaxis.set_major_formatter(_ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0.0, 1.05)
    # This is the NumPy logistic-regression anchor, not the optional PyTorch
    # BNN complement — the title must not claim otherwise.
    ax.set_title("FedGVI client loss resists label contamination", pad=8)
    ax.legend(fontsize=9.5, loc="lower left")

    # Peak gap annotation: find contamination level with largest robust-standard
    # margin. The margin is reported in the stats box only — a between-curve
    # arrow glyph is illegible at percent-level gaps and has no legend entry.
    curves = {k: [float(v) for v in c] for k, c in accuracy_by_config.items()}
    std_candidates = [k for k in curves if "nll" in k.lower() or "standard" in k.lower()]
    if len(curves) >= 2 and std_candidates:
        std_key = std_candidates[0]
        rob_key = [k for k in curves if k != std_key][0]
        gaps = [r - s for r, s in zip(curves[rob_key], curves[std_key])]
        peak_idx = max(range(len(gaps)), key=lambda i: gaps[i])
        peak_gap = gaps[peak_idx]
        annotate_stats_box(
            ax,
            f"Peak margin: {peak_gap:.1%}\nat {levels[peak_idx]:.0%} contamination",
            loc="upper right",
        )

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_bnn_robustness"]
