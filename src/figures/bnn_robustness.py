"""Logistic-regression robustness figure: held-out accuracy vs label contamination.

Draws the federated logistic-regression baseline
(:func:`fedference.bnn_baseline.fed_gvi_logreg`) held-out test accuracy as a
function of the per-client label-contamination fraction, for two exploratory
point-estimate configurations:

* **standard** — ``loss="nll"`` with L2 coefficient ``0.05``;
* **RCCE proxy** — ``loss="rcce"`` with L2 coefficient ``0.10``. The legacy
  ``divergence="AR"`` argument selects that coefficient but does not compute an
  Alpha-Renyi objective.

This figure is a small exploratory composite configuration proxy, not an
identified RCCE-only effect, FedGVI posterior, or source-protocol result. The
contrast is distinct from the server-side pooling heuristic of
``robust_influence_weights``. Pure ``matplotlib`` (Agg); the accuracy grid comes
from the analysis workflow, and this module only draws.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.ticker as _ticker

from ._common import (
    COLOR_MUTED,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
)

MANUSCRIPT_WIDTH_FRACTION = 0.80


def _configuration_style_role(label: str, comparison_index: int) -> tuple[str, bool]:
    """Return a neutral condition style without implying an aggregation method."""
    is_reference = "nll" in label.lower() or "standard" in label.lower()
    if is_reference:
        return "condition_reference", True
    if comparison_index == 0:
        return "condition_comparison", False
    return f"operating_point_{comparison_index}", False


def _direct_condition_label(label: str) -> str:
    """Return a compact, neutral curve label for placement inside the axes."""
    lowered = label.lower()
    if "nll" in lowered or "standard" in lowered:
        return "standard proxy"
    if "rcce" in lowered or "exploratory" in lowered:
        return "exploratory proxy"
    return label


def generate_bnn_robustness(
    accuracy_by_config: Mapping[str, Sequence[float]],
    contamination_levels: Sequence[float],
    *,
    accuracy_ci_by_config: Mapping[str, Sequence[Sequence[float]]] | None = None,
    selection_disclosure: str = (
        "Peak contamination selected within the displayed contamination sweep; "
        "q and points per class per client are configured inputs"
    ),
    project_root: Path | None = None,
    filename: str = "bnn_robustness.png",
) -> Path:
    """Render held-out accuracy curves vs contamination for client configurations.

    Args:
        accuracy_by_config: ``{config_label: [accuracy per contamination level]}``.
            Labels containing ``"nll"`` or ``"standard"`` use the neutral
            reference-condition style; other labels use neutral comparison or
            additional-operating-point styles.
        contamination_levels: Per-client label-flip fractions, in order.
        accuracy_ci_by_config: Optional ``{config_label: [[lo, hi], ...]}``
            intervals for shaded uncertainty bands.
        selection_disclosure: Source-owned description limiting within-sweep
            selection to the displayed peak-contamination summary.
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
            raise ValueError(f"curve for {label!r} has length {len(curve)}, expected {len(levels)}")
    if accuracy_ci_by_config is not None:
        for label, intervals in accuracy_ci_by_config.items():
            if label not in accuracy_by_config:
                raise ValueError(f"CI supplied for unknown curve {label!r}")
            if len(intervals) != len(levels):
                raise ValueError(f"CI for {label!r} has length {len(intervals)}, expected {len(levels)}")
            if any(len(pair) != 2 for pair in intervals):
                raise ValueError(f"CI for {label!r} must contain [lo, hi] pairs")

    apply_style()
    # The canonical manuscript embeds this figure at 80% width. A compact
    # seven-inch canvas keeps the 9.5-point annotations above the effective
    # 7-point floor without reducing typography.
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.14, right=0.94, top=0.76, bottom=0.34)
    comparison_index = 0
    style_by_label = {}
    for label, curve in accuracy_by_config.items():
        role, is_reference = _configuration_style_role(label, comparison_index)
        style = semantic_style(role)
        style_by_label[label] = style
        if not is_reference:
            comparison_index += 1
        if accuracy_ci_by_config is not None and label in accuracy_ci_by_config:
            intervals = accuracy_ci_by_config[label]
            lo = [float(pair[0]) for pair in intervals]
            hi = [float(pair[1]) for pair in intervals]
            ax.fill_between(
                levels,
                lo,
                hi,
                color=style.color,
                alpha=0.14,
                linewidth=0.0,
                zorder=1,
            )
        ax.plot(
            levels,
            [float(v) for v in curve],
            marker=style.marker,
            linestyle=style.dash,
            linewidth=style.linewidth,
            color=style.color,
            markeredgecolor=style.keyline,
            markerfacecolor=style.color if is_reference else "white",
            label=label,
            zorder=2 if is_reference else 3,
        )
    ax.set_xlabel("per-client label contamination fraction", labelpad=6)
    ax.set_ylabel("held-out test accuracy", labelpad=6)
    ax.yaxis.set_major_formatter(_ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0.0, 1.05)
    # This is the NumPy logistic-regression anchor, not the optional PyTorch
    # BNN complement — the title must not claim otherwise.
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2)

    # Direct labels preserve identity when the figure is printed in grayscale
    # or the legend is cropped. Use the penultimate operating point when
    # available: the displayed curves converge at the final point, so labeling
    # that endpoint would make the two names collide and obscure the data.
    label_index = -2 if len(levels) > 1 else -1
    for endpoint_index, (label, curve) in enumerate(accuracy_by_config.items()):
        style = style_by_label[label]
        ax.annotate(
            _direct_condition_label(label),
            xy=(levels[label_index], float(curve[label_index])),
            xytext=(8, -10 if endpoint_index % 2 == 0 else 10),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            color=style.keyline,
            clip_on=False,
        )

    # Peak gap annotation: find the contamination level with the largest
    # comparison-minus-reference margin. The margin is reported in the stats box only — a between-curve
    # arrow glyph is illegible at percent-level gaps and has no legend entry.
    curves = {k: [float(v) for v in c] for k, c in accuracy_by_config.items()}
    reference_candidates = [
        key for key in curves if "nll" in key.lower() or "standard" in key.lower()
    ]
    if len(curves) >= 2 and reference_candidates:
        reference_key = reference_candidates[0]
        comparison_key = [key for key in curves if key != reference_key][0]
        gaps = [
            comparison - reference
            for comparison, reference in zip(
                curves[comparison_key],
                curves[reference_key],
                strict=True,
            )
        ]
        peak_idx = max(range(len(gaps)), key=lambda i: gaps[i])
        peak_gap = gaps[peak_idx]
        annotate_stats_box(
            ax,
            f"Largest displayed margin: {peak_gap:.1%}\nat {levels[peak_idx]:.0%} contamination",
            loc="lower left",
        )

    fig.suptitle(
        "Client-loss comparison under label contamination",
        y=0.985,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.91,
        "Composite NLL/L2=0.05 versus RCCE/L2=0.10 point-estimate proxy",
        ha="center",
        va="center",
        fontsize=10,
        color=COLOR_MUTED,
    )
    fig.text(
        0.5,
        0.035,
        selection_disclosure
        + ".\nThe joint change cannot identify an RCCE-only effect; no Alpha-Renyi objective, "
        "calibration, universal robustness, posterior uncertainty, or source-protocol replication "
        "is established.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=COLOR_MUTED,
        wrap=True,
    )

    return save_figure(
        fig,
        figures_dir(project_root) / filename,
        manuscript_width_fraction=MANUSCRIPT_WIDTH_FRACTION,
    )


__all__ = ["generate_bnn_robustness"]
