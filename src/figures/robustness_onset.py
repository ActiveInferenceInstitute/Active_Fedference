"""Robustness-onset figure: naive vs robust accuracy vs rate, per mechanism.

Small-multiples (one panel per directional contamination mechanism) of the naive
log-linear-pool accuracy and the pooled display member's accuracy as the
contamination rate rises, from
:func:`fedference.experiments.run_robustness_onset`. The onset rate — where the
pooled robust display member first reliably exceeds naive — is marked. The panels show the qualitatively
different rate dependence: the additive confident-wrong/drift attacks degrade the
naive pool gradually and robust stays above past onset, while the multiplicative
byzantine attack opens a transient robustness window before escalating to a veto
cliff where both collapse. Pure ``matplotlib`` (Agg); the curves come from the
analysis workflow, this module only draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._common import (
    COLOR_MUTED,
    apply_style,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
)


def _endpoint_label_positions(first: float, second: float) -> tuple[float, float]:
    """Return boundary-safe endpoint-label positions with visible separation."""
    low, high = 0.08, 0.94
    first_position = min(high, max(low, first))
    second_position = min(high, max(low, second))
    minimum_gap = 0.13
    if abs(first_position - second_position) >= minimum_gap:
        return first_position, second_position

    midpoint = min(
        high - minimum_gap / 2,
        max(low + minimum_gap / 2, (first_position + second_position) / 2),
    )
    if first >= second:
        return midpoint + minimum_gap / 2, midpoint - minimum_gap / 2
    return midpoint - minimum_gap / 2, midpoint + minimum_gap / 2


def generate_robustness_onset(
    by_kind: dict,
    *,
    project_root: Path | None = None,
    filename: str = "robustness_onset.png",
) -> Path:
    """Render naive vs robust accuracy-vs-rate panels with onset markers.

    Args:
        by_kind: Mapping ``{kind: {rates, naive_curve, robust_curve, naive_ci,
            robust_ci, best_robust_method_by_rate, onset_rate}}`` from
            :func:`fedference.experiments.run_robustness_onset`. The source
            report supplies 95% seed-bootstrap bands; older curve-only mappings
            remain renderable without bands.
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If ``by_kind`` is empty.
    """
    if not by_kind:
        raise ValueError("by_kind must be non-empty")
    kinds = list(by_kind)
    apply_style()
    # Embedded at width=95% (~6.2 in): large fonts so effective text >= 7 pt.
    _FS_TICK, _FS_LABEL, _FS_TITLE, _FS_ANN = 13, 14, 13, 10
    fig, axes_array = plt.subplots(
        len(kinds),
        1,
        figsize=(6.5, max(4.8, 3.05 * len(kinds))),
        sharex=True,
        sharey=True,
    )
    fig.set_layout_engine("none")
    # Reserve a dedicated band beneath the title for the shared legend.  This
    # keeps the legend clear of the final panel's x-axis at manuscript scale.
    fig.subplots_adjust(left=0.14, right=0.79, top=0.82, bottom=0.10, hspace=0.44)
    axes = np.atleast_1d(axes_array).tolist()

    naive_style = semantic_style("naive")
    robust_style = semantic_style("heuristic_robust")
    rule_style = semantic_style("reference_rule")

    for ax, kind in zip(axes, kinds):
        cell = by_kind[kind]
        rates = np.asarray(cell["rates"], dtype=np.float64)
        naive_curve = np.asarray(cell["naive_curve"], dtype=np.float64)
        robust_curve = np.asarray(cell["robust_curve"], dtype=np.float64)
        naive_ci = np.asarray(cell.get("naive_ci", []), dtype=np.float64)
        robust_ci = np.asarray(cell.get("robust_ci", []), dtype=np.float64)
        has_ci = naive_ci.shape == (rates.size, 2) and robust_ci.shape == (rates.size, 2)
        if has_ci:
            ax.fill_between(
                rates,
                naive_ci[:, 0],
                naive_ci[:, 1],
                color=naive_style.color,
                alpha=0.10,
                linewidth=0,
                label="95 % seed bootstrap interval",
            )
            ax.fill_between(
                rates,
                robust_ci[:, 0],
                robust_ci[:, 1],
                color=robust_style.color,
                alpha=0.10,
                linewidth=0,
            )
        ax.plot(
            rates,
            naive_curve,
            marker=naive_style.marker,
            markerfacecolor=naive_style.color,
            markeredgecolor=naive_style.keyline,
            markersize=5,
            linewidth=naive_style.linewidth,
            color=naive_style.color,
            linestyle=naive_style.dash,
            label="reference log pool",
        )
        ax.plot(
            rates,
            robust_curve,
            marker=robust_style.marker,
            markerfacecolor="white",
            markeredgecolor=robust_style.keyline,
            markersize=5,
            linewidth=robust_style.linewidth,
            color=robust_style.color,
            linestyle=robust_style.dash,
            label="pooled display server preset",
        )
        onset = cell.get("onset_rate")
        if onset is not None:
            onset_value = float(onset)
            ax.axvline(
                onset_value,
                color=rule_style.color,
                linestyle=rule_style.dash,
                linewidth=rule_style.linewidth,
            )
            ax.axvspan(
                onset_value,
                min(float(rates.max()), onset_value + 0.035),
                color=robust_style.color,
                alpha=0.07,
            )
            ax.annotate(
                f"onset {onset:g}",
                xy=(float(onset), 0.05),
                xytext=(3, 0),
                textcoords="offset points",
                fontsize=_FS_ANN,
                color=rule_style.color,
                rotation=90,
                va="bottom",
            )
        else:
            ax.annotate(
                "no reliable onset",
                xy=(0.5, 0.05),
                xycoords="axes fraction",
                ha="center",
                fontsize=_FS_ANN,
                color=COLOR_MUTED,
            )
        label_x = float(rates[-1]) + 0.012
        naive_endpoint = float(naive_curve[-1])
        robust_endpoint = float(robust_curve[-1])
        naive_y, robust_y = _endpoint_label_positions(naive_endpoint, robust_endpoint)
        ax.annotate(
            "reference",
            xy=(float(rates[-1]), naive_endpoint),
            xytext=(label_x, naive_y),
            fontsize=_FS_ANN,
            color=naive_style.keyline,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": naive_style.keyline, "lw": 0.8},
            clip_on=False,
        )
        ax.annotate(
            "display preset",
            xy=(float(rates[-1]), robust_endpoint),
            xytext=(label_x, robust_y),
            fontsize=_FS_ANN,
            color=robust_style.keyline,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": robust_style.keyline, "lw": 0.8},
            clip_on=False,
        )
        if kind == "byzantine":
            ax.annotate(
                "veto cliff",
                xy=(rates[-2], robust_curve[-2]),
                xytext=(-70, 26),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": COLOR_MUTED, "lw": 0.9},
                fontsize=_FS_ANN,
                color=COLOR_MUTED,
            )
        ax.set_title(kind.replace("_", " "), fontsize=_FS_TITLE)
        if ax is axes[-1]:
            ax.set_xlabel("Contamination rate $\\epsilon$", labelpad=5)
        ax.set_xlim(float(rates.min()) - 0.02, float(rates.max()) + 0.12)
        ax.set_ylim(-0.04, 1.05)
        # --- stats box: onset rate + final accuracy gap ---
        naive_end = float(naive_curve[-1])
        robust_end = float(robust_curve[-1])
        gap = robust_end - naive_end
        onset_txt = f"onset \u03b5 = {float(onset):g}" if onset is not None else "no onset"
        ax.text(
            0.04,
            0.62,
            f"{onset_txt}\ngap@max = {gap:+.3f}",
            transform=ax.transAxes,
            fontsize=_FS_ANN + 1,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": COLOR_MUTED, "alpha": 0.85},
        )
        ax.tick_params(labelsize=_FS_TICK)
        ax.xaxis.label.set_size(_FS_LABEL)
    fig.supylabel(
        "Mean consensus accuracy $q(\\mathrm{true})$",
        x=0.02,
        fontsize=_FS_LABEL,
    )
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=_FS_ANN + 1,
        loc="upper center",
        bbox_to_anchor=(0.48, 0.915),
        ncol=min(3, len(labels)),
    )
    fig.suptitle(
        "Server-preset onset by contamination mechanism",
        fontsize=_FS_TITLE + 1,
        fontweight="bold",
    )

    path = save_figure(fig, figures_dir(project_root) / filename)
    from ._presentation_studies import onset_presentation

    onset_presentation(path, by_kind)
    return path


__all__ = ["generate_robustness_onset"]
