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
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.5 * len(kinds), 4.6), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.20, wspace=0.25)
    if len(kinds) == 1:
        axes = [axes]

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
        label_x = rates[-1] + 0.01
        naive_y = max(float(naive_curve[-1]), 0.06)
        robust_y = max(float(robust_curve[-1]), 0.06)
        if abs(naive_y - robust_y) < 0.09:
            mid = 0.5 * (naive_y + robust_y)
            naive_y, robust_y = mid - 0.045, mid + 0.045
        ax.text(
            label_x,
            naive_y,
            "reference",
            fontsize=_FS_ANN,
            color=naive_style.keyline,
            ha="left",
            va="center",
            clip_on=False,
        )
        ax.text(
            label_x,
            robust_y,
            "display preset",
            fontsize=_FS_ANN,
            color=robust_style.keyline,
            ha="left",
            va="center",
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
        ax.set_xlabel("Contamination rate $\\epsilon$", labelpad=5)
        ax.set_xlim(float(rates.min()) - 0.02, float(rates.max()) + 0.12)
        ax.set_ylim(0.0, 1.05)
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
        methods = cell.get("best_robust_method_by_rate", [])
        if methods:
            ax.text(
                0.04,
                0.46,
                "display preset selected within each rate",
                transform=ax.transAxes,
                fontsize=_FS_ANN,
                ha="left",
                va="top",
                color=COLOR_MUTED,
            )
        ax.tick_params(labelsize=_FS_TICK)
        ax.xaxis.label.set_size(_FS_LABEL)
    axes[0].set_ylabel("Mean consensus accuracy $q(\\mathrm{true})$", labelpad=5)
    axes[0].yaxis.label.set_size(_FS_LABEL)
    # Legend on the last panel: its lower-left corner is free of the
    # rotated onset-rate label that sits there in the byzantine panel.
    axes[-1].legend(fontsize=_FS_ANN + 1, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.suptitle(
        "Server-preset onset diagnostic by contamination mechanism", fontsize=_FS_TITLE + 1, fontweight="bold"
    )

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_robustness_onset"]
