"""Parameter recovery figure: recovered vs true acuity (Study 9).

Two-panel diagnostic of sensor-acuity recovery from the declared simulated
data. The scatter and absolute errors describe recovery on the configured
acuity grid; they do not establish global model identifiability.

Left panel  — scatter of recovered acuity (y) vs true acuity (x) with
              per-point 95 % empirical percentile-interval error bars and the
              identity line.
Right panel — bar chart of mean absolute error per acuity level, with an
              optional horizontal reference line at the global MAE.

Headless (Agg) matplotlib only; no infrastructure imports (layer contract).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from figures._common import (
    COLOR_GRID,
    COLOR_MUTED,
    apply_style,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
)
from figures._presentation_estimates import recovery_presentation

__all__ = ["generate_parameter_recovery"]

ArrayLike1D = Sequence[float] | np.ndarray


def generate_parameter_recovery(
    true_acuity: ArrayLike1D,
    recovered_acuity: ArrayLike1D,
    recovered_acuity_ci_lo: ArrayLike1D,
    recovered_acuity_ci_hi: ArrayLike1D,
    abs_error: ArrayLike1D,
    *,
    r_squared: float | None = None,
    mean_abs_error: float | None = None,
    n_trials: int | None = None,
    n_observations: int | None = None,
    project_root: str | Path | None = None,
    filename: str = "parameter_recovery.png",
) -> Path:
    """Generate a two-panel parameter-recovery figure for sensor acuity.

    Describes recovery on the configured simulation grid. Proximity to the
    identity line and observed absolute errors do not by themselves establish
    global model identifiability.

    Args:
        true_acuity: True (ground-truth) acuity values, one per condition.
        recovered_acuity: Posterior mean recovered acuity, matching length.
        recovered_acuity_ci_lo: Lower bound of the 95 % empirical percentile
            interval across independent trials, matching length.
        recovered_acuity_ci_hi: Upper bound of the 95 % empirical percentile
            interval across independent trials, matching length.
        abs_error: Mean absolute error |recovered - true| per acuity level,
            matching length.
        r_squared: Optional coefficient of determination (R²) of the
            recovered-vs-true regression; shown in the scatter title and
            summary line when provided.
        mean_abs_error: Optional global mean absolute error across all levels;
            drawn as a horizontal reference line on the error bar chart when
            provided.
        n_trials: Optional number of simulated trials per condition; shown in
            the stats box.
        n_observations: Optional number of observations per trial; shown in
            the stats box.
        project_root: Project root directory; defaults to the root inferred
            from this file's location.
        filename: Output filename under ``output/figures/``.

    Returns:
        Path to the written PNG file.

    Raises:
        ValueError: If *true_acuity* is empty or the four per-point arrays
            have inconsistent lengths.
    """
    # --- validation ---------------------------------------------------------
    n = len(true_acuity)
    if n == 0:
        raise ValueError("true_acuity must not be empty.")
    for name, seq in (
        ("recovered_acuity", recovered_acuity),
        ("recovered_acuity_ci_lo", recovered_acuity_ci_lo),
        ("recovered_acuity_ci_hi", recovered_acuity_ci_hi),
        ("abs_error", abs_error),
    ):
        if len(seq) != n:
            raise ValueError(
                f"Length mismatch: true_acuity has {n} elements but {name} has {len(seq)} elements."
            )

    # --- numpy arrays -------------------------------------------------------
    ta = np.asarray(true_acuity, dtype=np.float64)
    ra = np.asarray(recovered_acuity, dtype=np.float64)
    ci_lo = np.asarray(recovered_acuity_ci_lo, dtype=np.float64)
    ci_hi = np.asarray(recovered_acuity_ci_hi, dtype=np.float64)
    ae = np.asarray(abs_error, dtype=np.float64)

    lo_err = ra - ci_lo  # downward error bar lengths (non-negative)
    hi_err = ci_hi - ra  # upward error bar lengths (non-negative)

    # --- style --------------------------------------------------------------
    apply_style()
    recovery_style = semantic_style("estimate")
    reference_style = semantic_style("reference_rule")
    error_style = semantic_style("operating_point_1")

    fig, (ax_scatter, ax_error) = plt.subplots(1, 2, figsize=(7.8, 5.7), facecolor="white")
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.73, bottom=0.32, wspace=0.30)

    # ========================================================================
    # LEFT PANEL — scatter: recovered vs true
    # ========================================================================
    ax_scatter.errorbar(
        ta,
        ra,
        yerr=[lo_err, hi_err],
        fmt=recovery_style.marker,
        color=recovery_style.color,
        markerfacecolor="white",
        markeredgecolor=recovery_style.keyline,
        ecolor=COLOR_GRID,
        elinewidth=1.2,
        capsize=3,
        markersize=6,
        zorder=3,
        label="recovered (95 % percentile interval)",
    )

    # Identity line
    lim_lo = float(min(ta.min(), ra.min(), ci_lo.min()))
    lim_hi = float(max(ta.max(), ra.max(), ci_hi.max()))
    margin = (lim_hi - lim_lo) * 0.05
    id_range = [lim_lo - margin, lim_hi + margin]
    ax_scatter.plot(
        id_range,
        id_range,
        color=reference_style.color,
        linewidth=reference_style.linewidth,
        linestyle=reference_style.dash,
        zorder=2,
        label="identity reference",
    )

    ax_scatter.set_xlabel("True acuity $\\alpha$", labelpad=5)
    ax_scatter.set_ylabel("Mean recovered acuity $\\hat{\\alpha}$", labelpad=5)

    scatter_title = "Acuity recovery"
    if r_squared is not None:
        scatter_title += f"\nR² = {r_squared:.4f}"
    ax_scatter.set_title(scatter_title)

    ax_scatter.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.28))

    # Keep the study summary outside the data axes so intervals stay visible.
    stats_lines: list[str] = []
    if n_trials is not None:
        stats_lines.append(f"independent trials/acuity = {n_trials}")
    if n_observations is not None:
        stats_lines.append(f"observations/trial = {n_observations}")
    if mean_abs_error is not None:
        stats_lines.append(f"MAE = {mean_abs_error:.4f}")
    if r_squared is not None:
        stats_lines.append(f"R² = {r_squared:.4f}")

    if stats_lines:
        stats_text = "\n".join("; ".join(stats_lines[i:i + 2])
                               for i in range(0, len(stats_lines), 2))
        fig.text(0.5, 0.915, stats_text, ha="center", va="top", fontsize=9.5)

    # ========================================================================
    # RIGHT PANEL — bar chart of absolute error per acuity level
    # ========================================================================
    bar_width = 0.04
    ax_error.bar(
        ta,
        ae,
        width=bar_width,
        color="white",
        edgecolor=error_style.keyline,
        hatch=error_style.hatch,
        linewidth=1.1,
        alpha=0.80,
        zorder=3,
        label="|recovered − true|",
    )

    if mean_abs_error is not None:
        ax_error.axhline(
            mean_abs_error,
            color=reference_style.color,
            linewidth=reference_style.linewidth,
            linestyle=reference_style.dash,
            zorder=4,
            label=f"mean MAE = {mean_abs_error:.4f}",
        )
    ax_error.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.28))

    ax_error.set_xlabel("True acuity $\\alpha$", labelpad=5)
    ax_error.set_ylabel("|recovered − true| (mean absolute error)", labelpad=5)
    ax_error.set_title("Absolute recovery error\nper acuity level", pad=8)
    fig.suptitle("Parameter recovery: sensor acuity", fontsize=15, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.035,
        "Intervals vary independent simulation trials within each acuity condition;\n"
        "observations are nested within trial.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=COLOR_MUTED,
    )

    # ========================================================================
    # Save
    # ========================================================================
    out = figures_dir(Path(project_root) if project_root is not None else None)
    path = save_figure(
        fig,
        out / filename,
        manuscript_width_fraction=0.90,
    )
    recovery_presentation(path, ta, ra, ci_lo, ci_hi, ae, mean_abs_error, "\n".join(stats_lines))
    return path


if __name__ == "__main__":
    import numpy as _np

    _root = Path(__file__).resolve().parent.parent.parent
    _rng = _np.random.default_rng(42)
    _ta = _np.linspace(0.40, 0.95, 8)
    _noise = _rng.normal(0, 0.03, size=len(_ta))
    _ra = _np.clip(_ta + _noise, 0.0, 1.0)
    _half_ci = _np.abs(_rng.normal(0.05, 0.01, size=len(_ta)))
    _ae = _np.abs(_noise)
    out = generate_parameter_recovery(
        _ta,
        _ra,
        _ra - _half_ci,
        _ra + _half_ci,
        _ae,
        r_squared=0.987,
        mean_abs_error=float(_ae.mean()),
        n_trials=50,
        n_observations=200,
        project_root=_root,
    )
    print(out)
