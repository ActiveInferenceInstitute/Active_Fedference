"""Hierarchical POMDP figure (V2 upgrade): 2x3 six-panel belief dynamics.

Headless (``Agg``) matplotlib only; no ``infrastructure.*`` imports (layer
contract). Shares the project palette via :mod:`figures._common`.

Six panels — top row: 2-level (V2), bottom row: 3-level (V2 extension):

Top row (2-level):
- Top-left  (panel 0): L1 location posteriors (flat vs hierarchical) as bar-chart.
- Top-middle (panel 1): L2 context posterior evolution over alternating-min iters.
- Top-right (panel 2): Colony-fused L1 consensus under flat vs hierarchical.

Bottom row (3-level):
- Bottom-left  (panel 3): L1 location posteriors (flat vs 3-level) as bar-chart.
- Bottom-middle (panel 4): L2+L3 context/meta-context posteriors vs iteration.
- Bottom-right (panel 5): Measured final location-accuracy gap (hier - flat)
  for the 2-level and 3-level systems, as bars (one scalar per system).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from figures._common import (
    COLOR_GRID,
    COLOR_NAIVE,
    COLOR_ROBUST,
    COLOR_VARIATE,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def generate_hierarchical_pomdp(
    hier_report: dict | None = None,
    nl3_report: dict | None = None,
    project_root: str | Path | None = None,
    *,
    seed: int = 42,
    acuity: float = 0.85,
    n_agents: int = 4,
    n_trials: int = 20,
    n_iters: int = 4,
    filename: str = "hierarchical_pomdp.png",
    allow_illustrative_fallback: bool = False,
) -> Path:
    """2x3 six-panel figure for the V2 hierarchical POMDP study (2-level + 3-level).

    Args:
        project_root: Project root directory.
        seed: RNG seed for reproducibility.
        acuity: L1 sensor acuity.
        n_agents: Number of sentinel agents.
        n_trials: Trials for the accuracy-gap panel.
        n_iters: Alternating-minimization iterations per inference call.
        filename: Output filename under ``output/figures/``.
        allow_illustrative_fallback: Permit a seeded, synthetic accuracy-gap
            calculation when no executed reports are supplied. Keep this
            ``False`` for publication paths so the figure cannot silently
            replace measured report values with an illustrative calculation.

    Returns:
        Path to the written PNG file.
    """
    from fedference.aggregation import log_linear_pool
    from fedference.belief_updating import infer_states
    from fedference.pomdp import (
        N_CONTEXTS,
        N_LOCATIONS,
        N_META_CONTEXTS,
        build_3level_world,
        build_hierarchical_world,
        hierarchical_infer,
        nlevel_infer,
    )

    _EPS = 1e-12
    rng = np.random.default_rng(seed)

    # When the measured reports are supplied, the gap panel shows their final
    # measured scalar gaps (the reports carry no per-trial series), annotated
    # with the report's actual trial count (not the demo default).
    if hier_report is not None and "n_trials" in hier_report:
        n_trials = int(hier_report["n_trials"])

    # ---- 2-level world -------------------------------------------------------
    world2 = build_hierarchical_world(acuity=acuity)
    A2 = np.asarray(world2["L1"]["A"][0], dtype=np.float64)

    # Panel 1 data: L2 context posterior evolution over iters (obs=4, center).
    ctx_iters2: list[np.ndarray] = []
    for it in range(1, 8):
        r = hierarchical_infer(A2, obs=4, hier_world=world2, n_iters=it)
        ctx_iters2.append(r["q_ctx"].copy())
    ctx_arr2 = np.array(ctx_iters2)  # (n_iters, 2)

    # Panel 0 data: L1 posteriors (flat vs 2-level hier), single obs=4.
    obs_demo = 4
    flat_log_prior = np.log(np.full(N_LOCATIONS, 1.0 / N_LOCATIONS))
    q_loc_flat2 = infer_states(A2, obs_demo, flat_log_prior)
    q_loc_hier2 = hierarchical_infer(A2, obs=obs_demo, hier_world=world2, n_iters=n_iters)["q_loc"]

    # Panel 2 data: colony consensus (2-level), true state=4, context=alert.
    true_state2 = 4
    per_agent_obs2 = [
        int(rng.choice(A2.shape[0], p=np.clip(A2[:, true_state2], 0, None) /
                       np.clip(A2[:, true_state2], 0, None).sum()))
        for _ in range(n_agents)
    ]
    flat_local_posteriors2 = [
        infer_states(A2, o, flat_log_prior) for o in per_agent_obs2
    ]
    flat_consensus2 = log_linear_pool(flat_local_posteriors2)
    hier_local_posteriors2 = [
        hierarchical_infer(A2, o, world2, n_iters=n_iters)["q_loc"]
        for o in per_agent_obs2
    ]
    hier_consensus2 = log_linear_pool(hier_local_posteriors2)

    # ---- 3-level world -------------------------------------------------------
    world3 = build_3level_world(acuity=acuity)
    A3 = np.asarray(world3["L1"]["A"][0], dtype=np.float64)

    # Panel 4 data: L2+L3 posterior evolution over iters (obs=4, center).
    ctx_iters3_l2: list[np.ndarray] = []
    ctx_iters3_l3: list[np.ndarray] = []
    for it in range(1, 8):
        r3 = nlevel_infer(A3, obs=4, nlevel_world=world3, n_iters=it)
        ctx_iters3_l2.append(r3["q_levels"][1].copy())  # L2
        ctx_iters3_l3.append(r3["q_levels"][0].copy())  # L3
    ctx_arr3_l2 = np.array(ctx_iters3_l2)  # (n_iters, 2)
    ctx_arr3_l3 = np.array(ctx_iters3_l3)  # (n_iters, 2)

    # Panel 3 data: L1 posteriors (flat vs 3-level), single obs=4.
    q_loc_hier3 = nlevel_infer(A3, obs=obs_demo, nlevel_world=world3, n_iters=n_iters)["q_levels"][-1]

    # Panel 5 data: measured final accuracy gap per system. The executed
    # reports carry one scalar gap each — plotting them as a per-trial
    # trajectory would invent dynamics, so the publication panel is a two-bar
    # comparison. The fallback is available only when an explicit caller opts
    # into an illustrative, seeded synthetic figure (normally unit tests).
    gap2_list: list[float] = []
    gap3_list: list[float] = []
    if hier_report is not None and nl3_report is not None:
        gap2_list = [float(hier_report["location_accuracy_gap"])]
        gap3_list = [float(nl3_report["location_accuracy_gap"])]
    else:
        if hier_report is not None or nl3_report is not None:
            raise ValueError("hier_report and nl3_report must be supplied together")
        if not allow_illustrative_fallback:
            raise ValueError(
                "executed hierarchical reports are required; set "
                "allow_illustrative_fallback=True only for an illustrative figure"
            )
        l1_priors_l3 = world3["L2_priors_given_l3"]  # list[np.ndarray], len=N_META_CONTEXTS
        l1_priors_l2 = world3["L1_priors_given_l2"]  # list[np.ndarray], len=N_CONTEXTS
        l3_prior3 = np.asarray(world3["L3_prior"], dtype=np.float64)
        for t in range(n_trials):
            trial_rng = np.random.default_rng(seed + t + 2000)
            # Sample true state through 3-level chain.
            tl3 = int(trial_rng.choice(N_META_CONTEXTS, p=l3_prior3))
            tl2 = int(trial_rng.choice(N_CONTEXTS, p=np.asarray(l1_priors_l3[tl3], dtype=np.float64)))
            ts = int(trial_rng.choice(N_LOCATIONS, p=np.asarray(l1_priors_l2[tl2], dtype=np.float64)))
            col_2 = np.clip(A2[:, ts], 0, None)
            col_2 = col_2 / col_2.sum()
            # 2-level
            obs_list2 = [int(trial_rng.choice(A2.shape[0], p=col_2)) for _ in range(n_agents)]
            flat_b2 = [infer_states(A2, o, flat_log_prior) for o in obs_list2]
            hier_b2 = [hierarchical_infer(A2, o, world2, n_iters=n_iters)["q_loc"] for o in obs_list2]
            gap2_list.append(
                float(np.argmax(log_linear_pool(hier_b2)) == ts)
                - float(np.argmax(log_linear_pool(flat_b2)) == ts)
            )
            # 3-level
            col_3 = np.clip(A3[:, ts], 0, None)
            col_3 = col_3 / col_3.sum()
            obs_list3 = [int(trial_rng.choice(A3.shape[0], p=col_3)) for _ in range(n_agents)]
            hier_b3 = [nlevel_infer(A3, o, world3, n_iters=n_iters)["q_levels"][-1] for o in obs_list3]
            gap3_list.append(
                float(np.argmax(log_linear_pool(hier_b3)) == ts)
                - float(
                    np.argmax(
                        log_linear_pool(
                            [infer_states(A3, o, flat_log_prior) for o in obs_list3]
                        )
                    )
                    == ts
                )
            )

    final_gap2 = float(np.mean(gap2_list)) if gap2_list else 0.0
    final_gap3 = float(np.mean(gap3_list)) if gap3_list else 0.0

    # ---- Plot 2x3 layout -----------------------------------------------------
    apply_style()
    # Embedded at width=80% (~5.2 in) in the manuscript: keep the canvas
    # compact and the fonts large so effective tick text stays >= 7 pt.
    _FS_TICK, _FS_LABEL, _FS_TITLE, _FS_LEGEND, _FS_ANN = 14, 14, 13, 11, 11
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    x = np.arange(N_LOCATIONS)
    iters_x = np.arange(1, 8)

    # --- Panel 0 (top-left): L1 posteriors 2-level ---
    ax = axes[0, 0]
    ax.bar(x - 0.2, q_loc_flat2, 0.4, color=COLOR_NAIVE, label="Flat prior", alpha=0.85)
    ax.bar(x + 0.2, q_loc_hier2, 0.4, color=COLOR_ROBUST, label="2-level hier.", alpha=0.85)
    ax.axvline(x=obs_demo, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Location state")
    ax.set_ylabel("Posterior probability")
    ax.set_title("L1 posterior (2-level)")
    ax.legend(fontsize=_FS_LEGEND)
    annotate_stats_box(
        ax,
        f"acuity = {acuity:.2f}\nobs = {obs_demo}\nn_agents = {n_agents}",
        loc="upper right", fontsize=_FS_ANN,
    )

    # --- Panel 1 (top-middle): L2 context posterior evolution 2-level ---
    ax = axes[0, 1]
    ax.plot(iters_x, ctx_arr2[:, 0], "o-", color=COLOR_NAIVE,
            label=f"P({world2['context_labels'][0]})", linewidth=1.5)
    ax.plot(iters_x, ctx_arr2[:, 1], "s-", color=COLOR_ROBUST,
            label=f"P({world2['context_labels'][1]})", linewidth=1.5)
    ax.axhline(y=0.5, color=COLOR_GRID, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Alternating-min iteration")
    ax.set_ylabel("Context posterior P(ctx)")
    ax.set_title("L2 context belief (2-level)")
    ax.legend(fontsize=_FS_LEGEND)
    ax.set_ylim(0, 1)

    # --- Panel 2 (top-right): colony consensus 2-level ---
    ax = axes[0, 2]
    ax.bar(x - 0.2, flat_consensus2, 0.4, color=COLOR_NAIVE, label="Flat", alpha=0.85)
    ax.bar(x + 0.2, hier_consensus2, 0.4, color=COLOR_ROBUST, label="2-level hier.", alpha=0.85)
    ax.axvline(x=true_state2, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Location state")
    ax.set_ylabel("Colony consensus probability")
    ax.set_title(f"Colony consensus ({n_agents} agents)")
    # Both consensus distributions saturate at the true state; annotate the
    # peak values so the identical heights read as measured, not broken.
    for xoff, yoff, color, series in (
        (-0.2, 0.02, COLOR_NAIVE, flat_consensus2),
        (0.2, 0.12, COLOR_ROBUST, hier_consensus2),
    ):
        peak = int(np.argmax(series))
        ax.text(
            peak + xoff, float(series[peak]) + yoff, f"{float(series[peak]):.2f}",
            ha="center", va="bottom", fontsize=_FS_ANN - 1, color=color,
        )
    ax.set_ylim(0, 1.3)
    ax.legend(fontsize=_FS_LEGEND, loc="center left")

    # --- Panel 3 (bottom-left): L1 posteriors 3-level ---
    ax = axes[1, 0]
    ax.bar(x - 0.2, q_loc_flat2, 0.4, color=COLOR_NAIVE, label="Flat prior", alpha=0.85)
    ax.bar(x + 0.2, q_loc_hier3, 0.4, color=COLOR_VARIATE, label="3-level hier.", alpha=0.85)
    ax.axvline(x=obs_demo, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Location state")
    ax.set_ylabel("Posterior probability")
    ax.set_title("L1 posterior (3-level)")
    ax.legend(fontsize=_FS_LEGEND)

    # --- Panel 4 (bottom-middle): L2+L3 posterior evolution 3-level ---
    ax = axes[1, 1]
    ax.plot(iters_x, ctx_arr3_l2[:, 1], "s-", color=COLOR_ROBUST,
            label="L2 P(alert)", linewidth=1.5)
    ax.plot(iters_x, ctx_arr3_l3[:, 1], "^--", color=COLOR_VARIATE,
            label="L3 P(high_threat)", linewidth=1.5)
    ax.axhline(y=0.5, color=COLOR_GRID, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Alternating-min iteration")
    ax.set_ylabel("Level posterior")
    ax.set_title("L2/L3 belief (3-level)")
    ax.legend(fontsize=_FS_LEGEND)
    ax.set_ylim(0, 1)

    # --- Panel 5 (bottom-right): measured accuracy gap 2-level vs 3-level ---
    ax = axes[1, 2]
    bar_x = np.array([0.0, 1.0])
    ax.bar(bar_x, [final_gap2, final_gap3], 0.55,
           color=[COLOR_ROBUST, COLOR_VARIATE], alpha=0.85)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(["2-level", "3-level"])
    ax.set_ylabel("Accuracy gap (hier − flat)")
    ax.set_title("Measured accuracy gap")
    lo = min(final_gap2, final_gap3, 0.0)
    hi = max(final_gap2, final_gap3, 0.0)
    pad = max(0.01, 0.6 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    for bx, g in zip(bar_x, (final_gap2, final_gap3)):
        ax.annotate(f"{g:+.3f}", xy=(bx, g),
                    xytext=(0, 6 if g >= 0 else -14), textcoords="offset points",
                    ha="center", fontsize=_FS_ANN)
    ax.text(
        0.97, 0.96,
        f"measured over\nn_trials = {n_trials}",
        transform=ax.transAxes, fontsize=_FS_ANN, ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": COLOR_GRID, "alpha": 0.85},
    )

    for panel in axes.flat:
        panel.tick_params(labelsize=_FS_TICK)
        panel.xaxis.label.set_size(_FS_LABEL)
        panel.yaxis.label.set_size(_FS_LABEL)
        panel.title.set_size(_FS_TITLE)

    fig.suptitle(
        "V2 Hierarchical POMDP — 2-level (top) and 3-level (bottom) federation",
        fontsize=_FS_TITLE + 1,
    )

    out = figures_dir(Path(project_root) if project_root is not None else None)
    path = out / filename
    return save_figure(fig, path)


__all__ = ["generate_hierarchical_pomdp"]


if __name__ == "__main__":
    out = generate_hierarchical_pomdp(
        project_root=Path(__file__).resolve().parent.parent.parent,
        allow_illustrative_fallback=True,
    )
    print(out)
