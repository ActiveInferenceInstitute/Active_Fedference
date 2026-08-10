## Hierarchical POMDP: federated belief sharing across levels {#sec:results-hierarchical}

The flat sentinel world couples all agents at a single latent level — the
creature's location. A natural extension is a **2-level hierarchical POMDP** in
which location inference (Level 1, L1; {{HIER_N_LOCATIONS}} states) is coupled
to a global *context* variable (Level 2, L2; {{HIER_N_CONTEXTS}} states:
``quiet`` / ``alert``) that modulates the L1 prior. In the ``alert`` context the
creature is expected near the den (center cell); in the ``quiet`` context the
prior is uniform. Each sentinel runs alternating L1/L2 minimization
(`fedference.pomdp.hierarchical_infer`) to infer both its location belief
and the current context belief, then the colony federates both levels via a
log-linear pool.

We compare two conditions over {{HIER_N_TRIALS}} seeded trials with
{{HIER_N_AGENTS}} agents at sensor acuity {{HIER_ACUITY}}:

* **Flat** — agents ignore the hierarchy and infer location under a uniform
  prior;
* **Hierarchical** — agents run {{HIER_N_ITERS}} alternating-minimization
  iterations to couple L1 and L2 beliefs before federating.

The measured location accuracies are {{HIER_LOC_ACC_FLAT}} (flat) and
{{HIER_LOC_ACC_HIER}} (hierarchical), a gap of {{HIER_LOC_ACC_GAP}}. Across
{{HIER_N_SEEDS}} independent seeds the hierarchical location accuracy is
{{HIER_LOC_ACC_HIER_MEAN}} (SD {{HIER_LOC_ACC_HIER_STD}}; {{CI_PERCENT}} % CI {{HIER_LOC_ACC_HIER_CI_LO}}–{{HIER_LOC_ACC_HIER_CI_HI}})
versus flat {{HIER_LOC_ACC_FLAT_MEAN}} ({{CI_PERCENT}} % CI
{{HIER_LOC_ACC_FLAT_CI_LO}}–{{HIER_LOC_ACC_FLAT_CI_HI}}), a mean accuracy gap of
{{HIER_LOC_ACC_GAP_MEAN}} ({{CI_PERCENT}} % CI {{HIER_LOC_ACC_GAP_CI_LO}}–{{HIER_LOC_ACC_GAP_CI_HI}};
Wilcoxon signed-rank $p = {{HIER_WILCOX_PVALUE}}$, effect size
$r = {{HIER_EFFECT_SIZE}}$, {{HIER_EFFECT_LABEL}}). On location the gap is small
but statistically reliable in the *negative* direction — the paired test rejects
at $\alpha = {{CONFIG_POWER_ALPHA}}$ and the gap's confidence interval
({{HIER_LOC_ACC_GAP_CI_LO}}–{{HIER_LOC_ACC_GAP_CI_HI}}) excludes zero on the
negative side — so the hierarchy does not improve location accuracy in this
regime; if anything it pays a small, consistent location cost for carrying the
extra latent level. Its added value is that it *also*
infers the context latent, at accuracy {{HIER_CTX_ACC}} against a two-state
chance baseline of $0.5$. Two-level federation therefore runs L1/L2 inference
end-to-end and resolves context above chance while paying a small, reliable
location cost relative to the flat baseline ([@fig:hierarchical-pomdp]). Context beliefs across the
alternating-minimization iterations are shown in the top-middle panel: P(alert)
sits above the two-state chance line and is stable from the first iteration
onward when the observed
location is the center cell — the center-cell observation pins the context
posterior immediately, because the alert context-conditioned L1 prior is peaked
there. The full construction and
parameter sweep are detailed in the supplement ([@sec:supp-hierarchical]). For
the effect of acuity and colony size on these results, see
[@sec:results-sensitivity].

![Source relation: source-inspired original project diagnostic for a hierarchical
POMDP extension; estimand: posterior probabilities and final location-accuracy
gap in the declared seeded protocol; uncertainty: deterministic seeded run, so
no resampling interval is shown. Six-panel (2x3) visualization of the V2 hierarchical POMDP belief dynamics.
Top row shows the 2-level world; bottom row shows the 3-level extension.
Top-left panel: x-axis indexes the {{HIER_N_LOCATIONS}} location states;
y-axis shows posterior probability for the flat-prior and 2-level hierarchical
conditions given a single center-cell observation.
Top-middle panel: x-axis is alternating-minimization iteration number; y-axis
shows the L2 context posteriors P(quiet) and P(alert) under 2-level inference,
pinned by the center-cell observation and stable from the first iteration
onward.
Top-right panel: x-axis indexes location states; y-axis shows the colony L1
consensus probability after federating {{HIER_N_AGENTS}} agents, comparing flat
vs 2-level hierarchical.
Bottom-left panel: x-axis indexes location states; y-axis shows posterior
probability for the flat-prior and 3-level hierarchical conditions given a
single center-cell observation.
Bottom-middle panel: x-axis is alternating-minimization iteration number;
y-axis shows L2 P(alert) and L3 P(high_threat) under 3-level inference, stable
across iterations.
Bottom-right panel: two bars showing the measured final location-accuracy
gap (hierarchical minus flat) for the 2-level and 3-level systems, each a
single scalar measured over {{HIER_N_TRIALS}} trials, with a zero reference
line.
Deterministic seeded run (seed {{HIER_SEED}}), so bars carry no error
band.](../output/figures/hierarchical_pomdp.png){#fig:hierarchical-pomdp width=80%}
