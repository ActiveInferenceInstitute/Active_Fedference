## Structure learning: does the hierarchy earn its depth? {#sec:results-hierarchical-bmr}

Study 7 shows the {{HBMR_N_LEVELS}}-level agent runs and federates end to end,
but a deeper model is only warranted if the extra level carries information. We
close the loop with a structure-learning test: given a trained hierarchy and one
leaf observation, does Bayesian model reduction correctly decide whether the top
meta-context level should be *kept* or *pruned*?

We reduce at the level granularity
(:func:`fedference.bayesian_model_reduction.hierarchical_reduce`). For each
non-leaf level we measure its **Bayesian surprise**
$\mathrm{KL}(q_i \,\|\, \tilde p_i)$ — how far the leaf observation moves that
level's belief $q_i$ from its top-down prior $\tilde p_i$. A level whose belief
the data never move carries no structure and is prunable; an informative level
moves and is kept. This is an inference-derived divergence, not a model re-fit,
so it cannot manufacture a difference the generative model does not contain.

The test is directional by construction. We build two
{{HBMR_N_LEVELS}}-level worlds that differ *only* in the top level's conditioned
priors: a **degenerate** world whose meta-context is non-gating (both
meta-context states predict the same context distribution) and an
**informative** world whose meta-context sharply distinguishes the two contexts.
On the degenerate world the top level earns a Bayesian surprise of
{{HBMR_DEGEN_TOP_SURPRISE}} nats and is flagged prunable (recovers the
two-level structure: {{HBMR_DEGEN_PRUNES_TOP}}); on the informative world the
same level earns {{HBMR_INFORM_TOP_SURPRISE}} nats and is kept
({{HBMR_INFORM_KEEPS_TOP}}); [@fig:hierarchical-bmr] shows the per-level surprise
for both worlds side by side. Because the two worlds share every other
parameter, the opposite verdict is attributable to the meta-context's
information alone — the reduction discovers the right depth rather than assuming
it.

![Per-level Bayesian surprise and prune/keep decisions for two hierarchical worlds.
Source relation: original project BMR structure-learning diagnostic related to
the mechanism in Friston et al. Fig. 9; estimand: per-level Bayesian surprise
in nats and the resulting prune/keep decision; uncertainty: deterministic
schematic worlds, so no resampling interval is shown. Per-level Bayesian surprise for the two {{HBMR_N_LEVELS}}-level worlds. y-axis:
the non-leaf reduction targets, indexed top-down from the reduction routine as
level 0 = the meta-context (the topmost non-leaf level, L3 in the location-first
L1/L2/L3 convention used elsewhere) and level 1 = the context (L2); the leaf
location level (L1) is never a reduction target. x-axis: Bayesian surprise
$\mathrm{KL}(q \,\|\, \text{prior})$ in nats — the information the leaf
observation added at that level. Blue bars: the informative world (top level
kept). Grey bars: the degenerate world (top level prunable). The dashed red line
is the prune threshold; a level whose surprise falls below it is structurally
unnecessary. The degenerate meta-context (grey, level 0) sits at
{{HBMR_DEGEN_TOP_SURPRISE}} nats and is pruned, recovering the two-level model,
while the informative meta-context (blue, level 0) at {{HBMR_INFORM_TOP_SURPRISE}}
nats is retained; both worlds keep the context level (level 1). Deterministic
schematic worlds (no resampling), so no error band is
applicable.](../output/figures/hierarchical_bmr.png){#fig:hierarchical-bmr width=80%}

This is the same Beta-function model-reduction machinery that drives the
emergence study ([@sec:results-emergence], [@eq:bmr-deltaf]), lifted from pruning
redundant *states* within one level to pruning a redundant *level* of the
hierarchy — an honest, tested answer to "how deep should the generative model
be?" that the data, not the modeler, decides.
