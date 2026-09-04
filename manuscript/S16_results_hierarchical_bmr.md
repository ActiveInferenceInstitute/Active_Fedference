## Hierarchical information-gating diagnostic {#sec:results-hierarchical-bmr}

Study 7 shows that the {{HBMR_N_LEVELS}}-level agent runs and federates end to
end. This supplementary diagnostic asks a narrower question: under a configured
surprise threshold, does one leaf observation classify the top meta-context as
informative or non-gating? The output is a thresholded information-gating
diagnostic, not a Bayesian-model-evidence comparison or a data-driven proof of
the hierarchy's correct depth.

The implementation evaluates levels through
(:func:`fedference.bayesian_model_reduction.hierarchical_reduce`). For each
non-leaf level we measure its **Bayesian surprise**
$\mathrm{KL}(q_i \,\|\, \tilde p_i)$ — how far the leaf observation moves that
level's belief $q_i$ from its top-down prior $\tilde p_i$.

The routine labels a level below the configured threshold as prunable and a
level above it as kept. Because this is an inference-derived divergence rather
than a model re-fit, the labels are conditional on the specified worlds,
observation, and threshold.

The control is directional by construction. We build two
{{HBMR_N_LEVELS}}-level worlds that differ *only* in the top level's conditioned
priors: a **degenerate** world whose meta-context is non-gating (both
meta-context states predict the same context distribution) and an
**informative** world whose meta-context sharply distinguishes the two contexts.

In the degenerate world, the top level has Bayesian surprise
{{HBMR_DEGEN_TOP_SURPRISE}} nats and is flagged prunable (recovers the
two-level structure: {{HBMR_DEGEN_PRUNES_TOP}}). In the informative world, it
has {{HBMR_INFORM_TOP_SURPRISE}} nats and is kept
({{HBMR_INFORM_KEEPS_TOP}}).

[@fig:hierarchical-bmr] shows both configured worlds side by side. Their shared
lower-level parameters isolate the top-level gating contrast, but do not
establish a generally correct depth-selection procedure.

![Per-level Bayesian surprise and configured threshold classifications for two hierarchical worlds. Source relation: original project information-gating diagnostic related to, but not an implementation of, the BMR mechanism in Friston et al. Fig. 9; estimand: per-level Bayesian surprise in nats and its thresholded prune/keep label; uncertainty: deterministic schematic worlds, so no resampling interval applies. The x-axis is Bayesian surprise $\mathrm{KL}(q \,\|\, \text{prior})$ in nats, measuring information added by the leaf observation. The y-axis lists the non-leaf reduction targets top-down: level 0 is the meta-context (L3), level 1 is context (L2), and leaf location L1 is not reduced. Hatches, keylines, direct value-and-status labels, and vertical position distinguish the informative and non-gating worlds without color; the dark dotted rule is the declared classification threshold. The non-gating meta-context has {{HBMR_DEGEN_TOP_SURPRISE}} nats and is flagged prunable, leaving a two-level stack. The informative meta-context has {{HBMR_INFORM_TOP_SURPRISE}} nats and is kept; both worlds keep context. The unit is nats for each configured world-level target, with no independent replication unit, sample size, confidence interval, or error band. Falling below the configured threshold is a sign-control outcome, not proof of structural redundancy. The figure does not report model evidence, posterior refitting, universal depth recovery, structure emergence, or exact source-protocol replication.](../output/figures/hierarchical_bmr.png){#fig:hierarchical-bmr width=80%}

Unlike the Beta-function BMR sign control in [@sec:results-emergence] and
[@eq:bmr-deltaf], this routine thresholds per-level surprise. It therefore
checks the intended gating contrast in the declared worlds; it does not let the
data independently choose model depth.
