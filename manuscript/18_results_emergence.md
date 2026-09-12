## Configured BMR sign control passed {#sec:results-emergence}

The first two studies fixed the model structure. This third study instead checks
the sign of a configured Bayesian-model-reduction comparison. Its estimand is the
BMR free-energy difference for two candidate likelihood-column prunings. The
categorical diagnostic is related to the model-reduction mechanism discussed by
Friston et al. [@friston2024federated] and the post-hoc BMR lineage
[@friston2011post]. It is not a discovery claim about structure emergence.

The fixed posterior ranges over $n = {{EMERGENCE_N}}$ candidate states. One
likelihood column is unsupported by the configured evidence; another is a
supported control. BMR substitutes a reduced prior for each candidate and
computes $\Delta F$ [@eq:bmr-deltaf]. This is one deterministic closed-form
comparison, so no resampled sample, confidence interval, or paired test applies
[@smith2020active].

$\Delta F$ is positive for the declared redundant-column pruning, so the reduced
candidate is favored for this fixed posterior. **Redundant-column result:**
$\Delta F={{EMERGENCE_DELTA_F_REDUNDANT}}$; the configured reduction is accepted.

It is negative for the declared supported-column control. **Supported-column
control:** $\Delta F={{EMERGENCE_DELTA_F_SUPPORTED}}$; that reduction is
rejected.

**Configured sign-control disposition:** redundant accepted and supported
rejected, recorded as {{EMERGENCE_CONVERGENCE}}.

The observed sign pattern
$\Delta F_{\text{redundant}} > 0 > \Delta F_{\text{supported}}$ passes the
configured control: this fixed posterior favors pruning the declared redundant
column and rejects pruning the declared supported column. It does not show that
arbitrary data or model families will discover, prune, or retain the correct
structure.

![Configured Bayesian-model-reduction sign control. Source relation: source-mechanism analogue to the model-reduction mechanism in Friston et al. (2024), Fig. 9; estimand: BMR $\Delta F$ in nats for two declared pruning candidates. The x-axis is candidate likelihood-column pruning, ordered as the redundant column and supported-column control; the y-axis is $\Delta F$, where positive values favor the reduced model and negative values reject it. Dotted hatching and a dark teal keyline identify the favored redundant-column control, while cross-hatching and a dark neutral keyline identify rejected supported-column pruning; direct signed value labels and a dark dotted zero rule duplicate color. On the fixed posterior, redundant-column pruning gives $\Delta F = {{EMERGENCE_DELTA_F_REDUNDANT}}$ and supported-column pruning gives $\Delta F = {{EMERGENCE_DELTA_F_SUPPORTED}}$, so the configured sign control passes ({{EMERGENCE_CONVERGENCE}}). This is one deterministic closed-form comparison with no independent replication unit, resampling interval, confidence interval, or error bar. The configured signs do not establish universal structure emergence, consistent recovery of true structure, or exact reproduction of the source simulation.](../output/figures/emergence_bmr.png){#fig:emergence-bmr width=80% data-slide-manifest="../output/figures/emergence_bmr.slides.json"}

[@fig:emergence-bmr] contrasts the two configured prunings and reports the sign
control.

Studies 1–3 all ran in a *trusting* world, where every broadcast belief is
honest. The contamination sweep that follows removes that assumption, and it is
the point at which the three robustness axes of [@sec:robustness-axes-results]
begin to diverge.
