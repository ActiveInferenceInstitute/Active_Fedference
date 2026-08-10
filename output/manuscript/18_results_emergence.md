## Bayesian model reduction selects supported structure {#sec:results-emergence}

The first two studies fixed the model's structure and asked how well its beliefs
and learned parameters track the world; the third asks whether the model can also
shed structure it never needed. The estimand is a Bayesian-model-reduction
free-energy difference; the design is a categorical BMR diagnostic related to the
structure-emergence mechanism discussed by Friston et al. [@friston2024federated],
through the Bayesian-model-reduction lineage [@friston2011post]. A full Dirichlet
model carries a redundant state — one column the data never support — ranging over
$n = 4$ candidate states. Bayesian model reduction scores swapping
the prior for a *reduced* prior that prunes that column; the free-energy
difference $\Delta F$ is the model-reduction objective [@eq:bmr-deltaf]. This is a
single deterministic evidence comparison, so there is no resampled sample and, by
design, no confidence interval or paired test. The structure-learning frame here
is the discrete-state model-selection thread the active-inference community has
developed [@smith2020active], applied to a colony's shared generative model.

$\Delta F$ is positive for the correct (redundant) pruning — the simpler model
has more evidence and the run converges on it — and negative for the control
pruning of a well-supported column, which is correctly rejected:

- $\Delta F$, pruning the **redundant** column: 3.68
  (positive — reduction accepted)
- $\Delta F$, pruning a **supported** column (control):
  -27.67 (negative — reduction rejected)
- Emergence converged (redundant accepted, supported rejected):
  Yes

The sign pattern
$\Delta F_{\text{redundant}} > 0 > \Delta F_{\text{supported}}$ is the demonstrated
emergence verdict: the colony's generative model prunes the structure its data
never support and retains the structure they do.

![Bayesian-model-reduction (BMR) free-energy difference. Source relation: source-mechanism analogue to the model-reduction mechanism in Friston et al. (2024), Fig. 9; estimand: BMR $\Delta F$ in nats; uncertainty: deterministic closed-form comparison. $\Delta F$ for two candidate likelihood-column prunings in a colony with $n = 4$ hidden states. x-axis: the candidate pruning (redundant column vs. supported-column control); y-axis: $\Delta F$ in nats, where a positive value means the reduced model has more evidence and the pruning is accepted. The redundant-column bar is positive ($\Delta F = 3.68$ nats) — the data never supported this structure, so pruning it is the correct decision — while the supported-column control bar is negative ($\Delta F = -27.67$ nats), correctly rejected. The opposing signs constitute the emergence verdict (Yes). No error bar applies: BMR is a deterministic closed-form comparison on a single posterior.](../output/figures/emergence_bmr.png){#fig:emergence-bmr width=80%}

[@fig:emergence-bmr] contrasts the two prunings and annotates the convergence
verdict.

Studies 1–3 all ran in a *trusting* world, where every broadcast belief is
honest. The contamination sweep that follows removes that assumption, and it is
the point at which the three robustness axes of [@sec:robustness-axes-results]
begin to diverge.
