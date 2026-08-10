## Belief sharing lowers free energy at the project-pool corner {#sec:results-belief_sharing}

With the standard-Bayes client limit and project-pool recovery identity pinned
exactly ([@sec:results-recovery]), the study suite opens one step away from
them. The first study's estimand is the colony's mean variational free energy
under two communication conditions; the design is a categorical
source-mechanism analogue of the colony belief-sharing result
[@friston2024federated], implemented at the stated categorical recovery limits.
A colony of
{{BELIEF_SHARING_N_AGENTS}} sentinels each observe the same hidden creature
location through an independent noisy sensor (acuity {{BELIEF_SHARING_ACUITY}})
and form a one-step variational posterior. When the colony runs one federated
belief-sharing round — the standard log-linear pool, which is the `robustness=0`
corner of the aggregation identity [@eq:robust-identity] proven in
Theorem \ref{thm:belief-sharing-recovery} — each agent's posterior moves toward
the cross-agent consensus via the belief-sharing round [@eq:belief-round].
Under the theorem's shared-support, posterior-log-potential, and fixed-weight
assumptions, that pool specializes Eq. 7's message-combination term, not the
complete source protocol; when held incommunicado, agents keep their private
posteriors. Scoring each belief against the colony's joint evidence yields the
"two heads are better than one" reduction, reported here as earned quantities
rather than asserted:

- Mean variational free energy, **communicating**:
  $\bar F_{\text{share}} = {{BELIEF_SHARING_MEAN_F_COMMUNICATE}}$
  (across-seed {{CI_PERCENT}}% bootstrap CI
  $[{{BELIEF_SHARING_MEAN_F_SEED_CI_LO}}, {{BELIEF_SHARING_MEAN_F_SEED_CI_HI}}]$
  over $n = {{BELIEF_SHARING_N_SEEDS}}$ seeds). The single illustrative
  seed-{{BELIEF_SHARING_SEED}} run has its own colony mean
  ${{BELIEF_SHARING_SEED0_MEAN_F}}$, with a per-agent {{CI_PERCENT}}% bootstrap
  CI of $[{{BELIEF_SHARING_MEAN_F_CI_LO}}, {{BELIEF_SHARING_MEAN_F_CI_HI}}]$
  over its $n = {{BELIEF_SHARING_N}}$ agents — that interval characterizes the
  displayed run's per-agent spread, not the across-seed mean
- Mean variational free energy, **incommunicado**:
  $\bar F_{\text{solo}} = {{BELIEF_SHARING_MEAN_F_INCOMMUNICADO}}$
- Free-energy reduction from sharing:
  $\Delta \bar F = {{BELIEF_SHARING_DELTA_F}}$ (communicating is strictly lower)

The across-seed colony means above are computed over
$n = {{BELIEF_SHARING_N_SEEDS}}$ seeds. The communicating colony also reaches
higher mean true-state accuracy ({{BELIEF_SHARING_MEAN_ACCURACY}}) and lower mean
surprise ({{BELIEF_SHARING_MEAN_SURPRISE}}) than its members reach alone. Sharing
pulls each private posterior toward the joint minimizer, so the mean free energy
when communicating sits below the incommunicado value.

![Mean variational free energy of the sentinel colony. Source relation: source-mechanism analogue to the belief-sharing mechanism in Friston et al. (2024), Fig. 5; estimand: colony-mean variational free energy in nats; uncertainty: across-seed spread over independent seeds. The sentinel colony (${{BELIEF_SHARING_N_AGENTS}}$ agents, acuity ${{BELIEF_SHARING_ACUITY}}$) under two communication conditions. x-axis: condition (incommunicado vs. communicating — one standard belief-sharing round), plotted in that order; y-axis: colony-mean variational free energy in nats. Bars show the mean free energy averaged across $n = {{BELIEF_SHARING_N_SEEDS}}$ independent random seeds, with whiskers marking $\pm$one across-seed standard deviation and grey points overlaying the individual per-seed values. The communicating bar is strictly lower than the incommunicado bar, with $\Delta \bar F = {{BELIEF_SHARING_DELTA_F}}$ nats — the quantitative "two heads are better than one" result. Each seed is a fully deterministic run; the whiskers are the across-seed spread, not a bootstrap or resampling interval.](../output/figures/free_energy_comparison.png){#fig:free-energy width=80%}

[@fig:free-energy] reports the headline gap; [@fig:belief-heatmap] shows the
per-agent mechanism behind it.

![Single-panel belief heatmap over the hidden creature location. Source relation: original project diagnostic supporting the Study 1 analogue; estimand: posterior probability mass by hidden state; uncertainty: deterministic single-seed display. The hidden creature location (${{BELIEF_SHARING_N_AGENTS}}$ sentinels, acuity ${{BELIEF_SHARING_ACUITY}}$, seed ${{BELIEF_SHARING_SEED}}$). x-axis: hidden-state grid cell (creature location, ${{BELIEF_SHARING_N_STATES}}$ cells); rows: the ${{BELIEF_SHARING_N_AGENTS}}$ individual agents' private posteriors (one row per agent, dominant cell annotated), plus a bottom consensus row — separated by the divider line — holding the federated consensus fused from those posteriors by the cavity-exclusion round defined in the methods. Each private posterior concentrates only moderately on the cell its noisy observation suggests; the consensus row concentrates far more sharply on the true location, and that after-sharing concentration is the mechanism behind the free-energy reduction reported by the free-energy comparison. All cell values are deterministic posterior probabilities for the single displayed seed; no error band is applicable.](../output/figures/belief_heatmap.png){#fig:belief-heatmap width=80%}

### Three robustness axes remain distinct in the results {#sec:robustness-axes-results}

Before the contaminated studies, we fix the honesty boundary that governs every
robustness claim in this paper, because the belief-sharing round above is exactly
where the axes diverge once contamination is introduced. The robust extension of
belief sharing lives on **three distinct axes** that must not be conflated.

The **per-agent axis** is the generalized-Bayes update each sentinel runs locally:
the $\beta$-loss and robust cross-entropy clients of [@eq:beta-loss] and
[@eq:rcce-loss]. This axis follows the FedGVI objective [@mildner2025fedgvi] and
carries the cited bounded-influence result only under that theorem's matching
assumptions. The recovery limits of [@sec:results-recovery] show the
per-agent update reduces to standard Bayes; the separately stated server
identity returns the project log-linear pool.

The **server-side axis** is the `robust_aggregate` divergence-reweighting that
down-weights agents at pooling time. This is a complementary *heuristic*. Its only
proven property is the naive-recovery limit of
Theorem \ref{thm:belief-sharing-recovery} — at zero robustness it
equals the standard log-linear pool ([@eq:robust-identity]) — and it carries
**no** bounded-influence guarantee. No figure, table, or sentence in
[@sec:results-robustness] or [@sec:results-baseline] grants the server-side
heuristic the guarantee that belongs to the per-agent axis. The contaminated
sweep that follows reports the axes side by side and labels which is which at
every step.

The **variational server axis** is `variational_aggregate`. It is also
server-side, but it is not the same claim as the sharp heuristic: it descends the
stated aggregation free energy ([@eq:agg-free-energy]), carries the derived
effective-weight bound proved in the supplement, and pays for that property
with a conservative maximum-entropy bias. The contaminated sweep therefore
reports behavior for the sharp heuristic and cites the variational rule only
where the objective-backed guarantee is actually in force.
