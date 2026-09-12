## Belief sharing lowers free energy at the project-pool corner {#sec:results-belief_sharing}

With the standard-Bayes client limit and project-pool recovery identity pinned
exactly ([@sec:results-recovery]), the study suite opens one step away from
them. Its estimand is the paired, seed-level difference in colony-mean
variational free energy between incommunicado and communicating conditions. The
design is a categorical source-mechanism analogue of the colony belief-sharing
result [@friston2024federated].

A colony of {{BELIEF_SHARING_N_AGENTS}} sentinels observes one hidden location
through independent noisy sensors of acuity {{BELIEF_SHARING_ACUITY}}. Each
agent forms a one-step posterior. The communicating condition applies one
standard log-linear-pool round at the `robustness=0` recovery corner
([@eq:robust-identity], [@eq:belief-round]); the incommunicado condition retains
the paired private posteriors.

Under shared support, posterior-log-potential admission, and fixed weights, the
pool specializes Eq. 7's message-combination term. It does not reproduce the
complete source protocol. The paired comparison gives the following declared
quantities:

**Communicating mean.**
$\bar F_{\text{share}} = {{BELIEF_SHARING_MEAN_F_COMMUNICATE}}$
(across-seed {{CI_PERCENT}}% bootstrap CI
$[{{BELIEF_SHARING_MEAN_F_SEED_CI_LO}}, {{BELIEF_SHARING_MEAN_F_SEED_CI_HI}}]$
over $n = {{BELIEF_SHARING_N_SEEDS}}$ seeds). The single illustrative
seed-{{BELIEF_SHARING_SEED}} run has its own colony mean
${{BELIEF_SHARING_SEED0_MEAN_F}}$.

Its per-agent {{CI_PERCENT}}% bootstrap CI is
$[{{BELIEF_SHARING_MEAN_F_CI_LO}}, {{BELIEF_SHARING_MEAN_F_CI_HI}}]$ over its
$n = {{BELIEF_SHARING_N}}$ agents. That interval characterizes the displayed
run's per-agent spread, not the across-seed mean.

**Incommunicado mean.**
$\bar F_{\text{solo}} = {{BELIEF_SHARING_MEAN_F_INCOMMUNICADO}}$.

**Paired reduction from sharing.**
$\Delta \bar F = \bar F_{\text{solo}} - \bar F_{\text{share}}
= {{BELIEF_SHARING_DELTA_F}}$; positive values mean lower free energy under
communication.

Each seed supplies one matched pair of colony means. Seeds are the independent
unit; agents are nested within each seed and are not additional independent
replicates. Across the declared seeds, the communicating
condition also has higher mean true-state accuracy
({{BELIEF_SHARING_MEAN_ACCURACY}}) and lower mean surprise
({{BELIEF_SHARING_MEAN_SURPRISE}}). These are conditional results for the reduced
protocol.

![Paired colony-mean variational free energy under communication and isolation. Source relation: reduced categorical source-mechanism analogue of Friston et al. (2024), Fig. 5; estimand: seed-level $\Delta\bar F=\bar F_{\mathrm{solo}}-\bar F_{\mathrm{share}}$ in nats, where positive values indicate lower free energy with communication. In Panel A, the x-axis is communication condition and the y-axis is colony-mean free energy. Filled reference circles mark incommunicado values, open comparison diamonds mark communicating values, fine connecting slopes preserve within-seed pairing, and open diamonds joined by a dark line identify condition means. In Panel B, the x-axis is paired difference and the y-axis is jitter only, with one open diamond per seed, a translucent distribution envelope, a dark dotted zero rule, and an open-diamond mean with its {{CI_PERCENT}}% percentile-bootstrap interval. The mean difference is {{BELIEF_SHARING_DELTA_F}} nats across $n={{BELIEF_SHARING_N_SEEDS}}$ independent seeds. Agents and ordered steps remain nested within seed; the interval resamples seeds, not agents, worlds, or deployments. This reduced-protocol result does not establish a general communication benefit or exact source-protocol replication.](../output/figures/free_energy_comparison.png){#fig:free-energy width=80% data-slide-manifest="../output/figures/free_energy_comparison.slides.json"}

[@fig:free-energy] reports the headline gap; [@fig:belief-heatmap] shows the
per-agent mechanism behind it.

![Single-panel belief heatmap over the hidden creature location. Source relation: original project diagnostic supporting the Study 1 analogue; estimand: posterior probability mass by hidden state; uncertainty: deterministic single-seed display. The hidden creature location (${{BELIEF_SHARING_N_AGENTS}}$ sentinels, acuity ${{BELIEF_SHARING_ACUITY}}$, seed ${{BELIEF_SHARING_SEED}}$). x-axis: hidden-state grid cell (creature location, ${{BELIEF_SHARING_N_STATES}}$ cells); rows: the ${{BELIEF_SHARING_N_AGENTS}}$ individual agents' private posteriors (one row per agent, dominant cell annotated), plus a bottom consensus row — separated by the divider line — holding the federated consensus fused from those posteriors by the cavity-exclusion round defined in the methods. Each private posterior concentrates only moderately on the cell its noisy observation suggests; the consensus row concentrates more sharply on the true location. All cell values are deterministic posterior probabilities for the single displayed seed, so no error band applies. This mechanism display does not establish calibration, a general communication benefit, or exact source-protocol replication.](../output/figures/belief_heatmap.png){#fig:belief-heatmap width=80% data-slide-manifest="../output/figures/belief_heatmap.slides.json"}

### Three robustness axes remain distinct in the results {#sec:robustness-axes-results}

The authoritative three-axis guarantee map is [@sec:robustness-axes]. The
contaminated studies keep its source-conditional client update, heuristic server
reweighting, and objective-backed variational server rule visually and
inferentially separate. In particular, no result below transfers the client's
bounded-influence theorem or the variational rule's raw-weight bound to
`robust_aggregate`.
