## Conjugate Dirichlet likelihood-learning trajectory {#sec:results-language}

Where the first study fused fixed beliefs in a single round, the second records
the KL trajectory produced when a single agent updates the likelihood of its
shared world. The design is a categorical source-mechanism analogue of the
language-acquisition mechanism discussed by Friston et al.
[@friston2024federated].

Each configured seed runs an agent that learns the
likelihood of its shared world by conjugate Dirichlet updates over
{{LANGUAGE_NUM_STEPS}} count batches, the count update [@eq:dirichlet-update].
The recorded trajectory is
$\mathrm{KL}(\text{true } A \,\|\, \text{learned } A)$ before each batch — it
starts at the flat-prior value and declines monotonically toward zero as the
configured conjugate counts accumulate.

The KL here is the same divergence
whose $\alpha\to1$ Rényi limit is established by
Lemma \ref{lem:renyi-kl-limit}
([@eq:renyi-limit]), so the learning curve and the recovery limits measure the
same object.

The endpoint summary is: initial KL (flat prior), {{LANGUAGE_INITIAL_KL}};
final KL after {{LANGUAGE_NUM_STEPS}} batches, {{LANGUAGE_FINAL_KL}}; and total
KL reduction, {{LANGUAGE_KL_REDUCTION}}.

The trajectory contains {{LANGUAGE_N_POINTS}} ordered learning steps per seed.
Pointwise {{CI_PERCENT}}% intervals resample the
$n = {{LANGUAGE_N_SEEDS}}$ independent seeds, and the computed
monotone-decreasing indicator is {{LANGUAGE_MONOTONE}}.

The observed decline to a final KL of {{LANGUAGE_FINAL_KL}} is the bounded
result: under the tested count schedule, the learned likelihood moves toward
the true generative likelihood as conjugate counts accumulate. This finite
trajectory characterizes the configured update and is not a convergence-rate
result for arbitrary data-generating processes.

![Seed-mean KL divergence from the true likelihood A to the learned likelihood A. The plotted quantity is $\mathrm{KL}(\text{true }A \,\|\, \text{learned }A)$. Source relation: source-mechanism analogue to Friston et al.
(2024), Fig. 7; estimand: seed-mean KL in nats by ordered learning step. The
x-axis is the ordered Dirichlet count batch, from the flat prior at zero through
all {{LANGUAGE_NUM_STEPS}} batches ({{LANGUAGE_N_POINTS}} points per seed); the
y-axis is summed per-column KL divergence between the true likelihood and the
current expected likelihood, in nats. The solid line is the mean over
{{LANGUAGE_N_SEEDS}} independent configured seeds, and the shaded band is the
pointwise {{CI_PERCENT}}% percentile-bootstrap interval resampling seeds at each
learning step. The replication unit is seed, not the ordered trajectory points.
The mean curve falls monotonically from {{LANGUAGE_INITIAL_KL}} nats to
{{LANGUAGE_FINAL_KL}} nats (total reduction {{LANGUAGE_KL_REDUCTION}} nats,
computed from the unrounded endpoints); the computed monotone-decreasing check
is {{LANGUAGE_MONOTONE}}. This reduced categorical protocol is related to, but
does not exactly reproduce, the richer multi-episode protocol in Friston et al.
(2024).](../output/figures/language_kl_decay.png){#fig:language-kl width=80%}

[@fig:language-kl] plots the full learning curve and its CI band.
