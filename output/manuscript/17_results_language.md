## Conjugate Dirichlet likelihood-learning trajectory {#sec:results-language}

Where the first study fused fixed beliefs in a single round, the second records
the KL trajectory produced when a single agent updates the likelihood of its
shared world. The design is a categorical source-mechanism analogue of the
language-acquisition mechanism discussed by Friston et al.
[@friston2024federated].

Each configured seed runs an agent that learns the
likelihood of its shared world by conjugate Dirichlet updates over
24 count batches, the count update [@eq:dirichlet-update].
The recorded trajectory is
$\mathrm{KL}(\text{true } A \,\|\, \text{learned } A)$ before each batch — it
starts at the flat-prior value and declines monotonically toward zero as the
configured conjugate counts accumulate.

The KL here is the same divergence
whose $\alpha\to1$ Rényi limit is established by
Lemma \ref{lem:renyi-kl-limit}
([@eq:renyi-limit]), so the learning curve and the recovery limits measure the
same object.

The endpoint summary is: initial KL (flat prior), 3.4231;
final KL after 24 batches, 0.0027; and total
KL reduction, 3.4204.

The trajectory contains 25 ordered learning steps per seed.
Pointwise 95% intervals resample the
$n = 480$ independent seeds, and the computed
monotone-decreasing indicator is Yes.

The observed decline to a final KL of 0.0027 is the bounded
result: under the tested count schedule, the learned likelihood moves toward
the true generative likelihood as conjugate counts accumulate. This finite
trajectory characterizes the configured update and is not a convergence-rate
result for arbitrary data-generating processes.

![Seed-mean KL divergence from the true likelihood A to the learned likelihood A. The plotted quantity is $\mathrm{KL}(\text{true }A \,\|\, \text{learned }A)$. Source relation: source-mechanism analogue to Friston et al.
(2024), Fig. 7; estimand: seed-mean KL in nats by ordered learning step. The
x-axis is the ordered Dirichlet count batch, from the flat prior at zero through
all 24 batches (25 points per seed); the
y-axis is summed per-column KL divergence between the true likelihood and the
current expected likelihood, in nats. The solid line is the mean over
480 independent configured seeds, and the shaded band is the
pointwise 95% percentile-bootstrap interval resampling seeds at each
learning step. The replication unit is seed, not the ordered trajectory points.
The mean curve falls monotonically from 3.4231 nats to
0.0027 nats (total reduction 3.4204 nats,
computed from the unrounded endpoints); the computed monotone-decreasing check
is Yes. This reduced categorical protocol is related to, but
does not exactly reproduce, the richer multi-episode protocol in Friston et al.
(2024).](../output/figures/language_kl_decay.png){#fig:language-kl width=80% data-slide-manifest="../output/figures/language_kl_decay.slides.json"}

[@fig:language-kl] plots the full learning curve and its CI band.
