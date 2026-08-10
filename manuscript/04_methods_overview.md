# Methods: the federated active-inference stack {#sec:methods}

[]{#sec:methodology}

This section develops the federated generalized variational inference (FedGVI)
core in the discrete-categorical setting and defines the primitives whose
recovery limits [@sec:formalism] then states as numbered theorems. Every belief
here is a categorical pmf — a non-negative vector summing to one — so the
generalized-variational-inference machinery reduces to closed forms that are
exactly testable. All mathematics lives in `src/fedference/`; the prose names
the module and the identity that pins each claim.

The active-inference community has built a rich apparatus for federated belief
sharing: discrete-state-space agents that broadcast posteriors and fuse them
into a consensus [@dacosta2020active; @friston2024federated], message-passing
toolboxes that make exact-Bayes inference scalable [@heins2022pymdp;
@bagaev2023rxinfer], and collective and multi-agent formulations in which
ensembles coordinate by sharing observations and beliefs
[@heins2023collective; @albarracin2022epistemic; @kaufmann2021collective].
We accept that apparatus and extend it: the consensus rule the field uses is
exact-Bayes and trusting, with no account of what happens when an agent in the
ensemble is misspecified or adversarial. Outside active inference, the
federated-learning and robust-Bayes literatures address important parts of that question —
decentralized aggregation [@mcmahan2017communication], partitioned and
federated variational inference [@ashman2022partitioned; @bui2018partitioned],
and generalized, robustness-bearing Bayesian updating
[@bissiri2016general; @knoblauch2022generalized; @basu1998robust;
@zhang2018generalized] — but none has been carried into the
generative-model-bearing, action-selecting POMDP setting. The methodology below
is the bridge: it federates the FedGVI objective [@mildner2025fedgvi] per agent
inside an active-inference ensemble, proves the standard-Bayes client limits,
and tests the project-local zero-robustness log-linear-pool identity. Under the
qualified categorical bridge of [@sec:method-aggregation], that pool
specializes Eq. 7's message-combination term rather than the complete source
protocol. [@fig:system-overview] illustrates the three-axis architecture and
the recovery hierarchy.

## Federation protocol: local update, server fusion, broadcast {#sec:method-protocol}

A colony of $N$ agents shares a single latent factor $s \in \{1,\dots,n_s\}$
(in the sentinel scenario, the location of a creature on a grid of $n_s$ cells).
Each round proceeds in three steps:

1. **Local inference.** Agent $n$ observes $o_n$ and forms a local posterior
   $q_n(s)$ over the shared factor by a generalized-Bayes update against its own
   cavity (the colony belief with agent $n$'s previous contribution removed).
   This is where robustness enters per agent: the update minimizes a loss-plus-
   divergence objective, and the FedGVI choice of a bounded loss is what carries
   the source theorem's bounded-influence result under its matching assumptions.
2. **Broadcast.** Agent $n$ broadcasts $q_n(s)$, optionally with a scalar
   base weight $w_n \ge 0$.
3. **Aggregation.** The server (or, equivalently, each agent acting as its own
   server) fuses the broadcast beliefs into a consensus. Following sensory
   attenuation — "agents do not hear themselves" — an agent's heard consensus
   excludes its own message.

The protocol has two distinct places where robustness can live, and we keep them
separate throughout. The **per-agent generalized-Bayes update** in step 1 is
FedGVI-faithful at the stated primitive level: its formal bounded-influence claim
is conditional on the source theorem's assumptions. The
**server-side aggregation rule** in step 3 admits an optional
divergence-reweighting heuristic that down-weights agents far from the emerging
consensus; this heuristic is a complementary device whose positive formal
property is recovery of the naive consensus in its trusting limit, while a scoped
proposition rejects one declared separable objective class. [@sec:robustness-axes]
holds this boundary; no figure, table, or sentence in this work grants the
server-side heuristic the per-agent FedGVI guarantee.

## Notation for beliefs, losses, and divergences {#sec:method-notation}

The authoritative symbol and API contract is [@sec:supp-notation]. In the main
text, $q_n(s)$ denotes agent $n$'s local posterior, $q(s)$ the global
consensus, and $q_{-n}(s)$ the cavity after removing the site factor
$t_n(s)$. The prior is $\pi_0(s)$, while $\boldsymbol{\pi}$ is a policy.
The POMDP tensors are $A[o,s]$, $B[s',s,u]$, $C[o]$, and $D_0[s]$.
The aggregation weights are $w_n$ (raw/base), $a_n$ (raw variational
effective), and $\widetilde a_n$ (normalized influence). The server
robustness coefficient is $c$, the variational entropy weight is $\lambda$,
the Rényi order is $\alpha$, the density-power parameter is $\beta$, and
the robust cross-entropy parameter is $q_{\rm loss}$. The notation supplement
also defines the seed/trial nesting and all statistical quantities used below.

The study is run over a fixed ensemble of {{CONFIG_N_AGENTS}} agents sharing a
factor of {{CONFIG_N_LOCATIONS}} locations, with all randomness seeded at
{{EXPERIMENT_SEED}}; the full per-study configuration is tabulated in
[@tbl:study_params]. As an independent generative-model-free baseline, we also
implement FedGVI in a deterministic MLP complement trained with the
density-power $\beta$-loss — generalized variational inference with a
point-mass variational family ([@sec:results-baseline]). The remaining
methodology subsections develop each primitive in turn: the generalized-Bayes
update and its recovery to standard Bayes ([@sec:method-genbayes]), the
divergence family and its KL limit ([@sec:method-divergences]) and the robust
loss family and its NLL limit ([@sec:method-losses]), the aggregation identity
([@sec:method-aggregation]),
the lift to a belief-sharing round ([@sec:method-belief-sharing]), and the paired
statistics that earn every "robust beats naive" verdict
([@sec:methods-statistics]).
