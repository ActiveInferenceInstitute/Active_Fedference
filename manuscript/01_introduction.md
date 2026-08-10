# Introduction: from belief sharing to robust generalized Bayes {#sec:introduction}

A colony of active-inference agents that shares beliefs inherits both the power
and the fragility of the pool it uses: the same multiplication of reports that
sharpens an honest consensus lets a single confident, wrong member capture it.
Two research communities each hold part of the remedy but do not, on their own,
close the gap. The active-inference community gives agents generative models,
action selection, and colony-level belief sharing. The robust- and
federated-Bayes community gives generalized objectives and aggregation methods
for inference under misspecification or contamination. The cited literatures do
not, however, provide a single tested treatment of robust categorical belief
fusion for active-inference agents. This introduction states the problem and its
evidence boundary; [@sec:gap] scopes the reviewed gap, and [@sec:contributions]
lists the contributions together with what each result does not establish.

## Active inference supplies generative agents and shared beliefs {#sec:intro-active-inference}

Active inference casts perception, learning, and action as the minimization of
variational free energy under a generative model, a program that grew out of the
free-energy principle [@friston2010free] and its process-theory implementation
[@friston2017active]. In the discrete state-space setting the formalism is now
standardized: an agent carries the categorical tensors $A$
(likelihood), $B$ (transitions), $C$ (preferences), and $D_0$ (initial priors), infers
hidden states by minimizing variational free energy, and selects actions by
minimizing *expected* free energy — the sum of a risk term and an ambiguity term
that together trade off goal-seeking against uncertainty-resolving behavior
[@dacosta2020active]. This synthesis is the substrate we build on. Mature
toolboxes make it executable at scale: pymdp provides discrete-state active
inference in Python [@heins2022pymdp], and RxInfer delivers reactive message
passing for exact Bayesian inference [@bagaev2023rxinfer].

The community has also moved from single agents to *collectives*.
Surprise-minimizing ensembles reproduce collective animal behavior such as
schooling and flocking [@heins2023collective]; epistemic communities form when
agents share a generative model and exchange evidence
[@albarracin2022epistemic]; and collective intelligence has been framed directly
in active-inference terms [@kaufmann2021collective]. The narrow bridge studied
here starts from posterior broadcasts over one shared categorical factor — a
predator's location, say — and forms a project log-linear pool
([@eq:log-linear-pool]). Under the explicit finite-shared-support,
posterior-log-potential, and fixed-weight assumptions of
[@sec:method-aggregation], that weighted geometric pool is a specialization of
Friston et al.'s Eq. 7 message-combination term [@friston2024federated].
It is not a reconstruction of the source message construction, scheduling,
cavity policy, generative factors, or complete protocol. That representation
connects the qualified categorical bridge to the classical
logarithmic-pooling literature [@genest1986combining;
@genest1986externally], to modern Bayesian treatments of log-pool weights
[@carvalho2023logpooling], to product-of-experts geometry in machine learning
[@hinton2002products], and to distributed Bayesian estimators
[@tresp2000bayesian]. In the reproduced baseline, communication lowers mean free
energy relative to the matched incommunicado condition
([@sec:results-belief_sharing]).
Friston et al. [-@friston2024federated] crystallized the colony mechanism
into three worked simulations — communicating-colony free-energy convergence,
Dirichlet language acquisition, and Bayesian model reduction structure emergence —
whose mechanisms motivate three reduced categorical analogues in this
repository ([@sec:results-recovery]). They are not numerical or exact protocol
replications; a source-parity reconstruction remains future work before
evaluating source-level equivalence.
Alongside fusion, the community
has tools for growing the model itself: active inference connects naturally to
Bayesian optimal experimental design and model selection [@smith2020active], and
post-hoc Bayesian model reduction prunes redundant structure by comparing
free-energy bounds [@friston2011post] — the engine behind our emergence study
([@sec:results-emergence]).

The reviewed active-inference belief-sharing work does not systematically
characterize fusion under explicit contamination or intentionally wrong belief
broadcasts. The log-linear pool assumes reports are compatible with the shared
generative model, while the opinion-pooling literature makes the assumptions
about independence, weights, and external Bayesian coherence explicit
[@genest1986combining; @genest1986externally]. Fully Bayesian aggregation
sharpens the point: geometric pooling is normatively compelling under
dynamic-Bayesian rationality assumptions, not an assumption-free robustness
procedure [@dietrich2021fully]. The same product geometry can be brittle: if a
report assigns zero or near-zero mass to the true state, the product can assign
near-zero mass there too. That is the failure mode tested here, not a claim that
every federation or every product pool behaves identically.

## Robust and federated Bayes supplies bounded-influence updating {#sec:intro-robust-bayes}

A separate literature studies bounded-influence updating outside active
inference. **Federated learning** aggregates models trained on
decentralized data without pooling the data itself, the canonical algorithm being
FedAvg [@mcmahan2017communication]. Cast probabilistically, federated and
continual learning unify under **partitioned variational inference**, in which
each client owns a factor of a global approximate posterior and the server
combines factors in natural-parameter space [@bui2018partitioned;
@ashman2022partitioned]. This is a closely related factor algebra expressed in a
different vocabulary; the implementation tests the correspondence in the
categorical recovery limit rather than assuming that the two settings are
interchangeable.

The robustness this paper needs comes from **generalized Bayesian inference**,
which replaces the likelihood with a *loss* and the KL regularizer with a
*general divergence*, so that the posterior minimizes a generalized objective
rather than applying Bayes' rule literally [@bissiri2016general;
@jewson2018divergence; @knoblauch2022generalized]. This includes
Gibbs-posterior updates [@jiang2008gibbs], coarsened or tempered posteriors that
condition on neighborhoods rather than exact data [@miller2018coarsening], and
learning-rate/temperature choices designed for safe updating under
misspecification [@grunwald2012safe; @kleijn2012misspecification]. Choosing a
loss or divergence with a bounded-influence property — for example, the
density-power $\beta$-loss [@basu1998robust; @fujisawa2008robust;
@ghosh2015robust] or generalized cross-entropy [@zhang2018generalized] — can cap
the influence of a contaminated observation under the corresponding assumptions.
**FedGVI** [@mildner2025fedgvi] federates this idea:
each client runs a robust generalized-Bayes update against a *cavity* (the global
posterior with the client's own factor removed), and the server aggregates the
refreshed factors under a chosen server divergence. The result is federated
inference with divergence- and loss-specific robustness guarantees, demonstrated
on Bayesian neural networks under label contamination.

Crucially, standard Bayes is a *corner* of this generalized family — the case
where the loss is the negative log-likelihood and the divergence is KL. Friston
et al. do not claim FedGVI, generalized Bayes, $\beta$-divergence, or robustness;
this manuscript supplies the recasting. Robust generalized Bayes therefore does
not replace exact-Bayes belief fusion; it contains the standard pool as a tested
zero-robustness recovery limit. That containment is the hinge this paper tests,
and the recovery limits are stated formally as numbered results in
[@sec:formalism] and the central identity in [@sec:method-aggregation].

The historical framing is deliberately modest. The manuscript uses early
probability, inverse-probability, utility, and collective-judgment sources as a
conceptual genealogy for belief, evidence, expectation, and aggregation
[@pascal1654probability; @huygens1657ratiociniis; @bernoulli1713ars;
@demoivre1718doctrine; @bayes1763essay; @laplace1774memoire;
@bernoulli1738mensura; @condorcet1785essai]. These sources do not anticipate KL
divergence, product-of-experts learning, variational Bayes, or federated
optimization; the modern correspondence to FedGVI is the formal construction
proved and tested here.

Beyond this core, we evaluate a tempered objective family, a deterministic MLP
aggregation transfer, and disjoint-field-of-view communication. These are
boundary tests: the first probes the accuracy/weight-control trade-off, the
second tests API portability in one additional model class, and the third
separates a binary-complement null result from a larger-state-space case where
communication materially improves consensus.

## Questions, design, and evidence boundary {#sec:intro-questions}

The paper answers four scoped questions. First, does turning robustness off
recover the standard log-linear pool and closed-form Bayes update? Second, under
the declared confident-wrong broadcast mechanism, how does the server heuristic
change consensus accuracy across contamination rates? Third, what does the
objective-backed variational server rule guarantee, and what accuracy does it
trade away? Fourth, how do communication, hierarchy, sensitivity, and
parameter recovery behave in the accompanying categorical extensions? The first
question is answered algebraically and by machine-precision checks; the others
are conditional simulation results. The independent unit, resampling scheme,
and fixed hidden-state/attack-target estimand are specified in
[@sec:methods-experimental-design] and [@sec:methods-statistics].

### How to read the visual architecture {#sec:intro-visual-map}

The manuscript uses two complementary visual layers. The formal schematics
adapt the generative-model and posterior-sharing perspective of Friston et al.
[-@friston2024federated] to the categorical implementation: [@fig:generative-model-schema]
shows the private sensory report and $A/B/C/D$ substrate, [@fig:message-passing]
shows how three local posterior messages become server inputs, and
[@fig:pomdp-loop] shows the hidden-state, agent, and active-control context.
These diagrams are explanatory maps, not additional
empirical observations. The data-bearing figures then report the executed
recovery checks, conditional contamination sweep, and objective/descent
diagnostics; their captions identify the relevant uncertainty and resampling
unit. The graphical abstract in [@fig:graphical-abstract] compresses the same
logic into a recovery anchor, a federation pathway, and three explicitly
non-transferable robustness axes.

[@fig:system-overview] illustrates one configured failure-and-repair case: a
partially contaminated colony broadcasts beliefs (Panel A), the equal-weight
log-linear pool is pulled toward the attack target (Panel B), and the
server-side heuristic reweights the broadcasts (Panel C). It is a deterministic
schematic, not a universal robustness claim; the three axes and their distinct
guarantees are defined in [@sec:gap] and [@sec:robustness-axes].

![System overview. Source relation: original project schematic; estimand: displayed posterior-mass and influence-weight contrasts; uncertainty: none. {{SYSTEM_OVERVIEW_N_AGENTS}} agents with heterogeneous beliefs (blue = honest, red = adversarial) feed into naive pooling (Panel B, argmax pulled off-target) versus canonical `robust_aggregate` heuristic reweighting (Panel C, true state recovered). x-axis is hidden-state index (1–{{SYSTEM_OVERVIEW_N_STATES}}) in Panels B and C; y-axis: bar height indexes probability mass per state, and each agent in Panel A carries its own mini posterior bar chart (green column = true state). The true hidden state is {{SYSTEM_OVERVIEW_TRUE_STATE_DISPLAY}}; under {{SYSTEM_OVERVIEW_CONTAMINATION_PCT}}% contamination the equal-weight pool concentrates {{SYSTEM_OVERVIEW_NAIVE_ACC_PCT}}% of consensus mass on the true state (its argmax lands on the adversaries' state), while the heuristic concentrates {{SYSTEM_OVERVIEW_ROBUST_ACC_PCT}}% and recovers the correct argmax. This is a single deterministic schematic (no resampling, hence no error bars or CI); all percentages are computed from the pooled beliefs shown, and the panel does not claim the variational server's objective-backed weight-control result for `robust_aggregate`.](../output/figures/system_overview.png){#fig:system-overview width=100%}
