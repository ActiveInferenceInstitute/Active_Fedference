# Conclusion: a recovery-tested bridge with bounded claims {#sec:conclusion}

Active Fedference makes a deliberately narrow bridge between two bodies of work
that are usually discussed with different objects and different standards of
evidence. Active inference describes agents that infer and act within generative
models; federated generalized Bayes describes how losses, divergences, and
variational objectives can make decentralized inference less sensitive to bad
information [@friston2017active; @dacosta2020active; @bissiri2016general;
@knoblauch2022generalized]. This paper does not dissolve those distinctions. It
puts them into one categorical implementation, identifies the exact point at
which they coincide, and then measures the consequences of moving away from
that point.

## The durable result is a recovery contract {#sec:conclusion-recovery}

The central contribution is the recovery-tested contract. The KL/NLL client
limits of the declared generalized-Bayes construction recover the closed-form
Bayes update, while the zero-robustness server branch recovers the project
log-linear pool, with maximum measured deviations of
{{RECOVERY_POSTERIOR_MAXDIFF}} and {{RECOVERY_AGGREGATE_MAXDIFF}}
([@eq:standard-bayes]; [@eq:robust-identity]). This is stronger than a verbal
analogy because the limited identities are stated in the formalism, implemented
in the core, and checked by executable invariants. Under the explicit
shared-support, posterior-log-potential, and fixed-weight bridge, the server
pool specializes Eq. 7's message-combination term; it is not an assertion that
the active-inference and robust-Bayes literatures share all assumptions,
objectives, deployment meanings, or the complete source protocol.

That distinction matters historically and methodologically. Logarithmic pooling
has a substantial literature as an aggregation rule for expert distributions,
including its connections to external Bayesianity, product-of-experts
constructions, and KL-based opinion pooling [@genest1986combining;
@genest1986externally; @hinton2002products; @abbas2009kullback;
@dietrich2021fully]. FedGVI contributes a generalized-Bayes perspective in
which the loss and divergence determine what robustness means
[@mildner2025fedgvi]. The contribution here is therefore a scoped recasting:
the project's categorical log-linear-pool specialization is the non-robust
corner of the implemented family under its stated bridge assumptions, and the
corner becomes a testable boundary condition for future robust extensions. This
does not identify the complete source belief-sharing protocol with FedGVI.

## What the evidence establishes away from the corner {#sec:conclusion-evidence}

The standard-Bayes studies provide a necessary baseline rather than ornamental
background. Communication changes mean free energy by
{{BELIEF_SHARING_DELTA_F}} nats, Dirichlet learning reduces KL from
{{LANGUAGE_INITIAL_KL}} to {{LANGUAGE_FINAL_KL}}, and Bayesian model reduction
selects the declared redundant-pruning candidate. These results recover the
declared mechanisms that make belief sharing scientifically interesting while
keeping the source relationship bounded to the stated categorical protocol
[@friston2024federated]. The extension studies then show that communication is
not automatically beneficial in every information geometry: disjoint views can
make sharing valuable, whereas a complementary moving-world control can make
additional pooling unnecessary or mildly costly.

The contamination study adds a second lesson. The server-side heuristic is
regime-dependent: robust operating points can give up a little efficiency when
contamination is weak and recover that cost when the declared attack is severe.
At the most severe swept rate, the highest pooled robust mean reaches
{{SWEEP_BEST_ROBUST_ACCURACY}} against the standard pool's
{{SWEEP_NAIVE_ACCURACY}} ([@tbl:robustness_sweep]); at the verdict rate, the
matched, BH-adjusted comparison of [@sec:results-robustness] gives
{{SWEEP_BEST_VERDICT_ACCURACY_MEAN}} against
{{SWEEP_NAIVE_VERDICT_ACCURACY_MEAN}}. The result is
therefore an operating-point contrast, not a ranking that holds for every
contamination rate, attack target, hidden state, or calibration regime. This
interpretation follows the simulation-study principle that the data-generating
mechanism, estimand, and Monte Carlo unit should be declared before a numerical
result is treated as general evidence [@morris2019simulation].

The three robustness axes give the result its proper meaning. The client-side
generalized-Bayes update is the FedGVI-faithful, theorem-bearing axis under its
matching loss and regularity assumptions. The server-side `robust_aggregate`
is the sharp empirical heuristic; its proven property here is the recovery
limit, not a transferred client-side bounded-influence theorem. The
`variational_aggregate` is the conservative complement: it descends a stated
aggregation objective and has a proven raw effective-weight bound, but its
objective-backed control does not make it the peak-accuracy display leader. Separating
these axes prevents a familiar failure in interdisciplinary work: a guarantee
proved for one operator is silently attached to another operator because both
are described with the same word, “robust.”

## Why the bridge matters for active inference {#sec:conclusion-significance}

Belief sharing is not merely a communication convenience. In an active-inference
system, a shared posterior can change expected free-energy calculations, model
comparison, and subsequent action selection. A robustification at the fusion
step can therefore alter behavior even when each local generative model is
unchanged. Conversely, a server rule that suppresses an anomalous belief may
also suppress a rare but correct observation. The relevant scientific question
is not simply whether a robust curve rises; it is which assumptions about
competence, independence, support, and action-relevant uncertainty the fusion
rule encodes [@genest1986combining; @tresp2000bayesian; @heins2023collective].

This is why the recovery corner is a useful organizing device. It gives a
common reference behavior before robustness is introduced, makes the cost of a
server intervention measurable, and lets later work compare new aggregation
rules against an interpretable baseline. The bridge also keeps the distinction
between inference and infrastructure visible. The current federation tests
show that serialized beliefs can travel through the declared local transport
and return a bit-identical consensus, but they do not turn a mathematical
aggregation identity into a claim about secure, fault-tolerant, or privacy-
preserving deployment [@mcmahan2017communication; @blanchard2017krum;
@pillutla2022robust].

## What remains unproved {#sec:conclusion-boundaries}

The evidence does not establish universal Byzantine tolerance, truth recovery,
calibration, or an optimal robustness parameter. The primary intervals are
conditional on the fixed hidden state and attack target, and the nested trial
and seed structure is reduced at the declared unit rather than treated as a
larger independent sample [@koehler2009mcse; @loy2021lmeresampler]. The
categorical state space is the object of the proof and experiment; continuous
or hybrid state spaces are a separate mathematical extension. The neural
classification complement uses point-estimate weights, so it does not establish
full posterior FedGVI behavior at the scale of the source experiments.

These are not defects to be hidden by a more expansive title. They define the
proper contribution: an executable categorical bridge, a recovery certificate,
an explicit map of theorem-bearing and heuristic components, and conditional
evidence about contamination behavior. Robust-statistics language such as
influence and breakdown remains useful for describing the failure modes, but a
finite simulation sweep is not by itself a general estimator-level robustness
theorem [@huber2009robust]. The same discipline applies to historical and
conceptual scholarship: early work on probability, inverse inference, utility,
and collective judgment supplies lineage, not evidence for modern KL, FedGVI,
or adversarial-federation claims.

## A falsifiable research program {#sec:conclusion-program}

The next stage should be organized around boundary conditions rather than a
larger collection of demonstrations. First, derive a server objective whose
minimizer is competitive with `robust_aggregate` while retaining the
variational rule's effective-weight control. Recent work on closed-form
generalized variational objectives, logarithmic-pool weighting, and robust
divergence-weighted federation provides relevant mathematical constraints
[@nguyen2026closedformgvi; @carvalho2023logpooling; @li2022gammafl]. A useful
candidate must recover the standard log-linear pool at zero robustness, state
which quantity is bounded, and fail visibly when those assumptions are violated.

Second, broaden the primary estimand across attack targets, hidden states,
adaptive adversaries, calibration conditions, and model classes. The decisive
falsifier is not a lower average score on one new grid; it is failure of the
claimed robustness advantage, recovery identity, or stated uncertainty
calibration under a pre-registered extension of the data-generating mechanism.
Third, promote the point-estimate neural complement to a posterior-parameterized
FedGVI study at source-comparable scale, and test whether the client-side
bounded-loss behavior survives capacity, optimization, and posterior-family
changes. Finally, move from local transport to multi-machine execution with
explicit threat models, authentication, failure handling, and privacy claims;
none should be inferred from the current bit-identity result.

## Final position {#sec:conclusion-position}

The strongest conclusion is consequently neither that robust belief sharing has
been solved nor that the bridge is merely metaphorical. In the declared
categorical setting, the standard-Bayes client limits and project log-linear-pool
server identity are tested recovery conditions; away from them, server behavior
is measurable, regime-dependent, and partitioned into theorem-bearing,
objective-backed, and heuristic claims. That is a modest result, but it is a
useful one: it supplies a reproducible starting point from which stronger
theorems, broader state spaces, independent implementations, and real federated
deployments can be judged without losing the baseline they are meant to extend.
