## Related work: active inference, federated Bayes, and the scoped bridge {#sec:related-work}

This work sits at the boundary between two research communities with different
objects of inference. Each cited thread contributes a real component; this paper
extends the intersection without claiming that either literature is exhausted.
The positioning is therefore against the reviewed sources and their assumptions,
not against an absolute claim about everything the field has or has not done.

## Pre-modern probability, inverse probability, and collective judgment {#sec:related-historical}

These historical sources provide conceptual lineage, not evidence that early
authors anticipated KL minimization, active inference, or federation. Pascal and
Fermat, Huygens, Montmort, Bernoulli, and de Moivre made expectation and uncertain
evidence objects of calculation [@pascal1654probability;
@huygens1657ratiociniis; @montmort1708essay; @bernoulli1713ars;
@demoivre1718doctrine].

Bayes and Price framed inference from observed events to an unknown chance, and
Laplace generalized inference about causes [@bayes1763essay;
@laplace1774memoire]. Their relevance here is vocabulary for generative-model
inversion. The executable recovery identity is instead a modern, project-local
result of [@sec:method-aggregation] and [@sec:formalism].

Daniel Bernoulli's expected-utility treatment and the aggregation work of Borda
and Condorcet connect uncertain judgment to action and collective choice
[@bernoulli1738mensura; @borda1784elections; @condorcet1785essai]. They motivate
questions about competence, independence, weights, and stakes. They do not
formally support this paper's log-pool, generalized-Bayes, or robustness claims;
the full genealogy remains documented in the source audit.

## Active inference: generative agents, EFE, and colonies {#sec:related-aif}

The free-energy principle [@friston2010free] and discrete state-space process
theory [@friston2017active] connect generative models, variational inference,
and expected-free-energy action selection. The discrete synthesis
[@dacosta2020active], pymdp [@heins2022pymdp], and RxInfer
[@bagaev2023rxinfer] provide executable machinery. This project reimplements the
categorical substrate and its risk–ambiguity decomposition
([@eq:efe-decomposition], [@eq:efe-identity], [@fig:efe-decomp]), then evaluates
explicit robustness controls at belief fusion.

The fusion operator itself is not new as a mathematical object. Under the
explicit categorical posterior-log-potential and fixed-weight bridge of
[@sec:method-aggregation], the message-combination term of Friston's
belief-sharing equation is represented by a logarithmic opinion pool
[@genest1986combining; @genest1986externally] and a product-of-experts consensus
[@hinton2002products].

That limited representation is not a statement that the
complete source protocol is a log pool. Abbas' KL view of linear and log-linear
pools makes the same point in scoring-rule language: log-pooling can
be justified as a KL aggregation rule for expert distributions, not as a
contamination-robust estimator [@abbas2009kullback].

The Bayesian Committee
Machine adds the distributed-learning analogue: independent estimators trained
on data subsets can be combined by a product-style Bayesian rule, with
assumptions about conditional independence and prior accounting made explicit
[@tresp2000bayesian].

Fully Bayesian aggregation gives the social-choice
counterpart: geometric pooling of beliefs is singled out by dynamic Bayesian
rationality conditions, not by robustness to contaminated reports
[@dietrich2021fully].

That classical literature is useful precisely because it
names the hidden commitments — shared support, weight choice, prior accounting,
and external-Bayes coherence — that are easy to overlook when the same operation
appears as a colony update.

A parallel thread studies how active-inference agents coordinate as ensembles —
collective behavior and surprise minimization across a colony
[@heins2023collective], epistemic communities [@albarracin2022epistemic], and
collective intelligence [@kaufmann2021collective].

The belief-sharing thread [@friston2024federated] is one member of this family.
Agents fuse beliefs, and our colony scenario is a reduced categorical standard-
Bayes-limit analogue of its worked example ([@eq:belief-round],
[@sec:results-belief_sharing]).

Structure learning completes the conceptual frame. Active inference with
Bayesian optimal design selects and reduces models [@smith2020active], while
post-hoc BMR [@friston2011post] motivates our configured sign control
([@eq:bmr-deltaf], [@sec:results-emergence]).

Among the active-inference sources reviewed here, belief fusion is not
systematically characterized under explicit contamination or intentionally wrong
broadcasts, nor connected to the robustness theory used here. Fusion is treated
as trusting in that scoped comparison.

Friston et al. [-@friston2024federated] present communicating-colony free-energy
convergence, Dirichlet language acquisition, and Bayesian model reduction. We
use reduced categorical standard-Bayes-limit analogues of those mechanisms
([@sec:results-recovery]) before the robust extension. The third is a configured
sign control on a fixed posterior. None is an exact source-protocol or figure
reproduction.

## Robust and federated Bayes outside active inference {#sec:related-fl}

Federated learning aggregates models trained on decentralized data
[@mcmahan2017communication], and the probabilistic-federation line recasts that
aggregation as variational inference — partitioned variational inference
[@ashman2022partitioned] and its federated predecessor [@bui2018partitioned].
This is a different use of the word *federated* from Friston et al.'s federated
inference: Friston federates hidden-state beliefs among agents sharing a world
model, while machine-learning federated learning usually federates parameter or
predictive-model updates over decentralized datasets.

The bridge claimed here is
therefore algebraic and variational — a shared normalized product/log-pool
operator at the KL/NLL recovery corner — not a claim that either source paper
already solved the other's problem.

Robustness, meanwhile, has a mature Bayesian theory: general Bayesian updating
through a loss [@bissiri2016general], Gibbs posterior inference
[@jiang2008gibbs], safe learning-rate selection under misspecification
[@grunwald2012safe], coarsened posteriors for robustness to exact-data
conditioning [@miller2018coarsening], divergence-criteria posterior updating
[@jewson2018divergence], Bayesian misspecification asymptotics
[@kleijn2012misspecification].

The optimization-centric GVI view [@knoblauch2022generalized] and recent
closed-form characterizations [@nguyen2026closedformgvi] connect objectives to
approximating families. Bounded-influence approaches include density-power and
gamma-divergence estimation [@basu1998robust; @fujisawa2008robust;
@ghosh2015robust], robust-divergence variational inference
[@futami2018robustvi], and generalized cross-entropy
[@zhang2018generalized].
Huber and Ronchetti supply the robust-statistics vocabulary of influence and
breakdown [@huber2009robust], used here for boundedness and empirical failure
modes rather than theorem transfer. FedGVI [@mildner2025fedgvi] supplies the
per-agent generalized-Bayes synthesis with client and server divergence choices.

This literature provides decentralized aggregation and theorem-backed results in
its stated settings. Among the sources reviewed here, it does not evaluate
active-inference POMDP belief consensus—the generative-model-bearing,
action-selecting setting where beliefs drive behavior.

The 2025 preprint on convergence rates under prior misspecification
[@mildner2025rates] sharpens the current GVI context: bounded divergences can
support concentration and rates under explicit assumptions, but the result does
not transfer automatically to this repository's finite categorical state space
or to its server-side aggregation heuristic.

The federated-learning robustness literature also supplies important negative
space for this paper. Byzantine-tolerant gradient aggregation [@blanchard2017krum],
geometric-median robust aggregation [@pillutla2022robust], and
divergence-weighted gamma-mean aggregation [@li2022gammafl] attack corrupted
client updates directly. A recent Bayesian robust-aggregation preprint likewise
models unknown client honesty for federated model updates [@karakulev2025bayesian].

Those are close comparators for adversarial federation, but the object being
aggregated differs: they aggregate model-update vectors or posterior measures,
while this paper attacks categorical belief fusion and generalized-Bayes client
updates.

Robust subset-posterior combination is another nearby Bayesian route
[@minsker2017median], but it combines posterior measures across data shards
rather than active-inference belief broadcasts with a shared latent state.

We
use those sources to position the problem, not to import their guarantees into
`robust_aggregate`.

## The specific bridge added here {#sec:related-gap}

The scoped bridge evaluated by this manuscript is generalized-Bayes belief
fusion for active-inference ensembles. Concretely:

**Per-agent bounded-loss updates.** FedGVI-faithful $\beta$/rcce updates carry
the cited bounded-influence result only under its source assumptions
([@eq:beta-loss], [@eq:rcce-loss]).

**Server-side divergence reweighting.** The complementary heuristic has a stated
recovery limit ([@eq:robust-identity]). Its finite empirical behavior does not
borrow theorems from robust federated aggregation.

**Recovery of the standard-Bayes corner.** The KL/NLL client limits recover
Bayes, while the separate zero-robustness server identity returns the project
log-linear pool ([@eq:standard-bayes], [@eq:renyi-limit],
[@eq:robust-identity]). Under the categorical bridge, that pool specializes only
the Eq. 7 message-combination term [@friston2024federated].

**Additional executed studies.** The source simulations motivate mechanism
analogues, not protocol recovery checks. Added studies cover declared
contamination ([@sec:results-robustness]), active movement
([@sec:results-moving]), two- and three-level POMDPs
([@sec:results-hierarchical], [@sec:results-3level]), finite acuity-by-colony
sensitivity, and finite-grid acuity recovery ([@sec:results-sensitivity],
[@sec:results-parameter-recovery]).

**Declared paired analysis.** The configured comparison uses matched-pairs
Wilcoxon tests [@wilcoxon1945individual; @fay2010wilcoxon], Benjamini–Hochberg
FDR adjustment [@benjamini1995controlling], percentile-bootstrap intervals
[@efron1993bootstrap], rank/effect-size caveats [@nakagawa2007effect], and an
observed-effect planning approximation. These additions are project analyses,
not claims about the source paper.

**Objective-backed server control.** `variational_aggregate` descends its stated
free energy ([@eq:agg-free-energy]); a converged fixed point is coordinatewise
stationary. The rule has a raw effective-weight bound and a finite redescending
diagnostic ([@fig:bounded-influence]). Whether an equally defensible objective
has the sharper reverse-KL heuristic as its minimizer remains open.

The contribution is the bridge: an explicit connection between active-inference
belief consensus and robust federated generalized Bayes, with the standard pool
recovered exactly at the corner and the additional studies (the contamination
sweep plus several structural extensions) showing how the bridge behaves beyond
the Friston et al. baseline.
