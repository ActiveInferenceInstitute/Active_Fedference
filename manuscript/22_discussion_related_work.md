## Related work: active inference, federated Bayes, and the scoped bridge {#sec:related-work}

This work sits at the boundary between two research communities with different
objects of inference. Each cited thread contributes a real component; this paper
extends the intersection without claiming that either literature is exhausted.
The positioning is therefore against the reviewed sources and their assumptions,
not against an absolute claim about everything the field has or has not done.

## Pre-modern probability, inverse probability, and collective judgment {#sec:related-historical}

The pre-modern sources in this manuscript are not evidence that early
probability theorists anticipated KL minimization or federated learning. They
serve a narrower role: they show that the paper's recurring problems — expected
uncertain outcomes, inverse inference from effects to causes, utility-sensitive
action, and collective judgment — are old problems now implemented with modern
variational machinery.

The probability-calculus line begins with Pascal and Fermat's correspondence on
the problem of points [@pascal1654probability], Huygens' printed treatment of
reasoning in games of chance [@huygens1657ratiociniis], Montmort's combinatorial
analysis of games [@montmort1708essay], Bernoulli's *Ars conjectandi* and its
law-of-large-numbers logic [@bernoulli1713ars], and de Moivre's systematic
probability textbook [@demoivre1718doctrine]. For this paper, their relevance is
not that they contain active inference, but that they make expectation and
uncertain evidence objects of calculation.

The inverse-probability line is closer to the manuscript's formal spine. Bayes
and Price frame the problem of inferring an unknown chance from observed events
[@bayes1763essay], and Laplace generalizes the probability of causes given
events [@laplace1774memoire]. Read beside modern active inference, these sources
support the vocabulary of generative-model inversion: data are effects, latent
states or parameters are causes, and a posterior belief reconciles prior
commitments with observed evidence. That is a conceptual bridge only; the
actual identity with the FedGVI recovery corner is the modern result of
[@sec:method-aggregation] and [@sec:formalism].

Decision and aggregation enter through Daniel Bernoulli's expected-utility
treatment of risk [@bernoulli1738mensura] and the voting-theoretic work of Borda
and Condorcet [@borda1784elections; @condorcet1785essai]. They help name the
problem this paper revisits in a categorical active-inference colony: local
judgments must be combined, and the chosen aggregation rule encodes assumptions
about competence, independence, weights, and stakes. Those assumptions are
exactly what the modern log-pool and robustness literature make explicit.

## Active inference: generative agents, EFE, and colonies {#sec:related-aif}

The free-energy principle [@friston2010free] and its discrete state-space
process theory [@friston2017active] gave the field a unified account of
perception and action: hand-built generative models with $A$/$B$/$C$/$D$
matrices, variational free energy for inference, and expected free energy for
principled action selection. The discrete state-space synthesis
[@dacosta2020active] and toolboxes turned that account into scalable machinery —
the pymdp discrete-state library [@heins2022pymdp] and the RxInfer reactive
message-passing engine [@bagaev2023rxinfer]. *Yes:* this substrate is exactly
what we reimplement, and our EFE decomposition into risk and ambiguity
([@eq:efe-decomposition], [@eq:efe-identity], [@fig:efe-decomp]) is the
community's own action-selection objective. *And:* we add a robustness knob to
its belief-fusion step without leaving the framework.

The fusion operator itself is not new as a mathematical object. Under the
explicit categorical posterior-log-potential and fixed-weight bridge of
[@sec:method-aggregation], the message-combination term of Friston's
belief-sharing equation is represented by a logarithmic opinion pool
[@genest1986combining; @genest1986externally] and a product-of-experts consensus
[@hinton2002products]. That limited representation is not a statement that the
complete source protocol is a log pool. Abbas' KL view of linear and log-linear
pools makes the same point in scoring-rule language: log-pooling can
be justified as a KL aggregation rule for expert distributions, not as a
contamination-robust estimator [@abbas2009kullback]. The Bayesian Committee
Machine adds the distributed-learning analogue: independent estimators trained
on data subsets can be combined by a product-style Bayesian rule, with
assumptions about conditional independence and prior accounting made explicit
[@tresp2000bayesian]. Fully Bayesian aggregation gives the social-choice
counterpart: geometric pooling of beliefs is singled out by dynamic Bayesian
rationality conditions, not by robustness to contaminated reports
[@dietrich2021fully]. That classical literature is useful precisely because it
names the hidden commitments — shared support, weight choice, prior accounting,
and external-Bayes coherence — that are easy to overlook when the same operation
appears as a colony update.

A parallel thread studies how active-inference agents coordinate as ensembles —
collective behavior and surprise minimization across a colony
[@heins2023collective], epistemic communities [@albarracin2022epistemic], and
collective intelligence [@kaufmann2021collective]. The belief-sharing thread
[@friston2024federated] is one member of this family: agents reach consensus by
exact-Bayes fusion of one another's beliefs, and the colony belief-sharing
scenario we implement as a reduced categorical standard-Bayes-limit analogue
([@eq:belief-round], [@sec:results-belief_sharing]) is related to its worked
example. Structure learning rounds out the
picture: active inference with Bayesian optimal design selects and reduces models
within the same frame [@smith2020active], and post-hoc Bayesian model reduction
[@friston2011post] is the engine behind our emergence study
([@eq:bmr-deltaf], [@sec:results-emergence]). The cited active-inference line has
rich generative models, principled action, and ensembles that coordinate by
sharing beliefs. In the sources reviewed here, the belief-fusion rule is not
systematically characterized under explicit contamination or intentionally wrong
broadcasts, nor connected to the robustness theory used here. Fusion is treated
as exact-Bayes and trusting in that scoped comparison.

Friston et al. [-@friston2024federated] crystallized these ideas into three
worked simulations: (1) communicating-colony free-energy convergence, (2) Dirichlet
language acquisition, and (3) Bayesian model reduction structure emergence. We
use reduced categorical standard-Bayes-limit analogues of all three mechanisms
([@sec:results-recovery]) under the declared protocol before building the robust
extension. This is a mechanism-level comparison, not an exact source-protocol
or figure reproduction.

## Robust and federated Bayes outside active inference {#sec:related-fl}

Federated learning aggregates models trained on decentralized data
[@mcmahan2017communication], and the probabilistic-federation line recasts that
aggregation as variational inference — partitioned variational inference
[@ashman2022partitioned] and its federated predecessor [@bui2018partitioned].
This is a different use of the word *federated* from Friston et al.'s federated
inference: Friston federates hidden-state beliefs among agents sharing a world
model, while machine-learning federated learning usually federates parameter or
predictive-model updates over decentralized datasets. The bridge claimed here is
therefore algebraic and variational — a shared normalized product/log-pool
operator at the KL/NLL recovery corner — not a claim that either source paper
already solved the other's problem.

Robustness, meanwhile, has a mature Bayesian theory: general Bayesian updating
through a loss [@bissiri2016general], Gibbs posterior inference
[@jiang2008gibbs], safe learning-rate selection under misspecification
[@grunwald2012safe], coarsened posteriors for robustness to exact-data
conditioning [@miller2018coarsening], divergence-criteria posterior updating
[@jewson2018divergence], Bayesian misspecification asymptotics
[@kleijn2012misspecification], the optimization-centric generalized variational
inference view [@knoblauch2022generalized], recent closed-form characterizations of
unrestricted generalized variational objectives [@nguyen2026closedformgvi], and
the bounded-influence losses that make updates robust — density-power and
gamma-divergence estimation [@basu1998robust; @fujisawa2008robust;
@ghosh2015robust], robust-divergence variational inference
[@futami2018robustvi], and generalized cross-entropy [@zhang2018generalized].
Huber and Ronchetti supply the robust-statistics vocabulary of influence and
breakdown [@huber2009robust], which we use as vocabulary for boundedness and
empirical failure modes rather than as a theorem transfer. FedGVI
[@mildner2025fedgvi] is the synthesis we use per agent: a robust generalized-
Bayes objective with client and server divergence choices. This community
provides decentralized aggregation and theorem-backed results in its stated
settings. The cited papers do not evaluate active-inference POMDP belief
consensus — the generative-model-bearing, action-selecting setting where beliefs
drive behavior.

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
updates. Robust subset-posterior combination is another nearby Bayesian route
[@minsker2017median], but it combines posterior measures across data shards
rather than active-inference belief broadcasts with a shared latent state. We
use those sources to position the problem, not to import their guarantees into
`robust_aggregate`.

## The specific bridge added here {#sec:related-gap}

The gap this manuscript fills is robust, generalized-Bayes belief fusion for
active-inference ensembles. Concretely:

- **Per-agent bounded-influence updates.** FedGVI-faithful $\beta$/rcce updates
  that carry provable bounded influence ([@eq:beta-loss], [@eq:rcce-loss]).
- **Server-side divergence-reweighting heuristic.** A complementary aggregation
  heuristic ([@eq:robust-identity]) with a stated recovery limit, positioned
  alongside robust federated aggregation work without borrowing its theorems.
- **Provable recovery of the standard-Bayes client limit.** The client
  KL/NLL loss limits recover Bayes, and the separate zero-robustness server
  identity returns the project log-linear pool ([@eq:standard-bayes],
  [@eq:renyi-limit], [@eq:robust-identity]). Under the stated categorical
  bridge, that pool specializes only the Eq. 7 message-combination term
  [@friston2024federated].
- **Additional executed studies beyond the Friston et al. (2024) baseline.** The
  three Friston simulations motivate source-mechanism analogues rather than
  serving as source-protocol recovery checks; beyond them
  we contribute: (4) a contamination sweep in which a fraction of colony members
  are adversarial or misspecified ([@sec:results-robustness]); (5) a moving-world
  scenario with active expected-free-energy movement ([@sec:results-moving]);
  (6) a two-level hierarchical POMDP ([@sec:results-hierarchical]); (7) a three-level
  hierarchical POMDP ([@sec:results-3level]); (8) a sensitivity sweep over
  acuity $\times$ colony size ([@sec:results-sensitivity]); and (9) finite-grid
  parameter recovery for the declared acuity family
  ([@sec:results-parameter-recovery]).
- **Rigorous statistics.** Every verdict is produced by matched-pairs Wilcoxon
  signed-rank tests [@wilcoxon1945individual; @fay2010wilcoxon] deflated with
  Benjamini–Hochberg FDR [@benjamini1995controlling], reported with bootstrap
  confidence intervals [@efron1993bootstrap], rank/effect-size caveats
  [@nakagawa2007effect], and an observed-effect design-power approximation —
  none of which appear in Friston et al. (2024).
- **An objective-backed server-side aggregator with redescending weights.** The
  `variational_aggregate` rule descends a stated aggregation free energy
  ([@eq:agg-free-energy]) monotonically; any converged fixed point is
  coordinatewise stationary. The rule carries a
  proven raw effective-weight bound and empirical redescending response
  ([@fig:bounded-influence]), while leaving open
  whether the sharper reverse-KL server heuristic is itself the closed-form
  minimizer of an equally defensible objective.

The contribution is the bridge: an explicit connection between active-inference
belief consensus and robust federated generalized Bayes, with the standard pool
recovered exactly at the corner and the additional studies (the contamination
sweep plus several structural extensions) showing how the bridge behaves beyond
the Friston et al. baseline.
