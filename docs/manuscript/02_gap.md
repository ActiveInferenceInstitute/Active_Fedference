# Research gap and claim boundary {#sec:gap}

The two communities of [@sec:introduction] provide substantial pieces of the
problem, but the cited threads do not answer the same question. The gap is not a
claim that either field is incomplete; it is the untested intersection between
active-inference belief sharing and robust generalized Bayes. This section
describes that intersection thread by thread, states the scoped bridge evaluated
here, and records the evidence boundary that travels with it.

## Five reviewed threads and their open intersection {#sec:gap-threads}

Five threads in the reviewed literature run toward the gap but do not, in the
sources cited here, cross it — three from active inference and two from robust
Bayes.

**Thread 1 — Generative modeling and action** (active inference). Discrete
active inference is a mature, synthesized formalism with standardized tensors and
an expected-free-energy action rule [@friston2010free; @dacosta2020active], and it
is executable at scale through pymdp [@heins2022pymdp] and RxInfer
[@bagaev2023rxinfer]. *Boundary:* these sources do not by themselves equip the
active-inference inference step with the contamination mechanism evaluated here.

**Thread 2 — Collective belief coordination** (active inference). Colonies
coordinate by sharing beliefs and minimizing collective surprise, reproducing
flocking [@heins2023collective], epistemic-community formation
[@albarracin2022epistemic], and collective intelligence [@kaufmann2021collective].
*Boundary:* the cited collective-belief studies treat the broadcast posteriors as
trusted and do not evaluate the robustness of fusion to misspecified or
intentionally wrong members.

**Thread 3 — Belief sharing and structure growth** (active inference).
Federated belief sharing fuses posteriors into a hive-mind consensus
[@friston2024federated], and post-hoc Bayesian model reduction [@friston2011post]
together with optimal-design-flavored model selection [@smith2020active] grow and
prune the generative model. *Boundary:* in the cited belief-sharing line, fusion
is exact-Bayes and trusting; its connection to the robustness theory developed
for federated learning is not evaluated.

**Thread 4 — Federated and partitioned inference** (robust Bayes).
Federated learning aggregates decentralized models [@mcmahan2017communication],
and partitioned variational inference gives the factor-algebra framework that
unifies federated and continual learning
[@bui2018partitioned; @ashman2022partitioned]. *Boundary:* the cited examples
target parameter or predictive-model factors rather than the
generative-model-bearing, action-selecting POMDP belief consensus used here.

**Thread 5 — Generalized and robust Bayes** (robust Bayes).
Generalized
Bayesian updating replaces the likelihood–KL pair with a loss–divergence pair
[@bissiri2016general; @knoblauch2022generalized]; bounded losses — the
density-power $\beta$-divergence [@basu1998robust] and generalized cross-entropy
[@zhang2018generalized] — deliver bounded influence; and FedGVI
[@mildner2025fedgvi] federates the robust objective with provable guarantees.

*Boundary:* the cited robust-Bayes apparatus does not evaluate active-inference
POMDP belief consensus. Its behavior in the discrete categorical regime of the
worked belief-sharing example [@friston2024federated] is the scoped setting
evaluated here.

## The belief-fusion bridge evaluated here {#sec:gap-bridge}

Across these reviewed threads, the missing intersection is specific. The cited
active-inference sources provide belief fusion but not the contamination analysis
used here. The cited robust-Bayes sources provide robust inference but not this
acting-agent categorical consensus. We evaluate a bridge comprising three
distinct robustness axes and one recovery anchor.

The client axis is a FedGVI-faithful generalized-Bayes update using the
$\beta$- and rcce-losses ([@eq:beta-loss], [@eq:rcce-loss]). The heuristic
server axis reweights each agent by divergence from the current consensus
([@eq:robust-identity]). The objective-backed server axis is a conservative
variational rule with a stated aggregation free energy and raw effective-weight
bound ([@eq:agg-free-energy]).

The shared anchor recovers the standard-Bayes client limit and project
log-linear-pool server identity ([@sec:formalism-recovery]). Under the qualified
bridge of [@sec:method-aggregation], the latter specializes Friston et al.'s
Eq. 7 message-combination term [@friston2024federated], not the complete source
protocol. The algebra and executable residuals identify a recovery boundary,
not an equivalence of literatures.

Declared comparisons use paired Wilcoxon statistics
[@wilcoxon1945individual], Benjamini–Hochberg FDR
[@benjamini1995controlling], percentile-bootstrap intervals
[@efron1993bootstrap], and observed-effect design-power planning. Source-bound
reports make those analyses reproducible [@peng2011reproducible]; they do not
promote a conditional simulation into a theorem.

## Guarantee map: three robustness axes {#sec:robustness-axes}

A red-team review surfaced the paper's authoritative robustness taxonomy.
Robustness enters in **three** places, with different claim owners, evidence
classes, and permitted interpretations.

### Client-side: source-theorem-backed

The per-agent generalized-Bayes update uses a bounded loss inside
`generalized_posterior` ([@eq:beta-loss], [@eq:rcce-loss]). It is derived from
[@eq:gen-bayes] and limits to NLL/Bayes as the loss parameter approaches zero
(Corollary \ref{cor:closed-form-bayes}; Proposition
\ref{prop:robust-loss-recovery}). FedGVI's bounded-influence result transfers
only under the source theorem's loss, model, and contamination assumptions
[@mildner2025fedgvi].

### Server-side: heuristic

`robust_aggregate` discounts an agent by
$\exp(-c\,\mathrm{KL}(q_n \,\|\, q))$. Its proved positive property is the
recovery limit: at $c=0$, it equals the standard log-linear pool
([@eq:robust-identity], Theorem \ref{thm:belief-sharing-recovery}). It is not a
closed-form FedGVI-objective minimizer and does not inherit the client-side
bounded-influence result.

### Server-side: objective-backed and conservative

`variational_aggregate` applies exact block updates that do not increase the
stated free energy [@eq:agg-free-energy]. It recovers the log-linear pool in the
trusting limit and bounds each raw effective weight by its base weight. Its
declared tradeoff is a conservative consensus; the objective-backed property
does not imply peak-accuracy dominance.

The robustness results ([@sec:results-robustness]) apply this map, and
[@sec:limitations-scope] states the complete boundary. Effect sizes, intervals,
and planning quantities characterize the server heuristic's conditional
behavior; they do not transfer the client theorem or variational objective to
that heuristic. Later sections refer back to this taxonomy rather than redefine
it.
