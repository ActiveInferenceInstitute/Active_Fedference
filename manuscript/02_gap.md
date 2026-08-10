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

Across these five reviewed threads, the missing intersection is specific: the
active-inference sources provide belief fusion but not the contamination analysis
used here, while the robust-Bayes sources provide robust inference but not this
acting-agent categorical consensus. We evaluate the bridge with robust,
generalized-Bayes belief fusion for active-inference ensembles, comprising three
components and one recovery anchor. (i) A per-agent, FedGVI-faithful generalized-Bayes update carrying
bounded-influence robustness through the $\beta$- and rcce-losses
([@eq:beta-loss], [@eq:rcce-loss]). (ii) A complementary server-side
divergence-reweighting *heuristic* that discounts each agent by its divergence
from the emerging consensus ([@eq:robust-identity]). (iii) A conservative
server-side variational rule with a stated aggregation free energy and a
redescending raw effective-weight bound ([@eq:agg-free-energy]). The shared
anchor is recovery of the standard-Bayes client limit and the project
log-linear-pool server identity in their trusting limits
([@sec:formalism-recovery]). Under the qualified bridge of
[@sec:method-aggregation], the latter specializes Friston et al.'s Eq. 7
message-combination term [@friston2024federated], not the complete source
protocol. The algebraic result and machine-precision checks therefore identify
a recovery boundary rather than a replacement. Headline comparisons
use matched paired statistics —
paired Wilcoxon [@wilcoxon1945individual], Benjamini–Hochberg FDR
[@benjamini1995controlling], bootstrap confidence intervals
[@efron1993bootstrap], and observed-effect design-power planning — and are
certified for reproducibility [@peng2011reproducible].

## Guarantee map: three robustness axes {#sec:robustness-axes}

A red-team review surfaced a distinction we carry through the paper rather than
paper over: robustness enters in **three** places, and they do not have the same
theoretical standing.

1. **Client-side (rigorous).** The per-agent generalized-Bayes update, driven by
   a bounded loss inside `generalized_posterior` ([@eq:beta-loss],
   [@eq:rcce-loss]). It is derived from the stated objective [@eq:gen-bayes] and
   provably limits to negative-log-likelihood / Bayes — and hence to the standard
   pool — as the loss parameter goes to zero
   (Corollary \ref{cor:closed-form-bayes} +
   Proposition \ref{prop:robust-loss-recovery}). This axis inherits
   FedGVI's [@mildner2025fedgvi] bounded-influence result only under the source
   theorem's stated loss, model, and contamination assumptions.

2. **Server-side (heuristic).** The divergence-reweighting aggregator
   `robust_aggregate`, which discounts each agent by
   $\exp(-c\,\mathrm{KL}(q_n \,\|\, q))$. Only its *recovery* limit
   is proven — at $c=0$ it equals the standard log-linear pool
   ([@eq:robust-identity], Theorem \ref{thm:belief-sharing-recovery}); it is **not** the closed-form minimizer of
   a FedGVI objective. We present it as a complementary heuristic and never claim
   it inherits the bounded-influence bound.

3. **Server-side (objective-backed, conservative).** The variational aggregator
   `variational_aggregate` applies exact block updates that do not increase the
   stated free energy [@eq:agg-free-energy], recovers the same log-linear pool in
   the trusting limit, and bounds each raw effective weight by its base weight.
   Its honest cost is
   conservatism: it is not the sharp accuracy-maximizing heuristic.

The robustness sweep ([@sec:results-robustness]) reports these axes, and [@sec:limitations-scope]
states which claim rests on which. No figure, table, or sentence in this paper
grants the server-side heuristic the guarantee that belongs to the client-side
update or the variational server objective: the effect-size, confidence-interval,
and power enrichment that decorate the sweep characterize the heuristic's
*behavior*, not a per-agent or variational guarantee. This honesty is the point:
the client-side axis is source-theorem-backed, the sharp server heuristic is
useful but clearly labeled, and the variational server axis is rigorous but
conservative.
