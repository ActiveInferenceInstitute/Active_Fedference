# Discussion: what the evidence supports {#sec:discussion}

The 9 studies probe distinct parts of one categorical,
factor-based framework rather than repeating a single benchmark. Their common
result is narrow but useful: the client construction contains its
standard-Bayes limit and the server has a named project log-linear-pool recovery
corner, and behavior away from those limits can be measured under declared
contamination, sampling, and model assumptions. The recovery identities are
formal; the performance results are conditional simulation evidence.

## The recovery limit is the formal anchor {#sec:discussion-limit}

The coherence of the framework rests on the KL/NLL client limits and the
zero-robustness project-pool identity
([@eq:robust-identity], [@eq:standard-bayes]). This is an identity of the stated
categorical implementation, not an asymptotic claim about every generalized
Bayesian model or a reconstruction of the complete source protocol. The server
aggregator returns the log-linear pool ([@eq:log-linear-pool]) with maximum
deviation 0;
the generalized posterior returns the closed-form Bayes update with deviation
5.55e-17; and the Rényi divergence, $\beta$-loss, and
robust cross-entropy recover their KL/NLL limits with residuals
0, 0, and
0. Theorem
\ref{thm:belief-sharing-recovery}, Corollary \ref{cor:closed-form-bayes},
Lemma \ref{lem:renyi-kl-limit}, and Proposition
\ref{prop:robust-loss-recovery} state the formal limits; the recovery checks in
[@sec:results-recovery] are the executable falsification harness.

## What the study suite jointly shows {#sec:discussion-joint}

Read together, the suite separates into two kinds of result: identities that
hold exactly at the standard corner, and performance contrasts that are explicitly
conditional on the operating regime. The distinction is the point of the joint
reading — it says which claims travel and which are tied to the declared world.

Studies 1–3 sit at the standard corner. In the reduced categorical
source-mechanism analogue protocol,
belief sharing lowers mean variational free energy by
$\Delta \bar F = 3.3109$ nats
([@sec:results-belief_sharing], [@fig:free-energy]); Dirichlet language learning
reduces KL from 3.4231 to 0.0027 nats
([@sec:results-language], [@fig:language-kl]); and Bayesian model reduction
accepts the redundant-pruning candidate while rejecting the supported-column
control ([@sec:results-emergence], [@fig:emergence-bmr]). These are
mechanistic checks, not evidence that the implementation matches every detail
of the source simulations.

Study 4 steps away from the corner by holding the hidden state and attack target
fixed while redrawing matched contaminated colonies. The standard pool
degrades across the declared rate grid. The server-side heuristic is not
uniformly better: its robust members are similar to or slightly below the
standard pool at low contamination, then the selected pooled display member separates in its favor
under severe contamination. At the largest swept rate,
AR reaches 0.9880 against
0.6928 for the standard pool
([@tbl:robustness_sweep]; the sweep figure [@fig:robustness-sweep] plots the
separate matched-trial gallery colony, whose largest-rate separation is
smaller). This regime dependence is a
result, not a nuisance to be hidden: robustness can cost efficiency when the
attack is weak and pay off when the declared contamination is severe.

The structural extension studies (Studies 5–9) sharpen the conditional half of
that split rather than adding further corner checks, and they are reported with
their negative contrasts intact. Communication is not uniformly beneficial: the
cross-study summary ([@fig:cross-study-summary]) reports each study's federation
benefit in its native units with seed-level intervals — several studies clearly
positive, the moving-world EFE and two-level hierarchical studies approximately
zero — so pooling helps when views are
complementary and can be unnecessary or mildly costly when the agents already
agree. Adding hierarchical depth is held to the same standard — the two-level
stack does not beat the flat baseline on location accuracy (the paired gap is
a small, statistically reliable cost)
([@sec:results-hierarchical]), earning its place only by additionally resolving
the context latent above chance. The joint lesson is therefore not that
federation or depth is always worth its cost, but that the suite measures the
regimes in which each one is.

## What this simulation identifies—and what it does not {#sec:discussion-identifiability}

The primary robustness estimand is the matched-trial mean difference in consensus
accuracy conditional on the seeded true state and attack geometry. The
960 paired trials quantify Monte Carlo variation for that
estimand; they do not average over hidden states, attack targets, adaptive
adversaries, or real deployments. The cross-study layer reduces its matched
trials within seed before seed-level summaries, so clients and within-seed trials
are not silently counted as independent replicates. This follows simulation
study guidance to declare the estimand and Monte Carlo unit explicitly
[@morris2019simulation; @koehler2009mcse], and the bootstrap interval follows
the declared resampling unit rather than treating nested observations as a flat
sample [@loy2021lmeresampler].

The consequence is a more informative claim boundary. The sweep supports a
conditional statement about this contamination mechanism and these categorical
beliefs; it does not establish universal Byzantine tolerance, calibration, or
truth recovery. The result also does not identify a single universally best robustness
parameter independent of the operating regime: the standard pool is preferable
at the low-contamination cells in this run, while at least one robust member has
a BH-rejected positive contrast in the declared high-contamination verdict; the
tied display set and deterministic tie-break are reported separately. Those are
precisely the conditions a future
adaptive or deployment study must vary.

## The robustness verdict is conditional and statistically qualified {#sec:discussion-verdict}

The headline comparison is computed by the statistics module
([@sec:results-verdict]), not typed into the prose. Across
960 matched trial replicates nested within the fixed seeded world
at contamination rate 0.800, each robust server member is compared with the standard
pool by a Wilcoxon signed-rank test [@wilcoxon1945individual] and the declared
family of p-values is adjusted by Benjamini–Hochberg FDR
[@benjamini1995controlling]. A method wins only when the adjusted null is
rejected and its effect is positive, here at $q = 1.11 \times 10^{-158}$ with
effect size 1.0000 (large). The
observed-effect power and prospective sample-size calculation are planning
quantities, not independent confirmation of the result.

The paired design is appropriate for the controlled comparison because each
condition shares the seed, true state, and attack geometry. Its interpretation
remains limited: the signed-rank test concerns the distribution of paired
differences, not equality of raw means, and BH controls expected false-discovery
proportion within the declared family rather than the probability of any false
positive. The confidence intervals are percentile bootstrap intervals over the
matched-trial unit. Seed-level review-grid results are reported separately and
do not treat the primary trial count as a population of independent worlds.
These qualifications make the verdict narrower, but also
make it reproducible and falsifiable.

## Three robustness axes remain separate {#sec:discussion-axes}

The unifying narrative would be dishonest if it let the aggregation heuristic
inherit the guarantees of the client update or variational server objective.
The **client-side** $\beta$/rcce generalized-Bayes update is the FedGVI-faithful
axis [@mildner2025fedgvi]: it is derived from the generalized-Bayes objective
([@eq:gen-bayes]), limits to NLL/Bayes as its loss parameter tends to zero, and
carries the loss-specific bounded-influence result
([@eq:beta-loss], [@eq:rcce-loss]). The **server-side** `robust_aggregate`
divergence-reweighting is a complementary heuristic whose proven property is
the recovery limit only ([@eq:robust-identity]). The **variational server** axis
is objective-backed and supplies a proven raw effective-weight bound, with
conservative accuracy behavior. No figure, statistic, or sentence transfers a
client guarantee to the heuristic or a variational weight bound to the
heuristic's accuracy verdict; [@sec:limitations] states the boundary in full.

## Accuracy and effective-weight control can be traded explicitly {#sec:discussion-tempered}

The $F_\lambda$ family ([@sec:supp-tempered]) supplies empirical grid evidence
that accuracy and effective-weight control need not be a fixed binary choice
within the tested objective family. At
$\lambda = 1.0$ the implementation recovers the
current variational objective bit-for-bit; lower explored temperatures sharpen
the consensus toward the heuristic while preserving the stated raw-weight bound
under its assumptions. This is not a derivation of the heuristic from an
objective. The open problem is to identify an objective whose minimizer is both
competitive across contamination regimes and accompanied by a theorem that
survives beyond the present categorical construction.

## Why the boundary matters downstream {#sec:discussion-downstream}

The practical lesson is not that every robust rule should replace belief
sharing. It is that a multi-agent active-inference system can expose separate
controls for client updating, server reweighting, and objective-backed consensus,
while retaining a tested route back to the ordinary pool. That separation tells a
builder what can be promised: exact recovery at the named corner, conditional
empirical behavior under the declared attack, and no silent transfer of a
client-side or variational theorem to a different server heuristic. The result is
a usable research contract for extending belief-sharing systems without
confusing a simulation win with a general robustness guarantee.
