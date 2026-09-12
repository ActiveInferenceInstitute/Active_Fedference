# Contributions and evidence boundaries {#sec:contributions}

We make ten contributions, each paired with a theorem, figure, table, or
generated token and with an explicit boundary on what the evidence establishes.

They fall into four groups: the first two build the core and its recovery
contract; the next two report and statistically qualify the contaminated-consensus
result; the fifth and sixth make the robustness axes explicit and add the
objective-backed server rule; and the remaining four are scoped extensions —
parameter recovery, the tempered aggregation family, an aggregation-API transfer,
and a disjoint-observation communication test.

## 1. A discrete-categorical FedGVI core

A typed, deterministic, pure-NumPy/SciPy reimplementation of the FedGVI
[@mildner2025fedgvi] generalized-Bayes primitives — divergences, bounded robust
losses, the generalized posterior, the cavity/factor algebra, and robust
aggregation — in the discrete-categorical setting that active inference
[@dacosta2020active] uses.

The objective ([@eq:gen-bayes]) and its closed-form tempered-softmax solution
are stated in Definition \ref{def:generalized-bayes} and tested by the
recovery-limit probes. The whole core is zero-mock and reproducible
[@peng2011reproducible].

## 2. A recovery-tested connection between categorical pooling and robust Bayes

A recovery certificate shows that the client KL/negative-log-likelihood loss
limits recover Bayes and the server's zero-robustness branch recovers the
project log-linear pool.

Under the explicit shared-support, posterior-log-potential, and fixed-weight
assumptions in [@sec:method-aggregation], that pool specializes Friston et al.'s
Eq. 7 message-combination term [@friston2024federated], not the complete source
protocol.

Three recovery limits are pinned to bit-level residuals. The bounded losses
recover NLL/Bayes (Corollary \ref{cor:closed-form-bayes} + Proposition
\ref{prop:robust-loss-recovery}; $\le 0$ and
$\le 0$ maximum residual,
[@eq:standard-bayes]).

The Rényi divergence recovers KL (Lemma \ref{lem:renyi-kl-limit};
$\le 0$ residual, [@eq:renyi-limit]). The
server-side reweighting pool recovers the naive pool in its trusting limit
(Theorem \ref{thm:belief-sharing-recovery};
$\le 0$ residual, [@eq:robust-identity]).

Robustness is thereby a tested recovery-limit extension, not a replacement.

## 3. End-to-end evaluation of robust federated active inference

Three worked categorical source-mechanism analogues cover communicating
colonies reaching lower free energy ([@sec:results-belief_sharing]), Dirichlet
language acquisition ([@sec:results-language]), and a configured Bayesian-model-
reduction pruning sign control [@friston2011post] ([@sec:results-emergence]).

A contaminated-sentinel robustness sweep ([@sec:results-robustness]) shows the
naive pool degrading while at least one server-side robust member clears the
configured threshold at the most severe swept rate.

## 4. A statistically qualified server-side contrast

The "robust beats naive" conclusion for the declared contamination rate is
produced only by a matched-pairs Wilcoxon signed-rank test
[@wilcoxon1945individual]. Its multiplicity correction uses
Benjamini–Hochberg FDR [@benjamini1995controlling] across the divergence family.
We report the contrast with bootstrap confidence intervals
[@efron1993bootstrap] and observed-effect design-power planning.

Across 960 paired trials, the headline display method
(RKL; tied set: RKL, AR, beta, rcce) reaches
accuracy 0.9829 against the naive pool's
0.9021 at the verdict rate.

At that rate, $q = 1.11 \times 10^{-158}$, and the
rank-biserial-derived $d$-equivalent is saturated (r=+1). The
predeclared selection rule is largest positive rank-biserial effect_size; stable method order tie-break; the method with
the largest paired mean difference is
AR. Every headline number is a generated
token.

## 5. An explicit accounting of the robustness axes

The map in [@sec:robustness-axes] separates the client-side per-agent update,
which inherits FedGVI's bounded-influence result under its stated source
assumptions, from both server rules.

The sharp server-side reweighting heuristic has a recovery limit as its positive
formal property and a scoped no-go result for its declared separable objective
class.

The conservative variational server rule is objective-backed and has a
redescending effective-weight update, but is not the accuracy-maximizer.
Downstream users are told exactly which result is theoretically backed and
which is a labeled heuristic.

## 6. An objective-backed server aggregator with redescending weights

We derive an aggregation free energy [@eq:agg-free-energy] whose exact block
updates define the `variational_aggregate` rule ([@sec:method-variational],
[@sec:supp-variational]).

Each exact block update monotonically decreases a stated objective, and a
converged fixed point is coordinatewise stationary. The implementation keeps
the lowest observed objective among converged configured starts, or reports the
best unfinished trace as non-converged.

The rule recovers the standard log-linear pool in the trusting limit
([@eq:robust-identity]) and — unlike the sharp heuristic — carries a proven raw
effective-weight bound ([@fig:aggregation-descent], [@fig:bounded-influence]).

The honest cost is conservatism: it is a maximum-entropy-biased consensus and
trades peak point-accuracy for that control, so it complements rather than
replaces the sharp heuristic of contribution 5.

## 7. Executed finite-grid acuity-recovery experiment

At each value in the 0.60, 0.70, 0.80, 0.90 acuity grid, the study
generates 200 synthetic observations in each of
960 trials. It selects acuity by marginal-likelihood
grid search over the declared finite grid.

The observed mean absolute error is 0.0232 with
$R^2 = 0.9999$ ([@fig:parameter-recovery]).
Acuity-by-colony-size behavior belongs to the separate sensitivity study; it is
not a parameter-recovery result.

## 8. The tempered aggregation family

A one-parameter $\lambda>0$ generalization of the variational aggregate
([@sec:supp-tempered]) is the objective

$$
\begin{aligned}
F_\lambda(q,a)
&= \sum_n a_n\,\mathrm{CE}(q,q_n)-\lambda H(q)\\
&\quad +\frac{1}{c}\,\mathrm{KL}_{\mathrm{gen}}(a\Vert w).
\end{aligned}
$$ {#eq:contribution-tempered-family}

In [@eq:contribution-tempered-family], at
$\lambda = 1.0$, the temperature is unity
and the objective reduces to the standard variational aggregate bit-for-bit.
Lower $\lambda$ sharpens the variational $q$-block toward a maximizing state;
it does not algebraically recover `robust_aggregate`.

The raw effective-weight update and its bound are preserved for all $\lambda$.
The full derivation is in [@sec:supp-tempered].

## 9. An aggregation-API transfer demonstration

The same `robust_aggregate` API that governs the POMDP studies is exercised
unchanged with one deterministic MLP trained with the density-power $\beta$-loss
(16 hidden units, $\beta=0.5$; generalized variational
inference with a point-mass variational family) as the per-client model.

This supports portability of the server API to one additional model class
([@sec:results-baseline]) when the optional `torch` extra is installed
([@sec:methods-software]). Without it, the MLP run is skipped and its tokens
render accordingly.

## 10. Communication contrast under disjoint observations

A multi-agent extension ([@sec:results-disjoint-fov]) in which
3 agents each observe a 2-slot disjoint window
shows that belief sharing materially improves over isolated-agent accuracy in
the declared configuration. Isolated agents clear the 0.167
chance baseline but stay far below the communicating consensus, which itself
remains well short of full accuracy.

Across 128 seeds, isolated accuracy is 0.326 versus
communicating 0.493, a reproducible margin under the declared
matched-seed comparison (Wilcoxon $p = 9.35 \times 10^{-23}$). This is evidence
for the configured disjoint-observation protocol, not a universal communication
theorem.

The remainder of the paper proceeds as follows. [@sec:methods] develops the
FedGVI core and the recovery limits; [@sec:formalism] states the numbered
recovery theorems and the expected-free-energy identity; [@sec:methods-experimental-design]
fixes the configuration; [@sec:results] reports the 9 studies,
beginning with the recovery checks ([@sec:results-recovery]).

[@sec:discussion] and
[@sec:conclusion] synthesize; and [@sec:reproducibility] and [@sec:limitations-scope] document
determinism, scope, limitations, and the standing of each robustness axis.
