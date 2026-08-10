# Supplement: variational aggregation objective and weight control {#sec:supp-variational}

```{=latex}
\ifcsname proposition\endcsname
\else
\newtheorem{proposition}{Proposition}
\fi
```

This supplement gives the full derivation behind [@sec:method-variational]: the
server-side aggregator `robust_aggregate` is a heuristic, and a single change of
divergence direction turns it into block-coordinate descent on a stated free energy
with a derived, redescending effective-weight update. We work throughout with categorical local posteriors
$q_n(s)$ over the shared latent factor, base weights $w_n > 0$, robustness $c > 0$,
and the consensus $q$ on the probability simplex. Let $\lambda>0$ be the
entropy-weight coefficient, with the current default at
$\lambda={{TEMPERED_ENTROPY_WEIGHT_DEFAULT}}$.

## Why the sharp heuristic is not yet variational {#sec:supp-why-heuristic}

The sharp server rule is empirically strong, but this repository has not
established a variational certificate for it. The heuristic of
[@eq:robust-identity] alternates a *reverse-KL* weight update
$a_n \leftarrow w_n\exp(-c\,\mathrm{KL}(q_n \,\|\, q))$ with the log-linear
consensus $q \leftarrow \mathrm{softmax}(\sum_n a_n \log q_n)$. For the natural
direct objective
$\sum_n a_n \mathrm{KL}(q_n \,\|\, q) +
\tfrac{1}{c}\mathrm{KL_{gen}}(a\,\|\,w)$, the reverse-KL rule is the
$a$-minimizer, but the $q$-minimizer is the *arithmetic* (linear) pool
$q \propto \sum_n a_n q_n$, not the log-linear pool. The executable orientation
witness confirms this finite-simplex mismatch. The following proposition goes
further, while retaining a deliberately narrow scope.

$$
Q(a;s) \;=\; \operatorname{softmax}\!\left(\sum_n a_n\log s_n\right)
$$ {#eq:raw-log-pool-block}

The proposed raw $q$-block is the map in [@eq:raw-log-pool-block].

$$
F(q,a;s,w) \;=\; \sum_n a_n\,\mathrm{KL}(q\,\|\,s_n) + R(a,w) + G(q).
$$ {#eq:separable-server-objective}

\begin{proposition}[Scoped separable raw-log-pool no-go]\label{prop:raw-log-pool-no-go}
For categorical state dimension \(K\geq2\), no objective of the declared displayed form
with \(G\) continuously differentiable,
\(R\) independent of \(q\) and the \(s_n\), and \(G\) independent of \(a\) and
the \(s_n\), has \(Q(a;s)\) as its \(q\)-coordinate minimizer for every interior
raw \(a\in\mathbb{R}_{>0}^N\) and every interior local-posterior collection
\(s\). Consequently, this objective class cannot realize both block maps of the
implemented raw-weight heuristic.
\end{proposition}

*Proof sketch.* Fix any non-uniform interior $q$, a positive scalar
$\alpha$, and one local posterior constructed in [@eq:raw-log-pool-witness-source]:

$$
s_i^{(\alpha)} \;=\;
\frac{q_i^{1/\alpha}}{\sum_j q_j^{1/\alpha}}.
$$ {#eq:raw-log-pool-witness-source}

Then $Q(\alpha;s^{(\alpha)})=q$. Writing $\Pi$ for projection onto the
tangent space of the simplex, first-order stationarity of
[@eq:separable-server-objective] at that same $q$ requires
$\Pi[(\alpha-1)\log q+\nabla G(q)]=0$. The unit-scale construction forces
$\Pi\nabla G(q)=0$; any different positive scale then forces
$\Pi\log q=0$, contradicting the non-uniform choice of $q$. The executable
witness records both exact log-pool identities and the nonzero tangential
contradiction.

A companion witness also blocks the obvious normalized-weight escape within
the same natural data-term class: two interior consensuses yield the same
normalized reverse-KL weights but different forward-KL data-term differences,
so a $q$-independent differentiable $R(a,w)$ cannot satisfy both
simplex-stationarity equations. The implementation itself uses raw effective
weights, so this companion is a scope check rather than a description of the
production update.

| Formal artifact | Executable source | Scope |
| --- | --- | --- |
| Raw-log-pool contradiction | `server_theory.py`: raw witness | The declared raw $q$-block map over every interior input |
| Normalized-weight companion | `server_theory.py`: normalized witness | The normalized reparameterization of the same forward-KL data-term class |
| Typed source report | report `formal_no_go` field | Deterministic witness metadata, separate from the empirical attack grid |

: Formal MAJ-1 witness inventory. These are deterministic finite-simplex proof
artifacts, not empirical estimates; no resampling interval or deployment claim
is implied. {#tbl:server-theory-witness}

Table [@tbl:server-theory-witness] records the deterministic implementation
surfaces that bind this scoped result to the typed analysis report.

The proposition does **not** say that no objective of any kind exists. It does
not exclude nonseparable $q$--$a$ couplings, source-dependent terms,
non-differentiable constructions, or objectives that encode selected fixed
points without reproducing the update blocks for all interior inputs. Thus
[@sec:method-aggregation] retains the heuristic label and claims only the
recovery limit, the scoped negative result above, and conditional empirical
behavior — never a bounded-influence property or an objective-backed status.

## Aggregation free energy and its block minimizers {#sec:supp-derivation}

\begin{definition}[Aggregation free energy]\label{def:aggregation-free-energy}
For \(c>0\), \(\lambda>0\), consensus \(q\), and effective weights
\(a = (a_n)\), \(a_n \ge 0\), define \(F_\lambda(q,a)\) as in
(\ref{eq:agg-free-energy}) with the forward cross-entropy
\(\mathrm{CE}(q, q_n) = -\sum_i q_i \log q_{n,i}\), the consensus entropy \(H(q)\),
and the generalized KL \(\mathrm{KL_{gen}}(a\,\|\,w) = \sum_n[a_n\log(a_n/w_n)-a_n+w_n]\).
\end{definition}

**The $q$-block.** For $\lambda>0$, fixing $a$, the $q$-dependent part of
$F_\lambda$ is
$\sum_n a_n \mathrm{CE}(q,q_n) - \lambda H(q) = \sum_i q_i\big[\lambda\log q_i - \sum_n a_n \log q_{n,i}\big]$.
Adding a Lagrange multiplier for $\sum_i q_i = 1$ and differentiating gives
$\lambda\log q_i + \lambda - \sum_n a_n \log q_{n,i} + \mu = 0$, i.e.

$$
q_i \;\propto\; \exp\!\Big(\tfrac{1}{\lambda}\textstyle\sum_n a_n \log q_{n,i}\Big)
\;=\; \mathrm{softmax}\!\Big(\tfrac{1}{\lambda}\textstyle\sum_n a_n \log q_n\Big)_i,
$$ {#eq:agg-q-min}

the product of the weighted experts — the consensus update of [@eq:agg-updates].
At the default $\lambda={{TEMPERED_ENTROPY_WEIGHT_DEFAULT}}$, the $-H(q)$ term sharpens the weighted geometric mean
into the product-of-experts (the entropy bonus that makes the project's
log-linear pool a product rather than a geometric average).

**The $a$-block.** Fixing $q$, $\partial F/\partial a_n = \mathrm{CE}(q,q_n) + \tfrac{1}{c}\log(a_n/w_n) = 0$, so

$$
a_n \;=\; w_n\,\exp\!\big(-c\,\mathrm{CE}(q, q_n)\big),
$$ {#eq:agg-a-min}

the weight update of [@eq:agg-updates]. Because $\mathrm{CE}(q,q_n) = H(q) + \mathrm{KL}(q\,\|\,q_n)$,
the forward direction $\mathrm{KL}(q\,\|\,q_n)$ — not the heuristic's reverse
$\mathrm{KL}(q_n\,\|\,q)$ — is the one consistent with the consensus update.

Each block update is the *exact* minimizer of its block, so alternating them is
block-coordinate descent: $F$ is non-increasing at every half-step. When the
iterates converge, their fixed point is coordinatewise stationary.

The implementation keeps numerical failure handling outside that theorem. If
finite-precision underflow collapses all effective weights, it records a
fallback event, substitutes the declared base weights to return a valid
probability vector, and does not certify the substituted trajectory as
converged. Such a trace is diagnostic evidence about the solver boundary, not
an instance of the exact block-descent result.

## Formal properties of the conservative server rule {#sec:supp-theorem}

\begin{theorem}[Variational aggregation: descent, recovery, and effective-weight bound]\label{thm:variational-aggregation}
Let \(c>0\) and \(\lambda>0\). Each alternating update in
(\ref{eq:agg-q-min})–(\ref{eq:agg-a-min}) never increases \(F\). Any converged
fixed point is coordinatewise stationary. As \(c\to 0\) the generalized-KL penalty forces \(a_n \to w_n\)
and the consensus is the tempered log-linear pool; at the default
\(\lambda={{TEMPERED_ENTROPY_WEIGHT_DEFAULT}}\) it is
the log-linear pool (\ref{eq:log-linear-pool}) exactly, so the
variational aggregator shares the project log-linear-pool corner of
(\ref{eq:robust-identity}). Under the qualified bridge of
Section~\ref{sec:method-aggregation}, this is only the categorical
message-combination specialization, not the complete source protocol. Finally,
the effective weights satisfy
\(a_n = w_n\exp(-c\,\mathrm{CE}(q,q_n)) \le w_n\) with \(a_n \to 0\) as
\(\mathrm{KL}(q\,\|\,q_n)\to\infty\). Thus the raw effective-weight update is
bounded and redescending relative to the realized consensus. This statement does
not by itself establish a bounded influence function or finite gross-error
sensitivity for the normalized consensus estimator.
\end{theorem}

The objective $F$ is biconvex (each block convex, the coupling
$\sum_n a_n\mathrm{CE}(q,q_n)$ bilinear), so the result concerns monotone block
updates and converged coordinatewise fixed points, not guaranteed convergence to
or certification of a global minimum.

**The effective-weight regime, and why multi-start matters.** The weight bound
$a_n \le w_n$ is unconditional ($\mathrm{CE}(q,q_n)\ge 0$ always). The *collapse*
$a_n \to 0$ is driven by the agent's divergence *from the realized consensus*
$\mathrm{KL}(q\,\|\,q_n)$, and the consensus itself depends on the weights. Because
$F$ is biconvex, this couples into a subtlety an adversarial review of this work
surfaced: a *near-one-hot* saboteur (contamination rate $\to 1$) already captures
the product-of-experts, so a descent seeded *at* that pool stays in a
consensus-capture basin (high $F$) where the saboteur keeps its weight — even
against an honest majority. The repair is to search the stated objective more carefully:
`variational_aggregate` runs **multi-start** block-coordinate descent (the pool,
the uniform belief, and the arithmetic-mean seeds) and returns the lowest-observed-$F$
converged candidate. In the configured colony, the uniform/arithmetic seeds reach a lower-$F$
*vetoing* basin, so the saboteur is suppressed even at the simplex vertex
(pinned qualitatively by the near-vertex multi-start test) — the
{{VARIATIONAL_INFLUENCE_DROP_FACTOR}}$\times$ suppression of
[@fig:bounded-influence] is measured across the swept contamination grid,
whose most extreme point sits just below rate $1$. What remains
fundamental to *every* robust fusion rule, and is not claimed away: with no honest
majority — a colony split with no anchoring plurality — there is no truth to
recover. The observed suppression is conditional on the tested colonies and the
fixed point selected by a finite multi-start heuristic.

[@fig:descent-comparison] makes the capture and the escape concrete on a
near-vertex colony: the single (log-linear-pool) start settles at
$F = {{VARIATIONAL_SINGLE_START_F}}$ (the capture basin, where the saboteur keeps
its weight), while the multi-start descent reaches the genuinely lower
$F = {{VARIATIONAL_MULTI_START_F}}$ vetoing basin — a gap of
{{VARIATIONAL_CAPTURE_GAP}} nats that is exactly the difference between trusting
the natural seed and solving the stated objective properly.

![Variational free-energy descent on a near-vertex adversarial colony. Source relation: original project objective-descent diagnostic; estimand: free energy $F$ in nats by iteration; uncertainty: none for deterministic seeded runs. The figure compares the single (log-linear-pool) start versus the multi-start descent. The x-axis is the block-coordinate iteration; the y-axis is the free energy $F$ in nats. The single-start trajectory settles in the high-$F$ capture basin ($F = {{VARIATIONAL_SINGLE_START_F}}$, the saboteur retains weight); the multi-start trajectory reaches the lower-$F$ vetoing basin ($F = {{VARIATIONAL_MULTI_START_F}}$), a gap of {{VARIATIONAL_CAPTURE_GAP}} nats. Deterministic seeded runs, so no error band.](../output/figures/descent_comparison.png){#fig:descent-comparison width=80%}

## Numerical witnesses for descent and influence bounds {#sec:supp-witnesses}

The analysis pipeline runs `variational_aggregate` at robustness
$c = {{VARIATIONAL_ROBUSTNESS}}$ on a contaminated colony and records the free
energy after each iteration. The descent falls from
$F = {{VARIATIONAL_F_INITIAL}}$ to $F = {{VARIATIONAL_F_FINAL}}$ (a monotone drop
of ${{VARIATIONAL_DELTA_F}}$ over {{VARIATIONAL_ITERATIONS}} iterations,
converged: {{VARIATIONAL_CONVERGED}}); the largest single-step *increase* is
${{VARIATIONAL_MAX_ASCENT_MATH}}$, machine zero — the monotonicity of the theorem,
witnessed numerically and drawn in [@fig:aggregation-descent].

For the effective-weight diagnostic, one agent is drifted from healthy toward a confident-wrong
delta and its normalized influence is read at each drift. Clean, it carries
${{VARIATIONAL_INFLUENCE_CLEAN}}$ of the pool; at the most extreme swept
drift it carries below ${{VARIATIONAL_INFLUENCE_DIVERGED}}$ — a factor of
{{VARIATIONAL_INFLUENCE_DROP_FACTOR}} (computed from the unrounded influences,
not the display-rounded values above) below the fixed
{{VARIATIONAL_NAIVE_INFLUENCE}} the naive pool would still grant it
([@fig:bounded-influence]). This makes the redescending normalized-weight behavior
visible on the tested path; it is not an estimator-level B-robustness proof.

## Tempered aggregation family for the accuracy-guarantee trade {#sec:supp-tempered}

The aggregator of [@sec:supp-derivation] fixes the entropy term at unit weight.
Relaxing that single coefficient generates a one-parameter *tempered* family.
Introduce an entropy weight $\lambda > 0$ — the **inverse temperature is**
$1/\lambda$ — and minimize

$$
F_\lambda(q, a) \;=\; \sum_n a_n\,\mathrm{CE}(q, q_n)
\;-\; \lambda\,H(q)
\;+\; \tfrac{1}{c}\,\mathrm{KL_{gen}}(a \,\|\, w).
$$ {#eq:tempered-family}

Repeating the $q$-block derivation of [@eq:agg-q-min] with the entropy scaled by
$\lambda$ leaves the $a$-block **untouched** and tempers only the consensus update:

$$
q \;\propto\; \exp\!\Big(\tfrac{1}{\lambda}\textstyle\sum_n a_n \log q_n\Big),
\qquad
a_n \;=\; w_n\,\exp\!\big(-c\,\mathrm{CE}(q, q_n)\big).
$$ {#eq:tempered-updates}

The $\lambda\downarrow0$ endpoint is separately implemented as a deterministic
tied-argmax rule; it is not obtained by substituting $\lambda=0$ into
[@eq:tempered-family] or [@eq:tempered-updates].

The weight update is **independent of $\lambda$**: the bound $a_n \le w_n$ with
collapse $a_n \to 0$ as $\mathrm{KL}(q\,\|\,q_n) \to \infty$ is unchanged, so the
**raw effective-weight bound of [@sec:supp-theorem] holds for every** $\lambda > 0$.
At $\lambda = {{TEMPERED_ENTROPY_WEIGHT_DEFAULT}}$ the temperature is unity and
[@eq:tempered-updates] is **identical** to the current axis-3 aggregator
[@eq:agg-updates] — the default is bit-identical, not merely close. The $c \to 0$
recovery of [@sec:supp-theorem] generalizes to the *tempered* log-linear pool
$q \propto \exp(\tfrac{1}{\lambda}\sum_n w_n \log q_n)$; at $\lambda = {{TEMPERED_ENTROPY_WEIGHT_DEFAULT}}$ this is
exactly $\mathrm{softmax}(\sum_n w_n \log q_n)$ — the project's **log-linear
pool** [@eq:log-linear-pool]. Under the shared-support,
posterior-log-potential, and fixed-weight assumptions of
[@sec:method-aggregation], that pool is a categorical specialization of
Friston Eq. 7's message-combination term, not a reconstruction of the complete
source protocol. Positive-temperature members away from that default are
tempered pools, not Friston Eq. 7 itself; the project recovery checks,
including ISC-10, remain project-local.

A small empirical sweep over $\lambda \in \{ {{TEMPERED_LAMBDA_GRID}} \}$ on
{{TEMPERED_N_TRIALS}} contaminated colonies ({{TEMPERED_N_AGENTS}} agents,
{{TEMPERED_N_ADVERSARIAL}} adversarial) asks whether a single $\lambda^{\ast}$
makes the conservative aggregator narrow the gap to the sharp $\mathrm{robust\_aggregate}$
point-accuracy. The closest observed weight is $\lambda^{\ast} = {{TEMPERED_LAMBDA_STAR}}$
with an accuracy gap of {{TEMPERED_LAMBDA_STAR_DIFF}}.
**{{TEMPERED_HONEST_EXIT_SENTENCE}}** If no $\lambda$ closes that gap while
preserving the derived weight update, the result is the conservatism trade-off
of [@sec:limitations], not a defect to hide.
