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
with a derived, redescending effective-weight update.

We work throughout with categorical local posteriors
$q_n(s)$ over the shared latent factor, base weights $w_n > 0$, robustness $c > 0$,
and the consensus $q$ on the probability simplex. Let $\lambda>0$ be the
entropy-weight coefficient, with the current default at
$\lambda=1.0$.

## Why the sharp heuristic is not yet variational {#sec:supp-why-heuristic}

The sharp server rule is empirically strong, but this repository has not
established a variational certificate for it. The heuristic of
[@eq:robust-identity] alternates a *reverse-KL* weight update
$a_n \leftarrow w_n\exp(-c\,\mathrm{KL}(q_n \,\|\, q))$.

It couples that update with the log-linear
consensus $q \leftarrow \mathrm{softmax}(\sum_n a_n \log q_n)$. For the natural
direct objective
$\sum_n a_n \mathrm{KL}(q_n \,\|\, q) +
\tfrac{1}{c}\mathrm{KL_{gen}}(a\,\|\,w)$, the reverse-KL rule is the
$a$-minimizer.

The $q$-minimizer is the *arithmetic* (linear) pool
$q \propto \sum_n a_n q_n$, not the log-linear pool. The executable orientation
witness confirms this finite-simplex mismatch. The following proposition goes
further, while retaining a deliberately narrow scope.

$$
Q(a;s) \;=\; \operatorname{softmax}\!\left(\sum_n a_n\log s_n\right)
$$ {#eq:raw-log-pool-block}

The proposed raw $q$-block is the map in [@eq:raw-log-pool-block].

$$
\begin{aligned}
F(q,a;s,w)
&= \sum_n a_n\,\mathrm{KL}(q\,\|\,s_n)\\
&\quad + R(a,w) + G(q).
\end{aligned}
$$ {#eq:separable-server-objective}

\begin{proposition}[Scoped separable raw-log-pool no-go]\label{prop:raw-log-pool-no-go}
For \(K\geq2\), no objective in (\ref{eq:separable-server-objective}) satisfying
the declared separability and differentiability conditions has \(Q(a;s)\) as
its \(q\)-coordinate minimizer for every positive interior \(a\) and \(s\).
\end{proposition}

Precisely, $G$ is continuously differentiable and independent of $a,s$,
while $R$ is independent of $q,s$. Under those conditions, this objective
class cannot realize both block maps of the implemented raw-weight heuristic.

*Proof sketch.* Fix any non-uniform interior $q$, a positive scalar
$\alpha$, and one local posterior constructed in [@eq:raw-log-pool-witness-source]:

$$
s_i^{(\alpha)} \;=\;
\frac{q_i^{1/\alpha}}{\sum_j q_j^{1/\alpha}}.
$$ {#eq:raw-log-pool-witness-source}

Then $Q(\alpha;s^{(\alpha)})=q$. Writing $\Pi$ for projection onto the
tangent space of the simplex, first-order stationarity of
[@eq:separable-server-objective] at that same $q$ requires
$\Pi[(\alpha-1)\log q+\nabla G(q)]=0$.

The unit-scale construction forces
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

| Artifact | Implementation surface and declared scope |
| --- | --- |
| Raw contradiction | `server_theory.py`; interior inputs and raw $q$-block |
| Normalized companion | `server_theory.py`; forward-KL class and normalized weights |
| Typed report | `formal_no_go`; witness metadata, with the attack grid kept separate |

: Formal MAJ-1 witness inventory. These are deterministic finite-simplex proof
artifacts, not empirical estimates; no resampling interval or deployment claim
is implied. {#tbl:server-theory-witness}

The witness table in [@tbl:server-theory-witness] records the deterministic implementation
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
For \(c,\lambda>0\), consensus \(q\), and nonnegative effective weights
\(a=(a_n)\), define \(F_\lambda(q,a)\) by (\ref{eq:agg-free-energy}), where
\(\mathrm{CE}(q,q_n)=-\sum_i q_i\log q_{n,i}\), \(H(q)\) is entropy, and
\(\mathrm{KL}_{\rm gen}(a\|w)=\sum_n g_n\) for
\(g_n=a_n\log(a_n/w_n)-a_n+w_n\).
\end{definition}

**The $q$-block.** For $\lambda>0$, fixing $a$, the $q$-dependent part of
$F_\lambda$ is denoted $J(q;a)$ and can be written in either of two equivalent
forms:

$$
\begin{aligned}
J(q;a)
&=\sum_n a_n\,\mathrm{CE}(q,q_n)-\lambda H(q),\\
J(q;a)
&=\lambda\sum_i q_i\log q_i\\
&\quad-\sum_i\sum_n q_i a_n\log q_{n,i}.
\end{aligned}
$$ {#eq:agg-q-objective}

Adding a Lagrange multiplier to the block in [@eq:agg-q-objective] for
$\sum_i q_i = 1$ and differentiating gives
$\lambda\log q_i + \lambda - \sum_n a_n \log q_{n,i} + \mu = 0$, i.e.

$$
\begin{aligned}
q_i
&\propto \exp\!\Big(
  \tfrac{1}{\lambda}\sum_n a_n \log q_{n,i}\Big),\\
q_i
&= \mathrm{softmax}\!\Big(
  \tfrac{1}{\lambda}\sum_n a_n \log q_n\Big)_i.
\end{aligned}
$$ {#eq:agg-q-min}

the product of the weighted experts — the consensus update of [@eq:agg-updates].
At the default $\lambda=1.0$, the $-H(q)$ term sharpens the weighted geometric mean
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

## Conservative-server properties {#sec:supp-theorem}

\begin{theorem}[Properties]
\label{thm:variational-aggregation}
For \(c,\lambda>0\), (\ref{eq:agg-q-min})–(\ref{eq:agg-a-min}) make \(F\)
non-increasing. At convergence, \((q^*,a^*)\) is coordinatewise stationary.
As \(c\to0\), \(a_n\to w_n\) and \(q\) tends to the tempered log-linear pool;
\(\lambda=1.0\) gives
(\ref{eq:log-linear-pool}). Finally,
\(a_n=w_n\exp[-c\,\mathrm{CE}(q,q_n)]\le w_n\), and
\(\mathrm{KL}(q\,\|\,q_n)\to\infty\) gives \(a_n\to0\).
\end{theorem}

The variational aggregator therefore shares the project log-linear-pool corner
of (\ref{eq:robust-identity}). Under the qualified bridge of
[@sec:method-aggregation], this is only the categorical
message-combination specialization, not the complete source protocol.

The final inequality makes the raw effective-weight update bounded and
redescending relative to the realized consensus. It does not by itself
establish a bounded influence function or finite gross-error sensitivity for
the normalized consensus estimator.

The objective $F$ is biconvex (each block convex, the coupling
$\sum_n a_n\mathrm{CE}(q,q_n)$ bilinear), so the result concerns monotone block
updates and converged coordinatewise fixed points, not guaranteed convergence to
or certification of a global minimum.

**The effective-weight regime, and why multi-start matters.** The weight bound
$a_n \le w_n$ is unconditional ($\mathrm{CE}(q,q_n)\ge 0$ always). The *collapse*
$a_n \to 0$ is driven by the agent's divergence *from the realized consensus*
$\mathrm{KL}(q\,\|\,q_n)$, and the consensus itself depends on the weights.

Because
$F$ is biconvex, this couples into a subtlety an adversarial review of this work
surfaced: a *near-one-hot* saboteur (contamination rate $\to 1$) already captures
the product-of-experts, so a descent seeded *at* that pool stays in a
consensus-capture basin (high $F$) where the saboteur keeps its weight — even
against an honest majority.

The repair is to search the stated objective more carefully:
`variational_aggregate` runs **multi-start** block-coordinate descent (the pool,
the uniform belief, and the arithmetic-mean seeds) and returns the lowest-observed-$F$
converged candidate.

In the configured colony, the uniform/arithmetic seeds reach a lower-$F$
*vetoing* basin, so the saboteur is suppressed even at the simplex vertex
(pinned qualitatively by the near-vertex multi-start test) — the
267.1$\times$ suppression of
[@fig:bounded-influence] is measured across the swept contamination grid,
whose most extreme point sits just below rate $1$.

What remains
fundamental to *every* robust fusion rule, and is not claimed away: with no honest
majority — a colony split with no anchoring plurality — there is no truth to
recover.

The observed suppression is conditional on the tested colonies and the
fixed point selected by a finite multi-start heuristic.

[@fig:descent-comparison] makes the capture and the escape concrete on a
near-vertex colony: the single (log-linear-pool) start settles at
$F = 1.3092$ (the higher observed basin, where the
saboteur keeps its weight), while one of the configured alternative starts
reaches a lower observed basin at $F = -0.2305$ — a gap of
1.5397 nats. This finite multistart comparison shows that
the natural seed can be captured; it neither exhausts all basins nor certifies a
global optimum.

![Variational free-energy descent on a near-vertex adversarial colony. Source relation: original project objective-descent diagnostic; estimand: objective value $F$ in nats by block-coordinate iteration and configured initialization. The x-axis is iteration and the y-axis is $F$. A filled-circle solid path marks the single-start condition and an open-diamond dotted path marks the multistart condition; these neutral condition encodings do not imply different aggregation methods. Directly labeled terminal marks identify the higher and lower observed basins, a dark dotted rule marks the lower observed final level, and a double-ended annotation reports the terminal gap. The log-linear-pool single start settles at $F=1.3092$ with retained saboteur weight, whereas the lowest trajectory among the configured alternatives reaches $F=-0.2305$, a gap of 1.5397 nats. These are two deterministic traces from one configured colony, not independent stochastic replications, so no error bar, confidence interval, or resampling interval applies. Finite multistart descent does not certify a global optimum, enumerate every basin, or establish universal robustness.](../output/figures/descent_comparison.png){#fig:descent-comparison width=80% data-slide-manifest="../output/figures/descent_comparison.slides.json"}

## Numerical witnesses for descent and influence bounds {#sec:supp-witnesses}

The analysis pipeline runs `variational_aggregate` at robustness
$c = 1.50$ on a contaminated colony and records the free
energy after each iteration. The descent falls from
$F = 3.2458$ to $F = 2.3780$ (a monotone drop
of $0.8678$ over 11 iterations,
converged: Yes); the largest single-step *increase* is
$8.88 \times 10^{-16}$, machine zero — the monotonicity of the theorem,
witnessed numerically and drawn in [@fig:aggregation-descent].

For the effective-weight diagnostic, one agent is drifted from healthy toward a confident-wrong
delta and its normalized influence is read at each drift. Clean, it carries
$0.143$ of the pool; at the most extreme swept
drift it carries below $0.001$ — a factor of
267.1 (computed from the unrounded influences,
not the display-rounded values above) below the fixed
0.143 the naive pool would still grant it
([@fig:bounded-influence]).

This makes the redescending normalized-weight behavior
visible on the tested path; it is not an estimator-level B-robustness proof.

## Tempered aggregation family for the accuracy-guarantee trade {#sec:supp-tempered}

The aggregator of [@sec:supp-derivation] fixes the entropy term at unit weight.
Relaxing that single coefficient generates a one-parameter *tempered* family.
Introduce an entropy weight $\lambda > 0$ — the **inverse temperature is**
$1/\lambda$ — and minimize

$$
\begin{aligned}
F_\lambda(q, a)
&= \sum_n a_n\,\mathrm{CE}(q, q_n) - \lambda H(q)\\
&\quad + \tfrac{1}{c}\,\mathrm{KL_{gen}}(a \,\|\, w).
\end{aligned}
$$ {#eq:tempered-family}

Repeating the $q$-block derivation of [@eq:agg-q-min] with the entropy scaled by
$\lambda$ leaves the $a$-block **untouched** and tempers only the consensus update:

$$
\begin{aligned}
q
&\propto \exp\!\Big(
  \tfrac{1}{\lambda}\textstyle\sum_n a_n \log q_n\Big),\\
a_n
&= w_n\,\exp\!\big(-c\,\mathrm{CE}(q, q_n)\big).
\end{aligned}
$$ {#eq:tempered-updates}

The $\lambda\downarrow0$ endpoint is separately implemented as a deterministic
tied-argmax rule; it is not obtained by substituting $\lambda=0$ into
[@eq:tempered-family] or [@eq:tempered-updates].

The weight update is **independent of $\lambda$**: the bound $a_n \le w_n$ with
collapse $a_n \to 0$ as $\mathrm{KL}(q\,\|\,q_n) \to \infty$ is unchanged, so the
**raw effective-weight bound of [@sec:supp-theorem] holds for every** $\lambda > 0$.
At $\lambda = 1.0$ the temperature is unity and
[@eq:tempered-updates] is **identical** to the current axis-3 aggregator
[@eq:agg-updates] — the default is bit-identical, not merely close.

The $c \to 0$
recovery of [@sec:supp-theorem] generalizes to the *tempered* log-linear pool
$q \propto \exp(\tfrac{1}{\lambda}\sum_n w_n \log q_n)$; at $\lambda = 1.0$ this is
exactly $\mathrm{softmax}(\sum_n w_n \log q_n)$ — the project's **log-linear
pool** [@eq:log-linear-pool].

Under the shared-support,
posterior-log-potential, and fixed-weight assumptions of
[@sec:method-aggregation], that pool is a categorical specialization of
Friston Eq. 7's message-combination term, not a reconstruction of the complete
source protocol. Positive-temperature members away from that default are
tempered pools, not Friston Eq. 7 itself; the project recovery checks,
including ISC-10, remain project-local.

A small empirical sweep over $\lambda \in \{ 0.1, 0.2, 0.3, 0.5, 0.7, 1 \}$ on
10 contaminated colonies (5 agents,
2 adversarial) asks whether a single $\lambda^{\ast}$
makes the conservative aggregator narrow the gap to the sharp $\mathrm{robust\_aggregate}$
point-accuracy.

The closest observed weight is $\lambda^{\ast} = 0.3$
with an accuracy gap of 0.0008.
**A lambda* narrows the tested accuracy gap while preserving the stated weight bound on this grid.** If no $\lambda$ closes that gap while
preserving the derived weight update, the result is the conservatism trade-off
of [@sec:limitations], not a defect to hide.
