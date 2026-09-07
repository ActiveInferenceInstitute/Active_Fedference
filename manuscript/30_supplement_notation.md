# Supplemental notation contract {#sec:supp-notation}

This supplement is the authoritative notation contract for Active Fedference.
The methods, formalism, results, figures, report schemas, and API documentation
use these meanings even when a source paper uses a different symbol. A symbol
is not reused for a different mathematical object merely because the objects
are both probability vectors.

The implementation names in the final column are
the canonical names for new code and reports; old names survive only as warned,
parity-tested compatibility adapters.

## Probability objects and generative-model quantities

### States, posteriors, and site factors

| Symbol | Meaning |
|---|---|
| $s$ | Hidden categorical-state index. |
| $o$ | Observation/outcome index. |
| $q_n(s)$ | Agent $n$'s updated local posterior. |
| $q(s)$ | Server consensus posterior. |
| $q_{-n}(s)$ | Normalized cavity excluding site $n$. |
| $t_n(s)$ | Agent $n$'s natural-parameter site. |
| $m_n(s)$ | Bridge-only source log potential. |

Here $s\in\{1,\ldots,n_s\}$ and $o\in\{1,\ldots,n_o\}$ are indices,
not distributions. Each $q_n(s)$ is the local posterior over the shared latent
state after agent $n$'s update; $q(s)$ is the corresponding server consensus,
and $q_{-n}(s)$ removes agent $n$'s site contribution. The site $t_n(s)$ is a
natural-parameter factor. The bridge-only quantity satisfies
$m_n(s)=\log q_n(s)+\kappa_n$ for state-constant $\kappa_n$; it has no project
API and does not imply that every source-protocol message is a broadcast
posterior.

The corresponding code terms are `state`, `observation`,
`local_posteriors[n]`, `global_posterior` or `consensus`, `cavity(...)`, and
`site_factor`, in table order through $t_n(s)$. The bridge-only $m_n(s)$ has no
project API.

### Priors, policies, and POMDP quantities

| Symbol | Meaning |
|---|---|
| $\pi_0(s)$ | Hidden-state prior, not a policy. |
| $\boldsymbol{\pi}$ | Policy or action sequence. |
| $A[o,s]=P(o\mid s)$ | Observation-likelihood matrix. |
| $B[s',s,u]=P(s'\mid s,u)$ | State-transition tensor. |
| $C[o]$ | Outcome log-preference. |
| $D_0[s]$ | Initial POMDP state prior. |
| $q(o\mid\boldsymbol{\pi})$ | Policy-conditional predicted outcomes. |

The state index $s$, posterior $q(s)$, prior $\pi_0(s)$, and policy
$\boldsymbol{\pi}$ must not be conflated. In particular, the policy symbol is
bold when needed, never used for a prior, and the prior is never called a
policy. The uppercase POMDP tensors $A,B,C,D_0$ are model objects; they are not
posterior factors.
Each state-indexed column of $A$ is a pmf. The tensor $B$ is indexed by next
state, current state, and control. The preferred-outcome pmf is
$p_C(o)=\operatorname{softmax}(C)[o]$, and
$q(o\mid\boldsymbol{\pi})$ is the predicted-outcome distribution used in EFE
calculations.

The canonical code terms, in table order, are `prior` or `log_prior`, `policy`,
`likelihood`, `transition`, `log_preferences`, `initial_prior`, and
`predicted_outcomes`.

For the qualified relation to Friston et al.'s Eq. 7
[@friston2024federated], the shared support is finite, every $q_n(s)$ is
positive on it, the Eq. 7 softmax input is represented by the bridge-only
$m_n(s)$ above, and the declared weights $w_n$ are fixed rather than functions
of the emerging consensus.

Under exactly those assumptions, additive
$\kappa_n$ constants cancel under softmax and [@eq:log-linear-pool] is the
categorical posterior-log-potential specialization of the source
message-combination term. It neither reconstructs source message construction,
cavity/exclusion policy, scheduling, generative factors, nor the complete source
protocol.

## Divergences, losses, and scalar controls

### Generalized-Bayes and aggregation terms

| Symbol | Meaning |
|---|---|
| $\mathcal D(q\Vert p)$ | Regularizing divergence. |
| $L(s;o)$ | Statewise loss for observation $o$. |
| $\tau>0$ | Generalized-Bayes learning rate. |
| $w_n$ | Supplied non-negative base weight. |
| $a_n$ | Raw variational effective weight. |
| $\widetilde a_n=a_n/\sum_m a_m$ | Normalized influence weight. |

The regularizing divergence acts between distributions and may be KL, reverse
KL, or $\alpha$-Rényi. The positive learning rate/temperature $\tau$ multiplies
accumulated loss and accepts `learning_rate` only as a warned compatibility
alias.

The symbols $w_n$, $a_n$, and $\widetilde a_n$ are deliberately distinct.
The first is supplied before aggregation, the second is the variational raw
server output satisfying $0\le a_n\le w_n$, and the third is only its
normalized influence representation returned for interpretation and plotting.

Their canonical code terms, in table order, are `divergence`, `loss_by_state`,
`tau`, `base_weights[n]`, `raw_effective_weights[n]`, and
`normalized_effective_weights[n]`.

The server heuristic's reweighting is not a FedGVI client loss and does not
inherit a client-side robustness theorem. `robust_aggregate` is a server
heuristic with the tested $c=0$ recovery identity. `variational_aggregate`
owns the explicit finite-simplex objective and the raw-weight bound; that bound
is not an estimator-level B-robustness theorem.

### Robustness, divergences, and loss controls

| Symbol | Meaning |
|---|---|
| $c\ge0$ | Server divergence-reweighting coefficient. |
| $\lambda>0$ | Variational-objective entropy weight. |
| $\alpha>0$ | Rényi-divergence order. |
| $\beta\ge0$ | Density-power-loss parameter. |
| $q_{\rm loss}>0$ | Robust categorical-cross-entropy parameter. |
| $\rho\in[0,1]$ | Declared contamination rate. |

The coefficient $\lambda$ controls the variational objective and its
coordinate updates; $\lambda\downarrow0$ is a separate deterministic
tied-argmax endpoint. The parameter $q_{\rm loss}$ belongs to
$L_{q_{\rm loss}}$; its $q_{\rm loss}\downarrow0$ NLL limit is handled
separately, and the subscript prevents collision with posterior $q(s)$.
Contamination rate $\rho$ is always interpreted within the declared attack
mechanism.

The associated code terms are `robustness`, `entropy_weight`, `alpha`, `beta`,
`q_loss`, and `contamination_rate` or `rate`, in table order.

For the objective-backed server rule, the complete scalar-control contract is
defined for $c>0$ and $\lambda>0$:

$$
\begin{aligned}
F_\lambda(q,a)
 &= \sum_n a_n\,\mathrm{CE}(q,q_n)
    -\lambda H(q)\\
 &\qquad +\frac{1}{c}\,\mathrm{KL}_{\rm gen}(a\Vert w).
\end{aligned}
$$ {#eq:notation-variational-objective}

Its two coordinate updates are shown separately so each display remains
legible at the presentation typography floor:

$$
\begin{aligned}
q &\propto \exp\!\left(
\frac{1}{\lambda}\sum_n a_n\log q_n\right),\\
a_n &= w_n\exp\!\big[-c\,\mathrm{CE}(q,q_n)\big].
\end{aligned}
$$ {#eq:notation-variational-updates}

The coordinate updates [@eq:notation-variational-updates] use
$\lambda={{TEMPERED_ENTROPY_WEIGHT_DEFAULT}}$ by default; the `entropy_weight`
argument exposes the stated tempered family. The $c=0$ branch is handled as a
recovery limit outside the $c>0$ objective; at
$\lambda={{TEMPERED_ENTROPY_WEIGHT_DEFAULT}}$ it is the exact
project log-linear pool. The $\lambda\downarrow0$ endpoint is separately
implemented as a deterministic tied-argmax rule and is not obtained by
substituting $\lambda=0$ into the displayed objective or update.

## Cavity and factor algebra

For a positive-support global posterior and site term, the cavity operation is
defined in log space and then normalized:

$$
\begin{aligned}
q_{-n}(s)
&= \frac{q(s)/t_n(s)}{\sum_{s'}q(s')/t_n(s')}\\
&= \operatorname{softmax}\!\left(\log q(s)-\log t_n(s)\right).
\end{aligned}
$$ {#eq:notation-cavity}

The corresponding factor replacement is

$$
\begin{aligned}
\ell_n(s)
&= \log t_n^{\mathrm{old}}(s)\\
&\quad +\log q^{\mathrm{new}}(s)-\log q^{\mathrm{old}}(s),\\
t_n^{\mathrm{new}}(s)
&\leftarrow \operatorname{softmax}(\ell_n)(s).
\end{aligned}
$$ {#eq:notation-factor-replacement}

Here $\ell_n$ is the transient, unnormalized log-site vector; the softmax is
exactly the exponential normalization of that vector.

The code-level `cavity` adapter takes `global_posterior` and `site_factor`.
The `update_factor` adapter takes `old_site_factor`, `old_global_posterior`,
and `new_global_posterior`. The old keywords `posterior`, `factor`,
`old_factor`, `old_posterior`, and `new_posterior` are accepted only with a
`DeprecationWarning`; mixed canonical/old calls fail closed.

Recombination is
tested by normalizing $q_{-n}(s)t_n(s)$ and checking recovery of $q(s)$ to
floating-point tolerance. A transported site factor is represented as a pmf,
so its arbitrary positive natural-parameter scale is fixed by the explicit
normalization above.

## Statistical notation and nesting

| Symbol | Contractual meaning |
|---|---|
| $n_{\rm seed}$ | Number of independently seeded worlds/replicates in a declared cell; the inferential unit for seed-level summaries. |
| $n_{\rm trial}$ | Number of trials nested within one seed and cell; trials are averaged before seed-level inference. |
| $\Delta=b-a$ | Matched robust-minus-naive contrast for the same seed/trial or the declared seed-level reduction. |
| $r_{\rm rb}$ | Wilcoxon matched-pairs rank-biserial effect, primary standardized effect. |
| $d_{\rm eq}$ | $2r_{\rm rb}/\sqrt{1-r_{\rm rb}^2}$, a secondary rank-biserial-derived display/planning d-equivalent, not raw Cohen's $d$. |
| $\mathrm{CI}_{1-\alpha}$ | Percentile bootstrap interval for the named estimand, resampling the declared replication unit. |
| $\mathrm{MCSE}$ | Monte Carlo standard error/precision diagnostic for a simulation summary; it is not a confidence interval. |
| $\mathrm{MDE}$ | Observed-design minimum detectable effect diagnostic under its stated approximation; it is not confirmatory evidence. |
| $p,q$ | Raw p-value and BH-adjusted q-value; the family and ownership are declared with every report. |

For the robustness sweep, the primary result is $r_{\rm rb}$ and the matched
mean difference $\overline{\Delta}$ with its bootstrap interval. The d-equivalent
is retained only as a monotone secondary display and planning input. When
$|r_{\rm rb}|=1$, the transform diverges; reports use a finite sentinel and
captions disclose saturation rather than presenting a million-scale number as a
scientifically interpretable effect.

Power, prospective sample size, MCSE, and
MDE are observed-effect planning/precision diagnostics, not evidence that a
confirmatory effect exists.

The predeclared headline display rule is the largest positive $r_{\rm rb}$
among robust methods, with the declared method order as a deterministic tie
break. A report must also expose the complete tied-method set, the tie-break,
the method with the largest mean $\overline{\Delta}$, and the method with the
largest mean at the worst rate.

These are distinct summaries; none is a unique
scientific winner when the evidence is tied or conditional.

For the review grid, every configured robust method remains an inferential
member and a displayed rate-profile curve. No pooled-mean selection creates a
curve, interval, or hypothesis-test member for that surface.

## Code and manuscript naming map

Every arrow names a legacy surface followed by its canonical replacement.

**Belief inputs.** `beliefs` or `agent_beliefs` → `local_posteriors`; `weights`
→ `base_weights`. Both are warned keyword/property adapters.

**Aggregation outputs.** Result property `agent_weights` →
`normalized_effective_weights`; variational argument `agent_weights` →
`raw_effective_weights`. Both are warned adapters, but represent different
weight scales.

**Diagnostics.** `shared_beliefs` → `shared_posteriors`; `loss_vec` →
`loss_by_state`. These are warned diagnostics and keyword adapters.

**Statistics.** `cohens_d_from_rank_biserial` →
`d_equivalent_from_rank_biserial`; report field `cohens_d` → `d_equivalent`.
The function is a warned adapter; the report change is a versioned migration.

These adapters never silently reinterpret an argument or result. The
`agent_weights` result property preserves serialized wire compatibility, while
the variational objective and new reports use the canonical weight terms. New
reports are canonical and schema-versioned; readers fail closed on unsupported
versions. `d_equivalent_from_rank_biserial` returns the declared
rank-biserial-derived d-equivalent, not raw Cohen's $d$; `loss_vec` remains only
a warned generalized-Bayes keyword adapter.

The wire-level key `agent_weights` is preserved because it is a federation
transport contract, not a claim about the scale or meaning of the new result
fields. A future wire migration requires an explicit version and a fail-closed
reader; it must not silently reinterpret the key.

## Source and evidence boundaries

The Friston belief-sharing equations are source equations/protocol claims. Only
under the explicit finite-shared-support, posterior-log-potential, and
fixed-weight bridge above does the categorical log-linear pool specialize the
source message-combination term; the tested $c=0$ identity remains
project-local. The generalized-Bayes and loss limits are implementation
analogues checked in the finite categorical model.

The
contamination, gallery, onset, conditional-world, and review-grid quantities
are conditional simulation evidence over declared cells. None of these finite
surfaces is an external-data replication, a reconstructed source protocol, a
universal attack taxonomy, a causal intervention, or a proof of a server-side
robustness guarantee.

Open theory, calibration, protocol, continuous-state,
external-data, authenticated-federation, and clean-release work remains open in
the project TODO and claim-audit documents.
