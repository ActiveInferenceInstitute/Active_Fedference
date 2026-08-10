# Supplemental notation contract {#sec:supp-notation}

This supplement is the authoritative notation contract for Active Fedference.
The methods, formalism, results, figures, report schemas, and API documentation
use these meanings even when a source paper uses a different symbol. A symbol
is not reused for a different mathematical object merely because the objects
are both probability vectors. The implementation names in the final column are
the canonical names for new code and reports; old names survive only as warned,
parity-tested compatibility adapters.

## Probability objects and generative-model quantities

### States, posteriors, and site factors

| Symbol | Contractual meaning | Canonical implementation term |
|---|---|---|
| $s$ | A hidden categorical state; $s\in\{1,\ldots,n_s\}$ is an index, not a distribution. | `state` |
| $o$ | An observation/outcome index; $o\in\{1,\ldots,n_o\}$. | `observation` |
| $q_n(s)$ | Agent $n$'s local posterior over the shared latent state after its local update. | `local_posteriors[n]` |
| $q(s)$ | The server consensus posterior over the shared state. | `global_posterior` / `consensus` |
| $q_{-n}(s)$ | The normalized cavity posterior with agent $n$'s site contribution removed. | `cavity(...)` |
| $t_n(s)$ | Agent $n$'s site/factor term in natural-parameter space. | `site_factor` |
| $m_n(s)$ | Bridge-only source-equation log potential: $m_n(s)=\log q_n(s)+\kappa_n$ with state-constant $\kappa_n$. It is not a claim that every source-protocol message is a broadcast posterior. | No project API; notation for the qualified bridge only. |

### Priors, policies, and POMDP quantities

| Symbol | Contractual meaning | Canonical implementation term |
|---|---|---|
| $\pi_0(s)$ | A prior over hidden states. The subscript distinguishes a prior from a policy. | `prior` / `log_prior` |
| $\boldsymbol{\pi}$ | A policy or action sequence; it is not a prior and is bold when needed. | `policy` |
| $A[o,s]=P(o\mid s)$ | Observation likelihood matrix; each state-indexed column is a pmf. | `likelihood` |
| $B[s',s,u]=P(s'\mid s,u)$ | State-transition tensor indexed by next state, current state, and control. | `transition` |
| $C[o]$ | Log-preference over outcomes; $p_C(o)=\operatorname{softmax}(C)[o]$ is the preferred-outcome pmf. | `log_preferences` |
| $D_0[s]$ | Initial hidden-state prior in the POMDP. | `initial_prior` |
| $q(o\mid\boldsymbol{\pi})$ | Policy-conditional predicted outcome distribution in EFE calculations. | `predicted_outcomes` |

The state index $s$, posterior $q(s)$, prior $\pi_0(s)$, and policy
$\boldsymbol{\pi}$ must not be conflated. In particular, the policy symbol is
never used for a prior, and the prior is never called a policy. The uppercase
POMDP tensors $A,B,C,D_0$ are model objects; they are not posterior factors.

For the qualified relation to Friston et al.'s Eq. 7
[@friston2024federated], the shared support is finite, every $q_n(s)$ is
positive on it, the Eq. 7 softmax input is represented by the bridge-only
$m_n(s)$ above, and the declared weights $w_n$ are fixed rather than functions
of the emerging consensus. Under exactly those assumptions, additive
$\kappa_n$ constants cancel under softmax and [@eq:log-linear-pool] is the
categorical posterior-log-potential specialization of the source
message-combination term. It neither reconstructs source message construction,
cavity/exclusion policy, scheduling, generative factors, nor the complete source
protocol.

## Divergences, losses, and scalar controls

### Generalized-Bayes and aggregation terms

| Symbol | Contractual meaning | Canonical implementation term |
|---|---|---|
| $\mathcal D(q\Vert p)$ | A regularizing divergence between distributions, such as KL, reverse KL, or $\alpha$-Rényi. | `divergence` |
| $L(s;o)$ | Loss evaluated at state $s$ for observation $o$. | `loss_by_state` |
| $\tau>0$ | Generalized-Bayes learning rate/temperature multiplying the accumulated loss. | `tau` (`learning_rate` is a warned compatibility alias) |
| $w_n$ | Non-negative raw/base aggregation weight supplied for local posterior $q_n$. | `base_weights[n]` |
| $a_n$ | Raw variational server effective weight before normalization. In the variational rule $0\le a_n\le w_n$. | `raw_effective_weights[n]` |
| $\widetilde a_n=a_n/\sum_m a_m$ | Normalized influence weight returned for interpretation and plotting. | `normalized_effective_weights[n]` |

The symbols $w_n$, $a_n$, and $\widetilde a_n$ are deliberately distinct.
The first is supplied before aggregation, the second is the variational raw
server output, and the third is only its normalized influence representation.
The server heuristic's reweighting is not a FedGVI client loss and does not
inherit a client-side robustness theorem. `robust_aggregate` is a server
heuristic with the tested $c=0$ recovery identity. `variational_aggregate`
owns the explicit finite-simplex objective and the raw-weight bound; that bound
is not an estimator-level B-robustness theorem.

### Robustness, divergences, and loss controls

| Symbol | Contractual meaning | Canonical implementation term |
|---|---|---|
| $c\ge0$ | Server-side robustness coefficient used by the divergence-reweighting rule. | `robustness` |
| $\lambda>0$ | Entropy weight in the variational server objective and its coordinate updates. The $\lambda\downarrow0$ endpoint is a separate deterministic tied-argmax rule. | `entropy_weight` |
| $\alpha>0$ | Rényi divergence order. | `alpha` |
| $\beta\ge0$ | Density-power loss parameter. | `beta` |
| $q_{\rm loss}>0$ | Robust categorical cross-entropy parameter in $L_{q_{\rm loss}}$. The $q_{\rm loss}\downarrow0$ NLL limit is handled separately, and the subscript prevents collision with posterior $q(s)$. | `q_loss` |
| $\rho\in[0,1]$ | Contamination strength/rate in a declared attack mechanism. | `contamination_rate` / `rate` |

For the objective-backed server rule, the complete scalar-control contract is
defined for $c>0$ and $\lambda>0$:

$$
\begin{aligned}
F_\lambda(q,a)
 &= \sum_n a_n\,\mathrm{CE}(q,q_n)
    -\lambda H(q)
    +\frac{1}{c}\,\mathrm{KL}_{\rm gen}(a\Vert w),\\
q &\propto \exp\!\left(\frac{1}{\lambda}
                  \sum_n a_n\log q_n\right),\\
a_n &= w_n\exp\!\big[-c\,\mathrm{CE}(q,q_n)\big].
\end{aligned}
$$ {#eq:notation-variational-objective}

The implementation uses $\lambda={{TEMPERED_ENTROPY_WEIGHT_DEFAULT}}$ by default; the `entropy_weight`
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
q_{-n}(s)
  = \frac{q(s)/t_n(s)}{\sum_{s'}q(s')/t_n(s')}
  = \operatorname{softmax}\!\left(\log q(s)-\log t_n(s)\right).
$$ {#eq:notation-cavity}

The corresponding factor replacement is

$$
\log t_n^{\mathrm{new}}(s)
 = \log t_n^{\mathrm{old}}(s)
   +\log q^{\mathrm{new}}(s)-\log q^{\mathrm{old}}(s),
\qquad
t_n^{\mathrm{new}}(s)\leftarrow
\frac{\exp(\log t_n^{\mathrm{new}}(s))}
     {\sum_{s'}\exp(\log t_n^{\mathrm{new}}(s'))}.
$$ {#eq:notation-factor-replacement}

The code-level adapters are `cavity(global_posterior, site_factor)` and
`update_factor(old_site_factor, old_global_posterior,
new_global_posterior)`. The old keywords `posterior`, `factor`,
`old_factor`, `old_posterior`, and `new_posterior` are accepted only with a
`DeprecationWarning`; mixed canonical/old calls fail closed. Recombination is
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
scientifically interpretable effect. Power, prospective sample size, MCSE, and
MDE are observed-effect planning/precision diagnostics, not evidence that a
confirmatory effect exists.

The predeclared headline display rule is the largest positive $r_{\rm rb}$
among robust methods, with the declared method order as a deterministic tie
break. A report must also expose the complete tied-method set, the tie-break,
the method with the largest mean $\overline{\Delta}$, and the method with the
largest mean at the worst rate. These are distinct summaries; none is a unique
scientific winner when the evidence is tied or conditional.

For the review grid, every configured robust method remains an inferential
member and a displayed rate-profile curve. No pooled-mean selection creates a
curve, interval, or hypothesis-test member for that surface.

## Code and manuscript naming map

| Retired/ambiguous name | Canonical name | Compatibility rule |
|---|---|---|
| `beliefs`, `agent_beliefs` | `local_posteriors` | Warned keyword/property adapters; no silent reinterpretation. |
| `weights` | `base_weights` | Warned keyword adapter. The federation wire key `agent_weights` remains unchanged. |
| `agent_weights` result property | `normalized_effective_weights` | Warned property adapter; serialized wire compatibility is preserved. |
| Variational `agent_weights` argument | `raw_effective_weights` | Warned objective API adapter; reports use the canonical term. |
| `shared_beliefs` | `shared_posteriors` | Warned diagnostics property adapter. |
| `loss_vec` | `loss_by_state` | Warned generalized-Bayes keyword adapter. |
| `cohens_d_from_rank_biserial` | `d_equivalent_from_rank_biserial` | Warned function adapter; the returned value is not raw Cohen's $d$. |
| Report `cohens_d` | Report `d_equivalent` | New reports are canonical and schema-versioned; readers must fail closed on an unsupported version. |

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
analogues checked in the finite categorical model. The
contamination, gallery, onset, conditional-world, and review-grid quantities
are conditional simulation evidence over declared cells. None of these finite
surfaces is an external-data replication, a reconstructed source protocol, a
universal attack taxonomy, a causal intervention, or a proof of a server-side
robustness guarantee. Open theory, calibration, protocol, continuous-state,
external-data, authenticated-federation, and clean-release work remains open in
the project TODO and claim-audit documents.
