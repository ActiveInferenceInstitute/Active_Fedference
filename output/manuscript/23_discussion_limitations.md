## Limitations and claim boundaries {#sec:limitations}

The boundaries here define the contribution. The central goal is to show —
concretely and testably — that the categorical generalized-variational
construction inspired by FedGVI [@mildner2025fedgvi] has standard-Bayes client
limits and a project-local log-linear-pool server identity. Under the explicit
bridge in [@sec:method-aggregation], the latter specializes Friston et al.'s
Eq. 7 message-combination term [@friston2024federated], not the complete source
protocol; behavior away from those limits is evaluated only under declared
simulation conditions. The exact identities are formal; performance and
deployment claims remain conditional. Items beyond that boundary are named as
future work.

## Three robustness axes: theorem, heuristic, and objective {#sec:limitations-axes}

Robustness enters in three places with unequal standing — a distinction surfaced by
an adversarial review of this work and preserved deliberately throughout
([@sec:robustness-axes]).

1. **Client-side, source-theorem-backed.** A robust per-agent generalized-Bayes update with a
   bounded loss — $\beta$-loss [@basu1998robust] or generalized cross-entropy
   [@zhang2018generalized] — inside the generalized posterior. Density-power and
   gamma-divergence Bayes make clear that this is a loss/divergence-specific
   robustness property, not an automatic property of every generalized posterior
   [@fujisawa2008robust; @ghosh2015robust]. It is derived from the
   generalized-Bayes objective [@bissiri2016general; @jewson2018divergence;
   @knoblauch2022generalized],
   provably limits to NLL/Bayes (hence to the standard pool) as the loss
   parameter goes to zero ([@eq:beta-loss], [@eq:rcce-loss]), and is the
   theorem-bearing axis under the matching loss, divergence, and regularity
   assumptions of FedGVI. The manuscript does not re-prove that theorem for
   every possible categorical data-generating process. **This is the
   client-side mechanism implemented and evaluated here.**

2. **Server-side, heuristic.** The divergence-reweighting aggregator
   `robust_aggregate`. Its positive formal property is the recovery limit — robustness
   $0$ equals the standard log-linear pool ([@eq:robust-identity]). A scoped
   no-go rejects one declared separable objective class; it is *not* a
   closed-form minimizer of a FedGVI objective, and we never claim it inherits
   the bounded-influence bound. An empirical characterization
   ([@sec:results-heuristic-characterization]) makes this boundary concrete: the
   heuristic has a *finite* measured breakdown point (a colluding majority
   captures it), a witness against unconditional server resistance rather than
   a transferred client theorem. The robustness sweep reports it as a complementary
   heuristic, and [@fig:robust-weights] is labeled accordingly. Robust federated
   optimizers such as Krum and gamma-mean aggregation motivate the adversarial
   client problem [@blanchard2017krum; @li2022gammafl], but they do not certify
   this belief-pooling heuristic. It has BH-rejected positive contrasts in the
   configured accuracy verdict in [@sec:results-verdict], while the declared mechanism gallery and
   conditional-world study retain reversals that prohibit a universal
   accuracy claim.

3. **Server-side, objective-backed (conservative).** The variational aggregator
   `variational_aggregate` of [@sec:method-variational], derived in
   [@sec:supp-variational]. It runs exact closed-form block updates that descend
   the stated free energy [@eq:agg-free-energy] monotonically on each block, shares the recovery corner
   ([@eq:robust-identity]) with axis 2, and — unlike axis 2 — carries a proven
   raw effective-weight bound with empirical redescending behavior ([@fig:bounded-influence]).
   Its cost in the declared comparison is conservatism: the entropy
   regularization yields a more diffuse consensus, and the method does *not*
   win that configured peak-accuracy verdict. It closes the "is
   there *any* objective-backed server rule with raw-weight control" gap; it does not
   retroactively endow the sharp axis-2 heuristic with an objective.

The honest state is therefore a triangle, not a binary: axis 1 is
source-theorem-backed under matching assumptions, axis 2 has conditional wins
and reversals *without* a server-side objective, and axis 3 is objective-backed
*but* conservative. No experiment attributes axis 2's
accuracy win to a theoretical guarantee, and none attributes axis 3's weight bound
to a peak-accuracy claim. The remaining open problem — an objective whose
minimizer is the *sharp* reweighting itself, which would combine accuracy and guarantee in
one server rule — is named future work in [@sec:future]; the extended methods
that broaden the toolkit without touching these claims are cataloged in
[@sec:supp-extended].

## Scope boundaries that the evidence does not cross {#sec:limitations-scope}

- **The bridge is a recasting, not an upstream claim.** Friston et al.
  [@friston2024federated] supply a belief-sharing operator for agents with a
  shared world model; Mildner et al. [@mildner2025fedgvi] supply robust
  federated generalized variational inference. Neither paper claims the other.
  The contribution here is to recast the former as the tested KL/NLL
  zero-robustness recovery corner of the latter-inspired construction, then
  measure what changes when robust losses and server reweighting are introduced.
- **Historical sources are conceptual, not formal support.** The pre-modern
  sources added in [@sec:related-historical] support a genealogy of expectation,
  inverse inference, utility, and collective judgment
  [@huygens1657ratiociniis; @bayes1763essay; @laplace1774memoire;
  @bernoulli1738mensura; @condorcet1785essai]. They do not support any claim
  about KL, NLL, product-of-experts training, FedGVI, or `robust_aggregate`.
  Those claims rest only on the modern cited formalism and the tests reported
  in [@sec:results-recovery].
- **Single-machine federation.** Federation is validated with queue transport,
  a single-machine OS-process helper, and loopback TCP: agents serialize
  beliefs, the server aggregates, and consensus is broadcast back without
  changing the mathematics. The socket path now exercises frame integrity and
  file-backed digest-verified replay validation; an optional SQLite round-ID
  guard also survives local process restarts. These controls are still not a
  substitute for cross-host deployment, identity-bound mTLS, a shared
  multi-host replay domain, discovery, long-running worker orchestration, or
  fault tolerance; those steps are scoped as future work ([@sec:future]).
- **GPU / PyTorch Bayesian-neural-network experiments at the original FedGVI
  scale.** The bounded-influence result is anchored here by a small NumPy
  logistic-regression baseline ([@fig:bnn-robustness]), not the RTX-class runs of
  the source paper [@mildner2025fedgvi]. Even on that anchor the robust client's
  margin over the standard client is non-monotone rather than uniform: it opens in
  the moderate-to-high contamination range and then vanishes at the most extreme
  swept level, where both configurations collapse together with no reliable
  ordering ([@sec:results-baseline]). That terminal convergence is reported, not
  trimmed; the axis's rigorous standing rests on the recovery identities and the
  FedGVI theorem under its matching assumptions, not on the size or monotonicity
  of the empirical gap in this figure.
- **Classification baselines are point estimates, not full posteriors.** The
  NumPy logistic-regression baseline and the PyTorch MLP complement both use
  point-estimate weights (no posterior covariance is computed). A genuine
  mean-field variational family over the weights — diagonal-Gaussian $q(w)$ with
  a closed-form KL and a Monte-Carlo ELBO — is implemented as a tested primitive
  (`bnn_variational_torch.VariationalMLP`, recovering the point-estimate net
  exactly as its posterior variance vanishes), but the full paper-faithful
  FedGVI classification lane (that variational family trained under the
  contamination sweep, at GPU scale) is not: MCMC, non-diagonal structure, and
  stochastic-weight averaging remain unimplemented, and the full-posterior
  regime remains unverified.
- **Hierarchical depth does not by itself improve the base task.** The two- and
  three-level stacks ([@sec:results-hierarchical], [@sec:results-3level]) run
  alternating L1/L2(/L3) inference end-to-end and resolve their added context
  latents above chance, but on the shared location task they do not beat the
  flat baseline — the paired location-accuracy gap is a small, statistically
  reliable negative gap at the reported seed count
  ([@fig:hierarchical-pomdp]).
  Depth is therefore validated as executable and consensus-preserving, not as an
  accuracy improvement; whether a richer policy-and-horizon task family rewards
  hierarchy is future work ([@sec:future-hierarchical]).
- **Discrete POMDP only — continuous or hybrid state spaces are not addressed.**
  This work is discrete-categorical only, matching the community's worked POMDP
  example [@dacosta2020active; @friston2024federated]. The limit-as-proof
  contract is validated off the categorical case only for a one-dimensional
  Gaussian-mean conjugate slice; continuous-state active inference and Gaussian
  belief-sharing colonies remain untested.
- **Temperature and divergence calibration are fixed, not learned.** Coarsened
  posterior and safe-Bayes theory make clear that the learning-rate/temperature
  is part of the inference rule under misspecification [@miller2018coarsening;
  @grunwald2012safe; @kleijn2012misspecification]. This manuscript validates the
  configured losses and divergences, but it does not learn an optimal
  coarsening radius, temperature, or divergence schedule across agents.
- **Real multi-machine federation, networking, and privacy cryptography.**
  Federation here is *mathematical* (factor aggregation), not infrastructural —
  unlike communication-efficient or Byzantine-robust federated learning
  [@mcmahan2017communication; @blanchard2017krum].
- **New linguistic theory.** The language-acquisition study
  ([@sec:results-language]) reproduces the Dirichlet count mechanism
  ([@eq:dirichlet-update]) mechanically; it does not extend the linguistics.

## What the statistics can and cannot claim {#sec:limitations-stats}

The verdict is a paired comparison at a single high contamination rate
([@sec:results-verdict], [@tbl:robustness_verdict]). This concentrates power
where the effect is largest, which is honest about *where* robustness pays off
but does not characterize the full contamination curve as a continuous function;
the per-rate table ([@tbl:paired-by-rate]) reports the rest of the sweep without
claiming family-wide significance beyond what BH-FDR
[@benjamini1995controlling] certifies. BH-FDR controls expected false discovery
proportion within the declared family, not the chance of any false positive.
The matched-pairs Wilcoxon tests are rank tests under paired-difference
assumptions [@wilcoxon1945individual; @fay2010wilcoxon], not assumption-free
proofs about raw means. Confidence intervals are percentile bootstrap
[@efron1993bootstrap], not analytic, and inherit that method's small-sample
caveats at the lowest trial counts. The power values are observed-effect design
approximations useful for confirmatory sample-size planning; they are not
independent evidence for the verdict and do not cover model-specification
uncertainty outside the seeded simulation harness [@wasserstein2016asa].
