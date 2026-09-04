## Limitations and claim boundaries {#sec:limitations}

The boundaries here define the contribution. The central goal is to show —
concretely and testably — that the categorical generalized-variational
construction inspired by FedGVI [@mildner2025fedgvi] has standard-Bayes client
limits and a project-local log-linear-pool server identity. Under the explicit
bridge in [@sec:method-aggregation], the latter specializes Friston et al.'s
Eq. 7 message-combination term [@friston2024federated], not the complete source
protocol; behavior away from those limits is evaluated only under declared
simulation conditions.

The exact identities are formal; performance and
deployment claims remain conditional. Items beyond that boundary are named as
future work.

## Three robustness axes: theorem, heuristic, and objective {#sec:limitations-axes}

The authoritative definitions and claim owners are in
[@sec:robustness-axes]; this section records only what still cannot transfer.
A source-conditional client theorem does not certify either server rule, a
server recovery identity does not supply an objective or estimator-level
influence bound, and objective descent plus raw-weight control does not imply
truth recovery or peak-accuracy dominance. Likewise, conditional accuracy
contrasts cannot confer a theorem on the heuristic. Figure
[@fig:evidence-replication-map] makes that no-transfer boundary and its nested
replication units explicit.

The remaining methodological problem is a server objective that combines
competitive conditional performance with an appropriate robustness guarantee.
Related robust-federation work motivates the target
[@blanchard2017krum; @li2022gammafl] but does not certify either project server
rule. Any such result must also survive beyond the present categorical model,
declared attack grid, and single-host execution setting; future work and
extended methods remain separate ([@sec:future], [@sec:supp-extended]).

## Scope boundaries that the evidence does not cross {#sec:limitations-scope}

**The bridge is a recasting, not an upstream claim.** Friston et al.
[@friston2024federated] supply a belief-sharing operator for agents with a
shared world model; Mildner et al. [@mildner2025fedgvi] supply robust federated
generalized variational inference. Neither paper claims the other. The
contribution here is to recast the former as the tested KL/NLL zero-robustness
recovery corner of the latter-inspired construction, then measure changes under
robust losses and server reweighting.

**Historical sources are conceptual, not formal support.** The pre-modern
sources added in [@sec:related-historical] support a genealogy of expectation,
inverse inference, utility, and collective judgment
[@huygens1657ratiociniis; @bayes1763essay; @laplace1774memoire;
@bernoulli1738mensura; @condorcet1785essai]. They do not support claims about
KL, NLL, product-of-experts training, FedGVI, or `robust_aggregate`.

Those claims rest only on the modern cited formalism and the tests reported in
[@sec:results-recovery].

**Single-machine federation.** Federation is validated with queue transport, a
single-machine OS-process helper, and loopback TCP: agents serialize beliefs,
the server aggregates, and consensus is broadcast back without changing the
mathematics. The socket path exercises frame integrity and file-backed
digest-verified replay validation; an optional SQLite round-ID guard survives
local process restarts.

These controls do not substitute for cross-host deployment, identity-bound
mTLS, a shared multi-host replay domain, discovery, long-running worker
orchestration, or fault tolerance; those steps are scoped as future work
([@sec:future]).

**Source-scale posterior FedGVI classification remains open.** The displayed
NumPy logistic-regression baseline is an exploratory point-estimate comparison,
not the source paper's posterior-uncertainty BNN experiment
[@mildner2025fedgvi]. Its operating point was selected within the evaluated
synthetic sweep to expose a mid-contamination margin
([@fig:bnn-robustness]).

The curves lose reliable ordering at the highest rate. The source theorem, not
this selected finite sweep, carries the source-conditional robustness claim.

**Classification baselines are point estimates, not full posteriors.** The
NumPy logistic-regression baseline and the PyTorch MLP complement both use
point-estimate weights; no posterior covariance is computed.

A genuine mean-field variational family over the weights—diagonal-Gaussian
$q(w)$ with a closed-form KL and a Monte-Carlo ELBO—is implemented as a tested
primitive (`bnn_variational_torch.VariationalMLP`, recovering the point-estimate
net exactly as its posterior variance vanishes).

The source-comparable training and evaluation lane remains unexecuted:
leakage-free calibration, source-dataset parity, posterior-family comparisons,
and source-scale compute are still open.

**Hierarchical depth does not by itself improve the base task.** The two- and
three-level stacks ([@sec:results-hierarchical], [@sec:results-3level]) run
alternating L1/L2(/L3) inference end-to-end and resolve their added context
latents above chance. On the shared location task they do not beat the flat
baseline: the paired location-accuracy gap is a small, statistically reliable
negative gap at the reported seed count ([@fig:hierarchical-pomdp]).

Depth is therefore validated as executable and consensus-preserving, not as an
accuracy improvement; whether a richer policy-and-horizon task family rewards
hierarchy is future work ([@sec:future-hierarchical]).

**Discrete POMDP only—continuous or hybrid state spaces are not addressed.**
This work is discrete-categorical only, matching the community's worked POMDP
example [@dacosta2020active; @friston2024federated]. The limit-as-proof
contract is validated off the categorical case only for a one-dimensional
Gaussian-mean conjugate slice; continuous-state active inference and Gaussian
belief-sharing colonies remain untested.

**Temperature and divergence calibration are fixed, not learned.** Coarsened
posterior and safe-Bayes theory make clear that the learning-rate/temperature
is part of the inference rule under misspecification [@miller2018coarsening;
@grunwald2012safe; @kleijn2012misspecification]. This manuscript validates the
configured losses and divergences, but it does not learn an optimal coarsening
radius, temperature, or divergence schedule across agents.

**Real multi-machine federation, networking, and privacy cryptography.**
Federation here is *mathematical* (factor aggregation), not infrastructural—
unlike communication-efficient or Byzantine-robust federated learning
[@mcmahan2017communication; @blanchard2017krum].

**New linguistic theory.** The language-acquisition study
([@sec:results-language]) reproduces the Dirichlet count mechanism
([@eq:dirichlet-update]) mechanically; it does not extend the linguistics.

## What the statistics can and cannot claim {#sec:limitations-stats}

The verdict is a paired comparison at a single high contamination rate
([@sec:results-verdict], [@tbl:robustness_verdict]). This concentrates power
where the effect is largest, which is honest about *where* robustness pays off
but does not characterize the full contamination curve as a continuous function;
the per-rate effect and inference projections ([@tbl:paired-by-rate;
@tbl:paired-by-rate-inference]) report the rest of the sweep without
claiming family-wide significance beyond what BH-FDR
[@benjamini1995controlling] supports.

BH-FDR controls expected false discovery
proportion within the declared family, not the chance of any false positive.
The matched-pairs Wilcoxon tests are rank tests under paired-difference
assumptions [@wilcoxon1945individual; @fay2010wilcoxon], not assumption-free
proofs about raw means.

Confidence intervals are percentile bootstrap
[@efron1993bootstrap], not analytic, and inherit that method's small-sample
caveats at the lowest trial counts.

The power values are observed-effect design
approximations useful for confirmatory sample-size planning; they are not
independent evidence for the verdict and do not cover model-specification
uncertainty outside the seeded simulation harness [@wasserstein2016asa].
