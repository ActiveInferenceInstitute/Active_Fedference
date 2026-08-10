# Scholarly audit and evidence boundary

Updated 2026-08-01. This page records the literature pass used to tighten the
claims and simulation design of Active Fedference. It is a claim map, not a
literature review pretending that results in a neighboring field automatically
transfer to this repository.

The phase-indexed implementation plan is in
[`docs/todo/scholarship-and-phase-plan.md`](../todo/scholarship-and-phase-plan.md).

## Current primary-source recheck — 2026-08-01

The current pass rechecked the authoritative [Friston et al. (2024) open
record](https://pmc.ncbi.nlm.nih.gov/articles/PMC11139662/) and the final
[Mildner et al. (2025) PMLR record](https://proceedings.mlr.press/v267/mildner25a.html).
The former supports the federated belief-sharing setting and source equations;
the latter supports the FedGVI generalized-Bayesian/cavity/factor-replacement
and source-theorem lane. Neither source supports a universal contamination
result for this repository's server heuristic, a causal claim from the finite
simulations, or exact replication of the source figures without protocol
parity. The review therefore classifies source equations/protocols, executable
identities, implementation analogues, conditional simulation evidence, and
open research claims separately in the current claim ledger.

The companion [manuscript-wide claim audit](manuscript-claim-audit.md) maps the
load-bearing manuscript claims to their evidence status and records the
retitling/refactor decisions made after the adversarial review.

The current MAJ-1 scoped no-go proposition is a repository-derived
finite-simplex formal result. It is not attributed to Friston, Mildner, or the
closed-form-GVI preprint: those records constrain the interpretation of the
belief-sharing, FedGVI, and candidate-objective lanes, respectively. The
proposition excludes its declared separable class only and neither imports nor
creates a broader robustness or objective claim.

## Anchor sources

| Source | What it supports here | What it does not establish here |
| --- | --- | --- |
| [Friston et al. (2024), *Federated inference and belief sharing*](https://pmc.ncbi.nlm.nih.gov/articles/PMC11139662/) | The active-inference belief-sharing setting and the standard log-linear pooling limit. | Robustness to contaminated or adversarial broadcasts; statistical significance of this implementation. |
| [Mildner et al. (2025), *Federated Generalised Variational Inference*](https://proceedings.mlr.press/v267/mildner25a.html) | FedGVI's generalized-Bayesian federated objective, cavity construction, misspecification framing, and the source method reimplemented in the categorical lane. | A guarantee for Active Fedference's discrete-categorical active-inference architecture or its server heuristic. |
| [Mildner et al. (2025), arXiv version](https://arxiv.org/abs/2502.00846) | The accessible preprint record and version history for the same primary work. | A substitute for a repository-specific proof; the code must keep theorem-backed client updates distinct from heuristic server pooling. |
| [FedGVI public implementation](https://github.com/Terje-M/FedGVI), pinned by full revision in `research_registry.py` | The implementation-level protocol authority for variational family, site factors, cavity construction, factor replacement, data splits, rounds, stopping, predictive sampling, and ELBO sampling. | Evidence that an unresolved parity row is matched, or that the local M4 profile is exact source-scale CUDA replication. |

The pinned FashionMNIST shell uses run indices 1--5 against a six-value seed
table. Consequently, the executable source seeds are
`[676, 93, 215, 318, 242]`; the registry records the table, indices, and
effective sequence separately. This resolves the earlier roadmap copy of the
first five table entries without pretending the discrepancy was absent.
| [Mildner, Giampouras & Damoulas (2025), *Rates of Convergence of Generalised Variational Inference Posteriors under Prior Misspecification*](https://arxiv.org/abs/2510.03109) | Current theoretical context for bounded-divergence GVI and prior misspecification. | Evidence that the finite categorical implementation inherits the theorem's assumptions or rates. |
| [Koehler, Brown & Haneuse (2009), *On the Assessment of Monte Carlo Error in Simulation-Based Statistical Analyses*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3337209/) | The reason to report Monte Carlo error/precision separately from a confidence interval or p-value. | A universal sample-size number; the correct number depends on the estimand and observed simulation variance. |
| [Morris, White & Crowther (2019), *Using simulation studies to evaluate statistical methods*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/) | Simulation-design/reporting discipline: declare estimands, distinguish Monte Carlo error from the data-generating variance, and budget replication for precision. | Validation of this repository's particular number of seeds, bootstrap coverage, or contamination conclusions. |
| [Loy & Korobova (2021), *Bootstrapping Clustered Data in R using lmeresampler*](https://arxiv.org/abs/2106.06568) | The resampling-unit question: nested or clustered data require a resampling scheme that respects the declared dependence structure. | Evidence that the current trials are exchangeable across hidden worlds, or that a cluster bootstrap is required for every report here. |
| [Genest & Zidek (1986), *Combining Probability Distributions*](https://doi.org/10.1214/ss/1177013825) | The log-linear pool's established aggregation genealogy. | Contamination resistance or Byzantine tolerance of the ordinary pool. |
| [Efron & Tibshirani (1993), *An Introduction to the Bootstrap*](https://link.springer.com/book/10.1007/978-1-4899-4541-9) | Resampling as the basis for the deterministic percentile intervals in the reports. | Guaranteed interval coverage under every bounded, dependent, or nested simulation design. |

## Sources governing the next phases

These sources were added as design constraints for future work, not as evidence
that an open extension has already been completed.

| Future phase | Source bridge | Constraint carried into the plan |
| --- | --- | --- |
| Generative-model and hierarchy extensions | [Friston et al. (2017), *Deep temporal models and active inference*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5461873/); [de Vries & Friston (2017), *The role of generative models in perception*](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00095/full) | $A/B/C/D$ notation, temporal dependencies, and top-down message semantics must be declared before deeper hierarchies are compared. |
| Continuous/hybrid state spaces | [Bissiri et al. (2016), *Generalized Bayesian inference*](https://doi.org/10.1111/rssb.12158); [Futami et al. (2018), *Robust variational inference with gamma-divergence*](https://proceedings.mlr.press/v80/futami18a.html) | A Gaussian or hybrid implementation must restate the loss, divergence, normalization, and limiting argument; the categorical identity cannot be presumed to transfer. |
| POMDP semantics | [Kaelbling, Littman & Cassandra (1998), *Planning and acting in partially observable stochastic domains*](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf) | A diagram of a hidden-state/action loop is conceptual unless transitions, observations, policy selection, and execution are present in the experiment. |
| BNN scaling | [Blundell et al. (2015), *Weight uncertainty in neural networks*](https://proceedings.mlr.press/v37/blundell15.html); [Izmailov et al. (2021), *What are Bayesian neural network posteriors really like?*](https://proceedings.mlr.press/v139/izmailov21a.html) | A deterministic point-mass MLP is not a mean-field posterior; a faithful lane must expose posterior parameterization, objective, sampling, and source-protocol parity. |
| Server-objective theory | [Nguyen et al. (2026), *Closed-form solutions for generalized variational inference*](https://arxiv.org/abs/2606.25492) | Design input for candidate objective classes only. As a preprint, it does not prove that this repository's reverse-KL heuristic has an objective. |
| Hierarchical task design | [*Hierarchical Active Inference using Successor Representations* (2026)](https://arxiv.org/abs/2604.15679) | Design input for Four Rooms and Key-Door controls. It does not establish an advantage for this repository or replace task-level replication. |
| Authenticated transport | [TLS 1.3, RFC 8446](https://www.rfc-editor.org/rfc/rfc8446.html); [X.509 profile, RFC 5280](https://www.rfc-editor.org/rfc/rfc5280.html); [Python `ssl`](https://docs.python.org/3/library/ssl.html); [Docker Compose networking](https://docs.docker.com/compose/how-tos/networking/); [Docker Engine security](https://docs.docker.com/engine/security/) | The MAJ-4A emulator must require certificate-authenticated TLS, explicit trust-path/identity validation, protected key and replay state, and constrained container/daemon boundaries. These sources are design authorities, not evidence that MAJ-4A exists. |
| Privacy and adversarial robustness | [Bonawitz et al. (2017), *Practical secure aggregation for privacy-preserving machine learning*](https://arxiv.org/abs/1611.04482); [Abadi et al. (2016), *Deep learning with differential privacy*](https://arxiv.org/abs/1607.00133); [Blanchard et al. (2017), *Machine learning with adversaries*](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) | Authentication/integrity, secure aggregation, differential privacy, and Byzantine tolerance are non-equivalent properties and require separate threat models and evidence. |
| Simulation design and reproducibility | [Morris, White & Crowther (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/); [Koehler, Brown & Haneuse (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3337209/); [Wilkinson et al. (2016), FAIR guiding principles](https://doi.org/10.1038/sdata.2016.18) | Future sample sizes must be precision-planned, nested units must not be pseudoreplicated, and machine-readable provenance must accompany rendered claims. |

## Design decisions adopted in the repository

1. The independent Monte Carlo unit is the seed. Clients, episodes, observations,
   and repeated trials are nested within a seed and are reduced before inference.
2. The primary budgets are 240 independent seeds for the main across-seed
   belief-sharing evidence and 480 matched trials for the headline robustness
   contrast. The structural-extension/cross-study tier now uses 64 independent
   seeds, up from the former five-seed smoke summary, plus 20 matched trials per
   contamination rate in its robustness row.
3. Every multi-seed summary reports the mean, median, standard deviation, MCSE,
   percentile-bootstrap interval, and an approximate two-sided MDE. The MDE is a
   planning quantity conditional on the observed seed variance, not a claim of
   achieved external power.
4. Robustness curves retain the deterministic mechanistic trajectory, while the
   companion profile visual uses matched-trial means and 95% bootstrap intervals.
   This makes the single-colony curve and the inferential uncertainty visibly
   different objects.
5. BH-FDR remains scoped to the declared method or rate family. A positive
   effect plus an adjusted rejection is required for a server-side winner; no
   server statistic is presented as a proof of the per-client FedGVI guarantee.
6. The robustness profile is a conditional estimand: trials redraw heterogeneous
   colonies while holding the seeded hidden state and attack target fixed. The
   cross-study robustness row first reduces those matched trials within seed, then
   uses the seed as the independent unit.

7. The primary signed-rank effect is the matched-pairs rank-biserial
   correlation. The reported Cohen-$d$-like number is an explicitly labelled
   monotone display mapping, $2r/\sqrt{1-r^2}$, used for magnitude labels and
   planning only; it is not a raw-mean paired Cohen's $d$ and does not change
   the Wilcoxon test or BH-FDR decision.

## Explicit limitations

The repository is a controlled simulation study. Its results support executable
identities, behavior under the declared categorical contamination mechanisms,
and precision statements conditional on the generated world. They do not by
themselves establish deployment robustness, calibration on real data, resistance
to all Byzantine strategies, or transfer of continuous-space FedGVI theorems to
finite categorical state spaces. Those claims remain future work and are listed
as such in the manuscript limitations and roadmap.

The fixed hidden state and attack target are an important boundary: the within-rate
bootstrap intervals quantify Monte Carlo variation conditional on that world, not
generalization over hidden states or attack geometries. The 64-seed cross-study
layer partially widens the independent unit, but it still reuses the declared
data-generating mechanism and should not be read as a deployment sample.

The 2026-07-28 refresh rechecked the primary source records used by this audit:
Friston et al. is linked to its open PMC record, Mildner et al. to the final
PMLR paper, and the simulation-precision sources to their PMC records. The
repository continues to use those sources as design constraints and evidence
boundaries, not as proof that this implementation inherits their assumptions or
results.
