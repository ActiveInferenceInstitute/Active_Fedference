# Reproducibility: producer provenance and recovery checks {#sec:repro-production}

## Producer order, stale invalidation, and publication authority {#sec:repro-provenance-map}

The provenance map in [@fig:source-render-provenance] records the source-owned producer
contract rather than inferring it from the release manifest. Panel A places
the non-sequential fan-in dependencies in a labeled matrix, then orders the
temporal, gate, receipt, and authorization dependencies in a crossing-free
process path. Source, configuration, lockfile, figure-accessibility registries,
and the exact clean Template commit remain explicit inputs.

Panel B's fourteen coded dashed cards make reverse invalidation explicit without
crossing the forward process path. Each arrow starts at one or more immediate
stale targets on the right and points back to its changed owner on the left. A
changed owner makes the named immediate targets and their downstream reports,
figures, hydrated text, rendered surfaces, provenance, and release receipts
stale; previously green bytes are not carried forward.

GitHub and Zenodo are terminal authorization states, not automatic products of
a green build. The figure itself is generated from the source-owned pipeline
contract and never reads the downstream release manifest that later inventories
it, avoiding a circular dependency.

![Publication surfaces inherit source state; green gates remain distinct from release authority. Source relation: source-owned explanatory producer and invalidation contract; study status: deterministic provenance map, not an empirical result. Read Panel A first across the dependency matrix and then through the two-band process path. The matrix x-axis indexes seven downstream targets, while its rows name five inputs and three upstream producers; each populated cell carries a producer, gate, or receipt code. The adjacent crossing-free path orders provisional hydration, tests and coverage, final hydration, exact clean Template rendering, reader surfaces, surface checks, provenance, manifests, and the separately authorized GitHub and Zenodo terminals. Panel B groups fourteen changed owners into dashed reverse-dependency cards; each arrow points from immediate stale targets back to its changed owner. Panel C states the non-circular manifest boundary and four no-claim rules. Codes, direct labels, keylines, solid producer or receipt arrows, dashed gate arrows, and dashed reverse arrows duplicate color. The estimand is categorical dependency and authorization state; the unit is one pipeline or publication stage. There is no sampling uncertainty, sample size, or replication unit. The figure never reads the downstream manifest. A green build does not authorize publication, establish scientific validity, PDF/UA or WCAG conformance, or enlarge the claim surface through a release or DOI.](../output/figures/source_render_provenance.png){#fig:source-render-provenance data-slide-manifest="../output/figures/source_render_provenance.slides.json" width=95%}

## Recovery-limit certificate for the client and project-pool corners {#sec:repro-recovery}

The recovery identities are reproducibility checks: the client machinery must
return to standard Bayes at its KL/NLL loss limits, and the server heuristic
must return to the project's log-linear pool at zero robustness
([@eq:robust-identity], [@eq:standard-bayes]).

Under the explicit
shared-support, posterior-log-potential, and fixed-weight assumptions of
[@sec:method-aggregation], that pool specializes Eq. 7's message-combination
term; it does not reproduce the complete source protocol. These deviations are
computed on every build.

**Server recovery:** `robust_aggregate(robustness=0)` versus `log_linear_pool`
([@eq:log-linear-pool]) has maximum difference {{RECOVERY_AGGREGATE_MAXDIFF}}.

**Posterior recovery:** `generalized_posterior(KLD, NLL)` versus closed-form
Bayes has maximum difference {{RECOVERY_POSTERIOR_MAXDIFF}}.

**Divergence recovery:** Rényi divergence versus KL as $\alpha\to 1$ has
maximum difference {{RECOVERY_RENYI_MAXDIFF}}.

**Loss recoveries:** $\beta$-loss versus NLL as $\beta\to 0$ has maximum
difference {{RECOVERY_BETA_MAXDIFF}}, while rcce versus NLL as
$q_{\text{loss}}\to 0$ has maximum difference {{RECOVERY_RCCE_MAXDIFF}}.

Any drift in these limits beyond machine precision would mean the robust
generalization no longer
contains its standard-Bayes client limit and project-local log-linear-pool
server limit ([@sec:results-recovery]) — and would fail the core test suite
before this certificate could render.

The certificate covers the recovery
identity and the client-side result under the cited source theorem's matching
assumptions only;
the server-side `robust_aggregate` heuristic is certified here for its recovery
limit alone, not for any bounded-influence property ([@sec:limitations]).

All code is authored by {{CONFIG_FIRST_AUTHOR}} and licensed under the MIT license.
This is project version {{CONFIG_VERSION}}.
