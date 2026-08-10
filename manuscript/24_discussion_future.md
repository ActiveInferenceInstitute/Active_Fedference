## Future work: testing the open boundaries {#sec:future}

The staged research program follows directly from the boundaries in
[@sec:limitations]. It separates public-library reproducibility, server theory,
portable source-protocol work, task generalization, and deployment validation
so that success in one lane cannot silently certify another.

The order is deliberate. Simulation-study guidance recommends declaring the
estimand, data-generating mechanism, and Monte Carlo precision before treating
replication as evidence [@morris2019simulation], while Monte Carlo error should
be reported separately from an interval or a hypothesis test
[@koehler2009mcse]. For nested agents, trials, and seeds, the resampling unit
must respect the dependence structure [@loy2021lmeresampler]. Accordingly, the
phase plan records a primary unit and a falsifier for each extension; a larger
sample or more elaborate diagram is not itself a stronger claim.

The implementation registry also separates smoke, pilot, and confirmatory
profiles. Pilot worlds select budgets and calibration settings but never enter
confirmatory intervals or headline values. Each completed run must bind its
source bundle, configuration, dataset bytes, device, checkpoints, outputs, and
completion status into a verifiable receipt. A negative or null scientific
result remains a valid citable outcome when those implementation and provenance
gates pass; it blocks the intended positive claim, not the software release.

A separate Friston protocol lane will resolve a machine-readable parity matrix
before reconstructing source experiments in Python. Until every required source
parameter, routine, unit, and estimand is recovered, that lane will be described
as a paper-constrained reconstruction rather than an exact replication.

## Make the sharp server heuristic variational {#sec:future-server}

The asymmetry between the robustness axes ([@sec:robustness-axes]) is the most
consequential open problem. The client-side $\beta$/rcce update carries a
derived, loss-specific bounded-influence result under the matching assumptions;
the sharp server-side `robust_aggregate`
carries only its recovery limit
([@eq:robust-identity]); and the new variational aggregator
([@sec:method-variational], [@sec:supp-variational]) supplies a *rigorous*
server-side rule — exact block updates descending the stated free energy
[@eq:agg-free-energy] through non-increasing block updates, with a proven raw
effective-weight bound and the same
recovery corner. What it costs is conservatism: it is the maximum-entropy
consensus, not the sharp accuracy-maximizer. The remaining open problem is
therefore sharper than before — to write down a generalized variational objective
in the FedGVI family [@mildner2025fedgvi], informed by recent closed-form GVI
characterizations [@nguyen2026closedformgvi], logarithmic-pool weighting theory
[@carvalho2023logpooling], and robust divergence-weighted federated aggregation
[@li2022gammafl], whose closed-form minimizer is competitive with the empirical
reweighting across declared contamination regimes. That would combine axis 3's
effective-weight bound with axis 2's empirical sharpness in one server rule. The
recovery corner and the variational objective together supply two boundary
conditions any such unification must satisfy.

Any empirical choice of `robustness` or `entropy_weight` will be made on
separate calibration worlds using a proper log score, then frozen before
confirmatory evaluation. This guards against selecting an apparent leader
with evaluation truth and preserves null or reversed confirmatory outcomes.
The comparison family will include logarithmic and linear pools, the current
heuristic, the variational family, and a centered-log-ratio geometric-median
control; none inherits a parameter-space robust-federated-learning guarantee
merely by operating on beliefs.

## Promote the baseline to original FedGVI scale {#sec:future-scale}

The bounded-influence result is anchored locally by the small NumPy
logistic-regression baseline ([@fig:bnn-robustness]). Promoting it to the
GPU-scale Bayesian-neural-network experiments of the source paper
[@mildner2025fedgvi] — the experiments deferred here ([@sec:limitations]) — would
test whether the per-client robustness curve holds at the model capacity and
contamination regimes where federated learning [@mcmahan2017communication]
actually operates, and would connect the discrete-POMDP result to the
partitioned-VI line [@ashman2022partitioned; @bui2018partitioned]. The planned
comparison would also require posterior-parameterization parity with Bayesian
neural-network work, rather than treating the current deterministic point-mass
MLP as a posterior [@mildner2025fedgvi].

The portable lane preserves the source protocol's site factors, client cavity,
and factor-replacement update in natural coordinates. It distinguishes a
locally budgeted CPU/MPS profile from an exact source-scale CUDA profile that
remains external until suitable hardware is available. FashionMNIST anchors
protocol parity, while MNIST and KMNIST test portability. A separate
source-bound tabular pack will report proper-score effects per licensed dataset,
with training-only preprocessing and byte-, split-, and license-level
provenance; nested seeds will not be treated as independent datasets.

## Extend hierarchical federation beyond the current stack {#sec:future-hierarchical}

The governing caveat here is that added depth must be shown to *earn*
generalization, not merely to execute: on the current sentinel task the deeper
stacks match rather than beat the flat baseline on location
([@sec:limitations-scope]). The extension therefore has two distinct fronts —
carrying the recovery contract to deeper stacks, and finding a task family in
which depth actually pays.

The 2-level hierarchical POMDP ([@sec:results-hierarchical]) couples location
inference to a single global context. The generic N-level architecture
(:func:`fedference.pomdp.build_nlevel_world`) has already been exercised with a
3-level stack ([@sec:results-3level], [@sec:supp-3level]): a meta-context
variable (L3) gates the context prior (L2) which in turn gates the location
prior (L1). The empirical-prior top-down messages ([@eq:l3-to-l2-message],
[@eq:l2-to-l1-message]) remain valid variational steps at every depth, and the
log-linear-pool federation at each level is bit-identical to the in-process
result (Proposition \ref{prop:federation-bit-identity}). The natural next question is whether the
limit-as-proof contract of [@sec:limitations] survives still deeper hierarchies:
does the recovery corner (context prior → uniform) remain checkable to machine
precision when the L2 → L1 message is itself a function of an L3 belief coupled
to an L4 belief, and can message-passing engines such as RxInfer
[@bagaev2023rxinfer] carry the generic alternating-minimization at scale?
Structure learning already answers the dual question — how *deep* the model
should be — for the top level: hierarchical Bayesian model reduction
([@sec:results-hierarchical-bmr]) prunes a non-gating meta-context and keeps an
informative one, so the depth is decided by the evidence rather than assumed.
Extending that per-level reduction to a full breadth-and-depth search over the
generic N-level stack is the natural continuation.

The next task family is deliberately controlled rather than merely deeper:
partially observable Four Rooms and Key-Door will compare flat, oracle, learned,
shuffled, and non-gating hierarchies at matched horizons and compute. Task is
the higher-level replication unit. A gain in only one task will remain
task-specific, and the larger campaign will not begin until the hybrid
representation recovery gates pass.

## Move from process transport to true multi-machine federation {#sec:future-transport}

Promoting federation from the current queue-backed, single-machine process and
loopback-socket helpers to cross-host workers would retire the remaining
deployment caveat of [@sec:limitations] while preserving the bit-identical
consensus property proved in Proposition \ref{prop:federation-bit-identity}.
The `federation/` package and
federation tests already establish the API contract: a server collects
serialized beliefs from {{FEDERATION_N_WORKERS}} worker channels, fuses them
with `robust_aggregate` at robustness $c = {{FEDERATION_ROBUSTNESS}}$, and
broadcasts the consensus back over response channels, with bit-identity verified
at {{FEDERATION_BIT_IDENTICAL}}. The loopback socket path adds optional
pre-shared-key frame integrity and file-backed digest-verified replay validation.
The next systems step is an explicitly local Docker multi-node emulator with
mTLS by default, HMAC compatibility, checkpoint/restart, and reproducible
message-fault controls. It is not physical multi-host evidence. That later
claim requires receipts from distinct hosts, deployment-grade key management,
timeout policy, and long-running orchestration across process restarts, but it
does not require changing the mathematics. Secure aggregation,
differential privacy, and Byzantine tolerance are separate future threat models;
transport integrity alone would not establish any of them
[@blanchard2017krum; @pillutla2022robust].

## Move beyond categorical state spaces {#sec:future-continuous}

This work is discrete-categorical, matching the community's worked POMDP
[@dacosta2020active]. A first step is already in place: the closed-form Gaussian
KL and Rényi divergences of [@sec:supp-extended] show the divergence family — and
its $\alpha\to1$ recovery — carries over verbatim to Gaussian beliefs, scoped out
of the categorical experiments. Continuous-state Gaussian generative models would
test whether the limit-as-proof contract survives the move off categorical state
spaces — whether the recovery corner remains checkable to machine precision when
the belief simplex is replaced by a Gaussian belief, and whether message-passing
engines such as RxInfer [@bagaev2023rxinfer] can carry the robust update at
scale. Extending the structure-learning study [@smith2020active; @friston2011post]
into continuous models would, in parallel, test whether robust belief fusion and
robust *structure* fusion compose.

A minimal executable fixture now gates a discrete dynamics context over
continuous position and velocity, Gaussian observations, and bounded actions.
It is a representation and recovery surface, not confirmatory task evidence.
The full study must add discrete-only, continuous-only, and oracle-context
controls, singular-covariance and outlier checks, and held-out
posterior-predictive scoring before supporting a hybrid-task claim.
