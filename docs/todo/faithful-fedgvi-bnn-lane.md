# Faithful FedGVI BNN Lane

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Queue position: post-v1.1 scientific lane that may proceed independently of
  MAJ-8/MAJ-6 once its own parity and resource gates are fixed
- Owner surface: `VariationalMLP`, required PyTorch lanes, manuscript baseline section

## Rationale

A protocol-faithful, portable FedGVI neural lane is still needed to test whether
the client-side FedGVI robustness axis holds beyond small tabular models. The
source protocol is not a moment-matching server: the global posterior is the
prior times client site factors; each client trains against its cavity; and the
server replaces that client's factor with posterior-minus-cavity natural
parameters.

## Scope

The open scope is split:

- **MAJ-2A, local portable replication:** extend the verified synthetic CPU/MPS
  protocol to the source loss/divergence and cavity-conditioned client
  optimizer on FashionMNIST, MNIST, and KMNIST; pilot and freeze an M4 budget;
  execute the locked comparison with round-level checkpoints, resume
  equivalence, CPU references, and explicit MPS fallbacks.
- **MAJ-2B, external source-scale replication:** preserve the exact source
  configuration but execute it only on appropriate CUDA hardware. It is
  declarative and non-blocking for the local release.

The named profiles are `smoke`, `m4_confirmatory`, and `source_5090`. Smoke is
correctness-only. The portable and source-scale profiles must preserve the
pinned source revision, effective split-seed indexing, three-client setup,
contamination schedule, server rounds, early-stopping semantics, and predictive
sampling contract recorded by the protocol-parity artifact. Pilot only the
local training budget; never reinterpret a declarative source profile as an
executed result.

Outcome-bearing MAJ-2A implementation remains blocked until a versioned pilot
design freezes the numeric smallest effect of interest in held-out log-score
units, the seed-level MCSE stopping target, the complete primary/secondary
multiplicity family and adjustment rule, and the maximum rounds, local epochs,
posterior samples, seeds, wall time, and device compute budget. “Pilot-frozen”
placeholders in the registry are stop markers, not authority to choose those
values while inspecting confirmatory outcomes.

## Implementation Notes

Keep the NumPy logistic-regression baseline as the default proof surface. Torch
remains behind the `bnn` package extra and must not become a dependency of the
NumPy/SciPy core. The point-mass MLP is a control, not a posterior substitute.
Missing Torch is a setup failure for a certified BNN evidence run, not a skipped
claim. Never describe the cavity/site-factor server as moment matching.

## Acceptance Criteria

- Primary estimand: the paired held-out log-score difference between FedGVI and
  a matched PVI/NLL baseline at locked contamination levels. Accuracy and
  expected calibration error are secondary.
- Independent replication unit: an independently seeded end-to-end BNN run
  (data split, initialization, and training trajectory), not a per-batch or
  per-epoch measurement within one run.
- The tracked MAJ-2A design records the numeric SOEI and MCSE target, exact
  comparison family/correction, seed schedule, compute ceiling, and a stop
  disposition before `m4_confirmatory` can execute.
- Falsifier: if the paired proper-score interval includes zero, or its sign
  reverses versus the preregistered direction, at the declared contamination
  levels, the robustness claim for the BNN lane fails and is not published.
- Same-device repeatability and CPU/MPS directional or statistical parity
  within stated tolerances; cross-device bit identity is not required.
- Every unsupported MPS operation produces either a receipt-bearing CPU
  fallback or a hard failure.
- Checkpoint/resume and interrupted-run recovery match the uninterrupted run.
- No skipped PyTorch tests support manuscript claims; the required lane must
  pass before broader BNN claims are added.
- `VariationalMLP` recovery limits remain green: sigma-to-zero recovery,
  analytic KL agreement, Gibbs non-negativity, and ELBO term separation.

## Verification Probes

- Required artifacts and tests: `requires_torch` tests, the required Torch
  validation profile, protocol-parity artifact, device/fallback receipt, CPU
  reference, checkpoint chain, negative controls, and a paired-statistics sweep
  artifact under the declared configuration.
- Regenerated baseline figure and tokens (via `src/manuscript_variables.py`)
  only after the required lane is green.
- Documentation changes: update the manuscript baseline section and this lane's
  Claim-Boundary Constraints to state the BNN result and its estimand once the
  sweep lands.
- Required acceptance evidence: source-parity matrix, pilot budget receipt,
  complete paired run table, device/fallback and checkpoint chains, locked
  statistics, source-bound figures/exact-value tables, public PR checks, and
  merged public-main SHA.
- Negative controls reject a confirmatory run whose design retains a placeholder
  threshold/budget or whose runtime exceeds or mutates the frozen compute plan.

## Claim-Boundary Constraints

- Do not describe the NumPy logistic-regression baseline as a full FedGVI BNN
  reproduction.
- No-claim boundary: the bounded-influence guarantee belongs to the client-side
  FedGVI axis under its stated objective. This lane may not transfer that
  guarantee to `robust_aggregate`, may not claim an estimator-level
  B-robustness proof for normalized consensus, and may not claim the BNN lane
  wins on accuracy unless the paired statistics show it.

## Dependencies

- MAJ-2A requires PyTorch availability and explicit required-lane validation.
- MAJ-2A confirmatory execution remains blocked until the pilot-derived numeric
  design-freeze artifact passes schema, digest, and no-outcome-peeking checks.
- Existing cavity, factor-replacement, variational-family, determinism, and
  checkpoint primitives are prerequisite contracts recorded in source tests
  and `ISA.md`; they are not active TODO subitems.
- MAJ-2B additionally requires external CUDA resources and never blocks the
  M4-portable release.
- Primary estimand, unit, falsifier, and no-claim boundary trace to the
  [scholarship-indexed phase plan](scholarship-and-phase-plan.md).
