# Faithful FedGVI BNN Lane

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Owner surface: `VariationalMLP`, required PyTorch lanes, manuscript baseline section

## Rationale

A protocol-faithful, portable FedGVI neural lane is still needed to test whether
the client-side FedGVI robustness axis holds beyond small tabular models. The
source protocol is not a moment-matching server: the global posterior is the
prior times client site factors; each client trains against its cavity; and the
server replaces that client's factor with posterior-minus-cavity natural
parameters.

## Scope

Current capability: `bnn_fedgvi.py` implements the model-agnostic diagonal-
Gaussian site table, cavity, factor replacement, round counter, and atomic
checkpoint round trip. `VariationalMLP` provides the mean-field variational
family, including export/load of diagonal-Gaussian cavities and a
cavity-conditioned local optimizer. `torch_bnn.py` supplies explicit CPU/MPS
device selection, determinism, and fallback receipts, and the executable
synthetic pilot records held-out log score plus checkpoint/resume equivalence.
The parity matrix is pinned to FedGVI source revision
`5440352890037a81218285b8f4de81090861e9df`.

Residual scope is split:

- **MAJ-2A, local portable replication:** extend the verified synthetic CPU/MPS
  protocol to the source loss/divergence and cavity-conditioned client
  optimizer on FashionMNIST, MNIST, and KMNIST; pilot and freeze an M4 budget;
  execute the locked comparison with round-level checkpoints, resume
  equivalence, CPU references, and explicit MPS fallbacks.
- **MAJ-2B, external source-scale replication:** preserve the exact source
  configuration but execute it only on appropriate CUDA hardware. It is
  declarative and non-blocking for the local release.

The named profiles are `smoke`, `m4_confirmatory`, and `source_5090`. Smoke is
correctness-only. Source audit of the pinned FashionMNIST shell found an
important indexing detail: it executes run indices `[1, 2, 3, 4, 5]` against
the six-entry table `[42, 676, 93, 215, 318, 242]`, so the effective split seeds
are `[676, 93, 215, 318, 242]`. The registry preserves all three fields rather
than silently copying the table's first five entries. The portable profile
retains those effective seeds, three clients, contamination rates
`[0, 0.1, 0.2, 0.4, 0.6]`, and 25 server rounds, while its local training budget
must be frozen by pilot. The external profile preserves the source ceiling of
2,500 local epochs with ELBO early-stopping patience 10, 200
posterior-predictive samples, and 10 ELBO samples.

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
- MAJ-2B additionally requires external CUDA resources and never blocks the
  M4-portable release.
- Primary estimand, unit, falsifier, and no-claim boundary trace to the
  [scholarship-indexed phase plan](scholarship-and-phase-plan.md).
