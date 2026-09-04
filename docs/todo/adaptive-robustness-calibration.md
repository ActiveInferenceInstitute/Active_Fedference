# MAJ-8 — Frozen External-Data Calibration Design

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Queue position: first scientific tranche after verified v1.1 publication,
  the MIN-3 documentation handoff, and the one-time private synchronization
- Owner surface: preregistration policy, immutable partitions and seals, pilot
  execution, method-specific calibration results, frozen design digest, and
  confirmatory unlock

## Rationale

Selecting `robustness`, `entropy_weight`, or the confirmatory budget on final
evaluation outcomes leaks target information into the comparison. The existing
generic calibration primitives are prerequisites, not completion evidence for
the accepted three-dataset design. MAJ-8 must preregister the entire selection
policy, keep robust and variational families separate, seal final-test content,
and freeze an immutable source-bound design before MAJ-6 can read final-test
labels, predictions, or scores.

## Scope

### Preregistration tranche

Create `codex/maj8-external-calibration` from exact post-v1.1 public `main` and
commit the policy before pilot outcomes exist. Bind three equally weighted CC
BY 4.0 datasets—WDBC, Dry Bean, and Banknote Authentication—without row-count
weighting. Use deterministic class-stratified fit/calibration/final-test splits:

| Dataset | Fit | Calibration | Sealed final test |
| --- | ---: | ---: | ---: |
| WDBC | 318 | 80 | 171 |
| Dry Bean | 7,622 | 1,906 | 4,083 |
| Banknote | 768 | 192 | 412 |

Use five clients. The primary contaminated conditions use clients 0 and 1 at
rates 0.5 and 1.0; the clean guardrail uses no contaminated clients at rate
0.0. A four-of-five, rate-1.0 stress control uses a fixed 32-seed schedule and
is descriptive/falsifying only: it covers all three datasets and the two locked
method-versus-naive contrasts, for six stress contrasts, with clients 0–3
contaminated and client 4 honest. Partition, shard, and corruption random
streams are independent and receipt-bound. Matched corruption uniforms and
replacement labels make rate 0.5 nested within rate 1.0 across methods.

Use dataset order WDBC = 0, Dry Bean = 1, Banknote = 2 and the accepted seed
formulas:

```text
partition_seed(d)      = 310_000_000 + 10_000*d
pilot_seed(d, j)       = 320_000_000 + 10_000*d + j
confirmatory_seed(d,j) = 330_000_000 + 10_000*d + j
bootstrap_seed(cell)   = 340_000_000 + 100*cell_index
```

Pilot indices are `j=0..31`; the only permitted extension is `j=32..63`.
Confirmatory indices begin from the disjoint confirmatory formula. The policy
must also assign deterministic non-overlapping shard and corruption substreams
before pilot execution and expand every actual seed into the frozen artifact.

The canonical zero-based primary `cell_index` is the lexicographic product with
dataset outermost, method next, and contaminated rate innermost:
`cell_index = 4*d + 2*m + r`, where `d` follows WDBC/Dry Bean/Banknote,
`m=0` for robust and `m=1` for variational, and `r=0` for rate 0.5 and `r=1`
for rate 1.0. Thus the 12 primary cells occupy indices 0 through 11 without an
order inferred from a dictionary or execution scheduler. Let tranche `s=0`
denote the 32-seed stage and `s=1` the 64-seed stage. Clean and stress cell
indices are each `c_clean = c_stress = 2*d + m`, occupying 0 through 5. Freeze
the complete, pairwise-disjoint bootstrap namespaces as:

```text
s_u_bootstrap_seed(c_primary)       = 340_000_000 + 100*c_primary
eligibility_bootstrap_seed(s,m,d)   = 341_000_000 + 100_000*s + 10_000*m + 100*d
stability_bootstrap_seed(s)         = 342_000_000 + 100_000*s
primary_interval_seed(c_primary)    = 343_000_000 + 100*c_primary
primary_null_seed(c_primary)        = 343_000_000 + 100*c_primary + 1
clean_interval_seed(c_clean)        = 344_000_000 + 100*c_clean
clean_null_seed(c_clean)            = 344_000_000 + 100*c_clean + 1
stress_interval_seed(c_stress)      = 345_000_000 + 100*c_stress
```

The accepted `bootstrap_seed(cell)` formula is the first line and is reserved
for the 12 pilot `s_U` calculations. A clean-eligibility stream generates one
resample-index matrix shared by every candidate in that method/dataset/tranche.
A stability stream generates one dataset-first hierarchy of index draws shared
by every robust and variational candidate being compared in that tranche.
Confirmatory ordinary-interval, null-centered-tail, and descriptive-stress
streams remain separate. No code may derive bootstrap randomness from a
dictionary, execution order, candidate fingerprint, or a reused stream. The
policy resource and frozen design expand every named seed and schema validation
proves pairwise non-overlap.

Freeze these candidate families:

- naive: one untuned `AggregationConfig(method="naive")`;
- robust: `robustness` in `0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5`;
- variational: `robustness` in `0.1, 0.25, 0.5, 1.0, 1.5, 2.5` crossed
  with `entropy_weight` in `0.3, 0.5, 1.0, 2.0`;
- one variational recovery candidate at `robustness=0.0` and
  `entropy_weight=1.0`.

All iterative candidates use `max_iter=512` and `tol=1e-9`; variational uses
`multistart=True`. Any fallback or nonconvergence invalidates the pilot rather
than silently removing a configuration.

The primary score is per-observation `log(p_true)` in nats. Freeze SOEI 0.010,
MCSE target 0.005, clean noninferiority margin -0.010, one global robust and one
global variational selection, exact fingerprint tie-breaking, and 5,000
hierarchical bootstrap resamples. At a given completed pilot tranche, determine
clean eligibility before any selection-stability resampling. For each candidate
and each dataset, form paired seed-level candidate-minus-naive clean-calibration
log-score differences, resample those paired rows with replacement 5,000 times,
and take the 5th percentile of the resampled means as the one-sided 95% lower
bound using `numpy.quantile(..., 0.05, method="linear")`. A candidate is eligible
only if this bound is strictly greater than -0.010 on every dataset; equality
fails. This pilot-screening rule has no multiplicity or confirmatory-claim
interpretation. If either method family has no eligible candidate, stop without
freezing a design. Compute the eligible set once per completed tranche, retain
it unchanged in every stability resample for that tranche, and select only
within that set. Eligible candidates maximize the equally weighted mean over
the three datasets and two contaminated rates. Each selected fingerprint must
win at least 80% of resamples.

Selection stability uses a dataset-first hierarchical bootstrap. For each of
5,000 resamples, draw three dataset strata with replacement from the fixed
three-dataset pack; within each selected stratum, draw that dataset's pilot
seeds with replacement and reduce the nested client/rate observations to the
declared seed-level scores before candidate selection. Each selected dataset
occurrence contributes one equal-weight stratum estimate, so row count never
changes the pack estimand. Resampling only seeds while holding the three dataset
strata fixed is a different procedure and is not accepted for this stability
gate. Dataset resampling is a preregistered stability perturbation of this fixed,
equally weighted three-dataset pack; it does not define a dataset-population
estimand or support population-level generalization.

### Pilot and freeze tranche

The sequence is deterministic. First compute `E32`, the clean-eligible sets from
exactly the first 32 pilot seeds, then compute `S32`, the stability selection
over those fixed sets. If either family has no member in `E32`, stop. If both
selected fingerprints win at least 80% in `S32`, freeze them. Otherwise use the
single preregistered extension through 64 seeds, recompute clean eligibility
once as `E64` from all 64 seeds—`E64` supersedes rather than augments `E32`—and
then compute `S64` over the fixed `E64` sets. If either family has no member in
`E64`, or if either selected fingerprint remains below 80% in `S64`, stop
without a frozen design. Eligibility is never recomputed inside a stability
resample. Equality at the 80% threshold passes; no third tranche is permitted.

Fix the selected configuration before budget estimation; the budget bootstrap
must not reselect it. For each of the 12 primary cells, form its selected-method-
minus-naive seed-level paired differences and require at least two finite rows.
Compute the observed sample standard deviation with `numpy.std(..., ddof=1)`.
Using that cell's frozen bootstrap seed, draw 5,000 same-size paired-row
resamples with replacement, compute each resample's sample standard deviation
with `ddof=1`, and set `s_U` to
`numpy.quantile(sd_draws, 0.95, method="linear")`. A finite zero variance is
permitted; fewer than two rows or any nonfinite input, draw, or bound stops the
freeze. Let `N_raw = ceil((max s_U / 0.005)^2)`, round upward to a multiple of
16, apply a floor of 32 and cap of 256, and use the same `N` for every primary
cell. Stop without opening final-test rows if the upward-rounded uncapped
requirement exceeds 256.

Create a typed `FrozenCalibrationDesign` and separate robust/variational
calibration wrapper. Bind policy/version, datasets/partitions, candidates,
selected full configurations, expanded partition/shard/corruption/pilot/
confirmatory/bootstrap seeds, final `N`, non-circular registry-input
fingerprint, pilot source commit, lock digest, archive/member/split/final-test
seals, and canonical design digest. Package the policy/design resources and
require an exact digest-checked design reference in `ExperimentSpec`;
`confirmatory_ready=True` alone cannot unlock execution.

Use fresh write-once preregistration, pilot, and frozen-design namespaces. Pilot
and confirmatory CLI profiles reject tuning overrides.

## Implementation Notes

- Expand every realized seed in the design artifact. Pilot and confirmatory
  schedules are disjoint; no pilot row enters a confirmatory interval.
- Seal final-test indices and content hashes before the pilot. Pilot code can
  read fit/calibration partitions and seal digests, but cannot load final-test
  labels, predictions, or scores.
- Preserve every candidate, diagnostic, failure, fallback, and convergence
  record. Null or reversed pilot evidence remains reviewable.
- Keep design and registry hashing non-circular: the registry-input fingerprint
  excludes the eventual design reference; the registry may then bind the final
  design digest.
- The preregistration and pilot/freeze tranches are separate public PRs with
  new confidentiality audits and normal merge commits. After the
  preregistration PR merges and its public-main workflow passes, create
  `codex/maj8-frozen-design` from that exact merged public `main`; never reuse
  or rewrite the preregistration branch for the pilot and freeze.

## Acceptance Criteria

- Primary estimand: equal-weight mean calibration log score in nats per
  observation over three datasets and two contaminated rates, conditional on
  per-dataset clean noninferiority.
- Independent replication unit hierarchy: dataset is the external
  generalization unit. The pilot seed is the paired Monte
  Carlo/resampling unit nested within dataset, with clients, contamination
  rates, configurations, and observations nested within seed/cell. The
  dataset-first hierarchical bootstrap samples datasets and then their seeds;
  it does not treat seeds as independent datasets.
- Separate robust and variational fingerprints each satisfy the 80% stability
  threshold after at most the one allowed extension.
- Every source, license, archive, member, schema, preprocessing, split, seal,
  seed, configuration, lock, and source-commit digest verifies.
- The pilot has zero fallback and nonconvergence events and cannot access
  final-test rows.
- Final `N` is derived exactly from `N_raw`, rounded upward to a multiple of 16,
  and subject to the floor of 32. If that upward-rounded required `N` exceeds
  256, the campaign stops rather than clamping the budget and proceeding.
- Required acceptance evidence: policy resource, preregistration receipt,
  complete pilot table, stability analysis, clean-eligibility table, variance/
  `N` derivation, retained failures, frozen design and digest, installed-package
  round trip, no-peeking record, public PR checks, and merge SHA.

## Verification Probes

- Schema, source-license, archive/member hash, recovery, deterministic split,
  class allocation, corruption nesting, and seed-stream separation tests.
- No-peeking tests that make final-test access fail before a valid design
  unlock, plus design/registry tamper and digest-mismatch controls.
- Candidate-order permutation and exact-tie tests.
- Canonical primary-cell index/seed tests and a negative control that detects
  seed-only resampling in place of the required dataset-first hierarchy.
- Pilot/confirmatory override rejection and package-resource membership from
  wheel and source distribution.
- Write-once namespace collision and pilot/confirmatory seed-overlap controls.
- Full source, scientific, publication, package, installed-artifact, history,
  hosted PR, and post-merge gates.

## Claim-Boundary Constraints

- Falsifier: source/data/split/seal drift, test peeking, seed overlap, unstable
  selection after the extension, required `N` above 256, fallback,
  nonconvergence, invalid recovery/corruption controls, or a circular digest
  stops design freeze and MAJ-6.
- Prohibited claims: calibration does not make `robust_aggregate`
  objective-backed, does not prove estimator-level B-robustness for normalized
  consensus, does not establish deployment validity, and does not count as a
  confirmatory external-data result.
- A null or stopped pilot is publishable integrity evidence but cannot unlock
  final-test execution.

## Dependencies

Requires verified v1.1 GitHub/Zenodo publication, completion of MIN-3's public
documentation and repository-discovery handoff, and the one-time private
synchronization, plus the existing application/configuration/provenance/receipt
contracts, the scoped MAJ-1 vocabulary boundary, pinned external-data
specifications, and fresh scientific namespaces. MAJ-6 depends on the exact
frozen design produced here.
