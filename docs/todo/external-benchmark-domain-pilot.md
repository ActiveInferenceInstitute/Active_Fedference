# MAJ-6 — Three-Dataset Confirmatory Campaign

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Queue position: only after the exact MAJ-8 frozen design is merged and
  registry-verified
- Owner surface: sealed final-test execution, paired inference, outcome
  classification, source-bound report/receipt, figures, manuscript, and claim
  ledger

## Rationale

The pinned external-data loader and benchmark mechanics do not constitute a
confirmatory result. MAJ-6 asks whether the two separately calibrated server
methods improve held-out categorical log score against naive pooling across
WDBC, Dry Bean, and Banknote Authentication under the exact accepted design.
The campaign must preserve dataset as the highest external unit, retain null or
harmful outcomes, and prevent tuning after final-test rows become available.

## Scope

Create `codex/maj6-confirmatory` from the exact public `main` produced by the
normally merged `codex/maj8-frozen-design` PR and its green post-merge
workflow. Before reading final-test labels, reverify the source commit,
`uv.lock`, packaged policy, frozen-design digest, registry reference,
dataset/archive/member/schema/preprocessing/split/final-test hashes, seed
disjointness, solver policy, and a new write-once confirmatory namespace.
Reject every tuning or configuration override.

Execute the frozen common `N` for the 12 primary contrasts and the six clean
guardrail contrasts. Compute descriptive accuracy and expected calibration
error from those same rows; they do not receive a separate seed budget. The
families are:

- the primary family of 3 datasets × 2 locked methods × 2 contaminated rates,
  yielding 12 paired log-score contrasts against naive and Holm family-wise
  correction at alpha 0.05;
- a separate six-comparison clean one-sided noninferiority family against
  -0.010 nats per observation, with its own Holm correction;
- secondary descriptive accuracy and expected calibration error from the
  primary and clean rows.

Separately execute the four-of-five, rate-1.0 stress condition at exactly the
first 32 frozen confirmatory seeds per dataset for each of the two locked
method-versus-naive contrasts: 3 datasets × 2 methods = 6 descriptive stress
contrasts. Contaminate clients 0, 1, 2, and 3; client 4 remains honest. Bind its
own named corruption/bootstrap substreams and do not extend it to the common
`N`, include it in either Holm family, or use it in a positive pack-wide claim.

Use seed-cluster bootstrap intervals and seed-level MCSE. Report each primary
and clean cell's mean paired log-score difference, 95% interval, MCSE, common
`N`, adjusted decision, fallback count, nonconvergence count, and complete
design/source/lock/archive/member/partition hashes; report the corresponding
descriptive fields with `n=32` and no adjusted decision for stress. Do not
publish a pooled
dataset-population p-value. For every primary and clean cell let
`n = delta.size`, require `n == frozen_common_N`, and compute MCSE exactly as
`numpy.std(delta, ddof=1) / numpy.sqrt(n)`. Fewer than two rows, a row-count
mismatch, or any nonfinite contrast or MCSE is a hard stop. Every primary and
clean cell must have finite MCSE at most 0.005; equality passes and a value above
it fails. For each 32-seed stress contrast, report the same sample-SD-over-root-
`n` MCSE and the frozen descriptive percentile interval, but do not apply the
0.005 ceiling, a null-tail test, Holm correction, or a positive-claim gate.

The preregistered decision algorithm is fixed before final-test access. For a
primary cell, let the method-minus-naive paired seed contrast be $\Delta$ and
test the one-sided boundary null $H_0:\Delta\leq0$ against $H_1:\Delta>0$.
For a clean guardrail, test $H_0:\Delta\leq-0.010$ against
$H_1:\Delta>-0.010$. Within each dataset/cell, draw 5,000 bootstrap samples of
the paired seed rows with replacement. The reported two-sided interval is the
2.5th and 97.5th percentile of those ordinary seed-cluster means. For null
boundary $\delta_0$, construct the empirical null sample by centering the
paired seed contrasts at that boundary,
`delta_null = delta - mean(delta) + delta_0`, and resample those centered rows.
The one-sided tail value is
`(1 + count(mean(delta_null_bootstrap) >= mean(delta))) / (5000 + 1)`;
equality remains in the null-compatible tail. This null-centered bootstrap,
not the fraction of ordinary bootstrap means crossing the null, supplies the
values used by Holm. Apply Holm step-down correction separately to the 12
primary values and six clean values at alpha 0.05, sorting exact p-value ties
by the canonical dataset/method/rate cell key. Stop rejection at the first
failed Holm threshold. A primary statistical rejection is necessary but not
sufficient: the conditional-positive gate also requires observed mean
improvement at least 0.010 nats per observation, the MCSE/solver/provenance
gates below, and all required clean noninferiority decisions. These rules are
part of the frozen policy/design artifact; changing any rule requires a new
design version and unopened namespace.

Commit the raw report and receipt before adding result-dependent manuscript
language. Only then generate the 12-cell primary forest plot, separate clean-
noninferiority figure, solver/provenance matrix, dataset/rate small multiples,
clearly marked stress-control visual, exact-value tables, long descriptions,
captions, tokens, claim-ledger entries, and bounded outcome prose.

## Implementation Notes

- Load complete `robust_config` and `variational_config` objects from the frozen
  design. Compatibility scalar arguments remain smoke-only and cannot enter
  this campaign.
- The current generic benchmark summary's singleton-MCSE convention remains a
  compatibility smoke/pilot behavior and must not be called by the strict
  confirmatory runner. The confirmatory path owns the exact row-count, finite-
  value, `ddof=1`, and common-`N` checks above.
- Preserve every seed, failure, fallback, nonconvergence, recovery result, and
  hash mismatch. Never delete a row to improve the comparison.
- Keep fit/calibration/final partitions permanent. Opening sealed rows is a
  one-way campaign event and must be receipt-bound.
- Classify each result only with the frozen rules: pack-wide conditional
  positive, dataset/rate-only passing subset, harmful reversal, practical
  fixed-cell null, or inconclusive.
- Merge an integrity-valid null, harmful, or inconclusive result normally. Stop
  after the public confirmatory merge and present the observed result before
  selecting a scientific version, tag, DOI, GitHub release, or Zenodo record.

## Acceptance Criteria

- Primary estimand: per-observation paired held-out categorical log-score
  difference in nats between one locked method and naive pooling, separately
  for each dataset and contaminated rate.
- Independent replication unit hierarchy: dataset is the external
  replication/generalization unit. A confirmatory seed is the paired Monte
  Carlo and seed-cluster resampling unit nested within dataset;
  clients, rates, methods, and observations are nested further. The estimand is
  the fixed intersection of the three named datasets, never a pooled sample or
  a population-level dataset inference.
- Every primary cell reports its estimate, seed-cluster interval, MCSE, common
  `N`, Holm-adjusted disposition, fallback/nonconvergence counts, and complete
  provenance.
- A method earns a pack-wide conditional positive claim only when all six of
  its contaminated cells have MCSE at most 0.005, zero fallback/nonconvergence,
  intended-direction Holm results, mean improvement at least 0.010, and full
  provenance, and all three clean cells have MCSE at most 0.005, zero fallback/
  nonconvergence, complete provenance, and pass noninferiority. All 18 primary
  and clean cells must contain exactly the frozen common `N` paired rows.
- Mean at most -0.010 with an interval excluding zero is a harmful reversal; an
  interval wholly inside [-0.010, +0.010] is a practical fixed-cell null; all
  other non-passing cells are inconclusive.
- Required acceptance evidence: pre-open verification receipt, immutable raw
  result/receipt commit, complete primary and guardrail tables, secondary and
  stress records, source-bound figures/exact-value tables/long descriptions,
  claim-ledger disposition, public PR checks, merge SHA, and write-once
  namespace digest.

## Verification Probes

- Source, license, archive/member, schema, preprocessing, split, seal, registry,
  design, lock, and seed-disjointness verification before final-test access.
- Tuning/profile override rejection and design/config tamper controls.
- Matched-corruption nesting, recovery, fallback, nonconvergence, missing-data,
  moved-root, receipt-tamper, and namespace-reuse controls.
- Holm implementation, seed-cluster interval, MCSE, classification-rule, exact-
  value fallback, caption, figure, token, and no-pooled-inference tests.
- Primary/clean null-boundary, 5,000-resample percentile interval, bootstrap
  tail-value, equality/zero treatment, canonical tie-order, Holm stop-rule, and
  separate 0.010 practical-effect-gate tests.
- Full source, scientific, publication, installed-package, confidentiality,
  hosted PR, and post-merge gates.

## Claim-Boundary Constraints

- Falsifier: any source/design/hash mismatch, calibration-confirmatory overlap,
  unavailable or invalid dataset, failed recovery/split/schema/archive/
  corruption control, fallback, nonconvergence, namespace reuse, or final-test
  access before the pre-open receipt stops the scientific claim.
- Prohibited claims: no cell, caption, accuracy result, ECE result, or stress
  control can rescue a failed primary family; no dataset subset supports a
  pack-wide claim; no dataset pack establishes universal deployment validity,
  estimator-level B-robustness, physical multi-host operation, or source-
  protocol identity.
- Null, harmful, and inconclusive outcomes remain publishable when integrity
  gates pass.

## Dependencies

Requires the exact merged MAJ-8 frozen-design digest, unchanged public source
and environment lock, pinned CC BY 4.0 datasets, validated application and
receipt contracts, and a fresh write-once namespace. It cannot begin from a
standalone `confirmatory_ready` Boolean. Any scientific release decision is
downstream of the observed merged result.
