# External Benchmark Or Domain Pilot

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Owner surface: data provenance, preprocessing, benchmark tests, manuscript limitations

## Rationale

A careful external benchmark would make the project more legible to readers who
trust empirical stress tests more than synthetic sentinel worlds.

## Scope

Current capability: `research_registry.py` declares three CC BY 4.0 UCI
archives—Breast Cancer Wisconsin Diagnostic, Dry Bean, and Banknote
Authentication—with DOI, source URL, archive member, schema, preprocessing, and
pinned SHA-256. `external_data.py` performs hash-checked download and parsing
into a caller-owned cache. `benchmark.py` records train-only standardization,
split and input hashes, held-out log score, calibration, and accuracy for the
naive, robust, and variational server rules. It now also emits dataset-level
nested-seed summaries and recovery/split/archive negative controls. The
installed CLI can execute the pack into an explicit output directory and write a
verifiable receipt.

Residual scope is the preregistered pilot and confirmatory campaign, negative
controls, dataset-level inference, source-bound analysis report, manuscript
tokens, figures/tables, and release receipt.

## Implementation Notes

Preserve the pinned datasets and re-verify their bytes before each fresh run.
Keep preprocessing deterministic and train-split-owned. Record exclusions,
corrupted-cache controls, leakage checks, and failed runs rather than hiding
them. Report all three robustness axes with the Baseline Contract framing
intact: client-side FedGVI carries the cited bounded-influence result only under
the source theorem's matching assumptions; `robust_aggregate` is the sharp
server heuristic with only the recovery-limit guarantee; and
`variational_aggregate` is the objective-backed but conservative rule without
an estimator-level B-robustness proof.

## Acceptance Criteria

- Primary estimand: the per-dataset paired held-out log-score difference at
  locked contamination levels. Accuracy and expected calibration error are
  secondary.
- Independent replication unit: one licensed external dataset (at least three,
  each preprocessed deterministically from its own source).
- Falsifier: source bytes, schema, preprocessing, or split cannot be reproduced,
  or the locked proper-score effect is null or reversed. Either blocks a general
  uncertainty-robustness claim while preserving a publishable null result.
- The benchmark is reproducible and not cherry-picked.
- Licensing and data provenance are explicit.
- Manuscript frames the result as a conditional stress test, not a universal
  deployment claim.

## Verification Probes

- Required artifacts: the three pinned dataset specs, per-run archive/member/
  split hashes, deterministic preprocessing records, complete per-dataset
  effects, negative controls, evidence receipt, and resolved `{{BENCHMARK_*}}`
  tokens in `src/manuscript_variables.py`.
- Required tests: data provenance checks, deterministic preprocessing tests, the
  full source gate, and the manuscript provenance gate.
- Documentation changes: an explicit manuscript limitations passage and a
  provenance/licensing record for each dataset.

## Claim-Boundary Constraints

- No-claim boundary: do not use one benchmark, or the benchmark set, to claim
  general deployment readiness.
- Do not grant `robust_aggregate` any guarantee beyond the recovery limit, do
  not claim an estimator-level B-robustness proof for `variational_aggregate`,
  and do not claim any axis wins on accuracy unless the benchmark statistics
  show it.
- Do not imply true multi-machine federation exists.

## Dependencies

The executable data path exists. Confirmatory execution depends on MAJ-8
calibration, pilot-frozen budgets, and release-time license/attribution review.
