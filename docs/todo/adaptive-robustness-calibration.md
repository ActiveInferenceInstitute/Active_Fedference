# Adaptive Robustness Calibration

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Owner surface: calibration episodes, preregistration, experiment registry,
  confirmatory receipts, and claim ledger

## Rationale

`robustness` and `entropy_weight` change the scientific comparison. Selecting
them on the same worlds used for confirmatory scoring leaks evaluation truth
into the method and makes the reported contrast optimistic. The repository now
has a deterministic proper-score selector in `src/fedference/calibration.py`;
the open work is to bind real pilots and confirmatory runs to this separation.

## Scope

For each confirmatory experiment, declare a calibration-world generator,
candidate grid, proper scoring rule, smallest effect of interest, MCSE target,
and maximum budget. Run the grid only on calibration episodes, freeze the
selected `AggregationConfig` and its content hash, and reject any evaluation
episode identifier or episode-content hash that overlaps calibration. Preserve
all candidate scores, including null and reversed choices.

Current capability: `CalibrationEpisode`, `calibrate_aggregation`,
`evaluate_locked_aggregation`, and the `robustness-calibration` CLI pilot provide
deterministic candidate selection,
canonical fingerprints, a self-verifying canonical hash over episodes, world
families, complete candidate scores, and the selected configuration, plus
identifier and content-overlap rejection. Calibration and locked evaluation
reject nonconverged or fallback solver traces. The pilot report records the
complete candidate table and a deliberate evaluation-overlap rejection.
Residual work is experiment-specific budget freezing, confirmatory receipt
integration, and manuscript evidence.

## Implementation Notes

- Use mean held-out log score as the selector; do not select with accuracy and
  later describe the result as proper-score optimized.
- Keep `smoke`, `pilot`, and `confirmatory` profiles disjoint. Smoke runs test
  mechanics only. Pilot worlds choose settings and budgets. Neither contributes
  to confirmatory intervals or manuscript headline values.
- Break exact score ties with the canonical configuration fingerprint so caller
  iteration order cannot change the selected configuration.
- Record the complete candidate family, scores, world-family declarations, and
  selected configuration hash in the `RunReceipt`; do not retain only the
  winning parameters.

## Acceptance Criteria

- Primary estimand: mean held-out log score over independent calibration worlds.
- Independent replication unit: one independently generated calibration world;
  agents, states, and posterior samples remain nested.
- Smallest effect, MCSE stopping target, maximum budget, comparison family,
  falsifier, and no-claim outcome are frozen before confirmatory execution.
- Evaluation code fails closed when any episode identifier or content hash
  overlaps the calibration set, or when the selected configuration hash differs
  from the frozen receipt.
- Required artifacts and tests: calibration report, complete candidate table,
  frozen configuration, receipt, overlap negative control, deterministic tie
  test, and a deliberate config-tamper rejection.
- Documentation changes: each participating experiment page and the claim
  ledger name the calibration/evaluation boundary.

## Verification Probes

- `uv run --locked pytest tests/fedference/test_calibration.py -q`
- Run a deliberate episode-overlap control and require `ValueError`.
- Reorder the candidate grid and require the same selected fingerprint.
- Verify the confirmatory receipt against its artifact root with
  `fedference verify`.

## Claim-Boundary Constraints

- Explicit falsifier: evaluation overlap, a changed frozen configuration, or a
  confirmatory effect below or opposite the preregistered threshold blocks the
  intended scientific claim.
- Prohibited claims: calibration does not make `robust_aggregate`
  objective-backed, does not transfer parameter-space robust-FL guarantees to
  belief aggregation, and does not turn a scenario-specific effect into broad
  robustness.
- No-claim outcome: a null or reversed confirmatory result is publishable
  evidence when implementation and provenance gates pass.

## Dependencies

Uses the public `AggregationConfig`, experiment registry, and `RunReceipt`
contracts. The MAJ-1 scoped result fixes the server-rule vocabulary but neither
certifies an objective nor blocks calibration infrastructure and pilots.
