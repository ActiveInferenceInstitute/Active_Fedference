# Beyond Discrete-Categorical State Spaces

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Owner surface: representation, aggregation identities, methods prose

## Rationale

The current contribution is intentionally discrete-categorical. Continuous or
hybrid state spaces test whether the recovery-limit framing survives a harder
modeling regime, and whether the client-side FedGVI robustness axis and the
server aggregation rules extend without silently inheriting guarantees they do
not carry into the new representation.

## Scope

Current capability: `continuous_recovery.py` provides a one-dimensional
Gaussian recovery surface; `hybrid.py` provides mixed categorical/Gaussian
belief fusion; and `hybrid_tracking.py` exercises discrete context gating over
continuous position/velocity dynamics. `run_hybrid_tracking_comparison` now
executes matched naive, robust, discrete-only, continuous-only, and
oracle-context controls, records a next-position held-out predictive score, and
rejects singular covariance as a negative control. The executable task is a
bounded pilot, not a confirmatory benchmark.

Residual scope:

- Complete the comparison family: naive and robust hybrid fusion against
  discrete-only, continuous-only, and oracle-context controls under matched
  worlds and budgets.
- Promote the existing one-component Gaussian, zero-robustness,
  strictly-positive-covariance, and outlier-toggle unit checks into
  independently seeded publication gates; add the still-missing discrete-only
  equivalence and explicit singular-covariance failure receipt.
- Freeze pilot-selected budgets and robustness parameters before a
  confirmatory run.

## Implementation Notes

Extend from the minimal representation and tracking fixture; do not begin the
large hierarchy task family until every hybrid recovery gate is green. Any new
aggregation path must keep the same axis honesty as
the discrete case: the client-side FedGVI loss remains the only bounded-influence
axis, `robust_aggregate` keeps only its recovery-limit guarantee, and
`variational_aggregate` keeps its raw effective-weight bound without an
estimator-level B-robustness claim.

## Acceptance Criteria

- Primary estimand: held-out posterior-predictive log score over independently
  generated tracking worlds.
- Independent replication unit: one seeded continuous-state active-inference
  trial (fixed world configuration and seed), replicated across seeds.
- Falsifier: any failed recovery identity, singular-covariance failure, or a
  locked proper-score contrast below or opposite the preregistered effect
  prohibits a hybrid-robustness claim.
- Required artifacts and tests: representation and aggregation-identity tests,
  controlled tracking-world generator, complete baseline family, calibration
  and confirmatory receipts, and figures that separate representation mechanics
  from task complexity.
- Documentation changes: regenerated methods and limitations prose marking the
  new assumptions and recovery limits, with every numeric result carried by an
  existing `{{TOKEN}}` resolved by `src/manuscript_variables.py`.

## Verification Probes

- `uv run pytest tests/fedference/test_hybrid.py
  tests/fedference/test_hybrid_tracking.py -q`
- Singular-covariance, zero-robustness, discrete-only, and Gaussian-only
  negative controls.
- Regenerated methods and limitations prose.

## Claim-Boundary Constraints

- Do not imply the current discrete-categorical evidence already establishes
  continuous-state validity.
- No-claim boundary: do not claim the hybrid representation grants
  `robust_aggregate` a proven objective or bounded-influence guarantee, do not
  claim `variational_aggregate` gains an estimator-level B-robustness guarantee,
  and do not claim the continuous extension wins on accuracy unless statistics
  show it.

## Dependencies

The confirmatory task depends on the recovery gates and the MAJ-8 calibration
boundary. MAJ-5 remains blocked until those recovery tests pass.
