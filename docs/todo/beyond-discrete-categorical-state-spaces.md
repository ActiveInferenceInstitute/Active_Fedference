# Beyond Discrete-Categorical State Spaces

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Queue position: post-v1.1 research lane after its recovery and calibration
  prerequisites; blocks MAJ-5 confirmation
- Owner surface: representation, aggregation identities, methods prose

## Rationale

The current contribution is intentionally discrete-categorical. Continuous or
hybrid state spaces test whether the recovery-limit framing survives a harder
modeling regime, and whether the client-side FedGVI robustness axis and the
server aggregation rules extend without silently inheriting guarantees they do
not carry into the new representation.

## Scope

The open scope is a preregistered confirmatory hybrid-tracking benchmark:

- freeze the matched naive, robust, discrete-only, continuous-only, and
  oracle-context comparison family, proper-score estimand, budget, method
  settings, and multiple-comparison policy before evaluation;
- promote Gaussian, zero-robustness, strictly-positive-covariance,
  discrete-only, continuous-only, and outlier-toggle recovery controls into
  source-bound gates with an explicit singular-covariance failure receipt; and
- execute independently seeded held-out tracking worlds, retain every solver
  failure, and generate the report, receipt, exact-value table, figures, long
  descriptions, tokens, and bounded manuscript interpretation.

Confirmatory implementation remains blocked until a versioned pilot artifact
freezes the numerical held-out log-score SOEI, world-level MCSE target,
comparison/multiplicity rule, tracking horizon, number and schedule of worlds,
per-world episode budget, solver ceiling, and total compute budget. Recovery
fixtures may continue before that freeze, but they cannot be counted as a
confirmatory run or used to select thresholds from held-out outcomes.

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
- A source-addressed design artifact fixes the numeric SOEI/MCSE, horizon,
  world/episode schedule, multiplicity rule, solver budget, and stop rule before
  confirmatory worlds are generated.
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
- Required acceptance evidence: frozen design/configuration digest, recovery
  and singular-covariance records, complete matched-world table, locked proper-
  score inference, solver-health and provenance summary, source-bound visual
  artifacts, public PR checks, and merged public-main SHA.

## Verification Probes

- `uv run --locked pytest tests/fedference/test_hybrid.py
  tests/fedference/test_hybrid_tracking.py -q`
- Singular-covariance, zero-robustness, discrete-only, and Gaussian-only
  negative controls.
- Design-schema controls reject placeholder numeric thresholds, post-outcome
  budget edits, horizon/world drift, and receipts with the wrong design digest.
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

The one-dimensional Gaussian, mixed categorical/Gaussian fusion, and hybrid-
tracking pilot code are prerequisite contracts recorded in source tests and
`ISA.md`, not active TODO subitems. Confirmation depends on their recovery
gates and a MAJ-8-style calibration/freeze boundary. MAJ-5 remains blocked
until those gates pass. The hybrid confirmatory run remains blocked until its
pilot-derived numeric design freeze is tracked and digest-verified.
