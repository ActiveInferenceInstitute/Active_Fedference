# Deeper Hierarchy And Task Family

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Queue position: after MAJ-3 recovery/confirmation gates and task-family
  preregistration
- Owner surface: hierarchical POMDPs, task family, figures, manuscript discussion

## Rationale

Depth alone is not the gap. The open work is showing that the hierarchical
POMDP spine buys interpretable inference behavior beyond scale: richer policies,
longer horizons, and a compact non-sentinel task family that exercises a
distinct, nameable failure or validation mode.

## Scope

The open scope is to preregister and execute a confirmatory task-family design
for Four Rooms and Key-Door. Freeze task generators, policy set, horizon,
compute budget, flat/oracle/learned/shuffled/non-gating comparison family,
primary and secondary estimands, task-level replication rule, seed schedules,
negative controls, and multiplicity handling before outcome inspection. Retain
every task/seed/solver failure and generate source-bound report, receipt,
exact-value tables, figures, long descriptions, tokens, and bounded manuscript
interpretation.

The confirmatory target is the fixed intersection of exactly two named task
units—Four Rooms and Key-Door—with equal task weight after within-task
seed/episode reduction. It is not a random sample from a task population and
supports no population-level task inference. Confirmatory implementation
remains blocked until a versioned pilot design freezes numeric SOEI and MCSE
targets, per-task horizons, seed/episode schedules, matched-compute ceilings,
the primary and secondary multiplicity rules, and the maximum total budget.

## Implementation Notes

Preserve the generic N-level recovery spine; the depth-4 recovery and
belief-sharing identities must stay green as the invariant floor. Add
complexity only where it produces a new interpretable failure mode or a distinct
validation claim, never as scale for its own sake. Keep figures separating the
inference mechanism from incidental visual complexity. Treat task as the
higher-level unit and seeds/episodes as nested.

## Acceptance Criteria

- Primary estimand: episode success within a fixed horizon.
- Independent replication unit: one task in the task family (its own generator
  seed, world configuration, and policy set) — a claim replicates only if it
  holds across at least two independent task units, not across reseeds of one
  task.
- The pack estimand is the equal-weight fixed two-task intersection. Seeds and
  episodes quantify paired Monte Carlo variation within each task; they do not
  turn the two tasks into a sampled task population.
- Falsifier: if the learned hierarchy does not improve the locked primary
  estimand over matched flat, shuffled, and non-gating controls across both
  tasks, a general hierarchy advantage is withdrawn. A one-task result remains
  task-specific.
- N-level recovery and belief-sharing identities remain tested.
- Figures separate mechanism from visual complexity.
- Secondary outcomes are excess path length, free energy, calibration, and
  compute, corrected as a declared comparison family.
- Required acceptance evidence: preregistered task/design digest, matched-
  compute and recovery records, complete per-task result table, corrected
  secondary family, solver/provenance summary, source-bound visual artifacts,
  public PR checks, and merged public-main SHA.
- The preregistration contains exact numeric SOEI/MCSE targets, horizons,
  seed/episode counts, compute caps, adjustment rules, and stop conditions; no
  placeholder or result-selected setting can unlock confirmation.

## Verification Probes

- Hierarchy tests.
- Figure provenance tests.
- Updated discussion of where the mechanism does and does not generalize.
- Design-lock tests reject task substitution, task-weight changes, numeric
  placeholders, budget/horizon drift, and population-level task language.

## Required Artifacts And Tests

- Four Rooms and Key-Door generators under `src/fedference/` with per-task
  deterministic seeds, plus tests extending `tests/fedference/test_nlevel_depth.py`
  that pin the estimand and matched-compute controls per task unit.
- Figure generators and `output/figures/figure_registry.json` entries carrying
  `estimand`, `unit`, `uncertainty`, and `replication_unit` metadata for any new
  figure.
- Manuscript discussion updated via existing `{{TOKEN}}` variables resolved by
  `src/manuscript_variables.py`; no hardcoded result numbers.

## Documentation Changes

- Update the manuscript discussion section describing where the hierarchical
  mechanism generalizes and where it does not, referencing new figures by their
  registered labels only.
- Update `manuscript/SYNTAX.md` only if a new figure label is registered
  elsewhere first; this page adds no labels.

## Claim-Boundary Constraints

- Do not convert a deeper stack into an unqualified generalization claim; depth
  and task count are structural facts, not accuracy or robustness guarantees.
- No-claim boundary: this task family does not establish an estimator-level
  bounded-influence result for any server rule, does not grant the server
  heuristic a proven objective, and does not imply true multi-machine
  federation. Client-side FedGVI remains the only bounded-influence axis.

## Dependencies

The generic N-level spine and deterministic Four Rooms/Key-Door pilot are
prerequisite contracts recorded in source tests and `ISA.md`, not active TODO
subitems. Confirmation depends on all MAJ-3 hybrid recovery gates. Task
specifications, pilot budgets, and controls may be prepared earlier, but
confirmatory task-family execution must not begin before that boundary passes
and the numeric two-task design freeze is digest-bound.
