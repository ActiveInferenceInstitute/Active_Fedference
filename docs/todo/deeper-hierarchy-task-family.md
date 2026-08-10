# Deeper Hierarchy And Task Family

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Owner surface: hierarchical POMDPs, task family, figures, manuscript discussion

## Rationale

Depth alone is not the gap. The open work is showing that the hierarchical
POMDP spine buys interpretable inference behavior beyond scale: richer policies,
longer horizons, and a compact non-sentinel task family that exercises a
distinct, nameable failure or validation mode.

## Scope

Current capability: the generic N-level hierarchical POMDP spine
(`src/fedference/pomdp.py`, V2 extension) accepts arbitrary depth and is
exercised with structure-learning identities at depth 4
(`tests/fedference/test_nlevel_depth.py`). `hierarchy_tasks.py` now provides a
deterministic smoke/pilot for Four Rooms and Key-Door comparing flat inference,
oracle hierarchy, learned hierarchy, shuffled hierarchy, and a non-gating
hierarchical control at matched horizon and compute. Confirmatory task-unit
freezing and manuscript artifacts remain residual scope.

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
- Falsifier: if the learned hierarchy does not improve the locked primary
  estimand over matched flat, shuffled, and non-gating controls across both
  tasks, a general hierarchy advantage is withdrawn. A one-task result remains
  task-specific.
- N-level recovery and belief-sharing identities remain tested.
- Figures separate mechanism from visual complexity.
- Secondary outcomes are excess path length, free energy, calibration, and
  compute, corrected as a declared comparison family.

## Verification Probes

- Hierarchy tests.
- Figure provenance tests.
- Updated discussion of where the mechanism does and does not generalize.

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

Depends on all MAJ-3 hybrid recovery gates. Task specifications, pilot budgets,
and controls may be prepared earlier, but large task-family execution must not
begin before that boundary passes.
