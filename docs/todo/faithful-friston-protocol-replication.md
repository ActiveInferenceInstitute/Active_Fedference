# Faithful Friston protocol replication

## Status

- Priority class: Major
- State: Open
- Queue position: post-v1.1 source-parity lane; independent of the MAJ-8/MAJ-6
  external-data sequence
- Owner surface: source-protocol experiment lane, comparison records, figures, manuscript claims

## Rationale

The current categorical figures preserve selected mechanisms and estimands but
do not reproduce the source agents, modalities, episodes, mappings, or
parameter-learning protocol. A separately named lane is required before any
source figure-level numerical claim can be made.

## Scope

Recreate the source protocols corresponding to Friston et al. (2024) Eq. 2 and
Figs. 5, 7, and 9 in a separately named lane.

The open scope is source-protocol extraction and a separate Python lane
reconstructing agent count, modalities, episode/time horizon, mappings, priors,
learning updates, policy/action process, plotted estimand, and native unit. No
MATLAB or Octave dependency is introduced.

Execution remains blocked until one versioned source-extraction artifact has
resolved, for every target equation/figure, the independent unit, native unit,
aggregation/reduction order, deterministic or stochastic character, numerical
tolerance and comparison metric, seed or episode schedule, and maximum compute
budget. Unknown source details remain explicit and retain the paper-constrained
label; they cannot be filled from convenient project defaults after outcomes
are inspected.

## Implementation Notes

Keep the source-protocol implementation isolated from the current pure
categorical generators. Make protocol identity explicit in reports and figure
metadata so a reduced analogue cannot be mistaken for the faithful lane.
Resolve the existing matrix before implementation. List every deviation beside
the source parameter rather than hiding it in a convenience default. Preserve
unknowns explicitly when the paper or source routine does not determine them.

## Acceptance Criteria

- **Primary estimand:** the source-defined plotted quantity for each replicated
  figure (Eq. 2 and Figs. 5, 7, 9), declared in native units before any
  comparison.
- **Independent replication unit:** the source-defined independent agent,
  episode, or seed, explicitly separated from ordered trajectory points and
  nested trials.
- **Falsifier:** at least one source-protocol negative control whose expected
  direction or null result fails if the implementation silently changes the
  protocol.
- **Required artifacts and tests:** a machine-readable source-to-project
  comparison artifact (parameters, deviations, estimands, units); protocol-parity
  and deviation-schema tests; independent-unit and native-unit aggregation tests;
  negative-control and deterministic-rerun tests; declared numerical tolerances;
  and rendered figure checks under new filenames or an explicit protocol field.
- **Design-freeze gate:** the source-extraction artifact contains no unresolved
  unit, tolerance, comparison, or budget placeholder before faithful execution
  begins, and its digest is bound into every run receipt.
- **Documentation changes:** figure metadata, comparison reports, and the
  manuscript claim ledger declare protocol identity (faithful vs analogue) and
  are regenerated only after source and project protocols agree on the declared
  estimand. Current stable filenames and analogue captions stay intact until the
  new lane passes the full source and publication gates.
- **Required acceptance evidence:** resolved source matrix, source/version
  record, complete deviation table, deterministic execution receipt, negative-
  control result, native-unit exact-value tables, source-bound figures and long
  descriptions, public PR checks, and merged public-main SHA.

## Verification Probes

- Protocol-parity and deviation-schema tests.
- `uv run --locked pytest tests/fedference/test_protocol_parity.py -q`
- Independent-unit and native-unit aggregation tests.
- Source-protocol negative control and deterministic rerun tests.
- Negative tests reject missing units, outcome-selected tolerances, changed
  budgets, and receipts bound to a different source-extraction digest.
- Full source, manuscript, render, web, slide, and release gates.

## Claim-Boundary Constraints

- **No-claim boundary:** a positive result in the current reduced categorical
  lane cannot be promoted to evidence for source-protocol numerical identity.
- Exact protocol reproduction may be claimed only after parameter parity,
  estimand parity, and native-scale visual comparison are all verified.
- Do not relabel the current reduced categorical outputs as exact source
  reproductions. A matching qualitative direction is not parameter or figure
  identity, and a faithful lane must not inherit current source-inspired claims
  without its own estimand and uncertainty audit.
- Source figures remain analogues of Friston et al. (2024), never reproductions,
  until the faithful lane passes its gates.

## Dependencies

Depends on a complete, digest-bound source-protocol extraction and design
freeze, a separately named Python execution lane,
the existing machine-readable parity/deviation schema, and the figure metadata
and manuscript claim contracts. The current reduced categorical analogues are
tracked evidence outside this active queue. Missing source details force a
paper-constrained reconstruction label rather than an exact-replication claim.

[Back to roadmap](../../TODO.md)
