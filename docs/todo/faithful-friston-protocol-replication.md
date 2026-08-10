# Faithful Friston protocol replication

## Status

- Priority class: Major
- State: Open
- Owner surface: source-protocol experiment lane, comparison records, figures, manuscript claims

## Rationale

The current categorical figures preserve selected mechanisms and estimands but
do not reproduce the source agents, modalities, episodes, mappings, or
parameter-learning protocol. A separately named lane is required before any
source figure-level numerical claim can be made.

## Scope

Recreate the source protocols corresponding to Friston et al. (2024) Eq. 2 and
Figs. 5, 7, and 9 in a separately named lane.

Current capability: `protocol_parity.py` emits strict machine-readable matrices
for Eq. 2 and Figures 5, 7, and 9. Every unresolved source parameter keeps the
automatic claim label at “paper-constrained reconstruction”; “exact
replication” is available only when all required rows are matched. The
`run_friston_protocol_audit` pilot adds an analogue-relabeling negative control;
the shipped figures remain reduced categorical source-mechanism analogues.

Residual scope is source-protocol extraction and a separate Python lane
reconstructing agent count, modalities, episode/time horizon, mappings, priors,
learning updates, policy/action process, plotted estimand, and native unit. No
MATLAB or Octave dependency is introduced.

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
- **Documentation changes:** figure metadata, comparison reports, and the
  manuscript claim ledger declare protocol identity (faithful vs analogue) and
  are regenerated only after source and project protocols agree on the declared
  estimand. Current stable filenames and analogue captions stay intact until the
  new lane passes the full source and publication gates.

## Verification Probes

- Protocol-parity and deviation-schema tests.
- `uv run --locked pytest tests/fedference/test_protocol_parity.py -q`
- Independent-unit and native-unit aggregation tests.
- Source-protocol negative control and deterministic rerun tests.
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

Depends on source-protocol extraction, a separately named Python execution lane,
and the existing figure metadata and manuscript claim contracts. Missing source
details force a paper-constrained reconstruction label rather than blocking an
honestly labeled implementation.

[Back to roadmap](../../TODO.md)
