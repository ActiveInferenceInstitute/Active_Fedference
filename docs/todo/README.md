# Active Backlog Scope Index

This folder contains open Active Fedference work plus one standing governance
reference. The root
[`TODO.md`](../../TODO.md) owns ordering and classification; each page here owns
one implementable scope, prerequisites, acceptance evidence, falsifier, and
claim boundary. Closed work is removed from this folder after its durable
evidence is recorded in `ISA.md`, source-bound receipts, release records, or Git
history.

The authoritative public integration repository is
[`ActiveInferenceInstitute/Active_Fedference`](https://github.com/ActiveInferenceInstitute/Active_Fedference).
Public-main integration occurs through reviewed `codex/*` branches and normal
merge commits. The private/interim evidence remote remains a separate surface.

## Minor

| ID | Scoped page | Acceptance focus |
| --- | --- | --- |
| MIN-2 | [Clone-correct integration and release verification](release-and-verification-ladder.md) | Candidate history, exact-commit gates, two isolated clones, and eligible release verdict |
| MIN-3 | [Release documentation and repository-discovery handoff](release-documentation-handoff.md) | Official repository, release/DOI links, installed application path, and documentation consistency |

## Medium

| ID | Scoped page | Acceptance focus |
| --- | --- | --- |
| MED-4 | [Visual scholarship and accessible-slide integration](visual-scholarship-integration.md) | Exact clean Template renderer, source-bound visual/accessibility surfaces, public PR, and post-merge CI |
| MED-5 | [v1.1.0 release, publication, and private synchronization](v1-1-release-publication.md) | MED-5A release identity/PR, MIN-2 exact-final certification, MED-5B tag/assets/Zenodo, then MIN-3 closure before private sync |
| MED-6 | [Independent and cross-vendor reproduction](independent-reproduction.md) | Conjunctive final-bundle independent-human/Advisor and different-vendor execution evidence for ISC-89 |

## Major

| ID | Scoped page | Acceptance focus |
| --- | --- | --- |
| MAJ-8 | [Adaptive robustness calibration](adaptive-robustness-calibration.md) | Leakage-free preregistration, separate method selection, stability, and frozen design |
| MAJ-6 | [External three-dataset confirmation](external-benchmark-domain-pilot.md) | Design-locked paired inference, dataset-level reporting, solver health, and provenance |
| MAJ-2A/B | [Faithful FedGVI BNN lane](faithful-fedgvi-bnn-lane.md) | Portable source-protocol evidence and separately gated source-scale CUDA replication |
| MAJ-7 | [Faithful Friston protocol reconstruction](faithful-friston-protocol-replication.md) | Complete source-parameter, estimand, native-unit, and independent-unit parity |
| MAJ-3 | [Beyond discrete-categorical state spaces](beyond-discrete-categorical-state-spaces.md) | Recovery-gated, calibrated continuous/hybrid confirmation |
| MAJ-5 | [Deeper hierarchy and task family](deeper-hierarchy-task-family.md) | Preregistered cross-task confirmation with matched controls |
| MAJ-4A/B | [Authenticated and physical multi-machine federation](true-multi-machine-federation.md) | Local mTLS emulator first, then distinct-host receipts under a separate external gate |

## Classification rule

- **Minor** items are bounded maintenance or evidence checks, even when failure
  blocks a release.
- **Medium** items coordinate multiple engineering, rendering, repository, or
  publication surfaces without introducing a new scientific estimand.
- **Major** items add or test a scientific, protocol, representation, task, or
  deployment claim and require preregistration or external execution.

The active-page set is derived from the rows above; this index does not retain a
manual page count. A Minor item may require broad execution or coordination and
may block release—the tier describes the kind of work, not its duration or
criticality. MIN-3 is on the critical path after verified v1.1 publication and
before MAJ-8 scientific branching.

## Governing reference

[Scholarship-indexed phase and claim governance](scholarship-and-phase-plan.md)
is standing policy, not an active backlog row. It governs every proposed phase,
estimand, source, design, or claim change and includes the parked-track
definitions. Parked work has no version, execution priority, or authorization
until a separate review promotes it into the active tables.

## Identifier reconciliation

Several identifiers also appear in historical `ISA.md` iterations with older,
completed meanings. Those historical labels are immutable evidence and are not
reactivated. In this forward-only backlog, `MIN-2` means the clone-correct v1.1
integration/release verification ladder and `MAJ-7` means the faithful Friston
protocol-reconstruction lane accepted by the current plan. Always resolve an ID
through its linked active page; an identifier alone never implies status,
scope, or continuity with a retired ISA item.
