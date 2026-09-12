# True Multi-Machine Federation

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Queue position: post-v1.1 transport lane; MAJ-4A precedes the external
  MAJ-4B distinct-host campaign
- Owner surface: cross-host federation transport, long-running workers, deployment docs

## Rationale

Real federation requires networked transport across distinct hosts, failure
handling, and deployment boundaries beyond single-process or single-machine
demonstrations. Until cross-host operation exists, no claim of true
multi-machine federation is warranted.

## Scope

The open scope is explicitly split:

- **MAJ-4A:** a Docker-based local multi-node emulator with mTLS by default,
  HMAC compatibility mode, checkpoint/restart, and reproducible drop,
  duplicate, delay, replay, tamper, timeout, and out-of-order controls.
- **MAJ-4B:** validation receipts from physically distinct hosts, including
  deployment-grade key management and long-running restart behavior. This is
  a separately authorized external lane with no preassigned version.

Protocol-v1 frames remain unchanged. MAJ-4A must wrap the existing local server
and worker contract in a Docker-based, mTLS-default emulator with certificate
lifecycle, worker identity, checkpoint/restart, shared replay-domain semantics,
and deterministic drop/duplicate/delay/replay/tamper/timeout/out-of-order fault
controls. MAJ-4B then runs the frozen contract on physically distinct hosts
with approved key management and long-running restart behavior. The abuse paths
and trust assumptions are owned by the
[repository threat model](../security/active_fedference-threat-model.md).

Execution of either tranche remains blocked until its versioned protocol design
freezes the exact round count, serialized-consensus comparison metric and
tolerance, timeout/delay/drop/duplicate/replay/tamper/out-of-order schedules,
restart points, certificate lifetime/rotation cases, retry policy, and maximum
runtime/resource budget. The MAJ-4B design additionally freezes host topology,
clock-skew envelope, network path assumptions, and key custody/rotation
procedure. A placeholder “within tolerance” cannot authorize either campaign.

## Implementation Notes

The emulator and physical transport must remain adapters around the current server/worker
contract, never a new aggregation implementation. Preserve the existing lossless
serialization, optional per-frame HMAC integrity, file-backed digest-verified
replay validation, caller-owned persistent replay primitive, and in-process
reference unchanged. Cross-host work extends the transport adapter and
orchestration layers only. HMAC compatibility is shared-key integrity, not
per-worker identity or confidentiality.

## Acceptance Criteria

### MAJ-4A — authenticated local-container emulator

- Primary estimand: local-container transported consensus minus the in-process
  reference for identical serialized beliefs, reported under the frozen metric
  and tolerance for every declared no-fault and fault condition.
- Replication unit: one independently initialized local container round under a
  frozen schedule; messages and retry attempts within a round are nested.
- The mTLS-default emulator binds certificates to worker identities, rejects
  wrong roots, validity periods, usage/identity, plaintext, and HMAC downgrade,
  and reproduces the in-process reference or fails closed as preregistered.
- Restart-durable replay covers the frozen server/worker restart points and
  defines replay-state retention, backup, permissions, and multi-container
  replay domains.
- Required evidence: digest-bound design, pinned images, certificate policy,
  complete local fault matrix, restart/replay receipts, equivalence table,
  deployment guide, green PR checks, and merged public-main SHA.
- Falsifier: any accepted frame/round beyond the frozen consensus tolerance,
  unauthorized identity, replay acceptance, missing persistent state, or fault
  disposition contrary to the design blocks MAJ-4A.
- Physical hosts are not required to close MAJ-4A and local containers can
  support only the explicitly labeled local-emulator claim.

### MAJ-4B — physical distinct-host validation

- Primary estimand: physical-host transported consensus minus the same
  in-process reference, under the separately frozen metric/tolerance and host,
  clock, network, fault, restart, and key-management design.
- Replication unit: one federation deployment run spanning the declared set of
  physically distinct hosts; rounds/messages within a deployment are nested.
- Required evidence: exact host identities and software images, topology and
  clock records, measured network assumptions, external key custody/rotation,
  cross-host fault/restart receipts, reference comparisons, and external review.
- Falsifier: an undeclared topology or clock/network drift, key-custody failure,
  cross-host replay/restart failure, or accepted consensus beyond the frozen
  tolerance blocks the physical-host claim.
- MAJ-4B depends on completed MAJ-4A but is a distinct campaign; emulator
  receipts cannot be relabeled as physical multi-host evidence.

## Verification Probes

- MAJ-4A Docker emulator tests with mTLS-default and HMAC-compatibility profiles.
- Wrong-key, tamper, replay, duplicate, drop, delay, timeout, restart, and
  out-of-order controls.
- Replay-validation tests across process restarts.
- Persistent-state omission, divergent-state, corruption, and retention tests.
- Certificate-path, identity, key-rotation, and downgrade-negative tests.
- Design-lock tests reject missing round counts/tolerances, fault-schedule or
  restart drift, and resource-budget changes. MAJ-4B additionally exercises
  physical-host topology, clock-skew, network-path, and key-custody controls.
- Docs-contract tests for qualified true multi-machine claims.

## Claim-Boundary Constraints

Until MAJ-4A lands, claims remain limited to queue transport, single-machine OS
processes, and loopback TCP with versioned envelopes, optional HMAC frame
integrity, persisted digest-verified replay validation, and optional
restart-durable local round-ID rejection. After MAJ-4A, the claim may expand
only to local multi-node emulation. “Physical multi-host” remains prohibited
until MAJ-4B receipts exist.

Prohibited claims (no-claim boundary): do not state or imply true multi-machine
federation, cross-host operation, or a secure/private deployed channel before
the cross-host transport and its tests exist. Transport fidelity is never
evidence of mathematical novelty in the aggregation rules.

## Dependencies

MAJ-4A depends on the stable configuration hash, transport-envelope schema,
reviewed threat model, and the existing loopback/HMAC/digest-replay primitives
recorded in source tests and `ISA.md`; those local primitives are not active
TODO subitems.
MAJ-4B depends on MAJ-4A plus external hosts and an approved key-management
boundary. No cross-host execution is required for MAJ-4A acceptance, and no
MAJ-4A result can satisfy MAJ-4B.
