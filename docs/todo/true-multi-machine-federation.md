# True Multi-Machine Federation

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Major
- State: Open
- Owner surface: cross-host federation transport, long-running workers, deployment docs

## Rationale

Real federation requires networked transport across distinct hosts, failure
handling, and deployment boundaries beyond single-process or single-machine
demonstrations. Until cross-host operation exists, no claim of true
multi-machine federation is warranted.

## Scope

Residual scope is explicitly split:

- **MAJ-4A:** a Docker-based local multi-node emulator with mTLS by default,
  HMAC compatibility mode, checkpoint/restart, and reproducible drop,
  duplicate, delay, replay, tamper, timeout, and out-of-order controls.
- **MAJ-4B:** validation receipts from physically distinct hosts, including
  deployment-grade key management and long-running restart behavior. This is
  an external v1.x lane.

Current capability: loopback TCP now binds each payload to a versioned envelope
containing protocol version, round, worker, aggregation-configuration hash,
payload digest, and authentication mode. Optional HMAC framing, lossless
serialization, file-backed digest-verified replay, configuration-aware
aggregation, enforced loopback-only binding, an in-memory guard, a
SQLite-backed round-ID guard that survives local process restarts, and an
in-process reference are tested. The SQLite primitive is not a shared
multi-host replay domain. Containers, mTLS, certificate lifecycle, network
fault injection, long-running orchestration, and physical hosts remain open.
The prerequisite abuse paths and trust assumptions are recorded in the
[repository threat model](../security/active_fedference-threat-model.md).

## Implementation Notes

The emulator and physical transport must remain adapters around the current server/worker
contract, never a new aggregation implementation. Preserve the existing lossless
serialization, optional per-frame HMAC integrity, file-backed digest-verified
replay validation, caller-owned persistent replay primitive, and in-process
reference unchanged. Cross-host work extends the transport adapter and
orchestration layers only. HMAC compatibility is shared-key integrity, not
per-worker identity or confidentiality.

## Acceptance Criteria

- Primary estimand: the consensus belief produced by cross-host transport is
  identical (within serialization tolerance) to the in-process reference
  consensus over the same serialized input beliefs.
- Independent replication unit for MAJ-4A: one replicated local container
  round under a declared fault schedule. For MAJ-4B it is one federation run
  on physically distinct hosts. These units are never conflated.
- Falsifier: any host pair whose reconstructed consensus diverges from the
  in-process reference beyond serialization tolerance, or any replay that fails
  to reproduce a persisted log across a process restart, refutes the claim.
- An mTLS-default local emulator reproduces the in-process reference and fails
  closed under every declared network fault.
- A long-running worker/server runtime replays persisted logs across process
  restarts, makes the persistent replay guard mandatory, and defines retention,
  backup, permission, and multi-container replay-domain semantics.
- The mTLS profile requires and validates client certificates, binds one
  certificate identity to each declared worker, and rejects wrong trust roots,
  expired/not-yet-valid certificates, wrong usage or identity, and plaintext or
  HMAC downgrade attempts.
- Required artifacts and tests: end-to-end cross-host federation transport tests;
  replay-validation tests across restarts; docs-contract tests for qualified
  true multi-machine claims.
- Documentation changes: a deployment guide describing transport and key
  management, and updated claim-boundary language distinguishing transport
  fidelity from mathematical novelty.

## Verification Probes

- End-to-end cross-host federation transport tests.
- Docker emulator tests with mTLS-default and HMAC-compatibility profiles.
- Wrong-key, tamper, replay, duplicate, drop, delay, timeout, restart, and
  out-of-order controls.
- Replay-validation tests across process restarts.
- Persistent-state omission, divergent-state, corruption, and retention tests.
- Certificate-path, identity, key-rotation, and downgrade-negative tests.
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
and reviewed threat model.
MAJ-4B depends on MAJ-4A plus external hosts and an approved key-management
boundary.
