# AGENTS.md — `src/fedference/evidence_contracts/`

- Receipt schemas are versioned public contracts. Add a field only with a schema
  version change and a verification path that reads both versions; never
  silently reinterpret an existing field.
- Keep strict JSON strict: duplicate keys and non-finite constants stay errors.
  Schema validation keeps rejecting unexpected fields. Do not add a permissive
  fallback for convenience.
- Custody writes stay atomic and no-clobber. Never truncate or delete a prior
  artifact to make a retry succeed.
- A receipt must not contain credentials, tokens or machine-local paths.
- Verification returns named, structured failures. Do not catch and continue at
  this boundary.
- Tests are zero-mock and seeded; cover both accept and reject paths for every
  new schema branch.
- Parent rules: [../AGENTS.md](../AGENTS.md). Architecture:
  [`../../../docs/core/architecture.md`](../../../docs/core/architecture.md).
