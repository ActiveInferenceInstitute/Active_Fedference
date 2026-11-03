# AGENTS.md — `src/fedference/federation/`

- Bind loopback only. Never add an external interface, a public host default or
  a deployment helper here.
- Do not duplicate aggregation math. Call the validated domain operation and
  keep the result shape identical to the direct path.
- Keep replay digest-checked and guarded by the persistent round-ID store. A
  failed replay must not overwrite a prior record and must return a named error.
- Versioned envelopes may gain fields only with a version change and a reader
  for the prior version.
- Never put a real secret in source, tests or example frames. Use explicit
  non-secret fixtures and read any real key from the caller's environment.
- Tests use real child processes and real loopback sockets; no mocks.
- Parent rules: [../AGENTS.md](../AGENTS.md). Security boundary:
  [`../../../docs/security/active_fedference-threat-model.md`](../../../docs/security/active_fedference-threat-model.md).
