# `src/fedference/federation/` — local federation boundaries

One-machine transport adapters: spawned worker processes, a loopback socket
server, versioned envelopes, and digest-checked replay. This package is a
mechanics boundary, not a deployment or multi-machine protocol.

## Modules

| Module | Role |
| --- | --- |
| `process.py` | Spawned-process round orchestration. |
| `server.py` | Loopback server lifecycle and request handling. |
| `socket_transport.py` | Socket frames, optional HMAC, replay persistence. |
| `transport.py` | Versioned transport envelope and event ordering. |
| `worker.py` | Worker-side execution for a federated round. |

## Contract

- Every adapter consumes the same validated `AggregationConfig` and returns the
  same result shape as `aggregate_result`. Transport code must not duplicate
  aggregation math or reinterpret wire fields.
- Sockets bind loopback only. There is no external listener and no remote
  deployment path.
- Replay is digest-checked and guarded by a persistent round-ID store; a
  tampered or replayed frame is rejected, not repaired.
- Example HMAC keys are non-secret mechanics fixtures, not credential guidance.

## Check it

```bash
uv run --locked --extra dev pytest tests/fedference/test_transport_envelope.py \
  tests/fedference/test_socket_transport.py \
  tests/fedference/test_federation_end_to_end.py \
  tests/fedference/test_federation_health.py -q
```

Parent contract: [../AGENTS.md](../AGENTS.md).
