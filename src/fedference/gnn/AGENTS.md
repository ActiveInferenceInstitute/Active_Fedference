# AGENTS.md — `src/fedference/gnn/`

Parent contract: [../AGENTS.md](../AGENTS.md). Read [README.md](README.md) first.

- Never weaken the admission boundary: no skipped digest check, no fallback to
  a packaged cache after extraction failure, no `shell=True`, no network.
  Parser code runs as verified closure bytes under an isolated interpreter.
- Do not add model specifications or prose to [`../models/`](../models/). Those
  files are exact-name inputs consumed by `SENTINEL_ASSET`, `HYBRID_ASSET`,
  `SITES_ASSET`, `MOVING_ASSET` and `HIERARCHY_ASSET`.
- Keep the pinned upstream revision in `_documents.py` and the parser manifest
  `../models/sentinel.parser.json` in agreement. A refresh is a reviewed change
  with a passing `gnn` profile, not a drive-by edit.
- Keep `src/fedference/config/gnn-backend-lock.json` in step with the backend
  registry when a backend pin changes.
- Tests for required GNN paths must fail closed when the runtime is absent; do
  not convert `GNN_PROFILE_RUNTIME_REQUIRED` into a skip.
- New modules need a mirrored test under `tests/fedference/` or
  `tests/integration/`, and must not import Torch by default.
