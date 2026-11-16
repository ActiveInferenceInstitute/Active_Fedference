# `src/fedference/gnn/` — GNN admission and backend boundary

Closed, fail-closed admission of Generalized Notation Notation models, plus the
optional PyMDP backend. The model documents themselves live in
[`../models/`](../models/); they are exact-name inputs, not files to edit here.

## Modules

| Area | Modules |
| --- | --- |
| Document and source binding | `_documents.py`, `source.py`, `loader.py`, `admission.py`, `admission_values.py` |
| Sentinel and native models | `sentinel_admission.py`, `native.py`, `types.py` |
| Structured models | `structured.py`, `structured_admission.py`, `structured_cache.py`, `hierarchy.py`, `moving.py` |
| Scientific extensions | `scientific.py`, `scientific_admission.py`, `gaussian.py`, `sites.py`, `site_checkpoint.py` |
| Backend | `backend.py`, `backend_protocol.py`, `backend_registry.py`, `backend_replay.py`, `tracking.py` |

## What the loader guarantees

`load_gnn` runs the pinned upstream extractor in a subprocess against an
immutable caller-owned snapshot. It verifies every parser-closure digest in
`sentinel.parser.json` before and after execution, refuses absolute or `..`
paths, and never falls back to a packaged cache on extraction failure. The
upstream revision is recorded in `_documents.py`.

The optional PyMDP backend has its own source pin in
[`gnn-backend-lock.json`](../config/gnn-backend-lock.json). New backend runs
use its version 3 contract, taking the agent identity, dtype, devices, batch
size and policy horizon from the upstream simulation receipt. The runtime
record and replay must agree with that receipt and the locked profile.
Verification of earlier results still accepts the exact version 2 contract
and its original source and runtime lock.

Validation establishes admitted content and structural agreement. It never
certifies scientific validity.

## Models

`../models/` contains the standard `.gnn.md` sentinel specification (with its
`.ir.json` and `.parser.json` companions) and the `.fed.md` Fedference extension
specifications for the moving-observer, hierarchy, hybrid tracking and
FedGVI-sites models. The `.fed.md` documents are project extension payloads, not
standard GNN specifications; only `sentinel.gnn.md` is validated by the upstream
GNN importer. Read them; do not add prose files there.

## Check it

```bash
uv run --locked --extra dev pytest tests/test_gnn_interop.py tests/test_model_validation_cli.py -q
uv run --locked --extra dev pytest tests/integration --validation-profile=gnn -q
```

The `gnn` profile requires `GNN_ROOT` and `GNN_PARSER_DEPENDENCIES`; it refuses
with `GNN_PROFILE_RUNTIME_REQUIRED` rather than skipping. See
[../../../tests/integration/README.md](../../../tests/integration/README.md).
