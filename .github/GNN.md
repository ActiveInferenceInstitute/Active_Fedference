# GNN conformance

`gnn conformance (pinned)` is the `gnn-shard` job in
`.github/workflows/ci.yml`. It reads `.github/gnn-pin.json`, which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.2.0`
(`4a7cccce6d4d7cb483f00323ec5cdd79300f3464`). The shared importer reader first
shipped in `v5.1.0` (`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`).

## Shared reader

`scripts/gate.py gnn` runs the estate's canonical reader from the importer
release checkout and uses the Python interpreter prepared by `bootstrap`.
It passes that checkout's Git HEAD to the shared reader:

```bash
python -m gnn.importers check --repo . --expected-commit IMPORTER_CHECKOUT_HEAD --require-pointer \
    --workflow .github/workflows/ci.yml --job gnn-shard
```

The reader owns pin discovery and shape (`pin-missing`, `pin-unreadable`,
`pin-commit-shape`, `pin-commit-mismatch`, `pin-tag-mismatch`,
`pin-repository-mismatch`, `pin-schema-unversioned`), specification validation
(`spec-missing`, `spec-invalid`) and the pointer checks
(`pointer-pin-unnamed`, `pointer-revision-unnamed`,
`pointer-workflow-unnamed`, `pointer-job-unnamed`, `pointer-missing`). This
repository uses the upstream reader directly.

The workflow reads the immutable commit for each checkout before fetching
source. `.github/gnn-pin.json` owns importer conformance.
`src/fedference/config/gnn-backend-lock.json` owns the scientific runtime
source revision, file hashes, and dependency profiles.

## What the job does

1. Checks out the importer release into `.tmp/gnn-upstream` and the locked
   backend revision into `.tmp/gnn-backend-upstream`.
2. Runs the `scripts/gate.py bootstrap` arm, which qualifies a fresh runtime
   with `scripts/bootstrap_gnn.py --profile cpython312-cpu`. Bootstrap reads the
   materialized revision from the backend checkout's Git `HEAD` and refuses a
   backend lock `src/fedference/config/gnn-backend-lock.json` that names a
   different revision, records the materialized commit as `source_commit` in
   `bootstrap.json`, and verifies every declared source digest (backend
   closure, parser manifest, extension manifest) before copying anything.
3. Runs the reader and release version probe above, then the `gnn` validation
   profile shard (`scripts/ci_shards.py run --plan .tmp/ci-plan/plan.json
   --index 4`): the six modules named by `GNN_MODULES` in
   `scripts/validation_profiles.py`, `tests/test_gnn_interop.py`,
   `tests/fedference/test_gnn_sentinel.py`,
   `tests/fedference/test_gnn_scientific.py`,
   `tests/fedference/test_gnn_structured.py`,
   `tests/integration/test_gnn_model_cli.py` and
   `tests/integration/test_gnn_pymdp.py`.

With both checkouts at those paths, run the same arms locally:

```bash
uv run --locked python scripts/gate.py bootstrap
uv run --locked python scripts/gate.py plan
uv run --locked python scripts/gate.py gnn
```

Use `--upstream-root PATH` on `bootstrap` for a different backend checkout and
`--importer-root PATH` on `gnn`, `full`, or `all` for a different importer
checkout. The GNN test shard uses the backend `GNN_ROOT` selected by the
bootstrap environment.

`tests/fedference/test_gnn_sentinel.py` re-parses
`src/fedference/models/sentinel.gnn.md` with the pinned upstream parser through
`verify_sentinel_cache`, and requires the parsed values to equal the committed
`sentinel.ir.json` cache; it also rejects malformed sources, parser-closure
drift and cache tampering. `tests/test_gnn_interop.py` exports the native
sentinel world through `examples/06_gnn_sentinel.py` and verifies that
extraction against the pinned checkout, while the scientific, structured,
model-CLI and pymdp modules cover the `.fed.md` extension specifications and the
pinned backend runtime. The job does not scan the rest of the repository for
`*.gnn.md`; `sentinel.gnn.md` is the only such file in the checkout.

## Parser cache and backend provenance

The importer pin selects the release used by the shared conformance reader.
The packaged parser closures and their offline caches use independently
qualified `v5.1.0` source. Native PyMDP execution follows its own backend lock.

Advancing the release pin from `8f293d9ba810` to `75c834bd8519` updated the
parser manifests and `UPSTREAM_REVISION` in `src/fedference/gnn/_documents.py`.
The backend lock first advanced to the same release and now pins the PyMDP
producer commit. At the release change every digest in those closures was
recomputed from a clean tree of the pinned commit, and the three source-bound
caches (`sentinel.ir.json`, `hierarchy.fed.ir.json`,
`moving-observer.fed.ir.json`) were re-recorded: their parsed content is
identical and only their parser-provenance digest and payload digest moved.
