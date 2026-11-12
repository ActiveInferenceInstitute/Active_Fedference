# GNN conformance

`gnn conformance (pinned)` is the `gnn-shard` job in
`.github/workflows/ci.yml`. It reads `.github/gnn-pin.json`, which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.1.0`
(`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`). The previous pin was `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`), whose source tree predates the
estate's shared importer reader.

## Shared reader

`scripts/gate.py gnn` runs the estate's canonical reader from the pinned
runtime, the environment the `bootstrap` arm has already built:

```bash
python -m gnn.importers check --repo . --require-pointer \
    --workflow .github/workflows/ci.yml --job gnn-shard
```

The reader owns pin discovery and shape (`pin-missing`, `pin-unreadable`,
`pin-commit-shape`, `pin-commit-mismatch`, `pin-tag-mismatch`,
`pin-repository-mismatch`, `pin-schema-unversioned`), specification validation
(`spec-missing`, `spec-invalid`) and the pointer checks
(`pointer-pin-unnamed`, `pointer-revision-unnamed`,
`pointer-workflow-unnamed`, `pointer-job-unnamed`, `pointer-missing`). This
repository no longer reads its own pin: the interim `scripts/read_gnn_pin.py`
and the in-gate pin-shape check are deleted, and no copy of the reader is
vendored. `src/gnn/importers.py` is part of the verified backend source closure
in `src/fedference/config/gnn-backend-lock.json`, so the reader executes from
digest-verified bytes rather than from an unverified checkout.

The one remaining pin read is transport: `actions/checkout` needs a ref before
the pinned source exists, so the `Resolve the pinned upstream source revision`
step extracts `.github/gnn-pin.json`'s `commit` and refuses anything that is not
an immutable 40-character revision. It makes no schema, tag, repository or
pointer judgement.

## What the job does

1. Resolves the pinned commit and checks that revision out into
   `.tmp/gnn-upstream`.
2. Runs the `scripts/gate.py bootstrap` arm, which qualifies a fresh runtime
   with `scripts/bootstrap_gnn.py --profile cpython312-cpu`. Bootstrap reads the
   materialized revision from the source tree's own git `HEAD` and refuses a
   backend lock `src/fedference/config/gnn-backend-lock.json` that names a
   different revision, records the materialized commit as `source_commit` in
   `bootstrap.json`, and verifies every declared source digest (backend
   closure, parser manifest, extension manifest) before copying anything.
3. Runs the reader above, then the version probe and the `gnn` validation
   profile shard (`scripts/ci_shards.py run --plan .tmp/ci-plan/plan.json
   --index 4`): the six modules named by `GNN_MODULES` in
   `scripts/validation_profiles.py`, `tests/test_gnn_interop.py`,
   `tests/fedference/test_gnn_sentinel.py`,
   `tests/fedference/test_gnn_scientific.py`,
   `tests/fedference/test_gnn_structured.py`,
   `tests/integration/test_gnn_model_cli.py` and
   `tests/integration/test_gnn_pymdp.py`.

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

## What the pin advance re-derived

Advancing the pin from `8f293d9ba810` to `75c834bd8519` changed the revision in
the pin, in `src/fedference/gnn/_documents.py` (`UPSTREAM_REVISION`), in the
backend lock, and in both parser manifests. Every digest in those closures was
recomputed from a clean tree of the pinned commit, and the three source-bound
caches (`sentinel.ir.json`, `hierarchy.fed.ir.json`,
`moving-observer.fed.ir.json`) were re-recorded: their parsed content is
identical and only their parser-provenance digest and payload digest moved.
