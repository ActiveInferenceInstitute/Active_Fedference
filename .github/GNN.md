# GNN conformance

`gnn conformance (pinned)` is defined in `.github/workflows/ci.yml` as the
`gnn-shard` job and reads `.github/gnn-pin.json`, which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`).

The job rejects a pin whose schema is not `gnn-importer-pin/v1` or whose
`commit` is not an exact 40-character SHA, checks out that commit into
`.tmp/gnn-upstream`, and qualifies a fresh runtime with
`scripts/bootstrap_gnn.py --profile cpython312-cpu`. It then runs
`scripts/ci_shards.py run --plan .tmp/ci-plan/plan.json --index 4`, the
`gnn` validation profile: the six modules named by `GNN_MODULES` in
`scripts/validation_profiles.py`,
`tests/test_gnn_interop.py`,
`tests/fedference/test_gnn_sentinel.py`,
`tests/fedference/test_gnn_scientific.py`,
`tests/fedference/test_gnn_structured.py`,
`tests/integration/test_gnn_model_cli.py` and
`tests/integration/test_gnn_pymdp.py`.

`tests/fedference/test_gnn_sentinel.py` re-parses
`src/fedference/models/sentinel.gnn.md` with the pinned upstream parser
through `verify_sentinel_cache`, and requires the parsed values to equal
the committed `sentinel.ir.json` cache; it also rejects malformed sources,
parser-closure drift and cache tampering. `tests/test_gnn_interop.py`
exports the native sentinel world through `examples/06_gnn_sentinel.py` and
verifies that extraction against the pinned checkout, while the
scientific, structured, model-CLI and pymdp modules cover the `.fed.md`
extension specifications and the pinned backend runtime. The job does not
scan the rest of the repository for `*.gnn.md`; `sentinel.gnn.md` is the
only such file in the checkout.

CI checkout and local bootstrap share the interim local reader
`scripts/read_gnn_pin.py`. Bootstrap also rejects drift between that revision
and the qualified backend lock before provisioning, and records `source_commit`
in `bootstrap.json`. The estate pin above predates `gnn.importers`; this local
reader is not the shared-reader cutover and does not vendor that implementation.
Shared-reader adoption remains staged until a reader-bearing pin is qualified.
