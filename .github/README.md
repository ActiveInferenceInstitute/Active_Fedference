# `.github/` — CI and repository automation

GitHub-facing automation for
`ActiveInferenceInstitute/Active_Fedference`. This folder holds the CI
workflow, dependency-update configuration, and the shard-scheduling weights used
by the test planner. The authoritative project documentation remains
[`../README.md`](../README.md) and [`../docs/README.md`](../docs/README.md).

## Contents

| Path | Purpose |
| --- | --- |
| [`workflows/ci.yml`](workflows/ci.yml) | The `ci` workflow: fast source checks, the sharded native test suite, the pinned GNN conformance lane, Python 3.13 compatibility smoke, and the combined release gate. Every command sequence is a `scripts/gate.py` arm. |
| [`dependabot.yml`](dependabot.yml) | Weekly Dependabot updates for the `uv` lock and pinned GitHub Actions, with a 7-day cooldown. |
| [`gnn-pin.json`](gnn-pin.json) | Canonical GNN importer pin (`gnn-importer-pin/v1`): `ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.1.0`, commit `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`. |
| [`GNN.md`](GNN.md) | The pinned GNN conformance pointer: what the `gnn-shard` job parses, bootstraps, and runs. |
| [`ci/module-times.json`](ci/module-times.json) | Baseline per-module durations that `scripts/ci_shards.py` uses to balance whole test modules across shards. Scheduling weights only — not a hosted speed guarantee. |

The workflow is a thin caller: each step invokes one documented arm from
[`../scripts/gate.py`](../scripts/gate.py). A command change belongs in that
arm, not as a second copy in the workflow.

## Workflow jobs

The `ci` workflow runs on pushes to `main`, on every pull request, and on manual
dispatch.

1. **Fast source checks** — `scripts/gate.py lint`, `types`, `metadata`, and
   `docs` (Ruff, mypy, the metadata emitter-consistency check, and the
   Mermaid/docs contract).
2. **Complete test inventory** — `scripts/gate.py plan` collects every test and
   balances whole modules into four shards.
3. **Native-profile shards** (`0`–`3`) — `scripts/gate.py shard --index N` runs
   the balanced native test shards with coverage and timing evidence retained
   even on failure.
4. **GNN conformance (pinned)** — the `gnn-shard` job resolves the pinned
   commit from `.github/gnn-pin.json` into `.tmp/gnn-upstream`, then the
   `scripts/gate.py bootstrap` arm qualifies the `cpython312-cpu` runtime from
   that source and the runtime registry
   `src/fedference/config/gnn-backend-lock.json`. `scripts/gate.py gnn` runs the
   estate's shared reader (`python -m gnn.importers check --repo .
   --require-pointer --workflow .github/workflows/ci.yml --job gnn-shard`) from
   the pinned runtime, then the required GNN tests. The pin's shape, tag,
   repository, pointer and specification conformance are the reader's
   judgement; see [`GNN.md`](GNN.md).
5. **Python 3.13 compatibility smoke** — `scripts/gate.py fast` runs the
   non-slow, non-publication suite on Python 3.13.
6. **Gate** — `scripts/gate.py verify` requires every prerequisite through
   `scripts/ci_gate.py`, verifies complete shard execution, and combines
   coverage under the 90% floor; the remaining release probes (clean-checkout,
   numerical invariants, package/rendered-surface validation, pipeline
   freshness, and the release-bundle build and verify) run as separate gate
   steps.

The gate runs under `always() && !cancelled()`, so a superseded pull-request
head cannot record a permanent failure on the merge commit.

## Run the gate arms locally

```bash
uv run --locked python scripts/gate.py lint
uv run --locked python scripts/gate.py types
uv run --locked python scripts/gate.py metadata
uv run --locked python scripts/gate.py docs
uv run --locked python scripts/gate.py fast
uv run --locked python scripts/gate.py plan
uv run --locked python scripts/gate.py shard --index 0
uv run --locked python scripts/gate.py verify
uv run --locked python scripts/gate.py full
uv run --locked python scripts/gate.py all
```

`--dry-run` prints the exact commands without running them. `--keep-going`
continues after a failing arm command. The `full` and `all` arms are the
complete matrix and remain local/CI-only; `bootstrap` and `gnn` need the pinned
GNN runtime; `verify` needs the shard artifacts from a prior `shard` or `full`
run. `--project-root PATH` runs an arm in a standalone checkout.

See
[`../docs/reference/verification-commands.md`](../docs/reference/verification-commands.md)
for the complete probe list. Regenerate `ci/module-times.json` deliberately when
the test tree changes materially; it is a scheduling input, not release
evidence.
