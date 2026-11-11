# `.github/` — CI and repository automation

GitHub-facing automation for
`ActiveInferenceInstitute/Active_Fedference`. This folder holds the CI
workflow, dependency-update configuration, and the shard-scheduling weights used
by the test planner. The authoritative project documentation remains
[`../README.md`](../README.md) and [`../docs/README.md`](../docs/README.md).

## Contents

| Path | Purpose |
| --- | --- |
| [`workflows/ci.yml`](workflows/ci.yml) | The `ci` workflow: fast source checks, the sharded native test suite, the pinned GNN conformance lane, Python compatibility smoke, and the combined release gate. |
| [`dependabot.yml`](dependabot.yml) | Weekly Dependabot updates for the `uv` lock and pinned GitHub Actions, with a 7-day cooldown. |
| [`ci/module-times.json`](ci/module-times.json) | Baseline per-module durations that `scripts/ci_shards.py` uses to balance whole test modules across shards. Scheduling weights only — not a hosted speed guarantee. |

## Workflow jobs

The `ci` workflow runs on pushes to `main`, on every pull request, and on manual
dispatch.

1. **Fast source checks** — Ruff lint, the GitHub-compatible Mermaid contract
   (`scripts/validate_mermaid.py`), mypy, and the metadata emitter-consistency
   check (`scripts/emit_metadata.py --check`).
2. **Complete test inventory** — `scripts/ci_shards.py plan` collects every test
   and balances whole modules into four shards.
3. **Native-profile shards** (`0`–`3`) — the balanced native test shards with
   coverage and timing evidence retained even on failure.
4. **GNN conformance (pinned)** — reads the exact upstream commit from
   `src/fedference/config/gnn-backend-lock.json`, bootstraps the qualified
   `cpython312-cpu` runtime, and runs the required GNN tests.
5. **Python compatibility smoke** — the non-slow, non-publication suite on
   Python 3.10 and 3.13.
6. **Gate** — requires every prerequisite through `scripts/ci_gate.py`, combines
   shard coverage and enforces the 90% floor, rebuilds and byte-compares the
   wheel and source distribution, then runs the clean-checkout probe, numerical
   invariants, package/rendered-surface validation, pipeline freshness, and the
   release-bundle build and verify steps.

The gate runs under `always() && !cancelled()`, so a superseded pull-request
head cannot record a permanent failure on the merge commit.

## Run the key checks locally

```bash
uv run --locked --extra dev ruff check src/ tests/ scripts/ examples/ _fedference_build_backend.py
uv run --locked mypy src/
uv run --locked python scripts/validate_mermaid.py
uv run --locked python scripts/emit_metadata.py --check
uv run --locked python scripts/ci_shards.py plan --plan .tmp/ci-plan/plan.json --count 4
```

See
[`../docs/reference/verification-commands.md`](../docs/reference/verification-commands.md)
for the complete probe list. Regenerate `ci/module-times.json` deliberately when
the test tree changes materially; it is a scheduling input, not release
evidence.
