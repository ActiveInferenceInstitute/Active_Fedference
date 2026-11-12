# CI guidance — `.github/`

Read [`README.md`](README.md) first. This folder owns the GitHub Actions
workflow, the Dependabot configuration, and the shard-balance weights. The
authoritative CI behavior is [`workflows/ci.yml`](workflows/ci.yml).
`scripts/gate.py` owns the workflow's command arms, `scripts/ci_gate.py`
implements the prerequisite guard, and `scripts/ci_shards.py` implements test
planning, execution, and complete-run verification.

## Rules

- Do not weaken a required check into a skip or an `if:` guard to make a run
  pass. The pinned GNN conformance lane and the gate's prerequisite check are
  fail-closed: a missing runtime is a failure, not a skip.
- Keep the workflow pins intact — Python `3.12.15` for the primary lanes and the
  exact action versions already in use. Bump a pin deliberately and separately,
  then let the full gate re-verify it.
- Never hand-edit `ci/module-times.json` to change scheduling outcomes. It is a
  measured baseline keyed to a source commit; regenerate it from a real run and
  update `baseline_source`.
- The gate's `always() && !cancelled()` condition exists so a superseded
  pull-request head cannot record a permanent failure on the merge commit. Do
  not simplify it.
- Dependabot updates the `uv` lock and pinned GitHub Actions weekly with a 7-day
  cooldown. Merge an update only after the full gate passes, and re-verify the
  byte-identical distribution build when the update touches a reproducibility
  input.
- A workflow step's command sequence belongs in `scripts/gate.py`. Add or change
  an arm, document it in [`../scripts/README.md`](../scripts/README.md) and
  [`../scripts/AGENTS.md`](../scripts/AGENTS.md), and keep the `--dry-run`
  coverage in `tests/test_scripts_smoke.py`; never inline a second copy in the
  workflow.

## Validate

```bash
actionlint .github/workflows/ci.yml
uv run --locked python scripts/gate.py lint --dry-run
uv run --locked python scripts/ci_gate.py --results '{}'
```

`actionlint` requires the external binary. `scripts/gate.py lint --dry-run`
prints the lint arm's command without running it, so a malformed arm fails as a
syntax/import error before the workflow uses it. The `ci_gate.py` call with
empty results is a syntax/import smoke; the workflow passes the real `needs`
JSON.
