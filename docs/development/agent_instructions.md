# AI agent instructions — Active Fedference

Read this file before modifying any file in this repository.

## Rule 1: Read the hub first

| Document | Governs |
| --- | --- |
| **This file** | All modifications |
| [`../core/architecture.md`](../core/architecture.md) | File-boundary changes |
| [`testing_philosophy.md`](testing_philosophy.md) | Test changes |
| [`../manuscript/rendering_pipeline.md`](../manuscript/rendering_pipeline.md) | Manuscript or output changes |
| [`../manuscript/accessibility.md`](../manuscript/accessibility.md) | HTML/PDF/slide accessibility changes or claims |
| [`../security/active_fedference-threat-model.md`](../security/active_fedference-threat-model.md) | Federation transport, authentication, replay, persistence, or deployment changes |
| [`style_guide.md`](style_guide.md) | Source code changes |
| [`../manuscript/tokens-and-labels.md`](../manuscript/tokens-and-labels.md) | Manuscript `.md` edits |

Acceptance criteria: [`../../ISA.md`](../../ISA.md).

## Rule 2: Coverage gate — ≥90% on `src/`

Before modifying `src/fedference/`, locate existing tests for the symbol you touch.
After editing, run:

```bash
uv run pytest tests/ \
  --cov=src \
  --cov-fail-under=90 \
  --cov-report=term-missing \
  -v
```

Do not delete tests to satisfy the gate — add coverage for new branches.

## Rule 3: Thin orchestrator boundary

**`src/fedference/`** — domain layer with no `infrastructure.*` imports and
deterministic RNG via `np.random.default_rng(seed)`. Mathematical primitives
must remain side-effect free. File/network effects are confined to named,
caller-authorized boundary adapters: evidence receipts, external dataset/cache
loading, packaged benchmark resources, checkpoints, and socket replays.
Optional Torch modules remain behind package extras and outside the default
import graph.

**`src/analysis/workflow.py`** — orchestration only: calls `fedference.experiments`,
writes JSON, invokes figure generators. No new math here.

**`scripts/*.py`** — entry points only; delegate to `src/`.

**Forbidden in scripts/analysis:**

```python
# BAD — aggregation math belongs in fedference/
consensus = np.exp(np.sum(np.log(local_posteriors), axis=0))
```

**Correct:**

```python
from fedference.aggregation import log_linear_pool
consensus = log_linear_pool(local_posteriors)
```

## Rule 4: Show, not tell

**BAD:** "The test suite validates aggregation."

**GOOD:** "`tests/fedference/test_core_identities.py::test_robust_aggregate_at_zero_matches_log_linear_pool` pins the bit-identical *project-local* zero-robustness identity. The comparison to Friston Eq. 7 is a documented categorical posterior-log-potential specialization, not full protocol recovery."

## Rule 5: No hardcoded manuscript numbers

Every numeric in prose is a `{{TOKEN}}` from
`src/manuscript_variables.py::generate_variables()` (implementation lives in
the `src/manuscript_vars/` package). Add the token before adding the prose.
Numeric tokens used inside `$...$` math spans have `*_MATH` siblings that
render scientific notation as $M \times 10^{E}$ — use the `_MATH` sibling
there, never the plain token.

## Rule 6: Three robustness axes — keep them distinct

Never claim server-side `robust_aggregate` inherits FedGVI's per-client
bounded-influence result. Axis 1 (per-client `generalized_posterior` with
`rcce`/`β`-loss) carries the source theorem's result only under its matching
loss, model, and contamination assumptions. Axis 2
(`robust_aggregate`) is the sharp server heuristic — recovery-limit guarantee
only. Axis 3 (`variational_aggregate`) is objective-backed with a redescending
bound on the *raw* effective weights — conservative, and not by itself
estimator-level B-robustness of the normalized consensus. V1 extended axis 3
with a tempered family `F_λ` via `entropy_weight` — default `λ=1` is
bit-identical to the original. V2 adds hierarchical context inference via
`build_hierarchical_world` / `build_nlevel_world` (see Rule 8) — a modeling
dimension, not a fourth robustness guarantee. See
[`../core/conceptual-foundations.md`](../core/conceptual-foundations.md).

## Rule 7: Standalone repository and public target boundary (ISC-37)

This project is its own standalone repository. The private review mirror is
`https://github.com/docxology/active_fedference`; the intended public target is
`https://github.com/ActiveInferenceInstitute/Active_Fedference`. Never copy or
commit this project into the unrelated public template remote. A public GitHub
push and Zenodo publication remain explicit external release actions.

## Rule 8: N-level hierarchical inference — use LayerSpec

New multi-level worlds are built by passing a `list[LayerSpec]` to
`build_nlevel_world`. Do **not** add hardcoded depth logic to any caller.
`nlevel_infer` handles any depth automatically. Federation at each level reuses
`log_linear_pool` (existing function) — no new aggregation logic should be
introduced. `build_3level_world` and `build_hierarchical_world` are convenience
wrappers; prefer `build_nlevel_world` + `LayerSpec` for any new depth.

## Rule 9: Report JSON crosses the write boundary only via `_write_json`

Every report payload in `src/analysis/workflow.py` reaches disk through
`_write_json(payload, path, schema="<name>")`, which runs
`report_schemas.validate_report` *before* any file or directory is created; a
malformed payload raises `ReportSchemaError` naming the report and field.
Never bypass this boundary with a direct `json.dump` or `Path.write_text` for
report or figure-registry output. When adding a report: declare its `TypedDict`
and `_REPORT_SCHEMAS` entry in `src/analysis/report_schemas.py`, pass
`schema="<name>"` at the new call site, and declare any figure inputs in
`FIGURE_DEPENDENCY_CONTRACTS`; `tests/analysis/test_report_schemas.py` derives
accept/reject cases from the schema table automatically. Details:
[`../../src/analysis/README.md`](../../src/analysis/README.md).

## Visual and diagram contract

Mermaid is a source format, not a screenshot substitute. Keep diagrams in
fenced `mermaid` blocks so GitHub, the documentation pipeline, and a local
renderer consume the same source. Quote node labels containing parentheses,
brackets, colons, or other Mermaid syntax; prefer plain text and explicit
edges over renderer-specific HTML or embedded external images. Run both passes
before release review:

```bash
uv run python scripts/validate_mermaid.py
uv run python scripts/validate_mermaid.py --render --renderer npx \
  --output-dir .tmp/mermaid-render
```

Paper figures follow the same source-to-surface contract. A generator must
consume a typed report or explicitly named deterministic input, use the shared
palette/style/saver, declare metadata in `src/figures/_metadata.py`, and have
one manuscript embed with a self-contained caption. Captions state the axes,
estimand, units, source relation, uncertainty disposition, replication unit,
and no-claim boundary. Inspect native figure PNGs plus representative PDF,
HTML, and slide surfaces after regeneration. The HTML reader is the
accessibility-enhanced canonical surface; `qpdf` and text extraction do not
establish PDF/UA conformance for an untagged PDF.

## Pre-submit checklist

```bash
# Coverage + tests
uv run pytest tests/ \
  --cov=src --cov-fail-under=90 -q

# No mocks
uv run pytest tests/test_runtime_surface.py -q

# Domain layer free of infrastructure
rg -n "import infrastructure" src/fedference/ \
  && exit 1 || echo "Clean"

# Central identity (spine) — server axes
uv run python -c "
from fedference.aggregation import robust_aggregate, variational_aggregate, log_linear_pool
import numpy as np
b = [[.7,.3],[.6,.4]]
assert np.allclose(robust_aggregate(b, robustness=0).consensus, log_linear_pool(b))
assert np.allclose(variational_aggregate(b, robustness=0).consensus, log_linear_pool(b))
# V1: default entropy_weight=1.0 must be bit-identical to original variational path
assert np.allclose(variational_aggregate(b, robustness=0, entropy_weight=1.0).consensus, log_linear_pool(b))
"

# N-level hierarchical spine check
uv run python -c "
from fedference.pomdp import build_3level_world, nlevel_infer
import numpy as np
w = build_3level_world()
A = np.asarray(w['L1']['A'][0])
qs = nlevel_infer(A, 4, w)
assert all(abs(q.sum() - 1) < 1e-9 for q in qs['q_levels'])
print('OK: nlevel_infer produces valid PMFs at all levels')
"
```

Run from this repository root; pytest `conftest.py` handles the `src` path in
tests.
