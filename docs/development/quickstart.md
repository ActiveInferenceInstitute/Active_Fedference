# Quickstart

Runnable-in-order recipe from a fresh checkout to a green baseline, generated
reports and figures, hydrated manuscript tokens, and (optionally) a rendered
PDF. Steps 4-6 depend on their predecessors; do not reorder them.

## Prerequisites

From this repository root:

```bash
uv sync --locked --extra dev
```

## 1. Run tests with coverage gate

```bash
uv run --locked --extra dev pytest tests/ \
  --cov=src \
  --cov-fail-under=90 \
  -q
```

Expected: every collected test passes with zero failures; any skip must come
from an explicit dependency or unrendered-surface guard and remain visible in
the pytest summary. PyTorch is part of the required dev environment; coverage
must be >=90% on `src/`. This full pass is the authoritative gate and is slow —
real seeded experiments, no mocks.

For fast iteration before the full gate:

```bash
uv run --locked pytest tests/ -m "not slow" -q
```

## 2. Verify the central identity

```bash
uv run --locked python -c "
from fedference.aggregation import robust_aggregate, log_linear_pool
import numpy as np
b = [[0.7, 0.3], [0.6, 0.4]]
assert np.allclose(robust_aggregate(b, robustness=0.0).consensus,
                   log_linear_pool(b))
print('OK: project-local zero-robustness identity holds')
"
```

This check proves the implementation identity only. The relationship to Friston
et al. Eq. 7 is the documented categorical posterior-log-potential
specialization; it does not reconstruct the source factor graph, cavity
structure, or message schedule.

### Inspect the installed research registry

```bash
uv run --locked fedference list --json
uv run --locked fedference run server-theory \
  --profile smoke --seed 0 --output-dir .tmp/server-theory-smoke
uv run --locked fedference verify .tmp/server-theory-smoke/receipt.json
```

The explicit output directory protects the committed reviewer snapshot.
Confirmatory profiles remain blocked until their pilot design is frozen.

### Install the built package

To validate an artifact outside the editable checkout, use the documented
wheel/source-distribution smoke. It checks that the installed CLI and core
imports work without relying on the repository's `src/` path:

```bash
uv build --out-dir .tmp/dist
uv venv .tmp/package-env
uv pip install --python .tmp/package-env/bin/python .tmp/dist/*.whl
.tmp/package-env/bin/fedference list --json
.tmp/package-env/bin/python -c "from fedference import aggregate_result; print(aggregate_result([[.7, .3], [.6, .4]]).consensus)"
```

The source distribution retains the repository's documentation, manuscript,
scripts, and tests for archival reproduction. The wheel carries only runtime
modules and packaged compatibility data; it does not silently include the
committed reviewer snapshot under `output/`.

## 3. Run core experiments

```bash
uv run --locked python -c "
from fedference import experiments as e
seed = 20240601
print('belief sharing :', e.run_belief_sharing(seed)['mean_free_energy'])
print('language       :', e.run_language_acquisition(seed)['final_kl'])
print('emergence      :', e.run_emergence(seed)['convergence'])
print('robustness     :', e.run_robustness_sweep(seed)['any_robust_wins'])
mw = e.run_moving_world(seed)
print('moving world   :', mw['accuracy']['communicating'])
hw = e.run_hierarchical_world(seed)
print('hierarchical   :', hw['location_accuracy']['hierarchical'])
tw = e.run_3level_world(seed)
print('3-level        :', tw['location_accuracy']['nlevel3'])
"
```

`run_moving_world` (V4) uses disjoint-FOV sentinels on a 4-cell grid; isolated
agents cannot reach consensus without communication.

`run_hierarchical_world` (V2 Study 6) shows that 2-level federation — where L2
context beliefs modulate L1 location priors — matches flat inference on
location accuracy (the multiseed paired gap is not statistically significant)
while additionally resolving the context latent above chance. The current
significance numbers live in `output/reports/hierarchical_world.json` (written
by step 4), not in this doc.

`run_3level_world` (V2 Study 7) extends to a 3-level stack (L3 meta-context →
L2 context → L1 location) using the generic `build_nlevel_world` + `nlevel_infer`
API; `LayerSpec` declaratively specifies each level.

Parameters come from [`manuscript/config.yaml`](../../manuscript/config.yaml).

## 4. Full analysis pipeline (reports + figures)

From this repository root:

```bash
uv run --locked python scripts/02_run_analysis.py
```

Writes JSON reports under `output/reports/` and PNG/PDF figures under
`output/figures/`.

Every report payload is schema-validated at the write boundary: the pipeline
routes all report JSON through `workflow._write_json`, which checks the payload
against `src/analysis/report_schemas.py` before anything touches disk. A
malformed payload fails with a `ReportSchemaError` naming the report and field
— see [`../../src/analysis/README.md`](../../src/analysis/README.md).

The heuristic-characterization report (`heuristic_characterization.json`)
includes the scoped server-heuristic grid and explicit no-claim metadata; it is
a diagnostic, not a theorem or a Byzantine guarantee.

## 5. Hydrate manuscript tokens

```bash
# Provisional input only; final hydration needs the fresh full-suite receipt.
uv run --locked python scripts/z_generate_manuscript_variables.py --provisional-validation
```

This step reads the hierarchical reports (`hierarchical_world.json`,
`nlevel3_world.json`) written by step 4 and hydrates the `HIER_*` and
`NLEVEL3_*` token groups from them; it does not rerun the hierarchical
experiments itself, so step 4 must run first.

Inspect `output/data/manuscript_variables.json`. Numeric tokens that appear
inside math spans carry a `*_MATH` sibling that renders scientific notation as
$M \times 10^{E}$; prose contexts use the plain token.

For a source-current reviewer surface, use the
[two-pass rendering sequence](../manuscript/rendering_pipeline.md): it renders
these provisional inputs, runs the full-suite receipt, then performs final
receipt-backed hydration and a second render.

## 6. Render PDF (optional; requires LaTeX + pandoc)

```bash
TEMPLATE_REPO=/path/to/template
cd "$TEMPLATE_REPO"
uv run --locked python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
```

This targeted command assumes explicit hydration has already completed. The
source-current sequence runs it twice with the skip flag, so stage 03 cannot
attempt final hydration before the coverage receipt exists.

Or core pipeline in one shot:

```bash
TEMPLATE_REPO=/path/to/template
cd "$TEMPLATE_REPO"
uv run --locked python scripts/runner/execute_pipeline.py --project working/active_fedference --core-only
```

The combined core pipeline is exploratory template work, not a replacement for
the source-current two-pass sequence.

## Next steps

- Architecture: [`../core/architecture.md`](../core/architecture.md)
- Agent rules: [`agent_instructions.md`](agent_instructions.md)
- Full doc hub: [`../README.md`](../README.md)
- Acceptance contract: [`../../ISA.md`](../../ISA.md)
