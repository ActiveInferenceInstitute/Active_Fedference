# Verification commands

Copy-paste probes from [`../../ISA.md`](../../ISA.md) and project validation.
Run local project probes from this repository root.

## Test + coverage gate (authoritative)

```bash
uv run --extra dev pytest tests/ \
  --cov=src \
  --cov-fail-under=90
```

Expected: all collected tests pass with zero failures and zero skips. PyTorch is
part of the required dev environment, so missing PyTorch is a failing setup.

Fast and scoped profiles are available for iteration; they select real tests and
do not replace the full gate:

```bash
uv run pytest tests/ -m "not slow" -q
uv run pytest tests/ -m integration -q
uv run pytest tests/ -m publication -q
```

## Local validation profiles

```bash
uv run python scripts/validate_all.py quick
uv run python scripts/validate_all.py manuscript
uv run python scripts/validate_all.py package
uv run python scripts/validate_all.py torch
uv run python scripts/validate_all.py source
uv run python scripts/validate_all.py freshness
uv run python scripts/validate_all.py full
```

`--dry-run` prints every command without running it. `--keep-going` runs every
command in a profile before returning failure.

Profile scope:

- `quick`: docs contract, caption completeness, and the required Torch smoke.
- `manuscript`: cross-reference, caption, token-provenance, token-table, and
  manuscript-variable checks.
- `package`: web-figure mirroring, web cross-reference normalization, and
  package validation.
- `torch`: explicit required PyTorch lane via `uv run --extra dev`.
- `freshness`: the standalone successful test/coverage receipt plus the
  content-hashed analysis → hydration → render stage receipts.
- `source`: Ruff, mypy, invariants, domain-layer grep, and exact-set release
  build/verify.
- `full`: quick + manuscript + package + rendered-surface + freshness + source
  + full coverage gate.

## Public API, registry, and receipt verification

```bash
uv run fedference list --json

uv run fedference run server-theory \
  --profile smoke --seed 0 --output-dir .tmp/server-theory-smoke

uv run fedference verify .tmp/server-theory-smoke/receipt.json
```

Write-producing commands require an explicit empty directory and reject the
committed `output/` tree. Registry state is a work declaration, not a scientific
result. Smoke and pilot runs do not support manuscript claims.

The bounded single-machine research pilots use the same receipt contract:

```bash
uv run fedference run robustness-calibration \
  --profile pilot --seed 0 --output-dir .tmp/calibration-pilot
uv run fedference run fedgvi-bnn \
  --profile pilot --seed 0 --device cpu --output-dir .tmp/fedgvi-bnn-pilot
uv run fedference run hybrid-tracking \
  --profile pilot --seed 0 --output-dir .tmp/hybrid-pilot
uv run fedference run hierarchy-tasks \
  --profile pilot --seed 0 --seed 1 --output-dir .tmp/hierarchy-pilot
uv run fedference run friston-protocol \
  --profile pilot --seed 0 --output-dir .tmp/friston-parity-audit
```

These reports retain the calibration overlap, BNN checkpoint/device,
hybrid-control/singular-covariance, hierarchy-task, and Friston
analogue-relabeling negative controls. They remain pilot evidence until the
corresponding confirmatory budgets and manuscript contracts are frozen.

Configuration parity across the rich interface and legacy wrapper:

```bash
uv run python -c "
import numpy as np
from fedference import AggregationConfig, aggregate, aggregate_result
b = np.asarray([[.7, .3], [.6, .4]])
c = AggregationConfig(method='robust', robustness=1.5)
assert np.array_equal(aggregate_result(b, config=c).consensus, aggregate(b, config=c))
print(c.fingerprint)
"
```

## Wheel and source-distribution smoke

```bash
DIST_SMOKE=$(mktemp -d .tmp/distribution-smoke.XXXXXX)
export SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"
uv build --out-dir "$DIST_SMOKE/dist"
uv build --out-dir "$DIST_SMOKE/rebuilt"
shasum -a 256 "$DIST_SMOKE"/dist/* "$DIST_SMOKE"/rebuilt/*
uv venv "$DIST_SMOKE/wheel"
uv pip install --python "$DIST_SMOKE/wheel/bin/python" "$DIST_SMOKE"/dist/*.whl
"$DIST_SMOKE/wheel/bin/fedference" list --json
"$DIST_SMOKE/wheel/bin/python" -c "from fedference.benchmark import run_tabular_benchmark; assert run_tabular_benchmark(seed=0)['n_rows'] == 150"
uv venv "$DIST_SMOKE/sdist"
uv pip install --python "$DIST_SMOKE/sdist/bin/python" "$DIST_SMOKE"/dist/*.tar.gz
"$DIST_SMOKE/sdist/bin/fedference" list --json
"$DIST_SMOKE/sdist/bin/python" -c "from fedference.benchmark import run_tabular_benchmark; assert run_tabular_benchmark(seed=0)['n_rows'] == 150"
```

The default install must import the NumPy/SciPy core without Torch. The optional
BNN/MPS lane is installed separately with `uv sync --extra bnn`. The PEP 517
backend is exactly pinned and normalizes wheel/sdist archive metadata when
`SOURCE_DATE_EPOCH` is set. The two directories above must therefore contain
byte-identical wheel and sdist pairs; a differing digest is a release blocker,
even when both artifacts install successfully.

## Pinned external-data smoke

```bash
uv run fedference benchmark \
  --dataset-id uci-banknote --profile smoke --seed 42 \
  --cache-dir .tmp/uci-cache --output-dir .tmp/banknote-smoke
uv run fedference verify .tmp/banknote-smoke/receipt.json
uv run fedference verify .tmp/banknote-smoke/receipt.json --require-clean-git
```

The first verification checks archive/member hashes, schema, train-only
preprocessing, split hashing, bound `config.json`/`report.json` bytes, and
receipt consistency. The strict second form also matches the live full commit,
clean Git tree, and `uv.lock` digest against the receipt; pass
`--project-root /path/to/checkout` when running outside that checkout. Neither
is the confirmatory three-dataset evidence pack.

## Central identity (spine — both server axes + V1 default bit-identity)

```bash
uv run python -c "
from fedference.aggregation import robust_aggregate, variational_aggregate, log_linear_pool
import numpy as np
b = [[.7,.3],[.6,.4]]
assert np.allclose(robust_aggregate(b, robustness=0).consensus, log_linear_pool(b))
assert np.allclose(variational_aggregate(b, robustness=0).consensus, log_linear_pool(b))
assert np.allclose(variational_aggregate(b, robustness=0, entropy_weight=1.0).consensus, log_linear_pool(b))
print('OK')
"
```

Run from the repository root via `uv run`.

## N-level hierarchical spine (V2)

```bash
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

`test_hierarchical_pomdp.py` covers `LayerSpec` validation,
`build_hierarchical_world` / `build_3level_world` shapes, `hierarchical_infer` /
`nlevel_infer` PMF validity, and `run_hierarchical_world` / `run_3level_world`
smoke + determinism.

## Layer contract (no infra in domain)

```bash
grep -rn "import infrastructure" src/fedference/ \
  && { echo FAIL; exit 1; } || echo "Clean"
```

## MAJ-1 server-rule characterization

```bash
uv run python -c "
import json
from fedference.experiments import run_heuristic_characterization
r = run_heuristic_characterization(0)
assert r['claim_level'] == 'scoped_implementation_fact'
assert r['theory_status'] == 'open_no_global_objective'
assert r['formal_no_go']['status'] == 'proved_for_declared_class'
assert r['formal_no_go']['raw_q_block_witness']['tangential_contradiction_norm'] > 0
assert r['formal_no_go']['normalized_weight_companion']['forward_difference_gap'] > 0
assert r['grid']['negative_controls']['finite_search_is_not_a_global_breakdown_bound']
json.dumps(r)
print('OK: MAJ-1 grid and scoped no-go metadata are deterministic and claim-bounded')
"
```

## Repository mode

```bash
git remote -v
git rev-parse --show-toplevel
```

This checkout is a standalone repository (intended public release target:
ActiveInferenceInstitute/Active_Fedference). The top level should be this
repository root, not the sibling public template checkout.

## No mocks

```bash
uv run pytest tests/test_runtime_surface.py -q
```

The dedicated test scans executable Python under `src/`, `scripts/`, and
`tests/`, while excluding only its own pattern declarations. A raw recursive
grep is not the gate because it also matches the forbidden-name documentation
in `tests/PATTERNS.md` and the detector's own regex.

## Release bundle + provenance fingerprint

```bash
uv run python scripts/build_release.py
uv run python scripts/build_release.py --verify
```

Build writes `output/release/` (`manifest.json`, `sha256sums.txt`, a derived
`README.md`). Verify is exact-set: every listed artifact must exist with its
recorded byte size and SHA-256, and no unlisted file may appear under the
release roots. The manifest also records a provenance `fingerprint` — a
SHA-256 over the declared source, manuscript, documentation, producer-script,
dependency-lock, and claim-audit inputs — together with the pipeline profile,
generator version, and individual input digests. `--verify` recomputes these
from the current tree, so a bundle whose bytes all match but which was built
from a different producer or evidence state still fails as stale and names
changed inputs.

The CLI first requires fresh publication-profile analysis, test/coverage,
hydration, and render receipts. Its success establishes a local source-current
reviewer bundle; it does not replace isolated-clone reproduction, author
approval, DOI work, or external publication authority.

The default build is an unreleased, byte-reproducible snapshot and therefore
records `generated_at: null`. Only an approved release should add time metadata,
using `--timestamp YYYY-MM-DDTHH:MM:SSZ` or `SOURCE_DATE_EPOCH`; rebuilding the
same tree without either input must leave every bundle byte unchanged.

## Report schema write boundary

```bash
uv run python -c "
from analysis.report_schemas import ReportSchemaError, validate_report
try:
    validate_report('belief_sharing', {})
except ReportSchemaError:
    print('OK: schema gate rejects malformed report payloads')
else:
    raise SystemExit('FAIL: empty payload accepted')
"
```

`scripts/02_run_analysis.py` validates every report payload against its typed
schema at the single JSON write boundary (`src/analysis/report_schemas.py`)
and checks each figure generator's consumed fields against an explicit
dependency contract, so schema drift fails at write time rather than at
figure or token time.

## Complexity accounting and scaling diagnostic

The publication analysis writes the symbolic catalog and measured benchmark to
`output/reports/complexity_scaling.json` and the corresponding figure to
`output/figures/complexity_scaling.png`. Inspect both evidence layers with:

```bash
uv run python -c '
import json
from pathlib import Path
r = json.loads(Path("output/reports/complexity_scaling.json").read_text())
assert r["status"] == "ok"
assert {row["operation"] for row in r["analytic_specs"]} >= {
    "log_linear_pool", "robust_aggregate", "variational_aggregate",
    "share_round_naive", "infer_states", "federation_server_round",
}
assert all(value > 0 for row in r["measurements"] for value in row["median_seconds"])
print("OK: analytic complexity catalog and measured scaling report are present")
'
```

The analytic table is the claim-bearing implementation account. The timing
slopes are machine diagnostics only; rerun them after changing the benchmark
grid, NumPy/BLAS environment, or solver implementation, and do not interpret
the repeat range as a confidence interval.

## Analysis + token hydration

~~~bash
AF_REPO=/path/to/active_fedference
TEMPLATE_REPO=/path/to/template
export SOURCE_DATE_EPOCH="$(git -C "$AF_REPO" log -1 --format=%ct)"

cd "$AF_REPO"
uv run python scripts/02_run_analysis.py
uv run python scripts/z_generate_manuscript_variables.py --provisional-validation

# Provisional renderer pass: hydrate has already happened above.
cd "$TEMPLATE_REPO"
uv run python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
uv run python scripts/pipeline/stage_04_validate.py --project working/active_fedference
uv run python scripts/pipeline/stage_05_copy.py --project working/active_fedference

cd "$AF_REPO"
uv run --extra dev python scripts/validate_test_coverage.py
uv run python scripts/z_generate_manuscript_variables.py
if rg -n '\{\{[A-Z][A-Z0-9_]*\}\}' output/manuscript/; then
  echo UNRESOLVED
  exit 1
else
  echo OK
fi

# Final renderer pass: receipt-backed hydration has already happened above.
cd "$TEMPLATE_REPO"
uv run python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
uv run python scripts/pipeline/stage_04_validate.py --project working/active_fedference
uv run python scripts/pipeline/stage_05_copy.py --project working/active_fedference

cd "$AF_REPO"
uv run python scripts/prepare_web_package.py
uv run python scripts/validate_web_package.py
uv run python scripts/validate_rendered_surfaces.py
TEMPLATE_COMMIT="$(git -C "$TEMPLATE_REPO" rev-parse HEAD)"
TEMPLATE_DIFF_SHA256="$(git -C "$TEMPLATE_REPO" diff --no-ext-diff --binary HEAD | shasum -a 256 | awk '{print $1}')"
uv run python scripts/record_pipeline_stage.py render \
  --renderer "template-03-05 commit=$TEMPLATE_COMMIT diff_sha256=$TEMPLATE_DIFF_SHA256 source_date_epoch=$SOURCE_DATE_EPOCH"
uv run --extra dev python scripts/validate_test_coverage.py --verify
uv run python scripts/validate_pipeline_freshness.py
~~~

The full-suite wrapper writes a separate successful receipt to
`output/data/test_coverage_receipt.json`, binding its command, test/coverage
summary, environment, source-owned documentation, manuscript sources, release metadata, `ISA.md`, `uv.lock`, and fresh analysis-stage digests. It snapshots those bound inputs and analysis digests immediately before pytest and compares them again after the suite, refusing to attest any concurrent drift. Successful
publication-scale analysis and final hydration runs write content-bound stage
receipts to `output/data/pipeline_provenance.json`. The provisional template
pass is deliberately unrecorded. The final render receipt is recorded only
after final stages 03–05 and web preparation, because web preparation rewrites
`output/web/` and the render receipt must hash the actual reader surface.

Receipt schema 2 omits volatile completion times by default, so recording an
unchanged stage is byte-identical. Use a canonical `--timestamp` or
`SOURCE_DATE_EPOCH` only when an external event supplies that value.

The renderer label is provenance metadata for an external producer; the receipt
still hashes every declared render input and output. A fresh-clone evidence
probe is separate from the local dirty development workflow:

```bash
uv run python scripts/validate_clean_checkout.py
```

It must be run from a checkout containing the committed required paths; a dirty
tree is reported as a failure rather than silently promoted to release evidence.

For repeated subprocess smoke checks, the same pipeline accepts an explicit
bounded real-computation profile:

```bash
uv run python scripts/02_run_analysis.py --profile smoke
```

The smoke profile is not a publication snapshot and must not replace the
publication-scale regeneration before release.

## Web publication package

```bash
uv run python scripts/prepare_web_package.py
uv run python scripts/validate_web_package.py
```

The prepare step mirrors `output/figures/*` into `output/web/figures/` and
normalizes both source-style and renderer-style cross-reference markers in
generated HTML. Individual pages link numbered targets in the combined
manuscript; theorem-like references without HTML anchors retain their
auto-generated number as text. Validation fails on missing local assets,
unresolved typed references, broken internal fragments, or leaked Markdown
figure syntax. It also fails on missing document language/title, skip/main
navigation, image alternatives, figure captions, full-size-link labels, or
duplicate identifiers. This is the automated subset of
[`../manuscript/accessibility.md`](../manuscript/accessibility.md), not a WCAG
conformance declaration.

## Rendered manuscript and slide surfaces

```bash
uv run python scripts/validate_rendered_surfaces.py
```

This gate checks every manuscript and slide PDF with both `qpdf --check` and
`pdftotext`, requires matching PDF/TeX/log slide triplets, scans every retained
manuscript and slide log for missing glyphs, undefined references, and material
layout overflow, and validates the local web package. Every retained log is
evidence-bearing: obsolete logs may be removed only after a current,
source-bound render establishes that the producer no longer emits them.
Warnings in any retained log fail the gate. These probes do not establish
tagged PDF or PDF/UA conformance.

## Markdown pre-render

```bash
TEMPLATE_REPO=/path/to/template
cd "$TEMPLATE_REPO"
uv run python -m infrastructure.validation.cli prerender \
  projects/working/active_fedference/manuscript --repo-root .
```

## Template render pipeline

```bash
TEMPLATE_REPO=/path/to/template
cd "$TEMPLATE_REPO"
uv run python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
uv run python scripts/pipeline/stage_04_validate.py --project working/active_fedference
uv run python scripts/pipeline/stage_05_copy.py --project working/active_fedference
```

This targeted form assumes explicit hydration has already completed. For the
source-current provisional and final pair, use the two-pass commands above;
do not let stage 03 invoke a premature non-provisional hydration.

## Project invariants script

```bash
uv run python scripts/01_run_invariants.py
```

## See also

- Agent checklist: [`../development/agent_instructions.md`](../development/agent_instructions.md)
- Scoped TODO pages: [`../todo/README.md`](../todo/README.md)
- Full acceptance table: [`../../ISA.md`](../../ISA.md) § Verification
