# Verification commands

Copy-paste probes from [`../../ISA.md`](../../ISA.md) and project validation.
Run local project probes from this repository root.

## Test + coverage gate (authoritative)

```bash
uv run --locked --extra dev pytest tests/ \
  --cov=src \
  --cov-fail-under=90
```

Expected: all collected tests pass with zero failures. Any skip must come from
an explicit dependency or unrendered-surface guard and remain visible in the
test summary; PyTorch is part of the required dev environment, so missing
PyTorch is a failing setup.

Fast and scoped profiles are available for iteration; they select real tests and
do not replace the full gate:

```bash
uv run --locked pytest tests/ -m "not slow" -q
uv run --locked pytest tests/ -m integration -q
uv run --locked pytest tests/ -m publication -q
```

## Local validation profiles

```bash
uv run --locked python scripts/validate_all.py quick
uv run --locked python scripts/validate_all.py manuscript
uv run --locked python scripts/validate_all.py package
uv run --locked python scripts/validate_all.py torch
uv run --locked python scripts/validate_all.py source
uv run --locked python scripts/validate_all.py freshness
uv run --locked python scripts/validate_all.py full
```

`--dry-run` prints every command without running it. `--keep-going` runs every
command in a profile before returning failure.

Profile scope:

- `quick`: docs contract, caption completeness, and the required Torch smoke.
- `manuscript`: cross-reference, caption, token-provenance, token-table, and
  manuscript-variable checks.
- `package`: web-publication figure mirroring, cross-reference normalization,
  and prepared-web-package validation; it is not the wheel/sdist build lane.
- `torch`: explicit required PyTorch lane via `uv run --locked --extra dev`.
- `freshness`: the standalone successful test/coverage receipt plus the
  content-hashed analysis → hydration → render stage receipts.
- `source`: Ruff across source, tests, scripts, and runnable examples; mypy;
  invariants; domain-layer grep; and exact-set release build/verify.
- `full`: quick + manuscript + package + rendered-surface + freshness + source
  + full coverage gate.

## Run root-aware scripts from another checkout

Scripts default to the checkout containing the entry point. For a sibling
checkout or clean-clone probe, pass an explicit root; it takes precedence over
the `ACTIVE_FEDFERENCE_PROJECT_ROOT` review override:

```bash
uv run --locked python scripts/02_run_analysis.py \
  --project-root /path/to/active_fedference --profile smoke
uv run --locked python scripts/validate_outputs.py \
  --project-root /path/to/active_fedference
uv run --locked python scripts/validate_mermaid.py \
  --project-root /path/to/active_fedference
```

`validate_outputs.py` uses the canonical `analysis.artifacts.expected_artifacts()`
contract and fails
closed if that contract is unavailable; it never treats an arbitrary set of
files under `output/` as a valid substitute. `00_preflight.py` additionally
accepts `--template-root PATH` for the optional sibling template renderer.

## Public API, registry, and receipt verification

```bash
FEDFERENCE_VERIFY_ROOT="$(mktemp -d /tmp/active-fedference-verify.XXXXXX)"

# Primary labeled own-data path and application receipt.
uv run --locked fedference aggregate \
  --input examples/data/labeled_aggregation_request.json \
  --output-dir "$FEDFERENCE_VERIFY_ROOT/application" \
  --project-root .
uv run --locked fedference verify \
  "$FEDFERENCE_VERIFY_ROOT/application/receipt.json" \
  --require-nominal-solver

# Registered research path and research receipt.
uv run --locked fedference list --json

uv run --locked fedference run server-theory \
  --profile smoke --seed 0 \
  --output-dir "$FEDFERENCE_VERIFY_ROOT/server-theory" \
  --project-root .

uv run --locked fedference verify \
  "$FEDFERENCE_VERIFY_ROOT/server-theory/receipt.json"
```

Write-producing commands require an explicit empty directory and reject the
committed `output/` tree. An application receipt proves its declared
input/output/provenance integrity, not scientific validity or a downstream
decision. Registry state is a work declaration, not a scientific result. Smoke
and pilot runs do not support manuscript claims.

The bounded single-machine research pilots use the same receipt contract:

```bash
FEDFERENCE_PILOT_ROOT="$(mktemp -d /tmp/active-fedference-pilots.XXXXXX)"
uv run --locked fedference run robustness-calibration \
  --profile pilot --seed 0 \
  --output-dir "$FEDFERENCE_PILOT_ROOT/calibration"
uv run --locked fedference run fedgvi-bnn \
  --profile pilot --seed 0 --device cpu \
  --output-dir "$FEDFERENCE_PILOT_ROOT/fedgvi-bnn"
uv run --locked fedference run hybrid-tracking \
  --profile pilot --seed 0 \
  --output-dir "$FEDFERENCE_PILOT_ROOT/hybrid"
uv run --locked fedference run hierarchy-tasks \
  --profile pilot --seed 0 --seed 1 \
  --output-dir "$FEDFERENCE_PILOT_ROOT/hierarchy"
uv run --locked fedference run friston-protocol \
  --profile pilot --seed 0 \
  --output-dir "$FEDFERENCE_PILOT_ROOT/friston-parity"
```

These reports retain the calibration overlap, BNN checkpoint/device,
hybrid-control/singular-covariance, hierarchy-task, and Friston
analogue-relabeling negative controls. They remain pilot evidence until the
corresponding confirmatory budgets and manuscript contracts are frozen.

Configuration parity across the rich interface and legacy wrapper:

```bash
uv run --locked python -c "
import numpy as np
from fedference import AggregationConfig, aggregate, aggregate_result
b = np.asarray([[.7, .3], [.6, .4]])
c = AggregationConfig(method='robust', robustness=1.5)
assert np.array_equal(aggregate_result(b, config=c).consensus, aggregate(b, config=c))
print(c.fingerprint)
"
```

## Executable example ladder

Every numbered example is a real public-surface smoke program. Run the complete
ladder and its subprocess/output contract with:

```bash
mkdir -p .tmp
EXAMPLE_SMOKE=$(mktemp -d .tmp/example-smoke.XXXXXX)
uv run --locked python examples/01_minimal_aggregation.py
uv run --locked python examples/02_compare_aggregation_methods.py
uv run --locked python examples/03_federation_boundaries.py \
  --output-dir "$EXAMPLE_SMOKE/federation"
uv run --locked python examples/04_cli_receipt_workflow.py \
  --output-dir "$EXAMPLE_SMOKE/cli" --project-root .
uv run --locked python examples/05_labeled_application.py \
  --input examples/data/labeled_aggregation_request.json \
  --output-dir "$EXAMPLE_SMOKE/application" --project-root .
uv run --locked pytest tests/test_examples.py -q
```

The examples are linted and scanned by the no-placeholder/no-test-double gate.
They are archival source-distribution and provenance inputs, not wheel runtime
modules or generated reviewer artifacts.

## Wheel and source-distribution smoke

```bash
mkdir -p .tmp
CHECKOUT_ROOT="$(pwd -P)"
DIST_SMOKE="$(mktemp -d "$CHECKOUT_ROOT/.tmp/distribution-smoke.XXXXXX")"
INSTALLED_SMOKE="$(mktemp -d /tmp/active-fedference-installed.XXXXXX)"
export SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"
uv build --out-dir "$DIST_SMOKE/dist"
uv build --out-dir "$DIST_SMOKE/rebuilt"
cmp "$DIST_SMOKE"/dist/*.whl "$DIST_SMOKE"/rebuilt/*.whl
cmp "$DIST_SMOKE"/dist/*.tar.gz "$DIST_SMOKE"/rebuilt/*.tar.gz
sdist="$(find "$DIST_SMOKE/dist" -maxdepth 1 -name '*.tar.gz' -print -quit)"
for member in LICENSE docs/README.md docs/application-guide.md docs/development/modularity.md examples/README.md examples/01_minimal_aggregation.py examples/02_compare_aggregation_methods.py examples/03_federation_boundaries.py examples/04_cli_receipt_workflow.py examples/05_labeled_application.py examples/data/labeled_aggregation_request.json manuscript/config.yaml manuscript/config.yaml.example scripts/02_run_analysis.py src/fedference/py.typed src/fedference_cli/README.md tests/README.md; do
  tar -tzf "$sdist" | grep -Eq "/$member$"
done
cp examples/data/labeled_aggregation_request.json "$INSTALLED_SMOKE/request.json"
uv venv "$DIST_SMOKE/wheel-env"
uv pip install --python "$DIST_SMOKE/wheel-env/bin/python" "$DIST_SMOKE"/dist/*.whl
(
  cd "$INSTALLED_SMOKE"
  "$DIST_SMOKE/wheel-env/bin/python" -c "import sys; from importlib.metadata import version; from importlib.resources import files; import fedference; assert fedference.__version__ == version('active_fedference'); assert files('fedference').joinpath('py.typed').is_file(); assert 'torch' not in sys.modules"
  "$DIST_SMOKE/wheel-env/bin/fedference" aggregate --input request.json --output-dir wheel-run
  "$DIST_SMOKE/wheel-env/bin/fedference" verify wheel-run/receipt.json --require-nominal-solver
)
uv venv "$DIST_SMOKE/sdist-env"
uv pip install --python "$DIST_SMOKE/sdist-env/bin/python" "$DIST_SMOKE"/dist/*.tar.gz
(
  cd "$INSTALLED_SMOKE"
  "$DIST_SMOKE/sdist-env/bin/python" -c "import sys; from importlib.metadata import version; from importlib.resources import files; import fedference; assert fedference.__version__ == version('active_fedference'); assert files('fedference').joinpath('py.typed').is_file(); assert 'torch' not in sys.modules"
  "$DIST_SMOKE/sdist-env/bin/fedference" aggregate --input request.json --output-dir sdist-run
  "$DIST_SMOKE/sdist-env/bin/fedference" verify sdist-run/receipt.json --require-nominal-solver
)
```

The default install must import the NumPy/SciPy core without Torch. The optional
BNN/MPS lane is installed separately with `uv sync --locked --extra bnn`. The PEP 517
backend is exactly pinned and normalizes wheel/sdist archive metadata when
`SOURCE_DATE_EPOCH` is set. The two directories above must therefore contain
byte-identical wheel and sdist pairs; a differing digest is a release blocker,
even when both artifacts install successfully.

The source distribution is the archival source package: it includes `LICENSE`,
the modular `docs/`, `examples/`, `manuscript/`, `scripts/`, and `tests/` trees,
including the application guide, every numbered example, example data, and the
copyable `manuscript/config.yaml.example`, plus source-bound metadata and
acceptance files. The wheel is the typed runtime package; it includes
`fedference/py.typed`, importable modules, and packaged compatibility inputs,
but not examples or committed reviewer output. Both installed artifact probes
run outside the checkout and confirm that the default application import graph
does not load Torch.

## Pinned external-data smoke

```bash
FEDFERENCE_BENCHMARK_ROOT="$(mktemp -d /tmp/active-fedference-benchmark.XXXXXX)"
uv run --locked fedference benchmark \
  --dataset-id uci-banknote --profile smoke --seed 42 \
  --cache-dir "$FEDFERENCE_BENCHMARK_ROOT/cache" \
  --output-dir "$FEDFERENCE_BENCHMARK_ROOT/run"
uv run --locked fedference verify \
  "$FEDFERENCE_BENCHMARK_ROOT/run/receipt.json"
uv run --locked fedference verify \
  "$FEDFERENCE_BENCHMARK_ROOT/run/receipt.json" \
  --require-clean-git --project-root .
```

The first verification checks archive/member hashes, schema, train-only
preprocessing, split hashing, bound `config.json`/`report.json` bytes, and
receipt consistency. The strict second form also matches the live full commit,
clean Git tree, and `uv.lock` digest against the receipt; pass
`--project-root /path/to/checkout` when running outside that checkout. Neither
is the confirmatory three-dataset evidence pack.

## Central identity (spine — both server axes + V1 default bit-identity)

```bash
uv run --locked python -c "
from fedference.aggregation import robust_aggregate, variational_aggregate, log_linear_pool
import numpy as np
b = [[.7,.3],[.6,.4]]
assert np.allclose(robust_aggregate(b, robustness=0).consensus, log_linear_pool(b))
assert np.allclose(variational_aggregate(b, robustness=0).consensus, log_linear_pool(b))
assert np.allclose(variational_aggregate(b, robustness=0, entropy_weight=1.0).consensus, log_linear_pool(b))
print('OK')
"
```

Run from the repository root via `uv run --locked`.

## N-level hierarchical spine (V2)

```bash
uv run --locked python -c "
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
uv run --locked python -c "
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

This checkout is a standalone repository (public release target:
ActiveInferenceInstitute/Active_Fedference). The top level should be this
repository root, not the sibling public template checkout.

## No mocks

```bash
uv run --locked pytest tests/test_runtime_surface.py -q
```

The dedicated test scans executable Python under `src/`, `scripts/`, and
`examples/`, plus the test-double APIs under `tests/`, while excluding only its
own pattern declarations. A raw recursive grep is not the gate because it also
matches the forbidden-name documentation in `tests/PATTERNS.md` and the
detector's own regex.

## Zenodo linked-version boundary

Exercise both documented new-version service shapes through the real
standard-library HTTP client against loopback handlers; this probe uses no
production token and makes no request to Zenodo:

```bash
uv run --locked pytest tests/test_zenodo.py -q
uv run ruff check src/publication/zenodo.py scripts/zenodo_release.py \
  tests/test_zenodo.py
uv run mypy src/publication/zenodo.py scripts/zenodo_release.py
```

The acceptance matrix requires either exact legacy metadata/file inheritance
or current-service purpose-metadata parity with creation-date normalization,
an omitted-or-unchanged version, and an empty file set. Wrong lineage or DOI,
an arbitrary date, other metadata drift, and partial, inherited/mixed, or
unrelated files in the current-service shape must fail before any later draft
mutation or publication operation. The legacy shape may omit `created`; the
current shape requires strict RFC 3339 with an explicit offset and no more than
six fractional digits. Loopback recovery treats a source `latest_draft`
self-link as the published source rather than a draft, so it cannot preempt the
POST. Every draft link must be an absolute exact-origin URL on the configured
API base, including matching scheme, hostname, and effective port, with the
exact `/deposit/depositions/<positive-id>` suffix and at most one trailing
slash. The loopback negative matrix covers foreign/changed origins, relative
and protocol-relative forms, missing netloc, malformed schemes, credentials,
queries, fragments, wrong ports, arbitrary prefixes, extra slashes, and invalid
identifiers; the link is never followed directly. Recovery uses a distinct
source link when available; after the exact
already-exists HTTP shape, a self-link or absent link falls back to one bounded
`status=draft`/`all_versions=true` listing. Tests require full objects, local
exact concept filtering, exactly one distinct unsubmitted draft, identical full
revalidation, deterministic query encoding, and failures for zero, multiple,
truncated, malformed, partial, and wrong-concept-only results. Unrelated HTTP
failures remain failures. An echoed-token error probe inspects `repr`, `str`,
`vars`, and the exception dictionary and requires that neither the bearer token
nor a raw response payload is retained.

## Release bundle + provenance fingerprint

```bash
uv run --locked python scripts/build_release.py
uv run --locked python scripts/build_release.py --verify
```

Build writes `output/release/` (`manifest.json`, `sha256sums.txt`, a derived
`README.md`). Schema 4 declares an exact
`publication-payload-v1` scope: every listed payload artifact must exist with
its recorded byte size and SHA-256, no unlisted payload file may appear under
the release roots, and a listed downstream-control path is rejected even when
its own digest is correct. Construction and verification also reject symlinked
path components and filesystem aliases whose spelling is not an independently
discovered canonical payload path. The manifest also records a provenance
`fingerprint` — a
SHA-256 over the declared source, examples, manuscript, documentation,
producer-script, dependency-lock, and claim-audit inputs — together with the
pipeline profile, generator version, and individual input digests. `--verify`
recomputes these from the current tree, so a bundle whose bytes all match but
which was built from a different producer or evidence state still fails as
stale and names changed inputs.

Template-owned artifact/evidence/output-statistics/validation/
rendered-provenance reports and `output/reports/snapshots/` are downstream of
the payload manifest and are intentionally excluded from it. Verify those
after the release bytes exist by refreshing the artifact manifest in the
pinned Template checkout, running Template stages 04-05, and then running the
rendered, coverage-receipt, pipeline-freshness, and release `--verify` gates.
The Git tree, confidentiality evidence manifest, and isolated-clone comparison
bind both layers together.

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
uv run --locked python -c "
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
uv run --locked python -c '
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
uv run --locked python scripts/00_preflight.py --template-root "$TEMPLATE_REPO"
uv run --locked python scripts/02_run_analysis.py
uv run --locked python scripts/z_generate_manuscript_variables.py --provisional-validation

# Provisional renderer pass: hydrate has already happened above.
cd "$TEMPLATE_REPO"
uv run --locked python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
uv run --locked python scripts/pipeline/stage_04_validate.py --project working/active_fedference
uv run --locked python scripts/pipeline/stage_05_copy.py --project working/active_fedference

cd "$AF_REPO"
uv run --locked --extra dev python scripts/validate_test_coverage.py
uv run --locked python scripts/z_generate_manuscript_variables.py
# Hydrated Markdown must be token-free. Auxiliary config/preamble/BibTeX files
# remain source-exact and are validated by their consumer-specific producers.
if rg -n --glob '*.md' '\{\{[A-Z][A-Z0-9_]*\}\}' output/manuscript/; then
  echo UNRESOLVED
  exit 1
else
  echo OK
fi

# Final renderer pass: receipt-backed hydration has already happened above.
cd "$AF_REPO"
uv run --locked python scripts/00_preflight.py --template-root "$TEMPLATE_REPO"
cd "$TEMPLATE_REPO"
uv run --locked python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
uv run --locked python scripts/pipeline/stage_04_validate.py --project working/active_fedference
uv run --locked python scripts/pipeline/stage_05_copy.py --project working/active_fedference

cd "$AF_REPO"
uv run --locked python scripts/prepare_web_package.py
uv run --locked python scripts/validate_web_package.py
uv run --locked python scripts/validate_rendered_surfaces.py
uv run --locked python scripts/record_pipeline_stage.py render \
  --template-root "$TEMPLATE_REPO"
uv run --locked --extra dev python scripts/validate_test_coverage.py --verify
uv run --locked python scripts/validate_pipeline_freshness.py
~~~

The full-suite wrapper writes a separate successful receipt to
`output/data/test_coverage_receipt.json`, binding its command, test/coverage
summary, environment, source-owned documentation, manuscript sources, release metadata, `ISA.md`, `uv.lock`, and fresh analysis-stage digests. It snapshots those bound inputs and analysis digests immediately before pytest and compares them again after the suite, refusing to attest any concurrent drift. Successful
publication-scale analysis and final hydration runs write content-bound stage
receipts to `output/data/pipeline_provenance.json`. The provisional template
pass is deliberately unrecorded. The final render receipt is recorded only
after final stages 03–05 and web preparation, because web preparation rewrites
`output/web/` and the render receipt must hash the actual reader surface.

Receipt schema 4 omits volatile completion times by default, so recording an
unchanged stage is byte-identical. Use a canonical `--timestamp` or
`SOURCE_DATE_EPOCH` only when an external event supplies that value.

The renderer identity is accepted only from the explicit clean Template
checkout named by `--template-root`. Its canonical repository and exact commit
must match the source-owned lock in `manuscript/config.yaml`; the receipt stores
that structured identity without retaining a machine-local path or raw remote
URL. A fresh-clone evidence probe is separate from the local dirty development
workflow:

```bash
uv run --locked python scripts/validate_clean_checkout.py
```

It must be run from a checkout containing the committed required paths; a dirty
tree is reported as a failure rather than silently promoted to release evidence.
The probe also requires all five historical top-level release PDFs (v0.1.0
through v1.0.4) and verifies their bytes against
[`historical-release-pdfs.json`](historical-release-pdfs.json). The release
bundle preflight repeats that digest check, so tracking a substituted file does
not satisfy the immutable-release boundary.

For repeated subprocess smoke checks, the same pipeline accepts an explicit
bounded real-computation profile:

```bash
uv run --locked python scripts/02_run_analysis.py --profile smoke
```

The smoke profile is not a publication snapshot and must not replace the
publication-scale regeneration before release.

## Web publication package

```bash
uv run --locked python scripts/prepare_web_package.py
uv run --locked python scripts/validate_web_package.py
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

The publication validator intentionally accepts a smaller HTML language than a
general-purpose browser. It rejects every `<base>`, CDATA, and `<noscript>`
element because those constructs can change resource or parsing semantics that
the static publication contract does not model. External scripts require an
explicit HTTPS URL, syntactically valid SHA-256/384/512 integrity metadata, and
anonymous CORS mode; every integrity token must be supported and valid. This is
a fail-closed packaging policy. It validates metadata shape and browser-
effective attributes, but it does not fetch the remote response, prove that the
declared digest matches its bytes, or establish server-side CORS behavior.

## Rendered manuscript and slide surfaces

```bash
uv run --locked python scripts/validate_rendered_surfaces.py
```

This gate checks every manuscript and slide PDF with both `qpdf --check` and
`pdftotext`, requires matching PDF/TeX slide pairs, scans every local manuscript
and slide log for missing glyphs, undefined references, and material layout
overflow, and validates the local web package. When any slide logs are present,
the gate requires a complete PDF/TeX/log set and warnings fail the gate.
Renderer logs are retained in the external verification namespace, not the
public Git history: XeTeX and LuaLaTeX embed wall-clock and machine-local path
details even under a deterministic source epoch. A log-free clean checkout is
therefore valid after the producing run's log inspection is receipted. Other
LaTeX/Beamer build sidecars (`aux`, `bbl`, `blg`, `lof`, `lot`, `nav`, `out`,
`snm`, `toc`, and `vrb`) are likewise excluded from the durable public payload.
The manuscript-PDF branch
additionally enforces the source-requested tagged structure (`Tagged: yes`,
qpdf-visible `/Lang`, language, and `StructTreeRoot`). The validator accepts
the catalog `/Lang` when a Poppler build omits its optional `Language:` line.
These probes do not establish PDF/UA conformance; retain and inspect the
dedicated conformance report and complete the manual accessibility review
before making that claim.

## Markdown pre-render

```bash
TEMPLATE_REPO=/path/to/template
cd "$TEMPLATE_REPO"
uv run --locked python -m infrastructure.validation.cli prerender \
  projects/working/active_fedference/manuscript --repo-root .
```

## Template render pipeline

```bash
TEMPLATE_REPO=/path/to/template
cd "$TEMPLATE_REPO"
uv run --locked python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
uv run --locked python scripts/pipeline/stage_04_validate.py --project working/active_fedference
uv run --locked python scripts/pipeline/stage_05_copy.py --project working/active_fedference
```

This targeted form assumes explicit hydration has already completed. For the
source-current provisional and final pair, use the two-pass commands above;
do not let stage 03 invoke a premature non-provisional hydration.

## Project invariants script

```bash
uv run --locked python scripts/01_run_invariants.py
```

## See also

- Agent checklist: [`../development/agent_instructions.md`](../development/agent_instructions.md)
- Scoped TODO pages: [`../todo/README.md`](../todo/README.md)
- Full acceptance table: [`../../ISA.md`](../../ISA.md) § Verification
