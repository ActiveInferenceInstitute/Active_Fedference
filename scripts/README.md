# scripts/ — Quick reference

Thin orchestrators for Active Fedference. Domain and publication logic lives in
`../src/`; scripts own the command-line contract, checkout-root selection,
subprocess sequencing, stdout/status mapping, and release-boundary safety.
The installed CLI follows the same split; see
[`../src/fedference_cli/README.md`](../src/fedference_cli/README.md) and the
cross-layer [`../docs/development/modularity.md`](../docs/development/modularity.md).

## Why thin orchestrators are useful

The scripts are deliberately useful even though they do not contain the
research algorithms. They provide stable subprocess and CI boundaries with
predictable exit codes, explicit artifact paths, and one documented way to run
the same operation from the checkout, a sibling checkout, or a clean-clone
probe. The reusable implementation remains importable for unit and integration
tests, library callers, and the installed CLI. This separation also keeps
publication actions auditable: a script may require a receipt, validate a
surface, or refuse a stale input, but it must not invent a result or silently
discover a different output tree.

The thin boundary does not mean every script is a one-line wrapper. Specialized
validators such as `validate_mermaid.py` and `validate_outputs.py` own their
format-specific parsing and status reporting; their checks are still
dependency-light, source-owned, and fail closed. Numeric research logic stays
in `src/`.

| Script | Output | Status |
| --- | --- | --- |
| `02_run_analysis.py [--profile publication|smoke] [--project-root PATH]` | `output/reports/*.json`, `output/figures/*.png` | REQUIRED (stage 4) |
| `record_pipeline_stage.py render [--timestamp UTC] [--project-root PATH]` | `output/data/pipeline_provenance.json` | REQUIRED after the completed, validated external render boundary; analysis/hydration receipts are producer-owned |
| `validate_test_coverage.py [--verify] [--project-root PATH]` | `output/data/test_coverage_receipt.json` | REQUIRED full-suite receipt before final hydration |
| `z_generate_manuscript_variables.py [--provisional-validation] [--project-root PATH]` | `output/data/manuscript_variables.json`, `output/manuscript/` | REQUIRED; final non-draft mode consumes the test/coverage receipt, while provisional mode never records hydration |
| `prepare_web_package.py [--project-root PATH]` | Mirrored figures and numbered cross-format web references | REQUIRED for web package QA |
| `validate_web_package.py [--project-root PATH]` | Asset, reference-target, markup, and deterministic HTML-accessibility validation status | REQUIRED for web package QA |
| `validate_rendered_surfaces.py [--project-root PATH]` | PDF/slide/web text, layout-warning, link, asset, and HTML-accessibility checks | REQUIRED before release |
| `validate_all.py PROFILE [--project-root PATH]` | Local validation profiles | REQUIRED before release |
| `validate_pipeline_freshness.py [--project-root PATH]` | Content-hashed stage dependency validation | REQUIRED before release |
| `validate_clean_checkout.py [--project-root PATH]` | Clean Git/tracking/import probe | REQUIRED for fresh-checkout release evidence |
| `build_release.py [--verify] [--timestamp UTC] [--project-root PATH]` | Metadata-, surface-, and receipt-preflighted local reviewer bundle (default) or verifies an existing one | REQUIRED before release |
| `emit_metadata.py [--check\|--write] [--project-root PATH]` | Checks or regenerates the generated metadata surfaces (`publication.metadata`) | REQUIRED before release |
| `zenodo_release.py [--project-root PATH]` | Create a linked `--new-version-of` draft or `--reserve` one, update draft metadata, optionally correct published metadata in place with `--edit-published-metadata`, upload with explicit `--replace-existing`, verify, and explicitly publish; `--confirm-publish` is required | Release boundary |
| `validate_mermaid.py [--project-root PATH]` | Validate README/docs Mermaid fences; optionally render every block to SVG | Documentation/publication QA |
| `01_run_invariants.py [--project-root PATH]` | Invariant report (stdout) | Optional |
| `00_preflight.py [--project-root PATH] [--template-root PATH]` | Environment diagnostics | Optional |
| `summarize_tokens.py [--project-root PATH]` | Summary of all manuscript tokens and their resolved values (stdout) | Optional audit |
| `validate_outputs.py [--project-root PATH]` | Checks all expected Stage-02 figures/reports/variables exist and are non-empty | Optional audit |
| `generate_api_docs.py [--project-root PATH]` | `output/docs/api_reference.md` | Aesthetic |
| `_generate_api_docs.py [--project-root PATH]` | Compatibility shim for `generate_api_docs.py` | Aesthetic |

## Root selection

Every project-root-aware entry point defaults to the checkout containing the
script. `ACTIVE_FEDFERENCE_PROJECT_ROOT` is a validated test/review override;
an explicit `--project-root PATH` takes precedence over that environment
variable. Relative metadata, upload, verification, output, and scratch paths
are resolved against the selected root where the command owns those paths.
Invalid explicit roots fail rather than redirecting writes to the script's
checkout. `00_preflight.py` additionally accepts `--template-root PATH` for
the optional sibling template rendering environment.

```bash
uv run --locked python scripts/02_run_analysis.py                  # publication profile from config
uv run --locked python scripts/02_run_analysis.py --profile smoke # bounded real smoke profile
uv run --locked python scripts/z_generate_manuscript_variables.py --provisional-validation
uv run --locked --extra dev python scripts/validate_test_coverage.py
uv run --locked python scripts/z_generate_manuscript_variables.py
uv run --locked python scripts/validate_pipeline_freshness.py
uv run --locked python scripts/validate_clean_checkout.py --skip-imports
uv run --locked python scripts/prepare_web_package.py
uv run --locked python scripts/validate_web_package.py
```

To drive a sibling checkout without changing the working directory, add the
same explicit root to each command:

```bash
uv run --locked python scripts/02_run_analysis.py \
  --project-root /path/to/active_fedference --profile smoke
uv run --locked python scripts/validate_outputs.py \
  --project-root /path/to/active_fedference
```

Validate the shared GitHub/local Mermaid source before a documentation or
publication review:

```bash
uv run --locked python scripts/validate_mermaid.py
uv run --locked python scripts/validate_mermaid.py --render --renderer npx \
  --output-dir .tmp/mermaid-render
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

The `freshness` profile verifies the standalone successful test/coverage receipt
plus the content-hashed analysis → hydration → render dependency chain and the
producer-owned publication execution profile. A publication analysis sidecar
also binds a pre-run input snapshot and only the canonical
`manuscript/config.yaml` configuration may mint it. The `source` profile is the bounded source/release
gate: Ruff, mypy, the invariant report, the domain-layer grep, and exact-set
release build/verify. `full` runs quick, manuscript, package, rendered-surface,
freshness, source, and coverage gates.

Use `--dry-run` to print a profile without executing it and `--keep-going` to
run all commands in a profile before returning failure.

## Release bundle and provenance fingerprint

`build_release.py` first requires source-current generated metadata, live
PDF/slide/web surface validation, a publication-profile analysis, test/coverage
receipt, final hydration, and render receipt chain. It then writes
`output/release/` (manifest plus copied artifacts); `--verify` performs the
same local preflight before re-checking an existing bundle without writing. The
byte-manifest primitive lives in `src/publication/release_manifest.py`; its
direct API is not publication authorization.

`--provisional-validation` writes hydrated input only for the pre-test render;
it never records a hydration receipt and therefore cannot advance the
release-facing dependency chain.

The manifest records a deterministic provenance `fingerprint`: a SHA-256 over
the sorted `(path, content-sha256)` set of the declared source, manuscript,
documentation, producer-script, dependency-lock, and claim-audit inputs
(`fingerprint_inputs`), with no timestamps or file metadata in the hash. It
also records the pipeline profile, generator version, and individual input
digests. On `--verify`, per-artifact byte digests and those input digests are
checked; if the tree drifted since the bundle was built, verification fails
with an explicit changed-path diagnostic so a stale bundle cannot pass.
This proves a source-current local reviewer bundle, not clean-clone
reproducibility, author approval, DOI readiness, or external publication
authority.

The default unreleased build records `generated_at: null`; this deliberately
omits wall-clock time so two builds of the same evidence tree are byte-identical.
After release approval, provide a canonical UTC time with
`--timestamp YYYY-MM-DDTHH:MM:SSZ` or set the reproducible-build standard
`SOURCE_DATE_EPOCH`. Do not add an approval date to a reviewer snapshot.

## Stage receipt and sequence

Run stages in dependency order. The canonical explanation is in the
[rendering guide](../docs/manuscript/rendering_pipeline.md); the two renderer
invocations below are intentional: the first consumes provisional hydration
before the full-suite receipt, and the second consumes receipt-backed final
hydration.

```bash
AF_REPO=/path/to/active_fedference
TEMPLATE_REPO=/path/to/template
export SOURCE_DATE_EPOCH="$(git -C "$AF_REPO" log -1 --format=%ct)"

cd "$AF_REPO"
uv run --locked python scripts/02_run_analysis.py
uv run --locked python scripts/z_generate_manuscript_variables.py --provisional-validation

cd "$TEMPLATE_REPO"
uv run --locked python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
uv run --locked python scripts/pipeline/stage_04_validate.py --project working/active_fedference
uv run --locked python scripts/pipeline/stage_05_copy.py --project working/active_fedference

cd "$AF_REPO"
uv run --locked --extra dev python scripts/validate_test_coverage.py
uv run --locked python scripts/z_generate_manuscript_variables.py

cd "$TEMPLATE_REPO"
uv run --locked python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
uv run --locked python scripts/pipeline/stage_04_validate.py --project working/active_fedference
uv run --locked python scripts/pipeline/stage_05_copy.py --project working/active_fedference

cd "$AF_REPO"
uv run --locked python scripts/prepare_web_package.py
uv run --locked python scripts/validate_web_package.py
uv run --locked python scripts/validate_rendered_surfaces.py

TEMPLATE_COMMIT="$(git -C "$TEMPLATE_REPO" rev-parse HEAD)"
TEMPLATE_DIFF_SHA256="$(git -C "$TEMPLATE_REPO" diff --no-ext-diff --binary HEAD | shasum -a 256 | awk '{print $1}')"
uv run --locked python scripts/record_pipeline_stage.py render \
  --renderer "template-03-05 commit=$TEMPLATE_COMMIT diff_sha256=$TEMPLATE_DIFF_SHA256 source_date_epoch=$SOURCE_DATE_EPOCH"
uv run --locked --extra dev python scripts/validate_test_coverage.py --verify
uv run --locked python scripts/validate_pipeline_freshness.py
unset SOURCE_DATE_EPOCH
uv run --locked python scripts/build_release.py
uv run --locked python scripts/build_release.py --verify
```

The stage receipts store SHA-256 maps for declared stage inputs and outputs, so
a changed report blocks hydration/render freshness rather than being silently
treated as an up-to-date timestamp. `prepare_web_package.py` must precede
the render receipt because it mirrors assets and rewrites cross-references
under `output/web/`, which the receipt hashes. The standalone full-suite receipt also
stores matching pre- and post-test snapshots of its bound inputs and analysis
receipt; it refuses to attest a tree edited while pytest was running. The
clean-checkout probe is intentionally separate: it reports whether the current
Git tree is clean and clone-correct, and therefore adds no evidence when run in
a dirty development checkout.
Schema 2 records `recorded_at: null` by default, making a no-op stage receipt
byte-idempotent. A canonical `--timestamp` or `SOURCE_DATE_EPOCH` is optional
external metadata, not permission to skip content freshness.

Details: [`AGENTS.md`](AGENTS.md) · [`../docs/manuscript/rendering_pipeline.md`](../docs/manuscript/rendering_pipeline.md)
