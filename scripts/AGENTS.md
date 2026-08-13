# scripts/ — Active Fedference thin orchestrators

Pipeline-facing entry points. **No domain math** — delegate to `../src/`.
Thin scripts are a deliberate process boundary: they give CI and operators
stable arguments, exit codes, and artifact paths while keeping the reusable
implementation importable. Format-specific validators may parse and report
their format, but numeric research logic remains in `../src/`.
Conventions: [`CONVENTIONS.md`](CONVENTIONS.md) · Architecture:
[`../docs/core/architecture.md`](../docs/core/architecture.md).

## Scripts

| Script | Pipeline | Role |
| --- | --- | --- |
| `02_run_analysis.py` | Stage 4 (REQUIRED) | Calls `analysis.workflow.run_analysis_pipeline()` |
| `record_pipeline_stage.py` | Provenance (REQUIRED) | Validates the completed external PDF/slide/web surfaces, then records the render boundary; analysis and hydration receipts are producer-owned |
| `z_generate_manuscript_variables.py` | Pre-render (REQUIRED) | Final non-draft hydration requires fresh analysis plus the successful validation receipt; provisional mode is pre-test only and never records hydration |
| `validate_test_coverage.py` | Validation (REQUIRED) | Runs the full coverage gate and atomically writes/verifies `output/data/test_coverage_receipt.json` |
| `build_release.py` | Release (REQUIRED) | Requires current metadata, PDF/slide/web validation, and provenance before `build_release()` / `verify_release()`; default builds `output/release/`, `--verify` re-checks it |
| `emit_metadata.py` | Release (REQUIRED) | Checks (`--check`) or regenerates (`--write`) the generated metadata surfaces via `publication.metadata` |
| `zenodo_release.py` | Release boundary | Reserves, updates, uploads, verifies, or explicitly publishes the configured Zenodo deposition; publication requires confirmation |
| `validate_mermaid.py` | Documentation/publication | Checks every README/docs Mermaid fence and optionally renders every block to SVG |
| `validate_rendered_surfaces.py` | Release (REQUIRED) | Checks every manuscript/slide PDF structurally and textually, enforces slide triplets, rejects material LaTeX warnings, and validates web links/assets/accessibility invariants |
| `prepare_web_package.py` | Publication (REQUIRED) | Mirrors declared assets and builds numbered cross-format web references |
| `validate_web_package.py` | Publication (REQUIRED) | Checks mirrored assets, targets, rendered markup, and deterministic HTML accessibility invariants |
| `validate_all.py` | Validation (REQUIRED) | Runs named quick/manuscript/package/torch/source/freshness/full profiles |
| `validate_pipeline_freshness.py` | Release (REQUIRED) | Verifies the content-hashed analysis → hydration → render receipt chain |
| `validate_clean_checkout.py` | Release evidence | Probes Git cleanliness, required tracking, and importability from a fresh checkout |
| `validate_outputs.py` | Audit | Checks expected Stage-02 artifacts exist and are non-empty |
| `summarize_tokens.py` | Audit | Prints resolved manuscript-token provenance |
| `01_run_invariants.py` | Optional | Runs `invariants` checks |
| `00_preflight.py` | Optional | Chrome/LaTeX environment warnings |
| `generate_api_docs.py` | Aesthetic | Writes `output/docs/api_reference.md` |
| `_generate_api_docs.py` | Compatibility | Delegates to `generate_api_docs.py`; retained for older local automation |

All entries that read or write a checkout accept `--project-root PATH` (the
legacy underscored API-doc entry point delegates to the public one). The
explicit argument takes precedence over the validated
`ACTIVE_FEDFERENCE_PROJECT_ROOT` review override. Use the shared helper in
`src/project_paths.py`; do not duplicate root-selection logic. The canonical
artifact validator is `analysis.artifacts.expected_artifacts()` and
`validate_outputs.py` is intentionally fail-closed if it is unavailable.

## Stage 4 entry

```bash
uv run --locked python scripts/02_run_analysis.py
```

Prints artifact paths from `run_analysis_pipeline()`.

## Variable hydration

```bash
# Pre-test pass, only when a renderer needs hydrated input for the full suite.
uv run --locked python scripts/z_generate_manuscript_variables.py --provisional-validation
uv run --locked --extra dev python scripts/validate_test_coverage.py
# Final release-facing hydration consumes the fresh receipt.
uv run --locked python scripts/z_generate_manuscript_variables.py
```

## Rendering context

Full pipeline: [`../docs/manuscript/rendering_pipeline.md`](../docs/manuscript/rendering_pipeline.md).

Before a documentation or release review, validate the shared GitHub/local
Mermaid source and run the actual renderer probe:

```bash
uv run --locked python scripts/validate_mermaid.py
uv run --locked python scripts/validate_mermaid.py --render --renderer npx \
  --output-dir .tmp/mermaid-render
```

## Release provenance

`build_release.py` first requires fresh publication-profile analysis,
test/coverage, hydration, and render receipts, then records a deterministic
provenance `fingerprint` (SHA-256
over the declared source, manuscript, documentation, producer-script,
dependency-lock, and claim-audit `fingerprint_inputs`) in the manifest. It also
records the pipeline profile and generator version; `--verify` recomputes the
fingerprint and reports changed input paths when the evidence-bearing tree
drifts. Keep the fingerprint boundary in `publication.release_manifest`, not
here. Default unreleased builds omit `generated_at` so a no-op rebuild is
byte-identical. Supply `--timestamp` or `SOURCE_DATE_EPOCH` only for an approved
release. The guarded CLI establishes a local reviewer bundle only; it never
substitutes for clean-clone evidence, release authority, or approval.

The stage receipts in `output/data/pipeline_provenance.json` provide the
upstream/downstream freshness boundary. The separate
`output/data/test_coverage_receipt.json` binds a successful full test/coverage
gate to the source/test/documentation/manuscript/release-metadata/ISC/lock tree
and fresh publication-profile analysis-stage digests; it records matching pre- and post-suite
snapshots and rejects any bound-tree drift;
it is a direct hydration input, not a
graph dependency (otherwise the provisional render needed by the full suite
would form a cycle). Analysis and final hydration write their stage receipts
automatically after successful completion; a smoke profile cannot mint the
analysis receipt. Follow the source-current two-pass
sequence in [the rendering guide](../docs/manuscript/rendering_pipeline.md):
after the final sibling-template stages 03–05, prepare and validate the web
package, then record the external render boundary explicitly. Web preparation
rewrites `output/web/`, so recording it earlier would attest the wrong surface.

```bash
uv run --locked python scripts/record_pipeline_stage.py render \
  --renderer "template-03-05 commit=<SHA> diff_sha256=<SHA256> source_date_epoch=<EPOCH>"
uv run --locked python scripts/validate_pipeline_freshness.py
```

Receipt schema 2 omits wall-clock `recorded_at` values by default. Set
`SOURCE_DATE_EPOCH` or pass `--timestamp` only when an external release event
provides that time; content hashes, rather than time, establish freshness.

The renderer label is recorded as provenance metadata; the project can hash the
render inputs and outputs but cannot content-fingerprint the external template
implementation from this checkout. A clean-checkout probe is a separate release
evidence check and is expected to fail in an intentionally dirty development
tree.

## Smoke tests

`tests/test_scripts_smoke.py` runs the orchestrators as real subprocesses (no
mocks). The ordered analysis chain — `00_preflight.py`, `generate_api_docs.py`,
`01_run_invariants.py`, `02_run_analysis.py` (bounded smoke profile), and
`z_generate_manuscript_variables.py` — is asserted to exit clean, and
`validate_all.py` is exercised through its dry-run profiles. Every thin
orchestrator, those plus `prepare_web_package.py`,
`record_pipeline_stage.py`, `validate_test_coverage.py`, `validate_clean_checkout.py`,
`validate_pipeline_freshness.py`, `validate_rendered_surfaces.py`, and
`validate_web_package.py`, is additionally
asserted to stay logic-free (no numeric imports).

## See also

- [`README.md`](README.md)
- [`../docs/development/style_guide.md`](../docs/development/style_guide.md)
