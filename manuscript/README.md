# `active_fedference` — Manuscript package

This folder contains the manuscript source for Active Fedference and the
manuscript-driven validation gates that bind every claim to generated outputs.

## What this folder includes

- `config.yaml`: project metadata, rendering options, and experiment settings.
- `*.md`: section source files with `{{TOKEN}}` placeholders.
- `references.bib`: citation source.
- `preamble.md`: LaTeX helpers and package setup.
- `AGENTS.md`: manuscript-specific editing contract.
- `config.yaml.example`: local copy of the metadata template.

## Build sequence

1. Run the locked-core numerical invariants gate from the project root:

```bash
uv run python scripts/01_run_invariants.py
```

2. Run analyses:

```bash
uv run python scripts/02_run_analysis.py
```

3. Complete the source-current token, render, and provenance sequence:

~~~bash
AF_REPO=/path/to/active_fedference
TEMPLATE_REPO=/path/to/template
export SOURCE_DATE_EPOCH="$(git -C "$AF_REPO" log -1 --format=%ct)"

cd "$AF_REPO"
uv run python scripts/z_generate_manuscript_variables.py --provisional-validation

cd "$TEMPLATE_REPO"
uv run python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
uv run python scripts/pipeline/stage_04_validate.py --project working/active_fedference
uv run python scripts/pipeline/stage_05_copy.py --project working/active_fedference

cd "$AF_REPO"
uv run --extra dev python scripts/validate_test_coverage.py
uv run python scripts/z_generate_manuscript_variables.py

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
uv run python scripts/validate_pipeline_freshness.py
~~~

The first template pass consumes only provisional hydration. The second consumes
the full-suite receipt-backed hydration. Both explicitly skip automatic
hydration, and web preparation precedes the final receipt because it writes
`output/web/`. The canonical explanation, including release-bundle steps,
is in [`../docs/manuscript/rendering_pipeline.md`](../docs/manuscript/rendering_pipeline.md).

The HTML manuscript is the canonical accessibility-enhanced surface. The
current PDFs are not tagged or PDF/UA-conformant; see
[`../docs/manuscript/accessibility.md`](../docs/manuscript/accessibility.md).

Cover image is configured in `config.yaml` via `paper.cover.image` and currently
points to `cover_image.png` in this directory.

## Structure snapshot

```text
manuscript/
├── 00_abstract.md
├── 01_introduction.md
├── 02_gap.md ... 30_supplement_...
├── S*.md
├── config.yaml
├── config.yaml.example
├── preamble.md
├── references.bib
├── SYNTAX.md
├── README.md
└── cover_image.png
```

## Editing constraints

- Use `{{TOKEN}}` placeholders for all generated numbers.
- Keep every claim traceable in `SYNTAX.md` or validated in tests.
- Prefer references to `docs/..` and `ISA.md` before editing core claims.
- Preserve the accessibility/no-claim contract when editing figures, captions,
  HTML, PDF, or slide prose.
- For figure edits, run `uv run pytest tests/test_caption_completeness.py
  tests/figures/ -q` and inspect the native PNG plus a final rendered page;
  caption metadata, figure registry, and output filename must stay aligned.
- Never hand-edit `../output/manuscript/` or other generated reviewer
  surfaces; regenerate them in producer order after source changes.

## Verification checks

```bash
uv run pytest tests/test_xref_integrity.py \
  tests/test_caption_completeness.py \
  tests/test_token_provenance.py \
  tests/test_manuscript_variables.py -q
```
