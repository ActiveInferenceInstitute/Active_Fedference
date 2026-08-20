# Rendering pipeline: manuscript → PDF

Four-phase flow for Active Fedference. Run phases 1-2 from this repository root;
run phases 3-4 from the sibling template repository. A source-current
publication run uses two template passes: provisional hydration supplies the
first renderer input for the full suite, then receipt-backed final hydration
supplies the receipt-bearing pass.

Pin one build epoch before phase 1 and retain it through phase 4:

```bash
AF_REPO=/path/to/active_fedference
TEMPLATE_REPO=/path/to/template
export SOURCE_DATE_EPOCH="$(git -C "$AF_REPO" log -1 --format=%ct)"
```

This epoch is reproducibility metadata derived from the reviewed Active
Fedference source commit, not evidence of release approval or wall-clock
completion. Record the sibling renderer commit and diff separately in the
renderer label.

## Prerequisite: Mermaid / Chrome

Combined PDF rendering rasterises ```mermaid``` blocks via `mmdc`, which needs
`chrome-headless-shell`:

```bash
npx --yes puppeteer browsers install chrome-headless-shell
```

Symptom: `mmdc failed ... Could not find Chrome`. See
[`../operations/troubleshooting.md`](../operations/troubleshooting.md).

## Phase 1 — Analysis (pipeline stage 4)

**Script:** `scripts/02_run_analysis.py`

```bash
uv run --locked python scripts/02_run_analysis.py
```

**Logic:** `src/analysis/workflow.py::run_analysis_pipeline()`

**Outputs:** JSON reports in `output/reports/` and PNG/PDF figures in
`output/figures/`. Inventory:
[`../core/experiments-and-artifacts.md`](../core/experiments-and-artifacts.md).

## Phase 2 — Manuscript variables

**Script:** `scripts/z_generate_manuscript_variables.py`

```bash
# Provisional output is only for the renderer pass that precedes the full suite.
uv run --locked python scripts/z_generate_manuscript_variables.py --provisional-validation
```

**Logic:** `generate_variables()` (entry point `src/manuscript_variables.py`,
implementation in the `src/manuscript_vars/` package) reads
`output/reports/*.json` and `manuscript/config.yaml`, then
`src/manuscript_vars/render.py::render_manuscript_tree()` substitutes
`{{TOKEN}}` markers and writes resolved copies to `output/manuscript/`
(project-local logic; no `infrastructure.rendering` import is used in this
phase). Small magnitudes are emitted both as plain `.2e` tokens and as
`*_MATH` siblings in LaTeX scientific notation for `$...$` spans — see
[`tokens-and-labels.md`](tokens-and-labels.md).

The `HIER_*` and `NLEVEL3_*` token groups are strict loads from the hierarchical
and three-level reports generated in Phase 1. Variable hydration does not rerun
either study or silently synthesize missing report values.

**Outputs:**

- `output/data/manuscript_variables.json`
- `output/manuscript/*.md` (token-resolved)

All `{{TOKEN}}` placeholders must resolve before PDF render.

Do not run final non-draft hydration immediately after analysis: it requires
the successful full-suite receipt. The complete two-pass sequence below obtains
that receipt only after the provisional renderer pass.

## Source-current two-pass sequence

This is the authoritative sequence for a reviewer snapshot or release
candidate. Run code-quality and invariant gates before it, and keep both
repositories unchanged while it is running.

~~~bash
AF_REPO=/path/to/active_fedference
TEMPLATE_REPO=/path/to/template
export SOURCE_DATE_EPOCH="$(git -C "$AF_REPO" log -1 --format=%ct)"

cd "$AF_REPO"
uv run --locked python scripts/02_run_analysis.py
uv run --locked python scripts/z_generate_manuscript_variables.py --provisional-validation

# First template pass: its hydrated inputs are deliberately provisional.
cd "$TEMPLATE_REPO"
uv run --locked python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
uv run --locked python scripts/pipeline/stage_04_validate.py --project working/active_fedference
uv run --locked python scripts/pipeline/stage_05_copy.py --project working/active_fedference

# The full suite attests the source/manuscript/analysis tree used for final hydration.
cd "$AF_REPO"
uv run --locked --extra dev python scripts/validate_test_coverage.py
uv run --locked python scripts/z_generate_manuscript_variables.py

# Second template pass: this is the final hydrated manuscript surface.
cd "$TEMPLATE_REPO"
uv run --locked python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
uv run --locked python scripts/pipeline/stage_04_validate.py --project working/active_fedference
uv run --locked python scripts/pipeline/stage_05_copy.py --project working/active_fedference

# Prepare and validate the final web tree before declaring the render complete.
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

# An unreleased reviewer bundle has no release timestamp.
unset SOURCE_DATE_EPOCH
uv run --locked python scripts/build_release.py
uv run --locked python scripts/build_release.py --verify
~~~

The bundle CLI rejects a smoke or manually promoted analysis receipt and
requires the publication-profile analysis sidecar together with the final
test/coverage, hydration, and render chain. Its success is local reviewer
snapshot evidence, not external release approval.

Both `stage_03_render.py` calls pass `--skip-manuscript-hydration` because
hydration is explicit in this sequence. Without it, the first pass attempts
non-provisional hydration before the test receipt exists, and the second pass
would rewrite already-final hydrated input. The provisional pass is not itself
recorded as a render receipt.

`stage_05_copy.py` and `prepare_web_package.py` can write generated reader
surfaces; in particular, web preparation mirrors figures and normalizes
cross-references under `output/web/`. The final render receipt therefore comes
only after both have completed and the prepared web/PDF/slide surfaces have
validated. Its hashes then describe the actual release artifact rather than
the pre-package renderer output.

## Phase 3 — PDF, web, and slides render (pipeline stage 7)

**Script:** `scripts/pipeline/stage_03_render.py` from the template repository
root.

```bash
AF_REPO=/path/to/active_fedference
TEMPLATE_REPO=/path/to/template
cd "$TEMPLATE_REPO"
export SOURCE_DATE_EPOCH="$(git -C "$AF_REPO" log -1 --format=%ct)"
uv run --locked python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
```

**Inputs:**

- `output/manuscript/*.md` (substituted)
- `manuscript/config.yaml`, `preamble.md`, `references.bib`
- `output/figures/*.png`

**Outputs:** `output/pdf/active_fedference_combined.pdf` (copied to root
`output/working/active_fedference/pdf/` after the copy stage), per-section web HTML under
`output/web/`, and per-section Beamer slide decks under `output/slides/`.

The HTML manuscript is the canonical accessibility-enhanced reader surface.
The combined manuscript PDF is generated through the source-controlled tagged
producer and must pass the repository's `Tagged: yes`, qpdf `/Lang`, language,
and `StructTreeRoot` gates; Poppler's optional `pdfinfo` language line is not
the only accepted language evidence. Slide outputs are separate Beamer
surfaces. Tagged structure is not PDF/UA conformance. Read
[`accessibility.md`](accessibility.md) before changing the renderer or making
an accessibility claim.

### Slide cross-deck references

Each section deck is a standalone Beamer build, so a raw-LaTeX `\ref{...}`
whose `\label` lives in a different section's deck cannot resolve locally. The
template renderer runs a fail-open pre-pass
(`infrastructure/rendering/_slides_crossref.py` in the template repository)
that parses the combined manuscript's retained aux file
(`output/pdf/_combined_manuscript.aux`) into a label-to-printed-number map and
substitutes those numbers into cross-deck refs, so slide numbers match the
combined PDF exactly. Within-deck refs are left alone and numbered natively by
Beamer. Caveat: the aux map is an artifact of the most recent combined build,
so on the very first render of a project (no aux yet) cross-deck refs stay as
"??" until the next render pass.

### Web theorem rendering

Pandoc's HTML writer silently drops raw-LaTeX theorem-like environments
(theorem, lemma, proposition, corollary, definition) because their
`\newtheorem` definitions live in the LaTeX-only preamble. The template web
renderer rewrites them — web-only — into numbered `.theorem-box` Divs that
share one running counter, so web numbering matches the PDF's shared-counter
convention. A same-line `\label{...}` after the optional name (the standard
amsthm idiom) is consumed and becomes the Div's anchor id. The PDF path never
sees this rewrite; it consumes the original environments against the LaTeX
preamble.

Pre-flight markdown check:

```bash
TEMPLATE_REPO=/path/to/template
cd "$TEMPLATE_REPO"
uv run --locked python -m infrastructure.validation.cli markdown \
  projects/working/active_fedference/manuscript --repo-root .
```

## Phase 4 — Validation and copy (stages 8 and 11)

```bash
AF_REPO=/path/to/active_fedference
TEMPLATE_REPO=/path/to/template
cd "$TEMPLATE_REPO"
export SOURCE_DATE_EPOCH="$(git -C "$AF_REPO" log -1 --format=%ct)"
uv run --locked python scripts/pipeline/stage_04_validate.py --project working/active_fedference
uv run --locked python scripts/pipeline/stage_05_copy.py --project working/active_fedference
```

Keep the same `SOURCE_DATE_EPOCH` across stages 03–05 and both clean-clone
passes. It pins PDF creation metadata and validation-report time to the
reviewed Active Fedference source commit. Record the sibling renderer
commit/diff digest separately in the render label; an uncommitted external
overlay remains an explicit provenance limitation, never a source-bound
release claim.

The phase-3 command assumes that explicit hydration has already produced
`output/manuscript/`; use `--skip-manuscript-hydration` so the renderer does
not silently replace it. Follow the two-pass sequence above rather than
treating a one-off stage-03 call as a source-current release artifact. It
prepares the web package before recording the final render receipt and records
the template commit/diff label only after the final surface exists.

## Full template core pipeline

```bash
TEMPLATE_REPO=/path/to/template
cd "$TEMPLATE_REPO"
uv run --locked python scripts/runner/execute_pipeline.py --project working/active_fedference --core-only
```

The combined command is useful for exploratory template work, but it does not
express the provisional-render → full-suite-receipt → final-render order.
Do not use it as a replacement for the source-current sequence above.

## `manuscript/config.yaml` controls

| Block | Effect |
| --- | --- |
| `paper.*` | Title, authors, DOI metadata for PDF front matter |
| `experiment.*` | Seeds, agent counts, contamination grid, divergence labels |
| `rendering.section_breaks` | Combined-PDF page flow only: `true` (the default) inserts a break between source files; `false` lets fragments flow continuously. Use a source-authored raw-LaTeX `\newpage` for a required hard boundary. |
| `llm.*` | Optional Ollama review/translation stages (skipped if Ollama absent) |

Experiment keys mirror keyword arguments of `fedference.experiments` functions.

## Optional auxiliary scripts

| Script | Role |
| --- | --- |
| `scripts/00_preflight.py` | Chrome/LaTeX environment warnings |
| `scripts/generate_api_docs.py` | Writes `output/docs/api_reference.md` (not consumed by PDF stage) |

## See also

- Manuscript editing: [`../../manuscript/AGENTS.md`](../../manuscript/AGENTS.md)
- Tokens: [`tokens-and-labels.md`](tokens-and-labels.md)
- Verification commands: [`../reference/verification-commands.md`](../reference/verification-commands.md)
