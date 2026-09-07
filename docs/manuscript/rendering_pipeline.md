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
completion. The sibling renderer is locked separately by canonical repository,
exact commit, and required clean state in `manuscript/config.yaml`.

## Pipeline at a glance

The concise order is: source-owned inputs → typed analysis reports and figures
→ provisional hydration → full-suite coverage receipt → final hydration
→ clean, commit-locked Template rendering → surface validation and
freshness receipts → upstream release manifest → downstream Template control
manifests over the fixed release bytes → separately authorized publication.
The complete accessible producer graph, text equivalent, reverse-invalidation
rules, and non-circular manifest boundary are maintained in
[`experiments-and-artifacts.md`](../core/experiments-and-artifacts.md#source-to-render-production-and-invalidation-graph).
Any changed source, configuration, lock, registry, report, receipt, figure, or
renderer byte makes the dependent downstream artifacts stale; rerun from the
earliest changed producer rather than refreshing only the final surface.

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

Configuration string values are hydrated in `output/manuscript/config.yaml`
without editing the source template. YAML serialization preserves replacement
text as values, including quotes, colons, and newlines. Unresolved configuration
tokens reject the transaction before the previous hydrated tree is replaced.
The pinned Template renderer preserves this project-resolved configuration and
rejects any unresolved token that reaches its configuration boundary. Preamble
and BibTeX files retain their existing source-owned renderer contract.

The `HIER_*` and `NLEVEL3_*` token groups are strict loads from the hierarchical
and three-level reports generated in Phase 1. Variable hydration does not rerun
either study or silently synthesize missing report values.

**Outputs:**

- `output/data/manuscript_variables.json`
- `output/manuscript/*.md` (token-resolved)
- `output/manuscript/config.yaml` (configuration string values resolved)

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
uv run --locked python scripts/00_preflight.py --template-root "$TEMPLATE_REPO"
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
cd "$AF_REPO"
uv run --locked python scripts/00_preflight.py --template-root "$TEMPLATE_REPO"
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

uv run --locked python scripts/record_pipeline_stage.py render \
  --template-root "$TEMPLATE_REPO"
uv run --locked --extra dev python scripts/validate_test_coverage.py --verify
uv run --locked python scripts/validate_pipeline_freshness.py

# Build the upstream publication payload before Template writes its final
# artifact/validation control receipts. An unreleased reviewer bundle has no
# release timestamp; env -u leaves the exported render epoch available below.
env -u SOURCE_DATE_EPOCH uv run --locked python scripts/build_release.py
RELEASE_HASH_BEFORE="$(
  shasum -a 256 output/release/README.md output/release/manifest.json \
    output/release/sha256sums.txt | shasum -a 256 | awk '{print $1}'
)"
env -u SOURCE_DATE_EPOCH uv run --locked python scripts/build_release.py
RELEASE_HASH_AFTER="$(
  shasum -a 256 output/release/README.md output/release/manifest.json \
    output/release/sha256sums.txt | shasum -a 256 | awk '{print $1}'
)"
test "$RELEASE_HASH_BEFORE" = "$RELEASE_HASH_AFTER"

# Finalize the downstream control plane against the exact release bytes.
cd "$TEMPLATE_REPO"
uv run --locked python scripts/maintenance/refresh_artifact_manifests.py \
  --project working/active_fedference
uv run --locked python scripts/pipeline/stage_04_validate.py \
  --project working/active_fedference
uv run --locked python scripts/pipeline/stage_05_copy.py \
  --project working/active_fedference

# These gates are read-only at the fixed point. No payload producer runs after
# the downstream controls have been sealed.
cd "$AF_REPO"
uv run --locked python scripts/validate_web_package.py
uv run --locked python scripts/validate_rendered_surfaces.py
uv run --locked --extra dev python scripts/validate_test_coverage.py --verify
uv run --locked python scripts/validate_pipeline_freshness.py
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

Web preparation uses a source-preserving, no-clobber batch transaction. A
concurrent path edit or interrupted rollback fails with retained recovery
evidence instead of overwriting the other writer. The validator rejects HTML
`<base>`, CDATA, and `<noscript>` constructs and applies browser-effective
first-attribute semantics to resource, integrity, and CORS attributes. These
deliberately conservative restrictions keep static resource resolution
auditable; they are not a claim that the validator implements every browser
tree-building state or verifies remote script bytes.

The release manifest is schema-versioned as an upstream
`publication-payload-v1` inventory. It covers the scientific reports, data and
coverage receipts, hydrated manuscript, figures, PDF, slides, web surfaces,
metadata, and declared source fingerprint inputs. Template-owned artifact,
evidence, statistics, validation, rendered-provenance, and snapshot control
reports are produced after that payload and are deliberately outside its hash
set. They remain mandatory and are verified by the final Template validation,
Git-tree, confidentiality, and clean-clone gates. This one-way boundary avoids
the impossible cycle in which an artifact manifest hashes `output/release/`
while the release manifest hashes the artifact manifest that names it.

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
- `output/manuscript/config.yaml` (project-resolved configuration)
- source-owned `manuscript/preamble.md` and `manuscript/references.bib`
- `output/figures/*.png`

**Outputs:** `output/pdf/active_fedference_combined.pdf` (copied to root
`output/working/active_fedference/pdf/` after the copy stage), per-section web
HTML under `output/web/`, and paired per-section Reveal.js HTML and Beamer PDF
slide decks under `output/slides/`.

The validated HTML manuscript is the canonical accessibility-enhanced
manuscript reader, while Reveal.js is the accessibility-enhanced slide-reading
surface. The combined manuscript PDF is generated through the
source-controlled tagged producer and must pass the repository's `Tagged: yes`,
qpdf `/Lang`, language, and `StructTreeRoot` gates; Poppler's optional
`pdfinfo` language line is not the only accepted language evidence. Beamer
slide PDFs are explicitly untagged presentation derivatives. These roles do
not establish WCAG or PDF/UA conformance. Read
[`accessibility.md`](accessibility.md) before changing the renderer or making
an accessibility claim.

### Slide cross-deck references

Each section deck is a standalone Beamer build, so a raw-LaTeX `\ref{...}`
whose `\label` lives in a different section's deck cannot resolve locally.
During one canonical renderer invocation, the Template pipeline clears any
stale combined-manuscript aux file, builds the combined PDF, validates the new
aux label map, and then refreshes every enabled slide derivative in strict
mode. The refresh pass
(`infrastructure/rendering/_slides_crossref.py` in the Template repository)
substitutes the combined PDF's printed numbers into cross-deck references;
within-deck references remain native Beamer references. A missing, unreadable,
or unresolved non-section label fails the canonical refresh and removes the
affected derivative instead of publishing a stale deck. Direct standalone
slide rendering remains a deliberately fail-open authoring convenience: when
no aux map is available, unresolved section references are exposed as visible
`sec:*` labels. The transient aux sidecar remains inside the producer
workspace and is excluded from the public Git tree and release bundle. No
second pipeline invocation is required for canonical cross-deck resolution.

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
reviewed Active Fedference source commit. The source-owned
`rendering.template_renderer` lock names the canonical Template repository,
exact commit, and required clean-tree state. Release-facing rendering fails
closed unless the explicitly supplied Template checkout matches that lock;
the recorded renderer identity is path-free and contains no branch name,
machine-local checkout path, remote URL, or uncommitted-diff label. An
uncommitted renderer overlay is therefore inadmissible rather than a
publishable limitation.

The phase-3 command assumes that explicit hydration has already produced
`output/manuscript/`; use `--skip-manuscript-hydration` so the renderer does
not silently replace it. Follow the two-pass sequence above rather than
treating a one-off stage-03 call as a source-current release artifact. It
prepares the web package before recording the final render receipt and records
the locked, clean Template identity only after the final surface exists.

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
| `render.slides.profile` | `accessible` emits an atomic Reveal.js/Beamer pair and fails on unsafe density; omitted or `archive` preserves the Template default behavior for other consumers. |
| `render.slides.max_prose_words`, `max_table_rows` | At most 80 prose words per frame and eight displayed body rows per table in this project. Complete captions and tables remain in the linked HTML manuscript. |
| `render.slides.min_figure_area_percent` | Figure-led frames reserve at least 70% of usable space for the figure. |
| `render.slides.title_font_pt`, `body_font_pt`, `figure_label_font_pt` | Projected-text floors of 28, 20, and 16 points; the renderer diagnoses density instead of silently shrinking below them. |
| `render.slides.reader_href` | Relative link from each Reveal.js deck to the canonical HTML manuscript. |
| `rendering.section_breaks` | Combined-PDF page flow only: `true` (the default) inserts a break between source files; `false` lets fragments flow continuously. Use a source-authored raw-LaTeX `\newpage` for a required hard boundary. |
| `rendering.template_renderer` | Fail-closed lock to the canonical `docxology/template` repository, exact 40-character commit, and clean-tree state used by preflight and render receipts. |
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
