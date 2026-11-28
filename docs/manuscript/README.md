# `active_fedference` — Manuscript package

For publication compute, first complete the [bounded external stages](../development/staged-analysis.md) and set `WORKSPACE` to that frozen workspace. The import commands
below consume validated completions; they do not start missing reports.

This folder contains the manuscript source for Active Fedference and the
manuscript-driven validation gates that bind every claim to generated outputs.

## What this folder includes

- `config.yaml`: project metadata, rendering options, and experiment settings.
- `*.md`: section source files with `{{TOKEN}}` placeholders.
- `references.bib`: citation source.
- `preamble.md`: LaTeX helpers and package setup.
- `AGENTS.md`: manuscript-specific editing contract.
- `config.yaml.example`: local copy of the metadata template.

The reproducibility material spans `26_reproducibility.md` and
`26_reproducibility_provenance.md` in filename order. The first covers execution
and application integrity; the second retains the producer/invalidation map
and recovery checks. Each produces a separate complete slide deck within the
renderer resource limits. Their figures, captions, tokens, and body reading
order remain part of the combined manuscript.

## Build sequence

1. Run the locked-core numerical invariants gate from the project root:

```bash
uv run python scripts/01_run_invariants.py
```

2. Run analyses:

```bash
uv run --locked python scripts/analysis_stages.py import \
  --workspace "$WORKSPACE" --project-root "$PWD" --stage figures
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
uv run python scripts/record_pipeline_stage.py render \
  --template-root "$TEMPLATE_REPO"
uv run python scripts/validate_pipeline_freshness.py
~~~

The first template pass consumes only provisional hydration. The second consumes
the full-suite receipt-backed hydration. Both explicitly skip automatic
hydration, and web preparation precedes the final receipt because it writes
`output/web/`. The canonical explanation, including release-bundle steps,
is in [`../docs/manuscript/rendering_pipeline.md`](guides/rendering_pipeline.md).

The HTML manuscript is the canonical accessibility-enhanced surface. The
source-current combined manuscript PDF is generated with tagged structure and
must pass the repository's `Tagged: yes`, qpdf `/Lang`, language, and
`StructTreeRoot` gates; Poppler's optional `pdfinfo` language line is not the
only accepted language evidence.
the separate slide PDFs do not inherit that status. Tagged structure is not a
PDF/UA-conformance claim; see
[`../docs/manuscript/accessibility.md`](guides/accessibility.md).

Cover image is configured in `config.yaml` via `paper.cover.image` and currently
points to `cover_image.png` in this directory.

## Structure snapshot

```text
docs/manuscript/
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

## Opening conceptual pages

`00a_graphical_model.md`, `00b_graphical_robustness.md` and `00c_graphical_execution.md` own the new captions and ordered diagram pages. `paper.frontmatter` declares their position immediately after the unchanged overview cover and before the abstract. This is source intent pending the sibling renderer extension and actual complete-paper PDF/HTML placement checks; a standalone frontmatter proof does not satisfy those gates.

## Reading and reproduction guides

See the [manuscript guide](guides/README.md) for explanations of the source, publication pipeline, and evidence.

## Source files

- [`00_abstract.md`](00_abstract.md)
- [`00a_graphical_model.md`](00a_graphical_model.md)
- [`00b_graphical_robustness.md`](00b_graphical_robustness.md)
- [`00c_graphical_execution.md`](00c_graphical_execution.md)
- [`01_introduction.md`](01_introduction.md)
- [`02_gap.md`](02_gap.md)
- [`03_contributions.md`](03_contributions.md)
- [`04_methods_overview.md`](04_methods_overview.md)
- [`05_methods_generalized_bayes.md`](05_methods_generalized_bayes.md)
- [`06_methods_divergences_and_losses.md`](06_methods_divergences_and_losses.md)
- [`07_methods_aggregation.md`](07_methods_aggregation.md)
- [`08_methods_belief_sharing.md`](08_methods_belief_sharing.md)
- [`09_methods_generative_model.md`](09_methods_generative_model.md)
- [`10_methods_learning.md`](10_methods_learning.md)
- [`11_methods_contamination.md`](11_methods_contamination.md)
- [`12_methods_experimental_design.md`](12_methods_experimental_design.md)
- [`13_methods_statistics.md`](13_methods_statistics.md)
- [`14_formalism.md`](14_formalism.md)
- [`15_results_recovery.md`](15_results_recovery.md)
- [`16_results_belief_sharing.md`](16_results_belief_sharing.md)
- [`17_results_language.md`](17_results_language.md)
- [`18_results_emergence.md`](18_results_emergence.md)
- [`19_results_robustness.md`](19_results_robustness.md)
- [`20_results_baseline.md`](20_results_baseline.md)
- [`21_discussion_findings.md`](21_discussion_findings.md)
- [`22_discussion_related_work.md`](22_discussion_related_work.md)
- [`23_discussion_limitations.md`](23_discussion_limitations.md)
- [`24_discussion_future.md`](24_discussion_future.md)
- [`25_conclusion.md`](25_conclusion.md)
- [`26_reproducibility.md`](26_reproducibility.md)
- [`26_reproducibility_provenance.md`](26_reproducibility_provenance.md)
- [`27_supplement_aggregation_objective.md`](27_supplement_aggregation_objective.md)
- [`28_supplement_extended_methods.md`](28_supplement_extended_methods.md)
- [`29_supplement_federation_protocol.md`](29_supplement_federation_protocol.md)
- [`30_supplement_notation.md`](30_supplement_notation.md)
- [`99_references.md`](99_references.md)
- [`AGENTS.md`](AGENTS.md)
- [`S05_results_moving_world.md`](S05_results_moving_world.md)
- [`S06_supp_moving_world.md`](S06_supp_moving_world.md)
- [`S09_results_hierarchical_pomdp.md`](S09_results_hierarchical_pomdp.md)
- [`S10_supp_hierarchical_pomdp.md`](S10_supp_hierarchical_pomdp.md)
- [`S11_results_3level_pomdp.md`](S11_results_3level_pomdp.md)
- [`S12_supp_3level_pomdp.md`](S12_supp_3level_pomdp.md)
- [`S13_results_sensitivity.md`](S13_results_sensitivity.md)
- [`S14_supp_sensitivity.md`](S14_supp_sensitivity.md)
- [`S15_results_parameter_recovery.md`](S15_results_parameter_recovery.md)
- [`S16_results_hierarchical_bmr.md`](S16_results_hierarchical_bmr.md)
- [`S17_results_heuristic_characterization.md`](S17_results_heuristic_characterization.md)
- [`SYNTAX.md`](SYNTAX.md)
- [`preamble.md`](preamble.md)
- [`tokens-and-labels.md`](guides/tokens-and-labels.md)
