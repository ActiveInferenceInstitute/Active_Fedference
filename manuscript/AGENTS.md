---
title: "Manuscript directory: active_fedference"
type: "manuscript_guide"
version: "2.0"
---

# Manuscript (`manuscript/`)

Manuscript-specific editing rules for **Active Fedference** (robust federated
active inference: FedGVI applied to Friston et al. 2024 belief-sharing). Covers
file roles, the `{{VARIABLE}}` token protocol, the figure protocol, and the
numbered-formalism / cross-reference conventions. The acceptance contract is
[`../ISA.md`](../ISA.md); the syntax reference is [`SYNTAX.md`](SYNTAX.md).

## File Inventory (modular section set, render order = filename order)

| File | Role |
|---|---|
| `00_abstract.md` | Abstract; central identity plus evidence-scoped study and robustness headlines (result tokens) |
| `01_introduction.md` | Fragility of naive belief-sharing; FedGVI bridge; three-axis framing; contributions |
| `02_gap.md` / `03_contributions.md` | Scope, three-axis honesty, contribution map |
| `04_methods_overview.md` ... `13_methods_statistics.md` | Methods: generalized Bayes, divergences/losses, aggregation, belief sharing, POMDP world, learning, contamination, design, statistics |
| `14_formalism.md` | Consolidated recovery-limit table and EFE/tempered formalism |
| `15_results_recovery.md` ... `20_results_baseline.md` | Recovery checks, studies, robustness sweep, classification baseline |
| `21_discussion_findings.md` ... `25_conclusion.md` | Findings, related work, limitations, future work, conclusion |
| `26_reproducibility.md` | Determinism, environment fingerprint, reader-surface accessibility boundary, test/coverage evidence, artifact inventory |
| `27_supplement_*.md` ... `30_supplement_*.md`, `S*.md` | Supplements and extension studies; `30` is the authoritative notation contract |
| `99_references.md` | Bibliography pointer (`references.bib`, pandoc `--natbib`) |
| `config.yaml` / `config.yaml.example` | Paper metadata + `experiment:` block (mirrors `src/experiment_config.py`) |
| `preamble.md` | LaTeX injections: amsthm `\newtheorem` envs, caption styling, fonts |
| `references.bib` | BibTeX (incl. `friston2024federated`, `mildner2025fedgvi`) |
| `SYNTAX.md` | Citation / equation / figure / table / theorem label registry |
| `README.md` / `AGENTS.md` | Human quick-reference / this agent guide |

## {{VARIABLE}} Protocol — no hardcoded numbers

Every numeric value in prose **must** be a `{{TOKEN}}` resolved by
`src/manuscript_variables.py::generate_variables()`. Pipeline:

1. `scripts/z_generate_manuscript_variables.py` calls `generate_variables()`.
2. It reads the analysis JSON reports and config. Final non-draft hydration
   derives `ISC_TOTAL`/`ISC_PASSED` from `ISA.md` and loads `TEST_COUNT`,
   `COVERAGE_PERCENT`, and core environment tokens from the fresh successful
   `output/data/test_coverage_receipt.json`; it never re-collects tests while
   rendering. The receipt binds source, tests, final manuscript inputs,
   source-owned documentation and release metadata, `ISA.md`, `uv.lock`, and
   fresh analysis digests; it rejects pre/post-suite boundary drift. No
   signpost is hardcoded.
3. The mapping is written to `output/data/manuscript_variables.json`.
4. `infrastructure.rendering.manuscript_injection.write_resolved_manuscript_tree()`
   copies each `manuscript/*.md` → `output/manuscript/*.md`, substituting tokens.
5. The renderer consumes the substituted copies.

Token groups: provenance (`ISC_*`, `TEST_COUNT`, `COVERAGE_PERCENT`, versions,
`CONFIG_HASH`); config (`EXPERIMENT_SEED`, per-study parameters); results
(`BELIEF_SHARING_*`, `LANGUAGE_*`, `EMERGENCE_*`, `SWEEP_*`) with bootstrap CIs,
raw + BH-adjusted p-values, and standardized effect sizes; recovery residuals
(`RECOVERY_*`). **Detect unresolved tokens before rendering:**
`if rg -n '\{\{[A-Z][A-Z0-9_]*\}\}' output/manuscript/; then echo UNRESOLVED; exit 1; else echo OK; fi`.

## Figure Protocol

Figures are all referenced via pandoc-crossref `[@fig:label]` (never raw
`\ref{}` or hardcoded numbers), generated under `src/figures/` and wired into
`src/analysis/workflow.py` (run via `scripts/02_run_analysis.py`):

The complete figure-label registry lives in
[`SYNTAX.md`](SYNTAX.md) — that table is the source of truth; do not duplicate
it here. Representative rows:

| Label | PNG (`output/figures/`) | Generator (`src/figures/`) |
|---|---|---|
| `{#fig:belief-heatmap}` | `belief_heatmap.png` | `generate_belief_heatmap()` |
| `{#fig:robustness-sweep}` | `robustness_sweep.png` | `generate_robustness_sweep()` |
| `{#fig:bnn-robustness}` | `bnn_robustness.png` | `generate_bnn_robustness()` |
| `{#fig:generative-model-schema}` | `generative_model_schema.png` | `generate_generative_model_schema()` |
| `{#fig:message-passing}` | `message_passing.png` | `generate_message_passing()` |
| `{#fig:pomdp-loop}` | `pomdp_loop.png` | `generate_pomdp_loop()` |

**To add a figure:** add a thin generator to `src/figures/`, wire it into
`src/analysis/workflow.py`, embed a Markdown figure with a real
`../output/figures/<name>.png` path in the relevant results section, reference it
`[@fig:my-label]`, and add a
generator test under `tests/figures/`.

**Figure/caption review:** before final hydration, inspect every changed PNG at
native scale and representative final PDF/HTML/slide pages. The caption must
state the x-axis and y-axis (or panel/row equivalents), source relation,
estimand, units, uncertainty disposition, replication unit, and no-claim
boundary. Do not call a finite-grid span a confidence interval, do not let a
selected display method become an inferential winner, and do not transfer the
client FedGVI guarantee to either server rule. Use the repository-wide Mermaid
probe separately for diagrams embedded in README/docs.

## Numbered-formalism convention

- Display equations: `$$ … $$ {#eq:my-eq}`, referenced `[@eq:my-eq]`.
- Theorems/Definitions: amsthm environments (`\begin{theorem}…\end{theorem}`,
  `\begin{definition}…`) defined in `preamble.md`; every environment has a typed
  `\label{thm:...}` / `\label{lem:...}` / `\label{prop:...}` /
  `\label{cor:...}` / `\label{def:...}` and prose uses automatic `\ref`, never a
  hard-coded theorem counter. The three recovery limits in
  `14_formalism.md` MUST state exactly the identities pinned by ISC-2/5/10
  (`03_contributions.md` is the contributions section, not the formalism).
- Subsections: `## Title {#sec:my-sec}`, referenced `[@sec:my-sec]`.

## Section Modification Protocol

1. Update math/experimental description in the relevant `04_methods_*.md` /
   `14_formalism.md` / results file; keep claims numbered and the robustness-axis
   honesty intact.
2. Add/extend tests under the `tests/` mirror (`tests/fedference/…`, `tests/figures/…`).
3. Run the locked-core numerical invariants gate: `uv run python scripts/01_run_invariants.py`.
4. Regenerate analysis: `uv run python scripts/02_run_analysis.py`.
5. Provisionally hydrate for the pre-test renderer pass:
   `uv run python scripts/z_generate_manuscript_variables.py --provisional-validation`.
6. Render the provisional inputs from the template repo with explicit hydration
   skipped:
   ~~~bash
   TEMPLATE_REPO=/path/to/template
   cd "$TEMPLATE_REPO"
   uv run python scripts/pipeline/stage_03_render.py \
     --project working/active_fedference --skip-manuscript-hydration
   uv run python scripts/pipeline/stage_04_validate.py --project working/active_fedference
   uv run python scripts/pipeline/stage_05_copy.py --project working/active_fedference
   ~~~
7. Run `uv run --extra dev python scripts/validate_test_coverage.py`, then
   rerun final hydration without the provisional flag.
8. Verify resolved with the uppercase-token grammar shown above; a literal
   doubled-brace search is invalid because BibTeX uses doubled braces for case
   protection.
9. Render the receipt-backed final inputs from the template repo, again with
   explicit hydration skipped:
   ~~~bash
   TEMPLATE_REPO=/path/to/template
   cd "$TEMPLATE_REPO"
   uv run python scripts/pipeline/stage_03_render.py \
     --project working/active_fedference --skip-manuscript-hydration
   uv run python scripts/pipeline/stage_04_validate.py --project working/active_fedference
   uv run python scripts/pipeline/stage_05_copy.py --project working/active_fedference
   ~~~
10. Return to the project root; prepare and validate the web package before
    recording the final render receipt, because preparation writes
    `output/web/`. Use the exact source epoch, template label, and
    release-bundle tail in
    [`../docs/manuscript/rendering_pipeline.md`](../docs/manuscript/rendering_pipeline.md).

## Future-phase evidence protocol

Open extensions are governed by the scholarship-indexed phase plan in
[`../docs/todo/scholarship-and-phase-plan.md`](../docs/todo/scholarship-and-phase-plan.md).
Before adding a new result, record the source assumptions, primary estimand,
independent unit, falsifier, required artifact, and no-claim boundary. A larger
model, a network transport, or a new dataset does not inherit the categorical
recovery identity or the client-side FedGVI guarantee automatically.

The optional `experiment.bnn_torch` block in `config.yaml` is an explicit
execution profile for the PyTorch complement (`analysis.workflow` reads it); it
does not silently alter the frozen categorical `ExperimentConfig`. Test
fixtures may use a smaller real profile to keep repeated pipeline tests within
their budget, while the publication configuration remains the declared full
profile.

## RASP Conventions

1. No boilerplate "In summary"/"In conclusion" closers unless genuinely needed.
2. Every number in the results sections must trace to a `{{TOKEN}}` written by
   `scripts/02_run_analysis.py` / `generate_variables()` — never hardcode.
3. The sharp aggregation reweighting is a **heuristic**; never let a figure,
   statistic, or sentence grant it the β/rcce guarantee or the variational
   rule's objective-backed weight control. Carry the three-axis rule: per-client FedGVI is rigorous,
   `robust_aggregate` is sharp but heuristic, and `variational_aggregate` is
   rigorous but conservative.

## See also

- [`README.md`](README.md) · [`SYNTAX.md`](SYNTAX.md) — label registry
- [`../docs/manuscript/README.md`](../docs/manuscript/README.md) — rendering & tokens hub
- [`../docs/manuscript/accessibility.md`](../docs/manuscript/accessibility.md) — HTML/PDF accessibility and no-claim contract
- [`../ISA.md`](../ISA.md) — acceptance criteria (ISC)
- [`../src/manuscript_variables.py`](../src/manuscript_variables.py) — token logic
- [`../scripts/z_generate_manuscript_variables.py`](../scripts/z_generate_manuscript_variables.py) — thin orchestrator
