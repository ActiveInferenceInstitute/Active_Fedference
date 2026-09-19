# Reproducibility: execution record and application integrity {#sec:reproducibility}

This section is a machine-verifiable reproducibility certificate. Every value
below is computed by the analysis pipeline and injected at render time,
establishing a chain of custody from configuration through code to publication.

The discipline is the one that gates this project's CI: every prose number is a
generated token, every token is emitted by one generator function, and any drift
between narrative and computed result fails the build before a green PDF exists
[@peng2011reproducible].

## Determinism contract for seeded scientific results {#sec:repro-determinism}

Reproducing every reported number requires only two recorded inputs: the global
seed pinned here and the software environment fingerprinted in the next
subsection. The determinism contract fixes the first — it states exactly what is
held constant, what is asserted to machine tolerance, and what is deliberately
not claimed byte-identical.

**Seed.** The global seed is 0, threaded through every
`np.random.default_rng(seed)`; the global `np.random` state is never used.

**Assertions.** Recovery identities use exact or machine-tolerance assertions;
seeded study reports are regression-tested for repeatability under the recorded
software environment. Rendered PDF/HTML/slide containers are validated as fresh
publication products but are not claimed byte-identical across toolchain
versions.

**Execution.** No mocks are used: every test is a genuine computation on small
categorical distributions or a seeded simulation under this repository's
explicit no-mocks policy.

## Application integrity and solver-health receipts {#sec:repro-application-integrity}

The labeled own-data path produces a different evidentiary object from a
research report. It validates one-dimensional categorical inputs, hashes the
canonical semantic request, delegates once to the shared aggregation domain,
and records numerical health without adding a decision or interpretation.

The application flow in [@fig:application-integrity-flow] reads from top to bottom. Panel A follows
validation and canonical hashing; Panel B places the exact two-by-two
`solver_status` key below the result classifier; Panel C separates the upper
atomic-write row from the lower verification lane. Invalid inputs or unsafe
destinations fail before directory creation. Non-nominal numerical results
retain their artifacts and return a distinct status rather than being silently
promoted or discarded.

An application receipt can verify artifact integrity and, when requested and
available, source equivalence. Requiring nominal solver health adds a numerical
gate. None of those levels establishes calibration, domain suitability,
scientific validity, decision correctness, or acceptance.

![The labeled application path separates fail-closed request and destination checks from retained non-nominal results. Source relation: source-owned labeled-aggregation, solver-health, artifact, and receipt contracts; status: deterministic implementation map, not an experiment. Panel A follows labeled JSON or Python input through strict schema and one-dimensional shape validation, canonical semantic hashing, a destination-and-required-provenance safety gate, and one delegation to the shared aggregation result. Its two dashed rejection branches distinguish an invalid request from an unsafe destination or unavailable required provenance; both exit 2 before directory creation. Panel B classifies convergence and fallback state, then gives the exact two-by-two `solver_status` key: `nominal`, `converged_with_fallback`, `not_converged`, and `not_converged_with_fallback`. Panel C carries the canonical request and rich result into `request.json` and `result.json`, binds both into `receipt.json`, and enters verification. Solid arrows encode data dependency; the dotted sequence records atomic write order; dashed paths mark rejection or a retained warning. The x-axis is left-to-right execution within rows; rows order panels and verification levels. The estimand is validation, artifact, and solver state; the unit is one categorical software state per invocation. No sampling uncertainty or independent replication unit applies. Exit 1 retains non-nominal artifacts, and a mismatch reports its exact finding. Receipt integrity, source equivalence, and nominal health do not establish calibration, domain suitability, scientific validity, downstream decision correctness, or acceptance.](../output/figures/application_integrity_flow.png){#fig:application-integrity-flow data-slide-manifest="../output/figures/application_integrity_flow.slides.json" width=95%}

## Environment fingerprint for the reported run {#sec:repro-environment}

The second reproduction input is the exact toolchain. Every field below is
captured by the successful full test-and-coverage receipt before final
variable generation rather than transcribed by hand. The receipt is bound to
the source, tests, manuscript, source-owned documentation, release metadata,
ISC tree, dependency lock, and fresh analysis receipt.

It rejects any
pre/post-suite drift in that boundary, so a reader matching this environment
and the seed above can reproduce the seeded results; the config hash lets them
confirm they are running the configuration from which this manuscript was
rendered.

| Field | Value |
|---|---|
| Python | 3.13.11 |
| NumPy | 2.4.2 |
| SciPy | 1.18.0 |
| PyTorch (MLP complement) | not installed |
| Platform | Darwin arm64 |
| Config hash (SHA-256, first 16) | e707fcb26b557c57 |
| Reproducible build epoch (UTC) | 2026-09-19T00:00:00Z |

: Software and configuration fingerprint for the hydrated manuscript. The build epoch is derived from `SOURCE_DATE_EPOCH`; an unreleased build records an explicit omitted sentinel rather than wall-clock time. {#tbl:repro_env}

The exact environment used for the reported run is recorded in [@tbl:repro_env].

## Reader-surface accessibility boundary {#sec:repro-accessibility}

The validated HTML manuscript is the canonical accessibility-enhanced reading
surface. Its source gate requires a page language and title, a skip link and
main landmark, non-empty image alternatives, figure captions, labelled
full-size links, unique identifiers, resolved references, and present local
assets on every generated page.

These deterministic checks are not a claim of
WCAG conformance: alternative-text quality, contrast, keyboard behavior,
reading order, reflow, mathematics, and assistive-technology behavior still
require manual review.

The combined manuscript PDF is generated through the source-controlled
LuaLaTeX/tagpdf path and is released only when `pdfinfo` reports `Tagged: yes`,
qpdf exposes a non-empty `/Lang` and `StructTreeRoot`, and the source-bound
language check passes.

Some Poppler builds omit the language line from
`pdfinfo` even when `/Lang` is present. The
separate slide PDFs are checked structurally, textually, through renderer logs
during the producing run, and by raster inspection, but do not inherit the
manuscript tagging status. Renderer logs are retained with the external
verification evidence rather than the public source tree because they contain
environment-local timestamps and paths.

Tagged structure is not PDF/UA
conformance: a PDF/UA claim
requires a dedicated conformance report plus screen-reader and reading-order
review; `qpdf` structure checks and successful text extraction alone are
insufficient.

## Test and coverage evidence for the claim surface {#sec:repro-tests}

**Acceptance criteria.** 268 total, 266 passing.

**Project test suite.** 2860 collected cases; the bound successful
receipt records zero failed cases. The project no-mocks policy remains a
separately executable source contract.

**Combined line and branch coverage on `src/`.** 90.58%,
achieved by the bound full gate with branch measurement enabled. The shared
coverage threshold is $\ge 90\%$ on this combined measure.

To regenerate this evidence from a clean checkout, run the project suite under
the pinned development environment. Local execution and hosted CI are separate
results; the hosted checks must pass for the exact reviewed commit before
public integration. Both use the following coverage gate:

```bash
uv run --locked --extra dev pytest tests/ \
  --cov=src --cov-fail-under=90
```

For a release-facing hydration, use the receipt-producing wrapper after any
required provisional pre-test render, then rerun hydration without its
provisional flag:

```bash
uv run --locked --extra dev python scripts/validate_test_coverage.py
uv run --locked python scripts/\
z_generate_manuscript_variables.py
```

## Artifact inventory for figures, data, and reports {#sec:repro-artifacts}

| Category | Count |
|---|---|
| Figures | 1226 |
| Data files | 5 |
| Reports | 36 |
| Total | 1267 |

: Top-level generated files in `output/figures`, `output/data`, and
`output/reports` at token-hydration time. The generated release manifest is the
source of truth for the larger recursive publication bundle. Artifacts are
regenerable reviewer snapshots and must not be hand-edited.
{#tbl:repro_artifacts}

The top-level artifact counts in [@tbl:repro_artifacts] complement the recursive,
checksum-bearing release manifest.
