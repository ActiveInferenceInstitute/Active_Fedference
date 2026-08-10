# Reproducibility: execution record and recovery checks {#sec:reproducibility}

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

- Global seed: {{EXPERIMENT_SEED}}, threaded through every
  `np.random.default_rng(seed)`; the global `np.random` state is never used.
- Recovery identities use exact or machine-tolerance assertions; seeded study
  reports are regression-tested for repeatability under the recorded software
  environment. Rendered PDF/HTML/slide containers are validated as fresh
  publication products but are not claimed byte-identical across toolchain
  versions.
- No mocks anywhere: every test is a genuine computation on small categorical
  distributions or a seeded simulation under this repository's explicit
  no-mocks policy.

## Environment fingerprint for the reported run {#sec:repro-environment}

The second reproduction input is the exact toolchain. Every field below is
captured by the successful full test-and-coverage receipt before final
variable generation rather than transcribed by hand. The receipt is bound to
the source, tests, manuscript, source-owned documentation, release metadata,
ISC tree, dependency lock, and fresh analysis receipt. It rejects any
pre/post-suite drift in that boundary, so a reader matching this environment
and the seed above can reproduce the seeded results; the config hash lets them
confirm they are running the configuration from which this manuscript was
rendered.

| Field | Value |
|---|---|
| Python | {{PYTHON_VERSION}} |
| NumPy | {{NUMPY_VERSION}} |
| SciPy | {{SCIPY_VERSION}} |
| PyTorch (MLP complement) | {{PYTORCH_VERSION}} |
| Platform | {{PLATFORM}} |
| Config hash (SHA-256, first 16) | {{CONFIG_HASH}} |
| Reproducible build epoch (UTC) | {{GENERATION_TIMESTAMP}} |

: Software and configuration fingerprint for the hydrated manuscript. The build epoch is derived from `SOURCE_DATE_EPOCH`; an unreleased build records an explicit omitted sentinel rather than wall-clock time. {#tbl:repro_env}

The exact environment used for the reported run is recorded in [@tbl:repro_env].

## Reader-surface accessibility boundary {#sec:repro-accessibility}

The validated HTML manuscript is the canonical accessibility-enhanced reading
surface. Its source gate requires a page language and title, a skip link and
main landmark, non-empty image alternatives, figure captions, labelled
full-size links, unique identifiers, resolved references, and present local
assets on every generated page. These deterministic checks are not a claim of
WCAG conformance: alternative-text quality, contrast, keyboard behavior,
reading order, reflow, mathematics, and assistive-technology behavior still
require manual review.

The combined manuscript and slide PDFs are checked structurally, textually,
through retained renderer logs, and by raster inspection. They are not claimed
to be tagged or PDF/UA-conformant. A future accessible-PDF claim requires a
tagged producer, a dedicated conformance report, and screen-reader and
reading-order review; `qpdf` structure checks and successful text extraction
alone are insufficient.

## Test and coverage evidence for the claim surface {#sec:repro-tests}

- Acceptance criteria: {{ISC_TOTAL}} total, {{ISC_PASSED}} passing.
- Project test suite: {{TEST_COUNT}} collected cases; the bound successful
  receipt records zero failed cases. The project no-mocks policy remains a
  separately executable source contract.
- Line coverage on `src/`: {{COVERAGE_PERCENT}}% (achieved by the bound full
  gate; $\ge 90\%$ line coverage is enforced in CI, while branch coverage is
  tracked separately in CI).

To regenerate this evidence from a clean checkout, run the project suite under
the pinned development environment; the same invocation is the CI gate, so a
passing local run and a green build are the same event:

```bash
uv run --extra dev pytest tests/ \
  --cov=src --cov-fail-under=90
```

For a release-facing hydration, use the receipt-producing wrapper after any
required provisional pre-test render, then rerun hydration without its
provisional flag:

```bash
uv run --extra dev python scripts/validate_test_coverage.py
uv run python scripts/z_generate_manuscript_variables.py
```

## Artifact inventory for figures, data, and reports {#sec:repro-artifacts}

| Category | Count |
|---|---|
| Figures | {{ARTIFACT_FIGURES}} |
| Data files | {{ARTIFACT_DATA_FILES}} |
| Reports | {{ARTIFACT_REPORTS}} |
| Total | {{ARTIFACT_TOTAL}} |

: Top-level generated files in `output/figures`, `output/data`, and
`output/reports` at token-hydration time. The generated release manifest is the
source of truth for the larger recursive publication bundle. Artifacts are
regenerable reviewer snapshots and must not be hand-edited.
{#tbl:repro_artifacts}

The top-level artifact counts in [@tbl:repro_artifacts] complement the recursive,
checksum-bearing release manifest.

## Recovery-limit certificate for the client and project-pool corners {#sec:repro-recovery}

The recovery identities are reproducibility checks: the client machinery must
return to standard Bayes at its KL/NLL loss limits, and the server heuristic
must return to the project's log-linear pool at zero robustness
([@eq:robust-identity], [@eq:standard-bayes]). Under the explicit
shared-support, posterior-log-potential, and fixed-weight assumptions of
[@sec:method-aggregation], that pool specializes Eq. 7's message-combination
term; it does not reproduce the complete source protocol. These deviations are
computed on every build:

- `robust_aggregate(robustness=0)` versus `log_linear_pool` ([@eq:log-linear-pool]):
  {{RECOVERY_AGGREGATE_MAXDIFF}}
- `generalized_posterior(KLD, NLL)` versus closed-form Bayes:
  {{RECOVERY_POSTERIOR_MAXDIFF}}
- Rényi divergence versus KL as $\alpha\to 1$: {{RECOVERY_RENYI_MAXDIFF}}
- $\beta$-loss versus NLL as $\beta\to 0$: {{RECOVERY_BETA_MAXDIFF}}
- rcce versus NLL as $q_{\rm loss}\to 0$: {{RECOVERY_RCCE_MAXDIFF}}

Any drift in these limits beyond machine precision would mean the robust
generalization no longer
contains its standard-Bayes client limit and project-local log-linear-pool
server limit ([@sec:results-recovery]) — and would fail the core test suite
before this certificate could render. The certificate covers the recovery
identity and the client-side result under the cited source theorem's matching
assumptions only;
the server-side `robust_aggregate` heuristic is certified here for its recovery
limit alone, not for any bounded-influence property ([@sec:limitations]).

All code is authored by {{CONFIG_FIRST_AUTHOR}} and licensed under the MIT license.
This is project version {{CONFIG_VERSION}}.
