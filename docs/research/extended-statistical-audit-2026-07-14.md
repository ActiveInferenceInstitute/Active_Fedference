# Extended statistical and numerical audit

Updated 2026-08-10. This document preserves dated artifact-level audit records
and states the source contract for the pending enlarged rerun. It is not an
audit of the current generated reports, manuscript tokens, or rendered
publication surfaces: the checked-out configuration and review-grid producer
now differ from the 2026-08-01 snapshot. The historical simulations remain
conditional on their declared data-generating mechanisms and do not generalize
beyond them. The v0.1.0 GitHub/Zenodo publication on 2026-08-10 is documented
separately and does not convert this dated audit into a source-current
scientific evidence report.

## Status and source of truth

The historical audit was performed after the then-current analysis and
template-rendering pipeline completed. Its machine-readable evidence was:

- the dated `output/reports/*.json` for study results and statistical metadata;
- the dated `output/data/manuscript_variables.json` for hydrated manuscript
  values;
- the old validation and manifest records for the earlier publication checks;
  and
- the source tests and generated figure registry that existed at that time.

For the checked-out source, [`manuscript/config.yaml`](../../manuscript/config.yaml),
[`src/experiment_config.py`](../../src/experiment_config.py), the report
schemas, and the executable tests are the source contract. A source-current
claim requires a fresh analysis receipt, test-and-coverage receipt, final
hydration, render receipt, and surface validation. Until those producers run in
order, the tracked output is historical evidence only.

## Current source contract — rerun pending

The publication configuration now requests 480 primary seeds, 128
structural-extension seeds, 960 matched robustness-sweep trials, 64 seeds × 24
nested trials for conditional-world, gallery, and onset surfaces, and 64 seeds
× 200 observations for the categorical BNN complement. These are configured
budgets, not observations or claims that a new run has completed.

The bounded `robustness_review_grid` is configured for 160 deterministic seeds
× 24 nested trials at rates `0, .2, .4, .5, .6, .7, .8, .9`. On a fresh run it
must emit schema `1.1`, retain every configured non-KLD method in every
directional attack × rate cell, calculate each method's own paired
robust-minus-KLD contrast and seed-bootstrap interval, and record a precision
receipt. Its configured maximum MCSE target is 0.01, checked across all signed
method × rate × directional-mechanism cells; the producer fails instead of
writing a report if the target is not met. The rate-panel inference is therefore
selection-free, with BH families owned by attack mechanism × method across the
declared rates. The two-sided paired test, directional planning calculation,
rank-biserial-derived display transform, and finite saturation convention retain
their distinct roles.

The contamination gallery and robustness onset remain deliberately
pooled-best/descriptive surfaces. Their finite patterns, intervals, and onset
summaries are exploratory displays, not post-selection inference. The review
grid is the all-method selection-free inferential surface. Its executed report
also records the configured FDR level, planning alpha, and planning alternative;
those values are consumed by the producer rather than being decorative
metadata. The review-grid configuration does not promote its finite attack cells to independent
populations, prove a universal robustness result, or close the research phase.

## Historical publication-profile evidence — 2026-08-01

The following was recorded for the earlier source/API/formalism snapshot. It
does not validate the currently configured budgets or revised review-grid
producer. The primary robustness report then had schema version `2.0`, 480 matched
trials, seven agents, two contaminated agents, and verdict rate `0.8`. The
headline fields are reproducibly `RKL|AR|beta|rcce` for the tied rank-biserial
set, `RKL` for the stable method-order display choice, `AR` for the largest
mean difference, and `beta` for the best method at the worst rate. The report
also retains the matched mean differences and percentile intervals; the
million-scale values appear only as finite saturation sentinels for the
secondary d-equivalent display transform.

The historical `robustness_review_grid` report is schema version `1.0` and records 64
deterministic seeds, 12 nested trials per seed/cell, the rates
`0, .2, .4, .5, .6, .7, .8, .9`, and the union of clean, confident-wrong,
permutation, Byzantine, drift, label-noise, and uniform mechanisms. Its
payload predates the source's all-method figure and precision-receipt contract.
The completed hydration, focused 55-test suite, 1,393-test/90.08%-coverage
suite, template stages, visual/browser review, package validation,
distribution comparison, and release verification named in the old receipt are
therefore historical local facts. They cannot be carried forward as fresh
validation. The old HTML accessibility-enhancement and untagged-PDF status are
also historical build facts; clean-clone replication, DOI publication, remote
push, and author signoff remained separate gates.

## Historical replication structure and estimands

The following table describes the 2026-08-01 snapshot, not the checked-out
configuration. Its headline robustness sweep had **480 matched trials** with seven agents, two
contaminated agents, a fixed true state, and a fixed attack target. The primary
independent Monte Carlo unit for that cross-study inference was the seed; trials
and clients were nested within the declared seed/world protocol. The other
snapshot budgets were:

| Surface | Executed budget | Independent unit | Primary quantity | Interpretation |
|---|---:|---|---|---|
| Belief sharing | 240 seeds; 7 agents | seed | communicating minus incommunicado free energy | categorical source-mechanism analogue under this protocol |
| Robustness sweep | 480 matched trials; 7 agents; 2 contaminated | trial conditional on one fixed seeded world; seed-level extension only in the cross-study tier | accuracy and robust-minus-naive accuracy | conditional contamination result |
| Contamination gallery | 16 seeds × 20 trials; rate 0.6 | seed after nested trial reduction | best robust minus naive accuracy by attack mechanism | exploratory mechanism contrast |
| Robustness onset | 6 seeds × 12 trials | seed after nested trial reduction | rate at which the robust curve first wins | exploratory onset diagnostic |
| Moving world | 64 seeds; 480 trials in the executed tier | seed | isolated, communicating, and EFE-guided accuracy | field-of-view and action-loop contrast |
| Disjoint field of view | 64 seeds; 200 trials in the executed tier | seed | communicating versus isolated accuracy; EFE versus random | communication-benefit control and navigation null |
| Hierarchical world | 64 seeds; 480 trials | seed | location gap plus context accuracy | hierarchy test, not a location-improvement theorem |
| Three-level world | 64 seeds; 480 trials | seed | location gap plus context/meta-context accuracy | parameterized extension with marginal meta-context performance |
| BNN robustness | 20 seeds × 200 observations per condition | seed | held-out accuracy contrast across contamination | small client-side complement |
| PyTorch MLP | one deterministic configured endpoint | endpoint | accuracy and simplex deviation | API-transfer smoke complement, not inferential evidence |
| Acuity recovery | 480 trials × 200 observations | trial under a finite grid | absolute error and $R^2$ | finite-grid diagnostic, not generic identifiability |

The 64-seed cross-study summary was a structural precision tier. Its study rows
use different metrics and units; it is not an omnibus hypothesis test and must
not be read as one pooled effect estimate.

## Historical report observations and their boundaries

### Belief sharing is lower free energy in the declared categorical control

The communicating condition has mean free energy 13.091566, compared with
16.467587 for the incommunicado condition, for a recomputed gap of 3.376021
nats. This supports a scoped source-mechanism analogue. It does not
establish that communication is necessary in every active-inference task or
that the same gap persists under a different observation model.

### The standard pool degrades sharply; robust methods are conditional complements, not universal winners

At the executed verdict rate of 0.8, the matched-trial means and percentile
intervals are:

| Method | Mean accuracy | 95% interval | Mean robust-minus-naive difference | 95% interval for difference |
|---|---:|---|---:|---|
| KLD / standard pool | 0.902564 | [0.898707, 0.906351] | — | — |
| RKL | 0.982981 | [0.982686, 0.983266] | 0.080417 | [0.076782, 0.083969] |
| AR | 0.986746 | [0.986479, 0.987020] | 0.084182 | [0.080616, 0.087844] |
| beta | 0.978486 | [0.978162, 0.978808] | 0.075922 | [0.072484, 0.079529] |
| rcce | 0.980804 | [0.980496, 0.981105] | 0.078240 | [0.074736, 0.081701] |

The four declared robust contrasts reject their paired nulls after the report's
BH-FDR procedure at this verdict rate; their rank-biserial effect is saturated
at one for these tied directional comparisons. That saturation is a property
of the observed ordering, not evidence of an infinite or universal effect. The
headline method token is RKL because the report's selection rule resolves the
rank-effect tie; AR has the largest observed mean and mean difference in this
snapshot. These are deliberately separate statements.

Across contamination rates 0, 0.225, 0.45, 0.675, and 0.9, KLD falls from
0.999698 to 0.692802. The robust profiles remain near 0.98–0.99 at the final
rate in this mechanism. This supports the claim that the declared robust
losses protect the tested consensus under this severe fixed attack setup; it
does not imply a dominance ordering at low contamination, a bounded-influence
theorem for `robust_aggregate`, or protection against arbitrary attacks.

The report's prospective sample-size calculation is useful for planning but is
not confirmatory evidence: an observed effect can make a required sample size
look trivially small. The 480-trial budget should therefore be described as
precision for the declared conditional mechanism, not as 480 independent
worlds.

### The contamination gallery is the critical negative control

At rate 0.6, the mechanism-specific gallery gives the following best-robust
contrasts:

| Attack mechanism | Mean difference | Interval disposition | Reliable win? | Audit reading |
|---|---:|---|---|---|
| confident wrong | +0.003851 | positive in the report | yes; win fraction 1.000 | narrow confident-misinformation benefit |
| drift | +0.003851 | positive in the report | yes; win fraction 1.000 | same directional result under this drift generator |
| Byzantine | +0.028282 | interval includes zero | no; win fraction 0.625 | exploratory, inconclusive |
| label noise | −0.005155 | negative in the report | no; win fraction 0.000 | negative control against universal superiority |
| uniform | −0.003759 | negative in the report | no; win fraction 0.000 | negative control against universal superiority |

This table is why the manuscript uses “regime-dependent” language. The
positive confident-wrong and drift rows do not rescue the Byzantine claim, and
the label-noise and uniform rows actively falsify a blanket “robust always
wins” summary. The gallery is exploratory because it uses only 16 independent
seeds after nested trials.

### The onset sweep is non-monotone and attack-dependent

The onset diagnostic finds first-win rates of 0.4 for the declared Byzantine
generator and 0.6 for confident-wrong/drift. The Byzantine robust curve wins at
intermediate rates but loses again at the highest rates; confident-wrong and
drift win only from their later onset onward in this run. This falsifies a
monotone protection story for the tested generators. With six independent
seeds, it remains an exploratory diagnostic rather than a stable threshold
estimate.

### Communication benefits depend on observability

The moving-world study has isolated accuracy 0.999186, communicating accuracy
0.982650, and EFE-guided accuracy 0.983659; its EFE-versus-isolated free-energy
gap is −0.345886. The binary/overlapping-view condition therefore functions as
a cost/null control, not evidence that communication is necessary.

The disjoint-field-of-view extension changes the estimand: isolated accuracy
is 0.331484 and communicating accuracy is 0.498984, against a chance baseline
of one sixth. This supports a conditional communication benefit when agents
possess complementary views. EFE-guided accuracy is 0.975156 versus random
0.977422, with a non-significant EFE-versus-random comparison in this report
($p=0.231894$). The communication result and the EFE navigation result must
not be conflated.

### Hierarchy adds latent structure but does not improve location

The two-level hierarchical location mean is 0.974674 versus 0.980371 for the
flat baseline, with mean gap −0.005697 and paired $p=1.75\times10^{-5}$. The
three-level location mean is 0.976237 versus 0.979590, with mean gap −0.003353
and paired $p=7.54\times10^{-4}$. These results do not support a location
improvement claim. They support the more modest claim that the implementation
can execute the declared hierarchy and report context-sensitive outcomes,
while the tested location objective is slightly worse than flat.

The three-level meta-context accuracy is close to its binary chance reference
and is described as marginal rather than as successful deep hierarchical
inference. A larger diagram or a parameterized constructor does not establish
arbitrary-depth scientific validity.

### The client-side BNN complement is non-monotone and small in scope

In the 20-seed BNN sweep, the largest robust-minus-standard accuracy margin is
0.0386875 at contamination 0.35. At the terminal contamination condition the
contrast reverses to −0.0087813. This supports a regime-dependent client-side
complement and is inconsistent with a claim that the robust client method
dominates at every contamination level. The one deterministic PyTorch MLP
endpoint is an implementation complement only; it is not a source-scale
mean-field FedGVI reproduction.

### Parameter recovery and heuristic characterization remain diagnostics

The acuity study reports mean absolute error 0.026825 and $R^2=0.989286$ on a
finite tested grid with 480 trials and 200 observations. This supports
finite-grid recovery under the declared model, not generic identifiability.

The heuristic characterization measures influence and finite breakdown
witnesses for one configured categorical setup. Both the sharp heuristic and
the variational comparison have finite measured capture thresholds in that
setup. This is evidence against silently claiming bounded influence for the
server heuristic; it is not a theorem about all simplex dimensions, attack
classes, or hyperparameters.

## Statistical interpretation and caveats

1. **Nested replication.** The 480 headline trials are not 480 independent
   hidden worlds. Trial-level intervals are conditional on the fixed world,
   true state, attack target, agent count, and protocol. Seed-level summaries
   are the appropriate higher-level precision surface for structural studies.
2. **Matched tests.** Wilcoxon tests and BH-FDR adjustments are appropriate to
   the declared matched contrasts, but their interpretation remains conditional
   on the generated pairs and the predeclared comparison family. Tiny $p$-values
   do not broaden the estimand.
3. **Effect-size saturation.** The report uses a finite sentinel for a
   d-equivalent display when the rank-biserial effect saturates at ±1; it is
   not a raw Cohen's d estimate.
   The manuscript formatter renders this as “saturated” rather than as a
   literal million-scale effect. Rank-biserial saturation says all observed
   paired differences have the same direction; it does not imply infinite
   practical effect.
4. **Intervals and MCSE.** Percentile-bootstrap intervals and Monte Carlo
   standard errors quantify variation over the declared resampling unit. They
   do not quantify uncertainty over unmodeled worlds, attacks, datasets, or
   deployment environments.
5. **Multiple metrics.** The cross-study summary mixes accuracy, free energy,
   parameter error, and other quantities. No pooled omnibus significance claim
   is licensed by that table.
6. **Small exploratory tiers.** The gallery and onset runs are intentionally
   smaller than the headline sweep. Their role is to reveal failure modes and
   shape the next experiment, not to certify a universal attack taxonomy.

## Decisions for the next phases

The historical evidence supported retaining the paper's bounded primary claims,
subject to a fresh result review after the enlarged rerun:

- the formal KL/NLL/robustness-zero recovery identity;
- the scoped categorical belief-sharing bridge;
- conditional robustness gains under declared severe contamination; and
- the observability-dependent communication result in the disjoint-view
  extension.

It rejects or narrows the following formulations:

- “robust always wins” or “universal robustness”;
- “Byzantine robustness” based on the historical gallery alone;
- “communication is necessary” without the disjoint-view qualifier;
- “hierarchy improves location accuracy”;
- generic acuity identifiability;
- arbitrary-depth empirical scaling;
- model-class universality from the MLP complement; and
- transfer of the client-side FedGVI bounded-influence guarantee to
  `robust_aggregate`.

The recommended order remains:

1. MAJ-1: characterize or refute an objective for the sharp server heuristic;
2. MAJ-2: run a protocol-matched mean-field BNN FedGVI study;
3. MAJ-4: specify authenticated transport separately from privacy and
   Byzantine robustness;
4. MAJ-3: extend the recovery identity to continuous or hybrid beliefs;
5. MAJ-5: preregister richer hierarchical tasks and controls; and
6. MAJ-6: use an independently sourced benchmark only after the estimand and
   threat model are stable.

## Historical reproduction and release checks — 2026-07-17

The artifact snapshot was regenerated and checked with the project workflow,
the full source/test gate, token hydration, template stages 03–05, and the
template output validator. The regenerated release surface contains a 75-page
combined PDF, 42 section HTML surfaces, 42 slide decks, and 33 figure
artifacts. The committed `output/` snapshot contains 487 files, while the
release manifest contains 404 artifacts. Those counts describe reproducibility
of the publication build; they are not scientific sample sizes.

For a fresh audit, use the receipt-aware producer order rather than treating a
historical hydrate or test total as reusable:

```bash
uv run --locked ruff check src tests
uv run --locked python scripts/01_run_invariants.py
uv run --locked python scripts/02_run_analysis.py
uv run --locked python scripts/z_generate_manuscript_variables.py --provisional-validation
uv run --locked --extra dev python scripts/validate_test_coverage.py
uv run --locked python scripts/z_generate_manuscript_variables.py
cd /path/to/template
uv run --locked python scripts/pipeline/stage_03_render.py \
  --project working/active_fedference --skip-manuscript-hydration
uv run --locked python scripts/pipeline/stage_04_validate.py \
  --project working/active_fedference
uv run --locked python scripts/pipeline/stage_05_copy.py \
  --project working/active_fedference
```

The source reports and claim ledger remain authoritative if a future rerun
changes any snapshot number. Regenerate the final hydration and rendering
receipts after the test receipt; then update this audit or add a dated successor
rather than silently replacing the conditional interpretation.

## Subsequent MAJ-1 diagnostic extension

The server-rule characterization includes a deterministic scenario grid over
state dimension, honest-agent count, robustness, four named contamination
mechanisms, and balanced versus adversary-downweighted base weights. The new
rows are a counterexample-search instrument. They do not change the independent
unit to a population of worlds, and an uncaptured row means only that no witness
was found within the declared adversary budget. The empirical portion therefore
retains the status `scoped_implementation_fact` and `open_no_global_objective`.
Separately, the current source schema and formal-witness producer define an
exact scoped no-go witness for the declared separable raw-log-pool objective
class. A fresh report is required before treating its presence as a current
artifact fact. The formal proposition is not inferred from the grid and does
not rule out broader coupled objectives.

## Verifier and roadmap reassessment — 2026-07-15

The follow-up RedTeam pass found one verifier design defect and three genuine
forward evidence gaps. The script-smoke test was marked as smoke/publication
but called the publication-scale analysis entry point without an explicit
profile. That was a runtime/oracle-budget defect: an eventually passing test
could still consume an inappropriate publication budget or be interrupted. The
entry point now validates an explicit `publication` or `smoke` profile,
defaults to publication for release use, and the smoke test invokes the same
real pipeline with bounded settings. The targeted profile and subprocess
controls passed **2/2**; this repair does not alter any scientific estimand or
aggregation claim.

At that 2026-07-15 checkpoint the roadmap was reclassified by evidence role.
The scholarship-indexed phase plan is a governance control rather than a
seventh major scientific upgrade. Three genuine forward gaps were then scoped:
(i) vary hidden state, attacked target, and observation geometry while
preserving seed-level independence; (ii) add calibration/proper-score
sensitivity with overconfidence and oracle controls; and (iii) bind release
manifests to source/config fingerprints so byte-integrity checks cannot certify
an internally consistent but stale bundle. The first two evidence slices and
the implementation for the third are now landed below; none is treated as
evidence for a universal robustness theorem.

## Historical statistical and artifact closure — 2026-07-28

The corrected machine-local run completed **1,149 tests** with **92.48%** source
coverage. The statistical interpretation remains conditional on the declared
trial/seed replication units, fixed worlds and attack targets, and the
percentile-bootstrap/Wilcoxon/BH procedures specified above. The parameter-
recovery interval is explicitly an empirical percentile interval across
independent trials, not a bootstrap or Bayesian credible interval; the gallery
and onset displays use seed-level bootstrap confidence intervals with pooled
method selection. Prospective power respects the direction of the stated
alternative, so opposite-direction effects cannot produce an artificially
small sample-size recommendation. No new population, deployment, calibration,
or universal robustness claim is inferred from the rerun.

The regenerated publication surface contains an 80-page combined PDF, 42
manuscript sections, 42 slide decks, 43 HTML surfaces including the index, and
29 registered figure stems with 58 deterministic PNG/PDF assets. The package,
rendered-surface, freshness, output, metadata, and release-manifest gates
passed; `qpdf --check` passed all 43 publication PDFs. These counts are
publication-build reproducibility metadata, not scientific sample sizes.

## Historical conditional-world and belief-quality extensions — 2026-07-28

The two medium roadmap slices were then executable and source-bound. The
conditional-world report contains 40 pre-registered cells: two hidden states,
two observability levels, five attack mechanisms, and two adversarial-weight
settings. Each cell reduces 12 nested trials to one value per each of 16
independent seeds before the primary contrast or its percentile-bootstrap
confidence interval is formed. The report's controls pass exactly: the
robustness-zero path recovers the log-linear pool, all attack targets differ
from their true states, and the seed is the declared independent unit.

The result is intentionally mixed. At low observability, label-noise cells
show a large positive robust true-state-mass contrast, while permutation cells
are negative; at higher observability, confident-wrong cells are positive at
full adversarial weight but can turn negative at half weight. Clean and
uniform cells are near zero. This exercises the roadmap falsifier and narrows
the claim: the extension is a finite-grid attack-geometry diagnostic, not a
universal robustness result, a breakdown point, or a theorem for
`robust_aggregate`.

The belief-quality report evaluates a six-cell subset at the higher
observability setting. Categorical log score is the primary measure (higher is
better); Brier score and ECE are secondary diagnostics. The oracle, uniform,
and confidently-wrong controls satisfy the predeclared ordering and use 16
independent seeds with 12 nested trials per seed. Every tested robust-minus-
naive log-score contrast is negative, including the confident-wrong attack;
the correct conclusion is therefore no demonstrated belief-quality advantage
for the robust consensus in this subset. Accuracy and proper-score results are
not substituted for one another, and the report does not imply calibration
under shift or decision optimality.

The new hybrid module is a separate representation slice: it tests a
categorical log pool plus Gaussian precision pooling and a finite robust
fixed-point diagnostic, with exact zero-robustness recovery. It does not close
the continuous/hybrid active-inference phase because no full hybrid task,
objective-backed robust rule, or deployment experiment has been added.
