# Visual claim audit

Updated 2026-08-13. The numerical and render records dated 2026-07-17,
2026-07-28, and 2026-08-01 are historical snapshots. The checked-out source
declares an enlarged experiment configuration and a revised selection-free
review-grid producer. A visual is current only when its named report and the
analysis, hydration, render, and surface-validation records bind the same
source inputs; this ledger does not substitute an older snapshot for that
evidence. The canonical embed path is the PNG named in
`manuscript/SYNTAX.md`; generators using `save_figure_pair` also retain a PDF
companion for archival use. The registry/embed/generator consistency gate
checks the filename relationship directly.

The v0.1.0 source/PDF was published on 2026-08-10; publication does not alter
the evidence class assigned to any visual. This ledger records what each
manuscript visual is allowed to communicate. It
keeps the Friston et al. (2024) generative-model/message-sharing connection
visible without allowing a schematic, server heuristic, or conditional
simulation to inherit a guarantee it does not own.

## 2026-08-13 source-current refresh

The source-current reviewer snapshot was regenerated after the publication
analysis, receipt-backed hydration, and the template 03–05 renderer passes.
The combined manuscript is 75 pages, with 43 manuscript HTML surfaces, 43
slide PDFs, and 30 registered figure PNG/PDF pairs. The rendered-surface and
web-package gates passed; `qpdf --check` reports no syntax or stream errors.
The archived combined PDF for that refresh reported `Tagged: no`, so the HTML
surface was the accessibility-enhanced surface and PDF/UA conformance was not
claimed. The current source now requests tagged structure for the combined
manuscript; the regenerated surface and its receipts must be checked before
the status is changed.

Visual review found one concrete layout defect in the cross-study summary:
small negative effects placed their value labels immediately left of zero,
where they could collide with long study labels. The figure producer now puts
near-zero negative labels in a lightly boxed lane just right of zero, while
keeping material negative effects at their interval endpoints. The regenerated
figure was inspected after the fix alongside the review grid, calibration,
complexity, graphical-abstract, and BNN robustness figures; no clipping,
overlap, unreadable glyphs, or misleading uncertainty encoding was observed.

This refresh records rendering and legibility evidence only; it does not
promote any scientific claim beyond the source-bound estimands and claim
ledger.

## Current visual source contract

`robustness_review_grid` is a conditional finite-grid diagnostic backed by
schema `1.1`. Its left panel reports a half min--max span over the finite
conditional cell, not a confidence interval. Its rate panels draw every
predeclared non-KLD method separately, with that method's own 95% seed-bootstrap
interval; they do not consume legacy pooled-display or winner-selection fields.
The report retains every signed method × rate × directional-mechanism cell and
fails if the configured maximum MCSE target is not met. Captions name the
estimand, seed/trial nesting, uncertainty semantics, and the absence of
universal, causal, breakdown, or independence claims.

The contamination-gallery and robustness-onset surfaces remain
pooled-best/descriptive displays. Their pooled-best summaries and intervals may
show finite, exploratory patterns, but they are not a post-selection
inferential procedure. The review grid is the dedicated selection-free,
all-method inference surface. A checked-in figure is evidence for this contract
only when its analysis report, hydration receipt, render receipt, and validated
surface are source-current together.

Formal figure repairs use `D_0` for the initial state prior, bold
`\boldsymbol{\pi}` for policy, and `p_C` for preferred outcomes. Generative,
message-passing, POMDP, and system-overview visuals remain deterministic
formal/mechanistic schematics unless their captions explicitly bind them to a
data-bearing report; no schematic receives an empirical CI.

| Visual | Mode | Evidence source | Permitted reading | Explicit boundary |
| --- | --- | --- | --- | --- |
| `graphical-abstract.png` | Deterministic formal/mechanistic schematic | `system_overview` metadata and pooled beliefs | Recovery anchor, broadcast pathway, and three-axis map | Not a benchmark, CI, significance result, or transfer of the variational bound to `robust_aggregate` |
| `generative_model_schema.png` | Deterministic formal schematic | Categorical `A/B/C/D`, state-inference equation, hierarchy API | Temporal, hierarchical, and factorial model structure | Does not claim every displayed dependency is estimated in every study |
| `message_passing.png` | Deterministic mechanistic schematic | Aggregation and belief-sharing equations | Local update → broadcast → server-fusion routes | Client-side FedGVI, heuristic server, and variational server have separate claim ownership |
| `pomdp_loop.png` | Deterministic model schematic | Categorical POMDP and moving-world implementation | Hidden state → observation → belief → action → transition, plus federation | Flat studies use an inference-sharing subset; moving-world results execute the action/transition extension |
| `system_overview.png` | Deterministic configured schematic | `SYSTEM_OVERVIEW_METADATA` | One derived failure-and-repair example | Not universal robustness evidence and not a variational-server result |
| `complexity_scaling.png` | Implementation-derived diagnostic | `complexity_scaling.json`, `src/fedference/complexity.py`, and the public-path timing experiment | Dense-path orders plus seeded machine-local scaling observations | Dotted lines are normalized guides; min–max bars are not confidence intervals; no cross-machine, network, FLOP, or universal asymptotic claim |
| `conditional_world.png` | Conditional empirical finite-grid diagnostic | `conditional_world.json` and `run_conditional_world_generalization` | Seed-level true-state-mass contrasts across hidden state, observability, attack, and adversarial-weight cells | Mixed cell signs are retained; no universal attack or breakdown claim |
| `robustness_review_grid.png` | Conditional finite-grid, all-method inferential diagnostic on a fresh run | Fresh schema-`1.1` `robustness_review_grid.json` with its precision receipt | Every predeclared non-KLD robust-minus-KLD contrast by directional mechanism and rate, with method-specific seed-bootstrap intervals | No winner selection, pooled inference, universal attack claim, or claim before a source-current report and render exist |
| `belief_quality.png` | Conditional empirical proper-score diagnostic | `belief_quality.json` and `run_belief_quality_sensitivity` | Categorical log-score controls with Brier/ECE secondary diagnostics and seed-level uncertainty | Controls validate score direction only; no calibration-under-shift, decision-optimality, or robust-score advantage claim |
| `sensitivity_heatmap.png` | Deterministic seeded sensitivity summary | `SENSITIVITY_NOISE_FLOOR` plus the executed cell means | Acuity × colony-size gaps and the declared near-zero display convention | Hatching is a visualization threshold, not a test of statistical significance or a zero-effect claim |
| Data-bearing result figures | Conditional empirical or formal diagnostic | JSON reports and core tests | Only the statistic, estimand, and run named in the caption | CIs quantify the declared resampling unit; they do not generalize across unmodeled worlds or deployments |

## Source-to-project comparison

| Source surface | Project surface | Material protocol difference | Permitted interpretation | Prohibited claim |
| --- | --- | --- | --- | --- |
| Friston et al. Eq. 2 | `efe_decomposition` signed waterfall | Categorical state--outcome information gain is displayed; parameter-learning terms from the source equation are not added | Formal specialization of the executable algebraic identity | Exact reproduction of the full source decomposition |
| Friston et al. Fig. 5 | `free_energy_comparison`, `message_passing` | Reduced categorical colony and one-round summary rather than the source multi-agent, multi-panel protocol | Source-mechanism analogue of belief sharing and its declared free-energy estimand | Figure-level reproduction or transfer of source numerical values |
| Friston et al. Fig. 7 | `language_kl_decay` | One categorical likelihood trajectory aggregated over configured seeds rather than the source auditory mappings and episode protocol | Source-mechanism analogue of the language-acquisition estimand; seed is the replication unit | Exact episode-level replication or CI over ordered time points |
| Friston et al. Fig. 9 | `emergence_bmr` | Deterministic categorical BMR pruning diagnostic rather than the source multi-agent language-naive protocol | Diagnostic related to the model-reduction mechanism | Exact source simulation or universal structure-emergence claim |

## Red-team checks

- A caption for a schematic must say `schematic`, `formal`, or `mechanistic` and
  state that no uncertainty interval applies.
- A caption for a stochastic or seed-aggregated figure must name the interval,
  sample/resampling unit, and estimand.
- Gallery and onset captions must call their pooled-best displays descriptive;
  they must not borrow the review grid's selection-free inference language.
- A review-grid caption must identify its all-method, selection-free rate
  curves and must not call a finite-grid min--max span a confidence interval.
- `robust_aggregate` may be described as a server heuristic with conditional
  empirical behavior and a recovery limit; it may not be described as carrying
  the per-client FedGVI bounded-influence guarantee.
- `variational_aggregate` may be described as objective-backed and conservative;
  its weight-control property may not be converted into universal truth recovery
  or peak-accuracy dominance.
- The POMDP loop must remain visibly and textually distinct from the subset of
  paths executed in each experiment.

The figures are inspired by the explanatory function of the cited active-
inference diagrams, but their geometry, palette, labels, and assets are original
to this project.

## Current visual and caption closure — 2026-08-09

The current source-bound pass reviewed all 30 registered figure stems at native
PNG/PDF scale, the combined manuscript at page scale, representative slide
surfaces, and the generated HTML asset/alt-text surface. The final second-pass
bundle contains 30 PNG/PDF figure pairs, 43 slide PDFs with matching TeX/log
records, 44 HTML surfaces including the index, 60 checked web assets, and a
75-page combined PDF. The figure registry, manuscript embeds, captions, and
generator outputs agree on the same figure stems; the typed caption/figure
contract and rendered-surface checks pass.

The review specifically corrected the hierarchical POMDP diagnostic annotation
so it no longer overlaps the lower bar/value, raised the small quantitative
legend/note labels to the shared figure font floor, and changed the associated
caption from a source-like “schematic” implication to an explicitly
source-inspired original-project diagnostic. Captions were checked for their
estimand, units, replication/resampling unit, uncertainty semantics, and
no-claim boundary. Formal and mechanistic diagrams remain explicitly
uncertainty-free schematics; finite-grid spans remain descriptive ranges rather
than confidence intervals; and robustness axes remain distinct.

All five Mermaid blocks in the root README and `docs/` render to SVG with the
same source contract used by the static validator. The combined PDF passes
`qpdf --check`; the archived PDF reported `Tagged: no`, so the HTML surface
remained the accessibility-enhanced canonical surface and no PDF/UA
conformance was claimed. The source-current combined PDF has an additional
tagging/language/structure gate, but tagged structure is still not PDF/UA
conformance. These are render and claim-boundary facts, not additional
empirical evidence.

## Historical render-surface record

The 2026-07-17 reconciliation inspected the figures at native PNG scale, the
combined PDF at page scale, the per-section HTML asset/alt-text scale, and
representative slide scale. The publication-default render contains 42
manuscript sections, 42 slide decks, 43 web HTML surfaces including the index,
26 registered figures with deterministic PNG/PDF companions, and a 76-page
combined PDF. The output validator, web-package validator, and template stages
03–05 all passed. The visual checks found no clipping or overlap in the
equations, arrows, role badges, Figure 6 endpoint annotation, or caption
blocks. Slide-scale figures are necessarily reduced by the template layout, so
the archival PNG/PDF remains the preferred surface for reading fine notation.
This is a legibility record, not a new empirical result or a substitute for the
caption/evidence gates.

The source-to-project relationship is bounded throughout: Figure 6 is a formal
categorical specialization of Eq. 2; Studies 1–3 are reduced
source-mechanism/protocol analogues rather than exact Friston figure
reproductions; and the future faithful lane is scoped in
[`faithful-friston-protocol-replication.md`](../todo/faithful-friston-protocol-replication.md).

## Historical render-surface closure — 2026-07-28

The corrected machine-local render was inspected at native figure scale,
combined-PDF page scale, representative slide scale, and generated HTML
asset/alt-text scale. It contains 42 manuscript sections, 42 slide decks, 43
HTML surfaces including the index, 29 registered figure stems with 58
deterministic PNG/PDF companions, and an 80-page combined PDF. The dedicated
rendered-surface gate passed all 42 slide PDFs, 42 TeX sources, 42 logs, 43 HTML
surfaces, and 58 web assets: no overfull boxes, missing characters, raw citation
markers, unresolved references, or broken web assets were reported. `qpdf
--check` passed all 43 publication PDFs. Targeted visual QA confirmed that the
new seed-bootstrap bars/bands, pooled-method annotations, empirical
parameter-recovery intervals, finite-grid attack geometry, proper-score
controls, and finite-grid complexity diagnostic are legible and not clipped.
These are surface-quality facts only and do not upgrade any visual from
schematic to empirical evidence.

## Historical render-surface closure — 2026-08-01

The then-final second-pass template render processed 43 manuscript files into 43
slide PDFs, 43 TeX sources, 43 slide logs, the combined 84-page PDF, and 44
HTML surfaces including the index. The registered figure surface contains 30
PNG/PDF pairs (60 assets) plus the figure registry. Template stages 03, 04,
and 05 passed; the dedicated rendered-surface gate passed with no unresolved
references, raw citation markers, missing characters, broken web assets, or
material layout findings. The web package gate passed with 44 HTML files and
60 checked figure assets, and the loopback browser pass returned HTTP 200 with
zero console errors, a functioning skip link, and all requested figure assets
served successfully. The four console warnings were MathJax component-version
warnings only.

Native inspection of the robustness review-grid figure and notation, EFE,
robustness-sweep/onset, complexity, generative-schema, and message-passing
surfaces found no clipping or illegible annotations at the reviewed scales.
The archived combined PDF passes `qpdf --check`, but `pdfinfo` reports
`Tagged: no`; the HTML surface was therefore described as accessibility-
enhanced and no PDF/UA conformance was claimed. These are build and legibility
facts, not new scientific evidence. The current source-controlled render must
be validated separately through the tagged-PDF gate. This render predates the
enlarged configuration and the selection-free review-grid figure contract
above, so it cannot serve as their current visual-validation record.
