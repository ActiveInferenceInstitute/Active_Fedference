# Visual Scholarship And Accessible-Slide Integration

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Medium
- State: Open
- Queue position: immediate integration tranche before the v1.1 release-only
  identity change
- Owner surface: figure contracts, captions and long descriptions, exact-value
  fallbacks, manuscript prose, HTML, slides, Template renderer lock, and public
  PR integration

## Rationale

The visual-scholarship source changes span two repositories and several reader
surfaces. They become current evidence only when the opt-in accessible Template
profile is reviewed independently, Active Fedference pins the exact merged
renderer commit, source-owned figures and prose agree, generated artifacts are
rebuilt in producer order, and the exact public candidate passes both local and
hosted review. A visually attractive scratch render or a green source-only test
is not sufficient.

## Scope

Use the exact reviewed and merged Template renderer commit recorded in
`manuscript/config.yaml`, from a separate clean checkout. Preserve the user-owned dirty Template
checkout unchanged. The accessible profile must fail clearly on unsafe density
while leaving the archive profile unchanged for other consumers.

Complete Active Fedference's approved sanitized history on
`codex/v1.1-visual-scholarship`, created from refreshed public `main`. The
following work areas describe dependency order, not a fixed commit count:

1. align visual claims, estimands, uncertainty, replication units, and neutral
   scholarly wording;
2. add the application-integrity, evidence/replication, and source-to-render
   figures plus shared semantic styles;
3. integrate self-contained captions, concise alternatives, structured long
   descriptions, exact-value fallbacks, accessible Mermaid/text equivalents,
   and reader-surface documentation; and
4. regenerate publication artifacts from the final source and exact renderer
   lock.

The authorized continuation preserves the existing visual commits and permits
additive corrective commits for source, tests, and documentation. Keep those
corrections separate from `build: refresh visual publication artifacts`, which
contains generated artifacts only, and explain their scope in the public PR.
The private evidence branch remains unchanged. Do not substitute cherry-picking
of private commits, a compact replay, squash, rebase, or a history rewrite to
force an artificial commit count. Any further history-policy change requires
new approval; a clean scan does not grant that authority.

The three explanatory figures do not add experimental observations. Existing
data-bearing figures may change composition or encoding only when the typed
report remains the numerical source. All registered PNG/PDF pairs, HTML pages,
reveal.js decks, Beamer derivatives, and the combined PDF remain source-bound.

## Deferred Frontmatter Reader Surfaces

The three explanatory opening plates (`frontmatter_model`, `frontmatter_robustness`,
`frontmatter_execution`; manuscript `00a`-`00c`) keep their caption, concise
alternative, registered PNG/PDF pair, slide manifest, and ordered placement, but
they are not declared complex and carry no structured long description until
their reader surfaces exist. The web-package contract requires a labeled
long-description disclosure in the generated HTML for every figure registry
entry that carries a structured long description, and the pinned Template
producer has not rendered a chapter page for `00a`-`00c`; declaring the plates
complex therefore demands a reader surface that does not exist yet.

The three descriptions are preserved verbatim below and return to source in one
reviewed change:

1. Render `manuscript/00a_graphical_model.md`,
   `manuscript/00b_graphical_robustness.md`, and
   `manuscript/00c_graphical_execution.md` under `output/web/` with the pinned
   Template producer, so each page carries its plate's captioned disclosure.
2. Add `frontmatter_model`, `frontmatter_robustness`, and `frontmatter_execution`
   back to `COMPLEX_FIGURE_GENERATORS` in `src/analysis/visual_contracts.py`.
3. Restore the three structured long descriptions below into
   `src/figures/_metadata.py::_build_metadata`, together with the plates'
   complex-figure assertions in `tests/figures/test_frontmatter.py`.
4. Regenerate `output/figures/figure_registry.json` through
   `analysis.workflow._write_figure_registry`, then run
   `uv run --locked python scripts/prepare_web_package.py` and
   `uv run --locked pytest tests/test_web_publication_contract.py tests/figures/test_frontmatter.py -q`.
5. Record the rendered page paths, the regenerated registry, and the green
   publication gate in this page.

Preserved structured long descriptions, one per plate:

### `frontmatter_model`

```text
Reading order: top to bottom. Solid arrows show model or data flow; dashed arrows distinguish communication or prerequisites.

Generative model: D supplies initial state beliefs. B maps state and action to the next state. A maps hidden states to observations. C supplies log preferences over outcomes.

Observe outcomes, infer a local categorical posterior, evaluate policies with expected free energy, then act and observe again. Solid arrows distinguish the closed perception and action cycle.

Share posterior beliefs, not raw observations. Dashed message edges take admitted posteriors to fusion and return recipient-specific cavity beliefs, excluding the recipient's own message.

The Eq. 7 bridge assumes shared positive support, posterior log potentials, and fixed weights. It specializes categorical combination; it does not reconstruct the complete source protocol.
```

### `frontmatter_robustness`

```text
Reading order: top to bottom. Solid arrows show model or data flow; dashed arrows distinguish communication or prerequisites.

Client FedGVI: generalized variational updates combine a divergence and a loss. Bounded-loss influence claims require the source's matching model and regularity assumptions. KL/NLL with beta zero recovers the project-local Bayes update.

Heuristic server: robust_aggregate discounts reports by disagreement with the emerging consensus. At robustness zero it exactly returns log_linear_pool. Recovery only; no transferred client theorem.

Variational server: variational_aggregate alternates consensus and raw effective-weight updates on a stated objective. Descent and raw-weight bounds do not prove estimator B-robustness.

Choose one server rule after the client update. Guarantees stay with their own mathematical construction. No universal robustness or source protocol replication follows from this diagram.
```

### `frontmatter_execution`

```text
Reading order: top to bottom. Solid arrows show model or data flow; dashed arrows distinguish communication or prerequisites.

A GNN model document is admitted through a pinned parser, or the packaged sentinel uses its source-bound cache. Validate dimensions, axis order, support, and model matrices before consumption.

Native execution consumes GNN-derived A, B, C, D values through native_world and native_posterior. The sentinel exchange example compares extracted matrices against the native model.

Optional upstream execution requires an explicit source root, qualified runtime and backend contract, followed by real execution and parity checks. Dashed arrows mark prerequisites, not a backend support or qualification claim.

Receipts bind declared sources, inputs and outputs. Integrity evidence does not establish scientific validity. Parser success is not execution; matrix parity is not protocol replication; scientific conclusions still require appropriate models, controls and scoped evidence.
```

## Implementation Notes

- Use the shared figure saver, metadata registry, accessibility registry,
  typed report schemas, artifact inventory, manuscript syntax ledger, and
  freshness inputs. Do not write publication files directly.
- Use hue-independent markers, dashes, hatches, direct labels, minimum text
  sizes, and contrast thresholds. Reflow dense figures instead of shrinking
  typography.
- Keep HTML the accessibility-enhanced canonical reader. Describe the combined
  PDF only as having verified tagged structure; describe Beamer PDFs as
  untagged derivatives. Do not claim WCAG or PDF/UA conformance.
- Perform a new candidate-history confidentiality audit after the source and
  generated commits are fixed. Push the exact SHA to both Fedference remotes,
  open a public PR, and merge only through a normal merge commit after the
  exact head and merge ref are green.

## Acceptance Criteria

- Primary estimand: not applicable; this tranche changes explanation,
  accessibility, and evidence binding without creating a scientific effect.
- Independent replication unit: not applicable; the audit/review unit is one
  registered figure or diagram together with its typed source, metadata,
  caption, alternative, applicable fallback, and each rendered surface. One
  slide frame is the density-review unit.
- The Template PR passes its unit, integration, type, lint, density, geometry,
  and real Active Fedference render probes and is merged normally.
- Active Fedference pins the exact Template merge commit and rejects a dirty or
  mismatched renderer.
- Every complex figure has a structured long description; every declared
  quantitative figure has a source-generated exact-value fallback; every
  Mermaid block has an accessible title, description, and nearby text
  equivalent.
- All figures pass native, grayscale, final-PDF-scale, and HTML review; every
  slide page passes contact-sheet review, with focused checks of formalism,
  results, tables, figures, and references.
- Browser checks cover keyboard focus, contextual full-size links, landmarks,
  and document-body reflow at 200% and 400% zoom.
- Required acceptance evidence: Template merge SHA and CI URLs, Fedference
  source and generated commit SHAs, figure/accessibility manifests, PDF and
  browser reports, confidentiality disposition, public PR checks, public merge
  SHA, and green post-merge public-main workflow.
- The public branch preserves the approved sanitized history and additive
  corrective commits from the refreshed public-main base. Its audit records
  that base and every newly reachable object, while the private evidence
  branch remains unchanged. Source corrections and generated artifacts are
  separate commits; no history rewrite is used to impose a commit count.

## Verification Probes

- Figure, caption, exact-value, palette, report-schema, and manuscript-claim
  tests.
- `uv run --locked python scripts/validate_mermaid.py` and the renderer probe.
- Real clean-Template PDF, HTML, reveal.js, and Beamer generation.
- Browser zoom/keyboard/focus checks and slide-density validation.
- `pdfinfo`, qpdf catalog inspection, extracted text, raster/contact-sheet
  reads, `/Lang`, and non-empty `StructTreeRoot` checks.
- Full non-slow, integration, publication, example, coverage, Ruff, mypy,
  freshness, release-manifest, reproducible-build, and installed-package gates.

## Claim-Boundary Constraints

- Falsifier: any figure/report/caption disagreement, inaccessible color-only
  encoding, undersized effective text, stale artifact, renderer mismatch,
  browser body overflow, unresolved slide density, red hosted check, or open
  confidentiality finding blocks integration.
- Prohibited claims: the new figures do not establish calibration, domain
  suitability, scientific validity, universal robustness, source-protocol
  identity, physical multi-host federation, WCAG conformance, PDF/UA
  conformance, or a downstream decision.
- A release-quality visual surface does not upgrade the evidence class of the
  quantity it depicts.

## Dependencies

Depends on green public Active Fedference `main`, the current typed report and
figure contracts, a separate clean Template worktree, and MIN-2's
candidate-specific history and verification gates. MED-5 starts only after the
visual merge and its post-merge workflow are green.
