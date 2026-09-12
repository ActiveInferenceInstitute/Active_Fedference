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
