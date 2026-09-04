# Clone-Correct Integration And Release Verification Ladder

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Minor
- State: Open
- Queue position: candidate-specific checks before each public push and the
  authoritative two-clone campaign before v1.1 tagging
- Owner surface: tracking set, renderer identity, public-history review,
  reproducible distributions, isolated-clone receipts, release-verdict
  eligibility, and ISC-242

## Rationale

Release evidence is commit-specific. Passing tests on an earlier source commit,
in a dirty worktree, or against an unpinned renderer cannot certify a later
candidate. Analysis, hydration, figures, rendering, package builds, and release
manifests are valid only when their declared inputs and outputs bind to the
exact reviewed tree in producer order.

The active v1.1 path has two different verification scales. A public feature or
release branch first needs a complete candidate-history audit plus the
applicable local and hosted gates. The final public-main merge commit that
would be tagged then needs two isolated, sequential, write-once clone runs.
Neither scale replaces the required release verdict or publication authority,
and neither creates new scientific evidence.

The Minor classification describes this lane as verification rather than a new
scientific or product capability. It does not describe runtime or importance:
the final clone campaign is deliberately broad, coordination-heavy, and
release-blocking.

## Scope

### Candidate gate before each public branch push

For the v1.1 public application/visual lineage, the authorized history policy is
the reviewed four-commit sanitized replay based directly on refreshed public
`main`. The retained private evidence branch remains unchanged. A clean scan
does not authorize a different replay, cherry-pick, squash, rebase, or history
rewrite. Any history-policy change requires new approval.

1. Fetch both remotes and record the exact public base, private base, candidate
   commit, tree, branch, renderer commit, and worktree status. Rebuild or stop
   when a base moved unexpectedly.
2. Audit every commit, tree, current/deleted path, and blob newly reachable in
   `public/main..candidate`. Run a full-history secret scan; review authors,
   emails, commit messages, machine-local paths, internal drafts, proprietary
   data, generated artifacts, Gitlinks, and every object at or above 50 MiB;
   hard-block an object above GitHub's 100 MiB limit.
3. Require a clean candidate worktree and an explicit clean Template checkout
   at the exact configured renderer commit. A nearby dirty Template checkout is
   never an implicit dependency.
4. Run targeted tests during editing, then the full applicable source,
   publication, render, browser, slide, package, and reproducibility gates.
5. Push the fixed candidate SHA non-forcibly to the required `codex/*` branches.
   Merge only through a public PR after the exact head and merge ref are green,
   using a normal merge commit; require a green post-merge public-main workflow.

### Exact-final-SHA release campaign

After MED-5A's release-only PR merges and public `main` is green, recapture the
exact merge commit rather than certifying the PR head. The release-only PR must
therefore precede this final certification; MIN-2 is not a prerequisite for
creating or merging that PR. Immediately before
each clone, verify at least 40 GiB of safe headroom and confirm that no active
producer or protected data/caches would be disturbed. Use a new write-once
namespace beneath
`/Volumes/blue/active_fedference-verification/v1.1.0-<FINAL_SHA>/` with separate
`clone-a`, `clone-b`, and sibling receipt directories. This is an approved,
non-confidential operational example; it names no credential, user home,
dataset location, or proprietary input.

Run clone A and clone B sequentially without reusing either checkout. Give each
clone its own clean renderer checkout at the configured Template commit. Each
run must independently execute:

- exact source/tree/lock/renderer and tracking-set capture;
- Ruff, mypy, invariants, layer-boundary, non-slow, integration, publication,
  example, and full coverage gates;
- publication analysis, typed report validation, figure generation,
  provisional hydration, source-bound coverage receipt, and final hydration;
- clean PDF, HTML, reveal.js, and Beamer rendering;
- Mermaid, bibliography, cross-reference, token, caption, figure, accessibility,
  browser, slide-density, freshness, scale-guard, and release-manifest checks;
- post-suite proof that tests did not mutate the committed publication snapshot;
- combined-PDF structure, text, `/Lang`, non-empty `StructTreeRoot`, qpdf, and
  formalism/results raster review, described only as tagged structure;
- wheel and source-distribution installation in separate empty environments,
  installed application/CLI smoke, package membership, `py.typed`, and no
  default Torch import; and
- two builds of each distribution under one explicit `SOURCE_DATE_EPOCH`, with
  byte-identical pairs and cross-clone comparison wherever determinism is
  declared.

Retain command, environment, source, renderer, lock, output, distribution,
rendering, visual-review, and result receipts outside the disposable clone
trees.

### Release-certification verdict

Technical eligibility for this v1.1 release checkpoint permits either an
identified human or a genuinely different-vendor reviewer. A local subagent
does not qualify. The selected v1.1 path is specifically to present the exact
final SHA, tree, manifest, claim ledger, no-claim boundaries, two-clone evidence,
distribution hashes, PDF evidence, and unresolved items to Daniel Ari Friedman
and record his verdict as identified owner-author human review, not as
independent external replication or cross-vendor review.

The structured result is `pass`, `concerns`, or `fail`. `pass` supplies the
technical release verdict but does not itself supply publication authority;
it permits tagging only under the separately authorized MED-5B sequence;
every concern requires disposition or a reviewed repair and complete
recertification; `fail` blocks release. This release verdict does not close ISC-89 or the broader independent-reproduction lane.

## Implementation Notes

- Use repository commands of record and retain full failing output. Never
  weaken a gate, delete evidence, change a frozen threshold, or switch to an
  unapproved history policy to manufacture a pass.
- Keep generated output producer-owned. Regenerate in declared order rather
  than hand-editing a stale report, figure, manuscript, receipt, or manifest.
- `scripts/validate_clean_checkout.py` is the tracking/import front door;
  `scripts/validate_pipeline_freshness.py` is the upstream/downstream freshness
  front door; `scripts/build_release.py --verify` checks the exact release
  bundle.
- Free-space status is volatile and confers no authority to delete active data,
  caches, worktrees, or outputs.
- Development identity remains a PEP 440 development version with an empty DOI,
  `(forthcoming)`, and no release date or project DOI URL. Final identity is set
  only in the reviewed release-only PR.

## Acceptance Criteria

- Primary estimand: not applicable; this item establishes exact-source release
  integrity rather than a scientific quantity.
- Independent replication unit: not applicable; audit/review units are one
  fixed public candidate for the branch gate and each fresh clone plus complete
  ladder for the final campaign. Both clone units are required.
- The load-bearing tracked set includes every imported module, test, source
  document, figure/accessibility contract, example, and packaged resource.
- The candidate audit covers all newly reachable history and records zero open
  publishability concerns, object-size disposition, tool versions, public base,
  candidate SHA/tree, and evidence-manifest digest.
- Each final clone independently regenerates and validates the declared source,
  report, render, accessibility, distribution, and manifest chain.
- Clone outputs and distributions agree byte-for-byte where the repository
  declares determinism.
- The v1.1 application/visual history is exactly the approved four-commit
  sanitized replay from refreshed public `main`; the private evidence lineage
  is unchanged, and any policy change has separate approval.
- One eligible structured `pass` binds to the exact public-main commit and
  evidence bundle. The selected path is the owner-author review, which is
  labeled honestly, remains distinct from publication authority, and does not
  close ISC-89.
- Required acceptance evidence: candidate-history record, hosted check URLs,
  post-merge workflow URL, two clone receipt sets, cross-clone comparison,
  release-verdict record, and an ISC-242 update only after every required
  component exists.

## Verification Probes

- `git ls-files` audit against the load-bearing import, docs, test, figure,
  example, and package set.
- Full commit/blob history and secret/size/path audit for each fixed candidate.
- `uv run --locked python scripts/validate_clean_checkout.py`.
- The complete command floor in `TODO.md`, real clean-renderer production,
  browser/slide/PDF visual review, and package-installed smoke.
- `uv run --locked python scripts/validate_pipeline_freshness.py` and
  `uv run --locked python scripts/build_release.py --verify` after final
  regeneration.
- Cross-clone report/output/distribution digest comparison and structured
  reviewer-verdict schema validation.

## Claim-Boundary Constraints

- Falsifier: a moved base, dirty tree, renderer mismatch, unresolved history
  finding, failed source/render/package check, clone divergence, missing
  receipt, ineligible verdict, undispositioned concern, or `fail` blocks the
  affected push, merge, tag, or release step.
- Prohibited claims: clean clones, fingerprints, green CI, tagged structure, or
  a release verdict do not establish scientific validity, universal robustness,
  source-protocol identity, physical multi-host execution, WCAG, or PDF/UA.
- Do not treat a local subagent as an eligible reviewer or an owner-author
  verdict as independent reproduction.

## Dependencies

The candidate gate depends on a fixed branch/tree, current remote heads, the
approved sanitized-history policy, and an exact clean renderer. The final
campaign begins only after MED-5A merges the release-only identity/artifact PR
and public `main` is green; it then depends on safe external headroom and current
toolchain access. After this exact-final-SHA ladder and the selected owner-author
`pass`, MED-5B owns tag, GitHub, and Zenodo publication; MIN-3 then verifies and
closes the live documentation handoff before MED-5B performs the one-time
tree-identical private synchronization. MAJ-8 begins only after that sequence.
