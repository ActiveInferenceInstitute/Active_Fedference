# Clone-Correct Integration And Release Verification Ladder

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Minor
- State: Open — exact-commit fresh-clone certification, release-verdict,
  broader independent-reproduction, and author-authority gates remain for the
  next fully reviewed release wave
- Owner surface: git tracking set, release workflow, reviewer eligibility,
  broader cross-vendor verification, publication accessibility decision

## Rationale

Clean-checkout evidence is commit-specific. A prior commit's passing
`validate_clean_checkout.py` result never certifies a later development
overlay, even when the later source suite is green. Analysis, hydration,
rendering, and release receipts are likewise trusted only when their declared
input and output hashes match the reviewed tree in producer order.

The latest published snapshot is v1.0.4. The current v1.1 application work is
an unreleased development line and deliberately carries neither a version DOI
nor a release date. A scoped feature-branch push and review PR may expose an
exact candidate to hosted checks only after the approved sanitized-history
confidentiality gate below passes. It does not integrate the candidate into private or public
`main` and does not create a release. Hosted checks govern the development PR.
After that PR and the separate release-identity PR are merged, the complete
release ladder requires two genuinely isolated clones of the exact final
public-main commit before a tag is created. Those clone runs establish
regeneration and isolated-environment behavior for the release-identity
commit; they do not substitute for the separate reviewer-verdict or
release-authority gates, and they do not close the broader cross-vendor and
independent-reproduction lanes.

## Scope

1. **Tracking set.** The prior load-bearing source, documentation, tests,
   generated reports, figures, and reviewer snapshot were committed; the
   current development additions must be tracked and re-probed before that
   statement becomes current again. The sibling
   template repository's rendering changes remain owned by that repository.
   Keep the executable tracking/import probe in the release ladder.
2. **Exact-history confidentiality before a public push.** Refresh the public
   base, fix the exact candidate SHA, and inspect every object newly reachable
   in `public/main..candidate`, not only the candidate tree. Enumerate current
   and deleted paths and blobs; scan the full range for secrets; review commit
   messages and author identities; inspect machine-local paths, confidential
   drafts, proprietary inputs, and generated artifacts; review every object at
   or above 50 MiB; and hard-block any object over GitHub's 100 MiB limit.
   Retain the public-base SHA, candidate SHA, tool versions, findings, and
   disposition. Any unresolved publishability concern blocks the public branch
   push and PR. The approved v1.1 policy uses a four-commit sanitized replay
   based directly on refreshed public `main`; the retained private evidence
   branch is not rewritten. Any further history-policy change requires new
   approval and complete commit-bound regeneration.
3. **Fresh-clone verification (the rigorous "check everything" procedure).**
   Run this authoritative two-clone campaign only after the development PR and
   release-identity PR have merged and public `main` is green, on the exact
   final public-main SHA that would be tagged.
   Recheck free space and provision at least 40 GiB of safe working headroom
   without reclaiming active research data or caches absent a separate
   ownership check. From each of two isolated clean checkouts, run the full
   ladder in order:
   - verify the content-bound analysis → hydration → render receipt chain;
   - source gates: Ruff, mypy, invariants, layer-boundary grep;
   - full coverage suite at the declared gate floor;
   - report scale guard (publication n-fields) after the suite — this must pass
     *after* the suite, proving the suite does not touch the committed snapshot;
   - publication regeneration (analysis, token hydration) and byte-comparison of
     regenerated reports against the committed snapshot;
   - render of all three surfaces and the count-based invariants: rendered
     theorem-box count vs source environment count, zero unresolved reference
     markers in the slide text layer, zero unresolved tokens;
   - raster reads of at least one formalism page and one results page of the
     PDF (text extraction cannot see math-rendering or scale regressions);
   - release build plus fingerprint verification;
   - build the wheel and source distribution, install each into its own empty
     environment, and run the installed CLI plus a core import/aggregation
     smoke;
   - build both distribution formats twice with the same
     `SOURCE_DATE_EPOCH` and require byte-identical pairs. The custom PEP 517
     wrapper normalizes archive order, owner, and time metadata and the
     setuptools build backend is exactly pinned; backend-version or checkout
     mtime drift is a real release failure.
   Use candidate-specific write-once paths beneath
   `/Volumes/blue/active_fedference-verification/v1.1.0-<FINAL_SHA>/`, with
   `clone-a` and `clone-b` as separate checkout roots and receipts retained in
   sibling directories outside both clones. This mount layout is an approved,
   non-confidential operational example: it exposes no credential, user home,
   dataset location, or proprietary input and is allowed by the public-history
   audit.
4. **Release-certification verdict.** Obtain exactly one structured
   `pass`/`concerns`/`fail` verdict on the exact final SHA, tree, manifest,
   claim ledger, no-claim boundaries, and two-clone evidence. The eligible
   reviewer is either an identified human or a genuinely different-vendor
   model. A local subagent does not qualify. Daniel Ari Friedman's selected
   verdict is recorded as an identified owner-author human review, not as
   independent external replication or cross-vendor review. `pass` permits
   tagging; every `concerns` item requires disposition or repair through a new
   public PR and complete recertification; `fail` blocks the release. This
   release-specific verdict does not close ISC-89 or the broader cross-vendor
   and independent-reproduction lane.
5. **Accessibility disposition.** Treat the validated HTML manuscript as the
   canonical accessibility-enhanced publication surface. The combined
   manuscript PDF must be emitted by the source-controlled tagged producer and
   pass the `Tagged: yes`, qpdf `/Lang`, language, and `StructTreeRoot` gates.
   The validator accepts catalog language when Poppler omits its optional
   `Language:` line. Slide PDFs are separate convenience surfaces. Tagged
   structure is not PDF/UA conformance; a PDF/UA statement still requires a
   dedicated conformance validator and manual review.

## Implementation Notes

- The suite-write scaffolding is already in place: subprocess smoke tests
  redirect through the validated `ACTIVE_FEDFERENCE_PROJECT_ROOT` override and
  the scale guard is the standing tripwire. The ladder's post-suite guard step
  exists to keep that property continuously proven.
- Do not weaken any gate to make a fresh-clone run pass; a red step is a real
  finding about the release, not about the ladder.
- `scripts/validate_clean_checkout.py` is now the executable front door for the
  clone/tracking/import check. `scripts/validate_pipeline_freshness.py` is the
  executable front door for upstream/downstream artifact freshness. Both are
  release-evidence probes; neither upgrades scientific claims.
- Exact run counts, hashes, rendering receipts, and the current development
  disposition belong in `ISA.md` and generated verification artifacts. They
  are intentionally not copied into this forward-looking page, where they
  would become stale after the next source change.
- Other live research campaigns may be active on the workstation or external
  volumes. No cache or data reclamation is authorized by this roadmap item.
  Free space is volatile and must be rechecked immediately before each
  isolated clone/render run; the 40 GiB threshold is a planning floor, not
  deletion authority.
- For a new unreleased draft, the exact package/manuscript version ends in
  `.devN`, `publication.doi` is the empty string,
  `publication.doi_status` is exactly `(forthcoming)`, and `date_released` is
  null. The split preserves plain cover text without inventing an unassigned
  DOI resolver link. Generated CFF, Zenodo, and CodeMeta surfaces omit their DOI
  and release-date fields, and the package URL table omits a version-specific
  DOI. The immutable v1.0.4 release retains its DOI/date on its own tag and
  artifacts; development metadata must not copy them forward before approval.

## Acceptance Criteria

- Primary estimand: not applicable — this is release-integrity engineering; no
  scientific quantity is measured. Success is judged by ladder outcomes on a
  clean checkout, not by a statistical contrast.
- Independent/replication unit: one fresh clone plus one complete ladder run;
  two isolated clone units must reproduce the required results.
- Every load-bearing file is tracked; the post-commit probe confirms that
  `git ls-files` covers every import, docs-contract target, and test in the
  verified baseline.
- The full ladder passes from a fresh clone with no machine-local dependencies
  beyond the declared toolchain.
- Wheel and source-distribution installs expose `fedference aggregate`, `list`,
  `run`, `benchmark`, `verify`, and `replay`, retain `py.typed`, and run a
  labeled own-data application without importing Torch in the default
  NumPy/SciPy path.
- Before any public development-branch push, the sanitized-history scan covers all
  newly reachable commits and blobs, including deleted paths, and records an
  explicit publishability disposition with no unresolved concerns.
- A future draft carries no DOI or release date until confidentiality, license,
  attribution, and author approval are complete. Published v1.0.4 metadata is
  immutable release identity, not a development default.
- The eventual committed `output/` snapshot matches a post-commit regeneration at
  publication scale (value-identical reports, declared volatile fields aside).
- One eligible structured release-certification verdict exists. The evidence
  identifies whether it came from a human or genuinely different-vendor
  reviewer and never relabels an owner-author human verdict as independent
  external replication. ISC-89 and the broader independent-reproduction lane
  remain open after an owner-author `pass`.
- Each decision-queue item has a recorded decision and, where accepted, its
  implementation and test.

## Verification Probes

- Falsifier: a fresh clone that fails import, the suite, the post-suite scale
  guard, regeneration parity, a rendered-surface count invariant, or fingerprint
  verification falsifies the release claim; a missing, ineligible, `concerns`
  without disposition, or `fail` release verdict blocks tagging. An
  owner-author `pass` still cannot establish independent verification.
- Falsifier: a changed report, manuscript input, or rendered surface that still
  passes `validate_pipeline_freshness.py` without the dependent receipt being
  regenerated falsifies the stage-order guard.
- `git ls-files` audit against the recorded load-bearing set.
- `git fetch --prune public`, followed by a commit- and blob-level audit of the
  fixed `public/main..candidate` range, a full-history secret scan, size review,
  and recorded disposition before the public branch push.
- `uv run --locked python scripts/validate_clean_checkout.py` from the clean clone and
  `uv run --locked python scripts/validate_pipeline_freshness.py` after stages 03–05.
- The ordered ladder commands in `TODO.md` "Gates For Any Item" plus the
  fresh-clone additions above, each with captured output.
- Raster page reads and count-based rendered-surface checks as listed in Scope.

## Claim-Boundary Constraints

Prohibited claims (no-claim boundary):

- No claim that release integrity, fingerprints, or a green ladder add
  scientific evidence for any estimand; they establish reproducibility of the
  already-declared results only.
- No use of a green fresh-clone run to promote `robust_aggregate` beyond its
  recovery-limit guarantee or `variational_aggregate` beyond its raw
  effective-weight bound.
- No treating a local subagent as an eligible reviewer, or an owner-author
  human verdict as independent external or cross-vendor replication.

## Dependencies

None blocking; interacts with the implemented release fingerprint and receipt
capability in `src/publication/release_manifest.py` and
`src/publication/pipeline_freshness.py`, and the phase ordering owned by
[`scholarship-and-phase-plan.md`](scholarship-and-phase-plan.md).
Future public release waves additionally require confidentiality, third-party
attribution, license, and author approval outside local validation. Those
governance checks are not retroactively claimed as scientific evidence for the
published v1.0.4 artifact.
