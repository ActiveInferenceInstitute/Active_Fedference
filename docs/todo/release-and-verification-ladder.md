# Clone-Correct Release And Full Verification Ladder

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Minor
- State: Open — independent fresh-clone, cross-vendor, and author-authority
  gates remain for the next fully reviewed release wave
- Owner surface: git tracking set, release workflow, cross-vendor verification,
  publication accessibility decision

## Rationale

Clean-checkout evidence is commit-specific. A prior commit's passing
`validate_clean_checkout.py` result never certifies a later development
overlay, even when the later source suite is green. Analysis, hydration,
rendering, and release receipts are likewise trusted only when their declared
input and output hashes match the reviewed tree in producer order.

The v0.1.0 GitHub/Zenodo publication is complete, but the complete ladder still
requires two genuinely isolated clones of the exact
reviewed commit. Those clone runs establish regeneration and
independent-environment behavior; they do not substitute for the separate
cross-vendor or release-authority gates.
Separately, the cross-vendor audit gate has never produced a structured verdict
(the local Codex installation has two binaries with a version/cache skew), so
one independent verification lane remains unexercised.

## Scope

1. **Tracking set.** The prior load-bearing source, documentation, tests,
   generated reports, figures, and reviewer snapshot were committed; the
   current development additions must be tracked and re-probed before that
   statement becomes current again. The sibling
   template repository's rendering changes remain owned by that repository.
   Keep the executable tracking/import probe in the release ladder.
2. **Fresh-clone verification (the rigorous "check everything" procedure).**
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
3. **Cross-vendor verdict.** Repair the codex installation (reconcile the two
   installed binaries and the models-cache version skew) or use another
   non-Anthropic lane, and obtain one structured pass/concerns/fail verdict on
   the claim surface. A no-signal result is recorded as unsatisfied, never
   substituted.
4. **Accessibility disposition.** Treat the validated HTML manuscript as the
   canonical accessibility-enhanced publication surface. The combined PDF and
   slide PDFs remain structurally and visually checked convenience surfaces,
   but they must not be described as tagged or PDF/UA-conformant until the
   sibling rendering producer emits tagged files and a dedicated conformance
   validator passes them.

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
- For a new unreleased draft, metadata omits `date_released`, CFF
  `date-released`, Zenodo `publication_date`, and CodeMeta `dateModified`.
  The published v0.1.0 surfaces intentionally contain those release fields;
  a future draft must not copy them forward before approval.

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
- Wheel and source-distribution installs expose `fedference list`, `run`,
  `benchmark`, `verify`, and `replay` without importing Torch in the default
  NumPy/SciPy path.
- A future draft carries no release date until confidentiality, license,
  attribution, and author approval are complete. The published v0.1.0 metadata
  is not a draft.
- The eventual committed `output/` snapshot matches a post-commit regeneration at
  publication scale (value-identical reports, declared volatile fields aside).
- One structured cross-vendor verdict exists, or the gate is explicitly
  recorded as unsatisfied with the failure mode captured.
- Each decision-queue item has a recorded decision and, where accepted, its
  implementation and test.

## Verification Probes

- Falsifier: a fresh clone that fails import, the suite, the post-suite scale
  guard, regeneration parity, a rendered-surface count invariant, or fingerprint
  verification falsifies the release claim; a verdictless cross-vendor run
  falsifies the independent-verification claim.
- Falsifier: a changed report, manuscript input, or rendered surface that still
  passes `validate_pipeline_freshness.py` without the dependent receipt being
  regenerated falsifies the stage-order guard.
- `git ls-files` audit against the recorded load-bearing set.
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
- No treating a same-family review as a cross-vendor verdict.

## Dependencies

None blocking; interacts with the implemented release fingerprint and receipt
capability in `src/publication/release_manifest.py` and
`src/publication/pipeline_freshness.py`, and the phase ordering owned by
[`scholarship-and-phase-plan.md`](scholarship-and-phase-plan.md).
Future public release waves additionally require confidentiality, third-party
attribution, license, and author approval outside local validation. Those
governance checks are not retroactively claimed as scientific evidence for the
published v0.1.0 artifact.
