# v1.1.0 Release, Publication, And Private Synchronization

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Medium
- State: Open
- Queue position: after the visual-scholarship merge and before MAJ-8
- Owner surface: release identity, public PR, annotated tag, GitHub release,
  linked Zenodo version, and the single private-main synchronization

## Rationale

The development line must not acquire a final version, DOI, or release date
until its source and visual base is integrated and green. GitHub publication,
Zenodo draft mutation, irreversible Zenodo publication, and private-main
synchronization are separate state transitions with different evidence and
authority requirements.

## Scope

### MED-5A — linked draft and release-identity integration

From the exact green visual merge on public `main`, create the release-only
branch `codex/v1.1.0-release`. Create or recover the linked Zenodo next-version
draft from published record `21972644`, inspect it before mutation, and bind
the returned reserved DOI to the release identity. Change `1.1.0.dev0` to
`1.1.0`, set the actual UTC release date, regenerate lifecycle metadata and
artifacts, add the new version/DOI-named PDF without replacing historical
PDFs, and update the immutable-PDF ledger.

MED-5A also owns every tracked current-release documentation edit: canonical
repository identity, deterministic `v1.1.0` tag/release URLs, reserved DOI,
GitHub-asset installation commands, checksums workflow, application entry
points, and accessibility boundaries. These values must be correct in the
tagged source, wheel, sdist, PDF, and HTML; MIN-3 later verifies the live records
but does not retrofit an already published release payload.

Push the exact release branch to both Fedference remotes, open a public
release-only PR, require all hosted jobs on the exact head and merge ref, merge
normally, and require green post-merge CI. This integration step precedes the
authoritative MIN-2 campaign because the exact public-main merge commit—not the
release PR head—must be certified.

### MIN-2 bridge — certify the exact final main commit

Run MIN-2's two sequential clean-clone ladders against that exact public-main
merge commit and obtain the selected Daniel Ari Friedman structured verdict.
The owner-author verdict is the chosen eligible technical checkpoint; it is not
independent or cross-vendor reproduction, does not close ISC-89, and is distinct
from authorization to tag or publish.

### MED-5B — immutable publication and final private synchronization

Only after an exact-SHA owner-author `pass`, create and verify the annotated tag
and GitHub release assets under the already granted release authority. Update
and verify the unpublished linked Zenodo draft, then pause for fresh explicit
approval before the irreversible publication call. After independent
post-publication verification, complete MIN-3 through a reviewed normal-merge
public PR, including removal of its closed forward-backlog entry. Only then
merge the resulting final public `main` into private `origin/main` once with a
normal merge commit whose tree is exactly the public tree. No public repository
commit may intervene between that synchronization and the MAJ-8 branch point.

No PyPI publication is included.

## Implementation Notes

- Use current approved creator, ORCID, affiliation, publication email, title,
  abstract, MIT license, attribution, repository, and UTC date policy. Stop on
  any drift that requires a new owner decision.
- The public repository is
  `https://github.com/ActiveInferenceInstitute/Active_Fedference`; `origin` is
  the private/interim evidence remote, not the public release destination.
- Stage only the canonical PDF, reproducible wheel, reproducible source
  distribution, `manifest.json`, and a sorted `SHA256SUMS.txt`. Verify every
  downloaded asset again.
- Replace only the inherited same-named generic PDF in the unsubmitted Zenodo
  draft. Never mutate v1.0.4, move a tag, replace published Zenodo bytes, or
  edit published metadata as a shortcut.
- If the release date or final public-main SHA changes, use another reviewed
  PR and repeat all dependent generation and certification.

## Acceptance Criteria

- Primary estimand: not applicable; success is exact identity, artifact, and
  publication-state agreement rather than a scientific contrast.
- Independent replication unit: not applicable; audit/review units are one
  exact public-main merge commit, each of two isolated clone runs, one immutable
  tag object, one downloaded release asset, and one linked Zenodo version
  record.
- The release-only PR contains identity, historical-PDF-ledger, tracked
  current-release documentation, and derived artifact changes only and merges
  normally after exact-head checks pass. Its tracked documentation is limited
  to final repository/release links, installation and checksum instructions,
  application entry points, and accessibility boundaries.
- One identified owner-author structured `pass` is bound to the exact certified
  public-main SHA. It permits tagging but is not independent replication and
  does not close ISC-89.
- GitHub tag object, peeled commit, release target, filenames, sizes, API
  digests, local hashes, installability, package contents, and public CI agree.
- The tagged and packaged documentation already contains the final repository,
  version, deterministic tag/release URLs, reserved DOI, installation route,
  application path, and accessibility boundaries before publication.
- The Zenodo record has the approved metadata and relation, exactly one PDF,
  and PDF bytes identical to the canonical GitHub PDF; publication occurs only
  after a fresh explicit approval.
- MIN-3 closes through a green normal-merge public PR after publication and
  before the single private synchronization; its closed row/page do not linger
  in the forward-only backlog.
- The private synchronization is non-forced, preserves both histories, and
  produces a tree exactly equal to final public `main` after MIN-3 closes.
- Required acceptance evidence: public and private merge SHAs, tag object and
  peeled commit, release URL and asset hashes, Zenodo record/DOI/concept chain,
  PDF checksum, CI URLs, clone-receipt digests, owner verdict, and the remaining
  ISC-89/scientific boundaries.

## Verification Probes

- Lifecycle-aware metadata emission and validation in both development and
  final modes.
- MIN-2's two-clone exact-commit ladder, installed wheel/sdist smoke, and
  reproducible-build comparison.
- GitHub API plus downloaded-byte verification for every release asset.
- Zenodo draft metadata/file verification before publication and independent
  DOI, relation, file, checksum, tag, and CI checks after publication.
- Public/private tree comparison and immutable tag-object comparison after the
  private synchronization.

## Claim-Boundary Constraints

- Falsifier: identity drift, a stale or red PR head, clone divergence, an
  ineligible or non-`pass` verdict, tag mismatch, asset/checksum mismatch,
  unexpected Zenodo state or file, absent publication approval, or unequal
  public/private trees blocks the affected state transition.
- Prohibited claims: a version, green workflow, GitHub release, DOI, receipt,
  or tagged PDF does not establish universal robustness, calibration,
  scientific validity, deployment readiness, WCAG, PDF/UA, or any MAJ-6 result.
- Publication must not alter the immutable v1.0.4 tag, release, Zenodo record,
  metadata, or PDF bytes.

## Dependencies

MED-5A depends on MED-4's merged, green public-main result, current release
metadata approvals, and a safely editable linked Zenodo draft. MIN-2 then
depends on the exact release-identity merge commit. MED-5B depends on MIN-2's
full exact-final-SHA evidence and owner-author `pass`, plus a later fresh
approval for the irreversible Zenodo publication call. MIN-3 follows verified
publication, precedes the single private synchronization, and must close before
MAJ-8 begins.
