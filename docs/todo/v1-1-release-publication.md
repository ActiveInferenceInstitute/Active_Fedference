# v1.1.0 Release, Publication, And Private Synchronization

[Back to roadmap](../../TODO.md)

## Status

- Priority class: Medium
- State: Open
- Queue position: first on the post-publication path, before MIN-3 closure and
  MAJ-8
- Owner surface: published release identity, retrospective exact-SHA
  certification, the historical publication-approval record, and the single
  private-main synchronization

## Rationale

The development line must not acquire a final version, DOI, or release date
until its source and visual base is integrated and green. In v1.1.0 that order
was not followed: the release-only merge, annotated tag, GitHub release, and
Zenodo publication completed on 2026-09-19/20 before the exact-SHA campaign.
Those remain separate state transitions with different evidence and authority
requirements, and the published bytes cannot be retrofitted.

## Scope

### MED-5A — linked draft and release-identity integration

MED-5A completed: the release-only branch `codex/v1.1.0-release` merged as
`c759541` ("Merge release v1.1.0", 2026-09-19), changed the identity to `1.1.0`,
bound the DOI `10.5281/zenodo.22149133`, regenerated lifecycle metadata and
artifacts, added the version/DOI-named PDF without replacing historical PDFs,
and updated the immutable-PDF ledger. The tracked current-release documentation
already carries the canonical repository identity, deterministic tag/release
URLs, DOI, GitHub-asset installation commands, checksums workflow, application
entry points, and accessibility boundaries.

The remaining MED-5A obligation is evidence, not a new identity: confirm that
the tagged source, wheel, sdist, PDF, and HTML agree on those values, and that
the release-only PR was a normal merge with all checks then available on the
exact head and merge ref. Hosted Actions is billing-locked org-wide, so a
historical hosted-check claim must be read from the recorded run rather than
re-run. MIN-3 later verifies the live records but does not retrofit an already
published release payload. Do not re-create the identity or the release PR.

### MIN-2 bridge — certify the exact final main commit

Run MIN-2's two sequential clean-clone ladders against `c759541` and obtain the
selected Daniel Ari Friedman structured verdict. The owner-author verdict is the
chosen eligible technical checkpoint; it is not independent or cross-vendor
reproduction, does not close ISC-89, and is distinct from authorization to tag
or publish. Because the tag and publication already exist, the verdict now
certifies the published commit retrospectively and cannot authorize a new tag.

### MED-5B — immutable publication and final private synchronization

The tag, GitHub release, and Zenodo version are already published: annotated tag
object `c1f1a33`, peeled commit `c759541`, release assets re-downloaded and
re-hashed in the release-state record, and Zenodo record 22149133 with the
matching PDF. MED-5B now owns verification and the single private
synchronization, not another publication. Record the historical publication
approval if an external record exists; otherwise retain an explicit gap. Never
re-publish, re-upload, or replace the published bytes to reconstruct history.

After independent post-publication verification, complete MIN-3 through a
reviewed normal-merge public PR, including removal of its closed forward-backlog
entry. Only then merge the resulting final public `main` into private
`origin/main` once with a normal merge commit whose tree is exactly the public
tree. No public repository commit may intervene between that synchronization
and the MAJ-8 branch point.

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
- Treat the published Zenodo version and GitHub assets as immutable. Never
  mutate v1.0.4 or v1.1.0, move a tag, replace published Zenodo bytes, or edit
  published metadata as a shortcut.
- Any later source or identity change requires a new version through a fresh
  reviewed release-only PR; it never re-cuts or re-publishes v1.1.0.

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
  public-main SHA. It certifies the published commit but is not independent
  replication and does not close ISC-89.
- GitHub tag object, peeled commit, release target, filenames, sizes, API
  digests, local hashes, installability, package contents, and the recorded
  exact-commit CI run agree. Hosted Actions is billing-locked, so no new hosted
  run is required or claimed.
- The tagged and packaged documentation already contains the final repository,
  version, deterministic tag/release URLs, DOI, installation route, application
  path, and accessibility boundaries.
- The Zenodo record has the approved metadata and relation, exactly one PDF,
  and PDF bytes identical to the canonical GitHub PDF; the historical
  publication approval is attested from an external record or retained as an
  explicit gap.
- MIN-3 closes through a reviewed normal-merge public PR after publication and
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
- Published Zenodo record verification against the downloaded GitHub PDF, plus
  independent DOI, relation, file, checksum, tag, and recorded-CI checks.
- Public/private tree comparison and immutable tag-object comparison after the
  private synchronization.

## Claim-Boundary Constraints

- Falsifier: identity drift, a stale or red PR head, clone divergence, an
  ineligible or non-`pass` verdict, tag mismatch, asset/checksum mismatch,
  unexpected Zenodo state or file, unattested historical publication approval,
  or unequal public/private trees blocks the affected state transition.
- Prohibited claims: a version, green workflow, GitHub release, DOI, receipt,
  or tagged PDF does not establish universal robustness, calibration,
  scientific validity, deployment readiness, WCAG, PDF/UA, or any MAJ-6 result.
- Publication must not alter the immutable v1.0.4 tag, release, Zenodo record,
  metadata, or PDF bytes.

## Dependencies

MED-5A and the irreversible publication are historical: the release-only merge
`c759541`, tag `v1.1.0`, GitHub release, and Zenodo record 22149133 exist and
must not be recreated. The remaining MED-5 work depends on MIN-2's retrospective
exact-commit evidence and owner-author `pass`, an external record of the
historical publication approval (or an explicit gap), and a final public `main`
that includes MIN-3. MIN-3 follows verified publication, precedes the single
private synchronization, and must close before MAJ-8 begins. MED-4 no longer
gates this release; its visual work is post-v1.1.0 documentation work that must
merge before the private synchronization or be scheduled under a new version.
